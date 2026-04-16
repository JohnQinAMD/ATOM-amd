# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
ZMQ-based bootstrap for KV transfer between disaggregated prefill and decode workers.

The bootstrap protocol exchanges MoRI IOEngine descriptors (EngineDesc + MemoryDesc)
so that the prefill worker can RDMA-WRITE KV blocks directly into the decode worker's
GPU memory. The `bootstrap_room` (a random 63-bit ID assigned by the Dynamo
PrefillRouter) is used as the transfer_uid for RDMA completion tracking.

Protocol:
    1. Decode starts a ZMQ ROUTER socket on its bootstrap_port.
    2. Prefill connects (ZMQ DEALER) and sends:
       [engine_desc_packed, mem_desc_packed, src_block_ids, bootstrap_room]
    3. Decode replies:
       [engine_desc_packed, mem_desc_packed, dst_block_ids]
    4. Both sides call KVTransferOp.register_peer() with the received descriptors.
    5. Prefill calls send_blocks(); decode polls for completion.
"""

import logging
import pickle
import socket
import struct
from typing import Optional

import zmq

logger = logging.getLogger("atom")


def _get_local_ip() -> str:
    """Best-effort local IP for RDMA advertisement."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class KVBootstrapServer:
    """Decode-side bootstrap server. Accepts prefill connections and exchanges
    RDMA descriptors for one KV transfer session at a time.

    The server listens on a ZMQ ROUTER socket. For each incoming request
    (identified by bootstrap_room), it reserves destination blocks, replies
    with its own RDMA descriptors, and returns the dst_block_ids for the
    caller to feed into the decode-only scheduler path.
    """

    def __init__(self, port: int = 0):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        if port == 0:
            port = self._sock.bind_to_random_port("tcp://*")
        else:
            self._sock.bind(f"tcp://*:{port}")
        self._port = port
        self._host = _get_local_ip()
        logger.info(f"KVBootstrapServer listening on {self._host}:{self._port}")

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def accept(self, timeout_ms: int = 60000) -> Optional[dict]:
        """Wait for one prefill connection and return its metadata.

        Returns dict with keys:
            identity: ZMQ routing identity (for reply)
            engine_desc: packed bytes
            mem_desc: packed bytes
            src_block_ids: list[int]
            bootstrap_room: int
        """
        if self._sock.poll(timeout_ms):
            frames = self._sock.recv_multipart()
            # ROUTER prepends identity frame
            identity = frames[0]
            payload = pickle.loads(frames[1])
            payload["identity"] = identity
            return payload
        return None

    def reply(self, identity: bytes, engine_desc: bytes, mem_desc: bytes,
              dst_block_ids: list[int]) -> None:
        """Send decode-side descriptors back to the prefill worker."""
        payload = pickle.dumps({
            "engine_desc": engine_desc,
            "mem_desc": mem_desc,
            "dst_block_ids": dst_block_ids,
        })
        self._sock.send_multipart([identity, payload])

    def close(self):
        self._sock.close(linger=0)
        self._ctx.term()


class KVBootstrapClient:
    """Prefill-side bootstrap client. Connects to a decode worker's bootstrap
    server, sends RDMA descriptors + block info, and receives the decode
    side's descriptors + destination block IDs.
    """

    def __init__(self):
        self._ctx = zmq.Context()

    def exchange(
        self,
        decode_host: str,
        decode_port: int,
        engine_desc: bytes,
        mem_desc: bytes,
        src_block_ids: list[int],
        bootstrap_room: int,
        timeout_ms: int = 60000,
    ) -> Optional[dict]:
        """Connect to decode bootstrap server, exchange descriptors.

        Returns dict with keys:
            engine_desc: packed bytes (decode side)
            mem_desc: packed bytes (decode side)
            dst_block_ids: list[int] (where to write on decode side)
        """
        sock = self._ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect(f"tcp://{decode_host}:{decode_port}")

        try:
            payload = pickle.dumps({
                "engine_desc": engine_desc,
                "mem_desc": mem_desc,
                "src_block_ids": src_block_ids,
                "bootstrap_room": bootstrap_room,
            })
            sock.send(payload)
            reply = pickle.loads(sock.recv())
            return reply
        except zmq.Again:
            logger.error(
                f"KV bootstrap exchange timed out ({timeout_ms}ms) "
                f"connecting to {decode_host}:{decode_port}"
            )
            return None
        finally:
            sock.close(linger=0)

    def close(self):
        self._ctx.term()


class KVTransferManager:
    """High-level manager combining KVTransferOp + bootstrap for one worker.

    Prefill usage:
        mgr = KVTransferManager(kv_cache, layout, role="prefill")
        # For each request:
        mgr.send_kv(block_table, decode_host, decode_port, bootstrap_room)

    Decode usage:
        mgr = KVTransferManager(kv_cache, layout, role="decode",
                                bootstrap_port=12345)
        # For each request:
        dst_block_ids = mgr.recv_kv(num_blocks, block_manager)
    """

    def __init__(self, kv_cache, layout, role, bootstrap_port=0, host=""):
        from atom.model_engine.kv_transfer import KVTransferOp

        self.role = role
        self.layout = layout
        self.transfer_op = KVTransferOp(
            kv_cache, layout, host=host or _get_local_ip(), port=0
        )

        if role == "decode":
            self._server = KVBootstrapServer(port=bootstrap_port)
            self._client = None
        elif role == "prefill":
            self._server = None
            self._client = KVBootstrapClient()
        else:
            raise ValueError(f"role must be 'prefill' or 'decode', got {role!r}")

    @property
    def bootstrap_host(self) -> str:
        if self._server:
            return self._server.host
        return _get_local_ip()

    @property
    def bootstrap_port(self) -> int:
        if self._server:
            return self._server.port
        return 0

    def send_kv(
        self,
        src_block_ids: list[int],
        decode_host: str,
        decode_port: int,
        bootstrap_room: int,
    ) -> None:
        """Prefill side: exchange descriptors with decode, then RDMA-WRITE KV blocks."""
        assert self.role == "prefill"

        reply = self._client.exchange(
            decode_host,
            decode_port,
            self.transfer_op.engine_desc_packed,
            self.transfer_op.local_mem_packed,
            src_block_ids,
            bootstrap_room,
        )
        if reply is None:
            raise RuntimeError("KV bootstrap exchange failed (timeout)")

        peer_key = self.transfer_op.register_peer(
            reply["engine_desc"], reply["mem_desc"]
        )
        dst_block_ids = reply["dst_block_ids"]

        self.transfer_op.send_blocks(
            peer_key, src_block_ids, dst_block_ids, bootstrap_room
        )

    def recv_kv(self, block_manager, timeout_ms=60000) -> dict:
        """Decode side: accept prefill connection, reserve blocks, receive KV.

        Returns dict with:
            dst_block_ids: list[int] — blocks now populated with KV data
            bootstrap_room: int — the session ID
        """
        assert self.role == "decode"

        req = self._server.accept(timeout_ms=timeout_ms)
        if req is None:
            raise RuntimeError("KV bootstrap accept timed out")

        num_blocks = len(req["src_block_ids"])
        dst_block_ids = block_manager.reserve_blocks(num_blocks)

        peer_key = self.transfer_op.register_peer(
            req["engine_desc"], req["mem_desc"]
        )

        self._server.reply(
            req["identity"],
            self.transfer_op.engine_desc_packed,
            self.transfer_op.local_mem_packed,
            dst_block_ids,
        )

        # Wait for prefill's RDMA writes to arrive
        bootstrap_room = req["bootstrap_room"]
        import time
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self.transfer_op.wait_for_transfer(peer_key, bootstrap_room):
                break
            time.sleep(0.001)  # 1ms poll
        else:
            raise RuntimeError(
                f"KV transfer recv timed out after {timeout_ms}ms, room={bootstrap_room}"
            )

        logger.info(
            f"KV transfer received: {num_blocks} blocks, room={bootstrap_room}"
        )
        return {
            "dst_block_ids": dst_block_ids,
            "bootstrap_room": bootstrap_room,
        }

    def close(self):
        if self._server:
            self._server.close()
        if self._client:
            self._client.close()
        self.transfer_op.close()
