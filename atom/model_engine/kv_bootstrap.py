# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
ZMQ-based bootstrap for KV transfer between disaggregated prefill and decode workers.

Design: bidirectional PUSH/PULL keyed by bootstrap_room.

  Decode worker                       Prefill worker
  -------------                       --------------
  bind PULL on :dec_port              bind PULL on :pre_port
  PUSH(request {room, my_desc, my_mem, dec_block_ids, reply_endpoint}) ─────►  ── puller thread queues by room
                                                                              │
  ◄────── PUSH(reply {room, pre_desc, pre_mem, pre_block_ids})  ── after   ◄──┘
                                                                  prefill
                                                                  RDMA-WRITEs

  wait_for_transfer(peer_key, room) ───────────────────────────────► polling on mori

Why not ROUTER/DEALER: ZMQ DEALER identities are per-connection. A timed-out
DEALER socket's identity is permanently dead. Routing replies by ZMQ identity
meant that a retry (new DEALER, new identity) would never see the reply even
if the server had queued the earlier identity and then replied to it. Keying
everything by `bootstrap_room` in the payload removes this class of bugs.

`bootstrap_room` is a 63-bit random ID assigned by the Dynamo PrefillRouter
(or locally by the prefill_handler if the router didn't set one). It is also
used as the MoRI `transfer_uid` so RDMA completion tracking reuses the same ID.
"""

import logging
import pickle
import socket
import threading
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


class _Puller:
    """Background PULL socket that buffers incoming messages by bootstrap_room."""

    def __init__(self, port: int = 0):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.LINGER, 0)
        if port == 0:
            port = self._sock.bind_to_random_port("tcp://*")
        else:
            self._sock.bind(f"tcp://*:{port}")
        self._port = port
        self._host = _get_local_ip()

        self._by_room: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="kv-bootstrap-puller"
        )
        self._thread.start()
        logger.info(f"KV bootstrap puller listening on {self._host}:{self._port}")

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def _loop(self) -> None:
        while self._running:
            try:
                if not self._sock.poll(timeout=1000):
                    continue
                payload = pickle.loads(self._sock.recv())
                room = payload.get("bootstrap_room", 0)
                kind = payload.get("kind", "?")
                with self._cv:
                    self._by_room[room] = payload
                    self._cv.notify_all()
                logger.debug(f"Puller: buffered {kind} for room={room}")
            except Exception as e:
                if self._running:
                    logger.warning(f"Puller loop error: {e}")

    def pop(self, room: int, timeout_s: float) -> Optional[dict]:
        """Block until a message for `room` is available, or `timeout_s` elapses."""
        import time
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                msg = self._by_room.pop(room, None)
                if msg is not None:
                    return msg
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=min(5.0, remaining))

    def close(self) -> None:
        self._running = False
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


class _Pusher:
    """Cached PUSH sockets to remote PULL endpoints (endpoint → socket)."""

    def __init__(self):
        self._ctx = zmq.Context.instance()
        self._sockets: dict[str, "zmq.Socket"] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get(self, endpoint: str) -> tuple["zmq.Socket", threading.Lock]:
        with self._global_lock:
            if endpoint not in self._sockets:
                sock = self._ctx.socket(zmq.PUSH)
                sock.setsockopt(zmq.LINGER, 1000)
                sock.setsockopt(zmq.SNDHWM, 64)
                sock.connect(endpoint)
                self._sockets[endpoint] = sock
                self._locks[endpoint] = threading.Lock()
            return self._sockets[endpoint], self._locks[endpoint]

    def send(self, endpoint: str, payload: dict) -> None:
        sock, lock = self._get(endpoint)
        with lock:
            sock.send(pickle.dumps(payload))

    def close(self) -> None:
        with self._global_lock:
            for sock in self._sockets.values():
                try:
                    sock.close(linger=0)
                except Exception:
                    pass
            self._sockets.clear()
            self._locks.clear()


class KVTransferManager:
    """Per-worker KV transfer manager. Binds a PULL on bootstrap_port for
    incoming messages and holds a pool of PUSH sockets for outgoing replies /
    requests. All handshake messages are keyed by `bootstrap_room`.

    Prefill worker usage (role="decode", i.e. the RDMA-WRITE sender):
        mgr = KVTransferManager(kv_cache, layout, role="decode",
                                bootstrap_port=19876)
        mgr.serve_and_send_kv(src_block_ids, bootstrap_room)

    Decode worker usage (role="prefill", i.e. the RDMA-WRITE receiver):
        mgr = KVTransferManager(kv_cache, layout, role="prefill")
        mgr.connect_and_recv_kv(prefill_host, prefill_port,
                                dst_block_ids, bootstrap_room)

    The `role` names are inherited from the legacy API and are the inverse of
    the worker's disagg_mode; see init_kv_transfer in model_runner.py.
    """

    def __init__(self, kv_cache, layout, role, bootstrap_port=0, host="",
                 kv_scale=None):
        from atom.model_engine.kv_transfer import KVTransferOp

        if role not in ("prefill", "decode"):
            raise ValueError(f"role must be 'prefill' or 'decode', got {role!r}")
        self.role = role
        self.layout = layout
        self.transfer_op = KVTransferOp(
            kv_cache, layout, host=host or _get_local_ip(), port=0,
            kv_scale=kv_scale,
        )
        self._puller = _Puller(port=bootstrap_port)
        self._pusher = _Pusher()

    @property
    def bootstrap_host(self) -> str:
        return self._puller.host

    @property
    def bootstrap_port(self) -> int:
        return self._puller.port

    def start_accept_loop(self) -> None:
        """No-op retained for backward compatibility. The puller thread starts
        at __init__ so there's nothing separate to spin up."""
        return

    # ------------------------------------------------------------------
    # Prefill worker (role="decode"): receive handshake → reply → RDMA-WRITE
    # ------------------------------------------------------------------

    def serve_and_send_kv(
        self,
        src_block_ids: list[int],
        bootstrap_room: int,
        timeout_ms: int = 120000,
    ) -> None:
        assert self.role == "decode", (
            "serve_and_send_kv is the prefill-worker (role='decode') path"
        )

        req = self._puller.pop(bootstrap_room, timeout_s=timeout_ms / 1000.0)
        if req is None:
            raise RuntimeError(
                f"KV bootstrap: no decode handshake for room={bootstrap_room}"
            )

        peer_key = self.transfer_op.register_peer(
            req["engine_desc"], req["mem_desc"],
            peer_scale_mem_packed=req.get("scale_mem_desc"),
        )
        dst_block_ids = req["block_ids"]
        reply_endpoint = req["reply_endpoint"]

        reply = {
            "kind": "reply",
            "bootstrap_room": bootstrap_room,
            "engine_desc": self.transfer_op.engine_desc_packed,
            "mem_desc": self.transfer_op.local_mem_packed,
            "scale_mem_desc": self.transfer_op.local_scale_mem_packed,
            "block_ids": src_block_ids,
        }
        self._pusher.send(reply_endpoint, reply)

        self.transfer_op.send_blocks(
            peer_key, src_block_ids, dst_block_ids, bootstrap_room
        )
        logger.info(
            f"KV transfer: sent {len(src_block_ids)} blocks to {reply_endpoint}, "
            f"room={bootstrap_room}"
        )

    # ------------------------------------------------------------------
    # Decode worker (role="prefill"): send handshake → wait reply → wait RDMA
    # ------------------------------------------------------------------

    def connect_and_recv_kv(
        self,
        prefill_host: str,
        prefill_port: int,
        dst_block_ids: list[int],
        bootstrap_room: int,
        timeout_ms: int = 120000,
    ) -> dict:
        assert self.role == "prefill", (
            "connect_and_recv_kv is the decode-worker (role='prefill') path"
        )

        prefill_endpoint = f"tcp://{prefill_host}:{prefill_port}"
        reply_endpoint = f"tcp://{self._puller.host}:{self._puller.port}"

        request = {
            "kind": "request",
            "bootstrap_room": bootstrap_room,
            "engine_desc": self.transfer_op.engine_desc_packed,
            "mem_desc": self.transfer_op.local_mem_packed,
            "scale_mem_desc": self.transfer_op.local_scale_mem_packed,
            "block_ids": dst_block_ids,
            "reply_endpoint": reply_endpoint,
        }
        self._pusher.send(prefill_endpoint, request)
        logger.info(
            f"KV bootstrap: sent request to {prefill_endpoint}, "
            f"room={bootstrap_room}, reply_endpoint={reply_endpoint}"
        )

        reply = self._puller.pop(bootstrap_room, timeout_s=timeout_ms / 1000.0)
        if reply is None:
            raise RuntimeError(
                f"KV bootstrap: no prefill reply for room={bootstrap_room} "
                f"after {timeout_ms/1000:.0f}s"
            )

        peer_key = self.transfer_op.register_peer(
            reply["engine_desc"], reply["mem_desc"],
            peer_scale_mem_packed=reply.get("scale_mem_desc"),
        )

        self._wait_transfer(peer_key, bootstrap_room, timeout_ms)
        logger.info(
            f"KV transfer received: {len(dst_block_ids)} blocks, "
            f"room={bootstrap_room}"
        )
        return {"dst_block_ids": dst_block_ids, "bootstrap_room": bootstrap_room}

    # ------------------------------------------------------------------
    # Split-phase recv (used by DP>1 in engine_core): phase-1 accept +
    # phase-2 finalize after EngineCore allocates blocks. With PUSH/PULL the
    # "accept" phase doesn't truly exist, so we just buffer the request and
    # let the caller allocate blocks, then send the actual handshake in
    # recv_kv_finalize.
    # ------------------------------------------------------------------

    def recv_kv_no_alloc(self, timeout_ms: int = 120000) -> dict:
        """Decode worker: poll for ANY buffered request (any room).
        Returns {pending, num_blocks, bootstrap_room} so the caller can
        allocate blocks and then call recv_kv_finalize.

        Note: with PUSH/PULL there is no "accept" step — this simply blocks
        on the puller until something arrives. Retained for DP>1 parity.
        """
        assert self.role == "prefill", "recv_kv_no_alloc is decode-side"
        # This path is only used by DP>1 where non-zero ranks don't own the
        # bootstrap_room ahead of time. We peek at any buffered message.
        import time
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._puller._cv:
            while True:
                # Grab any available message
                if self._puller._by_room:
                    room, msg = next(iter(self._puller._by_room.items()))
                    del self._puller._by_room[room]
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("KV bootstrap: no request received")
                self._puller._cv.wait(timeout=min(5.0, remaining))
        return {
            "pending": msg,
            "num_blocks": len(msg["block_ids"]),
            "bootstrap_room": msg["bootstrap_room"],
        }

    def recv_kv_finalize(self, pending: dict, dst_block_ids: list[int],
                         timeout_ms: int = 120000) -> dict:
        """Finalize recv_kv_no_alloc: send our reply and wait for RDMA."""
        assert self.role == "prefill", "recv_kv_finalize is decode-side"

        # pending is a request dict that arrived through the puller
        peer_key = self.transfer_op.register_peer(
            pending["engine_desc"], pending["mem_desc"],
            peer_scale_mem_packed=pending.get("scale_mem_desc"),
        )
        # With split protocol, the "prefill worker" is actually the one
        # waiting on our reply; but in DP>1 engine_core.py's path, non-zero
        # ranks receive from their corresponding prefill rank. The reply
        # semantics are unchanged: we send our desc+block_ids, then wait.
        reply = {
            "kind": "reply",
            "bootstrap_room": pending["bootstrap_room"],
            "engine_desc": self.transfer_op.engine_desc_packed,
            "mem_desc": self.transfer_op.local_mem_packed,
            "scale_mem_desc": self.transfer_op.local_scale_mem_packed,
            "block_ids": dst_block_ids,
        }
        self._pusher.send(pending["reply_endpoint"], reply)
        self._wait_transfer(peer_key, pending["bootstrap_room"], timeout_ms)
        return {
            "dst_block_ids": dst_block_ids,
            "bootstrap_room": pending["bootstrap_room"],
        }

    # ------------------------------------------------------------------

    def _wait_transfer(self, peer_key: str, bootstrap_room: int,
                       timeout_ms: int) -> None:
        """Poll for RDMA transfer completion on the local MoRI engine."""
        import time
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            if self.transfer_op.wait_for_transfer(peer_key, bootstrap_room):
                return
            time.sleep(0.001)  # 1ms poll
        raise RuntimeError(
            f"KV transfer timed out after {timeout_ms}ms, "
            f"room={bootstrap_room}"
        )

    def close(self) -> None:
        try:
            self._puller.close()
        except Exception:
            pass
        try:
            self._pusher.close()
        except Exception:
            pass
        try:
            self.transfer_op.close()
        except Exception:
            pass
