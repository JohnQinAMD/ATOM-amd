# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
KV cache block transfer for disaggregated serving via MoRI IOEngine RDMA.

Prefill worker writes populated KV blocks to decode worker's GPU memory
in a single batched RDMA operation. No kernel development — uses MoRI's
existing IOEngine read/write primitives over ibverbs (ionic/mlx5/bnxt).

Usage:
    # Prefill side
    mgr = KVTransferManager(kv_cache_tensor, kv_cache_info, role="prefill",
                            bootstrap_port=12345)
    mgr.start_bootstrap()
    # ... after prefill completes ...
    mgr.send_blocks(src_block_ids, dst_block_ids, peer_engine_key, room_id)

    # Decode side
    mgr = KVTransferManager(kv_cache_tensor, kv_cache_info, role="decode",
                            bootstrap_port=12346)
    mgr.connect_to_prefill(prefill_host, prefill_port)
    dst_ids = mgr.recv_blocks(num_blocks, room_id)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger("atom")

try:
    from mori import cpp as mori_cpp
    from mori.io.engine import IOEngine

    MORI_IO_AVAILABLE = True
except ImportError:
    MORI_IO_AVAILABLE = False


@dataclass
class KVCacheLayout:
    """Describes the memory layout of atom's KV cache tensor for offset computation."""

    use_mla: bool
    num_layers: int
    num_blocks: int
    block_size: int  # tokens per block
    # MLA: kv_cache shape [num_layers, num_blocks, block_size, latent_dim]
    # Non-MLA: kv_cache shape [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    element_size: int  # bytes per element
    # For MLA
    latent_dim: int = 576
    # For non-MLA
    num_kv_heads: int = 0
    head_dim: int = 0

    @property
    def block_bytes(self) -> int:
        """Bytes per single (layer, block) slot."""
        if self.use_mla:
            return self.block_size * self.latent_dim * self.element_size
        else:
            return self.block_size * self.num_kv_heads * self.head_dim * self.element_size

    @property
    def layer_bytes(self) -> int:
        """Bytes per layer (all blocks)."""
        return self.num_blocks * self.block_bytes

    def block_offset(self, layer_idx: int, block_id: int) -> int:
        """Byte offset of a specific (layer, block) in the kv_cache tensor."""
        if self.use_mla:
            # [num_layers, num_blocks, block_size, latent_dim]
            return layer_idx * self.layer_bytes + block_id * self.block_bytes
        else:
            # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
            # K and V are in dim 0 — both need to be transferred
            # Return offset of K; caller adds k_v_stride for V
            return layer_idx * self.layer_bytes + block_id * self.block_bytes

    @property
    def kv_pair_stride(self) -> int:
        """For non-MLA: byte offset between K cache and V cache (dim 0 stride)."""
        if self.use_mla:
            return 0  # MLA has K+V fused
        return self.num_layers * self.layer_bytes


class KVTransferOp:
    """Point-to-point KV block transfer using MoRI IOEngine RDMA.

    Operates on a single registered KV cache tensor. Computes byte offsets
    per (layer, block_id) pair and issues batched RDMA writes.

    Both prefill and decode workers hold a KVTransferOp. The prefill side
    calls send_blocks(); the decode side waits for completion.
    """

    def __init__(
        self,
        kv_cache: torch.Tensor,
        layout: KVCacheLayout,
        host: str = "",
        port: int = 0,
    ):
        if not MORI_IO_AVAILABLE:
            raise ImportError(
                "mori.io is required for KV transfer but not installed. "
                "Install mori with IO support enabled."
            )

        self.layout = layout
        self.kv_cache = kv_cache

        config = mori_cpp.IOEngineConfig(host=host, port=port)
        self._engine = IOEngine(f"kv-transfer-{id(self)}", config)
        self._engine.create_backend(mori_cpp.BackendType.RDMA)

        # Register the entire KV cache tensor for RDMA
        self._local_mem = self._engine.register_torch_tensor(kv_cache)
        self._engine_desc = self._engine.get_engine_desc()

        self._sessions: dict[str, "mori_cpp.IOEngineSession"] = {}
        self._remote_mems: dict[str, "mori_cpp.MemoryDesc"] = {}

        logger.info(
            f"KVTransferOp initialized: {kv_cache.shape}, "
            f"{kv_cache.nbytes / 1e9:.2f} GB, engine={self._engine_desc.key}"
        )

    @property
    def engine_desc_packed(self) -> bytes:
        return self._engine_desc.pack()

    @property
    def local_mem_packed(self) -> bytes:
        return self._local_mem.pack()

    def register_peer(
        self, peer_engine_desc_packed: bytes, peer_mem_desc_packed: bytes
    ) -> str:
        """Register a remote peer and create a session for KV transfer."""
        peer_desc = mori_cpp.EngineDesc.unpack(peer_engine_desc_packed)
        peer_mem = mori_cpp.MemoryDesc.unpack(peer_mem_desc_packed)

        self._engine.register_remote_engine(peer_desc)
        session = self._engine.create_session(self._local_mem, peer_mem)
        if session is None:
            raise RuntimeError(
                f"Failed to create RDMA session with peer {peer_desc.key}"
            )

        peer_key = peer_desc.key
        self._sessions[peer_key] = session
        self._remote_mems[peer_key] = peer_mem
        logger.info(f"Registered peer {peer_key} for KV transfer")
        return peer_key

    def send_blocks(
        self,
        peer_key: str,
        src_block_ids: list[int],
        dst_block_ids: list[int],
        transfer_uid: int,
    ) -> None:
        """RDMA WRITE KV blocks from local GPU to remote GPU.

        For each layer, writes the KV data at src_block_ids to the
        corresponding dst_block_ids on the peer.
        """
        session = self._sessions[peer_key]
        layout = self.layout
        num_blocks = len(src_block_ids)

        local_offsets = []
        remote_offsets = []
        sizes = []

        for layer_idx in range(layout.num_layers):
            for i in range(num_blocks):
                src_off = layout.block_offset(layer_idx, src_block_ids[i])
                dst_off = layout.block_offset(layer_idx, dst_block_ids[i])
                local_offsets.append(src_off)
                remote_offsets.append(dst_off)
                sizes.append(layout.block_bytes)

                if not layout.use_mla:
                    # Non-MLA: also transfer V cache
                    v_stride = layout.kv_pair_stride
                    local_offsets.append(src_off + v_stride)
                    remote_offsets.append(dst_off + v_stride)
                    sizes.append(layout.block_bytes)

        logger.info(
            f"Sending {num_blocks} blocks × {layout.num_layers} layers "
            f"({len(sizes)} RDMA writes, "
            f"{sum(sizes) / 1e6:.1f} MB total) uid={transfer_uid}"
        )

        statuses = session.batch_write(
            local_offsets, remote_offsets, sizes, transfer_uid
        )

        # Wait for all writes to complete
        for i, status in enumerate(statuses):
            status.Wait()
            if status.Failed():
                raise RuntimeError(
                    f"KV transfer write {i} failed: {status.Message()}"
                )

        logger.info(f"KV transfer send complete: uid={transfer_uid}")

    def wait_for_transfer(
        self, peer_key: str, transfer_uid: int, timeout_ms: int = 30000
    ) -> bool:
        """Wait for an inbound KV transfer to complete (decode side)."""
        status = self._engine.pop_inbound_transfer_status(peer_key, transfer_uid)
        if status is not None:
            return status.Succeeded()
        # Not yet arrived — this is a simple poll; callers should loop
        return False

    def close(self):
        """Deregister memory and clean up RDMA resources."""
        try:
            self._engine.deregister_memory(self._local_mem)
        except Exception as e:
            logger.warning(f"Failed to deregister KV cache memory: {e}")
        self._sessions.clear()
        self._remote_mems.clear()
        logger.info("KVTransferOp closed")
