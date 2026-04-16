# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Standalone test for KV block transfer via MoRI IOEngine RDMA.

Run with 2 GPUs:
    python tests/test_kv_transfer.py

Validates:
1. KVCacheLayout offset computation (unit, no GPU)
2. KVTransferOp register + send_blocks on real GPU memory (2 GPUs)
3. Full KVTransferManager prefill→decode roundtrip (2 processes)
"""

import logging
import multiprocessing as mp
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger("test_kv_transfer")


def test_layout_offsets():
    """Test KVCacheLayout byte offset computation (no GPU needed)."""
    from atom.model_engine.kv_transfer import KVCacheLayout

    # MLA layout: [num_layers, num_blocks, block_size, 576]
    layout = KVCacheLayout(
        use_mla=True, num_layers=61, num_blocks=1024, block_size=1,
        element_size=1, latent_dim=576,
    )
    assert layout.block_bytes == 1 * 576 * 1  # 576
    assert layout.layer_bytes == 1024 * 576
    assert layout.block_offset(0, 0) == 0
    assert layout.block_offset(0, 1) == 576
    assert layout.block_offset(1, 0) == 1024 * 576
    assert layout.block_offset(1, 5) == 1024 * 576 + 5 * 576

    # Non-MLA layout: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    layout2 = KVCacheLayout(
        use_mla=False, num_layers=32, num_blocks=512, block_size=16,
        element_size=2, num_kv_heads=8, head_dim=128,
    )
    assert layout2.block_bytes == 16 * 8 * 128 * 2  # 32768
    assert layout2.layer_bytes == 512 * 32768
    # K at block (layer=0, block=0) offset = 0
    assert layout2.block_offset(0, 0) == 0
    # V stride (dim 0) = num_layers * layer_bytes
    assert layout2.kv_pair_stride == 32 * 512 * 32768

    logger.info("PASS: test_layout_offsets")


def test_transfer_op_2gpu():
    """Test KVTransferOp with 2 GPUs on same node (requires 2+ GPUs)."""
    if torch.cuda.device_count() < 2:
        logger.warning("SKIP: test_transfer_op_2gpu requires 2 GPUs")
        return

    try:
        from atom.model_engine.kv_transfer import KVCacheLayout, KVTransferOp
    except ImportError as e:
        logger.warning(f"SKIP: MoRI not available: {e}")
        return

    num_layers = 4
    num_blocks = 8
    block_size = 2
    latent_dim = 16  # small for test
    dtype = torch.float16

    layout = KVCacheLayout(
        use_mla=True, num_layers=num_layers, num_blocks=num_blocks,
        block_size=block_size, element_size=dtype.itemsize, latent_dim=latent_dim,
    )

    # Allocate KV caches on 2 GPUs
    shape = (num_layers, num_blocks, block_size, latent_dim)
    src_kv = torch.randn(shape, dtype=dtype, device="cuda:0")
    dst_kv = torch.zeros(shape, dtype=dtype, device="cuda:1")

    # Fill source blocks 0, 2, 5 with known patterns
    src_block_ids = [0, 2, 5]
    for layer in range(num_layers):
        for bid in src_block_ids:
            src_kv[layer, bid] = float(layer * 100 + bid)

    # Create transfer ops
    prefill_op = KVTransferOp(src_kv, layout, host="127.0.0.1", port=0)
    decode_op = KVTransferOp(dst_kv, layout, host="127.0.0.1", port=0)

    # Exchange descriptors
    peer_key = prefill_op.register_peer(
        decode_op.engine_desc_packed, decode_op.local_mem_packed
    )

    # Transfer blocks 0,2,5 → dst blocks 1,3,6
    dst_block_ids = [1, 3, 6]
    prefill_op.send_blocks(peer_key, src_block_ids, dst_block_ids, transfer_uid=42)

    # Verify
    torch.cuda.synchronize()
    for layer in range(num_layers):
        for src_bid, dst_bid in zip(src_block_ids, dst_block_ids):
            expected = src_kv[layer, src_bid].cpu()
            actual = dst_kv[layer, dst_bid].cpu()
            if not torch.allclose(expected, actual):
                logger.error(
                    f"MISMATCH layer={layer} src_block={src_bid} dst_block={dst_bid}"
                )
                logger.error(f"  expected: {expected}")
                logger.error(f"  actual:   {actual}")
                raise AssertionError("KV transfer data mismatch")

    prefill_op.close()
    decode_op.close()
    logger.info("PASS: test_transfer_op_2gpu")


def _worker_process(role, port, result_queue, shape, dtype_str, src_block_ids):
    """Worker for multiprocess KVTransferManager test."""
    import torch
    from atom.model_engine.kv_transfer import KVCacheLayout
    from atom.model_engine.kv_bootstrap import KVTransferManager

    num_layers, num_blocks, block_size, latent_dim = shape
    dtype = getattr(torch, dtype_str)

    layout = KVCacheLayout(
        use_mla=True, num_layers=num_layers, num_blocks=num_blocks,
        block_size=block_size, element_size=dtype.itemsize, latent_dim=latent_dim,
    )

    if role == "decode":
        gpu_id = 1
        torch.cuda.set_device(gpu_id)
        kv_cache = torch.zeros(shape, dtype=dtype, device=f"cuda:{gpu_id}")
        mgr = KVTransferManager(kv_cache, layout, role="decode", bootstrap_port=port)
        result_queue.put(("decode_ready", mgr.bootstrap_host, mgr.bootstrap_port))

        # Fake block manager for testing
        class FakeBlockManager:
            def __init__(self, num_blocks):
                self._free = list(range(num_blocks))
            def reserve_blocks(self, n):
                reserved = self._free[:n]
                self._free = self._free[n:]
                return reserved

        bm = FakeBlockManager(num_blocks)
        result = mgr.recv_kv(bm, timeout_ms=30000)
        # Read transferred data and check
        dst_ids = result["dst_block_ids"]
        data = {}
        for layer in range(num_layers):
            for dst_bid in dst_ids:
                data[(layer, dst_bid)] = kv_cache[layer, dst_bid].cpu().tolist()
        result_queue.put(("decode_done", dst_ids, data))
        mgr.close()

    elif role == "prefill":
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        kv_cache = torch.randn(shape, dtype=dtype, device=f"cuda:{gpu_id}")
        # Fill known pattern
        for layer in range(num_layers):
            for bid in src_block_ids:
                kv_cache[layer, bid] = float(layer * 100 + bid)

        mgr = KVTransferManager(kv_cache, layout, role="prefill")

        # Wait for decode to be ready
        msg = result_queue.get(timeout=15)
        assert msg[0] == "decode_ready"
        decode_host, decode_port = msg[1], msg[2]

        mgr.send_kv(src_block_ids, decode_host, decode_port, bootstrap_room=12345)
        # Send expected data for verification
        expected = {}
        for layer in range(num_layers):
            for bid in src_block_ids:
                expected[(layer, bid)] = kv_cache[layer, bid].cpu().tolist()
        result_queue.put(("prefill_done", expected))
        mgr.close()


def test_manager_multiprocess():
    """Full prefill→decode KV transfer test with 2 processes."""
    if torch.cuda.device_count() < 2:
        logger.warning("SKIP: test_manager_multiprocess requires 2 GPUs")
        return

    try:
        from atom.model_engine.kv_transfer import KVCacheLayout  # noqa: F401
    except ImportError as e:
        logger.warning(f"SKIP: MoRI not available: {e}")
        return

    shape = (4, 8, 2, 16)  # (layers, blocks, block_size, latent_dim)
    src_block_ids = [0, 2, 5]
    port = 19876

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    decode_proc = ctx.Process(
        target=_worker_process,
        args=("decode", port, q, shape, "float16", src_block_ids),
    )
    prefill_proc = ctx.Process(
        target=_worker_process,
        args=("prefill", port, q, shape, "float16", src_block_ids),
    )

    decode_proc.start()
    prefill_proc.start()

    # Collect results
    results = {}
    for _ in range(3):  # decode_ready, decode_done, prefill_done
        try:
            msg = q.get(timeout=60)
            results[msg[0]] = msg[1:]
        except Exception as e:
            logger.error(f"Timeout waiting for worker result: {e}")
            break

    prefill_proc.join(timeout=10)
    decode_proc.join(timeout=10)

    if "prefill_done" not in results or "decode_done" not in results:
        logger.error(f"Missing results: got keys {list(results.keys())}")
        raise AssertionError("Multiprocess test incomplete")

    expected_data = results["prefill_done"][0]
    dst_ids = results["decode_done"][0]
    actual_data = results["decode_done"][1]

    # Verify: src_block_ids → dst_block_ids mapping
    for layer in range(shape[0]):
        for src_bid, dst_bid in zip(src_block_ids, dst_ids):
            expected = expected_data[(layer, src_bid)]
            actual = actual_data[(layer, dst_bid)]
            if expected != actual:
                logger.error(
                    f"MISMATCH layer={layer} src={src_bid} dst={dst_bid}"
                )
                raise AssertionError("Multiprocess KV transfer data mismatch")

    logger.info("PASS: test_manager_multiprocess")


if __name__ == "__main__":
    test_layout_offsets()
    test_transfer_op_2gpu()
    test_manager_multiprocess()
    logger.info("ALL TESTS PASSED")
