# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import logging
import pickle
import queue
import threading
import time
from contextlib import ExitStack
from typing import List

import torch
import zmq
from atom.config import Config, ParallelConfig
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence
from atom.utils import init_exit_handler, make_zmq_socket
from atom.utils.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)

logger = logging.getLogger("atom")


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used within EngineCore.
    SHUTDOWN = b"\x05"
    # Stream output for callbacks
    STREAM = b"\x06"
    # Signal that EngineCore is fully initialized and ready
    READY = b"\x07"


class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[List[Sequence]]()
        self.stream_output_queue = (
            queue.Queue()
        )  # Queue for streaming intermediate outputs
        self.input_address = input_address
        self.output_address = output_address
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()
        self.input_thread = threading.Thread(
            target=self.process_input_sockets, args=(self.input_address,), daemon=True
        )
        self.input_thread.start()

        self.profile_enbaled = config.torch_profiler_dir is not None
        self.mark_trace = getattr(config, "mark_trace", False)
        init_exit_handler(self)
        self._init_data_parallel(config)

        # Initialize model runner processes
        try:
            good = False
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tensor_parallel_size,
                "atom.model_engine.model_runner.ModelRunner",
                config,
            )
            block_info = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            num_blocks = block_info["num_kvcache_blocks"]
            config.mamba_equiv_per_req = block_info.get("mamba_equiv_per_req", 0)
            config.num_mamba_groups = block_info.get("num_mamba_groups", 0)
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            config.num_kvcache_blocks = num_blocks
            self.kv_cache_info = self.runner_mgr.call_func(
                "get_kv_cache_info", wait_out=True
            )
            if not config.enforce_eager:
                # Start profiler before cudagraph capture only if mark-trace is enabled.
                if self.profile_enbaled and self.mark_trace:
                    self.runner_mgr.call_func(
                        "start_profiler", "capture_graph", wait_out=True
                    )
                cap_cost, bs = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{bs} cost: {cap_cost:.2f} seconds"
                )
                if self.profile_enbaled and self.mark_trace:
                    # Persist a dedicated capture-graph trace immediately.
                    self.runner_mgr.call_func("stop_profiler", wait_out=True)
            good = True
        finally:
            logger.info(
                f"{self.label}: load model runner {'success' if good else 'failed'}"
            )
            if not good:
                self._finalizer()

        self.scheduler = Scheduler(config)

        # Disaggregated KV transfer — initialized inside ModelRunner subprocess
        # because the kv_cache tensor can't be pickled across process boundaries.
        # We use runner_mgr.call_func("init_kv_transfer") to create the
        # KVTransferManager in ModelRunner, which has direct access to the tensor.
        self._disagg_mode = config.disagg_mode
        self._kv_transfer_bootstrap = None
        self._last_dp_kv_recv_info = None  # Track KV recv for DP broadcast
        if config.disagg_mode in ("prefill", "decode"):
            bootstrap_result = self.runner_mgr.call_func(
                "init_kv_transfer",
                config.disagg_mode,
                config.disagg_bootstrap_port,
                wait_out=True,
            )
            if bootstrap_result:
                self._kv_transfer_bootstrap = bootstrap_result
                logger.info(
                    f"{self.label}: KV transfer initialized in ModelRunner: "
                    f"mode={config.disagg_mode}, "
                    f"bootstrap={bootstrap_result.get('host')}:{bootstrap_result.get('port')}"
                )
            else:
                logger.warning(
                    f"{self.label}: KV transfer init failed (MoRI unavailable?)"
                )

        # Start input thread AFTER model is loaded so the "ready" signal
        # is sent only when the engine is truly ready to accept requests
        # self.input_thread = threading.Thread(
        #     target=self.process_input_sockets, args=(self.input_address,), daemon=True
        # )
        # self.input_thread.start()

        # We can not start input thread here since dp need to sync with other ranks,
        # Otherwise, DP will hang always.
        # Thus we add new signal READY to notify CoreManager

        self._send_ready_signal()
        logger.info(f"{self.label}: EngineCore fully initialized and ready")

    def _send_ready_signal(self):
        # Include kv_cache_info so CoreManager can expose it to LLMEngine
        ready_data = {
            "kv_cache_info": getattr(self, "kv_cache_info", {}),
            "kv_transfer_bootstrap": self._kv_transfer_bootstrap,
        }
        self.output_queue.put_nowait(("READY", ready_data))

    def _init_data_parallel(self, config: Config):
        pass

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self.runner_mgr.keep_monitoring = False
        try:
            self.runner_mgr.call_func("exit")
        except Exception:
            pass  # shared memory may already be freed
        for proc in self.runner_mgr.procs:
            try:
                alive = proc.is_alive()
            except ValueError:
                continue  # process object already closed by CoreManager
            if alive:
                proc.join(timeout=5)
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait([get_exit_sequence()])
        self.output_thread.join(timeout=0.5)

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine: EngineCore = None
        try:
            if config.parallel_config.data_parallel_size > 1:
                engine = DPEngineCoreProc(config, input_address, output_address)
            else:
                engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception as e:
            logger.error(f"run_engine: exception: {e}", exc_info=True)
            raise e
        finally:
            if engine is not None:
                engine.exit()

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self.pull_and_process_input_queue()
            if shutdown:
                break
            if not self.scheduler.is_finished():
                self._process_engine_step()

    def _process_engine_step(self):
        if not self.scheduler.has_requests():
            return False
        scheduled_batch, seqs = self.scheduler.schedule()
        # if scheduled_batch is None:
        #     return False
        fwd_out = self.runner_mgr.call_func("forward", scheduled_batch, wait_out=True)
        seqs = seqs.values()
        # Pass stream_output_queue to postprocess for streaming callbacks
        finished_seqs = self.scheduler.postprocess(
            seqs, fwd_out, stream_output_queue=self.stream_output_queue
        )

        # Send stream outputs to main process via output_queue
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                # Send stream outputs as intermediate results
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass

        # Disagg: send KV for prefill-only sequences after postprocess.
        # Each rank independently sends its own KV slice. For DP (multi-rank),
        # the bootstrap_info is broadcast from rank 0. The prefill rank N
        # connects to decode rank N at port=base_port+N.
        if self._disagg_mode == "prefill" and finished_seqs:
            for seq in finished_seqs:
                if seq.disagg_mode == "prefill_only" and seq.disagg_bootstrap_info:
                    try:
                        binfo = seq.disagg_bootstrap_info
                        # For multi-rank: adjust port for this rank
                        decode_port = binfo["decode_port"]
                        if hasattr(self, "dp_rank") and self.dp_rank > 0:
                            decode_port += self.dp_rank
                        success = self.runner_mgr.call_func(
                            "kv_send",
                            list(seq.block_table),
                            binfo["decode_host"],
                            decode_port,
                            binfo["bootstrap_room"],
                            wait_out=True,
                        )
                        if success:
                            logger.info(
                                f"KV transfer sent for seq {seq.id}: "
                                f"{len(seq.block_table)} blocks, "
                                f"room={binfo['bootstrap_room']}"
                            )
                        else:
                            logger.error(
                                f"KV transfer send returned False for seq {seq.id}"
                            )
                    except Exception as e:
                        logger.error(f"KV transfer failed for seq {seq.id}: {e}")
                    finally:
                        # Deallocate blocks after transfer completes
                        self.scheduler.deallocate_seq(seq)

        if finished_seqs:
            self.output_queue.put_nowait(finished_seqs)
        return True

    def pull_and_process_input_queue(self):
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    logger.debug(f"{self.label}: input_queue get exit engine")
                    return True
                # Disagg decode: receive KV blocks before scheduling.
                # Uses split protocol: ModelRunner accepts the prefill
                # connection and exchanges RDMA descriptors, then EngineCore
                # allocates blocks (it owns the block_manager), then
                # ModelRunner finalizes the RDMA transfer.
                if (
                    seq.disagg_mode == "decode_only"
                    and self._disagg_mode == "decode"
                    and seq.disagg_bootstrap_info is not None
                ):
                    try:
                        binfo = seq.disagg_bootstrap_info
                        prefill_host = binfo.get("bootstrap_host", "")
                        prefill_port = binfo.get("bootstrap_port", 0)
                        bootstrap_room = binfo.get("bootstrap_room", 0)

                        # Calculate blocks needed for this prompt
                        block_size = self.scheduler.block_manager.block_size
                        num_blocks = (seq.num_prompt_tokens + block_size - 1) // block_size

                        # Allocate blocks with MTP headroom
                        mtp_headroom = self.scheduler.mtp_k + 1
                        dst_block_ids = self.scheduler.block_manager.reserve_blocks_with_headroom(
                            num_blocks, headroom=mtp_headroom,
                        )

                        # Decode connects to prefill's bootstrap server to receive KV
                        recv_result = self.runner_mgr.call_func(
                            "kv_recv",
                            prefill_host, prefill_port,
                            dst_block_ids, bootstrap_room,
                            wait_out=True,
                        )
                        if not recv_result:
                            raise RuntimeError("kv_recv failed")

                        self.scheduler.block_manager.allocate_specific_blocks(
                            seq, dst_block_ids
                        )
                        seq.num_cached_tokens = seq.num_prompt_tokens
                        # Track for DP broadcast (non-zero ranks need to know)
                        self._last_dp_kv_recv_info = {
                            "num_blocks": num_blocks,
                            "seq_id": seq.id,
                        }
                        logger.info(
                            f"KV transfer received for seq {seq.id}: "
                            f"{len(dst_block_ids)} blocks"
                        )
                    except Exception as e:
                        logger.error(
                            f"KV transfer recv failed for seq {seq.id}: {e}"
                        )
                        seq.status = SequenceStatus.FINISHED
                        seq.leave_reason = "kv_transfer_failed"
                        continue
                recv_reqs.append(seq)
        if len(recv_reqs) > 0:
            logger.info(f"{self.label}: put {len(recv_reqs)} reqs to scheduler")
            self.scheduler.extend(recv_reqs)
        return False

    def process_input_sockets(self, input_address: str):
        """Input socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")
            alive = True

            while alive:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    obj = input_socket.recv(copy=False)
                    request_type, reqs = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.ADD:
                        req_ids = [req.id for req in reqs]
                        logger.debug(
                            f"{self.label}: input get {request_type} {req_ids}"
                        )
                        self.input_queue.put_nowait(reqs)
                    elif request_type == EngineCoreRequestType.UTILITY:
                        # Handle utility commands like start_profile/stop_profile
                        cmd = reqs.get("cmd") if isinstance(reqs, dict) else None
                        logger.debug(f"{self.label}: input get UTILITY command: {cmd}")
                        if cmd == "start_profile":
                            self.start_profiler()
                        elif cmd == "stop_profile":
                            self.stop_profiler()
                        elif cmd == "get_mtp_stats":
                            self.print_mtp_statistics()
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(f"{self.label}: input get {request_type}")
                        self.input_queue.put_nowait([get_exit_sequence()])
                        alive = False
                        reason = request_type
            logger.debug(f"{self.label}: input thread exit due to {reason}")

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                item = self.output_queue.get()
                if isinstance(item, tuple) and item[0] == "STREAM":
                    # Send stream outputs
                    stream_outputs = item[1]
                    obj = pickle.dumps((EngineCoreRequestType.STREAM, stream_outputs))
                    socket.send(obj)
                    continue

                if isinstance(item, tuple) and item[0] == "READY":
                    # Send READY signal with kv_cache_info and transfer bootstrap
                    obj = pickle.dumps((EngineCoreRequestType.READY, item[1]))
                    socket.send(obj)
                    logger.debug(f"{self.label}: sent READY signal")
                    continue

                # Regular finished sequences
                seqs = item
                valid_seqs = [
                    seq for seq in seqs if seq.status != SequenceStatus.EXIT_ENGINE
                ]
                num_valid = len(valid_seqs)
                if num_valid > 0:
                    obj = pickle.dumps((EngineCoreRequestType.ADD, valid_seqs))
                    socket.send(obj)
                    logger.info(f"{self.label}: output send {num_valid} reqs")
                if len(valid_seqs) != len(seqs):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.debug(
                        f"{self.label}: output send {EngineCoreRequestType.SHUTDOWN}"
                    )
                    break

    def start_profiler(self):
        if self.profile_enbaled:
            self.runner_mgr.call_func("start_profiler", wait_out=True)

    def stop_profiler(self):
        if self.profile_enbaled:
            logger.info("Profiler stopping...")
            t0 = time.monotonic()
            self.runner_mgr.call_func("stop_profiler", wait_out=True)
            logger.info("Profiler stopped in %.1fs", time.monotonic() - t0)

    def print_mtp_statistics(self):
        if self.scheduler.spec_stats is not None:
            self.scheduler.spec_stats._log()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available (MTP not enabled or no tokens processed)\n"
            )


class DPEngineCoreProc(EngineCore):
    def __init__(self, config: Config, input_address: str, output_address: str):
        # self.dp_group = config.parallel_config.dp_group
        self.dp_rank = config.parallel_config.data_parallel_rank
        # self.dp_group = config.parallel_config.stateless_init_dp_group()
        super().__init__(config, input_address, output_address)
        # Initialize to True so first iteration reaches all_reduce
        self.engines_running = True
        self._shutting_down = False
        self._mtp_enabled = config.speculative_config is not None

    def _init_data_parallel(self, config: Config):
        dp_rank = config.parallel_config.data_parallel_rank
        dp_size = config.parallel_config.data_parallel_size
        local_dp_rank = config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_group = config.parallel_config.stateless_init_dp_group()

    def exit(self):
        super().exit()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self.pull_and_process_input_queue()

            local_is_prefill, local_num_tokens, local_num_reqs = (
                self.scheduler.get_next_batch_info()
            )
            local_unfinished = not self.scheduler.is_finished()

            (
                global_has_prefill,
                global_max_tokens,
                global_max_reqs,
                global_has_unfinished,
                global_shutdown,
            ) = self._sync_dp_state(
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_unfinished,
                shutdown,
            )

            if global_shutdown and not global_has_unfinished:
                logger.info(
                    f"{self.label}: All DP ranks agreed to shutdown, exiting busy_loop"
                )
                break

            if not global_has_unfinished and not self.engines_running:
                self.engines_running = False
                continue

            if global_has_prefill and not local_is_prefill:
                # We must do dummy prefill to sync here
                # Since we want to split mori output in moe, we need to make dp all run prefill or all run decode
                dummy_reqs = min(
                    global_max_reqs, 2
                )  # dummy reqs at 2: just enough for TBO agreement, avoid wasting compute.
                logger.info(
                    f"{self.label}: Running dummy prefill ({global_max_tokens} tokens, {dummy_reqs} reqs) "
                    f"to sync with other DP ranks doing prefill"
                )
                self._execute_dummy_prefill(global_max_tokens, dummy_reqs)
            else:
                executed = self._process_engine_step()
                if not executed:
                    self._execute_dummy_batch()
                elif self._mtp_enabled and not global_has_prefill:
                    # After decode step with MTP: verify all DP ranks have
                    # same batch size to prevent draft token divergence.
                    num_running = len(self.scheduler.running)
                    self._verify_mtp_batch_consistency(num_running)

            self.engines_running = global_has_unfinished

    def pull_and_process_input_queue(self):
        """Override for DP: broadcast disagg bootstrap_info from rank 0 to all ranks.

        Rank 0 receives requests from Dynamo via ZMQ. For decode-only sequences,
        rank 0's bootstrap_info is broadcast to all DP ranks so each rank
        independently receives its KV slice from the corresponding prefill rank.
        """
        # Only rank 0 actually receives requests from Dynamo
        if self.dp_rank == 0:
            result = super().pull_and_process_input_queue()
        else:
            result = False

        # For disagg decode: rank 0 already processed the recv in the parent
        # call. Non-zero ranks need to participate if rank 0 did a recv.
        if self._disagg_mode == "decode":
            # Broadcast whether rank 0 did a KV recv this iteration
            did_recv = [getattr(self, "_last_dp_kv_recv_info", None)]
            try:
                torch.distributed.broadcast_object_list(
                    did_recv, src=0, group=self.dp_group
                )
            except RuntimeError:
                return result

            recv_info = did_recv[0]
            if recv_info is not None and self.dp_rank != 0:
                # Non-zero ranks: do their own KV recv using rank-specific
                # bootstrap ports. The prefill side connects to port+rank.
                try:
                    accept_result = self.runner_mgr.call_func(
                        "kv_recv", 0, wait_out=True,
                    )
                    if accept_result and "pending" in accept_result:
                        pending = accept_result["pending"]
                        num_blocks = accept_result["num_blocks"]

                        mtp_headroom = self.scheduler.mtp_k + 1
                        dst_block_ids = self.scheduler.block_manager.reserve_blocks_with_headroom(
                            num_blocks, headroom=mtp_headroom,
                        )

                        self.runner_mgr.call_func(
                            "kv_recv_finalize",
                            pending, dst_block_ids,
                            wait_out=True,
                        )

                        # Create and schedule the decode-only sequence on this rank
                        # (mirroring what rank 0 did via the parent class)
                        for seq in list(self.scheduler.waiting) + list(self.scheduler.running):
                            if (
                                seq.disagg_mode == "decode_only"
                                and not hasattr(seq, "_dp_kv_received")
                            ):
                                self.scheduler.block_manager.allocate_specific_blocks(
                                    seq, dst_block_ids
                                )
                                seq.num_cached_tokens = seq.num_prompt_tokens
                                seq._dp_kv_received = True
                                logger.info(
                                    f"{self.label}: DP rank {self.dp_rank} KV recv "
                                    f"for seq {seq.id}: {len(dst_block_ids)} blocks"
                                )
                                break
                except Exception as e:
                    logger.error(
                        f"{self.label}: DP rank {self.dp_rank} KV recv failed: {e}"
                    )
            # Clear the flag for next iteration
            self._last_dp_kv_recv_info = None

        return result

    def _execute_dummy_batch(self):
        return self.runner_mgr.call_func("dummy_execution", wait_out=True)

    def _execute_dummy_prefill(self, num_tokens: int, num_reqs: int = 1):
        return self.runner_mgr.call_func(
            "dummy_prefill_execution", num_tokens, num_reqs, wait_out=True
        )

    def _sync_dp_state(
        self,
        local_is_prefill: bool,
        local_num_tokens: int,
        local_num_reqs: int,
        local_has_unfinished: bool,
        local_shutdown: bool = False,
    ) -> tuple[bool, int, int, bool, bool]:
        if self._shutting_down:
            return (
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_has_unfinished,
                True,
            )

        try:
            # Pack all state: [is_prefill, num_tokens, num_reqs, has_unfinished, shutdown]
            state_tensor = torch.tensor(
                [
                    1 if local_is_prefill else 0,
                    local_num_tokens,
                    local_num_reqs,
                    1 if local_has_unfinished else 0,
                    1 if local_shutdown else 0,
                ],
                dtype=torch.int64,
                device="cpu",
            )
            torch.distributed.all_reduce(
                state_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_has_prefill = state_tensor[0].item() == 1
            global_max_tokens = state_tensor[1].item()
            global_max_reqs = state_tensor[2].item()
            global_has_unfinished = state_tensor[3].item() == 1
            global_shutdown = state_tensor[4].item() == 1
            return (
                global_has_prefill,
                global_max_tokens,
                global_max_reqs,
                global_has_unfinished,
                global_shutdown,
            )
        except RuntimeError as e:
            logger.warning(f"{self.label}: _sync_dp_state failed: {e}")
            # If sync fails, assume shutdown to prevent hang
            self._shutting_down = True
            return (
                local_is_prefill,
                local_num_tokens,
                local_num_reqs,
                local_has_unfinished,
                True,
            )

    def _verify_mtp_batch_consistency(self, batch_size: int) -> None:
        """Defensive check: verify all DP ranks have same batch size for MTP.

        MTP draft generation runs autoregressive steps that must produce
        identical results across DP ranks (since MoRI dispatch/combine
        requires all ranks to participate in every MoE forward). If batch
        sizes diverge, draft tokens will be misaligned.
        """
        if self._shutting_down or not self._mtp_enabled:
            return
        try:
            tensor = torch.tensor([batch_size], dtype=torch.int64, device="cpu")
            # Use MIN to detect if any rank has fewer seqs
            min_tensor = tensor.clone()
            torch.distributed.all_reduce(
                min_tensor, op=torch.distributed.ReduceOp.MIN, group=self.dp_group
            )
            max_tensor = tensor.clone()
            torch.distributed.all_reduce(
                max_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            if min_tensor.item() != max_tensor.item():
                logger.warning(
                    f"{self.label}: MTP batch size mismatch across DP ranks: "
                    f"local={batch_size}, min={min_tensor.item()}, max={max_tensor.item()}. "
                    f"This may cause draft token divergence."
                )
        except RuntimeError:
            pass  # DP group may be torn down during shutdown

    def _sync_shutdown_state(self, local_should_shutdown: bool) -> bool:
        try:
            tensor = torch.tensor(
                [local_should_shutdown], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_should_shutdown = bool(tensor.item())
            return global_should_shutdown
        except RuntimeError as e:
            # If all_reduce fails, it means other ranks are shutting down
            logger.warning(
                f"{self.label}: Shutdown sync failed, assuming shutdown: {e}"
            )
            return True

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        if self._shutting_down:
            logger.info(f"{self.label}: Skipping DP sync during shutdown")
            return local_unfinished
        try:
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        except RuntimeError as e:
            # Handle case where other ranks have already shut down
            logger.warning(f"{self.label}: DP sync failed during shutdown: {e}")
            return local_unfinished
