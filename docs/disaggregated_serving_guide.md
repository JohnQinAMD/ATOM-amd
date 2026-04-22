# Disaggregated Serving

ATOM supports **disaggregated prefill/decode serving**: the expensive
prompt-encoding phase (prefill) runs on dedicated GPUs, and the resulting
KV cache is streamed via RDMA to separate GPUs that serve the
generation loop (decode). This decouples the two workloads so each can be
provisioned, scaled, and batched independently.

- **CLI flag:** `--disagg-mode {null,prefill,decode}` (default `null` — aggregated)
- **Transport:** MoRI IOEngine RDMA for KV blocks + ZMQ PUSH/PULL for handshakes
- **Integrated with:** Dynamo's `PrefillRouter` (works on `sglang`, `vllm`, and
  `atom` backends uniformly)

---

## 1. Data path

```
    Client
      │
      ▼
  Dynamo frontend  ──(1)──►  Prefill worker (atom, --disagg-mode prefill)
                                    │
                                    │ (2) RDMA WRITE (MoRI IOEngine)
                                    ▼
                              Decode worker (atom, --disagg-mode decode)
                                    │
                                    │ (3) token stream
                                    ▼
                              Dynamo frontend
                                    │
                                    ▼
                                  Client
```

1. **Frontend → prefill:** the `PrefillRouter` forwards the request (with a
   fresh `bootstrap_room` ID) to the chosen prefill worker.
2. **Prefill → decode (RDMA):** prefill computes the KV cache, reads the decode
   worker's RDMA descriptors via the ZMQ bootstrap exchange, then RDMA-WRITES
   the KV blocks directly into decode's registered GPU memory.
3. **Decode → frontend:** decode schedules the sequence straight into its
   decode loop (no re-prefill), generates tokens, and streams them back.

Prefill and decode may live on different nodes; the RDMA path is GPU-Direct over
Pensando AINIC (RoCEv2). Only small ZMQ control messages traverse TCP.

---

## 2. KV transfer bootstrap protocol

Bootstrap is a **one-shot handshake per request**, keyed by `bootstrap_room` (a
63-bit ID assigned by the Dynamo PrefillRouter). The protocol uses
`zmq.PUSH` / `zmq.PULL` — intentionally *not* `ROUTER/DEALER` — so each message
carries its own routing key in the payload rather than depending on ephemeral
ZMQ socket identities. This eliminates a class of bugs where retries created
fresh DEALER identities that the server could no longer address.

### Messages (both keyed by `bootstrap_room`)

**Decode → prefill (handshake request):**
```
{
  "kind":            "request",
  "bootstrap_room":  <63-bit id>,
  "engine_desc":     <MoRI engine descriptor, bytes>,
  "mem_desc":        <decode-side registered KV memory descriptor, bytes>,
  "scale_mem_desc":  <optional FP8 scale tensor descriptor, bytes>,
  "block_ids":       [<decode dst block id>, ...],
  "reply_endpoint":  "tcp://<decode_host>:<decode_puller_port>",
}
```

**Prefill → decode (handshake reply):**
```
{
  "kind":            "reply",
  "bootstrap_room":  <same id>,
  "engine_desc":     <MoRI engine descriptor, bytes>,
  "mem_desc":        <prefill-side registered KV memory descriptor, bytes>,
  "scale_mem_desc":  <optional FP8 scale tensor descriptor, bytes>,
  "block_ids":       [<prefill src block id>, ...],
}
```

After reply, prefill RDMA-WRITEs `block_ids[src] → block_ids[dst]` using
`bootstrap_room` as the MoRI `transfer_uid` so completion can be polled on the
decode side.

### Why PUSH/PULL (not ROUTER/DEALER)

With `ROUTER/DEALER`, the server reply is routed by the ephemeral ZMQ identity
the client's socket had when it sent. If the client's socket is closed
(timeout, retry, or process restart) before the server replies, the reply is
dropped silently even when RDMA itself succeeds. Clients creating a fresh
DEALER per attempt always miss replies addressed to earlier identities.

With `PUSH/PULL`, every message is self-contained; the receiver's puller
buffers by `bootstrap_room`, and the sender's reply is routed by the
`reply_endpoint` it learned from the request payload. Sockets are cached per
endpoint and reused, so retries don't churn identities. This matches the
pattern used by SGLang's disaggregation bootstrap (PUSH/PULL + HTTP
rendezvous).

---

## 3. Scheduler path for a disaggregated decode

A decode-only sequence arrives at the atom scheduler with:
- `seq.disagg_mode == "decode_only"`
- `seq.block_table` already populated (blocks reserved by `engine_core` before
  the RDMA recv, via `block_manager.reserve_blocks_with_headroom`, then bound
  via `allocate_specific_blocks`)
- `seq.num_cached_tokens == seq.num_prompt_tokens` (all prompt tokens already
  cached because the KV is already present in the local paged KV)

The scheduler takes a **separate fast path** for such sequences: it skips the
prefill batch entirely, moves the sequence straight to `self.running` with
`type = DECODE`, and lets the next decode loop pick it up. This avoids two
bugs that hit the generic path:

1. `block_manager.allocate(seq)` asserts `not seq.block_table` — would fire on
   the already-bound block list.
2. `model_runner.prepare_model()` asserts `total_tokens_num > 0` — would fire
   on a 0-token prefill batch (all tokens are cached).

See `atom/model_engine/scheduler.py`, the prefill-loop branch starting at the
decode-only check.

---

## 4. MTP headroom

When MTP (multi-token prediction) is enabled on the decode side, the first
decode step requires `mtp_k + 1` additional blocks beyond the ones holding the
received prefill KV. EngineCore reserves blocks through
`block_manager.reserve_blocks_with_headroom(num_blocks, headroom=mtp_k+1)` so
that the decode doesn't OOM on its very first MTP step.

---

## 5. Multi-rank (TP / DP) disagg

Each ModelRunner rank binds its own ZMQ puller on `bootstrap_port + rank`.
The Dynamo PrefillRouter advertises `bootstrap_port` (rank 0); non-zero ranks
derive their port by adding their rank index.

For DP > 1, only DP rank 0 receives the request from Dynamo. Rank 0
`torch.distributed.broadcast_object_list`s the decode's KV-recv metadata to
all other ranks; each rank then runs its own RDMA recv in parallel,
corresponding to the prefill rank with the same rank index. See
`DPEngineCoreProc.pull_and_process_input_queue()`.

With MTP + DP, batch sizes must stay consistent across ranks (since MoE
dispatch/combine expects every rank to participate in every forward).
`_verify_mtp_batch_consistency` all-reduces MIN/MAX batch sizes and warns on
divergence.

---

## 6. FP8 KV cache

When `--kv_cache_dtype fp8`, each layer has a separate per-block scale tensor
in addition to the quantised KV tensor. Both are registered with MoRI for RDMA
and transferred in parallel: `KVCacheLayout.has_kv_scale` gates an additional
`batch_write` with `transfer_uid = bootstrap_room + 1` to avoid UID
collisions.

---

## 7. AINIC tuning

MoRI's RDMA backend is initialised with a configuration tuned for AMD's
Pensando AINIC NICs:

```python
RdmaBackendConfig(
    qp_per_transfer=4,
    num_worker_threads=2,
    poll_cq_mode=PollCqMode.POLLING,
)
```

Two worker threads are enough to saturate line rate (~48 GB/s per NIC); four
QPs keep the NIC's completion queue busy without contention.

---

## 8. Launch

### 8.1 Single node, 2× GPU (Qwen3-0.6B, smoke test)

```bash
# Terminal 1 — etcd + NATS + Dynamo frontend + prefill worker (GPU 0)
etcd &
nats-server -p 4222 -js &
python3 -m dynamo.frontend --http-port 8888 &
HIP_VISIBLE_DEVICES=0 python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/Qwen3-0.6B \
    -tp 1 \
    --block-size 16 \
    --kv_cache_dtype bf16 \
    --disagg-mode prefill \
    --disagg-bootstrap-port 19876

# Terminal 2 — decode worker (GPU 1)
ETCD_ENDPOINTS=http://localhost:2379 \
NATS_SERVER=nats://localhost:4222 \
HIP_VISIBLE_DEVICES=1 python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/Qwen3-0.6B \
    -tp 1 \
    --block-size 16 \
    --kv_cache_dtype bf16 \
    --disagg-mode decode \
    --disagg-bootstrap-port 19877

# Issue a request
curl -X POST http://localhost:8888/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/hf/Qwen3-0.6B","prompt":"The capital of France is","max_tokens":10}'
```

### 8.2 Multi-node with Pensando AINIC

Pensando AINIC NICs ("ionic" driver) expose RDMA over RoCEv2 and are
**IPv6-only in the deepep-a66 fabric**: the RoCEv2 GID[1] is a link-local
ULA that every host on the fabric can reach. MoRI's RDMA backend talks to
the ibverbs device directly, so you don't pick a GID explicitly — just
make sure the container has `--device=/dev/infiniband` and that
`ibv_devices` shows `ionic_0..ionic_N`.

**Per-node prerequisites (each worker node):**

```bash
# In the container (atom-disagg image): confirm RDMA devices are visible
ls /dev/infiniband/   # expect: rdma_cm, uverbs0..N
ibv_devices           # expect: ionic_0, ionic_1, ...
# If libionic was updated on the host, sync the container ABI (one-time):
/usr/local/bin/fix-ionic-abi.sh

# MoRI shared-memory heap (affects RDMA buffer pool)
export MORI_SHMEM_HEAP_SIZE=32G

# AITER kernel log suppression
export AITER_LOG_LEVEL=WARNING
```

Docker invocation (required flags) — run on each participating node:

```bash
docker run --rm --name atom-disagg \
    --network=host --privileged --ipc=host --shm-size=256G \
    --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
    --group-add video \
    -e MORI_SHMEM_HEAP_SIZE=32G \
    -e AITER_LOG_LEVEL=WARNING \
    -v /path/to/huggingface:/hf \
    amdprimus/dynamo-rocm-atom-disagg:latest \
    bash
```

> **Why `--network=host`:** MoRI's RDMA path binds directly to the host
> ibverbs device; the `bootstrap_host` it publishes is the host IP.
> Bridged networking breaks cross-node RDMA.

#### Single prefill node + single decode node (DeepSeek-R1 FP8, TP=8)

Two nodes, 8 GPUs each, 1 prefill worker on node 0 (TP=8), 1 decode
worker on node 1 (TP=8). Node 0 also runs etcd/NATS/Dynamo frontend.

```bash
# ── Node 0 (137.220.63.58): prefill + control plane ────────────────
etcd --listen-client-urls=http://0.0.0.0:2379 \
     --advertise-client-urls=http://137.220.63.58:2379 > /tmp/etcd.log 2>&1 &
nats-server -a 0.0.0.0 -p 4222 -js > /tmp/nats.log 2>&1 &

python3 -m dynamo.frontend --http-port 8000 > /tmp/frontend.log 2>&1 &

ETCD_ENDPOINTS=http://137.220.63.58:2379 \
NATS_SERVER=nats://137.220.63.58:4222 \
python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/DeepSeek-R1-0528 \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --disagg-mode prefill \
    --disagg-bootstrap-port 19876 \
    > /tmp/prefill.log 2>&1 &

# ── Node 1 (108.61.229.36): decode ─────────────────────────────────
ETCD_ENDPOINTS=http://137.220.63.58:2379 \
NATS_SERVER=nats://137.220.63.58:4222 \
python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/DeepSeek-R1-0528 \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --disagg-mode decode \
    --disagg-bootstrap-port 19877 \
    > /tmp/decode.log 2>&1 &

# ── Any node with network access to node 0 ─────────────────────────
curl -X POST http://137.220.63.58:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/hf/DeepSeek-R1-0528","prompt":"The capital of France is","max_tokens":16,"temperature":0.0}'
```

Each rank (TP=8) binds its own ZMQ puller on
`--disagg-bootstrap-port + rank` (so rank 0 listens on 19876, rank 1 on
19877, …). Only the rank-0 port is advertised via Dynamo; the others are
derived implicitly.

#### Multi-node prefill / multi-node decode (DeepSeek-R1, TP=16, DP=2)

Two 8-GPU nodes per side (16 GPUs total per role = 2 DP ranks of TP=8),
mirroring the sglang "DeepSeek Multi-Node" topology.

```bash
# ── Prefill node 0 (rank 0) ───────────────────────────────────────
ETCD_ENDPOINTS=http://${CTRL_IP}:2379 \
NATS_SERVER=nats://${CTRL_IP}:4222 \
python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/DeepSeek-R1-0528 \
    -tp 8 --data-parallel-size 2 \
    --kv_cache_dtype fp8 --block-size 16 \
    --enable-expert-parallel \
    --disagg-mode prefill \
    --disagg-bootstrap-port 19876

# ── Prefill node 1 (rank 1) — same command on the peer host ──────
#    (dp_rank 1 will bind its puller on 19877 = 19876 + 1)

# ── Decode node 0 (rank 0) ───────────────────────────────────────
ETCD_ENDPOINTS=http://${CTRL_IP}:2379 \
NATS_SERVER=nats://${CTRL_IP}:4222 \
python3 -m dynamo.atom \
    --namespace dynamo \
    --model /hf/DeepSeek-R1-0528 \
    -tp 8 --data-parallel-size 2 \
    --kv_cache_dtype fp8 --block-size 16 \
    --enable-expert-parallel \
    --disagg-mode decode \
    --disagg-bootstrap-port 29876

# ── Decode node 1 (rank 1) — same command on the peer host ───────
```

For DP > 1, only rank 0 receives the request from Dynamo; rank 0
broadcasts the decode-KV-recv metadata to non-zero ranks (via
`torch.distributed.broadcast_object_list`), and each decode rank N
connects to the corresponding prefill rank N on
`prefill_bootstrap_port + N`.

#### Verifying the AINIC path before serving

```bash
# On each node: RDMA device present + link up
ibv_devices       # ionic_0..ionic_N listed
ibstat ionic_0    # State: Active, LinkLayer: Ethernet, Rate: 200 Gb/sec

# End-to-end bandwidth check between node pair (MoRI sanity)
# (Run the MoRI IOEngine loopback test bundled with mori-amd)
python3 -m mori.benchmarks.io_loopback \
    --peer ${peer_host} --size 256MB

# Dynamo worker log: look for these lines after worker startup
grep "KV bootstrap puller listening" /tmp/prefill.log
grep "KV transfer initialized" /tmp/prefill.log
```

#### Known AINIC caveats

- **IPv6-only fabric (deepep-a66):** the RoCEv2 GID is a link-local ULA;
  don't pass `--disaggregation-ib-device` or similar IPv4-only flags.
  The MoRI IOEngine selects the device automatically.
- **libionic ABI drift:** the container may ship an older libionic than
  the host driver. Run `/usr/local/bin/fix-ionic-abi.sh` inside the
  container once after launch to symlink to the host ABI (the
  provided ATOM disagg container includes this script).
- **Per-device memory cap:** ionic enforces ~250 MB total `ibv_reg_mr`
  per device. The KV transfer registers one big MR per worker (the
  full `kv_cache` tensor), which for 266 GB KV caches requires the
  host to have `CAP_SYS_RAWIO` (present via `--privileged`).
- **FP8 KV cache:** the per-block scale tensor is a second MR; factor
  that into the MR budget when deploying very large models.

---

## 9. Code map

| Concern | File |
|---------|------|
| ZMQ bootstrap (PUSH/PULL, per-room routing) | `atom/model_engine/kv_bootstrap.py` |
| MoRI IOEngine register / batch_write / wait | `atom/model_engine/kv_transfer.py` |
| `KVCacheLayout` + FP8 scale tensor offsets | `atom/model_engine/kv_transfer.py` |
| `ModelRunner.init_kv_transfer/kv_send/kv_recv` | `atom/model_engine/model_runner.py` |
| EngineCore disagg scheduling + DP broadcast | `atom/model_engine/engine_core.py` |
| `reserve_blocks_with_headroom`, MTP headroom | `atom/model_engine/block_manager.py` |
| Decode-only fast path in scheduler | `atom/model_engine/scheduler.py` |
| `prev_batch is None` handling on first decode step | `atom/model_engine/model_runner.py` |

---

## 10. Known limitations

- **Decode worker restart:** the Dynamo frontend's `PrefillRouter`
  activator was one-shot at the time atom disagg landed; restarting a decode
  worker while prefill is still alive needed a frontend restart to re-link.
  The Dynamo-side fix (persistent `watch::Sender<Option<Endpoint>>`) is
  landing separately; once deployed, decode-worker restarts will re-link
  automatically.
- **Prefill DP > 1 heterogeneous rank-count:** this path broadcasts from
  rank 0 and assumes matching prefill/decode rank counts. DP-mismatched
  deployments are not yet supported.
- **Prefix caching + disagg:** decode-only seqs bypass the block manager's
  prefix-cache lookup (their KV arrived from prefill, not from cache).
  Mixing disagg with prefix-caching decoders isn't supported yet.
