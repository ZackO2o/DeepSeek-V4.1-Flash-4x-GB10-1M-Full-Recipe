---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
  - deepseek
  - deepseek-v4.1-flash
  - vllm
  - dgx-spark
  - gb10
  - sm121
  - tensor-parallel
  - cuda-graphs
  - dspark
  - speculative-decoding
  - engram
  - 1m-context
  - vision
  - tool-calling
language:
  - en
  - zh
pipeline_tag: text-generation
---

# DeepSeek-V4.1-Flash on 4× NVIDIA DGX Spark (GB10)
### One configuration with **1M context · CUDA graphs · vision · tool calling · DSpark speculative decoding**

**Chinese version: [README.zh-CN.md](README.zh-CN.md) · 中文文档见 [README.zh-CN.md](README.zh-CN.md)**

This repository publishes a *measured* serving recipe for `deepseek-ai/DeepSeek-V4.1-Flash`
(FP8 checkpoint, 48 shards, 4-way tensor parallel) on **four NVIDIA GB10 systems
("DGX Spark", SM 12.1, 128 GB unified memory each)** with vLLM.

It is field notes plus raw numbers. The interesting part is not "it boots" —
plenty of recipes do — it is **which combination of features actually fits in
4× 128 GB simultaneously**, and what the KV-cache pool really does as a function
of `--gpu-memory-utilization`. Everything in the tables below was measured on
our own hardware; nothing is extrapolated from a single node.

| | |
|---|---|
| **Model** | `deepseek-ai/DeepSeek-V4.1-Flash` (official FP8 weights, unchanged) |
| **Checkpoint** | 48 safetensors shards, ~510 GB total (two ~101 GB shards are the **Engram** tables) |
| **Hardware** | 4 × GB10 / SM 12.1, 128 GB unified memory, TP4 over RoCE |
| **Engine** | vLLM (`vllm/vllm-openai:nightly-...` base) + a **pinned Python tree** + 7 upstream patch files + 1 patch of ours |
| **Context** | **1,048,576 tokens** (`--max-model-len 1048576`) — measured, not aspirational |
| **Features at 1M** | ✅ CUDA graphs (`FULL_AND_PIECEWISE`) ✅ vision (`--limit-mm-per-prompt {"image":4}`) ✅ tool calling + reasoning parser ✅ DSpark spec decode (k=5) ✅ disk-backed Engram with node-local rows |
| **KV pool at 1M** | **1,716,692 tokens = 1.64× a 1M request** (gmu 0.83) |
| **Decode** | single stream **50.9 tok/s** mixed / **67.7 coding** / counting ceiling **84.2**; aggregate **125 tok/s** at C6 |
| **Quality gates** | needle PASS at 280K / 300K / 600K / 900K / 1M · garble gate 30/30 clean · vision+tools 7/7 |
| **License** | MIT (this repository). Model weights keep DeepSeek's own license. |

---

## 1. What this recipe adds

Compared with the excellent public work it builds on (see [Credits](#12-credits)),
this repository contributes:

1. **All four features at once, at 1M.** Most published 4× GB10 configurations trade
   something away — graphs *or* vision, 300 K *or* speed, text-only *or* tools. The
   configuration here keeps **1M max-model-len + FULL_AND_PIECEWISE CUDA graphs +
   vision + tool/reasoning parsers + DSpark k=5 + disk-backed Engram with node-local
   rows** live in the same process. The KV pool measurement (`1.64×`) is the reason
   it works: it is capacity, not feature conflicts, that decides this.
2. **A measured `gpu-memory-utilization` → KV-pool curve**, including the failure
   boundary (a 600 K window does *not* boot at `0.80`, and does at `0.82`) and the
   observation that the pool is **not** a pure function of the flag — it moves with
   host memory state, so **headroom, not the exact number, is what you tune**.
3. **A graph startup-state patch for a pinned tree.** Two environment-gated fixes in
   `v1/worker/gpu_worker.py` (skip the throwaway graph-memory profiling pass; clear
   dummy-warmup state after capture, in place, so graph buffer addresses survive),
   re-derived for the tree we build against. See
   [`patches/`](patches/README.md) — including the trap of mounting a patch file
   authored for a *newer* tree.
4. **Operational discipline that saved us twice**, published as scripts:
   a **four-node preflight** before any boot, and a **failure evidence collector**
   that grabs every rank's container log before the next boot destroys it.
5. **A quality-quantity pairing**: throughput tables *and* needle-in-haystack /
   garble / vision-tool gates, so "fast" cannot be published without "not broken".

---

## 2. Hardware and network

| | |
|---|---|
| Nodes | 4 × GB10 (`SM 12.1`), 128 GB unified memory each |
| Interconnect | direct RoCE ring, point-to-point links, `/30` per link, MTU 9000 |
| NCCL | `NCCL_IB_HCA` = the RoCE HCA, `NCCL_IB_GID_INDEX=3`, `NCCL_NET=IB`, RoCE v2 |
| Management link | used for `--master-addr` rendezvous and weight/NFS traffic control |

Two things we learned the hard way:

* **GID index 3 must be non-zero** on every port you intend to use. A port that
  enumerates GIDs full of zeros will fail NCCL with `errno 61` / `ibv_modify_qp`
  timeouts. Re-plugging/re-initialising the interface (or a cold power cycle) fixes it.
* **Serve weights over the fast fabric, not the management network.** Reading a 510 GB
  checkpoint through a 1 GbE management link takes hours; the same read over the
  RoCE ring is minutes. In our layout the head exports the checkpoint and each worker
  also keeps its own **node-local Engram rows** (see §5).

---

## 3. Checkpoint layout and the Engram tables

```
DeepSeek-V4.1-Flash/
├── config.json                 # architectures: ["DeepseekV41ForCausalLM"]
├── model.safetensors.index.json
├── tokenizer.json / tokenizer_config.json
├── model-00001-of-00048.safetensors   ...  model-00046-of-00048.safetensors
├── model-00047-of-00048.safetensors   # ~101 GB — Engram table
└── model-00048-of-00048.safetensors   # ~101 GB — Engram table
```

The two 101 GB shards are **not** dense weights. They hold the Engram n-gram memory
tables; the serving path can read them from disk instead of holding them resident,
and each rank only needs **its own row range**. In our deployment:

* the head serves the checkpoint to the workers read-only over the fabric,
* each worker has a ~48 GB **node-local sparse copy of its own rows** on NVMe and
  serves from that (`DSV41_ENGRAM_DIR`), verified row-by-row against the source,
* the model itself then fits in ~82 GB per rank, leaving room for the KV cache.

This is the single biggest memory decision in the whole recipe: without disk-backed
Engram with local rows, there is no 1M KV pool to talk about.

---

## 4. Launch configuration (the recipe)

Environment and flags that produced the numbers in §6. Addresses/interfaces are
placeholders — substitute your own.

```bash
# per node: rank 0..3
NODE_IPS=(<IP0> <IP1> <IP2> <IP3>)
FABRIC_IFACE=<roce-iface>        # e.g. the 200G link name
IB_HCA=<roce-hca>
MODEL_PATH=/srv/models/DeepSeek-V4.1-Flash
ENGRAM_DIR=/srv/engram-local/DeepSeek-V4.1-Flash   # workers only
```

```bash
docker run --gpus all --network host --ipc host \
  --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband --oom-score-adj 500 \
  -v "$MODEL_PATH:/models/DeepSeek-V4.1-Flash:ro" \
  -v "$ENGRAM_DIR:/engram-local:ro" \
  # ... plus the patch files bind-mounted over the vLLM Python tree ...
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
  -e DSV41_ENGRAM_DIR=/engram-local \
  -e DSV41_SKIP_GRAPH_MEMORY_PROFILE=1 \
  -e DSV41_CLEAR_STATE_AFTER_CAPTURE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_NET=IB -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IFACE" -e TORCH_CUDA_ARCH_LIST=12.1a \
  "$IMAGE" /models/DeepSeek-V4.1-Flash \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 --port 8888 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.83 \
    --max-model-len 1048576 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --block-size 128 \
    --engram-config '{"cpu_offload": false}' \
    --default-chat-template-kwargs '{"thinking": false}' \
    --limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1 \
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
    --reasoning-parser deepseek_v41 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,
                           "draft_sample_method":"probabilistic",
                           "rejection_sample_method":"block",
                           "enable_adaptive_verification":false}' \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE",
                           "cudagraph_capture_sizes":[...]}' \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$RANK" \
    --master-addr "${NODE_IPS[0]}" --master-port <PORT>
```

Notes that matter:

* **Capture sizes must match the speculative configuration.** With `k` speculative
  tokens and `max-num-seqs S`, the batch sizes the scheduler can produce are
  `k, 2k, … Sk` and `k+1, 2(k+1), … S(k+1)`. Generate the capture list from those,
  do not copy a list from a different `k` — a captured-shape mismatch on the sparse
  MLA path is a hang, not an error.
* **`--language-model-only` vs vision**: omit it to keep the vision tower; keep it for
  the text-only variant. Turning vision on costs KV-pool capacity (§6.4) because the
  multimodal buffers and the processor cache come out of the same unified memory.
* **`--default-chat-template-kwargs '{"thinking": false}'`**: with thinking on by
  default, short requests spend their whole budget in the reasoning field and the
  `content` field arrives empty. Clients can still opt in per request.
* **Boot order**: start workers first (highest rank → lowest), then the head, with a
  long engine-ready timeout. The head owns the rendezvous; if a worker never joins,
  the head blocks on the store and you get a *timeout*, not a clear error (§9).

---

## 5. The graph startup-state patch

The base image plus the pinned tree is not quite enough for graphs at large
`max-model-len` on this platform. Two things happen during startup that we patch:

| Change | Why |
|---|---|
| Skip the throwaway **graph-memory profiling pass** (`DSV41_SKIP_GRAPH_MEMORY_PROFILE=1`) | The profiler runs a full dummy pass to *estimate* graph memory before the real capture. At large `max-model-len` that transient allocation is what pushes a tight node over the edge — the engine then dies during initialisation with an unhelpful message. |
| Move the V2 **kernel warmup before capture**, and clear dummy-warmup state **after capture, in place** (`DSV41_CLEAR_STATE_AFTER_CAPTURE=1`) | A workspace resize after capture frees memory the captured graphs still point at; and dummy warmup can leave NaNs in KV slots that masked attention later reads. Clearing (rather than reallocating) preserves graph buffer addresses. |

`patches/gpu_worker_cachefix.py` is our re-derivation of those changes **for the
tree we build against**, `vim/v1/worker/gpu_worker.py`. See
[`patches/README.md`](patches/README.md) for the full diff rationale and the
licence/attribution notes — including why you must **not** mount a version of this
file that was authored for a newer vLLM tree.

---

## 6. Measured results

All numbers below are from our own four-GB10 deployment. "thinking off",
temperature 0, fixed prompt set, tokens counted from the server's own usage block.

### 6.1 Throughput at 1M configuration (gmu 0.83, k=5, vision+tools+graphs)

| concurrency | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| C1 | 44.7 | **50.9** | 0.38 |
| C2 | 72.4 | 42.0 | 0.42 |
| C3 | 84.8 | 32.4 | 0.45 |
| C6 | **125.1** | 24.2 | 0.55 |

Per-stream tok/s by category (C1 … C6):

| category | C1 | C2 | C3 | C6 |
|---|---|---|---|---|
| coding | 67.7 | 60.9 | 52.1 | 34.7 |
| json | 44.1 | 49.3 | 29.5 | 23.7 |
| narrative | 27.5 | 19.4 | 16.1 | 11.7 |
| prose | 31.1 | 22.5 | 17.9 | 13.1 |
| math | **74.0** | 51.1 | 42.5 | 32.3 |
| reasoning | 56.9 | 41.5 | 31.9 | 22.2 |
| summary | 33.0 | 26.7 | 23.2 | 14.4 |
| format | 73.1 | 64.7 | 45.7 | 41.8 |
| **counting ceiling** | **84.2** | 75.1 | 45.3 | 43.0 |

The counting category is excluded from aggregates; it is the practical
decode ceiling of this configuration.

### 6.2 Cold prefill (unique prefix, no cache reuse)

| prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|
| 2,950 | 2.0 | 1,449.9 |
| 46,810 | 32.4 | 1,445.8 |

### 6.3 Long-context: needle retrieval and prefill rate

Single needle at 50 % depth, exact-match answer, cold prefix:

| context | config | prompt tokens | TTFT (s) | prefill tok/s | result |
|---|---|---|---|---|---|
| 280K | 600K window, graphs | 278,118 | 204.0 | 1,363.2 | PASS |
| 300K | 1M window, eager | 298,172 | 293.9 | 1,014.4 | PASS |
| 600K | 600K window, graphs | 596,196 | 529.5 | 1,125.9 | PASS |
| 600K | 1M window, eager | 609,017 | 714.8 | 852.0 | PASS |
| 900K | **1M window, graphs** | 894,548 | 1,012.0 | 884.0 | PASS |
| 1M | 1M window, eager | 993,435 | 1,243.3 | 799.1 | PASS |

Retrieval was exact in every case. Note the **prefill rate is configuration
dependent**: the same 600K prompt prefills at ~1,126 tok/s with graphs and ~852 tok/s
eager. If your client timeout is shorter than `TTFT`, you will lose the answer no
matter how good the model is — size the client for `context / prefill_rate`.

### 6.4 KV-cache pool vs `--gpu-memory-utilization` (the part nobody publishes)

Unified-memory machines do not behave like discrete GPUs here. Measured pool sizes
at the moments each configuration booted:

| gmu | max-model-len | vision/tools | graphs | KV pool (tokens) | pool / window |
|---|---|---|---|---|---|
| 0.80 | 300,000 | on | yes | 648,717 | 2.16× |
| 0.80 | 600,000 | on | **boot failed** | — | — |
| 0.82 | 600,000 | on | yes | 780,110 … 1,149,256 | 1.30× … 1.92× |
| 0.83 | 1,048,576 | on | yes | **1,716,692** | **1.64×** |
| 0.83 | 1,048,576 | text-only, eager | no | 1,529,317 … 1,627,978 | 1.46× … 1.55× |

Two conclusions we would have got wrong by reasoning instead of measuring:

* **The pool is not a pure function of the flag.** The same `0.82` gave us two pools
  ~370 K tokens apart on different boots, because the pool is computed from *free
  memory at init time* and that depends on host state. Tune for **headroom**
  (we aim ≥1.5× the window), not for a magic gmu number.
* **A 600 K window did not boot at `0.80` but boots at `0.82`** — a ~2.4 GB
  difference (≈500 K tokens of KV) was the whole gap. Combined with the graph/patch
  work in §5, this is the difference between "600 K is impossible with vision" and
  "1M with vision works".

### 6.5 Decode at context depth (1M window, eager, 1024-token generations)

| task | prompt tokens | TTFT (s) | prefill tok/s | **decode tok/s** |
|---|---|---|---|---|
| code | 1,015,402 | 1,129.4 | 899.1 | **52.5** |
| prose | 1,015,382 | 995.2 | 1,020.3 | **24.3** |

Decode at depth is content dependent (speculative acceptance differs between code and
prose) — publish both, or your numbers are unfalsifiable.

### 6.6 Quality gates

| gate | result |
|---|---|
| garble gate (30 structured generations, temperatures 0 / 0.7 / 1.0, 6 in flight) | **30/30 clean** |
| vision tasks (colour order, two images, grid quadrant) | **3/3 PASS** |
| tool calls (single call, result round-trip, two parallel calls, forced `tool_choice`) | **4/4 PASS** |
| NaN probe (arithmetic / factual / self-intro at temperature 0) | no NaN, correct |
| needle-in-haystack | PASS at every context in §6.3 |

A note on why the gates exist: during development we had a configuration that
**booted clean, passed profiling and graph capture, and then emitted repeated garbage
from the first token** — see §9. Throughput tables cannot detect that; gates can.

---

## 7. Reproducing the measurements

```bash
# throughput grid: concurrency 1/2/3/6 across 8 categories + cold prefill
python3 bench/v41bench.py --base http://127.0.0.1:8888/v1 \
        --model deepseek-v4.1-flash --levels 1,2,3,6 \
        --prefill 2000,8000,32000 --out ./results

# decode at depth (code + prose, 1024 generated tokens, 4 reporting windows)
python3 tools/ctx_decode_bench.py --base http://127.0.0.1:8888/v1 \
        --model deepseek-v4.1-flash --targets 1000000 --maxtok 1024
```

`tools/` in this repository contains our own scripts only: the decode-at-depth
benchmark, the four-node preflight, and the failure-evidence collector.

---

## 8. Operating the fleet

* **Preflight before every boot** (`tools/preflight.sh`): verify, *on every rank*,
  that the image exists, the NCCL library is present, the checkpoint and the local
  Engram rows are mounted, and that **every patch file** referenced by the mount list
  exists locally. Patch files are bind-mounted per node: a patch directory that exists
  only on the head turns into three workers that never start while the head waits
  for them (§9).
* **Start workers before the head**, highest rank first, then the head, with
  `VLLM_ENGINE_READY_TIMEOUT_S` large (we use 3600). Weight loading is ~4–5 minutes
  per rank plus graph capture; expect ~10–15 minutes to a ready endpoint.
* **Keep the local Engram copy in sync with the checkpoint revision.** Rows are
  rank- and revision-specific; do not reuse a copy across revisions.
* **One deployment per node set.** The four nodes are a single TP4 group; a second
  model on the same nodes will fight for the same unified memory.

---

## 9. Pitfalls (each of these cost us real time)

1. **A patch file authored for a newer tree.** Whole-file patches must match the tree
   they are mounted over. If a patch expects symbols your tree lacks, you get an
   import error — annoying but honest. The dangerous variant is the opposite: a tree
   newer than the anchors your patches were written against, which can produce an
   engine that *boots*, passes profiling, and then emits repeated garbage tokens from
   the first position, with the speculative drafter accepting nothing. Pin the Python
   tree **and** the compiled extension to the same commit as the patch set.
2. **Patch directory only on the head.** Symptom on the head: the container starts,
   loads weights, then blocks in the process-group store and finally dies with
   `DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined`.
   Real cause: the workers' launch step exited with `PATCH MISSING ...` and nothing was
   listening. Check the worker launch output *before* diagnosing memory.
3. **Losing the evidence.** The most informative logs are on the workers, and the next
   boot removes those containers. Collect every rank's container log *immediately*
   after a failure (`tools/capture-allnode-logs.sh`).
4. **Sizing the window by KV capacity alone.** `pool ≥ window` is necessary, not
   sufficient: startup also needs transient memory (profiling pass, capture, warmup).
   Leave headroom, or make the transient passes cheaper (§5).
5. **Rendezvous ports across restarts.** Rapid successive boots should use a distinct
   `--master-port` (or ensure the previous store is gone), otherwise a stale store can
   hold the address while a new group tries to form on it.
6. **Trusting a single throughput number.** Code and prose decode differ by ~2× at
   depth because speculative acceptance differs. Report both, and pair every
   throughput table with a quality gate.
7. **Assuming the model knows its own configuration.** Ask it "what is your max
   context?" and it may answer from training data. Read `max_model_len` from the API,
   not from the model.

---

## 10. Frequently asked: 300K vs 600K vs 1M

With the recipe above, the question is capacity, not features:

| window | KV pool (measured) | pool / window | single-stream C1 | notes |
|---|---|---|---|---|
| 300K | 648,717 (gmu 0.80, vision+graphs) | 2.16× | 48.3 | most concurrent long requests |
| 600K | 780,110 – 1,149,256 (gmu 0.82) | 1.30× – 1.92× | ~50 | did **not** boot at gmu 0.80 |
| 1M | **1,716,692 (gmu 0.83)** | **1.64×** | **50.9** | vision + tools + graphs all on |

Because the pool is in the same order of magnitude across these windows while the
window itself varies by 3.5×, **the window you can serve is decided by the pool, and
the pool is decided by how much memory you leave for it** — not by any feature
trade-off. That is why we publish the pool column everywhere.

---

## 11. Environment we tested on

| Component | Version / note |
|---|---|
| Nodes | 4 × GB10, 128 GB unified memory each |
| CUDA arch | `12.1a` |
| Base image | `vllm/vllm-openai:nightly-<pin>` (kept private as an exact digest in the recipe) |
| Build | extension compiled for `sm121a`, `MAX_JOBS=2`, `FLASHINFER_NVCC_THREADS=1`, `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Container limits | `--memory 112g --memory-swap 112g --shm-size 32g --oom-score-adj 500` |
| Host tuning | `vm.min_free_kbytes` raised, watermark scale factor raised, NFS server threads raised, GPU clocks unlocked, container processes write their caches to NVMe |

---

## 12. Credits

Standing on other people's work, clearly stated:

* **DeepSeek** — the model, the checkpoint and the draft layers
  (`deepseek-ai/DeepSeek-V4.1-Flash`).
* **Tech2Wild/Kai (tonyd2wild)** — the foundational four-node Spark recipe: patch set,
  disk-backed Engram staging, worker-first boot order, image chain and benchmark
  protocol. Most of what we run is their structure; we changed how it is parameterised
  and how much of it fits at 1M.
* **0xTank** — the graph startup-state fix (skip the profiling pass; clear state after
  capture) and the compact output-projection idea. The patch in `patches/` is our
  re-derivation of those changes for the tree we build against.
* **vLLM, FlashInfer, Triton, PyTorch, NVIDIA** and their contributors — the engine,
  kernels and toolchain.
* **The wider DGX Spark community** publishing quantizations and recipes for this
  hardware; the pool/window data in §6.4 exists because their numbers made us
  suspicious of our own.

If you use this recipe, cite the people above first. The value we add is measurement
and operational discipline.

---

## 13. License

MIT — see [LICENSE](LICENSE). The model weights and any upstream source files retain
their own licences; see [NOTICE.md](NOTICE.md).

This repository contains **no** model weights, **no** credentials, **no** private
addresses and **no** operational secrets. Everything here is a configuration,
a measurement, or a script.
