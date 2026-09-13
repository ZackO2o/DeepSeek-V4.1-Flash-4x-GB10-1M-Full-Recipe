# 2026-09-13 — collective transport A/B, the KV-pool cost of the window, and one negative result

Two changes were tested under a controlled protocol on the same four-node ring, same
checkpoint, same server build, same day:

1. **Collective transport** — swap the NCCL build and its environment for the
   switchless-ring variant published by **FujitsuPolycom/sparkring** (Apache-2.0),
   together with dual-HCA use and a fixed channel count.
2. **Engram read path** — their `BALANCED` hash-column assignment plus packed
   single-read shards, against a deployment that already stages per-rank Engram rows
   locally.

Everything else was held constant: `1M` window, CUDA graphs (`FULL_AND_PIECEWISE`),
vision + tool calling, DSpark `k=5`, block 128, `gpu-memory-utilization 0.84`,
one rank per node, TP=4 over the direct ring.

Note on scope: both sides of every comparison below ran the **same checkpoint**; the
transport and Engram conclusions are properties of the communication and I/O paths, not
of a particular weight set, but absolute throughput can move a few percent between
checkpoints, so treat the deltas as the portable part.

The rule for this report: **a number is only claimed when it survives repetition.**
A single throughput grid on this hardware has a run-to-run spread of **±5–9 % per
concurrency level**, so the baseline was run **twice** before any conclusion was drawn.

---

## 1. Transport A/B — what changed and what it bought

| Metric | baseline (run 1) | baseline (run 2) | switchless transport | read |
|---|---:|---:|---:|---|
| C1 aggregate / per-stream tok/s | 43.26 / 48.91 | 46.10 / 52.51 | 47.87 / 53.31 | inside noise |
| C2 aggregate / per-stream | 74.15 / 43.01 | 71.49 / 40.81 | 79.34 / 45.78 | +7…11 % |
| C3 aggregate / per-stream | 85.75 / 33.60 | 93.21 / 35.81 | 97.00 / 36.96 | inside noise |
| **C6 aggregate / per-stream** | 136.76 / 26.17 | 134.31 / 26.01 | **153.33 / 29.30** | **+12…14 %** |
| cold prefill 2K / 8K / 32K tok/s | 1259 / 934 / 1309 | 1636 / 1450 / — | 1639 / 1821 / 1908 | top of band |
| 2-stream aggregate (smoke) | 123.0 | — | **144.6** | **+17.6 %** |
| KV pool at 1M window | 1,700,113 | — | 1,742,812 | +2.5 % |

**What we claim:** the **C6 gain is real** (both baseline runs sit below it, and C6 is
exactly where TP=4 all-reduce cost shows up), and the 2-stream aggregate moves with it.
**What we do not claim:** C1/C3 improvements, and any precise prefill percentage — the
baseline prefill numbers are themselves spread over 934–1636 tok/s across runs.

**What did not change:** single-stream latency (C1) — a single request has almost no
collective traffic to optimise. If your workload is one request at a time, this change
buys you nothing. It pays off under concurrency.

### The configuration that produced it

```
# transport selection (all four ranks identical)
NCCL_ALGO=Ring
NCCL_NET=IB
NCCL_IB_HCA=<roce-netdev-a>,<roce-netdev-b>   # both RoCE devices, one per ring direction
NCCL_IB_GID_INDEX=<gid>                        # non-zero GID only; all-zero = no RoCE
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_SUBNET_PREFIX_LEN=24
NCCL_SKIP_TREE_CONNECT=1
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
NCCL_PROTO=LL,LL128,Simple
NCCL_P2P_LEVEL=SYS
NCCL_CROSS_NIC=1
NCCL_IB_MERGE_NICS=0
NCCL_CUMEM_ENABLE=0
NCCL_SWITCHLESS_RING_ONLY=1                    # only exists in the patched NCCL below
```

The switchless build also needs its own `libnccl.so.2` (2.30.7) preloaded in the
container (`LD_PRELOAD`) instead of the wheel-shipped NCCL.

**Where the library comes from (no need to pull their multi-gigabyte image):** their
GitHub **release asset** `native-runtime-files-<date>.tar` contains
`native/libnccl.so.2.30.7` — aarch64, `NCCL version 2.30.7 compiled with CUDA 13.3`,
with the `ncclParamSwitchlessRingOnly` symbol present. Their source patches live in
`spark_transport/nccl/*.patch` and their build recipe in
`runtime/sparkring/jovian-r33/build_nccl.sh` if you prefer to build it yourself.

⚠️ **Channel count matters.** With this fabric and a different channel setting we have
previously seen `ibv_modify_qp` timeouts during NCCL init. `4` is what is verified
here; treat anything else as an experiment you validate on your own ring.

---

## 2. KV pool vs `--gpu-memory-utilization`, at the full 1M window

Same model, same features (graphs + vision + tools + `k=5`), same window
(1,048,576). Only the memory-utilization fraction differs. The pool token count is
1:1 with request length — a 1M request occupies 1M pool tokens, not a multiple.

| `--gpu-memory-utilization` | observed KV pool (tokens) | full-1M requests resident | setups measured |
|---|---:|---:|---:|
| 0.83 | 1,484,371 – 1,597,326 | 1.42 – 1.52× | 4 |
| **0.84** | **1,700,113 – 1,870,320** | **1.62 – 1.78×** | 3 |

Two operational notes:

* the same configuration lands on a **different pool size on a different boot**
  (≈±7 %), because the graph/profiling allocations and fragmentation differ; do not
  read a single boot as "the" pool size;
* raising the fraction is a **window-preserving** way to buy concurrency. Shrinking
  the window to buy pool space is the opposite trade and costs you context.

---

## 3. The negative result: Engram packing did not transfer

SparkRing documents a prefill gain of **+12…18 %** and −13 % burst TTFT for
`BALANCED` hash-column assignment combined with packed single-read shards.

Measured here: **no gain** (every level inside the noise band, both directions).
The reason is the baseline it was measured against. Their comparison is *packed shard
(single pread per row) vs raw checkpoint shards (two preads per row, weight and scale
≈24 GB apart)*. Our deployment already stages each rank's Engram rows on local NVMe,
so it was **already** on a single-read path — there was no second pread to remove.

**Lesson (the general one):** before adopting a published optimisation, identify its
baseline. If your setup already has the property the optimisation adds, the published
number is not available to you. Publishing the negative result costs nothing and saves
someone else the boot cycle.

---

## 4. How these measurements were taken

* **Grid**: 8 prompt categories + a counting-ceiling prompt, each run at concurrency
  1/2/3/6; tokens counted from the server's `usage.completion_tokens`; temperature 0,
  thinking off; identical prompt set across boots (cache-cold by construction).
* **Cold prefill**: unique filler prefixes per target length; TTFT measured as
  first-token delta, prefill rate = prompt tokens / TTFT.
* **KV pool**: read from the engine's own startup line, not estimated.
* **Acceptance gates** (all must pass before a configuration is kept): smoke + NaN
  probe with `logprobs` (NaN logits make the API refuse to serialise), a 30-generation
  garble gate across three temperatures, a 30-item correctness set, a vision +
  tool-calling suite (7 checks, deterministic images generated in-process), a
  refusal-rate probe, and a 12-item capability set.
* **Repetition**: any claim that changes a configuration is measured **twice on the
  baseline** and once on the candidate; if the candidate does not beat both baseline
  runs, it is reported as noise. This is the single discipline that keeps a
  "we improved X by 15 %" claim honest on this hardware.

---

## 5. Serving telemetry over the public path

Same configuration as §6.7, measured through the deployment's public gateway (one hop
of TLS + tunnel + proxy in front of the engine), so these are numbers a client actually
sees rather than engine-side counters:

| what | value | note |
|---|---:|---|
| steady-state single-stream decode | **~50–53 tok/s** | engine-side C1 per-stream, repeated runs |
| single-stream ceiling (fast/counting content) | **~86 tok/s** | best-case category |
| end-to-end, 700 generated tokens | **~55 tok/s** | public path, includes first-token wait |
| end-to-end, 273 generated tokens | ~27 tok/s | short outputs are diluted by TTFT — not a decode figure |
| 2 concurrent streams (aggregate) | **~132 tok/s** | measured live |
| 6 concurrent streams (aggregate) | **~153 tok/s** | best observed after the transport swap |
| prefill, 3.5K-token prompt | **~1,960 tok/s** | long-context prefill lands in the 900–1,900 band |
| first-token latency, short prompt | **~1.4 s** | public path round trip |
| KV pool at the 1M window | **1,870,320 tokens** | 1.78× the window |

Reading it: **single-stream decode is ~50 tok/s and does not move with the transport
work**; the transport work shows up as concurrency (2 streams ≈132, 6 streams ≈153).
Short generations over the public path look slower than the engine number because the
first token dominates — measure decode with ≥500 generated tokens or read the engine
side figure.

## 6. Credits

* **FujitsuPolycom/sparkring** (Apache-2.0) — the switchless-ring NCCL patch set,
  the prebuilt `libnccl.so.2.30.7` artifact used here, the dual-HCA channel
  configuration, the Engram `BALANCED`/packed-shard work (reported above as **not**
  transferring to our setup), and their soak/validation methodology.
* **Tech2Wild/Kai** — the four-node foundation this recipe builds on.
* **0xTank** — the graph startup-state fix used in this configuration.

Reproduce at your own risk on your own ring; the numbers above are ours, on our
hardware, with the checkpoints and builds named in the top-level README.
