# 2026-10-10 — the SGLang ring lane on a second fleet: four HCAs, RoCEnante over the mesh, NFS weights

Contributed by [colonel-otto/4x-dgx-spark-ring](https://github.com/colonel-otto/4x-dgx-spark-ring) (recipe, logs and
scripts in `dsv41-flash/`). Same engine and profile as
[`2026-09-28-sglang-switchless-ring-tp4.md`](2026-09-28-sglang-switchless-ring-tp4.md) (MiaAI `canary-roce`
production line, TP4, EP 1, DSpark k=5, 1M context, 16 running requests, chunk 4096, 0.80 static, 8M pin), on a
different four-GB10 switchless ring, with three changes. Our harness differs from yours and from sparkDash, so read
per-cell numbers as indicative across projects; the A/B columns below are like-for-like within this fleet.

## 1. What is different from the 2026-09-28 lane

| | 2026-09-28 lane | this fleet |
|---|---|---|
| NCCL | sparkring switchless overlay, 2 HCAs (one PCIe half per cable) | switchless-nccl v0.0.1 + [ondigo-winder 4-HCA patch](https://github.com/ondigo-winder/SWITCHLESS-NCCL-RING-4-GB10): all 4 RDMA functions, 4 channels |
| TP collectives | NCCL (`SGLANG_ROCE_ALLREDUCE=0`) | RoCEnante ≤80 KB over sparkring's hardware-forwarded opposite-node paths (`DSV41_ROCE_RING=1`), NCCL above |
| Weights | node-local copies (`NFS_SHARE=0`) | `NFS_SHARE=0` launcher mode, but each worker's volume is NFS from the head over the fabric; opposite node routed through a neighbour |
| Engram | per-rank packed shards, local | same; packed once on the head for all ranks, copied to each node |
| Kernels / driver | 6.17 / 580.173 on all four | 6.17 / 580.173 on three, 7.0 / 580.178 on one |
| Clocks | pinned `-lgc 2400` | not pinned (nodes ran ~2.4–2.5 GHz) |

Two points from the 2026-09-28 note this fleet answers differently:
- *"The RoCE all-reduce path must be off on a ring"* holds only without the opposite-node paths. With sparkring's
  `cx7_hairpin_diagonal` mesh (planner in MiaAI `scripts/ring_mesh/`, default `hairpin_queue_size` 1024, 80 KB cap,
  no devlink re-init) RoCEnante runs on the ring: 61 KB `ib_write_lat` to the opposite node 10.0 µs, all forwarded
  packets in the neighbour's hardware.
- *"A ring node that reads the checkpoint across the fabric competes with its collectives"*: here the checkpoint is read
  only at load (fast loader, ~120 s per rank, 889 MB/s to the routed opposite node); serving traffic is unaffected
  because Engram rows are never read over NFS.

## 2. Results (thinking off, sampled temp 0.6, 128-token answers, two passes, unique prompt prefixes)

| | NCCL only, 4 HCAs | + RoCEnante mesh |
|---|---:|---:|
| qeval (MiaAI, 75 tasks, temp 0) | 72/75, median 73.0 / 76.9 tok/s | 72/75, median 79.8 tok/s |
| C1 decode (per stream) | 114.8 / 115.7 | 126.7 / 132.0 |
| C4 / C8 / C16 aggregate (median of passes) | 187 / 256 / 400 | 183 / 288 / 414 |
| TTFT C8 / C16 | 630 / 1168 ms | 790 / 1335 ms |
| Cold TTFT 32k / 131k (repetitive filler) | 6.0 / 25.1 s | 6.0 / 24.3 s |
| Cold prefill, natural text ~28k / ~110k | | 4.4–4.8k / 4.2–4.7k tok/s |
| KV pool | 6,647,808 | 6,404,864–6,503,424 |

Short answers (128 tokens), so C1 here is not the long-answer idle case your boundary table warns about. All runs fail
the same three qeval tasks as MiaAI's published runs (`code_interval_intersect`, `json_escape`, `math_m9`).

## 3. `DSV41_CACHE_GIB=0` on this fleet

Your §4 measured +9 % KV pool at 0. Here, on the mesh line, two passes each against 4 GiB:

| | 4 GiB | 0 |
|---|---:|---:|
| KV pool | 6,503,424 | 6,738,944 (+3.6 %) |
| qeval median | 79.8 | 82.3 |
| C1 decode | 126.7 / 132.0 | 127.9 / 123.9 |
| C16 aggregate | 405.1 / 423.6 | 389.4 / 387.6 |
| natural-text prefill ~28k / ~110k | 4.4–4.8k / 4.2–4.7k | 4.6k / 4.6k |
| repetitive-filler TTFT 32k / 131k | 6.0 / 24.3 s | 6.8 / 27.6 s |

Agrees with MiaAI#21 that the cache's prefill gain is a filler artefact (natural text unchanged); on this fleet the pool
gain was smaller and C16 slipped ~6 %, so we kept 4 GiB. At a 6.4M+ token pool the extra KV does not change what fits.
