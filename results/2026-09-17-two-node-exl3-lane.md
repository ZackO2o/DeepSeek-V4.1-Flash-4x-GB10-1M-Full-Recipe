# A two-node lane: EXL3 2.9 bpw, prefilled Engram, and the knob that moved the memory wall (2026-09-17)

This note is **not** a second 4-node measurement. It is a different lane, published because two of
its findings are about *why* the 1M configuration fits, not about how fast it goes.

## The lane

| | |
|---|---|
| Nodes | **2 × GB10**, 128 GB unified memory each (121.69 GiB visible) |
| Parallelism | TP = 2 (so ~98 GiB of weights per rank) |
| Base pack | **EXL3 2.9 bpw**, 39 shards, `total_size` **210,713,013,432** (196.1 GiB) — a quantized pack, not native FP8 |
| Context | **600,000** (`--max-model-len 600000`) |
| Features | vision + tools on, `cudagraph_mode FULL_AND_PIECEWISE`, DSpark k=3, thinking off |
| Engram | **packed to node-local NVMe** (`start.sh pack`, 47.2 GiB per table per rank ≈ 94 GiB/node, 456 MiB/s, ~8 min) |
| Ancillary | cooperative MoE kernel, built locally and gated (see below) |

Everything here was measured on this lane. Absolute decode/prefill numbers are **not** comparable to
the 4-node native-FP8 tables in the main README — the interesting part is the direction of the two
effects below.

## Finding 1 — `--max-num-batched-tokens` moved the memory wall further than `--gpu-memory-utilization` did

We hit the same wall the main README documents ("600 K does not boot at 0.80"): our preflight came up
short by about **1 GiB**. Walking GMU down did not fix it — 0.88 → 0.86 → 0.84 shifted the wall rather
than removing it (each step bought back roughly what the smaller pool gave up).

What actually restored headroom was raising the batched-token budget from **1024 → 1536**:

| `MAX_NUM_BATCHED_TOKENS` | `MemAvailable` at boot (head) |
|---|---|
| 1024 | **0.09 GiB** |
| **1536** | **3.50 GiB** |

A *larger* chunk left *more* headroom. The plausible mechanism is scheduling rounds: a bigger chunk
means fewer prefill scheduling iterations and less transient workspace alive at the same time. This
also happens to be the value a vision tower needs (a single image item can cost ~1025 tokens in a
chunk), so on this model family it is not a trade — it is the value you want anyway.

**Caveat, stated plainly:** measured on a 2-node lane with a quantized pack. I have not reproduced it
on 4 nodes with native FP8, which is why it is an issue comment / results note rather than a change to
the main tuning table.

## Finding 2 — `MemAvailable` on GB10 is a floor, not a gauge

After a full-length prefill on this lane the head reports **2.35 GiB** `MemAvailable`; during serving it
frequently reads **0.00 GiB** while the server answers normally, with **zero** real OOM kills
(`journalctl -k` free of `oom-kill` / `Out of memory: Killed`).

The reason is the one the main README already gives for the pool behaving oddly: on GB10 the ~98 GiB of
weights live in **driver allocations that are not charged to any process RSS**, so the kernel's
"available" estimate has little left to count and pins near zero. Two consequences:

* **A watchdog keyed on `MemAvailable` needs a fleet-specific threshold.** The same quantity reads
  **10.3 GiB** on the 4-node fleet (issue #1) and **~0** here. A threshold ported between fleets will
  either never fire or fire constantly.
* **The signals that do survive the port:** a real `oom-kill` line from the kernel, a container
  health/restart transition, and the low-water mark *after a long prefill* (which is where the floor
  actually is — not at boot).

## Measured on this lane

| Quantity | Value |
|---|---|
| Weights per rank | ~98 GiB of the 121.69 GiB node budget |
| `MemAvailable` boot → after a long request | 3.50 → **2.35 GiB** |
| prefill, packed Engram, zero cached tokens | **966–1021 tok/s** (25 K and 100 K prompt tokens) |
| decode, single stream | **35.9–37.9 tok/s** |
| needle, exact match | **128 K PASS** (97.9 s) |
| KV pool | 2.14 M tokens at 600 K (≈ 3.6× the window) |

One ordering effect worth recording: **before** the Engram tables were packed to node-local NVMe, a
**32 K** needle probe failed outright on this lane. After packing, **128 K** passed. On a two-node lane
the Engram row-store is not a throughput nicety — for us it was the difference between "long context
does not work" and "long context works".

## Local extension gate (for anyone serving a quantized pack with a self-built kernel)

If you build a vendor extension yourself (we built the cooperative-MoE kernel because the release
artifact was not published), the vendor's own gate is the thing to run, per node, before believing the
build:

* the scripted integration gate must report `54` checks with `status: pass` **and** exit 0 **on each
  node** — one node passing does not imply the other;
* the chained pins must all be updated in one pass (a binary pin, the adapter's own embedded
  `SHA256`, and the adapter hash that the profile generator verifies);
* a locally rebuilt `.so` is **not** expected to hash-match the release pin — the vendor says so
  explicitly, and three builds of the same source produced three different hashes. The gate, not the
  hash, is what licenses the substitution.

## What this note does not claim

* It does not claim the 4-node tables are wrong — different lane, different base pack.
* It does not claim `MAX_NUM_BATCHED_TOKENS` is a universal lever; it is one measurement on one fleet.
* It does not claim a pool floor that transfers: our 600 K pool sits at 3.6× the window, and the main
  README's warning that *"every extra 0.5 GiB of KV pool costs ~1 GiB of head prefill margin"* applies
  here exactly as written.
