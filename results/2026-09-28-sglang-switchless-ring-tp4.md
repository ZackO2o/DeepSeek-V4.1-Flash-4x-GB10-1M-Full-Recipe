# 2026-09-28 — a switchless-ring TP4 lane for DeepSeek-V4.1-Flash (SGLang), measured against the upstream ring numbers

A second lane, same fleet class, **different engine**: SGLang's 1M TP4 profile on a
**switchless 4-node ring** (no RoCE switch — four GB10 nodes wired point to point).
This note publishes what we measured on it, how it compares with the ring numbers
the upstream project published for the *same harness*, and three operational findings
that cost us a night.

The README's headline configuration is the **vLLM** lane. This is the other one.
Nothing here replaces the vLLM numbers — the two are different engines and, more
importantly, **different measurement harnesses**, so they are not comparable with each
other (see §5).

## 1. The lane

| | |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4.1-Flash` (official FP8 checkpoint, unchanged) |
| Hardware | 4 × GB10 / SM 12.1, 128 GB unified memory each |
| Fabric | **switchless ring**, 200 GbE CX7, MTU 9000, one /30 per link |
| Engine | SGLang TP4, community `canary-roce` profile + `sparkring` switchless NCCL overlay (NCCL 2.30.7) |
| Context | `1,048,576` |
| KV pool | pinned `8,000,000`; **actually allocated 3,253,248 tokens** (see §4) |
| Running requests | `16` |
| Chunked prefill | `4096` |
| Spec decode | DSpark `k=5` |
| Memory fraction | `0.80` static (`+0.80` head) |
| Kernels / driver | `6.17.0-1029` / `6.17.0-1031-nvidia`, driver `580.173.02` on all four nodes |

Ring-only requirements that are easy to miss: the RoCE all-reduce path must be **off**
(`SGLANG_ROCE_ALLREDUCE=0`) because the ring carries collectives through the patched
NCCL overlay, the overlay must be a **flat** directory with the ring marker string in
it, and shared-storage weight loading should be off (`NFS_SHARE=0`) with node-local
copies — a ring node that reads the checkpoint across the fabric is competing with its
own collectives.

## 2. Measured decode (same harness as the upstream ring note)

Method: the upstream project's own `benchmarks/decode_window.py`, **unmodified**, run
against the fleet's gateway endpoint, greedy, thinking off, server idle
(`running=0 waiting=0`) before each wave. The aggregate column is wall-clock
`completion_tokens / elapsed` using the **server's** usage block; the per-stream column
is the harness's own character-scaled window estimate (see the caveat in §5).

| C | Aggregate tok/s | Median window tok/s / stream | TTFT | Success |
|---|---:|---:|---:|---:|
| 1 | **104.10** | 101.02 | 0.301 s | 1/1 |
| 2 | **147.56** | 70.68 | 0.409 s | 2/2 |
| 4 | **190.00** | 48.60 | 0.455 s | 4/4 |
| 8 | **245.98** | 33.80 | 0.518 s | 8/8 |

A repeat of the same sweep before the KV-pool change in §4 read
`105.1 / 139.19 / 201.70 / 252.40` aggregate at C1/C2/C4/C8 — i.e. the sweep is
reproducible to a few percent in both directions and the pool change did **not** move
decode speed (it is capacity, not speed).

## 3. Versus the upstream ring numbers, same harness

The upstream project publishes its own switchless-ring sweep, measured with the same
harness (`decode_window.py`) on a different four-node ring:

| C | Ours (aggregate) | Upstream ring (aggregate) | Delta |
|---|---:|---:|---:|
| 1 | **104.10** | 74.03 | **+41 %** |
| 2 | **147.56** | 111.48 | +32 % |
| 4 | **190.00** | 160.05 | +19 % |
| 8 | **245.98** | 224.84 | +9 % |

What is **held constant**: the same checkpoint family, the same harness, the same
metric definition, the same client path (HTTP streaming through a gateway), and the
same software generation of kernels/driver.

What is **not** held constant, and is why this is a comparison and not a claim of
architectural superiority:

1. **Checkout.** The upstream ring sweep was measured on an earlier revision of the
   SGLang tree (`chunked prefill 1024`, `max_running_requests 8`). Our lane runs the
   current `canary-roce` profile (`chunked prefill 4096`, chunked indexer, prefill
   sequence parallel, L2 prefetch, autotune-keep, `max_running_requests 16`). The
   TTFT gap in particular is mostly this.
2. **Clock policy.** Their nodes are capped by `nvidia-smi -lgc 0,2200` and measure
   **2177–2190 MHz**; ours are pinned with `-lgc 2400,2400` and measure
   **2392–2398 MHz** (+9.5 %). Their own documentation notes the cap is worth only
   about −0.7…−1.5 % on *bandwidth-bound* steps — so the clock explains a small part
   of the C1 gap, not all of it.
3. **Boxes.** Their number was taken on "Carlos's ring"; ours on ours. Cable, thermal
   and individual-node variation are not controlled and are not separable from the
   result.
4. **TTFT is the weakest column.** Their TTFT values were produced by a *legacy
   character-scaled estimator* (their own note says so); the harness as it stands today
   measures TTFT at the first non-empty content event, which is what our column is. Read
   the TTFT comparison as indicative, not as a 1.8× claim. **The aggregate column is the
   only strictly comparable one**, because both sides derive it from the server's final
   token usage.

## 3b. Cold prefill

Method: one request per size, a unique nonce per request (nothing cacheable), streaming,
`max_tokens=1` so the wall time is prefill plus one decode step; tok/s = the **server-reported**
`prompt_tokens` / TTFT. The filler is **high-entropy random words** — every token hashes to a
different Engram row — deliberately *not* the single-token synthetic filler some prefill tables use,
which on a `DSV41_CACHE_GIB=0` host mostly measures the row store rather than prefill (see §4).

| prompt tokens | TTFT | prefill tok/s |
|---:|---:|---:|
| 4,159 | 1.11 s | 3,732 |
| 16,327 | 3.53 s | 4,629 |
| 32,709 | 7.03 s | 4,656 |
| 65,176 | 13.91 s | 4,685 |
| 130,913 | 29.74 s | 4,402 |
| 261,416 | 64.11 s | 4,077 |

Peak is **~4,700 tok/s** in the 16k–64k band, and the curve is still flat where it matters: a longer
ladder on the same configuration measured 408,462 tokens in 109.1 s (**3,744 tok/s**) and
**818,377 tokens in 254.0 s (3,222 tok/s)** — no cliff through 800k tokens.

**A repeat-filler control** on the same configuration, at 25,250 and 100,868 tokens, read **5,041 and
5,054 tok/s** — the single-token filler is ~7 % *faster* here than the high-entropy one. That is a
different axis from the upstream report that motivated this method (they compared the *cache setting*
on the filler, where the filler was slow only with the cache off), so we record it as an observation
rather than a contradiction; repeated tokens also make the attention path degenerate.

**Where this lane sits.** The upstream project publishes **5,393–5,567 tok/s at 16k–128k and 4,971 at
262k** for its *ring* profile, measured with its own loader and its single-token filler. Our numbers are
about **15 % below** that (and ~7–9 % below it if read through our repeat-filler control) — the
**opposite direction from decode**, where this lane is ahead of the same project's ring figures. Our
numbers are TTFT-based from a local API client rather than their loader, so treat this as a lead rather
than a verdict: prefill is the first thing worth tuning on this lane, and the ring's large-prefill
collectives are where the upstream project says the fabric cost lands.

**Prefix cache, same lane:** repeating the identical 130,831-token request takes **0.51 s instead of
28.8 s** — a 56× TTFT reduction on a fully cached prefix.

## 4. KV pool: the pinned value is not the allocated value

`MAX_TOTAL_TOKENS` is pinned at `8,000,000`, but what the engine actually allocates
after CUDA-graph capture and the vision/tool path is far smaller. Two rows matter:

| `DSV41_CACHE_GIB` | KV pool allocated | host MemAvailable after boot |
|---|---:|---:|
| `4` (our template default) | 2,977,536 tokens | 13.67 GiB |
| **`0`** | **3,253,248 tokens** | **15.22 GiB** |

That is **+9 %** pool and ~1.5 GiB more headroom for the same configuration, from a
value the upstream project's current parameter table already carries as `0` for this
reason: on a GB10, **host RAM *is* GPU memory**, so a few GiB spent on a row cache are
worth more as boot/inference headroom than as hit rate. We had inherited `4` from an
older template.

Practical rule: measure the pool you get (`max_total_num_tokens` in the server's
startup log), not the one you asked for — the gap between the two is where "it booted
yesterday and not today" lives.

## 5. Operating a ring: three findings

### 5.1 Sample the NIC **carrier**, not the IB port state

One link of the ring was intermittently dropping. The IB port state on that device read
`4 (ACTIVE)` most of the time — the state that everyone checks — while the **NIC
carrier** on the same physical port was flapping `1 → 0 → 0 → 1 → 0` on a 4-second
sampling interval. The symptoms upstream of that were unmistakable and expensive: ~1300
`Got non-fatal async event` warnings from the RoCE async thread, `ibv_modify_qp` timeouts
during boot, and boots that died in the collective setup.

After physically re-seating the cable: carrier stable across repeated samples, port
speed back at 200 Gb/s, async-event warnings **1300 → 0**, and boots clean.

**Be honest about the value:** decode throughput did not move (58.9 → 58.6 on our own
quick harness). The win was removing a landmine, not gaining tok/s. If a ring boot is
flaky, sample `carrier` on every port before you blame NCCL — and sample repeatedly,
because one reading is worthless.

### 5.2 A hard-coded RoCE GID index expires when you touch the hardware

Our NCCL overlay was configured with `NCCL_IB_GID_INDEX=3`, which had been correct for
weeks. After the re-seat above, the **RoCEv2 IPv4 GID moved to index 4** on every node,
and the boot gate failed with *"no common IPv4 RoCE v2 GID"*. Nothing in the config
changed; the hardware state did.

Rule: after **any** cable, port, or power event, re-detect which GID index actually
carries the ring IPv4 address on every node and update the environment before booting.
Do not treat the index as a constant — it is a property of the current device state.

### 5.3 A soft NFS mount can decoy your health check

Two ring nodes reported their weight mount as present while `ls` on it blocked forever
— a stale entry from a soft mount that survived the share going away. A `mountpoint`
check says "mounted"; the boot then fails with *"weights missing/incomplete"*. Probes on
such a mount must be wrapped in `timeout`, and the health criterion must be
**artifact-shaped**, not mount-shaped: count the checkpoint shards (48 for this
checkpoint) instead of asking whether a path is a mount point.

## 6. Two negative results

1. **A third-party ring-only NCCL build did not transfer.** Another 4 × GB10 project
   ships a ring-optimized NCCL build; on our stack, boots with it fail with
   `NCCL error: invalid usage`. A single-variable test — **replace only the library,
   change no environment at all** — reproduces the failure, so the cause is the build's
   expectations rather than the tuning knobs that came bundled with it in the same
   experiment. That project documents a companion core-binding shim; we did not build
   it, and we did **not** isolate the extra NCCL knobs (buffer size, channel count,
   tuner threshold, protocol mask), so those remain **untested** here rather than
   "known-bad". Credit for the work stands — it simply does not drop into this lane.
2. **"Use all four HCA devices" presumes a network we do not run.** Advice to expose
   four RoCE devices per node instead of two fails our boot gate: it demands a common
   IPv4 RoCEv2 GID across all four devices, and the second plane's devices have no ring
   address on a single-plane ring. Splitting the ring across both planes (and giving
   both planes addresses) is a network reconfiguration, not a flag; we stayed at two
   devices and report the advice as inapplicable rather than wrong.

## 7. Reproduce

The benchmarks are the upstream project's, used unchanged; point yours at them rather
than re-implementing:

```bash
# idle cluster; thinking off; greedy
BASE_URL=http://127.0.0.1:8888 python3 benchmarks/decode_window.py
```

Our contributions are two probes, kept deliberately dumb:

- `tools/ring_link_health.sh` — per-port `carrier` / IB state / speed sampler
  (repeat N times; one reading proves nothing).
- `tools/post_reseat_recover.sh` — after a cable or power event: heal the weight mount
  by shard count, detect the current RoCEv2 IPv4 GID index, then restart and gate.

## 8. Current state

Serving `deepseek-v4.1-flash` at 1,048,576 context over the ring, KV pool 3,253,248
tokens, zero NCCL warnings across the last boots, decode as tabulated in §2
(reproduced twice). Vision, tool calling and the reasoning parser are live on the same
process; the ring change touches transport only.
