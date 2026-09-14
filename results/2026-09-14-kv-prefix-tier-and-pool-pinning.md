# 2026-09-14 — a restart-survivable KV prefix tier, pinning the KV pool, and two cross-repo results that did not transfer

Three pieces of work, measured on the same four-node TP=4 ring, same checkpoint, same
server build:

1. **A restart-survivable prefix cache on node-local NVMe.** The tier that gives
   multi-turn reuse was already in place on 2026-09-13; what was still broken is that
   its index lived in the engine's memory, so **every restart orphaned the KV sitting
   on disk** — and this configuration is restarted often. This note covers the fix, what
   it is worth, and the three ways the fix can silently corrupt output if written wrong.
2. **Pinning the KV pool** (`--kv-cache-memory-bytes`) so capacity stops being a
   lottery: the pool was observed at six different values across today's boots with
   identical flags and identical free memory. With the pin, it is one value, and it is
   **23 % larger** than the mean of the unpinned readings.
3. **A cross-repository transfer test.** A well-documented two-node recipe for a
   different model on the same GB10 hardware claims two easy wins. Both were tested
   under an interleaved A/B and **neither reproduced here**; the report is kept because
   the *conditions* under which each claim holds are now known, and because the test
   itself is a reusable protocol.

Rule for this report, same as the previous one: **a number is claimed only when it
survives repetition.** Where a claim did not survive, it is labelled as such.

---

## 1. The problem: a KV tier whose index dies with the process

Credit for the tier itself goes to **yunwei37/dgx-spark-4-ring-no-switch** (MIT):
a single-file, out-of-tree vLLM connector (`dsv41_kv_nvme.py`) that parks cacheable KV
groups in a per-node NVMe slot file and serves them back by block hash. Wiring it in
takes three CLI artifacts and no source-tree change, which is why it was adopted.

Measured on adoption (2026-09-13/14): a **92,429-token** prompt that had just been
processed (cold) took **59.87 s**; the *same* prompt resent took **0.65 s**, and the
same prompt with one sentence appended took **0.66 s** — roughly **92×**. That is the
multi-turn behaviour the engine's own prefix cache does not deliver on this model's
compressed-cache ring (its aggregate hit counters move, but a 92 K resend still costs
a full prefill).

The gap: the block table (hash → slot) is **in-process**. Restart the engine — which
happens on every configuration change here — and the NVMe file still holds gigabytes of
valid KV that nothing can address.

## 2. The fix: an out-of-tree cache policy that snapshots its index

vLLM supports plugging the offloading manager's eviction policy from outside the tree
(`cache_policy` + `cache_policy_module_path`, resolved by class name). The fix is a
policy subclass plus a spec subclass, mounted next to the tier module and required by
name in the connector config — **no vLLM source changes**:

```jsonc
"kv_connector_extra_config": {
  "spec_name": "PersistentNVMeOffloadingSpec",       // our subclass
  "spec_module_path": "kv_persist_policy",           // this repository's tools/
  "cache_policy": "PersistentLRUCachePolicy",        // our subclass
  "cache_policy_module_path": "kv_persist_policy",
  "nvme_dir": "/kv-offload", "nvme_bytes": 68719476736, "offload_prompt_only": false
}
```
```bash
-e KV_TIER_STATE_PATH=/kv-offload/kv-index.json   # where the index is written
-e KV_TIER_TAG=r0                                 # per-rank namespace
-e KV_TIER_FINGERPRINT=<checkpoint-and-config-id>  # invalidates on any change
```

The published file is [`tools/kv_persist_policy.py`](../tools/kv_persist_policy.py)
(MIT, ~180 lines).

**The three traps, because we hit all three:**

1. **Do not route through the stock `get_manager()`.** The base spec's manager reads its
   policy from a *different* config key (`eviction_policy`) and **never passes
   `cache_policy_module_path`**. Calling `super().get_manager()` therefore returns a
   manager with the built-in LRU policy while the snapshot thread runs against it and
   logs, every 20 s,
   `index snapshot failed ('LRUCachePolicy' object has no attribute '_dirty')`.
   The subclass must build the manager itself.
2. **Restore the allocator state, not just the hash table.** Key → slot alone is not
   enough: the manager's free-list and allocated-block counters must come back too, or
   a live block gets handed out again and a request reads **another request's KV
   (silently wrong output, no error)**. The published file restores
   `_free_list`, `_num_allocated_blocks` and `_num_evictable_cache_blocks`.
3. **Only snapshot blocks whose store completed**, and **fingerprint** the index
   (checkpoint, layout, slot geometry, KV dtype, tag). A half-written slot must never
   be addressable, and a KV page written under a different checkpoint must never be
   reused. Any failure in the restore path degrades to a cold cache rather than failing
   the boot.

## 3. What it is worth (measured)

| what | before | after |
|---|---|---|
| 92,429-token prompt, **first request after an engine restart** | 58 s (full prefill) | **0.86 s** |
| same, same-process resend | 0.65 s | 0.65 s |
| 20,202-token prompt over the public path, first call after restart | 13.6 s | **2.42 s** |
| boot log | — | `restored 183974 cached blocks from /kv-offload/kv-index.json` |

The index file is written every 20 s while traffic is flowing: **2.0 MB** after the first
window, **16.5 MB** with 191,962 cached blocks after a day of benchmarking. It is
bounded by the tier's slot file, not by request count — the eviction policy is still an
LRU over the same slot count, so a full tier evicts and the index shrinks with it.

Gate results for the pinned/persistent configuration are unchanged from the previous
report (garble 30/30, refusal behaviour as configured, quality 29/30).

## 4. Pinning the pool: measure first, then pin

Unpinned, with `--gpu-memory-utilization 0.83` held constant, the engine reported these
pools across today's boots: **1,460,513 / 1,638,036 / 2,202,903 / 2,537,044 / 2,609,348**
tokens. The knob that claims to size it is not the whole story — the engine's own
startup line shows why:

```
Free memory on device (105.9/121.69 GiB) on startup. Desired GPU memory utilization is (0.83, 101.0 GiB).
Actual usage is 91.5 GiB … Available KV cache memory: 8.85 GiB
```

On unified memory the driver needs pages that are **free**, not reclaimable, so what is
left after weights and graph capture moves with host state — and the pool moves with it.

`--kv-cache-memory-bytes` removes the lottery. **The sizing recipe is: read
`Available KV cache memory` from one unpinned boot, then pin slightly below it.**

| | unpinned | pinned at 8.5 GiB (9,126,805,504 B) |
|---|---|---|
| KV pool | 1.46 M – 2.61 M tokens (varies per boot) | **2,706,122 tokens, every boot** |
| server log | prints `Available KV cache memory` | does not print it (the pin sizes the pool) |
| decode c1 · code prompt | 58.9 / 62.7 tok/s | 63.0 / 60.3 tok/s |
| aggregate c6 | 132.8 / 144.5 tok/s | 168.4 / 168.9 tok/s |
| cold prefill, ~8 K TTFT | 10.1 / 10.7 s | 10.1 / 10.0 s |

An earlier attempt at this flag **failed to boot at three values** — the failure was
sizing above the real budget, not the flag. On unified memory an over-commit needs a
power cycle, so measure before pinning.

## 5. Boot hygiene: the other half of the pool lottery

If the pin removes variance *at the engine*, a gate removes it *before the engine*.
Adopted from the same two-node recipe (their `wait_mem` + page-cache eviction):

```bash
# after the previous containers are removed, before launching:
#   wait until every node reports MemAvailable >= 100 GiB, then evict page cache
sync; echo 3 > /proc/sys/vm/drop_caches     # root; without root:
#   find <weights dir> -name '*.safetensors' -exec dd if={} iflag=nocache count=0 \;
```

Measured: nodes report **106–110 GiB** available after the container dies and
**108–111 GiB** after eviction, and the boot then passes the gate immediately. The
reason it matters is the engine line in §4: unified memory takes tens of seconds to come
back after a container is killed, and launching into that window is how a boot ends up
with a phantom OOM or a smaller pool.

## 6. Cross-repo transfer test: two claims tested, neither reproduced

Source: **bilikaz/qwen38-flash-next-cluster-recipe** (two GB10 nodes, TP=2, a different
model, upstream vLLM 0.29 + six patches). Two of its host-level claims are cheap to test
and were tested here under an interleaved A/B.

### 6.1 `vm.compaction_proactiveness=0` — claim: ~10 % on a tightly-pinned serve

Their mechanism is specific and therefore falsifiable: background compaction migrates
pages, on a Spark the GPU's memory *is* those pages, so they see **a 4–5 s stall every
~37 s**. Test: four consecutive 1,500-token decodes (~66 s, spanning two such cycles)
with per-chunk inter-arrival times recorded.

| `compaction_proactiveness` | median gap | p99 | max | gaps > 0.5 s | tok/s |
|---|---|---|---|---|---|
| 20 (default) | 64 ms | 75 ms | **149 ms** | **0** | 89.0 / 90.6 / 90.9 / 92.5 |
| 0 | 64 ms | 114 ms | 708 ms | 1 | 91.6 / 91.7 / 92.3 / 81.5 |

**No stall exists here to remove.** The condition difference is visible in their own
write-up: they run with ~1 GB free; this deployment keeps 3–6 GB available, so the
compactor is not the actor it is on their box. The setting is kept (it is free, it is
persisted in `/etc/sysctl.d/`, and it is one command to revert) but **it produced no
measurable gain here** and is not claimed as one.

### 6.2 Pinning to the fast cores (`--cpuset-cpus`) — claim: +2–3 % at every concurrency

The topology claim checks out exactly: cores 0–4 and 10–14 run at 2808 MHz, cores 5–9
and 15–19 at 3900 MHz, so `--cpuset-cpus=5-9,15-19` is the right set. Interleaved A/B,
three pairs, containers updated in place:

| | single stream (fixed-length count task) | aggregate c6 |
|---|---|---|
| unbound | 84.6 tok/s | 166.9 tok/s |
| bound to 5–9,15–19 | 91.3 tok/s | 162.5 tok/s |

Both directions move; **ranges overlap** (79.1–92.0 vs 76.2–92.1 on single stream).
With bound-at-boot containers the same spread appeared (single 60.6/68.5, c6 151.7/156.5).
**A ±2–3 % claim cannot be resolved at this deployment's ±10 % run-to-run noise floor**,
and it is not published as a gain. It is wired in (the container flag is parameterised
and can be disabled) so that a future measurement can be taken at boot, where the
original was measured.

### 6.3 What did transfer

* the **boot memory gate** (§5),
* the **pool pin** (§4),
* and a **verification habit**: their viewer samples the HCA port counters against the
  interface's TCP byte counters to prove RDMA is actually being used instead of
  silently falling back to TCP. Same test here, published as
  [`tools/rdma_proof.sh`](../tools/rdma_proof.sh): during decoding all four nodes move
  **69–78 MB/s per direction on the RoCE ports with the management-NIC TCP counters at
  0** — RDMA is real, and the test is now part of acceptance.

## 7. Measurement discipline (two self-inflicted errors worth publishing)

1. **±10 % is this deployment's noise floor.** Same configuration, same prompt, same
   day: single-stream decode across repeated runs was 52–68 tok/s; code prompts and
   prose prompts differ by ~2× because speculative acceptance differs. Any claim of a
   few percent needs **interleaved A/B** (alternate the two configurations within one
   session) — a single before/after pair is worthless at this scale, which is exactly
   why §6.1 and §6.2 are reported as unresolved rather than as wins.
2. **Counting streamed chunks is not counting tokens.** With speculative decoding the
   server emits *several tokens per SSE delta*; a client that counts deltas
   under-reported a 384-token answer as 95 and produced a plausible-looking
   "14.9 tok/s". Request `stream_options: {"include_usage": true}` and divide the
   server's `completion_tokens` by the decode wall time. A second variant of the same
   class of bug — dividing a token count by a wall time that already contained the
   prefill, and then dividing again — produced a 12 tok/s aggregate on a stack that was
   actually doing 165.

## 8. Current state of this configuration

```
window 1,048,576 · CUDA graphs FULL_AND_PIECEWISE · vision + tools · DSpark k=5
KV pool 2,706,122 tokens (pinned, deterministic)   ·  prefix tier restart-survivable
single stream 60–63 tok/s (code) · 91 tok/s (repetitive text) · aggregate c6 169 tok/s
cold prefill ~10 s for a small unique prefix · queue 0/0 idle · GPU 49 °C, SM 2190 MHz
```

Unchanged by today's work: single-stream and aggregate throughput (the pin and the gate
buy capacity and determinism, not speed). What would move throughput is a different
engine's prefill scheduling — and that is a porting project, not a flag.
