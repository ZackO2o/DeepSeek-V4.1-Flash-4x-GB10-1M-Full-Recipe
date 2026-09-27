# 2026-09-28 — 4 × GB10 switchless-ring, cross-project reference table

**Read this as a reading list, not a leaderboard.** Every row below was published by
its own project, measured with that project's own harness, prompt set and checkout.
Different harnesses change the same metric by 20–25 %, so the numbers are **not
interchangeable** and this page does not rank them. What it is good for: finding the
project whose hardware and engine match yours, and knowing which number in its README
is the one to reproduce first.

Our own like-for-like comparison (same harness, both sides) lives in
[`2026-09-28-sglang-switchless-ring-tp4.md`](2026-09-28-sglang-switchless-ring-tp4.md);
this page is the wider context around it.

## 1. DeepSeek-V4.1-Flash on 4 × GB10

| Project | Engine / fabric | Published headline (as written by the project) |
|---|---|---|
| **MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks** (the recipe this repo builds on) | SGLang, switched production **and** a switchless-ring variant | Switched: prose C1 **87.7** / code C1 **124.8** tok/s, C1–C16 aggregate 87.7 / 120.4 / 163.6 / 237.6 / 342.7; cold prefill 4,059–5,925 tok/s (4k–262k); 1,011,084-token needle PASS; qeval 72/75. Ring variant with their own `decode_window.py`: C1 **74.03**, C8 **224.84** aggregate; 8 × 6,000-token streams → 215.81 tok/s wall-clock, engine batch peak 258.63 |
| **luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring** | vLLM (W4A4 MoE path), switchless ring | Pure-prefill peak **5,823.1 t/s** (65,536 × C2), 512 K single-stream 5,042.2; decode aggregate peak **572.6 t/s** (code, C16), C1 peak 89.19 |
| **nero-/deepseek-v41-flash-gb10-ring** | SGLang, switchless ring | Decode 81.8 / 76.7 / 78.8 tok/s; coding peak **97.5**, median 80.4; qeval 72/75; per-step breakdown published (profiler on rank 0) |
| **ChrisLou-bioinfo/dsv41-4x-spark-tutorial** | SGLang, ring (tutorial form) | Single stream **45**, 4-stream aggregate **103**, 128 K prefill 3.2 K tok/s, 1 M usable. Carries the same parameter set as this repo's ring lane (`0.80` / `MAX_RUNNING 8` / `8,000,000` pin / chunk 1024 / `TP_PAD 0`) |
| **yunwei37/dgx-spark-4-ring-no-switch** | vLLM, switchless ring, multi-model log | V4.1-Flash, DSpark k=5, gmu 0.83: **993,435-token needle correct at 1,048,576**, KV pool 2,058,026; **50.14 tok/s C1 mean, 68.06 coding, 144.71 aggregate at 8 requests**; 87–92 % production prefix-cache hit rate. Same repo also logs GLM-5.3 and Qwen3.8 lanes |
| **ours** (this repo) | SGLang, switchless ring, `canary-roce` profile + sparkring NCCL | C1 **104.10** / C2 147.56 / C4 190.00 / C8 **245.98** aggregate with the upstream harness; KV pool 3,253,248 tokens; details and method in the sibling note |

Models that appear at 27 B–30 B class: several of these projects also serve
`GLM-5.3-Flash` and `Qwen3.8-Flash-Next` on the same four-node rings, which is a useful
sanity check on what the fabric itself is worth:

| Project | Model | Published |
|---|---|---|
| ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless | GLM-5.3-Flash **NVFP4** | Single-stream **94.7 t/s peak / 75.9 avg** (thinking ON); 400 K cold prefill TTFT 191.4 s / 2,096.9 tps |
| yunwei37/dgx-spark-4-ring-no-switch | GLM-5.3-Flash FP8 (zai-org) | 240,000-token retrieval passed; **20.16 tok/s** single stream; MTP=5 variant 25.57 tok/s |
| yunwei37/dgx-spark-4-ring-no-switch | Qwen3.8-Flash-Next NVFP4, TP2 | **40.20 tok/s** single, **102.19** at 4 requests (262,144 window) |
| yunwei37/dgx-spark-4-ring-no-switch | GLM-5.3 Int4/Int8Mix | 8,192 bounded baseline: 12.90 tok/s mean (near-200 K prefill crossed the memory wall) |

## 2. The lanes we run

Same fleet, one gateway in front, model chosen per workload. Numbers below are ours,
measured on each lane with **that lane's own quick harness**, so they describe our
deployments rather than being comparable to the tables above:

| Lane | Model | Engine / hardware | What we use it for |
|---|---|---|---|
| Flagship long-context | `deepseek-ai/DeepSeek-V4.1-Flash` (official FP8) | SGLang TP4, 4 × GB10 switchless ring | 1 M-context work, vision + tools, coding; the subject of this repo |
| Second long-context lane | `deepseek-ai/DeepSeek-V4.1-Flash` | vLLM TP4, 4 × GB10 | A/B against the SGLang lane; carries the recipe in the main README |
| Fast 27 B-class | `Qwen3.8-Flash-Next` | SGLang TP2, 2 × GB10 ring | shorter-context bulk work |
| Cost lane | 27 B-class FP8, 2 × 2080 Ti | vLLM fork (FP8 KV, MTP; dequant path — SM 7.5 has no native FP8) | throughput per card on older hardware |
| Backup lane | `GLM-5.3-Flash` | EXL3 on 2 × GB10 | second vendor path, long context |

The point of publishing this: the *fabric* is the reusable part. Once the ring works,
the same four nodes serve a 1 M DeepSeek lane, a GLM lane or a Qwen lane with different
engines — and the cross-project rows above are where to look for the configuration of
each combination.

## 3. How to read a number from another project

1. **Find the harness.** If it is `decode_window.py`-style, per-stream values are
   character-scaled estimates; the aggregate column (server usage ÷ wall time) is the
   one that survives comparison. If the project says "engine batch peak", that is an
   internal counter, not end-to-end throughput — it will read higher than anything a
   client can observe.
2. **Find the checkout.** `chunked prefill 1024` vs `4096`, `max_running_requests 8` vs
   `16`, and spec-decode `k=3` vs `k=5` each move the result more than most hardware
   differences do. Two projects can publish 45 and 105 tok/s for the same model on the
   same boxes and both be honest.
3. **Find the clock policy.** GB10 defaults are not the same as a pinned `-lgc`; a node
   measured at 2,177–2,190 MHz against one at 2,392–2,398 MHz is a ~9.5 % gap before
   anything else is compared.
4. **Find the prompt.** A 256-token completion is dominated by TTFT and prefill; a
   6,000-token completion is not. Cross-project tables that mix both are not tables.
5. **Then reproduce one row.** Pick the number closest to your own config and re-run
   their harness on your boxes. That single action is worth more than every table here,
   including ours.

## 4. Sources

All values above were read from each project's own README / results documents on
**2026-09-28**. Repository names move — if a link 404s, search the project by its title
rather than trusting a stale path.

- MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks — `README.md`, `docs/tp4-switchless-ring-results.md`
- luxingcom/LuZ-0.1.7-DeepSeek-v4.1-Flash-DGXspark-TP4-Ring — `README.md`
- nero-/deepseek-v41-flash-gb10-ring — `README.md`, `RESULTS.md`
- ChrisLou-bioinfo/dsv41-4x-spark-tutorial — `README.md`
- yunwei37/dgx-spark-4-ring-no-switch — `README.md`
- ntxf31415/glm-5.3-flash-nvfp4-4x-dgx-spark-switchless — `README.md`
