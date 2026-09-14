#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Streaming client-side probe for a served LLM: throughput, first-token latency and
long-decode stalls -- measured so that the numbers are actually token counts.

Two traps this tool exists to avoid (both cost us a published number):

1. Counting SSE deltas is NOT counting tokens. With speculative decoding the server
   emits several tokens per chunk; a delta-counting client reported a 384-token answer
   as 95 tokens and printed a plausible-looking 14.9 tok/s. This tool asks for
   `stream_options: {"include_usage": true}` and divides the server's
   `completion_tokens` by the decode wall time.
2. Dividing a token count by a window that already contains the prefill (and then
   dividing again) produced "12 tok/s" on a stack that was doing 165. Decode rate here
   is always `completion_tokens / (stream_end - first_token)`.

It also reports the distribution of inter-chunk gaps, which is how an interference
mechanism (e.g. host-side page compaction on a unified-memory box) is confirmed or
ruled out: a claim of "a 4-5 s stall every ~37 s" must show up as gaps, or it is not
happening on that machine.

Usage:
  python3 stream_bench.py --base http://127.0.0.1:8888/v1 [--model M] [--conc 1,6]
                          [--max-tokens 384] [--rounds 2] [--gap-test 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
import urllib.request

CODE = ("Write a Python implementation of a thread-safe LRU cache with TTL expiry, then a "
        "short explanation of the locking strategy and two edge cases you handled.\n\nnonce=%s")
COUNT = ("Count from 1 upward, one number per line, as far as you can. nonce=%s")
PROSE = ("Explain how a B-tree differs from an LSM tree, in about 400 words. nonce=%s")
PROMPTS = {"code": CODE, "count": COUNT, "prose": PROSE}


def stream(base, model, prompt, max_tokens, timeout=900):
    """Returns (ttft_s, decode_tps, server_tokens, gaps)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    first = None
    end = None
    usage = 0
    gaps = []
    prev = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                end = time.time()
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"].get("completion_tokens", 0) or usage
            delta = ((d.get("choices") or [{}])[0].get("delta") or {})
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.time()
                if first is None:
                    first = now
                    ttft = now - t0
                else:
                    gaps.append(now - prev)
                prev = now
    if end is None:
        end = time.time()
    tps = usage / (end - first) if (first and usage and end > first) else 0.0
    return ttft or 0.0, tps, usage, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default=None, help="default: first entry of /v1/models")
    ap.add_argument("--prompt", default="code", choices=sorted(PROMPTS))
    ap.add_argument("--conc", default="1,6", help="comma-separated concurrency rungs")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--gap-test", type=int, default=0,
                    help="N long sequential decodes; reports stall distribution")
    ap.add_argument("--gap-tokens", type=int, default=1500)
    a = ap.parse_args()

    model = a.model
    if not model:
        with urllib.request.urlopen(a.base.rstrip("/") + "/models", timeout=20) as r:
            model = json.load(r)["data"][0]["id"]
    tpl = PROMPTS[a.prompt]
    print(f"=== {a.base} · model {model} · prompt '{a.prompt}' ===")

    for c in (int(x) for x in a.conc.split(",")):
        runs = []
        for _ in range(a.rounds):
            t0 = time.time()
            with cf.ThreadPoolExecutor(c) as ex:
                out = [f.result() for f in
                       [ex.submit(stream, a.base, model, tpl % random.random(), a.max_tokens)
                        for _ in range(c)]]
            wall = time.time() - t0
            agg = sum(x[2] for x in out) / wall
            runs.append((agg, statistics.median([x[0] for x in out]), sum(x[2] for x in out)))
        agg = statistics.median([r[0] for r in runs])
        ttft = statistics.median([r[1] for r in runs])
        print(f"  c={c:<2} aggregate {agg:7.1f} tok/s   "
              f"(rounds: {', '.join(f'{r[0]:.1f}' for r in runs)})   TTFT {ttft:.2f}s   "
              f"server tokens/round {runs[0][2]}")

    if a.gap_test:
        print(f"  --- stall probe: {a.gap_test} sequential decodes x {a.gap_tokens} tokens ---")
        allgaps = []
        for i in range(a.gap_test):
            ttft, tps, ntok, gaps = stream(a.base, model, COUNT % random.random(),
                                           a.gap_tokens, timeout=1200)
            allgaps += gaps
            over = [g for g in gaps if g > 1.0]
            print(f"    run {i+1}: {ntok} tok {tps:6.1f} tok/s · TTFT {ttft:.2f}s · "
                  f">1s gaps {len(over)}" + (f" (max {max(over):.2f}s)" if over else ""))
        if allgaps:
            s = sorted(allgaps)
            print(f"    gaps: median {statistics.median(allgaps)*1000:.0f}ms · "
                  f"p99 {s[int(len(s)*0.99)]*1000:.0f}ms · max {max(allgaps)*1000:.0f}ms · "
                  f">0.5s {len([g for g in allgaps if g > 0.5])} · >1s {len([g for g in allgaps if g > 1.0])}")
        else:
            print("    no gaps recorded (single chunk?)")


if __name__ == "__main__":
    sys.exit(main())
