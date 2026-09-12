#!/usr/bin/env python3
"""Decode-at-depth benchmark for long-context serving.

Fills a prompt to a target token count (a filler document whose only purpose is to
occupy context), asks for a long generation, and reports:

  * TTFT and prefill tokens/s (measured against the server-reported prompt_tokens)
  * end-to-end decode tokens/s (from the server's completion_tokens)
  * four *reporting windows* across the generation, so you can see whether decode
    decays as the KV grows

Why the windows are exact-but-opt-in: with speculative decoding a single streamed
chunk can carry several tokens, so "chunks per second" is not "tokens per second".
Passing ``--logprobs`` makes the server return one logprob entry per emitted token,
which lets this script count tokens per chunk exactly. Without it, windows fall back
to a chunk-based estimate and are labelled as such.

Usage:
  ctx_decode_bench.py --base http://127.0.0.1:8000/v1 --model <served-name> \
      --targets 300000,600000,1000000 --maxtok 1024 [--logprobs]
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
import urllib.request

PROSE = (
    "Grid-scale batteries have moved from pilot projects to a routine part of how electricity "
    "systems run. Their main job is to shift energy in time: they charge when solar and wind "
    "output is high and prices are low, then discharge in the evening peak when demand climbs "
    "and those sources fade. Operators also use them for services that last seconds rather than "
    "hours, such as holding grid frequency steady when a large power plant trips offline, because "
    "a battery can respond far faster than a spinning turbine. "
)

TASKS = {
    "code": ("\n\n# Continue in Python: implement a small module that aggregates battery "
             "charge/discharge events by time window, with data structures, IO, error handling "
             "and a __main__ example. Output code only.\n"),
    "prose": ("\n\nContinue the passage above with a detailed 600-word discussion of the cost "
              "and environmental arguments. Prose only, no lists.\n"),
}


def build_filler(target_tokens: int) -> str:
    tag = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(24))
    parts = [f"[request-id {tag}]\n\n"]
    size = 0
    i = 0
    # ~5.4 chars/token for this filler; the server's prompt_tokens is what we report.
    while size < target_tokens * 5.4:
        p = f"Chapter {i}. " + PROSE
        parts.append(p)
        size += len(p)
        i += 1
    return "".join(parts)


def run(base: str, model: str, prompt: str, maxtok: int, logprobs: bool, timeout: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": maxtok,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if logprobs:
        body["logprobs"] = 1
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    marks: list[tuple[float, int]] = []   # (elapsed_s, cumulative tokens)
    usage = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            content = delta.get("content")
            if not content:
                continue
            now = time.time() - t0
            if ttft is None:
                ttft = now
            n = 1
            lp = delta.get("logprobs") or {}
            if isinstance(lp.get("content"), list) and lp["content"]:
                n = len(lp["content"])
            tokens += n
            marks.append((round(now, 3), tokens))
    wall = time.time() - t0
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)
    completion_tokens = (usage or {}).get("completion_tokens", tokens)
    windows = []
    if len(marks) >= 8 and ttft:
        span = marks[-1][0] - marks[0][0]
        edges = [marks[0][0] + span * k / 4 for k in range(5)]
        for k in range(4):
            lo, hi = edges[k], edges[k + 1]
            inwin = [m for m in marks if lo <= m[0] < hi]
            if len(inwin) < 2:
                continue
            dt = inwin[-1][0] - inwin[0][0]
            dn = inwin[-1][1] - inwin[0][1]
            windows.append({
                "window": f"{k * 25}-{(k + 1) * 25}%",
                "tokens": dn,
                "seconds": round(dt, 2),
                "tps": round(dn / dt, 2) if dt > 0 else None,
            })
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": round(ttft, 3) if ttft else None,
        "wall_s": round(wall, 2),
        "prefill_tps": round(prompt_tokens / ttft, 1) if ttft else None,
        "decode_tps": (round((completion_tokens - 1) / (wall - ttft), 2)
                       if (ttft and completion_tokens > 1 and wall > ttft) else None),
        "windows_exact": bool(logprobs),
        "windows": windows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="300000,600000,1000000",
                    help="comma-separated target prompt token counts")
    ap.add_argument("--cats", default="code,prose")
    ap.add_argument("--maxtok", type=int, default=1024)
    ap.add_argument("--logprobs", action="store_true",
                    help="exact per-window token counts (server returns one logprob per token)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--out", default=None, help="write results as JSON to this path")
    a = ap.parse_args()

    targets = [int(x) for x in a.targets.split(",")]
    cats = a.cats.split(",")
    results = []
    for target in targets:
        filler = build_filler(target)
        for cat in cats:
            prompt = filler + TASKS[cat]
            print(f"\n== {target // 1000}K {cat}: submitting {time.strftime('%H:%M:%S')}", flush=True)
            try:
                r = run(a.base, a.model, prompt, a.maxtok, a.logprobs, a.timeout)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            rec = {"target": target, "cat": cat, **r}
            results.append(rec)
            print(f"  prompt={r['prompt_tokens']} tok | TTFT={r['ttft_s']}s | "
                  f"prefill={r['prefill_tps']} tok/s | decode={r['decode_tps']} tok/s | "
                  f"gen={r['completion_tokens']} tok | wall={r['wall_s']}s", flush=True)
            label = "" if r["windows_exact"] else "  (chunk-based estimate)"
            for w in r["windows"]:
                print(f"    window {w['window']}: {w['tokens']} tok / {w['seconds']}s = "
                      f"{w['tps']} tok/s{label}", flush=True)

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
