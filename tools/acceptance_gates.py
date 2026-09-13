#!/usr/bin/env python3
"""Acceptance gates for a served DeepSeek-V4.1-Flash endpoint (stdlib only).

Three suites, all judgeable without a human in the loop:

  1. garble   30 generations across 3 temperatures x 3 structured-output prompts;
              flags empty output, CJK leakage, repeated-token loops, loop fragments.
  2. quality  30 items with one unambiguous answer each (arithmetic, multi-step word
              problems, Python semantics, logic, format).
  3. ability  12 short items, the same class as (2) but fast enough to run on every
              boot of a configuration you are tuning.

Why these three and not a benchmark harness: they are cheap, deterministic at
temperature 0, and they catch the failure modes that actually appeared while tuning
this model on GB10 hardware — a checkpoint/engine mismatch that produced fluent-looking
garbage (caught by garble), and small capability regressions from weight surgery or a
mis-set engine flag (caught by quality/ability).

Usage:
    python3 acceptance_gates.py --base http://127.0.0.1:8000/v1 --model <served-name>
    python3 acceptance_gates.py --base ... --model ... --suites garble,quality

Exit code is non-zero if a gate fails, so it can gate a deploy script.
"""
import argparse
import json
import re
import statistics
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------- transport


API_KEY = None  # set from --api-key; many deployments front the engine with auth


def chat(base, model, messages, max_tokens=300, temperature=0.0, timeout=900, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    body.update(extra)
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def text_of(resp):
    """DeepSeek-V4.x can put the answer in `content` or in `reasoning_content`."""
    msg = resp["choices"][0]["message"]
    return (msg.get("content") or msg.get("reasoning_content") or "").strip()


# ---------------------------------------------------------------- suite 1: garble

GARBLE_PROMPTS = [
    ("json", "Return only a JSON object with keys name (string), count (integer), "
             "tags (array of 3 strings). No prose.", 120),
    ("tool", 'Extract to JSON: "Meet Ana at 3pm Tuesday at Cafe Rio for 45 minutes." '
             "Keys: who, time, day, place, duration_min.", 120),
    ("code", "Write a Python one-liner that reverses a string s. Only the code.", 60),
]


def garble_reason(t):
    if not t:
        return "EMPTY"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
    if cjk > len(t) * 0.10:
        return f"CJK {cjk}"
    words = t.split()
    if len(words) > 12:
        tok, n = Counter(words).most_common(1)[0]
        if n > len(words) * 0.4:
            return f"REPEAT {tok!r} x{n}"
    if re.search(r"(.{12,}?)\1{3,}", t):
        return "LOOPFRAG"
    return None


def suite_garble(base, model, workers=6):
    jobs = [(temp, name, p, mt, i)
            for temp in (0.0, 0.7, 1.0)
            for name, p, mt in GARBLE_PROMPTS
            for i in range(4 if temp else 2)]

    def run(job):
        temp, name, p, mt, i = job
        try:
            return job, text_of(chat(base, model, [{"role": "user", "content": p}], mt, temp))
        except Exception as e:
            return job, f"ERR {type(e).__name__}"

    bad = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (temp, name, p, mt, i), t in ex.map(run, jobs):
            why = garble_reason(t)
            if why:
                bad += 1
                print(f"  GARBLE t{temp} {name}#{i}: {why} | {t[:60]!r}")
    n = len(jobs)
    ok = bad <= n * 0.10
    print(f"  garble: {n - bad}/{n} clean ({bad} garbled) -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- suites 2 and 3

# (question, kind, expectation). kind: num | text | yes | json
QUALITY = [
    (3901, "num", "Compute 47 * 83. Answer with just the number."),
    (14002, "num", "Compute 9137 + 4865. Answer with just the number."),
    (653, "num", "Compute 1000 - 347. Answer with just the number."),
    (1024, "num", "What is 2 to the power of 10? Answer with just the number."),
    (998001, "num", "Compute 999 * 999. Answer with just the number."),
    (42.5, "num", "What is 17% of 250? Answer with just the number."),
    (864197532, "num", "Compute 987654321 - 123456789. Answer with just the number."),
    (12, "num", "What is the greatest common divisor of 48 and 180? Answer with just the number."),
    (150, "num", "A train travels at 60 km/h for 2.5 hours. How many km does it cover? Answer with just the number."),
    (12.75, "num", "I buy 3 items costing $4.25 each. What is the total in dollars? Answer with just the number."),
    (68, "num", "An $85 item is discounted by 20%. What is the discounted price in dollars? Answer with just the number."),
    (5, "num", "If 5 workers take 10 days to finish a job, how many days do 10 workers take at the same rate? Answer with just the number."),
    (40, "num", "A tank is filled at 4 litres per minute for 7 minutes, then at 3 litres per minute for 4 minutes. How many litres total? Answer with just the number."),
    ("14:20", "text", "It is 09:45. What time is it 4 hours and 35 minutes later? Answer as HH:MM only."),
    (3.1, "num", "I buy 7 apples at $0.60 each and 3 oranges at $0.90 each, and pay with a $10 bill. How much change in dollars? Answer with just the number."),
    (5, "num", "A car drives 240 km at 80 km/h, then 80 km at 40 km/h. What is the total driving time in hours? Answer with just the number."),
    (4, "num", "In Python, what is len([x for x in range(10) if x % 3 == 0])? Answer with just the number."),
    ("cba", "text", "In Python, what does 'abc'[::-1] evaluate to? Answer with just the resulting string."),
    ("[1, 2, 3]", "text", "In Python, what does sorted(set([3, 1, 3, 2])) evaluate to? Answer with just the list."),
    (55, "num", "In Python, what is sum(range(1, 11))? Answer with just the number."),
    ("31", "text", "In Python, what is str(10 // 3) + str(10 % 3)? Answer with just the resulting string."),
    ("False", "text", "In Python, what is bool([]) ? Answer with just True or False."),
    ("[0, 3, 6, 9]", "text", "In Python, what is list(range(0, 10, 3))? Answer with just the list."),
    (2, "num", "In Python, what is len('hello world'.split())? Answer with just the number."),
    (None, "yes", "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies? Answer yes or no."),
    (30, "num", "What is the next number in the sequence 2, 6, 12, 20, ? Answer with just the number."),
    (None, "yes", "If A > B and B > C, is A > C? Answer yes or no."),
    (9, "num", "Which number does not belong to the group of primes 3, 5, 7, 9, 11? Answer with just the number."),
    ({"a": 1, "b": [True, False]}, "json",
     'Return JSON only, no prose, reproducing this object exactly: {"a": 1, "b": [true, false]}'),
    (2027, "num", "What is the year in the ISO date 2027-03-14? Answer with just the number."),
]

ABILITY = [QUALITY[i] for i in (0, 1, 2, 5, 6, 7, 13, 16, 17, 19, 21, 28)]


def _numbers(t):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t.replace("$", " ").replace(",", " "))]


def judge(kind, expect, t):
    if kind == "num":
        return any(abs(n - expect) < 1e-6 for n in _numbers(t))
    if kind == "yes":
        return bool(re.search(r"\byes\b", t, re.I))
    if kind == "json":
        m = re.search(r"\{.*\}", t, re.S)
        try:
            return json.loads(m.group(0)) == expect if m else False
        except Exception:
            return False
    return str(expect).lower().replace(" ", "") in t.lower().replace(" ", "")


def suite_items(base, model, items, name, workers=6, max_tokens=300):
    def run(item):
        expect, kind, q = item
        try:
            return item, judge(kind, expect, text_of(chat(base, model, [{"role": "user", "content": q}], max_tokens)))
        except Exception as e:
            return item, f"ERR {type(e).__name__}"

    good, bad = 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (expect, kind, q), ok in ex.map(run, items):
            good += 1 if ok is True else 0
            if ok is not True:
                bad.append((q[:70], expect, ok))
    print(f"  {name}: {good}/{len(items)} passed")
    for q, e, got in bad:
        print(f"    miss: {q} | expected {e} | {got}")
    # gates the deploy on a clearly broken endpoint, not on a single hard item
    return good >= len(items) - 2


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--suites", default="garble,quality,ability")
    ap.add_argument("--api-key", default=None,
                    help="bearer token, if the endpoint is fronted by a gateway")
    a = ap.parse_args()

    global API_KEY
    API_KEY = a.api_key

    wanted = {s.strip() for s in a.suites.split(",") if s.strip()}
    results = {}
    print(f"endpoint {a.base} model {a.model}")
    if "garble" in wanted:
        print("-- garble suite")
        results["garble"] = suite_garble(a.base, a.model)
    if "quality" in wanted:
        print("-- quality suite (30 items)")
        results["quality"] = suite_items(a.base, a.model, QUALITY, "quality")
    if "ability" in wanted:
        print("-- ability suite (12 items)")
        results["ability"] = suite_items(a.base, a.model, ABILITY, "ability")

    print("\nsummary:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
