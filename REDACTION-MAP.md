# REDACTION-MAP — how the 2026-09-28 update keeps internal facts out

This file exists so the claim "no internal addresses, no internal hosts, no credentials"
can be **audited** instead of trusted. It lists the conventions the added material uses,
the categories that were scanned, and the residual hits.

## 1. Conventions used in this update

| Instead of | The material says | Where |
|---|---|---|
| node hostnames | `node0 node1 node2 node3` (rank order, node0 = head) | `tools/ring_link_health.sh`, `tools/post_reseat_recover.sh` |
| hard-coded addresses | environment variables: `NODES`, `IFACES`, `WEIGHTS_DIR`, `SHARDS_EXPECT`, `RING_IFACE`, `RESTART_CMD` | both new tools |
| real interface names | example values `ring0` / `ring1`, and "the ring interface" | both new tools, results note §5 |
| the fleet gateway's domain | "the fleet's gateway endpoint" / `BASE_URL=http://127.0.0.1:8888` (loopback, already published in this repo) | results note §2, §7 |
| the NCCL overlay's path | its **properties**: a flat directory containing the ring marker string | results note §1 |
| the fleet's host inventory and model variants | model **class** + engine + hardware class, e.g. "27 B-class FP8 on 2 × 2080 Ti" | cross-project note §2 |
| internal passwords, API keys, tokens, frp/litellm config | nothing — no credential, host or service name appears anywhere in the added material | — |

Known simplifications, stated rather than hidden:

* The two tools are written to be run against *your* ring, so they contain no project
  addresses at all; `ring0`/`ring1` are illustrative names, not ours.
* The KV-pool and decode numbers are ours and are labelled as such; the cross-project
  table quotes **other projects' published numbers verbatim**, with the method each of
  them used. Those rows are their claims, not our measurements.
* Model variants (quantisation, fine-tunes, safety tuning) are deliberately not
  described: the recipe is published as a deployment of the **official** checkpoint, and
  nothing in the added material identifies any other variant.

## 2. Categories scanned, with results

Scanned: (a) every file in the currently published tree, (b) every file added or changed
by this update, (c) **every blob in the repository history** (7 commits, 34 unique blobs),
including superseded versions of files.

| Category | Changed/added files | Published tree | History blobs |
|---|---:|---:|---:|
| private IPv4 (RFC1918) | **0** | 0 | 6 (see §3) |
| our ring / management subnets | **0** | 0 | 0 |
| public IPs of our own hosts | **0** | 0 | 0 |
| our domains | **0** | 0 | 0 |
| our hostnames | **0** | 0 | 0 |
| our ports (enumerated explicitly, not by range) | **0** | 0 | 0 |
| credential patterns (`ghp_`, `hf_`, `sk-`, `AKIA`, …) | **0** | 0 | 0 |
| `password|passwd|secret|api_key|token` assignments | **0** | 3 (benign, §3) | 3 (same) |
| variant markers (`uncensored`, `abliterated`, `heretic`, …) | **0** | 0 | 0 |
| internal ops terms (gateway product names, tunnel/proxy stacks) | **0** | 0 | 0 |

Informational categories, which are expected to have hits and were triaged one by one:
loopback/wildcard binds (`127.0.0.1`, `0.0.0.0`), port `8888` (already published in this
repo's launch examples), `node0..3` role names, container-internal paths such as
`/kv-offload`, the numeric default `7200` (a timeout, not a port), and third-party
handles in the credits and reference table.

## 3. Residual hits and their triage

1. **`tools/acceptance_gates.py` — `API_KEY =`, `API_KEY:` (3 hits, current tree and its
   history).** Variable names in an example script that reads the key from the
   environment; **no value is present**. Benign — it is the pattern that lets a reader
   authenticate, which is the opposite of a leak.
2. **`tools/capture-allnode-logs.sh`, `tools/preflight.sh` — `10.0.0.10` … `10.0.0.13`
   (6 hits, one historical commit `e39e66e`).** Example node addresses from an early
   revision. The **current** files carry role names (`node0 node1 node2 node3`) instead,
   which is why the current tree scans 0. Left in history deliberately: rewriting history
   to hide a generic `10.0.0.x` example would cost more (every future clone re-pulls the
   mirror) than it buys. Use `git log -p` if you want to see the fix itself.

## 4. Re-scan line

> Worktree + added/changed files: **residual sensitive hits = 0**, across the ten
> categories above. History blobs: 6 hits, all in one superseded revision, triaged in §3.
> Re-run `tools/`-adjacent audit before the next update — the fields most likely to
> reappear in a later commit are IP octets, subnets and digest prefixes.
