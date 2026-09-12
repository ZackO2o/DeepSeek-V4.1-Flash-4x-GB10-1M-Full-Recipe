#!/usr/bin/env bash
# Preflight for a multi-node TP deployment: verify, ON EVERY RANK, that everything the
# launcher will need actually exists locally. Run this before every boot.
#
# Why it exists: patch files are bind-mounted per node. A patch directory that exists
# only on the head produces a head that starts, loads weights, then blocks in the
# process-group store until it dies with
#     DistStoreError: Timed out after 601 seconds waiting for clients. 1/4 clients joined.
# The message blames the network; the cause is a missing file. This script fails fast.
#
# Config via environment (see --help / the block below):
#   NODE_IPS      space-separated node addresses, index = rank (rank 0 = head, local)
#   IMAGE         container image to check on every node
#   NCCL_DIR      directory containing the patched NCCL library (optional)
#   PATCH_DIR     patch directory that must be present on every node
#   EXTRA_PATHS   extra absolute paths that must exist on every node
#
# Usage:  NODE_IPS="..." IMAGE=... PATCH_DIR=... ./preflight.sh
set -u

NODE_IPS=${NODE_IPS:?set NODE_IPS, e.g. "10.0.0.10 10.0.0.11 10.0.0.12 10.0.0.13"}
IMAGE=${IMAGE:-}
NCCL_DIR=${NCCL_DIR:-}
PATCH_DIR=${PATCH_DIR:-}
EXTRA_PATHS=${EXTRA_PATHS:-}
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8"

fail=0
note() { printf '%s\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }

remote() {  # remote <rank> <command>
  local rank=$1; shift
  if [ "$rank" = "0" ]; then bash -lc "$*"; else ssh $SSH_OPTS "${NODE_IPS_ARR[$rank]}" "$*"; fi
}

read -r -a NODE_IPS_ARR <<< "$NODE_IPS"

note "== preflight: $((${#NODE_IPS_ARR[@]})) nodes =="
for rank in "${!NODE_IPS_ARR[@]}"; do
  ip=${NODE_IPS_ARR[$rank]}
  note "-- rank $rank @ $ip"
  if ! remote "$rank" "true" >/dev/null 2>&1; then
    bad "unreachable (ssh)"; continue
  fi
  ok "ssh reachable"

  if [ -n "$IMAGE" ]; then
    if remote "$rank" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then ok "image present: $IMAGE"; else bad "image missing: $IMAGE"; fi
  fi

  if [ -n "$NCCL_DIR" ]; then
    if remote "$rank" "test -f '$NCCL_DIR/libnccl.so.2'"; then ok "nccl lib: $NCCL_DIR"; else bad "nccl lib missing: $NCCL_DIR"; fi
  fi

  # The important one: every patch file must exist locally, on every rank.
  if [ -n "$PATCH_DIR" ]; then
    if remote "$rank" "test -f '$PATCH_DIR/mounts.txt'"; then ok "patch mounts: $PATCH_DIR/mounts.txt"; else bad "patch mounts missing: $PATCH_DIR/mounts.txt"; fi
    if remote "$rank" "test -f '$PATCH_DIR/mounts.txt'"; then
      while read -r file dest; do
        [ -z "${file:-}" ] && continue
        case "$file" in \#*) continue ;; esac
        if remote "$rank" "test -f '$PATCH_DIR/$file'"; then ok "patch file: $file -> $dest"; else bad "PATCH FILE MISSING on this rank: $PATCH_DIR/$file"; fi
      done < <(remote "$rank" "cat '$PATCH_DIR/mounts.txt'")
    fi
  fi

  for p in $EXTRA_PATHS; do
    if remote "$rank" "test -e '$p'"; then ok "path: $p"; else bad "path missing: $p"; fi
  done

  # Head memory check: the engine needs ~100 GiB free to load and size the KV pool.
  avail=$(remote "$rank" "awk '/MemAvailable:/ {printf \"%d\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo 0)
  if [ "${avail:-0}" -ge 95 ]; then ok "host memory available: ${avail} GiB"; else bad "host memory available: ${avail} GiB (want >= 95)"; fi
done

if [ "$fail" = "1" ]; then
  note ""
  note "PREFLIGHT FAILED — fix the items above before booting. Booting anyway usually means"
  note "one or more workers never start while the head waits for them (601 s store timeout)."
  exit 1
fi
note ""
note "PREFLIGHT OK"
