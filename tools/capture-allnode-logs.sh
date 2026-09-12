#!/usr/bin/env bash
# Failure-evidence collector for a multi-node deployment.
#
# Why it exists: the worker side of a startup failure is where the real error is, and
# it lives in the *worker's* container log. The next boot removes those containers, so
# the evidence disappears unless you grab it immediately. Symptom we hit repeatedly:
# the head log ended at
#     RuntimeError: Engine core initialization failed. ... Failed core proc(s): {}
# with no root cause at all, because the worker container that actually raised had
# already been deleted.
#
# Rule: call this BEFORE any subsequent boot, i.e. immediately after your launcher
# reports a failure.
#
#   NODE_IPS="10.0.0.10 10.0.0.11 ..." CONTAINER=vllm_dsv41 ./capture-allnode-logs.sh <tag>
#
# Output: ./allnode-<tag>/rank<N>.log per rank + a filtered error summary on stdout.
set -u

TAG=${1:?usage: capture-allnode-logs.sh <tag>}
NODE_IPS=${NODE_IPS:?set NODE_IPS (index = rank, rank 0 = head/local)}
CONTAINER=${CONTAINER:-vllm_dsv41}
OUT=${OUT:-./allnode-$TAG}
SSH_OPTS="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

read -r -a NODE_IPS_ARR <<< "$NODE_IPS"
mkdir -p "$OUT"

for rank in "${!NODE_IPS_ARR[@]}"; do
  ip=${NODE_IPS_ARR[$rank]}
  {
    echo "===== rank $rank @ $ip ====="
    if [ "$rank" = "0" ]; then
      docker logs "$CONTAINER" 2>&1 | tail -400
    else
      ssh $SSH_OPTS "$ip" "docker logs $CONTAINER 2>&1 | tail -400" 2>&1
    fi
  } > "$OUT/rank$rank.log"
  printf '  rank %s -> %s (%s lines)\n' "$rank" "$OUT/rank$rank.log" "$(wc -l < "$OUT/rank$rank.log")"
done

echo
echo "-- root causes (filtered) --"
grep -inE "out of memory|OutOfMemoryError|ValueError|AssertionError|RuntimeError|CUDA error|not enough|exceeds|too large|insufficient|illegal memory|PATCH MISSING|IMAGE MISSING|larger than|maximum number of tokens" \
  "$OUT"/rank*.log 2>/dev/null \
  | grep -viE "Duplicate NCCL|See root cause above|Engine core initialization failed|import_utils" \
  | head -30

echo
echo "saved: $OUT"
echo "note: also keep the launcher's own boot log — it shows which ranks launched at all"
echo "      (a worker that never launched never wrote a container log; the launcher log"
echo "      is the only place 'PATCH FILE MISSING' or 'rank N launch failed' appears)."
