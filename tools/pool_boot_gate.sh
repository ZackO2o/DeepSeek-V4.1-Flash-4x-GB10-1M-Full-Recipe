#!/usr/bin/env bash
# Boot gate for unified-memory hosts (DGX Spark / GB10): wait until every node really
# has its memory back, then evict the checkpoint's page cache, THEN start the engine.
#
# Why (measured, see results/2026-09-14-kv-prefix-tier-and-pool-pinning.md):
#   * The GPU driver on a unified-memory host needs pages that are FREE, not merely
#     "reclaimable". Right after a container is killed, MemAvailable takes tens of
#     seconds to come back; launching into that window is how a boot ends up with a
#     phantom CUDA OOM or with a KV pool a third smaller than the one it had before.
#   * The same flags produced KV pools of 1.46M / 1.64M / 2.20M tokens across boots on
#     identical hardware. Gating here + pinning the pool (--kv-cache-memory-bytes) is
#     what makes capacity reproducible.
#
# Config (env):
#   NODES      "node0 node1 node2 node3" rank order, node0 = head (local, run in place)
#   NEED_GIB   per-node MemAvailable required before launching     (default 100)
#   MAX_WAIT   seconds to keep polling before giving up (warning)   (default 180)
#
# Usage:  NODES="a b c d" NEED_GIB=100 bash pool_boot_gate.sh && ./your-boot.sh
set -u

NODES=${NODES:?set NODES to the node list in rank order, head first}
NEED_GIB=${NEED_GIB:-100}
MAX_WAIT=${MAX_WAIT:-180}
J="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"
MEM='awk "/^MemAvailable/{printf \"%d\", \$2/1048576}" /proc/meminfo'

echo "=== boot gate: MemAvailable >= ${NEED_GIB} GiB on every node, then evict page cache ==="
rc=0
for h in $NODES; do
  local_run=0; { [ "$h" = "$(hostname)" ] || [ "$h" = "local" ]; } && local_run=1
  t=0
  while :; do
    if [ "$local_run" = 1 ]; then m=$(eval "$MEM" 2>/dev/null)
    else m=$($J "root@$h" "$MEM" 2>/dev/null || echo 0); fi
    [ "${m:-0}" -ge "$NEED_GIB" ] && break
    if [ "$t" -ge "$MAX_WAIT" ]; then
      echo "  ! $h still short after ${MAX_WAIT}s (${m} GiB) -- another serve on the box? continuing anyway"
      rc=1; break
    fi
    [ "$t" = 0 ] && echo "  . $h waiting for memory to return (${m}/${NEED_GIB} GiB)"
    sleep 5; t=$((t + 5))
  done
  # evict OUR files from the page cache: root can drop everything, an unprivileged
  # user can do it per-file with dd(1) (POSIX_FADV_DONTNEED)
  if [ "$local_run" = 1 ]; then
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    m2=$(eval "$MEM" 2>/dev/null)
  else
    $J "root@$h" "sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null" >/dev/null 2>&1
    m2=$($J "root@$h" "$MEM" 2>/dev/null || echo 0)
  fi
  echo "  + $h available ${m} -> ${m2} GiB, page cache evicted"
done
exit $rc
