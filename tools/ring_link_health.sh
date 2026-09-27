#!/usr/bin/env bash
# Per-port health sampler for a switchless RoCE ring: sample the *NIC carrier*, the IB
# port state and the negotiated speed, on every node, several times, and fail if any
# port ever loses carrier.
#
# Why the carrier, and not just the IB state (measured, see
# results/2026-09-28-sglang-switchless-ring-tp4.md):
#   * A flapping ring link can read `4 (ACTIVE)` on the IB port *while* the NIC carrier
#     on the same physical port is dropping to 0 every few seconds. Checking only the IB
#     state is how a flapping link survives for days: you see a healthy port, and NCCL
#     sees ~1300 `Got non-fatal async event` warnings, `ibv_modify_qp` timeouts and boots
#     that die in collective setup.
#   * One reading proves nothing. Sample repeatedly; a single `1` is not health.
#
# What it does NOT tell you: throughput. Re-seating a flapping cable restored a clean
# link and did not move decode tok/s at all. The value here is removing a landmine.
#
# Config (env):
#   NODES    "node0 node1 node2 node3"  rank order, node0 = local (ssh as root)
#   IFACES   ring interfaces to sample, space separated (same names on each node), e.g.
#            "ring0 ring1"  -- list BOTH planes/ports per node, not just the one you
#            suspect: the interesting failures are on the port nobody watched
#   SAMPLES  how many readings per port (default 8)
#   INTERVAL seconds between readings (default 4)
#
# Exit: 0 = every port held carrier for every sample; 1 = at least one flap or error.

set -u
NODES="${NODES:?set NODES, e.g. NODES=\"node0 node1 node2 node3\"}"
IFACES="${IFACES:?set IFACES, e.g. IFACES=\"ring0 ring1\"}"
SAMPLES="${SAMPLES:-8}"
INTERVAL="${INTERVAL:-4}"

declare -A CAR=() IBT=()
flaps=0
printf '%-10s %-10s %-4s %-24s %-14s %s\n' NODE IFACE IDX CARRIER_TRACE IB_TRACE SPEED
for node in $NODES; do
  for ifc in $IFACES; do
    for i in $(seq 1 "$SAMPLES"); do
      # One ssh per sample keeps this dead simple and survives a node going away.
      out=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$node" "
        c=\$(cat /sys/class/net/$ifc/carrier 2>/dev/null || echo '?')
        s=\$(cat /sys/class/net/$ifc/speed 2>/dev/null || echo '?')
        # IB state via the netdev -> rdma link mapping, no name guessing
        d=\$(ls /sys/class/net/$ifc/device/infiniband 2>/dev/null | head -1)
        b=\$(cat /sys/class/infiniband/\$d/ports/1/state 2>/dev/null | awk '{print \$1}')
        echo \"\$c \$b \$s\"
      " 2>/dev/null)
      if [ -z "$out" ]; then
        printf '%-10s %-10s %-4s %-24s %-14s %s\n' "$node" "$ifc" "$i" "ssh-unreachable" "-" "-"
        flaps=$((flaps+1)); continue
      fi
      c=$(echo "$out" | awk '{print $1}')
      b=$(echo "$out" | awk '{print $2}')
      s=$(echo "$out" | awk '{print $3}')
      car_trace="${CAR[$node/$ifc]:-}$c"
      ib_trace="${IBT[$node/$ifc]:-}$b"
      CAR["$node/$ifc"]="$car_trace"; IBT["$node/$ifc"]="$ib_trace"
      [ "$c" != "1" ] && flaps=$((flaps+1))
      [ $i -eq "$SAMPLES" ] && \
        printf '%-10s %-10s %-4s %-24s %-14s %s\n' "$node" "$ifc" "-" "$car_trace" "$ib_trace" "$s Gb/s"
      sleep "$INTERVAL"
    done
  done
done

echo
if [ "$flaps" -eq 0 ]; then
  echo "PASS: every sampled port held carrier=1 for all $SAMPLES readings."
  exit 0
else
  echo "FAIL: $flaps reading(s) without carrier (or unreachable)."
  echo "  A port whose IB state stays 4 while carrier drops is an intermittent physical"
  echo "  link: re-seat both ends, then re-run this gate before chasing NCCL."
  echo "  AFTER any re-seat, re-check the RoCEv2 IPv4 GID index on every node"
  echo "  (tools/post_reseat_recover.sh) -- it can move."
  exit 1
fi
