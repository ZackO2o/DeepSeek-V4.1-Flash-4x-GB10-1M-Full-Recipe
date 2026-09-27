#!/usr/bin/env bash
# After a cable re-seat, a port move, or a power event on a switchless RoCE ring:
# heal the weight mount, re-detect the RoCEv2 IPv4 GID index, then boot and gate.
#
# Why this exists (measured, see results/2026-09-28-sglang-switchless-ring-tp4.md):
#   * A hard-coded GID index EXPIRES when the hardware state changes. Ours was index 3
#     for weeks; after re-seating one ring cable, the RoCEv2 IPv4 GID moved to index 4 on
#     every node, and the boot gate failed with "no common IPv4 RoCE v2 GID" while the
#     config had not changed by one character. Never treat the index as a constant.
#   * A soft NFS mount can LIE about being mounted. Two nodes reported the weight mount
#     present while `ls` on it blocked forever; `mountpoint` says "mounted", then boot
#     fails with "weights missing/incomplete". The only trustworthy criterion is
#     artefact-shaped: COUNT THE CHECKPOINT SHARDS, with a timeout on every probe.
#
# Config (env):
#   NODES        "node0 node1 node2 node3"  rank order, node0 = local (ssh as root)
#   WEIGHTS_DIR  path to the checkpoint as every node must see it
#   SHARDS_EXPECT expected number of *.safetensors shards (48 for this checkpoint)
#   RING_IFACE   ring interface name on each node (used to learn that node's ring IPv4)
#   RESTART_CMD  optional: command to (re)start the engine after healing, run on node0
#
# Exit: 0 = mounts healthy, one common GID index found on all nodes, optional restart OK.

set -u
NODES="${NODES:?set NODES}"
WEIGHTS_DIR="${WEIGHTS_DIR:?set WEIGHTS_DIR}"
SHARDS_EXPECT="${SHARDS_EXPECT:-48}"
RING_IFACE="${RING_IFACE:?set RING_IFACE}"
RESTART_CMD="${RESTART_CMD:-}"

ssh_do() { timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2" 2>/dev/null; }

echo "== 1. weight mount: artefact-shaped check (never trust mountpoint alone) =="
for node in $NODES; do
  # timeout everywhere: `ls` on a dead soft mount blocks forever.
  n=$(ssh_do "$node" "timeout 15 bash -c 'ls $WEIGHTS_DIR/*.safetensors 2>/dev/null | wc -l'")
  [ -z "$n" ] && n=0
  if [ "$n" -ge "$SHARDS_EXPECT" ]; then
    echo "  $node: $n shards  OK"
  else
    echo "  $node: $n shards (expected >= $SHARDS_EXPECT) -> forcing a remount"
    ssh_do "$node" "umount -f -l $WEIGHTS_DIR 2>/dev/null; mount $WEIGHTS_DIR 2>/dev/null"
    n2=$(ssh_do "$node" "timeout 15 bash -c 'ls $WEIGHTS_DIR/*.safetensors 2>/dev/null | wc -l'")
    [ -z "$n2" ] && n2=0
    echo "  $node: $n2 shards after remount"
    [ "$n2" -lt "$SHARDS_EXPECT" ] && echo "  !! $node still short: fix the share before booting"
  fi
done

echo
echo "== 2. RoCEv2 IPv4 GID index per node (it moves when hardware state changes) =="
common=""
for node in $NODES; do
  # Derive the expected IPv4 -> hex tail from THAT NODE's ring address; no addresses are
  # hard-coded here.
  found=$(ssh_do "$node" "
    ip4=\$(ip -4 -o addr show dev $RING_IFACE 2>/dev/null | awk '{print \$4}' | cut -d/ -f1)
    [ -z \"\$ip4\" ] && { echo 'no-ip'; exit 0; }
    IFS=. read -r a b c d <<< \"\$ip4\"
    tail=\$(printf '%02x%02x:%02x%02x' \$a \$b \$c \$d)
    for dev in /sys/class/infiniband/*; do
      [ -e \"\$dev\" ] || continue
      for g in \"\$dev\"/ports/1/gids/*; do
        idx=\$(basename \"\$g\")
        val=\$(cat \"\$g\" 2>/dev/null | tr 'A-Z' 'a-z')
        typ=\$(cat \"\$dev/ports/1/gid_attrs/types/\$idx\" 2>/dev/null)
        case \"\$val\" in
          *\$tail)
            case \"\$typ\" in *RoCE*v2*|*RoCEv2*) echo \"\$idx \$(basename \$dev)\";; esac ;;
        esac
      done
    done | head -1
  ")
  echo "  $node: ${found:-none}"
  idx=$(echo "$found" | awk '{print $1}')
  if [ -z "$idx" ]; then
    echo "  !! $node has no RoCEv2 IPv4 GID matching its $RING_IFACE address."
    echo "     If a cable/port moved, the ring address may now be on a different device."
  elif [ -z "$common" ]; then
    common="$idx"
    echo "     -> using NCCL_IB_GID_INDEX=$idx"
  elif [ "$common" != "$idx" ]; then
    echo "     !! index differs from other nodes ($common). NCCL needs ONE common index;"
    echo "        re-check which device carries the ring address on each node."
    common="MISMATCH"
  else
    echo "     -> agrees with NCCL_IB_GID_INDEX=$idx"
  fi
done

echo
if [ -n "$RESTART_CMD" ]; then
  echo "== 3. restart on node0 =="
  echo "  \$ $RESTART_CMD"
  timeout 900 ssh -o BatchMode=yes -o ConnectTimeout=10 "$(echo "$NODES" | awk '{print $1}')" "$RESTART_CMD"
else
  echo "== 3. restart skipped (set RESTART_CMD to do it here) =="
fi

echo
echo "Boot checklist reminder:"
echo "  * copy the detected index into the launch environment on EVERY node"
echo "  * the NCCL overlay must be a flat dir and contain the ring marker string"
echo "  * re-run tools/ring_link_health.sh after any physical work"
[ "$common" = "MISMATCH" ] && exit 1
exit 0
