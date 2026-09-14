#!/usr/bin/env bash
# Prove whether a multi-node deployment is actually using RDMA, or silently fell back
# to TCP sockets over the same cable.
#
# Why it is needed: NCCL will run happily over TCP on a ConnectX link and never say so.
# The environment variables look right, the serve works, and every collective is ~2x
# slower. A container also needs all three of  --device /dev/infiniband ,
# --cap-add IPC_LOCK , --ulimit memlock=-1:-1  before it can open the device at all.
#
# Method (as in the recipe by bilikaz/qwen38-flash-next-cluster-recipe): sample the HCA
# port counters and the interface TCP byte counters WHILE the model is decoding. RDMA
# moving + TCP flat = the fast path. HCA flat + TCP moving = you are on the slow path.
#
# Config (env):
#   NODES     "node0 node1 node2 node3"  rank order, node0 = head (local). ssh as root.
#   IF_MGMT   management interface name (the one in NCCL_SOCKET_IFNAME)
#   IF_RING   ring/interconnect interface name (optional; blank to skip)
#   API_URL   local OpenAI-compatible endpoint        (default http://127.0.0.1:8888/v1)
#   MODEL     served model name for the load request  (default: taken from /v1/models)
#   SECONDS_TO_SAMPLE  default 20
#
# Usage:  NODES="a b c d" IF_MGMT=enp1s0f0np0 IF_RING=renp1s0f1np1 bash rdma_proof.sh
set -u

NODES=${NODES:?set NODES to the node list in rank order, head first}
IF_MGMT=${IF_MGMT:-}
IF_RING=${IF_RING:-}
API_URL=${API_URL:-http://127.0.0.1:8888/v1}
SEC=${SECONDS_TO_SAMPLE:-20}
J="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10"

MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  MODEL=$(curl -s -m 10 "$API_URL/models" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
fi
[ -n "$MODEL" ] || { echo "cannot determine model name; set MODEL=..." >&2; exit 2; }

SNAP='
snap() {
  local hca_tx=0 hca_rx=0 d
  for d in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do
    [ -r "$d" ] && hca_tx=$((hca_tx + $(cat "$d")))
    d=${d%port_xmit_data}port_rcv_data; [ -r "$d" ] && hca_rx=$((hca_rx + $(cat "$d")))
  done
  local mg=0 rg=0
  [ -n "'"$IF_MGMT"'" ] && [ -r /sys/class/net/'"$IF_MGMT"'/statistics/tx_bytes ] && \
    mg=$(( $(cat /sys/class/net/'"$IF_MGMT"'/statistics/tx_bytes) + $(cat /sys/class/net/'"$IF_MGMT"'/statistics/rx_bytes) ))
  [ -n "'"$IF_RING"'" ] && [ -r /sys/class/net/'"$IF_RING"'/statistics/tx_bytes ] && \
    rg=$(( $(cat /sys/class/net/'"$IF_RING"'/statistics/tx_bytes) + $(cat /sys/class/net/'"$IF_RING"'/statistics/rx_bytes) ))
  echo "$((hca_tx/1048576)) $((hca_rx/1048576)) $((mg/1048576)) $((rg/1048576))"
}
snap'

echo "=== RDMA proof: sampling ${SEC}s while a decode is in flight (model $MODEL) ==="
# keep the engine busy for the whole window
( curl -s -m $((SEC + 60)) "$API_URL/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 upward, one number per line, as far as you can.\"}],\"max_tokens\":6000,\"temperature\":0}" >/dev/null 2>&1 ) &
LOAD=$!
BEFORE=$(mktemp); AFTER=$(mktemp)
for h in $NODES; do
  if [ "$h" = "$(hostname)" ] || [ "$h" = "local" ]; then bash -c "$SNAP" | sed "s/^/$h /" >> "$BEFORE"
  else $J "root@$h" "$SNAP" 2>/dev/null | sed "s/^/$h /" >> "$BEFORE"; fi
done
sleep "$SEC"
for h in $NODES; do
  if [ "$h" = "$(hostname)" ] || [ "$h" = "local" ]; then bash -c "$SNAP" | sed "s/^/$h /" >> "$AFTER"
  else $J "root@$h" "$SNAP" 2>/dev/null | sed "s/^/$h /" >> "$AFTER"; fi
done
kill $LOAD 2>/dev/null

fail=0
paste "$BEFORE" "$AFTER" | while read -r node atx arx amg arg node2 btx brx bmg brg; do
  h=$(printf '%d\n' $(( (btx-atx)/SEC ))); hr=$(( (brx-arx)/SEC ))
  t=$(( (bmg-amg)/SEC )); tr=$(( (brg-arg)/SEC ))
  printf '  %-16s HCA tx %5d MB/s  rx %5d MB/s   |   TCP mgmt %5d MB/s  ring %5d MB/s\n' \
         "$node" "$h" "$hr" "$t" "$tr"
done
rm -f "$BEFORE" "$AFTER"
echo "  verdict: HCA moving with TCP ~0 = RDMA is real."
echo "           HCA flat with TCP moving = the deployment is on TCP sockets (~2x slower)."
