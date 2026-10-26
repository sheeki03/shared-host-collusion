#!/usr/bin/env bash
# Relaunch v2: throughput-weighted shards (runs/shards-v2), new ledgers. Usage: deploy_v2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
deploy() {  # tag ip shard cap conc
  local tag=$1 ip=$2 shard=$3 cap=$4 conc=$5
  scp -q -o BatchMode=yes "$shard" "<user>@$ip:~/shared-host-collusion/runs/shards/$tag.json"
  scp -q -o BatchMode=yes runs/launch_remote.sh "<user>@$ip:~/launch_remote.sh"
  ssh -o BatchMode=yes "<user>@$ip" "bash ~/launch_remote.sh $tag $cap $conc" 2>&1 | grep -v '^\*\*\*' || true
}
deploy vps1v2 "<host-ip-vps1>" runs/shards-v2/vps1.json 550 18
deploy vps2v2 "<host-ip-vps2>" runs/shards-v2/vps2.json 150 6
cp runs/shards-v2/mac.json runs/shards/macv2.json
INSPECT_DOCKER_CLI_CONCURRENCY=40 nohup .venv/bin/python -m harness.run --plan full --shard-file runs/shards/macv2.json \
  --cap 450 --concurrency 24 --init-ledger --ledger runs/full/ledger-macv2.jsonl --log-dir runs/full/logs-macv2 \
  > runs/full/macv2.out 2>&1 < /dev/null &
echo "launched macv2 pid $!"
