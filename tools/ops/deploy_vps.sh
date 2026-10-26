#!/usr/bin/env bash
# Copy each VPS its shard and the launcher, then start its runner (local helper; runs/ is not committed).
set -euo pipefail
cd "$(dirname "$0")/.."
deploy() {
  local tag=$1 ip=$2 cap=$3 conc=$4
  ssh -o BatchMode=yes "<user>@$ip" "mkdir -p ~/shared-host-collusion/runs/shards" 2>&1 | grep -v '^\*\*\*' || true
  scp -q -o BatchMode=yes "runs/shards/$tag.json" "<user>@$ip:~/shared-host-collusion/runs/shards/$tag.json"
  scp -q -o BatchMode=yes runs/launch_remote.sh "<user>@$ip:~/launch_remote.sh"
  ssh -o BatchMode=yes "<user>@$ip" "bash ~/launch_remote.sh $tag $cap $conc" 2>&1 | grep -v '^\*\*\*' || true
}
deploy vps1 "<host-ip-vps1>" 400 18
deploy vps2 "<host-ip-vps2>" 150 6
