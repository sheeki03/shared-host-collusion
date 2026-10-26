#!/usr/bin/env bash
# v4 launch: pooled games of the slow Contabo runs on instabox2, instabox3 and vps1.
set -euo pipefail
cd "$(dirname "$0")/.."
launch() {  # tag dest cap conc
  local tag=$1 dest=$2 cap=$3 conc=$4
  ssh -o BatchMode=yes "$dest" "mkdir -p shared-host-collusion/runs/shards" 2>&1 | grep -v '^\*\*\*' || true
  scp -q -o BatchMode=yes "runs/shards/$tag.json" "$dest:shared-host-collusion/runs/shards/$tag.json"
  scp -q -o BatchMode=yes runs/launch_remote.sh "$dest:launch_remote.sh"
  ssh -o BatchMode=yes "$dest" "bash launch_remote.sh $tag $cap $conc" 2>&1 | grep -v '^\*\*\*' || true
}
launch instabox2-pool instabox2 400 32
launch instabox3-pool instabox3 200 14
launch vps1-pool "<user>@<host-ip-vps1>" 150 18
