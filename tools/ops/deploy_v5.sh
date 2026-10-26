#!/usr/bin/env bash
# v5 launch on the D13 commit (launch_remote.sh default REF): remaining confirmatory games on the three instabox hosts.
set -euo pipefail
cd "$(dirname "$0")/.."
launch() {  # tag dest cap conc
  local tag=$1 dest=$2 cap=$3 conc=$4
  scp -q -o BatchMode=yes "runs/shards/$tag.json" "$dest:shared-host-collusion/runs/shards/$tag.json"
  scp -q -o BatchMode=yes runs/launch_remote.sh "$dest:launch_remote.sh"
  ssh -o BatchMode=yes "$dest" "bash launch_remote.sh $tag $cap $conc"
}
launch instabox-v5 instabox 350 40
launch instabox2-v5 instabox2 300 32
launch instabox3-v5 instabox3 150 14
date
