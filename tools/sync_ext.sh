#!/usr/bin/env bash
# Read-only pull of every host's runs/ext (extension logs, ledgers, runner output), ext shard files and calibration
# output to runs/ext-remote/<host>/ (a tar stream over ssh; the instabox hosts have no rsync). Final pulls happen
# after the runners exit, so no .eval copy is torn; a copy taken while a runner writes may be.
set -uo pipefail
cd "$(dirname "$0")/.."
while read -r name dest; do
  mkdir -p "runs/ext-remote/$name"
  ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$dest" \
    'cd ~/shared-host-collusion/runs && tar cf - $(ls -d ext shards/*ext*.json calibration 2>/dev/null)' 2>/dev/null \
    | tar xf - -C "runs/ext-remote/$name"
  echo "$name: $(ls runs/ext-remote/$name/ext/logs-*/*.eval 2>/dev/null | wc -l | tr -d ' ') eval files"
done <<HOSTS
instabox instabox
instabox2 instabox2
instabox3 instabox3
vps1 root@161.97.142.53
HOSTS
