#!/usr/bin/env bash
# v4: stop the slow vps1 queue and split its (and vps2's) unfinished games across instabox2, instabox3 and vps1.
set -euo pipefail
cd "$(dirname "$0")/.."
printf '{"instabox": "instabox"}\n' > runs/dispatch_hosts.json      # no overflow chunk may land on vps1 meanwhile
for h in instabox2 instabox3; do echo "$h key: $(ssh -o BatchMode=yes "$h" 'stat -c "%a %s" .nanogpt-api-key')"; done
scp -q -o BatchMode=yes runs/hard_stop.sh "<user>@<host-ip-vps1>:~/hard_stop.sh"
ssh -o BatchMode=yes "<user>@<host-ip-vps1>" 'bash ~/hard_stop.sh' 2>&1 | grep -v '^\*\*\*' || true
.venv/bin/python runs/rebalance_v4.py instabox2=5000 instabox3=2500 vps1=1900
date
