#!/usr/bin/env bash
# Stop the dispatcher and every runner on the three instabox hosts (game containers only), for the D13 relaunch.
cd "$(dirname "$0")/.."
pkill -f "runs/dispatcher.py" && echo "dispatcher stopped" || echo "dispatcher not running"
for h in instabox instabox2 instabox3; do
  scp -q -o BatchMode=yes runs/hard_stop.sh "$h:hard_stop.sh"
  echo "$h: $(ssh -o BatchMode=yes "$h" 'bash hard_stop.sh 2>&1 | head -1')"
done
date
