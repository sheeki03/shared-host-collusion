#!/usr/bin/env bash
# Run hard_stop.sh on the Mac and both VPS.
cd "$(dirname "$0")/.."
for ip in "<host-ip-vps1>" "<host-ip-vps2>"; do
  scp -q -o BatchMode=yes runs/hard_stop.sh "<user>@$ip:~/hard_stop.sh"
done
bash runs/hard_stop.sh > runs/stop-mac.log 2>&1 &
for ip in "<host-ip-vps1>" "<host-ip-vps2>"; do
  echo "== $ip"; ssh -o BatchMode=yes "<user>@$ip" 'bash ~/hard_stop.sh' 2>&1 | grep -v '^\*\*\*'
done
wait
echo "== mac"; cat runs/stop-mac.log
