#!/usr/bin/env bash
# Wait until no confirmatory runner is alive on any machine; print container counts.
SSHO="-o BatchMode=yes -o ConnectTimeout=8"
remote() { ssh $SSHO "<user>@$1" 'echo $(pgrep -f "harness.run --plan full" | wc -l) $(docker ps -q | wc -l)' 2>/dev/null | tail -1; }
for i in $(seq 1 30); do
  m=$(pgrep -f "harness.run --plan full" | wc -l | tr -d ' ')
  c=$(docker ps -q | wc -l | tr -d ' ')
  v1=$(remote "<host-ip-vps1>")
  v2=$(remote "<host-ip-vps2>")
  echo "$(date +%T) mac proc=$m ctr=$c | vps1 proc,ctr=$v1 | vps2 proc,ctr=$v2"
  if [ "$m" = 0 ] && [ "${v1%% *}" = 0 ] && [ "${v2%% *}" = 0 ]; then exit 0; fi
  /bin/sleep 10
done
exit 1
