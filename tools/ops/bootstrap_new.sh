#!/usr/bin/env bash
# Bootstrap new instabox hosts in parallel at the frozen commit; logs to runs/vps-<host>-bootstrap.log
cd "$(dirname "$0")/.."
C=$(git rev-parse 'prereg-freeze^{commit}')
for h in "$@"; do
  (
    scp -q -o BatchMode=yes tools/vps_bootstrap.sh "$h:vps_bootstrap.sh" &&
    ssh -o BatchMode=yes "$h" "GATE_GAMES=6 bash vps_bootstrap.sh $C && mkdir -p shared-host-collusion/runs/shards shared-host-collusion/runs/full" \
      > "runs/vps-$h-bootstrap.log" 2>&1
    echo "$h exit $? : $(grep -E 'BOOTSTRAP OK|game_rounds_per_hour' "runs/vps-$h-bootstrap.log" | tr '\n' ' ')"
  ) &
done
wait
