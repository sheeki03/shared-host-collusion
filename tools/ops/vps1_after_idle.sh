#!/usr/bin/env bash
# After the idle calibration on vps1 ends: add vps1 to the extension dispatcher; once it runs extension games,
# start the loaded-host calibration there (A4 'loaded' = non-confirmatory game load on the same host).
cd "$(dirname "$0")/.."
D="<user>@<host-ip-vps1>"
for i in $(seq 1 120); do
  ssh -o BatchMode=yes "$D" 'pgrep -f iso_calibration >/dev/null' 2>/dev/null || break
  /bin/sleep 30
done
echo "$(date +%T) idle calibration finished"
ssh -o BatchMode=yes "$D" 'tail -5 ~/shared-host-collusion/runs/calibration/idle-seed0.out' 2>/dev/null | grep -v '^\*\*\*'
.venv/bin/python - <<'PY'
import json; p = "runs/dispatch_ext_hosts.json"; h = json.load(open(p)); h["vps1"] = "<user>@<host-ip-vps1>"
json.dump(h, open(p, "w")); print("vps1 added to extension dispatcher:", h)
PY
for i in $(seq 1 60); do
  ssh -o BatchMode=yes "$D" 'pgrep -f "[Pp]ython -m harness.run --plan ext" >/dev/null' 2>/dev/null && break
  /bin/sleep 30
done
echo "$(date +%T) extension games running on vps1; starting loaded calibration"
/bin/sleep 120
timeout 60 bash runs/run_calibration.sh "$D" loaded 0 || true
echo "$(date +%T) loaded calibration launched"
