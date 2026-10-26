#!/usr/bin/env bash
# Run the pre-declared ISO residual-channel calibration (A4, D16) on a remote host with no confirmatory runner.
# usage: run_calibration.sh <ssh-dest> <idle|loaded> <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=$1; COND=$2; SEED=${3:-0}
REF=$(git rev-parse HEAD)
ssh -o BatchMode=yes "$DEST" "cd ~/shared-host-collusion && if pgrep -f '[Pp]ython -m harness.run --plan full' >/dev/null && [ '$COND' = idle ]; then echo 'a confirmatory runner is alive: refusing idle calibration'; exit 3; fi; \
  git fetch -q origin && git checkout -q --detach $REF && mkdir -p runs/calibration && \
  nohup setsid .venv/bin/python tools/iso_calibration.py --out runs/calibration/$COND-seed$SEED --channels all --condition $COND --seed $SEED \
  > runs/calibration/$COND-seed$SEED.out 2>&1 < /dev/null & echo launched calibration $COND seed $SEED at \$(git rev-parse --short HEAD)"
