#!/usr/bin/env bash
# Restart the confirmatory L2 coder (resumable: already-coded items are skipped) over freshly synced logs.
# CONC sets the concurrency (default 8). Only the confirmatory coder (--out runs/analysis/coding) is stopped.
set -uo pipefail
cd "$(dirname "$0")/.."
pkill -f "out runs/analysis/coding --ledger" && echo "stopped old coder"; /bin/sleep 3
.venv/bin/python tools/sync_logs.py --timeout 600 >/dev/null 2>&1 || echo "sync had warnings"
DIRS=$(ls -d runs/full/logs-* runs/remote/*/logs-* 2>/dev/null | tr '\n' ' ')
nice -n 5 .venv/bin/python -m analysis.llm_coder label $DIRS --lenient --out runs/analysis/coding \
  --ledger runs/analysis/coding/ledger.jsonl --cap 150 --concurrency "${CONC:-8}" >> runs/analysis/coding/label.out 2>&1
echo "coder pass finished at $(date +%T)"
