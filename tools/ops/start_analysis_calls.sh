#!/usr/bin/env bash
# Owner-approved (02:20 IST) analysis model calls, incremental over completed games: L2 coder (cap $12) and S6 judges
# (cap $25), each on its own ledger under runs/analysis/. Resumable; rerun after the last games to complete coverage.
set -uo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python tools/sync_logs.py --timeout 600 >/dev/null 2>&1 || echo "sync had warnings"
DIRS=$(ls -d runs/full/logs-* runs/remote/*/logs-* 2>/dev/null | tr '\n' ' ')
echo "log dirs: $(echo $DIRS | wc -w)"
mkdir -p runs/analysis/coding runs/analysis/judges
nice -n 5 .venv/bin/python -m analysis.llm_coder label $DIRS --lenient --out runs/analysis/coding \
  --ledger runs/analysis/coding/ledger.jsonl --cap 12 --concurrency 6 > runs/analysis/coding/label.out 2>&1 &
echo "coder pid $!"
.venv/bin/python - <<'PY'
from pathlib import Path
from harness.ledger import BudgetLedger
p = Path("runs/analysis/judges/ledger.jsonl")
if not p.exists():
    BudgetLedger(p, 25.0)
print("judge ledger ready")
PY
SHC_LEDGER_PATH=runs/analysis/judges/ledger.jsonl SHC_BUDGET_CAP_USD=25.0 nice -n 5 .venv/bin/python -m analysis.monitors \
  --logs $DIRS --out runs/analysis --run-judges --judge-budget 25 --max-workers 6 --judge-concurrency 6 \
  > runs/analysis/judges/monitors.out 2>&1 &
echo "monitors pid $!"
wait
echo "both finished at $(date +%T)"
