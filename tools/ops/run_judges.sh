#!/usr/bin/env bash
# S6 monitors with paid judges over the synced confirmatory logs (D27: budget 60 USD = census of the evaluation set;
# judge ledger cap 80 USD after an audited raise). Resumable: judge replies and tirith verdicts are cached under
# runs/analysis/cache, so a re-run pays only for games not judged yet. Run tools/sync_logs.py first.
set -uo pipefail
cd "$(dirname "$0")/.."
DIRS=$(ls -d runs/full/logs-* runs/remote/*/logs-* 2>/dev/null | tr '\n' ' ')
SHC_LEDGER_PATH=runs/analysis/judges/ledger.jsonl SHC_BUDGET_CAP_USD=80.0 nice -n 5 .venv/bin/python -m analysis.monitors \
  --logs $DIRS --out runs/analysis --run-judges --judge-budget 60 --max-workers 6 --judge-concurrency "${CONC:-6}" \
  ${EXTRA:-} > runs/analysis/judges/monitors.out 2>&1
echo "monitors finished at $(date +%T)"
