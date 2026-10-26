#!/usr/bin/env bash
# E1 (ext-rep) L2 coder on its own ledger (prereg/EXTENSIONS.md E1 labels), over freshly pulled extension logs.
# Resumable; CONC sets the concurrency (default 8).
set -uo pipefail
cd "$(dirname "$0")/.."
pkill -f "out runs/analysis/coding-ext --ledger" && echo "stopped old E1 coder"; /bin/sleep 3
tools/sync_ext.sh >/dev/null 2>&1 || echo "ext sync had warnings"
DIRS=$(ls -d runs/ext-remote/*/ext/logs-* 2>/dev/null | tr '\n' ' ')
nice -n 5 .venv/bin/python -m analysis.llm_coder label $DIRS --cohort ext-rep --lenient --out runs/analysis/coding-ext \
  --ledger runs/analysis/coding-ext/ledger.jsonl --cap 30 --concurrency "${CONC:-8}" >> runs/analysis/coding-ext/label.out 2>&1
echo "E1 coder pass finished at $(date +%T)"
