#!/usr/bin/env bash
# The final confirmatory analysis, extensions, exploratory additions and report fill, in stages (README section 6).
# usage: tools/final_analysis.sh <stage> [<stage> ...]   stages, in order:
#   sync      pull every host's confirmatory logs/ledgers (tools/sync_logs.py) and extension logs (tools/sync_ext.sh)
#   audit     D12 game set counts and the sanitized run manifest; prints every ledger with open reservations
#   judges    S6 monitors with the judge census (D27: budget 60 USD, judge ledger cap 80 USD)       [paid]
#   code      final L2 coder pass over the confirmatory set and E1 (ext-rep) pass, both strict      [paid]
#   handcheck draw the pre-registered hand-check sheet (a person labels runs/analysis/handcheck/handcheck_sheet.csv)
#   agree     agreement of the filled hand-check sheet
#   outputs   analysis.report_outputs (confirmatory), then the excerpts by the pre-declared rules
#   extras    extensions (E1-E5), signalling (E5), A1-A3, the A4 matrix, run concurrently (none reads another's
#             output, nor report_outputs'), so `extras` may run alongside `judges outputs`
#   merge     tools/merge_results.py: report_outputs' results plus the extras into runs/analysis/final
#   fill      appendix from the code, then tools/fill_report.py --confirmatory on the merged results
# Account spend (cost quantity 3) comes from the owner: ACCOUNT_SPEND_USD and ACCOUNT_SPEND_SOURCE, if set, are passed
# to report_outputs. Everything writes under runs/analysis/ (git-ignored) and report/*.filled.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
HOSTS=(instabox instabox2 instabox3 vps1 vps2)
collect() {   # rebuilt before every stage, so files a sync stage just pulled are included
  LOGS=(runs/full/logs-*)
  LEDGERS=(runs/full/ledger-*.jsonl)
  for h in "${HOSTS[@]}"; do
    for d in runs/remote/$h/logs-*; do [ -d "$d" ] && LOGS+=("$d"); done
    for f in runs/remote/$h/ledger-*.jsonl; do [ -f "$f" ] && LEDGERS+=("$f"); done
  done
  EXT_LOGS=()
  for d in runs/ext-remote/*/ext/logs-*; do [ -d "$d" ] && EXT_LOGS+=("$d"); done
}
# Retired ledgers: launches that ended before finishing (their never-settled reservations are reported, not counted).
RETIRED=(runs/full/ledger-mac.jsonl runs/remote/vps1/ledger-vps1.jsonl runs/remote/vps2/ledger-vps2.jsonl   # D6
         runs/full/ledger-macv2.jsonl                                                                      # D10
         runs/remote/vps1/ledger-vps1v2.jsonl runs/remote/vps2/ledger-vps2v2.jsonl                         # D11
         runs/remote/instabox/ledger-instabox-spill-2.jsonl                                                # D13
         runs/remote/instabox/ledger-instabox-core.jsonl runs/remote/instabox/ledger-instabox-spill-3.jsonl
         runs/remote/instabox2/ledger-instabox2-pool.jsonl runs/remote/instabox3/ledger-instabox3-pool.jsonl  # D15
         runs/remote/instabox2/ledger-instabox2-kimi.jsonl                                                 # D24a
         runs/remote/instabox2/ledger-instabox2-kimi2.jsonl                                                # D25
         runs/remote/instabox2/ledger-instabox2-kimi3.jsonl                                                # D28
         runs/remote/instabox2/ledger-instabox2-kimi4.jsonl)                                               # D35
for f in ${EXTRA_RETIRED:-}; do RETIRED+=("$f"); done
CALIB=(--run contabo_idle=runs/ext-remote/vps1/calibration/idle-seed0/isocal-idle-seed0.json
       --run contabo_loaded=runs/ext-remote/vps1/calibration/loaded-seed0/isocal-loaded-seed0.json
       --run instabox_load_stopped=runs/ext-remote/instabox/calibration/loaded-seed0/isocal-loaded-seed0.json
       --run instabox_idle=runs/ext-remote/instabox/calibration/idle-seed0/isocal-idle-seed0.json
       --run instabox3_loaded=runs/ext-remote/instabox3/calibration/loaded-seed0/isocal-loaded-seed0.json
       --run mac_idle=runs/calibration/mac-idle-seed0/isocal-idle-seed0.json)

stage_sync() { $PY tools/sync_logs.py; tools/sync_ext.sh; }

stage_audit() {
  $PY -m analysis.game_set "${LOGS[@]}"
  local m=(--out-dir runs/analysis/manifest --overwrite)
  for f in runs/shards/*.json runs/shards-v2/*.json runs/remote/*/shards/*.json; do [ -f "$f" ] && m+=(--shard "$f"); done
  for f in "${LEDGERS[@]}"; do m+=(--ledger "$f"); done
  $PY tools/run_manifest.py "${m[@]}" "${LOGS[@]}"
  echo "retired ledgers (open reservations on any other ledger fail stage outputs):"; printf '  %s\n' "${RETIRED[@]}"
}

stage_judges() {
  SHC_LEDGER_PATH=runs/analysis/judges/ledger.jsonl SHC_BUDGET_CAP_USD=80.0 $PY -m analysis.monitors \
    --logs "${LOGS[@]}" --out runs/analysis --run-judges --judge-budget 60 --max-workers 6 --judge-concurrency 6
}

stage_code() {
  $PY -m analysis.llm_coder label "${LOGS[@]}" --out runs/analysis/coding \
    --ledger runs/analysis/coding/ledger.jsonl --cap 150 --concurrency 16
  $PY -m analysis.llm_coder label "${EXT_LOGS[@]}" --cohort ext-rep --out runs/analysis/coding-ext \
    --ledger runs/analysis/coding-ext/ledger.jsonl --cap 30 --concurrency 12
}

stage_handcheck() {
  $PY -m analysis.llm_coder handcheck "${LOGS[@]}" --out runs/analysis/coding --handcheck-dir runs/analysis/handcheck
}

stage_agree() {
  $PY -m analysis.llm_coder agreement --out runs/analysis/coding --handcheck-dir runs/analysis/handcheck
}

stage_outputs() {
  local a=(--full-design --accept-flagged-ledger --out runs/analysis --labels runs/analysis/coding/labels.csv
           --coding-summary runs/analysis/coding/summary.json --monitor-results runs/analysis/monitor_results.json)
  [ -f runs/analysis/handcheck/agreement.json ] && a+=(--coding-agreement runs/analysis/handcheck/agreement.json)
  [ -n "${ACCOUNT_SPEND_USD:-}" ] && a+=(--account-spend-usd "$ACCOUNT_SPEND_USD" --account-spend-source "$ACCOUNT_SPEND_SOURCE")
  for d in "${LOGS[@]}"; do a+=(--logs "$d"); done
  for f in "${LEDGERS[@]}"; do a+=(--ledger "$f"); done
  for f in "${RETIRED[@]}"; do a+=(--retired-ledger "$f"); done
  $PY -m analysis.report_outputs "${a[@]}"
  $PY tools/pick_excerpts.py "${LOGS[@]}"
}

stage_extras() {
  local lg=() cl=() el=() pids=() names=() bad=0 i
  for d in "${LOGS[@]}"; do lg+=(--logs "$d"); cl+=(--confirmatory-logs "$d"); done
  for d in "${EXT_LOGS[@]}"; do el+=(--logs "$d"); done
  mkdir -p runs/analysis/extras-logs
  run() { local n=$1; shift; "$@" > "runs/analysis/extras-logs/$n.log" 2>&1 & pids+=($!); names+=("$n"); }
  run ext $PY -m analysis.extensions "${el[@]}" "${cl[@]}" --labels runs/analysis/coding-ext/labels.csv \
    --out runs/analysis/ext
  run signal $PY -m analysis.signalling "${lg[@]}" --cohort confirmatory --out runs/analysis/signal
  run leak $PY -m analysis.leakage_probe "${lg[@]}" --labels runs/analysis/coding/labels.csv --out runs/analysis/x/leak
  run path $PY -m analysis.pathway "${lg[@]}" --labels runs/analysis/coding/labels.csv --out runs/analysis/x/path
  run digits $PY -m analysis.price_digits "${lg[@]}" --out runs/analysis/x/digits
  run isocal $PY tools/isocal_matrix.py "${CALIB[@]}" --out runs/analysis/isocal
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "  ${names[$i]}: ok"; else echo "  ${names[$i]}: FAILED (runs/analysis/extras-logs/${names[$i]}.log)"; bad=1; fi
  done
  [ "$bad" = 0 ]
}

stage_merge() {
  $PY tools/merge_results.py --base runs/analysis/results.json --extra runs/analysis/ext/ext_results.json \
    --extra runs/analysis/signal/signal_results.json --extra runs/analysis/x/leak/x_leak.json \
    --extra runs/analysis/x/path/x_path.json --extra runs/analysis/x/digits/x_digits.json \
    --extra runs/analysis/isocal/isocal_results.json --out runs/analysis/final
}

stage_fill() {
  $PY tools/prompt_diff.py appendix
  $PY tools/fill_report.py --confirmatory --results runs/analysis/final/results.json
  $PY tools/pick_excerpts.py --fill report/APPENDIX.md
}

[ $# -gt 0 ] || { sed -n 2,15p "$0"; exit 2; }
for s in "$@"; do
  declare -F "stage_$s" >/dev/null || { echo "unknown stage: $s" >&2; exit 2; }
  collect; echo "== $s $(date +%T)"; "stage_$s"
done
