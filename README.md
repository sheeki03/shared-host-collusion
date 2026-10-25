# Shared-host collusion study (Apart AI Collusion Research Sprint, Oct 2026)

Do co-located LLM agents that work for competing firms collude through the machine they share? This
repository holds all sprint work: harness, market engines, pre-registration, analysis and paper.

**Timeline.** Built for the Apart Research AI Collusion Research Sprint (23–25 Oct 2026; submissions close
25 Oct 23:59 AoE). Work started on 25 Oct 2026, ~00:58 IST. The commit history is the record of what was done
and when. The pre-registration is committed before any confirmatory run.
The study design is published in `prereg/` before any confirmatory run.

## Layout

| Path | What |
|---|---|
| `STATUS.md` | Step-by-step progress log with timestamps (read this to track the work) |
| `harness/` | Core game: `econ.py` (logit Bertrand benchmarks), `sandbox.py` (Docker pair), game loop, prompts, gate |
| `engines/` | Cross-domain market modules (auction, reporting, pricing re-framings, market making, bargaining) |
| `spike/` | 90-minute Inspect go/no-go spike |
| `prereg/` | `PREREGISTRATION.md` and its frozen hash manifest |
| `runs/pilot/`, `runs/full/` | Run logs (pilot seeds are never used in final estimates) |
| `analysis/` | Statistics, monitor replay, figures |
| `paper/` | Report drafts and figures for the PDF |
| `tools/` | Budget, model list, balance, Firecrawl search, preflight, tirith RC build |
| `bin/` | Pinned tirith 0.5.0 RC binary (PR #278 head `8856dc60`) and its sha256 |

## Preexisting vs sprint work
- **Preexisting (written before the sprint):** tirith (github.com/sheeki03/tirith, AGPL-3.0), used unmodified as an
  instrument. The binary is built from PR #278 head `8856dc60` in a separate worktree (`~/tirith-rc-8856dc60`).
- **Third-party (credited):** the Fish-style prompt starter by Lucia Sauer / Poseidon Research (MIT), UK AISI Inspect
  (MIT), ControlArena monitor prompt (MIT), MUTE nonce-test design (MIT).
- **Sprint work (25–26 Oct 2026):** everything else in this folder.

## Reproducing this study

Run everything from the repo root. Commands marked $0 make no paid model calls.

**1. Environment** (Python 3.13, [uv](https://github.com/astral-sh/uv); Docker with Compose v2 for the games and
sandbox tests):

```bash
uv venv -p 3.13 .venv
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Paid runs read the nanoGPT key from `~/.nanogpt-api-key` (chmod 600); it is never logged.

**2. tirith RC** (needed only for the gated arms and the S6 replay). The pinned binary is
`bin/tirith-0.5.0-rc-8856dc60`, built unmodified from tirith PR #278 head `8856dc60`. `tools/build_tirith_rc.sh`
builds it on the author's machine (its paths are hard-coded and it builds offline from an existing checkout). Elsewhere:

```bash
git clone https://github.com/sheeki03/tirith ../tirith && cd ../tirith
git fetch origin pull/278/head && git checkout --detach 8856dc60
cargo build --release --locked --bin tirith
cp target/release/tirith ../shared-host-collusion/bin/tirith-0.5.0-rc-8856dc60 && cd ../shared-host-collusion
shasum -a 256 bin/tirith-0.5.0-rc-8856dc60; cat bin/tirith-0.5.0-rc-8856dc60.sha256   # first fields must match
```

`harness/gate.py` fails closed (every command blocked and flagged) unless the binary's sha256 equals the pinned
`aaa5088571c82b3b1b10cde192f6cb07f2aea813d8b16b330041515a7a02c087`. A build with another toolchain may not be
byte-identical; if you pin your own hash instead, your gate is not the study's binary and should be reported as such.

**3. Offline self-tests ($0).**

```bash
.venv/bin/python -m pytest -q engines/tests analysis/tests   # pure Python
.venv/bin/python -m pytest -q harness/tests                  # Docker sandbox tests; the paid test runs only if SHC_REAL_CALLS is set
.venv/bin/python harness/tests/e2e_offline.py                # mock-LLM positive/negative control of the whole pipeline (Docker)
.venv/bin/python tools/bench_offline.py run-mock --confirm-docker --games 3 --rounds 12 --concurrency 3
.venv/bin/python -m harness.run --plan full --cap 1 --dry-run   # writes the 1,131 game specs, makes no calls
```

**4. Pilot** (paid; seeds 9000+, never used in estimates). `--init-ledger` creates a new ledger once and refuses an
existing one; resume a ledger by omitting it.

```bash
.venv/bin/python -m harness.run --plan pilot-probes --cap <usd> --concurrency <n> --init-ledger \
  --ledger runs/pilot/ledger.jsonl --log-dir runs/pilot/logs-probes
.venv/bin/python -m harness.run --plan pilot-smoke --cap <usd> --concurrency <n> --init-ledger \
  --ledger runs/pilot/ledger-smoke.jsonl --log-dir runs/pilot/logs-smoke
.venv/bin/python tools/pilot_report.py runs/pilot/logs-probes runs/pilot/logs-probes2 runs/pilot/logs-smoke
```

The study's pilot ran the 14 probes on a $60 ledger, a 9-game probe re-run after deviations D1–D2 on a new $20
ledger (`runs/pilot/logs-probes2`), and the 28 smoke games on a $15 ledger.

**5. Confirmatory run** (paid). The original core arms, prompts, engines and hypotheses are anchored at tag
`prereg-freeze` (commit `47a1c35`). Post-freeze provider retry, operational and extension changes make
`git diff --stat prereg-freeze -- harness engines` nonempty; `prereg/DEVIATIONS.md` records their timing and scope.
Host rebalancing later split some matched streams across machines. Every tirith-gated game ran on the host with the
pinned binary (the `,gate` host):

```bash
.venv/bin/python tools/make_shards_v2.py runs/shards-v2 - mac=0.34,gate vps1=0.52 vps2=0.14   # $0; '-' = exclude nothing
bash tools/vps_bootstrap.sh <commit>   # on each Linux runner: Docker, uv, pinned env, $0 mock gate (copy the key separately)
.venv/bin/python -m harness.run --plan full --shard-file runs/shards-v2/<host>.json --cap <usd> --concurrency <n> \
  --init-ledger --ledger runs/full/ledger-<host>.jsonl --log-dir runs/full/logs-<host>
```

`make_shards_v2.py` splits the core grids (deep, pairs, broad) by host weight and puts the 360 scenario games in four
overflow chunks (`spill-1.json` to `spill-4.json`, tier-1 framings first). Run each chunk with the same `harness.run`
command (`--shard-file runs/shards-v2/spill-<k>.json`, its own ledger and log directory) on whichever host is free.

How the study ran (deviation D6). A first split from `tools/make_shards.py` was stopped about 13 minutes in. The 6
games that had completed were kept and excluded from the relaunch. The relaunch split the core games three ways:

| Host | Games | Game-rounds | Cap | Games at once |
|---|---|---|---|---|
| Mac (all gated arms) | 257 | 10,280 | $450 | 24 |
| VPS-1 | 403 | 16,120 | $550 | 18 |
| VPS-2 | 105 | 4,200 | $150 | 6 |

The four scenario chunks (96, 84, 96 and 84 games) each went to the first free host through an overflow dispatcher,
with a $60 cap per chunk. The shard files, the dispatcher and the launch scripts are kept with the run outputs under
the ignored `runs/` folder (`runs/shards-v2/`, `runs/dispatcher.py`, `runs/deploy_v2.sh`, `runs/launch_remote.sh`).

With the 6 pre-relaunch games excluded (`runs/done-before-v2.json` in place of `-`), the `make_shards_v2.py` command
above reproduces the four chunks exactly, but not the core split: it gives 10,280 / 15,960 / 4,360 game-rounds where
the run used 10,280 / 16,120 / 4,200. The exact assignment requires the recorded shard files under local ignored
`runs/`; they are not currently in the public repository. The published generator reproduces the design and scenario
chunks, but not the final operational host assignment. Re-running a seed repeats its exogenous draws, move order and
canaries, while stochastic model responses can differ.

`tools/resume_plan.py` re-plans unfinished games for `--shard-file` / `--exclude-ids-file`. `tools/sync_logs.py`
copies a VPS's `runs/full/logs-<host>/` and its ledger to `runs/remote/<host>/` (read-only on the remote side).

**6. Analysis and report** ($0 except the two classifiers, which are paid and each need their own ledger and cap).
Run from the repo root once every runner has finished. The full confirmatory sequence:

```bash
# 1. Pull every remote host's logs and ledgers to runs/remote/<host>/ (read-only on the remote side).
.venv/bin/python tools/sync_logs.py
# 2. Every log directory and every game ledger (bash or zsh arrays). A stopped launch's ledger (D6, D10, D11) is
#    retired: its never-settled reservations are reported, never counted. Add any later stopped runner's ledger.
LOGS=(runs/full/logs-* runs/remote/*/logs-*)
LEDGERS=(runs/full/ledger-*.jsonl runs/remote/*/ledger-*.jsonl)
RETIRED=(runs/full/ledger-mac.jsonl runs/full/ledger-macv2.jsonl runs/remote/vps1/ledger-vps1.jsonl
         runs/remote/vps1/ledger-vps1v2.jsonl runs/remote/vps2/ledger-vps2.jsonl runs/remote/vps2/ledger-vps2v2.jsonl)
.venv/bin/python -m analysis.game_set "${LOGS[@]}"   # $0: counts, duplicates and the copy kept (ids, timestamps)
# 3. S6 monitor replay and judges -> runs/analysis/monitor_results.json, monitor_scores.csv, tables/monitors.md.
#    The judges are paid, on a dedicated ledger created once (outside the game-ledger globs above).
mkdir -p runs/analysis
.venv/bin/python -c "from harness.ledger import BudgetLedger; BudgetLedger('runs/analysis/judge-ledger.jsonl', 10.0)"
SHC_LEDGER_PATH=runs/analysis/judge-ledger.jsonl SHC_BUDGET_CAP_USD=10 \
  .venv/bin/python -m analysis.monitors --logs "${LOGS[@]}" --out runs/analysis --judge-budget 10 --run-judges
# 4. L2 / chat labels (paid; llm_coder creates its own ledger), the blinded hand-check sheet (a person fills
#    runs/analysis/handcheck/handcheck_sheet.csv), then the agreement.
.venv/bin/python -m analysis.llm_coder label "${LOGS[@]}" --out runs/analysis/coding \
  --ledger runs/analysis/coding/ledger-l2-coder.jsonl --cap <usd>
.venv/bin/python -m analysis.llm_coder handcheck "${LOGS[@]}" --out runs/analysis/coding \
  --handcheck-dir runs/analysis/handcheck
.venv/bin/python -m analysis.llm_coder agreement --out runs/analysis/coding --handcheck-dir runs/analysis/handcheck
# 5. Every estimate, table and figure -> runs/analysis/results.json, tables/, figures/, summary.md (fail-closed).
ARGS=(--full-design --accept-flagged-ledger --out runs/analysis --labels runs/analysis/coding/labels.csv
      --coding-summary runs/analysis/coding/summary.json --coding-agreement runs/analysis/handcheck/agreement.json
      --monitor-results runs/analysis/monitor_results.json)
for d in "${LOGS[@]}"; do ARGS+=(--logs "$d"); done
for f in "${LEDGERS[@]}"; do ARGS+=(--ledger "$f"); done       # a retired one listed here too is read once
for f in "${RETIRED[@]}"; do ARGS+=(--retired-ledger "$f"); done
.venv/bin/python -m analysis.report_outputs "${ARGS[@]}"
# 6. Excerpts by the pre-declared rules, the appendix from the code, then the filled report and appendix.
.venv/bin/python tools/pick_excerpts.py "${LOGS[@]}"     # -> runs/analysis/excerpts.md and excerpts.json
.venv/bin/python tools/prompt_diff.py appendix            # regenerates report/APPENDIX.md (--check detects drift)
.venv/bin/python tools/fill_report.py --confirmatory      # report/REPORT.md -> report/REPORT.filled.md
.venv/bin/python tools/pick_excerpts.py --fill report/APPENDIX.md   # -> report/APPENDIX.filled.md
```

Game set (deviation D12, `analysis/game_set.py`, used by both `analysis.monitors` and `analysis.report_outputs`):
complete games are admitted from logs of any status (stopped launches stay "started"); a game completed more than
once keeps the copy that finished first (ties: earlier log start, then log path), the others are listed in
`tables/duplicates` and `meta.n_duplicates_dropped`; errored samples of a game completed elsewhere are superseded,
not exclusions. Ledger copies that share a ledger ID (for example a second synced copy) are read once.
Every saved call is reconciled, including those of dropped and superseded samples (`meta.cost_all_usd`; the analysed
games alone are `meta.cost_kept_games_usd`). `--accept-flagged-ledger` (D8) admits attempts settled at their
reservation because the provider returned no cost (`transport_error:*`, `missing_cost:*`), never `over_est_max`.
Open reservations on a ledger not passed as `--retired-ledger` fail a confirmatory run (a runner still live).

`analysis.report_outputs --preliminary` (same `--logs` and ledgers, no labels needed) reports whatever games are
complete and never claims a result.
