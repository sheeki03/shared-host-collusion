# Status log

Times are IST. Newest entries at the bottom of each step.

## Step 0: unblock (done)
- 25 Oct ~02:55: Shivom submitted the pre-sprint survey and gave the go-ahead for Steps 1–2.

## Step 1: preflight (done, 25 Oct 02:53)
- Both API keys present, owner-only. Account balance checked.
- All 13 model IDs live on nanoGPT.
- Docker 29.4.3: 10 CPUs, 8 GB. `ubuntu:24.04` present. 3 unrelated containers already running (left alone).
- **Disk: 12 GB free.** The tirith build reuses the existing cache, and run logs are kept compressed.
- tirith RC commit `8856dc60` fetched. cargo 1.99.1.
- Official template link reachable (HTTP 200). Python 3.14 / 3.13, uv 0.12.5.

## Step 2: Inspect spike + tirith RC build (Inspect: GO at 03:00; tirith build running)
- tirith RC: detached worktree `~/tirith-rc-8856dc60`, offline build → `bin/tirith-0.5.0-rc-8856dc60` (log: `bin-build.log`).
- Inspect 0.3.283 in an isolated venv `.venv` (Python 3.13), with numpy, scipy, pandas, matplotlib, httpx, pyyaml and openai.
- **Raw API probe** (`spike/probe_raw.py`, 4 calls, ≈ $0.0013 in total): GLM 5.3, GPT-6.1 Sol, Opus 5.5 and GLM 5.2 all replied with valid JSON in 1.4–3.7 s.
  - `reasoning_effort` is honoured: GLM 5.2 "none" used 0 reasoning tokens; GLM 5.3 "low" used 28.
  - nanoGPT returns a USD charge in raw `usage.cost`; the Inspect spike did not preserve it. Step 3 later added a custom provider and a conservative local admission cap with per-attempt reconciliation.
- **Inspect spike** (`spike/spike_task.py`, GLM 5.3 low, 2 samples, 2:36 wall-clock, 2,784 tokens):
  - Shared scratch: a nonce crossed in both directions, and the file owner showed as uid 1001.
  - Private scratch: no crossing either way. Reader-before-writer control: empty.
  - Both Compose services used `network_mode: none`; the spike dynamically probed no DNS/TCP in firm A. Step 3 sandbox tests later probe both firms.
  - The 2-round mini game ran commands as the firm's uid and wrote an `.eval` log.
  - One price failed to parse in the spike's simple parser; the real harness retries once, then falls back to the last price.
  - Observation: agents list and `cat` `/srv/scratch/*` without being asked, so exposure to rival files will be common.
- **Decision: build on Inspect** (no fallback needed).
- **tirith RC built** (03:02, 7 min 38 s, offline, reusing the build cache):
  - `bin/tirith-0.5.0-rc-8856dc60` reports `tirith 0.5.0`.
  - sha256 `aaa5088571c82b3b1b10cde192f6cb07f2aea813d8b16b330041515a7a02c087`.
  - Source: PR #278 head `8856dc60`, detached worktree `~/tirith-rc-8856dc60`.
- Public repo created: https://github.com/sheeki03/shared-host-collusion (author: Shivom only).

## Step 2: done (25 Oct 03:05)

## Step 3 readiness review (25 Oct ~03:09)
- **GO** to begin Step 3 implementation and offline self-tests; **NO-GO** to the paid pilot until the gates in `STEP3_READINESS_REVIEW.md` pass.
- Inspect proved the shared/private storage route, but the shared spike's combined scorer failed on an unparseable price. Raw provider costs were not preserved as exact per-call costs in the Inspect log.
- The implementation checklist now requires a fail-closed global budget ledger, single-execution shell commands, independent trace labels, container/bulletin hardening, matched-game statistics, and five engines with ten scenario framings.
- Read-only review only: no new model calls. A redacted secret scan found no leaks in the three reachable commits.

## Step 3 implementation and verification (25 Oct, through ~14:55 IST)
- Five economic engines, the budget/provider/runner, sandbox/traces/gate, statistics and the confirmatory analysis bridge were built and pushed in separate commits. No new paid model calls were made.
- Verification: 340 engine tests, 58 analysis tests, and 202 harness tests passed; the one opt-in paid provider test was skipped. Targeted Docker trace, bulletin, inbox and process-cleanup tests passed. Ruff and a redacted working-tree secret scan passed.
- A previous 9-game mock E2E passed but one batch had repeated ~900-second Docker/Inspect stalls. New mock-only waves after trace caching completed without long calls: 3 games × 12 rounds in 216 s; 10 × 4 in 176 s; 20 × 6 in 483 s.
- **Phase 4 remains NO-GO for throughput:** 120 mock game-rounds in 483 s at 20 concurrent games projects to about 47 hours for the 41,640-round full grid before model calls. This is a measured short-wave extrapolation, not a full-length forecast. Work is continuing to reduce Docker operations per round; a new timed benchmark is required before the paid pilot.
- The wedged `shcv2explore-firm_a-1` container was left untouched. Healthy `product-*` containers were not restarted or modified.

## Phase 4 pilot handoff (25 Oct ~15:30 IST)
- **Code and offline/Docker gates: ready for the paid pilot; Step 4 remains on owner hold.** No paid model calls were made in this repair turn. The opt-in real-provider reconciliation test is the first tightly capped paid check when Step 4 is authorized.
- Final verification after the supervisor and batched-snapshot optimizations: **214 harness tests passed, 1 opt-in paid test skipped**; 340 engine tests and 58 analysis tests previously passed. Targeted Docker checks for root-owned inbox, protected bulletin, non-UTF8 filenames, trace labels, UID/capabilities, timeout and process cleanup passed. Ruff and redacted secret scan passed. Test containers cleaned up.
- Mock-only benchmark waves after optimization all passed with no sandbox call over 30 s: 3 × 12 rounds in 133 s; 20 × 6 in 249 s; a true 30 × 4 in 216 s with 30 active games and `INSPECT_DOCKER_CLI_CONCURRENCY=40`; 2 × 40 in 378 s. Product containers stayed healthy and were not modified.
- Runner dry runs generated **14 pilot probes, 28 pilot smoke games and 1,131 full-game specs**. Current read-only balance check: **$128.93**; the script now prints only the USD balance.
- **Full-grid schedule remains a separate decision after the pilot.** The true 30-game mock rate was 2,000 game-rounds/hour, implying roughly 21 h of Docker work for 41,640 planned rounds before real model latency, monitors and report writing. This short-wave projection does not promise deadline completion. The pilot must measure real cost, latency and throughput before funding/freezing any confirmatory run or deciding a contingency.
- The old wedged `shcv2explore-firm_a-1` remains untouched; restarting Docker Desktop would interrupt healthy product containers and was not needed for these tests.

## Step 4 pilot and Step 5 decisions (25 Oct, 21:46–22:23 IST)
- Docker Desktop had quit at 19:16; restarted at 21:50 (product containers came back on their own).
- Provider fixes found by the pilot: nanoGPT's pending-cost estimates (D1) and full-context reservations (D2). Pilot: probes 13/14 (Kimi outage, later back), smoke 28/28; output caps raised (D3). Pilot spend about $3.6.
- Owner decisions: full 40-round grid (D4), $750 top-up (balance $1,126.71), extra machines.
- **Pre-registration frozen at 47a1c35, 22:23:19 IST** (tag `prereg-freeze`).

## Step 6 confirmatory runs (from 22:24 IST; ongoing)
- Machines: the Mac (gated arms; done 00:50), Contabo VPS-1 (done 02:04), the owner's production VPS (retired 00:09 at load 47; site unaffected), three instabox VPS (8/8/4 vCPU).
- Relaunches and rebalancing: D6 (22:46), D10 (23:58), D11 (00:16), D15 (01:31). Network drops on instabox killed about 146 in-flight games (00:37–00:51); fixed by D13 and re-run with the same seeds (D15).
- 505 of 1,131 games complete at the 01:30 sync (D12 rule: 8 duplicates, first-finished copy kept). Remaining games run on the instabox hosts; ETA about 03:15–03:30 IST.
- Spend: about $464 real by 02:09; balance $664.55.

## Analysis, report and additions built overnight (timing and status in D7, D16–D18)
- Analysis machinery: report_outputs, game_set (D12), llm_coder (pinned rubric), monitors (S6), coding fixes, ledger union reconciliation (D8/D8a).
- Report draft `report/REPORT.md` (placeholders only for results), `report/APPENDIX.md` (generated), `report/references.bib` (53 BibTeX entries at this review), channel taxonomy (Appendix F), OpenAI Oct 2026 sandbox red-teaming context (D14 scan).
- Extensions E1–E5 (`prereg/EXTENSIONS.md`, harness D17): replication, 100-round horizon, ISO-ANY, secret exchange, price signalling; 216 games, run after the main grid.
- Exploratory A1–A4 (`prereg/EXPLORATORY.md`, D16): message leakage probe, note-to-board pathway, price-digit audit, ISO residual-channel calibration (idle run on VPS-1 started 02:04).

## Second-review checkpoint (Mon 26 Oct ~02:28 IST)
- Fresh read-only sync showed 671/1,131 complete games at 02:20; the last active host projected roughly 04:22 if its rate held. The progress tool was fixed to exclude copied ledgers as a ghost host; its local ledger settlements are not the actual account bill.
- A separate L2 coder and S6 judge job were already running on the partial log set. Owner approved a **$50 total L2 coder cap**; an atomic cap-change event raised its existing private ledger from $12 without resetting spend or starting another process. The final coding and judged-game sets must match the strict completed-game set after final sync.
- Review fixes underway: trace-unknown null denominators and host-consistent sensitivities, post-freeze/deviation wording, report containment/monitor/cost claims, and a fail-closed editorial export gate. The required LLM Usage Statement remains for the author to supply; raw-log release will be decided after submission.
- A private working copy of the official Google Docs template exists for layout QA. It contains a marked draft without final figures/results and is not shared or submitted. The PDF page count remains unverified pending an accessible PDF from the owner.

## Final collection and analysis preparation (Mon 26 Oct 04:35–06:05 IST)
- Main grid: 1,092 of 1,131 games complete under D12; the remaining 39 are the whole Kimi K3.1 Pro broad stream, which has never completed a game. Third attempt (D24a) hit its $60 cap during a Kimi outage; fourth (D25, new refusal retry budget) and fifth runs were stopped; the sixth runs at concurrency 16 (`instabox2-kimi4`, D28) after a replay diagnosis.
- Extensions: E2 (24/24), E1b (72/72), E3+E4 (47/48; one informative E4 exclusion, D24c) complete; E1a 67/72 at 05:55. A duplicate E1 tail runner was stopped before any game finished because it saturated the account's concurrency (D25).
- Classifiers: confirmatory L2 coder finished the current set (18,329 judgements; cap raised to $150, D26); E1 coder runs on its own ledger (`--cohort ext-rep`); S6 judges ran a census of 991 games ($11.55 this pass, D27). Both re-run on the final set after the Kimi games.
- A4 calibration: six runs over three host types (Contabo idle and loaded, instabox idle, instabox labelled loaded whose load was stopped, instabox3 loaded, Mac idle); matrix by `tools/isocal_matrix.py`. Memory accounting and page-cache timing are channels on every host so far; storage timing (the control) is a weak channel on the Mac only.
- Report: Section 4.6 summary; extensions now Appendix G; A1–A4 Appendix H; A4 matrix Table A1. The final sequence is `tools/final_analysis.sh` (stages sync, audit, judges, code, handcheck, agree, outputs, extras, fill), with `tools/merge_results.py` feeding `tools/fill_report.py`.
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider` at commit 234695a: 1,259 passed, 2 skipped (10 min 20 s). To be re-run after the last code change.
- Still with the owner: the 50 + 50 hand-check labels (sheet drawn once the final set is coded), the LLM Usage Statement, and the account balance and top-up history for cost quantity 3.

## Analysis cut and final analysis (Mon 26 Oct 08:20–08:40 IST)
- Cut at 08:20 (D35): every confirmatory game is complete except the 39-game Kimi K3.1 Pro stream, which completed no game in six launches (provider refusals under load) and is reported as missing. Extensions: E1 144/144, E2 24/24, E3 16/16, E4 28 analysed of 32 (one informative exclusion, D24; three Opus 5.5 SX-ISO games over the pre-registered 20% parse-failure rule).
- Final passes: L2 coder strict pass complete (18,311 items; $34.76 total), E1 coder ($6.27), S6 judges census of 991 games; confirmatory report_outputs (preliminary=False, 1,091/1,131), extensions, signalling, A1–A3, A4 matrix (six runs) merged into runs/analysis/final (8,607 keys).
- Unknown labels (877) block the P1 L2 and P2 claims as pre-registered; both are reported with declared bounds (D29, D29b). Analysis code was tagged `analysis-freeze` before outcomes were first read (D31); later analysis changes are D32–D35.
- Report: result text written; main text cut from about 15 to 8 estimated pages (arms and monitor tables, two figures, pilot and costs moved to appendices). Every result-dependent sentence was re-checked against the final results. `tools/export_docx.sh` writes the .docx for the Google Docs template.
- The confirmatory render (`tools/final_analysis.sh agree outputs extras fill`) waits only on the owner: the 100-item hand-check, the LLM Usage Statement and the account top-up total for cost quantity 3.

## Second external review and fixes (Mon 26 Oct 11:00–13:30 IST, D36)
- Two external reviews of commit 3c3b928. Findings verified against the code and logs before any change; every fix is in D36.
- Trace detector: a firm's own inbox (a separate mount that snapshots never list) was not an alternative source of matching text, so own-inbox JSON that a rival had copied into the shared folder counted as a content crossing. Fixed in `harness/traces.py` with regression tests; saved games are relabelled at analysis time by `analysis/relabel.py`, whose rebuilt inputs reproduce every saved label (1,323 samples) and whose regenerated inboxes match all 81,947 saved inbox reads. Effect: 2,021 content and 26 name labels removed in 118 games; no canary label changed.
- Unknown labels: the two D29 runs are now called imputations; the sharp range over every assignment of the unknowns is added for P1 and the P2 chat part.
- S6 monitors are compared on the judged games every monitor scored; Kimi games are "missing", not exclusions; the exclusion table has one row per planned game; machine names removed from report text, A4 kernel strings and excerpt headers; excerpts included in Appendix D; claims of absence reworded as "not detected"; spontaneous canary transfer in SHR-AWARE reported; OpenAI (2026c) wording and the bargaining citation corrected.
- Ops: `tools/sync_ext.sh` fails loudly and `--verify` confirmed the extension and calibration archive is byte-identical on all four hosts; `tools/final_analysis.sh` runs the extras concurrently; `tools/check_claims.py` saves the claim check; as-run relaunch scripts committed under `tools/ops/` (redacted).
- Re-analysis with the fixed labels: judges from cache ($0 new spend), outputs and extras re-run.
- Claim check on the re-analysed results: 43 PASS, 0 FAIL (`tools/check_claims.py`, saved to runs/analysis/final/claims.txt). Draft: abstract 236 words, main text 8 estimated pages.

## Submission package prepared (Mon 26 Oct 14:20 IST; D37–D39)
- Closed the standalone-loader route to pre-D36 labels and repaired paired-monitor output contracts and captions. Docker fixture failures were corrected without changing the sandbox boundary. Public VPS addresses now use logical destinations backed by private local configuration; Git history is retained.
- Verification: full suite **1,276 passed, 2 skipped**; **52 focused submission-tool checks passed**; **22 excerpt/prompt checks passed**. Strict report filling has **0 unresolved and 0 unfinished fields**; numerical claim check **43 PASS, 0 FAIL**.
- Primary statistical outputs identify clean committed source `a2176a8`. Extensions were regenerated from clean `f08d512` with unchanged results; figure export from `f08d512` preserves the statistical JSON and source tables byte for byte. The final assembly completed successfully after regeneration.
- D40: the 100-item labels described in D38 came from an AI agent and are withdrawn; no AI-review statistic is reported. The pre-registered hand-check was completed by an independent human reviewer on the same blinded sheet after outcomes were known and is reported through `coding.handcheck.*` (Section 4.0). Original classifier labels and primary unknown-label blocks are unchanged. Account-wide reconciliation stays deferred (D38).
- Account-wide balance/top-up reconciliation is deferred. Recorded analysed-game accounting costs are $880.96, including $369.26 without estimate flags, $393.70 in conservative estimates, and $118.01 in retry reservations. The separate API-key usage receipt is scoped and time-stamped; no whole-account bill is claimed.
- The combined report is **32 pages, with Sections 1–6 ending on page 8**. All pages were visually checked; the final five changed pages were rechecked. The local package at `runs/submission/` contains editable and PDF reports plus the complete cited-table archive. Google Docs import can change pagination and has not been performed.
- The public report is `report/SUBMISSION.md`, with the filled technical appendix, five figures, and twenty privacy-reviewed tables. Raw logs, individual review sheets, keys, provider diagnostics and local host configuration remain private. The submission itself has not been sent.

## Final submission build (Mon 26 Oct 14:30–15:30 IST; D40–D41)
- Hand-check (D40): the 100 labels Codex had produced with an AI agent (D38) were withdrawn. An independent human reviewer labelled the same blank blinded sheet; agreement with the classifier 0.95 (Wilson 95% CI 0.888–0.978), Cohen's κ 0.90 (bootstrap 0.80–0.98). Re-analysis changed only the `coding.handcheck.*` keys, `coding.status` and the analysis commit (`a4e6bb4`); all tables are unchanged.
- References: all 54 checked against live sources; none fabricated; four metadata corrections (D41).
- Template: the live Guidelines tab links the same Google Doc we used (byte-identical PDF export). Page 1 now uses the template's own title block. Everything before References ends on page 8; abstract within 150–250 words.
- tirith is named where the report evaluates it (abstract, introduction, contribution 3, Section 4.3, discussion, conclusion), with the repository link in Code and Data and the existing caveats and conflict-of-interest disclosure.
- Checks: strict fill 0 unresolved; claim check 43 PASS, 0 FAIL.
- Final independent review (four read-only checks: visual, consistency, sprint rules, privacy) fixed before release: linked-table captions now sit above their tables; the gate block rate is stated as including public-bulletin reads and evasion as undetected only where traces resolve; contribution 3, Appendix A, cross-references and a stale excerpt label (D38 → D40) corrected; reference URLs are clickable; Table A1 has a one-line summary. Full test suite passed; packager tests 57 passed.
