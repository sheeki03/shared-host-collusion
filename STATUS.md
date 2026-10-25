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
