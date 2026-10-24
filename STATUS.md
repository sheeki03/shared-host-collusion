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
  - nanoGPT returns the exact USD cost per call in `usage.cost`, so the spending cap can be exact.
- **Inspect spike** (`spike/spike_task.py`, GLM 5.3 low, 2 samples, 2:36 wall-clock, 2,784 tokens):
  - Shared scratch: a nonce crossed in both directions, and the file owner showed as uid 1001.
  - Private scratch: no crossing either way. Reader-before-writer control: empty.
  - No network in either container (no DNS, no TCP).
  - The 2-round mini game ran commands as the firm's uid and wrote an `.eval` log.
  - One price failed to parse in the spike's simple parser; the real harness retries once, then falls back to the last price.
  - Observation: agents list and `cat` `/srv/scratch/*` without being asked, so exposure to rival files will be common.
- **Decision: build on Inspect** (no fallback needed).
