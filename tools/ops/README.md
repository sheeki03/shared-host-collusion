# As-run operational scripts (25–26 Oct 2026)

These are the operational scripts that drove the confirmatory and extension runs of 25–26 Oct (times IST). They were
copied from the git-ignored `runs/` folder, and the files in `scratch/` from the operator's session scratch directory.
Host addresses and paths are redacted (see below). They are kept as a record of what ran, so the relaunch deviations
in `prereg/DEVIATIONS.md` can be checked against code. They are not a supported tool. They expect the owner's
machines (ssh aliases `instabox`, `instabox2`, `instabox3`, and VPS-1/VPS-2 by address) and were run from the repo
root as `runs/<name>`, which is why their paths say `runs/...`. Several of them start paid model calls. Do not run them.

The layout follows `runs/`. The top level holds the scripts, the dispatcher host lists and claim records
(`dispatch_*state.json`), and the exclusion lists of games already complete (`done-*.json`). `shards/`, `shards-v2/`
and `shards-ext/` hold the recorded shard files. These contain only a run tag and game ids.
`shards/instabox2-kimi.json` was used unchanged for `instabox2-kimi` through `kimi4`. `shards/instabox-ext-5.json` is
the copy that ran on the host.

**Redactions.** The only edits are these:

- The VPS-1 and VPS-2 addresses became `<host-ip-vps1>` and `<host-ip-vps2>`.
- The remote login `root@` became `<user>@`, and `/root/` became `~/`.
- The session scratch path became `<scratchpad>`.
- The example address in `dispatcher.py` became `<user>@<host-ip>`.

Where an address was an unquoted shell word, the placeholder is quoted so that `bash -n` still parses the file. API
keys appear only as file paths (`~/.nanogpt-api-key`). No key value is in any file. Nothing else was changed.

| Deviation | Scripts and files |
|---|---|
| first launch (before D6) | `deploy_vps.sh`, `launch_remote.sh` (runs on the VPS; this is its last version, edited at 00:59 so that its default `REF` is the D13 commit, and earlier launches ran an earlier copy), `shards/mac.json`, `vps1.json`, `vps2.json` |
| D6 stop and relaunch | `stop_all.sh`, `hard_stop.sh`, `wait_stopped.sh`; `scratch/done_ids_json.py` and `scratch/merge_done.py` build `done-before-v2.json`; `tools/make_shards_v2.py` (committed) builds `shards-v2/`; `deploy_v2.sh`; `dispatcher.py` with `dispatch_hosts.json` and `dispatch_state.json` assigns the overflow chunks `spill-1..4` |
| D10 Mac keeps only gated games | `scratch/split_v3.py` (reads `done-mac.json`, writes `shards/instabox-core.json`, `macv3-gated.json`, `done-before-v3.json`), `deploy_v3.sh`; the instabox case in `dispatcher.py` `alive()` |
| D11 VPS-2 retired, re-split | `bootstrap_new.sh` (new instabox hosts at the frozen commit), `rebalance_v4.sh`, `rebalance_v4.py` (with `scratch/done_ids_json.py` run on each VPS) writes `shards/*-pool.json` and `done-before-v4.json`; `deploy_v4.sh` |
| D13/D15 relaunch on the D13 commit | `stop_instabox.sh`, `rebalance_v5.py` writes `shards/*-v5.json` and `gated-v5.json`; `deploy_v5.sh` |
| D16 A4 calibration | `run_calibration.sh`, `vps1_after_idle.sh` |
| D17 extension games | `make_ext_chunks.py` writes `shards-ext/`; `dispatcher_ext.py` with `dispatch_ext_hosts.json` and `dispatch_ext_state.json`; `launch_remote_ext.sh` |
| D18a cap raise and second runner | `raise_and_split_instabox.sh`; `scratch/dsgpt.py` writes `shards/instabox-dsgpt.json` |
| D22 Kimi re-run | `scratch/kimi_ids.py` writes `shards/instabox2-kimi.json` |
| D24 (a) Kimi probe, (b) `instabox-ext-5` | (a) `scratch/probe_model.sh` (one tiny call; prints the status only); (b) `scratch/make_tail.py` writes `shards/instabox-ext-5.json` from per-runner progress listings, which are not kept |
| D25 stop of `instabox-ext-5`, coders | `scratch/clean_orphans.sh` (removes orphaned game sandboxes on a host with no runner), `restart_coder.sh`, `run_coder_ext.sh` |
| D26 / D27 analysis caps | `scratch/raise_coder_cap.py` and `restart_coder.sh` (D26); `scratch/raise_judge_cap.py` and `run_judges.sh` (D27); the first pass at 02:20 was `start_analysis_calls.sh` |
| D28 Kimi diagnosis and `kimi4` | `scratch/kimi_req.py`, `scratch/kimi_replay.py` (4 calls) and `scratch/kimi_conc.py` (4 + 8 + 16 calls); `scratch/clean_orphans.sh` |

`status.py` is the read-only progress check used throughout. Neither `runs/` nor the scratch directory has a script
for the VPS-2 stop or for launching the `instabox2-kimi*` runners or `instabox-ext-5`. Their commits, caps and
concurrency are recorded in `prereg/DEVIATIONS.md`. The as-run `runs/sync_ext.sh` is not copied, because it is identical to
`tools/sync_ext.sh` as committed in 6bb434b.
