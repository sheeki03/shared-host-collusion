#!/usr/bin/env bash
# usage: launch_remote_ext.sh <tag> <cap> <concurrency> <ref>   (runs on a VPS; shard at runs/shards/<tag>.json)
# Extension games (prereg/EXTENSIONS.md): separate ledgers and logs under runs/ext/, never runs/full.
set -euo pipefail
cd "$HOME/shared-host-collusion"
git fetch -q origin --tags && git checkout -q --detach "$4"
mkdir -p runs/ext runs/shards
test -s "$HOME/.nanogpt-api-key"
if pgrep -f "[Pp]ython -m harness.run --plan" >/dev/null; then echo "a runner is alive on this host; refusing"; exit 3; fi
nohup setsid .venv/bin/python -m harness.run --plan ext --ext-grid all --shard-file "runs/shards/$1.json" --cap "$2" \
  --concurrency "$3" --init-ledger --ledger "runs/ext/ledger-$1.jsonl" --log-dir "runs/ext/logs-$1" \
  > "runs/ext/$1.out" 2>&1 < /dev/null &
echo "launched $1 pid $! at $(git rev-parse --short HEAD)"
