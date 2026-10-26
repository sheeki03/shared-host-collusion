#!/usr/bin/env bash
# usage: launch_remote.sh <tag> <cap> <concurrency>   (runs on a VPS; repo at ~/shared-host-collusion; shard at runs/shards/<tag>.json)
set -euo pipefail
cd "$HOME/shared-host-collusion"
git fetch -q origin --tags && git checkout -q --detach "${REF:-595f444513e557d2ccdbf6fa5aa8aee3af3b9bc0}"
mkdir -p runs/full runs/shards
test -s "$HOME/.nanogpt-api-key"
DOCKER_RUN=""
docker info >/dev/null 2>&1 || DOCKER_RUN="sg docker -c"
if [ -n "$DOCKER_RUN" ]; then
  nohup setsid sg docker -c ".venv/bin/python -m harness.run --plan full --shard-file runs/shards/$1.json --cap $2 \
    --concurrency $3 --init-ledger --ledger runs/full/ledger-$1.jsonl --log-dir runs/full/logs-$1" \
    > "runs/full/$1.out" 2>&1 < /dev/null &
else
  nohup setsid .venv/bin/python -m harness.run --plan full --shard-file "runs/shards/$1.json" --cap "$2" \
    --concurrency "$3" --init-ledger --ledger "runs/full/ledger-$1.jsonl" --log-dir "runs/full/logs-$1" \
    > "runs/full/$1.out" 2>&1 < /dev/null &
fi
echo "launched $1 pid $! at $(git rev-parse --short HEAD)"
