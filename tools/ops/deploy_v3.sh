#!/usr/bin/env bash
# v3: the Mac keeps only the tirith-gated games; its remaining core games run on instabox beside the overflow chunks.
set -euo pipefail
cd "$(dirname "$0")/.."
scp -q -o BatchMode=yes runs/shards/instabox-core.json instabox:shared-host-collusion/runs/shards/instabox-core.json
scp -q -o BatchMode=yes runs/launch_remote.sh instabox:launch_remote.sh
ssh -o BatchMode=yes instabox "bash launch_remote.sh instabox-core 350 20"
INSPECT_DOCKER_CLI_CONCURRENCY=40 nohup .venv/bin/python -m harness.run --plan full --shard-file runs/shards/macv3-gated.json \
  --cap 150 --concurrency 12 --init-ledger --ledger runs/full/ledger-macv3-gated.jsonl --log-dir runs/full/logs-macv3-gated \
  > runs/full/macv3-gated.out 2>&1 < /dev/null &
echo "launched macv3-gated pid $!"
