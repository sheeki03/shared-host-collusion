#!/usr/bin/env bash
# D18: raise the instabox-v5 ledger cap (owner enabled auto-recharge) and run its not-yet-started DeepSeek V4 Pro and
# GPT-6.1 Sol scenario games on a second runner on the same host (a raised cap makes a not-yet-constructed model's
# ledger open fail, so instabox-v5 would fail those games instantly, before any API call).
set -euo pipefail
cd "$(dirname "$0")/.."
ssh -o BatchMode=yes instabox 'cd ~/shared-host-collusion && .venv/bin/python - <<PY
from harness.ledger import BudgetLedger
led = BudgetLedger("runs/full/ledger-instabox-v5.jsonl", 350.0, create=False)
led.raise_cap(650.0, reason="D18: owner enabled auto-recharge; projected shard cost above the 350 USD admission cap")
print("instabox-v5 cap raised; spent", round(led.spent, 2))
PY'
scp -q -o BatchMode=yes runs/shards/instabox-dsgpt.json instabox:shared-host-collusion/runs/shards/instabox-dsgpt.json
scp -q -o BatchMode=yes runs/launch_remote.sh instabox:launch_remote.sh
ssh -o BatchMode=yes instabox 'bash launch_remote.sh instabox-dsgpt 80 16'
date
