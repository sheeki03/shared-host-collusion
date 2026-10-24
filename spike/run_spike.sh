#!/bin/bash
# Run the Inspect spike against one nanoGPT model. Usage: run_spike.sh <nanogpt-model-id> <reasoning_effort>
set -eu
cd /Users/home/shared-host-collusion
export NANOGPT_API_KEY="$(cat ~/.nanogpt-api-key)"
export NANOGPT_BASE_URL="https://nano-gpt.com/api/v1"
MODEL="${1:-z-ai/glm-5.3}"
EFFORT="${2:-low}"
.venv/bin/inspect eval spike/spike_task.py --model "openai-api/nanogpt/$MODEL" --reasoning-effort "$EFFORT" \
  --max-tokens 400 --log-dir runs/spike-logs --max-samples 2 --no-log-realtime --display plain 2>&1 | tail -25
