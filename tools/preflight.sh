#!/bin/bash
# Step 1 preflight: read-only checks (no paid model calls, no downloads).
set -u
PY=/opt/homebrew/bin/python3.14
cd /Users/home/shared-host-collusion || exit 1

echo "== keys (presence and permissions only)"
for f in ~/.nanogpt-api-key ~/.firecrawl-api-key; do
  if [ -s "$f" ]; then printf '%s: present, perms %s\n' "$f" "$(stat -f %Lp "$f")"; else echo "$f: MISSING"; fi
done

echo "== nanoGPT balance"
$PY tools/balance.py 2>&1 | head -1 | grep -o '"usd_balance":"[0-9.]*"'

echo "== nanoGPT model IDs (fresh list)"
$PY tools/list_models.py "opus-5.5" "gpt-6.1-sol" "gemini-3.1-pro-preview" "glm-5.3" "deepseek-v4-pro" "sonnet-5.5" \
  "grok-4.7" "kimi-k3.1-pro" "qwen3.8-max-prime" "glm-5.2" "abliterated-model-large" "deepseek-v4.1-flash-1017" > /private/tmp/models_now.txt 2>&1
for id in anthropic/claude-opus-5.5 openai/gpt-6.1-sol google/gemini-3.1-pro-preview z-ai/glm-5.3 deepseek/deepseek-v4-pro \
          anthropic/claude-sonnet-5.5 x-ai/grok-4.7 moonshotai/kimi-k3.1-pro qwen/qwen3.8-max-prime z-ai/glm-5.2 \
          abliteration-ai/abliterated-model-large deepseek/deepseek-v4.1-flash-1017 abliteration-ai/abliterated-model-large-flash; do
  if grep -q "^$id " /private/tmp/models_now.txt; then echo "  ok   $id"; else echo "  MISSING $id"; fi
done

echo "== Docker"
docker info --format 'server {{.ServerVersion}} | cpus {{.NCPU}} | mem {{.MemTotal}}' 2>&1 | head -1
docker image inspect ubuntu:24.04 --format 'ubuntu:24.04 present ({{.Size}} bytes)' 2>&1 | head -1
echo "running containers: $(docker ps -q | wc -l | tr -d ' ')"
df -h / | tail -1 | awk '{print "disk free: " $4}'

echo "== tirith RC commit"
git -C /Users/home/tirith-next-cycle fetch -q origin release/v0.5.0 2>&1 | head -2
git -C /Users/home/tirith-next-cycle cat-file -t 8856dc60 2>&1 | head -1
git -C /Users/home/tirith-next-cycle log -1 --format='%h %cd %s' --date=short 8856dc60 2>&1 | head -1
command -v cargo >/dev/null && cargo --version

echo "== official template doc reachable"
curl -s -o /dev/null -w 'template HTTP %{http_code}\n' -L "https://docs.google.com/document/d/1Y5wd3VRbXU5N08BvAGiJUQTaWbMCSPjhzXsR-ozuehg/edit"

echo "== python + uv"
$PY --version; command -v uv && uv --version
