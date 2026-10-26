#!/usr/bin/env bash
# One tiny chat call to a nanoGPT model; prints HTTP status and a short body excerpt only (no key echo).
M="${1:?model id}"
curl -s -m 90 -o "<scratchpad>/ops/probe.json" -w "HTTP %{http_code} in %{time_total}s\n" \
  -H @<(printf 'Authorization: Bearer %s\nContent-Type: application/json\n' "$(cat ~/.nanogpt-api-key)") \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":400}" \
  https://nano-gpt.com/api/v1/chat/completions
head -c 400 "<scratchpad>/ops/probe.json"; echo
