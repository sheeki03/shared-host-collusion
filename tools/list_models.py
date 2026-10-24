"""List nanoGPT models (id, context, pricing) matching the families we care about. Key read from ~/.nanogpt-api-key."""
import json
import os
import pathlib
import sys

import httpx

KEY = pathlib.Path(os.path.expanduser("~/.nanogpt-api-key")).read_text().strip()
BASE = "https://nano-gpt.com/api/v1"

r = httpx.get(f"{BASE}/models", params={"detailed": "true"}, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
r.raise_for_status()
data = r.json().get("data", [])
pathlib.Path("/private/tmp/nanogpt-models.json").write_text(json.dumps(data, indent=1))
print("total models:", len(data))
pats = [p.lower() for p in (sys.argv[1:] or [
    "opus", "sonnet", "claude", "gpt-6", "gpt-5", "glm", "deepseek", "gemini", "grok", "qwen3", "kimi",
    "abliterat", "llama-4", "mistral-large", "minimax"])]
for m in data:
    mid = m.get("id", "")
    if any(p in mid.lower() for p in pats):
        pr = m.get("pricing") or {}
        print(f"{mid:70s} ctx={m.get('context_length')} in={pr.get('prompt')} out={pr.get('completion')}")
