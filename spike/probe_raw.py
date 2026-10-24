"""Spike check 1 (raw HTTP): how nanoGPT accepts reasoning effort and reports usage. Three tiny calls per variant."""
import json
import os
import pathlib
import time

import httpx

KEY = pathlib.Path(os.path.expanduser("~/.nanogpt-api-key")).read_text().strip()
URL = "https://nano-gpt.com/api/v1/chat/completions"
MSG = [{"role": "system", "content": "Reply only with a JSON object."},
       {"role": "user", "content": 'Unit cost is $11.20. Reply {"price": <number>} with your price for a product whose '
                                   'competitive price is about $16.'}]
CASES = [
    ("z-ai/glm-5.3", {"reasoning_effort": "low"}),
    ("openai/gpt-6.1-sol", {"reasoning_effort": "low"}),
    ("anthropic/claude-opus-5.5", {"reasoning_effort": "low"}),
    ("z-ai/glm-5.2", {"reasoning_effort": "none"}),
]
out = []
for model, extra in CASES:
    body = {"model": model, "messages": MSG, "max_tokens": 300, **extra}
    t0 = time.time()
    try:
        r = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {KEY}"}, timeout=120)
        dt = time.time() - t0
        j = r.json()
        msg = ((j.get("choices") or [{}])[0].get("message") or {})
        rec = {"model": model, "extra": extra, "status": r.status_code, "secs": round(dt, 1),
               "content": (msg.get("content") or "")[:120], "finish": (j.get("choices") or [{}])[0].get("finish_reason"),
               "usage": j.get("usage"), "x_cost": {k: v for k, v in r.headers.items() if "cost" in k.lower()}}
        if r.status_code != 200:
            rec["error"] = str(j)[:300]
    except Exception as e:  # noqa: BLE001
        rec = {"model": model, "extra": extra, "error": repr(e)[:300]}
    out.append(rec)
    print(json.dumps(rec, ensure_ascii=False))
pathlib.Path(__file__).with_name("probe_raw.json").write_text(json.dumps(out, indent=1))
