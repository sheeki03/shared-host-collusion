# One-off diagnosis: rebuild the exact request of an errored Kimi sample's last call (attachments resolved) and replay
# it to nanoGPT as recorded and with single-parameter variants. Prints status and error code only; the key is read
# from ~/.nanogpt-api-key and never printed. About 4 small calls.
import copy, glob, json, pathlib
import httpx
from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log_sample

body = None
cands = [(f, s) for f in sorted(glob.glob("runs/full/logs-instabox2-kimi*/*.eval"))
         for s in read_eval_log_sample_summaries(f) if s.error]
for f, s in cands[:1]:
    if s.error:
        smp = read_eval_log_sample(f, s.id, s.epoch)
        calls = [ev for ev in smp.events if getattr(ev, "event", "") == "model" and getattr(ev, "call", None)]
        body = copy.deepcopy(calls[-1].call.request)
        att = smp.attachments or {}
        for m in body["messages"]:
            c = m.get("content")
            if isinstance(c, str) and c.startswith("attachment://"):
                m["content"] = att[c[len("attachment://"):]]
        print("sample", s.id, "| model calls before failure", len(calls))
        break
print("request bytes", len(json.dumps(body)), "| system chars", len(body["messages"][0]["content"]),
      "| user chars", len(body["messages"][1]["content"]))
key = (pathlib.Path.home() / ".nanogpt-api-key").read_text().strip()
variants = {"as recorded": {}, "no reasoning_effort": {"reasoning_effort": None},
            "max_tokens 400": {"max_tokens": 400}, "no usage.include": {"usage": None}}
for name, change in variants.items():
    b = copy.deepcopy(body)
    for k, v in change.items():
        if v is None:
            b.pop(k, None)
        else:
            b[k] = v
    r = httpx.post("https://nano-gpt.com/api/v1/chat/completions", json=b, timeout=180,
                   headers={"Authorization": f"Bearer {key}"})
    try:
        j = r.json()
    except ValueError:
        j = {}
    err = j.get("error")
    code = (err.get("code") if isinstance(err, dict) else err) if err else None
    print(f"{name:22s} HTTP {r.status_code} code={code} finish={((j.get('choices') or [{}])[0]).get('finish_reason')}")
