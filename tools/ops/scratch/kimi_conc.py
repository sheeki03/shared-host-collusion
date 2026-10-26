# One-off diagnosis: send the same real Kimi game request N times in parallel (N = 4, 8, 16) and count successes, to
# find the concurrency Kimi's upstream on nanoGPT tolerates. Status codes only; the key is never printed.
import asyncio, copy, glob, json, pathlib, time
import httpx
from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log_sample

cands = [(f, s) for f in sorted(glob.glob("runs/full/logs-instabox2-kimi*/*.eval"))
         for s in read_eval_log_sample_summaries(f) if s.error]
f, s = cands[0]
smp = read_eval_log_sample(f, s.id, s.epoch)
calls = [ev for ev in smp.events if getattr(ev, "event", "") == "model" and getattr(ev, "call", None)]
body = copy.deepcopy(calls[-1].call.request)
for m in body["messages"]:
    c = m.get("content")
    if isinstance(c, str) and c.startswith("attachment://"):
        m["content"] = smp.attachments[c[len("attachment://"):]]
key = (pathlib.Path.home() / ".nanogpt-api-key").read_text().strip()


async def one(client):
    r = await client.post("https://nano-gpt.com/api/v1/chat/completions", json=body, timeout=180,
                          headers={"Authorization": f"Bearer {key}"})
    try:
        err = r.json().get("error")
    except ValueError:
        err = None
    return r.status_code, (err.get("code") if isinstance(err, dict) else err)


async def main():
    async with httpx.AsyncClient() as client:
        for n in (4, 8, 16):
            t0 = time.monotonic()
            res = await asyncio.gather(*[one(client) for _ in range(n)])
            ok = sum(1 for st, _ in res if st == 200)
            codes = sorted({str(c) for st, c in res if st != 200})
            print(f"parallel {n:2d}: {ok}/{n} ok in {time.monotonic() - t0:.1f}s; failures {codes}")
            await asyncio.sleep(10)

asyncio.run(main())
