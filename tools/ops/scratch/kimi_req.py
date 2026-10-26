# From errored Kimi samples, save the exact request body of each sample's last model call (the one that failed),
# shortest and longest, for a one-off replay.
import glob, json, sys
from inspect_ai.log import read_eval_log_sample_summaries, read_eval_log_sample
out = sys.argv[1]
reqs = []
for f in sorted(glob.glob("runs/full/logs-instabox2-kimi*/*.eval")):
    for s in read_eval_log_sample_summaries(f):
        if not s.error:
            continue
        smp = read_eval_log_sample(f, s.id, s.epoch)
        calls = [ev for ev in smp.events if getattr(ev, "event", "") == "model" and getattr(ev, "call", None)]
        if calls and calls[-1].call.request:
            r = calls[-1].call.request
            reqs.append((len(json.dumps(r)), (s.error or "")[-160:], r, len(calls)))
        if len(reqs) >= 4:
            break
    if len(reqs) >= 4:
        break
reqs.sort(key=lambda r: r[0])
print("samples with a last call", len(reqs), "sizes", [r[0] for r in reqs], "model events", [r[3] for r in reqs])
for name, r in (("short", reqs[0]), ("long", reqs[-1])):
    json.dump(r[2], open(f"{out}/kimi_{name}.json", "w"))
    print(name, r[0], "bytes; messages", len(r[2]["messages"]), "keys", sorted(r[2]), "| err:", r[1].replace("\n", " "))
