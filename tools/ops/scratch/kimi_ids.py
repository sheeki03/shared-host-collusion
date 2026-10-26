import json, sys
from inspect_ai.log import list_eval_logs, read_eval_log
ok, bad = set(), set()
for info in list_eval_logs(sys.argv[1]):
    for s in read_eval_log(info).samples or []:
        (bad if s.error else ok).add(s.id)
ids = sorted(g for g in bad if g not in ok)
json.dump({"host": "instabox2-kimi", "game_ids": ids, "rounds": 0}, open(sys.argv[2], "w"), indent=1)
print(len(ids), "ids;", sorted({g.split(":")[1] for g in ids}))
