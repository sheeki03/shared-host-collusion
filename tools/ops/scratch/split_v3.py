import json, sys
sys.path.insert(0, ".")
from harness.arms import ARMS, full_design
from harness.run import interleave
D = {s.game_id: s for s in full_design()}
done = set(json.load(open("runs/done-mac.json"))) | set(json.load(open("runs/done-before-v2.json")))
mac = json.load(open("runs/shards/macv2.json"))["game_ids"]
rest = [g for g in mac if g not in done]
gated = [g for g in rest if ARMS[D[g].arm].get("gate")]
core = [g for g in rest if not ARMS[D[g].arm].get("gate")]
order = {"deep": 0, "pairs": 1, "broad": 2}
def ordered(ids):
    out = []
    for grid in ("deep", "pairs", "broad"):
        out += [s.game_id for s in interleave([D[g] for g in ids if D[g].grid == grid])]
    return out
for tag, ids in (("macv3-gated", gated), ("instabox-core", core)):
    ids = ordered(ids)
    json.dump({"host": tag, "game_ids": ids, "rounds": sum(D[g].rounds for g in ids)}, open(f"runs/shards/{tag}.json", "w"), indent=1)
    print(tag, len(ids), "games", sum(D[g].rounds for g in ids), "rounds")
print("mac v2 shard", len(mac), "| done on mac", len([g for g in mac if g in done]))
json.dump(sorted(done), open("runs/done-before-v3.json", "w"))
