import json, sys, glob
sys.path.insert(0, ".")
from harness.arms import full_design
from harness.run import interleave
from analysis import game_set
D = {s.game_id: s for s in full_design()}
shard = json.load(open("runs/shards/instabox-v5.json"))["game_ids"]
dirs = sorted(set(glob.glob("runs/full/logs-*") + glob.glob("runs/remote/*/logs-*")))
done = set(game_set.scan(dirs).kept)
ids = [g for g in shard if D[g].model_id in ("deepseek/deepseek-v4-pro", "openai/gpt-6.1-sol") and g not in done]
pos = {g: i for i, g in enumerate(shard)}
print(len(ids), "games;", sum(D[g].rounds for g in ids), "rounds; grids:", sorted({D[g].grid for g in ids}),
      "; first queue position", min(pos[g] for g in ids) if ids else None, "of", len(shard))
json.dump({"host": "instabox-dsgpt", "game_ids": ids, "rounds": sum(D[g].rounds for g in ids)},
          open("runs/shards/instabox-dsgpt.json", "w"), indent=1)
