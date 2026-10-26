"""v5 (D13 relaunch): every confirmatory game not yet complete anywhere and not in vps1-pool's live queue, split
across the three instabox hosts by weight; whole (model, scenario) streams kept together; core grids first.

Usage: .venv/bin/python runs/rebalance_v5.py instabox=45 instabox2=40 instabox3=15
Reads only sample ids/status/timestamps (analysis.game_set.scan), never outcomes.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import game_set  # noqa: E402
from harness.arms import ARMS, full_design  # noqa: E402
from harness.run import interleave  # noqa: E402

D = {s.game_id: s for s in full_design()}
dirs = sorted(set(glob.glob(str(ROOT / "runs/full/logs-*")) + glob.glob(str(ROOT / "runs/remote/*/logs-*"))))
gs = game_set.scan(dirs)
complete = set(gs.kept)
vps1_queue = set(json.loads((ROOT / "runs/shards/vps1-pool.json").read_text())["game_ids"])
remaining = [g for g in D if g not in complete and g not in vps1_queue]
gated = [g for g in remaining if ARMS[D[g].arm].get("gate")]
print(f"log dirs {len(dirs)} | complete {len(complete)} | duplicates dropped {gs.n_duplicates_dropped} | "
      f"vps1-pool queue {len(vps1_queue)} (of which complete {len(vps1_queue & complete)})")
print(f"remaining for instabox: {len(remaining)} games, {sum(D[g].rounds for g in remaining)} rounds; "
      f"gated among them: {len(gated)}")
weights = {a.split('=')[0]: float(a.split('=')[1]) for a in sys.argv[1:]}
pool = [g for g in remaining if g not in gated]
total = sum(D[g].rounds for g in pool)
target = {h: w / sum(weights.values()) * total for h, w in weights.items()}
load = {h: 0 for h in weights}
assign = defaultdict(list)
streams = defaultdict(list)
for g in pool:
    streams[(D[g].model_label, D[g].scenario_id)].append(g)
for key in sorted(streams, key=lambda k: -sum(D[g].rounds for g in streams[k])):
    h = min(weights, key=lambda x: load[x] / target[x])
    assign[h] += streams[key]
    load[h] += sum(D[g].rounds for g in streams[key])
for h in weights:
    ordered = []
    for grid in ("deep", "pairs", "broad", "scenario"):
        ordered += [s.game_id for s in interleave([D[g] for g in assign[h] if D[g].grid == grid])]
    tag = f"{h}-v5"
    (ROOT / "runs/shards" / f"{tag}.json").write_text(json.dumps({"host": tag, "game_ids": ordered,
                                                                 "rounds": load[h]}, indent=1))
    print(f"{tag}: {len(ordered)} games, {load[h]} rounds")
(ROOT / "runs/shards/gated-v5.json").write_text(json.dumps({"host": "gated-v5", "game_ids": gated,
                                                            "rounds": sum(D[g].rounds for g in gated)}, indent=1))
print(f"gated-v5 (Mac): {len(gated)} games")
