"""Split the 216 pre-declared extension games into 4 chunks for the dispatcher: long first (critical path), then rep in
two halves (whole model streams), then any + sx. Writes runs/shards-ext/ext-{1..4}.json."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.arms import ext_design  # noqa: E402

specs = ext_design("all")
by = defaultdict(list)
for s in specs:
    by[s.grid].append(s)
rep_models = sorted({s.model_label for s in by["ext-rep"]})
half = {m: i % 2 for i, m in enumerate(rep_models)}
rep_a = [s for s in by["ext-rep"] if half[s.model_label] == 0]
rep_b = [s for s in by["ext-rep"] if half[s.model_label] == 1]
# balance the two rep halves by splitting the remaining model's seeds if model counts are odd
if len(rep_models) % 2:
    last = rep_models[-1]
    moving = [s for s in rep_a if s.model_label == last]
    keep = [s for s in moving if s.seed % 2 == 0]
    rep_a = [s for s in rep_a if s.model_label != last] + keep
    rep_b = rep_b + [s for s in moving if s.seed % 2 == 1]
chunks = {"ext-1": by["ext-long"], "ext-2": rep_a, "ext-3": rep_b, "ext-4": by["ext-any"] + by["ext-sx"]}
out = ROOT / "runs" / "shards-ext"
out.mkdir(parents=True, exist_ok=True)
assert sum(len(v) for v in chunks.values()) == len(specs) == 216
assert len({s.game_id for v in chunks.values() for s in v}) == 216
for name, ss in chunks.items():
    (out / f"{name}.json").write_text(json.dumps({"host": name, "game_ids": [s.game_id for s in ss],
                                                  "rounds": sum(s.rounds for s in ss)}, indent=1))
    print(name, len(ss), "games", sum(s.rounds for s in ss), "rounds")
