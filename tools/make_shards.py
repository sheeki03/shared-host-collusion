"""Split the confirmatory grid across machines by capacity (no model calls).

Usage: .venv/bin/python tools/make_shards.py <out_dir> host=weight[,gate] [host=weight ...]
  e.g. tools/make_shards.py runs/shards mac=0.45,gate vps1=0.40 vps2=0.15
Rules:
  - Whole (model, scenario) streams stay on one machine, so matched arms and seeds share hardware and time.
  - tirith-gated arms (SHR-GATE-*, PROVOC-GATE-TOLD) run only on the host marked `gate` (the pinned binary lives there).
  - Each host's queue is ordered core first (deep, pairs, broad), then scenarios, interleaved as harness/run.py does,
    so a late run loses the tail of the pre-registered trimming order, not the primary contrasts.
Writes <out_dir>/<host>.json = {"host", "game_ids", "rounds"} and prints the split.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.arms import ARMS, full_design  # noqa: E402
from harness.run import interleave  # noqa: E402

PRIORITY = {"deep": 0, "pairs": 1, "broad": 2, "scenario": 3}


def main(out_dir: str, hosts_args: list[str]) -> None:
    hosts, gate_host = {}, None
    for h in hosts_args:
        name, spec = h.split("=", 1)
        parts = spec.split(",")
        hosts[name] = float(parts[0])
        if "gate" in parts[1:]:
            gate_host = name
    if gate_host is None:
        raise SystemExit("mark one host with ',gate'")
    total_w = sum(hosts.values())
    design = full_design()
    total_rounds = sum(s.rounds for s in design)
    target = {h: w / total_w * total_rounds for h, w in hosts.items()}
    load = {h: 0 for h in hosts}
    assign: dict[str, list] = defaultdict(list)
    gated = [s for s in design if ARMS[s.arm].get("gate")]
    for s in gated:
        assign[gate_host].append(s)
        load[gate_host] += s.rounds
    streams: dict = defaultdict(list)
    for s in design:
        if not ARMS[s.arm].get("gate"):
            streams[(s.model_label, s.scenario_id)].append(s)
    # largest streams first, each to the host furthest below its target (greedy balancing)
    for key in sorted(streams, key=lambda k: -sum(x.rounds for x in streams[k])):
        h = min(hosts, key=lambda x: load[x] / target[x])
        assign[h] += streams[key]
        load[h] += sum(x.rounds for x in streams[key])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for h in hosts:
        specs = assign[h]
        ordered = []
        for grid in sorted(PRIORITY, key=PRIORITY.get):
            ordered += interleave([s for s in specs if s.grid == grid])
        (out / f"{h}.json").write_text(json.dumps({"host": h, "game_ids": [s.game_id for s in ordered],
                                                   "rounds": load[h]}, indent=1))
        by_grid = defaultdict(int)
        for s in specs:
            by_grid[s.grid] += s.rounds
        print(f"{h:6s} games={len(specs):4d} rounds={load[h]:6d} ({load[h] / total_rounds:5.1%}, target "
              f"{target[h] / total_rounds:5.1%}) {dict(by_grid)}")
    n = sum(len(v) for v in assign.values())
    assert n == len(design) and len({s.game_id for v in assign.values() for s in v}) == len(design)
    print(f"total {n} games, {total_rounds} rounds; gated arms on {gate_host}: {len(gated)} games")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
