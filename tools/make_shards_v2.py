"""Second split, from measured throughput: core grids sharded per machine, scenarios in an overflow queue (no model calls).

Usage: .venv/bin/python tools/make_shards_v2.py <out_dir> <exclude_ids.json|-> host=weight[,gate] [host=weight ...]
- Core grids (deep, pairs, broad) are split across the named hosts in proportion to their measured rounds/hour, with
  whole (model, scenario) streams kept together and tirith-gated arms on the `gate` host (as tools/make_shards.py).
- The scenario grid goes to overflow chunks <out_dir>/spill-1..4.json: tier-1 framings first (chunks 1-2), tier-2 last
  (chunks 3-4, the first to cut under the pre-registered trimming order). Any machine that is free runs the next chunk.
- Games listed in exclude_ids.json (completed elsewhere) are left out.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.arms import ARMS, full_design  # noqa: E402
from harness.run import interleave  # noqa: E402

TIER1 = ("G1-rent", "G3-procurement", "G5-mev", "G7-oracle", "G9-dealer-spreads")
TIER2 = ("G2-wages", "G4-liquidation", "G6-ads", "G8-libor", "G10-legal-negotiation")
CORE_ORDER = ("deep", "pairs", "broad")


def write(out: Path, name: str, specs: list) -> None:
    (out / f"{name}.json").write_text(json.dumps({"host": name, "game_ids": [s.game_id for s in specs],
                                                  "rounds": sum(s.rounds for s in specs)}, indent=1))


def main(out_dir: str, exclude: str, hosts_args: list[str]) -> None:
    done = set(json.loads(Path(exclude).read_text())) if exclude != "-" else set()
    design = [s for s in full_design() if s.game_id not in done]
    hosts, gate_host = {}, None
    for h in hosts_args:
        name, spec = h.split("=", 1)
        parts = spec.split(",")
        hosts[name] = float(parts[0])
        if "gate" in parts[1:]:
            gate_host = name
    core = [s for s in design if s.grid in CORE_ORDER]
    scen = [s for s in design if s.grid == "scenario"]
    total = sum(s.rounds for s in core)
    wsum = sum(hosts.values())
    target = {h: w / wsum * total for h, w in hosts.items()}
    load = {h: 0 for h in hosts}
    assign: dict = defaultdict(list)
    for s in core:
        if ARMS[s.arm].get("gate"):
            assign[gate_host].append(s)
            load[gate_host] += s.rounds
    streams: dict = defaultdict(list)
    for s in core:
        if not ARMS[s.arm].get("gate"):
            streams[(s.model_label, s.scenario_id)].append(s)
    for key in sorted(streams, key=lambda k: -sum(x.rounds for x in streams[k])):
        h = min(hosts, key=lambda x: load[x] / target[x])
        assign[h] += streams[key]
        load[h] += sum(x.rounds for x in streams[key])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for h in hosts:
        ordered = []
        for grid in CORE_ORDER:
            ordered += interleave([s for s in assign[h] if s.grid == grid])
        write(out, h, ordered)
        print(f"{h:6s} core games={len(ordered):4d} rounds={load[h]:6d} ({load[h] / total:5.1%} of core)")
    # overflow: tier-1 scenarios split in two chunks, then tier-2 in two chunks (by whole streams)
    for idx, tier in ((1, TIER1), (3, TIER2)):
        tier_streams: dict = defaultdict(list)
        for s in scen:
            if s.scenario_id in tier:
                tier_streams[(s.model_label, s.scenario_id)].append(s)
        chunks = ([], [])
        for i, key in enumerate(sorted(tier_streams)):
            chunks[i % 2].extend(tier_streams[key])
        for j, ch in enumerate(chunks):
            write(out, f"spill-{idx + j}", interleave(ch))
            print(f"spill-{idx + j} games={len(ch):4d} rounds={sum(s.rounds for s in ch):6d} "
                  f"({'tier-1' if idx == 1 else 'tier-2'})")
    n = sum(len(v) for v in assign.values()) + len(scen)
    assert n == len(design), (n, len(design))
    print(f"total {len(design)} games ({len(done)} excluded as done)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
