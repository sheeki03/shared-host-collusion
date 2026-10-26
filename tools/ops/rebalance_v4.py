"""v4 rebalance: pool the unfinished games of the two slow Contabo runs and split them by measured speed.

Usage: .venv/bin/python runs/rebalance_v4.py host=weight [host=weight ...]
Reads completed game ids from the remote logs (read-only), excludes every game completed anywhere, keeps whole
(model, scenario) streams on one host, orders core grids first, writes runs/shards/<host>-pool.json.
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.arms import ARMS, full_design  # noqa: E402
from harness.run import interleave  # noqa: E402

D = {s.game_id: s for s in full_design()}
REMOTE = {"vps1": ("<user>@<host-ip-vps1>", ["runs/full/logs-vps1", "runs/full/logs-vps1v2"]),
          "vps2": ("<user>@<host-ip-vps2>", ["runs/full/logs-vps2", "runs/full/logs-vps2v2"])}
DONE_SCRIPT = "<scratchpad>/done_ids_json.py"


def remote_done(dest: str, dirs: list[str]) -> list[str]:
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", DONE_SCRIPT, f"{dest}:done_ids_json.py"], check=True)
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", dest,
                        "cd ~/shared-host-collusion && .venv/bin/python ~/done_ids_json.py " + " ".join(dirs)],
                       capture_output=True, text=True, timeout=300)
    line = [l for l in r.stdout.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


def main(args: list[str]) -> None:
    weights = {a.split("=")[0]: float(a.split("=")[1]) for a in args}
    done = set(json.loads((ROOT / "runs" / "done-before-v3.json").read_text()))
    for tag, (dest, dirs) in REMOTE.items():
        ids = remote_done(dest, dirs)
        print(f"{tag}: {len(ids)} completed games in its logs")
        done |= set(ids)
    pool = []
    for shard in ("vps1", "vps2"):  # the v2 shards these runs used (deployed remotely as vps1v2 / vps2v2)
        pool += [g for g in json.loads((ROOT / "runs" / "shards-v2" / f"{shard}.json").read_text())["game_ids"]
                 if g not in done]
    assert not any(ARMS[D[g].arm].get("gate") for g in pool), "gated games must stay on the Mac"
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
        (ROOT / "runs" / "shards" / f"{h}-pool.json").write_text(
            json.dumps({"host": f"{h}-pool", "game_ids": ordered, "rounds": load[h]}, indent=1))
        print(f"{h}-pool: {len(ordered)} games, {load[h]} rounds ({load[h] / total:.0%})")
    (ROOT / "runs" / "done-before-v4.json").write_text(json.dumps(sorted(done)))
    print(f"pool {len(pool)} games / {total} rounds; done anywhere so far {len(done)}")


if __name__ == "__main__":
    main(sys.argv[1:])
