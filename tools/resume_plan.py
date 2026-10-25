"""Plan a resume after a host stops: exclude-ids JSON and, optionally, rebalanced shards of the remaining games.

Usage (repo root, venv python; run tools/sync_logs.py first so the VPS logs are current):
  .venv/bin/python tools/resume_plan.py --out-dir runs/resume
      -> runs/resume/exclude-ids.json: every game completed on any host (input for harness/run.py --exclude-ids-file)
  .venv/bin/python tools/resume_plan.py --out-dir runs/resume --pool-hosts vps2 --rebalance mac=0.5,gate vps1=0.5 \
      --host vps1
      -> also runs/resume/vps1.json: vps1's share of the games vps2 did not finish (input for --shard-file)
Rules (same as tools/make_shards.py):
  - Remaining games of one (model, scenario) stream stay on one host. A stream that already has completed games is
    pinned to the host that ran them when that host is in --rebalance (same hardware), unless --no-pin.
  - tirith-gated arms (SHR-GATE-*, PROVOC-GATE-TOLD) go only to the host marked `gate` (the Mac). If no listed host
    is the gate host they are reported as unassigned, never placed elsewhere.
  - Each host's queue is ordered core first (deep, pairs, broad), then scenarios, interleaved like harness/run.py.
Failed games (sample error) are re-run with the same seed (PREREGISTRATION.md: "excluded and re-run with the same
seed if budget allows"), so they are NOT in the exclude list unless --exclude-failed.
Only pool hosts whose runs have stopped: games still in flight on a running host would otherwise run twice. Games
in flight on a stopped host are simply not complete, so they are re-run. --completed-file/--failed-file add ids
(JSON lists) to what the logs show, for manual corrections or offline tests. Writes only under --out-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.arms import ARMS, GameSpec, full_design  # noqa: E402
from harness.run import interleave  # noqa: E402

PRIORITY = {"deep": 0, "pairs": 1, "broad": 2, "scenario": 3}  # as tools/make_shards.py
SHARD_HOSTS = ("mac", "vps1", "vps2")


def parse_weights(args: list[str]) -> tuple[dict[str, float], str | None]:
    weights, gate = {}, None
    for h in args:
        name, spec = h.split("=", 1)
        parts = spec.split(",")
        weights[name] = float(parts[0])
        if weights[name] <= 0:
            raise ValueError(f"weight for {name} must be positive")
        if "gate" in parts[1:]:
            gate = name
    return weights, gate


def order_queue(specs: list[GameSpec]) -> list[GameSpec]:
    out = []
    for grid in sorted(PRIORITY, key=PRIORITY.get):
        out += interleave([s for s in specs if s.grid == grid])
    return out


def plan_resume(design: list[GameSpec], shards: dict[str, list[str]], completed: set[str], failed: set[str], *,
                weights: dict[str, float] | None = None, gate_host: str | None = None,
                pool_hosts: list[str] | None = None, preload: dict[str, int] | None = None,
                completed_on: dict[str, str] | None = None, pin_started: bool = True,
                exclude_failed: bool = False) -> dict:
    """Pure planning step. Returns {"exclude", "rerun_failed", "pool", "shards", "unassigned", "pinned", "notes"}."""
    by_id = {s.game_id: s for s in design}
    unknown = sorted((completed | failed) - set(by_id))
    completed = completed & set(by_id)
    failed = (failed & set(by_id)) - completed  # a game that failed once but later completed is complete
    exclude = sorted(completed | (failed if exclude_failed else set()))
    done = set(exclude)
    if pool_hosts:
        pool_ids = {g for h in pool_hosts for g in shards.get(h, [])}
    else:
        pool_ids = set(by_id)
    pool = [by_id[g] for g in sorted(pool_ids) if g in by_id and g not in done]
    result = {"exclude": exclude, "rerun_failed": sorted(failed - done), "pool": [s.game_id for s in pool],
              "shards": {}, "unassigned": [], "pinned": {}, "notes": []}
    if unknown:
        result["notes"].append(f"{len(unknown)} completed/failed ids are not in the design (pilot?) and were ignored")
    if not weights:
        return result

    load = {h: int((preload or {}).get(h, 0)) for h in weights}
    total = sum(s.rounds for s in pool) + sum(load.values())
    tw = sum(weights.values())
    target = {h: max(weights[h] / tw * total, 1e-9) for h in weights}
    assign: dict[str, list[GameSpec]] = defaultdict(list)
    gated = [s for s in pool if ARMS[s.arm].get("gate")]
    if gate_host in weights:
        assign[gate_host] += gated
        load[gate_host] += sum(s.rounds for s in gated)
    else:
        result["unassigned"] += [s.game_id for s in gated]
        if gated:
            result["notes"].append(f"{len(gated)} tirith-gated games need the gate host; none listed in --rebalance")
    streams: dict[tuple, list[GameSpec]] = defaultdict(list)
    for s in pool:
        if not ARMS[s.arm].get("gate"):
            streams[(s.model_label, s.scenario_id)].append(s)
    # where each stream already ran (completed games), for pinning
    ran_on: dict[tuple, Counter] = defaultdict(Counter)
    for g in completed:
        h = (completed_on or {}).get(g)
        if h:
            s = by_id[g]
            ran_on[(s.model_label, s.scenario_id)][h] += 1
    free = []
    for key in streams:
        hosts_seen = ran_on.get(key)
        home = hosts_seen.most_common(1)[0][0] if hosts_seen else None
        if pin_started and home in weights:
            assign[home] += streams[key]
            load[home] += sum(x.rounds for x in streams[key])
            result["pinned"][f"{key[0]}|{key[1]}"] = home
            if len(hosts_seen) > 1:
                result["notes"].append(f"stream {key} already ran on {dict(hosts_seen)}; pinned to {home}")
        else:
            free.append(key)
    for key in sorted(free, key=lambda k: (-sum(x.rounds for x in streams[k]), k)):
        h = min(weights, key=lambda x: (load[x] / target[x], x))
        assign[h] += streams[key]
        load[h] += sum(x.rounds for x in streams[key])
    for h in weights:
        q = order_queue(assign[h])
        result["shards"][h] = {"host": h, "game_ids": [s.game_id for s in q], "rounds": sum(s.rounds for s in q)}
    placed = [g for v in result["shards"].values() for g in v["game_ids"]]
    assert len(placed) == len(set(placed)), "a game was assigned twice"
    assert set(placed) | set(result["unassigned"]) == set(result["pool"]), "a pool game was lost"
    assert not set(placed) & done, "an excluded game was assigned"
    for h, v in result["shards"].items():
        if h != gate_host:
            assert not any(ARMS[by_id[g].arm].get("gate") for g in v["game_ids"]), "gated game off the gate host"
    return result


def from_logs(hosts: list[str]) -> tuple[set[str], set[str], dict[str, str]]:
    """Completed and failed ids from every host's logs (mac live, VPS from runs/remote), and where each completed."""
    from tools.progress import collect, design_rounds

    completed, failed, where = set(), set(), {}
    for r in collect(hosts, design_rounds()):
        for name, why in r.scan.unreadable.items():
            print(f"WARNING {r.host}: unreadable log {name}: {why}", file=sys.stderr)
        for g, x in r.scan.games.items():
            if x.ok:
                completed.add(g)
                where.setdefault(g, r.host)
            else:
                failed.add(g)
    return completed, failed - completed, where


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, help="where exclude-ids.json and shard files are written")
    ap.add_argument("--hosts", nargs="*", default=list(SHARD_HOSTS), help="hosts whose logs are scanned")
    ap.add_argument("--completed-file", help="extra completed game ids (JSON list)")
    ap.add_argument("--failed-file", help="extra failed game ids (JSON list)")
    ap.add_argument("--no-logs", action="store_true", help="use only --completed-file/--failed-file")
    ap.add_argument("--exclude-failed", action="store_true", help="do not re-run failed games")
    ap.add_argument("--rebalance", nargs="*", default=[], help="host=weight[,gate] ... (as tools/make_shards.py)")
    ap.add_argument("--pool-hosts", nargs="*", help="only re-plan the unfinished games of these hosts' shards")
    ap.add_argument("--shards-dir", default=str(ROOT / "runs" / "shards"), help="original shards")
    ap.add_argument("--preload", nargs="*", default=[], help="host=rounds already queued there (e.g. its own tail)")
    ap.add_argument("--host", help="write only this host's rebalanced shard")
    ap.add_argument("--no-pin", action="store_true", help="do not pin started streams to the host that ran them")
    a = ap.parse_args()

    design = full_design()
    shards = {}
    for h in SHARD_HOSTS:
        p = Path(a.shards_dir) / f"{h}.json"
        if p.exists():
            shards[h] = json.loads(p.read_text())["game_ids"]
    completed, failed, where = (set(), set(), {}) if a.no_logs else from_logs(a.hosts)
    if a.completed_file:
        completed |= set(json.loads(Path(a.completed_file).read_text()))
    if a.failed_file:
        failed |= set(json.loads(Path(a.failed_file).read_text()))
    weights, gate = parse_weights(a.rebalance) if a.rebalance else ({}, None)
    if a.host and a.host not in weights:
        ap.error("--host must be one of the --rebalance hosts")
    preload = {k: int(v) for k, v in (x.split("=", 1) for x in a.preload)}
    res = plan_resume(design, shards, completed, failed, weights=weights, gate_host=gate, pool_hosts=a.pool_hosts,
                      preload=preload, completed_on=where, pin_started=not a.no_pin,
                      exclude_failed=a.exclude_failed)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exclude-ids.json").write_text(json.dumps(res["exclude"], indent=1))
    rounds = {s.game_id: s.rounds for s in design}
    by_id = {s.game_id: s for s in design}
    print(f"completed {len(completed)} | failed (re-run) {len(res['rerun_failed'])} | excluded {len(res['exclude'])}"
          f" -> {out / 'exclude-ids.json'}")
    print(f"pool: {len(res['pool'])} unfinished games, {sum(rounds[g] for g in res['pool'])} rounds"
          f" ({'shards of ' + ', '.join(a.pool_hosts) if a.pool_hosts else 'whole design'})")
    for h, v in res["shards"].items():
        if a.host and h != a.host:
            continue
        (out / f"{h}.json").write_text(json.dumps(v, indent=1))
        n_streams = len({(by_id[g].model_label, by_id[g].scenario_id) for g in v["game_ids"]})
        n_gated = sum(1 for g in v["game_ids"] if ARMS[by_id[g].arm].get("gate"))
        print(f"  {h:6s} games={len(v['game_ids']):4d} rounds={v['rounds']:6d} streams={n_streams} gated={n_gated}"
              f" -> {out / f'{h}.json'}")
    if res["pinned"]:
        print(f"  pinned started streams: {len(res['pinned'])}")
    if res["unassigned"]:
        print(f"  UNASSIGNED: {len(res['unassigned'])} games (see notes)")
    for n in res["notes"]:
        print(f"  note: {n}")
    if res["shards"]:
        print("run each with: harness/run.py --plan full --shard-file <out-dir>/<host>.json "
              "--exclude-ids-file <out-dir>/exclude-ids.json --ledger <that host's ledger> --log-dir <new dir> ...")


if __name__ == "__main__":
    main()
