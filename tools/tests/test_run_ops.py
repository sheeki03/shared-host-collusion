"""Synthetic tests for tools/resume_plan.py, tools/sync_logs.py and tools/progress.py (no network, no Docker)."""
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.arms import ARMS, full_design  # noqa: E402
from tools.progress import parse_ledger, scan_logs  # noqa: E402
from tools.resume_plan import PRIORITY, plan_resume  # noqa: E402
from tools.sync_logs import promote_ledger, validate_eval  # noqa: E402

DESIGN = full_design()
BY_ID = {s.game_id: s for s in DESIGN}
SHARDS = {h: json.loads((ROOT / "runs" / "shards" / f"{h}.json").read_text())["game_ids"] for h in ("mac", "vps1", "vps2")}
PILOT_LOG = sorted((ROOT / "runs" / "pilot" / "logs-smoke").glob("*.eval"))[0]


def synthetic_state():
    """mac finished its first 100 games, vps1 its first 50 (2 failed), vps2 died after 20 (1 failed, 1 failed-then-ok)."""
    completed, failed, where = set(), set(), {}
    for h, n in (("mac", 100), ("vps1", 50), ("vps2", 20)):
        for g in SHARDS[h][:n]:
            completed.add(g)
            where[g] = h
    failed |= {SHARDS["vps1"][50], SHARDS["vps1"][51], SHARDS["vps2"][20]}
    failed.add(SHARDS["vps2"][0])  # failed once, later completed: must count as complete
    return completed, failed, where


def test_shards_partition_design():
    ids = [g for v in SHARDS.values() for g in v]
    assert len(ids) == len(set(ids)) == len(DESIGN)


def test_exclude_only_completed_and_failed_rerun():
    completed, failed, where = synthetic_state()
    res = plan_resume(DESIGN, SHARDS, completed, failed)
    assert set(res["exclude"]) == completed
    assert SHARDS["vps2"][0] not in res["rerun_failed"]
    assert set(res["rerun_failed"]) == {SHARDS["vps1"][50], SHARDS["vps1"][51], SHARDS["vps2"][20]}
    assert len(res["pool"]) == len(DESIGN) - len(completed)
    res2 = plan_resume(DESIGN, SHARDS, completed, failed, exclude_failed=True)
    assert set(res2["exclude"]) == completed | set(res["rerun_failed"])


def check_plan(res, completed, gate_host, pool):
    placed = [g for v in res["shards"].values() for g in v["game_ids"]]
    assert len(placed) == len(set(placed))
    assert set(placed) | set(res["unassigned"]) == set(pool)
    assert not set(placed) & completed
    stream_host = {}
    for h, v in res["shards"].items():
        assert v["host"] == h and v["rounds"] == sum(BY_ID[g].rounds for g in v["game_ids"])
        pr = [PRIORITY[BY_ID[g].grid] for g in v["game_ids"]]
        assert pr == sorted(pr), "core grids must come before scenarios"
        for g in v["game_ids"]:
            s = BY_ID[g]
            if ARMS[s.arm].get("gate"):
                assert h == gate_host
                continue
            key = (s.model_label, s.scenario_id)
            assert stream_host.setdefault(key, h) == h, f"stream {key} split across hosts"


def test_rebalance_dead_vps2_onto_mac_and_vps1():
    completed, failed, where = synthetic_state()
    res = plan_resume(DESIGN, SHARDS, completed, failed, weights={"mac": 0.5, "vps1": 0.5}, gate_host="mac",
                      pool_hosts=["vps2"], completed_on=where)
    pool = [g for g in SHARDS["vps2"] if g not in completed]
    assert sorted(res["pool"]) == sorted(pool)
    check_plan(res, completed, "mac", pool)
    # vps2's started streams have no host in the weights, so they are free; both hosts receive work
    assert all(res["shards"][h]["game_ids"] for h in ("mac", "vps1"))


def test_pinning_keeps_started_streams_home():
    completed, failed, where = synthetic_state()
    res = plan_resume(DESIGN, SHARDS, completed, failed, weights={"mac": 0.45, "vps1": 0.4, "vps2": 0.15},
                      gate_host="mac", completed_on=where)
    check_plan(res, completed, "mac", res["pool"])
    for key, home in res["pinned"].items():
        label, scen = key.split("|")
        for h, v in res["shards"].items():
            for g in v["game_ids"]:
                s = BY_ID[g]
                if (s.model_label, s.scenario_id) == (label, scen) and not ARMS[s.arm].get("gate"):
                    assert h == home


def test_gated_games_never_leave_the_gate_host():
    completed, failed, where = synthetic_state()
    res = plan_resume(DESIGN, SHARDS, completed, failed, weights={"vps1": 1.0}, gate_host=None, pool_hosts=["mac"],
                      completed_on=where)
    gated = [g for g in res["pool"] if ARMS[BY_ID[g].arm].get("gate")]
    assert gated and set(res["unassigned"]) == set(gated)
    check_plan(res, completed, None, res["pool"])


def test_cli_writes_inputs_harness_run_accepts(tmp_path):
    completed, failed, _ = synthetic_state()
    (tmp_path / "c.json").write_text(json.dumps(sorted(completed)))
    (tmp_path / "f.json").write_text(json.dumps(sorted(failed)))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "resume_plan.py"), "--out-dir", str(out), "--no-logs",
                        "--completed-file", str(tmp_path / "c.json"), "--failed-file", str(tmp_path / "f.json"),
                        "--pool-hosts", "vps2", "--rebalance", "mac=0.5,gate", "vps1=0.5", "--host", "vps1"],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stderr
    assert not (out / "mac.json").exists()
    exclude = json.loads((out / "exclude-ids.json").read_text())
    shard = json.loads((out / "vps1.json").read_text())
    assert isinstance(exclude, list) and set(exclude) == completed
    # harness/run.py: shard ids must all be in the plan; exclusion is a set filter
    assert all(g in BY_ID for g in shard["game_ids"]) and shard["host"] == "vps1"
    assert [g for g in shard["game_ids"] if g not in set(exclude)] == shard["game_ids"]


def test_promote_ledger_holds_torn_tail_and_refuses_rewrites(tmp_path):
    staged, final = tmp_path / "staged.jsonl", tmp_path / "final.jsonl"
    staged.write_bytes(b'{"event": "init"}\n{"event": "reserve"}\n{"event": "se')
    ok, _, info = promote_ledger(staged, final)
    assert ok and info["held_back_bytes"] > 0 and final.read_bytes().count(b"\n") == 2
    staged.write_bytes(b'{"event": "init"}\n{"event": "reserve"}\n{"event": "settle"}\n')
    assert promote_ledger(staged, final)[0] and final.read_bytes().count(b"\n") == 3
    staged.write_bytes(b'{"event": "init"}\n{"event": "other"}\n')
    ok, why, _ = promote_ledger(staged, final)
    assert not ok and "append-only" in why and final.read_bytes().count(b"\n") == 3


def test_parse_ledger_counts(tmp_path):
    p = tmp_path / "l.jsonl"
    ev = [{"ts": "2026-10-25T16:00:00+00:00", "event": "init", "id": "x", "cap": "10.0"},
          {"ts": "2026-10-25T16:01:00+00:00", "event": "reserve", "id": "a", "est_max": 0.5, "sample_id": "g1#1"},
          {"ts": "2026-10-25T16:02:00+00:00", "event": "reserve", "id": "b", "est_max": 0.25, "sample_id": "g1#1"},
          {"ts": "2026-10-25T16:03:00+00:00", "event": "settle", "id": "a", "actual": 0.1, "sample_id": "g1#1"},
          {"ts": "2026-10-25T16:03:00+00:00", "event": "flag", "id": "a", "reason": "transport_error:ReadError"}]
    p.write_text("".join(json.dumps(e) + "\n" for e in ev) + '{"ts": "2026-10-25T16:0')
    led = parse_ledger(p)
    assert led.torn_tail and led.bad_lines == 0 and float(led.spent) == pytest.approx(0.1)
    assert led.n_open == 1 and float(led.outstanding) == 0.25 and led.calls_by_game["g1"] == 1
    assert led.flags["transport_error:ReadError"] == 1 and not led.breach


def test_validate_eval_rejects_torn_copy_and_scan_reads_good_one(tmp_path):
    good = tmp_path / PILOT_LOG.name
    shutil.copy2(PILOT_LOG, good)
    ok, info, _ = validate_eval(good)
    assert ok and info["n_samples"] > 0
    torn = tmp_path / "torn.eval"
    torn.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    assert not validate_eval(torn)[0]
    scan = scan_logs(tmp_path, {}, live=True)
    assert "torn.eval" in scan.unreadable and len(scan.games) == info["n_samples"]
