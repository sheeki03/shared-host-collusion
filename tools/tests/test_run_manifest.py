"""tools/run_manifest.py: sanitized game -> host/tag/launch/log manifest, coverage, shards and ledgers (no network,
no Docker, no model calls)."""
import csv
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis import game_set  # noqa: E402
from tools import run_manifest as rm  # noqa: E402

CANARY = "CX-0123456789ab"
HOSTNAME = "vmi-secret-prod-7"
PILOT_DIRS = [ROOT / "runs" / "pilot" / d for d in ("logs-probes", "logs-probes2", "logs-smoke")]


def _rows(rounds: int) -> list[dict]:
    return [{"t": t, "firm": f, "action": 1.5, "scripted": False, "parse_failed": False, "canary": CANARY,
             "rationale": "we should both hold the price"} for t in range(1, rounds + 1) for f in ("A", "B")]


def _sample(gid: str, arm: str, seed: int, *, done: str | None, error: str | None = None, host=HOSTNAME,
            model="m1", scen="core-pricing", grid="deep", rounds=2):
    meta = {"game_id": gid, "grid": grid, "model_label": model, "model_id": f"x/{model}", "effort": "low",
            "scenario_id": scen, "engine": "pricing", "arm": arm, "seed": seed, "rounds": rounds,
            "result": {"host": host, "rows": _rows(rounds) if error is None else [],
                       "chat": [{"text": f"secret plan {CANARY}"}], "calls": []}}
    return SimpleNamespace(id=gid, epoch=1, metadata=meta, completed_at=done, started_at="2026-10-25T17:00:00+00:00",
                           error=error)


def _log(eval_id: str, created: str, status: str, samples: list):
    return SimpleNamespace(status=status, eval=SimpleNamespace(eval_id=eval_id, run_id="run-" + eval_id,
                                                               created=created), samples=samples)


def _fake_runs(tmp_path):
    """Three logs: the local machine (stopped launch), and a remote host with two launches. g1 completes twice (the
    earlier finish is kept), g2 errors locally and completes remotely (superseded), g3 only errors."""
    runs = tmp_path / "runs"
    local, v1, v2 = runs / "full" / "logs-macv2", runs / "remote" / "vps1" / "logs-vps1", \
        runs / "remote" / "vps1" / "logs-vps1v2"
    for d in (local, v1, v2):
        d.mkdir(parents=True)
    err = f"ModelGenerateError('Request: {{\"messages\": [\"Your task {CANARY}\"]}}')"
    logs = {
        str(local / "a.eval"): _log("E1", "2026-10-25T16:54:00+00:00", "started", [
            _sample("deep:m1:core-pricing:SHR-AWARE:1000", "SHR-AWARE", 1000, done="2026-10-25T17:30:00+00:00"),
            _sample("deep:m1:core-pricing:ISO-AWARE:1000", "ISO-AWARE", 1000, done=None, error=err),
            _sample("pilot:m1:x", "SHR-AWARE", 9000, done="2026-10-25T17:31:00+00:00", grid="pilot"),
        ]),
        str(v1 / "b.eval"): _log("E2", "2026-10-25T16:55:00+00:00", "success", [
            _sample("deep:m1:core-pricing:SHR-AWARE:1000", "SHR-AWARE", 1000, done="2026-10-25T17:40:00+00:00",
                    host="other-host"),
            _sample("deep:m1:core-pricing:ISO-AWARE:1000", "ISO-AWARE", 1000, done="2026-10-25T17:41:00+00:00",
                    host="other-host"),
        ]),
        str(v2 / "c.eval"): _log("E3", "2026-10-25T18:00:00+00:00", "error", [
            _sample("deep:m1:core-pricing:SHR-AWARE:1001", "SHR-AWARE", 1001, done=None, error="RuntimeError('x')",
                    host="other-host"),
        ]),
    }

    def lister(d):
        return [SimpleNamespace(name=p) for p in sorted(logs) if str(Path(p).parent) == str(Path(d))]

    def reader(info):
        return logs[info.name]

    return [local, v1, v2], lister, reader


def test_host_and_tag_from_the_log_path():
    assert rm.host_and_tag("/x/runs/full/logs-macv3-gated/a.eval", "mac") == ("mac", "macv3-gated")
    assert rm.host_and_tag("file:///x/runs/remote/instabox2/logs-instabox2-v5/a.eval", "mac") == \
        ("instabox2", "instabox2-v5")
    assert rm.host_and_tag("/x/runs/pilot/logs-smoke/a.eval", "laptop") == ("laptop", "smoke")


def test_error_and_problem_categories_never_carry_text():
    msg = f"ModelGenerateError('Request: {{\"content\": \"Your task {CANARY}\"}}')"
    assert rm.error_class(msg) == "ModelGenerateError"
    assert rm.error_class(SimpleNamespace(message="BudgetExceeded('budget cap $60.0 ...')")) == "BudgetExceeded"
    assert rm.error_class(f"{CANARY}(x)") == "unclassified" and rm.error_class("free text here") == "unclassified"
    assert rm.problem_kind(f"sample error: {msg}", msg) == "sample_error:ModelGenerateError"
    assert rm.problem_kind("incomplete game: 3 rows for 2 rounds", None) == "incomplete_game"
    assert rm.problem_kind(game_set.PARSE_PREFIX + " preregistered 20% exclusion", None) == "parse_rate_above_threshold"
    assert rm.problem_kind("something new", None) == "incomplete" and rm.problem_kind(None, None) == ""
    assert rm.flag_kind("transport_error:ReadError") == "transport_error:ReadError"
    assert rm.flag_kind("missing_cost:http_429") == "missing_cost:http_429"
    assert rm.flag_kind("invalid_cost:{'secret': 1}") == "invalid_cost"
    assert rm.flag_kind("Weird reason: with text") == "other" and rm.flag_kind(None) == "unspecified"


def test_manifest_maps_games_to_hosts_and_applies_d12(tmp_path):
    dirs, lister, reader = _fake_runs(tmp_path)
    m = rm.build(dirs, design="none", lister=lister, reader=reader)
    g = m["games"]
    assert set(g) == {"deep:m1:core-pricing:SHR-AWARE:1000", "deep:m1:core-pricing:ISO-AWARE:1000",
                      "deep:m1:core-pricing:SHR-AWARE:1001"}           # the pilot sample is outside the cohort
    shr = g["deep:m1:core-pricing:SHR-AWARE:1000"]
    assert shr["state"] == "kept" and shr["n_samples"] == 2 and shr["n_complete"] == 2
    assert shr["kept"]["host"] == "mac" and shr["kept"]["tag"] == "macv2"   # finished first
    iso = g["deep:m1:core-pricing:ISO-AWARE:1000"]
    assert iso["kept"]["host"] == "vps1" and iso["kept"]["tag"] == "vps1"
    assert g["deep:m1:core-pricing:SHR-AWARE:1001"]["state"] == "incomplete_only"
    sel = {(r["game_id"], r["tag"]): r["selection"] for r in m["samples"]}
    assert sel[("deep:m1:core-pricing:SHR-AWARE:1000", "vps1")] == game_set.DUPLICATE
    assert sel[("deep:m1:core-pricing:ISO-AWARE:1000", "macv2")] == game_set.SUPERSEDED
    assert sel[("deep:m1:core-pricing:SHR-AWARE:1001", "vps1v2")] == game_set.INCOMPLETE
    probs = {r["problem"] for r in m["samples"]}
    assert probs == {"", "sample_error:ModelGenerateError", "sample_error:RuntimeError"}
    # the recorded hostnames become aliases, consistent across logs
    machines = {(r["tag"], r["machine"]) for r in m["samples"]}
    assert machines == {("macv2", "m1"), ("vps1", "m2"), ("vps1v2", "m2")} and m["totals"]["machines"] == 2
    # coverage: the 1000 matched set is complete but split across hosts; 1001 has no kept copy
    sets = {r["seed"]: r for r in m["matched_sets"]}
    assert sets[1000]["complete"] and sets[1000]["hosts"] == ["mac", "vps1"] and not sets[1000]["single_host"]
    assert sets[1001]["missing_arms"] == ["SHR-AWARE"] and not sets[1001]["complete"]
    seeds = {r["seed"]: r for r in m["seeds"]}
    assert seeds[1000]["kept"] == 2 and seeds[1001]["incomplete_only"] == 1
    assert m["totals"]["duplicates_dropped"] == 1 and m["totals"]["superseded_samples"] == 1
    assert m["totals"]["complete_matched_sets"] == 1 and m["totals"]["complete_matched_sets_single_host"] == 0
    launches = {x["tag"]: x for x in m["launches"]}
    assert launches["macv2"]["status"] == "started" and launches["macv2"]["n_in_cohort"] == 2
    assert launches["vps1v2"]["eval_id"] == "E3" and launches["vps1v2"]["run_id"] == "run-E3"


def test_outputs_hold_no_model_text_canaries_or_hostnames(tmp_path):
    dirs, lister, reader = _fake_runs(tmp_path)
    m = rm.build(dirs, design="none", lister=lister, reader=reader)
    files = rm.render(m)
    for name, text in files.items():
        for secret in (CANARY, HOSTNAME, "other-host", "secret plan", "Your task", "hold the price", "Request:",
                       str(tmp_path)):
            assert secret not in text, (name, secret)
    rows = list(csv.DictReader(io.StringIO(files["samples.csv"])))
    assert len(rows) == 5 and set(rows[0]) == set(rm.SAMPLE_COLUMNS)
    assert json.loads(files["run_manifest.json"])["schema"] == rm.SCHEMA
    # fail closed if any text ever carries a canary or an absolute home path
    m["notes"].append(f"report {CANARY}")
    with pytest.raises(rm.ManifestError, match="canary"):
        rm.render(m)
    m["notes"][-1] = f"log at {Path.home()}/x.eval"
    with pytest.raises(rm.ManifestError, match="home"):
        rm.render(m)


def test_design_coverage_counts_missing_games(tmp_path):
    from harness.arms import full_design

    dirs, lister, reader = _fake_runs(tmp_path)
    m = rm.build(dirs, design="full", lister=lister, reader=reader)
    t = m["totals"]
    n = len(full_design())
    assert t["planned"] == n and t["kept"] == 0 and t["incomplete_only"] == 0 and t["missing"] == n
    # the fake games (model m1) are not in the design: reported, never counted as coverage
    assert set(t["unplanned_observed"]) == {"deep:m1:core-pricing:SHR-AWARE:1000",
                                            "deep:m1:core-pricing:ISO-AWARE:1000",
                                            "deep:m1:core-pricing:SHR-AWARE:1001"}
    assert sum(r["planned"] for r in m["seeds"]) == n and sum(r["n_games"] for r in m["matched_sets"]) == n
    assert all(r["missing_arms"] == r["arms"] and not r["complete"] for r in m["matched_sets"])


def test_strict_fails_on_unreadable_and_staging_logs_lenient_notes_them(tmp_path):
    dirs, lister, reader = _fake_runs(tmp_path)
    stage = tmp_path / "runs" / "remote" / "vps1" / ".staging" / "full" / "logs-vps1"
    stage.mkdir(parents=True)

    def lister2(d):
        out = lister(d)
        return out + ([SimpleNamespace(name=str(stage / "z.eval"))] if str(d) == str(stage) else [])

    with pytest.raises(rm.ManifestError, match="staging"):
        rm.build([*dirs, stage], design="none", lister=lister2, reader=reader)
    m = rm.build([*dirs, stage], design="none", lister=lister2, reader=reader, strict=False)
    assert any("staging" in n for n in m["notes"])

    def broken(info):
        if info.name.endswith("c.eval"):
            raise OSError("mid-flush")
        return reader(info)

    with pytest.raises(rm.ManifestError, match="unreadable"):
        rm.build(dirs, design="none", lister=lister, reader=broken)
    m = rm.build(dirs, design="none", lister=lister, reader=broken, strict=False)
    assert "deep:m1:core-pricing:SHR-AWARE:1001" not in m["games"] and any("unreadable" in n for n in m["notes"])
    with pytest.raises(rm.ManifestError, match="missing"):
        rm.build([tmp_path / "nope"], design="none", lister=lister, reader=reader)


def test_write_refuses_log_dirs_and_silent_overwrite(tmp_path):
    dirs, lister, reader = _fake_runs(tmp_path)
    m = rm.build(dirs, design="none", lister=lister, reader=reader)
    with pytest.raises(rm.ManifestError, match="inside the input log directory"):
        rm.write(m, dirs[1] / "manifest", log_dirs=dirs)
    out = tmp_path / "manifest"
    paths = rm.write(m, out, log_dirs=dirs)
    assert sorted(p.name for p in paths) == ["matched_sets.csv", "run_manifest.json", "samples.csv", "seeds.csv"]
    with pytest.raises(rm.ManifestError, match="--overwrite"):
        rm.write(m, out, log_dirs=dirs)
    assert rm.write(m, out, log_dirs=dirs, overwrite=True) and not list(out.glob(".*.tmp"))


def test_shard_assignment_is_recorded(tmp_path):
    dirs, lister, reader = _fake_runs(tmp_path)
    sh = tmp_path / "shards"
    sh.mkdir()
    (sh / "mac.json").write_text(json.dumps({"host": "mac", "game_ids": ["deep:m1:core-pricing:SHR-AWARE:1000"],
                                             "rounds": 2}))
    (sh / "vps1.json").write_text(json.dumps({"host": "vps1", "game_ids": ["deep:m1:core-pricing:SHR-AWARE:1000",
                                                                           "deep:m1:core-pricing:SHR-AWARE:1001"]}))
    m = rm.build(dirs, design="none", lister=lister, reader=reader, shards=[sh / "mac.json", sh / "vps1.json"])
    g = m["games"]
    assert g["deep:m1:core-pricing:SHR-AWARE:1000"]["assigned_in"] == ["shards/mac.json", "shards/vps1.json"]
    assert g["deep:m1:core-pricing:ISO-AWARE:1000"]["assigned_in"] == []
    assert m["totals"]["kept_outside_their_shards"] == ["deep:m1:core-pricing:ISO-AWARE:1000"]
    assert [s["n_games"] for s in m["shards"]] == [1, 2] and all(len(s["sha256"]) == 64 for s in m["shards"])
    (sh / "bad.json").write_text(json.dumps({"host": "x"}))
    with pytest.raises(rm.ManifestError, match="game_ids"):
        rm.build(dirs, design="none", lister=lister, reader=reader, shards=[sh / "bad.json"])


def _ledger(lid: str, events: list[dict]) -> str:
    lines = [{"ts": "2026-10-25T16:00:00+00:00", "event": "init", "id": lid, "cap": "10.0"}] + events
    return "".join(json.dumps(e) + "\n" for e in lines)


def test_ledgers_report_open_reservations_and_copies(tmp_path):
    ev = [{"ts": "2026-10-25T16:01:00+00:00", "event": "reserve", "id": "a", "est_max": 0.5, "sample_id": "g#1"},
          {"ts": "2026-10-25T16:02:00+00:00", "event": "settle", "id": "a", "actual": 0.1},
          {"ts": "2026-10-25T16:02:00+00:00", "event": "flag", "id": "a", "reason": "missing_cost:http_503"},
          {"ts": "2026-10-25T16:03:00+00:00", "event": "reserve", "id": "b", "est_max": 0.25},
          {"ts": "2026-10-25T16:04:00+00:00", "event": "reserve", "id": "c", "est_max": 0.75},
          {"ts": "2026-10-25T16:05:00+00:00", "event": "release", "id": "c"},
          {"ts": "2026-10-25T16:06:00+00:00", "event": "flag", "id": "d", "reason": "invalid_cost:secret text"},
          {"ts": "2026-10-25T16:07:00+00:00", "event": "cap_change", "id": "L1", "old_cap": "10.0", "cap": "50.0",
           "reason": "owner approved, see chat"}]
    live = tmp_path / "remote" / "h" / "ledger-h.jsonl"
    old = tmp_path / "remote" / "ledgers" / "ledger-h.jsonl"
    other = tmp_path / "remote" / "h" / "ledger-x.jsonl"
    for p in (live, old, other):
        p.parent.mkdir(parents=True, exist_ok=True)
    full = _ledger("L1", ev)
    live.write_text(full)
    old.write_text(_ledger("L1", ev[:4]))           # an older synced copy: a byte prefix of the live file
    other.write_text(_ledger("L2", []))
    rows = {r["path"]: r for r in rm.read_ledgers([live, old, other, live])}
    assert len(rows) == 3
    lv, cp = rows["h/ledger-h.jsonl"], rows["ledgers/ledger-h.jsonl"]
    assert lv["n_open"] == 1 and lv["open_est_max_usd"] == "0.25" and lv["settled_usd"] == "0.1"
    assert lv["n_reserve"] == 3 and lv["n_settle"] == 1 and lv["n_release"] == 1 and lv["copy_of"] is None
    assert lv["flags"] == {"invalid_cost": 1, "missing_cost:http_503": 1}
    assert lv["cap_usd"] == "50.0" and lv["n_cap_changes"] == 1
    assert cp["copy_of"] == "h/ledger-h.jsonl" and cp["n_open"] == 1 and cp["conflicts_with"] is None
    assert rows["h/ledger-x.jsonl"]["copy_of"] is None
    text = json.dumps(list(rows.values()))
    assert "secret text" not in text and "owner approved" not in text
    # same ID, but neither file extends the other: flagged (the analysis refuses such a pair)
    old.write_text(_ledger("L1", [ev[3]]))
    rows = {r["path"]: r for r in rm.read_ledgers([live, old])}
    assert rows["ledgers/ledger-h.jsonl"]["conflicts_with"] == "h/ledger-h.jsonl"
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"event": "init", "id": "z", "cap": "1"}\nnot json\n{"event": "reserve", "id": "q"}\n')
    with pytest.raises(rm.ManifestError, match="cannot be read"):
        rm.read_ledgers([bad])


def test_cli_end_to_end_on_fake_logs(tmp_path, monkeypatch, capsys):
    dirs, lister, reader = _fake_runs(tmp_path)
    real_build = rm.build
    monkeypatch.setattr(rm, "build", lambda *a, **k: real_build(*a, **{**k, "lister": lister, "reader": reader}))
    out = tmp_path / "out"
    assert rm.main([*map(str, dirs), "--out-dir", str(out), "--design", "none"]) == 0
    printed = capsys.readouterr().out
    assert "matched sets 2, complete 1, complete on one host 0" in printed and (out / "run_manifest.json").exists()
    assert rm.main([*map(str, dirs), "--out-dir", str(out), "--design", "none"]) == 2   # no silent overwrite


PILOT_PRESENT = all(d.is_dir() and any(d.glob("*.eval")) for d in PILOT_DIRS)


@pytest.mark.skipif(not PILOT_PRESENT, reason="needs the ignored pilot logs under runs/pilot/ (author's machine only)")
def test_pilot_logs_manifest_matches_the_game_set_and_leaks_nothing(tmp_path):
    from inspect_ai.log import list_eval_logs, read_eval_log

    m = rm.build(PILOT_DIRS, cohort="pilot", design="none")
    kept = {g for g, v in m["games"].items() if v["state"] == "kept"}
    assert kept == set(game_set.scan(PILOT_DIRS, cohort="pilot").kept)
    assert m["totals"]["samples"] == sum(x["n_in_cohort"] for x in m["launches"])
    assert {x["host"] for x in m["launches"]} == {"mac"}
    assert {x["tag"] for x in m["launches"]} == {"probes", "probes2", "smoke"}
    secrets = set()
    for d in PILOT_DIRS:
        for info in list_eval_logs(str(d)):
            for s in read_eval_log(info, exclude_fields={"events", "messages"}).samples or []:
                res = (s.metadata or {}).get("result") or {}
                secrets |= {r.get("canary") for r in res.get("rows", []) if r.get("canary")}
                secrets |= {str(r.get("rationale"))[:40] for r in res.get("rows", []) if r.get("rationale")}
    assert secrets   # the pilot rows do carry canaries and rationales
    files = rm.render(m)
    for name, text in files.items():
        assert not any(s in text for s in secrets), name
    paths = rm.write(m, tmp_path / "manifest", log_dirs=PILOT_DIRS)
    assert all(p.parent == (tmp_path / "manifest").resolve() for p in paths)
