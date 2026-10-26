"""Synthetic tests for tools/resume_plan.py, tools/sync_logs.py and tools/progress.py (no network, no Docker).

`runs/` is ignored by git. Without it (a fresh public clone) the shard-based tests use the first split rebuilt by
tools/make_shards.py, and the tests that copy real pilot logs or parse the operational dispatcher are skipped.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.arms import ARMS, full_design  # noqa: E402
from tools import make_shards, progress, sync_logs  # noqa: E402
from tools.progress import parse_ledger, scan_logs  # noqa: E402
from tools.resume_plan import PRIORITY, plan_resume  # noqa: E402
from tools.sync_logs import promote_ledger, validate_eval  # noqa: E402

DESIGN = full_design()
BY_ID = {s.game_id: s for s in DESIGN}
SHARD_HOSTS = ("mac", "vps1", "vps2")
RUNS = ROOT / "runs"                     # ignored: present on the author's machine only
FIRST_SPLIT = ["mac=0.45,gate", "vps1=0.40", "vps2=0.15"]   # tools/make_shards.py's documented example


def generated_shards() -> dict[str, list[str]]:
    """The make_shards.py split rules (whole streams per host, gated arms on mac), built in a temporary directory."""
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        make_shards.main(d, FIRST_SPLIT)
        return {h: json.loads((Path(d) / f"{h}.json").read_text())["game_ids"] for h in SHARD_HOSTS}


def load_shards() -> dict[str, list[str]]:
    """The recorded first split (runs/shards/, the launch D6 stopped) when present, else the generated one. These
    tests only need a valid partition of the design, so both serve."""
    recorded = [RUNS / "shards" / f"{h}.json" for h in SHARD_HOSTS]
    if all(p.is_file() for p in recorded):
        return {h: json.loads(p.read_text())["game_ids"] for h, p in zip(SHARD_HOSTS, recorded)}
    return generated_shards()


def _first_eval(sub: str) -> Path | None:
    found = sorted((RUNS / "pilot" / sub).glob("*.eval"))
    return found[0] if found else None


SHARDS = load_shards()
PILOT_LOG = _first_eval("logs-smoke")        # 28 samples, all complete
PILOT_LOG_ERR = _first_eval("logs-probes")   # 14 samples, some failed
needs_pilot_logs = pytest.mark.skipif(
    PILOT_LOG is None or PILOT_LOG_ERR is None,
    reason="needs the ignored pilot logs runs/pilot/logs-smoke and runs/pilot/logs-probes (author's machine only)")
needs_dispatcher = pytest.mark.skipif(not (RUNS / "dispatcher.py").is_file(),
                                      reason="needs the ignored operational runs/dispatcher.py (author's machine only)")


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


def test_generated_first_split_follows_the_shard_rules():
    """The fallback used in a fresh clone: a partition with whole (model, scenario) streams and gated arms on mac."""
    shards = generated_shards()
    ids = [g for v in shards.values() for g in v]
    assert len(ids) == len(set(ids)) == len(DESIGN) and all(len(shards[h]) > 52 for h in SHARD_HOSTS)
    home = {}
    for h, v in shards.items():
        pr = [PRIORITY[BY_ID[g].grid] for g in v]
        assert pr == sorted(pr)
        for g in v:
            s = BY_ID[g]
            if ARMS[s.arm].get("gate"):
                assert h == "mac"
            else:
                assert home.setdefault((s.model_label, s.scenario_id), h) == h


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
    shards_dir = tmp_path / "shards"         # the same shards as SHARDS (recorded or generated)
    shards_dir.mkdir()
    for h, ids in SHARDS.items():
        (shards_dir / f"{h}.json").write_text(json.dumps({"host": h, "game_ids": ids}))
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "resume_plan.py"), "--out-dir", str(out), "--no-logs",
                        "--completed-file", str(tmp_path / "c.json"), "--failed-file", str(tmp_path / "f.json"),
                        "--shards-dir", str(shards_dir),
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
    staged.write_bytes(b'{"event": "init"}\n')  # an older / cut-short copy is not a rewrite, but is not promoted
    ok, why, _ = promote_ledger(staged, final)
    assert not ok and "shorter" in why and final.read_bytes().count(b"\n") == 3


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


@needs_pilot_logs
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



# ------------------------------------------------------------------------------------- sync_logs: many tags, hosts
def test_load_hosts_reads_dispatch_hosts(tmp_path):
    f = tmp_path / "dispatch_hosts.json"
    f.write_text(json.dumps({"mac": None, "vps1": "root@vps1.example", "vps2": "root@vps2.example",
                             "instabox": "instabox", "../evil": "x"}))
    assert sync_logs.load_hosts(f) == {"vps1": "root@vps1.example", "vps2": "root@vps2.example",
                                       "instabox": "instabox"}
    assert sync_logs.load_hosts(tmp_path / "missing.json") == sync_logs.DEFAULT_HOSTS


def test_load_hosts_uses_private_overrides_without_a_dispatch_file(tmp_path):
    f = tmp_path / "dispatch_hosts.json"
    f.write_text(json.dumps({"vps1": "old-alias", "instabox": "instabox"}))
    private = tmp_path / "ssh_hosts.json"
    private.write_text(json.dumps({"vps1": "operator@vps1.example", "vps2": "private-vps2",
                                   "../evil": "x", "bad": "-oProxyCommand=bad"}))
    assert sync_logs.load_hosts(f) == {"vps1": "operator@vps1.example", "vps2": "private-vps2",
                                      "instabox": "instabox"}
    f.unlink()
    assert sync_logs.load_hosts(f) == {"vps1": "operator@vps1.example", "vps2": "private-vps2"}


def test_classify_paths():
    c = sync_logs.classify
    assert c("full/logs-vps1v2/2026-10-25T16-54-07-00-00_shc-games_53Rf.eval") == "eval"
    assert c("full/logs-instabox-spill-1/a.eval") == "eval"
    assert c("full/ledger-vps1v2.jsonl") == "ledger" and c("full/instabox-spill-1.out") == "out"
    assert c("shards/vps1v2.json") == "shard"
    for bad in ("full/specs_full-all-vps1.json", "full/logs-x/.a.eval.tmp", "full/logs-x/.a.eval", "full/../x.out",
                "full/logs-/a.eval", "full/logs-x/sub/a.eval", "full/ledger-x.jsonl.conflict", "other/x.out"):
        assert c(bad) is None, bad
    assert sync_logs.local_rel("full/logs-a/x.eval") == "logs-a/x.eval"
    assert sync_logs.local_rel("shards/a.json") == "shards/a.json"
    assert [sync_logs.tag_of(x) for x in ("logs-vps1v2/x.eval", "ledger-vps1-spill-2.jsonl", "vps2.out",
                                          "shards/instabox-spill-1.json")] == \
        ["vps1v2", "vps1-spill-2", "vps2", "instabox-spill-1"]


def _tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime)
            for p in sorted(root.rglob("*")) if p.is_file()}


def _fake_remote(tmp_path) -> Path:
    """A remote home with v1, v2 and overflow tags (one torn log), plus files that must not be pulled."""
    if PILOT_LOG is None or PILOT_LOG_ERR is None:
        pytest.skip("needs the ignored pilot logs (author's machine only)")
    runs = tmp_path / "remote-home" / "shared-host-collusion" / "runs"
    full = runs / "full"
    for d in ("logs-vps1", "logs-vps1v2", "logs-vps1-spill-2"):
        (full / d).mkdir(parents=True)
    (runs / "shards").mkdir()
    shutil.copy2(PILOT_LOG, full / "logs-vps1" / PILOT_LOG.name)
    shutil.copy2(PILOT_LOG_ERR, full / "logs-vps1v2" / PILOT_LOG_ERR.name)
    data = PILOT_LOG.read_bytes()
    (full / "logs-vps1-spill-2" / "torn.eval").write_bytes(data[: len(data) // 2])
    (full / "logs-vps1v2" / ".hidden.eval").write_bytes(b"temp")
    (full / "ledger-vps1.jsonl").write_text('{"event": "init"}\n')
    (full / "ledger-vps1v2.jsonl").write_text('{"event": "init"}\n{"event": "reserve"}\n{"event": "se')
    (full / "vps1.out").write_text("v1 out\n")
    (full / "vps1v2.out").write_text("v2 out\n")
    (full / "specs_full-all-vps1.json").write_text("{}")
    (runs / "shards" / "vps1v2.json").write_text(json.dumps({"host": "vps1", "game_ids": ["a", "b"]}))
    (runs / "shards" / "vps1-spill-2.json").write_text(json.dumps({"host": "spill-2", "game_ids": ["c"]}))
    return runs


def _install_fake_transport(monkeypatch, runs: Path, rsync_ok: bool = True):
    calls = {"rsync": 0, "scp_files": []}

    def listing(dest, timeout):
        out = {}
        for p in runs.rglob("*"):
            rel = p.relative_to(runs).as_posix()
            if p.is_file() and sync_logs.classify(rel):
                st = p.stat()
                out[rel] = (st.st_size, st.st_mtime)
        return out, ""

    def fake_rsync(dest, stage, timeout, only=None):
        calls["rsync"] += 1
        if not rsync_ok:
            return False, "bash: line 1: rsync: command not found"
        for rel in (only or listing(dest, timeout)[0]):
            src, dst = runs / rel, stage / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size == src.stat().st_size and \
                    int(dst.stat().st_mtime) == int(src.stat().st_mtime) and not only:
                continue  # quick check: unchanged
            tmp = dst.with_name("." + dst.name + ".XXXX")  # rsync writes a temp file, then renames it
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        return True, ""

    real_run = sync_logs._run

    def fake_run(cmd, timeout):
        if cmd[0] != "scp":
            return real_run(cmd, timeout)
        prefix = "instabox:" + sync_logs.REMOTE_RUNS + "/"
        for a in cmd:
            if a.startswith(prefix):
                rel = a[len(prefix):]
                calls["scp_files"].append(rel)
                shutil.copy2(runs / rel, Path(cmd[-1]) / Path(rel).name)
        return 0, ""

    monkeypatch.setattr(sync_logs, "remote_listing", listing)
    monkeypatch.setattr(sync_logs, "rsync_pull", fake_rsync)
    monkeypatch.setattr(sync_logs, "_run", fake_run)
    return calls


@needs_pilot_logs
def test_sync_host_pulls_every_tag_read_only(tmp_path, monkeypatch):
    runs = _fake_remote(tmp_path)
    before = _tree_digest(runs)
    _install_fake_transport(monkeypatch, runs)
    local = tmp_path / "local"
    st = sync_logs.sync_host("vps1", "root@x", retries=1, retry_wait=0, deep=True, timeout=60, base_root=local)
    base = local / "vps1"
    assert (base / "logs-vps1" / PILOT_LOG.name).exists() and (base / "logs-vps1v2" / PILOT_LOG_ERR.name).exists()
    assert not (base / "logs-vps1-spill-2" / "torn.eval").exists()
    assert st["pending_invalid"] == ["logs-vps1-spill-2/torn.eval"]
    assert (base / "ledger-vps1v2.jsonl").read_bytes().count(b"\n") == 2  # torn tail held back
    assert (base / "vps1.out").read_text() == "v1 out\n" and (base / "vps1v2.out").exists()
    assert json.loads((base / "shards" / "vps1-spill-2.json").read_text())["game_ids"] == ["c"]
    assert not (base / "specs_full-all-vps1.json").exists() and not (base / "logs-vps1v2" / ".hidden.eval").exists()
    assert st["tags"]["vps1v2"]["evals_remote"] == 1 and st["tags"]["vps1v2"]["evals_local"] == 1
    assert st["tags"]["vps1-spill-2"]["evals_remote"] == 1 and "logs-vps1-spill-2/torn.eval" in st["missing_locally"]
    # one copy on disk: the promoted log is a hard link to its staged copy
    staged = base / ".staging" / "full" / "logs-vps1" / PILOT_LOG.name
    assert os.path.samefile(staged, base / "logs-vps1" / PILOT_LOG.name)
    assert _tree_digest(runs) == before  # nothing on the remote side changed

    # the runner rewrites a log mid-copy: the torn copy is staged, the last good copy stays as it was
    good = (base / "logs-vps1v2" / PILOT_LOG_ERR.name).read_bytes()
    remote_log = runs / "full" / "logs-vps1v2" / PILOT_LOG_ERR.name
    remote_log.write_bytes(good[: len(good) // 3])
    os.utime(remote_log, (remote_log.stat().st_atime, remote_log.stat().st_mtime + 5))
    # the ledger grows (append-only) and the spill log becomes whole
    with open(runs / "full" / "ledger-vps1v2.jsonl", "ab") as f:
        f.write(b'ttle"}\n')
    shutil.copy2(PILOT_LOG, runs / "full" / "logs-vps1-spill-2" / "torn.eval")
    st = sync_logs.sync_host("vps1", "root@x", retries=1, retry_wait=0, deep=True, timeout=60, base_root=local)
    assert (base / "logs-vps1v2" / PILOT_LOG_ERR.name).read_bytes() == good
    assert st["pending_invalid"] == [f"logs-vps1v2/{PILOT_LOG_ERR.name}"]
    assert (base / "logs-vps1-spill-2" / "torn.eval").exists()
    assert (base / "ledger-vps1v2.jsonl").read_bytes().count(b"\n") == 3

    # a rewritten (non-append) ledger is refused and parked, the last good copy kept
    (runs / "full" / "ledger-vps1v2.jsonl").write_text('{"event": "other"}\n')
    sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60, base_root=local)
    assert (base / "ledger-vps1v2.jsonl").read_bytes().count(b"\n") == 3
    assert (base / "ledger-vps1v2.jsonl.conflict").exists()


@needs_pilot_logs
def test_sync_host_scp_fallback_fetches_only_changed_files(tmp_path, monkeypatch):
    runs = _fake_remote(tmp_path)
    calls = _install_fake_transport(monkeypatch, runs, rsync_ok=False)
    local = tmp_path / "local"
    st = sync_logs.sync_host("instabox", "instabox", retries=0, retry_wait=0, deep=True, timeout=60,
                             base_root=local)
    base = local / "instabox"
    assert st["transport"] == "scp" and st.get("rsync_missing_until")
    assert (base / "logs-vps1" / PILOT_LOG.name).exists() and (base / "shards" / "vps1v2.json").exists()
    assert not list((base / ".staging").glob(".scp-tmp"))
    n_first, rsync_first = len(calls["scp_files"]), calls["rsync"]
    with open(runs / "full" / "vps1v2.out", "a") as f:
        f.write("more\n")
    sync_logs.sync_host("instabox", "instabox", retries=0, retry_wait=0, deep=True, timeout=60, base_root=local)
    assert calls["rsync"] == rsync_first  # rsync known missing: not retried for a while
    assert calls["scp_files"][n_first:] == ["full/vps1v2.out"]
    assert (base / "vps1v2.out").read_text() == "v2 out\nmore\n"


def test_old_single_tag_state_and_staging_are_migrated(tmp_path):
    stage = tmp_path / ".staging"
    (stage / "logs-vps1").mkdir(parents=True)
    (stage / "logs-vps1" / "a.eval").write_bytes(b"x")
    (stage / "ledger-vps1.jsonl").write_bytes(b"{}\n")
    sync_logs._migrate_staging(stage)
    assert (stage / "full" / "logs-vps1" / "a.eval").exists() and (stage / "full" / "ledger-vps1.jsonl").exists()
    st = sync_logs._migrate_state({"files": {"a.eval": {"size": 1}}, "ledger": {"lines": 1}, "out": {"bytes": 2}},
                                  "vps1")
    assert st["files"] == {"logs-vps1/a.eval": {"size": 1}}
    assert st["ledgers"] == {"ledger-vps1.jsonl": {"lines": 1}} and st["outs"] == {"vps1.out": {"bytes": 2}}


# --------------------------------------------------------------------------------- progress: tags, shards, ETA
def test_tag_kinds_and_shard_resolution(tmp_path):
    assert progress.tag_kind("vps1", "vps1") == ("v1", None)
    assert progress.tag_kind("vps1v2", "vps1") == ("v2", None)
    assert progress.tag_kind("instabox-spill-1", "instabox") == ("overflow", "spill-1")
    assert progress.tag_kind("vps1-spill-12", "vps1") == ("overflow", "spill-12")
    for d in ("runs/shards", "runs/shards-v2", "runs/remote/vps1/shards"):
        (tmp_path / d).mkdir(parents=True)

    def put(rel, ids):
        (tmp_path / rel).write_text(json.dumps({"host": "x", "game_ids": ids}))

    put("runs/shards-v2/vps1.json", ["derived"])
    put("runs/shards-v2/spill-3.json", ["chunk"])
    assert progress.shard_for(tmp_path, "vps1", "vps1v2", "v2", None) == (["derived"], "runs/shards-v2/vps1.json")
    put("runs/shards/vps1v2.json", ["mac-copy"])
    assert progress.shard_for(tmp_path, "vps1", "vps1v2", "v2", None)[0] == ["mac-copy"]
    put("runs/remote/vps1/shards/vps1v2.json", ["remote-copy"])
    assert progress.shard_for(tmp_path, "vps1", "vps1v2", "v2", None)[0] == ["remote-copy"]
    assert progress.shard_for(tmp_path, "vps1", "vps1-spill-3", "overflow", "spill-3")[0] == ["chunk"]
    assert progress.shard_for(tmp_path, "mac", "macv2", "v2", None) == ([], "no shard file found")


def test_assign_overflow_greedy_like_the_dispatcher():
    t0 = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    hosts = {"a": {"free_at": t0 + timedelta(hours=1), "kind": "units", "rate": 100.0},
             "b": {"free_at": t0 + timedelta(hours=2), "kind": "rounds", "rate": 50.0},
             "c": {"free_at": None, "kind": None, "rate": None}}
    work = {"units": 100, "rounds": 100}
    lag = timedelta(minutes=4)
    got = progress.assign_overflow(hosts, [("spill-2", work), ("spill-3", work), ("spill-4", work)], lag)
    assert [(c, h) for c, h, _, _ in got] == [("spill-2", "a"), ("spill-3", "b"), ("spill-4", "a")]
    assert got[0][3] == t0 + timedelta(hours=2) + lag
    assert got[1][3] == t0 + timedelta(hours=4) + lag
    assert hosts["a"]["free_at"] == t0 + timedelta(hours=3) + 2 * lag
    assert progress.assign_overflow({"c": hosts["c"]}, [("spill-9", work)]) == [("spill-9", None, None, None)]


def _ledger_lines(ids, t0):
    ev = [{"ts": t0.isoformat(), "event": "init", "id": "i", "cap": "100"}]
    for k, g in enumerate(ids):
        ts = (t0 + timedelta(minutes=k)).isoformat()
        ev += [{"ts": ts, "event": "reserve", "id": f"r{k}", "est_max": 0.1, "sample_id": f"{g}#1"},
               {"ts": ts, "event": "settle", "id": f"r{k}", "actual": 0.01, "sample_id": f"{g}#1"}]
    return "".join(json.dumps(e) + "\n" for e in ev)


@needs_pilot_logs
def test_collect_and_render_every_tag_host_and_chunk(tmp_path, monkeypatch):
    from inspect_ai.log import read_eval_log_sample_summaries

    ok_ids = [str(s.id) for s in read_eval_log_sample_summaries(str(PILOT_LOG))]
    err_ids = [str(s.id) for s in read_eval_log_sample_summaries(str(PILOT_LOG_ERR))]
    rounds = {g: 3 for g in ok_ids + err_ids + ["todo-1", "todo-2", "chunk-1", "chunk-2", "lost-1"]}
    units = {g: 6 for g in rounds}
    t0 = datetime(2026, 10, 25, 16, 20, tzinfo=timezone.utc)
    now = t0 + timedelta(minutes=40)
    r = tmp_path / "runs"
    for d in ("full/logs-mac", "full/logs-macv2", "shards", "shards-v2", "remote/vps1/logs-vps1v2",
              "remote/vps1/shards"):
        (r / d).mkdir(parents=True)
    shutil.copy2(PILOT_LOG, r / "full" / "logs-macv2" / PILOT_LOG.name)
    (r / "full" / "ledger-macv2.jsonl").write_text(_ledger_lines(ok_ids, t0))
    (r / "full" / "macv2.out").write_text("28 games -> specs\n")
    (r / "full" / "ledger-mac.jsonl").write_text(_ledger_lines([], t0))
    (r / "shards" / "mac.json").write_text(json.dumps({"host": "mac", "game_ids": ["todo-1", "lost-1"]}))
    (r / "shards" / "macv2.json").write_text(json.dumps({"host": "mac", "game_ids": ok_ids + ["todo-1"]}))
    shutil.copy2(PILOT_LOG_ERR, r / "remote" / "vps1" / "logs-vps1v2" / PILOT_LOG_ERR.name)
    (r / "remote" / "vps1" / "ledger-vps1v2.jsonl").write_text(_ledger_lines(err_ids, t0))
    (r / "remote" / "vps1" / "shards" / "vps1v2.json").write_text(
        json.dumps({"host": "vps1", "game_ids": err_ids + ["todo-2"]}))
    sync = (t0 + timedelta(minutes=35)).isoformat()
    (r / "remote" / "vps1" / "sync_state.json").write_text(json.dumps(
        {"host": "vps1", "logs_last_success": sync, "ledgers": {"ledger-vps1v2.jsonl": {"last_success": sync}}}))
    for i, ids in ((1, ["chunk-1"]), (2, ["chunk-2"])):
        (r / "shards-v2" / f"spill-{i}.json").write_text(json.dumps({"host": f"spill-{i}", "game_ids": ids}))
    (r / "dispatch_state.json").write_text(json.dumps({"claimed": {"spill-1": {"host": "instabox", "at": "23:05:09"}}}))
    (r / "dispatch_hosts.json").write_text(json.dumps({"mac": None, "vps1": "root@1.2.3.4", "instabox": "instabox"}))

    runs, hosts = progress.collect_runs(tmp_path, rounds, now)
    by_tag = {x.tag: x for x in runs}
    assert set(by_tag) == {"mac", "macv2", "vps1v2", "instabox-spill-1"}
    assert by_tag["mac"].retired and by_tag["instabox-spill-1"].placeholder
    assert by_tag["vps1v2"].shard_src == "runs/remote/vps1/shards/vps1v2.json"
    assert by_tag["vps1v2"].as_of == t0 + timedelta(minutes=35)
    text = progress.render(runs, hosts, tmp_path, rounds, units, now, 5, "balance $1.00")
    for needle in ("== mac (local)", "[macv2] v2", "[mac] v1 - retired", "== vps1 (root@1.2.3.4)", "[vps1v2] v2",
                   "== instabox (instabox)", "[instabox-spill-1] overflow - claimed", "spill-1 claimed by instabox",
                   "spill-2 queued (1 games, 3 rounds,", "failed samples", "== overall", "account balance $1.00",
                   "min for the last games"):
        assert needle in text, needle
    # lost-1: in the retired v1 shard only, so it is not pending anywhere and needs a re-run
    assert "lost-1" in text.split("== overall")[1]
    only = progress.render(runs, hosts, tmp_path, rounds, units, now, 5, None, only=["vps1"])
    assert "== vps1" in only and "== mac (local)" not in only and "== overall" in only
    # tools/resume_plan.py reads completed/failed ids through progress.collect(hosts, rounds): every tag of a host
    monkeypatch.setattr(progress, "ROOT", tmp_path)
    from tools.resume_plan import from_logs

    completed, failed, where = from_logs(["mac", "vps1"])
    assert set(ok_ids) <= completed and where[ok_ids[0]] == "mac"
    assert failed and failed <= set(err_ids) and not failed & completed


def test_progress_ignores_copied_ledgers_directory_as_a_host(tmp_path):
    """An archive of old ledgers is not a runner: it must not add spend, pending work, or an ETA lane."""
    t0 = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc)
    root = tmp_path / "study"
    runs_dir = root / "runs"
    host_dir = runs_dir / "remote" / "instabox"
    copies_dir = runs_dir / "remote" / "ledgers"
    for d in (host_dir, copies_dir, runs_dir / "shards"):
        d.mkdir(parents=True)
    (runs_dir / "dispatch_hosts.json").write_text(json.dumps({"instabox": "instabox"}))
    (runs_dir / "shards" / "instabox-v5.json").write_text(
        json.dumps({"host": "instabox", "game_ids": ["g1", "g2"]}))
    ledger = _ledger_lines(["g1"], t0)
    (host_dir / "ledger-instabox-v5.jsonl").write_text(ledger)
    (copies_dir / "ledger-instabox-v5.jsonl").write_text(ledger)
    (host_dir / "sync_state.json").write_text(json.dumps({"host": "instabox", "logs_last_success": t0.isoformat()}))
    # Even if an archive later gains metadata, its reserved name cannot turn it into a host.
    (copies_dir / "sync_state.json").write_text(json.dumps({"host": "ledgers"}))

    rounds = {"g1": 3, "g2": 3}
    runs, hosts = progress.collect_runs(root, rounds, t0 + timedelta(minutes=10), local_runners=set())
    assert {"mac", "instabox"} <= set(hosts) and "ledgers" not in hosts
    assert [(r.host, r.tag) for r in runs] == [("instabox", "instabox-v5")]
    text = progress.render(runs, hosts, root, rounds, {"g1": 6, "g2": 6},
                           t0 + timedelta(minutes=10), 2, None)
    assert "== ledgers" not in text
    assert "spent $0.01 across 1 ledgers" in text


# ------------------------------------------------------------------ adversarial checks: transport, freshness, ETA
def test_every_remote_command_only_reads(tmp_path, monkeypatch):
    """The listing, the rsync pulls and the scp fallback, as built: remote side is only a source, nothing deletes."""
    import re

    seen = []

    class Done:
        returncode, stdout, stderr = 0, "__listing_end__\n", ""

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return Done()

    monkeypatch.setattr(sync_logs.subprocess, "run", fake_run)
    sync_logs.remote_listing("instabox", 60)
    sync_logs.rsync_pull("instabox", tmp_path / "s", 60)
    sync_logs.rsync_pull("instabox", tmp_path / "s", 60, only=["full/logs-a/x.eval"])
    sync_logs.scp_pull("instabox", tmp_path / "s", ["full/ledger-a.jsonl", "shards/a.json"], None, 60)
    ssh, rs1, rs2, *scps = seen
    assert ssh[0] == "ssh" and ssh[-2] == "instabox"
    # the one remote shell command: cd, a find that only prints, echo, and ps piped into grep (all read-only)
    assert re.fullmatch(r"cd ~/shared-host-collusion/runs && \{ find full shards -maxdepth 2 -type f "
                        r"-printf '[^']*' 2>/dev/null; echo __listing_end__; if command -v ps >/dev/null 2>&1; "
                        r"then echo __ps_ok__; ps -eo args= 2>/dev/null \| grep -- '--ledger runs/full/ledge\[r\]-'; "
                        r"fi; echo __procs_end__; \}", ssh[-1]), ssh[-1]
    writes = {"--delete", "--del", "--delete-after", "--delete-before", "--delete-during", "--remove-source-files",
              "--inplace", "--append", "--append-verify", "-S", "--sparse"}
    for c in (rs1, rs2):
        assert c[0] == "rsync" and c[-2] == "instabox:shared-host-collusion/runs/" and c[-1] == f"{tmp_path / 's'}/"
        assert not {a.split("=")[0] for a in c} & writes
    assert len(scps) == 2  # one call per remote directory
    for c in scps:
        sources = [a for a in c if a.startswith("instabox:")]
        assert c[0] == "scp" and ":" not in c[-1] and len(sources) == 1 and c[-2] == sources[0]
        assert sources[0] in ("instabox:shared-host-collusion/runs/full/ledger-a.jsonl",
                              "instabox:shared-host-collusion/runs/shards/a.json")
    assert sync_logs.load_hosts(_hosts_file(tmp_path, {"x": "-oProxyCommand=evil"})) == sync_logs.DEFAULT_HOSTS


def _hosts_file(tmp_path, data):
    f = tmp_path / "hosts.json"
    f.write_text(json.dumps(data))
    return f


def test_local_rsync_renames_into_place_and_never_writes_through_the_promoted_link(tmp_path):
    """Each log is stored once (promoted copy = hard link to the staged copy), which is only safe if this machine's
    rsync writes a temp file and renames it. Also checks the pull filters with the real rsync."""
    if shutil.which("rsync") is None:
        pytest.skip("no rsync")
    src, stage = tmp_path / "src", tmp_path / "stage"
    (src / "full" / "logs-t").mkdir(parents=True)
    (src / "shards").mkdir()
    log = src / "full" / "logs-t" / "a.eval"
    log.write_bytes(b"v1" * 4000)
    (src / "full" / "ledger-t.jsonl").write_text("{}\n")
    (src / "full" / "specs_full-all-t.json").write_text("{}")
    (src / "full" / "logs-t" / ".a.eval.tmp").write_text("partial")
    (src / "shards" / "t.json").write_text("{}")
    stage.mkdir()

    def rsync(*flags):
        r = subprocess.run(["rsync", "-a", "-z", *flags, f"{src}/", f"{stage}/"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr

    rsync(*sync_logs.RSYNC_FILTERS)
    pulled = {p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()}
    assert pulled == {"full/logs-t/a.eval", "full/ledger-t.jsonl", "shards/t.json"}
    staged, final = stage / "full" / "logs-t" / "a.eval", tmp_path / "final.eval"
    os.link(staged, final)
    log.write_bytes(b"v1" * 3000 + b"v2" * 1000)  # the runner rewrote the log: rsync uses the old copy as delta basis
    os.utime(log, (log.stat().st_atime, log.stat().st_mtime + 10))
    rsync(*sync_logs.RSYNC_FILTERS)
    assert staged.read_bytes().endswith(b"v2") and final.read_bytes() == b"v1" * 4000
    assert not os.path.samefile(staged, final)
    os.unlink(final)
    os.link(staged, final)
    rsync("-I", *sync_logs._only_filters(["full/logs-t/a.eval"]))  # the retry path
    assert not os.path.samefile(staged, final) and final.read_bytes() == staged.read_bytes()


def _clock(monkeypatch):
    t = {"k": 0}

    def tick():
        t["k"] += 1
        return (datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc) + timedelta(seconds=t["k"])).isoformat()

    monkeypatch.setattr(sync_logs, "now_iso", tick)


@needs_pilot_logs
def test_partial_pull_is_not_reported_as_current(tmp_path, monkeypatch):
    runs = _fake_remote(tmp_path)
    _install_fake_transport(monkeypatch, runs)
    _clock(monkeypatch)
    local = tmp_path / "local"
    st = sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60, base_root=local)
    first = st["tags"]["vps1v2"]["data_at"]
    assert first and st["tags"]["vps1"]["data_at"] == first == st["remote_listed_at"]
    assert st["tags"]["vps1-spill-2"]["data_at"] is None  # its only log is torn: never current
    led_ok = st["ledgers"]["ledger-vps1v2.jsonl"]["last_success"]
    assert progress._remote_as_of(st, "vps1v2", datetime.now(timezone.utc))[0] == datetime.fromisoformat(first)

    # the ledger grows on the remote, but the pull reports success without bringing it (rsync exit 23 or 24)
    ledger = runs / "full" / "ledger-vps1v2.jsonl"
    with open(ledger, "ab") as f:
        f.write(b'ttle"}\n')
    os.utime(ledger, (ledger.stat().st_atime, ledger.stat().st_mtime + 5))
    monkeypatch.setattr(sync_logs, "rsync_pull", lambda dest, stage, timeout, only=None: (True, "code 23"))
    st = sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60, base_root=local)
    assert st["stale_after_pull"] == ["ledger-vps1v2.jsonl"]
    assert st["ledgers"]["ledger-vps1v2.jsonl"]["last_success"] == led_ok
    assert st["tags"]["vps1v2"]["data_at"] == first  # not moved forward
    assert st["tags"]["vps1"]["data_at"] == st["remote_listed_at"] > first  # unchanged files are current
    assert (local / "vps1" / "ledger-vps1v2.jsonl").read_bytes().count(b"\n") == 2


@needs_pilot_logs
def test_second_sync_of_a_host_is_skipped_while_one_runs(tmp_path, monkeypatch):
    import fcntl

    runs = _fake_remote(tmp_path)
    _install_fake_transport(monkeypatch, runs)
    base = tmp_path / "local" / "vps1"
    base.mkdir(parents=True)
    with open(base / ".lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60,
                                 base_root=tmp_path / "local")
        assert st["skipped"] and "skipped" in sync_logs.report("vps1", st).lower()
        assert not (base / "sync_state.json").exists() and not (base / ".staging").exists()
    st = sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60,
                             base_root=tmp_path / "local")
    assert not st.get("skipped") and (base / "logs-vps1" / PILOT_LOG.name).exists()


@needs_pilot_logs
def test_lost_state_never_replaces_a_log_with_a_smaller_copy(tmp_path):
    final, staged = tmp_path / "final" / "x.eval", tmp_path / "stage" / "x.eval"
    final.parent.mkdir()
    staged.parent.mkdir()
    shutil.copy2(PILOT_LOG, final)       # 28 samples
    shutil.copy2(PILOT_LOG_ERR, staged)  # 14 samples, valid
    rec: dict = {}
    assert not sync_logs._promote_eval(staged, final, rec, deep=True)
    assert "< last good 28" in rec["last_error"] and final.read_bytes() == PILOT_LOG.read_bytes()


def test_stale_rsync_temp_files_are_cleaned_from_staging_only(tmp_path):
    stage = tmp_path / ".staging"
    (stage / "full" / "logs-a").mkdir(parents=True)
    old, new = stage / "full" / "logs-a" / ".x.eval.Ab12Cd", stage / "full" / "logs-a" / ".y.eval.Ef34Gh"
    old.write_bytes(b"x")
    new.write_bytes(b"y")
    os.utime(old, (old.stat().st_atime, old.stat().st_mtime - 2 * sync_logs.STALE_TEMP))
    (stage / "full" / "logs-a" / "x.eval").write_bytes(b"z")
    assert sync_logs._staged(stage) == {"full/logs-a/x.eval": "eval"}
    assert not old.exists() and new.exists()


def test_runner_work_counts_games_done_elsewhere_and_finished_needs_a_final_log():
    t0 = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    scan = progress.LogScan()
    scan.games["a"] = progress.Game("a", True, 40, t0, t0 + timedelta(minutes=20), "", "l.eval")
    scan.games["b"] = progress.Game("b", False, 40, t0, t0 + timedelta(minutes=5), "boom", "l.eval")
    r = progress.Run("mac", "macv2", "v2", None, t0, ["a", "b", "c", "d"], "x", scan, progress.LedgerScan(),
                     {"exists": True, "n_specs": 4})
    # c was completed by a v1 runner; macv2 was given its whole shard, so it still runs c
    assert progress.pending_ids(r, {"a", "c"}) == ["c", "d"]
    r.out["n_specs"] = 2  # given fewer games than its shard: an exclude list skipped games done elsewhere
    assert progress.pending_ids(r, {"a", "c"}) == ["d"]
    assert not progress.run_finished({"started"}, {"finished": "completed 96 games; provider charges $1"})
    assert progress.run_finished({"success"}, {"finished": ""})
    assert progress.run_finished(set(), {"finished": "available budget $0 cannot reserve the requested 24-game"})
    assert not progress.run_finished(set(), {"exists": False})


def test_scan_out_reads_games_given_to_the_runner(tmp_path):
    p = tmp_path / "x.out"
    p.write_text("ledger ...\n96 games -> /data/home/shared-host-collusion/runs/full/specs_full-all-spill-1.json\n")
    assert progress.scan_out(p)["n_specs"] == 96


def test_tail_allowance_in_host_and_chunk_eta():
    assert progress.tail_allowance(30.0, [40, 40, 30]) == timedelta(seconds=600)
    assert progress.tail_allowance(None, [40]) == timedelta(0)
    t0 = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    lag = timedelta(minutes=4)
    hosts = {"a": {"free_at": t0, "kind": "rounds", "rate": 100.0, "spr": 30.0}}
    got = progress.assign_overflow(hosts, [("spill-2", {"units": 0, "rounds": 100, "game_rounds": [40, 40]})], lag)
    assert got == [("spill-2", "a", t0 + lag, t0 + lag + timedelta(hours=1, seconds=600))]


def test_spend_projection_observed_and_price_scaled(tmp_path):
    from decimal import Decimal

    t0 = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    models = {f"a{i}": ("cheap", "x/cheap") for i in range(4)} | {f"b{i}": ("dear", "x/dear") for i in range(4)} \
        | {"n1": ("new", "x/new"), "z1": ("ghost", "x/none")}
    units = {g: 10 for g in models}
    prices = {"x/cheap": 1.0, "x/dear": 4.0, "x/new": 2.0}
    scan, led = progress.LogScan(), progress.LedgerScan(exists=True, cap=Decimal(50))
    for g, c in (("a0", 1), ("a1", 1), ("a2", 1), ("b0", 8), ("b1", 8), ("b2", 8)):  # $/unit 0.1 and 0.8
        scan.games[g] = progress.Game(g, True, 10, t0, t0, "", "l.eval")
        led.cost_by_game[g] = Decimal(c)
        led.spent += Decimal(c)
    led.cost_by_game["b3"] = Decimal(3)  # in flight: $3 settled so far
    led.spent += Decimal(3)
    r = progress.Run("mac", "macv2", "v2", None, t0, list(models), "x", scan, led, {"exists": True})
    cm = progress.cost_model([r], units, models, prices)
    assert cm["cheap"] == (0.1, 0.1, 0.1, "observed") and cm["dear"][3] == "observed"
    # observed $/unit per unit of price: 0.1 / 1 and 0.8 / 4 -> median 0.15, range 0.1-0.2; "new" costs 2
    assert cm["new"][:3] == pytest.approx((0.3, 0.2, 0.4)) and cm["new"][3] == "price-scaled"
    assert "ghost" not in cm
    p = progress.project_cost(progress.pending_ids(r, set()), led.cost_by_game, models, units, cm)
    # a3: $1; b3: $8 - $3 settled = $5; n1: 10 units at the scaled rate; z1: no estimate
    assert p["c"] == pytest.approx(1 + 5 + 10 * cm["new"][0]) and p["unknown"] == 1 and p["scaled"] == 1
    spent = float(led.spent)
    assert progress.cap_flag(spent, 0.0, 1000.0, dict(p, unknown=0)) == ""
    assert progress.cap_flag(spent, 0.0, spent + p["c"] - 0.01, p) == " LIKELY TO HIT CAP"
    assert progress.cap_flag(spent, 0.0, spent + p["c"] + 0.01, p) == " may hit cap"  # high end, or no estimate
    assert progress.cap_flag(spent, 2.0, spent + p["c"] + 1.0, p) == " LIKELY TO HIT CAP"  # reservations count
    assert "games with no estimate" in progress.fmt_proj(spent, p)


def test_dispatcher_cap_is_read_without_running_it(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "dispatcher.py").write_text(
        'import sys\nCAP, CONC = {"mac": 60, "vps": 75}, {"mac": 24}\nraise SystemExit("must not run")\n')
    assert progress.dispatcher_caps(tmp_path) == {"mac": 60.0, "vps": 75.0}
    assert progress.chunk_cap(progress.dispatcher_caps(tmp_path), "mac") == 60.0
    assert progress.chunk_cap(progress.dispatcher_caps(tmp_path), "instabox") == 75.0


@needs_dispatcher
def test_the_operational_dispatcher_parses():
    assert progress.dispatcher_caps(ROOT)  # the real one parses
    assert progress.dispatcher_split_hosts(ROOT) == {"instabox"}  # the live dispatcher (D10)


def test_listing_parses_files_and_live_runners():
    out = ("10 1792951730.5 full/logs-vps1v2/a.eval\n5 1792951731 full/ledger-vps1v2.jsonl\n3 1 full/specs_x.json\n"
           "__listing_end__\n__ps_ok__\nsg docker -c .venv/bin/python -m harness.run --plan full --shard-file "
           "runs/shards/vps1v2.json --ledger runs/full/ledger-vps1v2.jsonl --log-dir runs/full/logs-vps1v2\n"
           ".venv/bin/python -m harness.run --plan full --ledger runs/full/ledger-instabox-spill-2.jsonl\n"
           "__procs_end__\n")
    listing, procs = sync_logs.parse_listing(out)
    assert listing == {"full/logs-vps1v2/a.eval": (10, 1792951730.5), "full/ledger-vps1v2.jsonl": (5, 1792951731.0)}
    assert procs == {"vps1v2", "instabox-spill-2"}
    assert sync_logs.parse_listing("1 1 full/x.out\n__listing_end__\n__procs_end__\n")[1] is None  # no ps
    assert sync_logs.parse_listing("1 1 full/x.out\n") == (None, None)  # cut short
    assert progress.RUNNER_LEDGER.pattern == sync_logs.RUNNER_LEDGER.pattern


@needs_pilot_logs
def test_runner_state_drives_retirement(tmp_path, monkeypatch):
    """A stopped runner (D10: macv2 split into instabox-core and macv3-gated) runs nothing more; its games are
    counted where they were moved, not twice."""
    runs = _fake_remote(tmp_path)
    calls = _install_fake_transport(monkeypatch, runs)
    real_listing = sync_logs.remote_listing
    monkeypatch.setattr(sync_logs, "remote_listing", lambda d, t: (*real_listing(d, t)[:2], {"vps1v2"}))
    st = sync_logs.sync_host("vps1", "root@x", retries=0, retry_wait=0, deep=True, timeout=60,
                             base_root=tmp_path / "local")
    assert st["tags"]["vps1v2"]["runner_alive"] is True and st["tags"]["vps1"]["runner_alive"] is False
    assert st["runners"]["tags"] == ["vps1v2"] and calls["rsync"] == 1

    t0 = datetime(2026, 10, 25, 18, 0, tzinfo=timezone.utc)
    r = tmp_path / "root" / "runs"
    for d in ("full/logs-macv2", "full/logs-macv3-gated", "shards", "remote"):
        (r / d).mkdir(parents=True)
    (r / "shards" / "macv2.json").write_text(json.dumps({"host": "mac", "game_ids": ["g1", "g2", "g3"]}))
    (r / "shards" / "macv3-gated.json").write_text(json.dumps({"host": "mac", "game_ids": ["g2", "g3"]}))
    for tag in ("macv2", "macv3-gated"):
        (r / "full" / f"ledger-{tag}.jsonl").write_text(_ledger_lines([], t0))
    got, _ = progress.collect_runs(tmp_path / "root", {"g1": 3, "g2": 3, "g3": 3}, t0, local_runners={"macv3-gated"})
    by = {x.tag: x for x in got}
    assert by["macv2"].retired and by["macv2"].alive is False and not by["macv3-gated"].retired
    assert progress.pending_ids(by["macv2"], set()) == [] and progress.pending_ids(by["macv3-gated"], set()) == ["g2", "g3"]
    text = progress.render(got, {"mac": None}, tmp_path / "root", {"g1": 3, "g2": 3, "g3": 3}, {"g1": 6, "g2": 6, "g3": 6},
                           t0, 5, None, models={})
    assert "[macv2] v2 - retired: runner stopped before finishing" in text
    assert "need a re-run (not done, not pending anywhere): 1, e.g. g1" in text
    # unknown process state (older sync state): the old rule, a v1 tag beside a newer tag is retired
    got, _ = progress.collect_runs(tmp_path / "root", {}, t0, local_runners=None)
    assert not any(x.retired for x in got)


def test_split_host_runs_chunks_beside_its_core_work():
    t0 = datetime(2026, 10, 25, 18, 30, tzinfo=timezone.utc)
    rounds = {f"c{i}": 30 for i in range(10)} | {f"o{i}": 30 for i in range(4)}
    units = {g: 60 for g in rounds}

    def run(tag, kind, ids, done):
        scan = progress.LogScan()
        for k, g in enumerate(done):
            end = t0 - timedelta(minutes=50 - 5 * k)
            scan.games[g] = progress.Game(g, True, 30, end - timedelta(minutes=10), end, "", "l.eval")
        led = progress.LedgerScan(exists=True, first_ts=t0 - timedelta(minutes=60))
        return progress.Run("instabox", tag, kind, "spill-1" if kind == "overflow" else None, t0, ids, "x", scan, led,
                            {"exists": True}, alive=True)

    core = run("instabox-core", "other", [f"c{i}" for i in range(10)], ["c0", "c1"])
    ovf = run("instabox-spill-1", "overflow", [f"o{i}" for i in range(4)], ["o0", "o1", "o2"])
    all_ok = {"c0", "c1", "o0", "o1", "o2"}
    whole = progress.plan_hosts([core, ovf], {"instabox": "instabox"}, rounds, units, t0, all_ok)["instabox"]
    split = progress.plan_hosts([core, ovf], {"instabox": "instabox"}, rounds, units, t0, all_ok,
                                {"instabox"})["instabox"]
    assert set(split.lanes) == {"core", "overflow"}
    assert split.lanes["overflow"]["pend"] == ["o3"] and len(split.lanes["core"]["pend"]) == 8
    # the overflow lane frees up long before all of the host's work is done, and the dispatcher sees only that lane
    assert split.ovf_free_at < split.lanes["core"]["end"] == split.own_free_at
    assert whole.ovf_free_at == whole.own_free_at and not whole.lanes


def test_dispatcher_split_hosts_read_from_alive(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "dispatcher.py").write_text(
        'def alive(ip, host=""):\n    pattern = f"runs/shards/{host}-spil[l]-" if host == "instabox" else "x"\n'
        '    return pattern\nraise SystemExit("never run")\n')
    assert progress.dispatcher_split_hosts(tmp_path) == {"instabox"}


def test_out_tail_never_shows_the_running_score(tmp_path):
    p = tmp_path / "x.out"
    p.write_text("play_game    | Steps:  57/96  59% | Samples:  56/ 96 | mean: 0.08 | accuracy=0.5 | docker: 23/24\n")
    tail = progress.scan_out(p)["tail"]
    assert "0.08" not in tail and "0.5" not in tail and "mean: (hidden)" in tail and "Samples:  56/ 96" in tail
