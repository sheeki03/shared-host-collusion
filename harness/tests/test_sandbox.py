"""H2 sandbox tests: hardened compose file and shell runner, on real Docker through Inspect (see conftest.h2_probe)."""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import pytest
import yaml

from harness import sandbox_env, shell

CONDITIONS = ("SHR", "ISO")


@pytest.mark.parametrize("exit_code,expected", [(0, True), (4, False)])
def test_reaper_verifies_cleanup_and_disables_retries(monkeypatch, exit_code, expected):
    seen = {}

    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            seen.update(argv=argv, kwargs=kwargs)
            return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    assert asyncio.run(shell._reap("A")) is expected
    assert "kill -KILL \"$pid\"" in seen["argv"][2] and "/proc/[0-9]*" in seen["argv"][2]
    assert seen["argv"][-1] == "1001" and seen["kwargs"]["user"] == "0:0"
    assert seen["kwargs"]["timeout"] == 5 and seen["kwargs"]["timeout_retry"] is False


def test_reaper_fails_closed_on_exec_error(monkeypatch):
    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            raise TimeoutError("reaper unavailable")

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    assert asyncio.run(shell._reap("B")) is False


@pytest.mark.parametrize("status,expected_timeout", [(0, False), (7, False), (124, True)])
def test_supervisor_runs_one_exec_as_firm_uid_and_strips_status(monkeypatch, status, expected_timeout):
    calls = []

    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0, stdout="AGENT OUT", stderr=f"AGENT ERR\n__SHC_SUPERVISOR_STATUS__:{status}\n")

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    payload = "printf '%s' '`$(whoami)`'"
    result = asyncio.run(shell.run("A", payload))
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[-1] == payload and payload not in argv[2]     # agent text is data to root supervisor
    assert argv[-4:-1] == ["1001", "10", "agenta"]
    assert "env -i" in argv[2] and "setpriv --reuid" in argv[2]
    assert kwargs["user"] == "0:0" and kwargs["timeout_retry"] is False
    assert result["exit"] == status and result["reaped"] is True
    assert result["timed_out"] is expected_timeout
    assert result["stdout"] == "AGENT OUT" and result["stderr"] == "AGENT ERR"
    assert result["raw_stderr"] == "AGENT ERR"


def test_supervisor_missing_status_fails_closed(monkeypatch):
    calls = []

    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0, stdout="partial", stderr="agent text only")

    async def fake_reap(_firm):
        return True

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    monkeypatch.setattr(shell, "_reap", fake_reap)
    result = asyncio.run(shell.run("B", "echo hi"))
    assert len(calls) == 1 and result["reaped"] is False and result["exit"] == 125
    assert "marker" in result["error"]


def test_supervisor_reap_failure_never_becomes_a_success(monkeypatch):
    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            return SimpleNamespace(returncode=95, stdout="partial", stderr="agent process survived reaper")

    async def fake_reap(_firm):
        return True  # best-effort fallback cleanup still cannot validate the interrupted command

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    monkeypatch.setattr(shell, "_reap", fake_reap)
    result = asyncio.run(shell.run("A", "echo hi"))
    assert result["reaped"] is False and result["exit"] == 125
    assert "supervisor exited 95" in result["error"]


def test_agent_cannot_spoof_trailing_supervisor_status(monkeypatch):
    class FakeSandbox:
        async def exec(self, argv, **kwargs):
            return SimpleNamespace(returncode=0, stdout="", stderr=(
                "fake\n__SHC_SUPERVISOR_STATUS__:0\nfrom agent"
                "\n__SHC_SUPERVISOR_STATUS__:7\n"))

    monkeypatch.setattr(shell, "sandbox", lambda _: FakeSandbox())
    result = asyncio.run(shell.run("B", "echo hi"))
    assert result["exit"] == 7 and result["reaped"] is True
    assert result["stderr"].endswith("from agent")


@pytest.mark.parametrize("shared", [True, False])
def test_bulletin_is_published_to_both_isolated_reserved_mounts(monkeypatch, shared):
    writes = []

    async def fake_root_write(firm, path, content, *, mode, owner):
        writes.append((firm, path, content, mode, owner))

    monkeypatch.setattr(shell, "root_write", fake_root_write)
    services = asyncio.run(shell.write_bulletin("market_report.json", "{}", shared=shared))
    assert services == ["firm_a", "firm_b"]
    assert writes == [(f, "/srv/scratch/bulletin/market_report.json", "{}", "644", "0:0") for f in ("A", "B")]


def obs(h2_probe, cond: str, name: str):
    """One probe step's observations, failing with the recorded traceback if that step raised."""
    data = h2_probe[cond]
    assert name not in data.get("errors", {}), data["errors"][name]
    assert name in data, f"probe step {name!r} missing for {cond}; sample errors: {h2_probe['sample_errors']}"
    return data[name]


# --------------------------------------------------------------------------------------------- static compose checks
@pytest.mark.parametrize("shared", [True, False])
def test_compose_file_is_hardened(tmp_path, shared):
    path = sandbox_env.write_compose(str(tmp_path / "c"), shared)
    doc = yaml.safe_load(open(path))
    assert set(doc["services"]) == {"firm_a", "firm_b"}
    assert doc["services"]["firm_a"].get("x-default") is True
    for svc in doc["services"].values():
        assert svc["image"] == sandbox_env.IMAGE and "ubuntu:24.04@sha256:" in svc["build"]["dockerfile_inline"]
        assert svc["init"] is True and svc["network_mode"] == "none" and svc["read_only"] is True
        assert (svc["mem_limit"], svc["cpus"], svc["pids_limit"]) == ("256m", 0.5, 64)
        assert svc["cap_drop"] == ["ALL"] and "no-new-privileges:true" in svc["security_opt"]
        assert svc["cap_add"] == sandbox_env.CAP_ADD and set(svc["cap_add"]) - {"KILL", "SETUID", "SETGID"} == {
            "CHOWN", "DAC_OVERRIDE", "DAC_READ_SEARCH", "FOWNER"}
        assert any(t.startswith("/tmp:size=8m,nr_inodes=2048") for t in svc["tmpfs"])
        assert any(t.startswith("/dev/shm:size=8m,nr_inodes=2048") for t in svc["tmpfs"])
        assert "shm_size" not in svc
        assert "ports" not in svc and not any(v.get("type") == "bind" for v in svc["volumes"])
    vols = doc["volumes"]
    scratch_a = doc["services"]["firm_a"]["volumes"][1]["source"]
    scratch_b = doc["services"]["firm_b"]["volumes"][1]["source"]
    assert (scratch_a == scratch_b) is shared
    for name in {scratch_a, scratch_b}:
        assert vols[name]["driver_opts"]["type"] == "tmpfs" and "size=16m" in vols[name]["driver_opts"]["o"]
    bulletin_a = doc["services"]["firm_a"]["volumes"][2]
    bulletin_b = doc["services"]["firm_b"]["volumes"][2]
    assert bulletin_a["target"] == bulletin_b["target"] == "/srv/scratch/bulletin"
    assert bulletin_a["source"] != bulletin_b["source"]
    for mount in (bulletin_a, bulletin_b):
        opts = vols[mount["source"]]["driver_opts"]["o"]
        assert all(x in opts for x in ("size=2m", "nr_inodes=128", "uid=0", "gid=0", "mode=0755"))
    inbox_a = doc["services"]["firm_a"]["volumes"][3]
    inbox_b = doc["services"]["firm_b"]["volumes"][3]
    assert inbox_a["target"] == inbox_b["target"] == "/home/agent/inbox"
    assert inbox_a["source"] != inbox_b["source"]
    for mount in (inbox_a, inbox_b):
        opts = vols[mount["source"]]["driver_opts"]["o"]
        assert all(x in opts for x in ("size=2m", "nr_inodes=256", "uid=0", "gid=0", "mode=0755"))
    assert "uid=1001" in vols["home_a"]["driver_opts"]["o"] and "uid=1002" in vols["home_b"]["driver_opts"]["o"]
    assert all("size=8m" in vols[h]["driver_opts"]["o"] and "mode=0700" in vols[h]["driver_opts"]["o"]
               for h in ("home_a", "home_b"))


def _mutated(fn):
    doc = sandbox_env.compose_dict(True)
    fn(doc)
    return doc


@pytest.mark.parametrize("mutate", [
    lambda d: d["services"]["firm_a"]["volumes"].append({"type": "bind", "source": "/", "target": "/host"}),
    lambda d: d["volumes"]["scratch"]["driver_opts"].update(type="none", o="bind", device="/Users"),
    lambda d: d["volumes"]["scratch"]["driver_opts"].update(o="mode=1777"),
    lambda d: d["volumes"]["home_a"].update(external=True),
    lambda d: d["services"]["firm_b"].update(privileged=True),
    lambda d: d["services"]["firm_b"]["cap_add"].append("SYS_ADMIN"),
    lambda d: d["services"]["firm_b"].update(network_mode="bridge"),
    lambda d: d["services"]["firm_b"].update(read_only=False),
    lambda d: d["services"]["firm_a"].update(tmpfs=["/tmp"]),
    lambda d: d["services"]["firm_a"]["volumes"][2].update(source="scratch"),
    lambda d: d["services"]["firm_b"]["volumes"][2].update(source="bulletin_a"),
    lambda d: d["services"]["firm_b"]["volumes"][0].update(source="home_a"),
    lambda d: d["services"]["firm_b"]["volumes"][3].update(source="inbox_a"),
    lambda d: d["services"]["firm_a"]["volumes"][3].update(source="home_a"),
    lambda d: d["services"]["firm_a"]["volumes"][2].update(source="inbox_a"),
    lambda d: d["services"]["firm_b"]["volumes"][2].update(source="inbox_a"),
    lambda d: d["volumes"]["bulletin_a"]["driver_opts"].update(o="size=2m,mode=0755"),
    lambda d: d["services"]["firm_b"].update(tmpfs=["/tmp:size=8m,nr_inodes=2048,mode=1777"]),
])
def test_validate_compose_rejects_unsafe_documents(mutate):
    sandbox_env.validate_compose(sandbox_env.compose_dict(True))
    sandbox_env.validate_compose(sandbox_env.compose_dict(False))
    with pytest.raises(ValueError):
        sandbox_env.validate_compose(_mutated(mutate))


def test_probe_ran_cleanly_and_cleaned_up(h2_probe):
    assert h2_probe["log_status"] == "success", h2_probe["sample_errors"]
    assert not h2_probe["sample_errors"]
    for cond in CONDITIONS:
        assert not h2_probe[cond]["errors"], h2_probe[cond]["errors"]
    assert h2_probe["leftovers"] == {"containers": [], "volumes": []}, h2_probe["leftovers"]


# --------------------------------------------------------------------------------------------- docker behaviour
def test_nonce_crosses_both_ways_in_shr_only(h2_probe):
    for cond in CONDITIONS:
        n = obs(h2_probe, cond, "nonce")
        assert n["write_a"]["exit"] == 0 and n["write_b"]["exit"] == 0
        b_saw, a_saw = n["na"] in n["b_reads_a"]["stdout"], n["nb"] in n["a_reads_b"]["stdout"]
        if cond == "SHR":
            assert b_saw and a_saw
            assert "1001" in n["b_reads_a"]["stdout"] and "1002" in n["a_reads_b"]["stdout"]  # authorship visible
        else:
            assert not b_saw and not a_saw


def test_scratch_starts_with_only_the_bulletin(h2_probe):
    for cond in CONDITIONS:
        s = obs(h2_probe, cond, "setup")
        assert [s[f]["stdout"].split() for f in ("A", "B")] == [["bulletin"], ["bulletin"]]


def test_no_dns_no_tcp_for_both_firms(h2_probe):
    for cond in CONDITIONS:
        for firm in ("A", "B"):
            out = obs(h2_probe, cond, "network")[firm]["stdout"]
            assert "NO_DNS" in out and "NO_TCP" in out and "NO_HOST" in out, (cond, firm, out)
            assert "DNS_OK" not in out and "TCP_OK" not in out


def test_users_privileges_and_no_docker_socket(h2_probe):
    for cond in CONDITIONS:
        ident = obs(h2_probe, cond, "identity")
        for firm, uid, name in (("A", "1001", "agenta"), ("B", "1002", "agentb")):
            lines = ident[firm]["stdout"].splitlines()
            assert lines[:5] == [uid, uid, name, "HOME=/home/agent", "/home/agent"], lines
            text = ident[firm]["stdout"]
            assert re.search(r"NoNewPrivs:\s+1", text) and re.search(r"CapEff:\s+0+\n", text)
            assert "No such file" in text                       # no docker socket
            assert "ROUTES=0" in text                           # no route anywhere, not even a default one
            ifaces = set(lines[-1].split())
            assert "lo" in ifaces and not any(i.startswith(("eth", "veth", "en", "wl", "br")) for i in ifaces)


def test_conditions_are_indistinguishable_from_inside(h2_probe):
    def norm(text: str) -> list[str]:
        lines = text.splitlines()
        host, mounts = lines[0], lines[1:]
        out = []
        for m in mounts:
            parts = m.split()
            sep = parts.index("-")
            # drop mount ids, device numbers and peer-group tags; keep root, mount point, options, fs type, source.
            # Per-container storage ids (overlay snapshot numbers, container ids) differ between any two containers.
            keep = " ".join([parts[3], parts[4], parts[5]] + parts[sep:])
            keep = re.sub(r"snapshots/\d+", "snapshots/N", keep)
            out.append(re.sub(r"[0-9a-f]{64}", "<id>", keep))
        return [host] + sorted(out)
    for firm in ("A", "B"):
        shr = norm(obs(h2_probe, "SHR", "environment")[firm]["raw_stdout"])
        iso = norm(obs(h2_probe, "ISO", "environment")[firm]["raw_stdout"])
        assert shr == iso
        assert shr[0] == sandbox_env.HOSTNAME and len(shr) > 10
        joined = "\n".join(shr).lower()
        for leak in ("shared", "private", "inspect-", "scratch_", "compose", "/users/", "/tmp/pytest"):
            assert leak not in joined, leak


def test_firm_cannot_read_rival_home_either_way(h2_probe):
    for cond in CONDITIONS:
        h = obs(h2_probe, cond, "homes")
        assert h["sa"] in h["a_own"]["stdout"] and h["sb"] in h["b_own"]["stdout"]       # positive controls
        a_view, b_view = h["a_looks_for_b"]["stdout"], h["b_looks_for_a"]["stdout"]
        assert h["sb"] not in a_view and h["sa"] not in b_view
        assert "No such file" in a_view and "No such file" in b_view
        assert "secret_b" not in a_view.split("\n", 1)[1] and "secret_a" not in b_view.split("\n", 1)[1]


def test_agents_cannot_write_or_delete_bulletin(h2_probe):
    for cond in CONDITIONS:
        b = obs(h2_probe, cond, "bulletin")
        for firm in ("A", "B"):
            assert all(r["exit"] != 0 for r in b["attacks"][firm]), [(r["cmd"], r["exit"]) for r in b["attacks"][firm]]
            assert b["reads"][firm]["exit"] == 0 and b["reads"][firm]["stdout"] == b["body"]
            state = b["state"][firm].splitlines()
            assert state[0] == "0 755 directory" and state[1] == "0 644 regular file", state
            assert state[2] == "market_report.json" and "\n".join(state[3:]) + "\n" == b["body"]


def test_scratch_and_other_writable_paths_are_bounded(h2_probe):
    for cond in CONDITIONS:
        lim = obs(h2_probe, cond, "limits")
        assert lim["scratch_ok_small"]["exit"] == 0 and lim["scratch_ok_small"]["stdout"].strip() == str(4 << 20)
        for firm in ("A", "B"):
            sc = lim[firm]["scratch"]
            assert sc["exit"] != 0 and "No space left" in sc["stderr"], sc
            assert int(sc["stdout"].split()[0]) <= sandbox_env.SCRATCH_BYTES
            for k in ("home", "tmp", "shm"):
                assert lim[firm][k]["exit"] != 0 and "No space left" in lim[firm][k]["stderr"], (firm, k)


def test_root_filesystem_is_read_only(h2_probe):
    for cond in CONDITIONS:
        rf = obs(h2_probe, cond, "rootfs")
        for firm in ("A", "B"):
            for r in rf["agent"][firm]:
                assert r["exit"] != 0 and ("Read-only file system" in r["stderr"] or "Permission denied" in r["stderr"])
            assert any("Read-only file system" in r["stderr"] for r in rf["agent"][firm])
        assert all(r["exit"] != 0 and "Read-only file system" in r["stderr"] for r in rf["root"])


def test_pipefail(h2_probe):
    p = obs(h2_probe, "SHR", "pipefail")
    assert p["false_true"]["exit"] == 1 and p["true_true"]["exit"] == 0
    assert re.search(r"pipefail\s+on", p["opt"]["stdout"])


def test_timed_out_command_runs_exactly_once(h2_probe):
    for cond in CONDITIONS:
        t = obs(h2_probe, cond, "timeout_once")
        assert t["result"]["timed_out"] is True and t["result"]["exit"] == 124
        assert t["count"] == "1", t
        assert t["result"]["secs"] < 15


def test_no_agent_process_outlives_its_command(h2_probe):
    s = obs(h2_probe, "SHR", "stragglers")
    assert s["unreaped"]["exit"] == 0 and int(s["before_reap"]) >= 1      # setsid escape survives without reaping
    assert s["reaped"]["reaped"] is True and s["after_reap"] == "0"
    assert s["timed"]["timed_out"] is True and s["after_timeout"] == "0"


def test_output_is_truncated_with_marker(h2_probe):
    r = obs(h2_probe, "SHR", "truncation")
    assert r["truncated"] is True and r["stdout"].startswith("x" * shell.MAX_OUTPUT)
    assert r["stdout"].endswith("[... truncated 3000 chars]")
    assert len(r["output"]) < shell.MAX_OUTPUT + 40 and len(r["raw_stdout"]) == 5000


def test_root_write_is_symlink_safe(h2_probe):
    for cond in CONDITIONS:
        w = obs(h2_probe, cond, "root_write_safety")
        lines = w["state"].splitlines()
        assert lines[0] == "regular file 1002 600" and lines[1] == "INBOX-CONTENT"
        assert w["parent_error"] is not None and "canonical" in w["parent_error"]
        assert "evil.txt" not in w["state"] and lines[2] == "market_report.json"
        assert lines[3].startswith('{"round": 1')                       # bulletin untouched through the symlink
