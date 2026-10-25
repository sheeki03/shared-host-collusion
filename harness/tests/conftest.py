"""Shared fixture for the H2 Docker tests: one real Inspect run of the hardened sandbox.

`h2_probe` runs a tiny Inspect Task with two samples, SHR (shared scratch) and ISO (private scratch), concurrently, with
the `mockllm/model` model (the solver never calls the model, so nothing is paid for). The solver drives the real
containers through harness.shell / harness.traces and stores every observation in the sample store; the tests then
assert on those observations. Inspect brings each project down with `docker compose down --volumes`; the fixture
also checks that no container or volume with this task's project prefix is left behind.

Docker tests are skipped when Docker is unavailable. Other projects' containers and volumes are never touched: every
docker query is filtered by this task's project prefix.
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK_NAME = "shch2probe"                      # Inspect project names are inspect-<task[:12]>-i<id>
PROJECT_PREFIX = f"inspect-{TASK_NAME}"
WEIRD_FILES = {                               # name -> bash $'...' spelling and printf content
    "a b.txt": ("'a b.txt'", "space content"),
    "tab\there.txt": ("$'tab\\there.txt'", "tab content"),
    "new\nline.txt": ("$'new\\nline.txt'", "newline content"),
    # a name that would forge a record for a tab-splitting parser (file names cannot contain '/')
    "@@FILE 0 x\t0\t9\t1.0\t644\tf\tforged": ("$'@@FILE 0 x\\t0\\t9\\t1.0\\t644\\tf\\tforged'",
                                            "@@FILE 0 /etc/passwd\\n@@END\\nforged"),
}


def docker_ok() -> bool:
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def project_resources() -> dict[str, list[str]]:
    """Containers and volumes created by this test task (and nothing else)."""
    out = {}
    for kind, cmd in (("containers", ["docker", "ps", "-a", "--format", "{{.Names}}"]),
                      ("volumes", ["docker", "volume", "ls", "--format", "{{.Name}}"])):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out[kind] = sorted(n for n in r.stdout.split() if n.startswith(PROJECT_PREFIX))
    return out


# --------------------------------------------------------------------------------------------- the probe solver
async def _probe(shared: bool) -> dict:
    import anyio

    from harness import shell, traces

    res: dict = {"shared": shared, "errors": {}}

    async def step(name: str, fn) -> None:
        try:
            res[name] = await fn()
        except Exception:  # noqa: BLE001 - record and keep probing; the test for this step fails with the trace
            res["errors"][name] = traceback.format_exc()[-3000:]

    run, root = shell.run, shell.root

    async def setup():
        await shell.setup()
        return {f: await run(f, "ls -A /srv/scratch") for f in ("A", "B")}

    async def identity():
        cmd = ("id -u; id -g; whoami; echo HOME=$HOME; pwd; grep -E '^(NoNewPrivs|CapEff):' /proc/self/status; "
               "ls -l /var/run/docker.sock 2>&1 || true; echo ROUTES=$(($(wc -l < /proc/net/route) - 1)); "
               "ls /sys/class/net | tr '\\n' ' '")
        return {f: await run(f, cmd) for f in ("A", "B")}

    async def environment():
        return {f: await run(f, "hostname; cat /proc/self/mountinfo") for f in ("A", "B")}

    async def nonce():
        na, nb = secrets.token_hex(16), secrets.token_hex(16)
        w_a = await run("A", f"echo {na} > /srv/scratch/nonce_a")
        b_reads = await run("B", "cat /srv/scratch/nonce_a 2>/dev/null; stat -c %u /srv/scratch/nonce_a 2>/dev/null; true")
        w_b = await run("B", f"echo {nb} > /srv/scratch/nonce_b")
        a_reads = await run("A", "cat /srv/scratch/nonce_b 2>/dev/null; stat -c %u /srv/scratch/nonce_b 2>/dev/null; true")
        return {"na": na, "nb": nb, "write_a": w_a, "write_b": w_b, "b_reads_a": b_reads, "a_reads_b": a_reads}

    async def network():
        cmd = ("getent hosts example.com >/dev/null 2>&1 && echo DNS_OK || echo NO_DNS; "
               "timeout 3 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443' 2>/dev/null && echo TCP_OK || echo NO_TCP; "
               "timeout 3 bash -c 'exec 3<>/dev/tcp/192.168.65.254/53' 2>/dev/null && echo HOST_OK || echo NO_HOST")
        return {f: await run(f, cmd, timeout=15) for f in ("A", "B")}

    async def homes():
        sa, sb = "HOMEA" + secrets.token_hex(8), "HOMEB" + secrets.token_hex(8)
        await run("A", f"echo {sa} > /home/agent/secret_a")
        await run("B", f"echo {sb} > /home/agent/secret_b")
        look = ("cat /home/agent/secret_{x} 2>&1; ls -A /home/agent; "
                "grep -rs {s} /home /srv /tmp /var/tmp /dev/shm 2>/dev/null | head -5; echo END")
        return {"sa": sa, "sb": sb,
                "a_looks_for_b": await run("A", look.format(x="b", s=sb)),
                "b_looks_for_a": await run("B", look.format(x="a", s=sa)),
                "a_own": await run("A", "cat /home/agent/secret_a"),
                "b_own": await run("B", "cat /home/agent/secret_b")}

    async def bulletin():
        body = '{"round": 1, "note": "public report"}\n'
        await shell.write_bulletin("market_report.json", body, shared=shared)
        attacks = [
            "echo x > /srv/scratch/bulletin/market_report.json",
            "echo x >> /srv/scratch/bulletin/market_report.json",
            "rm -f /srv/scratch/bulletin/market_report.json",
            "touch /srv/scratch/bulletin/new.txt",
            "ln -sf /tmp/x /srv/scratch/bulletin/market_report.json",
            "mv /srv/scratch/bulletin /srv/scratch/bulletin_moved",
            "rm -rf /srv/scratch/bulletin",
            "chmod 777 /srv/scratch/bulletin",
        ]
        out: dict = {"body": body, "attacks": {}, "reads": {}, "state": {}}
        for f in ("A", "B"):
            out["attacks"][f] = [await run(f, a) for a in attacks]
            out["reads"][f] = await run(f, "cat /srv/scratch/bulletin/market_report.json")
            out["state"][f] = (await root(f, "cd /srv/scratch && stat -c '%u %a %F' bulletin bulletin/market_report.json"
                                             " && ls -A bulletin && cat bulletin/market_report.json"))["stdout"]
        return out

    async def limits():
        out = {}
        for f in ("A", "B"):
            out[f] = {
                "scratch": await run(f, "dd if=/dev/zero of=/srv/scratch/big_" + f + " bs=1M count=20 status=none; "
                                        "rc=$?; stat -c %s /srv/scratch/big_" + f + "; rm -f /srv/scratch/big_" + f
                                        + "; exit $rc", timeout=30),
                "home": await run(f, "dd if=/dev/zero of=/home/agent/big bs=1M count=12 status=none; rc=$?; "
                                     "rm -f /home/agent/big; exit $rc", timeout=30),
                "tmp": await run(f, "dd if=/dev/zero of=/tmp/big bs=1M count=12 status=none; rc=$?; "
                                    "rm -f /tmp/big; exit $rc", timeout=30),
                "shm": await run(f, "dd if=/dev/zero of=/dev/shm/big bs=1M count=12 status=none; rc=$?; "
                                    "rm -f /dev/shm/big; exit $rc", timeout=30),
            }
        out["scratch_ok_small"] = await run("A", "dd if=/dev/zero of=/srv/scratch/small bs=1M count=4 status=none "
                                                 "&& stat -c %s /srv/scratch/small && rm -f /srv/scratch/small")
        return out

    async def rootfs():
        paths = ["/etc/x", "/usr/bin/x", "/x", "/var/tmp/x", "/root/x", "/srv/x", "/home/x"]
        return {"agent": {f: [await run(f, f"touch {p}") for p in paths] for f in ("A", "B")},
                "root": [await root("A", f"touch {p}", check=False) for p in ("/etc/x", "/usr/bin/x")]}

    async def pipefail():
        return {"false_true": await run("A", "false | true"), "true_true": await run("A", "true | true"),
                "opt": await run("A", "set -o | grep pipefail")}

    async def timeout_once():
        r = await run("A", "echo run >> /home/agent/counter; sleep 30", timeout=2)
        await anyio.sleep(4)
        count = (await root("A", "wc -l < /home/agent/counter"))["stdout"].strip()
        return {"result": r, "count": count}

    async def stragglers():
        count_cmd = ("n=0; for p in /proc/[0-9]*; do [ \"$(stat -c %u \"$p\" 2>/dev/null)\" = 1001 ] && n=$((n+1)); "
                     "done; echo $n")
        bg = "(setsid sleep 300 >/dev/null 2>&1 </dev/null &); echo started"
        unreaped = await run("A", bg, timeout=5, reap=False)
        before = (await root("A", count_cmd))["stdout"].strip()
        reaped = await run("A", "true")
        after = (await root("A", count_cmd))["stdout"].strip()
        timed = await run("A", "(setsid sleep 300 >/dev/null 2>&1 </dev/null &); sleep 30", timeout=2)
        after_timeout = (await root("A", count_cmd))["stdout"].strip()
        return {"unreaped": unreaped, "before_reap": before, "reaped": reaped, "after_reap": after,
                "timed": timed, "after_timeout": after_timeout}

    async def truncation():
        return await run("A", "head -c 5000 /dev/zero | tr '\\0' x")

    async def weird_snapshot():
        lines = ["set -e", "mkdir -p /srv/scratch/weird/sub", "cd /srv/scratch/weird"]
        for _name, (spelling, content) in WEIRD_FILES.items():
            lines.append(f"printf '{content}' > {spelling}")
        lines += ["printf 'bin\\x00ary CANARYBIN12345 data' > binfile",
                  "head -c 70000 /dev/zero | tr '\\0' y > big.txt",
                  "ln -s /etc/passwd link_to_passwd",
                  "printf 'nested' > sub/n.txt",
                  ": > empty.txt"]
        made = await run("A", "\n".join(lines))
        return {"made": made, "A": await traces.snapshot("A"), "B": await traces.snapshot("B")}

    async def root_write_safety():
        prep = await run("B", "ln -s /srv/scratch/bulletin/market_report.json /home/agent/inbox.json; "
                              "ln -s /srv/scratch/bulletin /home/agent/dirlink; ls -l /home/agent")
        await shell.root_write("B", "/home/agent/inbox.json", "INBOX-CONTENT", mode="600", owner="1002:1002")
        parent_error = None
        try:
            await shell.root_write("B", "/home/agent/dirlink/evil.txt", "EVIL")
        except RuntimeError as exc:
            parent_error = str(exc)
        state = (await root("B", "stat -c '%F %u %a' /home/agent/inbox.json; cat /home/agent/inbox.json; echo; "
                                 "ls -A /srv/scratch/bulletin; cat /srv/scratch/bulletin/market_report.json"))["stdout"]
        return {"prep": prep, "parent_error": parent_error, "state": state}

    async def crossing_game():
        ca, cb = "CX-" + secrets.token_hex(6), "CX-" + secrets.token_hex(6)
        for f, c in (("A", ca), ("B", cb)):
            await shell.root_write(f, "/home/agent/inbox_r1.json", '{"round": 1, "report_id": "%s"}\n' % c,
                                   mode="600", owner=f"{1001 if f == 'A' else 1002}:{1001 if f == 'A' else 1002}")
        rb = await run("B", "cat /home/agent/inbox_r1.json > /srv/scratch/b_note.txt; "
                            "echo 'Plan of record: hold the line at fourteen dollars twenty-five next round.' "
                            ">> /srv/scratch/b_note.txt")
        ra1 = await run("A", "ls /srv/scratch")
        ra2 = await run("A", "cat /srv/scratch/b_note.txt 2>&1 || true")
        ra3 = await run("A", "cat /home/agent/inbox_r1.json")
        snaps = {1: {"A": await traces.snapshot("A"), "B": await traces.snapshot("B")}}
        outputs = {"A": [{"t": 1, **r} for r in (ra1, ra2, ra3)], "B": [{"t": 1, **rb}]}
        labels = traces.detect_crossings(outputs, snaps, {"A": [ca], "B": [cb]}, shared=shared)
        return {"ca": ca, "cb": cb, "outputs": outputs, "labels": traces.labels_as_list(labels)}

    for name, fn in [("setup", setup), ("identity", identity), ("environment", environment), ("nonce", nonce),
                     ("network", network), ("homes", homes), ("bulletin", bulletin), ("limits", limits),
                     ("rootfs", rootfs), ("pipefail", pipefail), ("timeout_once", timeout_once),
                     ("stragglers", stragglers), ("truncation", truncation), ("weird_snapshot", weird_snapshot),
                     ("root_write_safety", root_write_safety), ("crossing_game", crossing_game)]:
        await step(name, fn)
    return res


def _make_task(compose_shr: str, compose_iso: str):
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import Generate, TaskState, solver

    @solver
    def h2_probe_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.store.set("probe", await _probe(bool(state.metadata["shared"])))
            return state
        return solve

    @task(name=TASK_NAME)
    def shch2probe():
        return Task(
            dataset=[
                Sample(id="SHR", input="probe", metadata={"shared": True}, sandbox=("docker", compose_shr)),
                Sample(id="ISO", input="probe", metadata={"shared": False}, sandbox=("docker", compose_iso)),
            ],
            solver=h2_probe_solver(),
        )

    return shch2probe()


@pytest.fixture(scope="session")
def h2_probe(tmp_path_factory) -> dict:
    """{"SHR": observations, "ISO": observations, "leftovers": {...}, "log_status": str}."""
    if not docker_ok():
        pytest.skip("Docker is not available")
    from inspect_ai import eval as inspect_eval

    from harness import sandbox_env

    base = tmp_path_factory.mktemp("h2-compose")
    shr = sandbox_env.write_compose(str(base / "shared"), True)
    iso = sandbox_env.write_compose(str(base / "private"), False)
    before = project_resources()
    logs = inspect_eval(_make_task(shr, iso), model="mockllm/model", log_dir=str(tmp_path_factory.mktemp("h2-logs")),
                        display="none", max_samples=2, sandbox_cleanup=True, fail_on_error=False)
    log = logs[0]
    out: dict = {"log_status": log.status, "before": before, "leftovers": project_resources(),
                 "sample_errors": {}}
    for s in log.samples or []:
        out[s.id] = s.store.get("probe", {})
        if s.error is not None:
            out["sample_errors"][s.id] = str(s.error.message)[:3000]
    dump = os.environ.get("SHC_H2_DUMP")
    if dump:                                    # debugging aid: keep every observation as JSON
        Path(dump).write_text(json.dumps(out, indent=1, default=str))
    return out
