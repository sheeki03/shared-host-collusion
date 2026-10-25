"""V2 infra verification: independent, adversarial checks of H1 (ledger, provider) and H2 (sandbox, traces, gate).

Written by the V2-infra-checker against harness/CONTRACT.md, separately from the builders' own tests. Nothing here
makes a paid call: the provider is driven through httpx.MockTransport and the Docker probe uses `mockllm/model`
without calling it.

Sections
1. Ledger: the cap holds across processes x threads with SIGKILLed writers; a missing or invalid cost is charged at
   est_max; an unwritable, non-regular (/dev/null), partially written or un-fsynced ledger fails closed.
2. Provider: raises before any HTTP call when the ledger env or file is missing or bad; the cap holds for 40
   concurrent generate() calls; a ledger that turns unwritable mid-call stops the run and keeps the reservation.
3. Gate: fails closed when the binary is missing, not executable or unpinned, and on command text that cannot be
   passed as argv (NUL, lone surrogate).
4. Snapshot parsing: hostile names and link targets round-trip through parse_find_output; garbage raises.
5. Docker probe (task `shcv2verify`, SHR and ISO concurrently): rival home, sticky scratch, bulletin, network,
   writable-path enumeration, scratch quota, sparse files, hostile names in a real snapshot, argv-invalid commands.
   Known open problems are recorded as strict xfails so they show up in every run until fixed.
"""
from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import random
import secrets
import signal
import subprocess
import sys
import time
import traceback
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import gate, traces  # noqa: E402
from harness.ledger import BudgetExceeded, BudgetLedger, LedgerError  # noqa: E402

IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def _money(x) -> Decimal:
    return Decimal(repr(x)) if isinstance(x, float) else Decimal(x)


def replay_ledger(path: Path, cap: float) -> dict:
    """Independent replay: verify persisted cap, audited changes, and every reservation's admission bound."""
    expected_cap = _money(cap)
    cap_d: Decimal | None = None
    ledger_id: str | None = None
    open_: dict[str, Decimal] = {}
    spent = Decimal(0)
    worst = Decimal(0)
    events = {"init": 0, "cap_change": 0, "reserve": 0, "settle": 0, "release": 0, "flag": 0, "torn": 0}
    for raw in path.read_bytes().split(b"\n"):
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except ValueError:
            events["torn"] += 1
            continue
        kind = ev["event"]
        events[kind] += 1
        if kind == "init":
            assert cap_d is None and ledger_id is None, "ledger initialized more than once"
            cap_d, ledger_id = _money(ev["cap"]), ev["id"]
            assert isinstance(ledger_id, str) and ledger_id
        elif kind == "cap_change":
            assert cap_d is not None and ev["id"] == ledger_id
            old, new = _money(ev["old_cap"]), _money(ev["cap"])
            assert old == cap_d and new > old and ev.get("reason"), "unaudited or invalid cap change"
            cap_d = new
        elif kind == "reserve":
            assert cap_d is not None, "reserve before initialization"
            open_[ev["id"]] = _money(ev["est_max"])
            total = spent + sum(open_.values(), Decimal(0))
            worst = max(worst, total)
            assert total <= cap_d, f"cap exceeded at reserve {ev['id']}: {total} > {cap_d}"
        elif kind in ("settle", "release"):
            est = open_.pop(ev["id"])
            if kind == "settle":
                actual = _money(ev["actual"])
                assert actual <= est, "a settle above est_max voids the conditional reservation guarantee"
                spent += actual
    assert cap_d == expected_cap, f"persisted cap {cap_d} != expected cap {expected_cap}"
    assert events["init"] == 1
    return {"spent": spent, "outstanding": sum(open_.values(), Decimal(0)), "worst": worst, "events": events,
            "open": len(open_)}


# =============================================================================================== 1. ledger
def _ledger_worker(path: str, cap: float, seed: int, threads: int) -> None:
    """Child process: `threads` threads, each with its own BudgetLedger object and event loop, reserving forever."""
    import threading

    def thread_main(k: int) -> None:
        rng = random.Random(seed * 100 + k)
        led = BudgetLedger(path, cap, create=False)

        async def loop() -> None:
            while True:
                est = rng.choice([0.001, 0.0025, 0.005, 0.01, 0.02])
                try:
                    tok = await led.reserve(est, {"model": "m", "sample_id": f"p{seed}t{k}", "call_kind": "x"})
                except BudgetExceeded:
                    await asyncio.sleep(0)
                    continue
                r = rng.random()
                if r < 0.15:
                    await led.release(tok)
                elif r < 0.3:
                    await led.settle(tok, None)                    # missing cost: charged at est_max
                elif r < 0.4:
                    pass                                           # left open (looks like an in-flight call)
                else:
                    await led.settle(tok, est * rng.choice([0.0, 0.3, 1.0]))

        asyncio.run(loop())

    ts = [threading.Thread(target=thread_main, args=(k,), daemon=True) for k in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def test_ledger_cap_holds_across_processes_threads_and_sigkill(tmp_path):
    path, cap = tmp_path / "ledger.jsonl", 0.37
    BudgetLedger(path, cap)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_ledger_worker, args=(str(path), cap, s, 2), daemon=True) for s in range(3)]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and BudgetLedger(path, cap, create=False).remaining > 0.02:
        time.sleep(0.2)
    time.sleep(1.0)                                   # let the workers contend at the cap for a while
    for p in procs:
        os.kill(p.pid, signal.SIGKILL)                # writers die without cleanup, possibly mid-call
    for p in procs:
        p.join(10)
    fresh = BudgetLedger(path, cap, create=False)     # rebuild from the file after the crash
    rep = replay_ledger(path, cap)                    # asserts spent + outstanding <= cap at every reserve
    assert rep["events"]["reserve"] > 20 and rep["events"]["flag"] >= 1
    assert rep["worst"] > _money(cap) - _money(0.02), "the cap was never binding; the test proved nothing"
    assert fresh.spent == pytest.approx(float(rep["spent"])) and fresh.outstanding == pytest.approx(
        float(rep["outstanding"]))
    assert fresh.remaining >= 0
    with pytest.raises(BudgetExceeded):               # orphaned reservations still count against the cap
        asyncio.run(fresh.reserve(fresh.remaining + 0.001))


@pytest.mark.parametrize("bad_cost", [None, float("nan"), float("inf"), -0.01, "abc", True])
def test_ledger_missing_or_invalid_cost_is_charged_at_est_max(tmp_path, bad_cost):
    led = BudgetLedger(tmp_path / "l.jsonl", 1.0)

    async def go() -> float:
        tok = await led.reserve(0.05, {"model": "m"})
        return await led.settle(tok, bad_cost)

    assert asyncio.run(go()) == 0.05
    assert led.spent == pytest.approx(0.05) and led.outstanding == 0 and led.flags == 1
    lines = [json.loads(x) for x in (tmp_path / "l.jsonl").read_bytes().splitlines()]
    assert [e["event"] for e in lines] == ["init", "reserve", "settle", "flag"] and lines[3]["actual"] == 0.05


@pytest.mark.skipif(IS_ROOT, reason="file permissions do not bind root")
def test_ledger_unwritable_file_fails_closed(tmp_path):
    path = tmp_path / "l.jsonl"
    led = BudgetLedger(path, 1.0)
    tok = asyncio.run(led.reserve(0.1))
    path.chmod(0o444)
    try:
        with pytest.raises(LedgerError):
            BudgetLedger(path, 1.0, create=False)
        with pytest.raises(LedgerError):
            asyncio.run(led.reserve(0.1))
        with pytest.raises(LedgerError):
            asyncio.run(led.settle(tok, 0.01))
        with pytest.raises(LedgerError):
            _ = led.remaining
    finally:
        path.chmod(0o644)
    again = BudgetLedger(path, 1.0, create=False)
    assert again.outstanding == pytest.approx(0.1) and again.spent == 0     # nothing was lost or double-counted


@pytest.mark.parametrize("target", ["/dev/null", "dir"])
def test_ledger_refuses_non_regular_files(tmp_path, target):
    """Fixed by V2: /dev/null used to be accepted, so appends vanished and every reserve was admitted (fail-open)."""
    path = "/dev/null" if target == "/dev/null" else str(tmp_path)
    if target == "/dev/null" and not os.path.exists(path):
        pytest.skip("no /dev/null")
    with pytest.raises(LedgerError):
        led = BudgetLedger(path, 0.05, create=False)
        for _ in range(10):
            asyncio.run(led.reserve(0.01))


def test_ledger_partial_write_fails_closed_and_is_repaired(tmp_path, monkeypatch):
    from harness import ledger as ledger_mod

    path = tmp_path / "l.jsonl"
    led = BudgetLedger(path, 1.0)
    real_write = os.write

    def half_then_enospc(fd, data):
        real_write(fd, bytes(data[: len(data) // 2]))
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ledger_mod.os, "write", half_then_enospc)
    with pytest.raises(LedgerError):
        asyncio.run(led.reserve(0.3))                       # no token: the caller must not make the call
    monkeypatch.setattr(ledger_mod.os, "write", real_write)
    other = BudgetLedger(path, 1.0, create=False)
    assert other.outstanding == 0                           # a torn reserve never counts and never returned a token
    tok = asyncio.run(other.reserve(0.2))                   # the next exclusive writer closes off the torn line
    asyncio.run(other.settle(tok, 0.1))
    rep = replay_ledger(path, 1.0)
    assert rep["events"]["torn"] == 1 and rep["events"]["flag"] == 1 and rep["spent"] == Decimal("0.1")


def test_ledger_fsync_failure_keeps_the_reservation_counted(tmp_path, monkeypatch):
    from harness import ledger as ledger_mod

    path = tmp_path / "l.jsonl"
    led = BudgetLedger(path, 1.0)

    def bad_fsync(fd):
        raise OSError(5, "I/O error")

    monkeypatch.setattr(ledger_mod.os, "fsync", bad_fsync)
    with pytest.raises(LedgerError):
        asyncio.run(led.reserve(0.25))
    monkeypatch.undo()
    # The line landed, so the reservation counts against the cap even though no token was handed out (conservative).
    assert BudgetLedger(path, 1.0, create=False).outstanding == pytest.approx(0.25)


# =============================================================================================== 2. provider
MODEL = "z-ai/glm-5.2"


def _ok_body(cost: float | None) -> dict:
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "x", "model": MODEL, "usage": usage,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}


class _Rec:
    def __init__(self, fn=None):
        self.requests: list[httpx.Request] = []
        self.fn = fn

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fn is not None:
            return self.fn(request)
        return httpx.Response(200, json=_ok_body(1e-6))


def _model(rec, key: str, **kw):
    import harness.nanogpt_provider  # noqa: F401  (registers "nanogpt")
    from inspect_ai.model import get_model

    return get_model(f"nanogpt/{MODEL}", memoize=False, transport=httpx.MockTransport(rec), key_path=key,
                     backoff_s=0.0, **kw)


@pytest.fixture
def keyfile(tmp_path) -> str:
    p = tmp_path / "key"
    p.write_text("sk-verify-not-a-real-key\n")
    return str(p)


def _bad_env(case: str, tmp_path: Path, monkeypatch) -> None:
    good = tmp_path / "ledger.jsonl"
    BudgetLedger(good, 2.0)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(good))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", "2.0")
    if case == "no_path":
        monkeypatch.delenv("SHC_LEDGER_PATH")
    elif case == "no_cap":
        monkeypatch.delenv("SHC_BUDGET_CAP_USD")
    elif case == "empty_path":
        monkeypatch.setenv("SHC_LEDGER_PATH", "")
    elif case == "empty_cap":
        monkeypatch.setenv("SHC_BUDGET_CAP_USD", "")
    elif case == "missing_file":
        monkeypatch.setenv("SHC_LEDGER_PATH", str(tmp_path / "typo.jsonl"))
    elif case == "directory":
        monkeypatch.setenv("SHC_LEDGER_PATH", str(tmp_path))
    elif case == "dev_null":
        monkeypatch.setenv("SHC_LEDGER_PATH", "/dev/null")
    elif case == "unreadable":
        good.chmod(0o000)
    elif case == "corrupt":
        good.write_bytes(b'{"event": "settle", "id": "nope", "actual": 0.0}\n')
    elif case.startswith("cap="):
        monkeypatch.setenv("SHC_BUDGET_CAP_USD", case[4:])
    else:
        raise AssertionError(case)


BAD_ENV_CASES = ["no_path", "no_cap", "empty_path", "empty_cap", "missing_file", "directory", "dev_null", "corrupt",
                 "cap=abc", "cap=nan", "cap=inf", "cap=-1"] + ([] if IS_ROOT else ["unreadable"])


@pytest.mark.parametrize("case", BAD_ENV_CASES)
def test_provider_raises_before_any_http_call_when_ledger_env_is_bad(case, tmp_path, monkeypatch, keyfile):
    _bad_env(case, tmp_path, monkeypatch)
    rec = _Rec()
    try:
        with pytest.raises(Exception):  # noqa: B017 - any error is acceptable as long as nothing was sent
            m = _model(rec, keyfile)
            from inspect_ai.model import ChatMessageUser, GenerateConfig
            asyncio.run(m.generate([ChatMessageUser(content="hi")], config=GenerateConfig(max_tokens=5)))
    finally:
        (tmp_path / "ledger.jsonl").chmod(0o644)
    assert rec.requests == []


@pytest.mark.skipif(IS_ROOT, reason="file permissions do not bind root")
@pytest.mark.parametrize("damage", ["delete", "read_only"])
def test_provider_ledger_damaged_after_construction_sends_nothing(damage, tmp_path, monkeypatch, keyfile):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    _bad_env("cap=2.0", tmp_path, monkeypatch)            # a valid env and ledger
    rec = _Rec()
    m = _model(rec, keyfile)
    ledger = tmp_path / "ledger.jsonl"
    if damage == "delete":
        ledger.unlink()
    else:
        ledger.chmod(0o444)
    try:
        with pytest.raises(LedgerError):
            asyncio.run(m.generate([ChatMessageUser(content="hi")], config=GenerateConfig(max_tokens=5)))
    finally:
        if ledger.exists():
            ledger.chmod(0o644)
    assert rec.requests == []


@pytest.mark.skipif(IS_ROOT, reason="file permissions do not bind root")
def test_provider_ledger_unwritable_mid_call_stops_and_keeps_reservation(tmp_path, monkeypatch, keyfile):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    _bad_env("cap=2.0", tmp_path, monkeypatch)
    ledger = tmp_path / "ledger.jsonl"

    def lock_ledger_then_answer(request: httpx.Request) -> httpx.Response:
        ledger.chmod(0o444)                               # the settle after this response cannot be written
        return httpx.Response(500, json={"error": "boom"})

    rec = _Rec(lock_ledger_then_answer)
    m = _model(rec, keyfile)
    try:
        with pytest.raises(LedgerError):
            asyncio.run(m.generate([ChatMessageUser(content="hi")], config=GenerateConfig(max_tokens=5)))
    finally:
        ledger.chmod(0o644)
    assert len(rec.requests) == 1                         # no retry once the ledger cannot record the attempt
    fresh = BudgetLedger(ledger, 2.0, create=False)
    assert len(fresh.open_reservations()) == 1 and fresh.outstanding > 0 and fresh.spent == 0


def test_provider_cap_holds_under_40_concurrent_calls(tmp_path, monkeypatch, keyfile):
    from inspect_ai.model import ChatMessageUser, GenerateConfig

    from harness.nanogpt_provider import estimate_max_cost, load_prices

    msgs = [ChatMessageUser(content="concurrency probe")]
    wire = [{"role": "user", "content": "concurrency probe"}]
    est = estimate_max_cost(load_prices()[MODEL], len(json.dumps(wire, ensure_ascii=False).encode("utf-8")), 50,
                            len(wire))
    cap = est * 7.5                                        # admits exactly 7 attempts when each settles at est_max
    path = tmp_path / "ledger.jsonl"
    BudgetLedger(path, cap)
    monkeypatch.setenv("SHC_LEDGER_PATH", str(path))
    monkeypatch.setenv("SHC_BUDGET_CAP_USD", repr(cap))

    requests: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        await asyncio.sleep(0.02)                         # keep many calls in flight at once
        body = _ok_body(None)
        body["usage"] = {}                                # no cost and no token counts -> settled at est_max
        return httpx.Response(200, json=body)

    import harness.nanogpt_provider  # noqa: F401
    from inspect_ai.model import get_model

    m = get_model(f"nanogpt/{MODEL}", memoize=False, transport=httpx.MockTransport(handler), key_path=keyfile,
                  backoff_s=0.0, retries=0)

    async def go():
        return await asyncio.gather(*(m.generate(msgs, config=GenerateConfig(max_tokens=50)) for _ in range(40)),
                                    return_exceptions=True)

    results = asyncio.run(go())
    ok = [r for r in results if not isinstance(r, BaseException)]
    refused = [r for r in results if isinstance(r, BudgetExceeded)]
    other = [r for r in results if isinstance(r, BaseException) and not isinstance(r, BudgetExceeded)]
    assert not other, other[:3]
    assert len(ok) == 7 and len(refused) == 33 and len(requests) == 7
    assert all(r.metadata["cost_source"] == "est_max" for r in ok)
    led = BudgetLedger(path, cap, create=False)
    assert led.spent <= cap and led.outstanding == 0 and led.flags == 7
    replay_ledger(path, cap)


# =============================================================================================== 3. gate
def _fake_tirith(tmp_path: Path, *, executable: bool = True, pin: bool = True) -> Path:
    b = tmp_path / "tirith-fake"
    b.write_text('#!/bin/sh\necho \'{"action": "allow", "findings": [], "policy_path_used": null}\'\n')
    b.chmod(0o755 if executable else 0o644)
    if pin:
        (tmp_path / "tirith-fake.sha256").write_text(f"{gate.binary_sha256(b)}  tirith-fake\n")
    return b


@pytest.fixture
def fresh_gate_cache():
    gate._self_test_cache.clear()
    yield
    gate._self_test_cache.clear()


@pytest.mark.parametrize("policy", sorted(gate.POLICIES))
def test_gate_fails_closed_when_binary_is_missing(policy, tmp_path, fresh_gate_cache):
    missing = tmp_path / "no-such-tirith"
    for cmd in ("echo hi", "ls /home/agent"):            # commands every policy would otherwise allow
        r = gate.check(cmd, policy, binary=missing)
        assert r["action"] == "block" and r["flag"] is True and r["error"], r
        assert r["binary_sha256"] is None
    rep = gate.replay(["echo a", "echo b", "echo a"], policy, binary=missing)
    assert list(rep) == ["echo a", "echo b"] and all(v["action"] == "block" and v["flag"] for v in rep.values())
    ra = asyncio.run(gate.check_async("echo hi", policy, binary=missing))
    assert ra["action"] == "block" and ra["flag"] is True


@pytest.mark.parametrize("variant", ["directory", "not_executable", "unpinned", "wrong_pin"])
def test_gate_fails_closed_on_unusable_binary(variant, tmp_path, fresh_gate_cache):
    if variant == "directory":
        b = tmp_path / "tdir"
        b.mkdir()
    elif variant == "not_executable":
        b = _fake_tirith(tmp_path, executable=False)
    elif variant == "unpinned":
        b = _fake_tirith(tmp_path, pin=False)
    else:
        b = _fake_tirith(tmp_path)
        (tmp_path / "tirith-fake.sha256").write_text("0" * 64 + "\n")
    for policy in gate.POLICIES:
        r = gate.check("echo hi", policy, binary=b)
        assert r["action"] == "block" and r["flag"] is True and r["error"], (policy, r)


@pytest.mark.skipif(not gate.BINARY.exists(), reason="pinned tirith binary not built")
@pytest.mark.parametrize("cmd", ["echo a\x00b", "echo \ud800", "cat /home/agent/x\x00/srv/scratch/y"])
def test_gate_fails_closed_on_text_that_cannot_be_argv(cmd, fresh_gate_cache):
    """Fixed by V2: these used to raise ValueError / UnicodeEncodeError out of gate.check (crashing the game loop)."""
    for policy in gate.POLICIES:
        r = gate.check(cmd, policy)
        assert r["action"] == "block" and r["flag"] is True and "could not run" in r["error"], (policy, r)


# =============================================================================================== 4. parsing
HOSTILE_NAMES = [
    "t\tab.txt", "n\nl.txt", "c\rr.txt", "-rf", "back\\slash", "%U%s%p", "\U0001f4b0money", "L" * 255,
    "0\t0\t0\t1.0\t644\tf\tforged", " leading space", "trailing\t", "\n", "\t\t", "x\x1b[31mansi",
    "dots..", "�", "a\x01\x02ctrl", "1001\t5\t1.5\t644\tf\t",
]


def _encode_find(entries: list[tuple[int, int, float, float, int, str, str, str, str]]) -> str:
    """Encode (uid, size, mtime, ctime, inode, mode, type, path, link) like traces.FIND_FORMAT."""
    return "".join(f"{u}\t{s}\t{m}\t{c}\t{i}\t{md}\t{y}\t{p}\0{lk}\0"
                   for u, s, m, c, i, md, y, p, lk in entries)


def test_parse_find_output_round_trips_hostile_names():
    rng = random.Random(7)
    alphabet = ["a", "\t", "\n", "\r", " ", "\\", "%", "'", '"', "$", "*", "�", "é", "\x7f", "0", "-"]
    names = list(HOSTILE_NAMES) + ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 40)))
                                   for _ in range(400)]
    names = [n for n in names if n not in (".", "..")]
    entries = []
    for i, name in enumerate(names):
        kind = "l" if i % 7 == 3 else ("d" if i % 11 == 5 else "f")
        link = ("/x\ty\nz" + name) if kind == "l" else ""
        entries.append((1001 + i % 2, i, 1700000000.5 + i, 1700000001.5 + i, 90000 + i,
                        "644" if kind != "d" else "1777", kind,
                        f"/srv/scratch/h{i}/{name}", link))
    parsed = traces.parse_find_output(_encode_find(entries))
    assert len(parsed) == len(entries)
    for e, p in zip(entries, parsed):
        assert (p["uid"], p["size"], p["mtime"], p["ctime"], p["inode"], p["mode"], p["type"], p["path"]) == e[:8]
        assert p["link"] == (e[8] if e[6] == "l" else None)
    assert not any(p["path"] == "/etc/passwd" for p in parsed)


@pytest.mark.parametrize("raw", [
    "1001\t1\t1.0\t644\tf\t/srv/scratch/x\0",                 # odd number of fields (link field missing)
    "1001\t1\t1.0\t644\t/srv/scratch/x\0\0",                  # too few header fields
    "u\t1\t1.0\t644\tf\t/srv/scratch/x\0\0",                  # non-numeric uid
    "1001\t1\t1.0\t644\tf\trelative/x\0\0",                   # path not absolute
    "1001\t1\t1.0\trwx\tf\t/srv/scratch/x\0\0",               # mode not octal
    "1001\t1\t1.0\t644\tff\t/srv/scratch/x\0\0",              # type not one letter
    "1001\t1\t1.0\tbad\t10\t644\tf\t/srv/scratch/x\0\0",      # invalid ctime
    "1001\t1\t1.0\t2.0\tbad\t644\tf\t/srv/scratch/x\0\0",      # invalid inode
])
def test_parse_find_output_rejects_garbage(raw):
    with pytest.raises(ValueError):
        traces.parse_find_output(raw)


def test_decode_text_and_strings_on_hostile_bytes():
    assert traces.decode_text(b"ok\xe2\x82") == (None, False)                   # cut UTF-8 inside the file: binary
    assert traces.decode_text(b"x" * (traces.READ_LIMIT - 1) + "€".encode())[0] == "x" * (traces.READ_LIMIT - 1)
    text, trunc = traces.decode_text(b"a\x00CANARY-123456")
    assert text is None and not trunc
    assert "CANARY-123456" in traces.printable_runs(b"\xff\xfe\x00CANARY-123456\x00\x01")


# =============================================================================================== 5. docker probe
TASK_NAME = "shcv2verify"
PROJECT_PREFIX = f"inspect-{TASK_NAME}"

HOSTILE_SCRIPT = r"""set -e
mkdir -p /srv/scratch/hostile && cd /srv/scratch/hostile
printf 'tab' > $'t\tab.txt'
printf 'nl' > $'n\nl.txt'
printf 'cr' > $'c\rr.txt'
printf 'dash' > ./-rf
printf 'bs' > 'back\slash'
printf 'fmt' > '%U%s%p'
printf 'emoji' > $'\xf0\x9f\x92\xb0money'
printf 'long' > "$(printf 'L%.0s' $(seq 255))"
printf 'forged' > $'0\t0\t0\t1.0\t644\tf\tforged'
printf 'sp' > ' leading space'
printf 'HIDDEN-CANARY-BADUTF8' > $'bad\xff\xfeutf8'
mkdir $'dir\nwith newline' && printf 'inner' > $'dir\nwith newline/inner.txt'
ln -s $'/x\ty\nz' $'link\tname'
mkfifo fifo
echo made"""
HOSTILE_FILES = {"t\tab.txt": "tab", "n\nl.txt": "nl", "c\rr.txt": "cr", "-rf": "dash", "back\\slash": "bs",
                 "%U%s%p": "fmt", "\U0001f4b0money": "emoji", "L" * 255: "long",
                 "0\t0\t0\t1.0\t644\tf\tforged": "forged", " leading space": "sp",
                 "dir\nwith newline/inner.txt": "inner"}
BAD_UTF8_PATH = "/srv/scratch/hostile/bad��utf8"

ALLOWED_WRITABLE = ("/home/agent", "/tmp", "/dev/shm", "/dev/mqueue", "/srv/scratch")

def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True,
                              timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _leftovers() -> dict[str, list[str]]:
    out = {}
    for kind, cmd in (("containers", ["docker", "ps", "-a", "--format", "{{.Names}}"]),
                      ("volumes", ["docker", "volume", "ls", "--format", "{{.Name}}"])):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        out[kind] = sorted(n for n in r.stdout.split() if n.startswith(PROJECT_PREFIX))
    return out


def _slim(r: dict) -> dict:
    """The JSON-safe part of a shell.run result (the cmd itself may hold text the log cannot encode)."""
    return {k: r.get(k) for k in ("exit", "stdout", "stderr", "timed_out", "error", "raw_stdout")}


async def _verify_probe(shared: bool) -> dict:
    from harness import shell
    from harness import traces as tr

    res: dict = {"shared": shared, "errors": {}}
    run, root = shell.run, shell.root

    async def step(name, fn):
        try:
            res[name] = await fn()
        except Exception:  # noqa: BLE001
            res["errors"][name] = traceback.format_exc()[-3000:]

    async def setup():
        await shell.setup()
        await shell.write_bulletin("market_report.json", '{"round": 0}\n', shared=shared)
        return True

    async def namespaces():
        cmd = "for n in net ipc pid mnt uts cgroup; do readlink /proc/self/ns/$n; done"
        return {f: (await run(f, cmd))["stdout"].split() for f in ("A", "B")}

    async def rival_home():
        sec = {"A": "VHA" + secrets.token_hex(8), "B": "VHB" + secrets.token_hex(8)}
        out: dict = {"sec": sec}
        for f in ("A", "B"):
            await run(f, f"echo {sec[f]} > /home/agent/vsecret_{f}")
            # the owner tries to export its home through scratch: a symlink (resolved in the reader's mount
            # namespace) and a hard link (another filesystem, so EXDEV)
            out[f"export_{f}"] = _slim(await run(f, f"ln -s /home/agent/vsecret_{f} /srv/scratch/link_{f}; "
                                                    f"ln /home/agent/vsecret_{f} /srv/scratch/hard_{f}"))
        for f, r in (("A", "B"), ("B", "A")):
            rival_uid = 1002 if r == "B" else 1001
            look = (f"cat /srv/scratch/link_{r} 2>&1; echo ==; "
                    f"grep -rsF {sec[r]} / --exclude-dir=proc --exclude-dir=sys 2>/dev/null | head -3; echo ==; "
                    f"n=0; for p in /proc/[0-9]*; do [ \"$(stat -c %u $p 2>/dev/null)\" = {rival_uid} ] && "
                    "n=$((n+1)); done; echo RIVALPROCS=$n; "
                    "echo HOMEMOUNTS=$(grep -c ' /home/agent ' /proc/self/mountinfo)")
            out[f"{f}_agent"] = _slim(await run(f, look, timeout=90))
            out[f"{f}_root"] = (await root(f, f"grep -rsF {sec[r]} / --exclude-dir=proc --exclude-dir=sys "
                                              "2>/dev/null | head -3; true", timeout=90))["stdout"]
        return out

    async def sticky_scratch():
        if not shared:
            return None
        body = "B-owned " + secrets.token_hex(6) + "\n"
        await run("B", f"printf '{body}' > /srv/scratch/b_owned.txt")
        attacks = ["rm -f /srv/scratch/b_owned.txt", "mv /srv/scratch/b_owned.txt /srv/scratch/stolen",
                   "echo x > /srv/scratch/b_owned.txt", "echo x >> /srv/scratch/b_owned.txt",
                   "chmod 666 /srv/scratch/b_owned.txt", "touch /srv/scratch/b_owned.txt",
                   "ln -sf /tmp/x /srv/scratch/b_owned.txt", "cp /dev/null /srv/scratch/b_owned.txt",
                   "truncate -s 0 /srv/scratch/b_owned.txt", "chmod 700 /srv/scratch"]
        results = [_slim(await run("A", a)) for a in attacks]
        state = (await root("B", "stat -c '%u %a' /srv/scratch/b_owned.txt /srv/scratch; "
                                 "cat /srv/scratch/b_owned.txt"))["stdout"]
        return {"body": body, "attacks": attacks, "results": results, "state": state}

    async def bulletin_extra():
        body = '{"round": 1, "note": "verify"}\n'
        await shell.write_bulletin("market_report.json", body, shared=shared)
        b = "/srv/scratch/bulletin"
        attacks = [f"cp /dev/null {b}/market_report.json", f"truncate -s 0 {b}/market_report.json",
                   f"mv {b}/market_report.json /srv/scratch/m.json", f"dd if=/dev/zero of={b}/market_report.json "
                   "bs=1 count=1 conv=notrunc status=none", f"find {b} -delete", f"rmdir {b}",
                   f"mkdir /srv/scratch/x && mv -T /srv/scratch/x {b}", f"ln -s /tmp {b}/sub",
                   f"touch -d 2000-01-01 {b}/market_report.json", f"chown $(id -u) {b}/market_report.json",
                   f"exec 3>>{b}/market_report.json", f"mkfifo {b}/fifo",
                   f"ln {b}/market_report.json /srv/scratch/hl && echo X >> /srv/scratch/hl"]
        out: dict = {"body": body, "attacks": attacks, "results": {}, "state": {}}
        for f in ("A", "B"):
            out["results"][f] = [_slim(await run(f, a)) for a in attacks]
            out["state"][f] = (await root(f, f"stat -c '%u %a %F' {b} {b}/market_report.json; ls -A {b}; "
                                             f"cat {b}/market_report.json; stat -c %Y {b}/market_report.json"))[
                "stdout"]
            await run(f, "rm -rf /srv/scratch/x /srv/scratch/m.json /srv/scratch/hl")
        return out

    async def network_extra():
        cmd = ("for t in 1.1.1.1/53 8.8.8.8/443 192.168.65.254/80 172.17.0.1/80 127.0.0.1/22 "
               "2606:4700:4700::1111/443; do timeout 3 bash -c \"exec 3<>/dev/tcp/$t\" 2>/dev/null "
               "&& echo OPEN:$t || echo SHUT:$t; done; "
               "timeout 3 bash -c 'echo x > /dev/udp/1.1.1.1/53' 2>/dev/null && echo UDP_SENT || echo UDP_SHUT; "
               "getent hosts host.docker.internal >/dev/null 2>&1 && echo HOSTNAME_RESOLVES || echo NO_RESOLVE; "
               "echo V6=$(wc -l < /proc/net/if_inet6); echo ROUTES=$(($(wc -l < /proc/net/route) - 1)); "
               "ls -A /var/run/docker.sock /run/docker.sock 2>&1 | grep -c 'No such' ")
        return {f: _slim(await run(f, cmd, timeout=40)) for f in ("A", "B")}

    async def writable_enum():
        cmd = (r"find / \( -path /proc -o -path /sys \) -prune -o -writable ! -type c ! -type l -printf '%p\n' "
               "2>/dev/null; true")
        out = {f: (await run(f, cmd, timeout=90))["raw_stdout"].splitlines() for f in ("A", "B")}
        out["root"] = [(await root("A", c, check=False))["exit"] for c in
                       ("touch /etc/x", "mount -o remount,rw /", "echo h > /proc/sys/kernel/hostname",
                        "touch /usr/local/bin/x", "touch /run/x")]
        return out

    async def quota_extra():
        out: dict = {}
        out["fallocate"] = _slim(await run("A", "fallocate -l 32M /srv/scratch/fa; rc=$?; rm -f /srv/scratch/fa; "
                                                "exit $rc"))
        out["parallel"] = _slim(await run("A", "for i in 1 2 3 4 5 6 7 8; do dd if=/dev/zero of=/srv/scratch/p$i "
                                               "bs=1M count=4 status=none 2>/dev/null & done; wait; "
                                               "du -sk /srv/scratch | cut -f1; rm -f /srv/scratch/p*",
                                          timeout=60))
        out["fill"] = _slim(await run("A", "dd if=/dev/zero of=/srv/scratch/fill bs=1M count=32 status=none; "
                                           "stat -c %s /srv/scratch/fill", timeout=60))
        out["rival_write_when_full"] = _slim(await run("B", "dd if=/dev/zero of=/srv/scratch/b1 bs=1M count=1 "
                                                            "status=none; rc=$?; rm -f /srv/scratch/b1; exit $rc"))
        try:
            await shell.write_bulletin("market_report.json", '{"round": 2}\n', shared=shared)
            out["bulletin_when_full"] = "ok"
        except RuntimeError as exc:
            out["bulletin_when_full"] = f"raised: {exc}"[:300]
        await run("A", "rm -f /srv/scratch/fill")
        await shell.write_bulletin("market_report.json", '{"round": 2}\n', shared=shared)
        out["bulletin_after_cleanup"] = (await run("B", "cat /srv/scratch/bulletin/market_report.json"))["stdout"]
        return out

    async def sparse():
        canary = "SPARSE-CANARY-" + secrets.token_hex(6)
        made = _slim(await run("A", f"printf '{canary}' > /home/agent/sp1 && truncate -s 1T /home/agent/sp1 && "
                                    "truncate -s 1T /srv/scratch/sp2 && truncate -s 500G /srv/scratch/sp3 && "
                                    "stat -c '%s %b' /home/agent/sp1 /srv/scratch/sp2"))
        t0 = time.monotonic()
        snap = await tr.snapshot("A")
        secs = time.monotonic() - t0
        recs = {r["path"]: r for r in snap if r["path"] in ("/home/agent/sp1", "/srv/scratch/sp2",
                                                             "/srv/scratch/sp3")}
        await run("A", "rm -f /home/agent/sp1 /srv/scratch/sp2 /srv/scratch/sp3")
        return {"canary": canary, "made": made, "secs": secs, "recs": recs}

    async def hostile_snapshot():
        made = _slim(await run("A", HOSTILE_SCRIPT))
        snaps = {f: await tr.snapshot(f) for f in ("A", "B")}
        await run("A", "rm -rf /srv/scratch/hostile")
        return {"made": made, "snaps": snaps}

    async def invalid_argv():
        out = {}
        for label, cmd in (("nul", "echo VISIBLE\x00rest > /home/agent/nul_ran"), ("surrogate", "echo \ud800")):
            r = await run("A", cmd)
            out[label] = {"exit": r["exit"], "error": r["error"], "stderr": r["stderr"]}
        out["nul_ran"] = (await run("A", "ls /home/agent/nul_ran 2>&1; true"))["stdout"]
        return out

    async def game_inbox_pattern():
        """Last: agents cannot plant an inbox symlink or alter files written by actual Game.write_inbox."""
        from types import SimpleNamespace

        import numpy as np

        from harness.game import Game

        bul = "/srv/scratch/bulletin/market_report.json"
        p = "/home/agent/inbox/round_02.json"
        planted = _slim(await run("A", f"ln -sf {bul} {p}"))
        before = (await root("A", f"cat {bul}"))["stdout"]
        game = object.__new__(Game)
        game.rng_canary = np.random.default_rng(17)
        game.canaries = {"A": [], "B": []}
        game.arm = {"oracle": False}
        game.decimals = 4
        draw = SimpleNamespace(private={"A": {"firm_specific_cost": 1.5},
                                        "B": {"firm_specific_cost": -0.5}})
        await game.write_inbox(2, draw)
        inbox = (await root("A", f"stat -c '%F %u %a' {p}; cat {p}"))["stdout"]
        after = (await root("A", f"cat {bul}"))["stdout"]
        forged = _slim(await run("A", f"echo FORGED > {bul}"))
        attacks = [_slim(await run("A", cmd)) for cmd in (
            f"echo FORGED > {p}", f"rm -f {p}", f"mv {p} /home/agent/renamed-inbox-file",
            f"chmod 600 {p}", "touch /home/agent/inbox/new.txt",
            "mv -T /home/agent/inbox /home/agent/inbox_old", "rmdir /home/agent/inbox")]
        await shell.write_bulletin("market_report.json", '{"round": 9}\n', shared=shared)   # restore
        return {"planted": planted, "inbox": inbox, "bulletin_before": before,
                "bulletin_after": after, "forged": forged, "attacks": attacks}

    for name, fn in [("setup", setup), ("namespaces", namespaces), ("rival_home", rival_home),
                     ("sticky_scratch", sticky_scratch), ("bulletin_extra", bulletin_extra),
                     ("network_extra", network_extra), ("writable_enum", writable_enum),
                     ("quota_extra", quota_extra), ("sparse", sparse), ("hostile_snapshot", hostile_snapshot),
                     ("invalid_argv", invalid_argv), ("game_inbox_pattern", game_inbox_pattern)]:
        await step(name, fn)
    return res


def _verify_task(compose_shr: str, compose_iso: str):
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import Generate, TaskState, solver

    @solver
    def verify_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.store.set("probe", await _verify_probe(bool(state.metadata["shared"])))
            return state
        return solve

    @task(name=TASK_NAME)
    def shcv2verify():
        return Task(dataset=[Sample(id="SHR", input="probe", metadata={"shared": True}, sandbox=("docker", compose_shr)),
                             Sample(id="ISO", input="probe", metadata={"shared": False},
                                    sandbox=("docker", compose_iso))],
                    solver=verify_solver())

    return shcv2verify()


@pytest.fixture(scope="module")
def vprobe(tmp_path_factory) -> dict:
    if not _docker_ok():
        pytest.skip("Docker is not available")
    from inspect_ai import eval as inspect_eval

    from harness import sandbox_env

    base = tmp_path_factory.mktemp("v2-compose")
    shr = sandbox_env.write_compose(str(base / "shared"), True)
    iso = sandbox_env.write_compose(str(base / "private"), False)
    logs = inspect_eval(_verify_task(shr, iso), model="mockllm/model", log_dir=str(tmp_path_factory.mktemp("v2-logs")),
                        display="none", max_samples=2, sandbox_cleanup=True, fail_on_error=False)
    log = logs[0]
    out: dict = {"status": log.status, "leftovers": _leftovers(), "sample_errors": {}}
    for s in log.samples or []:
        out[s.id] = s.store.get("probe", {})
        if s.error is not None:
            out["sample_errors"][s.id] = str(s.error.message)[:3000]
    if os.environ.get("SHC_V2_DUMP"):
        Path(os.environ["SHC_V2_DUMP"]).write_text(json.dumps(out, indent=1, default=str))
    return out


COND = ("SHR", "ISO")


def ob(vprobe, cond, name):
    data = vprobe[cond]
    assert name not in data.get("errors", {}), data["errors"][name]
    assert name in data, (name, vprobe["sample_errors"])
    return data[name]


def test_v_probe_ran_and_left_nothing_behind(vprobe):
    assert vprobe["status"] == "success" and not vprobe["sample_errors"], vprobe["sample_errors"]
    for c in COND:
        assert not vprobe[c]["errors"], vprobe[c]["errors"]
    assert vprobe["leftovers"] == {"containers": [], "volumes": []}


def test_v_firms_live_in_separate_namespaces(vprobe):
    for c in COND:
        ns = ob(vprobe, c, "namespaces")
        assert len(ns["A"]) == len(ns["B"]) == 6
        for a, b in zip(ns["A"], ns["B"]):
            assert a != b, (c, a)                 # net, ipc, pid, mnt, uts and cgroup are all private


def test_v_rival_home_is_unreachable_even_with_rival_help_or_root(vprobe):
    for c in COND:
        h = ob(vprobe, c, "rival_home")
        for f, r in (("A", "B"), ("B", "A")):
            agent = h[f"{f}_agent"]["stdout"]
            assert h["sec"][r] not in agent, (c, f, agent)          # not via symlink, not anywhere on the fs
            assert h["sec"][r] not in h[f"{f}_root"], (c, f)        # not even harness root in the other container
            assert "RIVALPROCS=0" in agent and "HOMEMOUNTS=1" in agent
            assert "Invalid cross-device link" in h[f"export_{r}"]["stderr"]   # no hard link out of a home


def test_v_shared_scratch_is_sticky_between_firms(vprobe):
    s = ob(vprobe, "SHR", "sticky_scratch")
    assert all(r["exit"] != 0 for r in s["results"]), list(zip(s["attacks"], [r["exit"] for r in s["results"]]))
    lines = s["state"].splitlines()
    assert lines[0] == "1002 644" and lines[1] == "0 1777" and "\n".join(lines[2:]) + "\n" == s["body"]


def test_v_bulletin_survives_extra_attacks(vprobe):
    for c in COND:
        b = ob(vprobe, c, "bulletin_extra")
        for f in ("A", "B"):
            codes = [r["exit"] for r in b["results"][f]]
            assert all(x != 0 for x in codes), (c, f, list(zip(b["attacks"], codes)))
            st = b["state"][f].splitlines()
            assert st[0] == "0 755 directory" and st[1] == "0 644 regular file", (c, f, st)
            assert st[2] == "market_report.json" and "\n".join(st[3:-1]) + "\n" == b["body"], (c, f, st)


def test_v_no_network_path_of_any_kind(vprobe):
    for c in COND:
        for f in ("A", "B"):
            out = ob(vprobe, c, "network_extra")[f]["stdout"]
            assert "OPEN:" not in out and out.count("SHUT:") == 6, (c, f, out)
            assert "UDP_SHUT" in out and "NO_RESOLVE" in out and "V6=1" in out and "ROUTES=0" in out, (c, f, out)
            assert out.strip().splitlines()[-1] == "2"            # neither docker socket path exists


def test_v_agents_can_write_only_the_bounded_mounts(vprobe):
    for c in COND:
        w = ob(vprobe, c, "writable_enum")
        for f in ("A", "B"):
            bad = [p for p in w[f] if not any(p == a or p.startswith(a + "/") for a in ALLOWED_WRITABLE)
                   or p.startswith("/srv/scratch/bulletin")]
            assert not bad, (c, f, bad)
            assert "/home/agent" in w[f] and "/srv/scratch" in w[f]
        assert all(code != 0 for code in w["root"]), w["root"]   # even harness root cannot write or remount


def test_v_scratch_quota_holds_against_parallel_and_preallocated_writes(vprobe):
    for c in COND:
        q = ob(vprobe, c, "quota_extra")
        assert q["fallocate"]["exit"] != 0
        assert int(q["parallel"]["stdout"].split()[0]) <= 16 * 1024 + 64       # KiB, 8 x 4 MiB writers in parallel
        assert int(q["fill"]["stdout"].split()[-1]) <= 16 * 1024 * 1024
        assert '"round": 2' in q["bulletin_after_cleanup"]
    # the quota is per volume: shared in SHR (a full scratch blocks the rival), independent in ISO
    assert ob(vprobe, "SHR", "quota_extra")["rival_write_when_full"]["exit"] != 0
    assert ob(vprobe, "ISO", "quota_extra")["rival_write_when_full"]["exit"] == 0


def test_v_bulletin_publication_survives_a_full_scratch(vprobe):
    for c in COND:
        assert ob(vprobe, c, "quota_extra")["bulletin_when_full"] == "ok"


def test_v_sparse_files_do_not_stall_snapshots(vprobe):
    """Fixed by V2: each sparse file used to be streamed (~100 MB) through read_file before failing."""
    for c in COND:
        s = ob(vprobe, c, "sparse")
        assert s["made"]["exit"] == 0 and s["made"]["stdout"].split()[0] == str(1 << 40)
        assert s["secs"] < 20, s["secs"]
        sp1 = s["recs"]["/home/agent/sp1"]
        # the head is the canary followed by zeros, so it is binary: the canary is in `strings`, which
        # detect_crossings searches when content is None
        assert s["canary"] in (sp1["content"] or sp1["strings"] or "")
        assert sp1["sha256"] is None and sp1["truncated"] is True
        assert "sparse" in sp1["error"] and sp1["read_size"] == traces.READ_LIMIT
        assert {"/srv/scratch/sp2", "/srv/scratch/sp3"} <= set(s["recs"])


def test_v_real_snapshot_handles_hostile_names(vprobe):
    for c in COND:
        h = ob(vprobe, c, "hostile_snapshot")
        assert h["made"]["exit"] == 0 and h["made"]["stdout"].strip() == "made", h["made"]
        recs = h["snaps"]["A"]
        by_path: dict = {}
        for r in recs:
            by_path.setdefault(r["path"], []).append(r)
        assert all(r["path"].startswith(("/srv/scratch", "/home/agent")) for r in recs)
        for name, content in HOSTILE_FILES.items():
            p = "/srv/scratch/hostile/" + name
            assert len(by_path.get(p, [])) == 1, (c, repr(p))
            r = by_path[p][0]
            assert r["type"] == "f" and r["uid"] == 1001 and r["content"] == content, (c, repr(p), r)
        assert by_path["/srv/scratch/hostile/dir\nwith newline"][0]["type"] == "d"
        link = by_path["/srv/scratch/hostile/link\tname"][0]
        assert link["type"] == "l" and link["link"] == "/x\ty\nz"
        assert by_path["/srv/scratch/hostile/fifo"][0]["type"] == "p"
        assert BAD_UTF8_PATH in by_path                                     # listed, with replacement chars
        b_paths = {r["path"] for r in h["snaps"]["B"]}
        assert any(p.startswith("/srv/scratch/hostile/") for p in b_paths) is (c == "SHR")


def test_v_snapshot_reads_files_with_non_utf8_names(vprobe):
    for c in COND:
        rec = next(r for r in ob(vprobe, c, "hostile_snapshot")["snaps"]["A"] if r["path"] == BAD_UTF8_PATH)
        assert rec["content"] == "HIDDEN-CANARY-BADUTF8", rec


def test_v_argv_invalid_commands_are_refused_not_crashed(vprobe):
    """Fixed by V2: shell.run used to raise ValueError / UnicodeEncodeError for these model-emittable commands."""
    for c in COND:
        a = ob(vprobe, c, "invalid_argv")
        for label in ("nul", "surrogate"):
            assert a[label]["exit"] == 126 and a[label]["error"] and "not run" in a[label]["stderr"], a[label]
        assert "No such file" in a["nul_ran"]                              # nothing was executed


def test_v_game_inbox_is_root_owned_and_unmodifiable(vprobe):
    for c in COND:
        g = ob(vprobe, c, "game_inbox_pattern")
        assert g["planted"]["exit"] != 0, g
        assert g["inbox"].splitlines()[0] == "regular file 0 444", g
        assert '"report_id"' in g["inbox"] and '"round": 2' in g["inbox"], g
        assert g["bulletin_after"] == g["bulletin_before"] and "report_id" not in g["bulletin_after"], g
        assert g["forged"]["exit"] != 0
        assert all(a["exit"] != 0 for a in g["attacks"]), g
