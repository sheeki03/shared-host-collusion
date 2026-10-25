"""Tests for harness.ledger.BudgetLedger (CONTRACT.md H1): concurrency, crash recovery, missing cost, cap safety."""
from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import random
from decimal import Decimal

import pytest

from harness.ledger import BudgetExceeded, BudgetLedger, LedgerError

LINE_KEYS = {"ts", "event", "id", "model", "est_max", "actual", "sample_id", "call_kind", "pid"}


def read_lines(path) -> list[dict]:
    with open(path, "rb") as f:
        return [v for x in f.read().splitlines() if x.strip() for v in [json.loads(x)]
                if v["event"] not in ("init", "cap_change")]


def all_lines(path) -> list[dict]:
    with open(path, "rb") as f:
        return [json.loads(x) for x in f.read().splitlines() if x.strip()]


def replay_check(path, cap: float) -> tuple[Decimal, Decimal]:
    """Independent replay of the file: spent + outstanding never exceeds the cap after any reserve."""
    spent, out, open_ = Decimal(0), Decimal(0), {}
    capd = Decimal(repr(cap))
    for ev in read_lines(path):
        if ev["event"] == "reserve":
            open_[ev["id"]] = Decimal(repr(ev["est_max"]))
            out += open_[ev["id"]]
            assert spent + out <= capd, f"cap exceeded at reserve {ev['id']}: {spent} + {out} > {capd}"
        elif ev["event"] in ("settle", "release"):
            out -= open_.pop(ev["id"])
            if ev["event"] == "settle":
                spent += Decimal(repr(ev["actual"]))
    return spent, out


async def _try_reserve(ledger: BudgetLedger, est: float, **meta) -> str | None:
    try:
        return await ledger.reserve(est, meta)
    except BudgetExceeded:
        return None


def test_reserve_settle_release_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = BudgetLedger(path, 1.0)

    async def go():
        t1 = await led.reserve(0.3, {"model": "m1", "sample_id": "s1", "call_kind": "firm", "extra": 7})
        assert led.outstanding == pytest.approx(0.3)
        assert await led.settle(t1, 0.1) == pytest.approx(0.1)
        t2 = await led.reserve(0.2, {"model": "m2"})
        await led.release(t2, reason="ConnectError")
    asyncio.run(go())
    assert led.spent == pytest.approx(0.1)
    assert led.outstanding == 0
    assert led.remaining == pytest.approx(0.9)
    lines = read_lines(path)
    assert [x["event"] for x in lines] == ["reserve", "settle", "reserve", "release"]
    for x in lines:
        assert LINE_KEYS <= set(x)
    assert lines[0]["meta"] == {"extra": 7}
    assert lines[1]["model"] == "m1" and lines[1]["sample_id"] == "s1" and lines[1]["call_kind"] == "firm"
    assert lines[1]["actual"] == 0.1 and lines[1]["est_max"] == 0.3
    assert lines[3]["reason"] == "ConnectError"


def test_200_parallel_reserves_admit_exactly_n(tmp_path):
    """Cap 0.37 with est 0.01 admits exactly 37 of 200 concurrent reserves, split over two objects on one path."""
    path = tmp_path / "ledger.jsonl"
    a, b = BudgetLedger(path, 0.37), BudgetLedger(path, 0.37)

    async def go():
        return await asyncio.gather(*[_try_reserve(a if i % 2 else b, 0.01) for i in range(200)])
    tokens = asyncio.run(go())
    ok = [t for t in tokens if t is not None]
    assert len(ok) == 37
    assert len(set(ok)) == 37
    assert a.outstanding == pytest.approx(0.37) and b.outstanding == pytest.approx(0.37)
    assert a.remaining == pytest.approx(0.0, abs=1e-12)
    replay_check(path, 0.37)


def test_exact_decimal_admission(tmp_path):
    """100 x $0.01 under a $1.00 cap: float summation would admit 99 (0.01*100 > 1.0 in binary); Decimal admits 100."""
    led = BudgetLedger(tmp_path / "ledger.jsonl", 1.0)

    async def go():
        return [await _try_reserve(led, 0.01) for _ in range(105)]
    assert sum(t is not None for t in asyncio.run(go())) == 100


def _mp_worker(path: str, cap: float, n: int, est: float) -> int:
    led = BudgetLedger(path, cap)

    async def go():
        return await asyncio.gather(*[_try_reserve(led, est) for _ in range(n)])
    return sum(t is not None for t in asyncio.run(go()))


def test_multiprocess_reserves_admit_exactly_n(tmp_path):
    """Four processes x 50 reserves on one file; a cap admitting 61 admits exactly 61 in total (fcntl.flock)."""
    path = str(tmp_path / "ledger.jsonl")
    BudgetLedger(path, 0.61)
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        counts = pool.starmap(_mp_worker, [(path, 0.61, 50, 0.01)] * 4)
    assert sum(counts) == 61
    assert len([x for x in read_lines(path) if x["event"] == "reserve"]) == 61
    assert BudgetLedger(path, 0.61).outstanding == pytest.approx(0.61)
    replay_check(path, 0.61)


def test_crash_recovery_rebuilds_from_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    a = BudgetLedger(path, 1.0)

    async def first():
        r1 = await a.reserve(0.2, {"model": "m"})
        r2 = await a.reserve(0.3, {"model": "m"})
        r3 = await a.reserve(0.4, {"model": "m", "sample_id": "x"})
        await a.settle(r1, 0.05)
        await a.release(r2)
        return r1, r3
    r1, r3 = asyncio.run(first())
    a = None  # "crash": drop the in-memory object while r3 is still in flight

    b = BudgetLedger(path, 1.0)
    assert b.spent == pytest.approx(0.05)
    assert b.outstanding == pytest.approx(0.4)
    assert [r.id for r in b.open_reservations()] == [r3]
    assert b.open_reservations()[0].sample_id == "x"

    async def second():
        with pytest.raises(BudgetExceeded):
            await b.reserve(0.56)               # 0.05 + 0.4 + 0.56 > 1.0: the orphaned reservation still counts
        await b.settle(r3, 0.02)                # a new process can settle the old process's token
        with pytest.raises(LedgerError):
            await b.settle(r1, 0.01)            # already settled
        with pytest.raises(LedgerError):
            await b.release("nope")             # unknown
    asyncio.run(second())
    c = BudgetLedger(path, 1.0)
    assert c.spent == pytest.approx(0.07) and c.outstanding == 0


def test_missing_and_invalid_cost_fail_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = BudgetLedger(path, 1.0)

    async def go():
        out = []
        for actual in (None, float("nan"), -1.0, "abc"):
            t = await led.reserve(0.05, {"model": "m"})
            out.append(await led.settle(t, actual, reason="missing_cost:http_200" if actual is None else None))
        t = await led.reserve(0.01)
        with pytest.raises(Exception, match="exceeded reserved maximum"):
            await led.settle(t, 0.02)  # bill is recorded, and the run halts rather than continuing
        return out
    assert asyncio.run(go()) == [0.05, 0.05, 0.05, 0.05]
    assert led.spent == pytest.approx(0.22)
    flags = [x for x in read_lines(path) if x["event"] == "flag"]
    assert [f["reason"].split(":")[0] for f in flags] == ["missing_cost", "invalid_cost", "invalid_cost",
                                                          "invalid_cost", "over_est_max"]
    assert flags[0]["reason"] == "missing_cost:http_200" and flags[0]["actual"] == 0.05
    assert led.flags == 5
    with pytest.raises(Exception, match="halted"):
        asyncio.run(led.reserve(0.01))


def test_cap_never_exceeded_randomised(tmp_path):
    """Random concurrent reserve/settle/release traffic from three objects; replay proves the cap always held."""
    path = tmp_path / "ledger.jsonl"
    cap = 2.5
    leds = [BudgetLedger(path, cap) for _ in range(3)]
    rng = random.Random(7)
    plan = [(rng.choice(leds), round(rng.uniform(0.001, 0.3), 6), rng.random()) for _ in range(300)]

    async def one(led, est, u):
        tok = await _try_reserve(led, est)
        if tok is None:
            return
        await asyncio.sleep(u / 500)
        if u < 0.15:
            await led.release(tok)
        elif u < 0.25:
            await led.settle(tok, None)
        else:
            await led.settle(tok, round(est * u, 6))

    async def go():
        await asyncio.gather(*[one(*p) for p in plan])
    asyncio.run(go())
    spent, out = replay_check(path, cap)
    assert out == 0
    assert leds[0].spent == pytest.approx(float(spent)) and float(spent) <= cap
    assert leds[1].outstanding == 0


def test_torn_tail_is_repaired_and_flagged(tmp_path):
    path = tmp_path / "ledger.jsonl"
    a = BudgetLedger(path, 1.0)

    async def go(led):
        return await led.reserve(0.1)
    t = asyncio.run(go(a))
    with open(path, "ab") as f:
        f.write(b'{"ts":"2026-10-25","event":"settle","id":"' + t.encode()[:5])  # writer died mid-line
    b = BudgetLedger(path, 1.0)                 # shared-lock rebuild ignores the unterminated tail
    assert b.outstanding == pytest.approx(0.1)  # the torn settle leaves its reservation outstanding (conservative)
    asyncio.run(go(b))                          # exclusive writer closes the torn line and flags it
    flags = [x for x in read_lines_lenient(path) if x.get("event") == "flag"]
    assert flags and flags[0]["reason"] == "torn_line"
    c = BudgetLedger(path, 1.0)
    assert c.outstanding == pytest.approx(0.2) and c.flags == 1


def read_lines_lenient(path) -> list[dict]:
    out = []
    for x in open(path, "rb").read().splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def test_malformed_ledger_fails_closed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    a = BudgetLedger(path, 1.0)

    async def go():
        return await a.reserve(0.1)
    asyncio.run(go())
    with open(path, "ab") as f:
        f.write(b"this is not json\n")
    with pytest.raises(LedgerError):
        BudgetLedger(path, 1.0)
    with pytest.raises(LedgerError):
        asyncio.run(go())
    with pytest.raises(LedgerError):     # sticky: the same object keeps refusing
        asyncio.run(go())


def test_missing_truncated_replaced_files(tmp_path):
    with pytest.raises(LedgerError):
        BudgetLedger(tmp_path / "absent.jsonl", 1.0, create=False)
    path = tmp_path / "ledger.jsonl"
    a = BudgetLedger(path, 1.0)

    async def go(led):
        return await led.reserve(0.1)
    asyncio.run(go(a))
    with open(path, "r+b") as f:
        f.truncate(10)
    with pytest.raises(LedgerError):
        asyncio.run(go(a))
    b = BudgetLedger(tmp_path / "l2.jsonl", 1.0)
    os.replace(tmp_path / "ledger.jsonl", tmp_path / "l2.jsonl")
    with pytest.raises(LedgerError):
        asyncio.run(go(b))


def test_settle_before_reserve_line_is_corrupt(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps({"ts": "x", "event": "settle", "id": "abc", "actual": 0.1, "est_max": 0.1}) + "\n")
    with pytest.raises(LedgerError):
        BudgetLedger(path, 1.0)


def test_ledger_survives_event_loop_change(tmp_path):
    led = BudgetLedger(tmp_path / "ledger.jsonl", 1.0)
    for _ in range(3):  # each asyncio.run is a new loop; the asyncio lock is rebound per loop
        async def go():
            await asyncio.gather(*[led.settle(await led.reserve(0.01), 0.01) for _ in range(5)])
        asyncio.run(go())
    assert led.spent == pytest.approx(0.15)


def test_cap_is_persisted_and_changes_are_audited(tmp_path):
    path = tmp_path / "ledger.jsonl"
    first = BudgetLedger(path, 1.0)
    assert all_lines(path)[0]["event"] == "init"
    with pytest.raises(LedgerError, match="cap is"):
        BudgetLedger(path, 2.0, create=False)
    first.raise_cap(2.0, reason="approved pilot increase")
    assert first.cap_usd == 2.0
    assert BudgetLedger(path, 2.0, create=False).cap_usd == 2.0
    events = all_lines(path)
    assert events[1]["event"] == "cap_change" and events[1]["reason"] == "approved pilot increase"
    with pytest.raises(LedgerError, match="cap is"):
        BudgetLedger(path, 1.0, create=False)


def test_over_estimate_records_bill_then_stops_all_new_calls(tmp_path):
    from harness.ledger import BudgetBreach

    path = tmp_path / "ledger.jsonl"
    led = BudgetLedger(path, 0.10)

    async def go():
        token = await led.reserve(0.09)
        with pytest.raises(BudgetBreach):
            await led.settle(token, 0.20)
        with pytest.raises(BudgetBreach):
            await led.reserve(0.001)
    asyncio.run(go())
    reopened = BudgetLedger(path, 0.10, create=False)
    assert reopened.spent == pytest.approx(0.20) and reopened.remaining == pytest.approx(-0.10)
    with pytest.raises(BudgetBreach):
        asyncio.run(reopened.reserve(0.001))
