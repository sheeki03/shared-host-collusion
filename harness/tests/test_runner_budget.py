"""Offline runner/ledger integration checks; no Docker or provider calls."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from harness import run
from harness.ledger import BudgetLedger, LedgerError


def test_runner_ledger_must_be_initialized_then_resumed(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(LedgerError):
        run.open_run_ledger(str(path), 10.0)
    first = run.open_run_ledger(str(path), 10.0, init=True)
    assert first.spent == 0 and first.ledger_id
    with pytest.raises(FileExistsError):
        run.open_run_ledger(str(path), 10.0, init=True)
    assert run.open_run_ledger(str(path), 10.0).ledger_id == first.ledger_id
    with pytest.raises(LedgerError, match="cap is"):
        run.open_run_ledger(str(path), 11.0)
    raised = run.open_run_ledger(str(path), 11.0, previous_cap=10.0, raise_reason="owner approved")
    assert raised.cap_usd == 11.0 and raised.ledger_id == first.ledger_id
    assert BudgetLedger(path, 11.0, create=False).cap_usd == 11.0


def test_saved_result_costs_are_checked(tmp_path):
    sample = SimpleNamespace(id="x", epoch=1, error=None, metadata={"result": {"calls": [
        {"total_cost_usd": 0.01}, {"total_cost_usd": 0.02}]}})
    log = SimpleNamespace(status="success", error=None, samples=[sample], location="unused")
    assert run.completed_cost([log]) == pytest.approx(0.03)
    sample.metadata["result"]["calls"].append({"cost_usd": 0.03})
    with pytest.raises(RuntimeError, match="missing total_cost_usd"):
        run.completed_cost([log])


def test_pilot_headroom_is_reported_before_calls():
    probes = run.pilot_probes()
    peak, wave = run.required_headroom(probes, 13)
    assert 0.3 < peak < 1.0 and 3 < wave < 5                  # byte-bounded reservations (Opus is the peak)
    assert run.required_headroom(run.pilot_smoke(), 12)[1] < 1.5


def test_failed_sample_is_not_a_successful_cli_result():
    sample = SimpleNamespace(id="broken", epoch=1, error=SimpleNamespace(message="boom"), metadata={})
    log = SimpleNamespace(status="success", error=None, samples=[sample], location="unused")
    with pytest.raises(RuntimeError, match="sample broken"):
        run.completed_cost([log])


# ------------------------------------------------------------------ replay-aware reconciliation (synthetic ledgers)

def _attempt(led, sample_id, cost, *, outcome="ok", reason=None):
    """One provider attempt as nanogpt_provider records it: reserve, then settle (or release), plus the saved dict."""
    async def go():
        token = await led.reserve(0.5, {"model": "z-ai/glm-5.3", "sample_id": sample_id, "call_kind": "subject"})
        if outcome == "released":
            await led.release(token, reason="never_sent:ConnectError")
            return {"reservation_id": token, "outcome": "released", "settled_usd": 0.0, "status": None}
        settled = await led.settle(token, cost, reason=reason)
        return {"reservation_id": token, "outcome": outcome, "settled_usd": settled, "status": 200}
    return asyncio.run(go())


def _call(step, attempts):
    return {"t": 1, "firm": "A", "step": step, "attempts": attempts, "attempt_count": len(attempts),
            "total_cost_usd": float(sum(a["settled_usd"] for a in attempts)),
            "cost_usd": attempts[-1]["settled_usd"], "reservation_id": attempts[-1]["reservation_id"]}


def _sample(game_id, calls, *, epoch=1, retries=0):
    return SimpleNamespace(id=game_id, epoch=epoch, error=None, metadata={"result": {"calls": calls}},
                           error_retries=[SimpleNamespace(message="ReadError") for _ in range(retries)] or None)


def _log(*samples):
    return SimpleNamespace(status="success", error=None, samples=list(samples), location="unused")


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = BudgetLedger(path, 10.0, create=True)
    # an earlier run on the same ledger: billed before the mark, so never part of this run's window
    _attempt(led, "old-game#1", 0.25)
    return path, led


def _near(a, b) -> bool:
    return abs(Decimal(a) - Decimal(b)) <= run.TOL


def _reconcile(path, led, mark, before, logs):
    return run.reconcile_run(logs, run.read_ledger_window(path, mark), ledger_delta=led.spent - before,
                             open_reservations=len(led.open_reservations()))


def test_reconciliation_matches_every_saved_attempt(ledger):
    path, led = ledger
    mark, before = run.ledger_mark(path), led.spent
    g1 = [_call("act#1", [_attempt(led, "g1#1", None, outcome="transport_error", reason="transport_error:ReadError"),
                          _attempt(led, "g1#1", 0.0125)]),
          _call("compliance", [_attempt(led, "g1#1", None, outcome="released"), _attempt(led, "g1#1", 0.001)])]
    g2 = [_call("act#1", [_attempt(led, "g2#1", 0.02)])]
    rec = _reconcile(path, led, mark, before, [_log(_sample("g1", g1), _sample("g2", g2))])
    assert rec.ok, rec.unexplained
    assert rec.saved_usd == rec.window_settled_usd == Decimal("0.5335")      # the ReadError settled at est_max 0.5
    assert _near(rec.ledger_delta_usd, "0.5335") and _near(rec.unexplained_usd, 0)
    assert rec.replayed_usd == 0 and rec.replayed_games == {} and rec.n_saved_attempts == 5
    assert rec.flags == {"transport_error": 1}
    assert run.completed_cost([_log(_sample("g1", g1), _sample("g2", g2))]) == pytest.approx(0.5335)


def _replayed_run(led):
    """g1 errored after two billed calls (one aborted at its full reservation), Inspect replayed it (same id and
    epoch) and saved only the final run's calls; g2 ran once."""
    _attempt(led, "g1#1", 0.03)
    _attempt(led, "g1#1", None, outcome="aborted", reason="aborted:CancelledError")
    _attempt(led, "g1#1", None, outcome="released")
    final = [_call("act#1", [_attempt(led, "g1#1", 0.04)]), _call("act#2", [_attempt(led, "g1#1", 0.01)])]
    g2 = [_call("act#1", [_attempt(led, "g2#1", 0.02)])]
    return final, g2


def test_replayed_sample_attempts_are_reported_separately(ledger):
    path, led = ledger
    mark, before = run.ledger_mark(path), led.spent
    final, g2 = _replayed_run(led)
    rec = _reconcile(path, led, mark, before, [_log(_sample("g1", final, retries=1), _sample("g2", g2))])
    assert rec.ok, rec.unexplained
    assert rec.window_settled_usd == Decimal("0.6")                # 0.03 + 0.5 + 0 + 0.04 + 0.01 + 0.02
    assert rec.saved_usd == rec.saved_calls_usd == Decimal("0.07")
    assert rec.replayed_usd == Decimal("0.53") and rec.replayed_attempts == 3
    assert rec.replayed_games == {"g1#1": {"replays": 1, "attempts": 3, "usd": Decimal("0.53")}}
    assert _near(rec.saved_usd + rec.replayed_usd, rec.ledger_delta_usd) and rec.flags == {"aborted": 1}
    d = rec.to_dict()
    assert d["ok"] and d["replayed_usd"] == pytest.approx(0.53) and json.dumps(d)
    # the old whole-run comparison (saved calls vs ledger delta) would have failed this run
    assert abs(run.completed_cost([_log(_sample("g1", final, retries=1), _sample("g2", g2))])
               - (led.spent - before)) > 1e-8


def test_unsaved_attempts_without_a_replay_are_unexplained(ledger):
    path, led = ledger
    mark, before = run.ledger_mark(path), led.spent
    final, g2 = _replayed_run(led)
    rec = _reconcile(path, led, mark, before, [_log(_sample("g1", final), _sample("g2", g2))])   # no error_retries
    assert not rec.ok and rec.replayed_usd == 0 and _near(rec.unexplained_usd, "0.53")
    assert any("which Inspect did not replay" in p for p in rec.unexplained)
    # a replayed g2 does not explain g1's unsaved attempts
    rec = _reconcile(path, led, mark, before, [_log(_sample("g1", final), _sample("g2", g2, retries=1))])
    assert not rec.ok and rec.replayed_games == {}


def test_foreign_unattributed_and_open_reservations_fail(ledger):
    path, led = ledger
    pending = asyncio.run(led.reserve(0.5, {"sample_id": "old-game#1"}))      # in flight at the mark
    mark, before = run.ledger_mark(path), led.spent
    g2 = [_call("act#1", [_attempt(led, "g2#1", 0.02)])]
    _attempt(led, "other-game#1", 0.05)                                       # a game not in this run's logs
    _attempt(led, None, 0.07)                                                 # no sample attribution
    asyncio.run(led.settle(pending, 0.11))                                    # reserved before the mark
    open_now = asyncio.run(led.reserve(0.5, {"sample_id": "g2#1"}))           # never closed
    rec = _reconcile(path, led, mark, before, [_log(_sample("g2", g2, retries=2))])
    text = "\n".join(rec.unexplained)
    assert not rec.ok and rec.open_reservations == 1 and _near(rec.unexplained_usd, "0.23")
    assert "not in this run's logs" in text and "with no sample_id" in text
    assert "reservations made before this run" in text and f"unsaved reservation {open_now}" in text


def test_saved_costs_must_match_the_ledger(ledger):
    path, led = ledger
    mark, before = run.ledger_mark(path), led.spent
    att = _attempt(led, "g1#1", 0.02)
    cases = {
        "settled_usd": [_call("a", [{**att, "settled_usd": 0.03}])],                   # saved amount differs
        "not its attempts' sum": [{**_call("a", [att]), "total_cost_usd": 0.05}],      # total above its attempts
        "no reservation since the start mark": [_call("a", [att, {"reservation_id": "f" * 32,
                                                                   "settled_usd": 0.0}])],
        "missing or duplicate reservation id": [_call("a", [att]), _call("b", [att])],
        "no per-attempt cost record": [{**_call("a", [att]), "attempts": []}],
    }
    for needle, calls in cases.items():
        rec = _reconcile(path, led, mark, before, [_log(_sample("g1", calls))])
        assert not rec.ok and any(needle in p for p in rec.unexplained), (needle, rec.unexplained)
    rec = _reconcile(path, led, mark, before, [_log(_sample("g1", [_call("a", [att])], epoch=2))])
    assert not rec.ok and any("was reserved for g1#1" in p for p in rec.unexplained)


def test_missing_costs_still_fail_closed(ledger):
    path, _ = ledger
    window = run.read_ledger_window(path, run.ledger_mark(path))
    no_total = _sample("g1", [{"step": "a", "attempts": []}])
    broken = SimpleNamespace(id="g2", epoch=1, error=SimpleNamespace(message="boom"), metadata={})
    for logs in ([_log(no_total)], [_log(broken)], [SimpleNamespace(status="error", error="stopped",
                                                                     samples=[_sample("g3", [])], location="x")]):
        with pytest.raises(RuntimeError, match="incomplete Inspect evaluation"):
            run.reconcile_run(logs, window, ledger_delta=0.0)


def test_ledger_window_rejects_unexplained_malformed_lines(ledger):
    path, led = ledger
    mark = run.ledger_mark(path)
    with open(path, "ab") as fh:
        fh.write(b'{"not json\n')
    assert run.read_ledger_window(path, mark).errors
    window = run.read_ledger_window(path, run.ledger_mark(path))
    assert window.events == [] and not window.errors
    with open(path, "ab") as fh:
        fh.write(b'{"event": "settle"')                                       # unterminated: not an event yet
    assert run.read_ledger_window(path, run.ledger_mark(path)).events == []
    assert run.read_ledger_window(path, os.path.getsize(path) + 10).errors   # shrank below the mark


# ------------------------------------------------------------------------------------------ main() end to end

def _offline_main(tmp_path, monkeypatch, play):
    import inspect_ai
    import harness.game

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(harness.game, "shc_games", lambda *a, **kw: "offline-task")
    for var in ("SHC_LEDGER_PATH", "SHC_BUDGET_CAP_USD"):     # main() sets them; restore them after the test
        monkeypatch.setenv(var, "unset")

    def fake_eval(*a, **kw):
        led = BudgetLedger(os.environ["SHC_LEDGER_PATH"], float(os.environ["SHC_BUDGET_CAP_USD"]), create=False)
        return play(led)

    monkeypatch.setattr(inspect_ai, "eval", fake_eval)
    ledger = tmp_path / "runs" / "ledger.jsonl"
    monkeypatch.setattr(sys, "argv", ["harness.run", "--plan", "pilot-probes", "--models", "glm-5.3",
                                      "--cap", "10", "--concurrency", "1", "--init-ledger",
                                      "--ledger", str(ledger)])
    return ledger, tmp_path / "runs" / "pilot" / "reconciliation_pilot-probes.json"


def test_non_dry_runner_smoke_uses_ledger_properties_and_checks_logs(tmp_path, monkeypatch):
    sample = SimpleNamespace(id="offline", epoch=1, error=None, metadata={"result": {"calls": []}})
    ledger, report = _offline_main(tmp_path, monkeypatch, lambda led: [_log(sample)])
    run.main()
    assert BudgetLedger(ledger, 10.0, create=False).spent == 0
    assert json.loads(report.read_text())["ok"]


def test_runner_passes_a_run_with_a_replayed_game(tmp_path, monkeypatch, capsys):
    gid = "pilot-probe:glm-5.3"

    def play(led):
        _attempt(led, f"{gid}#1", 0.03)                                       # the discarded first run
        return [_log(_sample(gid, [_call("act#1", [_attempt(led, f"{gid}#1", 0.04)])], retries=1))]

    ledger, report = _offline_main(tmp_path, monkeypatch, play)
    run.main()
    out = capsys.readouterr().out
    assert "replayed attempts $0.030000 (1 attempts on 1 replayed games" in out
    rep = json.loads(report.read_text())
    assert rep["ok"] and rep["replayed_usd"] == pytest.approx(0.03) and rep["saved_usd"] == pytest.approx(0.04)
    assert rep["ledger_delta_usd"] == pytest.approx(0.07) and rep["window_bytes"][0] > 0


def test_runner_fails_an_unsaved_attempt_without_a_replay(tmp_path, monkeypatch):
    gid = "pilot-probe:glm-5.3"

    def play(led):
        _attempt(led, f"{gid}#1", 0.03)
        return [_log(_sample(gid, [_call("act#1", [_attempt(led, f"{gid}#1", 0.04)])]))]

    ledger, report = _offline_main(tmp_path, monkeypatch, play)
    with pytest.raises(SystemExit, match="budget reconciliation failed.*unexplained \\$0.030000"):
        run.main()
    assert not json.loads(report.read_text())["ok"]
