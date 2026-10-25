"""Offline runner/ledger integration checks; no Docker or provider calls."""
from __future__ import annotations

import sys
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
    sample = SimpleNamespace(id="x", error=None, metadata={"result": {"calls": [
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
    sample = SimpleNamespace(id="broken", error=SimpleNamespace(message="boom"), metadata={})
    log = SimpleNamespace(status="success", error=None, samples=[sample], location="unused")
    with pytest.raises(RuntimeError, match="sample broken"):
        run.completed_cost([log])


def test_non_dry_runner_smoke_uses_ledger_properties_and_checks_logs(tmp_path, monkeypatch):
    import inspect_ai
    import harness.game

    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(harness.game, "shc_games", lambda *a, **kw: "offline-task")
    sample = SimpleNamespace(id="offline", error=None, metadata={"result": {"calls": []}})
    log = SimpleNamespace(status="success", error=None, samples=[sample], location="unused")
    monkeypatch.setattr(inspect_ai, "eval", lambda *a, **kw: [log])
    ledger = tmp_path / "runs" / "ledger.jsonl"
    monkeypatch.setattr(sys, "argv", ["harness.run", "--plan", "pilot-probes", "--models", "glm-5.3",
                                   "--cap", "10", "--concurrency", "1", "--init-ledger",
                                   "--ledger", str(ledger)])
    run.main()
    assert BudgetLedger(ledger, 10.0, create=False).spent == 0
