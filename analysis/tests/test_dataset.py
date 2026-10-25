"""Cohort and completeness gates for loading Inspect-shaped sample logs."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from analysis import dataset


def sample(grid: str, seed: int, *, bad: bool = False):
    rows = [
        {"game_id": f"{grid}:{seed}", "t": t, "firm": firm, "action": 12.0, "scripted": False,
         "parse_failed": False, "welfare": {"consumer_surplus": 1.0}, "collusion_index": 0.0}
        for t in range(1, 3) for firm in "AB"
    ]
    if bad:
        rows.pop()
    return SimpleNamespace(id=f"{grid}:{seed}", error=None, metadata={
        "game_id": f"{grid}:{seed}", "grid": grid, "model_label": "m", "model_id": "m/id", "effort": "low",
        "scenario_id": "core-pricing", "engine": "pricing", "arm": "SHR-AWARE", "seed": seed, "rounds": 2,
        "result": {"rows": rows, "calls": [], "chat": [], "chat_deliveries": [], "scratch_events": [],
                   "gate": [], "outputs": {"A": [], "B": []}},
    })


def patch_logs(monkeypatch, *samples):
    log = SimpleNamespace(status="success", samples=list(samples))
    monkeypatch.setattr(dataset, "list_eval_logs", lambda _: [SimpleNamespace(name="fake.eval")])
    monkeypatch.setattr(dataset, "read_eval_log", lambda _: log)


def test_confirmatory_default_excludes_pilot_selftest_and_seed_9000(monkeypatch):
    patch_logs(monkeypatch, sample("deep", 1000), sample("pilot", 9000), sample("selftest", 1), sample("deep", 9001))
    tables = dataset.load("unused")
    assert tables["games"]["game_id"].tolist() == ["deep:1000"]
    assert set(tables["rounds"]["game_id"]) == {"deep:1000"}
    assert len(tables["exclusions"]) == 3


def test_incomplete_confirmatory_is_never_admitted(monkeypatch):
    patch_logs(monkeypatch, sample("deep", 1000, bad=True), sample("deep", 1001))
    with pytest.raises(ValueError, match="invalid confirmatory games"):
        dataset.load("unused")
    tables = dataset.load("unused", allow_exclusions=True)
    assert tables["games"]["game_id"].tolist() == ["deep:1001"]
    assert "incomplete game" in tables["exclusions"].iloc[0]["reason"]


def test_pilot_mode_is_explicit(monkeypatch):
    patch_logs(monkeypatch, sample("deep", 1000), sample("pilot", 9000))
    assert dataset.load("unused", cohort="pilot")["games"]["game_id"].tolist() == ["pilot:9000"]


def test_parse_failure_exclusion(monkeypatch):
    s = sample("deep", 1000)
    s.metadata["result"]["rows"][0]["parse_failed"] = True
    patch_logs(monkeypatch, s)
    with pytest.raises(ValueError, match="parse-failure"):
        dataset.load("unused")


def test_sample_error_and_missing_evidence_are_not_negative_labels(monkeypatch):
    errored = sample("deep", 1000)
    errored.error = SimpleNamespace(message="sandbox failed")
    missing = sample("deep", 1001)
    del missing.metadata["result"]["chat_deliveries"]
    patch_logs(monkeypatch, errored, missing)
    with pytest.raises(ValueError, match="invalid confirmatory games"):
        dataset.load("unused")
    exclusions = dataset.load("unused", allow_exclusions=True)["exclusions"]
    assert len(exclusions) == 2
    assert exclusions.reason.str.contains("sample error").any()
    assert exclusions.reason.str.contains("structured communication evidence").any()
