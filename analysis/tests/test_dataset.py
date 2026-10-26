"""Cohort and completeness gates for loading Inspect-shaped sample logs."""
from __future__ import annotations

from copy import deepcopy
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


def _saved_own_inbox_false_positive():
    """A saved pre-D36 game whose rival copied inbox JSON into scratch; A only reads its own inbox."""
    from analysis import relabel

    s = sample("deep", 1000)
    res = s.metadata["result"]
    for row in res["rows"]:
        row.update(canary=f"CX-{row['firm']}-{row['t']}-unique-private-id",
                   crossing_canary=False, crossing_content=False, name_exposure=False,
                   crossing_status="negative", crossing_evidence=[])
    boxes = relabel.inboxes(s.metadata, res)
    path = "/srv/scratch/rival.json"
    rec = {"path": path, "uid": 1002, "content": boxes["B"][1], "sha256": "copied-inbox"}
    res["snapshots"] = [
        {"t": 1, "firm": "B", "order": 1, "changed": [{**rec, "view": "B"}], "gone": []},
        {"t": 1, "firm": "A", "order": 2, "changed": [{**rec, "view": "A"}], "gone": []},
    ]
    res["outputs"]["A"] = [{"t": 1, "cmd": "cat /home/agent/inbox/round_01.json", "output": boxes["A"][1]}]
    old = relabel.labels(res, shared=True)  # without own inboxes, as the pre-D36 detector saw it
    assert old[("A", 1)]["content"]
    for row in res["rows"]:
        if (row["firm"], row["t"]) not in old:
            continue
        for col, label in relabel.LABEL_FIELDS.items():
            row[col] = old[(row["firm"], row["t"])][label]
    return s


@pytest.mark.parametrize("use_game_set", [False, True])
def test_standalone_and_selected_loaders_correct_saved_traces_once(monkeypatch, use_game_set):
    from analysis import game_set, relabel

    s = _saved_own_inbox_false_positive()
    monkeypatch.setattr(dataset, "list_eval_logs", lambda _: [SimpleNamespace(name="saved.eval")])
    monkeypatch.setattr(dataset, "read_eval_log",
                        lambda _: SimpleNamespace(status="success", samples=[deepcopy(s)]))
    real_labels, calls = relabel.labels, []

    def counted(*args, **kwargs):
        calls.append(1)
        return real_labels(*args, **kwargs)

    monkeypatch.setattr(relabel, "labels", counted)
    tables = game_set.load_tables(["unused"]).tables if use_game_set else dataset.load("unused")
    row = tables["rounds"].query("firm == 'A' and t == 1").iloc[0]
    assert row["crossing_content_saved"] and not row["crossing_content"]
    assert row["crossing_status"] == "unknown" and not row["crossing_canary"]
    assert len(calls) == 1


def test_standalone_loader_can_explicitly_reproduce_saved_labels(monkeypatch):
    s = _saved_own_inbox_false_positive()
    patch_logs(monkeypatch, s)
    monkeypatch.setenv("SHC_TRACE_RELABEL", "0")
    row = dataset.load("unused")["rounds"].query("firm == 'A' and t == 1").iloc[0]
    assert row["crossing_content"] and row["crossing_status"] == "positive"
    assert "crossing_content_saved" not in row
