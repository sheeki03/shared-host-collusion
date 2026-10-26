"""D38 keeps AI review separate from human validation and reconciles cost components within their own scopes; the
D40 human route (--human-validation-completed) writes no review.ai.* key and keeps the account deferral."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis.coding import _payload_hash
from tools import final_review_inputs as fri


def put_json(path, value):
    path.write_text(json.dumps(value))
    return path


def put_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def inputs(tmp_path):
    key_rows, sheet_rows = [], []
    for i in range(100):
        label = "yes" if i < 50 else "no"
        payload = f"Constructed PRIVATE-CONTENT-CX-SECRET-{i}"
        key_rows.append({"candidate_id": f"PRIVATE-ID-{i}", "item_key": f"PRIVATE-ITEM-{i}", "kind": "chat",
                         "stratum": f"{label}/chat", "llm_label": label, "evidence_incomplete": "False", "payload": payload})
        sheet_rows.append({"candidate_id": f"PRIVATE-ID-{i}", "kind": "chat", "payload": payload,
                           "payload_sha256": _payload_hash(payload), "label": "no" if i < 10 else label,
                           "notes": "PRIVATE-REVIEW-NOTE"})
    key = put_csv(tmp_path / "key.csv", key_rows)
    sheet = put_csv(tmp_path / "sheet.csv", sheet_rows)
    reviewer = put_json(tmp_path / "reviewer.json", dict(fri.REQUIRED_REVIEWER))
    recorded = put_json(tmp_path / "recorded.json", {
        "n_games": 2, "n_calls": 7, "per_game_cost_mismatches": [],
        "cost_by_source": {"usage.cost": {"usd": 12.0, "attempts": 4},
                           "provider_estimate": {"usd": 3.0, "attempts": 3},
                           "retry_reservation:http_error": {"usd": 5.0, "attempts": 2}},
        "call_total_usd": 20.0, "attempts_total_usd": 20.0, "games_csv_total_usd": 20.0,
    })
    breakdown = [{"costUsd": 40.0, "refundedUsd": 0.0, "netCostUsd": 40.0, "requests": 5},
                 {"costUsd": 35.0, "refundedUsd": 5.0, "netCostUsd": 30.0, "requests": 2}]
    usage = put_json(tmp_path / "usage.json", {"usage": {
        "scope": "current_key", "timezone": "UTC", "from": "2026-10-24", "to": "2026-10-26",
        "asOf": "2026-10-26T07:35:29.807Z",
        "totals": {"costUsd": 75.0, "refundedUsd": 5.0, "netCostUsd": 70.0, "requests": 7},
        "byDay": breakdown, "byModel": breakdown,
    }, "balance": {"usd_balance": "PRIVATE-ACCOUNT-BALANCE"}})
    return sheet, key, reviewer, recorded, usage


def test_valid_ai_review_exports_only_aggregate_ai_keys(inputs):
    result = fri.build(*inputs, n_boot=100)
    assert result["review.ai.n"] == 100 and result["review.ai.n_agree"] == 90
    assert result["review.ai.raw_agreement"] == pytest.approx(0.9)
    assert result["review.ai.kappa"] == pytest.approx(0.8)
    assert result["review.ai.reviewer_type"] == "AI" and result["review.ai.human_validation"] is False
    assert result["review.human.status"] == "not performed (D38)"
    assert result["review.account.status"] == "deferred (D38)"
    assert result["review.ai.confusion.reviewer_no.classifier_yes"] == 10
    assert not any(k.startswith(("coding.", "meta.")) for k in result)
    public = json.dumps(result)
    assert "PRIVATE-" not in public and "human_yes" not in public
    assert all(not isinstance(v, (dict, list)) for v in result.values())


@pytest.mark.parametrize("field,value", [("reviewer_type", "human"), ("human_validation", True),
                                        ("human_validation", 0), ("blinded_to_classifier_labels", False),
                                        ("blinded_to_classifier_labels", "true")])
def test_review_metadata_cannot_misrepresent_ai_as_human(inputs, field, value):
    meta = dict(fri.REQUIRED_REVIEWER)
    meta[field] = value
    put_json(inputs[2], meta)
    with pytest.raises(ValueError, match="reviewer metadata requires"):
        fri.build(*inputs, n_boot=1)


def test_missing_metadata_fails_and_explicit_incomplete_stays_pending(inputs):
    inputs[2].unlink()
    with pytest.raises(ValueError, match="metadata must both be present"):
        fri.build(*inputs)
    result = fri.build(*inputs, allow_incomplete=True)
    assert result["review.ai.n"].startswith("pending")
    assert result["review.human.status"] == "not performed (D38)"
    put_json(inputs[2], {})
    with pytest.raises(ValueError, match="reviewer metadata requires"):
        fri.build(*inputs, allow_incomplete=True)


@pytest.mark.parametrize("change", ["blank", "bad_label", "payload", "missing", "incomplete_negative"])
def test_invalid_sheet_never_becomes_an_agreement_result(inputs, change):
    sheet, key = inputs[:2]
    with sheet.open() as f:
        rows = list(csv.DictReader(f))
    if change == "blank":
        rows[0]["label"] = ""
    elif change == "bad_label":
        rows[0]["label"] = "maybe"
    elif change == "payload":
        rows[0]["payload"] += "edited"
    elif change == "missing":
        rows.pop()
    else:
        with key.open() as f:
            keys = list(csv.DictReader(f))
        keys[0]["evidence_incomplete"] = "True"
        put_csv(key, keys)
    put_csv(sheet, rows)
    with pytest.raises(ValueError):
        fri.build(*inputs, n_boot=1)


def test_costs_reconcile_within_scopes_and_are_not_added(inputs):
    result = fri.cost_inputs(*inputs[3:])
    assert result["costs.recorded.nonestimated_usd"] == 12
    assert result["costs.recorded.provider_estimate_usd"] == 3
    assert result["costs.recorded.retry_reservation_usd"] == 5
    assert result["costs.recorded.total_usd"] == 20
    assert result["costs.key_usage.net_usd"] == 70
    assert result["costs.key_usage.scope"] == "current API key only"
    assert not any(k.startswith("meta.account_spend") for k in result)


@pytest.mark.parametrize("change", ["recorded_total", "recorded_count", "per_game", "source", "nan",
                                   "net", "breakdown", "request_count", "scope"])
def test_cost_mismatches_fail_closed(inputs, change):
    recorded, usage_path = inputs[3:]
    d, receipt = fri.read_object(recorded), fri.read_object(usage_path)
    if change == "recorded_total": d["games_csv_total_usd"] += 1
    elif change == "recorded_count": d["n_calls"] += 1
    elif change == "per_game": d["per_game_cost_mismatches"] = ["mismatch"]
    elif change == "source": d["cost_by_source"]["unknown"] = {"usd": 0, "attempts": 0}
    elif change == "nan": d["cost_by_source"]["usage.cost"]["usd"] = float("nan")
    elif change == "net": receipt["usage"]["totals"]["netCostUsd"] += 1
    elif change == "breakdown": receipt["usage"]["byModel"][0]["costUsd"] += 1
    elif change == "request_count": receipt["usage"]["byDay"][0]["requests"] += 1
    elif change == "scope": receipt["usage"]["scope"] = "account"
    put_json(recorded, d)
    put_json(usage_path, receipt)
    with pytest.raises(ValueError):
        fri.cost_inputs(recorded, usage_path)


def test_cli_writes_separate_result_and_cannot_overwrite_private_inputs(inputs, tmp_path):
    sheet, key, reviewer, recorded, usage = inputs
    before = {p: p.read_bytes() for p in inputs}
    args = ["--sheet", str(sheet), "--key", str(key), "--reviewer", str(reviewer),
            "--recorded-costs", str(recorded), "--provider-usage", str(usage)]
    out = tmp_path / "out" / "results.json"
    assert fri.main([*args, "--out", str(out)]) == 0
    assert json.loads(out.read_text())["review.ai.n"] == 100
    for p in inputs:
        assert p.read_bytes() == before[p]
        with pytest.raises(ValueError, match="must not overwrite"):
            fri.main([*args, "--out", str(p)])


@pytest.fixture
def human(inputs, tmp_path):
    sheet = inputs[0]
    human_sheet = tmp_path / "handcheck_sheet.csv"
    human_sheet.write_text(sheet.read_text().replace("PRIVATE-REVIEW-NOTE", "HUMAN-NOTE"))
    agreement = put_json(tmp_path / "agreement.json", {"n": 100, "raw_agreement": 0.9, "kappa": 0.8})
    return agreement, human_sheet


def test_human_route_reports_no_ai_review_and_keeps_account_deferred(inputs, human):
    agreement, human_sheet = human
    inputs[2].unlink()   # the AI reviewer metadata is not needed on the human route
    result = fri.build(*inputs, human_validation_completed=True, agreement=agreement, human_sheet=human_sheet)
    assert result["review.human.status"] == fri.HUMAN_STATUS
    assert result["review.account.status"] == "deferred (D38)"
    assert not any(k.startswith("review.ai.") for k in result)
    assert set(k for k in result if k.startswith("review.")) == {"review.human.status", "review.account.status"}
    assert result["costs.key_usage.scope"] == "current API key only" and result["costs.recorded.total_usd"] == 20
    assert not any(k.startswith(("coding.", "meta.")) for k in result)
    assert "PRIVATE-" not in json.dumps(result) and "HUMAN-NOTE" not in json.dumps(result)


@pytest.mark.parametrize("change", ["no_agreement", "no_sheet", "n", "n_bool", "nan_kappa", "raw_range", "ai_copy"])
def test_human_route_fails_closed(inputs, human, change):
    agreement, human_sheet = human
    if change == "no_agreement":
        agreement.unlink()
    elif change == "no_sheet":
        human_sheet.unlink()
    elif change == "n":
        put_json(agreement, {"n": 99, "raw_agreement": 0.9, "kappa": 0.8})
    elif change == "n_bool":
        put_json(agreement, {"n": True, "raw_agreement": 0.9, "kappa": 0.8})
    elif change == "nan_kappa":
        agreement.write_text('{"n": 100, "raw_agreement": 0.9, "kappa": NaN}')
    elif change == "raw_range":
        put_json(agreement, {"n": 100, "raw_agreement": 1.5, "kappa": 0.8})
    else:
        human_sheet.write_bytes(inputs[0].read_bytes())
    with pytest.raises(ValueError):
        fri.build(*inputs, human_validation_completed=True, agreement=agreement, human_sheet=human_sheet)


def test_cli_human_route_writes_no_ai_keys(inputs, human, tmp_path):
    sheet, key, reviewer, recorded, usage = inputs
    agreement, human_sheet = human
    args = ["--sheet", str(sheet), "--key", str(key), "--reviewer", str(reviewer),
            "--recorded-costs", str(recorded), "--provider-usage", str(usage), "--human-validation-completed",
            "--agreement", str(agreement), "--human-sheet", str(human_sheet)]
    out = tmp_path / "out" / "results.json"
    assert fri.main([*args, "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["review.human.status"] == fri.HUMAN_STATUS
    assert not any(k.startswith("review.ai.") for k in written)
    for p in (agreement, human_sheet):
        with pytest.raises(ValueError, match="must not overwrite"):
            fri.main([*args, "--out", str(p)])
