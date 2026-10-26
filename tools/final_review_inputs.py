"""Build public aggregate inputs for the D38 AI second review and deferred account reconciliation.

This is an AI-to-classifier comparison, never the pre-registered human hand-check. The private sheet is validated
against the frozen hand-check key using llm_coder.handcheck_agreement; its human-named internal fields are mapped
only to review.ai.*. No human sheet, coding.handcheck key, candidate text, identifier or review note is exported.
Cost components retain their actual scopes: saved analysed trajectories and an API-key usage receipt. Neither is
presented as a reconciled account-wide bill. Default output: runs/analysis/review_inputs/results.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AI = ROOT / "runs/analysis/ai-review"
DEFAULT_KEY = ROOT / "runs/analysis/coding/handcheck_key.csv"
DEFAULT_COSTS = ROOT / "runs/analysis/costs"
DEFAULT_OUT = ROOT / "runs/analysis/review_inputs/results.json"
AI_STATS = ("n", "n_agree", "raw_agreement", "raw_ci_lo", "raw_ci_hi", "kappa", "kappa_boot_ci_lo",
            "kappa_boot_ci_hi", "kappa_boot_valid", "n_boot")
REQUIRED_REVIEWER = {"reviewer_type": "AI", "human_validation": False, "blinded_to_classifier_labels": True}


def read_object(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return obj


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def agrees(a: float, b: float, name: str) -> None:
    if not math.isclose(a, b, rel_tol=0, abs_tol=1e-6):
        raise ValueError(f"{name} does not reconcile")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ai_review(sheet: Path, key: Path, reviewer: Path, *, allow_incomplete: bool = False,
              n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    if not sheet.exists() or not reviewer.exists():
        if not allow_incomplete:
            raise ValueError("AI review sheet and reviewer metadata must both be present")
        return {"review.ai.status": "pending (AI review not complete)",
                **{f"review.ai.{k}": "pending (AI review not complete)" for k in AI_STATS}}
    meta = read_object(reviewer)
    for k, expected in REQUIRED_REVIEWER.items():
        if k not in meta or type(meta[k]) is not type(expected) or meta[k] != expected:
            raise ValueError(f"reviewer metadata requires {k}={expected!r}")
    with key.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 100 or len({r.get("candidate_id") for r in rows}) != 100:
        raise ValueError("the frozen key must contain 100 unique review items")
    if any(r.get("llm_label") not in {"yes", "no"} for r in rows):
        raise ValueError("the frozen key must contain only classifier-positive or classifier-negative items")
    if sum(r["llm_label"] == "yes" for r in rows) != 50:
        raise ValueError("the frozen key must retain the declared 50 positive / 50 negative sample")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis.llm_coder import handcheck_agreement

    result = handcheck_agreement(sheet, key, n_boot=n_boot, seed=seed)
    if result["n"] != 100:
        raise ValueError("AI agreement must cover all 100 frozen review items")
    public = {"review.ai.status": "completed AI second review (D38)",
              **{f"review.ai.{k}": v for k, v in REQUIRED_REVIEWER.items()},
              "review.ai.sample": "50 classifier-positive and 50 classifier-negative items; descriptive sample agreement",
              "review.ai.sheet_sha256": sha256(sheet), "review.ai.reviewer_metadata_sha256": sha256(reviewer),
              "review.ai.human_validation": False}
    for k in AI_STATS:
        v = result[k]
        public[f"review.ai.{k}"] = v if not isinstance(v, float) or math.isfinite(v) else "n/a (not estimable)"
    public["review.ai.reviewer_unknown"] = result["human_unknown"]
    public["review.ai.classifier_unknown"] = result["llm_unknown"]
    for a in ("yes", "no", "unknown"):
        for b in ("yes", "no", "unknown"):
            public[f"review.ai.confusion.reviewer_{a}.classifier_{b}"] = result["confusion"][f"human_{a}"][f"llm_{b}"]
    return public


def cost_inputs(recorded_path: Path, usage_path: Path) -> dict[str, Any]:
    recorded, receipt = read_object(recorded_path), read_object(usage_path)
    sources = recorded.get("cost_by_source")
    if not isinstance(sources, dict) or not {"usage.cost", "provider_estimate"} <= set(sources):
        raise ValueError("recorded cost sources are missing")
    if recorded.get("per_game_cost_mismatches") != []:
        raise ValueError("recorded per-game costs have unresolved mismatches")
    usd: dict[str, float] = {}
    attempts: dict[str, int] = {}
    for name, data in sources.items():
        if name not in {"usage.cost", "provider_estimate"} and not name.startswith("retry_reservation:"):
            raise ValueError("unrecognized recorded cost source")
        if not isinstance(data, dict):
            raise ValueError("recorded cost source must be an object")
        usd[name] = number(data.get("usd"), f"{name}.usd")
        attempts[name] = count(data.get("attempts"), f"{name}.attempts")
    total = math.fsum(usd.values())
    for name in ("call_total_usd", "attempts_total_usd", "games_csv_total_usd"):
        agrees(total, number(recorded.get(name), name), name)
    n_calls = count(recorded.get("n_calls"), "n_calls")
    if n_calls != attempts["usage.cost"] + attempts["provider_estimate"]:
        raise ValueError("recorded call counts do not reconcile")
    usage = receipt.get("usage")
    if not isinstance(usage, dict) or usage.get("scope") != "current_key":
        raise ValueError("usage receipt must identify current_key scope")
    totals = usage.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("usage totals are missing")
    gross = number(totals.get("costUsd"), "usage.costUsd")
    refunds = number(totals.get("refundedUsd"), "usage.refundedUsd")
    net = number(totals.get("netCostUsd"), "usage.netCostUsd")
    agrees(gross - refunds, net, "usage cost minus refunds")
    n_requests = count(totals.get("requests"), "usage.requests")
    for group in ("byDay", "byModel"):
        parts = usage.get(group)
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"usage {group} breakdown is missing")
        for field in ("costUsd", "refundedUsd", "netCostUsd"):
            agrees(math.fsum(number(x.get(field), f"{group}.{field}") for x in parts),
                   number(totals[field], field), f"usage {group}.{field}")
        if sum(count(x.get("requests"), f"{group}.requests") for x in parts) != n_requests:
            raise ValueError(f"usage {group} request counts do not reconcile")
    start, end, as_of = usage.get("from"), usage.get("to"), usage.get("asOf")
    if date.fromisoformat(start) > date.fromisoformat(end):
        raise ValueError("usage date range is reversed")
    if datetime.fromisoformat(as_of.replace("Z", "+00:00")).tzinfo is None or usage.get("timezone") != "UTC":
        raise ValueError("usage receipt must record its UTC observation time")
    return {
        "costs.recorded.nonestimated_usd": usd["usage.cost"],
        "costs.recorded.provider_estimate_usd": usd["provider_estimate"],
        "costs.recorded.retry_reservation_usd": math.fsum(v for k, v in usd.items() if k.startswith("retry_reservation:")),
        "costs.recorded.total_usd": total, "costs.recorded.n_calls": n_calls,
        "costs.recorded.n_games": count(recorded.get("n_games"), "n_games"),
        "costs.recorded.scope": "final saved trajectories of analysed games only; excludes other attempts and analysis calls",
        "costs.recorded.source_sha256": sha256(recorded_path),
        "costs.key_usage.net_usd": net, "costs.key_usage.gross_usd": gross,
        "costs.key_usage.refunded_usd": refunds, "costs.key_usage.requests": n_requests,
        "costs.key_usage.window_start": start, "costs.key_usage.window_end": end,
        "costs.key_usage.as_of": as_of, "costs.key_usage.scope": "current API key only",
        "costs.key_usage.source_sha256": sha256(usage_path),
        "costs.key_usage.limitation": "whole UTC dates, final day partial; activity not matched request by request to study calls; not account-wide reconciliation",
    }


def build(sheet: Path, key: Path, reviewer: Path, recorded: Path, usage: Path, *, allow_incomplete: bool = False,
          n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    result = {"review.human.status": "not performed (D38)", "review.account.status": "deferred (D38)",
              **ai_review(sheet, key, reviewer, allow_incomplete=allow_incomplete, n_boot=n_boot, seed=seed),
              **cost_inputs(recorded, usage)}
    if any(not k.startswith(("review.", "costs.")) or isinstance(v, (dict, list)) for k, v in result.items()):
        raise ValueError("review inputs must be flat aggregate review/cost keys")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=Path, default=DEFAULT_AI / "review_sheet.csv")
    ap.add_argument("--reviewer", type=Path, default=DEFAULT_AI / "reviewer.json")
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--recorded-costs", type=Path, default=DEFAULT_COSTS / "analysed_game_cost_sources.json")
    ap.add_argument("--provider-usage", type=Path, default=DEFAULT_COSTS / "provider_usage_20261026.json")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--allow-incomplete", action="store_true", help="only missing AI files become explicit pending values; invalid data still fail")
    a = ap.parse_args(argv)
    protected = [a.sheet, a.reviewer, a.key, a.recorded_costs, a.provider_usage,
                 ROOT / "runs/analysis/handcheck/handcheck_sheet.csv", ROOT / "runs/analysis/handcheck/agreement.json",
                 ROOT / "runs/analysis/coding/labels.csv"]
    if a.out.resolve() in {p.resolve() for p in protected}:
        raise ValueError("output must not overwrite an input or human/coder validation artifact")
    result = build(a.sheet, a.key, a.reviewer, a.recorded_costs, a.provider_usage, allow_incomplete=a.allow_incomplete)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=a.out.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, a.out)
    print(f"{len(result)} public aggregate keys -> {a.out}; human validation not performed, account reconciliation deferred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
