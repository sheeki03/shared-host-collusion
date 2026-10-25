"""Turn Inspect .eval logs into analysis tables with an explicit cohort gate.

Outputs (CSV, one directory per call):
  rounds.csv   one row per firm-round (the shape analysis/SPEC.md expects, plus extra columns)
  games.csv    one row per game: arm, model, seed, mean collusion index (last 10 rounds), crossing counts, cost, errors
  calls.csv    one row per model call: tokens, exact cost, latency, finish reason
  chat.csv     chat messages and compliance verdicts
  gate.csv     gate verdicts
The default loader returns *only* complete confirmatory games. Pilot/self-test samples are recorded in exclusions and
never enter a confirmatory table; errored/incomplete confirmatory games raise unless the caller explicitly requests an
audited exclusions table. This prevents accidental pooling of pilot and confirmatory estimates.
Usage: .venv/bin/python -m analysis.dataset <log_dir> <out_dir> [--cohort confirmatory|pilot|all]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from analysis.coding import INTERNAL_COLUMNS, extract_candidates

SCALARS = ("action", "payoff", "own_priv", "own_com_err", "rival_priv", "rival_com_err", "collusion_index")
# Game-level "welfare" (what stats.welfare_effect compares): the round mean of each engine's primary buyer-side
# measure, signed so that higher is better for the other side of the market.
WELFARE = {"pricing": (("consumer_surplus", "worker_surplus"), 1.0), "auction": (("seller_revenue",), 1.0),
           "auction_procurement": (("buyer_cost",), -1.0), "reporting": (("settlement_distortion",), -1.0),
           "bargaining": (("total_surplus",), 1.0), "market_making": (("trader_cost_per_unit",), -1.0)}
CONFIRMATORY_GRIDS = frozenset({"deep", "broad", "pairs", "scenario"})
PILOT_SEED_BASE = 9000


def _sample_problem(base: dict, rows: list, err: str | None) -> str | None:
    """Why a selected game cannot enter confirmatory inference, or None if it is complete."""
    if err:
        return f"sample error: {err}"
    if not isinstance(rows, list) or not rows:
        return "missing firm-round rows"
    rounds = base.get("rounds")
    if not isinstance(rounds, int) or rounds <= 0:
        return "missing/invalid planned round count"
    if len(rows) != 2 * rounds:
        return f"incomplete game: {len(rows)} rows for {rounds} rounds"
    if {(r.get("t"), r.get("firm")) for r in rows} != {(t, f) for t in range(1, rounds + 1) for f in ("A", "B")}:
        return "missing or duplicate firm-round key"
    if any(isinstance(r.get("action"), bool) or not isinstance(r.get("action"), (int, float))
           or not math.isfinite(r["action"]) for r in rows):
        return "missing/non-finite action"
    llm = [r for r in rows if not r.get("scripted")]
    if not llm:
        return "no subject-model turns"
    if sum(bool(r.get("parse_failed")) for r in llm) / len(llm) > 0.20:
        return "parse-failure rate above preregistered 20% exclusion"
    return None


def _cohort_match(base: dict, cohort: str) -> bool:
    grid = base.get("grid")
    seed = base.get("seed")
    if cohort == "all":
        return True
    if cohort == "pilot":
        return grid == "pilot" or (isinstance(seed, int) and seed >= PILOT_SEED_BASE)
    return grid in CONFIRMATORY_GRIDS and isinstance(seed, int) and 1000 <= seed < PILOT_SEED_BASE


def primary_welfare(engine: str | None, w: dict) -> float | None:
    if not w:
        return None
    if engine == "auction" and "buyer_cost" in w:
        engine = "auction_procurement"
    keys, sign = WELFARE.get(engine or "", ((), 1.0))
    for k in keys:
        if isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool):
            return sign * float(w[k])
    return None


def load(log_dir: str, *, cohort: str = "confirmatory", allow_exclusions: bool = False) -> dict[str, pd.DataFrame]:
    """Load only the requested cohort; default confirmatory rejects incomplete selected games.

    ``allow_exclusions=True`` is for an audited analysis that publishes the returned ``exclusions`` table. It never
    admits an invalid game. ``cohort='all'`` is for offline validation/debugging, never a confirmatory estimate.
    """
    if cohort not in {"confirmatory", "pilot", "all"}:
        raise ValueError("cohort must be confirmatory, pilot or all")
    rounds, games, calls, chat, gate, scratch_events, outputs, candidates, exclusions = [], [], [], [], [], [], [], [], []
    bad_selected = []
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info)
        for s in log.samples or []:
            meta = s.metadata or {}
            res = meta.get("result") or {}
            base = {k: meta.get(k) for k in ("game_id", "grid", "model_label", "model_id", "effort", "scenario_id",
                                             "engine", "arm", "seed", "rounds")}
            err = s.error.message[:300] if s.error else None
            rows = res.get("rows", [])
            if not _cohort_match(base, cohort):
                exclusions.append({**base, "log": str(info.name), "reason": "outside requested cohort"})
                continue
            problem = _sample_problem(base, rows, err)
            if problem is None and log.status != "success":
                problem = f"evaluation status {log.status}"
            game_candidates = pd.DataFrame()
            if problem is None and (cohort != "all" or all(k in res for k in ("chat", "chat_deliveries", "scratch_events"))):
                try:
                    game_candidates = extract_candidates(base, res)
                except (TypeError, ValueError, KeyError) as exc:
                    problem = f"missing/malformed structured communication evidence: {exc}"
            if problem is not None:
                exclusions.append({**base, "log": str(info.name), "reason": problem})
                bad_selected.append((base.get("game_id"), problem))
                continue
            for r in rows:
                flat = {k: r.get(k) for k in r if not isinstance(r.get(k), (dict, list))}
                flat.update({k: v for k, v in base.items() if k not in flat})
                w = r.get("welfare") or {}
                flat.update({f"welfare_{k}": v for k, v in w.items() if isinstance(v, (int, float, bool))})
                flat["welfare"] = primary_welfare(meta.get("engine"), w)
                rounds.append(flat)
            llm = [r for r in rows if not r.get("scripted")]
            last = [r["collusion_index"] for r in llm if r.get("collusion_index") is not None
                    and r["t"] > (meta.get("rounds") or 0) - 10]
            cs = [((r.get("welfare") or {}).get("consumer_surplus")) for r in rows if r["firm"] == "A"]
            cs = [x for x in cs if isinstance(x, (int, float))]
            wel = [primary_welfare(meta.get("engine"), r.get("welfare") or {}) for r in rows if r["firm"] == "A"]
            wel = [x for x in wel if x is not None]
            gcalls = res.get("calls", [])
            games.append({**base, "log": str(info.name), "error": None, "n_rows": len(rows),
                          "collusion_index_last10": (sum(last) / len(last)) if last else None,
                          "consumer_surplus_mean": (sum(cs) / len(cs)) if cs else None,
                          "welfare": (sum(wel) / len(wel)) if wel else None,
                          "canary_crossings": sum(1 for r in llm if r.get("crossing_canary")),
                          "content_crossings": sum(1 for r in llm if r.get("crossing_content")),
                          "name_exposures": sum(1 for r in llm if r.get("name_exposure")),
                          "parse_failures": sum(1 for r in llm if r.get("parse_failed")),
                          "parse_failure_rate": (sum(1 for r in llm if r.get("parse_failed")) / len(llm)) if llm else None,
                          "blocked_commands": sum(r.get("n_blocked") or 0 for r in llm),
                          "cost_usd": sum((c.get("total_cost_usd") if c.get("total_cost_usd") is not None
                                           else c.get("cost_usd") or 0) for c in gcalls),
                          "cost_missing_calls": sum(1 for c in gcalls if c.get("total_cost_usd", c.get("cost_usd"))
                                                    is None),
                          "n_calls": len(gcalls),
                          "mean_call_secs": (sum(c.get("secs") or 0 for c in gcalls) / len(gcalls)) if gcalls else None})
            calls += [{**base, **c} for c in gcalls]
            chat += [{**base, **c} for c in res.get("chat", [])]
            gate += [{**base, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in g.items()}}
                     for g in res.get("gate", [])]
            scratch_events += [{**base, **e} for e in res.get("scratch_events", [])]
            outputs += [{**base, "firm": f, **o} for f, oo in (res.get("outputs") or {}).items() for o in oo]
            candidates.extend(game_candidates.to_dict("records"))
    if bad_selected and not allow_exclusions:
        raise ValueError(f"{len(bad_selected)} invalid {cohort} games; inspect exclusions before analysis: "
                         f"{bad_selected[:4]}")
    out = {"rounds": pd.DataFrame(rounds), "games": pd.DataFrame(games), "calls": pd.DataFrame(calls),
           "chat": pd.DataFrame(chat), "gate": pd.DataFrame(gate), "scratch_events": pd.DataFrame(scratch_events),
           "outputs": pd.DataFrame(outputs), "candidates": pd.DataFrame(candidates, columns=INTERNAL_COLUMNS),
           "exclusions": pd.DataFrame(exclusions)}
    for df in out.values():   # analysis/stats.py keys on scenario / model (PAIR_ON, game_table)
        if "scenario_id" in df:
            df["scenario"] = df["scenario_id"]
        if "model_label" in df:
            df["model"] = df["model_label"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cohort", choices=["confirmatory", "pilot", "all"], default="confirmatory")
    ap.add_argument("--allow-exclusions", action="store_true", help="write an audited exclusions table, never include invalid games")
    args = ap.parse_args()
    log_dir, out_dir = args.log_dir, args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load(log_dir, cohort=args.cohort, allow_exclusions=args.allow_exclusions)
    for name, df in tables.items():
        if name == "candidates":  # internal table may contain private text; export blinded sheet explicitly
            continue
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows -> {out_dir / (name + '.csv')}")


if __name__ == "__main__":
    main()
