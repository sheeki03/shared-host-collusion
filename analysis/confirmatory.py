"""Fail-closed bridge from immutable Inspect logs to the preregistered P1/P2/P3 family.

The treatment-blinded review sheet is private research data under ``runs/``. Before inference, every selected
confirmatory sample must be complete, present in the frozen spec manifest, and have adjudicated communication
candidates. The code never substitutes missing labels with negatives.
"""
from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis import coding, dataset, stats

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_RUNS = (ROOT / "runs").resolve()


def _private_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if not p.is_relative_to(PRIVATE_RUNS):
        raise ValueError(f"adjudication files must stay under ignored {PRIVATE_RUNS}")
    return p


def expected_ids(manifests: Iterable[str | Path]) -> set[str]:
    """Read frozen spec manifests; duplicate or non-confirmatory IDs fail closed."""
    ids: set[str] = set()
    for path in manifests:
        specs = json.loads(Path(path).read_text())
        if not isinstance(specs, list):
            raise ValueError(f"manifest {path} must be a list")
        for s in specs:
            if not isinstance(s, dict) or not isinstance(s.get("game_id"), str):
                raise ValueError(f"malformed game spec in {path}")
            if s.get("grid") not in dataset.CONFIRMATORY_GRIDS or not isinstance(s.get("seed"), int) \
                    or not 1000 <= s["seed"] < dataset.PILOT_SEED_BASE:
                raise ValueError(f"non-confirmatory game in manifest {path}: {s['game_id']}")
            if s["game_id"] in ids:
                raise ValueError(f"duplicate game ID across manifests: {s['game_id']}")
            ids.add(s["game_id"])
    if not ids:
        raise ValueError("at least one nonempty frozen spec manifest is required")
    return ids


def load_reviewable(log_dir: str, manifests: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    """Load complete confirmatory logs and prove exact manifest coverage before exporting/using labels."""
    want = expected_ids(manifests)
    tables = dataset.load(log_dir, cohort="confirmatory")
    games = tables["games"]
    got = set(games["game_id"]) if not games.empty else set()
    if got != want or (not games.empty and games["game_id"].duplicated().any()):
        raise ValueError(f"manifest/log mismatch: missing={len(want - got)}, extra={len(got - want)}, "
                         f"duplicate={int(games['game_id'].duplicated().sum()) if not games.empty else 0}")
    return tables


def prepare_sheet(log_dir: str, manifests: Iterable[str | Path], sheet_path: str | Path,
                  *, seed: int = 0) -> Path:
    """Export a treatment-blinded private review sheet. The sheet itself is never committed."""
    tables = load_reviewable(log_dir, manifests)
    return coding.export_blinded(tables["candidates"], _private_path(sheet_path), seed=seed)


def reconcile_call_costs(calls: pd.DataFrame, ledger_path: str | Path) -> Decimal:
    """Match every provider attempt in saved call metadata to its append-only ledger settlement/release.

    Retry charges are included. This is independent of the runner's whole-run reconciliation and detects a game log
    that omitted an attempt or counted only the successful final attempt.
    """
    if calls.empty:
        raise ValueError("no model calls to reconcile")
    charged: dict[str, Decimal] = {}
    released: set[str] = set()
    flagged: set[str] = set()
    with open(ledger_path, encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            rid = event.get("id")
            if event.get("event") == "settle" and rid:
                if rid in charged or rid in released:
                    raise ValueError(f"duplicate ledger closure for reservation {rid}")
                charged[rid] = Decimal(str(event["actual"]))
            elif event.get("event") == "release" and rid:
                if rid in charged or rid in released:
                    raise ValueError(f"duplicate ledger closure for reservation {rid}")
                released.add(rid)
            elif event.get("event") == "flag" and rid:
                flagged.add(rid)
    seen: set[str] = set()
    total = Decimal("0")
    for call in calls.itertuples(index=False):
        attempts = getattr(call, "attempts", None)
        saved_total = getattr(call, "total_cost_usd", None)
        if not isinstance(attempts, list) or saved_total is None:
            raise ValueError("paid call lacks per-attempt metadata or total_cost_usd")
        amount = Decimal("0")
        for attempt in attempts:
            rid = attempt.get("reservation_id")
            if not isinstance(rid, str) or rid in seen:
                raise ValueError("missing/duplicate reservation ID in saved calls")
            seen.add(rid)
            if rid in flagged:
                raise ValueError(f"ledger flagged paid attempt {rid}; audit before confirmatory analysis")
            if rid in charged:
                actual = charged[rid]
            elif rid in released:
                actual = Decimal("0")
            else:
                raise ValueError(f"no ledger closure for saved reservation {rid}")
            if abs(actual - Decimal(str(attempt.get("settled_usd")))) > Decimal("0.00000001"):
                raise ValueError(f"attempt cost differs from ledger for {rid}")
            amount += actual
        if abs(amount - Decimal(str(saved_total))) > Decimal("0.00000001"):
            raise ValueError("call total omitted or mispriced an attempt")
        total += amount
    return total


def analyze(log_dir: str, manifests: Iterable[str | Path], labels_path: str | Path, *,
            ledger_path: str | Path | None = None, n_boot: int = 2000, n_perm: int = 10000,
            seed: int = 0) -> tuple[stats.PrimaryAnalysis, pd.DataFrame]:
    """Return family tests and the coded game table; refuse missing/tampered/unknown labels."""
    tables = load_reviewable(log_dir, manifests)
    if ledger_path is not None:
        reconciled = reconcile_call_costs(tables["calls"], ledger_path)
        logged = sum(Decimal(str(x)) for x in tables["games"]["cost_usd"])
        if abs(reconciled - logged) > Decimal("0.00000001"):
            raise ValueError("game cost totals do not reconcile to provider attempts")
    labels = coding.load_labels(_private_path(labels_path), tables["candidates"], allow_unknown=False)
    round_rows, coded_chat = coding.merge_labels(tables["candidates"], labels, tables["rounds"])
    games = stats.game_table(round_rows)
    games = games.merge(coded_chat[["game_id", "chat_coordination", "n_delivered_coordination"]],
                        on="game_id", validate="one_to_one")
    meta = tables["games"][["game_id", "grid", "error", "n_rows"]].copy()
    if "grid" in games:
        games = games.drop(columns="grid")
    games = games.merge(meta, on="game_id", validate="one_to_one")
    if len(games) != len(meta) or games["error"].notna().any():
        raise ValueError("coded game table omitted or included invalid confirmatory games")
    result = stats.primary_analysis(games, n_boot=n_boot, n_perm=n_perm, seed=seed)
    return result, games


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("prepare", "analyze"))
    ap.add_argument("log_dir")
    ap.add_argument("sheet", help="private review CSV under runs/")
    ap.add_argument("--manifest", action="append", required=True, help="frozen spec JSON; repeat for grid splits")
    ap.add_argument("--ledger", help="append-only paid-call ledger; required to independently reconcile provider costs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "prepare":
        print(prepare_sheet(args.log_dir, args.manifest, args.sheet, seed=args.seed))
        return
    if not args.ledger:
        ap.error("--ledger is required for confirmatory analysis")
    result, games = analyze(args.log_dir, args.manifest, args.sheet, ledger_path=args.ledger, seed=args.seed)
    print(f"coded {len(games)} complete confirmatory games")
    for name in ("P1", "P2", "P3"):
        print(f"{name}: raw p={result.p_raw[name]:.6g}; Holm p={result.p_holm[name]:.6g}")


if __name__ == "__main__":
    main()
