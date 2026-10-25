"""Fail-closed bridge from immutable Inspect logs to the preregistered P1/P2/P3 family.

The treatment-blinded review sheet is private research data under ``runs/``. Before inference, every selected
confirmatory sample must be complete, present in the frozen spec manifest, and have adjudicated communication
candidates. The code never substitutes missing labels with negatives.

Ledger reconciliation (`reconcile_game_set`, used by analysis/report_outputs.py): every saved provider attempt of
every call set (the analysed games, and apart the dropped duplicates, superseded and excluded samples of DEVIATIONS
D12, which were billed too) must match one closure in the union of the ledgers (`read_ledgers`). Open reservations
are never spend; flags are admitted only as D8 allows (`ACCEPTED_FLAG_PREFIXES`, behind an explicit switch).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Mapping

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


# Flag reasons --accept-flagged-ledger admits on a saved attempt (DEVIATIONS D8): the provider returned no cost (a
# transport error, or an HTTP error without a cost), so the ledger settled the attempt at its full reservation. A
# charge above the reservation ("over_est_max", a breach) or an unreadable cost ("invalid_cost") is never admitted.
ACCEPTED_FLAG_PREFIXES = ("transport_error", "missing_cost")
TOL = Decimal("0.00000001")


@dataclass
class LedgerBook:
    """The union of several append-only ledgers (harness/ledger.py), read once.

    Files that carry the same ledger ID (its init line) are copies of one ledger (for example synced to two places):
    one must extend the other, and the longer one is used. A reservation ID in two different ledgers fails closed.
    """
    paths: list[str] = field(default_factory=list)            # the files used, one per ledger
    retired: set[str] = field(default_factory=set)            # the subset declared retired
    copies: dict[str, str] = field(default_factory=dict)      # an ignored shorter copy -> the file used
    charged: dict[str, Decimal] = field(default_factory=dict)  # reservation ID -> settled amount
    released: set[str] = field(default_factory=set)
    reserved: dict[str, tuple[Decimal, str]] = field(default_factory=dict)   # ID -> (est_max, ledger file)
    flags: dict[str, list[str]] = field(default_factory=dict)  # reservation ID -> flag reasons
    n_flags: int = 0                                          # every flag line, including torn_line (no ID)
    flag_reasons: Counter = field(default_factory=Counter)    # reason class (text before ':') -> lines

    @property
    def settled_usd(self) -> Decimal:
        return sum(self.charged.values(), Decimal("0"))

    def open_reservations(self) -> dict[str, tuple[Decimal, str]]:
        """Reservations never settled or released (a process stopped mid-call). They are never spend."""
        return {r: v for r, v in self.reserved.items() if r not in self.charged and r not in self.released}

    def open_summary(self) -> dict[str, tuple[int, Decimal]]:
        """{'retired': (n, est_max USD), 'live': (n, USD)}: open reservations on retired and on other ledgers."""
        out = {"retired": [0, Decimal("0")], "live": [0, Decimal("0")]}
        for est, path in self.open_reservations().values():
            k = "retired" if path in self.retired else "live"
            out[k][0] += 1
            out[k][1] += est
        return {k: (n, usd) for k, (n, usd) in out.items()}


def _ledger_events(path: Path) -> tuple[str | None, bytes, list[dict]]:
    """(ledger ID, raw bytes, events) of one file. A malformed line must be covered by a later torn_line flag at its
    offset (the harness closes off a line torn by a writer that died); a torn last line (a sync in progress) is
    ignored. Anything else fails closed."""
    raw = path.read_bytes()
    events, torn, pos = [], {}, 0
    lines = raw.split(b"\n")
    for i, line in enumerate(lines):
        start, pos = pos, pos + len(line) + 1
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
            if not isinstance(ev, dict):
                raise ValueError("not an object")
        except ValueError:
            if i == len(lines) - 1:
                continue                      # unterminated last line: not yet a ledger event
            torn[start] = i
            continue
        if ev.get("event") == "flag" and ev.get("reason") == "torn_line":
            torn.pop(ev.get("offset"), None)
        events.append(ev)
    if torn:
        raise ValueError(f"malformed ledger lines not closed off by a torn_line flag in {path}: offsets {sorted(torn)[:3]}")
    init = next((e for e in events if e.get("event") == "init"), None)
    return (init.get("id") if init else None), raw, events


def read_ledgers(paths: Iterable[str | Path], *, retired: Iterable[str | Path] = ()) -> LedgerBook:
    """Read every ledger once into a LedgerBook (see its docstring). `retired` ledgers belong to stopped launches
    (D6, D10, D11): their open reservations are reported, never counted, and never fail the reconciliation."""
    retired_set = {str(Path(p).resolve()) for p in retired}
    files: dict[str, None] = {}
    for p in [*paths, *retired]:
        files[str(Path(p).resolve())] = None
    groups: dict[str, list[tuple[str, bytes, list[dict]]]] = {}   # ledger ID (or path) -> its copies
    for f in files:
        lid, raw, events = _ledger_events(Path(f))
        groups.setdefault(lid or f, []).append((f, raw, events))
    book = LedgerBook()
    loaded: list[tuple[str, list[dict]]] = []
    for key, members in groups.items():
        members.sort(key=lambda m: (len(m[1]), m[0]))
        for (fa, ra, _), (fb, rb, _) in zip(members, members[1:]):   # each copy must extend the next shorter one
            if not rb.startswith(ra):
                raise ValueError(f"ledgers {fa} and {fb} carry the same ledger ID {key} but neither extends the other")
        used, _, events = members[-1]
        for f, _, _ in members[:-1]:
            book.copies[f] = used
        loaded.append((used, events))
        if any(f in retired_set for f, _, _ in members):   # retired when any copy of it is declared retired
            book.retired.add(used)
    for f, events in loaded:
        book.paths.append(f)
        for ev in events:
            kind, rid = ev.get("event"), ev.get("id")
            if kind == "flag":
                book.n_flags += 1
                reason = str(ev.get("reason") or "")
                book.flag_reasons[reason.split(":", 1)[0] or "unspecified"] += 1
                if rid:
                    book.flags.setdefault(rid, []).append(reason)
            elif kind == "reserve" and rid:
                if rid in book.reserved:
                    raise ValueError(f"reservation {rid} appears in two ledgers ({book.reserved[rid][1]}, {f})")
                book.reserved[rid] = (Decimal(str(ev.get("est_max") or 0)), f)
            elif kind in ("settle", "release") and rid:
                if rid in book.charged or rid in book.released:
                    raise ValueError(f"duplicate ledger closure for reservation {rid}")
                if kind == "settle":
                    book.charged[rid] = Decimal(str(ev["actual"]))
                else:
                    book.released.add(rid)
    return book


def flag_accepted(reasons: Iterable[str]) -> bool:
    """Whether every flag on one attempt is a conservative settlement --accept-flagged-ledger admits (D8)."""
    reasons = list(reasons)
    return bool(reasons) and all(str(r).startswith(ACCEPTED_FLAG_PREFIXES) for r in reasons)


def reconcile_attempts(calls: pd.DataFrame, book: LedgerBook, *, accept_flags: bool = False,
                       seen: set[str] | None = None) -> tuple[Decimal, int]:
    """(reconciled USD, flagged attempts admitted) for the saved calls `calls` against the ledger union `book`.

    Every attempt (retries included) must have exactly one closure: settled at the saved amount, or released (0). A
    reservation ID saved twice (also across call sets, through `seen`) fails, and so does a call whose total is not
    the sum of its attempts. A flagged attempt fails unless `accept_flags` and every flag reason on it is in
    ACCEPTED_FLAG_PREFIXES.
    """
    seen = set() if seen is None else seen
    total, n_flagged = Decimal("0"), 0
    for call in calls.itertuples(index=False):
        attempts = getattr(call, "attempts", None)
        saved_total = getattr(call, "total_cost_usd", None)
        if not isinstance(attempts, list) or saved_total is None or (isinstance(saved_total, float)
                                                                     and math.isnan(saved_total)):
            raise ValueError("paid call lacks per-attempt metadata or total_cost_usd")
        amount = Decimal("0")
        for attempt in attempts:
            rid = attempt.get("reservation_id")
            if not isinstance(rid, str) or rid in seen:
                raise ValueError("missing/duplicate reservation ID in saved calls")
            seen.add(rid)
            if rid in book.flags:
                if not (accept_flags and flag_accepted(book.flags[rid])):
                    raise ValueError(f"ledger flagged paid attempt {rid} ({', '.join(book.flags[rid])}); audit "
                                     "before confirmatory analysis")
                n_flagged += 1
            if rid in book.charged:
                actual = book.charged[rid]
            elif rid in book.released:
                actual = Decimal("0")
            else:
                raise ValueError(f"no ledger closure for saved reservation {rid}")
            if abs(actual - Decimal(str(attempt.get("settled_usd")))) > TOL:
                raise ValueError(f"attempt cost differs from ledger for {rid}")
            amount += actual
        if abs(amount - Decimal(str(saved_total))) > TOL:
            raise ValueError("call total omitted or mispriced an attempt")
        total += amount
    return total, n_flagged


def reconcile_call_costs(calls: pd.DataFrame, ledger_path: str | Path) -> Decimal:
    """Match every provider attempt in saved call metadata to its append-only ledger settlement/release.

    Retry charges are included. This is independent of the runner's whole-run reconciliation and detects a game log
    that omitted an attempt or counted only the successful final attempt.
    """
    if calls.empty:
        raise ValueError("no model calls to reconcile")
    return reconcile_attempts(calls, read_ledgers([ledger_path]))[0]


@dataclass
class Reconciliation:
    usd: dict[str, Decimal]          # call set -> reconciled USD (kept, duplicate, superseded, excluded)
    n_calls: dict[str, int]
    flagged_admitted: int            # saved attempts with an admitted flag (all call sets)
    book: LedgerBook

    @property
    def all_usd(self) -> Decimal:
        return sum(self.usd.values(), Decimal("0"))


def reconcile_game_set(call_sets: Mapping[str, pd.DataFrame], ledgers: Iterable[str | Path], *,
                       retired: Iterable[str | Path] = (), accept_flags: bool = False) -> Reconciliation:
    """Reconcile every saved call of every call set against the union of `ledgers` and `retired` (DEVIATIONS D12).

    `call_sets` maps a name to saved calls: 'kept' (the games analysed) must not be empty; the calls of dropped
    duplicates, superseded and excluded samples were billed too, so they are reconciled with the same rules (one
    shared set of seen reservation IDs) but kept apart. Open reservations are never spend; the caller decides what
    open reservations on ledgers not declared retired mean.
    """
    book = read_ledgers(ledgers, retired=retired)
    if "kept" in call_sets and call_sets["kept"].empty:
        raise ValueError("no model calls to reconcile")
    seen: set[str] = set()
    usd, n_calls, flagged = {}, {}, 0
    for name, calls in call_sets.items():
        total, nf = reconcile_attempts(calls, book, accept_flags=accept_flags, seen=seen) if not calls.empty \
            else (Decimal("0"), 0)
        usd[name], n_calls[name], flagged = total, int(len(calls)), flagged + nf
    return Reconciliation(usd=usd, n_calls=n_calls, flagged_admitted=flagged, book=book)


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
