"""Run games: pilot, confirmatory grid, or a subset.

Usage examples (from repo root, venv python):
  .venv/bin/python -m harness.run --plan pilot-probes --cap 40 --concurrency 13 --init-ledger
  .venv/bin/python -m harness.run --plan pilot-smoke  --cap 40 --concurrency 12
  .venv/bin/python -m harness.run --plan full --grid deep --cap 400 --previous-cap 40 \
    --raise-cap-reason 'owner-approved post-pilot funding' --concurrency 20
  .venv/bin/python -m harness.run --plan ext --ext-grid all --cap 1 --dry-run      # the 216 extension games

Extensions (prereg/EXTENSIONS.md): `--plan ext --ext-grid {rep,long,any,sx,all}` selects the pre-declared extension
games of harness/arms.py::ext_design() (144 / 24 / 16 / 32; 216 for all). Their ids start with `ext-` and never
collide with full_design(). They write specs to runs/ext/, log to runs/ext/logs and use runs/ext/ledger.jsonl unless
--log-dir / --ledger say otherwise (never the confirmatory runs/full or runs/ledger.jsonl defaults). With
`--ext-grid all` the 100-round ext-long games come first (the critical path), then the other grids interleaved.
--shard-file, --exclude-ids-file, --models and --arms work as for the other plans; --grid does not apply.

Budget: every subject/monitor call reserves the full advertised input-context cost plus bounded output under the
pinned price table before sending, then settles the provider-reported charge. This is a conservative local admission
limit conditional on provider context/prices; an above-reservation charge is recorded and halts future calls.
Pilot seeds (9000+) never enter confirmatory estimates. A ledger is initialized once and resumed explicitly.

End-of-run reconciliation (replay-aware; written Mon 26 Oct 2026, 03:59 IST, before the final analysis runs; the
commit that adds it is the declaration time).
Bookkeeping only and outcome-blind: it reads ledger events and the saved per-attempt cost metadata, never game
outcomes, and changes no game, prompt, selection rule or analysis. It applies to launches from this commit on; runners
already started from pinned commits keep the old whole-run check, and analysis/confirmatory.py's per-attempt
reconciliation remains the audit for every log. With Inspect's retry_on_error (D13) an errored game is replayed from
round 1 and only the final attempt's calls are saved, so the earlier attempt's billed calls are in the ledger but in no
saved sample; the old check (saved calls == ledger delta) then failed although nothing was missing. Now:
- Before the first call the runner marks the end of the last complete ledger line. After the eval, every reservation
  appended since the mark is attributed by its reservation id and the ledger's sample_id ("<sample id>#<epoch>"):
  (1) saved: the id is an attempt of a call saved in a final sample, its closure equals the saved settled_usd, the
      ledger sample_id is that sample's, and each call's total_cost_usd equals the sum of its attempts;
  (2) replayed: not saved, but its sample_id is a final sample that Inspect replayed (non-empty error_retries): an
      attempt billed in a discarded run of that game. Reported separately (USD, attempts, games), never a failure.
  (3) anything else is unexplained and fails the run: a saved attempt with no ledger reservation since the mark, no
      closure or a different amount; an unsaved reservation on a sample that was not replayed, is not in this run's
      logs or has no sample_id; a settlement of a reservation made before the mark (another process on the ledger);
      a malformed ledger line not closed off by a torn_line flag; an open reservation anywhere in the ledger; or a
      ledger spend delta that differs from the window's settlements or from saved + replayed by more than 1e-8 USD.
- Missing costs still fail closed exactly as before (completed_cost): an eval or sample error, a sample without its
  saved game result and call list, or a call without total_cost_usd stops the run before any reconciliation.
- The summary is printed and written to reconciliation_<tag>.json beside the spec file, also when it fails.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from harness.arms import (BROAD, CORE_ROUNDS, DEEP, EXT_GRID_NAMES, ORIGINAL_ARMS, PAIRS, PILOT_SEED_BASE,
                          SCENARIOS, GameSpec, ext_design, full_design)

ROOT = Path(__file__).resolve().parent.parent


def open_run_ledger(path: str, cap: float, *, init: bool = False, previous_cap: float | None = None,
                    raise_reason: str | None = None):
    """Open a paid-run ledger without silently resetting it or changing its persisted cap."""
    from harness.ledger import BudgetLedger

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if init:
        if previous_cap is not None or raise_reason:
            raise ValueError("--init-ledger cannot be combined with a cap increase")
        # O_EXCL makes an accidental second initialization fail even if another runner starts at the same time.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        return BudgetLedger(target, cap, create=True)
    if raise_reason:
        if previous_cap is None:
            raise ValueError("an audited cap increase needs --previous-cap")
        led = BudgetLedger(target, previous_cap, create=False)
        led.raise_cap(cap, reason=raise_reason)
        return led
    if previous_cap is not None:
        raise ValueError("--previous-cap requires --raise-cap-reason")
    return BudgetLedger(target, cap, create=False)


def required_headroom(specs: list[GameSpec], concurrency: int) -> tuple[float, float]:
    """Return largest per-call reservation and one-wave headroom at the requested game concurrency."""
    from harness.game import max_tokens_for
    from harness.nanogpt_provider import estimate_max_cost, load_prices

    prices = load_prices()
    # 64 KB / 12 messages bounds any game prompt (history capped at 5 rounds, notes at 2,000 chars, outputs 2,000).
    monitor = estimate_max_cost(prices["z-ai/glm-5.3"], 8_000, 80, 2)
    reservations = []
    for s in specs:
        if s.model_id not in prices:
            raise ValueError(f"no price bound for {s.model_id}")
        subject = estimate_max_cost(prices[s.model_id], 65_536, max_tokens_for(s.model_id), 12)
        reservations.append(max(subject, monitor))
    if not reservations:
        raise ValueError("the selected plan has no games")
    reservations.sort(reverse=True)
    return reservations[0], sum(reservations[:concurrency])


TOL = Decimal("0.00000001")      # USD; the same tolerance as analysis/confirmatory.py
INCOMPLETE = "incomplete Inspect evaluation; no result may be reported as complete:\n"


@dataclass
class SavedSample:
    key: str            # "<sample id>#<epoch>": the ledger's sample_id for this game's calls
    replays: int        # earlier attempts Inspect discarded and replayed (len(error_retries))
    calls: list[dict]   # the final attempt's saved calls (harness/game.py), each with total_cost_usd and attempts


def _saved_samples(logs) -> tuple[list[SavedSample], list[str]]:
    """The final saved samples of `logs`, and every reason the evaluation is incomplete (fail closed)."""
    from inspect_ai.log import read_eval_log

    out: list[SavedSample] = []
    errors: list[str] = []
    for log in logs:
        if log.status != "success" or log.error:
            errors.append(f"eval status {log.status}: {str(log.error)[:150]}")
        samples = log.samples or read_eval_log(log.location).samples or []
        if not samples:
            errors.append(f"no saved samples in {log.location}")
        for sample in samples:
            if sample.error:
                errors.append(f"sample {sample.id}: {str(sample.error.message)[:150]}")
                continue
            result = (sample.metadata or {}).get("result")
            if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
                errors.append(f"sample {sample.id}: missing game calls/results")
                continue
            calls = result["calls"]
            if any(not isinstance(c, dict) or c.get("total_cost_usd") is None for c in calls):
                errors.append(f"sample {sample.id}: call missing total_cost_usd")
            out.append(SavedSample(f"{sample.id}#{getattr(sample, 'epoch', None)}",
                                   len(getattr(sample, "error_retries", None) or []), calls))
    return out, errors


def completed_cost(logs) -> float:
    """Validate every sample and sum all provider attempts from saved per-game metadata."""
    samples, errors = _saved_samples(logs)
    if errors:
        raise RuntimeError(INCOMPLETE + "\n".join(errors[:20]))
    return sum(float(c["total_cost_usd"]) for s in samples for c in s.calls)


def _usd(x: Any) -> Decimal:
    """Exact Decimal of a saved or ledger USD amount (floats via repr, as harness/ledger.py sums them)."""
    if isinstance(x, bool) or not isinstance(x, (int, float, str, Decimal)):
        raise ValueError(f"not a USD amount: {x!r}")
    try:
        d = Decimal(repr(x)) if isinstance(x, float) else Decimal(x)
    except InvalidOperation as e:
        raise ValueError(f"not a USD amount: {x!r}") from e
    if not d.is_finite() or d < 0:
        raise ValueError(f"not a USD amount: {x!r}")
    return d


@dataclass
class LedgerWindow:
    """The ledger events appended between two marks (byte offsets at line boundaries)."""
    start: int
    end: int
    events: list[dict]
    errors: list[str]


def ledger_mark(path: str | os.PathLike) -> int:
    """Byte offset just past the last complete ledger line: where this run's reconciliation window starts."""
    return Path(path).read_bytes().rfind(b"\n") + 1


def read_ledger_window(path: str | os.PathLike, start: int) -> LedgerWindow:
    """Events on complete lines after `start`. An unterminated last line is not an event yet (as in harness/ledger.py);
    a malformed line counts only if a later torn_line flag closes it off, otherwise it is an error."""
    data = Path(path).read_bytes()
    if len(data) < start:
        return LedgerWindow(start, len(data), [], [f"ledger {path} shrank below the start mark {start}"])
    end = max(start, data.rfind(b"\n") + 1)
    events: list[dict] = []
    torn: set[int] = set()
    pos = start
    for raw in data[start:end].split(b"\n")[:-1]:
        try:
            ev = json.loads(raw)
            if not isinstance(ev, dict) or "event" not in ev:
                raise ValueError("not a ledger event")
        except ValueError:
            torn.add(pos)
        else:
            if ev.get("event") == "flag" and ev.get("reason") == "torn_line":
                torn.discard(ev.get("offset"))
            events.append(ev)
        pos += len(raw) + 1
    errors = [f"malformed ledger line(s) at offset(s) {sorted(torn)[:3]} not closed off by a torn_line flag"] \
        if torn else []
    return LedgerWindow(start, end, events, errors)


@dataclass
class RunReconciliation:
    """Saved calls of one run against the ledger events it appended (see the module docstring)."""
    ledger_delta_usd: Decimal                  # BudgetLedger settled spend after the run minus before it
    window_settled_usd: Decimal                # settlements on the ledger lines appended since the start mark
    saved_usd: Decimal                         # ledger closures of attempts saved in final samples
    saved_calls_usd: Decimal                   # the saved calls' own total_cost_usd (what completed_cost sums)
    replayed_usd: Decimal                      # billed attempts of discarded (replayed) game runs
    replayed_attempts: int
    replayed_games: dict[str, dict]            # sample key -> {replays, attempts, usd}
    n_samples: int
    n_saved_attempts: int
    open_reservations: int                     # anywhere in the ledger (never spend; fail closed)
    flags: dict[str, int]                      # flag lines in the window, by reason class (text before ':')
    unexplained: list[str] = field(default_factory=list)

    @property
    def unexplained_usd(self) -> Decimal:
        return self.ledger_delta_usd - self.saved_usd - self.replayed_usd

    @property
    def ok(self) -> bool:
        return not self.unexplained and abs(self.unexplained_usd) <= TOL and self.open_reservations == 0

    def to_dict(self) -> dict:
        return {"ok": self.ok, "ledger_delta_usd": float(self.ledger_delta_usd),
                "window_settled_usd": float(self.window_settled_usd), "saved_usd": float(self.saved_usd),
                "saved_calls_usd": float(self.saved_calls_usd), "replayed_usd": float(self.replayed_usd),
                "replayed_attempts": self.replayed_attempts, "replayed_games": {
                    k: {**v, "usd": float(v["usd"])} for k, v in sorted(self.replayed_games.items())},
                "unexplained_usd": float(self.unexplained_usd), "n_samples": self.n_samples,
                "n_saved_attempts": self.n_saved_attempts, "open_reservations": self.open_reservations,
                "flags": dict(sorted(self.flags.items())), "unexplained": self.unexplained}


def reconcile_run(logs, window: LedgerWindow, *, ledger_delta: float | Decimal,
                  open_reservations: int = 0) -> RunReconciliation:
    """Attribute every reservation in `window` to a saved attempt, a replayed attempt, or an unexplained difference.

    Raises RuntimeError, as completed_cost does, when the evaluation itself is incomplete (missing costs fail closed).
    """
    samples, errors = _saved_samples(logs)
    if errors:
        raise RuntimeError(INCOMPLETE + "\n".join(errors[:20]))
    problems = list(window.errors)
    reserves: dict[str, dict] = {}
    closures: dict[str, Decimal] = {}
    flags: Counter = Counter()
    window_settled = Decimal(0)
    foreign = [0, Decimal(0)]
    for ev in window.events:
        kind, rid = ev.get("event"), ev.get("id")
        if kind == "reserve":
            reserves[rid] = ev
        elif kind in ("settle", "release"):
            amount = Decimal(0)
            if kind == "settle":
                try:
                    amount = _usd(ev.get("actual"))
                except ValueError:
                    problems.append(f"ledger settle of {rid} has an unreadable amount {ev.get('actual')!r}")
                    continue
                window_settled += amount
            if rid not in reserves:
                foreign[0] += 1
                foreign[1] += amount
                continue
            closures[rid] = amount
        elif kind == "flag":
            flags[str(ev.get("reason") or "unspecified").split(":", 1)[0]] += 1
    if foreign[0]:
        problems.append(f"{foreign[0]} ledger closure(s) (${foreign[1]}) of reservations made before this run "
                        "(another process on this ledger?)")
    delta = ledger_delta if isinstance(ledger_delta, Decimal) else Decimal(repr(float(ledger_delta)))
    if abs(window_settled - delta) > TOL:
        problems.append(f"ledger spend delta ${delta} differs from the settlements appended since the start mark "
                        f"${window_settled}")

    by_key = {s.key: s for s in samples}
    used: set[str] = set()
    saved = saved_calls = Decimal(0)
    for s in samples:
        for call in s.calls:
            step = call.get("step")
            try:
                total = _usd(call.get("total_cost_usd"))
            except ValueError:
                problems.append(f"sample {s.key}: call {step} has an unreadable total_cost_usd")
                continue
            saved_calls += total
            attempts = call.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                problems.append(f"sample {s.key}: call {step} has no per-attempt cost record")
                continue
            amount, complete = Decimal(0), True
            for att in attempts:
                rid = att.get("reservation_id") if isinstance(att, dict) else None
                if not isinstance(rid, str) or not rid or rid in used:
                    problems.append(f"sample {s.key}: call {step} has a missing or duplicate reservation id {rid!r}")
                    complete = False
                    continue
                used.add(rid)
                res = reserves.get(rid)
                if res is None:
                    problems.append(f"sample {s.key}: saved attempt {rid} has no reservation since the start mark")
                    complete = False
                    continue
                if res.get("sample_id") is not None and res.get("sample_id") != s.key:
                    problems.append(f"sample {s.key}: saved attempt {rid} was reserved for {res.get('sample_id')}")
                if rid not in closures:
                    problems.append(f"sample {s.key}: saved attempt {rid} was never settled or released")
                    complete = False
                    continue
                try:
                    differs = abs(closures[rid] - _usd(att.get("settled_usd"))) > TOL
                except ValueError:
                    differs = True
                if differs:
                    problems.append(f"sample {s.key}: saved attempt {rid} settled_usd {att.get('settled_usd')!r} "
                                    f"differs from the ledger ${closures[rid]}")
                amount += closures[rid]
            if complete and abs(amount - total) > TOL:
                problems.append(f"sample {s.key}: call {step} total ${total} is not its attempts' sum ${amount}")
            saved += amount

    replayed: dict[str, dict] = {}
    other: dict[str, list] = defaultdict(lambda: [0, Decimal(0)])
    for rid, res in reserves.items():
        if rid in used:
            continue
        sid = res.get("sample_id")
        if rid not in closures:
            problems.append(f"unsaved reservation {rid} (sample {sid}) was never settled or released")
            continue
        s = by_key.get(sid) if isinstance(sid, str) else None
        if s is not None and s.replays > 0:
            rec = replayed.setdefault(sid, {"replays": s.replays, "attempts": 0, "usd": Decimal(0)})
            rec["attempts"] += 1
            rec["usd"] += closures[rid]
        else:
            why = ("with no sample_id" if not isinstance(sid, str) else
                   f"on sample {sid}, which is not in this run's logs" if s is None else
                   f"on sample {sid}, which Inspect did not replay")
            other[why][0] += 1
            other[why][1] += closures[rid]
    for why, (n, usd) in sorted(other.items()):
        problems.append(f"{n} billed attempt(s) (${usd}) in no saved call, {why}")
    return RunReconciliation(
        ledger_delta_usd=delta, window_settled_usd=window_settled, saved_usd=saved, saved_calls_usd=saved_calls,
        replayed_usd=sum((r["usd"] for r in replayed.values()), Decimal(0)),
        replayed_attempts=sum(r["attempts"] for r in replayed.values()), replayed_games=replayed,
        n_samples=len(samples), n_saved_attempts=len(used), open_reservations=open_reservations,
        flags=dict(flags), unexplained=problems)


def interleave(specs: list[GameSpec], seed: int = 0) -> list[GameSpec]:
    """Round-robin across (model, scenario) streams and, within each, across arms at the same seed, so matched arms
    run close together in time and no arm is systematically early or late."""
    rng = random.Random(seed)
    streams: dict = defaultdict(lambda: defaultdict(list))
    for s in specs:
        streams[(s.model_label, s.scenario_id)][s.seed].append(s)
    queues = []
    for key in sorted(streams):
        q = []
        for sd in sorted(streams[key]):
            batch = streams[key][sd][:]
            rng.shuffle(batch)
            q += batch
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def pilot_probes() -> list[GameSpec]:
    """5-round SHR-AWARE game per model configuration (availability, JSON, latency, billed reasoning)."""
    out, seen = [], set()
    for grid, models in (("deep", DEEP), ("broad", BROAD), ("pairs", PAIRS)):
        for label, mid, effort in models:
            if label in seen:
                continue
            seen.add(label)
            out.append(GameSpec(f"pilot-probe:{label}", "pilot", label, mid, effort, "core-pricing", "pricing",
                                "SHR-AWARE", PILOT_SEED_BASE, 5))
    return out


def pilot_smoke() -> list[GameSpec]:
    """5-round smoke games for every arm on one inexpensive model, plus one full 40-round game for prompt growth."""
    label, mid, effort = ("glm-5.3", "z-ai/glm-5.3", "low")
    out = [GameSpec(f"pilot-smoke:{label}:{arm}", "pilot", label, mid, effort, "core-pricing", "pricing", arm,
                    PILOT_SEED_BASE + 1, 5) for arm in ORIGINAL_ARMS]
    out.append(GameSpec(f"pilot-full:{label}:SHR-AWARE", "pilot", label, mid, effort, "core-pricing", "pricing",
                        "SHR-AWARE", PILOT_SEED_BASE + 2, CORE_ROUNDS))
    for scen, engine in SCENARIOS.items():
        out.append(GameSpec(f"pilot-scenario:{label}:{scen}", "pilot", label, mid, effort, scen, engine, "SHR-AWARE",
                            PILOT_SEED_BASE + 3, 3))
    return out


def ext_plan(ext_grid: str) -> list[GameSpec]:
    """The pre-declared extension games of one grid or all four, checked against the confirmatory ids."""
    specs = ext_design(ext_grid)
    ids = [s.game_id for s in specs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate extension game ids")
    clash = set(ids) & {s.game_id for s in full_design()}
    if clash or not all(s.grid.startswith("ext-") and s.game_id.startswith(s.grid + ":") for s in specs):
        raise SystemExit(f"extension ids must be ext-* and disjoint from full_design(); clash {sorted(clash)[:3]}")
    return specs


def select(plan: str, grid: str | None, models: list[str] | None, arms: list[str] | None,
           ext_grid: str | None = None) -> list[GameSpec]:
    if plan == "pilot-probes":
        specs = pilot_probes()
    elif plan == "pilot-smoke":
        specs = pilot_smoke()
    elif plan == "full":
        specs = full_design()
    elif plan == "ext":
        specs = ext_plan(ext_grid or "all")
    else:
        raise SystemExit(f"unknown plan {plan}")
    if grid:
        specs = [s for s in specs if s.grid == grid]
    if models:
        specs = [s for s in specs if s.model_label in models]
    if arms:
        specs = [s for s in specs if s.arm in arms]
    if plan == "ext":   # prereg/EXTENSIONS.md section 9: the 100-round games are launched first
        return interleave([s for s in specs if s.grid == "ext-long"]) + \
            interleave([s for s in specs if s.grid != "ext-long"])
    return interleave(specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pilot-probes", "pilot-smoke", "full", "ext"])
    ap.add_argument("--grid", choices=["deep", "broad", "pairs", "scenario"])
    ap.add_argument("--ext-grid", choices=[*EXT_GRID_NAMES, "all"],
                    help="with --plan ext (required there): one pre-declared extension grid, or all four")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--cap", type=float, required=True, help="hard USD cap for this ledger")
    ap.add_argument("--ledger", default=None,
                    help="default runs/ledger.jsonl (runs/ext/ledger.jsonl with --plan ext)")
    ap.add_argument("--init-ledger", action="store_true", help="create a new ledger once; never resets existing spend")
    ap.add_argument("--previous-cap", type=float, help="recorded cap before an audited increase")
    ap.add_argument("--raise-cap-reason", help="audited reason for increasing --previous-cap to --cap")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard-file", help="JSON {host, game_ids}: run only these games, in this order")
    ap.add_argument("--exclude-ids-file", help="JSON list of game_ids to skip (already completed elsewhere)")
    a = ap.parse_args()
    if a.concurrency < 1:
        ap.error("--concurrency must be positive")
    ext = a.plan == "ext"
    if ext and not a.ext_grid:
        ap.error("--plan ext needs --ext-grid {" + ",".join([*EXT_GRID_NAMES, "all"]) + "}")
    if a.ext_grid and not ext:
        ap.error("--ext-grid applies only to --plan ext")
    if ext and a.grid:
        ap.error("--grid does not apply to --plan ext (use --ext-grid)")
    if a.ledger is None:
        a.ledger = str(ROOT / "runs" / "ext" / "ledger.jsonl") if ext else str(ROOT / "runs" / "ledger.jsonl")

    specs = select(a.plan, a.grid, a.models, a.arms, a.ext_grid)
    tag = f"ext-{a.ext_grid}" if ext else (a.plan if a.plan != "full" else f"full-{a.grid or 'all'}")
    if a.shard_file:
        shard = json.loads(Path(a.shard_file).read_text())
        by_id = {s.game_id: s for s in specs}
        missing = [g for g in shard["game_ids"] if g not in by_id]
        if missing:
            raise SystemExit(f"shard names {len(missing)} games not in the selected plan, e.g. {missing[:3]}")
        specs = [by_id[g] for g in shard["game_ids"]]
        tag += f"-{shard['host']}"
    if a.exclude_ids_file:
        done = set(json.loads(Path(a.exclude_ids_file).read_text()))
        specs = [s for s in specs if s.game_id not in done]
        tag += "-resume"
    run_dir = ROOT / "runs" / ("ext" if ext else "pilot" if a.plan.startswith("pilot") else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / f"specs_{tag}.json"
    spec_path.write_text(json.dumps([s.to_dict() for s in specs], indent=1))
    print(f"{len(specs)} games -> {spec_path}")
    if ext:
        per = defaultdict(int)
        for s in specs:
            per[s.grid] += 1
        print("per grid: " + ", ".join(f"ext-{n} {per[f'ext-{n}']}" for n in EXT_GRID_NAMES)
              + f"; rounds {sum(s.rounds for s in specs):,}")
    if a.dry_run:
        return

    led = open_run_ledger(a.ledger, a.cap, init=a.init_ledger, previous_cap=a.previous_cap,
                          raise_reason=a.raise_cap_reason)
    peak, wave = required_headroom(specs, a.concurrency)
    print(f"ledger {a.ledger}: cap ${led.cap_usd:.2f}, spent ${led.spent:.4f}, "
          f"outstanding ${led.outstanding:.4f}, largest call reservation ${peak:.4f}, "
          f"one-wave headroom ${wave:.4f}")
    if led.remaining < wave:
        raise SystemExit(f"available budget ${led.remaining:.4f} cannot reserve the requested "
                         f"{a.concurrency}-game wave (${wave:.4f}); lower concurrency or explicitly increase cap")
    mark = ledger_mark(a.ledger)   # start of this run's reconciliation window, before any call of this run
    before = led.spent
    os.environ["SHC_LEDGER_PATH"] = a.ledger
    os.environ["SHC_BUDGET_CAP_USD"] = str(a.cap)
    from inspect_ai import eval as inspect_eval  # noqa: WPS433

    from harness.game import shc_games  # noqa: WPS433
    import harness.nanogpt_provider  # noqa: F401,WPS433  (fail early if the provider is missing)

    log_dir = a.log_dir or str(run_dir / "logs")
    logs = inspect_eval(shc_games(str(spec_path), compose_dir=str(ROOT / "runs" / "compose")),
                        model="mockllm/model",  # each game binds its own nanogpt model in the solver
                        log_dir=log_dir, max_samples=a.concurrency, max_sandboxes=a.concurrency,
                        fail_on_error=False, display="plain", log_realtime=False,
                        log_buffer=5,  # completed games reach the log file every 5 samples (crash loses little)
                        retry_on_error=2)  # a game that errors (e.g. network drop) is replayed from round 1, same seed (D13)
    completed_cost(logs)            # fail closed on any incomplete game or missing saved cost, as before
    window = read_ledger_window(a.ledger, mark)
    rec = reconcile_run(logs, window, ledger_delta=led.spent - before, open_reservations=len(led.open_reservations()))
    report = run_dir / f"reconciliation_{tag}.json"
    report.write_text(json.dumps({"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "ledger": a.ledger, "log_dir": log_dir, "games": len(specs),
                                  "window_bytes": [window.start, window.end], **rec.to_dict()}, indent=1))
    replays = (f"replayed attempts ${rec.replayed_usd:.6f} ({rec.replayed_attempts} attempts on "
               f"{len(rec.replayed_games)} replayed games, D13)")
    if not rec.ok:
        raise SystemExit(f"budget reconciliation failed: ledger delta ${rec.ledger_delta_usd:.6f}, saved attempts "
                         f"${rec.saved_usd:.6f}, {replays}, unexplained ${rec.unexplained_usd:.6f}, open "
                         f"reservations {rec.open_reservations} (details in {report}):\n" +
                         "\n".join(rec.unexplained[:20]))
    print(f"completed {len(specs)} games; provider charges ${rec.ledger_delta_usd:.6f} = saved attempts "
          f"${rec.saved_usd:.6f} + {replays}; ledger flags {led.flags}; reconciliation {report}")


if __name__ == "__main__":
    main()
