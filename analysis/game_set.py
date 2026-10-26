"""Which sample of each game the analysis uses: the outcome-blind duplicate rule of prereg/DEVIATIONS.md D12.

One helper for every reader of the confirmatory logs (analysis.report_outputs, analysis.monitors, and
`load_tables` for any other tool), so they all see exactly the same game set. The rule reads only sample status,
completeness and timestamps, never an outcome:

- A sample is *complete* when it has no sample error and holds every planned firm-round with a finite action
  (`dataset._sample_problem`). A game above the 20% parse-failure threshold counts as complete here; it is then a
  pre-registered exclusion (prereg section 5) of that game, so which copy is kept never depends on how a model played.
- Complete samples count whatever the final status of their log: operator stops and relaunches (D6, D10, D11) leave
  logs marked "started" or "cancelled" that hold finished games.
- A game_id with more than one complete sample keeps the sample that finished first (the sample's `completed_at` in
  the Inspect log). Ties go to the earlier log start (`eval.created`), then to the lexicographically smaller log path.
  Every other complete sample is a dropped duplicate and is listed (`GameSet.duplicates_frame`).
- Incomplete or errored samples of a game that has a complete sample anywhere are *superseded*, not exclusions.
  A game with no complete sample keeps its incomplete samples as exclusions (a harness error).

Fail-closed checks (`strict=True`, used in confirmatory mode): a complete sample of a duplicated game without a
parseable `completed_at` (the first-finished rule could not be applied), the same game twice in one log, and the same
Inspect log (eval_id) at two paths. Lenient mode sorts a missing timestamp last, uses the copy of a log with the most
in-cohort samples (then the smaller path), ignores the other copies, and reports each case in `GameSet.notes`.

CLI (counts and timestamps only, no outcome is read or printed):
  .venv/bin/python -m analysis.game_set runs/full/logs-* runs/remote/*/logs-*
"""
from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence
from unittest import mock

import pandas as pd

from analysis import dataset

PARSE_PREFIX = "parse-failure rate above"     # dataset._sample_problem: a complete game, excluded later (prereg 5)
KEPT, DUPLICATE, SUPERSEDED, INCOMPLETE, LOG_COPY = "kept", "duplicate", "superseded", "incomplete", "log copy"
# Inspect sample fields no analysis reads (dataset.load and monitors.load_games use metadata and error only).
HEAVY_FIELDS = frozenset({"events", "messages"})


class GameSetError(ValueError):
    """The duplicate rule could not be applied without a guess (strict mode)."""


@dataclass(frozen=True)
class SampleRef:
    """One in-cohort sample, described by what the rule may read: identity, status, completeness, timestamps."""
    log: str                     # the log's name as list_eval_logs gives it (a file URI or a path)
    game_id: str
    completed_at: str | None     # the sample's completed_at
    log_started: str | None      # the log's eval.created (or stats.started_at)
    log_status: str | None
    log_eval_id: str | None
    complete: bool
    problem: str | None          # dataset._sample_problem (None: complete with an acceptable parse rate)


@dataclass(frozen=True)
class Duplicate:
    game_id: str
    kept: SampleRef
    dropped: SampleRef


def _ts(value: Any) -> datetime | None:
    """An aware datetime from an ISO timestamp (naive means UTC), or None when missing or unparseable."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value not in (None, "") else None


def log_header(log: Any) -> tuple[str | None, str | None, str | None]:
    """(status, start time, eval_id) of an Inspect log (or a test stand-in with fewer attributes)."""
    ev = getattr(log, "eval", None)
    stats = getattr(log, "stats", None)
    started = _iso(getattr(ev, "created", None)) or _iso(getattr(stats, "started_at", None))
    return getattr(log, "status", None), started, getattr(ev, "eval_id", None)


def sample_problem(sample: Any) -> tuple[dict, dict, str | None]:
    """(base meta, result, dataset._sample_problem) of one sample: the same completeness rule as dataset.load."""
    meta = getattr(sample, "metadata", None) or {}
    res = meta.get("result") or {}
    base = {k: meta.get(k) for k in dataset_keys()}
    err = getattr(sample, "error", None)
    # exactly dataset.load's `s.error.message[:300] if s.error else None`, so both agree on every sample
    msg = (err.message if hasattr(err, "message") else str(err))[:300] if err else None
    return base, res, dataset._sample_problem(base, res.get("rows", []), msg)


def dataset_keys() -> tuple[str, ...]:
    return ("game_id", "grid", "model_label", "model_id", "effort", "scenario_id", "engine", "arm", "seed", "rounds")


def sample_ref(log: Any, log_name: str, sample: Any, *, cohort: str = "confirmatory") -> SampleRef | None:
    """The rule's view of one sample, or None when it is outside the cohort (dataset._cohort_match)."""
    base, _, problem = sample_problem(sample)
    if not dataset._cohort_match(base, cohort) or not isinstance(base.get("game_id"), str):
        return None
    status, started, eval_id = log_header(log)
    complete = problem is None or problem.startswith(PARSE_PREFIX)
    return SampleRef(log=str(log_name), game_id=base["game_id"], completed_at=_iso(getattr(sample, "completed_at", None)),
                     log_started=started, log_status=status, log_eval_id=eval_id, complete=complete, problem=problem)


def finish_key(ref: SampleRef) -> tuple:
    """Sort key of the D12 rule: finished first, then earlier log start, then log path. Missing times sort last."""
    done, start = _ts(ref.completed_at), _ts(ref.log_started)
    far = datetime.max.replace(tzinfo=timezone.utc)
    return (done is None, done or far, start is None, start or far, ref.log)


@dataclass
class GameSet:
    """The selected sample of every game, the dropped duplicates and the superseded samples."""
    refs: list[SampleRef]
    kept: dict[str, SampleRef] = field(default_factory=dict)          # game_id -> the complete sample that enters
    duplicates: list[Duplicate] = field(default_factory=list)
    superseded: list[SampleRef] = field(default_factory=list)
    incomplete: list[SampleRef] = field(default_factory=list)         # games without any complete sample
    ignored_logs: dict[str, str] = field(default_factory=dict)        # later copy of a log -> the copy used
    notes: list[str] = field(default_factory=list)
    _status: dict[tuple[str, str], str] = field(default_factory=dict, repr=False)

    def status(self, log: str, game_id: str) -> str | None:
        """kept / duplicate / superseded / incomplete / log copy, or None for a sample the rule never saw."""
        return self._status.get((str(log), str(game_id)))

    def admits(self, log: str, game_id: str) -> bool:
        """Whether this sample may enter the analysis tables (the kept copy, or the only, incomplete, sample: those
        stay visible as exclusions). Dropped duplicates, superseded samples and ignored log copies do not."""
        return self.status(log, game_id) in (KEPT, INCOMPLETE)

    @property
    def n_duplicates_dropped(self) -> int:
        return len(self.duplicates)

    @property
    def n_complete_samples(self) -> int:
        return sum(1 for r in self.refs if r.complete and str(r.log) not in self.ignored_logs)

    def duplicates_frame(self) -> pd.DataFrame:
        """One row per dropped duplicate: the game, the kept and the dropped copy with their timestamps."""
        cols = ["game_id", "kept_log", "kept_completed_at", "kept_log_started", "dropped_log", "dropped_completed_at",
                "dropped_log_started", "dropped_log_status"]
        rows = [{"game_id": d.game_id, "kept_log": short_log(d.kept.log), "kept_completed_at": d.kept.completed_at,
                 "kept_log_started": d.kept.log_started, "dropped_log": short_log(d.dropped.log),
                 "dropped_completed_at": d.dropped.completed_at, "dropped_log_started": d.dropped.log_started,
                 "dropped_log_status": d.dropped.log_status} for d in self.duplicates]
        return pd.DataFrame(rows, columns=cols)

    def reason(self, log: str, game_id: str) -> str | None:
        """The exclusions-table reason for a sample that does not enter (None for a kept or unknown sample)."""
        st = self.status(log, game_id)
        kept = self.kept.get(str(game_id))
        if st == DUPLICATE and kept is not None:
            mine = next((d.dropped for d in self.duplicates if d.game_id == game_id and d.dropped.log == str(log)), None)
            return (f"dropped duplicate (D12): the copy in {short_log(kept.log)} finished first "
                    f"({kept.completed_at}); this copy finished {mine.completed_at if mine else 'later'}")
        if st == SUPERSEDED and kept is not None:
            return f"superseded (D12): complete copy in {short_log(kept.log)}"
        if st == LOG_COPY:
            return f"second copy of log {short_log(self.ignored_logs.get(str(log), ''))}, ignored"
        return None


def short_log(name: str) -> str:
    """A log name without the file:// prefix, relative to the repository when it lies inside it."""
    s = str(name)
    if s.startswith("file://"):
        s = s[len("file://"):]
    root = str(Path(__file__).resolve().parent.parent) + "/"
    return s[len(root):] if s.startswith(root) else s


def select(refs: Iterable[SampleRef], *, strict: bool = True) -> GameSet:
    """Apply the D12 rule to every in-cohort sample of every log (see the module docstring)."""
    refs = list(refs)
    gs = GameSet(refs=refs)
    by_eval: dict[str, set[str]] = {}
    n_in_log: dict[str, int] = {}
    for r in refs:
        n_in_log[r.log] = n_in_log.get(r.log, 0) + 1
        if r.log_eval_id:
            by_eval.setdefault(r.log_eval_id, set()).add(r.log)
    for eval_id, logs in sorted(by_eval.items()):
        if len(logs) > 1:
            # lenient: the copy with the most in-cohort samples (a fresher sync), then the smaller path
            first, *rest = sorted(logs, key=lambda x: (-n_in_log.get(x, 0), x))
            msg = f"the same Inspect log (eval_id {eval_id}) is at {len(logs)} paths: {[short_log(x) for x in sorted(logs)]}"
            if strict:
                raise GameSetError(msg)
            gs.notes.append(msg + f"; using {short_log(first)}")
            gs.ignored_logs.update({x: first for x in rest})
    seen: set[tuple[str, str]] = set()
    groups: dict[str, list[SampleRef]] = {}
    for r in refs:
        if r.log in gs.ignored_logs:
            gs._status[(r.log, r.game_id)] = LOG_COPY
            continue
        if (r.log, r.game_id) in seen:
            msg = f"game {r.game_id} appears more than once in log {short_log(r.log)}"
            if strict:
                raise GameSetError(msg)
            gs.notes.append(msg + " (later samples ignored)")
            continue
        seen.add((r.log, r.game_id))
        groups.setdefault(r.game_id, []).append(r)
    for gid, group in groups.items():
        complete = sorted((r for r in group if r.complete), key=finish_key)
        if not complete:
            for r in group:
                gs._status[(r.log, gid)] = INCOMPLETE
                gs.incomplete.append(r)
            continue
        if len(complete) > 1:
            undated = [r for r in complete if _ts(r.completed_at) is None]
            if undated:
                msg = (f"game {gid}: {len(undated)} of its {len(complete)} complete samples have no completed_at "
                       f"({[short_log(r.log) for r in undated]}); the first-finished rule cannot be applied")
                if strict:
                    raise GameSetError(msg)
                gs.notes.append(msg + "; undated copies sorted last")
        keep = complete[0]
        gs.kept[gid] = keep
        gs._status[(keep.log, gid)] = KEPT
        for r in complete[1:]:
            gs._status[(r.log, gid)] = DUPLICATE
            gs.duplicates.append(Duplicate(gid, keep, r))
        for r in group:
            if not r.complete:
                gs._status[(r.log, gid)] = SUPERSEDED
                gs.superseded.append(r)
    gs.duplicates.sort(key=lambda d: (d.game_id, d.dropped.log))
    return gs


# ----------------------------------------------------------------------------------------------------------------------
# dataset.load over many logs with the rule applied (report_outputs; any tool that needs dataset tables)
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Loaded:
    tables: dict[str, pd.DataFrame]          # dataset.load tables of the admitted samples (games/exclusions: log, log_dir)
    game_set: GameSet
    hosts: dict[tuple[str, str], Any]        # (log, game_id) -> result.host
    statuses: dict[tuple[str, str], Any]     # (log, game_id) -> final status of its log
    other_calls: pd.DataFrame                # saved calls of in-cohort samples outside the analysis, column 'selection'
    notes: list[str]


TABLES = ("rounds", "games", "calls", "chat", "gate", "scratch_events", "outputs", "candidates", "exclusions")


def read_log(reader: Callable, info: Any, *, trace_relabel: bool = True) -> Any:
    """`reader(info)`; when the reader is Inspect's own read_eval_log, without the per-sample event and message
    streams (HEAVY_FIELDS), which no analysis reads and which are most of a log's size. Saved crossing labels are
    recomputed with the current trace detector (analysis/relabel.py, DEVIATIONS D36). ``trace_relabel=False`` is
    used only by load_tables' injected reader: dataset.load applies the same correction at its read boundary."""
    from analysis import relabel
    if str(getattr(reader, "__module__", "")).startswith("inspect_ai"):
        log = reader(info, exclude_fields=set(HEAVY_FIELDS))
    else:
        log = reader(info)
    return relabel.relabel_log(log) if trace_relabel else log


def load_tables(log_dirs: Sequence[str], *, cohort: str = "confirmatory", strict: bool = True,
                warn: Callable[[str], None] | None = None) -> Loaded:
    """`dataset.load` of every log in `log_dirs` (one log at a time, sorted by name) with the D12 rule applied.

    Complete samples are admitted whatever their log's status. Of a game with several complete samples only the one
    that finished first enters the tables; the others become `exclusions` rows (selection 'duplicate'), and their
    saved calls, with those of superseded and excluded samples, are in `other_calls` (they were billed, so they are
    reconciled against the ledgers, but never analysed). Each `exclusions` row carries a `selection` column (kept,
    duplicate, superseded, incomplete, log copy, or 'outside cohort'). `strict` (confirmatory) fails on an unreadable
    log and on the GameSetError cases; otherwise those are skipped with a warning.
    """
    notes: list[str] = []

    def note(msg: str) -> None:
        notes.append(msg)
        if warn is not None:
            warn(msg)
    real_list, real_read = dataset.list_eval_logs, dataset.read_eval_log
    refs: list[SampleRef] = []
    hosts: dict[tuple[str, str], Any] = {}
    statuses: dict[tuple[str, str], Any] = {}
    saved_calls: dict[tuple[str, str], list] = {}
    per_log: list[tuple[str, str, dict[str, pd.DataFrame]]] = []
    for d in log_dirs:
        for info in _infos_with(real_list, d):
            name = str(getattr(info, "name", info))

            def reader(_info, _name=name):
                try:
                    log = read_log(real_read, _info, trace_relabel=False)
                except Exception as exc:  # a live log can be mid-flush
                    if strict:
                        raise
                    note(f"unreadable log skipped: {short_log(_name)}: {type(exc).__name__}: {str(exc)[:120]}")
                    return SimpleNamespace(status="unreadable", samples=[])
                for s in log.samples or []:
                    ref = sample_ref(log, _name, s, cohort=cohort)
                    if ref is None:
                        continue
                    refs.append(ref)
                    res = (s.metadata or {}).get("result") or {}
                    hosts[(_name, ref.game_id)] = res.get("host")
                    statuses[(_name, ref.game_id)] = log.status
                    if isinstance(res.get("calls"), list) and res["calls"]:
                        saved_calls.setdefault((_name, ref.game_id), []).extend(res["calls"])
                # D12: complete samples are admitted from a log of any status; each sample is still checked.
                return SimpleNamespace(status="success", samples=log.samples)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(dataset, "list_eval_logs", lambda _d, _i=info: [_i]))
                stack.enter_context(mock.patch.object(dataset, "read_eval_log", reader))
                t = dataset.load(str(d), cohort=cohort, allow_exclusions=True)
            per_log.append((name, str(d), t))
    gs = select(refs, strict=strict)
    for n in gs.notes:
        note(n)
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in TABLES}
    other: list[dict] = []
    in_games: set[tuple[str, str]] = set()
    for name, d, t in per_log:
        games = t.get("games", pd.DataFrame())
        admitted = {g for g in (games["game_id"] if not games.empty else []) if gs.admits(name, g)}
        dropped = set(games["game_id"]) - admitted if not games.empty else set()
        in_games |= {(name, g) for g in admitted}
        for k in TABLES:
            df = t.get(k, pd.DataFrame())
            if k == "exclusions":
                continue
            if not df.empty and "game_id" in df.columns and dropped:
                df = df[~df["game_id"].isin(dropped)]
            if k == "games" and not df.empty:
                df = df.assign(log_dir=d)
            parts[k].append(df)
        ex = t.get("exclusions", pd.DataFrame())
        extra = []
        if dropped:
            base_cols = [c for c in dataset_keys() if c in games.columns]
            for rec in games[games["game_id"].isin(dropped)][base_cols].to_dict("records"):
                extra.append({**rec, "scenario": rec.get("scenario_id"), "model": rec.get("model_label"), "log": name,
                              "reason": gs.reason(name, rec["game_id"]) or "not selected (D12)"})
        ex = pd.concat([x for x in (ex, pd.DataFrame(extra)) if not x.empty], ignore_index=True) \
            if (not ex.empty or extra) else ex
        if not ex.empty:
            ex = ex.assign(log_dir=d)
            ex["selection"] = [gs.status(lg, g) or "outside cohort" for lg, g in zip(ex["log"].astype(str),
                                                                                      ex["game_id"].astype(str))]
            # a superseded sample, a log copy, or a dropped copy that dataset.load had already set aside (for
            # example above the parse-failure threshold): keep its own problem, and say why it is not an exclusion
            sup = ex["selection"].isin([SUPERSEDED, LOG_COPY]) | (
                (ex["selection"] == DUPLICATE) & ~ex["reason"].astype(str).str.startswith("dropped duplicate"))
            if sup.any():
                ex.loc[sup, "reason"] = [f"{r} [{gs.reason(lg, g)}]" for r, lg, g in
                                         zip(ex.loc[sup, "reason"], ex.loc[sup, "log"].astype(str),
                                             ex.loc[sup, "game_id"].astype(str))]
        parts["exclusions"].append(ex)
    for (name, gid), calls in saved_calls.items():
        if (name, gid) in in_games:
            continue                       # analysed: these are tables['calls']
        st = gs.status(name, gid)
        sel = st if st in (DUPLICATE, SUPERSEDED, LOG_COPY) else "excluded"
        if st == LOG_COPY:
            continue                       # the same calls as the copy that is used
        other += [{"game_id": gid, "log": name, "selection": sel, **c} for c in calls]
    tables = {k: (pd.concat([x for x in v if not x.empty], ignore_index=True) if any(not x.empty for x in v)
                  else (v[0] if v else pd.DataFrame())) for k, v in parts.items()}
    for k in TABLES:
        tables.setdefault(k, pd.DataFrame())
    if tables["candidates"].empty:
        from analysis.coding import INTERNAL_COLUMNS
        tables["candidates"] = pd.DataFrame(columns=INTERNAL_COLUMNS)
    return Loaded(tables=tables, game_set=gs, hosts=hosts, statuses=statuses,
                  other_calls=pd.DataFrame(other), notes=notes)


def _infos_with(real_list: Callable, log_dir: str) -> list:
    return sorted(real_list(str(log_dir)), key=lambda i: str(getattr(i, "name", i)))


# ----------------------------------------------------------------------------------------------------------------------
# Counts-only audit of real logs (no outcome is read): sample summaries, full reads only for duplicated games
# ----------------------------------------------------------------------------------------------------------------------

def scan(log_dirs: Sequence[str], *, cohort: str = "confirmatory") -> GameSet:
    """Build the GameSet of real logs cheaply: sample summaries for every sample; the full sample (without events
    and messages) only where the rule needs completeness, i.e. games with more than one error-free sample. A game
    with a single error-free summary is counted complete when its summary says completed; `load_tables` and
    `monitors.load_games` apply the full completeness check to every sample."""
    from inspect_ai.log import (list_eval_logs, read_eval_log, read_eval_log_sample,
                                read_eval_log_sample_summaries)
    headers: dict[str, Any] = {}
    summ: list[tuple[str, Any]] = []
    for d in log_dirs:
        for info in sorted(list_eval_logs(str(d)), key=lambda i: str(i.name)):
            headers[info.name] = read_eval_log(info, header_only=True)
            for s in read_eval_log_sample_summaries(info):
                summ.append((info.name, s))
    ok_count: dict[str, int] = {}
    for name, s in summ:
        if not s.error and dataset._cohort_match(s.metadata or {}, cohort):
            gid = (s.metadata or {}).get("game_id") or str(s.id)
            ok_count[gid] = ok_count.get(gid, 0) + 1
    refs: list[SampleRef] = []
    for name, s in summ:
        meta = s.metadata or {}
        if not dataset._cohort_match(meta, cohort):
            continue
        gid = meta.get("game_id") or str(s.id)
        log = headers[name]
        status, started, eval_id = log_header(log)
        if not s.error and ok_count.get(gid, 0) > 1:
            full = read_eval_log_sample(name, id=s.id, epoch=s.epoch, exclude_fields=set(HEAVY_FIELDS))
            ref = sample_ref(log, name, full, cohort=cohort)
            if ref is not None:
                refs.append(ref)
            continue
        problem = f"sample error: {str(s.error)[:300]}" if s.error else (None if s.completed else "not completed")
        refs.append(SampleRef(log=str(name), game_id=gid, completed_at=_iso(s.completed_at), log_started=started,
                              log_status=status, log_eval_id=eval_id, complete=problem is None, problem=problem))
    return select(refs, strict=False)


def audit_lines(gs: GameSet) -> list[str]:
    """Counts, duplicates and the kept copy, with ids and timestamps only."""
    logs: dict[str, dict[str, int]] = {}
    for r in gs.refs:
        c = logs.setdefault(r.log, {"samples": 0, "complete": 0, "status": r.log_status})
        c["samples"] += 1
        c["complete"] += int(r.complete)
    lines = [f"logs {len(logs)}; in-cohort samples {len(gs.refs)}; complete samples {gs.n_complete_samples}; "
             f"games with a complete sample {len(gs.kept)}; dropped duplicates {gs.n_duplicates_dropped}; "
             f"superseded samples {len(gs.superseded)}; games without a complete sample "
             f"{len({r.game_id for r in gs.incomplete})}"]
    for name, c in sorted(logs.items()):
        lines.append(f"  {short_log(name)}: status {c['status']}, samples {c['samples']}, complete {c['complete']}")
    for d in gs.duplicates:
        lines.append(f"  duplicate {d.game_id}: keep {short_log(d.kept.log)} finished {d.kept.completed_at}; "
                     f"drop {short_log(d.dropped.log)} finished {d.dropped.completed_at}")
    lines += [f"  note: {n}" for n in gs.notes]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D12 game-set audit: counts, duplicates and the kept copy (no outcomes)")
    ap.add_argument("log_dirs", nargs="+")
    ap.add_argument("--cohort", choices=("confirmatory", "pilot", "all"), default="confirmatory")
    a = ap.parse_args(argv)
    print("\n".join(audit_lines(scan(a.log_dirs, cohort=a.cohort))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
