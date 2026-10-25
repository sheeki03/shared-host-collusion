"""Sanitized run manifest: where every game ran, which copy the analysis uses, and what the logs cover ($0, read-only).

For each in-cohort sample of the given Inspect log directories it records:
  - the game (design keys only: grid, model label/id, effort, scenario, engine, arm, seed, rounds);
  - the host label and run tag, taken from the log's path: `<root>/runs/remote/<host>/logs-<tag>/` is a synced remote
    host, anything else (e.g. `runs/full/logs-<tag>/`) is the local machine (`--local-host`, default mac);
  - the launch: the Inspect log file (repository-relative path), its eval_id, run_id, creation time and final status;
  - the sample's start and finish times, whether it is complete, why not (a fixed category, or the exception class of
    a sample error, never its message), and its D12 selection from `analysis.game_set.select`: kept, duplicate,
    superseded, incomplete or log copy (the same completeness check as `game_set.load_tables`);
  - the recorded machine (`result.host`, the runner's hostname) as an opaque alias (m1, m2, ...), so samples that ran
    on one machine can be matched without publishing hostnames.
Coverage against the design (`--design full`: harness.arms.full_design): per seed and per matched set (model,
scenario, seed, as analysis.stats.PAIR_ON), with the host labels and machine aliases of the kept copies, so matched
sets split across hosts are visible. Optional inputs: shard files (`--shard`, each game's planned assignment and each
file's sha256) and ledgers (`--ledger`: per file its ID, event counts, settled USD, open reservations, flag categories,
and which files are byte-prefix copies of another with the same ledger ID; never per-call rows).

Never written: model text of any kind (messages, events, chat, notes, rationales, scratch contents), prompts, inbox
contents and canaries, error messages, ledger or cap-change reason text beyond a fixed category, hostnames, IP
addresses, and absolute paths (paths outside the repository are reduced to `<parent>/<name>`). The writer fails
closed if the serialized output contains a canary pattern (CX- and 12 hex digits) or the user's home directory.

Reads logs, shard files and ledgers only. Writes run_manifest.json, samples.csv, seeds.csv and matched_sets.csv into
`--out-dir`, which may not lie inside an input log directory; existing files are replaced only with `--overwrite`.
Strict by default (an unreadable log, a hidden staging copy or a D12 ambiguity fails); `--lenient` notes them instead
(for logs still being written).

Usage (from the repository root):
  .venv/bin/python tools/run_manifest.py --out-dir <dir> [--cohort confirmatory|pilot|all] [--design full|none]
      [--local-host mac] [--shard <shard.json> ...] [--ledger <ledger.jsonl> ...] [--lenient] [--overwrite]
      LOG_DIR [LOG_DIR ...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import game_set  # noqa: E402

SCHEMA = "shc-run-manifest/1"
CANARY_RE = re.compile(r"CX-[0-9a-fA-F]{12}")
ERROR_CLASS_RE = re.compile(r"\s*([A-Z][A-Za-z0-9_]{0,63})\(")
ERROR_CLASS_SUFFIX = ("Error", "Exception", "Exceeded", "Timeout", "Cancelled", "Interrupt", "Exit", "Unavailable")
DESIGN_KEYS = ("grid", "model_label", "model_id", "effort", "scenario_id", "engine", "arm", "seed", "rounds")
MATCH_KEYS = ("model_label", "scenario_id", "seed")          # analysis.stats.PAIR_ON (model, scenario, seed)
PROBLEMS = {                                                  # analysis.dataset._sample_problem -> category
    "missing firm-round rows": "missing_rows",
    "missing/invalid planned round count": "invalid_round_count",
    "missing or duplicate firm-round key": "bad_round_keys",
    "missing/non-finite action": "bad_action",
    "no subject-model turns": "no_subject_turns",
    "not completed": "not_completed",
}
SAMPLE_COLUMNS = ("game_id", "epoch", "grid", "model_label", "scenario_id", "arm", "seed", "host", "tag", "machine",
                  "log", "log_eval_id", "log_run_id", "log_created", "log_status", "started_at", "completed_at",
                  "complete", "problem", "selection")


class ManifestError(RuntimeError):
    """The manifest could not be built without a guess, or would not be sanitized."""


# ------------------------------------------------------------------------------------------------------- helpers
def rel_path(p: str | Path) -> str:
    """A repository-relative path, or `<parent>/<name>` for a path outside the repository (never absolute)."""
    s = str(p)
    if s.startswith("file://"):
        s = s[len("file://"):]
    path = Path(s).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return f"{path.parent.name}/{path.name}"


def host_and_tag(log_file: str | Path, local_host: str) -> tuple[str, str]:
    """(host label, run tag) from a log's path: runs/remote/<host>/logs-<tag>/x.eval, else the local host."""
    s = str(log_file)
    if s.startswith("file://"):
        s = s[len("file://"):]
    d = Path(s).parent
    tag = d.name[len("logs-"):] if d.name.startswith("logs-") and len(d.name) > len("logs-") else d.name
    host = d.parent.name if d.parent.parent.name == "remote" else local_host
    return host, tag


def error_class(err: Any) -> str:
    """The exception class of a sample error (e.g. ModelGenerateError), never its message."""
    if err is None:
        return ""
    msg = err.message if hasattr(err, "message") else str(err)
    m = ERROR_CLASS_RE.match(str(msg))
    return m.group(1) if m and m.group(1).endswith(ERROR_CLASS_SUFFIX) else "unclassified"


def problem_kind(problem: str | None, err: Any) -> str:
    """A fixed category for analysis.dataset._sample_problem's reason ('' when complete)."""
    if problem is None:
        return ""
    if problem.startswith(game_set.PARSE_PREFIX):
        return "parse_rate_above_threshold"
    if problem.startswith("sample error"):
        return f"sample_error:{error_class(err)}"
    if problem.startswith("incomplete game"):
        return "incomplete_game"
    return PROBLEMS.get(problem, "incomplete")


def flag_kind(reason: Any) -> str:
    """A ledger flag reason reduced to its category (`transport_error:ReadError`, `missing_cost:http_503`,
    `invalid_cost`): never the free text some reasons carry."""
    parts = str(reason or "unspecified").split(":")
    head = parts[0] if re.fullmatch(r"[a-z_]{1,40}", parts[0]) else "other"
    if len(parts) > 1 and head != "invalid_cost" and re.fullmatch(r"[A-Za-z0-9_]{1,40}", parts[1]):
        return f"{head}:{parts[1]}"
    return head


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(v: Any) -> str | None:
    return game_set._iso(v)


# --------------------------------------------------------------------------------------------------------- logs
@dataclass
class Scan:
    refs: list = field(default_factory=list)                   # game_set.SampleRef, in reading order
    rows: list[dict] = field(default_factory=list)             # one per ref, same order (sanitized fields)
    logs: list[dict] = field(default_factory=list)             # one per log file read
    notes: list[str] = field(default_factory=list)


def scan_logs(log_dirs: Sequence[str | Path], *, cohort: str = "confirmatory", local_host: str = "mac",
              strict: bool = True, lister: Callable | None = None, reader: Callable | None = None) -> Scan:
    """Read every log (without per-sample events and messages) and keep only the sanitized fields."""
    if lister is None or reader is None:
        from inspect_ai.log import list_eval_logs, read_eval_log
        lister = lister or list_eval_logs
        reader = reader or (lambda info: read_eval_log(info, exclude_fields=set(game_set.HEAVY_FIELDS)))
    out = Scan()
    aliases: dict[str, str] = {}
    for d in log_dirs:
        if not Path(d).is_dir():
            msg = f"log directory missing: {rel_path(d)}"
            if strict:
                raise ManifestError(msg)
            out.notes.append(msg)
            continue
        for info in sorted(lister(str(d)), key=lambda i: str(getattr(i, "name", i))):
            name = str(getattr(info, "name", info))
            local = name[len("file://"):] if name.startswith("file://") else name
            if any(part.startswith(".") for part in Path(local).parts[-4:-1]):
                msg = f"log in a hidden (staging) directory: {rel_path(name)}; pass the promoted logs-* directories"
                if strict:
                    raise ManifestError(msg)
                out.notes.append(msg + " (skipped)")
                continue
            try:
                log = reader(info)
            except Exception as exc:  # noqa: BLE001 - a live log can be mid-flush
                msg = f"unreadable log: {rel_path(name)}: {type(exc).__name__}"
                if strict:
                    raise ManifestError(msg) from exc
                out.notes.append(msg + " (skipped)")
                continue
            status, created, eval_id = game_set.log_header(log)
            run_id = getattr(getattr(log, "eval", None), "run_id", None)
            host, tag = host_and_tag(name, local_host)
            path = Path(local)
            entry = {"log": rel_path(name), "host": host, "tag": tag, "eval_id": eval_id, "run_id": run_id,
                     "created": created, "status": status,
                     "bytes": path.stat().st_size if path.is_file() else None,
                     "sha256": sha256_file(path) if path.is_file() else None,
                     "n_samples": 0, "n_in_cohort": 0}
            for s in getattr(log, "samples", None) or []:
                entry["n_samples"] += 1
                ref = game_set.sample_ref(log, name, s, cohort=cohort)
                if ref is None:
                    continue
                entry["n_in_cohort"] += 1
                meta = getattr(s, "metadata", None) or {}
                res = meta.get("result") if isinstance(meta.get("result"), dict) else {}
                recorded = res.get("host")
                machine = None
                if isinstance(recorded, str) and recorded:
                    machine = aliases.setdefault(recorded, f"m{len(aliases) + 1}")
                out.refs.append(ref)
                out.rows.append({
                    "game_id": ref.game_id, "epoch": getattr(s, "epoch", None),
                    **{k: meta.get(k) for k in DESIGN_KEYS},
                    "host": host, "tag": tag, "machine": machine, "log": entry["log"], "log_eval_id": eval_id,
                    "log_run_id": run_id, "log_created": created, "log_status": status,
                    "started_at": _iso(getattr(s, "started_at", None)), "completed_at": ref.completed_at,
                    "complete": bool(ref.complete), "problem": problem_kind(ref.problem, getattr(s, "error", None)),
                })
            out.logs.append(entry)
    return out


# ------------------------------------------------------------------------------------------------ shards, ledgers
def read_shards(paths: Iterable[str | Path]) -> tuple[list[dict], dict[str, list[str]]]:
    """([{path, sha256, bytes, host, n_games, rounds}], game_id -> shard paths listing it)."""
    files, assigned = [], defaultdict(list)
    for p in paths:
        path = Path(p)
        data = json.loads(path.read_text())
        ids = data.get("game_ids") if isinstance(data, dict) else data
        if not isinstance(ids, list) or not all(isinstance(g, str) for g in ids):
            raise ManifestError(f"shard file without a game_ids list: {rel_path(path)}")
        r = rel_path(path)
        files.append({"path": r, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                      "host": data.get("host") if isinstance(data, dict) else None, "n_games": len(ids),
                      "rounds": data.get("rounds") if isinstance(data, dict) else None})
        for g in ids:
            if r not in assigned[g]:
                assigned[g].append(r)
    return files, dict(assigned)


def read_ledgers(paths: Iterable[str | Path]) -> list[dict]:
    """Per-ledger counts with the analysis' own parser (analysis.confirmatory._ledger_events). A file whose bytes are
    a prefix of a longer file with the same ledger ID is marked `copy_of` it (the analysis reads only the longer)."""
    from analysis.confirmatory import _ledger_events

    seen: dict[str, None] = {}
    for p in paths:
        seen[str(Path(p).resolve())] = None
    loaded = []
    for f in seen:
        try:
            lid, raw, events = _ledger_events(Path(f))
        except (OSError, ValueError) as exc:
            raise ManifestError(f"ledger {rel_path(f)} cannot be read: {type(exc).__name__}") from exc
        loaded.append((f, lid, raw, events))
    by_id: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    for f, lid, raw, _ in loaded:
        if lid:
            by_id[lid].append((f, raw))
    used: dict[str, tuple[str, bytes]] = {}       # ledger ID -> the copy the analysis reads (longest, then path)
    for lid, members in by_id.items():
        used[lid] = sorted(members, key=lambda m: (len(m[1]), m[0]))[-1]
    out = []
    for f, lid, raw, events in loaded:
        reserved: dict[str, Decimal] = {}
        closed: set[str] = set()
        settled, n_settle, n_release, flags, caps, first, last = Decimal(0), 0, 0, Counter(), [], None, None
        for ev in events:
            kind, ts = ev.get("event"), ev.get("ts")
            first, last = first or ts, ts or last
            if kind == "init":
                caps.append(str(ev.get("cap")))
            elif kind == "cap_change":
                caps.append(str(ev.get("cap")))
            elif kind == "reserve" and ev.get("id"):
                reserved[ev["id"]] = Decimal(str(ev.get("est_max") or 0))
            elif kind == "settle" and ev.get("id"):
                closed.add(ev["id"])
                n_settle += 1
                settled += Decimal(str(ev.get("actual") or 0))
            elif kind == "release" and ev.get("id"):
                closed.add(ev["id"])
                n_release += 1
            elif kind == "flag":
                flags[flag_kind(ev.get("reason"))] += 1
        open_ids = [r for r in reserved if r not in closed]
        copy_of = conflict = None
        if lid and used[lid][0] != f:
            if used[lid][1].startswith(raw):
                copy_of = used[lid][0]
            else:
                conflict = rel_path(used[lid][0])   # the analysis refuses this pair (neither extends the other)
        out.append({"path": rel_path(f), "ledger_id": lid, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(), "cap_usd": caps[-1] if caps else None,
                    "n_cap_changes": max(len(caps) - 1, 0), "first_ts": first, "last_ts": last,
                    "n_reserve": len(reserved), "n_settle": n_settle, "n_release": n_release,
                    "settled_usd": str(settled), "n_open": len(open_ids),
                    "open_est_max_usd": str(sum((reserved[r] for r in open_ids), Decimal(0))),
                    "flags": dict(sorted(flags.items())), "copy_of": rel_path(copy_of) if copy_of else None,
                    "conflicts_with": conflict})
    return out


# ------------------------------------------------------------------------------------------------------ manifest
def build(log_dirs: Sequence[str | Path], *, cohort: str = "confirmatory", design: str = "full",
          local_host: str = "mac", strict: bool = True, shards: Sequence[str | Path] = (),
          ledgers: Sequence[str | Path] = (), lister: Callable | None = None,
          reader: Callable | None = None) -> dict:
    """The manifest as a JSON-ready dict (see the module docstring)."""
    scan = scan_logs(log_dirs, cohort=cohort, local_host=local_host, strict=strict, lister=lister, reader=reader)
    try:
        gs = game_set.select(scan.refs, strict=strict)
    except game_set.GameSetError as exc:
        raise ManifestError(f"D12 selection failed: {exc}") from exc
    for row, ref in zip(scan.rows, scan.refs):
        row["selection"] = gs.status(ref.log, ref.game_id) or "ignored (same game twice in one log)"
    planned: dict[str, dict] = {}
    if design == "full":
        from harness.arms import full_design
        from analysis import dataset

        for s in full_design():
            d = s.to_dict()
            if dataset._cohort_match(d, cohort):
                planned[s.game_id] = {k: d[k] for k in DESIGN_KEYS}
    shard_files, assigned = read_shards(shards) if shards else ([], {})
    ledger_rows = read_ledgers(ledgers) if ledgers else []

    by_game: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(scan.rows):
        by_game[row["game_id"]].append(i)
    games: dict[str, dict] = {}
    for gid in sorted(set(planned) | set(by_game)):
        idx = by_game.get(gid, [])
        kept = next((i for i in idx if scan.rows[i]["selection"] == game_set.KEPT), None)
        design_keys = planned.get(gid) or {k: scan.rows[idx[0]][k] for k in DESIGN_KEYS}
        state = "kept" if kept is not None else ("incomplete_only" if idx else "missing")
        games[gid] = {**design_keys, "planned": gid in planned, "state": state, "n_samples": len(idx),
                      "n_complete": sum(1 for i in idx if scan.rows[i]["complete"]),
                      "kept_sample": kept,
                      "kept": None if kept is None else {k: scan.rows[kept][k] for k in
                                                          ("host", "tag", "machine", "log", "completed_at")},
                      "samples": idx}
        if shards:
            games[gid]["assigned_in"] = assigned.get(gid, [])
    universe = planned if design == "full" else games

    def tally(keys: Iterable[str]) -> dict:
        c = Counter(games[g]["state"] for g in keys)
        return {"planned": sum(1 for g in keys if games[g]["planned"]), "kept": c["kept"],
                "incomplete_only": c["incomplete_only"], "missing": c["missing"]}

    seeds: dict[Any, list[str]] = defaultdict(list)
    sets: dict[tuple, list[str]] = defaultdict(list)
    for gid in universe:
        g = games[gid]
        seeds[g["seed"]].append(gid)
        sets[tuple(g[k] for k in MATCH_KEYS)].append(gid)
    seed_rows = []
    for seed in sorted(seeds, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x))):
        ids = seeds[seed]
        kept = [games[g]["kept"] for g in ids if games[g]["kept"]]
        seed_rows.append({"seed": seed, **tally(ids), "hosts": sorted({k["host"] for k in kept}),
                          "machines": sorted({k["machine"] for k in kept if k["machine"]})})
    set_rows = []
    for key in sorted(sets, key=lambda k: tuple(str(x) for x in k)):
        ids = sets[key]
        kept = [games[g]["kept"] for g in ids if games[g]["kept"]]
        arms = sorted({games[g]["arm"] for g in ids})
        kept_arms = sorted({games[g]["arm"] for g in ids if games[g]["kept"]})
        hosts = sorted({k["host"] for k in kept})
        machines = sorted({k["machine"] for k in kept if k["machine"]})
        set_rows.append({**dict(zip(MATCH_KEYS, key)), "grids": sorted({games[g]["grid"] for g in ids}),
                         "n_games": len(ids), "arms": arms, "kept_arms": kept_arms,
                         "missing_arms": [a for a in arms if a not in kept_arms],
                         "complete": kept_arms == arms, "hosts": hosts, "machines": machines,
                         "single_host": len(hosts) == 1 and len(machines) <= 1})
    tags: dict[str, Counter] = defaultdict(Counter)
    for row in scan.rows:
        c = tags[f"{row['host']}/{row['tag']}"]
        c["samples"] += 1
        c["complete"] += int(row["complete"])
        c[str(row["selection"])] += 1
    complete_sets = [r for r in set_rows if r["complete"]]
    totals = {**tally(universe), "observed_games": len(by_game),
              "unplanned_observed": sorted(g for g in by_game if planned and g not in planned),
              "samples": len(scan.rows), "complete_samples": sum(1 for r in scan.rows if r["complete"]),
              "duplicates_dropped": gs.n_duplicates_dropped, "superseded_samples": len(gs.superseded),
              "matched_sets": len(set_rows), "complete_matched_sets": len(complete_sets),
              "complete_matched_sets_single_host": sum(1 for r in complete_sets if r["single_host"]),
              "machines": len({r["machine"] for r in scan.rows if r["machine"]})}
    if shards:
        totals["kept_outside_their_shards"] = sorted(
            g for g, v in games.items() if v["kept"] and v.get("assigned_in") == [])
        totals["planned_not_in_any_shard"] = sorted(g for g in planned if not assigned.get(g))
    return {"schema": SCHEMA, "cohort": cohort, "design": design, "local_host": local_host, "strict": strict,
            "rule": "D12 (analysis.game_set.select): first-finished complete copy kept",
            "inputs": {"log_dirs": [rel_path(d) for d in log_dirs], "shards": [s["path"] for s in shard_files],
                       "ledgers": [x["path"] for x in ledger_rows]},
            "totals": totals, "notes": scan.notes + list(gs.notes), "launches": scan.logs,
            "by_tag": {k: dict(v) for k, v in sorted(tags.items())}, "seeds": seed_rows, "matched_sets": set_rows,
            "games": games, "samples": scan.rows, "shards": shard_files, "ledgers": ledger_rows}


def _csv(rows: list[dict], columns: Sequence[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (";".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()})
    return buf.getvalue()


def render(manifest: dict) -> dict[str, str]:
    """{file name: text} of every output file, after the sanitization check."""
    files = {
        "run_manifest.json": json.dumps(manifest, indent=1, sort_keys=False, default=str) + "\n",
        "samples.csv": _csv(manifest["samples"], SAMPLE_COLUMNS),
        "seeds.csv": _csv(manifest["seeds"], ("seed", "planned", "kept", "incomplete_only", "missing", "hosts",
                                              "machines")),
        "matched_sets.csv": _csv(manifest["matched_sets"], (*MATCH_KEYS, "grids", "n_games", "arms", "kept_arms",
                                                            "missing_arms", "complete", "hosts", "machines",
                                                            "single_host")),
    }
    home = str(Path.home())
    for name, text in files.items():
        if CANARY_RE.search(text):
            raise ManifestError(f"refusing to write {name}: it contains a canary pattern")
        if home and home != "/" and home in text:
            raise ManifestError(f"refusing to write {name}: it contains an absolute home path")
    return files


def write(manifest: dict, out_dir: str | Path, *, log_dirs: Sequence[str | Path] = (), overwrite: bool = False) -> list[Path]:
    out = Path(out_dir).resolve()
    for d in log_dirs:
        dd = Path(d).resolve()
        if out == dd or dd in out.parents:
            raise ManifestError(f"--out-dir {rel_path(out)} lies inside the input log directory {rel_path(dd)}")
    files = render(manifest)
    targets = [out / n for n in files]
    if not overwrite and any(t.exists() for t in targets):
        raise ManifestError(f"{rel_path(out)} already holds a manifest; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)
    for t in targets:
        tmp = t.with_name("." + t.name + ".tmp")
        tmp.write_text(files[t.name])
        tmp.replace(t)
    return targets


def summary_lines(m: dict) -> list[str]:
    t = m["totals"]
    lines = [f"games planned {t['planned']}, kept {t['kept']}, incomplete only {t['incomplete_only']}, "
             f"missing {t['missing']}; samples {t['samples']} (complete {t['complete_samples']}, dropped duplicates "
             f"{t['duplicates_dropped']}, superseded {t['superseded_samples']}); machines {t['machines']}",
             f"matched sets {t['matched_sets']}, complete {t['complete_matched_sets']}, complete on one host "
             f"{t['complete_matched_sets_single_host']}"]
    if t.get("unplanned_observed"):
        lines.append(f"observed games not in the design: {len(t['unplanned_observed'])}")
    if "kept_outside_their_shards" in t:
        lines.append(f"kept games listed in no given shard: {len(t['kept_outside_their_shards'])}; planned games in no "
                     f"given shard: {len(t['planned_not_in_any_shard'])}")
    for x in m["ledgers"]:
        lines.append(f"ledger {x['path']}: id {str(x['ledger_id'])[:8]}, open {x['n_open']} "
                     f"(${x['open_est_max_usd']} reserved)" + (f", copy of {x['copy_of']}" if x["copy_of"] else "")
                     + (f", CONFLICTS with {x['conflicts_with']}" if x["conflicts_with"] else ""))
    lines += [f"note: {n}" for n in m["notes"]]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sanitized run manifest: game -> host, tag, launch, log; per-seed and "
                                             "matched-set coverage ($0, read-only)")
    ap.add_argument("log_dirs", nargs="+", help="Inspect log directories (runs/full/logs-*, runs/remote/<host>/logs-*)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cohort", choices=("confirmatory", "pilot", "all"), default="confirmatory")
    ap.add_argument("--design", choices=("full", "none"), default="full",
                    help="coverage against harness.arms.full_design (full) or the observed games only (none)")
    ap.add_argument("--local-host", default="mac", help="host label of logs not under runs/remote/<host>/")
    ap.add_argument("--shard", action="append", default=[], help="a shard file (repeat): planned assignment")
    ap.add_argument("--ledger", action="append", default=[], help="a ledger file (repeat): counts and open reservations")
    ap.add_argument("--lenient", action="store_true", help="note unreadable logs and D12 ambiguities instead of failing")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        m = build(a.log_dirs, cohort=a.cohort, design=a.design, local_host=a.local_host, strict=not a.lenient,
                  shards=a.shard, ledgers=a.ledger)
        paths = write(m, a.out_dir, log_dirs=a.log_dirs, overwrite=a.overwrite)
    except ManifestError as exc:
        print(f"run_manifest: {exc}", file=sys.stderr)
        return 2
    print("\n".join(summary_lines(m)))
    print("wrote " + ", ".join(rel_path(p) for p in paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
