"""Compact progress report for the confirmatory run, per run tag and per host (read-only; prints no secrets).

Usage (repo root, venv python):
  .venv/bin/python tools/sync_logs.py && .venv/bin/python tools/progress.py
  .venv/bin/python tools/progress.py mac instabox --no-balance --failed 20     # print only these hosts / tags
Runs: every tag found as runs/full/logs-<tag>/, ledger-<tag>.jsonl or <tag>.out on the Mac (host mac; live files, each
.eval is copied to a temp file and validated before it is read, so a mid-write zip is retried, never trusted), and in
runs/remote/<host>/ as pulled by tools/sync_logs.py (already validated; "as of" is the listing time of the tag's last
complete pull). Tag kinds: v1 = <host> (stopped at the D6 relaunch), v2 = <host>v2, overflow = <host>-spill-N (claimed
and launched by runs/dispatcher.py), other = anything else (e.g. D10's macv3-gated and instabox-core).
Retired = will run nothing more: a tag that is not finished and has no harness.run process (matched by its --ledger
argument; on the Mac read with ps now, on a remote recorded by sync_logs at its last listing). Its completed games are
kept; its unfinished games count as pending only where another run or a queued chunk still has them. When the process
state is unknown (an older sync), a v1 tag next to a newer tag on the same host is taken as retired.
Shard used by a tag (the first that exists): runs/remote/<host>/shards/<tag>.json (the remote runner's own copy,
pulled by sync_logs), runs/shards/<tag>.json, then runs/shards-v2/<host>.json (v2) or runs/shards-v2/spill-N.json.
Round counts come from harness.arms.
Rates: rounds of games completed in the last 30 / 60 minutes before "as of" (windows clipped to the first ledger event,
so a young run is not under-counted), and an in-flight estimate: ledger calls settled in the window divided by the
calls per LLM firm-round measured on this host's completed games (pooled across hosts until a host has 3; PROVOC* arms
script one firm, so work is counted in LLM firm-rounds). Host rates use every tag on the host except retired v1 tags.
Inspect writes completed samples to the log every 5 games, so the last few completions are invisible until a flush.
ETA per host: its runners' remaining work at its rate, plus a tail allowance (half the median wall-clock length of a
game on that host: the last games run below full concurrency, and the dispatcher waits for the runner to exit), then
the overflow chunks runs/shards-v2/spill-*.json not claimed in runs/dispatch_state.json, handed out in order to
whichever host frees up first (as the dispatcher does, plus ~4 min dispatch lag; each chunk also gets the tail
allowance). Remaining work of a runner = the games of its shard it has neither completed nor failed, minus the in-flight
estimate: harness.run runs its whole shard (deploy_v2.sh, launch_remote.sh and the dispatcher pass no exclude list), so
a game already completed elsewhere is still run, and counted (if the runner's .out shows it was given fewer games than
its shard, an exclude list was used and games completed elsewhere are not counted). A queued chunk counts all its
games. A claimed chunk with no synced data yet counts as remaining work on its host. A run counts as finished when
every log header is final (or, with no readable log, when its .out has the runner's final line); a .out final line
next to a log copy that still says "started" means the copy is behind, so the run is not finished yet. harness.run
does not retry failed samples (fail_on_error=False), so a design game that is not done anywhere and not pending
anywhere is listed as needing a re-run. The balance line comes from tools/balance.py (USD balance only).
Spend at finish (per running tag, per queued chunk against the dispatcher's CAP, and in total against the balance):
settled spend plus, for each game still to run, its LLM firm-rounds times its model's $ per firm-round, less what that
game has already settled. A model's $ per firm-round is observed on its completed games (any tag, from 3 games), or
else scaled from its catalogue price in harness/model_prices.json by the observed $-per-firm-round-to-price ratios
(median; the range uses the lowest and highest ratio). "LIKELY TO HIT CAP" when the central estimate plus the open
reservations passes the cap (harness.run then refuses calls, and those games fail); "may hit cap" when the high end does.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.sync_logs import load_hosts, validate_eval  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
LOCAL_HOST = "mac"
MIN_CALIBRATION_GAMES = 3
DISPATCH_LAG = timedelta(minutes=4)  # dispatcher: two idle checks 120 s apart before it launches a chunk
INFLIGHT = timedelta(minutes=10)
STALE = timedelta(minutes=15)
WINDOWS = (30, 60)
SCORE_IN_TAIL = re.compile(r"\b((?:mean|accuracy|score|stderr|std)\s*[:=]\s*)[-+]?\d*\.?\d+(?:e[-+]?\d+)?", re.I)
RUNNER_LEDGER = re.compile(r"--ledger\s+\S*?ledger-([A-Za-z0-9][A-Za-z0-9._-]*)\.jsonl")
_PROBE = object()


def local_runner_tags() -> set[str] | None:
    """Run tags with a live harness.run process on this machine (from its --ledger argument); None if ps fails."""
    try:
        r = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return {m.group(1) for ln in r.stdout.splitlines() if "harness.run" in ln for m in RUNNER_LEDGER.finditer(ln)}


def design_rounds() -> dict[str, int]:
    from harness.arms import full_design

    return {s.game_id: s.rounds for s in full_design()}


def design_units() -> dict[str, int]:
    """LLM firm-rounds per game: PROVOC* arms script firm B, so they need half the model turns of other arms."""
    from harness.arms import ARMS, full_design

    return {s.game_id: s.rounds * (1 if ARMS[s.arm].get("rival") else 2) for s in full_design()}


def design_models() -> dict[str, tuple[str, str]]:
    """game id -> (model label, nanoGPT model id)."""
    from harness.arms import full_design

    return {s.game_id: (s.model_label, s.model_id) for s in full_design()}


def model_prices(root: Path) -> dict[str, float]:
    """nanoGPT model id -> catalogue $ per 1M input tokens + $ per 1M output tokens (harness/model_prices.json)."""
    try:
        data = json.loads((root / "harness" / "model_prices.json").read_text())["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    out = {}
    for mid, p in data.items() if isinstance(data, dict) else []:
        if isinstance(p, dict):
            try:
                out[mid] = float(p.get("catalog_in_per_m") or p.get("in_per_m") or 0) \
                    + float(p.get("catalog_out_per_m") or p.get("out_per_m") or 0)
            except (TypeError, ValueError):
                continue
    return {k: v for k, v in out.items() if v > 0}


def dispatcher_caps(root: Path) -> dict[str, float]:
    """runs/dispatcher.py CAP ({"mac": $, "vps": $}: the ledger cap of each overflow chunk run), read with ast (the
    dispatcher is not imported or run)."""
    import ast

    try:
        tree = ast.parse((root / "runs" / "dispatcher.py").read_text())
    except (OSError, SyntaxError, ValueError):
        return {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            names = [tgt] if isinstance(tgt, ast.Name) else list(getattr(tgt, "elts", []))
            values = [node.value] if isinstance(tgt, ast.Name) else list(getattr(node.value, "elts", []))
            for n, v in zip(names, values):
                if isinstance(n, ast.Name) and n.id == "CAP":
                    try:
                        cap = ast.literal_eval(v)
                        return {str(k): float(x) for k, x in cap.items()}
                    except (ValueError, TypeError, AttributeError):
                        return {}
    return {}


def parse_ts(x) -> datetime | None:
    if not x:
        return None
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def error_text(err) -> str:
    if err is None:
        return ""
    msg = getattr(err, "message", None) or str(err)
    lines = [ln.strip() for ln in str(msg).splitlines() if ln.strip()]
    return (lines[-1] if lines else str(msg))[:160]


# ------------------------------------------------------------------------------------------------------- logs
@dataclass
class Game:
    game_id: str
    ok: bool
    rounds: int
    started_at: datetime | None
    completed_at: datetime | None
    error: str
    log: str


@dataclass
class LogScan:
    games: dict[str, Game] = field(default_factory=dict)       # best record per game id (ok beats failed)
    failed_attempts: list[Game] = field(default_factory=list)  # every failed sample, even if re-run later
    duplicates_ok: int = 0
    statuses: dict[str, str] = field(default_factory=dict)     # log file -> header status
    unreadable: dict[str, str] = field(default_factory=dict)   # log file -> reason


def _read_summaries(path: Path, live: bool, tries: int = 3, wait: float = 3.0):
    """Return (status, summaries) or raise. A live file is copied to a temp file and validated first."""
    from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

    if not live:
        return read_eval_log(str(path), header_only=True).status, read_eval_log_sample_summaries(str(path))
    last = ""
    for k in range(tries):
        with tempfile.TemporaryDirectory(prefix="shc-progress-") as td:
            tmp = Path(td) / path.name
            shutil.copy2(path, tmp)
            ok, _info, why = validate_eval(tmp, deep=True)
            if ok:
                return (read_eval_log(str(tmp), header_only=True).status,
                        read_eval_log_sample_summaries(str(tmp)))
            last = why
        if k + 1 < tries:
            time.sleep(wait)
    raise RuntimeError(f"no valid copy after {tries} tries: {last}")


def scan_logs(log_dir: Path, rounds_by_id: dict[str, int], live: bool) -> LogScan:
    scan = LogScan()
    for path in sorted(log_dir.glob("*.eval")) if log_dir.exists() else []:
        try:
            status, summaries = _read_summaries(path, live)
        except Exception as e:  # noqa: BLE001
            scan.unreadable[path.name] = f"{type(e).__name__}: {str(e)[:140]}"
            continue
        scan.statuses[path.name] = status
        for s in summaries:
            gid = str(s.id)
            meta = s.metadata or {}
            rounds = rounds_by_id.get(gid) or int(meta.get("rounds") or 0)
            bad = error_text(s.error) or (f"limit:{s.limit}" if getattr(s, "limit", None) else "")
            if not bad and s.completed is False:
                bad = "sample not completed"
            g = Game(gid, not bad, rounds, parse_ts(s.started_at), parse_ts(s.completed_at), bad, path.name)
            if not g.ok:
                scan.failed_attempts.append(g)
            prev = scan.games.get(gid)
            if prev is None or (g.ok and not prev.ok):
                scan.games[gid] = g
            elif g.ok and prev.ok:
                scan.duplicates_ok += 1
    return scan


# ----------------------------------------------------------------------------------------------------- ledger
@dataclass
class LedgerScan:
    exists: bool = False
    cap: Decimal = Decimal(0)
    spent: Decimal = Decimal(0)
    outstanding: Decimal = Decimal(0)
    n_open: int = 0
    n_settle: int = 0
    n_release: int = 0
    flags: Counter = field(default_factory=Counter)
    breach: bool = False
    bad_lines: int = 0
    torn_tail: bool = False
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    settles: list[tuple[datetime, str]] = field(default_factory=list)  # (ts, game id)
    calls_by_game: Counter = field(default_factory=Counter)
    cost_by_game: dict = field(default_factory=dict)  # game id -> Decimal settled spend


def _money(x) -> Decimal:
    return Decimal(repr(x)) if isinstance(x, float) else Decimal(str(x))


def game_of(sample_id) -> str:
    return re.sub(r"#\d+$", "", str(sample_id or ""))


def parse_ledger(path: Path) -> LedgerScan:
    led = LedgerScan()
    if not path.exists():
        return led
    led.exists = True
    data = path.read_bytes()
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        led.torn_tail = True  # live writer mid-line, or a pulled copy cut short
    open_res: dict[str, Decimal] = {}
    for i, raw in enumerate(lines):
        try:
            ev = json.loads(raw)
        except ValueError:
            if not (led.torn_tail and i == len(lines) - 1):
                led.bad_lines += 1
            continue
        kind = ev.get("event")
        ts = parse_ts(ev.get("ts"))
        if ts:
            led.first_ts = led.first_ts or ts
            led.last_ts = ts
        try:
            if kind == "init":
                led.cap = _money(ev["cap"])
            elif kind == "cap_change":
                led.cap = _money(ev["cap"])
            elif kind == "reserve":
                open_res[ev["id"]] = _money(ev["est_max"])
            elif kind == "settle":
                open_res.pop(ev["id"], None)
                actual = _money(ev["actual"])
                led.spent += actual
                led.n_settle += 1
                gid = game_of(ev.get("sample_id"))
                led.calls_by_game[gid] += 1
                led.cost_by_game[gid] = led.cost_by_game.get(gid, Decimal(0)) + actual
                if ts:
                    led.settles.append((ts, gid))
            elif kind == "release":
                open_res.pop(ev["id"], None)
                led.n_release += 1
            elif kind == "flag":
                reason = str(ev.get("reason") or "?")
                led.flags[reason[:48]] += 1
                if reason == "over_est_max":
                    led.breach = True
        except (KeyError, TypeError, ValueError, ArithmeticError):
            led.bad_lines += 1
    led.n_open = len(open_res)
    led.outstanding = sum(open_res.values(), Decimal(0))
    if led.spent + led.outstanding > led.cap:
        led.breach = True
    return led


# --------------------------------------------------------------------------------------------------- runner .out
def scan_out(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    text = path.read_text(errors="replace")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    worded = [ln for ln in lines if len(re.findall(r"[A-Za-z0-9]", ln)) >= 3]  # skip separator lines
    tail = (worded or lines or [""])[-1].strip()
    # Inspect's progress line carries the running scorer mean (an outcome): never print it
    tail = SCORE_IN_TAIL.sub(lambda m: m.group(1) + "(hidden)", tail)[:140]
    m = re.search(r"^(\d+) games -> ", text, re.M)  # harness.run: games actually given to Inspect
    return {"exists": True, "n_specs": int(m.group(1)) if m else None,
            "sample_warnings": len(re.findall(r"WARNING .*sample=", text)),
            "tracebacks": text.count("Traceback (most recent call last)"),
            "finished": next((ln[:160] for ln in reversed(lines) if ln.startswith("completed ") or
                              "budget reconciliation failed" in ln or "incomplete Inspect evaluation" in ln or
                              "cannot reserve the requested" in ln), ""),
            "tail": tail}


# --------------------------------------------------------------------------------------------------- run tags
def tag_kind(tag: str, host: str) -> tuple[str, str | None]:
    """('v1'|'v2'|'overflow'|'other', overflow chunk name or None)."""
    m = re.fullmatch(re.escape(host) + r"-(spill-\d+)", tag)
    if m:
        return "overflow", m.group(1)
    if tag == host:
        return "v1", None
    if tag == f"{host}v2":
        return "v2", None
    return "other", None


def tags_in(d: Path) -> set[str]:
    """Run tags with a log dir, ledger or runner output directly in d."""
    tags: set[str] = set()
    if not d.is_dir():
        return tags
    for p in d.iterdir():
        n = p.name
        if p.is_dir() and n.startswith("logs-") and len(n) > 5:
            tags.add(n[5:])
        elif p.is_file() and n.startswith("ledger-") and n.endswith(".jsonl") and len(n) > 13:
            tags.add(n[7:-6])
        elif p.is_file() and n.endswith(".out") and len(n) > 4:
            tags.add(n[:-4])
    return tags


def load_ids(path: Path) -> list[str] | None:
    try:
        ids = json.loads(path.read_text())["game_ids"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return [str(g) for g in ids] if isinstance(ids, list) else None


def shard_for(root: Path, host: str, tag: str, kind: str, chunk: str | None) -> tuple[list[str], str]:
    """(game ids, source path) of the shard file a tag ran from."""
    cands = []
    if host != LOCAL_HOST:
        cands.append(root / "runs" / "remote" / host / "shards" / f"{tag}.json")
    cands.append(root / "runs" / "shards" / f"{tag}.json")
    if kind == "v2":
        cands.append(root / "runs" / "shards-v2" / f"{host}.json")
    if kind == "overflow" and chunk:
        cands.append(root / "runs" / "shards-v2" / f"{chunk}.json")
    for p in cands:
        ids = load_ids(p)
        if ids is not None:
            return ids, p.relative_to(root).as_posix()
    return [], "no shard file found"


def chunk_files(root: Path) -> dict[str, Path]:
    """Overflow chunks in dispatch order: {spill-N: path}."""
    d = root / "runs" / "shards-v2"
    found = [p for p in d.glob("spill-*.json") if re.fullmatch(r"spill-\d+", p.stem)] if d.is_dir() else []
    return {p.stem: p for p in sorted(found, key=lambda p: int(p.stem.split("-")[1]))}


def dispatch_hosts(root: Path) -> dict[str, str | None]:
    """runs/dispatch_hosts.json: the hosts runs/dispatcher.py hands overflow chunks to ({host: ssh dest or None})."""
    try:
        data = json.loads((root / "runs" / "dispatch_hosts.json").read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): v for k, v in data.items()} if isinstance(data, dict) else {}


def dispatch_claims(root: Path) -> dict[str, dict]:
    try:
        claimed = json.loads((root / "runs" / "dispatch_state.json").read_text()).get("claimed") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    return {str(k): (v if isinstance(v, dict) else {}) for k, v in claimed.items()}


@dataclass
class Run:
    host: str
    tag: str
    kind: str
    chunk: str | None
    as_of: datetime
    shard_ids: list[str]
    shard_src: str
    scan: LogScan
    ledger: LedgerScan
    out: dict
    note: str = ""
    placeholder: bool = False  # claimed overflow chunk with no synced data yet
    retired: bool = False      # will run nothing more: runner stopped before finishing (D6, D10), or v1 by default
    finished: bool = False     # runner printed its final line, or every log header is final
    alive: bool | None = None  # a harness.run process for this tag was seen (Mac: now; remote: at the last sync)


def _remote_as_of(state: dict, tag: str, now: datetime) -> tuple[datetime, str]:
    bad = [p for p in state.get("pending_invalid") or [] if p.startswith(f"logs-{tag}/")]
    note = f"{len(bad)} log copy(ies) still invalid at last sync" if bad else ""
    t = (state.get("tags") or {}).get(tag)
    at = parse_ts(t.get("data_at")) if isinstance(t, dict) else None
    if at:  # tools/sync_logs.py: listing time of the last pass in which all of the tag's files were current and valid
        return at, note
    if isinstance(t, dict) and "data_at" in t:
        note = "; ".join(x for x in (note, "never pulled completely; as of the last partial pull") if x)
    # older sync_state.json (no per-tag data_at): the host's last good log pass and the tag's last current ledger
    logs_ok = parse_ts(state.get("logs_last_success"))
    ledgers = state.get("ledgers") or {}
    led = ledgers.get(f"ledger-{tag}.jsonl") or (state.get("ledger") if tag == state.get("host") else None) or {}
    cands = [x for x in (logs_ok, parse_ts(led.get("last_success"))) if x]
    if not cands:
        return now, "never synced (run tools/sync_logs.py)"
    return min(cands), note


def collect_runs(root: Path, rounds_by_id: dict[str, int], now: datetime, local_runners=_PROBE
                 ) -> tuple[list[Run], dict[str, str | None]]:
    """Every run tag on every host, plus a placeholder for each claimed chunk with no data yet.
    local_runners: tags with a live runner on the Mac (default: read with ps; None = unknown).
    Returns (runs, {host: ssh destination or None for the Mac})."""
    if local_runners is _PROBE:
        local_runners = local_runner_tags()
    hosts: dict[str, str | None] = {LOCAL_HOST: None, **dispatch_hosts(root)}
    for h, d in load_hosts(root / "runs" / "dispatch_hosts.json").items():
        hosts.setdefault(h, d)
    remote_dir = root / "runs" / "remote"
    for p in sorted(remote_dir.iterdir()) if remote_dir.is_dir() else []:
        if p.is_dir() and not p.name.startswith("."):
            hosts.setdefault(p.name, "?")
    runs: list[Run] = []
    for h in hosts:
        if h == LOCAL_HOST:
            base, live, state = root / "runs" / "full", True, None
        else:
            base, live = remote_dir / h, False
            try:
                state = json.loads((base / "sync_state.json").read_text())
            except (OSError, ValueError):
                state = {}
        for tag in sorted(tags_in(base)):
            kind, chunk = tag_kind(tag, h)
            as_of, note = (now, "") if state is None else _remote_as_of(state, tag, now)
            ids, src = shard_for(root, h, tag, kind, chunk)
            if state is None:
                alive = (tag in local_runners) if local_runners is not None else None
            else:
                t = (state.get("tags") or {}).get(tag)
                alive = t.get("runner_alive") if isinstance(t, dict) else None
            runs.append(Run(h, tag, kind, chunk, as_of, ids, src, scan_logs(base / f"logs-{tag}", rounds_by_id, live),
                            parse_ledger(base / f"ledger-{tag}.jsonl"), scan_out(base / f"{tag}.out"), note,
                            alive=alive))
    files = chunk_files(root)
    for chunk, c in dispatch_claims(root).items():
        h = str(c.get("host") or "?")
        if not any(r.host == h and r.chunk == chunk for r in runs):
            ids = load_ids(files[chunk]) if chunk in files else None
            runs.append(Run(h, f"{h}-{chunk}", "overflow", chunk, now, ids or [],
                            files[chunk].relative_to(root).as_posix() if chunk in files else "no shard file found",
                            LogScan(), LedgerScan(), {"exists": False},
                            f"claimed at {c.get('at', '?')}", placeholder=True))
            hosts.setdefault(h, "?")
    for r in runs:
        r.finished = run_finished(set(r.scan.statuses.values()), r.out)
        if r.placeholder or r.finished:
            continue
        if r.alive is False:  # stopped before finishing (a relaunch moved its games, or it died): it runs no more
            r.retired = True
        elif r.alive is None:  # process state unknown (older sync state): a v1 tag next to a newer tag is retired
            r.retired = r.kind == "v1" and any(x.host == r.host and x.kind != "v1" for x in runs)
    return runs, hosts


def run_finished(statuses: set[str], out: dict) -> bool:
    """Every readable log header is final; with no readable log, the .out has the runner's final line. A synced .out
    can be ahead of the synced log copy, so a log that still says "started" wins over the .out."""
    return ("started" not in statuses) if statuses else bool(out.get("finished"))


def collect(hosts, rounds_by_id: dict[str, int], now: datetime | None = None) -> list[Run]:
    """Compatibility view for tools/resume_plan.py: every run tag (v1, v2, overflow) of the named hosts, each with
    .host and .scan (completed and failed samples)."""
    runs, _ = collect_runs(ROOT, rounds_by_id, now or datetime.now(timezone.utc))
    wanted = set(hosts)
    return [r for r in runs if r.host in wanted and not r.placeholder]


# ---------------------------------------------------------------------------------------------------- metrics
def calls_per_unit(runs: list[Run], units_by_id: dict[str, int], host: str | None) -> tuple[float | None, int]:
    """Ledger calls per LLM firm-round on completed games (monitor calls included), from each tag's own ledger."""
    calls = units = n = 0
    for r in runs:
        if host is not None and r.host != host:
            continue
        for g in r.scan.games.values():
            u = units_by_id.get(g.game_id)
            if g.ok and u and r.ledger.calls_by_game.get(g.game_id):
                calls += r.ledger.calls_by_game[g.game_id]
                units += u
                n += 1
    return ((calls / units) if units else None), n


def windows(runs: list[Run], as_of: datetime, cpu: float | None) -> dict:
    """Per window (30, 60 minutes and 'all' = since the first ledger event): completed games and rounds per hour,
    ledger calls per hour, and estimated LLM firm-rounds per hour in flight."""
    done = [g for r in runs for g in r.scan.games.values() if g.ok]
    settles = [ts for r in runs for ts, _ in r.ledger.settles]
    firsts = [r.ledger.first_ts for r in runs if r.ledger.first_ts]
    start = min(firsts) if firsts else as_of
    span = max((as_of - start).total_seconds() / 60, 0.0)
    out = {}
    for w in (*WINDOWS, "all"):
        eff = span if w == "all" else min(w, span)  # minutes of run inside the window
        lo = start - timedelta(seconds=1) if w == "all" else as_of - timedelta(minutes=w)
        fin = [g for g in done if g.completed_at and lo < g.completed_at <= as_of]
        calls = sum(1 for ts in settles if lo < ts <= as_of)
        k = 60 / eff if eff >= 5 else 0.0  # no rate from less than 5 minutes of data
        out[w] = {"games": len(fin), "eff_min": eff, "rounds_h": sum(g.rounds for g in fin) * k,
                  "calls_h": calls * k, "units_h": (calls / cpu * k) if cpu and k else None}
    return out


def pending_ids(r: Run, all_ok: set[str]) -> list[str]:
    """Shard games this runner will still run: neither completed nor failed in this run. harness.run runs its whole
    shard, so a game completed elsewhere is still run here, unless the runner was given fewer games than its shard
    (an exclude list), in which case games completed elsewhere are assumed excluded."""
    if r.retired or r.finished:
        return []
    seen = set(r.scan.games)
    n_specs = r.out.get("n_specs")
    excluded = all_ok if (n_specs is not None and n_specs < len(r.shard_ids)) else set()
    return [g for g in r.shard_ids if g not in seen and g not in excluded]


def sec_per_round(runs: list[Run], host: str | None) -> tuple[float | None, int]:
    """Median wall-clock seconds per round of the completed games of a host's current runs (None: all hosts)."""
    vals = [(g.completed_at - g.started_at).total_seconds() / g.rounds
            for r in runs if (host is None or r.host == host) and not r.retired and not r.placeholder
            for g in r.scan.games.values() if g.ok and g.started_at and g.completed_at and g.rounds]
    return (statistics.median(vals) if vals else None), len(vals)


def tail_allowance(spr: float | None, game_rounds: list[int]) -> timedelta:
    """Half the wall-clock length of a typical remaining game: the expected wait, after the work is done at full
    concurrency, for the last games still running."""
    if not spr or not game_rounds:
        return timedelta(0)
    return timedelta(seconds=0.5 * spr * statistics.median(game_rounds))


def cost_model(runs: list[Run], units_by_id: dict[str, int], models: dict[str, tuple[str, str]],
               prices: dict[str, float]) -> dict[str, tuple[float, float, float, str]]:
    """Model label -> ($ per LLM firm-round central, low, high, basis). "observed": settled spend of that model's
    completed games (any tag) per firm-round, once it has MIN_CALIBRATION_GAMES. Otherwise "price-scaled": its
    catalogue price (in + out per 1M tokens) times the median / lowest / highest observed ratio of $ per firm-round to
    price over the observed models."""
    cost: Counter = Counter()
    units: Counter = Counter()
    games: Counter = Counter()
    for r in runs:
        for g, x in r.scan.games.items():
            if x.ok and g in models and units_by_id.get(g) and g in r.ledger.cost_by_game:
                lab = models[g][0]
                cost[lab] += float(r.ledger.cost_by_game[g])
                units[lab] += units_by_id[g]
                games[lab] += 1
    label_price = {lab: prices.get(mid) for lab, mid in set(models.values())}
    seen = {lab: cost[lab] / units[lab] for lab in units if games[lab] >= MIN_CALIBRATION_GAMES and units[lab]}
    ratios = sorted(seen[lab] / label_price[lab] for lab in seen if label_price.get(lab))
    out: dict[str, tuple[float, float, float, str]] = {}
    for lab in label_price:
        if lab in seen:
            out[lab] = (seen[lab], seen[lab], seen[lab], "observed")
        elif ratios and label_price.get(lab):
            p = label_price[lab]
            out[lab] = (statistics.median(ratios) * p, ratios[0] * p, ratios[-1] * p, "price-scaled")
    return out


def project_cost(ids, spent_by_game: dict, models: dict[str, tuple[str, str]], units_by_id: dict[str, int],
                 cm: dict[str, tuple[float, float, float, str]]) -> dict:
    """Spend still to come for games `ids` (minus what is already settled on each in this run): central, low, high,
    games priced only from the catalogue, games with no estimate at all."""
    c = lo = hi = 0.0
    scaled = unknown = 0
    for g in ids:
        m = cm.get(models.get(g, ("?", "?"))[0])
        if not m:
            unknown += 1
            continue
        u = units_by_id.get(g, 0)
        already = float(spent_by_game.get(g, 0))
        c += max(m[0] * u - already, 0.0)
        lo += max(m[1] * u - already, 0.0)
        hi += max(m[2] * u - already, 0.0)
        scaled += m[3] == "price-scaled"
    return {"c": c, "lo": lo, "hi": hi, "scaled": scaled, "unknown": unknown}


def cap_flag(spent: float, outstanding: float, cap: float, proj: dict) -> str:
    """harness.run refuses a call once settled + outstanding + its reservation would pass the cap, so the open
    reservations come off the headroom; a refused call fails its game."""
    if not cap:
        return ""
    if spent + proj["c"] + outstanding > cap:
        return " LIKELY TO HIT CAP"
    if spent + proj["hi"] + outstanding > cap or proj["unknown"]:
        return " may hit cap"
    return ""


def fmt_proj(spent: float, proj: dict) -> str:
    rng = (f" (range ${spent + proj['lo']:.0f}-${spent + proj['hi']:.0f})" if proj["hi"] - proj["lo"] >= 0.5 else "")
    extra = []
    if proj["scaled"]:
        extra.append(f"{proj['scaled']} games priced from the catalogue only")
    if proj["unknown"]:
        extra.append(f"{proj['unknown']} games with no estimate")
    return f"~${spent + proj['c']:.0f}{rng}" + (f" [{'; '.join(extra)}]" if extra else "")


def assign_overflow(hosts: dict[str, dict], chunks: list[tuple[str, dict]], lag: timedelta = DISPATCH_LAG
                    ) -> list[tuple[str, str | None, datetime | None, datetime | None]]:
    """Greedy, as runs/dispatcher.py: each queued chunk (in order) goes to the host that frees up first.
    hosts: {h: {"free_at": datetime | None, "kind": "units" | "rounds" | None, "rate": per hour or None,
    "spr": seconds per round or None}} (free_at is advanced in place); chunks: [(name, {"units": u, "rounds": r,
    "game_rounds": [rounds per game]})]. Returns [(chunk, host, start, end)]; end includes the tail allowance."""
    out = []
    for name, work in chunks:
        cands = [h for h, x in hosts.items() if x.get("free_at") and x.get("rate") and x.get("kind")]
        if not cands:
            out.append((name, None, None, None))
            continue
        h = min(cands, key=lambda k: hosts[k]["free_at"])
        x = hosts[h]
        start = x["free_at"] + lag
        end = start + timedelta(hours=work[x["kind"]] / x["rate"]) \
            + tail_allowance(x.get("spr"), work.get("game_rounds") or [])
        x["free_at"] = end
        out.append((name, h, start, end))
    return out


def reason_key(err: str) -> str:
    """Failure reason with numbers collapsed, so 'spent $5.1' and 'spent $3.2' count as one reason."""
    return re.sub(r"\d+(?:\.\d+)?", "N", err)[:70]


def fmt_eta(t: datetime | None, now: datetime) -> str:
    if t is None:
        return "n/a"
    hrs = (t - now).total_seconds() / 3600
    return f"{t.astimezone(IST):%a %H:%M} IST ({hrs:+.1f} h)"


def _hm(t: datetime | None) -> str:
    return f"{t.astimezone(IST):%H:%M}" if t else "?"


@dataclass
class HostPlan:
    host: str
    dest: str | None
    as_of: datetime
    runs: list[Run]
    cpu: float | None
    basis: str
    win: dict
    pending: list[str]
    pending_rounds: int
    pending_units: int
    inflight_units: float
    kind: str | None
    rate: float | None
    rate_basis: str
    own_free_at: datetime | None
    spr: float | None = None
    tail: timedelta = timedelta(0)
    overflow: list = field(default_factory=list)
    eta: datetime | None = None
    # when the dispatcher counts only this host's overflow runners (instabox since D10), the overflow lane: its end
    # (free for the next chunk), rate kind and rate; else the same as the whole host
    ovf_free_at: datetime | None = None
    ovf_kind: str | None = None
    ovf_rate: float | None = None
    lanes: dict = field(default_factory=dict)  # lane name -> its _lane() result, only for split hosts


def _lane(cur: list[Run], places: list[Run], as_of: datetime, cpu: float | None, basis: str, spr: float | None,
          rounds_by_id: dict[str, int], units_by_id: dict[str, int], all_ok: set[str],
          fallback: dict | None = None) -> dict:
    """Remaining work, rate and finish time of runs that share one rate (cur: with data; places: claimed chunks with
    no data yet). fallback: a lane whose rate to use when these runs have none yet."""
    win = windows(cur, as_of, cpu)
    pend = sorted({g for r in cur + places for g in pending_ids(r, all_ok)})
    p_rounds = sum(rounds_by_id.get(g, 0) for g in pend)
    p_units = sum(units_by_id.get(g, 0) for g in pend)
    inflight_units = 0.0
    if cpu:
        for r in cur:
            fl = {g for ts, g in r.ledger.settles if ts > r.as_of - INFLIGHT} & set(pending_ids(r, all_ok))
            inflight_units += sum(min(r.ledger.calls_by_game[g] / cpu, units_by_id.get(g, 0)) for g in fl)
    kind = rate = None
    rate_basis = "no throughput observed yet"
    for wk, label in ((60, f"last {win[60]['eff_min']:.0f} min"), ("all", "whole run")):
        if cpu and win[wk]["units_h"]:
            kind, rate = "units", win[wk]["units_h"]
            rate_basis = f"ledger calls, {label}, {cpu:.2f} calls per LLM firm-round ({basis})"
            break
    if kind is None:
        for wk, label in ((60, f"last {win[60]['eff_min']:.0f} min"), ("all", "whole run")):
            if win[wk]["rounds_h"]:
                kind, rate, rate_basis = "rounds", win[wk]["rounds_h"], f"completed games, {label}"
                break
    if kind is None and fallback and fallback.get("rate"):
        kind, rate = fallback["kind"], fallback["rate"]
        rate_basis = f"too young for its own rate; the host's: {fallback['rate_basis']}"
    work = max(p_units - inflight_units, 0.0) if kind == "units" else float(p_rounds)
    tail = tail_allowance(spr, [rounds_by_id[g] for g in pend if rounds_by_id.get(g)]) if pend else timedelta(0)
    end = as_of if not pend else (as_of + timedelta(hours=work / rate) + tail if rate else None)
    return {"win": win, "pend": pend, "p_rounds": p_rounds, "p_units": p_units, "inflight_units": inflight_units,
            "kind": kind, "rate": rate, "rate_basis": rate_basis, "end": end, "tail": tail}


def plan_hosts(runs: list[Run], hosts: dict[str, str | None], rounds_by_id: dict[str, int],
               units_by_id: dict[str, int], now: datetime, all_ok: set[str],
               split_hosts: set[str] = frozenset()) -> dict[str, HostPlan]:
    """split_hosts: hosts the dispatcher treats as free for a chunk while non-overflow runs still run there."""
    pooled, pooled_n = calls_per_unit(runs, units_by_id, None)
    spr_pooled, spr_pooled_n = sec_per_round(runs, None)
    plans = {}
    for h, dest in hosts.items():
        hr = [r for r in runs if r.host == h]
        live = [r for r in hr if not r.retired]
        cur = [r for r in live if not r.placeholder]
        places = [r for r in live if r.placeholder]
        as_of = min((r.as_of for r in cur), default=None) or max((r.as_of for r in hr), default=now)
        cpu, n_cal = calls_per_unit(runs, units_by_id, h)
        basis = f"own {n_cal} games"
        if n_cal < MIN_CALIBRATION_GAMES:
            cpu, basis = (pooled, f"pooled {pooled_n} games") if pooled_n >= MIN_CALIBRATION_GAMES else (None, "")
        spr, n_spr = sec_per_round(runs, h)
        if n_spr < MIN_CALIBRATION_GAMES:
            spr = spr_pooled if spr_pooled_n >= MIN_CALIBRATION_GAMES else None
        args = (as_of, cpu, basis, spr, rounds_by_id, units_by_id, all_ok)
        whole = _lane(cur, places, *args)
        lanes: dict = {}
        own, ovf = whole["end"], whole
        if h in split_hosts:
            ovf = _lane([r for r in cur if r.kind == "overflow"], [r for r in places if r.kind == "overflow"], *args,
                        fallback=whole)
            core = _lane([r for r in cur if r.kind != "overflow"], [r for r in places if r.kind != "overflow"],
                         *args, fallback=whole)
            lanes = {"core": core, "overflow": ovf}
            ends = [x["end"] for x in (core, ovf)]
            own = None if any(e is None for e in ends) else max(ends)
        plans[h] = HostPlan(h, dest, as_of, hr, cpu, basis, whole["win"], whole["pend"], whole["p_rounds"],
                            whole["p_units"], whole["inflight_units"], whole["kind"], whole["rate"],
                            whole["rate_basis"], own, spr, whole["tail"], ovf_free_at=ovf["end"],
                            ovf_kind=ovf["kind"], ovf_rate=ovf["rate"], lanes=lanes)
    return plans


def dispatcher_split_hosts(root: Path) -> set[str]:
    """Hosts for which runs/dispatcher.py alive() counts only overflow runners (`host == "<name>"` beside a
    "-spil[l]-" pattern), read from its source; the dispatcher is not imported or run."""
    import ast

    try:
        src = (root / "runs" / "dispatcher.py").read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError, ValueError):
        return set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "alive":
            body = ast.get_source_segment(src, node) or ""
            if "spil" not in body:
                return set()
            return set(re.findall(r"host\s*==\s*[\"']([A-Za-z0-9][A-Za-z0-9._-]*)[\"']", body))
    return set()


# ----------------------------------------------------------------------------------------------------- report
def _ledger_line(led: LedgerScan, as_of: datetime, running: bool) -> str:
    age = (f"{max((as_of - led.last_ts).total_seconds(), 0) / 60:.0f} min before data time" if led.last_ts
           else "never")
    stale = running and led.last_ts and (as_of - led.last_ts) > STALE
    flags = ", ".join(f"{k}={v}" for k, v in led.flags.most_common()) or "none"
    return (f"ledger {'missing ' if not led.exists else ''}spent ${led.spent:.2f} / cap ${led.cap:.2f}, "
            f"outstanding ${led.outstanding:.2f} ({led.n_open} open), {led.n_settle} settled, {led.n_release} released, "
            f"last event {age}" + (" STALE?" if stale else "") + (" BREACH" if led.breach else "")
            + (f", {led.bad_lines} bad lines" if led.bad_lines else "") + f"; flags {flags}")


def _rate_line(win: dict, r_per_u: float) -> str:
    parts = []
    for w in WINDOWS:
        x = win[w]
        clip = f" (run {x['eff_min']:.0f} min old)" if x["eff_min"] < w else ""
        est = f", est {x['units_h'] * r_per_u:.0f} rounds/h in flight" if x["units_h"] is not None else ""
        parts.append(f"{w} min{clip}: {x['games']} games done = {x['rounds_h']:.0f} rounds/h, "
                     f"{x['calls_h']:.0f} calls/h{est}")
    return " | ".join(parts)


def chunk_cap(caps: dict[str, float], host: str) -> float:
    return caps.get("mac" if host == LOCAL_HOST else "vps", 0.0)


def run_projection(r: Run, all_ok: set[str], models, units_by_id, cm, caps) -> tuple[float, float, float, dict]:
    """(spent, outstanding, cap, projected spend still to come) of a run that will still spend."""
    proj = project_cost(pending_ids(r, all_ok), r.ledger.cost_by_game, models, units_by_id, cm)
    cap = float(r.ledger.cap) if r.ledger.exists else (chunk_cap(caps, r.host) if r.placeholder else 0.0)
    return float(r.ledger.spent), float(r.ledger.outstanding), cap, proj


def render_run(r: Run, plan: HostPlan, all_ok: set[str], pending_all: set[str], rounds_by_id: dict[str, int],
               units_by_id: dict[str, int], n_failed: int, models: dict | None = None, cm: dict | None = None,
               caps: dict | None = None) -> list[str]:
    shard = set(r.shard_ids)
    ok = {g for g, x in r.scan.games.items() if x.ok}
    ok_in = ok & shard
    elsewhere = (shard & all_ok) - ok
    pend = pending_ids(r, all_ok)
    status = ",".join(sorted(set(r.scan.statuses.values()))) or "none yet"
    if r.placeholder:
        state = "claimed, no data synced yet"
    elif r.retired:
        state = ("retired: runner stopped before finishing (no runner process" +
                 (" at the last sync)" if r.host != LOCAL_HOST else ")") if r.alive is False
                 else "retired at the D6 relaunch (runner process state unknown)")
    elif r.finished:
        state = "finished" + (f": {r.out['finished'][:90]}" if r.out.get("finished") else "")
    else:
        stale = r.ledger.last_ts and (r.as_of - r.ledger.last_ts) > STALE
        state = "STALE? (no ledger event for 15+ min)" if stale else ("running" if r.ledger.last_ts else "starting")
    head = f"  [{r.tag}] {r.kind} - {state} - shard {r.shard_src}"
    out = [head + (f" ({r.note})" if r.placeholder else f" - log status {status} - data as of {_hm(r.as_of)} IST"
                   + (f" ({r.note})" if r.note else ""))]
    done_rounds = sum(rounds_by_id.get(g, r.scan.games[g].rounds) for g in ok_in)
    shard_rounds = sum(rounds_by_id.get(g, 0) for g in r.shard_ids)
    failed_games = {g: x for g, x in r.scan.games.items() if not x.ok}
    in_flight = ({g for ts, g in r.ledger.settles if ts > r.as_of - INFLIGHT} & set(pend)) if not r.placeholder else set()
    redo = sum(1 for g in pend if g in all_ok)
    out.append(f"    games {len(ok_in)}/{len(r.shard_ids)} done here ({len(ok_in) / max(len(r.shard_ids), 1):.1%})"
               + (f", {len(elsewhere)} done elsewhere" if elsewhere else "")
               + (f", {len(ok - shard)} done outside the shard" if ok - shard else "")
               + f", {len(failed_games)} failed, ~{len(in_flight)} in flight, {len(pend)} still to run"
               + (f" ({redo} of them already done elsewhere)" if redo else "")
               + (f"; rounds {done_rounds}/{shard_rounds} ({done_rounds / max(shard_rounds, 1):.1%})"))
    if r.retired:
        rest = [g for g in r.shard_ids if g not in all_ok]
        resched = sum(1 for g in rest if g in pending_all)
        out.append(f"    of its {len(rest)} games not done anywhere: {resched} pending in other runs or queued chunks, "
                   f"{len(rest) - resched} not pending anywhere")
    elif not r.placeholder and not r.finished:
        rem_u = sum(units_by_id.get(g, 0) for g in pend)
        r_per_u = (sum(rounds_by_id.get(g, 0) for g in pend) / rem_u) if rem_u else 0.5
        out.append("    rate " + _rate_line(windows([r], r.as_of, plan.cpu), r_per_u))
    if not r.placeholder:
        out.append("    " + _ledger_line(r.ledger, r.as_of, not (r.retired or r.finished)))
    if not (r.retired or r.finished) and pend and models is not None:
        spent, outstanding, cap, proj = run_projection(r, all_ok, models, units_by_id, cm or {}, caps or {})
        out.append(f"    spend at finish {fmt_proj(spent, proj)} of cap ${cap:.0f}"
                   + cap_flag(spent, outstanding, cap, proj))
    if r.scan.failed_attempts:
        reasons = Counter(reason_key(x.error) for x in r.scan.failed_attempts)
        undone = [g for g in failed_games if g not in all_ok]
        rerun = [g for g in undone if g not in pending_all]
        out.append(f"    failed samples {len(r.scan.failed_attempts)} by reason: "
                   + "; ".join(f"{k} x{v}" for k, v in reasons.most_common(5))
                   + f"  [{len(failed_games) - len(undone)} since done, {len(undone) - len(rerun)} pending elsewhere, "
                     f"{len(rerun)} need a re-run]")
        for g in rerun[:n_failed]:
            out.append(f"      re-run {g}: {failed_games[g].error[:110]}")
    if r.scan.duplicates_ok:
        out.append(f"    {r.scan.duplicates_ok} duplicate ok samples in this tag's logs")
    o = r.out
    if o.get("exists"):
        out.append(f"    .out  {o['sample_warnings']} sample warnings, {o['tracebacks']} tracebacks"
                   + (f"; tail: {o['tail'][:100]}" if not o["finished"] else ""))
    for name, why in r.scan.unreadable.items():
        out.append(f"    UNREADABLE {name}: {why}")
    return out


def render(runs: list[Run], hosts: dict[str, str | None], root: Path, rounds_by_id: dict[str, int],
           units_by_id: dict[str, int], now: datetime, n_failed: int, balance: str | None,
           only: list[str] | None = None, models: dict[str, tuple[str, str]] | None = None) -> str:
    all_ok: set[str] = set()
    ok_tags: dict[str, list[str]] = {}
    for r in runs:
        ok_here = {g for g, x in r.scan.games.items() if x.ok}
        all_ok |= ok_here
        for g in ok_here:
            ok_tags.setdefault(g, []).append(r.tag)
    plans = plan_hosts(runs, hosts, rounds_by_id, units_by_id, now, all_ok, dispatcher_split_hosts(root))
    models = design_models() if models is None else models
    cm = cost_model(runs, units_by_id, models, model_prices(root))
    caps = dispatcher_caps(root)
    claims = dispatch_claims(root)
    files = chunk_files(root)
    queued = []
    queued_ids: set[str] = set()
    for chunk, p in files.items():
        if chunk in claims:
            continue
        ids = load_ids(p) or []  # the dispatcher launches a chunk without an exclude list: every game is run
        queued_ids |= set(ids)
        queued.append((chunk, {"units": sum(units_by_id.get(g, 0) for g in ids),
                               "rounds": sum(rounds_by_id.get(g, 0) for g in ids), "games": len(ids),
                               "done_elsewhere": sum(1 for g in ids if g in all_ok),
                               "game_rounds": [rounds_by_id[g] for g in ids if rounds_by_id.get(g)],
                               "proj": project_cost(ids, {}, models, units_by_id, cm)}))
    takers = set(dispatch_hosts(root)) or {h for h, d in hosts.items() if d != "?"}  # hosts the dispatcher feeds
    sim = {h: {"free_at": p.ovf_free_at, "kind": p.ovf_kind, "rate": p.ovf_rate, "spr": p.spr}
           for h, p in plans.items() if h in takers}
    assigned = assign_overflow(sim, queued)
    for chunk, h, start, end in assigned:
        if h:
            plans[h].overflow.append((chunk, start, end))
    for h, p in plans.items():
        ends = [p.own_free_at] + ([sim[h]["free_at"]] if h in sim and p.overflow else [])
        p.eta = None if any(e is None for e in ends) else max(ends)
    pending_all = {g for r in runs for g in pending_ids(r, all_ok)} | queued_ids

    out = [f"progress at {now.astimezone(IST):%a %d %b %H:%M} IST ({now:%H:%M} UTC)"]
    for h, p in plans.items():
        if only and h not in only and not any(r.tag in only for r in p.runs):
            continue
        out.append("")
        where = "local" if h == LOCAL_HOST else (p.dest or "?")
        out.append(f"== {h} ({where})  data as of {_hm(p.as_of)} IST, {len(p.runs)} run tag(s)")
        if not p.runs:
            out.append("  no run data (never synced, or nothing launched yet)")
        for r in sorted(p.runs, key=lambda x: (x.retired, x.finished, x.tag)):
            if only and h not in only and r.tag not in only:
                continue
            out.extend(render_run(r, p, all_ok, pending_all, rounds_by_id, units_by_id, n_failed, models, cm, caps))
        spent = sum((r.ledger.spent for r in p.runs), Decimal(0))
        r_per_u = p.pending_rounds / p.pending_units if p.pending_units else 0.5
        inflight_rounds = p.inflight_units * r_per_u
        out.append(f"  host  spent ${spent:.2f} over {sum(1 for r in p.runs if r.ledger.exists)} ledger(s); still to "
                   f"run {len(p.pending)} games = {p.pending_rounds} rounds (~{inflight_rounds:.0f} of them in progress)")
        if sum(1 for r in p.runs if not r.retired and not r.placeholder) > 1:  # else the same as the tag's line
            out.append("        rate " + _rate_line(p.win, r_per_u))
        if p.rate:
            shown = p.rate * (r_per_u if p.kind == "units" else 1)
            out.append(f"        work rate {shown:.0f} rounds/h [{p.rate_basis}]")
        else:
            out.append(f"        work rate n/a [{p.rate_basis}]")
        ovf = "; then " + ", ".join(f"{c} (projected {_hm(s)}-{_hm(e)} IST)" for c, s, e in p.overflow) \
            if p.overflow else ""
        if p.lanes:
            c, o = p.lanes["core"], p.lanes["overflow"]

            def lane_rate(x):
                return f"{x['rate'] * (r_per_u if x['kind'] == 'units' else 1):.0f} rounds/h" if x["rate"] else "n/a"
            out.append(f"        lanes (the dispatcher counts only overflow runners here): core {len(c['pend'])} games "
                       f"at {lane_rate(c)}, done {fmt_eta(c['end'], now)}; overflow {len(o['pend'])} games at "
                       f"{lane_rate(o)}, free {fmt_eta(o['end'], now)}{ovf}; ETA {fmt_eta(p.eta, now)}")
        else:
            tail = (f" (incl. {p.tail.total_seconds() / 60:.0f} min for the last games, {p.spr:.0f} s/round)"
                    if p.tail else "")
            out.append(f"        own work done {fmt_eta(p.own_free_at, now)}{tail}{ovf}; ETA {fmt_eta(p.eta, now)}")

    out.append("")
    parts = []
    for chunk in files:
        if chunk in claims:
            c = claims[chunk]
            parts.append(f"{chunk} claimed by {c.get('host', '?')} at {c.get('at', '?')}")
    for (chunk, w), (_c, h, s, e) in zip(queued, assigned):
        cap = chunk_cap(caps, h) if h else max(caps.values(), default=0.0)
        parts.append(f"{chunk} queued ({w['games']} games, {w['rounds']} rounds"
                     + (f", {w['done_elsewhere']} of them already done elsewhere" if w["done_elsewhere"] else "")
                     + f", spend {fmt_proj(0.0, w['proj'])} of cap ${cap:.0f}{cap_flag(0.0, 0.0, cap, w['proj'])}"
                     + ") -> "
                     + (f"projected on {h} {_hm(s)}-{_hm(e)} IST" if h else "no host with a known rate"))
    out.append("== overflow  " + ("; ".join(parts) if parts else "no chunk files"))

    design = set(rounds_by_id)
    design_ok = all_ok & design
    done_rounds = sum(rounds_by_id[g] for g in design_ok)
    total_rounds = sum(rounds_by_id.values()) or 1
    rerun = sorted(design - all_ok - pending_all)
    dups = sorted(g for g, t in ok_tags.items() if len(t) > 1)
    spent = sum((r.ledger.spent for r in runs), Decimal(0))
    n_ledgers = sum(1 for r in runs if r.ledger.exists)
    with_work = [p for p in plans.values() if p.pending or p.overflow]
    unknown = [p.host for p in with_work if p.eta is None] + [c for c, h, _s, _e in assigned if h is None]
    eta = max((p.eta for p in with_work if p.eta), default=None)
    out.append(f"== overall  games {len(design_ok)}/{len(design)}, rounds {done_rounds}/{total_rounds} "
               f"({done_rounds / total_rounds:.1%}); spent ${spent:.2f} across {n_ledgers} ledgers; "
               f"ETA (last host) {fmt_eta(eta, now) if eta and not unknown else 'n/a'}"
               + (f" (no estimate for {', '.join(unknown)})" if unknown else ""))
    redo = sorted(pending_all & all_ok)
    out.append(f"   pending {len(pending_all & design - all_ok)} games not done anywhere"
               + (f" (runners will also run {len(redo)} game(s) already done elsewhere, e.g. {redo[0]})" if redo else "")
               + f"; need a re-run (not done, not pending anywhere): {len(rerun)}"
               + (f", e.g. {', '.join(rerun[:3])}" if rerun else "")
               + (f"; completed in more than one tag: {len(dups)}" if dups else ""))
    if dups:
        pairs = Counter("+".join(sorted(ok_tags[g])) for g in dups)
        out.append("   duplicate completions by tags: " + ", ".join(f"{k} x{v}" for k, v in pairs.most_common())
                   + f" (e.g. {dups[0]}); the analysis must keep one per game")
    to_come = {"c": 0.0, "lo": 0.0, "hi": 0.0, "scaled": 0, "unknown": 0}
    for r in runs:
        if not (r.retired or r.finished):
            proj = run_projection(r, all_ok, models, units_by_id, cm, caps)[3]
            for k in to_come:
                to_come[k] += proj[k]
    for _chunk, w in queued:
        for k in to_come:
            to_come[k] += w["proj"][k]
    m = re.search(r"\$(-?[0-9.]+)", balance or "")
    short = m and float(m.group(1)) < to_come["hi"]
    out.append(f"   spend still to come {fmt_proj(0.0, to_come)} (running tags and queued chunks)"
               + (f"; account {balance}" if balance is not None else "")
               + (" - BALANCE MAY RUN OUT" if short else ""))
    return "\n".join(out)


def account_balance() -> str:
    try:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "balance.py")], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "balance unavailable"
    m = re.search(r"usd_balance=([-0-9.]+)", r.stdout or "")
    return f"balance ${float(m.group(1)):.2f}" if m else "balance unavailable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("only", nargs="*", help="hosts or run tags to print (default: all); totals always cover all")
    ap.add_argument("--failed", type=int, default=8, help="failed game ids needing a re-run to list per tag")
    ap.add_argument("--no-balance", action="store_true")
    a = ap.parse_args()
    now = datetime.now(timezone.utc)
    rounds_by_id = design_rounds()
    runs, hosts = collect_runs(ROOT, rounds_by_id, now)
    print(render(runs, hosts, ROOT, rounds_by_id, design_units(), now, a.failed,
                 None if a.no_balance else account_balance(), a.only or None))


if __name__ == "__main__":
    main()
