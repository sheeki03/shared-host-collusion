"""Compact progress report for the confirmatory run on every machine (read-only; prints no secrets).

Usage (repo root, venv python):
  .venv/bin/python tools/sync_logs.py && .venv/bin/python tools/progress.py
  .venv/bin/python tools/progress.py mac --no-balance --failed 20
Sources per host:
  mac   runs/full/logs-mac/*.eval, runs/full/ledger-mac.jsonl, runs/full/mac.out (live files; each .eval is copied to
        a temp file and validated before it is read, so a mid-write zip is retried, never trusted)
  vps*  runs/remote/<host>/... as pulled by tools/sync_logs.py (already validated); "as of" is the last sync
Shard (planned work): runs/shards/<host>.json {"host", "game_ids", "rounds"}; round counts come from harness.arms.
Rates: rounds of games that completed in the last 30 / 60 minutes before "as of" (windows are clipped to the
ledger's first event, so a run younger than the window is not under-counted), and an in-flight estimate: ledger
calls settled in the window divided by the calls per LLM firm-round measured on this host's completed games (pooled
across hosts until a host has 3; PROVOC* arms have one scripted firm, so work is counted in LLM firm-rounds). The
ETA uses the in-flight estimate once calibrated, otherwise the completed-game rate. Inspect writes completed samples to the log every few games, so the last few completions on a
host are invisible until the next flush. The balance line comes from tools/balance.py (USD balance only).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
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

from tools.sync_logs import validate_eval  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
HOSTS = ("mac", "vps1", "vps2")
MIN_CALIBRATION_GAMES = 3


def host_paths(host: str) -> dict:
    if host == "mac":
        base = ROOT / "runs" / "full"
        return {"logs": base / "logs-mac", "ledger": base / "ledger-mac.jsonl", "out": base / "mac.out",
                "live": True, "state": None}
    base = ROOT / "runs" / "remote" / host
    return {"logs": base / f"logs-{host}", "ledger": base / f"ledger-{host}.jsonl", "out": base / f"{host}.out",
            "live": False, "state": base / "sync_state.json"}


def design_rounds() -> dict[str, int]:
    from harness.arms import full_design

    return {s.game_id: s.rounds for s in full_design()}


def design_units() -> dict[str, int]:
    """LLM firm-rounds per game: PROVOC* arms script firm B, so they need half the model turns of other arms."""
    from harness.arms import ARMS, full_design

    return {s.game_id: s.rounds * (1 if ARMS[s.arm].get("rival") else 2) for s in full_design()}


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
                led.spent += _money(ev["actual"])
                led.n_settle += 1
                gid = game_of(ev.get("sample_id"))
                led.calls_by_game[gid] += 1
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
    tail = lines[-1][:140] if lines else ""
    return {"exists": True, "sample_warnings": len(re.findall(r"WARNING .*sample=", text)),
            "tracebacks": text.count("Traceback (most recent call last)"),
            "finished": next((ln[:160] for ln in reversed(lines) if ln.startswith("completed ") or
                              "budget reconciliation failed" in ln or "incomplete Inspect evaluation" in ln), ""),
            "tail": tail}


# ----------------------------------------------------------------------------------------------------- report
@dataclass
class HostReport:
    host: str
    as_of: datetime
    shard_ids: list[str]
    shard_rounds: int
    scan: LogScan
    ledger: LedgerScan
    out: dict
    sync_note: str = ""


def collect(hosts, rounds_by_id: dict[str, int], now: datetime | None = None) -> list[HostReport]:
    now = now or datetime.now(timezone.utc)
    reports = []
    for h in hosts:
        p = host_paths(h)
        try:
            shard = json.loads((ROOT / "runs" / "shards" / f"{h}.json").read_text())
            ids = list(shard["game_ids"])
        except (OSError, ValueError, KeyError):
            ids = []
        as_of, note = now, ""
        if p["state"] is not None:
            try:
                st = json.loads(p["state"].read_text())
                synced = parse_ts(st.get("logs_last_success"))
                led_synced = parse_ts((st.get("ledger") or {}).get("last_success"))
                if synced:
                    as_of = min(synced, led_synced or synced)
                if st.get("pending_invalid"):
                    note = f"{len(st['pending_invalid'])} log copy(ies) still invalid at last sync"
            except (OSError, ValueError):
                as_of, note = now, "never synced (run tools/sync_logs.py)"
        reports.append(HostReport(h, as_of, ids, sum(rounds_by_id.get(g, 0) for g in ids),
                                  scan_logs(p["logs"], rounds_by_id, p["live"]), parse_ledger(p["ledger"]),
                                  scan_out(p["out"]), note))
    return reports


def fmt_eta(t: datetime | None, now: datetime) -> str:
    if t is None:
        return "n/a"
    hrs = (t - now).total_seconds() / 3600
    return f"{t.astimezone(IST):%a %H:%M} IST ({hrs:+.1f} h)"


def rates(r: HostReport, cpu: float | None) -> dict:
    """Per window: completed games/rounds per hour, calls per hour, estimated LLM firm-rounds per hour in flight."""
    out = {}
    done = [g for g in r.scan.games.values() if g.ok]
    start = r.ledger.first_ts or r.as_of
    for w in (30, 60):
        eff = min(w, max((r.as_of - start).total_seconds() / 60, 0.0))  # minutes of run inside the window
        lo = r.as_of - timedelta(minutes=w)
        fin = [g for g in done if g.completed_at and lo < g.completed_at <= r.as_of]
        calls = sum(1 for ts, _ in r.ledger.settles if lo < ts <= r.as_of)
        k = 60 / eff if eff >= 5 else 0.0  # no rate from less than 5 minutes of data
        out[w] = {"games": len(fin), "eff_min": eff, "rounds_h": sum(g.rounds for g in fin) * k,
                  "calls_h": calls * k, "est_units_h": (calls / cpu * k) if cpu and k else None}
    return out


def calls_per_unit(reports: list[HostReport], units_by_id: dict[str, int], host: str | None
                   ) -> tuple[float | None, int]:
    """Ledger calls per LLM firm-round on completed games (monitor calls included)."""
    calls = units = n = 0
    for r in reports:
        if host is not None and r.host != host:
            continue
        for g in r.scan.games.values():
            u = units_by_id.get(g.game_id)
            if g.ok and u and r.ledger.calls_by_game.get(g.game_id):
                calls += r.ledger.calls_by_game[g.game_id]
                units += u
                n += 1
    return ((calls / units) if units else None), n


def render(reports: list[HostReport], rounds_by_id: dict[str, int], units_by_id: dict[str, int], now: datetime,
           n_failed: int, balance: str | None) -> str:
    out = [f"progress at {now.astimezone(IST):%a %d %b %H:%M} IST ({now:%H:%M} UTC)"]
    pooled_cpu, pooled_n = calls_per_unit(reports, units_by_id, None)
    all_ok: set[str] = set()
    for r in reports:
        all_ok |= {g for g, x in r.scan.games.items() if x.ok}
    etas, tot_spent, tot_rem_rounds = [], Decimal(0), 0
    for r in reports:
        shard = set(r.shard_ids)
        ok = {g: x for g, x in r.scan.games.items() if x.ok}
        ok_in = [g for g in ok if g in shard]
        failed_now = {g: x for g, x in r.scan.games.items() if not x.ok and g not in all_ok}
        done_rounds = sum(rounds_by_id.get(g, ok[g].rounds) for g in ok_in)
        # rounds done anywhere count against this shard (a resumed game may have finished on another host)
        rem_ids = [g for g in r.shard_ids if g not in all_ok]
        rem_rounds = sum(rounds_by_id.get(g, 0) for g in rem_ids)
        rem_units = sum(units_by_id.get(g, 0) for g in rem_ids)
        cpu, n_cal = calls_per_unit(reports, units_by_id, r.host)
        basis = f"own {n_cal} games"
        if n_cal < MIN_CALIBRATION_GAMES:
            cpu, basis = (pooled_cpu, f"pooled {pooled_n} games") if pooled_n >= MIN_CALIBRATION_GAMES else (None, "")
        rt = rates(r, cpu)
        in_flight = {gid for ts, gid in r.ledger.settles if ts > r.as_of - timedelta(minutes=10)} - set(ok)
        inflight_units = 0.0
        if cpu:
            inflight_units = sum(min(r.ledger.calls_by_game[g] / cpu, units_by_id.get(g, 0)) for g in in_flight)
        r_per_u = rem_rounds / rem_units if rem_units else 0.5  # game-rounds per LLM firm-round in the remaining mix
        inflight_rounds = inflight_units * r_per_u
        for w in (30, 60):
            eu = rt[w]["est_units_h"]
            rt[w]["est_rounds_h"] = eu * r_per_u if eu is not None else None
        if cpu and rt[60]["est_units_h"]:
            rate, rate_basis = rt[60]["est_rounds_h"], (f"ledger calls, {cpu:.2f} calls per LLM firm-round "
                                                         f"({basis}), last {rt[60]['eff_min']:.0f} min")
        elif rt[60]["rounds_h"]:
            rate, rate_basis = rt[60]["rounds_h"], f"completed games, last {rt[60]['eff_min']:.0f} min"
        else:
            rate, rate_basis = None, "no throughput observed yet"
        eta_rounds = max(rem_rounds - inflight_rounds, 0.0)
        eta = r.as_of + timedelta(hours=eta_rounds / rate) if rate and eta_rounds > 0 else (
            r.as_of if eta_rounds == 0 and r.shard_ids else None)
        if eta is not None:
            etas.append(eta)
        tot_spent += r.ledger.spent
        tot_rem_rounds += rem_rounds
        led = r.ledger
        status = ",".join(sorted(set(r.scan.statuses.values()))) or "no log"
        # staleness is judged at the data time (the last sync for a VPS), not at wall-clock time
        age = (f"{max((r.as_of - led.last_ts).total_seconds(), 0) / 60:.0f} min before data time"
               if led.last_ts else "never")
        stale = led.last_ts and (r.as_of - led.last_ts) > timedelta(minutes=15) and "started" in status
        out.append("")
        out.append(f"== {r.host}  [log {status}]  data as of {r.as_of.astimezone(IST):%H:%M} IST"
                   + (f"  ({r.sync_note})" if r.sync_note else ""))
        out.append(f"  games  {len(ok_in)}/{len(r.shard_ids)} done ({len(ok_in) / max(len(r.shard_ids), 1):.1%}), "
                   f"{len(failed_now)} failed, ~{len(in_flight)} in flight"
                   + (f", {len(ok) - len(ok_in)} done outside shard" if len(ok) > len(ok_in) else "")
                   + (f", {r.scan.duplicates_ok} duplicate ok samples" if r.scan.duplicates_ok else ""))
        out.append(f"  rounds {done_rounds}/{r.shard_rounds} done ({done_rounds / max(r.shard_rounds, 1):.1%}), "
                   f"{rem_rounds} remaining (~{inflight_rounds:.0f} of them in progress)")
        for w in (30, 60):
            x = rt[w]
            est = f", est {x['est_rounds_h']:.0f} rounds/h in flight" if x["est_rounds_h"] is not None else ""
            clip = f" (run {x['eff_min']:.0f} min old)" if x["eff_min"] < w else ""
            out.append(f"  last {w} min{clip}: {x['games']} games done = {x['rounds_h']:.0f} rounds/h, "
                       f"{x['calls_h']:.0f} calls/h{est}")
        out.append(f"  ETA    {fmt_eta(eta, now)}  [rate {rate:.0f} rounds/h from {rate_basis}]" if rate else
                   f"  ETA    {fmt_eta(eta, now)}  [{rate_basis}]")
        flags = ", ".join(f"{k}={v}" for k, v in led.flags.most_common()) or "none"
        out.append(f"  ledger {'missing' if not led.exists else ''}spent ${led.spent:.2f} / cap ${led.cap:.2f}, "
                   f"outstanding ${led.outstanding:.2f} ({led.n_open} open), {led.n_settle} settled, "
                   f"{led.n_release} released, last event {age}"
                   + (" STALE?" if stale else "") + (" BREACH" if led.breach else "")
                   + (f", {led.bad_lines} bad lines" if led.bad_lines else ""))
        out.append(f"  flags  {flags}")
        o = r.out
        if o.get("exists"):
            out.append(f"  .out   {o['sample_warnings']} sample warnings, {o['tracebacks']} tracebacks"
                       + (f"; finished: {o['finished']}" if o["finished"] else f"; tail: {o['tail'][:90]}"))
        for name, why in r.scan.unreadable.items():
            out.append(f"  UNREADABLE {name}: {why}")
        if failed_now:
            reasons = Counter(x.error[:60] for x in failed_now.values())
            out.append("  failed reasons: " + "; ".join(f"{k} x{v}" for k, v in reasons.most_common(5)))
            for g, x in list(failed_now.items())[:n_failed]:
                out.append(f"    {g}: {x.error[:120]}")
    total_games = len(rounds_by_id)
    total_rounds = sum(rounds_by_id.values())
    design_ok = all_ok & set(rounds_by_id)
    done_rounds = sum(rounds_by_id[g] for g in design_ok)
    out.append("")
    out.append(f"== overall  games {len(design_ok)}/{total_games}, rounds {done_rounds}/{total_rounds} "
               f"({done_rounds / total_rounds:.1%}); spent ${tot_spent:.2f} across ledgers; "
               f"ETA (last host) {fmt_eta(max(etas), now) if etas and len(etas) == len(reports) else 'n/a'}")
    if balance is not None:
        out.append(f"   account {balance}")
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
    ap.add_argument("hosts", nargs="*", help=f"subset of {', '.join(HOSTS)} (default: all)")
    ap.add_argument("--failed", type=int, default=8, help="failed game ids to list per host")
    ap.add_argument("--no-balance", action="store_true")
    a = ap.parse_args()
    hosts = a.hosts or list(HOSTS)
    if any(h not in HOSTS for h in hosts):
        ap.error(f"known hosts: {HOSTS}")
    now = datetime.now(timezone.utc)
    rounds_by_id = design_rounds()
    reports = collect(hosts, rounds_by_id, now)
    print(render(reports, rounds_by_id, design_units(), now, a.failed,
                 None if a.no_balance else account_balance()))


if __name__ == "__main__":
    main()
