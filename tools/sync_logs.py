"""Pull the VPS confirmatory logs, ledgers and runner output to the Mac (read-only on the remote side).

Usage (repo root, venv python):
  .venv/bin/python tools/sync_logs.py                 # both VPS, one pass (with retries for mid-write .eval files)
  .venv/bin/python tools/sync_logs.py vps1            # one host
  .venv/bin/python tools/sync_logs.py --watch 600     # repeat every 10 minutes until interrupted
Layout on the Mac, per host (vps1 = root@161.97.142.53, vps2 = root@75.119.141.172):
  runs/remote/<host>/logs-<host>/*.eval     last copy of each Inspect log that validated
  runs/remote/<host>/ledger-<host>.jsonl    complete ledger lines only (a torn last line is held back)
  runs/remote/<host>/<host>.out             runner stdout/stderr
  runs/remote/<host>/sync_state.json        what was pulled, when, and why a copy was rejected
  runs/remote/<host>/.staging/              rsync target; never read by the analysis
Remote side: rsync/scp only read. No --delete, no --remove-source-files, no remote command that writes.
An Inspect .eval is a zip that the runner rewrites every few samples, so a copy can be torn. Each staged copy is
validated (read_eval_log header_only, then the sample summaries, then a CRC check of every zip member) and only then
replaces the previous good copy. A copy with fewer samples than the last good copy of the same file is rejected.
Invalid copies are retried after --retry-wait seconds (up to --retries times); the last good copy stays in place.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/shared-host-collusion/runs/full"
HOSTS = {"vps1": "root@161.97.142.53", "vps2": "root@75.119.141.172"}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_eval(path: Path, deep: bool = True) -> tuple[bool, dict, str]:
    """Return (ok, info, error) for an Inspect .eval file that may have been copied mid-write."""
    from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries

    info: dict = {"size": path.stat().st_size if path.exists() else 0}
    try:
        if not zipfile.is_zipfile(path):
            return False, info, "not a zip (torn copy?)"
        header = read_eval_log(str(path), header_only=True)
        info["status"] = header.status
        info["started_at"] = header.stats.started_at if header.stats else None
        summaries = read_eval_log_sample_summaries(str(path))
        info["n_samples"] = len(summaries)
        info["n_errors"] = sum(1 for s in summaries if s.error)
        if deep:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
            if bad is not None:
                return False, info, f"CRC mismatch in member {bad}"
        return True, info, ""
    except Exception as e:  # noqa: BLE001 - any failure means "not a usable copy yet"
        return False, info, f"{type(e).__name__}: {str(e)[:160]}"


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stderr or "").strip()[-300:]
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s"
    except OSError as e:
        return 127, str(e)


def pull(remote: str, src: str, dest: Path, *, is_dir: bool, force_check: bool, timeout: int) -> tuple[bool, str]:
    """Copy remote:src into dest with rsync (read-only on the remote); fall back to scp."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    ssh = "ssh " + " ".join(SSH_OPTS)
    flags = ["-a", "-z", f"--timeout={min(timeout, 300)}"] + (["-I"] if force_check else [])
    if is_dir:
        dest.mkdir(parents=True, exist_ok=True)
        cmd = ["rsync", *flags, "-e", ssh, f"{remote}:{src}/", f"{dest}/"]
    else:
        cmd = ["rsync", *flags, "-e", ssh, f"{remote}:{src}", str(dest)]
    code, err = _run(cmd, timeout)
    # 23/24: partial transfer / vanished source files; whatever arrived is validated below.
    if code in (0, 23, 24):
        return True, err if code else ""
    scp = ["scp", "-p", "-q", *SSH_OPTS] + (["-r", f"{remote}:{src}/.", f"{dest}/"] if is_dir
                                             else [f"{remote}:{src}", str(dest)])
    code2, err2 = _run(scp, timeout)
    if code2 == 0:
        return True, f"rsync failed ({code}: {err[-120:]}); scp ok"
    return False, f"rsync {code}: {err[-150:]} | scp {code2}: {err2[-150:]}"


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_write(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def promote_ledger(staged: Path, final: Path) -> tuple[bool, str, dict]:
    """Keep complete lines only; refuse a copy that is not an append-only extension of the last good copy."""
    data = staged.read_bytes()
    cut = data.rfind(b"\n") + 1
    complete, torn = data[:cut], len(data) - cut
    bad = 0
    for line in complete.splitlines():
        try:
            json.loads(line)
        except ValueError:
            bad += 1
    info = {"bytes": len(complete), "lines": complete.count(b"\n"), "held_back_bytes": torn, "bad_lines": bad}
    if final.exists():
        old = final.read_bytes()
        if not complete.startswith(old):
            conflict = final.with_name(final.name + ".conflict")
            _atomic_write(conflict, complete)
            return False, f"not an append-only extension of the last good copy; saved to {conflict.name}", info
    _atomic_write(final, complete)
    return True, "", info


def sync_host(host: str, *, retries: int, retry_wait: float, deep: bool, timeout: int) -> dict:
    remote = HOSTS[host]
    base = ROOT / "runs" / "remote" / host
    stage = base / ".staging"
    logs_final = base / f"logs-{host}"
    logs_final.mkdir(parents=True, exist_ok=True)
    state_path = base / "sync_state.json"
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    state.setdefault("host", host)
    state.setdefault("files", {})
    state["last_attempt"] = now_iso()
    notes: list[str] = []

    # small files first: ledger and runner output
    for name, kind in ((f"ledger-{host}.jsonl", "ledger"), (f"{host}.out", "out")):
        ok, err = pull(remote, f"{REMOTE_ROOT}/{name}", stage / name, is_dir=False, force_check=False,
                       timeout=timeout)
        entry = state.setdefault(kind, {})
        entry["last_attempt"] = now_iso()
        if not ok:
            entry["last_error"] = err
            notes.append(f"{name}: pull failed: {err}")
            continue
        if kind == "ledger":
            good, why, info = promote_ledger(stage / name, base / name)
            entry.update(info)
            if not good:
                entry["last_error"] = why
                notes.append(f"{name}: {why}")
                continue
        else:
            _atomic_copy(stage / name, base / name)
            entry["bytes"] = (base / name).stat().st_size
        entry["last_success"] = now_iso()
        entry.pop("last_error", None)

    # Inspect logs, retried while any copy is torn
    pending = None
    for attempt in range(retries + 1):
        ok, err = pull(remote, f"{REMOTE_ROOT}/logs-{host}", stage / f"logs-{host}", is_dir=True,
                       force_check=attempt > 0, timeout=timeout)
        if not ok:
            notes.append(f"logs: pull failed (attempt {attempt + 1}): {err}")
            state["logs_last_error"] = err
            if attempt < retries:
                time.sleep(retry_wait)
            continue
        state.pop("logs_last_error", None)
        names = sorted(p.name for p in (stage / f"logs-{host}").glob("*.eval"))
        if pending is not None:
            names = [n for n in names if n in pending]
        pending = set()
        for name in names:
            staged, final = stage / f"logs-{host}" / name, logs_final / name
            st = staged.stat()
            rec = state["files"].setdefault(name, {})
            if rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime and final.exists():
                continue  # unchanged since the last good promotion
            good, info, why = validate_eval(staged, deep=deep)
            rec["last_checked"] = now_iso()
            if good and info.get("n_samples", 0) < rec.get("n_samples", 0):
                good, why = False, f"{info['n_samples']} samples < last good {rec['n_samples']}"
            if good:
                _atomic_copy(staged, final)
                rec.update({"size": st.st_size, "mtime": st.st_mtime, "n_samples": info.get("n_samples"),
                            "n_errors": info.get("n_errors"), "status": info.get("status"),
                            "started_at": info.get("started_at"), "promoted_at": now_iso()})
                rec.pop("last_error", None)
            else:
                rec["last_error"] = why
                pending.add(name)
        if not pending:
            break
        notes.append(f"logs: {len(pending)} torn/invalid copy(ies) on attempt {attempt + 1}: "
                     + "; ".join(f"{n[:40]}: {state['files'][n]['last_error']}" for n in sorted(pending)))
        if attempt < retries:
            time.sleep(retry_wait)
    if pending is not None and not pending:  # a pull succeeded and every copy validated
        state["logs_last_success"] = now_iso()
    state["pending_invalid"] = sorted(pending or [])
    state["last_notes"] = notes[-10:]
    _atomic_write(state_path, json.dumps(state, indent=1, default=str).encode())
    return state


def report(host: str, state: dict) -> str:
    files = state.get("files", {})
    n_samples = sum(f.get("n_samples") or 0 for f in files.values())
    led = state.get("ledger", {})
    out = [f"{host}: {len(files)} log file(s), {n_samples} saved samples, ledger {led.get('lines', '?')} lines"
           f"{' (torn tail held back)' if led.get('held_back_bytes') else ''}, logs synced "
           f"{state.get('logs_last_success') or 'never'}"]
    if state.get("pending_invalid"):
        out.append(f"  still invalid (last good copy kept): {', '.join(state['pending_invalid'])}")
    for note in state.get("last_notes", []):
        out.append(f"  note: {note}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hosts", nargs="*", help=f"subset of {', '.join(HOSTS)} (default: all)")
    ap.add_argument("--retries", type=int, default=3, help="re-pulls when an .eval copy is torn")
    ap.add_argument("--retry-wait", type=float, default=20.0, help="seconds between re-pulls")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per rsync/scp call")
    ap.add_argument("--no-deep", action="store_true", help="skip the per-member CRC check")
    ap.add_argument("--watch", type=float, default=0, help="repeat every N seconds until interrupted")
    a = ap.parse_args()
    hosts = a.hosts or list(HOSTS)
    unknown = [h for h in hosts if h not in HOSTS]
    if unknown:
        ap.error(f"unknown host(s) {unknown}; known: {list(HOSTS)}")
    while True:
        for h in hosts:
            st = sync_host(h, retries=a.retries, retry_wait=a.retry_wait, deep=not a.no_deep, timeout=a.timeout)
            print(report(h, st), flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
