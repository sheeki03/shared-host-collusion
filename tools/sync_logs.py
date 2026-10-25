"""Pull every confirmatory run's logs, ledgers, runner output and shard file from every remote host to the Mac
(read-only on the remote side).

Usage (repo root, venv python):
  .venv/bin/python tools/sync_logs.py                 # every remote host, one pass (retries for mid-write .eval files)
  .venv/bin/python tools/sync_logs.py vps1 instabox   # some hosts
  .venv/bin/python tools/sync_logs.py --watch 600     # repeat every 10 minutes until interrupted
Hosts: every non-null entry of runs/dispatch_hosts.json ({host: ssh destination or alias}; null is the Mac, which is
not synced), plus vps1 = root@161.97.142.53 and vps2 = root@75.119.141.172 if that file lacks them. The remote repo is
addressed home-relative (~/shared-host-collusion), so the /root and /data/home layouts both work.
Pulled from <remote home>/shared-host-collusion/runs/, for every run tag (v1 "vps1", v2 "vps1v2", overflow chunks
"<host>-spill-N", anything else the runner names):
  full/logs-*/*.eval   full/ledger-*.jsonl   full/*.out   shards/*.json
Layout on the Mac, per host, keeping the tag names:
  runs/remote/<host>/logs-<tag>/*.eval      last copy of each Inspect log that validated
  runs/remote/<host>/ledger-<tag>.jsonl     complete ledger lines only (a torn last line is held back)
  runs/remote/<host>/<tag>.out              runner stdout/stderr
  runs/remote/<host>/shards/<tag>.json      the shard file the remote runner was given
  runs/remote/<host>/sync_state.json        what was pulled and when, remote file counts, why a copy was rejected
  runs/remote/<host>/.staging/{full,shards}/  transfer target; never read by the analysis
Remote side: one read-only ssh command (a `find` listing, then `ps -eo args=` filtered to harness.run command lines, to
record which run tags have a live runner: sync_state.json "runners" and "tags".<tag>."runner_alive"), then rsync
reading only (scp where the remote has no rsync; it fetches only files whose size or mtime changed). No --delete, no
--remove-source-files, no remote command that writes.
An Inspect .eval is a zip that the runner rewrites every few samples, so a copy can be torn. Each staged copy is
validated (read_eval_log header_only, then the sample summaries, then a CRC check of every zip member) and only then
replaces the previous good copy. A copy with fewer samples than the last good copy of the same file is rejected.
Invalid copies are re-pulled (only those files) after --retry-wait seconds, up to --retries times; the last good copy
stays in place. Ledgers must be append-only extensions of the last good copy.
Freshness: a staged file counts as current only if its mtime (the remote's, preserved by rsync -a / scp -p) is at least
the one in this pass's listing, so a partial rsync (exit 23/24) or a failed scp never marks old data as new. Per tag,
sync_state.json "tags".<tag>."data_at" is the listing time of the last pass in which every listed file of the tag was
current and valid; tools/progress.py uses it as that tag's "data as of".
Disk: a promoted .eval is a hard link to its staged copy, so each log is stored once. rsync and the scp fallback write
a new file and rename it into .staging (never in place), which breaks the link and leaves the last good copy as it
was; do not add --inplace, --append or -S to the rsync flags (tools/tests/test_run_ops.py checks the local rsync).
One sync per host at a time (flock on runs/remote/<host>/.lock); a second one skips that host.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REMOTE_BASE = ROOT / "runs" / "remote"
HOSTS_FILE = ROOT / "runs" / "dispatch_hosts.json"
REMOTE_RUNS = "shared-host-collusion/runs"  # relative to the remote login directory (rsync/scp/ssh all resolve it so)
DEFAULT_HOSTS = {"vps1": "root@161.97.142.53", "vps2": "root@75.119.141.172"}
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4"]
RSYNC_FILTERS = ["--include=/full/", "--include=/full/logs-*/", "--include=/full/logs-*/*.eval",
                 "--include=/full/ledger-*.jsonl", "--include=/full/*.out", "--include=/shards/",
                 "--include=/shards/*.json", "--exclude=*"]
# Read-only: a file listing, then the command lines of live harness.run processes (the [r] keeps grep from matching
# its own command line), so tools/progress.py can tell a stopped runner from a running one.
LIST_CMD = (f"cd ~/{REMOTE_RUNS} && {{ find full shards -maxdepth 2 -type f -printf '%s %T@ %p\\n' 2>/dev/null; "
            "echo __listing_end__; if command -v ps >/dev/null 2>&1; then echo __ps_ok__; ps -eo args= 2>/dev/null "
            "| grep -- '--ledger runs/full/ledge[r]-'; fi; echo __procs_end__; }")
RUNNER_LEDGER = re.compile(r"--ledger\s+\S*?ledger-([A-Za-z0-9][A-Za-z0-9._-]*)\.jsonl")
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # no hidden/temp files, no '..', no glob characters
SAFE_DEST = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*")  # user@host or an ssh alias; never an ssh option
RSYNC_RECHECK = timedelta(hours=6)
STALE_TEMP = 3600  # seconds: a dot-file left in .staging by a killed rsync is removed after this


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_hosts(path: Path = HOSTS_FILE) -> dict[str, str]:
    """{host: ssh destination}: vps1/vps2 defaults updated by the non-null entries of runs/dispatch_hosts.json."""
    hosts = dict(DEFAULT_HOSTS)
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return hosts
    if isinstance(data, dict):
        for h, dest in data.items():
            if isinstance(dest, str) and SAFE_DEST.fullmatch(dest.strip()) and SAFE.fullmatch(str(h)):
                hosts[str(h)] = dest.strip()
    return hosts


def classify(rel: str) -> str | None:
    """Kind of a path relative to the remote runs/ dir: 'eval', 'ledger', 'out', 'shard', or None (not pulled)."""
    parts = rel.split("/")
    if not parts or not all(SAFE.fullmatch(p) for p in parts):
        return None
    if parts[0] == "full":
        if len(parts) == 3 and parts[1].startswith("logs-") and len(parts[1]) > 5 and parts[2].endswith(".eval"):
            return "eval"
        if len(parts) == 2:
            name = parts[1]
            if name.startswith("ledger-") and name.endswith(".jsonl") and len(name) > len("ledger-.jsonl"):
                return "ledger"
            if name.endswith(".out") and len(name) > len(".out"):
                return "out"
    elif parts[0] == "shards" and len(parts) == 2 and parts[1].endswith(".json") and len(parts[1]) > 5:
        return "shard"
    return None


def local_rel(rel: str) -> str:
    """Remote runs/-relative path -> path under runs/remote/<host>/ (full/ is flattened, shards/ kept)."""
    return rel[len("full/"):] if rel.startswith("full/") else rel


def tag_of(lrel: str) -> str:
    """Run tag of a local relative path (logs-<tag>/x.eval, ledger-<tag>.jsonl, <tag>.out, shards/<tag>.json)."""
    first = lrel.split("/")[0]
    if first.startswith("logs-"):
        return first[5:]
    if first == "shards":
        return lrel.split("/")[1][:-5]
    if first.startswith("ledger-") and first.endswith(".jsonl"):
        return first[7:-6]
    return first[:-4] if first.endswith(".out") else first


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


# ------------------------------------------------------------------------------------------------ transport
def parse_listing(stdout: str) -> tuple[dict[str, tuple[int, float]] | None, set[str] | None]:
    """LIST_CMD output -> ({runs/-relative path: (size, mtime)}, run tags with a live runner process or None if the
    process list is unavailable). (None, None) when the listing did not complete."""
    head, sep, rest = (stdout or "").partition("__listing_end__")
    if not sep:
        return None, None
    out: dict[str, tuple[int, float]] = {}
    for line in head.splitlines():
        m = re.fullmatch(r"(\d+) (\d+(?:\.\d+)?) (\S+)", line.strip())
        if m and classify(m.group(3)):
            out[m.group(3)] = (int(m.group(1)), float(m.group(2)))
    procs_text, sep2, _ = rest.partition("__procs_end__")
    if not sep2 or "__ps_ok__" not in procs_text:
        return out, None
    return out, {m.group(1) for m in RUNNER_LEDGER.finditer(procs_text.split("__ps_ok__", 1)[1])}


def remote_listing(dest: str, timeout: int) -> tuple[dict[str, tuple[int, float]] | None, str, set[str] | None]:
    """({runs/-relative path: (size, mtime)} of every pullable remote file, error, run tags with a live runner), via
    one read-only ssh command (find + ps); listing None on failure."""
    try:
        r = subprocess.run(["ssh", *SSH_OPTS, dest, LIST_CMD], capture_output=True, text=True,
                           timeout=min(timeout, 120))
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"listing failed: {type(e).__name__}", None
    listing, procs = parse_listing(r.stdout or "")
    if listing is None:
        return None, f"listing failed ({r.returncode}): {(r.stderr or '').strip()[-150:]}", None
    return listing, "", procs


def _only_filters(rels: list[str]) -> list[str]:
    inc: list[str] = []
    for rel in rels:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            d = "--include=/" + "/".join(parts[:i]) + "/"
            if d not in inc:
                inc.append(d)
        inc.append("--include=/" + rel)
    return inc + ["--exclude=*"]


def rsync_pull(dest: str, stage: Path, timeout: int, only: list[str] | None = None) -> tuple[bool, str]:
    """rsync <dest>:~/shared-host-collusion/runs/ (filtered) into stage; `only` re-pulls those files with -I."""
    stage.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a", "-z", f"--timeout={min(timeout, 300)}", *(["-I"] if only else []),
           *(_only_filters(only) if only else RSYNC_FILTERS), "-e", "ssh " + " ".join(SSH_OPTS),
           f"{dest}:{REMOTE_RUNS}/", f"{stage}/"]
    code, err = _run(cmd, timeout)
    # 23/24: partial transfer / vanished source files; whatever arrived is validated below.
    return code in (0, 23, 24), (err if code else "")


def scp_pull(dest: str, stage: Path, rels: list[str], listing: dict | None, timeout: int) -> tuple[bool, str]:
    """Fetch the given runs/-relative files into a temp dir, then rename each into stage (never written in place)."""
    tmp_root = stage / ".scp-tmp"
    shutil.rmtree(tmp_root, ignore_errors=True)
    groups: dict[str, list[str]] = defaultdict(list)
    for rel in rels:
        groups[str(PurePosixPath(rel).parent)].append(rel)
    errors = []
    for parent, items in sorted(groups.items()):
        tdir = tmp_root / parent
        tdir.mkdir(parents=True, exist_ok=True)
        code, err = _run(["scp", "-p", "-q", *SSH_OPTS, *[f"{dest}:{REMOTE_RUNS}/{r}" for r in items], f"{tdir}/"],
                         timeout)
        for rel in items:
            got = tmp_root / rel
            # after a failed scp, keep only files that arrived whole (size as listed); others may be cut short
            if got.is_file() and (code == 0 or (listing and listing.get(rel, (-1,))[0] == got.stat().st_size)):
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                os.replace(got, stage / rel)
        if code != 0:
            errors.append(f"{parent}: scp {code}: {err[-120:]}")
    shutil.rmtree(tmp_root, ignore_errors=True)
    return not errors, "; ".join(errors)


def _changed(stage: Path, listing: dict) -> list[str]:
    out = []
    for rel, (size, mtime) in sorted(listing.items()):
        p = stage / rel
        try:
            st = p.stat()
        except OSError:
            out.append(rel)
            continue
        if st.st_size != size or int(st.st_mtime) != int(mtime):
            out.append(rel)
    return out


def pull_all(dest: str, stage: Path, listing: dict | None, state: dict, timeout: int,
             only: list[str] | None = None) -> tuple[bool, str]:
    """rsync unless this remote is known to lack it; scp of changed (or `only`) files as the fallback."""
    skip_until = _parse(state.get("rsync_missing_until"))
    use_rsync = not (skip_until and datetime.now(timezone.utc) < skip_until and listing is not None)
    err = ""
    if use_rsync:
        ok, err = rsync_pull(dest, stage, timeout, only)
        if ok:
            state["transport"] = "rsync"
            state.pop("rsync_missing_until", None)
            return True, err
        if "not found" in err.lower():
            state["rsync_missing_until"] = (datetime.now(timezone.utc) + RSYNC_RECHECK).isoformat(timespec="seconds")
    if listing is None:
        return False, f"rsync failed ({err[-150:]}) and no remote listing for the scp fallback"
    ok, err2 = scp_pull(dest, stage, only if only else _changed(stage, listing), listing, timeout)
    state["transport"] = "scp"
    return ok, err2


def _parse(x) -> datetime | None:
    try:
        return datetime.fromisoformat(str(x)) if x else None
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------ promotion
def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _atomic_write(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def _atomic_link(src: Path, dst: Path) -> None:
    """dst becomes a hard link to src (one copy on disk); a plain copy if linking is impossible."""
    tmp = dst.with_name(dst.name + ".lnk")
    try:
        tmp.unlink(missing_ok=True)
        os.link(src, tmp)
        os.replace(tmp, dst)
    except OSError:
        tmp.unlink(missing_ok=True)
        _atomic_copy(src, dst)


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


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
            if old.startswith(complete):
                info.update(bytes=len(old), lines=old.count(b"\n"))
                return False, "staged copy is shorter than the last good copy (older or cut short); kept it", info
            conflict = final.with_name(final.name + ".conflict")
            _atomic_write(conflict, complete)
            return False, f"not an append-only extension of the last good copy; saved to {conflict.name}", info
    _atomic_write(final, complete)
    return True, "", info


def _migrate_state(state: dict, host: str) -> dict:
    """Older sync_state.json (one tag per host) -> per-tag keys relative to runs/remote/<host>/."""
    state["files"] = {(k if "/" in k else f"logs-{host}/{k}"): v for k, v in (state.get("files") or {}).items()}
    state.setdefault("ledgers", {})
    state.setdefault("outs", {})
    state.setdefault("shards", {})
    if isinstance(state.get("ledger"), dict):
        state["ledgers"].setdefault(f"ledger-{host}.jsonl", state.pop("ledger"))
    if isinstance(state.get("out"), dict):
        state["outs"].setdefault(f"{host}.out", state.pop("out"))
    return state


def _migrate_staging(stage: Path) -> None:
    """Older layout staged runs/full items directly under .staging/; move them under .staging/full/ (keeps rsync's
    delta basis)."""
    if not stage.exists():
        return
    for p in list(stage.iterdir()):
        if p.name in ("full", "shards", ".scp-tmp"):
            continue
        if (p.is_dir() and p.name.startswith("logs-")) or (p.is_file() and classify(f"full/{p.name}")):
            target = stage / "full" / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(p, target)


def _staged(stage: Path) -> dict[str, str]:
    """{runs/-relative path: kind} of every staged pullable file. Dot-files older than STALE_TEMP (temp files of an
    rsync that was killed at its timeout) are removed; only .staging is touched."""
    out = {}
    now = time.time()
    for sub in ("full", "shards"):
        d = stage / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.name.startswith("."):
                try:
                    if now - p.stat().st_mtime > STALE_TEMP:
                        p.unlink()
                except OSError:
                    pass
                continue
            rel = p.relative_to(stage).as_posix()
            kind = classify(rel)
            if kind:
                out[rel] = kind
    return out


def _stale(stage: Path, listing: dict | None, rels) -> set[str]:
    """Of `rels`, those whose staged copy is missing or older than this pass's listing (the pull did not reach them).
    Both mtimes are the remote's (rsync -a and scp -p preserve it), so clock skew does not matter."""
    if listing is None:
        return set()
    out = set()
    for rel in rels:
        if rel not in listing:
            continue
        size, mtime = listing[rel]
        try:
            st = (stage / rel).stat()
        except OSError:
            out.add(rel)
            continue
        if int(st.st_mtime) < int(mtime) or (int(st.st_mtime) == int(mtime) and st.st_size != size):
            out.add(rel)
    return out


def _promote_small(kind: str, staged: Path, final: Path, rec: dict, notes: list[str], name: str,
                   fresh: bool) -> None:
    """Promote a shard file, runner output or ledger; `fresh` = this pass's pull succeeded (data is current)."""
    st = staged.stat()
    rec["last_attempt"] = now_iso()
    if rec.get("staged_size") == st.st_size and rec.get("staged_mtime") == st.st_mtime and final.exists() \
            and "last_error" not in rec:
        if fresh:
            rec["last_success"] = rec["last_attempt"]  # unchanged on the remote since the last good promotion
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    if kind == "ledger":
        good, why, info = promote_ledger(staged, final)
        rec.update(info)
        if not good:
            rec["last_error"] = why
            notes.append(f"{name}: {why}")
            return
    elif kind == "shard":
        try:
            ids = json.loads(staged.read_text())["game_ids"]
            assert isinstance(ids, list)
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as e:
            rec["last_error"] = f"not a shard file: {type(e).__name__}"
            notes.append(f"{name}: {rec['last_error']}")
            return
        _atomic_copy(staged, final)
        rec["n_games"] = len(ids)
    else:
        _atomic_copy(staged, final)
    rec["bytes"] = final.stat().st_size
    rec.update(staged_size=st.st_size, staged_mtime=st.st_mtime)
    if fresh:
        rec["last_success"] = now_iso()
    rec.pop("last_error", None)


def _promote_eval(staged: Path, final: Path, rec: dict, deep: bool) -> bool:
    """Validate a staged .eval and hard-link it into place; return False (last good copy kept) if it is not usable."""
    st = staged.stat()
    if rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime and final.exists():
        fs = final.stat()
        if fs.st_size == rec["size"] and fs.st_mtime == rec["mtime"]:
            if not _same_file(staged, final):  # e.g. an -I re-pull rewrote an unchanged file: store it once again
                _atomic_link(final, staged)
            return True
    good, info, why = validate_eval(staged, deep=deep)
    rec["last_checked"] = now_iso()
    floor = rec.get("n_samples")
    if floor is None and final.exists() and not _same_file(staged, final):  # state lost: the good copy sets the floor
        ok_f, info_f, _ = validate_eval(final, deep=False)
        floor = info_f.get("n_samples") if ok_f else None
    if good and floor is not None and (info.get("n_samples") or 0) < floor and final.exists():
        good, why = False, f"{info['n_samples']} samples < last good {floor}"
    if not good:
        rec["last_error"] = why
        return False
    final.parent.mkdir(parents=True, exist_ok=True)
    _atomic_link(staged, final)
    rec.update({"size": st.st_size, "mtime": st.st_mtime, "n_samples": info.get("n_samples"),
                "n_errors": info.get("n_errors"), "status": info.get("status"),
                "started_at": info.get("started_at"), "promoted_at": now_iso()})
    rec.pop("last_error", None)
    return True


def _read_state(base: Path) -> dict:
    try:
        state = json.loads((base / "sync_state.json").read_text())
    except (OSError, ValueError):
        state = {}
    return state if isinstance(state, dict) else {}


def sync_host(host: str, dest: str, *, retries: int, retry_wait: float, deep: bool, timeout: int,
              base_root: Path = REMOTE_BASE) -> dict:
    """One sync pass for a host, under a per-host lock; a pass that finds the lock taken changes nothing."""
    base = Path(base_root) / host
    base.mkdir(parents=True, exist_ok=True)
    with open(base / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            state = _read_state(base)
            state["last_notes"] = ["another sync of this host is running; this pass skipped it (nothing changed)"]
            state["skipped"] = True
            return state
        try:
            return _sync_host_locked(host, dest, base, retries=retries, retry_wait=retry_wait, deep=deep,
                                     timeout=timeout)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _sync_host_locked(host: str, dest: str, base: Path, *, retries: int, retry_wait: float, deep: bool,
                      timeout: int) -> dict:
    stage = base / ".staging"
    _migrate_staging(stage)
    state_path = base / "sync_state.json"
    state = _migrate_state(_read_state(base), host)
    state.pop("skipped", None)
    state.update(host=host, dest=dest, last_attempt=now_iso())
    notes: list[str] = []

    listed_at = now_iso()
    res = remote_listing(dest, timeout)
    listing, lerr, procs = res[0], res[1], (res[2] if len(res) > 2 else None)
    if listing is None:
        notes.append(lerr)
    state["runners"] = {"at": listed_at, "tags": sorted(procs)} if procs is not None else None
    ok, err = pull_all(dest, stage, listing, state, timeout)
    if not ok:
        notes.append(f"pull failed: {err}")
        state["pull_last_error"] = err
    else:
        state.pop("pull_last_error", None)
        if err:
            notes.append(f"pull: {err[-160:]}")
    staged = _staged(stage)
    stale = _stale(stage, listing, listing or {})

    # small files: shard files, runner output, ledgers ("fresh" = the staged copy is current for this pass)
    sections = {"shard": "shards", "out": "outs", "ledger": "ledgers"}
    for rel, kind in sorted(staged.items()):
        if kind in sections:
            lrel = local_rel(rel)
            _promote_small(kind, stage / rel, base / lrel, state[sections[kind]].setdefault(lrel, {}), notes, lrel,
                           fresh=(rel not in stale) if listing is not None else ok)

    # Inspect logs, re-pulled (only the bad ones) while any copy is torn
    pending: set[str] = set()
    names = sorted(r for r, k in staged.items() if k == "eval")
    for attempt in range(retries + 1):
        pending = set()
        for rel in names:
            lrel = local_rel(rel)
            if not _promote_eval(stage / rel, base / lrel, state["files"].setdefault(lrel, {}), deep):
                pending.add(rel)
        if not pending:
            break
        notes.append(f"logs: {len(pending)} torn/invalid copy(ies) on attempt {attempt + 1}: "
                     + "; ".join(f"{local_rel(n)[:60]}: {state['files'][local_rel(n)].get('last_error')}"
                                 for n in sorted(pending)))
        if attempt == retries:
            break
        time.sleep(retry_wait)
        ok2, err2 = pull_all(dest, stage, listing, state, timeout, only=sorted(pending))
        if not ok2:
            notes.append(f"re-pull failed (attempt {attempt + 2}): {err2}")
        names = sorted(pending)
    stale = _stale(stage, listing, listing or {})  # again: the re-pulls may have caught up
    stale_evals = {r for r in stale if classify(r) == "eval"}
    if (ok or listing is not None) and not pending and not stale_evals:
        state["logs_last_success"] = now_iso()
    state["pending_invalid"] = sorted(local_rel(r) for r in pending)
    state["stale_after_pull"] = sorted(local_rel(r) for r in stale)
    if stale:
        notes.append(f"{len(stale)} file(s) not brought up to date by this pass: "
                     + ", ".join(sorted(local_rel(r) for r in stale)[:4]))

    # sync-correctness accounting per tag: remote files listed vs local good copies, and the time the tag's data is
    # known to be current as of (listing time of the last pass in which all its files were current and valid)
    prev_tags = state.get("tags") if isinstance(state.get("tags"), dict) else {}
    tags: dict[str, dict] = defaultdict(lambda: Counter())
    for lrel, rec in state["files"].items():
        if (base / lrel).exists():
            t = tags[tag_of(lrel)]
            t["evals_local"] += 1
            t["samples"] += rec.get("n_samples") or 0
    for lrel, rec in state["ledgers"].items():
        tags[tag_of(lrel)]["ledger_lines"] = rec.get("lines") or 0
        if rec.get("held_back_bytes"):
            tags[tag_of(lrel)]["ledger_torn_tail"] = 1
    not_current: set[str] = {tag_of(local_rel(r)) for r in pending | stale}
    for sec in ("ledgers", "outs", "shards"):
        not_current |= {tag_of(lrel) for lrel, rec in state[sec].items() if rec.get("last_error")}
    if listing is not None:
        missing = []
        for rel in listing:
            lrel = local_rel(rel)
            if classify(rel) == "eval":
                tags[tag_of(lrel)]["evals_remote"] += 1
            if not (base / lrel).exists():
                missing.append(lrel)
                not_current.add(tag_of(lrel))
        listed_tags = {tag_of(local_rel(rel)) for rel in listing}
        state["remote_listed_at"] = listed_at
        state["remote_files"] = len(listing)
        state["missing_locally"] = sorted(missing)
        if missing:
            notes.append(f"{len(missing)} remote file(s) without a good local copy: {', '.join(sorted(missing)[:4])}")
    else:
        listed_tags = set(tags) if ok else set()
    out_tags: dict[str, dict] = {}
    for t in sorted(set(tags) | listed_tags):
        rec = dict(tags.get(t) or {})
        prev = (prev_tags.get(t) or {}).get("data_at") if isinstance(prev_tags.get(t), dict) else None
        rec["data_at"] = listed_at if (t in listed_tags and t not in not_current) else prev
        # runner process seen at listing time (only for tags that had files then; None = unknown)
        rec["runner_alive"] = (t in procs) if (procs is not None and t in listed_tags) else None
        out_tags[t] = rec
    state["tags"] = out_tags
    state["last_notes"] = [" | ".join(x.strip() for x in n.splitlines() if x.strip()) for n in notes[-10:]]
    _atomic_write(state_path, json.dumps(state, indent=1, default=str).encode())
    return state


def report(host: str, state: dict) -> str:
    out = [f"{host} ({state.get('dest', '?')}, via {state.get('transport', '?')}): "
           + ("SKIPPED (locked), " if state.get("skipped") else "")
           + f"logs synced {state.get('logs_last_success') or 'never'}, "
             f"{state.get('remote_files', '?')} remote file(s) listed"]
    for tag, t in (state.get("tags") or {}).items():
        out.append(f"  {tag}: {t.get('evals_local', 0)}/{t.get('evals_remote', '?')} log(s) local/remote, "
                   f"{t.get('samples', 0)} saved samples, ledger {t.get('ledger_lines', '-')} lines"
                   + (" (torn tail held back)" if t.get("ledger_torn_tail") else "")
                   + f"; data as of {t.get('data_at') or 'never'}"
                   + {True: "; runner running", False: "; no runner process"}.get(t.get("runner_alive"), ""))
    if state.get("pending_invalid"):
        out.append(f"  still invalid (last good copy kept): {', '.join(state['pending_invalid'])}")
    for note in state.get("last_notes", []):
        out.append(f"  note: {note}")
    return "\n".join(out)


def main() -> None:
    hosts_all = load_hosts()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hosts", nargs="*", help=f"subset of {', '.join(hosts_all)} (default: all)")
    ap.add_argument("--retries", type=int, default=3, help="re-pulls when an .eval copy is torn")
    ap.add_argument("--retry-wait", type=float, default=20.0, help="seconds between re-pulls")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per rsync/scp call")
    ap.add_argument("--no-deep", action="store_true", help="skip the per-member CRC check")
    ap.add_argument("--watch", type=float, default=0, help="repeat every N seconds until interrupted")
    a = ap.parse_args()
    unknown = [h for h in a.hosts if h not in hosts_all]
    if unknown:
        ap.error(f"unknown host(s) {unknown}; known: {list(hosts_all)}")
    while True:
        hosts_all = load_hosts()  # a host added to runs/dispatch_hosts.json is picked up on the next pass
        for h in a.hosts or list(hosts_all):
            if h not in hosts_all:
                continue  # removed from runs/dispatch_hosts.json during --watch
            st = sync_host(h, hosts_all[h], retries=a.retries, retry_wait=a.retry_wait, deep=not a.no_deep,
                           timeout=a.timeout)
            print(report(h, st), flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
