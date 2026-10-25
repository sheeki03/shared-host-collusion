"""Shell runner for the firm sandboxes (H2).

Agent commands
- `run(firm, cmd)` makes one Inspect/Docker exec on the normal path. A root supervisor passes the command as a
  *positional argument* (never interpolated into its own shell), drops to the firm's uid:gid with a clean environment,
  executes `bash -o pipefail -c <cmd>` once under an inner timeout, then reaps and verifies all firm-UID processes
  before returning. The agent itself has no effective capabilities. Inspect has `timeout_retry=False`.
- The supervisor adds a status marker only after cleanup; Python removes it from the agent's stderr. Missing or
  malformed markers, failed cleanup, or an outer Inspect timeout fail closed and trigger a separate best-effort
  root reaper before the game aborts. The `reap=False` test path deliberately uses the old direct agent exec.
- Agents see stdout and stderr each cut at 2,000 characters with a marker; `output` is the combined agent view
  (stdout, then "[stderr]" and stderr), also cut at 2,000. The untruncated streams are kept for research up to 64 KB
  each in `raw_stdout` / `raw_stderr`.

Harness commands
- `root(firm, cmd)` runs as root (uid 0) for harness-only work. `root_write` writes a file atomically and
  symlink-safely (rename over the target, never through it; refuses symlinked parent directories). `setup()` and
  `write_bulletin()` build on it.
"""
from __future__ import annotations

import re
import time

from inspect_ai.util import OutputLimitExceededError, sandbox

from harness.sandbox_env import BULLETIN, FIRMS, HOME, INIT_SCRIPT, SERVICE, UID, USERNAME, user_spec

MAX_OUTPUT = 2000
RAW_CAP = 65536
DEFAULT_TIMEOUT = 10
ROOT_TIMEOUT = 60
AGENT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_BULLETIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SUPERVISOR_MARKER = "\n__SHC_SUPERVISOR_STATUS__:"


def _cut(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n[... truncated {len(text) - limit} chars]", True


def _argv_error(cmd: str) -> str | None:
    """Why `cmd` cannot be passed as an argv string (NUL byte, lone surrogate from JSON text), or None if it can."""
    if "\0" in cmd:
        return "command contains a NUL byte"
    try:
        cmd.encode("utf-8")
    except UnicodeEncodeError as exc:
        return f"command is not encodable as UTF-8: {exc.reason} at {exc.start}"
    return None


def agent_env(firm: str) -> dict[str, str]:
    """Environment for agent commands (identical across conditions apart from the firm's own user name)."""
    return {"HOME": HOME, "USER": USERNAME[firm], "LOGNAME": USERNAME[firm], "PATH": AGENT_PATH}


_REAP_FUNCTION = r"""
runnable_target() {
  [ -r "/proc/$1/status" ] || return 1
  uid= state=
  while IFS=$' \t' read -r field value rest; do
    case "$field" in
      Uid:) uid=$value ;;
      State:) state=$value ;;
    esac
    [ -n "$uid" ] && [ -n "$state" ] && break
  done < "/proc/$1/status"
  [ "$uid" = "$target" ] && [ "$state" != Z ] && [ "$state" != X ]
}
active() {
  for d in /proc/[0-9]*; do
    pid=${d##*/}
    runnable_target "$pid" && printf '%s\n' "$pid"
  done
}
reap_uid() {
  for pid in $(active); do
    # A process may exit after enumeration. Recheck UID immediately before signalling a reused PID.
    runnable_target "$pid" && kill -KILL "$pid" 2>/dev/null || :
  done
  for i in 1 2 3 4 5; do
    [ -z "$(active)" ] && return 0
    sleep 0.05
  done
  echo 'agent process survived reaper' >&2
  return 4
}
"""


_AGENT_SUPERVISOR = r"""
target=$1; seconds=$2; agent_name=$3; agent_cmd=$4
case "$target" in 1001|1002) : ;; *) echo 'bad agent uid' >&2; exit 92 ;; esac
command -v /usr/bin/setpriv >/dev/null || { echo 'setpriv missing' >&2; exit 93; }
cd /home/agent || exit 94
/usr/bin/timeout -k 5s "$seconds" /usr/bin/env -i HOME=/home/agent USER="$agent_name" LOGNAME="$agent_name" \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  /usr/bin/setpriv --reuid="$target" --regid="$target" --clear-groups --no-new-privs \
  /bin/bash -o pipefail -c "$agent_cmd"
agent_status=$?
__REAPER_FUNCTION__
reap_uid || exit 95
printf '\n__SHC_SUPERVISOR_STATUS__:%s\n' "$agent_status" >&2
exit 0
""".replace("__REAPER_FUNCTION__", _REAP_FUNCTION)


def _supervisor_status(stderr: str) -> tuple[int, str]:
    """Remove the root-only trailing status marker; any malformed/missing marker fails closed."""
    head, marker, tail = stderr.rpartition(_SUPERVISOR_MARKER)
    if not marker or not tail.endswith("\n") or not tail[:-1].isdigit():
        raise ValueError("supervisor status marker missing or malformed")
    status = int(tail[:-1])
    if not 0 <= status <= 255:
        raise ValueError("supervisor returned an invalid command status")
    return status, head


async def _reap(firm: str) -> bool:
    """Fallback root reaper for abnormal supervisor/Inspect failures; verify no firm process remains.

    Running ``kill -KILL -1`` as the agent also kills Inspect's same-UID timeout wrapper, making a successful cleanup
    appear to fail. This root process has only CAP_KILL in addition to the harness's normal minimal capabilities;
    it enumerates /proc and targets the specific numeric firm UID, never root or the rival UID. UID is checked again
    immediately before each signal. Shell kill has a residual check-to-signal PID-reuse race; its possible effect is
    limited to the current isolated container, and any cleanup anomaly aborts the game.
    """
    script = "target=$1\n" + _REAP_FUNCTION + "\nreap_uid\n"
    try:
        r = await sandbox(SERVICE[firm]).exec(["/bin/bash", "-c", script, "shc-reap", str(UID[firm])],
                                              user="0:0", cwd="/",
                                              timeout=5, timeout_retry=False)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - the caller must abort when cleanup cannot be verified
        return False


async def run(firm: str, cmd: str, timeout: int = DEFAULT_TIMEOUT, *, reap: bool = True) -> dict:
    """Run one agent command once. Returns {exit, stdout, stderr, truncated, secs, ...} (see module docstring)."""
    t0 = time.monotonic()
    timed_out = False
    error = None
    reaped = None if not reap else False
    bad_text = _argv_error(cmd)
    try:
        if bad_text is not None:  # exec would raise ValueError before running anything; refuse instead of crashing
            code, out, err, error = 126, "", "[command not run: it contains a NUL byte or invalid text]", bad_text
            if reap:
                reaped = await _reap(firm)
        elif reap:
            r = await sandbox(SERVICE[firm]).exec(
                ["/bin/bash", "-c", _AGENT_SUPERVISOR, "shc-supervisor", str(UID[firm]), str(timeout),
                 USERNAME[firm], cmd],
                user="0:0", cwd="/", timeout=timeout + 20, timeout_retry=False)
            out, err = r.stdout or "", r.stderr or ""
            if r.returncode == 0:
                try:
                    code, err = _supervisor_status(err)
                    timed_out = code == 124
                    reaped = True
                except ValueError as exc:
                    code, error = 125, str(exc)
            else:
                code, error = 125, f"root supervisor exited {r.returncode}"
        else:
            r = await sandbox(SERVICE[firm]).exec(["bash", "-o", "pipefail", "-c", cmd], user=user_spec(firm),
                                                  cwd=HOME, env=agent_env(firm), timeout=timeout,
                                                  timeout_retry=False)
            code, out, err = r.returncode, r.stdout or "", r.stderr or ""
    except TimeoutError as exc:
        timed_out = True
        code, out, err = 124, str(getattr(exc, "truncated_output", "") or ""), f"[timed out after {timeout}s]"
    except OutputLimitExceededError as exc:
        code, out, err, error = 125, str(getattr(exc, "truncated_output", "") or ""), "[output limit exceeded]", repr(exc)
    except UnicodeDecodeError as exc:
        code, out, err, error = 125, "", "[output was not valid UTF-8]", repr(exc)
    if reap and reaped is False:
        await _reap(firm)
    secs = round(time.monotonic() - t0, 3)
    stdout, t_out = _cut(out, MAX_OUTPUT)
    stderr, t_err = _cut(err, MAX_OUTPUT)
    combined = out + (("\n[stderr]\n" + err) if err.strip() else "")
    output, t_comb = _cut(combined, MAX_OUTPUT)
    raw_out, raw_t1 = _cut(out, RAW_CAP)
    raw_err, raw_t2 = _cut(err, RAW_CAP)
    return {
        "cmd": cmd, "exit": code, "stdout": stdout, "stderr": stderr, "output": output,
        "truncated": t_out or t_err or t_comb, "secs": secs, "timed_out": timed_out, "reaped": reaped,
        "raw_stdout": raw_out, "raw_stderr": raw_err, "raw_truncated": raw_t1 or raw_t2, "error": error,
    }


async def root(firm: str, cmd: str, *, input: str | bytes | None = None, timeout: int = ROOT_TIMEOUT,
               check: bool = True, args: list[str] | None = None) -> dict:
    """Run a harness-only command as root in the firm's container (pipefail, never retried).

    `args` become "$1", "$2", ... of the script. Raises RuntimeError on a non-zero exit when `check` is true.
    """
    t0 = time.monotonic()
    argv = ["/bin/bash", "-o", "pipefail", "-c", cmd, "shc-root", *(args or [])]
    r = await sandbox(SERVICE[firm]).exec(argv, input=input, user="0:0", cwd="/", timeout=timeout,
                                          timeout_retry=False)
    res = {"exit": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "secs": round(time.monotonic() - t0, 3)}
    if check and r.returncode != 0:
        raise RuntimeError(f"harness root command failed on {SERVICE[firm]} (exit {r.returncode}): "
                           f"{cmd[:200]!r}: {r.stderr[:300]}")
    return res


# Atomic, symlink-safe write. The parent directory is entered with `cd -P` and must already be its own canonical path
# (no symlinked component); the temp file is created there with O_EXCL (mktemp) and renamed over the target, so a
# planted symlink at the target is replaced rather than followed, and a directory at the target makes it fail.
_ROOT_WRITE = r"""
set -eu
target=$1; mode=$2; owner=$3
dir=${target%/*}; base=${target##*/}
[ -n "$dir" ] && [ -n "$base" ] && [ "$base" != . ] && [ "$base" != .. ] || { echo "bad target: $target" >&2; exit 3; }
cd -P -- "$dir"
[ "$(pwd -P)" = "$dir" ] || { echo "parent of $target is not a canonical directory" >&2; exit 3; }
tmp=$(mktemp -p . .shc-write.XXXXXXXX)
trap 'rm -f -- "$tmp"' EXIT
cat > "$tmp"
chown -- "$owner" "$tmp"
chmod -- "$mode" "$tmp"
mv -fT -- "$tmp" "$base"
trap - EXIT
"""


async def root_write(firm: str, path: str, content: str | bytes, *, mode: str = "644", owner: str = "0:0") -> None:
    """Write `content` to absolute `path` in the firm's container as root, atomically and symlink-safely."""
    if not path.startswith("/") or "\0" in path:
        raise ValueError(f"root_write needs an absolute path, got {path!r}")
    if not re.fullmatch(r"[0-7]{3,4}", mode) or not re.fullmatch(r"\d+:\d+", owner):
        raise ValueError(f"bad mode/owner {mode!r} {owner!r}")
    await root(firm, _ROOT_WRITE, input=content, args=[path, mode, owner])


async def setup(firms=FIRMS) -> None:
    """Idempotent per-sample initialisation (scratch mode, protected bulletin). Call before any agent command."""
    for firm in firms:
        await root(firm, INIT_SCRIPT)


async def write_bulletin(name: str, content: str | bytes, *, shared: bool) -> list[str]:
    """Publish identical public bytes in each firm's protected bulletin mount, independent of scratch sharing."""
    if not _BULLETIN_NAME.fullmatch(name):
        raise ValueError(f"bad bulletin file name {name!r}")
    for firm in FIRMS:
        await root_write(firm, f"{BULLETIN}/{name}", content, mode="644", owner="0:0")
    return [SERVICE[f] for f in FIRMS]
