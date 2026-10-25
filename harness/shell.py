"""Shell runner for the firm sandboxes (H2).

Agent commands
- `run(firm, cmd)` executes `bash -o pipefail -c <cmd>` once, as the firm's uid:gid, in /home/agent, through Inspect's
  `sandbox(service).exec(..., timeout=timeout, timeout_retry=False)`. Inspect is never allowed to re-run it: with
  `timeout_retry=False` neither its timeout retry nor its "CLI exited before reading stdin" retry fires, and this
  module never retries anything itself.
- On timeout Inspect's in-container `/usr/bin/timeout -k 5s` kills the command's process group and raises
  TimeoutError; the result then has exit 124, `timed_out=True` and whatever partial output Inspect kept.
- After every command (finished or timed out), a root harness reaper identifies only processes with this firm's
  numeric UID inside its isolated container, kills them, then verifies none runnable remain. Unlike a same-UID
  `kill -KILL -1`, this does not kill Inspect's timeout wrapper. A failure makes the game abort before another
  private-data write.
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


async def _reap(firm: str) -> bool:
    """Kill only the firm's UID from a root wrapper, then verify no runnable process remains.

    Running ``kill -KILL -1`` as the agent also kills Inspect's same-UID timeout wrapper, making a successful cleanup
    appear to fail. This root process has only CAP_KILL in addition to the harness's normal minimal capabilities;
    it enumerates /proc and targets the specific numeric firm UID, never root or the rival UID. UID is checked again
    immediately before each signal. Shell kill has a residual check-to-signal PID-reuse race; its possible effect is
    limited to the current isolated container, and any cleanup anomaly aborts the game.
    """
    script = r"""
target=$1
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
for pid in $(active); do
  # A process may exit after enumeration. Recheck UID immediately before signalling a reused PID.
  runnable_target "$pid" && kill -KILL "$pid" 2>/dev/null || :
done
for i in 1 2 3 4 5; do
  [ -z "$(active)" ] && exit 0
  sleep 0.05
done
echo 'agent process survived reaper' >&2
exit 4
"""
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
    bad_text = _argv_error(cmd)
    try:
        if bad_text is not None:  # exec would raise ValueError before running anything; refuse instead of crashing
            code, out, err, error = 126, "", "[command not run: it contains a NUL byte or invalid text]", bad_text
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
    secs = round(time.monotonic() - t0, 3)
    reaped = await _reap(firm) if reap else None
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
