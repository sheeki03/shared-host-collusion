"""Docker sandboxes for the two agents.

Each agent gets its own container (no network, small CPU/memory/pid limits) running as its own uid
(firm A = 1001, firm B = 1002), so every file in /shared carries its author. /shared is either one volume
mounted into both containers (shared conditions) or a separate private volume per container (isolated
conditions); the prompt text is identical either way.
"""
from __future__ import annotations

import subprocess
import uuid

IMAGE = "ubuntu:24.04"
UIDS = {"A": 1001, "B": 1002}
CMD_TIMEOUT_S = 10
MAX_OUTPUT = 2000


def _docker(*args: str, timeout: int = 60, input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, input=input_text)


class HostPair:
    """Two containers for one game, with /shared shared or private."""

    def __init__(self, run_id: str, shared: bool):
        self.run_id = run_id
        self.shared = shared
        tag = uuid.uuid4().hex[:8]
        self.names = {f: f"shc-{run_id}-{f}-{tag}".lower() for f in ("A", "B")}
        if shared:
            vol = f"shc-{tag}-shared"
            self.volumes = {"A": vol, "B": vol}
        else:
            self.volumes = {f: f"shc-{tag}-{f}-private" for f in ("A", "B")}

    def start(self) -> None:
        for v in set(self.volumes.values()):
            _docker("volume", "create", v)
        for firm, name in self.names.items():
            uid = UIDS[firm]
            r = _docker(
                "run", "-d", "--name", name, "--network", "none", "--memory", "256m", "--cpus", "0.5",
                "--pids-limit", "64", "--hostname", "pricing-host",
                "-v", f"{self.volumes[firm]}:/shared", IMAGE, "sh", "-c",
                f"useradd -u {uid} -m -d /home/agent agent{uid} 2>/dev/null; "
                f"chmod 1777 /shared; sleep infinity",
            )
            if r.returncode != 0:
                raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
        # Wait until the users exist.
        for firm, name in self.names.items():
            for _ in range(50):
                r = _docker("exec", name, "id", "-u", f"agent{UIDS[firm]}")
                if r.returncode == 0:
                    break

    def exec(self, firm: str, command: str) -> dict:
        name, uid = self.names[firm], UIDS[firm]
        try:
            r = _docker("exec", "-u", str(uid), "-w", "/home/agent", "-e", "HOME=/home/agent", name,
                        "timeout", str(CMD_TIMEOUT_S), "bash", "-c", command, timeout=CMD_TIMEOUT_S + 15)
            out = (r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else ""))
            code = r.returncode
        except subprocess.TimeoutExpired:
            out, code = "[timed out]", 124
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n[... truncated {len(out) - MAX_OUTPUT} chars]"
        return {"exit": code, "output": out}

    def read_notes(self, firm: str, limit: int = 1500) -> str:
        r = _docker("exec", self.names[firm], "sh", "-c", "head -c 4000 /home/agent/notes.md 2>/dev/null")
        text = r.stdout
        return text if len(text) <= limit else text[-limit:]

    def snapshot_shared(self) -> list[dict]:
        """Every file under /shared (as seen from A's container) with owner uid and the first 2 KB."""
        script = (
            "cd /shared 2>/dev/null || exit 0; "
            "find . -type f -print0 2>/dev/null | while IFS= read -r -d '' f; do "
            "printf '@@FILE %s %s\\n' \"$(stat -c %u \"$f\")\" \"$f\"; head -c 2048 \"$f\"; printf '\\n@@END\\n'; done"
        )
        snaps = []
        for firm in (("A",) if self.shared else ("A", "B")):
            r = _docker("exec", self.names[firm], "bash", "-c", script)
            cur = None
            for line in r.stdout.splitlines():
                if line.startswith("@@FILE "):
                    _, uid, path = line.split(" ", 2)
                    cur = {"view": firm, "uid": int(uid), "path": path, "content": ""}
                elif line == "@@END" and cur is not None:
                    snaps.append(cur)
                    cur = None
                elif cur is not None:
                    cur["content"] += line + "\n"
        return snaps

    def wipe_shared(self) -> None:
        for firm in (("A",) if self.shared else ("A", "B")):
            _docker("exec", self.names[firm], "sh", "-c", "find /shared -mindepth 1 -delete 2>/dev/null; true")

    def write_shared_as(self, firm: str, relpath: str, content: str) -> None:
        """Used by the scripted provocateur to write into /shared under its own uid."""
        _docker("exec", "-i", "-u", str(UIDS[firm]), self.names[firm], "sh", "-c", f"cat > /shared/{relpath}",
                input_text=content)

    def stop(self) -> None:
        for name in self.names.values():
            _docker("rm", "-f", name)
        for v in set(self.volumes.values()):
            _docker("volume", "rm", "-f", v)
