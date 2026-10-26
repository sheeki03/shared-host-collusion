"""Extension dispatcher: when a host has no harness runner alive (confirmed on two checks), start the next unclaimed
extension chunk there. ext-1 (24 x 100-round games, the critical path) only goes to hosts whose concurrency >= 24.
Hosts: runs/dispatch_ext_hosts.json {name: ssh-destination}; state: runs/dispatch_ext_state.json.
Run: nohup .venv/bin/python runs/dispatcher_ext.py > runs/dispatcher_ext.log 2>&1 &
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "dispatch_ext_state.json"
HOSTS = ROOT / "runs" / "dispatch_ext_hosts.json"
CHUNKS = ["ext-4", "ext-2", "ext-3", "ext-1"]  # E3+E4 first, E1 halves, 100-round E2 last (review item 10)
CONC = {"instabox": 40, "instabox2": 32, "instabox3": 14, "<user>@<host-ip-vps1>": 18}
CAP = {"ext-1": 60, "ext-2": 80, "ext-3": 80, "ext-4": 60}
NEEDS_CONC = {"ext-1": 24}
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
PATTERN = "[Pp]ython -m harness.run --plan"


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def busy(dest: str) -> bool | None:
    try:
        out = subprocess.run(SSH + [dest, f"pgrep -f '{PATTERN}' >/dev/null && echo yes || echo no"],
                             capture_output=True, text=True, timeout=60).stdout
    except subprocess.TimeoutExpired:
        return None
    lines = [l for l in out.splitlines() if l in ("yes", "no")]
    return None if not lines else lines[-1] == "yes"


def launch(host: str, dest: str, chunk: str, ref: str) -> None:
    tag = f"{host}-{chunk}"
    shard = ROOT / "runs" / "shards-ext" / f"{chunk}.json"
    subprocess.run(SSH + [dest, "mkdir -p shared-host-collusion/runs/shards"], check=True, timeout=60)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(shard),
                    f"{dest}:shared-host-collusion/runs/shards/{tag}.json"], check=True, timeout=60)
    subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(ROOT / "runs" / "launch_remote_ext.sh"),
                    f"{dest}:launch_remote_ext.sh"], check=True, timeout=60)
    r = subprocess.run(SSH + [dest, f"bash launch_remote_ext.sh {tag} {CAP[chunk]} {CONC.get(dest, 14)} {ref}"],
                       capture_output=True, text=True, timeout=180)
    log(f"remote {host}: {r.stdout.strip()[-160:]} {r.stderr.strip()[-160:]}")
    if "launched" not in r.stdout:
        raise RuntimeError("launch did not confirm")
    log(f"LAUNCHED {chunk} on {host}")


def main() -> None:
    ref = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log(f"extension code ref {ref[:7]}")
    state = json.loads(STATE.read_text()) if STATE.exists() else {"claimed": {}}
    idle: dict = {}
    while True:
        free = [c for c in CHUNKS if c not in state["claimed"]]
        if not free:
            log("all extension chunks claimed; exiting")
            return
        for host, dest in json.loads(HOSTS.read_text()).items():
            b = busy(dest)
            if b is None:
                log(f"{host}: unreachable")
                continue
            if b:
                idle.pop(host, None)
                continue
            if host not in idle:
                idle[host] = time.time()
                log(f"{host}: no runner (confirming next check)")
                continue
            free = [c for c in CHUNKS if c not in state["claimed"]
                    and CONC.get(dest, 14) >= NEEDS_CONC.get(c, 0)]
            if not free:
                continue
            chunk = free[0]
            state["claimed"][chunk] = {"host": host, "at": f"{datetime.now():%H:%M:%S}", "ref": ref[:7]}
            STATE.write_text(json.dumps(state, indent=1))
            try:
                launch(host, dest, chunk, ref)
            except Exception as exc:  # noqa: BLE001
                log(f"launch of {chunk} on {host} failed: {exc!r}; releasing")
                state["claimed"].pop(chunk, None)
                STATE.write_text(json.dumps(state, indent=1))
            idle.pop(host, None)
        time.sleep(90)


if __name__ == "__main__":
    main()
