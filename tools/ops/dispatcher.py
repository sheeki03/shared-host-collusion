"""Keep every machine busy: when a host's runner has exited, start the next unclaimed overflow chunk on it.

Run on the Mac:  nohup .venv/bin/python runs/dispatcher.py > runs/dispatcher.log 2>&1 &
Hosts are listed in runs/dispatch_hosts.json ({tag: ssh-destination-or-null}, e.g. "<user>@<host-ip>" or an ssh alias); add a new VPS there at any time (it gets chunks at
once). Claims are recorded in runs/dispatch_state.json so a chunk never runs twice. Chunks: runs/shards-v2/spill-1..4.
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "dispatch_state.json"
HOSTS = ROOT / "runs" / "dispatch_hosts.json"
CHUNKS = [f"spill-{i}" for i in range(1, 5)]
CAP, CONC = {"mac": 60, "vps": 60}, {"mac": 24, "<user>@<host-ip-vps1>": 18, "<user>@<host-ip-vps2>": 6, "instabox": 24,
                                     "instabox2": 28, "instabox3": 12}
PATTERN = "[Pp]ython -m harness.run --plan full"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def alive(ip, host: str = "") -> bool | None:
    # A host is busy while any of its runners is alive, except instabox, which runs core work beside the overflow
    # chunks: there only its overflow runners count.
    pattern = f"runs/shards/{host}-spil[l]-" if host == "instabox" else PATTERN  # [l]: never matches its own shell
    cmd = ["bash", "-c", f"pgrep -f '{pattern}' >/dev/null && echo yes || echo no"]
    if ip:
        cmd = SSH + [ip, f"pgrep -f '{pattern}' >/dev/null && echo yes || echo no"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=40).stdout
    except subprocess.TimeoutExpired:
        return None
    lines = [l for l in out.splitlines() if l in ("yes", "no")]
    return None if not lines else lines[-1] == "yes"


def launch(host: str, ip, chunk: str) -> None:
    tag = f"{host}-{chunk}"
    shard = ROOT / "runs" / "shards-v2" / f"{chunk}.json"
    if ip is None:
        (ROOT / "runs" / "shards" / f"{tag}.json").write_text(shard.read_text())
        cmd = (f"cd {ROOT} && INSPECT_DOCKER_CLI_CONCURRENCY=40 nohup .venv/bin/python -m harness.run --plan full "
               f"--shard-file runs/shards/{tag}.json --cap {CAP['mac']} --concurrency {CONC['mac']} --init-ledger "
               f"--ledger runs/full/ledger-{tag}.jsonl --log-dir runs/full/logs-{tag} > runs/full/{tag}.out 2>&1 "
               f"< /dev/null &")
        subprocess.run(["bash", "-c", cmd], check=True)
    else:
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(shard),
                        f"{ip}:shared-host-collusion/runs/shards/{tag}.json"], check=True, timeout=60)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(ROOT / "runs" / "launch_remote.sh"),
                        f"{ip}:launch_remote.sh"], check=True, timeout=60)
        r = subprocess.run(SSH + [ip, f"bash launch_remote.sh {tag} {CAP['vps']} {CONC.get(ip, 18)}"],
                           capture_output=True, text=True, timeout=120)
        log(f"remote: {r.stdout.strip()[-120:]} {r.stderr.strip()[-120:]}")
    log(f"LAUNCHED {chunk} on {host}")


def main() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {"claimed": {}}
    idle_since: dict = {}
    while True:
        hosts = json.loads(HOSTS.read_text())
        free = [c for c in CHUNKS if c not in state["claimed"]]
        if not free:
            log("all chunks claimed; dispatcher exits")
            return
        for host, ip in hosts.items():
            a = alive(ip, host)
            if a is None:
                log(f"{host}: unreachable")
                continue
            if a:
                idle_since.pop(host, None)
                continue
            # require two consecutive idle checks so a runner between phases is not mistaken for finished
            if host not in idle_since:
                idle_since[host] = time.time()
                log(f"{host}: runner not running (will confirm on next check)")
                continue
            free = [c for c in CHUNKS if c not in state["claimed"]]
            if not free:
                break
            chunk = free[0]
            state["claimed"][chunk] = {"host": host, "at": f"{datetime.now():%H:%M:%S}"}
            STATE.write_text(json.dumps(state, indent=1))
            try:
                launch(host, ip, chunk)
            except Exception as exc:  # noqa: BLE001
                log(f"launch of {chunk} on {host} failed: {exc!r}; releasing claim")
                state["claimed"].pop(chunk, None)
                STATE.write_text(json.dumps(state, indent=1))
            idle_since.pop(host, None)
        time.sleep(120)


if __name__ == "__main__":
    main()
