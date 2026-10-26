"""Progress of the confirmatory run on every machine (local helper; prints no secrets)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOSTS = {"instabox-v5": "instabox", "instabox-dsgpt": "instabox", "instabox2-v5": "instabox2", "instabox3-v5": "instabox3",
         "macv3-gated": None, "vps1-pool": "<user>@<host-ip-vps1>",
         "instabox-core": "instabox", "instabox-spill-2": "instabox", "instabox-spill-3": "instabox",
         "instabox-spill-4": "instabox", "instabox2-pool": "instabox2", "instabox3-pool": "instabox3",
         "instabox2-spill-3": "instabox2", "instabox2-spill-4": "instabox2", "instabox3-spill-3": "instabox3",
         "instabox3-spill-4": "instabox3", "vps1-spill-3": "<user>@<host-ip-vps1>", "vps1-spill-4": "<user>@<host-ip-vps1>"}
REMOTE = r"""
cd ~/shared-host-collusion 2>/dev/null || cd {root}
python3 - <<'PY'
import json, collections, os, glob, re
tag = "{tag}"
L = f"runs/full/ledger-{{tag}}.jsonl"
ev = [json.loads(l) for l in open(L)] if os.path.exists(L) else []
st = [e for e in ev if e["event"] == "settle"]
spent = sum(e["actual"] for e in st)
flags = collections.Counter(e.get("reason", "")[:40] for e in ev if e["event"] == "flag")
out = open(f"runs/full/{{tag}}.out", errors="replace").read() if os.path.exists(f"runs/full/{{tag}}.out") else ""
errs = len(re.findall(r"WARNING .*sample=", out))
logs = glob.glob(f"runs/full/logs-{{tag}}/*.eval")
print(json.dumps({{"settled_calls": len(st), "ledger_spent": round(spent, 4), "open": len([e for e in ev if e["event"]=="reserve"]) - len(st) - len([e for e in ev if e["event"]=="release"]),
                  "flags": dict(flags), "sample_warnings": errs, "eval_files": len(logs),
                  "load": os.getloadavg()[0], "tail": out.strip().splitlines()[-1][:120] if out.strip() else ""}}))
PY
docker ps -q | wc -l
"""


def run(tag, ip):
    script = REMOTE.format(tag=tag, root=ROOT)
    cmd = ["bash", "-c", script] if ip is None else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                                     ip, script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    lines = [l for l in r.stdout.splitlines() if not l.startswith("***")]
    return lines, r.stderr.strip()[-200:]


if __name__ == "__main__":
    for tag, ip in HOSTS.items():
        if len(sys.argv) > 1 and tag not in sys.argv[1:]:
            continue
        lines, err = run(tag, ip)
        print(f"== {tag}: containers={lines[-1].strip() if lines else '?'}")
        for l in lines[:-1]:
            print("  ", l)
        if err and not lines:
            print("   ERR", err)
