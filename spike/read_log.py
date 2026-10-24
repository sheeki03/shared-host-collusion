"""Print per-sample spike results from the newest .eval log in runs/spike-logs."""
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

logs = sorted(Path("/Users/home/shared-host-collusion/runs/spike-logs").glob("*.eval"))
log = read_eval_log(str(sys.argv[1] if len(sys.argv) > 1 else logs[-1]))
print("status:", log.status, "| error:", (log.error.message[:300] if log.error else None))
for s in log.samples or []:
    print(f"\n=== sample {s.id} | error: {s.error.message[:300] if s.error else None}")
    for name, sc in (s.scores or {}).items():
        print(f"score {name}: {sc.value} | {sc.explanation}")
    r = (s.metadata or {}).get("spike")
    if not r:
        continue
    for k in ("shared", "before_write_b_sees", "b_reads_a", "a_reads_b", "owner_seen_by_b", "network_a", "whoami_a"):
        print(f"  {k}: {r.get(k)!r}")
    for g in r.get("game", []):
        print(f"  round {g['round']} firm {g['firm']}: price={g['price']!r} commands={json.dumps(g['commands'])[:200]}")
        for o in g["outputs"]:
            print(f"      out: {o[:160]!r}")
