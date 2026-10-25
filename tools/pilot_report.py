"""Summarise pilot logs into the numbers the Step 5 decisions need (no model calls).

Usage: .venv/bin/python tools/pilot_report.py <log_dir> [<log_dir> ...]
Prints, per model configuration: games, rounds, calls, cost per call / per round, median and p90 call latency,
max input / output / reasoning tokens, finish reasons, parse failures; per arm: scratch commands and crossings;
prompt growth in long games; and full-grid cost projections from the measured cost per round.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

from harness.arms import full_design  # noqa: E402

SCRATCH = re.compile(r"/srv/scratch|\bscratch/")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main(dirs: list[str]) -> None:
    by_model = defaultdict(lambda: {"games": 0, "rounds": 0, "calls": [], "parse": 0, "llm_rows": 0, "errors": [],
                                    "secs_game": []})
    by_arm = defaultdict(lambda: {"games": 0, "scratch_cmds": 0, "cmds": 0, "canary": 0, "content": 0, "name": 0,
                                  "llm_rows": 0, "blocked": 0, "chat": 0, "chat_blocked": 0})
    growth = []
    for d in dirs:
        for info in list_eval_logs(d):
            log = read_eval_log(info)
            for s in log.samples or []:
                m = s.metadata or {}
                label, arm = m.get("model_label"), m.get("arm")
                res = m.get("result") or {}
                bm = by_model[label]
                if s.error:
                    bm["errors"].append(f"{s.id}: {str(s.error.message)[:100]}")
                    continue
                rows = [r for r in res.get("rows", []) if not r.get("scripted")]
                bm["games"] += 1
                bm["rounds"] += int(m.get("rounds") or 0)
                bm["calls"] += res.get("calls", [])
                bm["parse"] += sum(1 for r in rows if r.get("parse_failed"))
                bm["llm_rows"] += len(rows)
                if getattr(s, "total_time", None):
                    bm["secs_game"].append((s.total_time, int(m.get("rounds") or 0)))
                ba = by_arm[(m.get("scenario_id"), arm)]
                ba["games"] += 1
                ba["llm_rows"] += len(rows)
                for f, outs in (res.get("outputs") or {}).items():
                    for o in outs:
                        ba["cmds"] += 1
                        ba["scratch_cmds"] += bool(SCRATCH.search(o.get("cmd") or ""))
                        ba["blocked"] += bool(o.get("blocked"))
                ba["canary"] += sum(1 for r in rows if r.get("crossing_canary"))
                ba["content"] += sum(1 for r in rows if r.get("crossing_content"))
                ba["name"] += sum(1 for r in rows if r.get("name_exposure"))
                ba["chat"] += len(res.get("chat", []))
                ba["chat_blocked"] += sum(1 for c in res.get("chat", []) if c.get("blocked"))
                if int(m.get("rounds") or 0) >= 20:
                    per_t = defaultdict(list)
                    for c in res.get("calls", []):
                        if c.get("input_tokens"):
                            per_t[c["t"]].append(c["input_tokens"])
                    growth.append((s.id, {t: max(v) for t, v in sorted(per_t.items()) if t in (1, 5, 10, 20, 30, 40)}))

    print(f"{'model':28s} games rnds calls  $/call    $/round  lat_med lat_p90  in_max out_max rsn_max parse  finish")
    cost_round = {}
    for label, b in sorted(by_model.items(), key=lambda kv: str(kv[0])):
        calls = b["calls"]
        cost = [c.get("cost_usd") or 0 for c in calls]           # billed cost of the attempt that answered
        overhead = sum((c.get("total_cost_usd") or 0) - (c.get("cost_usd") or 0) for c in calls)  # fail-closed settles
        lat = [c.get("secs") for c in calls if c.get("secs") is not None]
        fin = defaultdict(int)
        for c in calls:
            fin[c.get("finish_reason")] += 1
        cr = sum(cost) / b["rounds"] if b["rounds"] else None
        cost_round[label] = cr
        print(f"{str(label):28s} {b['games']:5d} {b['rounds']:4d} {len(calls):5d}  "
              f"{(sum(cost) / len(cost)) if cost else 0:.5f}  {cr or 0:.5f}  {pct(lat, .5) or 0:7.1f} {pct(lat, .9) or 0:7.1f}  "
              f"{max((c.get('input_tokens') or 0) for c in calls) if calls else 0:6d} "
              f"{max((c.get('output_tokens') or 0) for c in calls) if calls else 0:7d} "
              f"{max((c.get('reasoning_tokens') or 0) for c in calls) if calls else 0:7d} "
              f"{b['parse']:3d}/{b['llm_rows']:<4d} {dict(fin)} ledger_overhead=${overhead:.3f}")
        for e in b["errors"]:
            print(f"    ERROR {e}")
    print("\nper (scenario, arm): games, commands, scratch commands, canary/content/name crossings (firm-rounds), "
          "gate blocks, chat (blocked)")
    for (scen, arm), a in sorted(by_arm.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        print(f"  {str(scen):22s} {str(arm):18s} g={a['games']:2d} cmds={a['cmds']:4d} scratch={a['scratch_cmds']:3d} "
              f"cross={a['canary']}/{a['content']}/{a['name']} of {a['llm_rows']} blocked={a['blocked']} "
              f"chat={a['chat']}({a['chat_blocked']})")
    if growth:
        print("\nprompt growth (max input tokens at round t):")
        for gid, g in growth:
            print(f"  {gid}: {g}")

    # projection: measured $/round per model (pilot rounds are shorter, so a 40-round game costs a bit more per round)
    design = full_design()
    proj, missing = 0.0, set()
    for s in design:
        cr = cost_round.get(s.model_label)
        if cr is None:
            missing.add(s.model_label)
            continue
        proj += cr * s.rounds
    print(f"\nfull grid ({len(design)} games, {sum(s.rounds for s in design)} rounds): measured-rate subject cost "
          f"≈ ${proj:,.2f}" + (f" (no pilot rate for {sorted(missing)})" if missing else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
