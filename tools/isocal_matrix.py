"""Shared-resource matrix of the A4 ISO residual-channel calibration (prereg/EXPLORATORY.md section 5): result keys
and a markdown table from the isocal-<condition>-seed<seed>.json files that tools/iso_calibration.py writes.

Usage (repo root, venv python):
  .venv/bin/python tools/isocal_matrix.py --run contabo_idle=runs/.../isocal-idle-seed0.json \
      --run mac_idle=runs/calibration/mac-idle-seed0/isocal-idle-seed0.json ... --out runs/analysis/isocal
Writes <out>/isocal_results.json (flat keys x.isocal.*) and <out>/tables/isocal_matrix.md ({{TABLE:isocal_matrix}}).

Keys, per run label <r> (the --run label: letters, digits and underscores) and channel <c> (memory, load, pagecache,
storage; primary statistic only, as the decision rule):
  x.isocal.<r>.{condition,platform,os,kernel,cpus,containers_start,load_start,load_end,complete,protocol_as_declared}
  x.isocal.<r>.<c>.{status,statistic,correct,n_test,accuracy,ci_lo,ci_hi,capacity,capacity_lower,channel}
  x.isocal.<r>.<c>.stage (the failed stage when status is not ok, e.g. setup:probe)
  x.isocal.n_runs, x.isocal.channel_runs.<c> (runs in which <c> is a channel), x.isocal.unavailable_runs.<c>
  (runs in which its setup failed, so neither channel nor null), x.isocal.label
The condition is the caller's label recorded by the tool; load_start and load_end (1-minute load average) say what the
host actually carried, so the text can say when a run labelled "loaded" lost its load. Nothing here is a model result:
a channel row shows capacity the ISO sandbox leaves in place, not that any agent used it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CHANNELS = ("memory", "load", "pagecache", "storage")
NAMES = {"memory": "Shared memory (`Shmem` in /proc/meminfo)", "load": "Load average (/proc/loadavg)",
         "pagecache": "Page cache (read timing of a shared image file)", "storage": "Storage write timing (control)"}
LABEL = re.compile(r"^[A-Za-z0-9_]+$")


def run_keys(label: str, doc: dict) -> dict:
    meta, out = doc["meta"], {}
    p = f"x.isocal.{label}"
    dk = meta.get("docker") or {}
    out[f"{p}.condition"] = meta.get("condition")
    out[f"{p}.platform"] = meta.get("platform")
    out[f"{p}.os"] = dk.get("OperatingSystem")
    out[f"{p}.kernel"] = dk.get("KernelVersion")
    out[f"{p}.cpus"] = dk.get("NCPU")
    out[f"{p}.containers_start"] = dk.get("ContainersRunning")
    out[f"{p}.load_start"] = round(float((meta.get("host_start") or {}).get("loadavg", [float("nan")])[0]), 2)
    out[f"{p}.load_end"] = round(float((meta.get("host_end") or {}).get("loadavg", [float("nan")])[0]), 2)
    out[f"{p}.complete"] = bool(doc.get("complete")) and not doc.get("interrupted")
    out[f"{p}.protocol_as_declared"] = bool(meta.get("protocol_as_declared"))
    for c in CHANNELS:
        ch = (doc.get("channels") or {}).get(c)
        q = f"{p}.{c}"
        if not ch or ch.get("status") != "ok" or not ch.get("analysis"):
            out[f"{q}.status"] = (ch or {}).get("status") or "not run"
            out[f"{q}.stage"] = ((ch or {}).get("error") or {}).get("stage") or "-"
            out[f"{q}.channel"] = "not available"
            continue
        a = ch["analysis"]
        s = a["stats"][a["primary"]]
        out.update({f"{q}.status": "ok", f"{q}.statistic": a["primary"], f"{q}.correct": int(s["correct"]),
                    f"{q}.n_test": int(s["n_test"]), f"{q}.accuracy": round(float(s["accuracy"]), 3),
                    f"{q}.ci_lo": round(float(s["ci_lower"]), 3), f"{q}.ci_hi": round(float(s["ci_upper"]), 3),
                    f"{q}.capacity": round(float(s["capacity_bits_per_round"]), 3),
                    f"{q}.capacity_lower": round(float(s["capacity_lower_bits_per_round"]), 3),
                    f"{q}.channel": "yes" if s["is_channel"] else "no"})
    return out


def table(labels: list[str], res: dict) -> str:
    def head(r: str) -> str:
        p = f"x.isocal.{r}"
        return (f"{res[f'{p}.os']}, kernel {res[f'{p}.kernel']}, {res[f'{p}.cpus']} CPUs; "
                f"labelled {res[f'{p}.condition']}, load {res[f'{p}.load_start']} → {res[f'{p}.load_end']}")

    rows = ["| Channel (primary statistic) | " + " | ".join(head(r) for r in labels) + " |",
            "|---|" + "---|" * len(labels)]
    for c in CHANNELS:
        cells = []
        for r in labels:
            q = f"x.isocal.{r}.{c}"
            if res[f"{q}.status"] != "ok":
                cells.append(f"not available (failed at {res[f'{q}.stage']})")
                continue
            cells.append(f"{res[f'{q}.correct']}/{res[f'{q}.n_test']} ({res[f'{q}.ci_lo']:.2f}–{res[f'{q}.ci_hi']:.2f}); "
                         f"{res[f'{q}.capacity']:.2f} bit/round (≥ {res[f'{q}.capacity_lower']:.2f}); "
                         f"**{res[f'{q}.channel']}**")
        rows.append(f"| {NAMES[c]} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=PATH to an isocal-*.json; repeat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res: dict = {"x.isocal.label": "exploratory (A4), LLM-free scripted calibration"}
    labels = []
    for spec in a.run:
        label, _, path = spec.partition("=")
        if not LABEL.match(label) or not path or label in labels:
            raise SystemExit(f"bad or repeated --run {spec!r}: want LABEL=PATH, LABEL of letters, digits, _")
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if not str(doc.get("meta", {}).get("tool", "")).endswith("iso_calibration.py"):
            raise SystemExit(f"{path} is not an iso_calibration.py result")
        res.update(run_keys(label, doc))
        labels.append(label)
    res["x.isocal.n_runs"] = len(labels)
    for c in CHANNELS:
        res[f"x.isocal.channel_runs.{c}"] = sum(res[f"x.isocal.{r}.{c}.channel"] == "yes" for r in labels)
        res[f"x.isocal.unavailable_runs.{c}"] = sum(res[f"x.isocal.{r}.{c}.channel"] == "not available" for r in labels)
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "isocal_results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    (out / "tables" / "isocal_matrix.md").write_text(table(labels, res), encoding="utf-8")
    print(f"{len(labels)} runs -> {out}/isocal_results.json, {out}/tables/isocal_matrix.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
