"""Merge the analysis outputs that report/REPORT.md cites into one results file and one tables folder for
tools/fill_report.py, which reads a single results.json and the tables/ (and figures/, excerpts.json) next to it.

Usage (repo root, venv python):
  .venv/bin/python tools/merge_results.py --base runs/analysis/results.json \
      --extra runs/analysis/ext/ext_results.json --extra runs/analysis/signal/signal_results.json \
      --extra runs/analysis/x/leak/x_leak.json ... --out runs/analysis/final
The base is analysis/report_outputs.py's results.json: its meta.* keys (cohort, preliminary flag, counts) are the only
meta keys (and its "notes" the only "notes") in the merged file. Each --extra is a flat {key: value} file from analysis/extensions.py, signalling.py,
leakage_probe.py, pathway.py, price_digits.py or tools/isocal_matrix.py; meta.* keys in an extra are dropped and
its "notes" are kept as notes.<file stem>. A key
present in two inputs must have the same value in both, or the merge fails (nothing is overwritten silently).
Tables: every *.md under <input dir>/tables/ is copied to <out>/tables/; the same table name from two inputs must
have identical content. The base's figures/ and excerpts.json are copied next to the merged file. <out>/merge.json
records each input's path, sha256 and key count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


class MergeError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def merge(base: Path, extras: list[Path], out: Path) -> dict:
    res = json.loads(base.read_text(encoding="utf-8"))
    if not isinstance(res, dict):
        raise MergeError(f"{base}: not a flat results object")
    origin = {k: str(base) for k in res}
    inputs = [{"path": str(base), "sha256": _sha(base), "keys": len(res), "role": "base"}]
    for x in extras:
        d = json.loads(x.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise MergeError(f"{x}: not a flat results object")
        kept = {k: v for k, v in d.items() if not k.startswith("meta.") and k != "notes"}
        if "notes" in d:  # each module's run notes, kept apart: notes.<file stem>
            kept[f"notes.{x.stem}"] = d["notes"]
        for k, v in kept.items():
            if k in res and res[k] != v:
                raise MergeError(f"key {k!r} differs between {origin[k]} and {x}")
            res[k], origin[k] = v, origin.get(k, str(x))
        inputs.append({"path": str(x), "sha256": _sha(x), "keys": len(kept), "role": "extra",
                       "dropped_meta": len(d) - len(kept)})
    (out / "tables").mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    for src in [base, *extras]:
        tdir = src.parent / "tables"
        for t in sorted(tdir.glob("*.md")) if tdir.is_dir() else []:
            if t.name in seen and seen[t.name].read_bytes() != t.read_bytes():
                raise MergeError(f"table {t.name} differs between {seen[t.name]} and {t}")
            seen[t.name] = t
            shutil.copyfile(t, out / "tables" / t.name)
    if (base.parent / "figures").is_dir():
        shutil.copytree(base.parent / "figures", out / "figures", dirs_exist_ok=True)
    if (base.parent / "excerpts.json").exists():
        shutil.copyfile(base.parent / "excerpts.json", out / "excerpts.json")
    (out / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True), encoding="utf-8")
    manifest = {"inputs": inputs, "tables": {n: str(p) for n, p in sorted(seen.items())}, "n_keys": len(res)}
    (out / "merge.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="analysis/report_outputs.py results.json")
    ap.add_argument("--extra", action="append", default=[], help="another flat results file; repeat")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.resolve() == Path(a.base).resolve().parent:
        raise SystemExit("--out must differ from the base's directory")
    m = merge(Path(a.base), [Path(x) for x in a.extra], out)
    print(f"{m['n_keys']} keys, {len(m['tables'])} tables -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
