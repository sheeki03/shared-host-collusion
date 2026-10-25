"""Render report/REPORT.md -> report/REPORT.filled.md from runs/analysis outputs.

Placeholders:
  {{KEY}}          a dotted key from runs/analysis/results.json (analysis/RESULT_KEYS.md), formatted for prose.
                   KEY.ci renders "[ci_lo, ci_hi]" from KEY.ci_lo / KEY.ci_hi; KEY.est_ci renders
                   "est [lo, hi]" using KEY.diff (or KEY.est / KEY.rate / KEY itself).
  {{TABLE:name}}   the markdown table runs/analysis/tables/<name>.md
  {{FIG:name}}     an image link to runs/analysis/figures/<name>.png (relative to the output file)
  {{EXCERPT:n}}    the transcript excerpt n from <results dir>/excerpts.json (tools/pick_excerpts.py); pilot excerpts
                   are rendered but count as unfinished
Formatting: integers with thousands separators; p-values (keys ending .p, .p_holm, ...) as 3 significant digits or
"< 0.001"; other floats 3 significant digits (2 decimals at or above 1, no decimals at or above 1,000); booleans
yes/no; strings as is (e.g. "pending").
A JSON null (a non-finite estimate) renders as "n/a" and counts as unfinished.
Unresolved placeholders are listed on stderr. With --confirmatory the tool exits 1 if any remain, if a value is
'pending' / 'insufficient data' / null, if results.json is marked preliminary or is not the confirmatory cohort, or
if an excerpt comes from pilot data.
Usage: .venv/bin/python tools/fill_report.py [--report report/REPORT.md] [--out report/REPORT.filled.md]
       [--results runs/analysis/results.json] [--confirmatory]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{\s*([^{}\s]+)\s*\}\}")
P_SUFFIX = re.compile(r"\.(p|p_holm|p_seedblock|p_family_raw|p_signflip)$")
UNFINISHED = ("pending", "insufficient data")


def fmt_number(v, *, p_value: bool = False) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "n/a"
        if p_value:
            return "< 0.001" if v < 0.001 else f"{v:.3g}"
        a = abs(v)
        if a < 1e-12:
            return "0"
        if a >= 1000:
            return f"{v:,.0f}"
        if a >= 1:
            return f"{v:.2f}"
        return f"{v:.3g}"
    return str(v)


def _base_value(results: dict, base: str):
    for k in (f"{base}.diff", f"{base}.est", f"{base}.rate", base):
        if k in results:
            return results[k]
    return None


def _finished(v) -> bool:
    return v is not None and not (isinstance(v, float) and not math.isfinite(v)) and \
        not (isinstance(v, str) and v in UNFINISHED)


def resolve(key: str, results: dict) -> tuple[str | None, bool]:
    """Return (text, finished). text None means unresolved. An exact key always wins over the .ci/.est_ci forms."""
    if key in results:
        v = results[key]
        if isinstance(v, list):
            return "; ".join(str(x) for x in v), True
        if isinstance(v, str):
            return v, _finished(v)
        return fmt_number(v, p_value=bool(P_SUFFIX.search(key))), _finished(v)
    if key.endswith(".est_ci") or key.endswith(".ci"):
        base = key.rsplit(".", 1)[0]
        if f"{base}.ci_lo" not in results and f"{base}.ci_hi" not in results:
            return None, False
        lo, hi = results.get(f"{base}.ci_lo"), results.get(f"{base}.ci_hi")
        if isinstance(lo, str) or isinstance(hi, str):
            s = lo if isinstance(lo, str) else hi
            return s, _finished(s)
        ci = f"[{fmt_number(lo)}, {fmt_number(hi)}]"
        ok = _finished(lo) and _finished(hi)
        if key.endswith(".ci"):
            return ci, ok
        est = _base_value(results, base)
        if isinstance(est, str):
            return est, _finished(est)
        return f"{fmt_number(est)} {ci}", ok and _finished(est)
    return None, False


def render(text: str, results: dict, tables_dir: Path, figures_dir: Path, out_path: Path,
           excerpts: dict | None = None) -> tuple[str, list[str], list[str]]:
    unresolved: list[str] = []
    unfinished: list[str] = []

    def sub(m: re.Match) -> str:
        token = m.group(1)
        if token.startswith("TABLE:"):
            name = token.split(":", 1)[1]
            f = tables_dir / f"{name}.md"
            if not f.exists():
                unresolved.append(token)
                return m.group(0)
            body = f.read_text(encoding="utf-8").strip()
            if body.startswith("_No data yet._"):
                unfinished.append(token)
            return body
        if token.startswith("EXCERPT:"):
            entry = (excerpts or {}).get(token)
            if not isinstance(entry, dict) or "markdown" not in entry:
                unresolved.append(token)
                return m.group(0)
            if entry.get("pilot"):
                unfinished.append(token)
            return str(entry["markdown"]).rstrip("\n")
        if token.startswith("FIG:"):
            name = token.split(":", 1)[1]
            f = figures_dir / f"{name}.png"
            if not f.exists():
                unresolved.append(token)
                return m.group(0)
            rel = os.path.relpath(f.resolve(), out_path.resolve().parent)
            return f"![{name}]({rel})"
        val, finished = resolve(token, results)
        if val is None:
            unresolved.append(token)
            return m.group(0)
        if not finished:
            unfinished.append(token)
        return val

    return PLACEHOLDER.sub(sub, text), unresolved, unfinished


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(ROOT / "report" / "REPORT.md"))
    ap.add_argument("--out", default=None, help="default: <report stem>.filled.md next to the report")
    ap.add_argument("--results", default=str(ROOT / "runs" / "analysis" / "results.json"))
    ap.add_argument("--tables", default=None, help="default: <results dir>/tables")
    ap.add_argument("--figures", default=None, help="default: <results dir>/figures")
    ap.add_argument("--excerpts", default=None, help="default: <results dir>/excerpts.json (tools/pick_excerpts.py)")
    ap.add_argument("--confirmatory", action="store_true", help="fail on any unresolved/pending placeholder")
    a = ap.parse_args(argv)
    report = Path(a.report)
    out = Path(a.out) if a.out else report.with_name(report.stem + ".filled.md")
    results_path = Path(a.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    tables = Path(a.tables) if a.tables else results_path.parent / "tables"
    figures = Path(a.figures) if a.figures else results_path.parent / "figures"
    ex_path = Path(a.excerpts) if a.excerpts else results_path.parent / "excerpts.json"
    excerpts = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.exists() else {}
    text, unresolved, unfinished = render(report.read_text(encoding="utf-8"), results, tables, figures, out, excerpts)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}; {len(unresolved)} unresolved, {len(unfinished)} pending/insufficient placeholders")
    for u in sorted(set(unresolved)):
        print(f"  unresolved: {{{{{u}}}}}", file=sys.stderr)
    for u in sorted(set(unfinished)):
        print(f"  pending: {{{{{u}}}}}", file=sys.stderr)
    if a.confirmatory:
        if results.get("meta.preliminary") is not False:
            print("  results.json is preliminary (meta.preliminary is not false)", file=sys.stderr)
            return 1
        if results.get("meta.cohort", "confirmatory") != "confirmatory":
            print(f"  results.json cohort is {results.get('meta.cohort')!r}, not confirmatory", file=sys.stderr)
            return 1
        if unresolved or unfinished:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
