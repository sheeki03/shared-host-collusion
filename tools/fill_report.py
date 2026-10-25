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
if an excerpt comes from pilot data. It also refuses editorial brackets, HTML comments, a missing/blank LLM Usage
Statement, and an abstract outside the required 150–250 words. A failed confirmatory render never leaves a filled
report that could be mistaken for a finished submission.
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
UNFINISHED = ("pending", "insufficient data", "n/a", "nan")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SECTION = re.compile(r"(?m)^##\s+(.+?)\s*$")
EDITORIAL_BRACKET = re.compile(r"(?<!!)\[([^\[\]\n]{1,240})\](?!\()")
EDITORIAL_WORD = re.compile(r"\b(?:TODO|TBD|AUTHOR TO WRITE|TO BE ADDED)\b", re.I)
WORD = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.U)


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
        not (isinstance(v, str) and v.strip().casefold() in UNFINISHED)


def resolve(key: str, results: dict) -> tuple[str | None, bool]:
    """Return (text, finished). text None means unresolved. An exact key always wins over the .ci/.est_ci forms."""
    if key in results:
        v = results[key]
        if isinstance(v, list):
            return "; ".join(str(x) for x in v), all(_finished(x) for x in v)
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
            if "_No data yet._" in body or re.search(r"\b(?:pending|insufficient data)\b", body, re.I):
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


def editorial_issues(text: str) -> list[str]:
    """Find human decisions that a placeholder-only check cannot catch in a submission draft."""
    issues: list[str] = []
    comments = HTML_COMMENT.findall(text)
    if comments or "<!--" in text or "-->" in text:
        issues.append(f"{len(comments) or 1} HTML editorial comment(s) remain")
    clean = HTML_COMMENT.sub("", text)

    sections = list(SECTION.finditer(clean))
    by_name = {m.group(1).strip().lower(): (m.end(), sections[i + 1].start() if i + 1 < len(sections) else len(clean))
               for i, m in enumerate(sections)}
    abstract = by_name.get("abstract")
    if abstract is None:
        issues.append("required Abstract section is missing")
    else:
        body = clean[abstract[0]:abstract[1]]
        # Links contribute their visible text; Markdown punctuation is not a word.
        n_words = len(WORD.findall(body))
        if not 150 <= n_words <= 250:
            issues.append(f"Abstract has {n_words} words; required range is 150–250")

    usage = by_name.get("llm usage statement")
    if usage is None:
        issues.append("required LLM Usage Statement section is missing")
    elif not WORD.search(clean[usage[0]:usage[1]]):
        issues.append("LLM Usage Statement is blank")

    # Prompt excerpts and transcript quotations can contain literal [AWARE] or [stderr].
    prose = re.sub(r"(?ms)^```[^\n]*\n.*?^```\s*$", "", clean)
    prose = re.sub(r"`[^`\n]*`", "", prose)
    prose = "\n".join(line for line in prose.splitlines() if not line.lstrip().startswith(">"))
    for m in EDITORIAL_BRACKET.finditer(prose):
        if re.search(r"[A-Za-z]", m.group(1)):
            issues.append(f"editorial bracket remains: [{m.group(1)[:80]}]")
    if EDITORIAL_WORD.search(prose):
        issues.append("TODO/TBD editorial text remains")
    return issues


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
    if report.resolve() == out.resolve():
        raise SystemExit("--out must not overwrite the report source")
    results_path = Path(a.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    tables = Path(a.tables) if a.tables else results_path.parent / "tables"
    figures = Path(a.figures) if a.figures else results_path.parent / "figures"
    ex_path = Path(a.excerpts) if a.excerpts else results_path.parent / "excerpts.json"
    excerpts = json.loads(ex_path.read_text(encoding="utf-8")) if ex_path.exists() else {}
    source = report.read_text(encoding="utf-8")
    # Editorial comments are not report content or a placeholder contract. Still block a final render until the
    # author removes them; resolving placeholders inside them would manufacture unrelated missing-key failures.
    source_comments = HTML_COMMENT.findall(source) if a.confirmatory else []
    content = HTML_COMMENT.sub("", source) if a.confirmatory else source
    text, unresolved, unfinished = render(content, results, tables, figures, out, excerpts)
    issues = editorial_issues(text) if a.confirmatory else []
    if source_comments:
        issues.insert(0, f"{len(source_comments)} HTML editorial comment(s) remain in source")
    for u in sorted(set(unresolved)):
        print(f"  unresolved: {{{{{u}}}}}", file=sys.stderr)
    for u in sorted(set(unfinished)):
        print(f"  pending: {{{{{u}}}}}", file=sys.stderr)
    for issue in issues:
        print(f"  editorial: {issue}", file=sys.stderr)
    if a.confirmatory:
        if results.get("meta.preliminary") is not False:
            print("  results.json is preliminary (meta.preliminary is not false)", file=sys.stderr)
            issues.append("preliminary results")
        if results.get("meta.cohort") != "confirmatory":
            print(f"  results.json cohort is {results.get('meta.cohort')!r}, not confirmatory", file=sys.stderr)
            issues.append("non-confirmatory cohort")
        if unresolved or unfinished or issues:
            # A stale successful-looking file must not survive a failed final render.
            out.unlink(missing_ok=True)
            print(f"blocked {out}; {len(unresolved)} unresolved, {len(unfinished)} pending placeholders, "
                  f"{len(issues)} editorial/cohort issues", file=sys.stderr)
            return 1
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}; {len(unresolved)} unresolved, {len(unfinished)} pending/insufficient placeholders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
