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
Unresolved placeholders are listed on stderr with their source line numbers. With --confirmatory the tool exits 1 if
any remain, if a value is 'pending' / 'insufficient data' / null, if results.json is marked preliminary or is not the
confirmatory cohort, or if an excerpt comes from pilot data. It also refuses the editorial items below, a
missing/blank LLM Usage Statement, and an abstract outside the required 150–250 words. A failed confirmatory render
never leaves a filled report that could be mistaken for a finished submission.

Confirmatory editorial gate, revision 2. Declared in this docstring on 26 Oct 2026 at 04:06 IST, before the final
analysis and the confirmatory render run (it answers point 4 of an external review). It reads only the report source
text and the form of the rendered values (status words, comment and placeholder syntax), never an estimate's sign,
size or significance, so it is outcome-blind. Every item is listed on stderr as "line N: ...", where N is the
line in the report source (for an item spanning lines, the line where it starts), sorted by line:
  - every HTML comment, each listed with its first words and tagged "author TODO" when it holds a TODO marker or starts
    with "author"; also an unclosed "<!--" and a stray "-->" (a broken comment edit);
  - unresolved square-bracket text: an editorial alternative such as "[did / did not]" or an instruction such as
    "[One sentence ...]", including brackets whose text runs over line breaks within a paragraph or list item,
    nested brackets (the outermost editorial bracket is listed once), and brackets holding placeholders, whose
    rendered values can themselves contain brackets (a "{{KEY.ci}}" inside an editorial bracket). Placeholders are
    judged by position, not by their rendered values; an unclosed "[" and an unmatched "]" are listed as well, and so
    are blank fill-ins ("[ ]", "[___]") and unchecked task boxes;
  - author-TODO markers in prose: TODO, TBD, TBC, FIXME, "AUTHOR TO WRITE", "TO BE ADDED / WRITTEN / DECIDED /
    CONFIRMED / DETERMINED / FILLED (IN)", XX or XXX (upper case) and "??";
  - literal unfinished values: "pending" or "insufficient data" in a value position, that is, a table data cell, after
    ":", "=", "(", "[", or a verb such as "is", "remains" or "still", or at the start of a line or list item, and not
    followed by a word ("Kappa: pending." and "| pending |" are refused; "pending the data-release decision" and
    "pending games per cell" are prose and pass). A value from results.json that starts with "pending" or
    "insufficient data" ("pending (judges not run)", "insufficient data: fewer than 10 games") is unfinished like the
    bare word; "n/a ..." (not applicable) is a finished value. A generated table is unfinished if it contains
    "_No data yet._" or "pending" / "insufficient data" outside its header row (a column named "pending", as in the
    coverage table, is a header, not a value);
  - a malformed or nested placeholder ("{{KEY", "KEY}}", "{{a{{b}}}}"), which would otherwise stay in the output as
    literal braces, and a result or table value that inserts "<!--", "{{" or a TODO marker.
Brackets that pass: markdown links, images, reference links and definitions, footnotes, pandoc citations
("[@key]"), numeric citations and intervals ("[1]", "[1, 3]", "[1–3]", "[0.1, 0.3]", "[1.2e-05, 0.3]", "[n/a, n/a]"),
half-open intervals ("[0, 1)", "(0, 1]"), intervals of short symbols ("[a, b]", "[ci_lo, ci_hi]"), indices written
straight after a word ("x[i]", "U[60, 100]"), label references ("[S1]", "[Table C11]", "[A1–A4]"), author-year
citations ("[Zolkowski et al. 2025; Fish et al. 2024]"), statistics ("[n = 12]", "[p < 0.001]"), "[sic]", "[…]",
checked task boxes ("- [x]"), and anything in fenced code, inline code or a blockquote (prompt excerpts and transcript
quotations contain literal "[AWARE]" or "[stderr]"). Table cells are scanned one cell at a time, so a bracket cannot
pair across "|". Every other square bracket that holds a letter is treated as editorial, so the gate errs towards
refusing.
Usage: .venv/bin/python tools/fill_report.py [--report report/REPORT.md] [--out report/REPORT.filled.md]
       [--results runs/analysis/results.json] [--confirmatory]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{\s*([^{}\s]+)\s*\}\}")
P_SUFFIX = re.compile(r"\.(p|p_holm|p_seedblock|p_family_raw|p_signflip)$")
UNFINISHED = ("pending", "insufficient data", "n/a", "nan")
# A status string that starts with one of these is unfinished whatever qualifier follows it.
UNFINISHED_PREFIX = re.compile(r"(?:pending|insufficient data)(?![\w-])", re.I)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
SECTION = re.compile(r"(?m)^##\s+(.+?)\s*$")
EDITORIAL_WORD = re.compile(
    r"\b(?:TODO|TBD|TBC|FIXME|AUTHOR TO WRITE|TO BE (?:ADDED|WRITTEN|DECIDED|CONFIRMED|DETERMINED|FILLED(?: IN)?))\b"
    r"|(?-i:\bX{2,3}\b)|\?{2,}", re.I)
WORD = re.compile(r"\b[\w]+(?:['’\-][\w]+)*\b", re.U)

# --- editorial gate patterns (see the module docstring) ---------------------------------------------------------------
FENCE = re.compile(r"(?ms)^[ \t]{0,3}(?P<f>(?P<c>[`~])(?P=c){2,})[^\n]*\n.*?^[ \t]{0,3}(?P=f)(?P=c)*[ \t]*$")
INLINE_CODE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)")
BLOCKQUOTE = re.compile(r"(?m)^[ \t]*>.*$")
LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
RULE_LINE = re.compile(r"^[ \t]*(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
TABLE_SEPARATOR = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$")
LETTER = re.compile(r"[^\W\d_]")
NUMERIC_TOKEN = re.compile(r"(?i)\d+(?:[.,]\d+)*(?:e[-+−]?\d+)?|\b(?:n/a|nan|inf)\b")
# Numbers in a symbolic interval take no thousands separators, so a comma is always a separator (linear matching).
_TERM = (r"[-+−–]?(?:\d[\d.]*(?:e[-+−]?\d+)?%?|∞|inf|nan|n/a|"
         r"[^\W\d_]{1,2}(?:_\w{1,10})?\d*(?:\s*[-+−*/]\s*\d+(?:\.\d+)?)?)")
_LABEL = r"[A-Z]{0,3}\d+(?:\.\d+)*[a-z]?(?:\([a-z]\))?"
_REF_WORD = r"(?:Tables?|Figures?|Figs?\.|Sections?|Secs?\.|§|Appendix|Appendices|Eqs?\.|Equations?|Refs?\.)"
_NAME = r"[A-Z][\w'’\-]*(?:\s+(?:[A-Z][\w'’\-]*|of|the|for|and|&|van|von|de|der|da|di|le|la))*"
_CITE = (rf"(?:(?:see|e\.g\.,?|cf\.|also)\s+)*{_NAME}(?:\s+et\s+al\.)?,?\s+(?:19|20)\d{{2}}[a-z]?"
         rf"(?:\s*,\s*(?:19|20)\d{{2}}[a-z]?)*(?:,\s*(?:pp?|ch|sec|§|Table|Fig)\.?\s*[\w.\-–]+)?")
_ASSIGN = (r"[^\W\d_]{1,3}(?:_\w{1,10})?\s*(?:<=|>=|=|<|>|≤|≥|≈|~|≠)\s*"
           r"(?:[-+−]?\d[\d,]*(?:\.\d+)?(?:e[-+−]?\d+)?%?|n/a|[^\W\d_]{1,2}(?:_\w{1,10})?)")
LEGIT_BRACKET = re.compile("|".join([
    rf"(?:{_REF_WORD}\s+)?{_LABEL}(?:\s*(?:[-–—,;]|and|&)\s*{_LABEL})*",       # [S1], [Table C11], [A1–A4]
    r"(?i:sic!?|…|\.{3}|emphasis (?:added|ours|in (?:the )?original)|original emphasis)",
    rf"{_TERM}(?:\s*[,;]\s*{_TERM})+",                                           # [a, b], [ci_lo, ci_hi]
    rf"{_CITE}(?:\s*;\s*{_CITE})*",                                              # [Zolkowski et al. 2025]
    rf"{_ASSIGN}(?:\s*[,;]?\s*{_ASSIGN})*",                                      # [n = 12], [p < 0.001]
]))
SINGLE_TERM = re.compile(_TERM)
INTERVAL = re.compile(rf"\s*{_TERM}\s*,\s*{_TERM}\s*")
PENDING_WORD = re.compile(r"(?i)\b(?:pending|insufficient data)\b")
VALUE_SLOT_END = re.compile(
    r"(?:[:=≈(\[|<>≤≥~]|\b(?:is|are|was|were|be|been|being|remains?|remained|stays?|stayed|still|currently|left|"
    r"marked(?: as)?|shown as|reported as|listed as))$", re.I)


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
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return False
    if isinstance(v, str):
        s = v.strip()
        return s.casefold() not in UNFINISHED and not UNFINISHED_PREFIX.match(s)
    return True


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


TABLE_PENDING = re.compile(r"(?i)\bpending\b")


def table_unfinished(body: str) -> bool:
    """A generated table holds an unfinished value: the no-data marker, or 'pending' outside the header row (a column
    *named* pending, as in the coverage table, is not a value). 'insufficient data' in a generated table cell is a
    final, honest status (e.g. an AUROC with too few games of one class) and does not block the render; in prose it
    still does, so the author states it in words (revision 3, 26 Oct 2026 04:36 IST)."""
    if "_No data yet._" in body:
        return True
    lines = body.splitlines()
    header = {i - 1 for i, line in enumerate(lines) if i and TABLE_SEPARATOR.match(line) and "|" in lines[i - 1]}
    return any(TABLE_PENDING.search(line) for i, line in enumerate(lines) if i not in header)


# Called once per placeholder: (token, offset of the placeholder in the rendered source, inserted text or None, kind)
# with kind "key", "table", "figure" or "excerpt".
OnSub = Callable[[str, int, "str | None", str], None]


def render(text: str, results: dict, tables_dir: Path, figures_dir: Path, out_path: Path,
           excerpts: dict | None = None, *, on_sub: OnSub | None = None) -> tuple[str, list[str], list[str]]:
    unresolved: list[str] = []
    unfinished: list[str] = []

    def note(m: re.Match, value: str | None, kind: str) -> None:
        if on_sub is not None:
            on_sub(m.group(1), m.start(), value, kind)

    def sub(m: re.Match) -> str:
        token = m.group(1)
        if token.startswith("TABLE:"):
            name = token.split(":", 1)[1]
            f = tables_dir / f"{name}.md"
            if not f.exists():
                unresolved.append(token)
                note(m, None, "table")
                return m.group(0)
            body = f.read_text(encoding="utf-8").strip()
            if table_unfinished(body):
                unfinished.append(token)
            note(m, body, "table")
            return body
        if token.startswith("EXCERPT:"):
            entry = (excerpts or {}).get(token)
            if not isinstance(entry, dict) or "markdown" not in entry:
                unresolved.append(token)
                note(m, None, "excerpt")
                return m.group(0)
            if entry.get("pilot"):
                unfinished.append(token)
            body = str(entry["markdown"]).rstrip("\n")
            note(m, body, "excerpt")
            return body
        if token.startswith("FIG:"):
            name = token.split(":", 1)[1]
            f = figures_dir / f"{name}.png"
            if not f.exists():
                unresolved.append(token)
                note(m, None, "figure")
                return m.group(0)
            rel = os.path.relpath(f.resolve(), out_path.resolve().parent)
            note(m, f"![{name}]({rel})", "figure")
            return f"![{name}]({rel})"
        val, finished = resolve(token, results)
        if val is None:
            unresolved.append(token)
            note(m, None, "key")
            return m.group(0)
        if not finished:
            unfinished.append(token)
        note(m, val, "key")
        return val

    return PLACEHOLDER.sub(sub, text), unresolved, unfinished


# ----------------------------------------------------------------------------------------------------------------------
# Confirmatory editorial gate


class Issue(NamedTuple):
    """One refused item. A NamedTuple, not a dataclass: analysis/tests load this file with importlib without
    registering it in sys.modules, which dataclasses need under postponed annotations."""
    line: int            # 1-based source line; 0 = the whole document
    text: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.text}" if self.line else self.text


class LineIndex:
    def __init__(self, text: str):
        self.starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]

    def __call__(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)


def _preview(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _blank(chars: list[str], start: int, end: int, fill: str = " ") -> None:
    """Overwrite chars[start:end] but keep newlines, so offsets and line numbers stay those of the source."""
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = fill


def comment_spans(source: str, line: LineIndex) -> tuple[list[tuple[int, int]], list[Issue]]:
    """Every HTML comment as (start, end), and one issue per comment, unclosed opener or stray closer."""
    spans: list[tuple[int, int]] = []
    issues: list[Issue] = []
    pos = 0
    while True:
        i, j = source.find("<!--", pos), source.find("-->", pos)
        if i < 0 and j < 0:
            break
        if j >= 0 and (i < 0 or j < i):
            issues.append(Issue(line(j), "stray '-->' outside an HTML comment (broken comment edit)"))
            pos = j + 3
            continue
        k = source.find("-->", i + 4)
        if k < 0:
            issues.append(Issue(line(i), f"unclosed HTML comment: <!-- {_preview(source[i + 4:], 60)}"))
            break
        body = source[i + 4:k]
        todo = EDITORIAL_WORD.search(body) or re.match(r"\s*author\b", body, re.I)
        tag = "author TODO" if todo else ("conditional wording" if re.match(r"\s*if\b", body, re.I) else "editorial")
        issues.append(Issue(line(i), f"HTML comment ({tag}) remains: {_preview(body)}"))
        spans.append((i, k + 3))
        pos = k + 3
    return spans, issues


def prose_view(content: str) -> str:
    """The text the gate reads: same length and line breaks as `content`, with placeholders replaced by a neutral
    value token ("0") and fenced code, inline code and blockquote lines blanked."""
    chars = list(content)
    for m in PLACEHOLDER.finditer(content):
        _blank(chars, m.start(), m.end())
        chars[m.start()] = "0"
    masked = "".join(chars)
    for pattern in (FENCE, INLINE_CODE, BLOCKQUOTE):
        chars = list(masked)
        for m in pattern.finditer(masked):
            _blank(chars, m.start(), m.end())
        masked = "".join(chars)
    return masked


def _segments(prose: str) -> list[tuple[int, int]]:
    """Spans inside which a bracket may open and close: a paragraph or list item (across its line breaks), a heading,
    or one table cell. Blank lines, headings, rules, table rows and new list items end a paragraph."""
    segs: list[tuple[int, int]] = []
    start = None
    pos = 0
    for raw in prose.splitlines(keepends=True):
        end = pos + len(raw)
        body = raw.strip()
        boundary = not body or body.startswith("#") or body.startswith("|") or RULE_LINE.match(raw) \
            or LIST_ITEM.match(raw)
        if boundary and start is not None:
            segs.append((start, pos))
            start = None
        if body.startswith("|"):
            cell = pos
            for k, ch in enumerate(raw):
                if ch == "|" and (k == 0 or raw[k - 1] != "\\"):
                    segs.append((cell, pos + k))
                    cell = pos + k + 1
            segs.append((cell, end))
        elif body.startswith("#"):
            segs.append((pos, end))
        elif body and not RULE_LINE.match(raw):
            if start is None:
                start = pos
        pos = end
    if start is not None:
        segs.append((start, pos))
    return segs


def _bracket_spans(prose: str) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    """Matched [ ... ] pairs (nesting allowed, half-open intervals skipped) and unmatched brackets, per segment."""
    spans: list[tuple[int, int]] = []
    stray: list[tuple[int, str]] = []
    for a, b in _segments(prose):
        stack: list[tuple[str, int]] = []
        for i in range(a, b):
            ch = prose[i]
            if ch in "[(":
                stack.append((ch, i))
            elif ch == ")":
                if stack and (stack[-1][0] == "(" or INTERVAL.fullmatch(prose[stack[-1][1] + 1:i])):
                    stack.pop()          # a parenthesis, or a half-open interval [a, b)
                # else: a stray ")" ("a)" enumerations); ignore it
            elif ch == "]":
                if stack and stack[-1][0] == "(" and INTERVAL.fullmatch(prose[stack[-1][1] + 1:i]):
                    stack.pop()          # half-open interval (a, b]
                    continue
                while stack and stack[-1][0] == "(":
                    stack.pop()          # an unclosed "(" inside the bracket
                if not stack:
                    stray.append((i, "]"))
                    continue
                spans.append((stack.pop()[1], i))
        stray.extend((p, "[") for c, p in stack if c == "[" and LETTER.search(prose[p + 1:b]))
    return spans, stray


def _bracket_ok(prose: str, p: int, q: int) -> bool:
    """Whether the bracket prose[p] ... prose[q] is legitimate markdown, a citation, a number or an interval."""
    nxt = prose[q + 1:q + 2]
    prev = prose[p - 1:p]
    line_start = prose.rfind("\n", 0, p) + 1
    before = prose[line_start:p]
    if nxt in ("(", "[") or prev == "]":
        return True                                    # inline link / image / reference link [text][ref]
    if nxt == ":" and not before.strip():
        return True                                    # reference definition "[ref]: url"
    c = " ".join(prose[p + 1:q].split())
    if c in ("x", "X") and LIST_ITEM.fullmatch(before):
        return True                                    # checked task box
    if not c or re.fullmatch(r"_+", c):
        return False                                   # blank fill-in, unchecked task box
    if c.startswith("^") or re.search(r"(?<![\w.])-?@[\w:.\-]+", c):
        return True                                    # footnote, pandoc citation
    if not LETTER.search(NUMERIC_TOKEN.sub("", c)):
        return True                                    # [1], [1, 3], [0.1, 0.3], [1.2e-05, 0.3]
    if len(c) > 200:
        return False                                   # no citation, label or interval is this long
    if prev and (prev.isalnum() or prev in ")]") and SINGLE_TERM.fullmatch(c):
        return True                                    # index notation x[i]
    return bool(LEGIT_BRACKET.fullmatch(c))


def _line_end(text: str, pos: int) -> int:
    e = text.find("\n", pos)
    return len(text) if e < 0 else e


def bracket_issues(prose: str, line: LineIndex, content: str | None = None) -> list[Issue]:
    """Editorial brackets in `prose` (prose_view); previews quote `content`, the source text at the same offsets."""
    text = prose if content is None else content
    spans, stray = _bracket_spans(prose)
    issues: list[Issue] = []
    reported: list[tuple[int, int]] = []
    for p, q in sorted(spans, key=lambda s: (s[0], -s[1])):
        if any(a <= p and q <= b for a, b in reported) or _bracket_ok(prose, p, q):
            continue
        reported.append((p, q))
        c = prose[p + 1:q]
        first, last = line(p), line(q)
        where = f" (runs to line {last})" if last != first else ""
        kind = "blank fill-in or unchecked box" if not c.strip() or re.fullmatch(r"\s*_+\s*", c) else \
            "unresolved editorial bracket"
        issues.append(Issue(first, f"{kind} [{_preview(text[p + 1:q])}]{where}"))
    for pos, ch in stray:
        if ch == "[":
            issues.append(Issue(line(pos), f"unclosed '[' (unfinished editorial bracket?): "
                                           f"[{_preview(text[pos + 1:_line_end(text, pos)], 60)}"))
        else:
            start = max(prose.rfind("\n", 0, pos) + 1, pos - 60)
            issues.append(Issue(line(pos), f"unmatched ']' (half-edited bracket?): {_preview(text[start:pos + 1])}"))
    return issues


def marker_issues(prose: str, line: LineIndex) -> list[Issue]:
    return [Issue(line(m.start()), f"author-TODO marker '{m.group(0)}' remains")
            for m in EDITORIAL_WORD.finditer(prose)]


def pending_value_issues(prose: str, line: LineIndex, content: str | None = None) -> list[Issue]:
    """Literal 'pending' / 'insufficient data' written as a value (see the module docstring), not as prose."""
    text = prose if content is None else content
    issues: list[Issue] = []
    lines = prose.split("\n")
    header_rows = {i - 1 for i, s in enumerate(lines) if i and TABLE_SEPARATOR.match(s) and "|" in lines[i - 1]}
    for m in PENDING_WORD.finditer(prose):
        ln = line(m.start())
        if ln - 1 in header_rows:
            continue
        ls = prose.rfind("\n", 0, m.start()) + 1
        le = prose.find("\n", m.end())
        after = prose[m.end():le if le >= 0 else len(prose)]
        if re.match(r"[\s*_]*[^\W_]", after):
            continue                                   # "pending the decision", "pending games"
        before = prose[ls:m.start()].rstrip(" \t*_")
        if before.strip() and not LIST_ITEM.fullmatch(before + " ") and not VALUE_SLOT_END.search(before):
            continue                                   # "with the decision pending, ..."
        issues.append(Issue(ln, f"unfinished value '{m.group(0)}' remains: {_preview(text[ls:_line_end(text, ls)])}"))
    return issues


def section_issues(rendered: str, content: str, line: LineIndex) -> list[Issue]:
    issues: list[Issue] = []
    head_line = {m.group(1).strip().lower(): line(m.start()) for m in SECTION.finditer(content)}
    sections = list(SECTION.finditer(rendered))
    by_name = {m.group(1).strip().lower(): (m.end(), sections[i + 1].start() if i + 1 < len(sections)
                                            else len(rendered)) for i, m in enumerate(sections)}
    abstract = by_name.get("abstract")
    if abstract is None:
        issues.append(Issue(0, "required Abstract section is missing"))
    else:
        # Links contribute their visible text; Markdown punctuation is not a word.
        n_words = len(WORD.findall(rendered[abstract[0]:abstract[1]]))
        if not 150 <= n_words <= 250:
            issues.append(Issue(head_line.get("abstract", 0), f"Abstract has {n_words} words; required range is "
                                                               f"150–250"))
    usage = by_name.get("llm usage statement")
    if usage is None:
        issues.append(Issue(0, "required LLM Usage Statement section is missing"))
    elif not WORD.search(rendered[usage[0]:usage[1]]):
        issues.append(Issue(head_line.get("llm usage statement", 0), "LLM Usage Statement is blank"))
    return issues


def value_issues(subs: list[tuple[str, int, str | None, str]], line: LineIndex) -> list[Issue]:
    """Rendered result values and tables must not smuggle in comments, placeholders or TODO markers. Excerpts quote
    logs verbatim (in code or quotes), so their text is not checked here."""
    issues: list[Issue] = []
    for token, pos, value, kind in subs:
        if value is None or kind not in ("key", "table"):
            continue
        for bad, what in (("<!--", "an HTML comment"), ("{{", "an unrendered placeholder")):
            if bad in value:
                issues.append(Issue(line(pos), f"{{{{{token}}}}} inserts {what}"))
        m = EDITORIAL_WORD.search(value) if kind == "key" else None
        if m:
            issues.append(Issue(line(pos), f"{{{{{token}}}}} renders the author-TODO marker '{m.group(0)}'"))
    return issues


def editorial_issues(content: str, rendered: str | None = None, *, line: LineIndex | None = None,
                     subs: list | None = None) -> list[Issue]:
    """Human decisions that a placeholder-only check cannot catch in a submission draft, sorted by source line.

    `content` is the report source with its HTML comments blanked (comment_spans lists those); offsets in it are
    source offsets. `rendered` is the filled text (defaults to `content`), used for the section checks."""
    line = line or LineIndex(content)
    prose = prose_view(content)
    issues = bracket_issues(prose, line, content) + marker_issues(prose, line) + \
        pending_value_issues(prose, line, content)
    # Leftover braces: "{{KEY" / "KEY}}" / "{{KEY NAME}}" never match PLACEHOLDER, and "{{{KEY}}}" or "{{a{{b}}}}"
    # leave literal braces around a rendered value. One issue per line.
    brace_at: dict[int, int] = {}
    for m in re.finditer(r"\{\{|\}\}", prose):
        brace_at.setdefault(line(m.start()), m.start())
    for m in PLACEHOLDER.finditer(content):
        if prose[m.start()] == "0" and (content[m.start() - 1:m.start()] == "{" or content[m.end():m.end() + 1] == "}"):
            brace_at.setdefault(line(m.start()), m.start())
    for ln, pos in brace_at.items():
        start = max(content.rfind("\n", 0, pos) + 1, pos - 30)
        issues.append(Issue(ln, f"malformed or nested placeholder: {_preview(content[start:pos + 40])}"))
    issues += section_issues(rendered if rendered is not None else content, content, line)
    issues += value_issues(subs or [], line)
    return sorted(set(issues))


def _lines_label(lines: list[int]) -> str:
    u = sorted(set(lines))
    return ("line " if len(u) == 1 else "lines ") + ", ".join(map(str, u))


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
    line = LineIndex(source)
    # Editorial comments are not report content or a placeholder contract. Still block a final render until the
    # author removes them; resolving placeholders inside them would manufacture unrelated missing-key failures.
    # They are blanked in place (newlines kept), so every offset below is still a source offset.
    issues: list[Issue] = []
    content = source
    if a.confirmatory:
        spans, issues = comment_spans(source, line)
        chars = list(source)
        for s, e in spans:
            _blank(chars, s, e)
        content = "".join(chars)
    subs: list[tuple[str, int, str | None, str]] = []
    text, unresolved, unfinished = render(content, results, tables, figures, out, excerpts,
                                          on_sub=lambda t, pos, v, kind: subs.append((t, pos, v, kind)))
    where: dict[str, list[int]] = defaultdict(list)
    for token, pos, _, _ in subs:
        where[token].append(line(pos))
    if a.confirmatory:
        issues = sorted(set(issues + editorial_issues(content, text, line=line, subs=subs)))
    for u in sorted(set(unresolved)):
        print(f"  unresolved: {{{{{u}}}}} ({_lines_label(where[u])})", file=sys.stderr)
    for u in sorted(set(unfinished)):
        print(f"  pending: {{{{{u}}}}} ({_lines_label(where[u])})", file=sys.stderr)
    for issue in issues:
        print(f"  editorial: {issue}", file=sys.stderr)
    if a.confirmatory:
        if results.get("meta.preliminary") is not False:
            print("  results.json is preliminary (meta.preliminary is not false)", file=sys.stderr)
            issues.append(Issue(0, "preliminary results"))
        if results.get("meta.cohort") != "confirmatory":
            print(f"  results.json cohort is {results.get('meta.cohort')!r}, not confirmatory", file=sys.stderr)
            issues.append(Issue(0, "non-confirmatory cohort"))
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
