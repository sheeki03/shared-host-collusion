"""Final submission gates for the report renderer; no model calls or PDF compiler."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import fill_report as fr


ABSTRACT = ("We study two simulated pricing agents with separate private inputs and a controlled shared storage "
            "path. The experiment records commands, prices and independent trace evidence while comparing matched "
            "market draws across storage conditions. We report uncertainty and limitations so each claim remains "
            "tied to the observed behaviour. ") * 4
USAGE = "The author used AI tools for code and drafting, reviewed their output, and verified all results."
FINAL = {"meta.preliminary": False, "meta.cohort": "confirmatory"}
CI = {**FINAL, "P1.l2.diff": 0.25, "P1.l2.ci_lo": 0.1, "P1.l2.ci_hi": 0.4}
BODY_LINE = 9          # report() puts the first line of `body` on source line 9


def report(body: str = "The confirmed results are reported with intervals.", *, abstract: str = ABSTRACT,
           usage: str = USAGE) -> str:
    return (f"# Study\n\n## Abstract\n\n{abstract}\n\n## 1. Results\n\n{body}\n\n"
            f"## LLM Usage Statement\n\n{usage}\n")


def render(tmp_path: Path, source: str, *, results: dict | None = None, final: bool = True) -> tuple[int, Path]:
    src = tmp_path / "REPORT.md"
    src.write_text(source)
    data = tmp_path / "results.json"
    data.write_text(json.dumps(results or FINAL))
    out = tmp_path / "REPORT.filled.md"
    args = ["--report", str(src), "--results", str(data), "--out", str(out)]
    if final:
        args.append("--confirmatory")
    return fr.main(args), out


def blocked(tmp_path, capsys, body: str, *, results: dict | None = None) -> str:
    """Render `body` as a final report, require a refusal with no output file, and return stderr."""
    rc, out = render(tmp_path, report(body), results=results)
    err = capsys.readouterr().err
    assert rc == 1 and not out.exists(), err
    return err


def passes(tmp_path, capsys, body: str, *, results: dict | None = None) -> Path:
    rc, out = render(tmp_path, report(body), results=results)
    err = capsys.readouterr().err
    assert rc == 0 and out.exists(), err
    return out


def test_valid_final_report_and_safe_markdown(tmp_path):
    assert 150 <= len(fr.WORD.findall(ABSTRACT)) <= 250
    body = "The 95% CI was [0.1, 0.3]. See [the source](https://example.org).\n\n```text\n[AWARE]\n```"
    rc, out = render(tmp_path, report(body))
    assert rc == 0 and out.exists() and "[0.1, 0.3]" in out.read_text()


@pytest.mark.parametrize("body", ["Agents [did / did not] communicate.",
                                   "[One sentence with the observed result.]",
                                   "<!-- choose after inspecting outcomes --> The result is complete.",
                                   "TODO: verify the result."])
def test_final_blocks_editorial_choices_and_removes_stale_output(tmp_path, body):
    out = tmp_path / "REPORT.filled.md"
    out.write_text("old finished-looking report")
    rc, out = render(tmp_path, report(body))
    assert rc == 1 and not out.exists()


@pytest.mark.parametrize("body, what", [
    # the bracketed text runs over a line break (the pre-revision regex stopped at "\n" and missed all of these)
    ("Agents [did / did\nnot] use the shared folder.", "[did / did not]"),
    ("Agents [did\n/ did not] use it.", "[did / did not]"),
    ("The gate blocked most commands. [One-sentence takeaway on containment\nversus inspection, stated only as "
     "far as S2 and S6 support it.]", "(runs to line 10)"),
    # a placeholder inside the bracket renders its own "[lo, hi]", so the bracket nests
    ("The effect was [{{P1.l2.diff}} with CI {{P1.l2.ci}}, or say that it was null].", "or say that it was null"),
    ("Agents [did [or did not]] communicate.", "[did [or did not]]"),
    # half-edited brackets
    ("Agents [did / did not use the shared folder.", "unclosed '['"),
    ("Agents did / did not] use the shared folder.", "unmatched ']'"),
    # fill-ins
    ("Agents used it in [ ] games.", "blank fill-in"),
    ("Agents used it in [___] games.", "blank fill-in"),
    ("Agents used it in [N] games.", "[N]"),
    ("- [ ] check the totals", "unchecked box"),
])
def test_editorial_brackets_block_across_lines_and_nesting(tmp_path, capsys, body, what):
    err = blocked(tmp_path, capsys, body, results=CI)
    hit = [s for s in err.splitlines() if s.startswith(f"  editorial: line {BODY_LINE}:")]
    assert hit and what in " ".join(hit), err


def test_bracket_in_a_table_cell_is_listed_on_its_row(tmp_path, capsys):
    body = "| arm | result |\n|---|---|\n| SHR | agents [did / did not] write |\n| ISO | [0.1, 0.3] |"
    err = blocked(tmp_path, capsys, body)
    assert f"line {BODY_LINE + 2}: unresolved editorial bracket [did / did not]" in err
    assert "[0.1, 0.3]" not in err


def test_nested_bracket_is_listed_once(tmp_path, capsys):
    err = blocked(tmp_path, capsys, "Agents [did [or did not]] communicate.")
    assert err.count("unresolved editorial bracket") == 1


def test_every_unresolved_item_is_listed_with_its_line(tmp_path, capsys):
    body = ("Agents [did / did not] communicate.\n"                                  # 9
            "<!-- author TODO: add the hand check -->\n"                              # 10
            "The gate blocked {{S2.block_rate}} of commands, [and / but] evasion\n"  # 11
            "was rare. TBD\n"                                                         # 12
            "Kappa: pending.\n"                                                       # 13
            "{{missing.key}} and {{missing.key}} again\n\n"                           # 14
            "<!-- if P3 is null: say so -->")                                         # 16
    err = blocked(tmp_path, capsys, body, results={**FINAL, "S2.block_rate": "pending"})
    expected = ["line 9: unresolved editorial bracket [did / did not]",
                "line 10: HTML comment (author TODO) remains: author TODO: add the hand check",
                "line 11: unresolved editorial bracket [and / but]",
                "line 12: author-TODO marker 'TBD' remains",
                "line 13: unfinished value 'pending' remains: Kappa: pending.",
                "line 16: HTML comment (conditional wording) remains: if P3 is null: say so"]
    editorial = [s.removeprefix("  editorial: ") for s in err.splitlines() if s.startswith("  editorial: ")]
    assert editorial == expected, err                     # every item, in source-line order, nothing else
    assert "  pending: {{S2.block_rate}} (line 11)" in err
    assert "  unresolved: {{missing.key}} (line 14)" in err
    assert "6 editorial/cohort issues" in err


def test_legitimate_brackets_do_not_block(tmp_path, capsys):
    body = ("Prior work [1] and [2, 3] and [4–6] found effects [@fish2024; @lee2026] (see also\n"
            "[Zolkowski et al. 2025; Fish et al. 2024]). The interval [0.1, 0.3] and [1.2e-05, 0.3], the CI\n"
            "{{P1.l2.ci}}, the bounds [{{P1.l2.ci_lo}}, {{P1.l2.ci_hi}}] and [n/a, 0.2].\n"
            "Prices lie in [0, 1) and costs in (0, 1] for x[i] and draws U[60, 100], with\n"
            "[a, b] and [ci_lo, ci_hi] bounds; [n = 12], [p < 0.001], [S1], [Table C11], [A1–A4],\n"
            "[Figure 2(c)], [sic], […]. See [the source](https://example.org), ![fig](fig.png),\n"
            "[the guide][ref] and a note[^1].\n\n"
            "[ref]: https://example.org/guide\n"
            "[^1]: A footnote.\n\n"
            "- [x] a checked box\n\n"
            "| arm | CI | rate |\n|---|---|---|\n| SHR | [0.1, 0.3] | [1.2e-05, 0.3] |\n\n"
            "`[AWARE]` is a prompt tag.\n\n"
            "> [stderr] quoted from a transcript\n\n"
            "```text\n[did / did not]\nTODO pending\n```")
    out = passes(tmp_path, capsys, body, results=CI)
    assert "[0.1, 0.4], the bounds [0.1, 0.4]" in out.read_text()


def test_generated_tables_with_numeric_brackets_pass(tmp_path, capsys):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "ci.md").write_text("| arm | CI |\n|---|---|\n| SHR | [1.20e-04, 0.0300] |\n")
    passes(tmp_path, capsys, "{{TABLE:ci}}")


@pytest.mark.parametrize("body", ["FIXME: confirm the totals.", "The effect was XX%.", "The rate was ??.",
                                  "This paragraph is TO BE WRITTEN.", "Spend was tbd at export."])
def test_author_todo_markers_block(tmp_path, capsys, body):
    err = blocked(tmp_path, capsys, body)
    assert f"line {BODY_LINE}: author-TODO marker" in err


@pytest.mark.parametrize("body", ["Kappa: pending.", "| item | value |\n|---|---|\n| spend | pending |",
                                  "The blinded hand check is still pending.", "- insufficient data",
                                  "Rate = **pending**", "The release (pending) follows."])
def test_literal_pending_values_block(tmp_path, capsys, body):
    err = blocked(tmp_path, capsys, body)
    assert "unfinished value" in err


def test_pending_in_prose_and_headers_passes(tmp_path, capsys):
    body = ("Raw transcripts are withheld pending the data-release decision.\n\n"
            "Table C11 lists planned, complete, trimmed and pending games per cell.\n\n"
            "With the decision pending, logs stay private. There were insufficient data to fit it.\n\n"
            "| arm | pending | complete |\n|---|---|---|\n| SHR | 0 | 12 |")
    passes(tmp_path, capsys, body)


@pytest.mark.parametrize("value, ok", [("pending (judges not run)", False),
                                       ("insufficient data: fewer than 10 games", False),
                                       ("Pending", False),
                                       ("n/a (supporting, outside the Holm family)", True),
                                       ("ORACLE excluded; gated arms excluded", True)])
def test_status_values_with_qualifiers(tmp_path, capsys, value, ok):
    rc, out = render(tmp_path, report("Status: {{x}}."), results={**FINAL, "x": value})
    err = capsys.readouterr().err
    assert (rc == 0) is ok and out.exists() is ok, err
    if not ok:
        assert f"  pending: {{{{x}}}} (line {BODY_LINE})" in err


def test_generated_table_pending_header_is_not_a_value(tmp_path, capsys):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "coverage.md").write_text(
        "| grid | arm | expected | complete | pending | coverage |\n|---|---|---|---|---|---|\n"
        "| core | SHR | 12 | 12 | 0 | 1.00 |\n")
    passes(tmp_path, capsys, "{{TABLE:coverage}}")
    (tmp_path / "tables" / "held.md").write_text("| monitor | tpr |\n|---|---|\n| judge | pending |\n")
    err = blocked(tmp_path, capsys, "{{TABLE:held}}")
    assert f"  pending: {{{{TABLE:held}}}} (line {BODY_LINE})" in err


@pytest.mark.parametrize("body", ["The effect was {{P1.l2 diff}}.", "The effect was {{P1.l2.diff}.",
                                  "The effect was {P1.l2.diff}}.", "The effect was {{{P1.l2.diff}}}."])
def test_malformed_and_nested_placeholders_block(tmp_path, capsys, body):
    err = blocked(tmp_path, capsys, body, results=CI)
    assert f"line {BODY_LINE}: malformed or nested placeholder" in err


@pytest.mark.parametrize("value, what", [("TBD", "author-TODO marker"), ("<!-- later -->", "HTML comment"),
                                         ("{{other.key}}", "unrendered placeholder")])
def test_values_cannot_smuggle_editorial_text(tmp_path, capsys, value, what):
    err = blocked(tmp_path, capsys, "Status: {{x}}.", results={**FINAL, "x": value})
    assert f"line {BODY_LINE}: {{{{x}}}}" in err and what in err


@pytest.mark.parametrize("body, what", [("<!-- unfinished note\nThe result is complete.", "unclosed HTML comment"),
                                        ("The result is complete. -->", "stray '-->'")])
def test_broken_comments_block(tmp_path, capsys, body, what):
    err = blocked(tmp_path, capsys, body)
    assert f"line {BODY_LINE}: {what}" in err


@pytest.mark.parametrize("words", [149, 251])
def test_final_enforces_abstract_word_range(tmp_path, capsys, words):
    rc, out = render(tmp_path, report(abstract="word " * words))
    assert rc == 1 and not out.exists()
    assert f"line 3: Abstract has {words} words" in capsys.readouterr().err


def test_final_requires_llm_usage_and_sections(tmp_path):
    for source in (report(usage="  "), "# Study\n\n## LLM Usage Statement\n\n" + USAGE,
                   "# Study\n\n## Abstract\n\n" + ABSTRACT):
        rc, out = render(tmp_path, source)
        assert rc == 1 and not out.exists()


def test_word_count_uses_filled_abstract_and_draft_stays_flexible(tmp_path):
    rc, out = render(tmp_path, report(abstract="{{expanded}}"),
                     results={"meta.preliminary": False, "meta.cohort": "confirmatory",
                              "expanded": "word " * 251})
    assert rc == 1 and not out.exists()
    rc, out = render(tmp_path, "# Draft\n\n{{missing}}\n<!-- still editing -->", final=False)
    assert rc == 0 and "{{missing}}" in out.read_text()


def test_draft_mode_ignores_editorial_text(tmp_path):
    rc, out = render(tmp_path, report("Agents [did / did\nnot] communicate. TODO"), final=False)
    assert rc == 0 and "[did / did\nnot]" in out.read_text()


def test_comment_placeholders_do_not_pollute_visible_key_contract(tmp_path, capsys):
    rc, out = render(tmp_path, report("<!-- {{NOT_A_RESULT_KEY}} -->\nThe result is complete."))
    err = capsys.readouterr().err
    assert rc == 1 and not out.exists()
    assert "HTML comment" in err and "unresolved: {{NOT_A_RESULT_KEY}}" not in err


def test_confirmatory_cohort_marker_is_explicit(tmp_path):
    rc, out = render(tmp_path, report(), results={"meta.preliminary": False})
    assert rc == 1 and not out.exists()


def test_pending_lists_and_tables_block_final_render(tmp_path):
    data = {"meta.preliminary": False, "meta.cohort": "confirmatory", "items": ["done", "Pending"]}
    rc, out = render(tmp_path, report("{{items}}"), results=data)
    assert rc == 1 and not out.exists()
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "x.md").write_text("| result |\n|---|\n| insufficient data |\n")
    rc, out = render(tmp_path, report("{{TABLE:x}}"))
    assert rc == 1 and not out.exists()


def test_out_cannot_replace_source(tmp_path):
    source = tmp_path / "REPORT.md"
    source.write_text(report())
    data = tmp_path / "results.json"
    data.write_text(json.dumps({"meta.preliminary": False, "meta.cohort": "confirmatory"}))
    with pytest.raises(SystemExit, match="must not overwrite"):
        fr.main(["--report", str(source), "--out", str(source), "--results", str(data), "--confirmatory"])
    assert source.read_text() == report()
