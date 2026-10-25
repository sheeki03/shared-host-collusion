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


def report(body: str = "The confirmed results are reported with intervals.", *, abstract: str = ABSTRACT,
           usage: str = USAGE) -> str:
    return (f"# Study\n\n## Abstract\n\n{abstract}\n\n## 1. Results\n\n{body}\n\n"
            f"## LLM Usage Statement\n\n{usage}\n")


def render(tmp_path: Path, source: str, *, results: dict | None = None, final: bool = True) -> tuple[int, Path]:
    src = tmp_path / "REPORT.md"
    src.write_text(source)
    data = tmp_path / "results.json"
    data.write_text(json.dumps(results or {"meta.preliminary": False, "meta.cohort": "confirmatory"}))
    out = tmp_path / "REPORT.filled.md"
    args = ["--report", str(src), "--results", str(data), "--out", str(out)]
    if final:
        args.append("--confirmatory")
    return fr.main(args), out


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


@pytest.mark.parametrize("words", [149, 251])
def test_final_enforces_abstract_word_range(tmp_path, words):
    rc, out = render(tmp_path, report(abstract="word " * words))
    assert rc == 1 and not out.exists()


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


def test_comment_placeholders_do_not_pollute_visible_key_contract(tmp_path, capsys):
    rc, out = render(tmp_path, report("<!-- {{NOT_A_RESULT_KEY}} -->\nThe result is complete."))
    err = capsys.readouterr().err
    assert rc == 1 and not out.exists()
    assert "HTML editorial comment" in err and "unresolved: {{NOT_A_RESULT_KEY}}" not in err


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
