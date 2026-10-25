"""Checks for tools/prompt_diff.py: section G of report/APPENDIX.md restates tools/pick_excerpts.py (no logs read)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import pick_excerpts as pe  # noqa: E402
import prompt_diff as pd_  # noqa: E402


def test_section_g_e3_rules_are_the_pickers_labels():
    rules = pd_.E3_EVASION_RULES + pd_.E3_FALLBACK_RULES
    assert tuple(t for t, _ in rules) == pe.E3_TIERS
    for tier, text in pd_.E3_EVASION_RULES:
        assert pe.E3_LABELS[tier] == (True, text)
    for tier, text in pd_.E3_FALLBACK_RULES:
        assert pe.E3_LABELS[tier] == (False, text)


def test_section_g_headings_and_wording_match_picker_revision_2():
    g = pd_.section_g()
    heads = re.findall(r"^### (G\.\d) (.*)$", g, re.M)
    assert [h for h, _ in heads] == ["G.1", "G.2", "G.3"]
    titles = dict(heads)
    assert titles["G.1"].startswith(pe.TITLES[1])
    assert titles["G.2"].startswith(pe.TITLES[2]) and "coordination" not in titles["G.2"].lower()
    assert titles["G.3"].startswith("Gate block followed by a trace-confirmed crossing")
    assert "not an evasion" in titles["G.3"]
    assert "Redacted, sometimes truncated quoted spans" in g
    assert "verbatim" not in g.lower() and "Delivered chat coordination" not in g
    assert "Timing:" in g and "Account:" in g and "cannot verify" in g
    assert "not an evasion under the pre-registered definition" in g
    assert "passed the lexical screen" in g and "No coded label is shown" in g
    for tier, text in pd_.E3_EVASION_RULES + pd_.E3_FALLBACK_RULES:
        assert f"- ({tier}) {text}" in g
    assert set(pe.E1_OTHER_ARMS) <= set(re.findall(r"[A-Z][A-Z-]+[A-Z]", g)) and all(a in g for a in pe.GATED_ARMS)
    assert [m.group(1) for m in pe.EXCERPT_RE.finditer(g)] == ["1", "2", "3"]


def test_appendix_section_g_is_the_last_section_and_placeholders_appear_once():
    text = (ROOT / "report" / "APPENDIX.md").read_text()
    assert text.count("## G. Transcript excerpts") == 1
    assert text.rstrip("\n").endswith("{{EXCERPT:3}}")
    assert [m.group(1) for m in pe.EXCERPT_RE.finditer(text)] == ["1", "2", "3"]
    assert pd_.section_g().rstrip("\n") in text
