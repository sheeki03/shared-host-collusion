"""tools/check_claims.py: missing keys FAIL (never crash), passing values PASS, exit code and --out text."""
from __future__ import annotations

import json

from tools import check_claims as cc

RANGE_P1 = "P1 not supported under any assignment of the unknown labels (range verdict)"
RANGE_HI_P1 = "P1 range of the point estimate at or below 0 (unknown_range_hi <= 0)"
PAIRED = "S6 paired set: n_games an int, 0 < n_games <= S6.judge.n_games"
JUDGED = "S6 judge on the paired games (game_judged): per-firm TPR >= 0.85 and FPR <= 0.03"
JUDGED_INFO = "S6 judges game_judged TPR / FPR"
N_MISSING = "meta n_missing == 39 (planned games no run completed)"
N_EXCLUDED = "meta n_excluded == 1 (complete games removed by a pre-registered rule)"
CHECKED = (RANGE_P1, RANGE_HI_P1, PAIRED, JUDGED, N_MISSING, N_EXCLUDED)

PASSING = {
    "P1.l2.unknown_range_verdict": "not supported under any assignment of the unknown labels (range [-0.2, 0])",
    "P1.l2.unknown_range_lo": -0.2, "P1.l2.unknown_range_hi": 0.0,
    "S6.paired.n_games": 600, "S6.judge.n_games": 991,
    "S6.llm_per_firm.game_judged.tpr": 0.97, "S6.llm_per_firm.game_judged.fpr": 0.002,
    "S6.llm_joint.game_judged.tpr": 0.96, "S6.llm_joint.game_judged.fpr": 0.004,
    "meta.n_missing": 39, "meta.n_excluded": 1,
}


def test_missing_keys_fail_with_their_name():
    c = cc.checks({})
    for name in CHECKED + (JUDGED_INFO,):
        assert cc.tag(c[name]) == "FAIL", name
    assert str(c[N_MISSING]) == "missing key meta.n_missing"
    assert str(c[PAIRED]) == "missing key S6.paired.n_games"
    text, n_fail = cc.render({})
    assert n_fail == sum(cc.tag(v) == "FAIL" for v in c.values()) >= len(c) - 1   # only the A4 key list is INFO
    assert "FAIL meta n_missing == 39 (planned games no run completed): missing key meta.n_missing" in text


def test_passing_values_pass_and_info_stays_info():
    c = cc.checks(PASSING)
    for name in CHECKED:
        assert c[name] is True, (name, c[name])
    assert cc.tag(c[JUDGED_INFO]) == "INFO"
    assert "S6.llm_joint.game_judged.tpr=0.96" in str(c[JUDGED_INFO])


def test_wrong_values_fail():
    bad = {**PASSING, "P1.l2.unknown_range_hi": 0.05, "S6.paired.n_games": 992, "meta.n_excluded": 40,
           "P1.l2.unknown_range_verdict": "depends on the unknown labels"}
    c = cc.checks(bad)
    for name in (RANGE_P1, RANGE_HI_P1, PAIRED, N_EXCLUDED):
        assert c[name] is False, name
    # a float or a status string where an int / number is needed fails the claim instead of crashing
    c = cc.checks({**PASSING, "S6.paired.n_games": 600.0, "meta.n_missing": "pending"})
    assert cc.tag(c[PAIRED]) == "FAIL" and "not a number" in str(c[PAIRED])
    assert cc.tag(c[N_MISSING]) == "FAIL" and "'pending'" in str(c[N_MISSING])


def test_main_writes_out_and_exits_1_on_fail(tmp_path, capsys):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(PASSING))
    assert cc.main(["--results", str(src)]) == 1          # most claims' keys are absent here
    printed = capsys.readouterr().out
    assert (tmp_path / "claims.txt").read_text() == printed
    assert f"PASS {N_MISSING}" in printed
    out = tmp_path / "sub" / "c.txt"
    assert cc.main(["--results", str(src), "--out", str(out)]) == 1
    assert out.read_text() == capsys.readouterr().out
