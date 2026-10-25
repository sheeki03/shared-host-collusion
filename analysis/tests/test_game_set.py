"""analysis/game_set.py: the outcome-blind duplicate-completion rule (DEVIATIONS D12), and a counts-only audit of the
real local logs (ids, statuses and timestamps only; no outcome is read)."""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

from analysis import game_set as G

ROOT = Path(__file__).resolve().parents[2]


def ref(log, gid="g1", done="2026-10-25T17:00:00+00:00", start="2026-10-25T16:54:00+00:00", *, complete=True,
        problem=None, eval_id=None, status="started"):
    return G.SampleRef(log=log, game_id=gid, completed_at=done, log_started=start, log_status=status,
                       log_eval_id=eval_id or f"ev-{log}", complete=complete, problem=problem)


def test_first_finished_copy_is_kept_whatever_the_log_order():
    gs = G.select([ref("a.eval", done="2026-10-25T18:00:00+00:00", start="2026-10-25T16:00:00+00:00"),
                   ref("b.eval", done="2026-10-25T17:00:00+00:00", start="2026-10-25T17:00:00+00:00")])
    assert gs.kept["g1"].log == "b.eval" and gs.status("a.eval", "g1") == G.DUPLICATE
    assert gs.n_duplicates_dropped == 1 and not gs.admits("a.eval", "g1") and gs.admits("b.eval", "g1")
    row = gs.duplicates_frame().iloc[0]
    assert row["kept_log"] == "b.eval" and row["dropped_log"] == "a.eval"
    assert "finished first" in gs.reason("a.eval", "g1")


def test_timestamps_compare_across_utc_offsets():
    # 22:40 IST is 17:10 UTC: earlier than 17:15 UTC although the string sorts later
    gs = G.select([ref("a.eval", done="2026-10-25T17:15:00+00:00"), ref("b.eval", done="2026-10-25T22:40:00+05:30")])
    assert gs.kept["g1"].log == "b.eval"


def test_ties_go_to_the_earlier_log_start_then_the_log_path():
    t = "2026-10-25T17:30:00+00:00"
    gs = G.select([ref("z.eval", done=t, start="2026-10-25T16:00:00+00:00"),
                   ref("a.eval", done=t, start="2026-10-25T17:00:00+00:00")])
    assert gs.kept["g1"].log == "z.eval"
    gs = G.select([ref("z.eval", done=t), ref("a.eval", done=t)])
    assert gs.kept["g1"].log == "a.eval"


def test_parse_excluded_copy_counts_as_complete_so_selection_is_outcome_blind():
    """A copy above the 20% parse-failure threshold finished first: it is kept (and excluded later under prereg
    section 5); the rule never swaps in the other copy because of how the model played."""
    gs = G.select([ref("a.eval", done="2026-10-25T17:00:00+00:00", problem="parse-failure rate above 20%"),
                   ref("b.eval", done="2026-10-25T17:30:00+00:00")])
    assert gs.kept["g1"].log == "a.eval" and gs.status("b.eval", "g1") == G.DUPLICATE


def test_superseded_and_incomplete_only_games():
    gs = G.select([ref("a.eval", "g1", done=None, complete=False, problem="sample error: cancelled"),
                   ref("b.eval", "g1"), ref("a.eval", "g2", done=None, complete=False, problem="sample error: x"),
                   ref("b.eval", "g2", done=None, complete=False, problem="incomplete game: 6 rows for 40 rounds")])
    assert gs.status("a.eval", "g1") == G.SUPERSEDED and [r.log for r in gs.superseded] == ["a.eval"]
    assert gs.status("a.eval", "g2") == gs.status("b.eval", "g2") == G.INCOMPLETE and "g2" not in gs.kept
    assert gs.admits("a.eval", "g2")            # it stays visible as an exclusion (a harness error)
    assert gs.n_duplicates_dropped == 0 and "complete copy in b.eval" in gs.reason("a.eval", "g1")


def test_fail_closed_cases_strict_and_lenient():
    undated = [ref("a.eval", done=None), ref("b.eval")]
    with pytest.raises(G.GameSetError, match="no completed_at"):
        G.select(undated)
    gs = G.select(undated, strict=False)
    assert gs.kept["g1"].log == "b.eval" and gs.notes        # an undated copy sorts last
    twice = [ref("a.eval"), ref("a.eval", done="2026-10-25T17:01:00+00:00")]
    with pytest.raises(G.GameSetError, match="more than once in log"):
        G.select(twice)
    copies = [ref("x/a.eval", eval_id="same"), ref("y/a.eval", eval_id="same")]
    with pytest.raises(G.GameSetError, match="same Inspect log"):
        G.select(copies)
    gs = G.select(copies, strict=False)
    assert gs.kept["g1"].log == "x/a.eval" and gs.status("y/a.eval", "g1") == G.LOG_COPY
    assert gs.n_duplicates_dropped == 0             # a second copy of one log is not a second completion


def test_single_complete_copy_needs_no_timestamp():
    gs = G.select([ref("a.eval", done=None)])
    assert gs.kept["g1"].log == "a.eval" and not gs.notes


REAL_DIRS = sorted(glob.glob(str(ROOT / "runs" / "full" / "logs-*")) + glob.glob(str(ROOT / "runs" / "remote" / "*" / "logs-*")))


@pytest.mark.skipif(not REAL_DIRS, reason="no local confirmatory log directories")
def test_counts_only_on_the_real_local_logs():
    """Counts, duplicates and which copy the rule keeps, on runs/full/logs-* and runs/remote/*/logs-* (sample
    summaries; full sample reads, without events, only for games with more than one error-free sample). Run with -s
    to see the audit. Nothing but ids, statuses and timestamps is read or printed."""
    gs = G.scan(REAL_DIRS)
    lines = G.audit_lines(gs)
    print("\n" + "\n".join(lines))
    for d in gs.duplicates:
        assert d.kept.complete and d.dropped.complete and d.kept.game_id == d.dropped.game_id
        assert G.finish_key(d.kept) <= G.finish_key(d.dropped)
        assert G._ts(d.kept.completed_at) is not None
    assert {d.game_id for d in gs.duplicates} <= set(gs.kept)
    multi = {}
    for r in gs.refs:
        if r.complete and r.log not in gs.ignored_logs:
            multi[r.game_id] = multi.get(r.game_id, 0) + 1
    assert gs.n_duplicates_dropped == sum(n - 1 for n in multi.values())
    assert all(gs.kept[r.game_id].complete for r in gs.superseded)


def test_lenient_log_copy_uses_the_fuller_copy_and_errors_match_dataset():
    """Lenient mode: of one Inspect log at two paths, the copy with more samples is used (a fresher sync), whatever
    the path order. And an error with an empty message is read exactly as dataset.load reads it."""
    from types import SimpleNamespace as NS
    copies = [ref("x/a.eval", "g1", eval_id="same"), ref("y/a.eval", "g1", eval_id="same"),
              ref("y/a.eval", "g2", eval_id="same")]
    gs = G.select(copies, strict=False)
    assert gs.ignored_logs == {"x/a.eval": "y/a.eval"} and gs.kept["g1"].log == "y/a.eval" and "g2" in gs.kept
    meta = {"game_id": "g", "rounds": 1, "result": {"rows": [{"t": 1, "firm": f, "action": 1.0} for f in "AB"]}}
    _, _, problem = G.sample_problem(NS(metadata=meta, error=NS(message="")))
    assert problem == G.dataset._sample_problem(meta, meta["result"]["rows"], "")
