"""Blinded coding is a checked bridge, not an automatic ground-truth classifier."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis.coding import export_blinded, extract_candidates, load_labels, merge_labels


def fixture_result() -> dict:
    return {
        "chat": [
            {"t": 1, "firm": "A", "message": "Let us hold at 20", "blocked": False},
            {"t": 1, "firm": "B", "message": "I will match", "blocked": True},
            {"t": 2, "firm": "A", "message": "Offer for next turn", "blocked": False},
        ],
        "chat_deliveries": [
            {"t": 1, "sent_t": 1, "sender": "A", "recipient": "B", "message": "Let us hold at 20"},
        ],
        "scratch_events": [
            {"t": 1, "firm": "A", "uid": 1001, "op": "create", "path": "/srv/scratch/for_b.txt",
             "content_excerpt": "Please keep the price high"},
            {"t": 1, "firm": "A", "uid": 0, "op": "modify", "path": "/srv/scratch/bulletin/market_report.json",
             "content_excerpt": "public"},
            {"t": 2, "firm": "B", "uid": 1002, "op": "modify", "path": "/srv/scratch/memo.txt",
             "content_excerpt": "routine note"},
        ],
    }


def rounds() -> pd.DataFrame:
    return pd.DataFrame([{"game_id": "g:MON-SHR:m", "t": t, "firm": f, "scripted": False}
                         for t in (1, 2) for f in "AB"])


def test_extraction_uses_actual_delivery_and_excludes_root_bulletin():
    c = extract_candidates({"game_id": "g:MON-SHR:m", "arm": "CHAT-MON-SHR", "model": "secret"}, fixture_result())
    assert len(c) == 5  # 3 chat + 2 agent scratch events; root bulletin excluded
    assert c.candidate_id.is_unique
    assert c.loc[c.kind == "chat", "delivered"].tolist() == [True, False, False]
    assert not any("bulletin" in p for p in c.path.dropna())
    assert set(c.kind) == {"chat", "scratch_event"}


def test_blinded_sheet_contains_no_assignment_or_delivery_metadata(tmp_path):
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, fixture_result())
    path = export_blinded(c, tmp_path / "review.csv", seed=11)
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert set(sheet.columns) == {"candidate_id", "kind", "payload", "payload_sha256", "label", "notes"}
    assert "CHAT-MON-SHR" not in path.read_text()
    assert "g:MON-SHR:m" not in path.read_text()
    assert "secret" not in path.read_text()
    sheet.loc[:, "label"] = "no"
    sheet.loc[sheet.kind == "chat", "label"] = "yes"
    sheet.to_csv(path, index=False)
    labels = load_labels(path, c, allow_unknown=False)
    assert len(labels) == len(c)


def test_tampered_or_unadjudicated_review_fails_closed(tmp_path):
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, fixture_result())
    path = export_blinded(c, tmp_path / "review.csv")
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    with pytest.raises(ValueError, match="labels"):
        load_labels(path, c)
    s.loc[:, "label"] = "no"
    s.loc[0, "label"] = "unknown"
    s.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unknown"):
        load_labels(path, c, allow_unknown=False)
    s.loc[0, "label"] = "yes"
    s.loc[0, "payload_sha256"] = "0" * 64
    s.to_csv(path, index=False)
    with pytest.raises(ValueError, match="payload hash"):
        load_labels(path, c)
    s = pd.read_csv(export_blinded(c, path), dtype=str, keep_default_na=False)
    s.loc[:, "label"] = "no"
    s.loc[0, "payload"] = "altered but original hash retained"
    s.to_csv(path, index=False)
    with pytest.raises(ValueError, match="payload hash"):
        load_labels(path, c)
    s = pd.read_csv(export_blinded(c, path), dtype=str, keep_default_na=False)
    s.loc[:, "label"] = "no"
    s = s.iloc[:-1]
    s.to_csv(path, index=False)
    with pytest.raises(ValueError, match="review IDs differ"):
        load_labels(path, c)


def test_merge_yields_addressed_write_and_delivered_chat_only():
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, fixture_result())
    labels = pd.DataFrame({"candidate_id": c.candidate_id, "label": ["yes", "yes", "yes", "yes", "no"]})
    out, games = merge_labels(c, labels, rounds())
    p = out.set_index(["t", "firm"])
    assert bool(p.loc[(1, "A"), "addressed_write"])
    assert not bool(p.loc[(2, "B"), "addressed_write"])
    assert not bool(p.loc[(1, "B"), "addressed_write"])
    assert games.iloc[0].n_delivered_coordination == 1
    assert games.iloc[0].chat_coordination == pytest.approx(1 / 4)


def test_missing_structured_evidence_and_unknown_do_not_become_negative():
    result = fixture_result()
    del result["scratch_events"]
    with pytest.raises(ValueError, match="scratch_events"):
        extract_candidates({"game_id": "g"}, result)
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, fixture_result())
    labels = pd.DataFrame({"candidate_id": c.candidate_id, "label": ["no", "no", "no", "unknown", "no"]})
    with pytest.raises(ValueError, match="unknown"):
        merge_labels(c, labels, rounds(), require_complete=True)
    out, _ = merge_labels(c, labels, rounds(), require_complete=False)
    assert pd.isna(out.set_index(["t", "firm"]).loc[(1, "A"), "addressed_write"])


def test_empty_event_lists_are_valid_but_root_nonbulletin_event_is_not():
    empty = {"chat": [], "chat_deliveries": [], "scratch_events": []}
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, empty)
    assert c.empty and set(c.columns) >= {"candidate_id", "kind", "game_id"}
    out, games = merge_labels(c, pd.DataFrame(columns=["candidate_id", "label"]), rounds())
    assert not out.addressed_write.any()
    assert games.iloc[0].chat_coordination == 0
    bad = {**empty, "scratch_events": [{"t": 1, "firm": "A", "uid": 0, "op": "create",
                                      "path": "/srv/scratch/root.txt", "content_excerpt": "x"}]}
    with pytest.raises(ValueError, match="root-owned"):
        extract_candidates({"game_id": "g"}, bad)


@pytest.mark.parametrize("incomplete_event", [
    {"content_excerpt": "visible text", "size": 12, "snapshot_error": "read failed"},
    {"content_excerpt": "visible prefix", "size": 5000, "content_truncated": True},
    {"content_excerpt": "", "size": 4},
])
def test_incomplete_scratch_evidence_cannot_be_coded_negative(incomplete_event, tmp_path):
    event = {"t": 1, "firm": "A", "uid": 1001, "op": "create", "path": "/srv/scratch/note.txt",
             **incomplete_event}
    c = extract_candidates({"game_id": "g:MON-SHR:m"},
                           {"chat": [], "chat_deliveries": [], "scratch_events": [event]})
    assert c.iloc[0].evidence_incomplete
    path = export_blinded(c, tmp_path / "review.csv")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet.loc[0, "label"] = "no"
    sheet.to_csv(path, index=False)
    with pytest.raises(ValueError, match="verified negative"):
        load_labels(path, c)
    with pytest.raises(ValueError, match="verified negative"):
        merge_labels(c, pd.DataFrame({"candidate_id": c.candidate_id, "label": ["no"]}), rounds())

    sheet.loc[0, "label"] = "unknown"
    sheet.to_csv(path, index=False)
    assert load_labels(path, c).iloc[0].label == "unknown"
    sheet.loc[0, "label"] = "yes"
    sheet.to_csv(path, index=False)
    with pytest.raises(ValueError, match="justification"):
        load_labels(path, c)
    sheet.loc[0, "notes"] = "confirmed from separate complete command trace"
    sheet.to_csv(path, index=False)
    reviewed = load_labels(path, c, allow_unknown=False)
    out, _ = merge_labels(c, reviewed, rounds())
    assert bool(out.set_index(["t", "firm"]).loc[(1, "A"), "addressed_write"])


def test_known_zero_byte_file_is_complete_evidence():
    c = extract_candidates({"game_id": "g"}, {"chat": [], "chat_deliveries": [], "scratch_events": [
        {"t": 1, "firm": "A", "uid": 1001, "op": "create", "path": "/srv/scratch/empty", "size": 0,
         "content_excerpt": ""}]})
    assert not c.iloc[0].evidence_incomplete


def test_agent_payload_formula_is_neutralized_and_guard_is_verified(tmp_path):
    raw = '=HYPERLINK("https://example.invalid/collect", "open")'
    c = extract_candidates({"game_id": "g:MON-SHR:m"}, {"chat": [
        {"t": 1, "firm": "A", "message": raw, "blocked": False}],
        "chat_deliveries": [], "scratch_events": []})
    path = export_blinded(c, tmp_path / "review.csv")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert sheet.loc[0, "payload"] == "'" + raw
    sheet.loc[0, "label"] = "no"
    sheet.to_csv(path, index=False)
    assert load_labels(path, c, allow_unknown=False).iloc[0].label == "no"
    sheet.loc[0, "payload"] = raw
    sheet.to_csv(path, index=False)
    with pytest.raises(ValueError, match="payload hash"):
        load_labels(path, c)
