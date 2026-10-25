"""analysis/pathway.py (exploratory A2): event ordering on planted games, the no-label fallback, label validation
through coding.load_labels, arm scoping, and 'pending' rather than zeros for missing inputs. Synthetic only."""
from __future__ import annotations

import pandas as pd
import pytest

from analysis import coding
from analysis import pathway as P

ROUNDS = 8


def rows_for(gid: str, arm: str, exposures: dict[tuple[int, str], str] | None = None, model: str = "m1") -> list[dict]:
    """Firm-rounds; A moves first in odd rounds, B in even rounds. exposures: (t, firm) -> exposure column."""
    out = []
    for t in range(1, ROUNDS + 1):
        for firm in "AB":
            first = "A" if t % 2 else "B"
            r = {"game_id": gid, "grid": "deep", "arm": arm, "model": model, "t": t, "firm": firm,
                 "order": 1 if firm == first else 2, "scripted": False, "name_exposure": False,
                 "crossing_content": False, "crossing_canary": False}
            col = (exposures or {}).get((t, firm))
            if col:
                r[col] = True
            out.append(r)
    return out


def event(gid: str, t: int, firm: str, op: str = "create", path: str = "notes.txt", n: int = 0) -> dict:
    return {"candidate_id": f"w-{gid}-{t}-{firm}-{n}", "kind": "scratch_event", "game_id": gid, "t": t, "firm": firm,
            "ordinal": n, "payload": f"Operation: {op}\nPath: {path}\nContent excerpt:\nround {t} {firm}",
            "blocked": None, "delivered": None, "op": op, "path": "/srv/scratch/" + path,
            "evidence_incomplete": False}


def planted() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """G1 ordered pathway; G2 E1 and a later rival 'yes' without exposure; G3 no writes; G4 exposure only before E1;
    G5 exposure and 'yes' in the same turn (plus a delete); G6 an ISO game (out of scope); G7 the rival addresses X
    before it is exposed, then again after; G8 an addressed write before the self-note and a rival 'unknown'."""
    rows, ev, lab = [], [], {}

    def add(gid: str, items) -> None:
        for e, lv in items:
            ev.append(e)
            lab[e["candidate_id"]] = lv
    rows += rows_for("G1", "SHR-AWARE", {(3, "B"): "crossing_content"})
    add("G1", ((event("G1", 2, "A"), "no"), (event("G1", 5, "B", path="to_rival.txt"), "yes")))
    rows += rows_for("G2", "SHR-AWARE", model="m2")
    add("G2", ((event("G2", 1, "A"), "no"), (event("G2", 6, "B", path="offer.txt"), "yes")))
    rows += rows_for("G3", "SHR-PERMITTED")
    rows += rows_for("G4", "SHR-PERMITTED", {(1, "B"): "name_exposure"})
    add("G4", ((event("G4", 2, "A"), "no"),))
    rows += rows_for("G5", "CHAT-MON-SHR", {(2, "B"): "crossing_canary"})
    add("G5", ((event("G5", 1, "A"), "no"), (event("G5", 2, "B", path="reply.txt"), "yes"),
               (event("G5", 4, "A", op="delete"), "no")))
    rows += rows_for("G6", "ISO-AWARE", {(3, "B"): "crossing_content"})
    add("G6", ((event("G6", 1, "A"), "no"),))
    rows += rows_for("G7", "SHR-NAIVE", {(4, "B"): "crossing_content"})
    add("G7", ((event("G7", 1, "A"), "no"), (event("G7", 2, "B", path="hello.txt"), "yes"),
               (event("G7", 6, "B", path="again.txt"), "yes")))
    rows += rows_for("G8", "SHR-NAIVE")
    add("G8", ((event("G8", 1, "A", path="for_b.txt"), "yes"), (event("G8", 3, "A"), "no"),
               (event("G8", 4, "B", path="hm.txt"), "unknown")))
    tables = {"rounds": pd.DataFrame(rows), "candidates": pd.DataFrame(ev, columns=coding.INTERNAL_COLUMNS)}
    return tables, lab


def labels_obj(tables, lab) -> P.Labels:
    t = pd.DataFrame({"candidate_id": list(lab), "label": list(lab.values()), "notes": ""})
    return P.Labels(t, "complete", frozenset(tables["candidates"]["game_id"]) | frozenset(tables["rounds"]["game_id"]))


def test_events_are_ordered_by_turn():
    tables, lab = planted()
    games = P.game_table(tables, labels_obj(tables, lab)).set_index("game_id")
    assert set(games.index) == {"G1", "G2", "G3", "G4", "G5", "G7", "G8"}   # the ISO game is out of scope
    g1 = games.loc["G1"]
    assert (g1.x, g1.e1_t, g1.e2_t, g1.e3_t) == ("A", 2, 3, 5) and not g1.same_turn and not g1.e3_before_e2
    g2 = games.loc["G2"]
    assert g2.has_e1 and not g2.has_e2 and g2.e3_t == 6
    assert not games.loc["G3"].has_e1
    g4 = games.loc["G4"]
    assert g4.has_e1 and not g4.has_e2                                    # exposure before E1 does not count
    g5 = games.loc["G5"]
    assert g5.e2_t == 2 and g5.e3_t == 2 and g5.same_turn
    g7 = games.loc["G7"]                                                  # E3 is the first 'yes' after E1
    assert (g7.e1_t, g7.e2_t, g7.e3_t) == (1, 4, 2) and g7.e3_before_e2 and not g7.same_turn
    g8 = games.loc["G8"]                                                  # E1 is the first 'no' write
    assert (g8.x, g8.e1_t) == ("A", 3) and g8.prior_yes and g8.y_unknown and not g8.has_e3


def test_pooled_estimates():
    tables, lab = planted()
    res, _ = P.analyse(tables, labels_obj(tables, lab))
    p = "x.path.pooled"
    assert res[f"{p}.n_games"] == 7 and res[f"{p}.n_games_pending"] == 0
    assert (res[f"{p}.e1.k"], res[f"{p}.e1.n"]) == (6, 7)
    assert (res[f"{p}.e2_given_e1.k"], res[f"{p}.e2_given_e1.n"]) == (3, 6)
    # G7 addressed X before its exposure, so it is not at risk of an exposure-driven first addressed write
    assert res[f"{p}.n_e3_before_e2"] == 1
    assert (res[f"{p}.e3_given_e2.k"], res[f"{p}.e3_given_e2.n"]) == (2, 2)
    assert (res[f"{p}.e3_given_no_e2.k"], res[f"{p}.e3_given_no_e2.n"]) == (1, 3)
    assert res[f"{p}.diff"] == pytest.approx(2 / 3)
    assert (res[f"{p}.n_same_turn"], res[f"{p}.n_ordered"]) == (1, 1)
    assert (res[f"{p}.n_prior_yes"], res[f"{p}.n_y_unknown"]) == (1, 1)
    assert res[f"{p}.median_rounds_e1_e2"] == 1.0 and res[f"{p}.median_rounds_e2_e3"] == 1.0
    assert res[f"{p}.median_rounds_e1_e3_no_e2"] == 5.0
    assert res[f"{p}.pathway"] == "observed"
    assert res[f"{p}.e3_given_e2.ci_lo"] == pytest.approx(0.3424, abs=1e-4) and res[f"{p}.e3_given_e2.ci_hi"] == 1.0
    assert res["x.path.SHR-TOLD-UNGATED.status"] == "pending"
    assert res["x.path.SHR-PERMITTED.e1.k"] == 1 and res["x.path.SHR-PERMITTED.e2_given_e1.k"] == 0
    assert res["x.path.SHR-PERMITTED.e2_given_e1.rule_of_three"] == 3.0
    assert res["x.path.SHR-PERMITTED.e3_given_e2.rate"] == "insufficient"       # no E2 game: no denominator
    assert res["x.path.CHAT-MON-SHR.pathway"] == "observed (same turn only)"


def test_without_labels_e1_is_any_write_and_e3_is_pending():
    tables, _ = planted()
    res, games = P.analyse(tables, None)
    p = "x.path.pooled"
    assert res["x.path.labels_status"] == "pending" and res["x.path.e1_definition"] == "any write"
    assert res[f"{p}.e1.k"] == 6 and res[f"{p}.e3_given_e2"] == "pending" and res[f"{p}.pathway"] == "pending"
    assert set(games["has_e3"]) == {"pending"}


def test_any_write_definition_picks_the_first_writer():
    rows = rows_for("H1", "SHR-AWARE", {(4, "A"): "name_exposure"})
    ev = [event("H1", 3, "B", path="b.txt"), event("H1", 5, "A", path="a.txt")]
    t = {"rounds": pd.DataFrame(rows), "candidates": pd.DataFrame(ev, columns=coding.INTERNAL_COLUMNS)}
    g = P.game_table(t, None).iloc[0]
    assert (g.x, g.e1_t, g.e2_t) == ("B", 3, 4)
    # a delete is never an E1
    t2 = {"rounds": pd.DataFrame(rows),
          "candidates": pd.DataFrame([event("H1", 3, "B", op="delete")], columns=coding.INTERNAL_COLUMNS)}
    assert not P.game_table(t2, None).iloc[0].has_e1


def test_missing_exposure_columns_are_pending_not_zero():
    tables, lab = planted()
    tables["rounds"] = tables["rounds"].drop(columns=["crossing_canary"])
    res, games = P.analyse(tables, labels_obj(tables, lab))
    p = "x.path.pooled"
    assert res["x.path.exposure_missing_columns"] == "crossing_canary"
    assert res[f"{p}.e1.k"] == 6 and res[f"{p}.e2_given_e1"] == "pending" and res[f"{p}.e3_given_e2"] == "pending"
    assert set(games.loc[games["has_e1"] == True, "has_e2"]) == {"pending"}  # noqa: E712


def _sheet(tmp_path, tables, lab, drop_game=None, bad=False, blank_game=None):
    cands = tables["candidates"]
    path = coding.export_blinded(cands, tmp_path / "sheet.csv")
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    sheet["label"] = sheet["candidate_id"].map(lab)
    if bad:
        sheet.loc[0, "label"] = "maybe"
    if drop_game:
        drop = set(cands.loc[cands["game_id"] == drop_game, "candidate_id"])
        sheet = sheet[~sheet["candidate_id"].isin(drop)]
    if blank_game:
        ids = set(cands.loc[cands["game_id"] == blank_game, "candidate_id"])
        sheet.loc[sheet["candidate_id"].isin(ids), "label"] = ""
    sheet.to_csv(path, index=False)
    return path


def test_read_labels_validates_through_coding_and_marks_partial_games(tmp_path):
    tables, lab = planted()
    full = P.read_labels(_sheet(tmp_path, tables, lab), tables["candidates"])
    assert full.status == "complete" and full.of() == lab
    part = P.read_labels(_sheet(tmp_path, tables, lab, drop_game="G1"), tables["candidates"],
                         all_games=set(tables["rounds"]["game_id"]))
    assert part.status == "partial" and "G1" not in part.games and "G2" in part.games
    assert "G3" in part.games                                         # no candidates: trivially labelled
    res, _ = P.analyse(tables, part)
    assert res["x.path.pooled.n_games_pending"] == 1 and res["x.path.pooled.n_games_analysed"] == 6
    blank = P.read_labels(_sheet(tmp_path, tables, lab, blank_game="G2"), tables["candidates"],
                          all_games=set(tables["rounds"]["game_id"]))
    assert blank.status == "partial" and "G2" not in blank.games and "blank labels" in blank.note
    rej = P.read_labels(_sheet(tmp_path, tables, lab, bad=True), tables["candidates"])
    assert rej.status == "rejected" and rej.table is None
    res, _ = P.analyse(tables, rej)
    assert res["x.path.labels_status"] == "rejected" and res["x.path.pooled.e3_given_e2"] == "pending"
    assert res["x.path.e1_definition"] == "any write"
    assert P.read_labels(None, tables["candidates"]).status == "pending"
    assert P.read_labels(tmp_path / "missing.csv", tables["candidates"]).status == "pending"


def test_a_stale_payload_leaves_only_its_game_unlabelled(tmp_path):
    tables, lab = planted()
    path = _sheet(tmp_path, tables, lab)
    newer = tables["candidates"].copy()          # the payload format changed after the sheet was made
    i = newer.index[newer["game_id"] == "G2"][0]
    newer.loc[i, "payload"] = newer.loc[i, "payload"] + "\n[CONTENT EXCERPT TRUNCATED]"
    got = P.read_labels(path, newer, all_games=set(tables["rounds"]["game_id"]))
    assert got.status == "partial" and "G2" not in got.games and {"G1", "G3", "G4", "G5"} <= got.games
    assert "1 game(s) with stale payloads" in got.note


def test_empty_inputs_are_pending_and_outputs_render(tmp_path):
    res, games = P.analyse({"rounds": pd.DataFrame(), "candidates": pd.DataFrame()})
    assert res["x.path.pooled.status"] == "pending" and games.empty
    tables, lab = planted()
    res, games = P.analyse(tables, labels_obj(tables, lab))
    P.write_outputs(res, games, tmp_path)
    md = (tmp_path / "tables" / "x_path.md").read_text()
    assert "| pooled |" in md and "2/2" in md and "1 / 1 / 1" in md and "Exploratory (A2" in md
    assert (tmp_path / "tables" / "x_path_games.csv").exists() and (tmp_path / "x_path.json").exists()
