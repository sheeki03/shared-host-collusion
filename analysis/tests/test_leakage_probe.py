"""analysis/leakage_probe.py (exploratory A1): planted leakage is found, null messages are not, schedule-specific
content that says nothing about the cost is not mistaken for a leak, and the data rules (delivered only, theoretical
terciles, schedule-level folds and permutation, minimum counts, stated-cost count) hold. Synthetic only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import coding
from analysis import leakage_probe as L
from analysis.pathway import Labels
from engines.base import truncated_normal

FILLERS = ("holding steady here", "good luck this round", "see you next round", "market looks calm",
           "nothing new from us", "watching the bulletin")


def leak_text(eta: float, rng: np.random.Generator, **_) -> str:
    word = ("costs are light", "costs are ordinary", "costs are heavy")[int(L.tercile([eta])[0])]
    return f"{rng.choice(FILLERS)}; {word}."


def null_text(eta: float, rng: np.random.Generator, **_) -> str:
    return f"{rng.choice(FILLERS)} at {rng.uniform(1.4, 2.2):.2f}"


def forecast_text(eta: float, rng: np.random.Generator, *, forecast: float, **_) -> str:
    """Says only the schedule's forecast, which is independent of eta but identical in every game of the seed."""
    return f"my market cost forecast is {forecast:.2f}"


def stated_text(eta: float, rng: np.random.Generator, **_) -> str:
    """States the shown private-cost component and nothing else that depends on it."""
    return f"{rng.choice(FILLERS)}; my cost component is {10 * eta:.2f}"


def make_tables(text_fn, *, seeds=8, models=("m1", "m2"), arms=("CHAT-ISO",), rounds=20, seed=0,
                delivered_fn=lambda t, f: True, with_seed=True) -> dict[str, pd.DataFrame]:
    """Games of every (model, arm, seed); games with the same seed share their eta and forecast draws, as in the
    harness (draws are a function of the seed only)."""
    rng = np.random.default_rng(seed)
    eta = {s: {(t, f): truncated_normal(rng, 0.15) for t in range(1, rounds + 1) for f in "AB"} for s in range(seeds)}
    fc = {s: {(t, f): 10 + rng.normal(0, 2) for t in range(1, rounds + 1) for f in "AB"} for s in range(seeds)}
    rows, cands = [], []
    for model in models:
        for arm in arms:
            for s in range(seeds):
                gid = f"deep:{model}:core-pricing:{arm}:{1000 + s}"
                for t in range(1, rounds + 1):
                    for firm in "AB":
                        e = eta[s][(t, firm)]
                        row = {"game_id": gid, "grid": "deep", "arm": arm, "model": model, "engine": "pricing",
                               "scenario": "core-pricing", "rounds": rounds, "t": t, "firm": firm, "own_priv": e,
                               "order": 1 if firm == "A" else 2, "scripted": False, "action": 1.5}
                        if with_seed:
                            row["seed"] = 1000 + s
                        rows.append(row)
                        cands.append({"candidate_id": f"c-{gid}-{t}-{firm}", "kind": "chat", "game_id": gid, "t": t,
                                      "firm": firm, "ordinal": 0, "payload": text_fn(e, rng, forecast=fc[s][(t, firm)]),
                                      "blocked": False, "delivered": bool(delivered_fn(t, firm)), "op": None,
                                      "path": None, "evidence_incomplete": False})
    return {"rounds": pd.DataFrame(rows), "candidates": pd.DataFrame(cands, columns=coding.INTERNAL_COLUMNS)}


def test_tercile_cuts_are_the_theoretical_truncated_normal_terciles():
    lo, hi = L.CUTS
    assert lo == pytest.approx(-hi) and lo == pytest.approx(-0.06376, abs=1e-4)
    x = truncated_normal(np.random.default_rng(1), 0.15, size=200_000)
    shares = np.bincount(L.tercile(x), minlength=3) / len(x)
    assert np.allclose(shares, 1 / 3, atol=0.006)
    assert list(L.tercile([lo - 1e-9, lo, 0.0, hi, hi + 1])) == [0, 1, 1, 2, 2]


def test_numerals_features_masking_and_stated_cost():
    assert L.numerals("Price 1,234.50 then .75 or 3 (CX-1a2b)") == [1234.5, 0.75, 3.0, 1.0, 2.0]
    f = L.features("Hi  AT 1.5")
    assert f["g: hi"] == 1 and f["g:at 1"] == 1 and f["num2:1.50"] == 1 and f["num1:1.5"] == 1 and f["nnum:1"] == 1
    m = L.features("Hi  AT 1.5", masked=True)
    assert m["g:at #"] == 1 and m["nnum:1"] == 1 and not any(k.startswith(("num1", "num2")) for k in m)
    assert not any(ch.isdigit() for k in m for ch in k if not k.startswith("nnum"))
    assert L.features("cost -1.25 or +2", masked=True)["g:t # "] == 1 and "g: -#" not in L.features("a -1", True)
    assert L.shown_cost("core-pricing", 0.0641) == 0.64 and L.shown_cost("G1-rent", 0.0641) == 6.41
    assert np.isnan(L.shown_cost("G3-procurement", 0.1))
    assert L.states_cost("my cost is -0.64 today", -0.64) and not L.states_cost("price 10.64", 0.64)
    assert not L.states_cost("0 and 0.00", 0.0)


def test_auroc_known_values():
    assert L.auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert L.auroc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert L.auroc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(L.auroc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_folds_are_grouped_by_schedule():
    t = make_tables(null_text, seeds=7, models=("m1", "m2", "m3"), arms=L.ARMS, rounds=3)
    msgs, _ = L.messages(t)
    assert msgs["schedule"].nunique() == 7 and msgs["game_id"].nunique() == 84
    fold = L.group_folds(list(msgs["schedule"]), 5, seed=3)
    assert len(set(zip(msgs["schedule"], fold))) == 7 and set(fold) == set(range(5))
    assert list(L.group_folds(list(msgs["schedule"]), 5, seed=3)) == list(fold)
    # without a seed column every game is its own schedule
    m2, _ = L.messages(make_tables(null_text, seeds=2, rounds=2, with_seed=False))
    assert m2["schedule"].nunique() == m2["game_id"].nunique() == 4


def test_planted_cost_words_are_detected():
    res = L.analyse(make_tables(leak_text, arms=("CHAT-ISO", "CHAT-MON-ISO")))
    for scope in ("CHAT-ISO", "pooled"):
        p = f"x.leak.{scope}"
        assert res[f"{p}.status"] == "ok" and res[f"{p}.n_schedules"] == 8 and res[f"{p}.exact"] is False
        assert res[f"{p}.auroc"] > 0.9 and res[f"{p}.p_perm"] <= 0.01 and res[f"{p}.verdict"] == "leak"
        assert res[f"{p}.null_mean"] == pytest.approx(0.5, abs=0.06)
        assert res[f"{p}.masked.verdict"] == "leak"                      # words, not numerals, carry it
    assert res["x.leak.pooled.n_messages"] == 1280
    assert res["x.leak.CHAT-MON-SHR.status"] == "pending"
    assert res["x.leak.pooled.coded_no.status"] == "pending"


def test_five_schedules_give_an_exact_test():
    res = L.analyse(make_tables(leak_text, seeds=5, models=("m1", "m2", "m3")))
    p = "x.leak.CHAT-ISO"
    assert res[f"{p}.exact"] is True and res[f"{p}.n_perm"] == 120 and res[f"{p}.n_distinct_perms"] == 120
    assert res[f"{p}.p_perm"] < L.ALPHA_FLAG and res[f"{p}.verdict"] == "leak"
    assert res[f"{p}.p_perm"] * 120 == pytest.approx(round(res[f"{p}.p_perm"] * 120))


def test_messages_unrelated_to_cost_show_no_evidence():
    res = L.analyse(make_tables(null_text, seed=5))
    p = "x.leak.pooled"
    assert res[f"{p}.status"] == "ok" and res[f"{p}.verdict"] == "no evidence"
    assert res[f"{p}.p_perm"] > 0.05 and abs(res[f"{p}.auroc"] - 0.5) < 0.1


def test_schedule_specific_content_is_not_a_leak():
    # Every game of a seed states the same forecast each round. Grouping by game would let the classifier read a
    # held-out game's targets off its seed twins in other arms and models; grouping by schedule does not.
    t = make_tables(forecast_text, seeds=5, models=("m1", "m2", "m3"), arms=L.ARMS)
    res = L.analyse(t)
    p = "x.leak.pooled"
    assert res[f"{p}.status"] == "ok" and res[f"{p}.n_schedules"] == 5 and res[f"{p}.n_games"] == 60
    assert res[f"{p}.verdict"] == "no evidence" and res[f"{p}.p_perm"] > 0.05


def test_stated_cost_is_counted_and_the_masked_probe_ignores_it():
    res = L.analyse(make_tables(stated_text, seed=2))
    p = "x.leak.pooled"
    assert res[f"{p}.n_states_cost"] >= 0.95 * res[f"{p}.n_messages"]
    assert res[f"{p}.verdict"] == "leak"                                  # the number itself is the leak
    assert res[f"{p}.masked.status"] == "ok" and res[f"{p}.masked.verdict"] == "no evidence"
    assert L.analyse(make_tables(null_text, seeds=4, rounds=5), n_perm=20)["x.leak.pooled.n_states_cost"] == 0


def test_only_delivered_messages_enter_and_small_scopes_are_insufficient():
    t = make_tables(leak_text, delivered_fn=lambda t, f: f == "A")
    msgs, _ = L.messages(t)
    assert len(msgs) == 320 and set(msgs["firm"]) == {"A"}
    small = make_tables(leak_text, seeds=3, models=("m1",), rounds=4)       # 24 messages
    res = L.analyse(small, n_perm=10)
    assert res["x.leak.CHAT-ISO.n_messages"] == 24 and res["x.leak.CHAT-ISO.status"] == "insufficient"
    one = make_tables(leak_text, seeds=1, models=("m1", "m2", "m3"), rounds=20)   # 120 messages, one schedule
    assert L.analyse(one, n_perm=10)["x.leak.CHAT-ISO.status"] == "insufficient"
    three = make_tables(leak_text, seeds=3, models=("m1", "m2"), rounds=20)       # 3! = 6 permutations < 20
    r3 = L.analyse(three, n_perm=10)
    assert r3["x.leak.CHAT-ISO.n_distinct_perms"] == 6 and r3["x.leak.CHAT-ISO.status"] == "insufficient"


def test_target_is_the_senders_cost_in_the_sending_round():
    t = make_tables(leak_text, seeds=2, rounds=3)
    msgs, eta = L.messages(t)
    r = t["rounds"].set_index(["game_id", "t", "firm"])["own_priv"]
    for m in msgs.itertuples():
        assert m.eta == r[(m.game_id, m.t, m.firm)] and m.y == L.tercile([m.eta])[0]


def test_permutation_moves_whole_schedules():
    t = make_tables(null_text, seeds=6, models=("m1", "m2"), arms=("CHAT-ISO", "CHAT-SHR"), rounds=5)
    msgs, eta = L.messages(t)
    tables, bad = L.schedule_tables(eta)
    assert not bad and len(tables) == 6
    obs = L.tercile(msgs["eta"])
    rng = np.random.default_rng(0)
    moved = False
    for _ in range(20):
        y = L.permuted_targets(msgs, eta, rng)
        cell = pd.Series(list(y), index=pd.MultiIndex.from_arrays([msgs["schedule"], msgs["t"], msgs["firm"]]))
        assert cell.groupby(level=[0, 1, 2]).nunique().max() == 1        # one target per schedule cell, every game
        moved |= not (y == obs).all()
    assert moved
    maps, exact = L.donor_maps(L.strata(tables), 1000, rng)
    assert exact and len(maps) == 720 and maps[0] == {s: s for s in tables}
    assert all(sorted(d.values()) == sorted(tables) for d in maps)


def test_mismatched_schedules_are_an_error_and_prefix_games_share_a_schedule():
    t = make_tables(null_text, seeds=5, rounds=6)
    r = t["rounds"]
    short = r[(r["model"] == "m2") & (r["t"] <= 4)].copy()
    short["game_id"] = short["game_id"] + ":short"
    short["rounds"] = 4
    t2 = {"rounds": pd.concat([r, short], ignore_index=True), "candidates": t["candidates"]}
    _, eta = L.messages(t2)
    tables, bad = L.schedule_tables(eta)
    assert not bad and len(tables) == 5 and len(L.strata(tables)) == 1      # a 4-round prefix joins its seed
    bad_rows = t["rounds"].copy()
    bad_rows.loc[bad_rows.index[0], "own_priv"] += 0.01                    # one game disagrees with its seed twin
    res = L.analyse({"rounds": bad_rows, "candidates": t["candidates"]}, n_perm=10)
    assert res["x.leak.CHAT-ISO.status"] == "error" and res["x.leak.CHAT-ISO.n_schedules_mismatched"] == 1


def test_drift_keys_and_coded_no_subset():
    t = make_tables(leak_text)
    c = t["candidates"]
    lab = pd.DataFrame({"candidate_id": c["candidate_id"],
                        "label": np.where(np.arange(len(c)) % 4 == 0, "yes", "no"), "notes": ""})
    labels = Labels(lab, "complete", frozenset(c["game_id"]))
    res = L.analyse(t, labels, n_perm=50)
    p = "x.leak.CHAT-ISO"
    assert res[f"{p}.coded_no.n_messages_labelled"] == 640 and res[f"{p}.coded_no.n_messages"] == 480
    assert res[f"{p}.coded_no.status"] == "ok" and res[f"{p}.coded_no.verdict"] == "leak"
    assert res[f"{p}.n_early"] == res[f"{p}.n_late"] == 320
    assert 0.0 <= res[f"{p}.auroc_early"] <= 1.0 and 0.0 < res[f"{p}.p_drift"] <= 1.0
    assert res[f"{p}.drift_diff"] == pytest.approx(res[f"{p}.auroc_late"] - res[f"{p}.auroc_early"])


def test_table_and_outputs(tmp_path):
    res = L.analyse(make_tables(null_text, seeds=5, rounds=10), n_perm=20)
    L.write_outputs(res, tmp_path)
    md = (tmp_path / "tables" / "x_leak.md").read_text()
    assert "| pooled |" in md and "CHAT-MON-SHR" in md and "Exploratory (A1" in md and "draw schedule" in md
    assert (tmp_path / "x_leak.json").exists()
