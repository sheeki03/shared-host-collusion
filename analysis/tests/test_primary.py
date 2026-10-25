"""Integrated four-arm P1/P2/P3 decision tests on complete, coded game tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import stats as S


def core_games(n_models: int = 2, n_seeds: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(173)
    arms = ("ISO-AWARE", "SHR-AWARE", "ISO-PERMITTED", "SHR-PERMITTED",
            "CHAT-ISO", "CHAT-SHR", "CHAT-MON-ISO", "CHAT-MON-SHR")
    rows = []
    for model in range(n_models):
        for k in range(n_seeds):
            key_noise = rng.normal(0, .03)
            for arm in arms:
                shr = "SHR" in arm
                mon = "MON" in arm
                permitted = "PERMITTED" in arm
                chat = arm.startswith("CHAT")
                rows.append(dict(grid="deep", model=f"m{model}", scenario="core-pricing", seed=1000 + k,
                                 game_id=f"m{model}:{k}:{arm}", arm=arm,
                                 addressed_write_rate=(.05 + .65 * (shr and permitted) + key_noise
                                                       if not chat else np.nan),
                                 chat_coordination=(.6 - .5 * mon + key_noise if chat else np.nan),
                                 beta=.05 + .4 * (arm == "SHR-AWARE") + .9 * (arm == "CHAT-MON-SHR")
                                 + key_noise))
    return pd.DataFrame(rows)


def test_primary_family_uses_beta_not_file_writes_for_displacement():
    games = core_games()
    result = S.primary_analysis(games, n_boot=499, n_perm=1024, seed=5)
    assert result.permission_carrier.estimate > .5
    assert result.displacement.part_i and result.displacement.part_ii and result.displacement.displacement
    assert "addressed_write_rate" not in result.displacement.part_ii_by_measure  # chat-arm writes were not coded
    assert result.spontaneous_carrier.estimate > .3
    assert set(result.p_raw) == set(result.p_holm) == {"P1", "P2", "P3"}
    assert all(0 <= p <= 1 for p in result.p_holm.values())


def test_pilot_or_unreviewed_primary_data_fails_closed():
    games = core_games()
    games.loc[0, "grid"] = "pilot"
    with pytest.raises(ValueError, match="pilot/self-test"):
        S.primary_analysis(games, n_boot=99, n_perm=128)
    games.loc[0, "grid"] = "deep"
    games.loc[games.arm == "SHR-PERMITTED", "addressed_write_rate"] = np.nan
    with pytest.raises(ValueError, match="adjudicated"):
        S.primary_analysis(games, n_boot=99, n_perm=128)


def test_addressed_writes_cannot_substitute_for_action_dependence():
    games = core_games()
    games.loc[games.arm == "CHAT-MON-SHR", "addressed_write_rate"] = .9
    games.loc[games.arm == "CHAT-MON-SHR", "beta"] = .05
    result = S.primary_analysis(games, n_boot=199, n_perm=256, seed=7)
    assert not result.displacement.part_ii
    assert not result.displacement.displacement


def test_factorial_estimate_and_seed_block_sensitivity():
    games = core_games()
    res = S.factorial_effect(games, "addressed_write_rate", treatment_t="SHR-PERMITTED",
                             treatment_c="ISO-PERMITTED", baseline_t="SHR-AWARE", baseline_c="ISO-AWARE",
                             cluster="seed", n_boot=199, n_perm=1024, seed=2)
    assert res.estimate == pytest.approx(.65, abs=.01)
    assert res.n_pairs == 20 and res.n_clusters == 10
