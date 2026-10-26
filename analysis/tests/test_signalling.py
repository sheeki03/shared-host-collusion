"""analysis/signalling.py: the follow regression, planted arm contrasts, benchmark capture from logs, cohorts, docs."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import dataset
from analysis import signalling as S
from analysis.tests.test_extensions import (PILOT_DIRS, REAL_LIST, REAL_READ, ROOT, SECTION, FakeLogs, _patterns,
                                            core_sample, meta_of, placeholders)

KW = dict(n_boot=199, n_perm=999, seed=11)
FOLLOW = {"CHAT-MON-SHR": 0.5, "CHAT-MON-ISO": 0.5}      # every other arm follows at 0.1
MODELS = ("glm-5.3", "deepseek-v4-pro", "gpt-6.1-sol")    # deep-grid labels: per-model keys are fixed per cohort
SEEDS = tuple(range(1000, 1008))


def follow_of(arm: str) -> float:
    return FOLLOW.get(arm, 0.1)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sig")
    mp = pytest.MonkeyPatch()
    fake = FakeLogs()
    fake.install(mp)
    rng = np.random.default_rng(99)
    d = tmp / "logs"
    d.mkdir()
    samples = [core_sample(meta_of("deep", m, arm, s, 40), rng, follow=follow_of(arm))
               for m in MODELS for s in SEEDS for arm in S.SIGNAL_ARMS]
    samples.append(core_sample(meta_of("deep", "glm-5.3", "PROVOC", 1000, 40), rng, follow=0.1))   # not compared
    rep = [core_sample(meta_of("ext-rep", "glm-5.3", arm, 2000, 40), rng, follow=follow_of(arm))
           for arm in S.SIGNAL_ARMS]
    fake.add(d, "a.eval", samples + rep)
    res = S.run(S.Config(log_dirs=[str(d)], out=str(tmp / "out"), **KW))
    yield SimpleNamespace(res=res, tmp=tmp, dir=d, fake=fake, rng=rng)
    mp.undo()


def _chain(follow: float, rounds: int, rng, noise: float = 0.0) -> pd.DataFrame:
    """Two firms whose benchmark deviations follow dev_j,t = 0.3 dev_j,t-1 + follow dev_i,t-1 + 0.5 eta_j,t + e."""
    dev, rows = {"A": 0.0, "B": 0.0}, []
    for t in range(1, rounds + 1):
        eta = {f: float(rng.normal()) for f in "AB"}
        new = {f: 0.2 + 0.3 * dev[f] + follow * dev["B" if f == "A" else "A"] + 0.5 * eta[f]
               + float(rng.normal(0, noise)) for f in "AB"}
        for f in "AB":
            bench = 14.7 + float(rng.normal())
            rows.append(dict(game_id="g", model="m", scenario="core-pricing", arm="CHAT-ISO", seed=1000, t=t, firm=f,
                             action=bench + new[f], bench_competitive=bench, own_priv=eta[f], scripted=False,
                             fallback_action=False))
        dev = new
    return pd.DataFrame(rows)


def test_follow_regression_exact_without_noise():
    df = _chain(0.6, 30, np.random.default_rng(1))
    d = S.follow_by_direction(df)
    assert set(d["leader"]) == {"A", "B"} and (d["n"] == 29).all()
    assert np.allclose(d["follow"], 0.6, atol=1e-8)
    g = S.follow_by_game(df)
    assert g["follow"].iloc[0] == pytest.approx(0.6) and g["n_directions"].iloc[0] == 2


def test_fallback_and_missing_benchmark_rows_are_dropped():
    df = _chain(0.4, 30, np.random.default_rng(2))
    df.loc[(df["t"] == 10) & (df["firm"] == "A"), "fallback_action"] = True
    df.loc[(df["t"] == 20) & (df["firm"] == "B"), "bench_competitive"] = np.nan
    d = S.follow_by_direction(df).set_index("follower")
    assert d.loc["A", "n"] == 27            # its fallback row (t = 10) and t = 21 (no leader lag: B lacks t = 20)
    assert d.loc["B", "n"] == 27            # t = 20 (no dev), t = 21 (no own lag); A's t = 21 lacks the leader lag
    assert np.allclose(d["follow"], 0.4, atol=1e-8)
    short = S.follow_by_game(_chain(0.4, 4, np.random.default_rng(3)))
    assert np.isnan(short["follow"].iloc[0]) and short["n_directions"].iloc[0] == 0   # 3 rows, 4 coefficients


def test_planted_monitor_contrast_recovered(world):
    r = world.res
    assert r["signal.label"].startswith("exploratory") and r["signal.meta.cohort"] == "confirmatory"
    assert r["signal.meta.n_games"] == len(MODELS) * len(SEEDS) * 6 and r["signal.meta.n_core_games_all_arms"] == 145
    assert r["signal.meta.n_rows_missing_benchmark"] == 0
    assert abs(r["signal.mon_vs_chat.diff"] - 0.4) < 0.12 and r["signal.mon_vs_chat.ci_lo"] > 0
    assert r["signal.mon_vs_chat.n_pairs"] == 24 and isinstance(r["signal.mon_vs_chat.p_seedblock"], float)
    assert abs(r["signal.mon_vs_aware.diff"] - 0.4) < 0.12
    assert r["signal.chat_vs_aware.ci_lo"] < 0 < r["signal.chat_vs_aware.ci_hi"]
    assert abs(r["signal.pair.CHAT-MON-ISO__CHAT-ISO.diff"] - 0.4) < 0.15
    assert abs(r["signal.arm.CHAT-MON-SHR.all.mean"] - 0.5) < 0.12
    assert r["signal.arm.ISO-AWARE.deepseek-v4-pro.n_games"] == 8
    assert r["signal.arm.ISO-AWARE.opus-5_5.mean"] == "pending"          # a design model with no game: pending
    assert not any(".m1." in k for k in r)
    games = pd.read_csv(world.tmp / "out" / "signal_games.csv")
    assert not games["game_id"].str.startswith("ext-").any() and len(games) == 145


def test_benchmark_comes_from_the_kept_copy(tmp_path, world):
    d = tmp_path / "dup"
    d.mkdir()
    rng = np.random.default_rng(4)
    meta = meta_of("deep", "m1", "CHAT-ISO", 1000, 6)
    first = core_sample(meta, rng, follow=0.1)
    later = core_sample(meta, rng, follow=0.1)
    later.completed_at = "2026-10-26T09:00:00+00:00"
    for r in later.metadata["result"]["rows"]:
        r["benchmark"]["competitive"] = {"A": -1.0, "B": -1.0}
    first.metadata["result"]["rows"][0]["benchmark"] = {}          # one row without a saved benchmark
    world.fake.add(d, "x1.eval", [first])
    world.fake.add(d, "x2.eval", [later], created="2026-10-26T08:00:00+00:00")
    loaded = S.load_rounds([str(d)])
    rows = loaded.rounds.sort_values(["t", "firm"])
    want = [r["benchmark"].get("competitive", {}).get(r["firm"], np.nan)
            for r in sorted(first.metadata["result"]["rows"], key=lambda r: (r["t"], r["firm"]))]
    assert np.allclose(rows["bench_competitive"].to_numpy(float), want, equal_nan=True)
    assert loaded.n_missing_benchmark == 1 and loaded.game_set.n_duplicates_dropped == 1
    kept = sorted(first.metadata["result"]["rows"], key=lambda r: (r["t"], r["firm"]))
    assert np.allclose(rows["action"].to_numpy(float), [r["action"] for r in kept])


def test_ext_cohort_reads_ext_rep_only(world, tmp_path):
    r = S.run(S.Config(log_dirs=[str(world.dir)], out=str(tmp_path / "ext"), cohort="ext", **KW))
    assert r["signal.meta.cohort"] == "ext" and r["signal.meta.n_games"] == 6
    assert r["signal.mon_vs_chat.diff"] == "insufficient data"        # one matched key: fewer than 2 units
    assert r["signal.mon_vs_chat.p_seedblock"] == "insufficient data"
    assert r["signal.arm.CHAT-ISO.glm-5_3.n_games"] == 1 and "signal.arm.CHAT-ISO.opus-5_5.mean" not in r
    games = pd.read_csv(tmp_path / "ext" / "signal_games.csv")
    assert games["game_id"].str.startswith("ext-rep:").all()


def test_ext_cohort_is_ext_rep_game_set(world, tmp_path):
    """The ext cohort uses analysis.extensions' loader and design check: same exclusions, same fail-closed rule."""
    d = tmp_path / "e"
    d.mkdir()
    rng = np.random.default_rng(12)
    ok = core_sample(meta_of("ext-rep", "glm-5.3", "CHAT-ISO", 2001, 40), rng, follow=0.1)
    legacy = core_sample(meta_of("ext-rep", "glm-5.3", "CHAT-SHR", 2001, 40), rng, follow=0.1)
    del legacy.metadata["result"]["chat"]
    odd = core_sample(meta_of("ext-rep", "glm-5.3", "CHAT-MON-ISO", 2099, 40), rng, follow=0.1)   # seed off-design
    world.fake.add(d, "e.eval", [ok, legacy, odd])
    with pytest.raises(S.SignallingError, match="outside the declared design"):
        S.run(S.Config(log_dirs=[str(d)], out=str(tmp_path / "o1"), cohort="ext", **KW))
    r = S.run(S.Config(log_dirs=[str(d)], out=str(tmp_path / "o2"), cohort="ext", preliminary=True, **KW))
    games = pd.read_csv(tmp_path / "o2" / "signal_games.csv")
    assert list(games["game_id"]) == [ok.id] and r["signal.mon_vs_chat.diff"] == "pending"


def test_single_seed_cluster_seed_block_is_insufficient():
    games = pd.DataFrame([dict(game_id=f"g{m}{a}", model=m, scenario="core-pricing", arm=a, seed=1000, grid="deep",
                               follow=(0.5 if "MON" in a else 0.1) + 0.01 * i)
                          for i, m in enumerate(MODELS) for a in S.SIGNAL_ARMS])
    res, _ = S.analyse(games, n_boot=99, n_perm=99, seed=0, models=MODELS)
    assert isinstance(res["signal.mon_vs_chat.diff"], float) and res["signal.mon_vs_chat.n_pairs"] == 3
    assert res["signal.mon_vs_chat.p_seedblock"] == "insufficient data"      # one seed: no sign-flip test
    res2, _ = S.analyse(games[~games["arm"].eq("CHAT-MON-ISO")], n_boot=99, n_perm=99, seed=0, models=MODELS)
    assert res2["signal.mon_vs_chat.diff"] == "pending" and res2["signal.mon_vs_chat.p_seedblock"] == "pending"
    assert isinstance(res2["signal.pair.CHAT-MON-SHR__CHAT-SHR.diff"], float)


def test_follow_uses_own_benchmark_per_round_and_firm():
    """dev is action minus that firm's own benchmark in that round: a benchmark shift common to the firm's rows that
    is offset in its action leaves follow unchanged; swapping the benchmarks between firms changes it."""
    df = _chain(0.6, 30, np.random.default_rng(5), noise=0.1)
    base = S.follow_by_direction(df).set_index("follower")["follow"]
    shifted = df.assign(action=df["action"] + 3.0 * (df["firm"] == "A") + 0.1 * df["t"],
                        bench_competitive=df["bench_competitive"] + 3.0 * (df["firm"] == "A") + 0.1 * df["t"])
    assert np.allclose(S.follow_by_direction(shifted).set_index("follower")["follow"], base, atol=1e-10)
    swapped = df.copy()
    swapped["bench_competitive"] = df.groupby("t")["bench_competitive"].transform(lambda s: s[::-1].to_numpy())
    assert not np.allclose(S.follow_by_direction(swapped).set_index("follower")["follow"], base, atol=1e-3)


def test_pilot_cohort_needs_preliminary(world, tmp_path):
    with pytest.raises(S.SignallingError, match="preliminary"):
        S.run(S.Config(log_dirs=[str(world.dir)], out=str(tmp_path / "p"), cohort="pilot"))
    r = S.run(S.Config(log_dirs=[str(world.dir)], out=str(tmp_path / "p"), cohort="pilot", preliminary=True, **KW))
    assert r["signal.meta.n_games"] == 0 and r["signal.mon_vs_chat.diff"] == "pending"


def test_every_written_key_is_documented(world):
    pats = _patterns((ROOT / "analysis" / "EXTENSION_KEYS.md").read_text(), "signal.")
    bad = sorted(k for k in world.res if k != "notes" and not any(p.match(k) for p in pats))
    assert not bad, bad[:10]
    assert (world.tmp / "out" / "tables" / "signal_contrasts.md").read_text().startswith("_Price-following")


@pytest.mark.skipif(not SECTION.exists(), reason="report/REPORT.md not present")
def test_report_placeholders_resolve(world):
    want = placeholders("signal.")
    assert want and not sorted(want - set(world.res)), sorted(want - set(world.res))


@pytest.mark.skipif(not all(p.exists() for p in PILOT_DIRS), reason="pilot logs not present")
def test_real_pilot_logs_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "list_eval_logs", REAL_LIST)
    monkeypatch.setattr(dataset, "read_eval_log", REAL_READ)
    r = S.run(S.Config(log_dirs=[str(p) for p in PILOT_DIRS], out=str(tmp_path / "pilot"), cohort="pilot",
                       preliminary=True, **KW))
    assert r["signal.meta.n_rows"] > 0 and r["signal.meta.n_rows_missing_benchmark"] == 0
    assert r["signal.meta.n_core_games_all_arms"] >= 17
