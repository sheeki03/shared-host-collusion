"""V1-econ-checker: independent verification of analysis/stats.py against analysis/SPEC.md.

Reference implementations here are written from the textbook definitions (explicit OLS, CR1 sandwich, brute-force
wild-cluster restricted bootstrap with every sign vector refitted, Holm by definition, Wilson by formula), not from
stats.py internals. Simulation checks use my own data-generating processes with fixed seeds.

Run: .venv/bin/python -m pytest analysis/tests/test_verify_stats.py -q -s   (-s prints measured sizes/coverage)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import stats as S  # noqa: E402

BAND = (0.03, 0.08)


def say(msg: str) -> None:
    print(f"[verify-stats] {msg}")


# ------------------------------------------------------------------------------------------------ synthetic data
def panel(G: int, T: int, beta: float, rng: np.random.Generator, *, rho_rival: float = 0.3, rho_own: float = 0.0,
          leak_mover: int = 2) -> pd.DataFrame:
    """Firm-round table. action = beta * rival_priv (leak_mover rows only) + 0.5 own_priv + 0.2 own_com_err
    + rho_rival * rival_action_{t-1} + rho_own * own_action_{t-1} + game effect + game-specific-scale noise.
    History feeds back through the lag terms; signals are i.i.d. N(0, 1) per firm-round."""
    recs = []
    for g in range(G):
        effect, scale = rng.normal(0, 1), rng.uniform(0.5, 1.5)
        last = {"A": 0.0, "B": 0.0}
        for t in range(1, T + 1):
            first = "A" if rng.random() < 0.5 else "B"
            priv = {f: rng.normal() for f in "AB"}
            com = {f: rng.normal() for f in "AB"}
            act = {}
            for f in "AB":
                r = "B" if f == "A" else "A"
                order = 1 if f == first else 2
                a = 0.5 * priv[f] + 0.2 * com[f] + rho_rival * last[r] + rho_own * last[f] + effect \
                    + rng.normal(0, scale)
                act[f] = a + (beta * priv[r] if order == leak_mover else 0.0)
            for f in "AB":
                r = "B" if f == "A" else "A"
                recs.append(dict(model="m", scenario="s", arm="X", seed=g, game_id=f"g{g:03d}", t=t, firm=f,
                                 order=1 if f == first else 2, action=act[f], own_priv=priv[f], own_com_err=com[f],
                                 rival_priv=priv[r], rival_com_err=com[r], fallback_action=False))
            last = act
    return pd.DataFrame(recs)


def brute_design(df: pd.DataFrame, order: int = 2, lags: bool = True):
    """Rows of `order`, with own and rival previous-round actions looked up in a dict (own implementation)."""
    act = {(r.game_id, r.t, r.firm): r.action for r in df.itertuples()}
    other = {"A": "B", "B": "A"}
    rows = []
    for r in df.itertuples():
        if r.order != order:
            continue
        own_lag = act.get((r.game_id, r.t - 1, r.firm))
        riv_lag = act.get((r.game_id, r.t - 1, other[r.firm]))
        if lags and (own_lag is None or riv_lag is None):
            continue
        rows.append((r.game_id, r.action, r.rival_priv, r.own_priv, r.own_com_err, own_lag, riv_lag))
    return pd.DataFrame(rows, columns=["game_id", "y", "x", "c1", "c2", "own_lag", "riv_lag"])


def ols_cr1(y: np.ndarray, X: np.ndarray, cl: np.ndarray, j: int = 1) -> tuple[float, float]:
    """Explicit OLS with the CR1 sandwich: G/(G-1) * (N-1)/(N-K) * (X'X)^-1 [sum_g X_g'u_g u_g'X_g] (X'X)^-1."""
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ X.T @ y
    u = y - X @ b
    N, K = X.shape
    groups = pd.unique(cl)
    meat = np.zeros((K, K))
    for g in groups:
        m = cl == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(groups)
    V = G / (G - 1) * (N - 1) / (N - K) * xtx_inv @ meat @ xtx_inv
    return float(b[j]), float(np.sqrt(V[j, j]))


def brute_wcr_p(y, X, cl, beta0: float, j: int = 1) -> float:
    """WCR-t: impose beta = beta0, refit the full model and CR1 on every one of the 2^G Rademacher sign vectors."""
    b, se = ols_cr1(y, X, cl, j)
    t_obs = abs(b - beta0) / se
    Xr = np.delete(X, j, axis=1)
    yr = y - beta0 * X[:, j]
    gam = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    ur = yr - Xr @ gam
    fitted = Xr @ gam + beta0 * X[:, j]
    codes, uniq = pd.factorize(cl)
    hits, total = 0, 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(uniq)):
        v = np.array(signs)[codes]
        bs, ses = ols_cr1(fitted + v * ur, X, cl, j)
        hits += abs(bs - beta0) / ses >= t_obs * (1 - 1e-10)
        total += 1
    return hits / total


# ======================================================================================== 1. pooled_beta, beta_by_game
def test_pooled_beta_matches_explicit_ols_cr1_and_brute_force_wild_bootstrap():
    df = panel(8, 15, 0.3, np.random.default_rng(1))
    d = brute_design(df)
    X = np.column_stack([np.ones(len(d)), d.x, d.c1, d.c2, d.own_lag, d.riv_lag])
    y, cl = d.y.to_numpy(), d.game_id.to_numpy()
    b, se = ols_cr1(y, X, cl)
    res = S.pooled_beta(df, n_boot=2000, seed=3)
    assert res.n == len(d) and res.n_games == 8 and res.n_boot == 256          # 2^8 sign vectors enumerated
    assert res.beta == pytest.approx(b, rel=1e-10)
    assert res.se == pytest.approx(se, rel=1e-10)
    tc = sps.t.ppf(0.975, 7)
    assert res.ci_cr1 == pytest.approx((b - tc * se, b + tc * se), rel=1e-10)
    assert res.p_cr1 == pytest.approx(2 * sps.t.sf(abs(b / se), 7), rel=1e-10)
    assert res.p_boot == pytest.approx(brute_wcr_p(y, X, cl, 0.0), abs=1e-12)
    for end, inside in ((res.ci_boot[0], +1), (res.ci_boot[1], -1)):            # test-inversion endpoints
        eps = 1e-4 * se
        assert brute_wcr_p(y, X, cl, end + inside * eps) >= 0.05
        assert brute_wcr_p(y, X, cl, end - inside * eps) < 0.05


def test_beta_by_game_matches_explicit_per_game_ols():
    df = panel(5, 25, 0.3, np.random.default_rng(2))
    out = S.beta_by_game(df).set_index("game_id")
    d = brute_design(df)
    mx, m1, m2 = d.x.mean(), d.c1.mean(), d.c2.mean()                            # pooled centring (documented)
    for gid, g in d.groupby("game_id"):
        X = np.column_stack([g.x - mx, g.c1 - m1, g.c2 - m2])
        yy = (g.y - g.own_lag).to_numpy()
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        u = yy - X @ coef
        se = np.sqrt(u @ u / (len(g) - 3) * np.linalg.inv(X.T @ X)[0, 0])
        assert out.loc[gid, "beta"] == pytest.approx(coef[0], rel=1e-10)
        assert out.loc[gid, "se"] == pytest.approx(se, rel=1e-10)
        assert out.loc[gid, "n"] == len(g)
    classic = S.beta_by_game(df, baseline="none").set_index("game_id")          # spec's literal form
    d0 = brute_design(df, lags=False)
    for gid, g in d0.groupby("game_id"):
        X = np.column_stack([np.ones(len(g)), g.x, g.c1, g.c2])
        coef, *_ = np.linalg.lstsq(X, g.y.to_numpy(), rcond=None)
        assert classic.loc[gid, "beta"] == pytest.approx(coef[1], rel=1e-10)
    for col in ("beta", "se", "n", "t"):                                        # spec return fields
        assert col in out.columns


def test_planted_beta_coverage_and_first_mover_placebo():
    """Spec: beta_by_game and pooled_beta recover beta = 0.3 within the CI in >= 90% of 200 simulations; the
    first-mover placebo is null when the leak is second-mover only."""
    rng = np.random.default_rng(10)
    cover_cr1 = cover_boot = 0
    game_hits = game_n = 0
    placebo_beta, placebo_rej = [], 0
    n_sims = 200
    for _ in range(n_sims):
        df = panel(30, 20, 0.3, rng)
        r = S.pooled_beta(df, n_boot=999, seed=rng)
        cover_cr1 += r.ci_cr1[0] <= 0.3 <= r.ci_cr1[1]
        cover_boot += r.ci_boot[0] <= 0.3 <= r.ci_boot[1]
        g = S.beta_by_game(df)
        half = sps.t.ppf(0.975, g["df"]) * g["se"]
        game_hits += int(((g["beta"] - half <= 0.3) & (0.3 <= g["beta"] + half)).sum())
        game_n += len(g)
        p = S.pooled_beta(df, mover="first", n_boot=199, seed=rng, ci=False)
        placebo_beta.append(p.beta)
        placebo_rej += p.p_cr1 < 0.05
    say(f"coverage of planted 0.3: pooled CR1 {cover_cr1 / n_sims:.3f}, pooled wild {cover_boot / n_sims:.3f}, "
        f"per game {game_hits / game_n:.3f}; placebo rejection {placebo_rej / n_sims:.3f}, "
        f"mean {np.mean(placebo_beta):+.4f}")
    assert cover_cr1 / n_sims >= 0.90 and cover_boot / n_sims >= 0.90 and game_hits / game_n >= 0.90
    pb = np.array(placebo_beta)
    assert abs(pb.mean()) <= 4 * pb.std(ddof=1) / np.sqrt(n_sims)
    assert placebo_rej / n_sims <= 0.10


def test_null_size_with_feedback_default_estimators():
    """Spec: under the null the test size is within [0.03, 0.08]. Strong history feedback, no leak."""
    rng = np.random.default_rng(20)
    n_sims = 500
    pb, pc, betas, fe = [], [], [], []
    game_rej = game_n = 0
    for i in range(n_sims):
        df = panel(24, 16, 0.0, rng, rho_rival=0.6, rho_own=0.3)
        r = S.pooled_beta(df, n_boot=999, seed=rng, ci=False)
        pb.append(r.p_boot)
        pc.append(r.p_cr1)
        betas.append(r.beta)
        g = S.beta_by_game(df)
        game_rej += int((g["p"] < 0.05).sum())
        game_n += len(g)
        if i < 200:
            fe.append(S.pooled_beta(df, game_fe=True, lag_controls=False, n_boot=99, seed=rng, ci=False).beta)
    size_b, size_c = np.mean(np.array(pb) < 0.05), np.mean(np.array(pc) < 0.05)
    b, f = np.array(betas), np.array(fe)
    say(f"null size: wild {size_b:.3f}, CR1 {size_c:.3f}, per game {game_rej / game_n:.3f}; mean beta {b.mean():+.4f} "
        f"(MC SE {b.std(ddof=1) / np.sqrt(n_sims):.4f}); classic game-FE mean beta {f.mean():+.4f} "
        f"(MC SE {f.std(ddof=1) / np.sqrt(f.size):.4f})")
    assert BAND[0] <= size_b <= BAND[1] and BAND[0] <= size_c <= BAND[1]
    assert BAND[0] <= game_rej / game_n <= BAND[1]
    assert abs(b.mean()) <= 4 * b.std(ddof=1) / np.sqrt(n_sims)


# ======================================================================================== 2. adaptive_null, real engines
def _engines():
    from engines.auction import AuctionEngine
    from engines.bargaining import BargainingEngine
    from engines.market_making import MarketMakingEngine
    from engines.pricing import PricingEngine
    from engines.reporting import ReportingEngine
    return [(PricingEngine(), "core-pricing"), (AuctionEngine(), "G6-ads"), (ReportingEngine(), "G7-oracle"),
            (MarketMakingEngine(), "G9-dealer-spreads"), (BargainingEngine(), "G10-legal-negotiation")]


@pytest.mark.parametrize("idx", range(5))
def test_simulate_null_games_runs_on_every_real_engine(idx):
    eng, sid = _engines()[idx]

    def policy(obs: S.PolicyInput, rng: np.random.Generator) -> float:
        lo, hi = obs.bounds
        return float(rng.uniform(lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)))
    df = S.simulate_null_games(policy, eng, n_games=3, rounds=8, seed=7, scenario_id=sid)
    assert len(df) == 3 * 8 * 2
    per_round = df.groupby(["game_id", "t"])
    assert (per_round["order"].apply(sorted).map(tuple) == (1, 2)).all()
    piv = df.pivot_table(index=["game_id", "t"], columns="firm", values=["own_priv", "rival_priv"])
    assert np.allclose(piv[("own_priv", "A")], piv[("rival_priv", "B")])
    gt = S.game_table(df)                                   # engines without com_err must still give a beta
    assert gt["beta"].notna().all()


def test_adaptive_null_end_to_end_with_pricing_engine():
    from engines.pricing import PricingEngine
    pol = S.noisy_fixed_markup(lambda p: p["market_cost_forecast"] + p["firm_specific_cost"], 0.47, 0.5)
    res = S.adaptive_null(pol, PricingEngine(), n_games=6, rounds=5, seed=3, scenario_id="core-pricing", n_sims=3,
                          n_boot=99)
    assert res.n_sims == 3 and np.all(np.isfinite(res.beta)) and 0 <= res.size_boot <= 1


# ======================================================================================== 3. matched / stratified
def games_frame(diffs: np.ndarray, base: float = 1.0, arm_t: str = "T", arm_c: str = "C") -> pd.DataFrame:
    recs = []
    for i, dlt in enumerate(diffs):
        for arm, val in ((arm_t, base + dlt), (arm_c, base)):
            recs.append(dict(model="m" + str(i % 2), scenario="s" + str(i % 3), seed=i, arm=arm, y=val,
                             game_id=f"{arm}{i}"))
    return pd.DataFrame(recs)


def test_matched_effect_exact_sign_flip_by_hand():
    r = S.matched_effect(games_frame(np.array([1.0, 2.0, 3.0, 4.0])), "y", "T", "C")
    assert r.estimate == pytest.approx(2.5) and r.n_pairs == 4
    assert r.p_signflip == pytest.approx(2 / 16)          # only all-plus and all-minus reach |sum| = 10
    r = S.matched_effect(games_frame(np.array([1.0, -1.0, 2.0, -2.0])), "y", "T", "C")
    assert r.estimate == pytest.approx(0.0) and r.p_signflip == pytest.approx(1.0)


def test_matched_effect_null_uniform_and_planted_coverage():
    rng = np.random.default_rng(30)
    ps, cover = [], 0
    for _ in range(300):
        ps.append(S.matched_effect(games_frame(rng.normal(0, 1, 40)), "y", "T", "C", n_boot=500, n_perm=999,
                                   seed=rng).p_signflip)
    for _ in range(200):
        r = S.matched_effect(games_frame(rng.normal(0.5, 1, 40)), "y", "T", "C", n_boot=2000, n_perm=199, seed=rng)
        cover += r.ci[0] <= 0.5 <= r.ci[1]
    ks = sps.kstest(ps, "uniform").pvalue
    say(f"matched_effect: KS p of null sign-flip p-values {ks:.3f}; coverage of planted 0.5 {cover / 200:.3f}")
    assert ks > 0.01 and cover / 200 >= 0.90


def test_stratified_and_welfare_effects_by_hand():
    rng = np.random.default_rng(40)
    diffs = rng.normal(0.3, 1.0, 60)
    df = games_frame(diffs)
    units = df.pivot_table(index=["model", "scenario", "seed"], columns="arm", values="y").reset_index()
    units["d"] = units["T"] - units["C"]
    g = units.groupby(["model", "scenario"])["d"]
    m, v, n = g.mean(), g.var(ddof=1), g.size()
    eq = S.stratified_effect(df, "y", "T", "C", seed=1)
    assert eq.estimate == pytest.approx(m.mean(), rel=1e-12)
    assert eq.se == pytest.approx(np.sqrt(np.sum(v / n)) / len(m), rel=1e-12)
    iv = S.stratified_effect(df, "y", "T", "C", weights="ivw", seed=1)
    w = (n / v) / (n / v).sum()
    assert iv.estimate == pytest.approx(float((w * m).sum()), rel=1e-12)
    df["transfer_observed"] = df["seed"] % 2 == 0
    we = S.welfare_effect(df, outcome="y", arm_t="T", arm_c="C", seed=2)
    assert we.primary.estimate == pytest.approx(diffs.mean(), rel=1e-12) and we.primary.n_pairs == 60
    bt = we.by_transfer.set_index("transfer")
    assert bt.loc[True, "mean"] == pytest.approx(diffs[::2].mean(), rel=1e-12)
    assert bt.loc[False, "mean"] == pytest.approx(diffs[1::2].mean(), rel=1e-12)


# ======================================================================================== 4. displacement
def quads(rng, chat_drop: float, storage_rise: float, n: int = 40) -> pd.DataFrame:
    recs = []
    for i in range(n):
        for arm, mon, shr in (("CHAT-ISO", 0, 0), ("CHAT-SHR", 0, 1), ("CHAT-MON-ISO", 1, 0), ("CHAT-MON-SHR", 1, 1)):
            recs.append(dict(model="m", scenario="s", seed=i, arm=arm, game_id=f"{arm}-{i}",
                             chat_coordination=1.0 - chat_drop * mon + rng.normal(0, 0.5),
                             beta=0.1 + storage_rise * mon * shr + rng.normal(0, 0.5),
                             addressed_write_rate=0.2 + storage_rise * mon * shr + rng.normal(0, 0.5)))
    return pd.DataFrame(recs)


def test_displacement_detects_both_parts_and_needs_both():
    """A claim needs part (i) (one-sided, nominal 2.5%) and part (ii); with one part absent the claim rate should
    be near or below 2.5%. 100 simulations per case; the bound 0.08 allows for Monte Carlo noise."""
    rng = np.random.default_rng(50)
    n = 100
    hits = {"both": 0, "only_i": 0, "only_ii": 0}
    for _ in range(n):
        hits["both"] += S.displacement_test(quads(rng, 1.0, 1.0), n_boot=499, n_perm=199, seed=rng).displacement
        r_i = S.displacement_test(quads(rng, 1.0, 0.0), n_boot=499, n_perm=199, seed=rng)
        r_ii = S.displacement_test(quads(rng, 0.0, 1.0), n_boot=499, n_perm=199, seed=rng)
        hits["only_i"] += r_i.displacement
        hits["only_ii"] += r_ii.displacement
        assert r_i.part_i and r_ii.part_ii
    say(f"displacement claims out of {n}: {hits}")
    assert hits["both"] >= 0.95 * n and hits["only_i"] <= 0.08 * n and hits["only_ii"] <= 0.08 * n


# ======================================================================================== 5. rates, multiplicity, power
def holm_by_definition(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def test_holm_known_values_and_definition():
    assert S.holm([0.01, 0.02, 0.03, 0.04, 0.05]) == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    rng = np.random.default_rng(60)
    for _ in range(50):
        p = np.round(rng.uniform(0, 0.3, rng.integers(1, 12)), 3)            # rounding makes ties
        assert S.holm(p) == pytest.approx(holm_by_definition(p))
    with pytest.raises(ValueError):
        S.holm([0.1, float("nan")])


def test_wilson_rule_of_three_fisher():
    # Newcombe (1998), Table I: 81/263 -> 0.2553-0.3662; 15/148 -> 0.0624-0.1605; 0/20 -> 0-0.1611
    for (k, n), (lo, hi) in (((81, 263), (0.2553, 0.3662)), ((15, 148), (0.0624, 0.1605)), ((0, 20), (0.0, 0.1611))):
        assert S.wilson_ci(k, n) == pytest.approx((lo, hi), abs=6e-5)
    z = sps.norm.ppf(0.975)
    for k, n in ((3, 10), (50, 50), (7, 1000)):
        ph = k / n
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        assert S.wilson_ci(k, n) == pytest.approx((max(0, c - h), min(1, c + h)), rel=1e-12)
    for n in (30, 100, 1000):
        exact = 1 - 0.05 ** (1 / n)
        assert exact <= S.rule_of_three_upper(n) <= 1.1 * exact
    p = [0.01, 0.2, 0.5, 0.03]
    x, pv = S.fisher_combine(p)
    ref = sps.combine_pvalues(p, method="fisher")
    assert x == pytest.approx(ref.statistic, rel=1e-12) and pv == pytest.approx(ref.pvalue, rel=1e-12)


def test_power_mde_normal_formula_and_t_version_by_simulation():
    zsum = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)
    assert S.power_mde(100, 1.0) == pytest.approx(zsum * np.sqrt(2 / 100), rel=1e-12)
    assert S.power_mde(25, 2.0, paired=True) == pytest.approx(zsum * 2.0 / 5.0, rel=1e-12)
    rng = np.random.default_rng(70)
    n, reps = 8, 40_000
    mde = S.power_mde(n, 1.0, method="t")
    a, b = rng.normal(0, 1, (reps, n)), rng.normal(mde, 1, (reps, n))
    power = np.mean(sps.ttest_ind(b, a, axis=1).pvalue < 0.05)
    mde_p = S.power_mde(10, 1.0, method="t", paired=True)
    power_p = np.mean(sps.ttest_1samp(rng.normal(mde_p, 1, (reps, 10)), 0.0, axis=1).pvalue < 0.05)
    say(f"power at the t-based MDE by simulation: two-sample {power:.4f}, paired {power_p:.4f} (target 0.80)")
    assert abs(power - 0.8) < 0.01 and abs(power_p - 0.8) < 0.01


# ======================================================================================== 6. game_table
def test_game_table_rates_and_scripted_exclusion():
    df = panel(2, 6, 0.0, np.random.default_rng(80))
    df["addressed_write"] = (df["t"] == 2) & (df["firm"] == "A")
    df["crossing_canary"] = (df["game_id"] == "g000") & (df["t"] == 3)
    df["crossing_content"] = False
    df["scripted"] = df["firm"] == "B"
    df.loc[df["scripted"], "addressed_write"] = True             # a scripted rival's writes are not counted
    gt = S.game_table(df).set_index("game_id")
    assert gt.loc["g000", "addressed_write_rate"] == pytest.approx(1 / 6)
    assert bool(gt.loc["g000", "transfer_observed"]) and not bool(gt.loc["g001", "transfer_observed"])
    assert (gt["n_rounds"] == 6).all()
