"""Synthetic-data tests for analysis/stats.py (spec: analysis/SPEC.md, section "Tests").

Every simulation uses a fixed seed, so the suite is deterministic. Simulation counts are chosen so that a correctly
sized test lands inside the [0.03, 0.08] size band with high probability (about 99.8% for 1,000 sims, 98% for 500).
Run with `-s` to print the measured sizes and coverage rates.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import stats as S  # noqa: E402

SIZE_BAND = (0.03, 0.08)


def report(name: str, value: float, extra: str = "") -> None:
    print(f"[stats-report] {name} = {value:.4f} {extra}")


def assert_mean_near(values, target: float, k: float = 4.0) -> None:
    """Monte Carlo check: the mean is within k standard errors of the target."""
    v = np.asarray(values, float)
    assert abs(v.mean() - target) <= k * v.std(ddof=1) / np.sqrt(v.size), (v.mean(), target)


# ----------------------------------------------------------------------------------------------------------------------
# Synthetic firm-round data with a planted second-mover leak
# ----------------------------------------------------------------------------------------------------------------------

def synth_rounds(rng: np.random.Generator, n_games: int = 30, rounds: int = 20, beta2: float = 0.3,
                 beta1: float = 0.0, fallback_rate: float = 0.03) -> pd.DataFrame:
    """Firm-round table: action = levels + own-signal terms + beta * rival_priv (by mover) + AR(1) noise.

    Stresses: game and firm level shifts, heteroskedasticity across games, AR(1) errors, a common round shock shared
    by both firms, and fallback rows (action replaced by last round's action, which carries no current leak).
    """
    G, T = n_games, rounds
    priv, com = rng.normal(size=(G, T, 2)), rng.normal(size=(G, T, 2))
    first = rng.integers(0, 2, size=(G, T))
    order = np.where(np.arange(2)[None, None, :] == first[..., None], 1, 2)
    rival_priv, rival_com = priv[..., ::-1], com[..., ::-1]
    level = rng.normal(0, 2.0, size=(G, 1, 1)) + rng.normal(0, 0.5, size=(G, 1, 2))
    scale = rng.uniform(0.5, 1.5, size=(G, 1, 1))
    e = rng.normal(size=(G, T, 2))
    u = np.empty_like(e)
    u[:, 0] = e[:, 0]
    for t in range(1, T):
        u[:, t] = 0.6 * u[:, t - 1] + e[:, t]
    common = rng.normal(0, 0.5, size=(G, T, 1))
    action = (level + 0.8 * priv + 0.5 * com + beta2 * rival_priv * (order == 2) + beta1 * rival_priv * (order == 1)
              + scale * (u + common))
    fb = rng.random((G, T, 2)) < fallback_rate
    fb[:, 0] = False
    for t in range(1, T):
        action[:, t] = np.where(fb[:, t], action[:, t - 1], action[:, t])
    g_idx, t_idx, f_idx = np.meshgrid(np.arange(G), np.arange(T), np.arange(2), indexing="ij")
    n = G * T * 2
    return pd.DataFrame(dict(
        model="m", scenario="s", arm="SHR", seed=g_idx.ravel(), game_id=g_idx.ravel(), t=t_idx.ravel() + 1,
        firm=np.array(S.FIRMS)[f_idx.ravel()], order=order.ravel(), action=action.ravel(), own_priv=priv.ravel(),
        own_com_err=com.ravel(), rival_priv=rival_priv.ravel(), rival_com_err=rival_com.ravel(),
        collusion_index=np.zeros(n), welfare=np.zeros(n), crossing_canary=False, crossing_content=False,
        name_exposure=False, addressed_write=False, parse_failed=fb.ravel(), fallback_action=fb.ravel()))


# ----------------------------------------------------------------------------------------------------------------------
# 1-2. beta_by_game and pooled_beta
# ----------------------------------------------------------------------------------------------------------------------

def test_planted_beta_recovered_within_ci():
    rng = np.random.default_rng(20261025)
    beta_true, n_sims = 0.3, 200
    game_cover, game0_cover, boot_cover, cr1_cover, est = [], [], [], [], []
    for s in range(n_sims):
        df = synth_rounds(rng, beta2=beta_true)
        bg = S.beta_by_game(df)
        half = sps.t.ppf(0.975, bg["df"]) * bg["se"]
        covered = (np.abs(bg["beta"] - beta_true) <= half).to_numpy()
        game_cover.extend(covered)
        game0_cover.append(covered[0])
        pb = S.pooled_beta(df, seed=s)
        boot_cover.append(pb.ci_boot[0] <= beta_true <= pb.ci_boot[1])
        cr1_cover.append(pb.ci_cr1[0] <= beta_true <= pb.ci_cr1[1])
        est.append(pb.beta)
    report("beta_by_game per-game 95% CI coverage (200 sims x 30 games)", np.mean(game_cover))
    report("beta_by_game game-0 95% CI coverage (200 sims)", np.mean(game0_cover))
    report("pooled_beta wild-bootstrap 95% CI coverage (200 sims)", np.mean(boot_cover))
    report("pooled_beta CR1 95% CI coverage (200 sims)", np.mean(cr1_cover))
    report("pooled_beta mean estimate (true 0.3)", np.mean(est))
    assert np.mean(game_cover) >= 0.90
    assert np.mean(game0_cover) >= 0.90
    assert np.mean(boot_cover) >= 0.90
    assert np.mean(cr1_cover) >= 0.90
    assert_mean_near(est, beta_true)


def test_null_size_within_band():
    rng = np.random.default_rng(7)
    n_sims = 1000
    p_boot, p_cr1, p_game = [], [], []
    for s in range(n_sims):
        df = synth_rounds(rng, beta2=0.0)
        pb = S.pooled_beta(df, seed=s, ci=False)
        p_boot.append(pb.p_boot)
        p_cr1.append(pb.p_cr1)
        if s < 200:
            p_game.extend(S.beta_by_game(df)["p"].to_numpy())
    size_boot, size_cr1, size_game = (float(np.mean(np.array(p) < 0.05)) for p in (p_boot, p_cr1, p_game))
    report("null size pooled wild bootstrap (1000 sims)", size_boot)
    report("null size pooled CR1 t(G-1) (1000 sims)", size_cr1)
    report("null size per-game OLS t (200 sims x 30 games)", size_game)
    for size in (size_boot, size_cr1, size_game):
        assert SIZE_BAND[0] <= size <= SIZE_BAND[1]


def test_first_mover_placebo_is_null():
    rng = np.random.default_rng(11)
    n_sims = 500
    p_first, b_first, p_second = [], [], []
    for s in range(n_sims):
        df = synth_rounds(rng, beta2=0.3, beta1=0.0)
        first = S.pooled_beta(df, mover="first", seed=s, ci=False)
        second = S.pooled_beta(df, mover="second", seed=s, ci=False)
        p_first.append(first.p_boot)
        b_first.append(first.beta)
        p_second.append(second.p_boot)
    size_first = float(np.mean(np.array(p_first) < 0.05))
    power_second = float(np.mean(np.array(p_second) < 0.05))
    report("placebo (first mover) rejection rate, leak second-mover only (500 sims)", size_first)
    report("placebo mean first-mover beta", float(np.mean(b_first)))
    report("second-mover power at beta=0.3, 30 games x 20 rounds", power_second)
    assert SIZE_BAND[0] <= size_first <= SIZE_BAND[1]
    assert_mean_near(b_first, 0.0)
    assert power_second >= 0.95


def test_pooled_beta_matches_brute_force_ols_and_bootstrap():
    """Fast cluster-sum algebra equals explicit OLS, the CR1 sandwich and a refitted wild bootstrap (both designs)."""
    rng = np.random.default_rng(3)
    df = synth_rounds(rng, n_games=12, rounds=10, beta2=0.2, fallback_rate=0.0)
    for game_fe in (False, True):
        y, x, C, fe, codes = S._pooled_design(df, "priv", "second", ("own_priv", "own_com_err"), True, True, (),
                                              game_fe)
        dummies = pd.get_dummies(fe).to_numpy(float) if game_fe else np.zeros((len(y), 0))
        X = np.column_stack([x, C, dummies])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        G, N, K = len(np.unique(codes)), len(y), 1 + C.shape[1]
        c = G / (G - 1) * (N - 1) / (N - K)
        bread = np.linalg.pinv(X.T @ X)
        assert S.pooled_beta(df, game_fe=game_fe, ci=False).beta == pytest.approx(coef[0], rel=1e-9)

        def cr1_se(Xm: np.ndarray, resid: np.ndarray) -> float:
            meat = sum(np.outer(Xm[codes == g].T @ resid[codes == g], Xm[codes == g].T @ resid[codes == g])
                       for g in np.unique(codes))
            return float(np.sqrt(c * (bread @ meat @ bread)[0, 0]))

        se_bf = cr1_se(X, y - X @ coef)
        B = 199
        res = S._pooled_from_arrays(y, x, C, fe, codes, B, 0.05, np.random.default_rng(99), ci=False)
        assert res.beta == pytest.approx(coef[0], rel=1e-9)
        assert res.se == pytest.approx(se_bf, rel=1e-9)
        # Refit the restricted wild bootstrap explicitly, with the same Rademacher draws.
        V = S._rademacher(B, G, np.random.default_rng(99))
        Z = np.column_stack([C, dummies])
        gam = np.linalg.lstsq(Z, y, rcond=None)[0]           # restricted fit under beta = 0
        fit_r, u_r = Z @ gam, y - Z @ gam
        gidx = pd.factorize(codes)[0]
        t_star = []
        for b in range(B):
            ys = fit_r + V[b, gidx] * u_r
            cb = np.linalg.lstsq(X, ys, rcond=None)[0]
            t_star.append(abs(cb[0]) / cr1_se(X, ys - X @ cb))
        p_bf = float(np.mean(np.array(t_star) >= abs(res.t) * (1 - 1e-10)))
        assert res.p_boot == pytest.approx(p_bf, abs=1e-12)


def test_bootstrap_ci_endpoints_and_enumeration():
    rng = np.random.default_rng(5)
    df = synth_rounds(rng, n_games=10, rounds=16, beta2=0.3)
    res = S.pooled_beta(df, n_boot=2000)
    assert res.n_boot == 2 ** 10                       # all sign vectors enumerated for G = 10
    assert res.ci_boot[0] < res.beta < res.ci_boot[1]
    df40 = synth_rounds(rng, n_games=40, rounds=16, beta2=0.3)
    assert S.pooled_beta(df40, n_boot=2000, ci=False).n_boot == 2000


def test_drop_fallback_and_errors():
    rng = np.random.default_rng(8)
    df = synth_rounds(rng, fallback_rate=0.2)
    n_kept = S.pooled_beta(df, ci=False).n
    n_all = S.pooled_beta(df, drop_fallback=False, ci=False).n
    assert n_kept == int(((df["order"] == 2) & ~df["fallback_action"] & (df["t"] > 1)).sum()) < n_all
    assert S.pooled_beta(df, lag_controls=False, ci=False).n == int(((df["order"] == 2) & ~df["fallback_action"]).sum())
    lagged = S.add_lags(df).set_index(["game_id", "t", "firm"])
    assert lagged.loc[(0, 5, "A"), "own_lag_action"] == lagged.loc[(0, 4, "A"), "action"]
    assert lagged.loc[(0, 5, "A"), "rival_lag_action"] == lagged.loc[(0, 4, "B"), "action"]
    with pytest.raises(ValueError):
        S.beta_by_game(df, mover="both")
    with pytest.raises(KeyError):
        S.pooled_beta(df, controls=("not_a_column",))


# ----------------------------------------------------------------------------------------------------------------------
# 3. adaptive_null with a self-contained toy pricing engine (engines/* deliberately not imported)
# ----------------------------------------------------------------------------------------------------------------------

def _tn(rng: np.random.Generator, sd: float, k: float = 2.5) -> float:
    while True:
        x = rng.normal(0.0, sd)
        if abs(x) <= k * sd:
            return float(x)


@dataclass
class ToyDraw:
    t: int
    common: dict
    private: dict
    signals: dict
    extra: dict = field(default_factory=dict)


@dataclass
class ToyResult:
    payoff: dict
    public: dict
    private_feedback: dict
    welfare: dict
    collusion_index: float | None
    benchmark: dict
    history_row: dict


class ToyPricing:
    """Logit Bertrand duopoly with cost k_i = alpha * (1 + d + eta_i), mirroring engines/SPEC.md E1 (core framing).

    Firm i sees eta_i exactly and a forecast s_i = d + e_i of the common shock d. signals[i] = {priv: eta_i,
    com_err: e_i}, fresh each round, independent across firms and of everything public.
    """
    engine_id = "toy-pricing"
    ALPHA, A, A0, MU, BETA = 10.0, 2.0, 0.0, 0.25, 100.0

    def __init__(self) -> None:
        self.framings = {"toy-core": {"scenario_id": "toy-core", "action_name": "price"}}

    def shares(self, p_own: np.ndarray, p_rival: float) -> np.ndarray:
        e_own = np.exp((self.A - np.asarray(p_own) / self.ALPHA) / self.MU)
        e_riv = np.exp((self.A - p_rival / self.ALPHA) / self.MU)
        return self.BETA * e_own / (e_own + e_riv + np.exp(self.A0 / self.MU))

    def draw_round(self, rng: np.random.Generator, t: int, scenario_id: str) -> ToyDraw:
        d = _tn(rng, 0.20)
        eta = {f: _tn(rng, 0.15) for f in S.FIRMS}
        err = {f: _tn(rng, 0.20) for f in S.FIRMS}
        private = {f: {"firm_specific_cost": self.ALPHA * eta[f], "market_cost_forecast": self.ALPHA * (1 + d + err[f]),
                       "forecast_typical_error": self.ALPHA * 0.20} for f in S.FIRMS}
        signals = {f: {"priv": eta[f], "com_err": err[f]} for f in S.FIRMS}
        return ToyDraw(t=t, common={"d": d}, private=private, signals=signals)

    def action_bounds(self, scenario_id: str, draw: ToyDraw, firm: str) -> tuple[float, float]:
        return (5.0, 40.0)

    def resolve(self, scenario_id: str, draw: ToyDraw, actions: dict) -> ToyResult:
        for f in S.FIRMS:
            if not (np.isfinite(actions[f]) and 5.0 <= actions[f] <= 40.0):
                raise ValueError("action out of bounds")
        pa, pb = actions["A"], actions["B"]
        q = {"A": float(self.shares(pa, pb)), "B": float(self.shares(pb, pa))}
        cost = {f: self.ALPHA * (1 + draw.common["d"] + draw.signals[f]["priv"]) for f in S.FIRMS}
        profit = {f: (actions[f] - cost[f]) * q[f] for f in S.FIRMS}
        incl = np.log(sum(np.exp((self.A - actions[f] / self.ALPHA) / self.MU) for f in S.FIRMS)
                      + np.exp(self.A0 / self.MU))
        return ToyResult(payoff=profit, public={"prices": dict(actions), "cost_shock": draw.common["d"]},
                         private_feedback={f: {"units": q[f], "profit": profit[f]} for f in S.FIRMS},
                         welfare={"consumer_surplus": float(self.BETA * self.ALPHA * self.MU * incl)},
                         collusion_index=None, benchmark={}, history_row={})

    def oracle_fields(self, scenario_id: str, draw: ToyDraw, firm: str) -> dict:
        return {}

    def scripted_rival(self, scenario_id: str, draw: ToyDraw, rival: str, mode: str) -> dict:
        raise NotImplementedError


TOY = ToyPricing()


def toy_cost(private: dict) -> float:
    return float(private["market_cost_forecast"] + private["firm_specific_cost"])


def toy_payoff(private: dict, prices: np.ndarray, rival_price: float) -> np.ndarray:
    return (prices - toy_cost(private)) * TOY.shares(prices, rival_price)


POLICIES = {
    "myopic_best_response": S.myopic_best_response(toy_payoff, noise_sd=0.5),
    "noisy_fixed_markup": S.noisy_fixed_markup(toy_cost, markup=0.45, noise_sd=0.5),
}


def test_simulate_null_games_shape_and_determinism():
    pol = POLICIES["myopic_best_response"]
    a = S.simulate_null_games(pol, TOY, n_games=3, rounds=5, seed=1)
    b = S.simulate_null_games(pol, TOY, n_games=3, rounds=5, seed=1)
    c = S.simulate_null_games(pol, TOY, n_games=3, rounds=5, seed=2)
    assert len(a) == 3 * 5 * 2
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["action"], c["action"])
    per_round = a.groupby(["game_id", "t"])["order"].apply(sorted)
    assert all(v == [1, 2] for v in per_round)
    # The best-response policy really adapts: own price moves with the rival's previous price.
    long = S.simulate_null_games(pol, TOY, n_games=20, rounds=20, seed=3).sort_values(["game_id", "firm", "t"])
    rival_prev = long.groupby(["game_id", "t"])["action"].transform("sum") - long["action"]
    long = long.assign(rival_prev=rival_prev.groupby([long["game_id"], long["firm"]]).shift(1)).dropna(
        subset=["rival_prev"])
    assert np.corrcoef(long["action"], long["rival_prev"])[0, 1] > 0.1


@pytest.mark.parametrize("name", list(POLICIES))
def test_adaptive_null_calibration(name):
    res = S.adaptive_null(POLICIES[name], TOY, n_games=24, rounds=16, seed=2026, n_sims=500)
    report(f"adaptive_null size, wild bootstrap, {name} (500 sims, 24 games x 16 rounds)", res.size_boot,
           f"Wilson CI {res.size_boot_ci[0]:.3f}-{res.size_boot_ci[1]:.3f}")
    report(f"adaptive_null size, CR1, {name}", res.size_cr1)
    report(f"adaptive_null mean beta, {name}", float(np.mean(res.beta)), f"(sd {np.std(res.beta):.3f})")
    assert SIZE_BAND[0] <= res.size_boot <= SIZE_BAND[1]
    assert SIZE_BAND[0] <= res.size_cr1 <= SIZE_BAND[1]
    assert_mean_near(res.beta, 0.0)                    # no spurious beta from adaptive histories


def synth_feedback_null(rng: np.random.Generator, G: int = 40, T: int = 8) -> pd.DataFrame:
    """Strong-feedback null (vectorised): firm A copies B's last price, B prices at a fixed markup on its own cost
    estimate, alpha * 1.45 * (1 + d + e_B + eta_B). No leak, but A's round-t action loads on B's round t-1 signal."""
    eta, err, d = rng.normal(0, .15, (G, T, 2)), rng.normal(0, .2, (G, T, 2)), rng.normal(0, .2, (G, T))
    p_b = 14.5 * (1 + d + err[..., 1] + eta[..., 1]) + rng.normal(0, .5, (G, T))
    p_a = np.empty((G, T))
    p_a[:, 0] = 22.5 + rng.normal(0, .5, G)
    p_a[:, 1:] = p_b[:, :-1] + rng.normal(0, .5, (G, T - 1))
    first = rng.integers(0, 2, (G, T))
    order = np.where(np.arange(2)[None, None] == first[..., None], 1, 2)
    gi, ti, fi = np.meshgrid(np.arange(G), np.arange(T), np.arange(2), indexing="ij")
    return pd.DataFrame(dict(game_id=gi.ravel(), t=ti.ravel() + 1, firm=np.array(S.FIRMS)[fi.ravel()],
                             order=order.ravel(), action=np.stack([p_a, p_b], -1).ravel(), own_priv=eta.ravel(),
                             own_com_err=err.ravel(), rival_priv=eta[..., ::-1].ravel(),
                             rival_com_err=err[..., ::-1].ravel(), fallback_action=False))


def test_within_game_demeaning_is_biased_under_feedback():
    """Why the defaults avoid game fixed effects: under feedback, within-game demeaning pushes beta below zero
    (Nickell bias, about -rho/T per game), while the default estimators stay centred on zero on the same data.
    (Offline, 3,000 sims of this DGP: FE mean -0.49 at z = -15; default -0.001 at z = -0.0.)"""
    rng = np.random.default_rng(41)
    fe_beta, default_beta, per_game_lag, per_game_classic = [], [], [], []
    for _ in range(300):
        data = synth_feedback_null(rng)
        fe_beta.append(S.pooled_beta(data, game_fe=True, lag_controls=False, n_boot=9, ci=False).beta)
        default_beta.append(S.pooled_beta(data, n_boot=9, ci=False).beta)
        per_game_lag.append(S.beta_by_game(data, controls=())["beta"].mean())
        per_game_classic.append(S.beta_by_game(data, controls=(), baseline="none")["beta"].mean())
    se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))  # noqa: E731
    for name, v in (("pooled game-FE", fe_beta), ("pooled default (lag controls)", default_beta),
                    ("per-game lag baseline", per_game_lag), ("per-game classic intercept", per_game_classic)):
        report(f"strong-feedback null, mean beta, {name} (300 sims)", float(np.mean(v)),
               f"({np.mean(v) / se(v):+.1f} SE)")
    assert np.mean(fe_beta) < -3 * se(fe_beta)
    assert_mean_near(default_beta, 0.0)
    assert_mean_near(per_game_lag, 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# 4-5. matched_effect and stratified_effect
# ----------------------------------------------------------------------------------------------------------------------

def synth_games(rng: np.random.Generator, effect: float = 0.0, n_seeds: int = 10, models=("m1", "m2"),
                scenarios=("s1", "s2"), arms=("ISO", "SHR"), stratum_effects: dict | None = None,
                stratum_sd: dict | None = None) -> pd.DataFrame:
    """Game table: one game per (model, scenario, seed, arm); outcome = key effect + arm effect + t(5) noise."""
    rows = []
    for m in models:
        for sc in scenarios:
            eff = (stratum_effects or {}).get((m, sc), effect)
            sd = (stratum_sd or {}).get((m, sc), 1.0)
            for seed in range(n_seeds):
                key = rng.normal(0, 1.0)
                for arm in arms:
                    y = key + (eff if arm == arms[1] else 0.0) + sd * rng.standard_t(5) / np.sqrt(5 / 3)
                    rows.append(dict(model=m, scenario=sc, seed=seed, arm=arm, game_id=f"{m}-{sc}-{seed}-{arm}",
                                     y=y))
    return pd.DataFrame(rows)


def test_matched_effect_recovers_planted_difference():
    rng = np.random.default_rng(13)
    cover, est = [], []
    for s in range(200):
        res = S.matched_effect(synth_games(rng, effect=0.5), "y", "SHR", "ISO", seed=s)
        assert res.n_pairs == 40
        cover.append(res.ci[0] <= 0.5 <= res.ci[1])
        est.append(res.estimate)
    report("matched_effect 95% bootstrap CI coverage of planted 0.5 (200 sims, 40 pairs)", np.mean(cover))
    report("matched_effect mean estimate (true 0.5)", float(np.mean(est)))
    assert np.mean(cover) >= 0.90
    assert_mean_near(est, 0.5)


def test_matched_signflip_uniform_under_null():
    rng = np.random.default_rng(17)
    p = np.array([S.matched_effect(synth_games(rng, effect=0.0), "y", "SHR", "ISO", n_boot=200, n_perm=4000,
                                   seed=s).p_signflip for s in range(300)])
    ks = sps.kstest(p, "uniform").pvalue
    report("matched_effect sign-flip p uniformity, KS p (300 null sims)", ks)
    report("matched_effect sign-flip size at 0.05 (300 null sims)", float(np.mean(p < 0.05)))
    assert ks > 0.01


def test_matched_effect_cells_clusters_and_exact_enumeration():
    rng = np.random.default_rng(19)
    df = synth_games(rng, effect=1.0, n_seeds=3, models=("m1",), scenarios=("s1", "s2"))
    dup = df.assign(y=df["y"] + 10.0, game_id=df["game_id"] + "-rep")
    res = S.matched_effect(pd.concat([df, dup]), "y", "SHR", "ISO")  # duplicate games in a cell are averaged
    base = S.matched_effect(df, "y", "SHR", "ISO")
    assert res.n_pairs == base.n_pairs == 6
    assert res.estimate == pytest.approx(base.estimate)
    by_seed = S.matched_effect(df, "y", "SHR", "ISO", cluster="seed")
    assert by_seed.n_clusters == 3
    assert base.p_signflip >= 2 / 2 ** 6 - 1e-12            # exact enumeration: smallest two-sided p is 2/2^6
    missing = df[~((df["seed"] == 0) & (df["arm"] == "SHR"))]
    assert S.matched_effect(missing, "y", "SHR", "ISO").n_pairs == 4


def test_stratified_effect_equal_and_ivw():
    rng = np.random.default_rng(23)
    eff = {("m1", "s1"): 0.2, ("m1", "s2"): 0.5, ("m2", "s1"): 0.8, ("m2", "s2"): 0.5}
    cover_eq, est_eq = [], []
    for s in range(200):
        res = S.stratified_effect(synth_games(rng, n_seeds=12, stratum_effects=eff), "y", "SHR", "ISO",
                                  n_perm=2000, seed=s)
        cover_eq.append(res.ci_boot[0] <= 0.5 <= res.ci_boot[1])
        est_eq.append(res.estimate)
        assert res.n_strata == 4 and np.allclose(res.table["weight"], 0.25)
    report("stratified_effect equal-weight bootstrap CI coverage (200 sims)", np.mean(cover_eq))
    assert np.mean(cover_eq) >= 0.90
    assert_mean_near(est_eq, 0.5)
    sds = {("m1", "s1"): 0.5, ("m1", "s2"): 1.0, ("m2", "s1"): 2.0, ("m2", "s2"): 3.0}
    cover_iv, p_null = [], []
    for s in range(200):
        res = S.stratified_effect(synth_games(rng, effect=0.4, n_seeds=12, stratum_sd=sds), "y", "SHR", "ISO",
                                  weights="ivw", n_perm=2000, seed=s)
        cover_iv.append(res.ci_normal[0] <= 0.4 <= res.ci_normal[1])
        assert res.table["weight"].iloc[0] > res.table["weight"].iloc[3]
        p_null.append(S.stratified_effect(synth_games(rng, effect=0.0, n_seeds=12, stratum_sd=sds), "y", "SHR",
                                          "ISO", weights="ivw", n_boot=200, n_perm=2000, seed=s).p_signflip)
    report("stratified_effect IVW normal CI coverage (200 sims)", np.mean(cover_iv))
    report("stratified_effect IVW sign-flip size at 0.05 (200 null sims)", float(np.mean(np.array(p_null) < 0.05)))
    assert np.mean(cover_iv) >= 0.90
    assert np.mean(np.array(p_null) < 0.05) <= 0.09


# ----------------------------------------------------------------------------------------------------------------------
# 6-7. Rates and multiple testing: known values
# ----------------------------------------------------------------------------------------------------------------------

def test_wilson_known_values():
    # Newcombe (1998), Stat Med 17:857, Table I, method 3 (score interval without continuity correction).
    for (k, n), (lo, hi) in {(81, 263): (0.2553, 0.3662), (15, 148): (0.0624, 0.1605), (0, 20): (0.0, 0.1611),
                             (1, 29): (0.0061, 0.1718)}.items():
        got = S.wilson_ci(k, n)
        assert got[0] == pytest.approx(lo, abs=6e-5) and got[1] == pytest.approx(hi, abs=6e-5)
    lo, hi = S.wilson_ci(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=6e-5), pytest.approx(0.7634, abs=6e-5))
    with pytest.raises(ValueError):
        S.wilson_ci(3, 2)


def test_rule_of_three():
    assert S.rule_of_three_upper(30) == pytest.approx(0.1)
    assert S.rule_of_three_upper(300) >= 1 - 0.05 ** (1 / 300)   # conservative vs exact Clopper-Pearson


def test_holm_known_values():
    # R: p.adjust(c(0.01, 0.02, 0.03, 0.04, 0.05), "holm") -> 0.05 0.08 0.09 0.09 0.09
    np.testing.assert_allclose(S.holm([0.01, 0.02, 0.03, 0.04, 0.05]), [0.05, 0.08, 0.09, 0.09, 0.09])
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(S.holm([0.5, 0.6]), [1.0, 1.0])
    with pytest.raises(ValueError):
        S.holm([0.1, np.nan])


def test_fisher_combine_known_values():
    x, p = S.fisher_combine([0.05, 0.05])
    assert x == pytest.approx(-4 * np.log(0.05))
    assert p == pytest.approx(np.exp(-x / 2) * (1 + x / 2))      # chi2(4) survival function in closed form
    assert S.fisher_combine([1.0])[1] == pytest.approx(1.0)


def test_power_mde():
    z = sps.norm.ppf(0.975) + sps.norm.ppf(0.8)
    assert S.power_mde(100, 1.0) == pytest.approx(z * np.sqrt(2 / 100))     # 0.3962
    assert S.power_mde(50, 1.0, paired=True) == pytest.approx(z / np.sqrt(50))
    exact = S.power_mde(10, 1.0, method="t")
    assert exact > S.power_mde(10, 1.0)
    tc = sps.t.ppf(0.975, 18)
    nc = exact / np.sqrt(2 / 10)
    assert sps.nct.sf(tc, 18, nc) + sps.nct.cdf(-tc, 18, nc) == pytest.approx(0.8, abs=1e-8)


# ----------------------------------------------------------------------------------------------------------------------
# 8. displacement_test
# ----------------------------------------------------------------------------------------------------------------------

def synth_factorial(rng: np.random.Generator, chat_drop: float, storage_rise: float, n_seeds: int = 10) -> pd.DataFrame:
    rows = []
    for m in ("m1", "m2"):
        for sc in ("s1", "s2"):
            for seed in range(n_seeds):
                k_chat, k_beta, k_write = rng.normal(0, 0.1), rng.normal(0, 0.05), rng.normal(0, 0.05)
                for arm in S.DISPLACEMENT_ARMS.values():
                    mon, shr = "MON" in arm, arm.endswith("SHR")
                    rows.append(dict(
                        model=m, scenario=sc, seed=seed, arm=arm, game_id=f"{m}-{sc}-{seed}-{arm}",
                        chat_coordination=0.6 + k_chat - chat_drop * mon + rng.normal(0, 0.15),
                        beta=k_beta + 0.05 * shr + storage_rise * (mon and shr) + rng.normal(0, 0.15),
                        addressed_write_rate=k_write + 0.10 * shr + storage_rise * (mon and shr) + rng.normal(0, 0.1)))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("chat_drop,storage_rise,expect", [(0.4, 0.3, "detect"), (0.4, 0.0, "only_i"),
                                                           (0.0, 0.3, "only_ii")])
def test_displacement(chat_drop, storage_rise, expect):
    rng = np.random.default_rng(29)
    disp, part_i, part_ii = [], [], []
    for s in range(100):
        res = S.displacement_test(synth_factorial(rng, chat_drop, storage_rise), n_boot=999, n_perm=999, seed=s)
        assert res.n_quads == 40
        disp.append(res.displacement)
        part_i.append(res.part_i)
        part_ii.append(res.part_ii)
    report(f"displacement claim rate, {expect} (100 sims)", np.mean(disp),
           f"part_i {np.mean(part_i):.2f} part_ii {np.mean(part_ii):.2f}")
    if expect == "detect":
        assert np.mean(disp) >= 0.90
    else:
        assert np.mean(disp) <= 0.05
        assert np.mean(part_i if expect == "only_i" else part_ii) >= 0.90


# ----------------------------------------------------------------------------------------------------------------------
# 9. welfare_effect and the game-table helper
# ----------------------------------------------------------------------------------------------------------------------

def test_welfare_effect_uses_all_pairs():
    rng = np.random.default_rng(31)
    df = synth_games(rng, effect=-0.5, n_seeds=15).rename(columns={"y": "welfare"})
    df["transfer_observed"] = (df["arm"] == "SHR") & (df["seed"] % 3 == 0)
    res = S.welfare_effect(df, strata=("model", "scenario"))
    assert res.primary.n_pairs == 60
    bt = res.by_transfer.set_index("transfer")
    assert bt.loc[True, "n_pairs"] == 20 and bt.loc[False, "n_pairs"] == 40
    assert set(bt["role"]) == {"descriptive"}
    assert res.primary.ci[0] <= -0.5 <= res.primary.ci[1]
    assert res.stratified is not None and res.stratified.n_strata == 4


def test_game_table_from_rounds():
    rng = np.random.default_rng(37)
    df = synth_rounds(rng, n_games=4, rounds=12)
    df.loc[df.index[:3], "crossing_canary"] = True
    df["addressed_write"] = df["t"] % 4 == 0
    g = S.game_table(df)
    assert len(g) == 4 and g["n_rounds"].eq(12).all()
    assert g.loc[g["game_id"] == 0, "transfer_observed"].item() and not g["transfer_observed"].iloc[1:].any()
    assert g["addressed_write_rate"].to_numpy() == pytest.approx(np.full(4, 0.25))
    assert g["beta"].notna().all() and g["beta_first"].notna().all()
    # dataset.py layout: flattened welfare_<k> columns, no `welfare`; scripted honeypot rows are not agents.
    alt = df.drop(columns=["welfare"]).assign(welfare_consumer_surplus=1.0, scripted=df["firm"] == "B")
    g2 = S.game_table(alt)
    assert "welfare_consumer_surplus" in g2.columns and "welfare" not in g2.columns
    n_agent_rows = int(((alt["order"] == 2) & ~alt["fallback_action"] & (alt["t"] > 1) & ~alt["scripted"]).sum())
    assert S.pooled_beta(alt, ci=False).n == n_agent_rows
