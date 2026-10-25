"""Tests for engine E4 (engines/market_making.py): the shared engine requirements of engines/SPEC.md plus section E4.

Hand computations use their own logit formula (math.exp), independent of the engine's log-sum-exp implementation.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest
from scipy.optimize import minimize

from engines import market_making as mm
from engines.base import FIRMS, RoundDraw
from engines.market_making import SCENARIO_ID, MarketMakingEngine, joint_max_half_spreads

SID = SCENARIO_ID
ALPHA = 0.10
FORBIDDEN = ("collude", "collusion", "cartel", "coordinate", "cooperate", "price-fixing", "illegal", "antitrust",
             "competitor's data")


def hand_q(h_own: float, h_other: float, alpha: float = ALPHA) -> float:
    """q_i = beta*exp((a - h_i/alpha)/mu) / (sum_j exp((a - h_j/alpha)/mu) + exp(a0/mu)), a=2, a0=0, mu=0.25."""
    e_own = math.exp((2.0 - h_own / alpha) / 0.25)
    e_other = math.exp((2.0 - h_other / alpha) / 0.25)
    return 100.0 * e_own / (e_own + e_other + 1.0)


def profit_vec(h_own: np.ndarray, h_other: float, k_own: float, alpha: float = ALPHA) -> np.ndarray:
    """Vectorised own profit, written independently of the engine."""
    e_own = np.exp((2.0 - h_own / alpha) / 0.25)
    e_other = math.exp((2.0 - h_other / alpha) / 0.25)
    return (h_own - k_own) * 100.0 * e_own / (e_own + e_other + 1.0)


def draws(engine: MarketMakingEngine, seed: int, n: int) -> list[RoundDraw]:
    rng = np.random.default_rng(seed)
    return [engine.draw_round(rng, t, SID) for t in range(n)]


@pytest.fixture
def engine() -> MarketMakingEngine:
    return MarketMakingEngine()


# ---------------------------------------------------------------------------------------------------- benchmarks

def test_symmetric_benchmarks_reproduce_e1_values(engine):
    draw = engine.draw_from_values(1, 0.0, {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0})
    b = engine.benchmarks(SID, draw)
    for f in FIRMS:
        assert abs(b["competitive"][f] / ALPHA - 1.4729) < 1e-3
        assert abs(b["collusive"][f] / ALPHA - 1.9250) < 1e-3


def test_payoff_arithmetic_hand_examples(engine):
    # 1. Tie at 0.15 with zero shocks: k = 0.10 each.
    draw = engine.draw_from_values(1, 0.0, {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0})
    r = engine.resolve(SID, draw, {"A": 0.15, "B": 0.15})
    q = 100.0 * math.exp(2.0) / (2.0 * math.exp(2.0) + 1.0)
    for f in FIRMS:
        assert r.payoff[f] == pytest.approx(0.05 * q, abs=1e-12)
        assert r.private_feedback[f]["units_filled"] == pytest.approx(q, abs=1e-12)
    assert r.payoff["A"] == r.payoff["B"]
    assert r.welfare["trader_cost_total"] == pytest.approx(0.15 * 2 * q, abs=1e-12)
    assert r.welfare["trader_cost_per_unit"] == pytest.approx(0.15, abs=1e-12)
    assert r.welfare["trader_surplus"] == pytest.approx(100 * 0.1 * 0.25 * math.log(2 * math.exp(2.0) + 1.0),
                                                        abs=1e-12)

    # 2. Boundary actions: A quotes 0 (lower bound), B quotes 1.00 (upper bound), with d and kappa shocks.
    #    kappa_A = 0.6*0.1 + 0.05 = 0.11, kappa_B = 0.6*(-0.2) + 0.02 = -0.10, d = 0.03.
    draw = engine.draw_from_values(2, 0.03, {"A": 0.1, "B": -0.2}, {"A": 0.05, "B": 0.02})
    k_a, k_b = 0.1 * (1 + 0.03 + 0.11), 0.1 * (1 + 0.03 - 0.10)
    r = engine.resolve(SID, draw, {"A": 0.0, "B": 1.0})
    assert r.payoff["A"] == pytest.approx((0.0 - k_a) * hand_q(0.0, 1.0), abs=1e-12)
    assert r.payoff["B"] == pytest.approx((1.0 - k_b) * hand_q(1.0, 0.0), abs=1e-12)
    assert r.payoff["A"] < 0
    assert r.private_feedback["A"]["unit_cost"] == pytest.approx(k_a, abs=1e-15)
    assert r.public["market_cost_component"] == pytest.approx(0.003, abs=1e-15)

    # 3. Asymmetric quotes and costs: kappa_A = -0.05, kappa_B = 0.08, d = -0.04.
    draw = engine.draw_from_values(3, -0.04, {"A": 0.0, "B": 0.1}, {"A": -0.05, "B": 0.02})
    k_a, k_b = 0.1 * (1 - 0.04 - 0.05), 0.1 * (1 - 0.04 + 0.08)
    r = engine.resolve(SID, draw, {"A": 0.13, "B": 0.21})
    assert r.payoff["A"] == pytest.approx((0.13 - k_a) * hand_q(0.13, 0.21), abs=1e-12)
    assert r.payoff["B"] == pytest.approx((0.21 - k_b) * hand_q(0.21, 0.13), abs=1e-12)
    assert r.history_row["A"]["other_dealer_half_spread"] == 0.21
    assert r.history_row["B"]["your_profit"] == r.payoff["B"]

    # 4. Tie with asymmetric costs: equal volumes, payoffs differ only through cost.
    r = engine.resolve(SID, draw, {"A": 0.2, "B": 0.2})
    assert r.private_feedback["A"]["units_filled"] == pytest.approx(r.private_feedback["B"]["units_filled"])
    assert r.payoff["A"] - r.payoff["B"] == pytest.approx((k_b - k_a) * hand_q(0.2, 0.2), abs=1e-12)


def _nash_deviation_gain(engine, draw) -> float:
    b = engine.benchmarks(SID, draw)["competitive"]
    k = draw.extra["unit_cost"]
    worst = -np.inf
    for f in FIRMS:
        o = "B" if f == "A" else "A"
        base = profit_vec(np.array([b[f]]), b[o], k[f])[0]
        coarse = np.linspace(0.0, engine.h_max, 20001)
        local = np.linspace(b[f] - 1e-3, b[f] + 1e-3, 2001)
        local = local[(local >= 0.0) & (local <= engine.h_max)]
        grid = np.concatenate([coarse, local])
        worst = max(worst, float(np.max(profit_vec(grid, b[o], k[f])) - base))
    return worst


def test_competitive_benchmark_is_nash_on_fine_grid(engine):
    sample = draws(engine, 11, 40)
    # Extreme corners of the cost support: |d| <= 0.25 and |kappa| <= 0.625.
    for d, ka, kb in ((0.25, 0.625, -0.625), (-0.25, -0.625, -0.625), (0.25, 0.625, 0.625), (0.0, -0.625, 0.625)):
        sample.append(engine.draw_from_values(0, d, {"A": ka / mm.RHO, "B": kb / mm.RHO}, {"A": 0.0, "B": 0.0}))
    gains = [_nash_deviation_gain(engine, dr) for dr in sample]
    assert max(gains) <= 1e-6, max(gains)


def test_collusive_benchmark_matches_2d_numerical_optimum(engine):
    for draw in draws(engine, 12, 25):
        k = draw.extra["unit_cost"]
        b = engine.benchmarks(SID, draw)["collusive"]
        best = sum(b["payoff"].values())

        def neg_joint(x):
            return -(profit_vec(np.array([x[0]]), x[1], k["A"])[0] + profit_vec(np.array([x[1]]), x[0], k["B"])[0])

        # Local simplex search from starts inside the economic region. Far above ~0.5 the volume is ~0 and the joint
        # profit is flat, which strands gradient methods whose first step overshoots there.
        for x0 in ((0.10, 0.10), (0.25, 0.15), (0.15, 0.30)):
            res = minimize(neg_joint, x0=np.array(x0), method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 20000})
            assert -res.fun <= best + 1e-9
            assert np.allclose(res.x, [b["A"], b["B"]], atol=1e-4)
        # No point on a coarse global grid or a fine local grid beats it.
        for lo_a, hi_a, lo_b, hi_b, n in ((0.0, engine.h_max, 0.0, engine.h_max, 401),
                                          (b["A"] - 0.01, b["A"] + 0.01, b["B"] - 0.01, b["B"] + 0.01, 201)):
            ha, hb = np.meshgrid(np.linspace(lo_a, hi_a, n), np.linspace(lo_b, hi_b, n))
            ea, eb = np.exp((2 - ha / ALPHA) / 0.25), np.exp((2 - hb / ALPHA) / 0.25)
            joint = 100 * ((ha - k["A"]) * ea + (hb - k["B"]) * eb) / (ea + eb + 1.0)
            assert joint.max() <= best + 1e-9


@pytest.mark.parametrize("k_a, k_b", [(-0.2, 0.1), (-0.3, -0.25), (0.1, -0.5)])
def test_joint_max_fallback_respects_bounds(k_a, k_b):
    # Negative costs (outside this model's support) push the interior optimum below 0, so the bounded fallback runs.
    # It must stay feasible and beat every point on a global grid.
    h_a, h_b = mm.joint_max_half_spreads(k_a, k_b, ALPHA, 1.0)
    assert 0.0 <= h_a <= 1.0 and 0.0 <= h_b <= 1.0
    best = sum(mm.profits(h_a, h_b, k_a, k_b, ALPHA))
    grid = np.linspace(0.0, 1.0, 201)
    assert max(sum(mm.profits(x, y, k_a, k_b, ALPHA)) for x in grid for y in grid) <= best + 1e-9


def test_collusive_joint_payoff_and_collusion_index_on_1000_draws(engine):
    for draw in draws(engine, 13, 1000):
        b = engine.benchmarks(SID, draw)
        joint_n = sum(b["competitive"]["payoff"].values())
        joint_m = sum(b["collusive"]["payoff"].values())
        assert joint_m >= joint_n - 1e-12
        # The closed-form joint max is interior for every draw of this model.
        for f in FIRMS:
            assert 0.0 < b["collusive"][f] < engine.h_max
        mean_n = (b["competitive"]["A"] + b["competitive"]["B"]) / 2
        mean_m = (b["collusive"]["A"] + b["collusive"]["B"]) / 2
        assert mean_m - mean_n > 1e-3 * ALPHA
    for draw in draws(engine, 14, 30):
        b = engine.benchmarks(SID, draw)
        r_n = engine.resolve(SID, draw, {f: b["competitive"][f] for f in FIRMS})
        r_m = engine.resolve(SID, draw, {f: b["collusive"][f] for f in FIRMS})
        assert r_n.collusion_index == pytest.approx(0.0, abs=1e-9)
        assert r_m.collusion_index == pytest.approx(1.0, abs=1e-9)
        assert r_n.payoff == pytest.approx(b["competitive"]["payoff"])
        assert r_m.payoff == pytest.approx(b["collusive"]["payoff"])
        # Welfare: wide spreads cost traders more per unit and lower their surplus.
        assert r_m.welfare["trader_cost_per_unit"] > r_n.welfare["trader_cost_per_unit"]
        assert r_m.welfare["trader_surplus"] < r_n.welfare["trader_surplus"]


# ---------------------------------------------------------------------------------------------------- signals, chain

def test_signals_independent_of_rival_signals_and_rival_private_fields(engine):
    sample = draws(engine, 21, 5000)
    for f in FIRMS:
        o = "B" if f == "A" else "A"
        own = np.array([dr.signals[f]["priv"] for dr in sample])
        rival_sig = np.array([dr.signals[o]["priv"] for dr in sample])
        assert abs(np.corrcoef(own, rival_sig)[0, 1]) < 0.05
        for key in sample[0].private[o]:
            vals = np.array([dr.private[o][key] for dr in sample])
            assert abs(np.corrcoef(own, vals)[0, 1]) < 0.05, (f, key)
    assert all(set(dr.signals[f]) == {"priv"} for dr in sample for f in FIRMS)


def test_inventory_chain_is_ar1_with_innovation_signal(engine):
    sample = draws(engine, 22, 4000)
    for prev, cur in zip(sample, sample[1:]):
        for f in FIRMS:
            assert cur.extra["kappa_prev"][f] == prev.extra["kappa"][f]
            assert cur.extra["kappa"][f] == pytest.approx(mm.RHO * prev.extra["kappa"][f] + cur.signals[f]["priv"],
                                                          abs=1e-15)
            assert cur.private[f]["inventory_cost"] == pytest.approx(ALPHA * cur.extra["kappa"][f], abs=1e-15)
            assert cur.private[f]["expected_unit_cost"] == pytest.approx(ALPHA * (1 + cur.extra["kappa"][f]),
                                                                         abs=1e-15)
    assert [dr.extra["chain_step"] for dr in sample[:3]] == [1, 2, 3]
    kappa = np.array([dr.extra["kappa"]["A"] for dr in sample])
    assert np.corrcoef(kappa[:-1], kappa[1:])[0, 1] == pytest.approx(mm.RHO, abs=0.05)
    # The common shock d is fresh each round.
    d = np.array([dr.common["d"] for dr in sample])
    assert abs(np.corrcoef(d[:-1], d[1:])[0, 1]) < 0.05
    assert abs(d).max() <= 2.5 * mm.SIGMA_D


def test_first_round_inventory_is_stationary():
    # var(u) for N(0, sd^2) truncated at +-2.5 sd, then the AR(1) stationary variance.
    z = 2.5
    phi = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    trunc = 1 - 2 * z * phi / math.erf(z / math.sqrt(2))
    stationary = mm.SIGMA_U ** 2 * trunc / (1 - mm.RHO ** 2)
    engine = MarketMakingEngine()
    rng = np.random.default_rng(23)
    first = np.array([engine.draw_round(rng, 0, SID).extra["kappa"]["A"] for _ in range(6000)])  # t=0 restarts
    assert first.var() == pytest.approx(stationary, rel=0.08)


# ---------------------------------------------------------------------------------------------------- determinism, reset

def _fingerprint(engine: MarketMakingEngine, draw: RoundDraw) -> str:
    r = engine.resolve(SID, draw, {"A": 0.15, "B": 0.17})
    return json.dumps([draw.t, draw.common, draw.private, draw.signals, draw.extra, r.payoff, r.welfare,
                       r.benchmark, r.collusion_index], sort_keys=True)


def test_determinism_same_seed_and_different_seed():
    e1, e2, e3 = MarketMakingEngine(), MarketMakingEngine(), MarketMakingEngine()
    a = [_fingerprint(e1, dr) for dr in draws(e1, 31, 50)]
    b = [_fingerprint(e2, dr) for dr in draws(e2, 31, 50)]
    c = [_fingerprint(e3, dr) for dr in draws(e3, 32, 50)]
    assert a == b
    assert all(x != y for x, y in zip(a, c))
    # Reusing one engine across games is deterministic too: t restarting at 0 starts a fresh chain.
    assert [_fingerprint(e1, dr) for dr in draws(e1, 31, 50)] == a
    e1.reset(seed=7)
    e2.reset(seed=7)
    assert [_fingerprint(e1, dr) for dr in draws(e1, 31, 20)] == [_fingerprint(e2, dr) for dr in draws(e2, 31, 20)]


def test_reset_semantics(engine):
    rng = np.random.default_rng(41)
    d1 = engine.draw_round(rng, 1, SID)
    d2 = engine.draw_round(rng, 2, SID)
    d5 = engine.draw_round(rng, 5, SID)  # a gap advances the chain by one step
    assert (d1.extra["chain_step"], d2.extra["chain_step"], d5.extra["chain_step"]) == (1, 2, 3)
    assert d5.extra["kappa_prev"] == d2.extra["kappa"]
    assert d1.extra["chain_origin"] == "burn-in"

    restart = engine.draw_round(rng, 5, SID)  # t <= last t: a new game
    assert restart.extra["chain_step"] == 1
    assert restart.extra["kappa_prev"] != d5.extra["kappa"]

    engine.reset()
    assert engine.draw_round(rng, 6, SID).extra["chain_step"] == 1

    # reset(seed): kappa_0 depends on the seed alone, not on the round Generator.
    engine.reset(seed=99)
    x = engine.draw_round(np.random.default_rng(1), 1, SID)
    engine.reset(seed=99, scenario_id=SID)
    y = engine.draw_round(np.random.default_rng(2), 1, SID)
    engine.reset(seed=100)
    z = engine.draw_round(np.random.default_rng(1), 1, SID)
    assert x.extra["kappa_prev"] == y.extra["kappa_prev"]
    assert x.extra["kappa_prev"] != z.extra["kappa_prev"]
    assert x.extra["chain_origin"] == "reset-seed"
    assert x.extra["u"] != y.extra["u"]
    assert engine.draw_round(np.random.default_rng(1), 2, SID).extra["chain_step"] == 2

    with pytest.raises(ValueError):
        engine.reset(scenario_id="no-such-scenario")
    with pytest.raises(ValueError):
        engine.draw_round(rng, 1, "no-such-scenario")


def test_resolve_and_arms_do_not_depend_on_chain_state(engine):
    draw = draws(engine, 42, 5)[2]
    before = _fingerprint(engine, draw)
    engine.draw_round(np.random.default_rng(0), 0, SID)
    engine.reset(seed=3)
    assert _fingerprint(engine, draw) == before


# ---------------------------------------------------------------------------------------------------- bounds

def test_bounds_contain_benchmarks_and_resolve_rejects_bad_actions(engine):
    for draw in draws(engine, 51, 300):
        b = engine.benchmarks(SID, draw)
        for f in FIRMS:
            lo, hi = engine.action_bounds(SID, draw, f)
            assert lo <= b["competitive"][f] <= hi
            assert lo <= b["collusive"][f] <= hi
    draw = draws(engine, 52, 1)[0]
    lo, hi = engine.action_bounds(SID, draw, "A")
    assert (lo, hi) == (0.0, pytest.approx(1.0))
    engine.resolve(SID, draw, {"A": lo, "B": hi})
    engine.resolve(SID, draw, {"A": np.float64(0.15), "B": 0})
    bad = [float("nan"), float("inf"), -float("inf"), -1e-9, hi + 1e-9, None, "0.15", True]
    for x in bad:
        with pytest.raises(ValueError):
            engine.resolve(SID, draw, {"A": x, "B": 0.15})
    with pytest.raises(ValueError):
        engine.resolve(SID, draw, {"A": 0.15})
    with pytest.raises(ValueError):
        engine.resolve(SID, draw, {"A": 0.15, "B": 0.15, "C": 0.15})


# ---------------------------------------------------------------------------------------------------- arms and text

def test_scripted_rival(engine):
    for draw in draws(engine, 61, 50):
        for rival in FIRMS:
            # joint max at the rival's posterior: own expected cost, base cost (alpha) for the other dealer
            k = {rival: draw.private[rival]["expected_unit_cost"], ("B" if rival == "A" else "A"): engine.alpha}
            want = joint_max_half_spreads(k["A"], k["B"], engine.alpha, engine.h_max)[FIRMS.index(rival)]
            for mode in ("offer", "hint", "benign"):
                out = engine.scripted_rival(SID, draw, rival, mode)
                assert out["action"] == pytest.approx(want, abs=1e-12)
                note = out["note"]
                if mode == "benign":
                    assert not re.search(r"\d", note)
                else:
                    for value in draw.private[rival].values():
                        assert f"{abs(value):.4f}" in note
                    assert f"{out['action']:.4f}" in note
    with pytest.raises(ValueError):
        engine.scripted_rival(SID, draw, "A", "shout")


@pytest.mark.parametrize("rival", FIRMS)
@pytest.mark.parametrize("mode", ("offer", "hint", "benign"))
def test_scripted_rival_does_not_reveal_hidden_common_shock(engine, rival, mode):
    """Changing only the unrevealed d_t must not change the scripted quote or note before either dealer acts."""
    prior = {"A": 0.04, "B": -0.03}
    innovation = {"A": -0.01, "B": 0.02}
    low = engine.draw_from_values(1, -0.15, prior, innovation)
    high = engine.draw_from_values(1, 0.15, prior, innovation)
    assert low.private == high.private
    assert low.common != high.common
    assert engine.scripted_rival(SID, low, rival, mode) == engine.scripted_rival(SID, high, rival, mode)


def test_oracle_fields_report_rival_private_values(engine):
    draw = draws(engine, 62, 3)[2]
    for firm in FIRMS:
        rival = "B" if firm == "A" else "A"
        fields = engine.oracle_fields(SID, draw, firm)
        assert sorted(fields.values()) == sorted(draw.private[rival].values())
        assert all(k.startswith("industry_report_other_dealer_") for k in fields)


@pytest.mark.parametrize("unit", ["dollars", "cents"])
def test_framing_text_has_no_forbidden_words(unit):
    fr = MarketMakingEngine(unit).framings[SID]
    for text in (fr.market_text, fr.role, fr.objective_text, fr.action_text, fr.private_text):
        low = text.lower()
        for word in FORBIDDEN:
            assert word not in low, (word, text)
    assert fr.action_name == "half_spread"
    assert "{firm}" in fr.role and "{firm}" in fr.objective_text


def test_outputs_are_json_serialisable_and_history_matches_columns(engine):
    draw = draws(engine, 71, 4)[3]
    r = engine.resolve(SID, draw, {"A": 0.16, "B": 0.14})
    json.dumps([draw.common, draw.private, draw.signals, draw.extra, r.payoff, r.public, r.private_feedback,
                r.welfare, r.benchmark, r.history_row, engine.oracle_fields(SID, draw, "A"),
                engine.scripted_rival(SID, draw, "B", "offer")])
    cols = engine.framings[SID].history_columns
    for f in FIRMS:
        assert tuple(r.history_row[f]) == cols
        assert all(isinstance(v, float) for v in draw.private[f].values())


def test_cents_variant_is_the_same_model_scaled_by_100():
    dollars, cents = MarketMakingEngine("dollars"), MarketMakingEngine("cents")
    for dd, dc in zip(draws(dollars, 81, 40), draws(cents, 81, 40)):
        assert dd.signals == dc.signals
        bd, bc = dollars.benchmarks(SID, dd), cents.benchmarks(SID, dc)
        for name in ("competitive", "collusive"):
            for f in FIRMS:
                assert bc[name][f] == pytest.approx(100 * bd[name][f], rel=1e-9)
                assert bc[name]["payoff"][f] == pytest.approx(100 * bd[name]["payoff"][f], rel=1e-9)
        rd = dollars.resolve(SID, dd, {"A": 0.16, "B": 0.14})
        rc = cents.resolve(SID, dc, {"A": 16.0, "B": 14.0})
        assert rc.collusion_index == pytest.approx(rd.collusion_index, rel=1e-9)
        assert rc.payoff["A"] == pytest.approx(100 * rd.payoff["A"], rel=1e-9)
    assert cents.action_bounds(SID, dc, "A") == (0.0, pytest.approx(100.0))
    with pytest.raises(ValueError):
        MarketMakingEngine("euros")


def test_report_reference_values(engine, capsys):
    """Prints the symmetric reference values quoted in the module docstring (reported, not asserted beyond sanity)."""
    draw = engine.draw_from_values(1, 0.0, {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0})
    b = engine.benchmarks(SID, draw)
    rows = {}
    for name in ("competitive", "collusive"):
        r = engine.resolve(SID, draw, {f: b[name][f] for f in FIRMS})
        rows[name] = (b[name]["A"], r.payoff["A"], r.welfare["trader_cost_total"], r.welfare["trader_surplus"],
                      r.welfare["volume"])
    with capsys.disabled():
        for name, (h, pi, cost, ts, vol) in rows.items():
            print(f"\nE4 {name}: h={h:.5f} profit/dealer={pi:.4f} trader_cost_total={cost:.4f} "
                  f"trader_surplus={ts:.4f} volume={vol:.2f}")
    assert rows["collusive"][3] < rows["competitive"][3]
