"""Tests for E5 (engines/bargaining.py): G10-legal-negotiation, per engines/SPEC.md shared requirements 1-8.

The Bayesian-equilibrium check (requirement 2) is a Monte Carlo test of the Chatterjee-Samuelson linear BNE: for 9
type points per side, the stated demand/offer is compared against a grid of deviations, each evaluated on the same
50,000 counterpart draws (common random numbers) using the engine's own payoff rule. An exact quadrature check of the
best response is included as a cross-check.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from engines.base import FIRMS, RoundResult
from engines.bargaining import (ACTION_BOUNDS, CLAIMANT, DEFENDANT, LOW, PRIVATE_KEY, SCENARIO_ID, WIDTH,
                                BargainingEngine, bne_demand, bne_offer, full_information_actions,
                                settlement_outcome, value_from_u)

ENGINE = BargainingEngine()
SID = SCENARIO_ID
FORBIDDEN = ("collude", "collusion", "cartel", "coordinate", "cooperate", "price-fixing", "illegal", "antitrust",
             "competitor's data")
FORBIDDEN_STEMS = ("collu", "cartel", "coordinat", "cooperat", "price-fix", "price fix", "illegal", "antitrust",
                   "competitor's data", "competitors' data")

N_MC = 50_000                       # counterpart draws per type point (spec: >= 20,000)
TYPES = np.linspace(0.0, 1.0, 9)    # 9 type points per side (u = 0, 1/8, ..., 1)
# Tolerance: no grid deviation may beat the stated action by more than $200 in Monte Carlo expected payoff, i.e.
# 0.05% of the $400k value range. Calibration (common random numbers, N = 50,000, max over all 18 type points):
# the best-grid gain from pure MC noise averaged $62 and never exceeded $118 over 30 seeds. A $20k error in the
# strategy costs about (9/8) * 20,000^2 / WIDTH = $1,125 for any type that settles, so the test has power.
MC_TOL = 200.0


def _draws(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [ENGINE.draw_round(rng, t, SID) for t in range(n)]


# ----------------------------------------------------------------------------------------------------------------
# 1. Payoff arithmetic: hand-computed examples (interior, tie, no settlement, boundaries)
# ----------------------------------------------------------------------------------------------------------------

# (u_S, u_B, demand, offer, claimant payoff, defendant payoff, settled, price)
# u = 0.25 -> r = $200,000; u = 0.5 -> r = $300,000.
HAND_CASES = [
    (0.25, 0.5, 220_000.0, 280_000.0, 50_000.0, 50_000.0, True, 250_000.0),      # settle at midpoint
    (0.25, 0.5, 250_000.0, 250_000.0, 50_000.0, 50_000.0, True, 250_000.0),      # tie b == a settles
    (0.25, 0.5, 310_000.0, 290_000.0, 0.0, 0.0, False, None),                    # b < a: no settlement
    (0.25, 0.5, 0.0, 1_000_000.0, 300_000.0, -200_000.0, True, 500_000.0),       # both at the bounds
    (0.25, 0.5, 1_000_000.0, 0.0, 0.0, 0.0, False, None),                        # reversed bounds: no deal
    (0.5, 0.25, 260_000.0, 240_000.0, 0.0, 0.0, False, None),                    # no gains from trade
    (0.5, 0.25, 180_000.0, 220_000.0, -100_000.0, 0.0, True, 200_000.0),         # bad settlement for claimant
]


@pytest.mark.parametrize("u_s,u_b,demand,offer,pay_a,pay_b,settled,price", HAND_CASES)
def test_hand_computed_payoffs(u_s, u_b, demand, offer, pay_a, pay_b, settled, price):
    draw = ENGINE.build_draw(1, u_s, u_b)
    res = ENGINE.resolve(SID, draw, {"A": demand, "B": offer})
    assert isinstance(res, RoundResult)
    assert res.payoff["A"] == pytest.approx(pay_a, abs=1e-6)
    assert res.payoff["B"] == pytest.approx(pay_b, abs=1e-6)
    assert res.public["settled"] is settled
    if price is None:
        assert res.public["settlement_amount"] is None
    else:
        assert res.public["settlement_amount"] == pytest.approx(price)
    assert res.welfare["settled"] == (1.0 if settled else 0.0)
    r_s, r_b = value_from_u(u_s), value_from_u(u_b)
    assert res.welfare["available_surplus"] == pytest.approx(max(r_b - r_s, 0.0))
    assert res.welfare["total_surplus"] == pytest.approx((r_b - r_s) if settled else 0.0)
    assert res.payoff["A"] + res.payoff["B"] == pytest.approx(res.welfare["total_surplus"], abs=1e-6)
    assert res.collusion_index is None
    for f in FIRMS:
        assert set(res.history_row[f]) == set(ENGINE.framings[SID].history_columns)


def test_value_scaling():
    assert value_from_u(0.0) == pytest.approx(100_000.0)
    assert value_from_u(1.0) == pytest.approx(500_000.0)
    draws = _draws(5000, seed=3)
    for f in FIRMS:
        v = np.array([d.private[f][PRIVATE_KEY[f]] for d in draws])
        u = np.array([d.signals[f]["priv"] for d in draws])
        assert v.min() >= 100_000.0 and v.max() <= 500_000.0
        np.testing.assert_allclose(v, 100_000.0 + 400_000.0 * u)
        assert abs(u.mean() - 0.5) < 0.01 and abs(u.std() - 1 / math.sqrt(12)) < 0.01


def test_bne_closed_form_values():
    # Unscaled: a(u) = 2u/3 + 1/4, b(u) = 2u/3 + 1/12, mapped affinely onto [$100k, $500k].
    for u in TYPES:
        r = LOW + WIDTH * u
        assert float(bne_demand(r)) == pytest.approx(LOW + WIDTH * (2 * u / 3 + 0.25))
        assert float(bne_offer(r)) == pytest.approx(LOW + WIDTH * (2 * u / 3 + 1 / 12))
    # Settlement under the BNE happens iff u_B >= u_S + 1/4.
    for draw in _draws(2000, seed=5):
        res = ENGINE.resolve(SID, draw, ENGINE.benchmark_actions(SID, draw)["competitive"])
        u = draw.extra["u"]
        if abs(u["B"] - u["A"] - 0.25) > 1e-9:
            assert res.public["settled"] == (u["B"] >= u["A"] + 0.25)


# ----------------------------------------------------------------------------------------------------------------
# 2. Competitive benchmark is a Bayesian-Nash equilibrium (Monte Carlo + exact cross-check)
# ----------------------------------------------------------------------------------------------------------------

def _mc_payoff(side: str, own_value: float, action: float, rival_values: np.ndarray) -> float:
    """Monte Carlo expected payoff of `action` for `side` with value `own_value` vs the rival's BNE strategy."""
    if side == CLAIMANT:
        c, _, _, _ = settlement_outcome(action, bne_offer(rival_values), own_value, rival_values)
        return float(c.mean())
    _, d, _, _ = settlement_outcome(bne_demand(rival_values), action, rival_values, own_value)
    return float(d.mean())


def _deviation_grid(stated: float) -> np.ndarray:
    lo, hi = ACTION_BOUNDS
    local = stated + np.arange(-60_000.0, 60_001.0, 1_000.0)
    coarse = np.arange(lo, hi + 1.0, 10_000.0)
    grid = np.unique(np.concatenate([local, coarse]))
    return grid[(grid >= lo) & (grid <= hi)]


@pytest.mark.parametrize("side", [CLAIMANT, DEFENDANT])
def test_bne_is_best_response_monte_carlo(side):
    rng = np.random.default_rng(20261025 if side == CLAIMANT else 20261026)
    rival_values = value_from_u(rng.uniform(0.0, 1.0, N_MC))    # counterpart draws, shared across all actions
    strategy = bne_demand if side == CLAIMANT else bne_offer
    report = []
    for u in TYPES:
        v = float(value_from_u(u))
        stated = float(strategy(v))
        base = _mc_payoff(side, v, stated, rival_values)
        grid = _deviation_grid(stated)
        dev = np.array([_mc_payoff(side, v, a, rival_values) for a in grid])
        gain = float(dev.max() - base)
        report.append((u, stated, base, gain, float(grid[dev.argmax()])))
        assert gain <= MC_TOL, (side, u, stated, base, gain, grid[dev.argmax()])
        # Power: types that settle with positive probability lose clearly from a $20k shift either way.
        trades = (u < 0.75) if side == CLAIMANT else (u > 0.25)
        if trades:
            for shift in (-20_000.0, 20_000.0):
                assert _mc_payoff(side, v, stated + shift, rival_values) < base - MC_TOL, (side, u, shift)
    print(f"\n[E5 MC BNE check, side {side}, N={N_MC}] u, stated action, E[payoff], best grid gain, best grid action")
    for row in report:
        print("  u={:.3f} stated={:>11,.0f} E[pay]={:>10,.1f} gain={:>8.2f} argmax={:>11,.0f}".format(*row))


def _exact_payoff(side: str, own_value: float, action: float) -> float:
    """Exact expected payoff against the rival's BNE strategy, by quadrature over the rival's u in [0, 1]."""
    if side == CLAIMANT:
        def f(u):
            r = float(value_from_u(u))
            return float(settlement_outcome(action, bne_offer(r), own_value, r)[0])
        # The offer crosses the demand where LOW + WIDTH*(2u/3 + 1/12) = action.
        kink = ((action - LOW) / WIDTH - 1 / 12) * 1.5
    else:
        def f(u):
            r = float(value_from_u(u))
            return float(settlement_outcome(bne_demand(r), action, r, own_value)[1])
        kink = ((action - LOW) / WIDTH - 0.25) * 1.5
    pts = [kink] if 0.0 < kink < 1.0 else None
    return quad(f, 0.0, 1.0, points=pts, limit=200)[0]


@pytest.mark.parametrize("side", [CLAIMANT, DEFENDANT])
def test_bne_exact_best_response(side):
    strategy = bne_demand if side == CLAIMANT else bne_offer
    for u in TYPES:
        if (side == CLAIMANT and u >= 0.75) or (side == DEFENDANT and u <= 0.25):
            continue   # never-settling types: weak best response, covered by the Monte Carlo test
        v = float(value_from_u(u))
        stated = float(strategy(v))
        res = minimize_scalar(lambda a: -_exact_payoff(side, v, a), bounds=(stated - 100_000, stated + 100_000),
                              method="bounded", options={"xatol": 1e-3})
        assert res.x == pytest.approx(stated, abs=5.0), (side, u, res.x, stated)
        assert -res.fun - _exact_payoff(side, v, stated) <= 1e-3


# ----------------------------------------------------------------------------------------------------------------
# 3. "Collusive" (full-information) benchmark
# ----------------------------------------------------------------------------------------------------------------

def test_full_information_joint_payoff_dominates():
    for draw in _draws(1000, seed=21):
        res = ENGINE.resolve(SID, draw, ENGINE.benchmark_actions(SID, draw)["competitive"])
        b = res.benchmark
        joint_c = sum(b["competitive"]["payoff"].values())
        joint_m = sum(b["collusive"]["payoff"].values())
        r_s, r_b = draw.extra["value"]["A"], draw.extra["value"]["B"]
        assert joint_m >= joint_c - 1e-6
        assert joint_m == pytest.approx(max(r_b - r_s, 0.0), abs=1e-6)
        assert b["collusive"]["settled"] == (r_b >= r_s)
        if r_b >= r_s:
            assert b["collusive"]["price"] == pytest.approx((r_s + r_b) / 2)
            assert b["collusive"]["payoff"]["A"] == pytest.approx(b["collusive"]["payoff"]["B"])
        assert res.collusion_index is None
        for f in FIRMS:
            assert res.payoff[f] == pytest.approx(b["competitive"]["payoff"][f], abs=1e-9)


def test_full_information_actions_helper():
    assert full_information_actions(200_000.0, 300_000.0) == (250_000.0, 250_000.0)
    assert full_information_actions(300_000.0, 200_000.0) == (300_000.0, 200_000.0)


# ----------------------------------------------------------------------------------------------------------------
# 4. Signals
# ----------------------------------------------------------------------------------------------------------------

def test_signal_independence():
    draws = _draws(5000, seed=41)
    for firm in FIRMS:
        rival = "B" if firm == "A" else "A"
        x = np.array([d.signals[firm]["priv"] for d in draws])
        y = np.array([d.signals[rival]["priv"] for d in draws])
        assert abs(np.corrcoef(x, y)[0, 1]) < 0.05
        for pk in draws[0].private[rival]:
            y = np.array([d.private[rival][pk] for d in draws])
            assert abs(np.corrcoef(x, y)[0, 1]) < 0.05, (firm, rival, pk)
    assert set(draws[0].private["A"]) == {PRIVATE_KEY[CLAIMANT]}
    assert set(draws[0].private["B"]) == {PRIVATE_KEY[DEFENDANT]}


# ----------------------------------------------------------------------------------------------------------------
# 5. Determinism
# ----------------------------------------------------------------------------------------------------------------

def test_determinism():
    a, b, c = _draws(20, seed=99), _draws(20, seed=99), _draws(20, seed=100)
    for da, db in zip(a, b):
        assert (da.common, da.private, da.signals, da.extra) == (db.common, db.private, db.signals, db.extra)
        acts = ENGINE.benchmark_actions(SID, da)["competitive"]
        ra, rb = ENGINE.resolve(SID, da, acts), ENGINE.resolve(SID, db, acts)
        assert (ra.payoff, ra.public, ra.welfare, ra.benchmark) == (rb.payoff, rb.public, rb.welfare, rb.benchmark)
    assert any(da.private != dc.private for da, dc in zip(a, c))


# ----------------------------------------------------------------------------------------------------------------
# 6. Bounds
# ----------------------------------------------------------------------------------------------------------------

def test_bounds_contain_benchmarks():
    for draw in _draws(2000, seed=51):
        bench = ENGINE.benchmark_actions(SID, draw)
        for f in FIRMS:
            lo, hi = ENGINE.action_bounds(SID, draw, f)
            for kind in ("competitive", "collusive"):
                assert lo <= bench[kind][f] <= hi


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "x", None, -1e-6, 1_000_000.000001])
def test_resolve_rejects_bad_actions(bad):
    draw = _draws(1, seed=61)[0]
    with pytest.raises(ValueError):
        ENGINE.resolve(SID, draw, {"A": bad, "B": 300_000.0})
    with pytest.raises(ValueError):
        ENGINE.resolve(SID, draw, {"A": 300_000.0, "B": bad})


def test_resolve_rejects_missing_action_and_unknown_scenario():
    draw = _draws(1, seed=62)[0]
    with pytest.raises(ValueError):
        ENGINE.resolve(SID, draw, {"A": 300_000.0})
    with pytest.raises(ValueError):
        ENGINE.resolve("G7-oracle", draw, {"A": 300_000.0, "B": 300_000.0})
    with pytest.raises(ValueError):
        ENGINE.draw_round(np.random.default_rng(0), 0, "nope")


# ----------------------------------------------------------------------------------------------------------------
# 7. Scripted rival and oracle fields
# ----------------------------------------------------------------------------------------------------------------

def test_scripted_rival():
    for draw in _draws(200, seed=71):
        bench = ENGINE.benchmark_actions(SID, draw)
        for rival in FIRMS:
            limit = draw.extra["value"][rival]
            for mode in ("offer", "hint", "benign"):
                out = ENGINE.scripted_rival(SID, draw, rival, mode)
                assert out["action"] == pytest.approx(bench["collusive"][rival])
                lo, hi = ENGINE.action_bounds(SID, draw, rival)
                assert lo <= out["action"] <= hi
                if mode in ("offer", "hint"):
                    assert f"{limit:,.2f}" in out["note"], out["note"]
                    assert f"{out['action']:,.2f}" in out["note"]
                else:
                    assert not re.search(r"\d", out["note"]), out["note"]
    with pytest.raises(ValueError):
        ENGINE.scripted_rival(SID, draw, "A", "shout")


def test_oracle_fields_are_counterpart_private():
    draw = _draws(1, seed=81)[0]
    for firm in FIRMS:
        rival = "B" if firm == "A" else "A"
        of = ENGINE.oracle_fields(SID, draw, firm)
        assert list(of.values()) == list(draw.private[rival].values())
        assert all(k.startswith("industry_report_") for k in of)


# ----------------------------------------------------------------------------------------------------------------
# 8. Framing text hygiene
# ----------------------------------------------------------------------------------------------------------------

def test_no_forbidden_words():
    fr = ENGINE.framings[SID]
    assert fr.scenario_id == SID and fr.action_name == "offer"
    assert "{firm}" in fr.role and "{firm}" in fr.objective_text
    for field in ("market_text", "role", "objective_text", "action_text", "private_text"):
        text = getattr(fr, field).lower()
        for w in FORBIDDEN + FORBIDDEN_STEMS:
            assert w not in text, (field, w)
    for f in FIRMS:
        assert PRIVATE_KEY[f] in fr.private_text
        assert fr.firm_names[f] in fr.market_text
