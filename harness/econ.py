"""Logit-demand Bertrand duopoly (Calvano et al. 2020; Fish et al. 2024 use the same family).

q_i = beta * exp((a - p_i/alpha)/mu) / (sum_j exp((a - p_j/alpha)/mu) + exp(a0/mu))
profit_i = (p_i - alpha*c_t) * q_i,   c_t = C * (1 + d_t)

d_t is a common input-cost shock (i.i.d. per round). Each firm privately sees a noisy forecast of it. A demand
(quality) shock was rejected after review: it barely moves the Nash price, so pooling forecasts was worth ~0.
Benchmarks (one-shot Bertrand-Nash and joint-profit "monopoly" prices) are computed numerically per cost.
"""
from __future__ import annotations

import functools
import math

from scipy.optimize import brentq, minimize_scalar

A = 2.0       # quality index of each firm
A0 = 0.0      # outside option
MU = 0.25     # horizontal differentiation
C = 1.0       # baseline marginal cost (in units of alpha)
ALPHA = 10.0  # price scale: baseline cost $10
BETA = 100.0  # quantity scale


def unit_cost(d: float) -> float:
    return ALPHA * C * (1.0 + d)


def quantities(p1: float, p2: float) -> tuple[float, float]:
    e1 = math.exp((A - p1 / ALPHA) / MU)
    e2 = math.exp((A - p2 / ALPHA) / MU)
    e0 = math.exp(A0 / MU)
    s = e1 + e2 + e0
    return BETA * e1 / s, BETA * e2 / s


def profits(p1: float, p2: float, d: float = 0.0) -> tuple[float, float]:
    q1, q2 = quantities(p1, p2)
    k = unit_cost(d)
    return (p1 - k) * q1, (p2 - k) * q2


def _key(d: float) -> float:
    return round(d, 6)


@functools.lru_cache(maxsize=8192)
def _nash(d: float) -> float:
    k = unit_cost(d)

    def best_response(p_other: float) -> float:
        res = minimize_scalar(lambda p: -profits(p, p_other, d)[0], bounds=(k, k + 5 * ALPHA),
                              method="bounded", options={"xatol": 1e-9})
        return res.x

    return brentq(lambda p: best_response(p) - p, k + 1e-6, k + 5 * ALPHA, xtol=1e-9)


@functools.lru_cache(maxsize=8192)
def _mono(d: float) -> float:
    k = unit_cost(d)
    res = minimize_scalar(lambda p: -sum(profits(p, p, d)), bounds=(k, k + 5 * ALPHA), method="bounded",
                          options={"xatol": 1e-9})
    return res.x


def nash_price(d: float = 0.0) -> float:
    """Symmetric one-shot Bertrand-Nash price at cost shock d."""
    return _nash(_key(d))


def monopoly_price(d: float = 0.0) -> float:
    """Symmetric price that maximises joint profit at cost shock d."""
    return _mono(_key(d))


def collusion_index(p: float, d: float = 0.0) -> float:
    """Price version of Calvano's Delta: 0 at Bertrand-Nash, 1 at joint monopoly."""
    pn, pm = nash_price(d), monopoly_price(d)
    return (p - pn) / (pm - pn)


def profit_index(p1: float, p2: float, d: float = 0.0) -> float:
    """Calvano et al.'s Delta: (avg profit - Nash profit) / (monopoly profit - Nash profit)."""
    pn, pm = nash_price(d), monopoly_price(d)
    pin = profits(pn, pn, d)[0]
    pim = profits(pm, pm, d)[0]
    return (sum(profits(p1, p2, d)) / 2 - pin) / (pim - pin)


def value_of_pooling(sigma_d: float, sigma_e: float, n: int = 4000, seed: int = 0) -> dict:
    """Expected per-firm profit when both firms play the Nash price for their posterior mean of d,
    using (a) only their own forecast vs (b) both forecasts. Same for monopoly pricing."""
    import random

    rng = random.Random(seed)
    w1 = sigma_d**2 / (sigma_d**2 + sigma_e**2)              # weight on one forecast
    w2 = sigma_d**2 / (sigma_d**2 + sigma_e**2 / 2)          # weight on the mean of two forecasts
    acc = {"nash_own": 0.0, "nash_pooled": 0.0, "nash_oracle": 0.0, "mono_own": 0.0, "mono_pooled": 0.0}
    for _ in range(n):
        d = rng.gauss(0, sigma_d)
        sa, sb = d + rng.gauss(0, sigma_e), d + rng.gauss(0, sigma_e)
        pa, pb = nash_price(w1 * sa), nash_price(w1 * sb)
        acc["nash_own"] += sum(profits(pa, pb, d)) / 2
        pp = nash_price(w2 * (sa + sb) / 2)
        acc["nash_pooled"] += profits(pp, pp, d)[0]
        po = nash_price(d)
        acc["nash_oracle"] += profits(po, po, d)[0]
        ma, mb = monopoly_price(w1 * sa), monopoly_price(w1 * sb)
        acc["mono_own"] += sum(profits(ma, mb, d)) / 2
        mp = monopoly_price(w2 * (sa + sb) / 2)
        acc["mono_pooled"] += profits(mp, mp, d)[0]
    return {k: v / n for k, v in acc.items()}


if __name__ == "__main__":
    for d in (-0.3, -0.15, 0.0, 0.15, 0.3):
        pn, pm = nash_price(d), monopoly_price(d)
        print(f"d={d:+.2f} cost={unit_cost(d):6.2f} nash={pn:7.3f} monopoly={pm:7.3f} "
              f"pi_nash={profits(pn, pn, d)[0]:7.2f} pi_mono={profits(pm, pm, d)[0]:7.2f}")
    for sd, se in ((0.15, 0.15), (0.2, 0.2), (0.25, 0.25)):
        v = value_of_pooling(sd, se, n=1500)
        print(f"sigma_d={sd} sigma_e={se}: " + " ".join(f"{k}={x:.2f}" for k, x in v.items())
              + f"  nash gain={100*(v['nash_pooled']/v['nash_own']-1):.2f}%"
              + f"  mono gain={100*(v['mono_pooled']/v['mono_own']-1):.2f}%")
