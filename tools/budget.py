"""Budget and power estimates for the sprint plan. Prices are nanoGPT USD per 1M tokens (read 25 Oct 2026)."""
from math import sqrt

from scipy.stats import norm

import sys

ROUNDS = 30
CALLS_PER_AGENT_ROUND = 2            # step 1 (shell) + step 2 (price)
IN_TOK, OUT_TOK = 3000, 600          # per call; OUT includes visible reasoning; hidden reasoning may add more
if len(sys.argv) == 3:               # e.g. `budget.py 1500 250` for the slim prompt budget
    IN_TOK, OUT_TOK = int(sys.argv[1]), int(sys.argv[2])

MODELS = {  # id: (label, in $/M, out $/M, group)
    "anthropic/claude-opus-5.5": ("Claude Opus 5.5", 4, 20, "frontier"),
    "anthropic/claude-sonnet-5.5": ("Claude Sonnet 5.5", 2, 10, "frontier"),
    "openai/gpt-6.1-sol": ("GPT-6.1 Sol", 2, 10, "frontier"),
    "openai/gpt-6-astra": ("GPT-6 Astra", 10, 50, "frontier"),
    "google/gemini-3.1-pro-preview": ("Gemini 3.1 Pro", 2, 12, "frontier"),
    "x-ai/grok-4.7": ("Grok 4.7", 2, 6, "frontier"),
    "moonshotai/kimi-k3.1-pro": ("Kimi K3.1 Pro", 2, 10, "frontier"),
    "qwen/qwen3.8-max-prime": ("Qwen 3.8 Max", 4, 12, "frontier"),
    "z-ai/glm-5.3": ("GLM 5.3", 0.6, 1.1, "frontier"),
    "deepseek/deepseek-v4-pro": ("DeepSeek V4 Pro", 0.7, 1.4, "frontier"),
    "z-ai/glm-5.3-uncensored": ("GLM 5.3 uncensored", 1.25, 2.25, "refusal-removed"),
    "abliteration-ai/abliterated-model-large": ("Abliterated Large", 3, 5, "refusal-removed"),
    "abliteration-ai/abliterated-model-large-flash": ("Abliterated Large Flash", 1, 2, "refusal-removed"),
}

# cells: name -> (seeds, llm_agents_per_game)
FULL = {"ISO-AWARE": (8, 2), "SHR-AWARE": (8, 2), "CHAT": (5, 2), "SHR-PERMITTED": (5, 2), "ORACLE": (3, 2),
        "SHR-NAIVE": (4, 2), "PROVOC": (4, 1), "SHR-GATE-TOLD": (4, 2)}
CORE = {"ISO-AWARE": (8, 2), "SHR-AWARE": (8, 2), "CHAT": (3, 2), "SHR-PERMITTED": (3, 2)}


def run_cost(pin: float, pout: float, agents: int) -> float:
    calls = ROUNDS * CALLS_PER_AGENT_ROUND * agents
    return calls * (IN_TOK * pin + OUT_TOK * pout) / 1e6


def grid_cost(pin: float, pout: float, cells: dict) -> tuple[float, int]:
    total, runs = 0.0, 0
    for seeds, agents in cells.values():
        total += seeds * run_cost(pin, pout, agents)
        runs += seeds
    return total, runs


print(f"Assumptions: {ROUNDS} rounds, {CALLS_PER_AGENT_ROUND} calls/agent/round, {IN_TOK} in + {OUT_TOK} out tokens/call\n")
print(f"{'model':26s} {'$/run':>7s} {'FULL $':>8s} {'CORE $':>8s}")
tot_full = tot_core = 0.0
rows = []
for mid, (label, pin, pout, grp) in MODELS.items():
    rc = run_cost(pin, pout, 2)
    f, nf = grid_cost(pin, pout, FULL)
    c, nc = grid_cost(pin, pout, CORE)
    tot_full += f
    tot_core += c
    rows.append((label, rc, f, c))
    print(f"{label:26s} {rc:7.2f} {f:8.1f} {c:8.1f}")
print(f"{'TOTAL (all 13 models)':26s} {'':7s} {tot_full:8.1f} {tot_core:8.1f}")
print(f"runs per model: FULL={grid_cost(1,1,FULL)[1]}  CORE={grid_cost(1,1,CORE)[1]}")

# Recommended mix: FULL on a core set, CORE on the rest
full_set = {"Claude Opus 5.5", "GPT-6.1 Sol", "Gemini 3.1 Pro", "GLM 5.3", "DeepSeek V4 Pro", "Abliterated Large",
            "GLM 5.3 uncensored"}
mix = sum(f if lab in full_set else c for lab, _, f, c in rows)
print(f"\nRECOMMENDED mix (FULL on {len(full_set)} models, CORE on the rest): ${mix:.0f}")
no_astra = sum((f if lab in full_set else c) for lab, _, f, c in rows if lab != "GPT-6 Astra")
print(f"RECOMMENDED without GPT-6 Astra: ${no_astra:.0f}")

# Power: two-sample, two-sided alpha=0.05, power 0.8 -> minimum detectable effect in SD units
z = norm.ppf(0.975) + norm.ppf(0.8)
for n in (4, 5, 8, 12):
    print(f"MDE per model, {n} vs {n} runs: {z*sqrt(2/n):.2f} SD")
for k in (8, 10, 12):
    n = 8 * k
    print(f"MDE pooled over {k} models (8 vs 8 each => {n} vs {n}): {z*sqrt(2/n):.2f} SD")
