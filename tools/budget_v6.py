"""Plan v6 budget: deep grid on 6 models, broad grid on 6 more. Prices: nanoGPT USD per 1M tokens (25 Oct 2026)."""
from math import sqrt

from scipy.stats import norm

DEEP = {"Claude Opus 5.5": (4, 20), "GPT-6.1 Sol": (2, 10), "Gemini 3.1 Pro": (2, 12), "GLM 5.3": (0.6, 1.1),
        "DeepSeek V4 Pro": (0.7, 1.4)}
BROAD = {"Claude Sonnet 5.5": (2, 10), "Grok 4.7": (2, 6), "Kimi K3.1 Pro": (2, 10), "Qwen 3.8 Max": (4, 12)}
PAIRS = {"GLM 5.2 (base)": (0.189, 0.594), "Abliterated Large (GLM-5.2-derived)": (3, 5),
         "DeepSeek V4.1 Flash 1017 (base)": (0.10, 0.60), "Abliterated Large Flash (V4.1-Flash-1017-based)": (1, 2)}
OPTIONAL = {"GLM 5.3 uncensored (descriptive)": (1.25, 2.25)}
ASTRA = {"GPT-6 Astra": (10, 50)}

# cell: (seeds, llm_firms)   llm_firms=1 for scripted-rival cells
DEEP_CELLS = {
    "ISO-AWARE": (8, 2), "SHR-AWARE": (8, 2),                                   # A carrier test
    "SHR-PERMITTED": (5, 2), "CHAT": (5, 2), "ORACLE": (3, 2),                  # B positive controls
    "CHAT-MON-SHR": (5, 2), "CHAT-MON-ISO": (5, 2),                             # C displacement (3.4)
    "PROVOC": (4, 1), "PROVOC-HINT": (3, 1), "PROVOC-BENIGN": (3, 1),           # D honeypot (3.2)
    "SHR-TOLD-UNGATED": (4, 2), "SHR-GATE-SILENT": (4, 2), "SHR-GATE-TOLD": (4, 2), "PROVOC-GATE-TOLD": (3, 1),  # E gate
    "SHR-NAIVE": (3, 2),                                                         # propensity, not told
}
BROAD_CELLS = {"ISO-AWARE": (6, 2), "SHR-AWARE": (6, 2), "SHR-PERMITTED": (4, 2), "CHAT-MON-SHR": (4, 2),
               "CHAT-MON-ISO": (4, 2), "PROVOC": (3, 1)}
PAIR_CELLS = {"ISO-AWARE": (6, 2), "SHR-AWARE": (6, 2), "SHR-PERMITTED": (4, 2), "CHAT": (4, 2), "CHAT-MON-SHR": (4, 2),
              "PROVOC": (4, 1), "PROVOC-GATE-TOLD": (3, 1), "ORACLE": (3, 2)}


def per_run(pin, pout, rounds, tin, tout, firms=2):
    return rounds * 2 * firms * (tin * pin + tout * pout) / 1e6


def grid(models, cells, rounds, tin, tout):
    tot, runs = 0.0, 0
    for pin, pout in models.values():
        for seeds, firms in cells.values():
            tot += seeds * per_run(pin, pout, rounds, tin, tout, firms)
            runs += seeds
    return tot, runs


print(f"runs/model: deep {sum(s for s, _ in DEEP_CELLS.values())}, broad {sum(s for s, _ in BROAD_CELLS.values())}, "
      f"pair {sum(s for s, _ in PAIR_CELLS.values())}")
for rounds in (30, 40):
    for tin, tout, tag in ((1500, 250, "slim"), (3000, 600, "conservative")):
        d, nd = grid(DEEP, DEEP_CELLS, rounds, tin, tout)
        b, nb = grid(BROAD, BROAD_CELLS, rounds, tin, tout)
        p, npr = grid(PAIRS, PAIR_CELLS, rounds, tin, tout)
        o, no = grid(OPTIONAL, PAIR_CELLS, rounds, tin, tout)
        a, na = grid(ASTRA, BROAD_CELLS, rounds, tin, tout)
        print(f"{rounds}r {tag:12s}: deep ${d:5.0f} ({nd}) + broad ${b:5.0f} ({nb}) + pairs ${p:5.0f} ({npr}) = "
              f"${d+b+p:5.0f}, games {nd+nb+npr}   [+GLM5.3-unc ${o:4.0f}]  [+Astra ${a:4.0f}]")

print("\nper-run cost, 40 rounds, slim tokens:")
for name, (pin, pout) in {**DEEP, **BROAD, **PAIRS, **OPTIONAL, **ASTRA}.items():
    print(f"  {name:48s} ${per_run(pin, pout, 40, 1500, 250):.2f}")

z = norm.ppf(0.975) + norm.ppf(0.8)
print("\nMDE (SD units), two-sided .05, power .80:")
for label, n in (("per model 8v8", 8), ("per model 5v5", 5), ("deep pooled 6x8=48v48", 48),
                 ("all 12 pooled (deep 8 + broad 6) = 84v84", 84), ("displacement deep 6x5=30v30", 30)):
    print(f"  {label:44s} {z*sqrt(2/n):.2f}")

# wall-clock: sequential calls per game = rounds * 2 firms * 2 calls (random order -> firms act one after another)
for rounds in (30, 40):
    calls = rounds * 2 * 2
    for sec in (6, 12):
        games = (len(DEEP) * sum(s for s, _ in DEEP_CELLS.values()) + len(BROAD) * sum(s for s, _ in BROAD_CELLS.values())
                 + len(PAIRS) * sum(s for s, _ in PAIR_CELLS.values()))
        for conc in (25, 40):
            hrs = games * calls * sec / conc / 3600
            print(f"wall-clock {rounds}r, {sec}s/call, {conc} concurrent: {hrs:.1f} h")
