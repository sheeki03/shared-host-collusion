"""Run games: pilot, confirmatory grid, or a subset.

Usage examples (from repo root, venv python):
  .venv/bin/python -m harness.run --plan pilot-probes --cap 5 --concurrency 13
  .venv/bin/python -m harness.run --plan pilot-smoke  --cap 10 --concurrency 12
  .venv/bin/python -m harness.run --plan full --grid deep --cap 400 --concurrency 20

Budget: every subject/monitor call is reserved against a global append-only ledger before it is sent and settled
with the provider's exact cost (harness/ledger.py). The cap is enforced across all concurrent games; a call that
would exceed it raises and the game errors (fail closed). Pilot seeds (9000+) never enter confirmatory estimates.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from harness.arms import (ARMS, BROAD, CORE_ROUNDS, DEEP, PAIRS, PILOT_SEED_BASE, SCENARIO_MODELS, SCENARIOS, GameSpec,
                          full_design)

ROOT = Path(__file__).resolve().parent.parent


def interleave(specs: list[GameSpec], seed: int = 0) -> list[GameSpec]:
    """Round-robin across (model, scenario) streams and, within each, across arms at the same seed, so matched arms
    run close together in time and no arm is systematically early or late."""
    rng = random.Random(seed)
    streams: dict = defaultdict(lambda: defaultdict(list))
    for s in specs:
        streams[(s.model_label, s.scenario_id)][s.seed].append(s)
    queues = []
    for key in sorted(streams):
        q = []
        for sd in sorted(streams[key]):
            batch = streams[key][sd][:]
            rng.shuffle(batch)
            q += batch
        queues.append(q)
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def pilot_probes() -> list[GameSpec]:
    """5-round SHR-AWARE game per model configuration (availability, JSON, latency, billed reasoning)."""
    out, seen = [], set()
    for grid, models in (("deep", DEEP), ("broad", BROAD), ("pairs", PAIRS)):
        for label, mid, effort in models:
            if label in seen:
                continue
            seen.add(label)
            out.append(GameSpec(f"pilot-probe:{label}", "pilot", label, mid, effort, "core-pricing", "pricing",
                                "SHR-AWARE", PILOT_SEED_BASE, 5))
    return out


def pilot_smoke() -> list[GameSpec]:
    """5-round smoke games for every arm on one inexpensive model, plus one full 40-round game for prompt growth."""
    label, mid, effort = ("glm-5.3", "z-ai/glm-5.3", "low")
    out = [GameSpec(f"pilot-smoke:{label}:{arm}", "pilot", label, mid, effort, "core-pricing", "pricing", arm,
                    PILOT_SEED_BASE + 1, 5) for arm in ARMS]
    out.append(GameSpec(f"pilot-full:{label}:SHR-AWARE", "pilot", label, mid, effort, "core-pricing", "pricing",
                        "SHR-AWARE", PILOT_SEED_BASE + 2, CORE_ROUNDS))
    for scen, engine in SCENARIOS.items():
        out.append(GameSpec(f"pilot-scenario:{label}:{scen}", "pilot", label, mid, effort, scen, engine, "SHR-AWARE",
                            PILOT_SEED_BASE + 3, 3))
    return out


def select(plan: str, grid: str | None, models: list[str] | None, arms: list[str] | None) -> list[GameSpec]:
    if plan == "pilot-probes":
        specs = pilot_probes()
    elif plan == "pilot-smoke":
        specs = pilot_smoke()
    elif plan == "full":
        specs = full_design()
    else:
        raise SystemExit(f"unknown plan {plan}")
    if grid:
        specs = [s for s in specs if s.grid == grid]
    if models:
        specs = [s for s in specs if s.model_label in models]
    if arms:
        specs = [s for s in specs if s.arm in arms]
    return interleave(specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pilot-probes", "pilot-smoke", "full"])
    ap.add_argument("--grid", choices=["deep", "broad", "pairs", "scenario"])
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--cap", type=float, required=True, help="hard USD cap for this ledger")
    ap.add_argument("--ledger", default=str(ROOT / "runs" / "ledger.jsonl"))
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    specs = select(a.plan, a.grid, a.models, a.arms)
    tag = a.plan if a.plan != "full" else f"full-{a.grid or 'all'}"
    run_dir = ROOT / "runs" / ("pilot" if a.plan.startswith("pilot") else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / f"specs_{tag}.json"
    spec_path.write_text(json.dumps([s.to_dict() for s in specs], indent=1))
    print(f"{len(specs)} games -> {spec_path}")
    if a.dry_run:
        return

    os.environ["SHC_LEDGER_PATH"] = a.ledger
    os.environ["SHC_BUDGET_CAP_USD"] = str(a.cap)
    from inspect_ai import eval as inspect_eval  # noqa: WPS433

    from harness.game import shc_games  # noqa: WPS433
    import harness.nanogpt_provider  # noqa: F401,WPS433  (fail early if the provider is missing)

    log_dir = a.log_dir or str(run_dir / "logs")
    inspect_eval(shc_games(str(spec_path), compose_dir=str(ROOT / "runs" / "compose")),
                 model="mockllm/model",  # placeholder: each game binds its own nanogpt model in the solver
                 log_dir=log_dir, max_samples=a.concurrency, max_sandboxes=a.concurrency,
                 fail_on_error=False, display="plain", log_realtime=False)


if __name__ == "__main__":
    main()
