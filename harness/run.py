"""Run games: pilot, confirmatory grid, or a subset.

Usage examples (from repo root, venv python):
  .venv/bin/python -m harness.run --plan pilot-probes --cap 40 --concurrency 13 --init-ledger
  .venv/bin/python -m harness.run --plan pilot-smoke  --cap 40 --concurrency 12
  .venv/bin/python -m harness.run --plan full --grid deep --cap 400 --previous-cap 40 \
    --raise-cap-reason 'owner-approved post-pilot funding' --concurrency 20
  .venv/bin/python -m harness.run --plan ext --ext-grid all --cap 1 --dry-run      # the 216 extension games

Extensions (prereg/EXTENSIONS.md): `--plan ext --ext-grid {rep,long,any,sx,all}` selects the pre-declared extension
games of harness/arms.py::ext_design() (144 / 24 / 16 / 32; 216 for all). Their ids start with `ext-` and never
collide with full_design(). They write specs to runs/ext/, log to runs/ext/logs and use runs/ext/ledger.jsonl unless
--log-dir / --ledger say otherwise (never the confirmatory runs/full or runs/ledger.jsonl defaults). With
`--ext-grid all` the 100-round ext-long games come first (the critical path), then the other grids interleaved.
--shard-file, --exclude-ids-file, --models and --arms work as for the other plans; --grid does not apply.

Budget: every subject/monitor call reserves the full advertised input-context cost plus bounded output under the
pinned price table before sending, then settles the provider-reported charge. This is a conservative local admission
limit conditional on provider context/prices; an above-reservation charge is recorded and halts future calls.
Pilot seeds (9000+) never enter confirmatory estimates. A ledger is initialized once and resumed explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from harness.arms import (BROAD, CORE_ROUNDS, DEEP, EXT_GRID_NAMES, ORIGINAL_ARMS, PAIRS, PILOT_SEED_BASE,
                          SCENARIOS, GameSpec, ext_design, full_design)

ROOT = Path(__file__).resolve().parent.parent


def open_run_ledger(path: str, cap: float, *, init: bool = False, previous_cap: float | None = None,
                    raise_reason: str | None = None):
    """Open a paid-run ledger without silently resetting it or changing its persisted cap."""
    from harness.ledger import BudgetLedger

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if init:
        if previous_cap is not None or raise_reason:
            raise ValueError("--init-ledger cannot be combined with a cap increase")
        # O_EXCL makes an accidental second initialization fail even if another runner starts at the same time.
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        return BudgetLedger(target, cap, create=True)
    if raise_reason:
        if previous_cap is None:
            raise ValueError("an audited cap increase needs --previous-cap")
        led = BudgetLedger(target, previous_cap, create=False)
        led.raise_cap(cap, reason=raise_reason)
        return led
    if previous_cap is not None:
        raise ValueError("--previous-cap requires --raise-cap-reason")
    return BudgetLedger(target, cap, create=False)


def required_headroom(specs: list[GameSpec], concurrency: int) -> tuple[float, float]:
    """Return largest per-call reservation and one-wave headroom at the requested game concurrency."""
    from harness.game import max_tokens_for
    from harness.nanogpt_provider import estimate_max_cost, load_prices

    prices = load_prices()
    # 64 KB / 12 messages bounds any game prompt (history capped at 5 rounds, notes at 2,000 chars, outputs 2,000).
    monitor = estimate_max_cost(prices["z-ai/glm-5.3"], 8_000, 80, 2)
    reservations = []
    for s in specs:
        if s.model_id not in prices:
            raise ValueError(f"no price bound for {s.model_id}")
        subject = estimate_max_cost(prices[s.model_id], 65_536, max_tokens_for(s.model_id), 12)
        reservations.append(max(subject, monitor))
    if not reservations:
        raise ValueError("the selected plan has no games")
    reservations.sort(reverse=True)
    return reservations[0], sum(reservations[:concurrency])


def completed_cost(logs) -> float:
    """Validate every sample and sum all provider attempts from saved per-game metadata."""
    from inspect_ai.log import read_eval_log

    total = 0.0
    errors = []
    for log in logs:
        if log.status != "success" or log.error:
            errors.append(f"eval status {log.status}: {str(log.error)[:150]}")
        samples = log.samples or read_eval_log(log.location).samples or []
        if not samples:
            errors.append(f"no saved samples in {log.location}")
        for sample in samples:
            if sample.error:
                errors.append(f"sample {sample.id}: {str(sample.error.message)[:150]}")
                continue
            result = (sample.metadata or {}).get("result")
            if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
                errors.append(f"sample {sample.id}: missing game calls/results")
                continue
            for call in result["calls"]:
                cost = call.get("total_cost_usd")
                if cost is None:
                    errors.append(f"sample {sample.id}: call missing total_cost_usd")
                else:
                    total += float(cost)
    if errors:
        raise RuntimeError("incomplete Inspect evaluation; no result may be reported as complete:\n" +
                           "\n".join(errors[:20]))
    return total


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
                    PILOT_SEED_BASE + 1, 5) for arm in ORIGINAL_ARMS]
    out.append(GameSpec(f"pilot-full:{label}:SHR-AWARE", "pilot", label, mid, effort, "core-pricing", "pricing",
                        "SHR-AWARE", PILOT_SEED_BASE + 2, CORE_ROUNDS))
    for scen, engine in SCENARIOS.items():
        out.append(GameSpec(f"pilot-scenario:{label}:{scen}", "pilot", label, mid, effort, scen, engine, "SHR-AWARE",
                            PILOT_SEED_BASE + 3, 3))
    return out


def ext_plan(ext_grid: str) -> list[GameSpec]:
    """The pre-declared extension games of one grid or all four, checked against the confirmatory ids."""
    specs = ext_design(ext_grid)
    ids = [s.game_id for s in specs]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate extension game ids")
    clash = set(ids) & {s.game_id for s in full_design()}
    if clash or not all(s.grid.startswith("ext-") and s.game_id.startswith(s.grid + ":") for s in specs):
        raise SystemExit(f"extension ids must be ext-* and disjoint from full_design(); clash {sorted(clash)[:3]}")
    return specs


def select(plan: str, grid: str | None, models: list[str] | None, arms: list[str] | None,
           ext_grid: str | None = None) -> list[GameSpec]:
    if plan == "pilot-probes":
        specs = pilot_probes()
    elif plan == "pilot-smoke":
        specs = pilot_smoke()
    elif plan == "full":
        specs = full_design()
    elif plan == "ext":
        specs = ext_plan(ext_grid or "all")
    else:
        raise SystemExit(f"unknown plan {plan}")
    if grid:
        specs = [s for s in specs if s.grid == grid]
    if models:
        specs = [s for s in specs if s.model_label in models]
    if arms:
        specs = [s for s in specs if s.arm in arms]
    if plan == "ext":   # prereg/EXTENSIONS.md section 9: the 100-round games are launched first
        return interleave([s for s in specs if s.grid == "ext-long"]) + \
            interleave([s for s in specs if s.grid != "ext-long"])
    return interleave(specs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, choices=["pilot-probes", "pilot-smoke", "full", "ext"])
    ap.add_argument("--grid", choices=["deep", "broad", "pairs", "scenario"])
    ap.add_argument("--ext-grid", choices=[*EXT_GRID_NAMES, "all"],
                    help="with --plan ext (required there): one pre-declared extension grid, or all four")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--cap", type=float, required=True, help="hard USD cap for this ledger")
    ap.add_argument("--ledger", default=None,
                    help="default runs/ledger.jsonl (runs/ext/ledger.jsonl with --plan ext)")
    ap.add_argument("--init-ledger", action="store_true", help="create a new ledger once; never resets existing spend")
    ap.add_argument("--previous-cap", type=float, help="recorded cap before an audited increase")
    ap.add_argument("--raise-cap-reason", help="audited reason for increasing --previous-cap to --cap")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard-file", help="JSON {host, game_ids}: run only these games, in this order")
    ap.add_argument("--exclude-ids-file", help="JSON list of game_ids to skip (already completed elsewhere)")
    a = ap.parse_args()
    if a.concurrency < 1:
        ap.error("--concurrency must be positive")
    ext = a.plan == "ext"
    if ext and not a.ext_grid:
        ap.error("--plan ext needs --ext-grid {" + ",".join([*EXT_GRID_NAMES, "all"]) + "}")
    if a.ext_grid and not ext:
        ap.error("--ext-grid applies only to --plan ext")
    if ext and a.grid:
        ap.error("--grid does not apply to --plan ext (use --ext-grid)")
    if a.ledger is None:
        a.ledger = str(ROOT / "runs" / "ext" / "ledger.jsonl") if ext else str(ROOT / "runs" / "ledger.jsonl")

    specs = select(a.plan, a.grid, a.models, a.arms, a.ext_grid)
    tag = f"ext-{a.ext_grid}" if ext else (a.plan if a.plan != "full" else f"full-{a.grid or 'all'}")
    if a.shard_file:
        shard = json.loads(Path(a.shard_file).read_text())
        by_id = {s.game_id: s for s in specs}
        missing = [g for g in shard["game_ids"] if g not in by_id]
        if missing:
            raise SystemExit(f"shard names {len(missing)} games not in the selected plan, e.g. {missing[:3]}")
        specs = [by_id[g] for g in shard["game_ids"]]
        tag += f"-{shard['host']}"
    if a.exclude_ids_file:
        done = set(json.loads(Path(a.exclude_ids_file).read_text()))
        specs = [s for s in specs if s.game_id not in done]
        tag += "-resume"
    run_dir = ROOT / "runs" / ("ext" if ext else "pilot" if a.plan.startswith("pilot") else "full")
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / f"specs_{tag}.json"
    spec_path.write_text(json.dumps([s.to_dict() for s in specs], indent=1))
    print(f"{len(specs)} games -> {spec_path}")
    if ext:
        per = defaultdict(int)
        for s in specs:
            per[s.grid] += 1
        print("per grid: " + ", ".join(f"ext-{n} {per[f'ext-{n}']}" for n in EXT_GRID_NAMES)
              + f"; rounds {sum(s.rounds for s in specs):,}")
    if a.dry_run:
        return

    led = open_run_ledger(a.ledger, a.cap, init=a.init_ledger, previous_cap=a.previous_cap,
                          raise_reason=a.raise_cap_reason)
    peak, wave = required_headroom(specs, a.concurrency)
    print(f"ledger {a.ledger}: cap ${led.cap_usd:.2f}, spent ${led.spent:.4f}, "
          f"outstanding ${led.outstanding:.4f}, largest call reservation ${peak:.4f}, "
          f"one-wave headroom ${wave:.4f}")
    if led.remaining < wave:
        raise SystemExit(f"available budget ${led.remaining:.4f} cannot reserve the requested "
                         f"{a.concurrency}-game wave (${wave:.4f}); lower concurrency or explicitly increase cap")
    before = led.spent
    os.environ["SHC_LEDGER_PATH"] = a.ledger
    os.environ["SHC_BUDGET_CAP_USD"] = str(a.cap)
    from inspect_ai import eval as inspect_eval  # noqa: WPS433

    from harness.game import shc_games  # noqa: WPS433
    import harness.nanogpt_provider  # noqa: F401,WPS433  (fail early if the provider is missing)

    log_dir = a.log_dir or str(run_dir / "logs")
    logs = inspect_eval(shc_games(str(spec_path), compose_dir=str(ROOT / "runs" / "compose")),
                        model="mockllm/model",  # each game binds its own nanogpt model in the solver
                        log_dir=log_dir, max_samples=a.concurrency, max_sandboxes=a.concurrency,
                        fail_on_error=False, display="plain", log_realtime=False,
                        log_buffer=5,  # completed games reach the log file every 5 samples (crash loses little)
                        retry_on_error=2)  # a game that errors (e.g. network drop) is replayed from round 1, same seed (D13)
    used = completed_cost(logs)
    billed = led.spent - before
    if led.open_reservations() or abs(used - billed) > 1e-8:
        raise SystemExit(f"budget reconciliation failed: saved calls ${used:.6f}, ledger delta ${billed:.6f}, "
                         f"open reservations {len(led.open_reservations())}")
    print(f"completed {len(specs)} games; provider charges ${billed:.6f}; ledger flags {led.flags}")


if __name__ == "__main__":
    main()
