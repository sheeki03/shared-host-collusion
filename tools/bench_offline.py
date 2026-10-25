"""Inspect/Docker timing gate with a scripted mock model and zero provider calls.

Read existing logs (safe, no Docker):
    .venv/bin/python tools/bench_offline.py analyze runs/selftest/logs

Run a new, explicitly requested mock-only Docker wave:
    .venv/bin/python tools/bench_offline.py run-mock --confirm-docker --games 3 --rounds 12 --concurrency 3

The benchmark never uses nanoGPT or a paid model. It creates one .eval log per wave under ignored runs/bench/.
Its concurrency projections assume ideal linear scaling and are deliberately labelled optimistic. The measured
game-rounds/hour and long-call counts, rather than tiny raw API probes, are the Phase 4 schedule gate.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOTAL_PLANNED_GAME_ROUNDS = 771 * 40 + 360 * 30  # 41,640 across the v6.4 full design
TARGET_CONCURRENCIES = (3, 10, 20, 30)
DOCKER_CLI_SLOTS = (10, 20, 30, 40)


def _instant(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _peak_overlap(intervals: Iterable[tuple[datetime, datetime]]) -> int:
    boundaries: list[tuple[datetime, int]] = []
    for start, end in intervals:
        if end >= start:
            boundaries.extend(((start, 1), (end, -1)))
    # End events precede starts at the same instant, so touching intervals do not overlap.
    count = peak = 0
    for _, change in sorted(boundaries, key=lambda pair: (pair[0], pair[1])):
        count += change
        peak = max(peak, count)
    return peak


def summarize_log(log: Any, *, label: str) -> dict[str, Any]:
    """Summarize one Inspect EvalLog without loading prompts or model outputs into the report."""
    start = _instant(getattr(getattr(log, "stats", None), "started_at", None))
    end = _instant(getattr(getattr(log, "stats", None), "completed_at", None))
    elapsed = (end - start).total_seconds() if start and end else None
    samples = list(getattr(log, "samples", None) or [])
    completed_rounds = 0
    failed_samples = 0
    sample_intervals: list[tuple[datetime, datetime]] = []
    sandbox_intervals: list[tuple[datetime, datetime]] = []
    action_intervals: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    timings: dict[str, list[float]] = defaultdict(list)
    slowest: list[dict[str, Any]] = []
    model_secs: list[float] = []
    missing_event_end = 0
    for sample in samples:
        if getattr(sample, "error", None):
            failed_samples += 1
        result = (getattr(sample, "metadata", None) or {}).get("result") or {}
        rows = result.get("rows") or []
        completed_rounds += len({row.get("t") for row in rows if isinstance(row, dict) and row.get("t") is not None})
        model_secs.extend(float(call["secs"]) for call in result.get("calls", [])
                          if isinstance(call, dict) and isinstance(call.get("secs"), (int, float)))
        sample_start = _instant(getattr(sample, "started_at", None))
        sample_end = _instant(getattr(sample, "completed_at", None))
        if sample_start and sample_end:
            sample_intervals.append((sample_start, sample_end))
        for event in getattr(sample, "events", None) or []:
            if getattr(event, "event", None) != "sandbox":
                continue
            event_start = _instant(getattr(event, "timestamp", None))
            event_end = _instant(getattr(event, "completed", None))
            if not event_start or not event_end:
                missing_event_end += 1
                continue
            seconds = max(0.0, (event_end - event_start).total_seconds())
            action = str(getattr(event, "action", None) or "unknown")
            timings[action].append(seconds)
            sandbox_intervals.append((event_start, event_end))
            action_intervals[action].append((event_start, event_end))
            slowest.append({"sample": str(getattr(sample, "id", "?")), "action": action,
                            "seconds": round(seconds, 3), "started_at": event_start.isoformat()})
    all_timings = [seconds for values in timings.values() for seconds in values]
    peak_samples = _peak_overlap(sample_intervals)
    throughput = completed_rounds * 3600 / elapsed if elapsed and elapsed > 0 else None
    return {
        "label": label, "status": str(getattr(log, "status", "unknown")),
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "samples": len(samples), "failed_samples": failed_samples,
        "completed_game_rounds": completed_rounds,
        "peak_sample_concurrency": peak_samples,
        "peak_sandbox_call_concurrency": _peak_overlap(sandbox_intervals),
        "game_rounds_per_hour": round(throughput, 3) if throughput is not None else None,
        "model_call_count": len(model_secs), "model_seconds_total": round(sum(model_secs), 3),
        "sandbox_call_count": len(all_timings), "sandbox_events_missing_completion": missing_event_end,
        "sandbox_seconds_total": round(sum(all_timings), 3),
        "sandbox_calls_per_game_round": round(len(all_timings) / completed_rounds, 3) if completed_rounds else None,
        "sandbox_seconds_per_game_round": round(sum(all_timings) / completed_rounds, 3) if completed_rounds else None,
        "sandbox_over_30s": sum(s > 30 for s in all_timings),
        "sandbox_over_60s": sum(s > 60 for s in all_timings),
        "sandbox_over_300s": sum(s > 300 for s in all_timings),
        "sandbox_by_action": {
            action: {"count": len(values), "p50_seconds": round(statistics.median(values), 3),
                     "p95_seconds": round(_percentile(values, .95) or 0, 3),
                     "max_seconds": round(max(values), 3),
                     "peak_concurrent": _peak_overlap(action_intervals[action]),
                     "over_30s": sum(s > 30 for s in values), "over_60s": sum(s > 60 for s in values)}
            for action, values in sorted(timings.items())
        },
        "slowest_sandbox_calls": sorted(slowest, key=lambda item: item["seconds"], reverse=True)[:8],
    }


def optimistic_projection(summary: dict[str, Any]) -> dict[int, float | None]:
    """Best-case linear scaling; not a deadline promise, especially when stalls were observed."""
    rate = summary.get("game_rounds_per_hour")
    observed = summary.get("peak_sample_concurrency")
    if not rate or not observed:
        return {c: None for c in TARGET_CONCURRENCIES}
    return {c: round(TOTAL_PLANNED_GAME_ROUNDS / (rate * c / observed), 2)
            for c in TARGET_CONCURRENCIES}


def docker_slot_lower_bound(summary: dict[str, Any]) -> dict[int, float | None]:
    """Idealized minimum time from sandbox-call occupancy alone, excluding provider and scheduling overhead."""
    seconds_per_round = summary.get("sandbox_seconds_per_game_round")
    if seconds_per_round is None:
        return {slots: None for slots in DOCKER_CLI_SLOTS}
    return {slots: round(TOTAL_PLANNED_GAME_ROUNDS * seconds_per_round / slots / 3600, 2)
            for slots in DOCKER_CLI_SLOTS}


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2))
    print("optimistic_linear_hours_for_41640_game_rounds_by_concurrency:",
          json.dumps(optimistic_projection(summary), sort_keys=True))
    print("idealized_docker_cli_slot_lower_bound_hours:",
          json.dumps(docker_slot_lower_bound(summary), sort_keys=True))
    if summary["sandbox_over_60s"] or summary["failed_samples"] or summary["status"] != "success":
        print("NOT READY: long sandbox calls, sample errors, or failed eval make this projection unreliable.")
    else:
        print("Scaling warning: no stalls in this wave does not prove 10/20/30-game concurrency will scale linearly.")


def analyze(path: Path) -> int:
    from inspect_ai.log import read_eval_log

    paths = [path] if path.is_file() else sorted(path.rglob("*.eval"))
    if not paths:
        raise SystemExit(f"no .eval logs found at {path}")
    bad = False
    for candidate in paths:
        summary = summarize_log(read_eval_log(str(candidate)), label=str(candidate))
        print_summary(summary)
        bad |= bool(summary["failed_samples"] or summary["status"] != "success" or summary["sandbox_over_60s"])
    return 2 if bad else 0


def run_mock(args: argparse.Namespace) -> int:
    if not args.confirm_docker:
        raise SystemExit("run-mock starts Docker containers; repeat with --confirm-docker")
    if not 1 <= args.games <= 30 or not 1 <= args.concurrency <= 30 or not 1 <= args.rounds <= 40:
        raise SystemExit("safety bounds: games 1..30, concurrency 1..30, rounds 1..40")
    if args.concurrency > args.games:
        print(f"note: effective sample concurrency is at most {args.games}")

    # The only model instantiated here is a scripted in-process mock. No provider keys are read or used.
    from inspect_ai import eval as inspect_eval
    from inspect_ai.model import get_model
    from harness.game import shc_games
    from harness.tests.e2e_offline import POLICY, mock_policy

    POLICY["mode"] = args.policy
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    wave = ROOT / "runs" / "bench" / f"{stamp}-{args.arm.lower()}-{args.policy}-{args.games}x{args.rounds}"
    wave.mkdir(parents=True, exist_ok=False)
    specs = [{"game_id": f"bench:{stamp}:{args.arm}:{args.policy}:{seed}", "grid": "selftest",
              "model_label": "mock", "model_id": "mockllm/model", "effort": "low",
              "scenario_id": "core-pricing", "engine": "pricing", "arm": args.arm,
              "seed": seed, "rounds": args.rounds}
             for seed in range(args.seed_start, args.seed_start + args.games)]
    spec_path = wave / "specs.json"
    spec_path.write_text(json.dumps(specs, indent=2))
    task = shc_games(str(spec_path), compose_dir=str(wave / "compose"),
                     monitor_model="active", subject_override="active")
    model = get_model("mockllm/model", custom_outputs=mock_policy)
    started = time.perf_counter()
    logs = inspect_eval(task, model=model, log_dir=str(wave / "logs"),
                        max_samples=args.concurrency, display="none", fail_on_error=False)
    local_elapsed = round(time.perf_counter() - started, 3)
    summaries = [summarize_log(log, label=f"{wave}/logs") for log in logs]
    if len(summaries) != 1:
        print(f"warning: expected one EvalLog, got {len(summaries)}")
    for summary in summaries:
        summary["invoker_elapsed_seconds"] = local_elapsed
        print_summary(summary)
    (wave / "timing_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"mock-only timing artifacts: {wave}")
    return 2 if any(s["failed_samples"] or s["status"] != "success" or s["sandbox_over_60s"]
                    for s in summaries) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    view = sub.add_parser("analyze", help="read existing Inspect .eval files; never starts Docker")
    view.add_argument("path", nargs="?", type=Path, default=ROOT / "runs" / "selftest" / "logs")
    run = sub.add_parser("run-mock", help="explicitly start Docker games with an in-process mock (no paid calls)")
    run.add_argument("--confirm-docker", action="store_true", help="required opt-in to start Docker containers")
    run.add_argument("--arm", choices=("SHR-AWARE", "ISO-AWARE"), default="SHR-AWARE")
    run.add_argument("--policy", choices=("leaky", "honest"), default="honest")
    run.add_argument("--games", type=int, default=3)
    run.add_argument("--rounds", type=int, default=12)
    run.add_argument("--concurrency", type=int, default=3)
    run.add_argument("--seed-start", type=int, default=9500)
    args = parser.parse_args(argv)
    return analyze(args.path) if args.action == "analyze" else run_mock(args)


if __name__ == "__main__":
    sys.exit(main())
