#!/usr/bin/env python3
"""Check live progress of matching parallel benchmark workers."""

import argparse
import sys
import time

from eleusis.analysis.live_progress import collect_live_progress
from eleusis.benchmark_run_store import BENCHMARK_RUN_DATABASE_NAME


def _print_report(pattern: str) -> None:
    """Render one progress report for matching worker folders."""
    workers = collect_live_progress(pattern)
    if not workers:
        print(
            f"Looking for: results/{pattern}/"
            f"{{results.json,{BENCHMARK_RUN_DATABASE_NAME}}}"
        )
        return

    total_completed = 0
    total_rounds = 0
    total_successful = 0
    total_score = 0

    print("=" * 70)
    print("ELEUSIS BENCHMARK PROGRESS")
    print("=" * 70)
    print()

    for worker in workers:
        if worker.error is not None:
            print(f"  {worker.name}: progress unavailable ({worker.error})")
            print()
            continue

        completed = worker.completed
        total = worker.total
        percentage = completed / total * 100 if total else 0.0

        total_completed += completed
        total_rounds += total
        total_successful += worker.successful
        total_score += worker.score

        if completed > 0 and completed < total:
            remaining = (total - completed) * worker.duration_seconds / completed
            eta = f"~{remaining / 3600:.1f}h remaining"
        elif completed >= total:
            eta = "DONE"
        else:
            eta = "starting..."

        bar_length = 30
        filled = int(bar_length * completed / total) if total else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        detail = ""
        if worker.active_round_number is not None:
            detail = (
                f" | active Round {worker.active_round_number}, "
                f"{worker.committed_turns} committed Turns"
            )
        print(f"  {worker.name}")
        print(
            f"    [{bar}] {completed}/{total} ({percentage:.0f}%) | "
            f"{worker.successful} wins | score {worker.score} | {eta}{detail}"
        )
        print()

    if total_rounds:
        overall_percentage = total_completed / total_rounds * 100
        success_rate = (
            total_successful / total_completed * 100 if total_completed else 0.0
        )
        average_score = total_score / total_completed if total_completed else 0.0
        print("-" * 70)
        print(
            f"  OVERALL: {total_completed}/{total_rounds} rounds "
            f"({overall_percentage:.0f}%)"
        )
        if total_completed:
            print(
                f"  Success rate: {success_rate:.1f}% "
                f"({total_successful}/{total_completed})"
            )
            print(f"  Avg score: {average_score:.1f}")
        print()


def _watch(pattern: str, interval: float) -> None:
    """Redraw the progress report every interval seconds until interrupted."""
    interval = max(1.0, interval)
    try:
        while True:
            sys.stdout.write("\x1b[2J\x1b[H")
            refreshed_at = time.strftime("%H:%M:%S")
            print(f"refreshed {refreshed_at} | every {interval:g}s | Ctrl+C to exit")
            _print_report(pattern)
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print()


def main() -> None:
    """Print progress summaries, preferring SQLite for new-format Runs."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--pattern",
        default="solo_evaluation_*w*qwen*",
        help=("Glob pattern for results/ worker folders (default: %(default)s)"),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep redrawing the report until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Seconds between redraws in watch mode (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.watch:
        _watch(args.pattern, args.interval)
    else:
        _print_report(args.pattern)


if __name__ == "__main__":
    main()
