#!/usr/bin/env python3
"""Check live progress of matching parallel benchmark workers."""

import glob
import json
from pathlib import Path
from typing import cast

from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
    BenchmarkRunStoreError,
)


def _matching_worker_folders() -> list[Path]:
    """Find matching legacy exports and authoritative SQLite worker folders."""
    artifact_patterns = (
        "results/solo_evaluation_*w*qwen*/results.json",
        f"results/solo_evaluation_*w*qwen*/{BENCHMARK_RUN_DATABASE_NAME}",
    )
    return sorted(
        {
            Path(artifact).parent
            for pattern in artifact_patterns
            for artifact in glob.glob(pattern)
        }
    )


def _legacy_worker_progress(folder: Path) -> dict[str, object]:
    """Read one historical JSON-only worker progress projection."""
    with (folder / "results.json").open() as results_file:
        data = json.load(results_file)
    checkpoint = cast(dict[str, object], data["checkpoint"])
    statistics = cast(dict[str, object], data.get("statistics", {}))
    return {
        "completed": checkpoint["completed_rounds"],
        "total": checkpoint["total_rounds"],
        "successful": statistics.get("successful_rounds", 0),
        "score": statistics.get("total_score", 0),
        "duration_seconds": statistics.get("total_wall_clock_seconds", 0.0),
        "active_detail": None,
    }


def _sqlite_worker_progress(folder: Path) -> dict[str, object]:
    """Read live Round and committed-Turn progress from authoritative SQLite."""
    store = BenchmarkRunStore(folder)
    progress = store.read_progress()
    summary = store.read_derived_summary()
    usage = cast(dict[str, object], summary["usage"])
    active_detail = None
    if progress.active_round_number is not None:
        active_detail = (
            f"active Round {progress.active_round_number}, "
            f"{progress.committed_turns} committed Turns"
        )
    return {
        "completed": progress.completed_rounds,
        "total": progress.total_rounds,
        "successful": summary["successful_rounds"],
        "score": summary["total_score"],
        "duration_seconds": usage["duration_seconds"],
        "active_detail": active_detail,
    }


def main() -> None:
    """Print progress summaries, preferring SQLite for new-format Runs."""
    folders = _matching_worker_folders()
    if not folders:
        print("No benchmark results found yet.")
        print(
            "Looking for: results/solo_evaluation_*w*qwen*/"
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

    for folder in folders:
        try:
            progress = (
                _sqlite_worker_progress(folder)
                if (folder / BENCHMARK_RUN_DATABASE_NAME).is_file()
                else _legacy_worker_progress(folder)
            )
        except (
            BenchmarkRunStoreError,
            OSError,
            json.JSONDecodeError,
            KeyError,
        ) as error:
            print(f"  {folder.name}: progress unavailable ({error})")
            print()
            continue

        completed = cast(int, progress["completed"])
        total = cast(int, progress["total"])
        successful = cast(int, progress["successful"])
        score = cast(int, progress["score"])
        duration_seconds = cast(float, progress["duration_seconds"])
        active_detail = progress["active_detail"]
        percentage = completed / total * 100 if total else 0.0

        total_completed += completed
        total_rounds += total
        total_successful += successful
        total_score += score

        if completed > 0 and completed < total:
            remaining = (total - completed) * duration_seconds / completed
            eta = f"~{remaining / 3600:.1f}h remaining"
        elif completed >= total:
            eta = "DONE"
        else:
            eta = "starting..."

        bar_length = 30
        filled = int(bar_length * completed / total) if total else 0
        bar = "█" * filled + "░" * (bar_length - filled)
        detail = f" | {active_detail}" if isinstance(active_detail, str) else ""
        print(f"  {folder.name}")
        print(
            f"    [{bar}] {completed}/{total} ({percentage:.0f}%) | "
            f"{successful} wins | score {score} | {eta}{detail}"
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


if __name__ == "__main__":
    main()
