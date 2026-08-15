#!/usr/bin/env python3
"""Check progress of parallel benchmark workers."""

import glob
import json


def main() -> None:
    """Print progress summaries for matching Qwen evaluation runs."""
    results = sorted(glob.glob("results/solo_evaluation_*w*qwen*/results.json"))

    if not results:
        print("No benchmark results found yet.")
        print("Looking for: results/solo_evaluation_*w*qwen*/results.json")
        return

    total_completed = 0
    total_rounds = 0
    total_successful = 0
    total_score = 0

    print("=" * 70)
    print("ELEUSIS BENCHMARK PROGRESS")
    print("=" * 70)
    print()

    for path in results:
        folder = path.split("/")[-2]
        with open(path) as f:
            data = json.load(f)

        chk = data["checkpoint"]
        completed = chk["completed_rounds"]
        total = chk["total_rounds"]
        stats = data.get("statistics", {})
        success = stats.get("successful_rounds", 0)
        score = stats.get("total_score", 0)
        pct = (completed / total * 100) if total > 0 else 0

        total_completed += completed
        total_rounds += total
        total_successful += success
        total_score += score

        # Estimate remaining time from wall clock
        wall = stats.get("total_wall_clock_seconds", 0)
        if completed > 0 and completed < total:
            per_round = wall / completed
            remaining = (total - completed) * per_round
            eta = f"~{remaining / 3600:.1f}h remaining"
        elif completed >= total:
            eta = "DONE"
        else:
            eta = "starting..."

        bar_len = 30
        filled = int(bar_len * completed / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"  {folder}")
        print(
            f"    [{bar}] {completed}/{total} ({pct:.0f}%) | {success} wins | score"
            f" {score} | {eta}"
        )
        print()

    if total_rounds > 0:
        overall_pct = total_completed / total_rounds * 100
        success_rate = (
            (total_successful / total_completed * 100) if total_completed > 0 else 0
        )
        avg_score = total_score / total_completed if total_completed > 0 else 0

        print("-" * 70)
        print(
            f"  OVERALL: {total_completed}/{total_rounds} rounds ({overall_pct:.0f}%)"
        )
        if total_completed > 0:
            print(
                f"  Success rate: {success_rate:.1f}%"
                f" ({total_successful}/{total_completed})"
            )
            print(f"  Avg score: {avg_score:.1f}")
        print()


if __name__ == "__main__":
    main()
