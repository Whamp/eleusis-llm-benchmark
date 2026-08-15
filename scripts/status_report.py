#!/usr/bin/env python3
"""Status report for in-progress benchmark evaluations.

Merges results from parallel workers into a single model's data, identifies
completed rules (rules with all rounds finished across workers), filters
reference results to those same rules, and runs the full analysis pipeline.

A rule is "completed" when all workers have finished their round for that rule.
Incomplete rules are excluded from both the in-progress and reference models.

Usage:
    uv run python scripts/status_report.py \
        --reference results/260312_all_models_corrected \
        results/solo_evaluation_*_w*_rys-qwen3.5-27b-fp8-xl
"""

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from eleusis.analysis import analyze_folder
from eleusis.evaluation_results import (
    ConsumedRule,
    EvaluationResults,
    EvaluationStatistics,
    SavedRound,
    parse_evaluation_results,
)
from eleusis.game.rule_library import RuleLibraryEntry

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse worker folders and status-report output options."""
    parser = argparse.ArgumentParser(
        description="Generate status report for in-progress benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run python scripts/status_report.py \
    --reference results/260312_all_models_corrected \
    results/solo_evaluation_*_w*_rys-qwen3.5-27b-fp8-xl
""",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference folder with completed benchmark results",
    )
    parser.add_argument(
        "workers",
        nargs="+",
        type=Path,
        help="Worker result folders to merge",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder (default: status/ inside first worker folder)",
    )
    return parser.parse_args()


def get_integer_metric(metrics: dict[str, object], key: str) -> int:
    """Return an integer metric value, treating missing or invalid values as zero."""
    value = metrics.get(key, 0)
    return value if isinstance(value, int) else 0


def load_worker_results(folders: list[Path]) -> list[EvaluationResults]:
    """Load results.json from each worker folder."""
    results = []
    for folder in folders:
        rfile = folder / "results.json"
        if not rfile.exists():
            logger.warning(f"No results.json in {folder.name}, skipping")
            continue
        try:
            with rfile.open() as results_file:
                results.append(parse_evaluation_results(json.load(results_file)))
        except json.JSONDecodeError:
            logger.warning(f"Corrupt results.json in {folder.name}, skipping")
    return results


def _merge_worker_statistics(rounds: list[SavedRound]) -> EvaluationStatistics:
    """Recompute aggregate statistics from merged worker rounds."""
    statistics: EvaluationStatistics = {
        "total_score": 0,
        "successful_rounds": 0,
        "failed_rounds": 0,
        "total_turns": 0,
        "total_failed_guesses": 0,
        "total_output_tokens": 0,
        "total_reasoning_tokens": 0,
        "total_answer_tokens": 0,
        "total_wall_clock_seconds": 0.0,
        "total_retries": 0,
        "retry_by_cause": {},
    }
    for round_data in rounds:
        statistics["total_score"] += round_data["score"]
        key = "successful_rounds" if round_data["success"] else "failed_rounds"
        statistics[key] += 1
        statistics["total_turns"] += round_data["turn_count"]
        statistics["total_failed_guesses"] += round_data["failed_guesses"]
        statistics["total_wall_clock_seconds"] += round_data["wall_clock_seconds"]
        player_usage = round_data["llm_usage"].get("player", {})
        statistics["total_output_tokens"] += get_integer_metric(
            player_usage, "output_tokens"
        )
        statistics["total_reasoning_tokens"] += get_integer_metric(
            player_usage, "reasoning_tokens"
        )
        statistics["total_answer_tokens"] += get_integer_metric(
            player_usage, "answer_tokens"
        )
        for turn in round_data["turns"]:
            statistics["total_retries"] += turn["retry_count"]
            for retry in turn["retry_causes"]:
                cause_value = retry.get("cause", "unknown")
                cause = cause_value if isinstance(cause_value, str) else "unknown"
                counts = statistics["retry_by_cause"]
                counts[cause] = counts.get(cause, 0) + 1
    return statistics


def _merge_consumed_rules(
    worker_results: list[EvaluationResults],
) -> list[ConsumedRule]:
    """Deduplicate consumed rules by rule name across workers."""
    merged: list[ConsumedRule] = []
    seen: set[str | None] = set()
    for worker in worker_results:
        for rule in worker["checkpoint"]["rules_consumed"]:
            if rule["name"] not in seen:
                merged.append(rule)
                seen.add(rule["name"])
    return merged


def _merge_rule_library(
    worker_results: list[EvaluationResults],
) -> list[RuleLibraryEntry]:
    """Deduplicate embedded library entries by description across workers."""
    merged: list[RuleLibraryEntry] = []
    seen: set[str] = set()
    for worker in worker_results:
        for rule in worker["checkpoint"]["rules_library"]:
            if rule["description"] not in seen:
                merged.append(rule)
                seen.add(rule["description"])
    return merged


def merge_workers(
    worker_results: list[EvaluationResults],
) -> EvaluationResults:
    """Merge multiple worker results into a single result set.

    Combines rounds, aggregates statistics, renumbers rounds sequentially. Config and
    checkpoint are taken from the first worker.
    """
    base = copy.deepcopy(worker_results[0])

    # Collect all rounds from all workers
    all_rounds = []
    for w in worker_results:
        all_rounds.extend(w.get("rounds", []))

    # Sort by rule_name then batch_round_index for deterministic ordering
    all_rounds.sort(
        key=lambda r: (r.get("rule_name", ""), r.get("batch_round_index", 0))
    )

    # Renumber sequentially
    for i, r in enumerate(all_rounds):
        r["round_number"] = i + 1

    base["rounds"] = all_rounds

    base["statistics"] = _merge_worker_statistics(all_rounds)
    base["checkpoint"]["rules_consumed"] = _merge_consumed_rules(worker_results)
    base["checkpoint"]["rules_library"] = _merge_rule_library(worker_results)

    return base


def find_completed_rules(merged: EvaluationResults, num_workers: int) -> set[str]:
    """Find rule descriptions that have all expected rounds (one per worker)."""
    rule_counts = Counter(r["rule_description"] for r in merged["rounds"])
    return {desc for desc, count in rule_counts.items() if count >= num_workers}


def filter_to_rules(
    results: EvaluationResults, completed_rules: set[str]
) -> EvaluationResults:
    """Filter a results dict to only include rounds for the given rules."""
    filtered = copy.deepcopy(results)
    filtered["rounds"] = [
        r for r in filtered["rounds"] if r["rule_description"] in completed_rules
    ]

    # Recompute statistics
    stats = filtered.get("statistics", {})
    stats["total_score"] = sum(r.get("score", 0) for r in filtered["rounds"])
    stats["successful_rounds"] = sum(1 for r in filtered["rounds"] if r.get("success"))
    stats["failed_rounds"] = sum(1 for r in filtered["rounds"] if not r.get("success"))
    stats["total_turns"] = sum(r.get("turn_count", 0) for r in filtered["rounds"])
    stats["total_failed_guesses"] = sum(
        r.get("failed_guesses", 0) for r in filtered["rounds"]
    )

    return filtered


def main() -> int:
    """Merge completed worker rules and run comparative analysis."""
    args = parse_args()

    if not args.reference.exists():
        print(f"Error: Reference folder not found: {args.reference}")
        return 1

    # Load worker results
    logger.info("Loading worker results...")
    worker_results = load_worker_results(args.workers)
    if not worker_results:
        print("No results found in any worker folder.")
        print("Workers may still be on their first round.")
        return 1

    logger.info(f"Loaded {len(worker_results)} worker(s)")
    num_workers = len(args.workers)

    # Merge into single result set
    merged = merge_workers(worker_results)
    model_name = merged["config"]["player"]
    model_key = merged["config"]["player_model"]

    # Append (in-progress) to avoid collision with reference models of same name
    merged["config"]["player"] = f"{model_name} (in-progress)"
    logger.info(
        f"Renamed to '{merged['config']['player']}' to distinguish from reference"
    )

    total_rounds = len(merged["rounds"])
    logger.info(f"Merged: {model_name} ({model_key}), {total_rounds} rounds")

    # Find completed rules
    completed_rules = find_completed_rules(merged, num_workers)
    all_rules_seen = {r["rule_description"] for r in merged["rounds"]}
    incomplete_rules = all_rules_seen - completed_rules

    if not completed_rules:
        print("No completed rules yet.")
        print(
            f"Workers have data for {len(all_rules_seen)} rule(s), but none have all"
            f" {num_workers} rounds."
        )
        return 1

    logger.info(f"Completed: {len(completed_rules)}/26 rules")
    if incomplete_rules:
        logger.info(f"Incomplete: {len(incomplete_rules)} rule(s) waiting for workers")

    # Filter merged results to only completed rules
    filtered_merged = filter_to_rules(merged, completed_rules)

    # Set up output folder
    status_folder = args.output or args.workers[0] / "status"
    status_folder.mkdir(parents=True, exist_ok=True)

    # Save merged in-progress results as a solo_evaluation_* subfolder
    safe_name = model_key.lower().replace("-", "_").replace(".", "_")
    in_progress_dir = status_folder / f"solo_evaluation_in_progress_{safe_name}"
    in_progress_dir.mkdir(exist_ok=True)
    with (in_progress_dir / "results.json").open("w") as results_file:
        json.dump(filtered_merged, results_file, indent=2)

    # Copy and filter reference results
    ref_count = 0
    ref_folder = Path(args.reference)
    for sub in sorted(ref_folder.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("solo_evaluation_")):
            continue
        rfile = sub / "results.json"
        if not rfile.exists():
            continue

        with rfile.open() as results_file:
            ref_data = parse_evaluation_results(json.load(results_file))

        ref_filtered = filter_to_rules(ref_data, completed_rules)
        ref_dir = status_folder / sub.name
        ref_dir.mkdir(exist_ok=True)
        with (ref_dir / "results.json").open("w") as results_file:
            json.dump(ref_filtered, results_file, indent=2)
        ref_count += 1

    logger.info(f"Reference models: {ref_count}")
    logger.info(
        f"Comparing on {len(completed_rules)} rules ({len(completed_rules) * 3} rounds"
        " per model)"
    )

    # Run full analysis
    logger.info("")
    analyze_folder(status_folder)

    print(f"\nStatus report: {status_folder}")
    print(f"Completed rules: {len(completed_rules)}/26")
    print(f"In-progress model rounds: {len(filtered_merged['rounds'])}")
    print(f"Reference models: {ref_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
