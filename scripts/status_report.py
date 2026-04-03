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
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from eleusis.analysis import analyze_folder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
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
        "--reference", required=True, type=Path,
        help="Reference folder with completed benchmark results",
    )
    parser.add_argument(
        "workers", nargs="+", type=Path,
        help="Worker result folders to merge",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output folder (default: status/ inside first worker folder)",
    )
    return parser.parse_args()


def load_worker_results(folders: list[Path]) -> list[dict]:
    """Load results.json from each worker folder."""
    results = []
    for folder in folders:
        rfile = folder / "results.json"
        if not rfile.exists():
            logger.warning(f"No results.json in {folder.name}, skipping")
            continue
        try:
            with open(rfile) as f:
                results.append(json.load(f))
        except json.JSONDecodeError:
            logger.warning(f"Corrupt results.json in {folder.name}, skipping")
    return results


def merge_workers(worker_results: list[dict]) -> dict:
    """Merge multiple worker results into a single result set.

    Combines rounds, aggregates statistics, renumbers rounds sequentially.
    Config and checkpoint are taken from the first worker.
    """
    base = json.loads(json.dumps(worker_results[0]))  # deep copy

    # Collect all rounds from all workers
    all_rounds = []
    for w in worker_results:
        all_rounds.extend(w.get("rounds", []))

    # Sort by rule_name then batch_round_index for deterministic ordering
    all_rounds.sort(key=lambda r: (r.get("rule_name", ""), r.get("batch_round_index", 0)))

    # Renumber sequentially
    for i, r in enumerate(all_rounds):
        r["round_number"] = i + 1

    base["rounds"] = all_rounds

    # Recompute statistics from merged rounds
    stats = {
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

    for round_data in all_rounds:
        stats["total_score"] += round_data.get("score", 0)
        if round_data.get("success"):
            stats["successful_rounds"] += 1
        else:
            stats["failed_rounds"] += 1
        stats["total_turns"] += round_data.get("turn_count", 0)
        stats["total_failed_guesses"] += round_data.get("failed_guesses", 0)
        stats["total_wall_clock_seconds"] += round_data.get("wall_clock_seconds", 0)

        player_usage = round_data.get("llm_usage", {}).get("player", {})
        stats["total_output_tokens"] += player_usage.get("output_tokens", 0)
        stats["total_reasoning_tokens"] += player_usage.get("reasoning_tokens", 0)
        stats["total_answer_tokens"] += player_usage.get("answer_tokens", 0)

        for turn in round_data.get("turns", []):
            stats["total_retries"] += turn.get("retry_count", 0)
            for retry_info in turn.get("retry_causes", []):
                cause = retry_info.get("cause", "unknown")
                stats["retry_by_cause"][cause] = stats["retry_by_cause"].get(cause, 0) + 1

    base["statistics"] = stats

    # Merge rules_consumed (deduplicate by name)
    seen_rules = set()
    merged_consumed = []
    for w in worker_results:
        for rule in w.get("checkpoint", {}).get("rules_consumed", []):
            if rule["name"] not in seen_rules:
                merged_consumed.append(rule)
                seen_rules.add(rule["name"])
    base["checkpoint"]["rules_consumed"] = merged_consumed

    # Merge rules_library (deduplicate by description)
    seen_descs = set()
    merged_library = []
    for w in worker_results:
        for rule in w.get("checkpoint", {}).get("rules_library", []):
            if rule["description"] not in seen_descs:
                merged_library.append(rule)
                seen_descs.add(rule["description"])
    base["checkpoint"]["rules_library"] = merged_library

    return base


def find_completed_rules(merged: dict, num_workers: int) -> set[str]:
    """Find rule descriptions that have all expected rounds (one per worker)."""
    rule_counts = Counter(r["rule_description"] for r in merged["rounds"])
    return {desc for desc, count in rule_counts.items() if count >= num_workers}


def filter_to_rules(results: dict, completed_rules: set[str]) -> dict:
    """Filter a results dict to only include rounds for the given rules."""
    filtered = json.loads(json.dumps(results))  # deep copy
    filtered["rounds"] = [
        r for r in filtered["rounds"]
        if r["rule_description"] in completed_rules
    ]

    # Recompute statistics
    stats = filtered.get("statistics", {})
    stats["total_score"] = sum(r.get("score", 0) for r in filtered["rounds"])
    stats["successful_rounds"] = sum(1 for r in filtered["rounds"] if r.get("success"))
    stats["failed_rounds"] = sum(1 for r in filtered["rounds"] if not r.get("success"))
    stats["total_turns"] = sum(r.get("turn_count", 0) for r in filtered["rounds"])
    stats["total_failed_guesses"] = sum(r.get("failed_guesses", 0) for r in filtered["rounds"])

    return filtered


def main():
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
    logger.info(f"Renamed to '{merged['config']['player']}' to distinguish from reference")

    total_rounds = len(merged["rounds"])
    logger.info(f"Merged: {model_name} ({model_key}), {total_rounds} rounds")

    # Find completed rules
    completed_rules = find_completed_rules(merged, num_workers)
    all_rules_seen = set(r["rule_description"] for r in merged["rounds"])
    incomplete_rules = all_rules_seen - completed_rules

    if not completed_rules:
        print("No completed rules yet.")
        print(f"Workers have data for {len(all_rules_seen)} rule(s), but none have all {num_workers} rounds.")
        return 1

    logger.info(f"Completed: {len(completed_rules)}/26 rules")
    if incomplete_rules:
        logger.info(f"Incomplete: {len(incomplete_rules)} rule(s) waiting for workers")

    # Filter merged results to only completed rules
    filtered_merged = filter_to_rules(merged, completed_rules)

    # Set up output folder
    if args.output:
        status_folder = args.output
    else:
        status_folder = args.workers[0] / "status"
    status_folder.mkdir(parents=True, exist_ok=True)

    # Save merged in-progress results as a solo_evaluation_* subfolder
    safe_name = model_key.lower().replace("-", "_").replace(".", "_")
    in_progress_dir = status_folder / f"solo_evaluation_in_progress_{safe_name}"
    in_progress_dir.mkdir(exist_ok=True)
    with open(in_progress_dir / "results.json", "w") as f:
        json.dump(filtered_merged, f, indent=2)

    # Copy and filter reference results
    ref_count = 0
    ref_folder = Path(args.reference)
    for sub in sorted(ref_folder.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("solo_evaluation_")):
            continue
        rfile = sub / "results.json"
        if not rfile.exists():
            continue

        with open(rfile) as f:
            ref_data = json.load(f)

        ref_filtered = filter_to_rules(ref_data, completed_rules)
        ref_dir = status_folder / sub.name
        ref_dir.mkdir(exist_ok=True)
        with open(ref_dir / "results.json", "w") as f:
            json.dump(ref_filtered, f, indent=2)
        ref_count += 1

    logger.info(f"Reference models: {ref_count}")
    logger.info(f"Comparing on {len(completed_rules)} rules ({len(completed_rules) * 3} rounds per model)")

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
