#!/usr/bin/env python3
"""Validate scoring fields in results.json files against raw turn data.

Checks that score, floored_score, and no_stakes_score are correctly computed
from the underlying round data (turn_count, failed_guesses, turns list).

Score convention: finding rule on turn N (1-indexed) gives max_turns - N + 1 points.
- Turn 1 = 30 points (best case)
- Turn 30 = 1 point (last turn)
"""

import argparse
import json
from pathlib import Path


def find_first_correct_turn(turns: list[dict]) -> int | None:
    """Find first turn where a guess (formal or shadow) was correct."""
    for turn in turns:
        guess_attempt = turn.get("guess_attempt")
        if guess_attempt and guess_attempt.get("correct"):
            return turn["turn_number"]
    return None


def check_round(round_data: dict, config: dict) -> dict:
    """Check a single round's scoring fields against raw data.

    Returns dict with any discrepancies found.
    """
    max_turns = config.get("max_turns", 30)
    penalty = config.get("wrong_guess_penalty", 2)

    success = round_data.get("success", False)
    failed_guesses = round_data.get("failed_guesses", 0)
    turn_count = round_data.get("turn_count", 0)
    turns = round_data.get("turns", [])

    # Stored values
    stored_score = round_data.get("score")
    stored_floored = round_data.get("floored_score")
    stored_no_stakes = round_data.get("no_stakes_score")
    stored_first_correct = round_data.get("first_correct_turn")

    # Recompute from raw data
    computed_first_correct = find_first_correct_turn(turns)

    if success:
        computed_score = max_turns - turn_count + 1 - (penalty * failed_guesses)
    else:
        computed_score = -(penalty * failed_guesses)

    computed_floored = max(0, computed_score)

    computed_no_stakes = (
        max_turns - computed_first_correct + 1
        if computed_first_correct is not None
        else 0
    )

    # Check for discrepancies
    issues = {}

    if stored_score != computed_score:
        issues["score"] = {"stored": stored_score, "computed": computed_score}

    if stored_floored != computed_floored:
        issues["floored_score"] = {"stored": stored_floored, "computed": computed_floored}

    if stored_no_stakes != computed_no_stakes:
        issues["no_stakes_score"] = {"stored": stored_no_stakes, "computed": computed_no_stakes}

    if stored_first_correct != computed_first_correct:
        issues["first_correct_turn"] = {"stored": stored_first_correct, "computed": computed_first_correct}

    return {
        "round_number": round_data.get("round_number"),
        "success": success,
        "turn_count": turn_count,
        "failed_guesses": failed_guesses,
        "first_correct_turn": computed_first_correct,
        "computed": {
            "score": computed_score,
            "floored_score": computed_floored,
            "no_stakes_score": computed_no_stakes,
        },
        "stored": {
            "score": stored_score,
            "floored_score": stored_floored,
            "no_stakes_score": stored_no_stakes,
        },
        "issues": issues,
    }


def check_results_file(filepath: Path, verbose: bool = False) -> dict:
    """Check all rounds in a results.json file."""
    with open(filepath) as f:
        data = json.load(f)

    config = data.get("config", {})
    rounds = data.get("rounds", [])

    results = {
        "file": str(filepath),
        "total_rounds": len(rounds),
        "rounds_with_issues": 0,
        "all_issues": [],
    }

    for round_data in rounds:
        check = check_round(round_data, config)

        if verbose or check["issues"]:
            rn = check["round_number"]
            print(f"\nRound {rn}:")
            print(f"  success: {check['success']}")
            print(f"  turn_count: {check['turn_count']}")
            print(f"  failed_guesses: {check['failed_guesses']}")
            print(f"  first_correct_turn: {check['first_correct_turn']}")
            print(f"  Stored:   score={check['stored']['score']}, "
                  f"floored={check['stored']['floored_score']}, "
                  f"no_stakes={check['stored']['no_stakes_score']}")
            print(f"  Computed: score={check['computed']['score']}, "
                  f"floored={check['computed']['floored_score']}, "
                  f"no_stakes={check['computed']['no_stakes_score']}")

        if check["issues"]:
            results["rounds_with_issues"] += 1
            results["all_issues"].append(check)
            print(f"  *** ISSUES: {list(check['issues'].keys())}")
            for field, vals in check["issues"].items():
                print(f"      {field}: stored={vals['stored']} != computed={vals['computed']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate scoring fields in results.json files"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to results.json file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all rounds, not just those with issues",
    )
    args = parser.parse_args()

    path = args.path

    if not path.exists():
        print(f"Error: {path} not found")
        return 1

    if not path.is_file():
        print(f"Error: {path} is not a file")
        return 1

    print(f"Checking: {path}")
    results = check_results_file(path, verbose=args.verbose)

    print(f"\n{'=' * 60}")
    print(f"Summary: {results['total_rounds']} rounds checked, "
          f"{results['rounds_with_issues']} with issues")

    if results["rounds_with_issues"] == 0:
        print("All scores are correct!")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
