"""Evaluate rules by measuring acceptance rates with random cards."""

import argparse
import json
import logging
import random
from pathlib import Path

from eleusis.cards import Card, Suit
from eleusis.python_rule import PythonRule

logger = logging.getLogger(__name__)


def simulate_random_plays(rule: PythonRule, num_plays: int = 50) -> dict:
    """Simulate random card plays and return statistics.

    Args:
        rule: The rule to evaluate
        num_plays: Number of random cards to try

    Returns:
        Dict with acceptance_rate and other stats
    """
    # Create all 52 cards
    all_cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]

    # Track statistics
    total_plays = 0
    total_accepted = 0
    mainline = []

    # Try random cards
    for _ in range(num_plays):
        # Pick a random card
        card = random.choice(all_cards)

        # Evaluate against current mainline
        accepted = rule.evaluate(card, mainline)

        total_plays += 1
        if accepted:
            total_accepted += 1
            # Add to mainline for next evaluation
            mainline.append(card)

    acceptance_rate = total_accepted / total_plays if total_plays > 0 else 0.0

    return {
        "total_plays": total_plays,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "mainline_length": len(mainline),
    }


def evaluate_rule(
    rule_dict: dict,
    num_simulations: int = 10,
    plays_per_simulation: int = 50,
) -> dict:
    """Evaluate a single rule across multiple simulations.

    Args:
        rule_dict: Dict with name, description, code
        num_simulations: Number of simulations to run for averaging
        plays_per_simulation: Number of random card plays per simulation

    Returns:
        Dict with averaged statistics
    """
    name = rule_dict["name"]
    description = rule_dict["description"]
    code = rule_dict["code"]

    logger.info(f"Evaluating rule: {name}")
    logger.info(f"Description: {description}")

    rule = PythonRule(description, code)

    # Run multiple simulations
    sim_results = []
    for sim_num in range(num_simulations):
        logger.debug(f"  Simulation {sim_num + 1}/{num_simulations}")
        result = simulate_random_plays(rule, plays_per_simulation)
        sim_results.append(result)

    # Compute averages
    avg_acceptance_rate = sum(r["acceptance_rate"] for r in sim_results) / num_simulations
    avg_mainline_length = sum(r["mainline_length"] for r in sim_results) / num_simulations

    logger.info(f"  Acceptance rate: {avg_acceptance_rate:.1%}")
    logger.info(f"  Avg mainline length: {avg_mainline_length:.1f}")

    return {
        "avg_acceptance_rate": avg_acceptance_rate,
        "avg_mainline_length": avg_mainline_length,
        "num_simulations": num_simulations,
        "plays_per_simulation": plays_per_simulation,
    }


def main():
    """Main entry point for rule evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Eleusis rules by acceptance rate")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("rules.json"),
        help="Path to rule library JSON (default: rules.json)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10,
        help="Number of simulations per rule for averaging (default: 10)",
    )
    parser.add_argument(
        "--plays-per-simulation",
        type=int,
        default=50,
        help="Number of random card plays per simulation (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for updated rules (default: overwrites input library)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s - %(message)s",
    )

    # Load rule library
    logger.info(f"Loading rule library from {args.library}")
    with open(args.library) as f:
        data = json.load(f)
        rules = data.get("rules", [])

    logger.info(f"Found {len(rules)} rules to evaluate")
    logger.info("")

    # Evaluate each rule and merge results
    for i, rule_dict in enumerate(rules, 1):
        logger.info(f"[{i}/{len(rules)}] {rule_dict['name']}")
        evaluation_results = evaluate_rule(
            rule_dict,
            num_simulations=args.num_simulations,
            plays_per_simulation=args.plays_per_simulation,
        )
        # Merge evaluation results into rule dict
        rule_dict.update(evaluation_results)
        logger.info("")

    # Add evaluation metadata to the data structure
    data["evaluation_params"] = {
        "num_simulations": args.num_simulations,
        "plays_per_simulation": args.plays_per_simulation,
    }

    # Determine output path
    output_path = args.output if args.output else args.library

    # Save updated rules back to JSON
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Updated rules saved to {output_path}")
    logger.info("")
    logger.info("Summary:")
    logger.info("-" * 60)
    for rule in rules:
        logger.info(f"{rule['name']:40s} {rule['avg_acceptance_rate']:6.1%}")
    logger.info("-" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
