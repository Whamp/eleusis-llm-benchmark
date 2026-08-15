"""Offline shadow evaluation for saved benchmark results.

Takes results from a run with shadow_mode=offline (which records tentative rules
without evaluating them) and evaluates shadow guesses offline, producing augmented
results with shadow correctness metrics.

Usage:
    uv run python scripts/evaluate_shadows.py --results results/run_123/results.json \
        --output results/run_123/results_with_shadows.json

The script reads each round's turns, finds unevaluated shadow entries
(guess_attempt.shadow=True, guess_attempt.evaluated=False), compiles and
simulates the guessed rules against the actual rule, and writes augmented results.
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from eleusis.benchmark_config import BenchmarkConfig, parse_benchmark_config
from eleusis.evaluation_results import (
    SavedRound,
    TurnRecord,
    parse_evaluation_results,
)
from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.game.validator import RuleValidator
from eleusis.llm import BaseLLMClient, create_client_from_config

logger = logging.getLogger(__name__)

# Map string suit names to Suit enum
_SUIT_MAP = {
    "hearts": Suit.HEARTS,
    "h": Suit.HEARTS,
    "♥": Suit.HEARTS,
    "diamonds": Suit.DIAMONDS,
    "d": Suit.DIAMONDS,
    "♦": Suit.DIAMONDS,
    "clubs": Suit.CLUBS,
    "c": Suit.CLUBS,
    "♣": Suit.CLUBS,
    "spades": Suit.SPADES,
    "s": Suit.SPADES,
    "♠": Suit.SPADES,
}


def _parse_card(card_str: str) -> Card:
    """Parse a card string like '4H', '10S', 'KD' into a Card object."""
    card_str = card_str.strip().upper()

    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}

    if len(card_str) == 2:
        rank_part, suit_part = card_str[0], card_str[1]
    elif len(card_str) == 3:
        rank_part, suit_part = card_str[:2], card_str[2]
    else:
        raise ValueError(f"Cannot parse card: {card_str}")

    rank = rank_map[rank_part] if rank_part in rank_map else int(rank_part)

    suit = _SUIT_MAP[suit_part.lower()]
    return Card(rank, suit)


def evaluate_shadow_turns(
    turns: list[TurnRecord],
    actual_rule: Rule,
    mainline: list[Card],
    rule_compiler_client: BaseLLMClient,
    num_simulations: int = 100,
    turns_per_simulation: int = 40,
    simulation_seed: int = 42,
    compiler_max_retries: int | None = None,
) -> list[TurnRecord]:
    """Evaluate unevaluated shadow entries in a list of turn dicts.

    Args:
        turns: List of turn data dicts (from a round's results).
        actual_rule: The secret rule for this round.
        mainline: The mainline card sequence at end of round.
        rule_compiler_client: Client for compiling guessed rules to code.
        num_simulations: Simulation runs for rule comparison.
        turns_per_simulation: Turns per simulation.
        simulation_seed: Seed for simulation RNG.
        compiler_max_retries: Max retries for rule compilation.

    Returns:
        Deep copy of turns with shadow entries evaluated (correct, reasoning, etc.).
    """
    validator = RuleValidator()
    augmented = copy.deepcopy(turns)

    for turn in augmented:
        ga = turn.get("guess_attempt")
        if not ga:
            continue
        if not ga.get("shadow") or ga.get("evaluated") is not False:
            continue

        guess_text = ga["guess"]
        is_correct, reasoning, metadata = validator.compare_rules(
            actual_rule=actual_rule,
            guessed_rule_desc=guess_text,
            current_mainline=mainline,
            rule_compiler_client=rule_compiler_client,
            num_simulations=num_simulations,
            turns_per_simulation=turns_per_simulation,
            simulation_seed=simulation_seed,
            compiler_max_retries=(
                compiler_max_retries if compiler_max_retries is not None else 2
            ),
        )

        complexity = metadata["complexity_metrics"]
        ga["correct"] = is_correct
        ga["reasoning"] = reasoning
        ga["guessed_code"] = metadata["guessed_code"]
        ga["node_count"] = complexity["node_count"] if complexity else None
        ga["cyclomatic_complexity"] = complexity["cyclomatic"] if complexity else None
        ga["evaluated"] = True

    return augmented


def _evaluate_round(
    round_data: SavedRound,
    rule_compiler_client: BaseLLMClient,
    config: BenchmarkConfig,
) -> SavedRound:
    """Evaluate shadow entries for a single round."""
    actual_rule = Rule(
        description=round_data["rule_description"],
        code=round_data["rule_code"],
    )

    # Reconstruct mainline from turn data (last turn's mainline_state)
    turns = round_data.get("turns", [])
    if not turns:
        return round_data

    # Use the last turn's mainline_state to reconstruct mainline
    last_mainline_str = turns[-1].get("mainline_state", "")
    mainline = []
    if last_mainline_str:
        for card_str in last_mainline_str.replace("[", "").replace("]", "").split(","):
            card_str = card_str.strip()
            if card_str:
                try:
                    mainline.append(_parse_card(card_str))
                except (ValueError, KeyError):
                    logger.warning(f"Could not parse mainline card: {card_str}")

    rule_compiler_cfg = config.get("rule_compiler", {})
    num_simulations = rule_compiler_cfg.get("num_simulations", 100)
    turns_per_simulation = rule_compiler_cfg.get("turns_per_simulation", 40)
    simulation_seed = rule_compiler_cfg.get("simulation_seed")
    if simulation_seed is None:
        simulation_seed = 42
    compiler_max_retries = rule_compiler_cfg.get("max_retries")

    augmented_turns = evaluate_shadow_turns(
        turns=turns,
        actual_rule=actual_rule,
        mainline=mainline,
        rule_compiler_client=rule_compiler_client,
        num_simulations=num_simulations,
        turns_per_simulation=turns_per_simulation,
        simulation_seed=simulation_seed,
        compiler_max_retries=compiler_max_retries,
    )

    result: SavedRound = copy.deepcopy(round_data)
    result["turns"] = augmented_turns

    # Recompute first_shadow_correct_turn
    first_shadow_correct = None
    for turn in augmented_turns:
        ga = turn.get("guess_attempt")
        if (
            ga
            and ga.get("shadow")
            and ga.get("correct")
            and first_shadow_correct is None
        ):
            first_shadow_correct = turn["turn_number"]
    result["first_shadow_correct_turn"] = first_shadow_correct

    return result


def main() -> None:
    """Evaluate offline shadow guesses and write augmented results."""
    parser = argparse.ArgumentParser(
        description="Evaluate shadow guesses offline from saved benchmark results."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results.json from an offline shadow run",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file for rule compiler settings (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        help="Output path for augmented results (default: <results>_shadows.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    with results_path.open() as results_file:
        results = parse_evaluation_results(json.load(results_file))

    # Load config for rule compiler settings
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    with config_path.open() as config_file:
        config = parse_benchmark_config(yaml.safe_load(config_file))

    # Create rule compiler client
    rule_compiler_cfg = config["rule_compiler"]
    max_tokens = config["llm"]["max_tokens"]
    llm_seed = config["llm"]["seed"]

    rule_compiler_client = create_client_from_config(
        rule_compiler_cfg,
        max_tokens=max_tokens,
        role="rule_compiler_shadow",
        seed=llm_seed,
    )

    # Process each round
    rounds = results.get("rounds", [])
    augmented_rounds = []
    for i, round_data in enumerate(rounds):
        rule_desc = round_data.get("rule_description", "unknown")
        logger.info(f"Processing round {i + 1}/{len(rounds)}: {rule_desc}")
        augmented = _evaluate_round(round_data, rule_compiler_client, config)
        augmented_rounds.append(augmented)

    results["rounds"] = augmented_rounds

    # Write output
    output_path = args.output
    if not output_path:
        output_path = str(results_path).replace(".json", "_shadows.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as output_file:
        json.dump(results, output_file, indent=2)

    logger.info(f"Augmented results written to: {output_path}")


if __name__ == "__main__":
    main()
