"""Evaluate rules by measuring acceptance rates with random players."""

import argparse
import json
import logging
from pathlib import Path

from eleusis.game_engine import GameEngine
from eleusis.game_state import GameState
from eleusis.llm_player import RandomScientist
from eleusis.python_rule import PythonRule

logger = logging.getLogger(__name__)


def play_game_with_random_players(
    rule: PythonRule,
    num_players: int = 3,
    max_turns: int = 40,
    cards_per_scientist: int = 12,
) -> dict:
    """Play a game with random players and return statistics.

    Args:
        rule: The rule to evaluate
        num_players: Number of scientist players
        max_turns: Maximum number of turns
        cards_per_scientist: Initial hand size

    Returns:
        Dict with acceptance_rate and other stats
    """
    # Create game state
    player_names = ["RuleMaker"] + [f"RandomPlayer{i}" for i in range(1, num_players + 1)]
    game_state = GameState(player_names, rule_maker_index=0)

    # Create game engine
    engine = GameEngine(
        game_state,
        rule,
        cards_per_scientist=cards_per_scientist,
        # Minimal penalties for random players
        card_reject_penalty=1,
        no_play_incorrect_penalty=1,
        no_play_correct_reduction=1,
        correct_guess_bonus=0,  # No guessing
    )

    # Create random players
    players = [RandomScientist(name) for name in player_names[1:]]

    # Track statistics
    total_plays = 0
    total_accepted = 0

    # Play game
    for turn_num in range(max_turns):
        current_player_index = game_state.current_turn_index

        if current_player_index == 0:  # Skip rule-maker
            game_state.advance_turn()
            continue

        player = players[current_player_index - 1]
        action = player.get_action(game_state)

        # Play turn
        result = engine.play_turn(action)

        # Track statistics
        if "card" in result:
            total_plays += 1
            if result.get("accepted", False):
                total_accepted += 1

        # Check if game ended
        if result.get("game_over", False):
            break

    acceptance_rate = total_accepted / total_plays if total_plays > 0 else 0.0

    return {
        "total_plays": total_plays,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "turns_played": turn_num + 1,
    }


def evaluate_rule(
    rule_dict: dict,
    num_games: int = 5,
    num_players: int = 3,
    max_turns: int = 40,
) -> dict:
    """Evaluate a single rule across multiple games.

    Args:
        rule_dict: Dict with name, description, code
        num_games: Number of games to play for averaging
        num_players: Number of random players
        max_turns: Maximum turns per game

    Returns:
        Dict with averaged statistics
    """
    name = rule_dict["name"]
    description = rule_dict["description"]
    code = rule_dict["code"]

    logger.info(f"Evaluating rule: {name}")
    logger.debug(f"Description: {description}")

    rule = PythonRule(description, code)

    # Run multiple games
    game_results = []
    for game_num in range(num_games):
        logger.debug(f"  Game {game_num + 1}/{num_games}")
        result = play_game_with_random_players(rule, num_players, max_turns)
        game_results.append(result)

    # Compute averages
    avg_acceptance_rate = sum(r["acceptance_rate"] for r in game_results) / num_games
    avg_plays = sum(r["total_plays"] for r in game_results) / num_games
    avg_accepted = sum(r["total_accepted"] for r in game_results) / num_games

    logger.info(f"  Acceptance rate: {avg_acceptance_rate:.1%}")
    logger.info(f"  Avg plays per game: {avg_plays:.1f}")

    return {
        "name": name,
        "description": description,
        "avg_acceptance_rate": avg_acceptance_rate,
        "avg_plays_per_game": avg_plays,
        "avg_accepted_per_game": avg_accepted,
        "num_games_evaluated": num_games,
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
        "--num-games",
        type=int,
        default=5,
        help="Number of games per rule for averaging (default: 5)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=3,
        help="Number of random players (default: 3)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=40,
        help="Maximum turns per game (default: 40)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rule_evaluation_results.json"),
        help="Output JSON file (default: rule_evaluation_results.json)",
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

    # Evaluate each rule
    results = []
    for i, rule_dict in enumerate(rules, 1):
        logger.info(f"[{i}/{len(rules)}] {rule_dict['name']}")
        result = evaluate_rule(
            rule_dict,
            num_games=args.num_games,
            num_players=args.num_players,
            max_turns=args.max_turns,
        )
        results.append(result)
        logger.info("")

    # Save results
    output_data = {
        "evaluation_params": {
            "num_games_per_rule": args.num_games,
            "num_players": args.num_players,
            "max_turns": args.max_turns,
        },
        "results": results,
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {args.output}")
    logger.info("")
    logger.info("Summary:")
    logger.info("-" * 60)
    for result in results:
        logger.info(f"{result['name']:40s} {result['avg_acceptance_rate']:6.1%}")
    logger.info("-" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
