"""Evaluate a single LLM player in solo pattern discovery mode."""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game_engine_solo import Rule
from eleusis.game_runner_solo import play_round_solo
from eleusis.logging_utils import setup_logging

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/solo_evaluation_{timestamp}.txt"
setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def save_evaluation_results(evaluation_results: dict, timestamp: str) -> str:
    """Save evaluation results to JSON file (incremental)."""
    Path(f"results/solo_evaluation_{timestamp}").mkdir(parents=True, exist_ok=True)
    output_file = f"results/solo_evaluation_{timestamp}/results.json"
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    return output_file


def main():
    """Evaluate single player across multiple rounds."""

    # Load solo config
    solo_config = config.get("solo_game", config.get("game"))
    num_rounds = solo_config.get("num_rounds", 10)
    rounds_per_rule = config["rule_source"].get("rounds_per_rule", 1)
    player_cfg = solo_config["player"]

    logger.info("=" * 80)
    logger.info(f"SOLO MODE EVALUATION - {num_rounds} ROUNDS")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Rounds per rule: {rounds_per_rule}")
    logger.info(f"  - Game Master: {config['models']['game_master']['display_name']}")
    logger.info(f"  - Player: {player_cfg['display_name']}")
    logger.info("")

    # Initialize evaluation tracking
    evaluation_results = {
        'timestamp': timestamp,
        'config': {
            'num_rounds': num_rounds,
            'rounds_per_rule': rounds_per_rule,
            'game_master': config['models']['game_master']['display_name'],
            'player': player_cfg['display_name'],
            'hand_size': solo_config.get('hand_size', 12),
            'max_turns': solo_config.get('max_turns', 40),
            'wrong_guess_penalty': solo_config.get('wrong_guess_penalty', 3),
        },
        'rounds': [],
        'statistics': {
            'total_score': 0,
            'successful_rounds': 0,
            'failed_rounds': 0,
            'total_turns': 0,
            'total_failed_guesses': 0,
        }
    }

    # Play all rounds
    current_rule = None
    for round_num in range(1, num_rounds + 1):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} / {num_rounds}")
        logger.info("=" * 80)
        logger.info("")

        # Generate new rule every N rounds
        if (round_num - 1) % rounds_per_rule == 0:
            logger.info("Generating new rule for this batch of rounds...")
            current_rule = None  # Force new rule generation
        else:
            logger.info(f"Reusing rule from previous round (batch {(round_num-1)//rounds_per_rule + 1})")

        # Play round
        result = play_round_solo(
            config=config,
            round_number=round_num,
            rule=current_rule,
        )

        # Update current rule for reuse
        if current_rule is None:
            current_rule = Rule(result['rule_description'], result['rule_code'])

        # Update evaluation results
        evaluation_results['rounds'].append({
            'round_number': round_num,
            'turn_count': result['turn_count'],
            'rule_description': result['rule_description'],
            'rule_code': result['rule_code'],
            'success': result['success'],
            'score': result['score'],
            'failed_guesses': result['failed_guesses'],
            'game_over_reason': result['game_over_reason'],
            'llm_usage': result['llm_usage'],
            'turns': result['turns'],
        })

        # Update statistics
        evaluation_results['statistics']['total_score'] += result['score']
        if result['success']:
            evaluation_results['statistics']['successful_rounds'] += 1
        else:
            evaluation_results['statistics']['failed_rounds'] += 1
        evaluation_results['statistics']['total_turns'] += result['turn_count']
        evaluation_results['statistics']['total_failed_guesses'] += result['failed_guesses']

        # Save incrementally after each round
        output_file = save_evaluation_results(evaluation_results, timestamp)
        logger.info(f"Progress saved to: {output_file}")

        # Log round summary
        logger.info("")
        logger.info(f"Round {round_num} complete:")
        logger.info(f"  Turns: {result['turn_count']}")
        logger.info(f"  Success: {'YES' if result['success'] else 'NO'}")
        logger.info(f"  Score: {result['score']}")
        logger.info(f"  Failed guesses: {result['failed_guesses']}")
        logger.info(f"  Rule: {result['rule_description'][:80]}...")
        logger.info("")

    # Calculate final statistics
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    stats = evaluation_results['statistics']
    success_rate = (stats['successful_rounds'] / num_rounds) * 100
    avg_score = stats['total_score'] / num_rounds
    avg_turns = stats['total_turns'] / num_rounds
    avg_turns_success = (
        sum(r['turn_count'] for r in evaluation_results['rounds'] if r['success']) / stats['successful_rounds']
        if stats['successful_rounds'] > 0 else 0
    )
    avg_failed_guesses = stats['total_failed_guesses'] / num_rounds

    evaluation_results['statistics']['success_rate'] = success_rate
    evaluation_results['statistics']['average_score'] = avg_score
    evaluation_results['statistics']['average_turns'] = avg_turns
    evaluation_results['statistics']['average_turns_when_successful'] = avg_turns_success
    evaluation_results['statistics']['average_failed_guesses'] = avg_failed_guesses

    logger.info(f"Player: {player_cfg['display_name']}")
    logger.info(f"Rounds played: {num_rounds}")
    logger.info("")
    logger.info(f"Success rate: {success_rate:.1f}% ({stats['successful_rounds']}/{num_rounds})")
    logger.info(f"Average score: {avg_score:.1f}")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info(f"Average turns (successful rounds only): {avg_turns_success:.1f}")
    logger.info(f"Average failed guesses per round: {avg_failed_guesses:.1f}")
    logger.info(f"Total score: {stats['total_score']}")

    # Save final results
    output_file = save_evaluation_results(evaluation_results, timestamp)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
