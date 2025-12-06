"""Evaluate a single LLM player in solo pattern discovery mode."""

import argparse
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


def load_checkpoint(resume_folder: str) -> dict | None:
    """Load checkpoint from results.json in resume folder."""
    results_path = Path(resume_folder) / "results.json"
    if not results_path.exists():
        logger.error(f"No results.json found in {resume_folder}")
        return None

    try:
        with open(results_path) as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in results.json: {e}")
        return None

    # Validate checkpoint structure
    if "checkpoint" not in checkpoint:
        logger.warning("No checkpoint field in results.json - may be from older version")
        logger.warning("Converting to checkpoint format...")
        # Try to create checkpoint from old format
        checkpoint = convert_old_format_to_checkpoint(checkpoint)
        if not checkpoint:
            return None

    return checkpoint


def convert_old_format_to_checkpoint(old_results: dict) -> dict | None:
    """Convert old results format (without checkpoint) to new format."""
    if not old_results.get('rounds'):
        logger.error("No rounds found in results.json")
        return None

    # Extract info from last round
    rounds = old_results['rounds']
    last_round = rounds[-1]

    checkpoint_data = {
        'completed_rounds': len(rounds),
        'total_rounds': old_results.get('config', {}).get('num_rounds', 50),
        'rule_factory_state': {
            'selection': 'sequential',  # Assume sequential
            'current_index': len(rounds),  # Best guess
        },
        'current_rule': {
            'description': last_round.get('rule_description'),
            'code': last_round.get('rule_code'),
            'rounds_used_in_batch': 1,
            'rounds_per_rule': 1,
        },
        'rules_consumed': []
    }

    old_results['checkpoint'] = checkpoint_data
    return old_results


def validate_resume_config(checkpoint_config: dict, current_config: dict) -> bool:
    """Validate that critical config matches between checkpoint and current run."""
    critical_keys = [
        "num_rounds", "max_turns", "hand_size", "wrong_guess_penalty"
    ]

    for key in critical_keys:
        checkpoint_val = checkpoint_config.get(key)
        current_val = current_config.get(key)
        if checkpoint_val != current_val:
            logger.error(f"Config mismatch: {key} changed from {checkpoint_val} to {current_val}")
            logger.error("Cannot resume with different configuration")
            return False

    return True


def restore_rule_from_checkpoint(rule_data: dict | None) -> Rule | None:
    """Restore Rule object from checkpoint data."""
    if not rule_data:
        return None
    description = rule_data.get('description')
    code = rule_data.get('code')
    if not description or not code:
        return None
    return Rule(description, code)


def load_and_filter_rules_from_library(config: dict) -> list[dict]:
    """Load all rules from library and filter by acceptance rate.

    Returns list of rule dicts with 'description', 'code', 'name', etc.
    """
    import json
    from pathlib import Path

    rule_source_cfg = config["rule_source"]
    library_path = Path(rule_source_cfg["library_path"])

    if not library_path.exists():
        logger.error(f"Rule library not found: {library_path}")
        return []

    with open(library_path) as f:
        data = json.load(f)

    all_rules = data.get("rules", [])
    min_acceptance = rule_source_cfg.get("min_acceptance", 0.0)
    max_acceptance = rule_source_cfg.get("max_acceptance", 1.0)

    # Filter rules by acceptance rate if present
    filtered_rules = []
    for rule in all_rules:
        if "avg_acceptance_rate" in rule:
            rate = rule["avg_acceptance_rate"]
            if min_acceptance <= rate <= max_acceptance:
                filtered_rules.append(rule)
        else:
            # No acceptance rate info, include by default
            filtered_rules.append(rule)

    logger.info(f"Loaded {len(all_rules)} rules from library, {len(filtered_rules)} match acceptance criteria")
    return filtered_rules


def main():
    """Evaluate single player across multiple rounds."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate single LLM in solo pattern discovery mode')
    parser.add_argument('--resume', type=str, help='Path to resume folder (e.g., results/solo_evaluation_20251205_151306)')
    args = parser.parse_args()

    # Load solo config
    solo_config = config.get("solo_game", config.get("game"))
    num_rounds = solo_config.get("num_rounds", 10)
    rounds_per_rule = config["rule_source"].get("rounds_per_rule", 1)
    player_cfg = solo_config["player"]

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if not checkpoint:
            logger.error("Failed to load checkpoint")
            return

        if not validate_resume_config(checkpoint['config'], solo_config):
            return

        # Check if selection mode is sequential
        rule_selection = checkpoint['checkpoint']['rule_factory_state']['selection']
        if rule_selection != 'sequential':
            logger.error("Resume only supported for sequential rule selection")
            logger.error(f"Checkpoint uses: {rule_selection}")
            return

        # Check if already completed
        completed = checkpoint['checkpoint']['completed_rounds']
        total = checkpoint['checkpoint']['total_rounds']
        if completed >= total:
            logger.info(f"Evaluation already complete ({completed}/{total} rounds)")
            return

    # Initialize or resume
    if checkpoint:
        # Resume mode
        timestamp_val = checkpoint['timestamp']
        evaluation_results = checkpoint
        start_round = checkpoint['checkpoint']['completed_rounds'] + 1

        # Restore current rule from checkpoint (not from library)
        current_rule = restore_rule_from_checkpoint(
            checkpoint['checkpoint'].get('current_rule')
        )

        rule_factory_index = checkpoint['checkpoint']['rule_factory_state']['current_index']

        logger.info("=" * 80)
        logger.info(f"RESUMING SOLO MODE EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Resuming from round {start_round} / {checkpoint['checkpoint']['total_rounds']}")
        logger.info(f"Rule factory index: {rule_factory_index}")
        logger.info(f"Rounds per rule: {rounds_per_rule}")
        if current_rule:
            logger.info(f"Reusing rule: {current_rule.description()[:80]}...")
        logger.info(f"  - Game Master: {config['models']['game_master']['display_name']}")
        logger.info(f"  - Player: {player_cfg['display_name']}")
        logger.info("")
    else:
        # Fresh start
        timestamp_val = timestamp
        start_round = 1
        current_rule = None
        rule_factory_index = config["rule_source"].get("index", 0)

        logger.info("=" * 80)
        logger.info(f"SOLO MODE EVALUATION - {num_rounds} ROUNDS")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Rounds per rule: {rounds_per_rule}")
        logger.info(f"  - Game Master: {config['models']['game_master']['display_name']}")
        logger.info(f"  - Player: {player_cfg['display_name']}")
        logger.info("")

        # Load all rules from library upfront (for self-contained checkpoint)
        logger.info("Loading rules library...")
        all_rules_library = load_and_filter_rules_from_library(config)
        logger.info(f"Stored {len(all_rules_library)} rules in checkpoint for resume support")
        logger.info("")

        # Initialize evaluation tracking
        evaluation_results = {
            'timestamp': timestamp_val,
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
            },
            'checkpoint': {
                'completed_rounds': 0,
                'total_rounds': num_rounds,
                'rule_factory_state': {
                    'selection': config["rule_source"]["selection"],
                    'current_index': rule_factory_index,
                    'min_acceptance': config["rule_source"].get("min_acceptance", 0.0),
                    'max_acceptance': config["rule_source"].get("max_acceptance", 1.0),
                },
                'current_rule': None,
                'rules_consumed': [],
                'rules_library': all_rules_library,  # Store ALL rules for self-contained resume
            }
        }

    # Play all rounds from start_round to num_rounds
    for round_num in range(start_round, num_rounds + 1):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} / {num_rounds}")
        logger.info("=" * 80)
        logger.info("")

        # Determine if new rule needed
        need_new_rule = (round_num - 1) % rounds_per_rule == 0

        if need_new_rule:
            logger.info("Generating new rule for this batch of rounds...")
            current_rule = None  # Force new rule generation
        else:
            logger.info(f"Reusing rule from previous round (batch {(round_num-1)//rounds_per_rule + 1})")

        # Track if we're generating a new rule this round
        generated_new_rule = current_rule is None

        # Play round with rule_factory_index
        result = play_round_solo(
            config=config,
            round_number=round_num,
            rule=current_rule,
            start_rule_index=rule_factory_index if need_new_rule else None,
        )

        # Update current rule for reuse and track consumption
        if generated_new_rule:
            current_rule = Rule(result['rule_description'], result['rule_code'])

            # Increment rule_factory_index and store consumed rule (only for newly generated rules)
            rule_factory_index += 1

            # Add newly consumed rule to rules_consumed list
            if 'rules_consumed' not in evaluation_results.get('checkpoint', {}):
                if 'checkpoint' not in evaluation_results:
                    evaluation_results['checkpoint'] = {}
                evaluation_results['checkpoint']['rules_consumed'] = []

            evaluation_results['checkpoint']['rules_consumed'].append({
                'index': rule_factory_index - 1,
                'description': current_rule.description(),
                'code': current_rule.get_code(),
            })

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

        # Update checkpoint (preserve rules_library from initial load)
        evaluation_results['checkpoint'] = {
            'completed_rounds': round_num,
            'total_rounds': num_rounds,
            'rule_factory_state': {
                'selection': config["rule_source"]["selection"],
                'current_index': rule_factory_index,
                'min_acceptance': config["rule_source"].get("min_acceptance", 0.0),
                'max_acceptance': config["rule_source"].get("max_acceptance", 1.0),
            },
            'current_rule': {
                'description': current_rule.description() if current_rule else None,
                'code': current_rule.get_code() if current_rule else None,
                'rounds_used_in_batch': (round_num - 1) % rounds_per_rule + 1,
                'rounds_per_rule': rounds_per_rule,
            },
            'rules_consumed': evaluation_results.get('checkpoint', {}).get('rules_consumed', []),
            'rules_library': evaluation_results.get('checkpoint', {}).get('rules_library', []),  # Preserve loaded library
        }

        # Save incrementally after each round
        output_file = save_evaluation_results(evaluation_results, timestamp_val)
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
    output_file = save_evaluation_results(evaluation_results, timestamp_val)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
