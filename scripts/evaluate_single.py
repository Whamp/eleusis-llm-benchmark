"""Evaluate a single LLM player in solo pattern discovery mode."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game_engine_solo import Rule
from eleusis.game_runner_solo import play_round_solo
from eleusis.logging_utils import setup_logging

# Load environment variables
load_dotenv()

# Module-level variables (initialized in main)
config = None
logger = None
log_file = None
timestamp = None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate single LLM in solo pattern discovery mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults from config.yaml
  python scripts/evaluate_single.py

  # Override player model
  python scripts/evaluate_single.py --player "openrouter:anthropic/claude-haiku"

  # Run 20 rounds with a specific model and custom tag
  python scripts/evaluate_single.py --player "openrouter:google/gemini-flash" --num-rounds 20 --tag gemini

  # Start from rule index 10
  python scripts/evaluate_single.py --rule-index 10

  # Resume interrupted evaluation
  python scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
"""
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume', type=str,
                        help='Path to resume folder (e.g., results/solo_evaluation_20251205_151306)')
    parser.add_argument('--player', type=str,
                        help='Player model spec (e.g., "openrouter:anthropic/claude-haiku")')
    parser.add_argument('--num-rounds', type=int,
                        help='Number of rounds to play')
    parser.add_argument('--rule-index', type=int,
                        help='Starting rule index (for sequential selection)')
    parser.add_argument('--max-turns', type=int,
                        help='Maximum turns per round')
    parser.add_argument('--tag', type=str,
                        help='Tag to append to output folder name for identification')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / config_path
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config."""
    game_config = config["game"]

    # Player model override
    if args.player:
        config["model"] = args.player

    # Number of rounds override
    if args.num_rounds is not None:
        game_config["num_rounds"] = args.num_rounds

    # Max turns override
    if args.max_turns is not None:
        game_config["max_turns"] = args.max_turns

    # Rule index override
    if args.rule_index is not None:
        config["rules"]["index"] = args.rule_index

    return config


def _model_spec_to_display_name(model_spec: str) -> str:
    """Convert model spec to readable display name."""
    # Remove provider prefix
    if ":" in model_spec:
        _, model_name = model_spec.split(":", 1)
    else:
        model_name = model_spec

    # Extract last part after /
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Clean up common suffixes and format
    model_name = model_name.replace("-", " ").replace("_", " ")
    model_name = re.sub(r'\s+', ' ', model_name).strip()
    return model_name.title()


def generate_output_tag(args, player_name: str) -> str:
    """Generate output folder tag based on CLI args or player name."""
    if args.tag:
        return args.tag

    # Create sanitized tag from player name
    tag = player_name.lower()
    tag = re.sub(r'[^a-z0-9]+', '_', tag)
    tag = tag.strip('_')[:30]  # Limit length
    return tag


def save_evaluation_results(evaluation_results: dict, folder_name: str) -> str:
    """Save evaluation results to JSON file (incremental)."""
    Path(f"results/{folder_name}").mkdir(parents=True, exist_ok=True)
    output_file = f"results/{folder_name}/results.json"
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

    rules_cfg = config["rules"]
    library_path = Path(rules_cfg["library_path"])

    if not library_path.exists():
        logger.error(f"Rule library not found: {library_path}")
        return []

    with open(library_path) as f:
        data = json.load(f)

    all_rules = data.get("rules", [])
    min_acceptance = rules_cfg.get("min_acceptance", 0.0)
    max_acceptance = rules_cfg.get("max_acceptance", 1.0)

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
    global config, logger, log_file, timestamp

    # Parse command-line arguments
    args = parse_args()

    # Load and configure
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Load config sections (after overrides applied)
    game_config = config["game"]
    rules_cfg = config["rules"]
    num_rounds = game_config.get("num_rounds", 10)
    rounds_per_rule = rules_cfg.get("rounds_per_rule", 1)

    # Derive player display name from model spec
    player_model = config["model"]
    player_display_name = _model_spec_to_display_name(player_model)
    game_master_display_name = _model_spec_to_display_name(config["game_master"]["model_name"])

    # Generate output tag for this run
    output_tag = generate_output_tag(args, player_display_name)

    # Setup logging with tag in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/solo_evaluation_{timestamp}_{output_tag}.txt"
    setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if not checkpoint:
            logger.error("Failed to load checkpoint")
            return

        if not validate_resume_config(checkpoint['config'], game_config):
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
        # Resume mode - use original folder name from checkpoint
        folder_name = checkpoint.get('folder_name', f"solo_evaluation_{checkpoint['timestamp']}")
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
        logger.info(f"  - Game Master: {game_master_display_name}")
        logger.info(f"  - Player: {player_display_name}")
        logger.info("")
    else:
        # Fresh start
        folder_name = f"solo_evaluation_{timestamp}_{output_tag}"
        start_round = 1
        current_rule = None
        rule_factory_index = rules_cfg.get("index", 0)

        logger.info("=" * 80)
        logger.info(f"SOLO MODE EVALUATION - {num_rounds} ROUNDS")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Output folder: results/{folder_name}")
        logger.info(f"Rounds per rule: {rounds_per_rule}")
        logger.info(f"  - Game Master: {game_master_display_name}")
        logger.info(f"  - Player: {player_display_name}")
        logger.info("")

        # Load all rules from library upfront (for self-contained checkpoint)
        logger.info("Loading rules library...")
        all_rules_library = load_and_filter_rules_from_library(config)
        logger.info(f"Stored {len(all_rules_library)} rules in checkpoint for resume support")
        logger.info("")

        # Initialize evaluation tracking
        evaluation_results = {
            'timestamp': timestamp,
            'folder_name': folder_name,
            'config': {
                'num_rounds': num_rounds,
                'rounds_per_rule': rounds_per_rule,
                'game_master': game_master_display_name,
                'game_master_model': config['game_master']['model_name'],
                'player': player_display_name,
                'player_model': player_model,
                'hand_size': game_config.get('hand_size', 12),
                'max_turns': game_config.get('max_turns', 40),
                'wrong_guess_penalty': game_config.get('wrong_guess_penalty', 3),
                'max_continuation_attempts': config['llm'].get('max_continuation_attempts', 3),
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
                    'selection': rules_cfg["selection"],
                    'current_index': rule_factory_index,
                    'min_acceptance': rules_cfg.get("min_acceptance", 0.0),
                    'max_acceptance': rules_cfg.get("max_acceptance", 1.0),
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
                'selection': rules_cfg["selection"],
                'current_index': rule_factory_index,
                'min_acceptance': rules_cfg.get("min_acceptance", 0.0),
                'max_acceptance': rules_cfg.get("max_acceptance", 1.0),
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
        output_file = save_evaluation_results(evaluation_results, folder_name)
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

    logger.info(f"Player: {player_display_name}")
    logger.info(f"Rounds played: {num_rounds}")
    logger.info("")
    logger.info(f"Success rate: {success_rate:.1f}% ({stats['successful_rounds']}/{num_rounds})")
    logger.info(f"Average score: {avg_score:.1f}")
    logger.info(f"Average turns: {avg_turns:.1f}")
    logger.info(f"Average turns (successful rounds only): {avg_turns_success:.1f}")
    logger.info(f"Average failed guesses per round: {avg_failed_guesses:.1f}")
    logger.info(f"Total score: {stats['total_score']}")

    # Save final results
    output_file = save_evaluation_results(evaluation_results, folder_name)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
