"""Evaluate a single LLM player in solo pattern discovery mode."""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game import Rule
from eleusis.llm import create_client
from eleusis.runner import play_round
from eleusis.utils import model_spec_to_display_name, setup_logging

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
  # Run evaluation with a model
  python scripts/evaluate_single.py --model "claude-opus"

  # Test 20 rules with a specific model and custom tag
  python scripts/evaluate_single.py --model "gpt-5.2" --num-rules 20 --tag gpt

  # Start from rule index 10
  python scripts/evaluate_single.py --model "gpt-5.2" --rule-index 10

  # Resume interrupted evaluation (model taken from checkpoint)
  python scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
"""
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    parser.add_argument('--resume', type=str,
                        help='Path to resume folder (e.g., results/solo_evaluation_20251205_151306)')
    parser.add_argument('--model', type=str,
                        help='Model key from models.yaml (required unless --resume)')
    parser.add_argument('--num-rules', type=int,
                        help='Number of distinct rules to test')
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

    # Number of rules override
    if args.num_rules is not None:
        game_config["num_rules"] = args.num_rules

    # Max turns override
    if args.max_turns is not None:
        game_config["max_turns"] = args.max_turns

    # Rule index override
    if args.rule_index is not None:
        config["rules"]["index"] = args.rule_index

    return config


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
        logger.error("No checkpoint field in results.json - incompatible format")
        return None

    return checkpoint


def restore_rule_from_checkpoint(rule_data: dict | None) -> Rule | None:
    """Restore Rule object from checkpoint data."""
    if not rule_data:
        return None
    description = rule_data.get('description')
    code = rule_data.get('code')
    if not description or not code:
        return None
    return Rule(description, code)


def reconstruct_config_from_checkpoint(checkpoint: dict) -> dict:
    """Reconstruct full config dict from checkpoint data for self-contained resume."""
    cfg = checkpoint['config']
    chk = checkpoint['checkpoint']

    return {
        'model': cfg['player_model'],
        'game': {
            'num_rules': cfg['num_rules'],
            'num_rounds_per_rule': cfg['num_rounds_per_rule'],
            'max_turns': cfg['max_turns'],
            'hand_size': cfg['hand_size'],
            'wrong_guess_penalty': cfg['wrong_guess_penalty'],
            'seed': cfg.get('seed'),
        },
        'llm': {
            'max_tokens': cfg.get('llm_max_tokens', 8192),
            'temperature': cfg.get('llm_temperature', 0.7),
            'seed': cfg.get('llm_seed'),
            'max_llm_retries': cfg.get('llm_max_retries', 3),
        },
        'rule_compiler': {
            'model': cfg['rule_compiler_model'],
            'temperature': cfg.get('rule_compiler_temperature', 0.8),
        },
        'rules': {
            'library_path': None,  # Not needed, rules embedded in checkpoint
            'selection': chk['rule_factory_state']['selection'],
            'index': chk['rule_factory_state']['current_index'],
        },
    }


def load_rules_from_library(config: dict) -> list[dict]:
    """Load all rules from library.

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

    rules = data.get("rules", [])
    logger.info(f"Loaded {len(rules)} rules from library")
    return rules


def preflight_check(model_key: str) -> None:
    """Run pre-flight model check. Fails fast on issues.

    Args:
        model_key: Model key from models.yaml (e.g., "claude-opus", "deepseek-r1")

    Raises:
        SystemExit: If pre-flight check fails
    """
    import time

    logger.info("Running pre-flight model check...")

    try:
        start = time.time()
        client = create_client(model_key, max_tokens = 16384)
        # Simple connectivity test
        response = client.generate("Say 'hello' and nothing else.")
        latency = time.time() - start
    except Exception as e:
        logger.error(f"Pre-flight check failed: {e}")
        raise SystemExit(1)

    logger.info(f"  Provider: {client.provider_name}")
    logger.info(f"  Model: {client.model_name}")
    logger.info(f"  Latency: {latency:.2f}s")
    logger.info(f"  Response: {response[:100]}...")


def main():
    """Evaluate single player across multiple rounds."""
    global config, logger, log_file, timestamp

    # Parse command-line arguments
    args = parse_args()

    # Temporarily setup minimal logging for early messages
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    temp_logger = logging.getLogger(__name__)

    # Check required arguments
    if not args.resume and not args.model:
        temp_logger.error("--model is required (unless using --resume)")
        return

    # Load checkpoint OR config.yaml (resume is self-contained, no config.yaml needed)
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if not checkpoint:
            temp_logger.error("Failed to load checkpoint")
            return

        # Reconstruct full config from checkpoint (self-contained resume)
        config = reconstruct_config_from_checkpoint(checkpoint)

        # Validate CLI model override if provided
        if args.model and args.model != checkpoint['config']['player_model']:
            temp_logger.error("Model mismatch:")
            temp_logger.error(f"  Checkpoint: {checkpoint['config']['player_model']}")
            temp_logger.error(f"  CLI --model: {args.model}")
            temp_logger.error("Cannot resume with different model")
            temp_logger.error("Remove --model flag to use checkpoint's model, or start a new evaluation")
            return

        # Check if selection mode is sequential
        rule_selection = checkpoint['checkpoint']['rule_factory_state']['selection']
        if rule_selection != 'sequential':
            temp_logger.error("Resume only supported for sequential rule selection")
            temp_logger.error(f"Checkpoint uses: {rule_selection}")
            return

        # Check if already completed
        completed = checkpoint['checkpoint']['completed_rounds']
        total = checkpoint['checkpoint']['total_rounds']
        if completed >= total:
            temp_logger.info(f"Evaluation already complete ({completed}/{total} rounds)")
            return

        # Extract values from checkpoint for display/logging
        player_model = config['model']
        player_display_name = checkpoint['config']['player']
        rule_compiler_display_name = checkpoint['config']['rule_compiler']
        num_rounds_per_rule = config['game']['num_rounds_per_rule']
    else:
        # Fresh start - load config.yaml and set model from CLI
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
        config["model"] = args.model  # Model comes from CLI, not config
        player_model = args.model
        player_display_name = model_spec_to_display_name(player_model)
        rule_compiler_display_name = model_spec_to_display_name(config["rule_compiler"]["model"])
        num_rounds_per_rule = config["game"].get("num_rounds_per_rule", 1)

    # Load config sections
    game_config = config["game"]
    rules_cfg = config["rules"]

    # Generate output tag - from checkpoint folder name when resuming
    if checkpoint:
        checkpoint_folder = checkpoint.get('folder_name', f"solo_evaluation_{checkpoint['timestamp']}")
        # Extract tag from folder name for log file consistency
        output_tag = checkpoint_folder.replace(f"solo_evaluation_{checkpoint['timestamp']}_", "")
        if output_tag == checkpoint_folder:  # No tag in folder name
            output_tag = player_display_name.lower().replace(" ", "_")[:30]
    else:
        output_tag = generate_output_tag(args, player_display_name)

    # Setup logging with tag in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(exist_ok=True)
    log_file = f"logs/solo_evaluation_{timestamp}_{output_tag}.txt"
    setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)

    # Run pre-flight model check (fails fast if model doesn't respond)
    logger.info("=" * 80)
    logger.info("PRE-FLIGHT MODEL CHECK")
    logger.info("=" * 80)
    preflight_check(player_model)  # Fails fast if model doesn't respond
    logger.info("Pre-flight check passed!")
    logger.info("")

    # Determine num_rules and num_rounds - from checkpoint when resuming, else from config
    if checkpoint:
        num_rounds = checkpoint['checkpoint']['total_rounds']
        num_rules = config['game']['num_rules']  # From reconstruct_config_from_checkpoint
    else:
        num_rules = game_config.get("num_rules", 10)
        # Handle num_rules=0: use entire rule library
        if num_rules == 0:
            all_rules = load_rules_from_library(config)
            num_rules = len(all_rules)
            game_config["num_rules"] = num_rules
            logger.info(f"num_rules=0: using entire library ({num_rules} rules)")
        # Validate: rules library has enough rules for num_rules
        else:
            all_rules = load_rules_from_library(config)
            if len(all_rules) < num_rules:
                logger.error(f"Not enough rules: {len(all_rules)} available, {num_rules} requested")
                return
        # Compute total rounds
        num_rounds = num_rules * num_rounds_per_rule
        game_config["num_rounds"] = num_rounds  # Store for compatibility

    # Initialize or resume
    if checkpoint:
        # Resume mode - use original folder name from checkpoint
        folder_name = checkpoint.get('folder_name', f"solo_evaluation_{checkpoint['timestamp']}")
        evaluation_results = checkpoint
        start_round = checkpoint['checkpoint']['completed_rounds'] + 1

        # Ensure new statistics fields exist (for backwards compatibility with old checkpoints)
        stats = evaluation_results.setdefault('statistics', {})
        stats.setdefault('total_output_tokens', 0)
        stats.setdefault('total_reasoning_tokens', 0)
        stats.setdefault('total_answer_tokens', 0)
        stats.setdefault('total_wall_clock_seconds', 0.0)
        stats.setdefault('total_retries', 0)
        stats.setdefault('retry_by_cause', {})

        # Restore current rule from checkpoint (not from library)
        current_rule = restore_rule_from_checkpoint(
            checkpoint['checkpoint'].get('current_rule')
        )

        # Restore current rule name from last consumed rule
        rules_consumed_list = checkpoint['checkpoint'].get('rules_consumed', [])
        current_rule_name = rules_consumed_list[-1]['name'] if rules_consumed_list else None

        # Load rules library and consumed rules from checkpoint
        checkpoint_rules_library = checkpoint['checkpoint'].get('rules_library')
        rules_consumed = checkpoint['checkpoint'].get('rules_consumed', [])
        completed_rounds = checkpoint['checkpoint']['completed_rounds']
        total_rounds = checkpoint['checkpoint']['total_rounds']

        # Validate: consumed rules count matches expected for completed rounds
        # With num_rounds_per_rule > 1, we consume fewer rules than rounds
        expected_consumed = (completed_rounds + num_rounds_per_rule - 1) // num_rounds_per_rule
        if len(rules_consumed) != expected_consumed:
            logger.error(f"Mismatch: {len(rules_consumed)} rules consumed but expected {expected_consumed} for {completed_rounds} rounds")
            return

        # Validate: library has enough rules for total rounds
        expected_rules_needed = (total_rounds + num_rounds_per_rule - 1) // num_rounds_per_rule
        if len(checkpoint_rules_library) < expected_rules_needed:
            logger.error(f"Not enough rules: {len(checkpoint_rules_library)} in library but need {expected_rules_needed}")
            return

        # Filter to unconsumed rules only (match by name)
        consumed_names = {r['name'] for r in rules_consumed}
        unconsumed_rules = [r for r in checkpoint_rules_library if r['name'] not in consumed_names]

        # Start from index 0 of the filtered (unconsumed) list
        checkpoint_rules_library = unconsumed_rules
        rule_factory_index = 0

        logger.info("=" * 80)
        logger.info(f"RESUMING SOLO MODE EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Resuming from round {start_round} / {total_rounds}")
        logger.info(f"Rules consumed: {len(rules_consumed)}, unconsumed: {len(unconsumed_rules)}")
        logger.info(f"Rounds per rule: {num_rounds_per_rule}")
        if current_rule:
            logger.info(f"Reusing rule: {current_rule.description()[:80]}...")
        logger.info(f"  - Rule Compiler: {rule_compiler_display_name}")
        logger.info(f"  - Player: {player_display_name}")
        logger.info("")
    else:
        # Fresh start
        folder_name = f"solo_evaluation_{timestamp}_{output_tag}"
        start_round = 1
        current_rule = None
        current_rule_name = None
        rule_factory_index = rules_cfg.get("index", 0)
        checkpoint_rules_library = None  # Will load from file

        logger.info("=" * 80)
        logger.info(f"SOLO MODE EVALUATION - {num_rounds} ROUNDS")
        logger.info("=" * 80)
        logger.info(f"Log file: {log_file}")
        logger.info(f"Output folder: results/{folder_name}")
        logger.info(f"Rounds per rule: {num_rounds_per_rule}")
        logger.info(f"  - Rule Compiler: {rule_compiler_display_name}")
        logger.info(f"  - Player: {player_display_name}")
        logger.info("")

        # Load all rules from library upfront (for self-contained checkpoint)
        logger.info("Loading rules library...")
        all_rules_library = load_rules_from_library(config)
        logger.info(f"Stored {len(all_rules_library)} rules in checkpoint for resume support")
        logger.info("")

        # Initialize evaluation tracking
        evaluation_results = {
            'timestamp': timestamp,
            'folder_name': folder_name,
            'config': {
                'num_rules': num_rules,
                'num_rounds_per_rule': num_rounds_per_rule,
                'rule_compiler': rule_compiler_display_name,
                'rule_compiler_model': config['rule_compiler']['model'],
                'rule_compiler_temperature': config['rule_compiler'].get('temperature', 0.8),
                'player': player_display_name,
                'player_model': player_model,
                'hand_size': game_config.get('hand_size', 12),
                'max_turns': game_config.get('max_turns', 40),
                'wrong_guess_penalty': game_config.get('wrong_guess_penalty', 3),
                'seed': game_config.get('seed'),
                'llm_max_tokens': config['llm'].get('max_tokens', 8192),
                'llm_temperature': config['llm'].get('temperature', 0.7),
                'llm_seed': config['llm'].get('seed'),
                'llm_max_retries': config['llm'].get('max_llm_retries', 3),
            },
            'rounds': [],
            'statistics': {
                'total_score': 0,
                'successful_rounds': 0,
                'failed_rounds': 0,
                'total_turns': 0,
                'total_failed_guesses': 0,
                'total_output_tokens': 0,
                'total_reasoning_tokens': 0,
                'total_answer_tokens': 0,
                'total_wall_clock_seconds': 0.0,
                'total_retries': 0,
                'retry_by_cause': {},
            },
            'checkpoint': {
                'completed_rounds': 0,
                'total_rounds': num_rounds,
                'rule_factory_state': {
                    'selection': rules_cfg["selection"],
                    'current_index': rule_factory_index,
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

        # Determine if new rule needed and batch round index
        batch_round_index = (round_num - 1) % num_rounds_per_rule
        need_new_rule = batch_round_index == 0

        if need_new_rule:
            logger.info("Generating new rule for this batch of rounds...")
            current_rule = None  # Force new rule generation
        else:
            batch_num = (round_num - 1) // num_rounds_per_rule + 1
            logger.info(f"Reusing rule from previous round (batch {batch_num})")

        # Track if we're generating a new rule this round
        generated_new_rule = current_rule is None

        # Play round with rule_factory_index and batch_round_index (for unique seeding)
        result = play_round(
            config=config,
            round_number=round_num,
            rule=current_rule,
            start_rule_index=rule_factory_index if need_new_rule else None,
            rules_list=checkpoint_rules_library,
            batch_round_index=batch_round_index,
        )

        # Update current rule for reuse and track consumption
        if generated_new_rule:
            current_rule = Rule(result['rule_description'], result['rule_code'])
            rule_metadata = result.get('rule_metadata', {}) or {}
            current_rule_name = rule_metadata.get('name')

            # Increment rule_factory_index for RuleFactory (used for start position on next rule)
            rule_factory_index += 1

            # Add newly consumed rule to rules_consumed list
            if 'rules_consumed' not in evaluation_results.get('checkpoint', {}):
                if 'checkpoint' not in evaluation_results:
                    evaluation_results['checkpoint'] = {}
                evaluation_results['checkpoint']['rules_consumed'] = []

            evaluation_results['checkpoint']['rules_consumed'].append({
                'name': current_rule_name,
                'description': current_rule.description(),
                'code': current_rule.get_code(),
                'rounds_completed': 1,
            })
        else:
            # Reusing rule - increment rounds_completed for current rule
            if evaluation_results.get('checkpoint', {}).get('rules_consumed'):
                evaluation_results['checkpoint']['rules_consumed'][-1]['rounds_completed'] += 1

        # Update evaluation results
        evaluation_results['rounds'].append({
            'round_number': round_num,
            'rule_name': current_rule_name,
            'batch_round_index': batch_round_index,
            'turn_count': result['turn_count'],
            'rule_description': result['rule_description'],
            'rule_code': result['rule_code'],
            'success': result['success'],
            'score': result['score'],
            'failed_guesses': result['failed_guesses'],
            'game_over_reason': result['game_over_reason'],
            'llm_usage': result['llm_usage'],
            'turns': result['turns'],
            'wall_clock_seconds': result['wall_clock_seconds'],
        })

        # Update statistics
        evaluation_results['statistics']['total_score'] += result['score']
        if result['success']:
            evaluation_results['statistics']['successful_rounds'] += 1
        else:
            evaluation_results['statistics']['failed_rounds'] += 1
        evaluation_results['statistics']['total_turns'] += result['turn_count']
        evaluation_results['statistics']['total_failed_guesses'] += result['failed_guesses']

        # Aggregate token counts from player LLM usage
        player_usage = result['llm_usage'].get('player', {})
        evaluation_results['statistics']['total_output_tokens'] += player_usage.get('output_tokens', 0)
        evaluation_results['statistics']['total_reasoning_tokens'] += player_usage.get('reasoning_tokens', 0)
        evaluation_results['statistics']['total_answer_tokens'] += player_usage.get('answer_tokens', 0)
        evaluation_results['statistics']['total_wall_clock_seconds'] += result.get('wall_clock_seconds', 0)

        # Aggregate retry counts from turns
        for turn in result.get('turns', []):
            retry_count = turn.get('retry_count', 0)
            evaluation_results['statistics']['total_retries'] += retry_count
            for retry_info in turn.get('retry_causes', []):
                cause = retry_info.get('cause', 'unknown')
                retry_by_cause = evaluation_results['statistics']['retry_by_cause']
                retry_by_cause[cause] = retry_by_cause.get(cause, 0) + 1

        # Update checkpoint (preserve rules_library from initial load)
        evaluation_results['checkpoint'] = {
            'completed_rounds': round_num,
            'total_rounds': num_rounds,
            'rule_factory_state': {
                'selection': rules_cfg["selection"],
                'current_index': rule_factory_index,
            },
            'current_rule': {
                'description': current_rule.description() if current_rule else None,
                'code': current_rule.get_code() if current_rule else None,
                'rounds_used_in_batch': (round_num - 1) % num_rounds_per_rule + 1,
                'num_rounds_per_rule': num_rounds_per_rule,
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
    logger.info("")
    logger.info(f"Total output tokens: {stats['total_output_tokens']:,}")
    logger.info(f"Total reasoning tokens: {stats['total_reasoning_tokens']:,}")
    logger.info(f"Total answer tokens: {stats['total_answer_tokens']:,}")
    logger.info(f"Total wall clock time: {stats['total_wall_clock_seconds']:.1f}s")
    if stats.get('total_retries', 0) > 0:
        logger.info("")
        logger.info(f"Total LLM retries: {stats['total_retries']}")
        for cause, count in stats.get('retry_by_cause', {}).items():
            logger.info(f"  - {cause}: {count}")

    # Save final results
    output_file = save_evaluation_results(evaluation_results, folder_name)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
