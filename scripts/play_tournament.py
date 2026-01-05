"""Play a tournament of multiple Eleusis rounds."""

import json
import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game_engine import Rule
from eleusis.game_runner import play_round
from eleusis.logging_utils import setup_logging

# Load environment variables
load_dotenv()

# Load configuration (tournament mode uses config_tournament.yaml)
config_path = Path(__file__).parent.parent / "config_tournament.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/tournament_log_{timestamp}.txt"
setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def save_tournament_results(tournament_results: dict, timestamp: str) -> str:
    """Save tournament results to JSON file (incremental)."""
    Path(f"results/tournament_results_{timestamp}").mkdir(parents=True, exist_ok=True)
    output_file = f"results/tournament_results_{timestamp}/results.json"
    with open(output_file, 'w') as f:
        json.dump(tournament_results, f, indent=2)
    return output_file


def main():
    """Play a full tournament with multiple rounds."""

    # Load tournament config
    num_rounds = config["rule_source"]["tournament_rounds"]
    rounds_per_rule = config["rule_source"].get("rounds_per_rule", 1)
    player_configs = config["models"]["players"]

    logger.info("=" * 80)
    logger.info(f"ELEUSIS TOURNAMENT - {num_rounds} ROUNDS")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Rounds per rule: {rounds_per_rule}")
    logger.info(f"  - Game Master: {config['models']['game_master']['display_name']}")
    for i, player_cfg in enumerate(player_configs, 1):
        logger.info(f"  - Scientist {i}: {player_cfg['display_name']}")
    logger.info("")

    # Initialize tournament tracking
    tournament_results = {
        'timestamp': timestamp,
        'config': {
            'num_rounds': num_rounds,
            'rounds_per_rule': rounds_per_rule,
            'game_master': config['models']['game_master']['display_name'],
            'scientists': [cfg['display_name'] for cfg in player_configs],
        },
        'rounds': [],
        'cumulative_scores': {},
        'win_counts': {},
    }

    # Initialize player stats (scientists only, no RuleMaker)
    player_names = [cfg['display_name'] for cfg in player_configs]
    for name in player_names:
        tournament_results['cumulative_scores'][name] = 0
        tournament_results['win_counts'][name] = 0

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
        result = play_round(
            config=config,
            round_number=round_num,
            rule=current_rule,
        )

        # Update current rule for reuse
        if current_rule is None:
            current_rule = Rule(result['rule_description'], result['rule_code'])

        # Update tournament stats
        tournament_results['rounds'].append({
            'round_number': round_num,
            'turn_count': result['turn_count'],
            'rule_description': result['rule_description'],
            'rule_code': result['rule_code'],
            'winning_player': result['winning_player'],
            'scores': result['scores'],
            'game_over_reason': result['game_over_reason'],
            'llm_usage': result['llm_usage'],
            'turns': result['turns'],
        })

        # Update cumulative scores (only scientists, no RuleMaker)
        for player, score in result['scores'].items():
            if player in tournament_results['cumulative_scores']:
                tournament_results['cumulative_scores'][player] += score

        # Update win counts
        if result['winning_player']:
            tournament_results['win_counts'][result['winning_player']] += 1

        # Save incrementally after each round
        output_file = save_tournament_results(tournament_results, timestamp)
        logger.info(f"Progress saved to: {output_file}")

        # Log round summary
        logger.info("")
        logger.info(f"Round {round_num} complete:")
        logger.info(f"  Turns: {result['turn_count']}")
        logger.info(f"  Winner: {result['winning_player'] or 'None'}")
        logger.info(f"  Rule: {result['rule_description'][:80]}...")
        logger.info("")

    # Calculate final rankings
    logger.info("=" * 80)
    logger.info("TOURNAMENT SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    logger.info("Cumulative Scores (lower is better):")
    sorted_by_score = sorted(
        tournament_results['cumulative_scores'].items(),
        key=lambda x: x[1]
    )
    for rank, (player, score) in enumerate(sorted_by_score, 1):
        wins = tournament_results['win_counts'][player]
        logger.info(f"  {rank}. {player}: {score} points ({wins} wins)")

    logger.info("")
    logger.info("Win Counts:")
    sorted_by_wins = sorted(
        tournament_results['win_counts'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for rank, (player, wins) in enumerate(sorted_by_wins, 1):
        score = tournament_results['cumulative_scores'][player]
        logger.info(f"  {rank}. {player}: {wins} wins ({score} points)")

    # Save results to JSON (final save)
    output_file = save_tournament_results(tournament_results, timestamp)

    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
