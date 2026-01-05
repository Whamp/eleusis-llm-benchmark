"""Play a single round of Eleusis with LLM players."""

import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game_runner import play_round
from eleusis.logging_utils import setup_logging

# Load environment variables from .env
load_dotenv()

# Load configuration (tournament mode uses config_tournament.yaml)
config_path = Path(__file__).parent.parent / "config_tournament.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/game_log_{timestamp}.txt"
setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def main():
    """Play a single round of Eleusis."""

    # Load configuration
    player_configs = config["models"]["players"]

    # Log setup
    logger.info(f"Log file: {log_file}")
    logger.info(f"  - Game-Master: {config['models']['game_master']['display_name']}")
    for i, player_cfg in enumerate(player_configs, 1):
        logger.info(f"  - Scientist {i}: {player_cfg['display_name']}")
    logger.info("")

    # Play single round
    result = play_round(
        config=config,
        round_number=1,
    )

    # Log final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("GAME SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Turns: {result['turn_count']}")
    logger.info(f"Secret rule: {result['rule_description']}")
    logger.info("")

    logger.info("Final scores (lower is better):")
    for name, score in sorted(result['scores'].items(), key=lambda x: x[1]):
        logger.info(f"  {name}: {score}")

    if result['winning_player']:
        logger.info(f"\nWinner: {result['winning_player']}")

    logger.info("")
    logger.info(f"Game log: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
