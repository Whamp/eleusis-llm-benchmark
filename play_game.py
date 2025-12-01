"""Play a full game of Eleusis with LLM players."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

from eleusis.game_engine import GameEngine
from eleusis.game_state import GameState
from eleusis.llm_client import HuggingFaceClient
from eleusis.llm_player import LLMRuleMaker, LLMScientist
from eleusis.rules import RuleValidator

# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/game_log_{timestamp}.txt"

# Configure logging with dual levels: DEBUG to file, INFO to console
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)  # DEBUG for file
file_handler.setFormatter(logging.Formatter("%(message)s"))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # INFO for console
console_handler.setFormatter(logging.Formatter("%(message)s"))

logging.basicConfig(
    level=logging.DEBUG,  # Root logger at DEBUG to capture everything
    handlers=[file_handler, console_handler],
)

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)



def play_full_game():
    """Play a complete game."""
    logger.info(f"Log file: {log_file}")
    logger.info(f"  - Rule-maker: {config['models']['rule_maker']['display_name']}")
    logger.info(f"  - Scientist 1: {config['models']['scientist_1']['display_name']}")
    logger.info(f"  - Scientist 2: {config['models']['scientist_2']['display_name']}")
    logger.info(f"  - Scientist 3: {config['models']['scientist_3']['display_name']}")
    logger.info("=" * 80)
    logger.info("")

    # Initialize clients from config
    try:
        logger.info("Initializing LLM clients...")

        rule_maker_cfg = config["models"]["rule_maker"]
        rule_maker_client = HuggingFaceClient(
            model_name=rule_maker_cfg["name"],
            temperature=rule_maker_cfg["temperature"],
        )
        logger.info("✓ Rule-maker client initialized")

        scientist1_cfg = config["models"]["scientist_1"]
        scientist1_client = HuggingFaceClient(
            model_name=scientist1_cfg["name"],
            temperature=scientist1_cfg["temperature"],
        )

        scientist2_cfg = config["models"]["scientist_2"]
        scientist2_client = HuggingFaceClient(
            model_name=scientist2_cfg["name"],
            temperature=scientist2_cfg["temperature"],
        )

        scientist3_cfg = config["models"]["scientist_3"]
        scientist3_client = HuggingFaceClient(
            model_name=scientist3_cfg["name"],
            temperature=scientist3_cfg["temperature"],
        )
        logger.info("✓ All scientist clients initialized")
        logger.info("")

    except Exception as e:
        logger.error(f"✗ Failed to initialize clients: {e}")
        logger.error("Make sure HF_TOKEN environment variable is set!")
        return

    # Create validator
    logger.info("=" * 80)
    logger.info("PHASE 1: RULE GENERATION")
    logger.info("=" * 80)
    logger.info("")

    validator = RuleValidator(referee_client=None)

    # Generate rule
    logger.info("Rule-maker is creating a secret rule...")
    logger.info("")
    max_rule_attempts = config["game"]["max_rule_generation_attempts"]
    max_tokens = config["game"]["max_tokens"]
    rule_maker = LLMRuleMaker(
        rule_maker_client, validator, max_attempts=max_rule_attempts, max_tokens=max_tokens
    )
    rule = rule_maker.generate_rule()

    if not rule:
        logger.error("✗ Failed to generate valid rule")
        return

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"SECRET RULE: {rule.description()}")
    logger.info("=" * 80)
    logger.info("")

    # Create game
    logger.info("=" * 80)
    logger.info("PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info("")

    players = ["RuleMaker", "Scientist1", "Scientist2", "Scientist3"]
    game_state = GameState(players, rule_maker_index=0)
    cards_per_scientist = config["game"]["cards_per_scientist"]
    correct_guess_bonus = config["game"]["correct_guess_bonus"]
    engine = GameEngine(
        game_state,
        rule,
        rule_validator=None,
        cards_per_scientist=cards_per_scientist,
        correct_guess_bonus=correct_guess_bonus,
    )

    # Setup game
    engine.setup_game()
    logger.info("✓ Game setup complete")
    logger.info(f"✓ Starter card placed: {game_state.mainline.get_last()}")
    logger.info(f"✓ Each scientist has {cards_per_scientist} cards")
    logger.info(f"✓ Deck has {game_state.deck.remaining_count()} cards remaining")
    logger.info("")

    # Create scientist players
    guess_threshold = config["game"]["scientist_guess_threshold"]
    max_llm_retries = config["game"]["max_llm_retries"]
    scientists = [
        LLMScientist(
            "Scientist1",
            scientist1_client,
            guess_threshold=guess_threshold,
            max_retries=max_llm_retries,
            max_tokens=max_tokens,
        ),
        LLMScientist(
            "Scientist2",
            scientist2_client,
            guess_threshold=guess_threshold,
            max_retries=max_llm_retries,
            max_tokens=max_tokens,
        ),
        LLMScientist(
            "Scientist3",
            scientist3_client,
            guess_threshold=guess_threshold,
            max_retries=max_llm_retries,
            max_tokens=max_tokens,
        ),
    ]

    # Game loop
    logger.info("=" * 80)
    logger.info("PHASE 3: GAME PLAY")
    logger.info("=" * 80)
    logger.info("")

    max_turns = config["game"]["max_turns"]
    turn_count = 0

    while turn_count < max_turns and not engine.is_game_over():
        current_player_state = game_state.get_current_player()
        player_name = current_player_state.name

        player = next((p for p in scientists if p.name == player_name), None)
        if not player:
            logger.error(f"No player found for {player_name}")
            break

        # Log turn header with compact board
        logger.info("-" * 80)
        logger.info(f"TURN {turn_count + 1}: {player_name}")
        logger.info("-" * 80)
        logger.info(f"Board: {game_state.to_compact_string()}")
        logger.info(f"Hand size {current_player_state.hand.size()} cards")
        logger.info(f"Deck remaining: {game_state.deck.remaining_count()} cards")
        logger.info("")

        # Get action
        can_guess = current_player_state.hand.size() > 0
        try:
            action = player.get_action(game_state, can_guess=can_guess)
        except Exception as e:
            logger.error(f"Error getting action: {e}", exc_info=True)
            game_state.advance_turn()
            turn_count += 1
            continue

        # Play turn
        result = engine.play_turn(action)

        # Log result
        logger.info(f"Action: {result['action']}")
        if "card" in result:
            logger.info(f"Card played: {result['card']}")
            logger.info(f"Result: {'ACCEPTED ✓' if result.get('accepted') else 'REJECTED ✗'}")

        elif "correct" in result:
            logger.info(f"No-play: {'CORRECT ✓' if result['correct'] else 'INCORRECT ✗'}")
            if "forced_card" in result:
                logger.info(f"Forced card: {result['forced_card']}")

        if "guess" in result:
            logger.info("")
            logger.info("RULE GUESS!")
            logger.info(f"Guess: {result['guess']}")
            logger.info(f"Verdict: {'CORRECT ✓✓✓' if result.get('correct') else 'INCORRECT ✗'}")
            if result.get("correct"):
                logger.info("=" * 80)
                logger.info(f"GAME OVER! {player_name} won!")
                logger.info("=" * 80)
                break

        logger.info("")
        turn_count += 1

        # Pause after turn if configured
        if config["game"]["pause_after_turn"]:
            input(f"[Turn {turn_count} complete] Press Enter to continue...")

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("GAME SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Turns: {turn_count}")
    logger.info(f"Secret rule: {rule.description()}")
    logger.info(f"Final board: {game_state.to_compact_string()}")
    logger.info("")

    # Sidelines summary
    if game_state.sidelines:
        logger.info("Rejected cards by position:")
        for pos, sideline in sorted(game_state.sidelines.items()):
            cards = [str(c) for c in sideline.get_cards()]
            logger.info(f"  Position {pos}: {cards}")
        logger.info("")

    scores = engine.calculate_scores()
    logger.info("Final scores (lower is better):")
    for name, score in sorted(scores.items(), key=lambda x: x[1]):
        logger.info(f"  {name}: {score}")

    logger.info("")
    logger.info(f"Game log: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    play_full_game()
