"""Test script to run a single game and verify API connections."""

import logging
import sys

from dotenv import load_dotenv

from eleusis.game_engine import GameEngine
from eleusis.game_state import GameState
from eleusis.llm_client import HuggingFaceClient, RefereeClient
from eleusis.llm_player import LLMRuleMaker, LLMScientist, RandomScientist
from eleusis.rules import AlternatingColorsRule, RuleValidator

# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def test_with_predefined_rule():
    """Test game with a predefined rule (no LLM rule generation)."""
    logger.info("=" * 60)
    logger.info("TEST 1: Predefined Rule (Alternating Colors)")
    logger.info("=" * 60)

    # Create game state
    players = ["RuleMaker", "Scientist1", "Scientist2", "Scientist3"]
    game_state = GameState(players, rule_maker_index=0)

    # Use predefined rule
    rule = AlternatingColorsRule()
    logger.info(f"Rule: {rule.description()}")

    # Create referee client
    try:
        referee = RefereeClient()
        validator = RuleValidator(referee_client=referee)
        logger.info("✓ Referee client initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize referee: {e}")
        logger.info("Continuing without referee...")
        validator = None

    # Create game engine
    engine = GameEngine(game_state, rule, rule_validator=validator)

    # Setup game
    engine.setup_game()
    logger.info(f"✓ Game setup complete. Starter card: {game_state.mainline.get_last()}")

    # Create random scientists for testing
    scientists = [
        RandomScientist("Scientist1"),
        RandomScientist("Scientist2"),
        RandomScientist("Scientist3"),
    ]

    # Run a few turns
    max_turns = 10
    turn_count = 0

    logger.info("\nStarting game loop...")
    while turn_count < max_turns and not engine.is_game_over():
        current_player_state = game_state.get_current_player()
        player_name = current_player_state.name

        # Find corresponding player object
        player = next((p for p in scientists if p.name == player_name), None)
        if not player:
            logger.error(f"No player found for {player_name}")
            break

        # Get action
        action = player.get_action(game_state)

        # Play turn
        logger.info(f"\n--- Turn {turn_count + 1}: {player_name} ---")
        result = engine.play_turn(action)
        logger.info(f"Result: {result}")

        turn_count += 1

    # Show final state
    logger.info("\n" + "=" * 60)
    logger.info("Game Summary")
    logger.info("=" * 60)
    logger.info(f"Turns played: {turn_count}")
    logger.info(f"Mainline: {[str(c) for c in game_state.mainline.get_all()]}")
    for player_state in game_state.players:
        if not player_state.is_rule_maker:
            logger.info(f"{player_state.name}: {player_state.hand.size()} cards in hand")

    logger.info("\n✓ Test 1 completed successfully!\n")


def test_with_llm_rule():
    """Test game with LLM-generated rule."""
    logger.info("=" * 60)
    logger.info("TEST 2: LLM-Generated Rule")
    logger.info("=" * 60)

    # Initialize HuggingFace client
    try:
        # Use a good model for testing (Qwen is fast and reliable)
        llm_client = HuggingFaceClient(
            model_name="Qwen/Qwen3-4B-Thinking-2507", temperature=0.7
        )
        logger.info("✓ HuggingFace client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize HuggingFace client: {e}")
        logger.info("Skipping LLM rule test")
        logger.info("Note: Make sure HF_TOKEN environment variable is set")
        return

    # Initialize referee
    try:
        referee = RefereeClient()
        validator = RuleValidator(referee_client=referee)
        logger.info("✓ Referee client initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize referee: {e}")
        logger.info("Continuing without referee...")
        validator = RuleValidator()

    # Generate rule
    logger.info("\nGenerating rule with LLM...")
    rule_maker = LLMRuleMaker(llm_client, validator, max_attempts=3)
    rule = rule_maker.generate_rule()

    if not rule:
        logger.error("✗ Failed to generate valid rule")
        return

    logger.info(f"✓ Rule generated: {rule.description()}")

    # Create game state
    players = ["RuleMaker", "LLMScientist1", "RandomScientist1", "RandomScientist2"]
    game_state = GameState(players, rule_maker_index=0)

    # Create game engine
    engine = GameEngine(game_state, rule, rule_validator=validator)

    # Setup game
    engine.setup_game()
    logger.info(f"✓ Game setup complete. Starter card: {game_state.mainline.get_last()}")

    # Create players (1 LLM scientist, 2 random for speed)
    scientists = [
        LLMScientist("LLMScientist1", llm_client, guess_threshold=5),
        RandomScientist("RandomScientist1"),
        RandomScientist("RandomScientist2"),
    ]

    # Run a few turns
    max_turns = 15
    turn_count = 0

    logger.info("\nStarting game loop...")
    while turn_count < max_turns and not engine.is_game_over():
        current_player_state = game_state.get_current_player()
        player_name = current_player_state.name

        # Find corresponding player object
        player = next((p for p in scientists if p.name == player_name), None)
        if not player:
            logger.error(f"No player found for {player_name}")
            break

        # Get action (allow guessing after successful plays)
        action = player.get_action(game_state)

        # Play turn
        logger.info(f"\n--- Turn {turn_count + 1}: {player_name} ---")
        logger.info(f"Hand size: {current_player_state.hand.size()}")
        result = engine.play_turn(action)
        logger.info(f"Result: {result}")

        # Record play for LLM scientist
        if isinstance(player, LLMScientist) and "card" in result:
            # Note: We can't easily reconstruct the card here, skip recording
            pass

        if result.get("correct") and "guess" in result:
            logger.info(f"GAME OVER: {player_name} guessed the rule correctly!")
            break

        turn_count += 1

    # Show final state
    logger.info("\n" + "=" * 60)
    logger.info("Game Summary")
    logger.info("=" * 60)
    logger.info(f"Turns played: {turn_count}")
    logger.info(f"Mainline: {[str(c) for c in game_state.mainline.get_all()]}")
    logger.info(f"Secret rule was: {rule.description()}")
    for player_state in game_state.players:
        if not player_state.is_rule_maker:
            logger.info(f"{player_state.name}: {player_state.hand.size()} cards in hand")

    logger.info("\n✓ Test 2 completed!\n")


def main():
    """Run all tests."""
    logger.info("Starting Eleusis Test Suite\n")

    # Test 1: Predefined rule with random players (basic sanity check)
    try:
        test_with_predefined_rule()
    except Exception as e:
        logger.error(f"Test 1 failed with error: {e}", exc_info=True)

    # Test 2: LLM-generated rule (API connection test)
    try:
        test_with_llm_rule()
    except Exception as e:
        logger.error(f"Test 2 failed with error: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
