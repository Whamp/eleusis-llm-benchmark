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
    # Validate player configuration
    player_configs = config["models"]["players"]
    if len(player_configs) < 1:
        logger.error("✗ Must have at least 1 scientist player in config")
        return

    logger.info(f"Log file: {log_file}")
    logger.info(f"  - Rule-maker: {config['models']['rule_maker']['display_name']}")
    for i, player_cfg in enumerate(player_configs, 1):
        logger.info(f"  - Scientist {i}: {player_cfg['display_name']}")
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

        # Initialize scientist clients dynamically
        scientist_clients = []
        for i, player_cfg in enumerate(player_configs, 1):
            client = HuggingFaceClient(
                model_name=player_cfg["name"],
                temperature=player_cfg["temperature"],
            )
            scientist_clients.append(client)

        logger.info(f"✓ {len(scientist_clients)} scientist client(s) initialized")
        logger.info("")

    except Exception as e:
        logger.error(f"✗ Failed to initialize clients: {e}")
        logger.error("Make sure HF_TOKEN environment variable is set!")
        return

    # Create validator with referee
    logger.info("=" * 80)
    logger.info("PHASE 1: RULE GENERATION")
    logger.info("=" * 80)
    logger.info("")

    # Initialize referee client for rule comparison
    referee_client = None
    try:
        from eleusis.llm_client import RefereeClient

        max_tokens_referee = config["game"]["max_tokens_referee"]
        referee_client = RefereeClient(max_tokens=max_tokens_referee)
        logger.info("✓ Referee client initialized for rule comparison")
    except Exception as e:
        logger.warning(f"Could not initialize referee: {e}")
        logger.info("Continuing without referee (guesses will not be validated)")

    validator = RuleValidator(referee_client=referee_client)

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

    # Create player names dynamically
    num_scientists = len(player_configs)
    player_names = ["RuleMaker"] + [f"Scientist{i}" for i in range(1, num_scientists + 1)]

    game_state = GameState(player_names, rule_maker_index=0)
    cards_per_scientist = config["game"]["cards_per_scientist"]
    correct_guess_bonus = config["game"]["correct_guess_bonus"]
    card_reject_penalty = config["game"]["card_reject_penalty"]
    no_play_incorrect_penalty = config["game"]["no_play_incorrect_penalty"]
    no_play_correct_reduction = config["game"]["no_play_correct_reduction"]
    engine = GameEngine(
        game_state,
        rule,
        rule_validator=validator,
        cards_per_scientist=cards_per_scientist,
        correct_guess_bonus=correct_guess_bonus,
        card_reject_penalty=card_reject_penalty,
        no_play_incorrect_penalty=no_play_incorrect_penalty,
        no_play_correct_reduction=no_play_correct_reduction,
    )

    # Setup game
    engine.setup_game()
    logger.info("✓ Game setup complete")
    logger.info(f"✓ Starter card placed: {game_state.mainline.get_last()}")
    logger.info(f"✓ Each scientist has {cards_per_scientist} cards")
    logger.info(f"✓ Deck has {game_state.deck.remaining_count()} cards remaining")
    logger.info("")

    # Create scientist players dynamically
    max_llm_retries = config["game"]["max_llm_retries"]
    scientists = []
    for i, client in enumerate(scientist_clients, 1):
        scientist = LLMScientist(
            f"Scientist{i}",
            client,
            max_retries=max_llm_retries,
            max_tokens=max_tokens,
        )
        scientists.append(scientist)

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
        logger.info(f"Deck remaining: {game_state.deck.remaining_count()} cards")

        # Pretty print player's hand
        hand_cards = current_player_state.hand.get_all_cards()
        hand_str = ", ".join([str(c) for c in hand_cards])
        logger.info(f"Hand ({len(hand_cards)} cards): {hand_str}")
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

        # Log ACTION details from LLM response
        if player.last_action_response:
            reasoning = player.last_action_response.get("reasoning", "")
            tentative_rule = player.last_action_response.get("tentative_rule", "")
            guess_if_accepted = player.last_action_response.get("guess_rule_if_accepted", False)

            if reasoning:
                logger.info(f"Reasoning: {reasoning}")
            if tentative_rule:
                logger.info(f"Tentative rule: {tentative_rule}")
            logger.info(f"Will guess if accepted: {guess_if_accepted}")
            logger.info("")

        # Play turn
        play_result = engine.play_turn(action)

        # Log result
        logger.info(f"Action: {play_result['action']}")
        if "card" in play_result:
            logger.info(f"Card played: {play_result['card']}")
            logger.info(f"Result: {'ACCEPTED ✓' if play_result.get('accepted') else 'REJECTED ✗'}")

        elif "correct" in play_result:
            logger.info(f"No-play: {'CORRECT ✓' if play_result['correct'] else 'INCORRECT ✗'}")
            if "forced_card" in play_result:
                logger.info(f"Forced card: {play_result['forced_card']}")

        # Check if player wants to guess after successful action
        # (BEFORE clearing last_action_response)
        should_guess = (
            play_result.get("can_guess", False)
            and player.last_action_response
            and player.last_action_response.get("guess_rule_if_accepted", False)
        )
        guess_text = (
            player.last_action_response.get("tentative_rule", "")
            if player.last_action_response
            else ""
        )

        # Now record the play result (this clears last_action_response)
        player.record_action_result(play_result)

        # Execute the guess if needed (using saved values)
        if should_guess and guess_text:
            from eleusis.game_engine import GuessRuleAction

            logger.info("")
            logger.info(f"{player_name} is guessing the rule based on tentative_rule...")
            result = engine.play_turn(GuessRuleAction(guess_text))
        else:
            result = play_result

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
