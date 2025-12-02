"""Play a full game of Eleusis with LLM players."""

import logging
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from eleusis.game_engine import GameEngine
from eleusis.game_state import GameState
from eleusis.llm_client import HuggingFaceClient
from eleusis.llm_player import LLMScientist
from eleusis.logging_utils import setup_logging
from eleusis.rule_factory import RuleFactory
from eleusis.rules import RuleValidator

# Load environment variables from .env
load_dotenv()

# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Logging setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/game_log_{timestamp}.txt"
setup_logging(log_file=log_file, console_level=logging.INFO, file_level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx info logs
logger = logging.getLogger(__name__)


def play_full_game():
    """Play a complete game."""

    # ---------------
    # (0) Initialize
    # ---------------

    player_configs = config["models"]["players"]

    logger.info(f"Log file: {log_file}")
    logger.info(f"  - Rule-maker: {config['models']['rule_maker']['display_name']}")
    for i, player_cfg in enumerate(player_configs, 1):
        logger.info(f"  - Scientist {i}: {player_cfg['display_name']}")
    logger.info("")

    rule_maker_cfg = config["models"]["rule_maker"]
    rule_maker_client = HuggingFaceClient(
        model_name=rule_maker_cfg["name"],
        temperature=rule_maker_cfg["temperature"],
    )
    logger.info("✓ Rule-maker client initialized")

    scientist_clients = []
    for i, player_cfg in enumerate(player_configs, 1):
        client = HuggingFaceClient(
            model_name=player_cfg["name"],
            temperature=player_cfg["temperature"],
        )
        scientist_clients.append(client)
    logger.info(f"✓ {len(scientist_clients)} scientist client(s) initialized")

    # Initialize referee client for rule comparison
    from eleusis.llm_client import RefereeClient

    referee_cfg = config["models"]["referee"]
    max_tokens_referee = config["game"]["max_tokens_referee"]
    referee_client = RefereeClient(
        model_name=referee_cfg["name"],
        temperature=referee_cfg["temperature"],
        max_tokens=max_tokens_referee,
    )
    logger.info(f"✓ Referee client initialized: {referee_cfg['display_name']}")


    # --------------------
    # (1) Rule generation
    # --------------------

    logger.info("=" * 80)
    logger.info("PHASE 1: RULE GENERATION")
    logger.info("=" * 80)
    logger.info("")

    validator = RuleValidator(referee_client=referee_client)

    rule_source_cfg = config["rule_source"]
    mode = rule_source_cfg["mode"]

    max_tokens = config["game"]["max_tokens"]
    min_acceptance = rule_source_cfg.get("min_acceptance", 0.0)
    max_acceptance = rule_source_cfg.get("max_acceptance", 1.0)

    logger.info(f"Rule source mode: {mode}")
    logger.info(f"Acceptance rate bounds: [{min_acceptance:.2%}, {max_acceptance:.2%}]")

    if mode == "library":
        logger.info("Loading rule from library...")
        start_index = rule_source_cfg.get("index", 0)
        rule_factory = RuleFactory(
            mode="library",
            library_path=rule_source_cfg["library_path"],
            selection=rule_source_cfg["selection"],
            min_acceptance=min_acceptance,
            max_acceptance=max_acceptance,
            start_index=start_index,
        )
    else:  # llm mode
        logger.info("Rule-maker is creating a secret rule...")
        rule_factory = RuleFactory(
            mode="llm",
            llm_client=rule_maker_client,
            validator=validator,
            min_acceptance=min_acceptance,
            max_acceptance=max_acceptance,
        )

    rule = rule_factory.create_rule()

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"SECRET RULE: {rule.description()}")
    logger.info("=" * 80)
    logger.info("")

    # ---------------
    # (2) Game setup
    # ---------------

    logger.info("=" * 80)
    logger.info("PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info("")

    player_names = ["RuleMaker"] + [player_cfg['display_name'] for player_cfg in player_configs]
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
    for i, client in enumerate(scientist_clients):
        scientist = LLMScientist(
            player_configs[i]["display_name"],
            client,
            max_retries=max_llm_retries,
            max_tokens=max_tokens,
        )
        scientists.append(scientist)
        logger.info(f"✓ scientist player(s) # {i+1} initialized: {scientist.name}")
    logger.info("")


    # --------------
    # (3) MAIN LOOP
    # --------------

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

        # Log turn header
        logger.info("-" * 80)
        logger.info(f"TURN {turn_count + 1}: {player_name}")
        logger.info("-" * 80)
        logger.info(f"Board: {game_state.to_compact_string()}")
        logger.info(f"Deck remaining: {game_state.deck.remaining_count()} cards")
        hand_cards = current_player_state.hand.get_all_cards()
        hand_str = ", ".join([str(c) for c in hand_cards])
        logger.info(f"Hand ({len(hand_cards)} cards): {hand_str}")
        logger.info("")

        try:
            action = player.get_action(game_state)
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

        # Check if player wants to guess (before action is processed)
        will_guess = (
            player.last_action_response
            and player.last_action_response.get("guess_rule_if_accepted", False)
        )

        play_result = engine.play_turn(action, advance_turn=False)

        # Log result
        logger.info(f"Action: {play_result['action']}")
        if "card" in play_result:       # Card play
            logger.info(f"Card played: {play_result['card']}")
            logger.info(f"Result: {'ACCEPTED ✓' if play_result.get('accepted') else 'REJECTED ✗'}")

        elif "correct" in play_result:  # No-play
            logger.info(f"No-play: {'CORRECT ✓' if play_result['correct'] else 'INCORRECT ✗'}")
            if "forced_card" in play_result:
                logger.info(f"Forced card: {play_result['forced_card']}")

        # Check if guess should execute (action was successful and player wanted to guess)

        guess_text = (
            player.last_action_response.get("tentative_rule", "")
            if player.last_action_response
            else ""
        )

        # Record the play result (this clears last_action_response)
        player.record_action_result(play_result)

        # Execute the guess if needed (will advance turn)
        if will_guess and guess_text:
            from eleusis.game_engine import GuessRuleAction

            logger.info("")
            logger.info(f"{player_name} is guessing the rule based on tentative_rule...")
            result = engine.play_turn(GuessRuleAction(guess_text))  # This will advance turn
        else:
            # No guess - manually advance turn
            game_state.advance_turn()
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

    scores = engine.calculate_scores()
    logger.info("Final scores (lower is better):")
    for name, score in sorted(scores.items(), key=lambda x: x[1]):
        logger.info(f"  {name}: {score}")

    logger.info("")
    logger.info(f"Game log: {log_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    play_full_game()
