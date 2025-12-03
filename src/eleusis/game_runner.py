"""Game runner for playing rounds of Eleusis."""

import logging

from eleusis.game_engine import GameEngine, GuessRuleAction, Rule
from eleusis.game_master import GameMaster
from eleusis.game_state import GameState
from eleusis.llm_client import HuggingFaceClient
from eleusis.player import LLMScientist
from eleusis.rule_factory import RuleFactory
from eleusis.rules import RuleValidator

logger = logging.getLogger(__name__)


def play_round(
    config: dict,
    round_number: int,
    rule: Rule | None = None,
    max_turns: int | None = None,
) -> dict:
    """Play a single round of Eleusis.

    Args:
        config: Full game configuration dict
        round_number: Current round number (for logging)
        rule: Optional rule to reuse (if None, generate/load new rule)
        max_turns: Optional override for max turns

    Returns:
        dict with round_number, turn_count, rule_description, rule_code,
        winning_player, scores, game_over_reason
    """
    from eleusis.game_engine import GuessRuleAction

    # --------------------
    # (1) Client initialization
    # --------------------

    player_configs = config["models"]["players"]
    max_tokens = config["models"]["max_tokens"]

    game_master_cfg = config["models"]["game_master"]
    game_master_client = HuggingFaceClient(
        model_name=game_master_cfg["name"],
        temperature=game_master_cfg["temperature"],
        max_tokens=max_tokens,
    )

    scientist_clients = []
    for player_cfg in player_configs:
        client = HuggingFaceClient(
            model_name=player_cfg["name"],
            temperature=player_cfg["temperature"],
            max_tokens=max_tokens,
        )
        scientist_clients.append(client)

    # --------------------
    # (2) Rule generation
    # --------------------

    if rule is None:
        logger.info("=" * 80)
        logger.info(f"[Round {round_number}] PHASE 1: RULE GENERATION")
        logger.info("=" * 80)
        logger.info("")

        # Initialize game master for rule operations
        game_master = GameMaster(
            llm_client=game_master_client,
            max_retry_attempts=config["game"].get("max_rule_retry_attempts", 3),
        )
        logger.info("✓ Game master initialized")

        validator = RuleValidator(referee_client=game_master_client)

        rule_source_cfg = config["rule_source"]
        mode = rule_source_cfg["mode"]

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
            logger.info("Game master is creating a secret rule...")
            rule_factory = RuleFactory(
                mode="llm",
                game_master=game_master,
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
    else:
        logger.info(f"[Round {round_number}] Using provided rule")
        game_master = GameMaster(
            llm_client=game_master_client,
            max_retry_attempts=config["game"].get("max_rule_retry_attempts", 3),
        )
        validator = RuleValidator(referee_client=game_master_client)

    # ---------------
    # (3) Game setup
    # ---------------

    logger.info("=" * 80)
    logger.info(f"[Round {round_number}] PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info("")

    player_configs = config["models"]["players"]
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
        game_master=game_master,
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
    max_llm_retries = config["models"]["max_llm_retries"]
    scientists = []
    for i, client in enumerate(scientist_clients):
        scientist = LLMScientist(
            player_configs[i]["display_name"],
            client,
            max_retries=max_llm_retries,
        )
        scientists.append(scientist)
        logger.info(f"✓ scientist player(s) # {i+1} initialized: {scientist.name}")
    logger.info("")

    # --------------
    # (4) MAIN LOOP
    # --------------

    logger.info("=" * 80)
    logger.info(f"[Round {round_number}] PHASE 3: GAME PLAY")
    logger.info("=" * 80)
    logger.info("")

    max_turns_limit = max_turns or config["game"]["max_turns"]
    turn_count = 0
    game_over_reason = "max_turns"

    while turn_count < max_turns_limit and not engine.is_game_over():
        current_player_state = game_state.get_current_player()
        player_name = current_player_state.name

        player = next((p for p in scientists if p.name == player_name), None)
        if not player:
            logger.error(f"No player found for {player_name}")
            break

        # Log turn header
        logger.info("=" * 80)
        logger.info(f"TURN {turn_count + 1}: {player_name}")
        logger.info("=" * 80)
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
            confidence_level = player.last_action_response.get("confidence_level", "")
            guess_if_accepted = player.last_action_response.get("guess_rule_if_accepted", False)

            logger.info(f"Reasoning: {reasoning}")
            logger.info(f"Tentative rule: {tentative_rule}")
            logger.info(f"Confidence level: {confidence_level}")
            logger.info(f"Will guess if accepted: {guess_if_accepted}")
            logger.info("")

        # Check if player wants to guess (before action is processed)
        will_guess = (
            player.last_action_response
            and player.last_action_response.get("guess_rule_if_accepted", False)
        )

        play_result = engine.play_turn(action, advance_turn=False)
        can_guess = "card" in play_result and play_result.get("accepted", False) or "correct" in play_result and play_result.get("correct", False)
        logger.info(f"Can guess this turn: {'YES' if can_guess else 'NO'}")

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
        if will_guess and can_guess and guess_text:
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
                game_over_reason = "correct_guess"
                break

        logger.info("")
        turn_count += 1

        # Pause after turn if configured
        if config["game"]["pause_after_turn"]:
            input(f"[Turn {turn_count} complete] Press Enter to continue...")

    # Check if game ended due to deck empty
    if engine.is_game_over() and game_over_reason != "correct_guess":
        game_over_reason = "deck_empty"

    # -------------
    # (5) Scoring
    # -------------

    scores = engine.calculate_scores()
    winning_player = engine.winning_guesser if engine.rule_guessed else None

    # Remove RuleMaker from scores (only track scientists)
    if "RuleMaker" in scores:
        del scores["RuleMaker"]

    return {
        'round_number': round_number,
        'turn_count': turn_count,
        'rule_description': rule.description(),
        'rule_code': rule.get_code(),
        'winning_player': winning_player,
        'scores': scores,
        'game_over_reason': game_over_reason,
    }
