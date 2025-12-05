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
        role="game_master"
    )

    scientist_clients = []
    for player_cfg in player_configs:
        client = HuggingFaceClient(
            model_name=player_cfg["name"],
            temperature=player_cfg["temperature"],
            max_tokens=max_tokens,
            role=player_cfg["display_name"]
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
        game_master = GameMaster(llm_client=game_master_client)
        logger.info("✓ Game master initialized")

        # Reset usage stats for new round
        game_master_client.reset_usage_stats()
        for client in scientist_clients:
            client.reset_usage_stats()

        validator = RuleValidator()

        rule_source_cfg = config["rule_source"]

        min_acceptance = rule_source_cfg.get("min_acceptance", 0.0)
        max_acceptance = rule_source_cfg.get("max_acceptance", 1.0)

        logger.info("Loading rule from library...")
        logger.info(f"Acceptance rate bounds: [{min_acceptance:.2%}, {max_acceptance:.2%}]")

        start_index = rule_source_cfg.get("index", 0)
        rule_factory = RuleFactory(
            library_path=rule_source_cfg["library_path"],
            selection=rule_source_cfg["selection"],
            min_acceptance=min_acceptance,
            max_acceptance=max_acceptance,
            start_index=start_index,
        )

        rule = rule_factory.create_rule()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"SECRET RULE: {rule.description()}")
        logger.info("=" * 80)
        logger.info("")
    else:
        logger.info(f"[Round {round_number}] Using provided rule")
        game_master = GameMaster(llm_client=game_master_client)
        validator = RuleValidator()

    # ---------------
    # (3) Game setup
    # ---------------

    logger.info("=" * 80)
    logger.info(f"[Round {round_number}] PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info("")

    player_configs = config["models"]["players"]
    player_names = [player_cfg['display_name'] for player_cfg in player_configs]
    game_state = GameState(player_names)

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
    turn_data_list = []  # Track all turn data

    while turn_count < max_turns_limit and not engine.is_game_over():
        current_player_state = game_state.get_current_player()
        player_name = current_player_state.name

        # Capture state BEFORE action
        mainline_before = game_state.to_compact_string()
        hand_before = [str(c) for c in current_player_state.hand.get_all_cards()]

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

        # Safety check: player should not start turn with empty hand
        if len(hand_cards) == 0:
            logger.error(f"{player_name} started turn with empty hand - should not happen!")
            # This indicates a bug in the mandatory guess logic

        try:
            action = player.get_action(game_state)
        except Exception as e:
            logger.error(f"Error getting action: {e}", exc_info=True)
            game_state.advance_turn()
            turn_count += 1
            continue

        # Log ACTION details from LLM response
        if player.last_action_response:
            reasoning_summary = player.last_action_response.get("reasoning_summary", "")
            tentative_rule = player.last_action_response.get("tentative_rule", "")
            confidence_level = player.last_action_response.get("confidence_level", "")
            guess_if_accepted = player.last_action_response.get("guess_rule_if_accepted", False)

            logger.info(f"Reasoning: {reasoning_summary}")
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

        # Check for mandatory guess (hand reached 0 after successful action)
        player_hand_size_after = current_player_state.hand.size()
        action_was_successful = (
            play_result.get("success", False) and
            (play_result.get("accepted", False) or play_result.get("correct", False))
        )
        mandatory_guess = action_was_successful and player_hand_size_after == 0

        # Get tentative rule if available
        guess_text = (
            player.last_action_response.get("tentative_rule", "")
            if player.last_action_response
            else ""
        )

        # Handle mandatory guess without tentative_rule: draw penalty instead
        if mandatory_guess and not guess_text:
            logger.warning(
                f"{player_name} reached empty hand but no tentative_rule provided - "
                f"drawing 1 penalty card instead of forced guess"
            )
            if not game_state.deck.is_empty():
                drawn = game_state.deck.draw()
                current_player_state.hand.add_card(drawn)
                logger.info(f"{player_name} drew penalty card: {drawn}")
            mandatory_guess = False  # Cancel the forced guess

        # Update will_guess to include mandatory guess
        will_guess = will_guess or mandatory_guess

        # Log if mandatory
        if mandatory_guess:
            logger.info(f"{player_name} has empty hand after successful action - MANDATORY GUESS")

        # Create turn data entry
        turn_data = {
            "turn_number": turn_count + 1,
            "player": player_name,
            "mainline_state": mainline_before,
            "hand": hand_before,
            "llm_response": player.last_action_response.copy() if player.last_action_response else {},
            "action_result": {
                "action": play_result.get("action"),
                "card": play_result.get("card"),
                "accepted": play_result.get("accepted"),
                "correct": play_result.get("correct"),
                "forced_card": play_result.get("forced_card"),
                "success": play_result.get("success"),
            },
            "guess_attempt": None  # Will be filled if guess happens
        }

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
        # (guess_text already extracted above for mandatory guess check)

        # Record the play result (this clears last_action_response)
        player.record_action_result(play_result)

        # Execute the guess if needed (will advance turn)
        if will_guess and can_guess and guess_text:
            logger.info("")
            logger.info(f"{player_name} is guessing the rule based on tentative_rule...")
            result = engine.play_turn(GuessRuleAction(guess_text))  # This will advance turn

            # Add guess data to turn
            if "guess" in result:
                turn_data["guess_attempt"] = {
                    "guess": result["guess"],
                    "correct": result.get("correct", False),
                    "reasoning": result.get("reasoning", "")
                }
        else:
            # No guess - manually advance turn
            game_state.advance_turn()
            result = play_result

        # Store turn data
        turn_data_list.append(turn_data)

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

    # Collect LLM usage statistics
    llm_usage = {
        "game_master": game_master_client.get_usage_stats(),
        "scientists": {}
    }
    for i, client in enumerate(scientist_clients):
        player_name = player_configs[i]["display_name"]
        llm_usage["scientists"][player_name] = client.get_usage_stats()

    return {
        'round_number': round_number,
        'turn_count': turn_count,
        'rule_description': rule.description(),
        'rule_code': rule.get_code(),
        'winning_player': winning_player,
        'scores': scores,
        'game_over_reason': game_over_reason,
        'llm_usage': llm_usage,
        'turns': turn_data_list,
    }
