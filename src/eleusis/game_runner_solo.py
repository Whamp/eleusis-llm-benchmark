"""Game runner for solo pattern discovery mode."""

import logging
import re

from eleusis.game_engine_solo import GameEngineSolo, GuessRuleAction, Rule
from eleusis.game_master import GameMaster
from eleusis.game_state import GameState
from eleusis.llm_client import create_client
from eleusis.player import LLMScientist
from eleusis.rule_factory import RuleFactory
from eleusis.rules import RuleValidator
from eleusis.prompts_solo import get_solo_action_selection_prompt

logger = logging.getLogger(__name__)


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


class LLMScientistSolo(LLMScientist):
    """Scientist player for solo mode with modified prompt."""

    def __init__(self, name: str, llm_client, max_retries: int = 3, engine=None, max_turns: int = 40):
        """Initialize with engine reference for accessing failed_guess_count."""
        super().__init__(name, llm_client, max_retries)
        self.engine = engine
        self.max_turns = max_turns

    def _select_move(self, game_state, current_player):
        """Select a card to play using solo mode prompt."""
        hand_cards = current_player.hand.get_all_cards()
        if not hand_cards:
            # Should not happen in solo mode with constant hand size
            logger.error("Empty hand in solo mode - should not happen!")
            import random
            from eleusis.game_engine_solo import PlayCardAction
            return PlayCardAction(random.choice(hand_cards)) if hand_cards else None

        hand_dicts = [c.to_dict() for c in hand_cards]
        compact_board = game_state.to_compact_string()
        deck_remaining = game_state.deck.remaining_count()
        failed_guesses = game_state.failed_rule_guesses

        # Calculate current turn number (total turns taken so far)
        current_turn = game_state.round_number

        # Get failed guess count from engine
        failed_guess_count = self.engine.failed_guess_count if self.engine else 0

        prompt = get_solo_action_selection_prompt(
            compact_board=compact_board,
            hand_cards=hand_dicts,
            deck_remaining=deck_remaining,
            play_history=self.play_history,
            failed_guesses=failed_guesses,
            current_turn=current_turn,
            max_turns=self.max_turns,
            failed_guess_count=failed_guess_count,
        )

        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.generate(prompt, xml_tag="ACTION", return_dict=True)

                # Store response for history tracking
                self.last_action_response = response

                # Parse card field
                card_value = response.get("card", "").strip()

                # Find the card in hand
                from eleusis.game_engine_solo import PlayCardAction
                card = self._parse_card(card_value, hand_cards)
                if card:
                    logger.info(f"{self.name} plays {card}")
                    tentative = response.get("tentative_rule", "")
                    if tentative:
                        logger.debug(f"{self.name}'s tentative rule: {tentative}")
                    return PlayCardAction(card)

            except Exception as e:
                logger.warning(f"Move selection attempt {attempt + 1} failed: {e}")

        # Fallback: play random card
        logger.warning(f"{self.name} using random fallback")
        import random
        from eleusis.game_engine_solo import PlayCardAction
        return PlayCardAction(random.choice(hand_cards))


def play_round_solo(
    config: dict,
    round_number: int,
    rule: Rule | None = None,
    max_turns: int | None = None,
    start_rule_index: int | None = None,
) -> dict:
    """Play a single round of solo pattern discovery.

    Args:
        config: Full game configuration dict
        round_number: Current round number (for logging)
        rule: Optional rule to reuse (if None, generate/load new rule)
        max_turns: Optional override for max turns
        start_rule_index: Starting index for RuleFactory (for resume support)

    Returns:
        dict with round_number, turn_count, rule_description, rule_code,
        success, score, game_over_reason
    """

    # --------------------
    # (1) Client initialization
    # --------------------

    game_config = config["game"]
    llm_config = config["llm"]
    game_master_cfg = config["game_master"]

    player_model = config["model"]
    player_display_name = _model_spec_to_display_name(player_model)

    max_tokens = llm_config["max_tokens"]
    max_continuation = llm_config.get("max_continuation_attempts", 3)

    game_master_client = create_client(
        game_master_cfg["model_name"],
        temperature=game_master_cfg["temperature"],
        max_tokens=max_tokens,
        role="game_master",
        max_continuation_attempts=max_continuation,
    )

    scientist_client = create_client(
        player_model,
        temperature=llm_config["temperature"],
        max_tokens=max_tokens,
        role=player_display_name,
        max_continuation_attempts=max_continuation,
    )

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
        scientist_client.reset_usage_stats()

        validator = RuleValidator()

        rules_cfg = config["rules"]

        min_acceptance = rules_cfg.get("min_acceptance", 0.0)
        max_acceptance = rules_cfg.get("max_acceptance", 1.0)

        logger.info("Loading rule from library...")
        logger.info(f"Acceptance rate bounds: [{min_acceptance:.2%}, {max_acceptance:.2%}]")

        # Use provided start_rule_index for resume support, otherwise use config default
        start_index = start_rule_index if start_rule_index is not None else rules_cfg.get("index", 0)
        logger.info(f"Rule factory starting at index: {start_index}")

        rule_factory = RuleFactory(
            library_path=rules_cfg["library_path"],
            selection=rules_cfg["selection"],
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

    player_name = player_display_name
    game_state = GameState([player_name])

    hand_size = game_config.get('hand_size', 12)
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)

    engine = GameEngineSolo(
        game_state,
        rule,
        game_master=game_master,
        rule_validator=validator,
        hand_size=hand_size,
        wrong_guess_penalty=wrong_guess_penalty,
    )

    # Setup game
    engine.setup_game()

    logger.info("✓ Game setup complete")
    logger.info(f"✓ Starter card placed: {game_state.mainline.get_last()}")
    logger.info(f"✓ Player has {hand_size} cards (constant hand size)")
    logger.info(f"✓ Deck has {game_state.deck.remaining_count()} cards remaining")
    logger.info("")

    # Determine max_turns_limit first
    max_turns_limit = max_turns or game_config.get("max_turns", 40)

    # Create scientist player
    max_llm_retries = llm_config["max_llm_retries"]
    scientist = LLMScientistSolo(
        player_name,
        scientist_client,
        max_retries=max_llm_retries,
        engine=engine,
        max_turns=max_turns_limit,
    )
    logger.info(f"✓ Solo player initialized: {scientist.name}")
    logger.info("")

    # --------------
    # (4) MAIN LOOP
    # --------------

    logger.info("=" * 80)
    logger.info(f"[Round {round_number}] PHASE 3: GAME PLAY")
    logger.info("=" * 80)
    logger.info("")
    turn_count = 0
    game_over_reason = "max_turns"
    turn_data_list = []

    while turn_count < max_turns_limit and not engine.is_game_over():
        current_player_state = game_state.get_current_player()

        # Capture state BEFORE action
        mainline_before = game_state.to_compact_string()
        hand_before = [str(c) for c in current_player_state.hand.get_all_cards()]

        # Log turn header
        logger.info("=" * 80)
        logger.info(f"[Round {round_number}] TURN {turn_count + 1}: {player_name}")
        logger.info("=" * 80)
        logger.info(f"Board: {game_state.to_compact_string()}")
        logger.info(f"Deck remaining: {game_state.deck.remaining_count()} cards")
        hand_cards = current_player_state.hand.get_all_cards()
        hand_str = ", ".join([str(c) for c in hand_cards])
        logger.info(f"Hand ({len(hand_cards)} cards): {hand_str}")
        logger.info("")

        # Update the round_number to track turn for prompt
        game_state.round_number = turn_count + 1

        try:
            action = scientist.get_action(game_state)
        except Exception as e:
            logger.error(f"Error getting action: {e}", exc_info=True)
            game_state.advance_turn()
            turn_count += 1
            continue

        # Log ACTION details from LLM response
        if scientist.last_action_response:
            reasoning_summary = scientist.last_action_response.get("reasoning_summary", "")
            tentative_rule = scientist.last_action_response.get("tentative_rule", "")
            confidence_level = scientist.last_action_response.get("confidence_level", "")
            guess_rule = scientist.last_action_response.get("guess_rule", False)

            logger.info(f"Reasoning: {reasoning_summary}")
            logger.info(f"Tentative rule: {tentative_rule}")
            logger.info(f"Confidence level: {confidence_level}")
            logger.info(f"Will guess: {guess_rule}")
            logger.info("")

        # Check if player wants to guess
        will_guess = (
            scientist.last_action_response
            and scientist.last_action_response.get("guess_rule", False)
        )

        # Get tentative rule if available
        guess_text = (
            scientist.last_action_response.get("tentative_rule", "")
            if scientist.last_action_response
            else ""
        )

        play_result = engine.play_turn(action, advance_turn=False)

        # Create turn data entry
        turn_data = {
            "turn_number": turn_count + 1,
            "player": player_name,
            "mainline_state": mainline_before,
            "hand": hand_before,
            "llm_response": scientist.last_action_response.copy() if scientist.last_action_response else {},
            "action_result": {
                "action": play_result.get("action"),
                "card": play_result.get("card"),
                "accepted": play_result.get("accepted"),
                "success": play_result.get("success"),
            },
            "guess_attempt": None
        }

        # Log result
        logger.info(f"Action: {play_result['action']}")
        if "card" in play_result:
            logger.info(f"Card played: {play_result['card']}")
            logger.info(f"Result: {'ACCEPTED ✓' if play_result.get('accepted') else 'REJECTED ✗'}")

        # Record the play result
        scientist.record_action_result(play_result)

        # Execute the guess if needed
        if will_guess and guess_text:
            logger.info("")
            logger.info(f"{player_name} is guessing the rule...")
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
                logger.info(f"GAME OVER! {player_name} won on turn {turn_count + 1}!")
                logger.info("=" * 80)
                game_over_reason = "correct_guess"
                break

        logger.info("")
        turn_count += 1

        # Pause after turn if configured
        if game_config.get("pause_after_turn", False):
            input(f"[Turn {turn_count} complete] Press Enter to continue...")

    # -------------
    # (5) Scoring
    # -------------

    scores = engine.calculate_scores(max_turns_limit, turn_count)
    success = engine.rule_guessed
    final_score = scores[player_name]

    # Collect LLM usage statistics
    llm_usage = {
        "game_master": game_master_client.get_usage_stats(),
        "player": scientist_client.get_usage_stats(),
    }

    return {
        'round_number': round_number,
        'turn_count': turn_count,
        'rule_description': rule.description(),
        'rule_code': rule.get_code(),
        'success': success,
        'score': final_score,
        'failed_guesses': engine.failed_guess_count,
        'game_over_reason': game_over_reason,
        'llm_usage': llm_usage,
        'turns': turn_data_list,
    }
