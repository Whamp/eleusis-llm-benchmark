"""Game runner for pattern discovery mode."""

import hashlib
import logging
import time

from eleusis.game import GameEngine, GameState, GuessRuleAction, Rule, RuleFactory, RuleValidator
from eleusis.llm import LLMScientist, create_client
from eleusis.utils import model_spec_to_display_name

__all__ = ["play_round"]

logger = logging.getLogger(__name__)


def play_round(
    config: dict,
    round_number: int,
    rule: Rule | None = None,
    max_turns: int | None = None,
    start_rule_index: int | None = None,
    rules_list: list[dict] | None = None,
) -> dict:
    """Play a single round of pattern discovery.

    Args:
        config: Full game configuration dict
        round_number: Current round number (for logging)
        rule: Optional rule to reuse (if None, generate/load new rule)
        max_turns: Optional override for max turns
        start_rule_index: Starting index for RuleFactory (for resume support)
        rules_list: Pre-loaded rules list (for resume, takes precedence over library file)

    Returns:
        dict with round_number, turn_count, rule_description, rule_code,
        success, score, game_over_reason, wall_clock_seconds
    """
    round_start_time = time.time()

    # --------------------
    # (1) Client initialization
    # --------------------

    game_config = config["game"]
    llm_config = config["llm"]
    rule_compiler_cfg = config["rule_compiler"]

    player_model = config["model"]
    player_display_name = model_spec_to_display_name(player_model)

    max_tokens = llm_config["max_tokens"]
    llm_seed = llm_config.get("seed")

    rule_compiler_client = create_client(
        rule_compiler_cfg["model"],
        temperature=rule_compiler_cfg["temperature"],
        max_tokens=max_tokens,
        role="rule_compiler",
        seed=llm_seed,
    )

    scientist_client = create_client(
        player_model,
        temperature=llm_config["temperature"],
        max_tokens=max_tokens,
        role=player_display_name,
        seed=llm_seed,
    )

    # --------------------
    # (2) Rule generation
    # --------------------

    rule_metadata = None

    if rule is None:
        logger.info("=" * 80)
        logger.info(f"[Round {round_number}] PHASE 1: RULE LOADING")
        logger.info("=" * 80)
        logger.info("")

        logger.info("✓ Rule compiler client initialized")

        rule_compiler_client.reset_usage_stats()
        scientist_client.reset_usage_stats()

        validator = RuleValidator()

        rules_cfg = config["rules"]

        logger.info("Loading rule from library...")

        start_index = start_rule_index if start_rule_index is not None else rules_cfg.get("index", 0)
        logger.info(f"Rule factory starting at index: {start_index}")

        rule_factory = RuleFactory(
            library_path=rules_cfg["library_path"] if rules_list is None else None,
            selection=rules_cfg["selection"],
            start_index=start_index,
            rules_list=rules_list,
        )

        rule, rule_metadata = rule_factory.create_rule_with_metadata()

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"SECRET RULE: {rule.description()}")
        logger.info("=" * 80)
        logger.info("")
    else:
        logger.info(f"[Round {round_number}] Using provided rule")
        validator = RuleValidator()

    # ---------------
    # (3) Game setup
    # ---------------

    logger.info("=" * 80)
    logger.info(f"[Round {round_number}] PHASE 2: GAME SETUP")
    logger.info("=" * 80)
    logger.info("")

    player_name = player_display_name
    game_state = GameState(player_name)

    hand_size = game_config.get('hand_size', 12)
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)

    engine = GameEngine(
        game_state,
        rule,
        rule_compiler_client=rule_compiler_client,
        rule_validator=validator,
        hand_size=hand_size,
        wrong_guess_penalty=wrong_guess_penalty,
    )

    # Compute round seed from rule code for reproducibility
    # Use hashlib instead of hash() because Python's hash() is randomized per process
    base_seed = config["game"].get("seed")
    if base_seed is not None:
        rule_hash = int(hashlib.md5(rule.get_code().encode()).hexdigest(), 16) & 0xFFFFFFFF
        round_seed = (base_seed + rule_hash) & 0xFFFFFFFF
        logger.info(f"Using round seed: {round_seed} (base_seed={base_seed}, rule_hash={rule_hash})")
    else:
        round_seed = None

    engine.setup_game(round_seed=round_seed)

    logger.info("✓ Game setup complete")
    logger.info(f"✓ Starter card placed: {game_state.mainline.get_last()}")
    logger.info(f"✓ Player has {hand_size} cards (constant hand size)")
    logger.info(f"✓ Deck has {game_state.deck.remaining_count()} cards remaining")
    logger.info("")

    max_turns_limit = max_turns or game_config.get("max_turns", 40)

    max_llm_retries = llm_config["max_llm_retries"]
    scientist = LLMScientist(
        player_name,
        scientist_client,
        max_retries=max_llm_retries,
        engine=engine,
        max_turns=max_turns_limit,
    )
    logger.info(f"✓ Player initialized: {scientist.name}")
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
        player = game_state.player

        mainline_before = game_state.to_compact_string()
        hand_before = [str(c) for c in player.hand.get_all_cards()]

        logger.info("=" * 80)
        logger.info(f"[Round {round_number}] TURN {turn_count + 1}: {player_name}")
        logger.info("=" * 80)
        logger.info(f"Board: {game_state.to_compact_string()}")
        logger.info(f"Deck remaining: {game_state.deck.remaining_count()} cards")
        hand_cards = player.hand.get_all_cards()
        hand_str = ", ".join([str(c) for c in hand_cards])
        logger.info(f"Hand ({len(hand_cards)} cards): {hand_str}")
        logger.info("")

        game_state.turn_number = turn_count + 1

        # Track generate_metrics count before LLM call for per-turn token tracking
        gen_metrics_before = len(scientist_client.generate_metrics)

        try:
            action = scientist.get_action(game_state)
        except Exception as e:
            logger.error(f"Error getting action: {e}", exc_info=True)
            turn_count += 1
            continue

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

        will_guess = (
            scientist.last_action_response
            and scientist.last_action_response.get("guess_rule", False)
        )

        guess_text = (
            scientist.last_action_response.get("tentative_rule", "")
            if scientist.last_action_response
            else ""
        )

        play_result = engine.play_turn(action)

        # Extract per-turn token metrics from any new generate_metrics
        gen_metrics_after = len(scientist_client.generate_metrics)
        turn_tokens = {"output_tokens": 0, "reasoning_tokens": 0, "answer_tokens": 0}
        if gen_metrics_after > gen_metrics_before:
            for gm in scientist_client.generate_metrics[gen_metrics_before:gen_metrics_after]:
                turn_tokens["output_tokens"] += gm.total_output_tokens
                turn_tokens["reasoning_tokens"] += gm.total_reasoning_tokens
                turn_tokens["answer_tokens"] += gm.total_answer_tokens

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
            "guess_attempt": None,
            "tokens": turn_tokens,
            "retry_count": scientist.last_retry_count,
            "retry_causes": scientist.last_retry_causes.copy(),
        }

        logger.info(f"Action: {play_result['action']}")
        if "card" in play_result:
            logger.info(f"Card played: {play_result['card']}")
            logger.info(f"Result: {'ACCEPTED ✓' if play_result.get('accepted') else 'REJECTED ✗'}")

        scientist.record_action_result(play_result)

        if will_guess and guess_text:
            logger.info("")
            logger.info(f"{player_name} is guessing the rule...")
            result = engine.play_turn(GuessRuleAction(guess_text))

            if "guess" in result:
                complexity = result.get("complexity_metrics") or {}
                turn_data["guess_attempt"] = {
                    "guess": result["guess"],
                    "correct": result.get("correct", False),
                    "reasoning": result.get("reasoning", ""),
                    "guessed_code": result.get("guessed_code"),
                    "node_count": complexity.get("node_count"),
                    "cyclomatic_complexity": complexity.get("cyclomatic"),
                }
        else:
            result = play_result

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

        if game_config.get("pause_after_turn", False):
            input(f"[Turn {turn_count} complete] Press Enter to continue...")

    # -------------
    # (5) Scoring
    # -------------

    score = engine.calculate_score(max_turns_limit, turn_count)
    success = engine.rule_guessed

    llm_usage = {
        "rule_compiler": rule_compiler_client.get_usage_stats(),
        "player": scientist_client.get_usage_stats(),
    }

    return {
        'round_number': round_number,
        'turn_count': turn_count,
        'rule_description': rule.description(),
        'rule_code': rule.get_code(),
        'rule_metadata': rule_metadata,
        'success': success,
        'score': score,
        'failed_guesses': engine.failed_guess_count,
        'game_over_reason': game_over_reason,
        'llm_usage': llm_usage,
        'turns': turn_data_list,
        'wall_clock_seconds': round(time.time() - round_start_time, 2),
    }
