"""Turn execution and result assembly for one Eleusis benchmark round."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from eleusis.evaluation_results import TurnRecord
from eleusis.game import GameEngine, GameState, GuessRuleAction, Rule
from eleusis.game.engine import Action, PlayCardAction
from eleusis.game.rule_library import RuleMetadata
from eleusis.llm.base import BaseLLMClient
from eleusis.normalization import (
    NormalizedActionResponse,
    compute_schema_compliance_rate,
    normalize_action_response,
)
from eleusis.player import LLMScientist

if TYPE_CHECKING:
    from eleusis.game.validator import RuleComparisonMetadata
    from eleusis.runner import RoundResult

logger = logging.getLogger(__name__)

ActionErrorHandler = Callable[
    [Exception, LLMScientist, GameState],
    tuple[PlayCardAction, dict[str, str | None]],
]


@dataclass
class RoundRuntime:
    """Initialized collaborators and settings for one benchmark round."""

    round_number: int
    start_time: float
    rule: Rule
    rule_metadata: RuleMetadata | None
    engine: GameEngine
    game_state: GameState
    scientist: LLMScientist
    scientist_client: BaseLLMClient
    rule_compiler_client: BaseLLMClient
    player_name: str
    max_turns: int
    shadow_mode: str
    pause_after_turn: bool
    results_folder: str | None
    handle_action_error: ActionErrorHandler
    completed_turn_committer: (
        Callable[[RoundRuntime, list[TurnRecord]], None] | None
    ) = None


@dataclass(frozen=True)
class TurnRecordInput:
    """Captured state and outcomes needed to persist one turn."""

    turn_index: int
    mainline_before: str
    hand_before: list[str]
    normalized: NormalizedActionResponse
    play_result: dict[str, object]
    tokens: dict[str, int]
    error_info: dict[str, str | None] | None


def _request_turn_action(
    runtime: RoundRuntime,
) -> tuple[Action, dict[str, str | None] | None, int]:
    """Request one action while allowing unexpected player failures to propagate."""
    metrics_before = len(runtime.scientist_client.generate_metrics)
    action = runtime.scientist.get_action(runtime.game_state)
    return action, None, metrics_before


def _turn_token_metrics(client: BaseLLMClient, metrics_before: int) -> dict[str, int]:
    """Aggregate generation metrics emitted during one player action."""
    tokens = {"output_tokens": 0, "reasoning_tokens": 0, "answer_tokens": 0}
    for metric in client.generate_metrics[metrics_before:]:
        tokens["output_tokens"] += metric.total_output_tokens
        tokens["reasoning_tokens"] += metric.total_reasoning_tokens
        tokens["answer_tokens"] += metric.total_answer_tokens
    return tokens


def _log_action_response(scientist: LLMScientist) -> None:
    """Log normalized user-facing fields from the latest structured response."""
    response = scientist.last_action_response
    if not response:
        return
    logger.info(f"Reasoning: {response.get('reasoning_summary', '')}")
    logger.info(f"Tentative rule: {response.get('tentative_rule', '')}")
    logger.info(f"Confidence level: {response.get('confidence_level', '')}")
    logger.info(f"Will guess: {response.get('guess_rule', False)}")
    logger.info("")


def _build_turn_record(
    runtime: RoundRuntime,
    turn_input: TurnRecordInput,
) -> TurnRecord:
    """Build the persisted record for one card play."""
    scientist = runtime.scientist
    return {
        "turn_number": turn_input.turn_index + 1,
        "player": runtime.player_name,
        "mainline_state": turn_input.mainline_before,
        "hand": turn_input.hand_before,
        "llm_response": (
            scientist.last_action_response.copy()
            if scientist.last_action_response
            else {}
        ),
        "model_attempts": list(scientist.last_model_attempts),
        "confidence_level_raw": turn_input.normalized["confidence_level_raw"],
        "confidence_level": turn_input.normalized["confidence_level"],
        "schema_errors": turn_input.normalized["schema_errors"],
        "action_result": {
            "action": turn_input.play_result.get("action"),
            "card": turn_input.play_result.get("card"),
            "accepted": turn_input.play_result.get("accepted"),
            "success": turn_input.play_result.get("success"),
        },
        "guess_attempt": None,
        "tokens": turn_input.tokens,
        "retry_count": scientist.last_retry_count,
        "retry_causes": [dict(retry) for retry in scientist.last_retry_causes],
        "error": turn_input.error_info,
    }


def _apply_formal_guess(
    runtime: RoundRuntime,
    turn_record: TurnRecord,
    guess_text: str,
) -> dict[str, object]:
    """Evaluate and record one formal rule guess."""
    logger.info("")
    logger.info(f"{runtime.player_name} is guessing the rule...")
    result = runtime.engine.play_turn(GuessRuleAction(guess_text))
    guess_value = result.get("guess")
    if not isinstance(guess_value, str):
        return result
    reasoning = result.get("reasoning", "")
    metadata = cast("RuleComparisonMetadata", result)
    complexity = metadata["complexity_metrics"] or {}
    turn_record["guess_attempt"] = {
        "version": 1,
        "kind": "formal",
        "guess": guess_value,
        "correct": result.get("correct") is True,
        "reasoning": reasoning if isinstance(reasoning, str) else "",
        "guessed_code": metadata["guessed_code"],
        "node_count": complexity.get("node_count"),
        "cyclomatic_complexity": complexity.get("cyclomatic"),
        "compilation": {
            "status": metadata["compilation_status"],
            "attempt_count": metadata["compilation_attempts"],
            "cache_hit": metadata["compilation_cache_hit"],
            "artifact_provider": metadata["compilation_provider"],
            "rule_compilation_attempts": None,
        },
        "equivalence": {
            "num_simulations": runtime.engine.num_simulations,
            "turns_per_simulation": runtime.engine.turns_per_simulation,
            "simulation_seed": runtime.engine.simulation_seed,
            "cache_hit": metadata["equivalence_cache_hit"],
            "comparisons": metadata["simulation_comparisons"],
            "mismatches": metadata["simulation_mismatches"],
            "duration_seconds": metadata["simulation_duration_seconds"],
        },
    }
    return result


def _apply_shadow_guess(
    runtime: RoundRuntime,
    turn_record: TurnRecord,
    guess_text: str,
    confidence: int | None,
) -> None:
    """Record or evaluate one qualifying tentative rule guess."""
    if confidence is None or confidence < 5 or not guess_text:
        return
    if runtime.shadow_mode == "offline":
        turn_record["guess_attempt"] = {
            "guess": guess_text,
            "shadow": True,
            "evaluated": False,
        }
        return
    if runtime.shadow_mode != "online":
        return
    logger.info("")
    logger.info(f"Shadow evaluation for tentative rule (confidence={confidence})...")
    is_correct, reasoning, metadata = runtime.engine.evaluate_rule(guess_text)
    logger.info(
        f"Shadow evaluation result: {'CORRECT ✅' if is_correct else 'INCORRECT ❌'}"
    )
    complexity = metadata["complexity_metrics"]
    turn_record["guess_attempt"] = {
        "guess": guess_text,
        "correct": is_correct,
        "reasoning": reasoning,
        "guessed_code": metadata["guessed_code"],
        "node_count": complexity["node_count"] if complexity else None,
        "cyclomatic_complexity": complexity["cyclomatic"] if complexity else None,
        "shadow": True,
    }


def _execute_turn(
    runtime: RoundRuntime,
    turn_index: int,
) -> tuple[TurnRecord, dict[str, object]]:
    """Execute one card play and any associated formal or shadow guess."""
    state = runtime.game_state
    player = state.player
    state.turn_number = turn_index + 1
    mainline_before = state.to_compact_string()
    hand_before = [str(card) for card in player.hand.get_all_cards()]
    logger.info("=" * 80)
    logger.info(
        f"[Round {runtime.round_number}] TURN {turn_index + 1}: {runtime.player_name}"
    )
    logger.info("=" * 80)
    logger.info(f"Board: {mainline_before}")
    logger.info(f"Deck remaining: {state.deck.remaining_count()} cards")
    logger.info(f"Hand ({len(hand_before)} cards): {', '.join(hand_before)}\n")

    action, error_info, metrics_before = _request_turn_action(runtime)
    if runtime.results_folder and runtime.scientist.last_prompt:
        prompt_file = Path(runtime.results_folder) / "last_prompt.md"
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(runtime.scientist.last_prompt)
    _log_action_response(runtime.scientist)
    normalized = normalize_action_response(runtime.scientist.last_action_response)
    response = runtime.scientist.last_action_response or {}
    guess_value = response.get("tentative_rule", "")
    guess_text = guess_value if isinstance(guess_value, str) else ""

    play_result = runtime.engine.play_turn(action)
    turn_record = _build_turn_record(
        runtime,
        TurnRecordInput(
            turn_index=turn_index,
            mainline_before=mainline_before,
            hand_before=hand_before,
            normalized=normalized,
            play_result=play_result,
            tokens=_turn_token_metrics(runtime.scientist_client, metrics_before),
            error_info=error_info,
        ),
    )
    logger.info(f"Action: {play_result['action']}")
    if "card" in play_result:
        logger.info(f"Card played: {play_result['card']}")
        verdict = "ACCEPTED ✓" if play_result.get("accepted") else "REJECTED ✗"
        logger.info(f"Result: {verdict}")
    runtime.scientist.record_action_result(play_result)

    if response.get("guess_rule") is True and guess_text:
        result = _apply_formal_guess(runtime, turn_record, guess_text)
    else:
        result = play_result
        _apply_shadow_guess(
            runtime,
            turn_record,
            guess_text,
            normalized["confidence_level"],
        )
    return turn_record, result


def execute_round_turn(
    runtime: RoundRuntime,
    turn_index: int,
) -> tuple[TurnRecord, dict[str, object]]:
    """Execute one complete Turn through the Round orchestration path."""
    return _execute_turn(runtime, turn_index)


def execute_round_turns(
    runtime: RoundRuntime,
    *,
    completed_turns: list[TurnRecord] | None = None,
    next_turn_index: int = 0,
) -> tuple[int, str, list[TurnRecord]]:
    """Run from the next uncommitted Turn until a terminal Round outcome."""
    turns = list(completed_turns or [])
    if next_turn_index != len(turns):
        raise ValueError(
            "Round execution resume invalid: next Turn index does not match "
            "completed Turn records"
        )
    resumed_correct_guess = runtime.engine.rule_guessed
    turn_count = next_turn_index - 1 if resumed_correct_guess else next_turn_index
    game_over_reason = "correct_guess" if resumed_correct_guess else "max_turns"
    while turn_count < runtime.max_turns and not runtime.engine.is_game_over():
        turn_record, result = execute_round_turn(runtime, turn_count)
        turns.append(turn_record)
        if runtime.completed_turn_committer is not None:
            runtime.completed_turn_committer(runtime, turns)
        if result.get("correct") and "guess" in result:
            logger.info("\nRULE GUESS!")
            logger.info(f"Guess: {result['guess']}")
            logger.info("Verdict: CORRECT ✅✅✅")
            logger.info("=" * 80)
            logger.info(f"{runtime.player_name} FOUND RULE ON TURN {turn_count + 1}!")
            logger.info("=" * 80)
            game_over_reason = "correct_guess"
            break
        if "guess" in result:
            logger.info("\nRULE GUESS!")
            logger.info(f"Guess: {result['guess']}")
            logger.info("Verdict: INCORRECT ❌❌❌")
        logger.info("")
        turn_count += 1
        if runtime.pause_after_turn:
            input(f"[Turn {turn_count} complete] Press Enter to continue...")
    return turn_count, game_over_reason, turns


def _first_correct_turns(
    turns: list[TurnRecord],
) -> tuple[int | None, int | None, int | None]:
    """Find first formal, shadow, and either-kind correct turn numbers."""
    formal = None
    shadow = None
    for turn in turns:
        guess = turn["guess_attempt"]
        if not guess or not guess.get("correct"):
            continue
        if guess.get("shadow", False) and shadow is None:
            shadow = turn["turn_number"]
        elif not guess.get("shadow", False) and formal is None:
            formal = turn["turn_number"]
    values = [turn for turn in (formal, shadow) if turn is not None]
    return formal, shadow, min(values) if values else None


def build_round_result(
    runtime: RoundRuntime,
    turn_count: int,
    game_over_reason: str,
    turns: list[TurnRecord],
) -> RoundResult:
    """Build the serializable result after round execution finishes."""
    formal_turn, shadow_turn, first_turn = _first_correct_turns(turns)
    score = runtime.engine.calculate_score(runtime.max_turns, turn_count)
    no_stakes_score = runtime.max_turns - first_turn + 1 if first_turn else 0
    return {
        "round_number": runtime.round_number,
        "turn_count": turn_count + 1,
        "rule_description": runtime.rule.description(),
        "rule_code": runtime.rule.get_code(),
        "rule_metadata": runtime.rule_metadata,
        "success": runtime.engine.rule_guessed,
        "score": score,
        "floored_score": max(0, score),
        "no_stakes_score": no_stakes_score,
        "first_correct_turn": first_turn,
        "first_formal_correct_turn": formal_turn,
        "first_shadow_correct_turn": shadow_turn,
        "failed_guesses": runtime.engine.failed_guess_count,
        "game_over_reason": game_over_reason,
        "schema_compliance_rate": compute_schema_compliance_rate(turns),
        "llm_usage": {
            "rule_compiler": runtime.rule_compiler_client.get_usage_stats(),
            "player": runtime.scientist_client.get_usage_stats(),
        },
        "turns": turns,
        "wall_clock_seconds": round(time.time() - runtime.start_time, 2),
    }
