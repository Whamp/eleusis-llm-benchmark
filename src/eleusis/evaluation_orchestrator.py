"""Round-loop orchestration and finalization for a single-model evaluation."""

import argparse
import logging
from dataclasses import dataclass

from eleusis.evaluation_startup import resolve_evaluation_startup
from eleusis.evaluation_state import EvaluationState, initialize_evaluation_state
from eleusis.evaluation_support import (
    get_integer_metric,
    save_evaluation_results,
)
from eleusis.game import Rule
from eleusis.runner import RoundResult, play_round

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoundSelection:
    """Rule selection decision and seed batch index for one round."""

    batch_round_index: int
    need_new_rule: bool
    suite_rule_name: str | None


def _select_round_rule(state: EvaluationState, round_number: int) -> RoundSelection:
    """Choose rule reuse/loading and batch index for the next round."""
    startup = state.startup
    if startup.suite_cases:
        suite_rule_name, batch_index = startup.suite_cases[round_number - 1]
        need_new_rule = state.current_rule_name != suite_rule_name
        logger.info(
            "%s rule: %s (batch index %s)",
            "Suite" if need_new_rule else "Reusing",
            suite_rule_name,
            batch_index,
        )
        if need_new_rule:
            state.current_rule = None
        return RoundSelection(batch_index, need_new_rule, suite_rule_name)
    offset = startup.game_config.get("batch_round_offset")
    if offset is not None:
        return RoundSelection(offset, True, None)
    batch_index = (round_number - 1) % startup.num_rounds_per_rule
    return RoundSelection(batch_index, batch_index == 0, None)


def _execute_round(
    state: EvaluationState,
    round_number: int,
    selection: RoundSelection,
) -> RoundResult:
    """Invoke the runner with the cursor appropriate to the selected mode."""
    if selection.need_new_rule:
        state.current_rule = None
    generated_new_rule = state.current_rule is None
    if state.startup.suite_cases and selection.need_new_rule:
        if selection.suite_rule_name is None:
            raise RuntimeError("Suite round selection requires a rule name")
        start_index = state.rule_name_to_index[selection.suite_rule_name]
        rules_list = state.checkpoint_rules_library or state.all_rules_library
    else:
        start_index = state.rule_factory_index if selection.need_new_rule else None
        rules_list = state.checkpoint_rules_library
    result = play_round(
        config=state.startup.config,
        round_number=round_number,
        rule=state.current_rule,
        start_rule_index=start_index,
        rules_list=rules_list,
        batch_round_index=selection.batch_round_index,
        results_folder=(
            str(state.run_store.run_folder)
            if state.run_store is not None
            else f"results/{state.folder_name}"
        ),
        run_store=state.run_store,
    )
    _update_current_rule(state, result, generated_new_rule)
    return result


def _update_current_rule(
    state: EvaluationState,
    result: RoundResult,
    generated_new_rule: bool,
) -> None:
    """Advance or reuse the current rule and consumed-rule checkpoint entry."""
    if not generated_new_rule:
        consumed = state.results["checkpoint"]["rules_consumed"]
        if consumed:
            consumed[-1]["rounds_completed"] += 1
        return
    state.current_rule = Rule(result["rule_description"], result["rule_code"])
    metadata = result.get("rule_metadata") or {}
    name = metadata.get("name")
    state.current_rule_name = name if isinstance(name, str) else None
    state.rule_factory_index += 1
    state.results["checkpoint"]["rules_consumed"].append(
        {
            "name": state.current_rule_name,
            "description": state.current_rule.description(),
            "code": state.current_rule.get_code(),
            "rounds_completed": 1,
        }
    )


def _append_round_result(
    state: EvaluationState,
    round_number: int,
    selection: RoundSelection,
    result: RoundResult,
) -> None:
    """Append the serializable subset of one completed runner result."""
    state.results["rounds"].append(
        {
            "round_number": round_number,
            "rule_name": state.current_rule_name,
            "batch_round_index": selection.batch_round_index,
            "turn_count": result["turn_count"],
            "rule_description": result["rule_description"],
            "rule_code": result["rule_code"],
            "success": result["success"],
            "score": result["score"],
            "floored_score": result["floored_score"],
            "no_stakes_score": result["no_stakes_score"],
            "first_correct_turn": result["first_correct_turn"],
            "failed_guesses": result["failed_guesses"],
            "game_over_reason": result["game_over_reason"],
            "llm_usage": result["llm_usage"],
            "turns": result["turns"],
            "wall_clock_seconds": result["wall_clock_seconds"],
        }
    )


def _update_evaluation_statistics(
    state: EvaluationState,
    result: RoundResult,
) -> None:
    """Accumulate score, token, timing, and retry statistics."""
    statistics = state.results["statistics"]
    statistics["total_score"] += result["score"]
    success_key = "successful_rounds" if result["success"] else "failed_rounds"
    statistics[success_key] += 1
    statistics["total_turns"] += result["turn_count"]
    statistics["total_failed_guesses"] += result["failed_guesses"]
    player_usage = result["llm_usage"].get("player", {})
    statistics["total_output_tokens"] += get_integer_metric(
        player_usage, "output_tokens"
    )
    statistics["total_reasoning_tokens"] += get_integer_metric(
        player_usage, "reasoning_tokens"
    )
    statistics["total_answer_tokens"] += get_integer_metric(
        player_usage, "answer_tokens"
    )
    statistics["total_wall_clock_seconds"] += result["wall_clock_seconds"]
    for turn in result["turns"]:
        statistics["total_retries"] += turn["retry_count"]
        for retry in turn["retry_causes"]:
            cause_value = retry.get("cause", "unknown")
            cause = cause_value if isinstance(cause_value, str) else "unknown"
            retry_counts = statistics["retry_by_cause"]
            retry_counts[cause] = retry_counts.get(cause, 0) + 1


def _update_evaluation_checkpoint(
    state: EvaluationState,
    round_number: int,
) -> None:
    """Advance the resume cursor while preserving consumed and library records."""
    previous = state.results["checkpoint"]
    current_rule = state.current_rule
    rounds_per_rule = state.startup.num_rounds_per_rule
    state.results["checkpoint"] = {
        "completed_rounds": round_number,
        "total_rounds": state.startup.num_rounds,
        "rule_factory_state": {
            "selection": state.startup.rules_config["selection"],
            "current_index": state.rule_factory_index,
        },
        "current_rule": {
            "description": current_rule.description() if current_rule else None,
            "code": current_rule.get_code() if current_rule else None,
            "rounds_used_in_batch": (round_number - 1) % rounds_per_rule + 1,
            "num_rounds_per_rule": rounds_per_rule,
        },
        "rules_consumed": previous["rules_consumed"],
        "rules_library": previous["rules_library"],
    }


def _run_evaluation_round(state: EvaluationState, round_number: int) -> None:
    """Select, execute, aggregate, checkpoint, and save one round."""
    logger.info("=" * 80)
    logger.info("ROUND %s / %s", round_number, state.startup.num_rounds)
    logger.info("=" * 80 + "\n")
    selection = _select_round_rule(state, round_number)
    result = _execute_round(state, round_number, selection)
    _append_round_result(state, round_number, selection, result)
    _update_evaluation_statistics(state, result)
    _update_evaluation_checkpoint(state, round_number)
    output_file = (
        state.run_store.ensure_current_export()
        if state.run_store is not None
        else save_evaluation_results(state.results, state.folder_name)
    )
    logger.info("Progress saved to: %s", output_file)
    logger.info(
        "Round %s complete: turns=%s success=%s score=%s "
        "failed_guesses=%s rule=%s...\n",
        round_number,
        result["turn_count"],
        result["success"],
        result["score"],
        result["failed_guesses"],
        result["rule_description"][:80],
    )


def _finalize_evaluation(state: EvaluationState) -> None:
    """Compute final averages, save results, and log the evaluation summary."""
    statistics = state.results["statistics"]
    rounds = state.startup.num_rounds
    success_rate = statistics["successful_rounds"] / rounds * 100
    average_score = statistics["total_score"] / rounds
    average_turns = statistics["total_turns"] / rounds
    successful_turns = sum(
        result["turn_count"] for result in state.results["rounds"] if result["success"]
    )
    average_success_turns = (
        successful_turns / statistics["successful_rounds"]
        if statistics["successful_rounds"]
        else 0
    )
    average_failed = statistics["total_failed_guesses"] / rounds
    statistics["success_rate"] = success_rate
    statistics["average_score"] = average_score
    statistics["average_turns"] = average_turns
    statistics["average_turns_when_successful"] = average_success_turns
    statistics["average_failed_guesses"] = average_failed
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("Player: %s", state.startup.player_display_name)
    logger.info("Rounds played: %s", rounds)
    logger.info(
        "Success rate: %.1f%% (%s/%s)",
        success_rate,
        statistics["successful_rounds"],
        rounds,
    )
    logger.info("Average score: %.1f", average_score)
    logger.info("Average turns: %.1f", average_turns)
    logger.info("Average turns (successful rounds only): %.1f", average_success_turns)
    logger.info("Average failed guesses per round: %.1f", average_failed)
    logger.info("Total score: %s", statistics["total_score"])
    logger.info("Total output tokens: %s", statistics["total_output_tokens"])
    logger.info("Total reasoning tokens: %s", statistics["total_reasoning_tokens"])
    logger.info("Total answer tokens: %s", statistics["total_answer_tokens"])
    logger.info("Total wall clock time: %.1fs", statistics["total_wall_clock_seconds"])
    if statistics["total_retries"]:
        logger.info("Total LLM retries: %s", statistics["total_retries"])
        for cause, count in statistics["retry_by_cause"].items():
            logger.info("  - %s: %s", cause, count)
    output_file = (
        state.run_store.ensure_current_export()
        if state.run_store is not None
        else save_evaluation_results(state.results, state.folder_name)
    )
    logger.info("Results saved to: %s", output_file)
    logger.info("Log file: %s", state.startup.log_file)


def run_evaluation(args: argparse.Namespace) -> None:
    """Resolve, initialize, execute, and finalize one evaluation session."""
    startup = resolve_evaluation_startup(args)
    if startup is None:
        return
    state = initialize_evaluation_state(startup)
    if state is None:
        return
    for round_number in range(state.start_round, startup.num_rounds + 1):
        _run_evaluation_round(state, round_number)
    _finalize_evaluation(state)
