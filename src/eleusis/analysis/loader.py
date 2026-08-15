"""Data loading utilities for analysis module."""

import logging
from pathlib import Path

import pandas as pd

from .benchmark_run_artifact import (
    HistoricalRunCompatibilityError,
    read_analysis_run_artifact,
)
from .legacy_records import LegacyResults, RuleLookup
from .utils import compute_counting_cutoff

logger = logging.getLogger(__name__)


def load_results(folder: Path) -> tuple[LegacyResults, list[str]]:
    """Load all results.json files from a results folder.

    Returns (results, folder_names).
    """
    results = []
    folder_names = []
    for subfolder in sorted(folder.iterdir()):
        if subfolder.is_dir() and subfolder.name.startswith("solo_evaluation_"):
            try:
                artifact = read_analysis_run_artifact(subfolder)
            except HistoricalRunCompatibilityError as error:
                logger.warning("Skipped %s: %s", subfolder.name, error)
                continue
            data = artifact.analysis_document
            if data is None:
                logger.warning(
                    "Skipped %s: compatibility policy produced no analysis view",
                    subfolder.name,
                )
                continue
            data["_folder"] = subfolder.name
            results.append(data)
            folder_names.append(subfolder.name)
            logger.info(
                "Loaded %s (%s%s)",
                subfolder.name,
                artifact.source_format,
                ", partial" if artifact.is_partial else "",
            )
    return results, folder_names


def build_rules_lookup(results: LegacyResults) -> RuleLookup:
    """Build rules lookup from embedded rules_library in results files."""
    rules_lookup = {}
    for result in results:
        checkpoint = result.get("checkpoint", {})
        rules_lib = checkpoint.get("rules_library", [])
        for rule in rules_lib:
            desc = rule.get("description")
            if desc and desc not in rules_lookup:
                rules_lookup[desc] = rule
    return rules_lookup


def _compute_counting_round_metrics(
    turns: list[dict[str, object]],
    cutoff_turn: int | None,
    round_success: bool,
    actual_turn_count: int,
) -> tuple[int, int, bool]:
    """Compute failed guesses, effective turns, and success before the cutoff."""
    failed_guesses = 0
    success_turn = None
    for turn in turns:
        guess_attempt = turn.get("guess_attempt")
        if not isinstance(guess_attempt, dict) or guess_attempt.get("shadow", False):
            continue
        if guess_attempt.get("correct") is True:
            turn_number = turn.get("turn_number")
            success_turn = turn_number if isinstance(turn_number, int) else None
            break
        if guess_attempt.get("correct") is False:
            turn_number = turn.get("turn_number")
            if isinstance(turn_number, int) and (
                cutoff_turn is None or turn_number < cutoff_turn
            ):
                failed_guesses += 1
    counting_success = round_success and (
        cutoff_turn is None or (success_turn is not None and success_turn < cutoff_turn)
    )
    counting_turn_count = (
        min(actual_turn_count, cutoff_turn) if cutoff_turn else actual_turn_count
    )
    return failed_guesses, counting_turn_count, counting_success


def build_rounds_dataframe(
    results: LegacyResults, rules_lib: RuleLookup
) -> pd.DataFrame:
    """Build DataFrame with one row per round."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        model_spec = result["config"]["player_model"]
        for round_data in result["rounds"]:
            rule_desc = round_data["rule_description"]
            rule_info = rules_lib.get(rule_desc, {})
            player_usage = round_data["llm_usage"]["player"]

            # Handle both old (total_tokens) and new (output_tokens) formats
            output_tokens = player_usage.get(
                "output_tokens", player_usage.get("total_tokens", 0)
            )
            reasoning_tokens = player_usage.get("reasoning_tokens", 0) or 0
            answer_tokens = player_usage.get(
                "answer_tokens", output_tokens - reasoning_tokens
            )

            max_turns = (
                result["config"]["game"]["max_turns"]
                if "game" in result["config"]
                else result["config"].get("max_turns", 30)
            )

            # Compute counting cutoff (turn where score becomes guaranteed <= 0)
            turns = round_data.get("turns", [])
            penalty = (
                result["config"]["game"]["wrong_guess_penalty"]
                if "game" in result["config"]
                else result["config"].get("wrong_guess_penalty", 2)
            )
            cutoff_turn = compute_counting_cutoff(turns, max_turns, penalty=penalty)

            (
                counting_failed_guesses,
                counting_turn_count,
                counting_success,
            ) = _compute_counting_round_metrics(
                turns,
                cutoff_turn,
                round_data["success"],
                round_data["turn_count"],
            )

            rows.append(
                {
                    "model": model,
                    "model_spec": model_spec,
                    "round_number": round_data["round_number"],
                    "success": round_data["success"],
                    "score": round_data["score"],
                    "turn_count": round_data["turn_count"],
                    "failed_guesses": round_data["failed_guesses"],
                    "game_over_reason": round_data["game_over_reason"],
                    "rule_description": rule_desc,
                    "max_turns": max_turns,
                    # Token metrics
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "answer_tokens": answer_tokens,
                    "wall_clock_seconds": round_data.get("wall_clock_seconds", 0),
                    # Rule complexity metrics
                    "cyclomatic_complexity": rule_info.get("cyclomatic_complexity"),
                    "node_count": rule_info.get("node_count"),
                    "avg_acceptance_rate": rule_info.get("avg_acceptance_rate"),
                    # Counting cutoff metrics
                    "counting_cutoff_turn": cutoff_turn,
                    "counting_failed_guesses": counting_failed_guesses,
                    "counting_turn_count": counting_turn_count,
                    "counting_success": counting_success,
                }
            )
    return pd.DataFrame(rows)


def build_turns_dataframe(results: LegacyResults) -> pd.DataFrame:
    """Build a turn-level DataFrame for calibration and model analysis."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        for round_data in result["rounds"]:
            rule_desc = round_data["rule_description"]
            max_turns = (
                result["config"]["game"]["max_turns"]
                if "game" in result["config"]
                else result["config"].get("max_turns", 30)
            )
            turns = round_data.get("turns", [])
            penalty = (
                result["config"]["game"]["wrong_guess_penalty"]
                if "game" in result["config"]
                else result["config"].get("wrong_guess_penalty", 2)
            )
            cutoff_turn = compute_counting_cutoff(turns, max_turns, penalty=penalty)

            for turn in turns:
                llm_resp = turn.get("llm_response", {})
                guess_attempt = turn.get("guess_attempt")

                # Determine if this is a shadow guess
                is_shadow = False
                guess_correct = None
                tentative_node_count = None
                tentative_cyclomatic = None

                if guess_attempt:
                    is_shadow = guess_attempt.get("shadow", False)
                    guess_correct = guess_attempt.get("correct")
                    tentative_node_count = guess_attempt.get("node_count")
                    tentative_cyclomatic = guess_attempt.get("cyclomatic_complexity")

                # Determine if this turn counts for analysis (before cutoff)
                turn_num = turn["turn_number"]
                counts_for_analysis = cutoff_turn is None or turn_num < cutoff_turn

                # Per-turn token usage
                turn_tokens = turn.get("tokens", {})
                output_tokens = turn_tokens.get("output_tokens", 0)

                action_result = turn.get("action_result")
                card_accepted = (
                    action_result.get("accepted")
                    if isinstance(action_result, dict)
                    else None
                )
                rows.append(
                    {
                        "model": model,
                        "round_number": round_data["round_number"],
                        "turn_number": turn_num,
                        "confidence_level": llm_resp.get("confidence_level"),
                        "guess_rule": llm_resp.get("guess_rule", False),
                        "guess_correct": guess_correct,
                        "is_shadow": is_shadow,
                        "card_accepted": card_accepted,
                        "tentative_rule": llm_resp.get("tentative_rule"),
                        "actual_rule": rule_desc,
                        "round_success": round_data["success"],
                        # Tentative rule complexity (from guess_attempt)
                        "tentative_node_count": tentative_node_count,
                        "tentative_cyclomatic": tentative_cyclomatic,
                        # Counting cutoff
                        "counts_for_analysis": counts_for_analysis,
                        # Token usage
                        "output_tokens": output_tokens,
                    }
                )
    return pd.DataFrame(rows)
