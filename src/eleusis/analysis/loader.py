"""Data loading utilities for analysis module."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_results(folder: Path) -> tuple[list[dict], list[str]]:
    """Load all results.json files from a results folder.

    Returns (results, folder_names).
    """
    results = []
    folder_names = []
    for subfolder in sorted(folder.iterdir()):
        if subfolder.is_dir() and subfolder.name.startswith("solo_evaluation_"):
            results_file = subfolder / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["_folder"] = subfolder.name
                    results.append(data)
                    folder_names.append(subfolder.name)
                    logger.info(f"Loaded: {subfolder.name}")
    return results, folder_names


def build_rules_lookup(results: list[dict]) -> dict[str, dict]:
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


def build_rounds_dataframe(results: list[dict], rules_lib: dict) -> pd.DataFrame:
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
            output_tokens = player_usage.get("output_tokens", player_usage.get("total_tokens", 0))
            reasoning_tokens = player_usage.get("reasoning_tokens", 0) or 0
            answer_tokens = player_usage.get("answer_tokens", output_tokens - reasoning_tokens)

            rows.append({
                "model": model,
                "model_spec": model_spec,
                "round_number": round_data["round_number"],
                "success": round_data["success"],
                "score": round_data["score"],
                "turn_count": round_data["turn_count"],
                "failed_guesses": round_data["failed_guesses"],
                "game_over_reason": round_data["game_over_reason"],
                "rule_description": rule_desc,
                # Token metrics
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "answer_tokens": answer_tokens,
                "wall_clock_seconds": round_data.get("wall_clock_seconds", 0),
                # Rule complexity metrics
                "cyclomatic_complexity": rule_info.get("cyclomatic_complexity"),
                "node_count": rule_info.get("node_count"),
                "avg_acceptance_rate": rule_info.get("avg_acceptance_rate"),
            })
    return pd.DataFrame(rows)


def build_turns_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build DataFrame with one row per turn (for calibration and per-model analysis)."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        for round_data in result["rounds"]:
            rule_desc = round_data["rule_description"]
            for turn in round_data["turns"]:
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

                rows.append({
                    "model": model,
                    "round_number": round_data["round_number"],
                    "turn_number": turn["turn_number"],
                    "confidence_level": llm_resp.get("confidence_level"),
                    "guess_rule": llm_resp.get("guess_rule", False),
                    "guess_correct": guess_correct,
                    "is_shadow": is_shadow,
                    "card_accepted": turn.get("action_result", {}).get("accepted"),
                    "tentative_rule": llm_resp.get("tentative_rule"),
                    "actual_rule": rule_desc,
                    "round_success": round_data["success"],
                    # Tentative rule complexity (from guess_attempt)
                    "tentative_node_count": tentative_node_count,
                    "tentative_cyclomatic": tentative_cyclomatic,
                })
    return pd.DataFrame(rows)
