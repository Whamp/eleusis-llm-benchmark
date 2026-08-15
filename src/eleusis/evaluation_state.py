"""Fresh and resumed mutable state for a single-model evaluation."""

import logging
from dataclasses import dataclass

from eleusis.evaluation_results import EvaluationResults
from eleusis.evaluation_startup import EvaluationStartup
from eleusis.evaluation_support import (
    load_rules_from_library,
    restore_rule_from_checkpoint,
)
from eleusis.game import Rule
from eleusis.game.rule_library import RuleLibraryEntry

logger = logging.getLogger(__name__)


@dataclass
class EvaluationState:
    """Mutable cursor, current rule, and persisted results for an evaluation."""

    startup: EvaluationStartup
    results: EvaluationResults
    folder_name: str
    start_round: int
    current_rule: Rule | None
    current_rule_name: str | None
    rule_factory_index: int
    checkpoint_rules_library: list[RuleLibraryEntry] | None
    all_rules_library: list[RuleLibraryEntry]
    rule_name_to_index: dict[str, int]


def _ensure_resume_statistics(results: EvaluationResults) -> None:
    """Add aggregate fields absent from historical checkpoints."""
    statistics = results["statistics"]
    statistics.setdefault("total_output_tokens", 0)
    statistics.setdefault("total_reasoning_tokens", 0)
    statistics.setdefault("total_answer_tokens", 0)
    statistics.setdefault("total_wall_clock_seconds", 0.0)
    statistics.setdefault("total_retries", 0)
    statistics.setdefault("retry_by_cause", {})


def _validate_resume_library(startup: EvaluationStartup) -> bool:
    """Validate consumed-rule and library counts against the checkpoint cursor."""
    checkpoint = startup.checkpoint
    if checkpoint is None:
        return False
    checkpoint_data = checkpoint["checkpoint"]
    consumed = checkpoint_data["rules_consumed"]
    completed_rounds = checkpoint_data["completed_rounds"]
    expected_consumed = (
        completed_rounds + startup.num_rounds_per_rule - 1
    ) // startup.num_rounds_per_rule
    if len(consumed) != expected_consumed:
        logger.error(
            "Mismatch: %s rules consumed but expected %s for %s rounds",
            len(consumed),
            expected_consumed,
            completed_rounds,
        )
        return False
    expected_rules = (
        checkpoint_data["total_rounds"] + startup.num_rounds_per_rule - 1
    ) // startup.num_rounds_per_rule
    if len(checkpoint_data["rules_library"]) < expected_rules:
        logger.error(
            "Not enough rules: %s in library but need %s",
            len(checkpoint_data["rules_library"]),
            expected_rules,
        )
        return False
    return True


def _initialize_resume_state(startup: EvaluationStartup) -> EvaluationState | None:
    """Restore mutable evaluation state from a validated checkpoint."""
    checkpoint = startup.checkpoint
    if checkpoint is None or not _validate_resume_library(startup):
        return None
    _ensure_resume_statistics(checkpoint)
    checkpoint_data = checkpoint["checkpoint"]
    consumed = checkpoint_data["rules_consumed"]
    current_rule = restore_rule_from_checkpoint(checkpoint_data["current_rule"])
    state = EvaluationState(
        startup=startup,
        results=checkpoint,
        folder_name=checkpoint.get(
            "folder_name", f"solo_evaluation_{checkpoint['timestamp']}"
        ),
        start_round=checkpoint_data["completed_rounds"] + 1,
        current_rule=current_rule,
        current_rule_name=consumed[-1]["name"] if consumed else None,
        rule_factory_index=checkpoint_data["rule_factory_state"]["current_index"],
        checkpoint_rules_library=checkpoint_data["rules_library"],
        all_rules_library=[],
        rule_name_to_index={},
    )
    logger.info("=" * 80)
    logger.info("RESUMING SOLO MODE EVALUATION")
    logger.info("=" * 80)
    logger.info("Log file: %s", startup.log_file)
    logger.info("Resuming from round %s / %s", state.start_round, startup.num_rounds)
    logger.info(
        "Rules consumed: %s, rule_factory_index: %s",
        len(consumed),
        state.rule_factory_index,
    )
    if current_rule:
        logger.info("Reusing rule: %s...", current_rule.description()[:80])
    return state


def _new_evaluation_results(
    startup: EvaluationStartup,
    folder_name: str,
    rules_library: list[RuleLibraryEntry],
    rule_factory_index: int,
) -> EvaluationResults:
    """Build the initial self-contained evaluation result/checkpoint document."""
    config = startup.config
    game = startup.game_config
    compiler = config["rule_compiler"]
    llm = config["llm"]
    return {
        "timestamp": startup.timestamp,
        "folder_name": folder_name,
        "config": {
            "num_rules": startup.num_rules,
            "num_rounds_per_rule": startup.num_rounds_per_rule,
            "rule_compiler": startup.rule_compiler_display_name,
            "rule_compiler_provider": compiler["provider"],
            "rule_compiler_model_id": compiler["model_id"],
            "rule_compiler_reasoning_format": compiler.get(
                "reasoning_format", "separate_field"
            ),
            "rule_compiler_temperature": compiler.get("temperature", 0.7),
            "rule_compiler_max_retries": compiler.get("max_retries", 10),
            "rule_compiler_num_simulations": compiler.get("num_simulations", 100),
            "rule_compiler_turns_per_simulation": compiler.get(
                "turns_per_simulation", 40
            ),
            "rule_compiler_simulation_seed": compiler.get("simulation_seed"),
            "player": startup.player_display_name,
            "player_model": startup.player_model,
            "hand_size": game["hand_size"],
            "max_turns": game["max_turns"],
            "wrong_guess_penalty": game["wrong_guess_penalty"],
            "seed": game["seed"],
            "llm_max_tokens": llm["max_tokens"],
            "llm_temperature": llm["temperature"],
            "llm_seed": llm["seed"],
            "llm_max_retries": llm["max_llm_retries"],
            "batch_round_offset": game.get("batch_round_offset"),
            "suite": startup.suite_name,
        },
        "rounds": [],
        "statistics": {
            "total_score": 0,
            "successful_rounds": 0,
            "failed_rounds": 0,
            "total_turns": 0,
            "total_failed_guesses": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "total_answer_tokens": 0,
            "total_wall_clock_seconds": 0.0,
            "total_retries": 0,
            "retry_by_cause": {},
        },
        "checkpoint": {
            "completed_rounds": 0,
            "total_rounds": startup.num_rounds,
            "rule_factory_state": {
                "selection": startup.rules_config["selection"],
                "current_index": rule_factory_index,
            },
            "current_rule": None,
            "rules_consumed": [],
            "rules_library": rules_library,
        },
    }


def _initialize_fresh_state(startup: EvaluationStartup) -> EvaluationState:
    """Load the library and initialize a fresh evaluation checkpoint."""
    folder_name = f"solo_evaluation_{startup.timestamp}_{startup.output_tag}"
    all_rules = load_rules_from_library(startup.config)
    name_to_index = {
        name: index
        for index, rule in enumerate(all_rules)
        if isinstance((name := rule.get("name")), str)
    }
    rule_factory_index = startup.rules_config["index"]
    logger.info("=" * 80)
    title = f"SOLO MODE EVALUATION - {startup.num_rounds} ROUNDS"
    if startup.suite_name:
        title += f" (suite: {startup.suite_name})"
    logger.info(title)
    logger.info("=" * 80)
    logger.info("Log file: %s", startup.log_file)
    logger.info("Output folder: results/%s", folder_name)
    logger.info("Stored %s rules for resume support\n", len(all_rules))
    return EvaluationState(
        startup=startup,
        results=_new_evaluation_results(
            startup,
            folder_name,
            all_rules,
            rule_factory_index,
        ),
        folder_name=folder_name,
        start_round=1,
        current_rule=None,
        current_rule_name=None,
        rule_factory_index=rule_factory_index,
        checkpoint_rules_library=None,
        all_rules_library=all_rules,
        rule_name_to_index=name_to_index,
    )


def initialize_evaluation_state(
    startup: EvaluationStartup,
) -> EvaluationState | None:
    """Create fresh state or restore resume state for the evaluation loop."""
    return (
        _initialize_resume_state(startup)
        if startup.checkpoint
        else _initialize_fresh_state(startup)
    )
