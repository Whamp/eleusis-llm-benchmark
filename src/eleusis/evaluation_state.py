"""Fresh and resumed mutable state for a single-model evaluation."""

import logging
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path

from eleusis.benchmark_run_manifest import create_benchmark_run_manifest
from eleusis.benchmark_run_store import ActiveStoredRound, BenchmarkRunStore
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
    run_store: BenchmarkRunStore | None = None


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


def _rules_from_run_manifest(
    manifest: dict[str, object],
) -> list[RuleLibraryEntry]:
    """Recover the ordered embedded rule library needed by resumed orchestration."""
    schedule = manifest["schedule"]
    if not isinstance(schedule, list):
        raise TypeError("Benchmark Run resume schedule must be a list")
    rules: list[RuleLibraryEntry] = []
    seen: set[tuple[str | None, str, str]] = set()
    for scheduled in schedule:
        if not isinstance(scheduled, dict):
            raise TypeError("Benchmark Run resume schedule entry must be an object")
        description = scheduled["rule_description"]
        code = scheduled["rule_code"]
        name = scheduled["rule_name"]
        if not isinstance(description, str) or not isinstance(code, str):
            raise TypeError(
                "Benchmark Run resume requires an embedded deterministic Round rule"
            )
        key = (name if isinstance(name, str) else None, description, code)
        if key in seen:
            continue
        seen.add(key)
        rule: RuleLibraryEntry = {"description": description, "code": code}
        if key[0] is not None:
            rule["name"] = key[0]
        rules.append(rule)
    return rules


def _restore_active_round_cursor(
    active: ActiveStoredRound,
) -> tuple[Rule, str | None, int] | None:
    """Validate and restore the orchestration cursor for one active Round."""
    turns = active.record["turns"]
    next_turn_index = active.continuation["next_turn_index"]
    if (
        not isinstance(turns, list)
        or not isinstance(next_turn_index, int)
        or len(turns) != next_turn_index
    ):
        logger.error(
            "Benchmark Run resume incompatible: active Round Record and checkpoint "
            "disagree on completed Turns"
        )
        return None
    rule_payload = active.continuation["rule"]
    if not isinstance(rule_payload, dict):
        logger.error("Benchmark Run resume incompatible: active rule is malformed")
        return None
    description = rule_payload["description"]
    code = rule_payload["code"]
    if not isinstance(description, str) or not isinstance(code, str):
        logger.error("Benchmark Run resume incompatible: active rule is malformed")
        return None
    secret_rule = active.record["secret_rule"]
    current_rule_name = (
        secret_rule.get("name")
        if isinstance(secret_rule, dict) and isinstance(secret_rule.get("name"), str)
        else None
    )
    return Rule(description, code), current_rule_name, next_turn_index


def _initialize_sqlite_resume_state(
    startup: EvaluationStartup,
) -> EvaluationState | None:
    """Restore orchestration state from the active SQLite Round Checkpoint."""
    run_store = startup.run_store
    manifest = startup.run_manifest
    if run_store is None or manifest is None:
        return None
    progress = run_store.read_progress()
    active = run_store.read_resumable_round()
    completed_rounds = run_store.read_completed_rounds()
    if progress.is_complete:
        logger.info(
            "Benchmark Run already complete (%s/%s Rounds)",
            progress.completed_rounds,
            progress.total_rounds,
        )
        return None
    if progress.next_round_number is None:
        logger.error("Benchmark Run resume incompatible: next Round is unavailable")
        return None
    rules = _rules_from_run_manifest(manifest)
    folder_name = run_store.run_folder.name
    results = _new_evaluation_results(
        startup,
        folder_name,
        rules,
        startup.rules_config["index"] + len(completed_rounds),
    )
    start_round = progress.next_round_number
    current_rule: Rule | None = None
    current_rule_name: str | None = None
    next_turn_index = 0
    if active is not None:
        restored_cursor = _restore_active_round_cursor(active)
        if restored_cursor is None:
            return None
        current_rule, current_rule_name, next_turn_index = restored_cursor
        start_round = active.round_number
    versions = manifest["versions"]
    logger.info("=" * 80)
    logger.info("RESUMING AUTHORITATIVE BENCHMARK RUN")
    logger.info("=" * 80)
    if active is None:
        logger.info(
            "Next scheduled Round %s after %s completed Rounds; schema versions %s",
            start_round,
            len(completed_rounds),
            versions,
        )
    else:
        logger.info(
            "Active Round %s: %s committed Turns; schema versions %s",
            active.round_number,
            next_turn_index,
            versions,
        )
    return EvaluationState(
        startup=startup,
        results=results,
        folder_name=folder_name,
        start_round=start_round,
        current_rule=current_rule,
        current_rule_name=current_rule_name,
        rule_factory_index=startup.rules_config["index"] + len(completed_rounds),
        checkpoint_rules_library=rules,
        all_rules_library=rules,
        rule_name_to_index={
            name: index
            for index, rule in enumerate(rules)
            if isinstance((name := rule.get("name")), str)
        },
        run_store=run_store,
    )


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
    """Load the library and initialize a fresh authoritative Benchmark Run."""
    folder_name = f"solo_evaluation_{startup.timestamp}_{startup.output_tag}"
    configured_game_seed = startup.game_config["seed"]
    if configured_game_seed is None:
        effective_seed = secrets.randbits(32)
        startup.game_config["seed"] = effective_seed
        startup.config["game"]["seed"] = effective_seed
        logger.info("Generated effective Benchmark Run game seed: %s", effective_seed)
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
    manifest = create_benchmark_run_manifest(
        startup,
        all_rules,
        run_id=str(uuid.uuid4()),
        configured_game_seed=configured_game_seed,
    )
    run_store = BenchmarkRunStore.create(
        Path("results") / folder_name,
        manifest,
    )
    startup.run_store = run_store
    startup.run_manifest = manifest
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
        run_store=run_store,
    )


def initialize_evaluation_state(
    startup: EvaluationStartup,
) -> EvaluationState | None:
    """Create fresh state or restore resume state for the evaluation loop."""
    if startup.run_store is not None:
        return _initialize_sqlite_resume_state(startup)
    if startup.checkpoint:
        return _initialize_resume_state(startup)
    return _initialize_fresh_state(startup)
