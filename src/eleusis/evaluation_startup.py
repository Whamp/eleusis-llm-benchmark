"""Configuration, suite, logging, and round-count startup for one evaluation."""

import argparse
import copy
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from eleusis.benchmark_config import BenchmarkConfig, GameConfig, RulesConfig
from eleusis.benchmark_run_manifest import (
    BenchmarkRunManifestIncompatibilityError,
    restore_benchmark_run_config,
    verify_benchmark_run_resume_compatibility,
)
from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
    BenchmarkRunStoreError,
)
from eleusis.evaluation_support import (
    apply_cli_overrides,
    generate_output_tag,
    load_config,
    load_rules_from_library,
    preflight_check,
)
from eleusis.suites import resolve_suite
from eleusis.utils import model_spec_to_display_name, setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationStartup:
    """Resolved inputs and display metadata for one evaluation session."""

    args: argparse.Namespace
    config: BenchmarkConfig
    player_model: str
    player_display_name: str
    rule_compiler_display_name: str
    num_rounds_per_rule: int
    suite_name: str | None
    suite_cases: list[tuple[str, int]] | None
    game_config: GameConfig
    rules_config: RulesConfig
    output_tag: str
    timestamp: str
    log_file: str
    num_rounds: int
    num_rules: int
    run_store: BenchmarkRunStore | None = None
    run_manifest: dict[str, object] | None = None


def _load_sqlite_resume_startup(
    args: argparse.Namespace,
    run_store: BenchmarkRunStore,
) -> tuple[
    BenchmarkConfig,
    dict[str, object],
    str,
    str,
    str,
    int,
    str | None,
    list[tuple[str, int]] | None,
]:
    """Validate and reconstruct startup values from an authoritative Run store."""
    manifest = run_store.read_manifest()
    model_identity = manifest["model_identity"]
    compiler_identity = manifest["compiler_identity"]
    if not isinstance(model_identity, dict) or not isinstance(compiler_identity, dict):
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run resume incompatible: model identities are malformed"
        )
    stored_model = model_identity["model_key"]
    if not isinstance(stored_model, str):
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run resume incompatible: model_identity.model_key is malformed"
        )
    if args.model and args.model != stored_model:
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run resume incompatible: scientific_config.model changed"
        )
    current_config = load_config(args.config)
    config = restore_benchmark_run_config(manifest, current_config)
    comparison_config = apply_cli_overrides(copy.deepcopy(config), args)
    comparison_config["model"] = stored_model
    if args.suite is not None:
        comparison_config["suite"] = args.suite
    verify_benchmark_run_resume_compatibility(manifest, comparison_config)
    schedule = manifest["schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise BenchmarkRunManifestIncompatibilityError(
            "Benchmark Run resume incompatible: schedule is empty"
        )
    cases: list[tuple[str, int]] = []
    for scheduled in schedule:
        if not isinstance(scheduled, dict):
            raise BenchmarkRunManifestIncompatibilityError(
                "Benchmark Run resume incompatible: schedule entry is malformed"
            )
        name = scheduled["rule_name"]
        batch_index = scheduled["batch_round_index"]
        if not isinstance(name, str) or not isinstance(batch_index, int):
            cases = []
            break
        cases.append((name, batch_index))
    suite_value = config.get("suite")
    suite_name = suite_value if isinstance(suite_value, str) else None
    rounds_per_rule = config["game"].get("num_rounds_per_rule", 1)
    return (
        config,
        manifest,
        stored_model,
        str(model_identity["display_name"]),
        str(compiler_identity["display_name"]),
        rounds_per_rule,
        suite_name,
        cases or None,
    )


def _resolve_suite_cases(
    args: argparse.Namespace,
    config: BenchmarkConfig,
) -> tuple[str | None, list[tuple[str, int]] | None]:
    """Resolve and optionally partition a named suite for this worker."""
    suite_name = args.suite or config.get("suite")
    if not suite_name:
        return None, None
    cases = resolve_suite(suite_name)
    if args.batch_round_offset is not None:
        cases = [case for case in cases if case[1] == args.batch_round_offset]
    logger.info("Suite %r: %s rounds", suite_name, len(cases))
    return suite_name, cases


def _configure_evaluation_logging(
    args: argparse.Namespace,
    player_display_name: str,
) -> tuple[str, str, str]:
    """Resolve output tag and configure timestamped console/file logging."""
    output_tag = generate_output_tag(args, player_display_name)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    Path("logs").mkdir(exist_ok=True)
    log_file = f"logs/solo_evaluation_{timestamp}_{output_tag}.txt"
    setup_logging(
        log_file=log_file,
        console_level=logging.INFO,
        file_level=logging.DEBUG,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return output_tag, timestamp, log_file


def _resolve_round_counts(
    config: BenchmarkConfig,
    suite_cases: list[tuple[str, int]] | None,
    num_rounds_per_rule: int,
) -> tuple[int, int] | None:
    """Resolve and validate rule and round counts for the selected mode."""
    game_config = config["game"]
    if suite_cases:
        num_rounds = len(suite_cases)
        num_rules = len(dict.fromkeys(name for name, _index in suite_cases))
        game_config["num_rounds"] = num_rounds
        return num_rules, num_rounds
    all_rules = load_rules_from_library(config)
    num_rules = game_config.get("num_rules", 10)
    if num_rules == 0:
        num_rules = len(all_rules)
        game_config["num_rules"] = num_rules
        logger.info("num_rules=0: using entire library (%s rules)", num_rules)
    elif len(all_rules) < num_rules:
        logger.error(
            "Not enough rules: %s available, %s requested", len(all_rules), num_rules
        )
        return None
    num_rounds = num_rules * num_rounds_per_rule
    game_config["num_rounds"] = num_rounds
    return num_rules, num_rounds


def resolve_evaluation_startup(args: argparse.Namespace) -> EvaluationStartup | None:
    """Resolve all startup inputs and fail fast before evaluation state mutation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.resume and not args.model:
        logger.error("--model is required (unless using --resume)")
        return None
    run_store: BenchmarkRunStore | None = None
    run_manifest: dict[str, object] | None = None
    database_path = (
        Path(args.resume) / BENCHMARK_RUN_DATABASE_NAME if args.resume else None
    )
    sqlite_resume = database_path is not None and database_path.is_file()
    if args.resume and not sqlite_resume:
        results_path = Path(args.resume) / "results.json"
        if results_path.is_file():
            logger.error(
                "Historical JSON-only Benchmark Run cannot be resumed. "
                "Start a new Run; legacy import is not implemented."
            )
        else:
            logger.error(
                "Benchmark Run resume unavailable: neither %s nor results.json exists",
                BENCHMARK_RUN_DATABASE_NAME,
            )
        return None
    if sqlite_resume:
        try:
            run_store = BenchmarkRunStore(Path(args.resume))
            (
                config,
                run_manifest,
                player_model,
                player_name,
                compiler_name,
                rounds_per_rule,
                suite_name,
                suite_cases,
            ) = _load_sqlite_resume_startup(args, run_store)
        except (
            BenchmarkRunManifestIncompatibilityError,
            BenchmarkRunStoreError,
            OSError,
            ValidationError,
            yaml.YAMLError,
        ) as error:
            logger.error("%s", error)
            return None
    else:
        config = apply_cli_overrides(load_config(args.config), args)
        config["model"] = args.model
        player_model = args.model
        player_name = model_spec_to_display_name(player_model)
        compiler_name = model_spec_to_display_name(config["rule_compiler"]["model_id"])
        rounds_per_rule = config["game"].get("num_rounds_per_rule", 1)
        suite_name, suite_cases = _resolve_suite_cases(args, config)
    output_tag, timestamp, log_file = _configure_evaluation_logging(args, player_name)
    logger.info("=" * 80)
    logger.info("PRE-FLIGHT MODEL CHECK")
    logger.info("=" * 80)
    preflight_check(player_model)
    logger.info("Pre-flight check passed!\n")
    if run_manifest is not None:
        schedule = run_manifest["schedule"]
        assert isinstance(schedule, list)
        num_rounds = len(schedule)
        num_rules = len(
            {
                scheduled["rule_name"]
                for scheduled in schedule
                if isinstance(scheduled, dict)
            }
        )
    else:
        counts = _resolve_round_counts(config, suite_cases, rounds_per_rule)
        if counts is None:
            return None
        num_rules, num_rounds = counts
    return EvaluationStartup(
        args=args,
        config=config,
        player_model=player_model,
        player_display_name=player_name,
        rule_compiler_display_name=compiler_name,
        num_rounds_per_rule=rounds_per_rule,
        suite_name=suite_name,
        suite_cases=suite_cases,
        game_config=config["game"],
        rules_config=config["rules"],
        output_tag=output_tag,
        timestamp=timestamp,
        log_file=log_file,
        num_rounds=num_rounds,
        num_rules=num_rules,
        run_store=run_store,
        run_manifest=run_manifest,
    )
