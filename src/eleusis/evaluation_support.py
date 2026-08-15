"""Configuration, persistence, and preflight helpers for single evaluations."""

import argparse
import json
import logging
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from eleusis.benchmark_config import BenchmarkConfig, parse_benchmark_config
from eleusis.evaluation_results import (
    CurrentRuleCheckpoint,
    EvaluationResults,
    parse_evaluation_results,
)
from eleusis.game import Rule
from eleusis.game.rule_library import RuleLibraryEntry, parse_rule_library_entries
from eleusis.llm import create_client

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> BenchmarkConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / config_path
    with path.open() as config_file:
        return parse_benchmark_config(yaml.safe_load(config_file))


def apply_cli_overrides(
    config: BenchmarkConfig, args: argparse.Namespace
) -> BenchmarkConfig:
    """Apply CLI argument overrides to config."""
    game_config = config["game"]

    # Number of rules override
    if args.num_rules is not None:
        game_config["num_rules"] = args.num_rules

    # Max turns override
    if args.max_turns is not None:
        game_config["max_turns"] = args.max_turns

    # Rule index override
    if args.rule_index is not None:
        config["rules"]["index"] = args.rule_index

    # Batch round offset: run 1 round per rule with a specific batch index
    if args.batch_round_offset is not None:
        game_config["num_rounds_per_rule"] = 1
        game_config["batch_round_offset"] = args.batch_round_offset

    return config


def generate_output_tag(args: argparse.Namespace, player_name: str) -> str:
    """Generate output folder tag based on CLI args or player name."""
    if args.tag:
        return args.tag

    tag = player_name.lower()
    tag = re.sub(r"[^a-z0-9]+", "_", tag)
    tag = tag.strip("_")[:30]
    return tag


def get_integer_metric(metrics: dict[str, object], key: str) -> int:
    """Return an integer metric value, treating missing or invalid values as zero."""
    value = metrics.get(key, 0)
    return value if isinstance(value, int) else 0


def save_evaluation_results(
    evaluation_results: EvaluationResults, folder_name: str
) -> str:
    """Save evaluation results to JSON file (incremental)."""
    Path(f"results/{folder_name}").mkdir(parents=True, exist_ok=True)
    output_file = f"results/{folder_name}/results.json"
    with Path(output_file).open("w") as results_file:
        json.dump(evaluation_results, results_file, indent=2)
    return output_file


def load_checkpoint(resume_folder: str) -> EvaluationResults | None:
    """Load checkpoint from results.json in resume folder."""
    results_path = Path(resume_folder) / "results.json"
    if not results_path.exists():
        logger.error(f"No results.json found in {resume_folder}")
        return None

    try:
        with results_path.open() as results_file:
            checkpoint = parse_evaluation_results(json.load(results_file))
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid JSON in results.json: {e}")
        return None

    return checkpoint


def restore_rule_from_checkpoint(
    rule_data: CurrentRuleCheckpoint | dict[str, object] | None,
) -> Rule | None:
    """Restore Rule object from checkpoint data."""
    if not rule_data:
        return None
    description = rule_data.get("description")
    code = rule_data.get("code")
    if not isinstance(description, str) or not isinstance(code, str):
        return None
    return Rule(description, code)


def reconstruct_config_from_checkpoint(
    checkpoint: EvaluationResults,
) -> BenchmarkConfig:
    """Reconstruct full config dict from checkpoint data for self-contained resume."""
    cfg = checkpoint["config"]
    chk = checkpoint["checkpoint"]

    return parse_benchmark_config(
        {
            "model": cfg["player_model"],
            "game": {
                "num_rules": cfg["num_rules"],
                "num_rounds_per_rule": cfg["num_rounds_per_rule"],
                "max_turns": cfg["max_turns"],
                "hand_size": cfg["hand_size"],
                "wrong_guess_penalty": cfg["wrong_guess_penalty"],
                "seed": cfg.get("seed"),
                "batch_round_offset": cfg.get("batch_round_offset"),
            },
            "llm": {
                "max_tokens": cfg.get("llm_max_tokens"),
                "temperature": cfg.get("llm_temperature"),
                "seed": cfg.get("llm_seed"),
                "max_llm_retries": cfg.get("llm_max_retries"),
            },
            "rule_compiler": {
                "provider": cfg["rule_compiler_provider"],
                "model_id": cfg["rule_compiler_model_id"],
                "reasoning_format": cfg.get(
                    "rule_compiler_reasoning_format", "separate_field"
                ),
                # hf_provider intentionally omitted - allows backup providers
                "temperature": cfg.get("rule_compiler_temperature"),
                "max_retries": cfg.get("rule_compiler_max_retries", 10),
                "num_simulations": cfg.get("rule_compiler_num_simulations", 100),
                "turns_per_simulation": cfg.get(
                    "rule_compiler_turns_per_simulation", 40
                ),
                "simulation_seed": cfg.get("rule_compiler_simulation_seed"),
            },
            "rules": {
                "library_path": None,  # Not needed, rules embedded in checkpoint
                "selection": chk["rule_factory_state"]["selection"],
                "index": chk["rule_factory_state"]["current_index"],
            },
        }
    )


def load_rules_from_library(config: BenchmarkConfig) -> list[RuleLibraryEntry]:
    """Load all rules from library.

    Returns list of rule dicts with 'description', 'code', 'name', etc.
    """
    import json
    from pathlib import Path

    rules_cfg = config["rules"]
    library_path_value = rules_cfg["library_path"]
    if library_path_value is None:
        raise ValueError("Rule library path is required when loading rules from disk")
    library_path = Path(library_path_value)

    if not library_path.exists():
        logger.error(f"Rule library not found: {library_path}")
        return []

    with library_path.open() as rules_file:
        data = json.load(rules_file)
    if not isinstance(data, dict):
        raise TypeError("Rule library must contain a JSON object")

    rules = parse_rule_library_entries(data.get("rules", []))
    logger.info(f"Loaded {len(rules)} rules from library")
    return rules


def preflight_check(model_key: str) -> None:
    """Run pre-flight model check.

    Fails fast on issues.

    Args:
            model_key: Model key from models.yaml (e.g., "claude-opus", "deepseek-r1")

    Raises:
            SystemExit: If pre-flight check fails
    """
    import time

    logger.info("Running pre-flight model check...")

    try:
        start = time.time()
        client = create_client(model_key, max_tokens=16384)
        # Simple connectivity test
        response = client.generate("Say 'hello' and nothing else.")
        latency = time.time() - start
    # Preflight is the top-level boundary for arbitrary provider failures.
    except Exception as error:
        logger.error(f"Pre-flight check failed: {error}")
        raise SystemExit(1) from error

    logger.info(f"  Provider: {client.provider_name}")
    logger.info(f"  Model: {client.model_name}")
    logger.info(f"  Latency: {latency:.2f}s")
    logger.info(f"  Response: {response[:100]}...")
