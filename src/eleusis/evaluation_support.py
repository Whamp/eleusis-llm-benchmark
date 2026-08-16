"""Configuration, persistence, and preflight helpers for single evaluations."""

import argparse
import json
import logging
import re
from pathlib import Path

import yaml

from eleusis.benchmark_config import BenchmarkConfig, parse_benchmark_config
from eleusis.evaluation_results import EvaluationResults
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
