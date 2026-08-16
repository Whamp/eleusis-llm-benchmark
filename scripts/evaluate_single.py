"""Evaluate a single LLM player in solo pattern discovery mode."""

import argparse

from dotenv import load_dotenv

from eleusis.evaluation_support import (
    apply_cli_overrides,
    generate_output_tag,
    get_integer_metric,
    load_config,
    load_rules_from_library,
    logger,
    preflight_check,
    save_evaluation_results,
)

__all__ = [
    "apply_cli_overrides",
    "generate_output_tag",
    "get_integer_metric",
    "load_config",
    "load_rules_from_library",
    "logger",
    "main",
    "parse_args",
    "preflight_check",
    "save_evaluation_results",
]

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate single LLM in solo pattern discovery mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with a model
  python scripts/evaluate_single.py --model "claude-opus"

  # Test 20 rules with a specific model and custom tag
  python scripts/evaluate_single.py --model "gpt-5.2" --num-rules 20 --tag gpt

  # Start from rule index 10
  python scripts/evaluate_single.py --model "gpt-5.2" --rule-index 10

  # Resume interrupted evaluation (model taken from checkpoint)
  python scripts/evaluate_single.py --resume results/solo_evaluation_20251205_151306
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to resume folder (e.g., results/solo_evaluation_20251205_151306)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model key from models.yaml (required unless --resume)",
    )
    parser.add_argument(
        "--num-rules", type=int, help="Number of distinct rules to test"
    )
    parser.add_argument(
        "--rule-index", type=int, help="Starting rule index (for sequential selection)"
    )
    parser.add_argument("--max-turns", type=int, help="Maximum turns per round")
    parser.add_argument(
        "--tag", type=str, help="Tag to append to output folder name for identification"
    )
    parser.add_argument(
        "--batch-round-offset",
        type=int,
        default=None,
        help=(
            "Run only 1 round per rule using this batch index for seeding. "
            "Use with parallel workers: worker 0 gets offset 0, worker 1 gets 1, etc. "
            "Each offset produces a different deck shuffle for the same rule."
        ),
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help=(
            "Named benchmark suite from suites.yaml "
            "(e.g. full_26x3, screen_26x1, stress_12x1). "
            "Overrides num_rules/num_rounds_per_rule/batch_round_offset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate one configured model across fresh or resumed rounds."""
    from eleusis.evaluation_orchestrator import run_evaluation

    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
