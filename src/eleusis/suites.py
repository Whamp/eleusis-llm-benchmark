"""Named benchmark suite resolution.

Loads suite definitions from suites.yaml and expands them into
(rule_name, batch_round_index) pairs for the runner.
"""

import json
from pathlib import Path

import yaml

_DEFAULT_SUITES_PATH = Path(__file__).parent.parent.parent / "suites.yaml"
_DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent / "rules.json"


def load_suites(suites_path: Path = _DEFAULT_SUITES_PATH) -> dict:
    """Load suite definitions from YAML file.

    Returns dict mapping suite name to its definition.
    """
    with open(suites_path) as f:
        return yaml.safe_load(f)


def resolve_suite(
    suite_name: str,
    suites_path: Path = _DEFAULT_SUITES_PATH,
    rules_path: Path = _DEFAULT_RULES_PATH,
) -> list[tuple[str, int]]:
    """Expand a named suite into (rule_name, batch_round_index) pairs.

    Args:
        suite_name: Name of the suite (e.g. "full_26x3", "screen_26x1").
        suites_path: Path to suites.yaml.
        rules_path: Path to rules.json (used when rules == "all").

    Returns:
        List of (rule_name, batch_round_index) tuples in execution order.

    Raises:
        ValueError: If suite_name is not found in suites.yaml.
    """
    suites = load_suites(suites_path)

    if suite_name not in suites:
        available = ", ".join(sorted(suites.keys()))
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {available}")

    suite_def = suites[suite_name]
    batch_indices = suite_def["batch_round_indices"]

    # Resolve rule list
    rules_spec = suite_def["rules"]
    if rules_spec == "all":
        with open(rules_path) as f:
            library = json.load(f)
        rule_names = [r["name"] for r in library["rules"]]
    else:
        rule_names = list(rules_spec)

    # Expand: for each rule, iterate over batch indices
    cases = []
    for rule_name in rule_names:
        for batch_idx in batch_indices:
            cases.append((rule_name, batch_idx))

    return cases
