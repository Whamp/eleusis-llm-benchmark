"""Factory for creating rules from library."""

import json
import logging
import random
from pathlib import Path

from eleusis.game_engine import Rule

__all__ = ["RuleFactory"]

logger = logging.getLogger(__name__)


class RuleFactory:
    """Factory for creating rules from pre-generated library."""

    def __init__(
        self,
        library_path: str | None = None,
        selection: str = "random",
        start_index: int = 0,
        rules_list: list[dict] | None = None,
    ) -> None:
        """Initialize rule factory.

        Args:
            library_path: Path to rules.json file (used if rules_list not provided)
            selection: "sequential" or "random"
            start_index: Starting index for sequential selection
            rules_list: Pre-loaded list of rule dicts (takes precedence over library_path)
        """
        self.library_path = library_path
        self.selection = selection
        self._library_index = start_index
        self._library_rules: list[dict] | None = None

        if rules_list is not None:
            self._library_rules = rules_list
            logger.info(f"Using pre-loaded rules list ({len(rules_list)} rules)")
        elif library_path is not None:
            self._load_library()
        else:
            raise ValueError("Either library_path or rules_list must be provided")

        # Validate start_index is within bounds
        if self._library_index >= len(self._library_rules):
            raise IndexError(
                f"Rule index {self._library_index} out of bounds. "
                f"Library has {len(self._library_rules)} rules (indices 0-{len(self._library_rules)-1})."
            )

    def _load_library(self) -> None:
        """Load rule library from JSON file."""
        path = Path(self.library_path)
        if not path.exists():
            raise FileNotFoundError(f"Rule library not found: {path}")

        with open(path) as f:
            data = json.load(f)

        self._library_rules = data.get("rules", [])
        if not self._library_rules:
            raise ValueError(f"No rules found in library: {path}")

        logger.info(f"Loaded {len(self._library_rules)} rules from {path}")

    def create_rule_with_metadata(self) -> tuple[Rule, dict]:
        """Create rule and return with full metadata from library.

        Returns:
            Tuple of (Rule, metadata_dict) where metadata contains:
            - name: Unique rule name from library
            - description: Rule description
            - code: Rule Python code
        """
        if self.selection == "random":
            rule_dict = random.choice(self._library_rules)
        else:  # sequential
            rule_dict = self._library_rules[self._library_index]
            self._library_index = (self._library_index + 1) % len(self._library_rules)

        description = rule_dict["description"]
        code = rule_dict["code"]
        name = rule_dict.get("name", "library_rule")

        logger.info(f"Using library rule: {name}")
        logger.info(f"Description: {description}")

        if "avg_acceptance_rate" in rule_dict:
            rate = rule_dict["avg_acceptance_rate"]
            logger.info(f"Pre-evaluated acceptance rate: {rate:.2%}")

        logger.debug(f"Code:\n{code}")

        metadata = {
            "name": rule_dict.get("name"),
            "description": description,
            "code": code,
        }
        return Rule(description, code), metadata
