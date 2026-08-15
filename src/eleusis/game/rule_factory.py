"""Rule selection from validated pre-generated libraries."""

import json
import logging
import random
from pathlib import Path
from typing import Literal

from eleusis.game.engine import Rule
from eleusis.game.rule_library import (
    RuleLibraryEntry,
    RuleMetadata,
    parse_rule_library_entries,
)

logger = logging.getLogger(__name__)


class RuleFactory:
    """Select executable rules from a validated pre-generated library."""

    def __init__(
        self,
        library_path: str | None = None,
        selection: Literal["random", "sequential"] = "random",
        start_index: int = 0,
        rules_list: list[RuleLibraryEntry] | None = None,
    ) -> None:
        """Initialize from a path or an already validated rules list."""
        self.library_path = library_path
        self.selection = selection
        self._library_index = start_index
        if rules_list is not None:
            self._library_rules = rules_list
            logger.info(f"Using pre-loaded rules list ({len(rules_list)} rules)")
        elif library_path is not None:
            self._library_rules = self._load_library(library_path)
        else:
            raise ValueError("Either library_path or rules_list must be provided")
        if self._library_index >= len(self._library_rules):
            raise IndexError(
                f"Rule index {self._library_index} out of bounds. Library has"
                f" {len(self._library_rules)} rules (indices"
                f" 0-{len(self._library_rules) - 1})."
            )

    def _load_library(self, library_path: str) -> list[RuleLibraryEntry]:
        """Load and validate a rule library from a JSON file."""
        path = Path(library_path)
        if not path.exists():
            raise FileNotFoundError(f"Rule library not found: {path}")
        with path.open() as rule_file:
            data = json.load(rule_file)
        if not isinstance(data, dict):
            raise TypeError("Rule library must contain a JSON object")
        library_rules = parse_rule_library_entries(data.get("rules", []))
        if not library_rules:
            raise ValueError(f"No rules found in library: {path}")
        logger.info(f"Loaded {len(library_rules)} rules from {path}")
        return library_rules

    def create_rule_with_metadata(self) -> tuple[Rule, RuleMetadata]:
        """Create a selected rule and return its full library metadata."""
        if self.selection == "random":
            rule_dict = random.choice(self._library_rules)
        else:
            rule_dict = self._library_rules[self._library_index]
            self._library_index = (self._library_index + 1) % len(self._library_rules)
        description = rule_dict["description"]
        code = rule_dict["code"]
        name = rule_dict.get("name", "library_rule")
        logger.info(f"Using library rule: {name}")
        logger.info(f"Description: {description}")
        if "avg_acceptance_rate" in rule_dict:
            logger.info(
                f"Pre-evaluated acceptance rate: {rule_dict['avg_acceptance_rate']:.2%}"
            )
        logger.debug(f"Code:\n{code}")
        metadata: RuleMetadata = {
            "name": rule_dict.get("name"),
            "description": description,
            "code": code,
        }
        return Rule(description, code), metadata
