"""Factory for creating rules from library."""

import json
import logging
import random
from pathlib import Path

from eleusis.game_engine_solo import Rule

logger = logging.getLogger(__name__)


class RuleFactory:
    """Factory for creating rules from pre-generated library."""

    def __init__(
        self,
        library_path: str,
        selection: str = "random",
        min_acceptance: float = 0.0,
        max_acceptance: float = 1.0,
        start_index: int = 0,
    ) -> None:
        """Initialize rule factory."""
        self.library_path = library_path
        self.selection = selection
        self.min_acceptance = min_acceptance
        self.max_acceptance = max_acceptance
        self._library_index = start_index
        self._library_rules: list[dict] | None = None

        self._load_library()

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

    def _is_acceptance_rate_valid(self, rate: float) -> bool:
        """Check if acceptance rate is within configured bounds."""
        return self.min_acceptance <= rate <= self.max_acceptance

    def create_rule(self) -> Rule:
        """Create rule from pre-generated library, filtering by acceptance rate."""
        max_attempts = len(self._library_rules)

        for attempt in range(max_attempts):
            if self.selection == "random":
                rule_dict = random.choice(self._library_rules)
            else:  # sequential
                rule_dict = self._library_rules[self._library_index]
                self._library_index = (self._library_index + 1) % len(self._library_rules)

            # Check if rule has pre-evaluated acceptance rate
            if "avg_acceptance_rate" in rule_dict:
                acceptance_rate = rule_dict["avg_acceptance_rate"]
                if not self._is_acceptance_rate_valid(acceptance_rate):
                    logger.info(
                        f"Skipping rule '{rule_dict.get('name', 'unknown')}' "
                        f"(acceptance rate {acceptance_rate:.2%} outside bounds "
                        f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}])"
                    )
                    continue

            description = rule_dict["description"]
            code = rule_dict["code"]
            name = rule_dict.get("name", "library_rule")

            logger.info(f"Using library rule: {name}")
            logger.info(f"Description: {description}")

            if "avg_acceptance_rate" in rule_dict:
                rate = rule_dict["avg_acceptance_rate"]
                logger.info(f"Pre-evaluated acceptance rate: {rate:.2%}")

            logger.debug(f"Code:\n{code}")

            return Rule(description, code)

        # If we exhausted all rules without finding one in bounds
        raise RuntimeError(
            f"No rules found in library with acceptance rate in range "
            f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}]"
        )
