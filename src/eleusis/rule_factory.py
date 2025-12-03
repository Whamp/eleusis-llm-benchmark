"""Factory for creating rules from various sources."""

import json
import logging
import random
from pathlib import Path

from eleusis.cards import Card, Suit
from eleusis.game_engine import Rule
from eleusis.game_master import GameMaster
from eleusis.rules import RuleValidator

logger = logging.getLogger(__name__)


class RuleFactory:
    """Factory for creating rules from LLM or pre-generated library."""

    def __init__(
        self,
        mode: str,
        library_path: str | None = None,
        selection: str = "random",
        game_master: GameMaster | None = None,
        validator: RuleValidator | None = None,
        min_acceptance: float = 0.0,
        max_acceptance: float = 1.0,
        start_index: int = 0,
    ) -> None:
        """Initialize rule factory."""
        self.mode = mode
        self.library_path = library_path
        self.selection = selection
        self.game_master = game_master
        self.validator = validator
        self.min_acceptance = min_acceptance
        self.max_acceptance = max_acceptance
        self._library_index = start_index
        self._library_rules: list[dict] | None = None

        if mode == "library":
            if not library_path:
                raise ValueError("library_path required for library mode")
            self._load_library()
        elif mode == "llm":
            if not game_master or not validator:
                raise ValueError("game_master and validator required for llm mode")
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'llm' or 'library'")

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

    def _evaluate_acceptance_rate(
        self, rule: Rule, num_simulations: int = 5, plays_per_simulation: int = 50
    ) -> float:
        """Evaluate rule's acceptance rate by simulating random plays.

        Args:
            rule: The rule to evaluate
            num_simulations: Number of simulations to run
            plays_per_simulation: Number of random card plays per simulation

        Returns:
            Average acceptance rate across simulations
        """
        all_cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]
        sim_results = []

        for _ in range(num_simulations):
            total_plays = 0
            total_accepted = 0
            mainline = []

            for _ in range(plays_per_simulation):
                card = random.choice(all_cards)
                accepted = rule.evaluate(card, mainline)

                total_plays += 1
                if accepted:
                    total_accepted += 1
                    mainline.append(card)

            acceptance_rate = total_accepted / total_plays if total_plays > 0 else 0.0
            sim_results.append(acceptance_rate)

        avg_acceptance = sum(sim_results) / num_simulations
        return avg_acceptance

    def _is_acceptance_rate_valid(self, rate: float) -> bool:
        """Check if acceptance rate is within configured bounds."""
        return self.min_acceptance <= rate <= self.max_acceptance

    def create_rule(self) -> Rule:
        """Create a rule based on the configured mode."""
        if self.mode == "library":
            return self._create_from_library()
        else:
            return self._create_from_llm()

    def _create_from_library(self) -> Rule:
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

    def _create_from_llm(self) -> Rule:
        """Create rule by generating with LLM, ensuring acceptance rate is within bounds."""
        max_generation_attempts = 5

        for attempt in range(max_generation_attempts):
            logger.info(
                f"Generating rule using GameMaster "
                f"(attempt {attempt + 1}/{max_generation_attempts})..."
            )

            rule = self.game_master.create_rule()

            # Validate rule
            validation_result = self.validator.validate_rule(rule)
            if not validation_result.valid:
                logger.warning(f"Rule validation failed: {validation_result.issues}")
                continue

            # Evaluate acceptance rate
            logger.info("Evaluating acceptance rate...")
            acceptance_rate = self._evaluate_acceptance_rate(rule, num_simulations=5)
            logger.info(f"Acceptance rate: {acceptance_rate:.2%}")

            if self._is_acceptance_rate_valid(acceptance_rate):
                logger.info(
                    f"✓ Rule within acceptable range "
                    f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}]"
                )
                return rule
            else:
                logger.warning(
                    f"✗ Rule acceptance rate {acceptance_rate:.2%} outside bounds "
                    f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}], regenerating..."
                )
                continue

        raise RuntimeError(
            f"Failed to generate rule with acceptance rate in range "
            f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}] "
            f"after {max_generation_attempts} attempts"
        )
