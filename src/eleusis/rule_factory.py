"""Factory for creating rules from various sources."""

import json
import logging
import random
from pathlib import Path

from eleusis.cards import Card, Suit
from eleusis.game_engine import Rule
from eleusis.llm_client import HuggingFaceClient
from eleusis.llm_player import LLMRuleMaker
from eleusis.python_rule import PythonRule
from eleusis.rules import RuleValidator

logger = logging.getLogger(__name__)


class RuleFactory:
    """Factory for creating rules from LLM or pre-generated library."""

    def __init__(
        self,
        mode: str,
        library_path: str | None = None,
        selection: str = "random",
        llm_client: HuggingFaceClient | None = None,
        validator: RuleValidator | None = None,
        min_acceptance: float = 0.0,
        max_acceptance: float = 1.0,
    ) -> None:
        """Initialize rule factory.

        Args:
            mode: "llm" or "library"
            library_path: Path to rules JSON file (required for library mode)
            selection: "random" or "sequential" (for library mode)
            llm_client: LLM client for generating rules (required for llm mode)
            validator: Validator for generated rules (required for llm mode)
            min_acceptance: Minimum acceptance rate for rules (0.0-1.0)
            max_acceptance: Maximum acceptance rate for rules (0.0-1.0)
        """
        self.mode = mode
        self.library_path = library_path
        self.selection = selection
        self.llm_client = llm_client
        self.validator = validator
        self.min_acceptance = min_acceptance
        self.max_acceptance = max_acceptance
        self._library_index = 0
        self._library_rules: list[dict] | None = None

        if mode == "library":
            if not library_path:
                raise ValueError("library_path required for library mode")
            self._load_library()
        elif mode == "llm":
            if not llm_client or not validator:
                raise ValueError("llm_client and validator required for llm mode")
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
        self, rule: PythonRule, num_simulations: int = 5, plays_per_simulation: int = 50
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

            return PythonRule(description, code)

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
                f"Generating rule using LLM "
                f"(attempt {attempt + 1}/{max_generation_attempts})..."
            )
            rule_maker = LLMRuleMaker(
                self.llm_client,
                self.validator,
                max_attempts=3,
                max_tokens=8192,
            )

            rule = rule_maker.generate_rule()
            if not rule:
                logger.warning("Failed to generate valid rule, retrying...")
                continue

            # Only evaluate PythonRules (LLMGeneratedRules are too slow to evaluate)
            if isinstance(rule, PythonRule):
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
            else:
                # LLMGeneratedRule - skip evaluation (too slow)
                logger.warning(
                    "Generated LLMGeneratedRule (not PythonRule), "
                    "skipping acceptance rate check"
                )
                return rule

        raise RuntimeError(
            f"Failed to generate rule with acceptance rate in range "
            f"[{self.min_acceptance:.2%}, {self.max_acceptance:.2%}] "
            f"after {max_generation_attempts} attempts"
        )
