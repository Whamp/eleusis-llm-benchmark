"""Rule validation and factory for Eleusis."""

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.game.metrics import code_complexity

__all__ = ["ValidationResult", "RuleValidator", "RuleFactory"]

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of rule validation."""
    valid: bool
    deterministic: bool
    works_with_empty_mainline: bool
    issues: list[str]


class RuleValidator:
    """Validates rules and compares guessed rules to actual rules."""

    def __init__(self):
        self._shadow_cache: dict[tuple, tuple[bool, str, dict]] = {}

    def clear_shadow_cache(self) -> None:
        """Clear the shadow evaluation cache, forcing re-simulation on next call."""
        self._shadow_cache.clear()

    def validate_rule(self, rule: Rule, num_test_cases: int = 3) -> ValidationResult:
        """Validate that a rule meets requirements."""
        issues = []

        deterministic = self._test_determinism(rule)
        if not deterministic:
            issues.append("Rule is not deterministic")

        works_with_empty = self._test_empty_mainline(rule)
        if not works_with_empty:
            issues.append("Rule does not work with empty mainline")

        test_issues = self._run_test_scenarios(rule, num_test_cases)
        issues.extend(test_issues)

        return ValidationResult(
            valid=len(issues) == 0,
            deterministic=deterministic,
            works_with_empty_mainline=works_with_empty,
            issues=issues,
        )

    def _test_determinism(self, rule: Rule, num_tests: int = 3) -> bool:
        """Test if rule gives consistent results."""
        test_scenarios = [
            (Card(5, Suit.HEARTS), []),
            (Card(8, Suit.SPADES), [Card(5, Suit.HEARTS)]),
            (Card(2, Suit.DIAMONDS), [Card(5, Suit.HEARTS), Card(8, Suit.SPADES)]),
        ]

        for card, mainline in test_scenarios:
            results = []
            for _ in range(num_tests):
                result = rule.evaluate(card, mainline)
                results.append(result)

            if len(set(results)) > 1:
                logger.warning(f"Non-deterministic results for {card} with mainline {mainline}")
                return False

        return True

    def _test_empty_mainline(self, rule: Rule) -> bool:
        """Test if rule works with empty mainline."""
        try:
            test_cards = [
                Card(1, Suit.HEARTS),
                Card(7, Suit.SPADES),
                Card(13, Suit.DIAMONDS),
            ]
            for card in test_cards:
                rule.evaluate(card, [])
            return True
        except Exception as e:
            logger.error(f"Rule failed with empty mainline: {e}")
            return False

    def _run_test_scenarios(self, rule: Rule, num_tests: int) -> list[str]:
        """Run random test scenarios to check rule behavior."""
        issues = []

        for _ in range(num_tests):
            mainline_length = random.randint(0, 5)
            mainline = []
            for _ in range(mainline_length):
                card = Card(random.randint(1, 13), random.choice(list(Suit)))
                mainline.append(card)

            test_card = Card(random.randint(1, 13), random.choice(list(Suit)))

            try:
                rule.evaluate(test_card, mainline)
            except Exception as e:
                issues.append(f"Rule evaluation failed on random scenario: {e}")
                break

        return issues

    def compare_rules(
        self,
        actual_rule,
        guessed_rule_desc: str,
        current_mainline: list[Card],
        rule_compiler_client,
        num_simulations: int = 10,
        turns_per_simulation: int = 20,
        simulation_seed: int = 42,
        compiler_max_retries: int = 2,
    ) -> tuple[bool, str, dict]:
        """Compare rules using simulation-based comparison.

        Results are cached by (actual_rule_code, guessed_rule_desc,
        num_simulations, turns_per_simulation, simulation_seed) so identical
        shadow evaluations within or across rounds are not re-simulated.
        """
        cache_key = (
            actual_rule.get_code(),
            guessed_rule_desc,
            num_simulations,
            turns_per_simulation,
            simulation_seed,
        )
        if cache_key in self._shadow_cache:
            logger.debug(f"Shadow cache hit for tentative rule: {guessed_rule_desc[:60]}")
            return self._shadow_cache[cache_key]

        compile_result = rule_compiler_client.convert_rule_to_code(
            guessed_rule_desc,
            max_retries=compiler_max_retries,
        )

        guessed_code = compile_result["code"]
        compilation_status = compile_result["status"]
        compilation_attempts = compile_result["attempts"]

        sim_start = time.perf_counter()
        sim_equivalent, sim_reasoning, comparisons, mismatches = (
            self.check_equivalence_by_simulation(
                actual_rule,
                guessed_rule_desc,
                current_mainline,
                num_simulations,
                turns_per_simulation,
                guessed_code,
                simulation_seed,
            )
        )
        sim_duration = time.perf_counter() - sim_start
        logger.info(
            f"Rule comparison: {num_simulations} sims × {turns_per_simulation} turns, "
            f"{comparisons} comparisons in {sim_duration:.3f}s"
        )

        # Compute complexity metrics for guessed rule code
        complexity_metrics = code_complexity(guessed_code) if guessed_code else None

        result = (sim_equivalent, sim_reasoning, {
            "simulation_comparisons": comparisons,
            "simulation_mismatches": mismatches,
            "simulation_duration_seconds": round(sim_duration, 3),
            "guessed_code": guessed_code,
            "complexity_metrics": complexity_metrics,
            "compilation_status": compilation_status,
            "compilation_attempts": compilation_attempts,
        })
        self._shadow_cache[cache_key] = result
        return result

    def check_equivalence_by_simulation(
        self,
        actual_rule: Rule,
        guessed_rule_text: str,
        current_mainline: list[Card],
        num_simulations: int = 10,
        turns_per_simulation: int = 20,
        preconverted_code: str | None = None,
        simulation_seed: int = 42,
    ) -> tuple[bool, str, int, int]:
        """Check if two rules are equivalent by simulating gameplay."""
        if not preconverted_code:
            return False, "No code provided for guessed rule", 0, 0

        try:
            guessed_rule = Rule(guessed_rule_text, preconverted_code)
        except Exception as e:
            logger.error(f"Failed to create Rule from guessed code: {e}")
            return False, f"Guessed rule code has syntax errors: {e}", 0, 0

        # Use seeded RNG for reproducible simulations
        rng = random.Random(simulation_seed)

        all_cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]

        total_comparisons = 0
        mismatches = 0

        for sim_num in range(num_simulations):
            simulated_mainline = list(current_mainline)

            for turn_num in range(turns_per_simulation):
                accepted_cards_actual = []

                for card in all_cards:
                    try:
                        actual_result = actual_rule.evaluate(card, simulated_mainline)
                        guessed_result = guessed_rule.evaluate(card, simulated_mainline)

                        total_comparisons += 1

                        if actual_result != guessed_result:
                            mismatches += 1
                            logger.info(
                                f"Mismatch at sim {sim_num+1}, turn {turn_num+1}: "
                                f"card={card}, actual={actual_result}, guessed={guessed_result}"
                            )
                            mismatch_msg = (
                                f"Mismatch at sim {sim_num+1}, turn {turn_num+1}: "
                                f"card={card}, actual={actual_result}, guessed={guessed_result}"
                            )
                            return False, mismatch_msg, total_comparisons, mismatches

                        if actual_result:
                            accepted_cards_actual.append(card)

                    except Exception as e:
                        logger.error(f"Evaluation error for {card}: {e}")
                        mismatches += 1
                        total_comparisons += 1

                if mismatches > 0:
                    reasoning = (
                        f"Rules differ: {mismatches}/{total_comparisons} comparisons mismatched. "
                        f"Stopped at simulation {sim_num+1}, turn {turn_num+1}."
                    )
                    return False, reasoning, total_comparisons, mismatches

                if not accepted_cards_actual:
                    logger.debug(
                        f"No cards accepted at sim {sim_num+1}, turn {turn_num+1}, "
                        f"ending simulation"
                    )
                    break

                chosen_card = rng.choice(accepted_cards_actual)
                simulated_mainline.append(chosen_card)

        if mismatches == 0:
            reasoning = f"Rules appear equivalent: {total_comparisons} comparisons, all matched"
            return True, reasoning, total_comparisons, mismatches
        else:
            reasoning = f"Rules differ: {mismatches}/{total_comparisons} comparisons mismatched"
            return False, reasoning, total_comparisons, mismatches


class RuleFactory:
    """Factory for creating rules from pre-generated library."""

    def __init__(
        self,
        library_path: str | None = None,
        selection: str = "random",
        start_index: int = 0,
        rules_list: list[dict] | None = None,
    ) -> None:
        """Initialize rule factory."""
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
        """Create rule and return with full metadata from library."""
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
