"""Rule system for Eleusis: LLM-generated rules and validation."""

import logging
import random
from dataclasses import dataclass

from eleusis.cards import Card, Suit
from eleusis.game_engine import Rule
from eleusis.llm_client import HuggingFaceClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of rule validation."""

    valid: bool
    deterministic: bool
    works_with_empty_mainline: bool
    issues: list[str]


class RuleValidator:
    """Validates LLM-generated rules."""

    def __init__(self, referee_client: HuggingFaceClient | None = None) -> None:
        """Initialize validator with optional referee client."""
        self.referee_client = referee_client

    def validate_rule(self, rule: Rule, num_test_cases: int = 3) -> ValidationResult:
        """Validate that a rule meets requirements.

        Works with any Rule implementation (LLMGeneratedRule, PythonRule, etc.).
        """
        issues = []

        # Test 1: Determinism - same input should give same output
        deterministic = self._test_determinism(rule)
        if not deterministic:
            issues.append("Rule is not deterministic")

        # Test 2: Empty mainline - rule must work with first card
        works_with_empty = self._test_empty_mainline(rule)
        if not works_with_empty:
            issues.append("Rule does not work with empty mainline")

        # Test 3: Run various test scenarios
        test_issues = self._run_test_scenarios(rule, num_test_cases)
        issues.extend(test_issues)

        valid = len(issues) == 0

        return ValidationResult(
            valid=valid,
            deterministic=deterministic,
            works_with_empty_mainline=works_with_empty,
            issues=issues,
        )

    def _test_determinism(self, rule: Rule, num_tests: int = 3) -> bool:
        """Test if rule gives consistent results."""
        # Test a few random scenarios multiple times
        test_scenarios = [
            (Card(5, Suit.HEARTS), []),
            (Card(8, Suit.SPADES), [Card(5, Suit.HEARTS)]),
            (Card(2, Suit.DIAMONDS), [Card(5, Suit.HEARTS), Card(8, Suit.SPADES)]),
        ]

        for card, mainline in test_scenarios:
            # Evaluate same scenario multiple times
            results = []
            for _ in range(num_tests):
                result = rule.evaluate(card, mainline)
                results.append(result)

            # Check if all results are the same
            if len(set(results)) > 1:
                logger.warning(f"Non-deterministic results for {card} with mainline {mainline}")
                return False

        return True

    def _test_empty_mainline(self, rule: Rule) -> bool:
        """Test if rule works with empty mainline."""
        try:
            # Try evaluating a few cards with empty mainline
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

        # Generate random scenarios
        for _ in range(num_tests):
            # Random mainline length 0-5
            mainline_length = random.randint(0, 5)
            mainline = []

            for _ in range(mainline_length):
                card = Card(random.randint(1, 13), random.choice(list(Suit)))
                mainline.append(card)

            # Random test card
            test_card = Card(random.randint(1, 13), random.choice(list(Suit)))

            try:
                rule.evaluate(test_card, mainline)
            except Exception as e:
                issues.append(f"Rule evaluation failed on random scenario: {e}")
                break

        return issues

    def check_equivalence(
        self, rule1_text: str, rule2_text: str, mainline_text: str
    ) -> tuple[bool, str]:
        """Check if two rules are equivalent using referee LLM."""
        if not self.referee_client:
            raise ValueError("Referee client not configured")

        return self.referee_client.check_rule_equivalence(rule1_text, rule2_text, mainline_text)

    def compare_rules(
        self,
        actual_rule,
        guessed_rule_desc: str,
        current_mainline: list[Card],
        game_master,
        num_simulations: int = 2,
        turns_per_simulation: int = 10,
    ) -> tuple[bool, str, dict]:
        """Compare rules using simulation (authoritative) and optionally LLM (debugging)."""
        # Step 1: Convert guessed rule to code
        guessed_code = game_master.convert_rule_to_code(guessed_rule_desc)

        # Step 2: Run simulation comparison (authoritative)
        sim_equivalent, sim_reasoning, comparisons, mismatches = (
            self.check_equivalence_by_simulation(
                actual_rule,
                guessed_rule_desc,
                current_mainline,
                num_simulations,
                turns_per_simulation,
                guessed_code,
            )
        )

        # Step 3: Optionally run LLM comparison (debugging only)
        llm_equivalent, llm_reasoning = game_master.compare_rules_with_llm(
            actual_rule.description(),
            guessed_rule_desc,
            " ".join([str(c) for c in current_mainline]),
        )

        # Step 4: Return simulation verdict with metadata
        return sim_equivalent, sim_reasoning, {
            "llm_verdict": llm_equivalent,
            "llm_reasoning": llm_reasoning,
            "simulation_comparisons": comparisons,
            "simulation_mismatches": mismatches,
        }

    def check_equivalence_by_simulation(
        self,
        actual_rule: Rule,
        guessed_rule_text: str,
        current_mainline: list[Card],
        num_simulations: int = 2,
        turns_per_simulation: int = 10,
        preconverted_code: str | None = None,
    ) -> tuple[bool, str, int, int]:
        """Check if two rules are equivalent by simulating gameplay.
        """
        # Use preconverted code if provided, otherwise convert now
        if preconverted_code:
            logger.debug("Using preconverted guessed rule code")
            guessed_code = preconverted_code
        else:
            if not self.referee_client:
                raise ValueError("Referee client not configured for code conversion")

            logger.debug(f"Converting guessed rule to Python code: {guessed_rule_text}")
            guessed_code = self.referee_client.convert_rule_to_code(guessed_rule_text)

        if not guessed_code:
            return False, "Failed to convert guessed rule to Python code", 0, 0

        # Create Rule from guessed code
        try:
            guessed_rule = Rule(guessed_rule_text, guessed_code)
        except Exception as e:
            logger.error(f"Failed to create Rule from guessed code: {e}")
            return False, f"Guessed rule code has syntax errors: {e}", 0, 0

        # Generate all 52 cards
        all_cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]

        total_comparisons = 0
        mismatches = 0

        # Run N simulations
        for sim_num in range(num_simulations):
            # Start with current mainline
            simulated_mainline = list(current_mainline)

            # Run K turns
            for turn_num in range(turns_per_simulation):
                # Try all 52 cards
                accepted_cards_actual = []
                accepted_cards_guessed = []

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
                            logger.debug(f"  simulated_mainline={simulated_mainline}")
                            mismatch_msg = (
                                f"Mismatch at sim {sim_num+1}, turn {turn_num+1}: "
                                f"card={card}, actual={actual_result}, guessed={guessed_result}"
                            )
                            return False, mismatch_msg, total_comparisons, mismatches

                        if actual_result:
                            accepted_cards_actual.append(card)
                        if guessed_result:
                            accepted_cards_guessed.append(card)

                    except Exception as e:
                        logger.error(f"Evaluation error for {card}: {e}")
                        mismatches += 1
                        total_comparisons += 1

                # If there are mismatches, rules are not equivalent
                if mismatches > 0:
                    reasoning = (
                        f"Rules differ: {mismatches}/{total_comparisons} comparisons mismatched. "
                        f"Stopped at simulation {sim_num+1}, turn {turn_num+1}."
                    )
                    return False, reasoning, total_comparisons, mismatches

                # Pick a random accepted card from actual rule to add to mainline
                # If no cards accepted, stop simulation
                if not accepted_cards_actual:
                    logger.debug(
                        f"No cards accepted at sim {sim_num+1}, turn {turn_num+1}, "
                        f"ending simulation"
                    )
                    break

                chosen_card = random.choice(accepted_cards_actual)
                simulated_mainline.append(chosen_card)

        # If we got here, no mismatches found
        if mismatches == 0:
            reasoning = f"Rules appear equivalent: {total_comparisons} comparisons, all matched"
            return True, reasoning, total_comparisons, mismatches
        else:
            reasoning = f"Rules differ: {mismatches}/{total_comparisons} comparisons mismatched"
            return False, reasoning, total_comparisons, mismatches