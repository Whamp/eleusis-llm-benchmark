"""Rule validation and factory for Eleusis."""

import logging
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from typing_extensions import TypedDict

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.game.metrics import CodeComplexity, code_complexity
from eleusis.game.rule_library import (
    RuleLibraryEntry,
    parse_rule_library_entries,
)
from eleusis.llm.base import BaseLLMClient

__all__ = [
    "RuleLibraryEntry",
    "RuleValidator",
    "ValidationResult",
    "parse_rule_library_entries",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationTurnRequest:
    """Rules, cards, and location for one simulated comparison turn."""

    actual_rule: Rule
    guessed_rule: Rule
    mainline: list[Card]
    all_cards: list[Card]
    simulation_number: int
    turn_number: int


@dataclass(frozen=True)
class SimulationTurnResult:
    """Accepted cards and comparison outcome from one simulation turn."""

    accepted_cards: list[Card]
    comparisons: int
    mismatches: int
    mismatch_message: str | None


class RuleComparisonMetadata(TypedDict):
    """Compilation provenance and simulation evidence for one guessed rule."""

    simulation_comparisons: int
    simulation_mismatches: int
    simulation_duration_seconds: float
    guessed_code: str | None
    complexity_metrics: CodeComplexity | None
    compilation_status: str
    compilation_attempts: int
    compilation_cache_hit: bool
    compilation_provider: str | None
    equivalence_cache_hit: bool


ShadowCacheKey = tuple[str, str, int, int, int]
ShadowCacheValue = tuple[bool, str, RuleComparisonMetadata]


@dataclass
class ValidationResult:
    """Result of rule validation."""

    valid: bool
    deterministic: bool
    works_with_empty_mainline: bool
    issues: list[str]


class RuleValidator:
    """Validates rules and compares guessed rules to actual rules."""

    def __init__(self) -> None:
        """Initialize an empty cache for deterministic shadow comparisons."""
        self._shadow_cache: dict[ShadowCacheKey, ShadowCacheValue] = {}

    def clear_shadow_cache(self) -> None:
        """Clear the shadow evaluation cache, forcing re-simulation on next call."""
        self._shadow_cache.clear()

    def snapshot_validator_cache(self) -> list[dict[str, object]]:
        """Capture reusable simulation verdicts without provider-specific state."""
        return [
            {
                "key": {
                    "actual_rule_code": key[0],
                    "guessed_rule_description": key[1],
                    "num_simulations": key[2],
                    "turns_per_simulation": key[3],
                    "simulation_seed": key[4],
                },
                "correct": value[0],
                "reasoning": value[1],
                "metadata": dict(value[2]),
            }
            for key, value in sorted(self._shadow_cache.items())
        ]

    def restore_validator_cache(
        self,
        payloads: Sequence[Mapping[str, object]],
    ) -> None:
        """Restore validated reusable simulation verdicts into this validator."""
        restored: dict[ShadowCacheKey, ShadowCacheValue] = {}
        for payload in payloads:
            key_payload = cast(Mapping[str, object], payload["key"])
            key: ShadowCacheKey = (
                cast(str, key_payload["actual_rule_code"]),
                cast(str, key_payload["guessed_rule_description"]),
                cast(int, key_payload["num_simulations"]),
                cast(int, key_payload["turns_per_simulation"]),
                cast(int, key_payload["simulation_seed"]),
            )
            restored[key] = (
                cast(bool, payload["correct"]),
                cast(str, payload["reasoning"]),
                cast(
                    RuleComparisonMetadata,
                    dict(cast(Mapping[str, object], payload["metadata"])),
                ),
            )
        self._shadow_cache = restored

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
                logger.warning(
                    f"Non-deterministic results for {card} with mainline {mainline}"
                )
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
        except Exception as e:  # ruff: ignore[blind-except]
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
            except Exception as e:  # ruff: ignore[blind-except]
                issues.append(f"Rule evaluation failed on random scenario: {e}")
                break

        return issues

    def compare_rules(
        self,
        actual_rule: Rule,
        guessed_rule_desc: str,
        current_mainline: list[Card],
        rule_compiler_client: BaseLLMClient,
        num_simulations: int = 10,
        turns_per_simulation: int = 20,
        simulation_seed: int = 42,
        compiler_max_retries: int = 2,
    ) -> tuple[bool, str, RuleComparisonMetadata]:
        """Compare rules using simulation-based comparison.

        Results are cached by (actual_rule_code, guessed_rule_desc, num_simulations,
        turns_per_simulation, simulation_seed) so identical shadow evaluations within or
        across rounds are not re-simulated.
        """
        # current_mainline excluded: simulation-based comparison generates
        # independent random card sequences, so the mainline does not affect results.
        cache_key = (
            actual_rule.get_code(),
            guessed_rule_desc,
            num_simulations,
            turns_per_simulation,
            simulation_seed,
        )
        if cache_key in self._shadow_cache:
            logger.debug(
                f"Shadow cache hit for tentative rule: {guessed_rule_desc[:60]}"
            )
            correct, reasoning, cached_metadata = self._shadow_cache[cache_key]
            metadata: RuleComparisonMetadata = {
                **cached_metadata,
                "equivalence_cache_hit": True,
            }
            return correct, reasoning, metadata

        compile_result = rule_compiler_client.convert_rule_to_code(
            guessed_rule_desc,
            max_retries=compiler_max_retries,
        )

        guessed_code = compile_result["code"]
        compilation_status = compile_result["status"]
        compilation_attempts = compile_result["attempts"]
        compilation_cache_hit = compile_result["cache_hit"]
        compilation_provider = compile_result["provider_used"]

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
            f"Rule comparison: {num_simulations} sims x {turns_per_simulation} turns, "
            f"{comparisons} comparisons in {sim_duration:.3f}s"
        )

        # Compute complexity metrics for guessed rule code
        complexity_metrics = code_complexity(guessed_code) if guessed_code else None

        result: ShadowCacheValue = (
            sim_equivalent,
            sim_reasoning,
            {
                "simulation_comparisons": comparisons,
                "simulation_mismatches": mismatches,
                "simulation_duration_seconds": round(sim_duration, 3),
                "guessed_code": guessed_code,
                "complexity_metrics": complexity_metrics,
                "compilation_status": compilation_status,
                "compilation_attempts": compilation_attempts,
                "compilation_cache_hit": compilation_cache_hit,
                "compilation_provider": compilation_provider,
                "equivalence_cache_hit": False,
            },
        )
        self._shadow_cache[cache_key] = result
        return result

    @staticmethod
    def _compare_simulation_turn(
        request: SimulationTurnRequest,
    ) -> SimulationTurnResult:
        """Compare both rules against every card for one simulated turn."""
        accepted_cards: list[Card] = []
        comparisons = 0
        mismatches = 0
        for card in request.all_cards:
            try:
                actual_result = request.actual_rule.evaluate(card, request.mainline)
                guessed_result = request.guessed_rule.evaluate(card, request.mainline)
                comparisons += 1
                if actual_result != guessed_result:
                    message = (
                        f"Mismatch at sim {request.simulation_number + 1}, "
                        f"turn {request.turn_number + 1}: card={card}, "
                        f"actual={actual_result}, guessed={guessed_result}"
                    )
                    logger.info(message)
                    return SimulationTurnResult(
                        accepted_cards, comparisons, mismatches + 1, message
                    )
                if actual_result:
                    accepted_cards.append(card)
            # Generated rule bodies can raise arbitrary exceptions.
            except Exception as error:  # ruff: ignore[blind-except]
                logger.error(f"Evaluation error for {card}: {error}")
                comparisons += 1
                mismatches += 1
        return SimulationTurnResult(accepted_cards, comparisons, mismatches, None)

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
        except Exception as e:  # ruff: ignore[blind-except]
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
                turn_result = self._compare_simulation_turn(
                    SimulationTurnRequest(
                        actual_rule=actual_rule,
                        guessed_rule=guessed_rule,
                        mainline=simulated_mainline,
                        all_cards=all_cards,
                        simulation_number=sim_num,
                        turn_number=turn_num,
                    )
                )
                total_comparisons += turn_result.comparisons
                mismatches += turn_result.mismatches
                if turn_result.mismatch_message:
                    return (
                        False,
                        turn_result.mismatch_message,
                        total_comparisons,
                        mismatches,
                    )
                if mismatches > 0:
                    reasoning = (
                        f"Rules differ: {mismatches}/{total_comparisons} comparisons"
                        f" mismatched. Stopped at simulation {sim_num + 1}, turn"
                        f" {turn_num + 1}."
                    )
                    return False, reasoning, total_comparisons, mismatches

                if not turn_result.accepted_cards:
                    logger.debug(
                        f"No cards accepted at sim {sim_num + 1}, turn {turn_num + 1}, "
                        "ending simulation"
                    )
                    break

                chosen_card = rng.choice(turn_result.accepted_cards)
                simulated_mainline.append(chosen_card)

        reasoning = (
            f"Rules appear equivalent: {total_comparisons} comparisons, all matched"
        )
        return True, reasoning, total_comparisons, mismatches
