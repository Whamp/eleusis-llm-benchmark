"""Tests for shadow evaluation caching.

Verifies:
- Identical (actual_rule, tentative_rule, sim_budget) shadow evals are simulated once
- Cache key changes when simulation budget changes
- Different tentative rules are evaluated independently
"""

from __future__ import annotations

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.game.validator import RuleValidator
from eleusis.llm.base import BaseLLMClient, RuleCompileResult
from tests.conftest import FakeLLMClient


class FakeShadowCompiler(FakeLLMClient):
    """Fake compiler that tracks convert_rule_to_code calls."""

    def __init__(self, code_map: dict[str, str]) -> None:
        """code_map: rule_text -> code to return."""
        super().__init__()
        self._code_map = code_map
        self.compile_calls: list[str] = []

    def convert_rule_to_code(
        self,
        rule_text: str,
        max_retries: int = 1,
        fallback_clients: list[BaseLLMClient] | None = None,
        max_total_attempts: int = 5,
    ) -> RuleCompileResult:
        """Return the configured code while recording each compile request."""
        del max_retries, fallback_clients, max_total_attempts
        self.compile_calls.append(rule_text)
        code = self._code_map.get(rule_text)
        if code is None:
            return {
                "code": None,
                "status": "exhausted",
                "attempts": 1,
                "sleep_cycles": 0,
                "provider_used": None,
                "cache_hit": False,
            }
        return {
            "code": code,
            "status": "success",
            "attempts": 1,
            "sleep_cycles": 0,
            "provider_used": "fake/fake-model",
            "cache_hit": False,
        }


class TestShadowEvaluationCache:
    """Shadow evaluation cache deduplicates identical evaluations."""

    def _make_validator(self) -> RuleValidator:
        return RuleValidator()

    def _even_rule(self) -> Rule:
        return Rule("Only even ranks", "return card.rank % 2 == 0")

    def _mainline(self) -> list[Card]:
        return [Card(4, Suit.HEARTS)]

    def test_identical_shadow_evals_simulated_once(self) -> None:
        """Same (actual_rule, tentative_rule, sim_budget) should simulate only once."""
        validator = self._make_validator()
        actual_rule = self._even_rule()
        compiler = FakeShadowCompiler({"Only even ranks": "return card.rank % 2 == 0"})
        mainline = self._mainline()

        result1 = validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )
        result2 = validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        # Both should return same result
        assert result1[0] == result2[0]  # is_correct
        # Compiler should only be called once (cached on second call)
        assert len(compiler.compile_calls) == 1

    def test_cache_key_changes_with_sim_budget(self) -> None:
        """Different simulation budgets should be evaluated independently."""
        validator = self._make_validator()
        actual_rule = self._even_rule()
        compiler = FakeShadowCompiler({"Only even ranks": "return card.rank % 2 == 0"})
        mainline = self._mainline()

        validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )
        validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=10,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        # Different sim budgets -> both compiled & simulated
        assert len(compiler.compile_calls) == 2

    def test_different_tentative_rules_evaluated_independently(self) -> None:
        """Different tentative rule texts should not share cache entries."""
        validator = self._make_validator()
        actual_rule = self._even_rule()
        compiler = FakeShadowCompiler(
            {
                "Only even ranks": "return card.rank % 2 == 0",
                "Only red cards": 'return card.color == "red"',
            }
        )
        mainline = self._mainline()

        validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )
        validator.compare_rules(
            actual_rule,
            "Only red cards",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        assert len(compiler.compile_calls) == 2

    def test_cached_failure_reused(self) -> None:
        """Failed compilations should also be cached."""
        validator = self._make_validator()
        actual_rule = self._even_rule()
        # Compiler will fail for this rule text
        compiler = FakeShadowCompiler({})
        mainline = self._mainline()

        result1 = validator.compare_rules(
            actual_rule,
            "Unknown rule",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )
        result2 = validator.compare_rules(
            actual_rule,
            "Unknown rule",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        assert result1[0] == result2[0]  # both fail
        assert result1[0] is False
        assert len(compiler.compile_calls) == 1  # only compiled once

    def test_clear_shadow_cache(self) -> None:
        """clear_shadow_cache() should force re-evaluation."""
        validator = self._make_validator()
        actual_rule = self._even_rule()
        compiler = FakeShadowCompiler({"Only even ranks": "return card.rank % 2 == 0"})
        mainline = self._mainline()

        validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )
        validator.clear_shadow_cache()
        validator.compare_rules(
            actual_rule,
            "Only even ranks",
            mainline,
            compiler,
            num_simulations=5,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        assert len(compiler.compile_calls) == 2
