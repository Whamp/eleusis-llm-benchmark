"""Tests for named benchmark suite resolution."""

from pathlib import Path
from typing import ClassVar

import pytest

# The module we're testing — will be created in the GREEN phase
from eleusis.suites import load_suites, resolve_suite

SUITES_PATH = Path(__file__).parent.parent / "suites.yaml"


class TestSuiteDefinitions:
    """Verify suites.yaml contains correct suite definitions."""

    def test_suites_yaml_exists(self) -> None:
        """Verify suites yaml exists."""
        assert SUITES_PATH.exists(), "suites.yaml must exist at project root"

    def test_suites_yaml_has_three_suites(self) -> None:
        """Verify suites yaml has three suites."""
        suites = load_suites(SUITES_PATH)
        assert set(suites.keys()) == {"full_26x3", "screen_26x1", "stress_12x1"}


class TestFullSuite:
    """full_26x3: all 26 rules x 3 batch indices = 78 cases."""

    def test_full_expands_to_78_cases(self) -> None:
        """Verify full expands to 78 cases."""
        cases = resolve_suite("full_26x3", suites_path=SUITES_PATH)
        assert len(cases) == 78

    def test_full_uses_all_26_rules(self) -> None:
        """Verify full uses all 26 rules."""
        cases = resolve_suite("full_26x3", suites_path=SUITES_PATH)
        rule_names = list(
            dict.fromkeys(name for name, _ in cases)
        )  # preserve order, dedupe
        assert len(rule_names) == 26

    def test_full_uses_batch_indices_0_1_2(self) -> None:
        """Verify full uses batch indices 0 1 2."""
        cases = resolve_suite("full_26x3", suites_path=SUITES_PATH)
        # Each rule should appear with batch indices 0, 1, 2
        from collections import Counter

        rule_counts = Counter(name for name, _ in cases)
        for rule, count in rule_counts.items():
            assert count == 3, f"Rule {rule} should appear 3 times, got {count}"
        batch_indices = {idx for _, idx in cases}
        assert batch_indices == {0, 1, 2}

    def test_full_rules_in_library_order(self) -> None:
        """Rules should appear in the same order as rules.json."""
        import json

        rules_json = Path(__file__).parent.parent / "rules.json"
        with open(rules_json) as f:
            library_names = [r["name"] for r in json.load(f)["rules"]]

        cases = resolve_suite("full_26x3", suites_path=SUITES_PATH)
        # Extract unique rule names preserving first-appearance order
        seen = set()
        suite_order = []
        for name, _ in cases:
            if name not in seen:
                seen.add(name)
                suite_order.append(name)

        assert suite_order == library_names


class TestScreenSuite:
    """screen_26x1: all 26 rules x batch index [1] = 26 cases."""

    def test_screen_expands_to_26_cases(self) -> None:
        """Verify screen expands to 26 cases."""
        cases = resolve_suite("screen_26x1", suites_path=SUITES_PATH)
        assert len(cases) == 26

    def test_screen_uses_batch_index_1(self) -> None:
        """Verify screen uses batch index 1."""
        cases = resolve_suite("screen_26x1", suites_path=SUITES_PATH)
        batch_indices = {idx for _, idx in cases}
        assert batch_indices == {1}


class TestStressSuite:
    """stress_12x1: exactly 12 named rules x batch index [1] = 12 cases."""

    EXPECTED_RULES: ClassVar[tuple[str, ...]] = (
        "only_aces",
        "face_cards_only",
        "different_suit",
        "prime_ranks_only",
        "no_spades",
        "ranks_one_to_seven",
        "red_up_black_down",
        "paired_ranks_distinct",
        "alternating_groups",
        "face_card_imposes_suit",
        "face_cards_red_number_cards_black",
        "rank_non_decreasing_start_ace",
    )

    def test_stress_expands_to_12_cases(self) -> None:
        """Verify stress expands to 12 cases."""
        cases = resolve_suite("stress_12x1", suites_path=SUITES_PATH)
        assert len(cases) == 12

    def test_stress_uses_exact_rules(self) -> None:
        """Verify stress uses exact rules."""
        cases = resolve_suite("stress_12x1", suites_path=SUITES_PATH)
        rule_names = [name for name, _ in cases]
        assert rule_names == list(self.EXPECTED_RULES)

    def test_stress_uses_batch_index_1(self) -> None:
        """Verify stress uses batch index 1."""
        cases = resolve_suite("stress_12x1", suites_path=SUITES_PATH)
        batch_indices = {idx for _, idx in cases}
        assert batch_indices == {1}


class TestSuiteErrors:
    """Error handling for unknown suite names."""

    def test_unknown_suite_raises_error(self) -> None:
        """Verify unknown suite raises error."""
        with pytest.raises(ValueError, match="Unknown suite"):
            resolve_suite("nonexistent_suite", suites_path=SUITES_PATH)
