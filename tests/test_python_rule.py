"""Tests for Python-based rule implementation."""

import pytest

from eleusis.cards import Card, Suit
from eleusis.python_rule import PythonRule


class TestPythonRule:
    """Tests for PythonRule class."""

    def test_valid_code_execution(self) -> None:
        """Test that valid Python code executes correctly."""
        code = """
if not mainline:
    return True
last_card = mainline[-1]
return card.color != last_card.color
"""
        rule = PythonRule("Alternating colors", code)

        # First card should be accepted
        card1 = Card(5, Suit.HEARTS)  # Red
        assert rule.evaluate(card1, [])

        # Opposite color should be accepted
        card2 = Card(8, Suit.SPADES)  # Black
        assert rule.evaluate(card2, [card1])

        # Same color should be rejected
        card3 = Card(3, Suit.DIAMONDS)  # Red
        assert not rule.evaluate(card3, [card1])

    def test_even_ranks_rule(self) -> None:
        """Test rule that only accepts even ranks."""
        code = "return card.is_even"
        rule = PythonRule("Even ranks only", code)

        assert rule.evaluate(Card(2, Suit.HEARTS), [])
        assert rule.evaluate(Card(10, Suit.CLUBS), [])
        assert not rule.evaluate(Card(5, Suit.DIAMONDS), [])
        assert not rule.evaluate(Card(13, Suit.SPADES), [])  # King

    def test_syntax_error_handling(self) -> None:
        """Test that syntax errors are caught and handled."""
        code = "if not mainline\n    return True"  # Missing colon
        rule = PythonRule("Broken rule", code)

        # Rule should reject all cards when compilation fails
        assert not rule.evaluate(Card(5, Suit.HEARTS), [])
        assert not rule.evaluate(Card(8, Suit.SPADES), [Card(5, Suit.HEARTS)])

    def test_runtime_error_handling(self) -> None:
        """Test that runtime errors are caught and handled."""
        code = """
if not mainline:
    return True
# This will cause IndexError if mainline is actually empty somehow
return card.rank > mainline[10].rank
"""
        rule = PythonRule("Risky rule", code)

        # First card should work
        assert rule.evaluate(Card(5, Suit.HEARTS), [])

        # Should handle runtime error gracefully
        mainline = [Card(3, Suit.CLUBS)]
        assert not rule.evaluate(Card(8, Suit.SPADES), mainline)

    def test_empty_mainline_handling(self) -> None:
        """Test rules correctly handle empty mainline."""
        code = """
if not mainline:
    return card.rank <= 7
last_card = mainline[-1]
return card.rank > last_card.rank
"""
        rule = PythonRule("First card rank <= 7, then increasing", code)

        # First card with rank <= 7 should be accepted
        assert rule.evaluate(Card(5, Suit.HEARTS), [])
        assert rule.evaluate(Card(7, Suit.CLUBS), [])

        # First card with rank > 7 should be rejected
        assert not rule.evaluate(Card(8, Suit.DIAMONDS), [])
        assert not rule.evaluate(Card(13, Suit.SPADES), [])

        # After first card, increasing ranks
        mainline = [Card(5, Suit.HEARTS)]
        assert rule.evaluate(Card(8, Suit.CLUBS), mainline)
        assert not rule.evaluate(Card(3, Suit.DIAMONDS), mainline)

    def test_safe_globals_no_imports(self) -> None:
        """Test that imports are not allowed."""
        code = """
import os
return True
"""
        rule = PythonRule("Malicious import", code)

        # Should fail at compilation due to NameError
        assert not rule.evaluate(Card(5, Suit.HEARTS), [])

    def test_safe_globals_no_open(self) -> None:
        """Test that file operations are not allowed."""
        code = """
with open('test.txt', 'w') as f:
    f.write('test')
return True
"""
        rule = PythonRule("Malicious file I/O", code)

        # Should reject due to NameError (open not defined)
        assert not rule.evaluate(Card(5, Suit.HEARTS), [])

    def test_safe_globals_no_eval(self) -> None:
        """Test that eval/exec are not allowed."""
        code = """
eval('malicious code')
return True
"""
        rule = PythonRule("Malicious eval", code)

        # Should reject due to NameError (eval not defined)
        assert not rule.evaluate(Card(5, Suit.HEARTS), [])

    def test_allowed_builtins(self) -> None:
        """Test that allowed builtins work correctly."""
        code = """
if not mainline:
    return True
# Use allowed builtins
total_ranks = sum(c.rank for c in mainline)
return total_ranks % 2 == 0
"""
        rule = PythonRule("Sum of ranks is even", code)

        # Empty mainline
        assert rule.evaluate(Card(5, Suit.HEARTS), [])

        # After one card (rank 5), sum=5 (odd), should reject
        mainline = [Card(5, Suit.HEARTS)]
        assert not rule.evaluate(Card(3, Suit.CLUBS), mainline)

        # After two cards (ranks 5,3), sum=8 (even), should accept
        mainline = [Card(5, Suit.HEARTS), Card(3, Suit.CLUBS)]
        assert rule.evaluate(Card(2, Suit.DIAMONDS), mainline)

    def test_complex_rule_with_any_all(self) -> None:
        """Test rule using any() and all() builtins."""
        code = """
if not mainline:
    return True
# Accept if any previous card was even
return any(c.is_even for c in mainline)
"""
        rule = PythonRule("Any previous card was even", code)

        # First card
        assert rule.evaluate(Card(5, Suit.HEARTS), [])

        # After odd card, should reject
        mainline = [Card(5, Suit.HEARTS)]
        assert not rule.evaluate(Card(3, Suit.CLUBS), mainline)

        # After even card, should accept
        mainline = [Card(4, Suit.DIAMONDS)]
        assert rule.evaluate(Card(7, Suit.SPADES), mainline)

    def test_get_code_method(self) -> None:
        """Test that get_code() returns the original code."""
        code = "return card.is_even"
        rule = PythonRule("Even ranks", code)

        assert rule.get_code() == code

    def test_description_method(self) -> None:
        """Test that description() returns the rule description."""
        description = "Only accept even ranked cards"
        code = "return card.is_even"
        rule = PythonRule(description, code)

        assert rule.description() == description
