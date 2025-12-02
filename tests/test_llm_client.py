"""Tests for LLM client unwrapping logic."""

from unittest.mock import MagicMock

from eleusis.llm_client import RefereeClient


class TestRefereeClientUnwrapping:
    """Test the _unwrap_function_definition method."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock RefereeClient without needing actual API keys
        self.client = MagicMock(spec=RefereeClient)
        self.client._unwrap_function_definition = RefereeClient._unwrap_function_definition.__get__(
            self.client, RefereeClient
        )

    def test_unwrap_full_function_definition(self):
        """Test unwrapping a complete function definition."""
        code = """def is_card_accepted(card, mainline):
    if not mainline:
        return True
    return card.is_even"""

        expected = """if not mainline:
    return True
return card.is_even"""

        result = self.client._unwrap_function_definition(code)
        assert result == expected

    def test_no_unwrap_needed_for_body_only(self):
        """Test that function body without def passes through unchanged."""
        code = """if not mainline:
    return True
return card.is_even"""

        result = self.client._unwrap_function_definition(code)
        assert result == code

    def test_unwrap_with_helper_function_inside(self):
        """Test unwrapping outer function but preserving inner helper."""
        code = """def is_card_accepted(card, mainline):
    def helper(x):
        return x % 2
    if not mainline:
        return True
    return helper(card.rank)"""

        expected = """def helper(x):
    return x % 2
if not mainline:
    return True
return helper(card.rank)"""

        result = self.client._unwrap_function_definition(code)
        assert result == expected

    def test_unwrap_function_with_complex_logic(self):
        """Test unwrapping function with complex nested logic."""
        code = """def is_card_accepted(card, mainline):
    # Helper to check odd rank
    def is_odd_rank(rank):
        return rank % 2 == 1

    if not mainline:
        return True

    last_card = mainline[-1]
    opposite_color = card.color != last_card.color
    return opposite_color and is_odd_rank(card.rank)"""

        expected = """# Helper to check odd rank
def is_odd_rank(rank):
    return rank % 2 == 1

if not mainline:
    return True

last_card = mainline[-1]
opposite_color = card.color != last_card.color
return opposite_color and is_odd_rank(card.rank)"""

        result = self.client._unwrap_function_definition(code)
        assert result == expected

    def test_unwrap_with_leading_whitespace(self):
        """Test unwrapping when there's leading whitespace."""
        code = """
def is_card_accepted(card, mainline):
    if not mainline:
        return True
    return card.is_even
"""

        expected = """if not mainline:
    return True
return card.is_even"""

        result = self.client._unwrap_function_definition(code)
        assert result == expected

    def test_unwrap_simple_helper_at_start(self):
        """Test unwrapping when a simple helper function is at the start."""
        # Even helper functions at the start get unwrapped
        code = """def helper(x):
    return x % 2"""

        expected = """return x % 2"""

        result = self.client._unwrap_function_definition(code)
        assert result == expected
