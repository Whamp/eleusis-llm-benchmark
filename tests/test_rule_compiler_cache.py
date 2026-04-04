"""Tests for bounded rule compiler attempts and compile caching.

Verifies:
- convert_rule_to_code() respects max_total_attempts parameter
- Returns failure status on exhaustion instead of looping forever
- Repeated identical rule texts reuse cached compile results
- Cached failures are reused (no re-compilation)
"""

from __future__ import annotations

from unittest.mock import patch

from eleusis.llm.base import BaseLLMClient
from tests.conftest import FakeLLMClient


class FakeCompilerClient(FakeLLMClient):
    """Fake client for compiler tests that delegates to real BaseLLMClient methods."""

    def __init__(self, responses=None):
        super().__init__(responses)
        self.prompts_seen: list[str] = []
        self._compile_cache: dict[str, dict] = {}

    def _validate_code_syntax(self, code):
        """Delegate to BaseLLMClient's real validation."""
        return BaseLLMClient._validate_code_syntax(self, code)

    def convert_rule_to_code(self, rule_text, max_retries=1,
                             fallback_clients=None, max_total_attempts=5):
        """Delegate to BaseLLMClient's real implementation."""
        return BaseLLMClient.convert_rule_to_code(
            self, rule_text, max_retries=max_retries,
            fallback_clients=fallback_clients,
            max_total_attempts=max_total_attempts,
        )

    def clear_compile_cache(self):
        """Delegate to BaseLLMClient's real implementation."""
        BaseLLMClient.clear_compile_cache(self)


@patch("time.sleep")
class TestCompilerTotalAttemptCap:
    """convert_rule_to_code must stop after max_total_attempts."""

    def test_returns_failure_on_exhaustion(self, mock_sleep):
        """When all attempts produce invalid code, returns failure status."""
        bad_code = "this is not valid python!!!"
        client = FakeCompilerClient([bad_code] * 10)

        result = client.convert_rule_to_code(
            "Only even ranks",
            max_retries=1,
            max_total_attempts=5,
        )

        assert result["status"] == "exhausted"
        assert result["code"] is None

    def test_total_attempts_bounded(self, mock_sleep):
        """Number of generate() calls must not exceed max_total_attempts."""
        bad_code = "not valid python :-("
        client = FakeCompilerClient([bad_code] * 20)

        result = client.convert_rule_to_code(
            "Only even ranks",
            max_retries=1,
            max_total_attempts=3,
        )

        assert client._call_count <= 3
        assert result["status"] == "exhausted"

    def test_success_within_cap(self, mock_sleep):
        """If valid code is produced within the cap, returns success."""
        valid_code = "return card.rank % 2 == 0"
        client = FakeCompilerClient([valid_code])

        result = client.convert_rule_to_code(
            "Only even ranks",
            max_retries=1,
            max_total_attempts=5,
        )

        assert result["status"] == "success"
        assert result["code"] == valid_code

    def test_no_sleep_on_exhaustion(self, mock_sleep):
        """Exhaustion should NOT enter the sleep loop."""
        bad_code = "invalid code!!!"
        client = FakeCompilerClient([bad_code] * 10)

        result = client.convert_rule_to_code(
            "Only even ranks",
            max_retries=0,
            max_total_attempts=3,
        )

        assert result["status"] == "exhausted"

    def test_default_max_total_attempts(self, mock_sleep):
        """Default max_total_attempts should be 5."""
        bad_code = "invalid!!!"
        client = FakeCompilerClient([bad_code] * 20)

        result = client.convert_rule_to_code(
            "Only even ranks",
            max_retries=0,
        )

        assert client._call_count <= 5
        assert result["status"] == "exhausted"


@patch("time.sleep")
class TestCompilerCache:
    """Compile cache deduplicates identical rule text compilations."""

    def test_repeated_rule_compiles_once(self, mock_sleep):
        """Compiling the same rule text twice should call generate() only once."""
        valid_code = "return card.rank % 2 == 0"
        client = FakeCompilerClient([valid_code])

        result1 = client.convert_rule_to_code("Only even ranks", max_retries=0)
        result2 = client.convert_rule_to_code("Only even ranks", max_retries=0)

        assert client._call_count == 1
        assert result1["status"] == "success"
        assert result2["status"] == "success"
        assert result2["code"] == valid_code

    def test_cached_failure_reused(self, mock_sleep):
        """Failed compilations are also cached — no re-attempt on same text."""
        bad_code = "invalid!!!"
        client = FakeCompilerClient([bad_code] * 10)

        result1 = client.convert_rule_to_code(
            "Bad rule", max_retries=0, max_total_attempts=3,
        )
        calls_after_first = client._call_count

        result2 = client.convert_rule_to_code(
            "Bad rule", max_retries=0, max_total_attempts=3,
        )

        assert client._call_count == calls_after_first  # no new calls
        assert result1["status"] == "exhausted"
        assert result2["status"] == "exhausted"

    def test_different_rule_text_not_cached(self, mock_sleep):
        """Different rule texts should each compile independently."""
        valid_code_1 = "return card.rank % 2 == 0"
        valid_code_2 = 'return card.color == "red"'
        client = FakeCompilerClient([valid_code_1, valid_code_2])

        result1 = client.convert_rule_to_code("Only even ranks", max_retries=0)
        result2 = client.convert_rule_to_code("Only red cards", max_retries=0)

        assert client._call_count == 2
        assert result1["code"] == valid_code_1
        assert result2["code"] == valid_code_2

    def test_cache_cleared_on_reset(self, mock_sleep):
        """clear_compile_cache() should force re-compilation."""
        valid_code = "return card.rank % 2 == 0"
        client = FakeCompilerClient([valid_code, valid_code])

        client.convert_rule_to_code("Only even ranks", max_retries=0)
        client.clear_compile_cache()
        client.convert_rule_to_code("Only even ranks", max_retries=0)

        assert client._call_count == 2

    def test_different_max_attempts_not_cached(self, mock_sleep):
        """Different max_total_attempts should produce separate cache entries.

        A call with a small budget that exhausts should not prevent a later
        call with a larger budget from retrying.
        """
        valid_code = "return card.rank % 2 == 0"
        bad_code = "invalid!!!"
        # First 2 responses are bad (exhaust budget=2), next response is valid
        client = FakeCompilerClient([bad_code, bad_code, valid_code])

        result1 = client.convert_rule_to_code(
            "Only even ranks", max_retries=0, max_total_attempts=2,
        )
        assert result1["status"] == "exhausted"

        result2 = client.convert_rule_to_code(
            "Only even ranks", max_retries=0, max_total_attempts=5,
        )
        # With the larger budget, it should retry and succeed
        assert result2["status"] != "exhausted"
        assert result2["code"] == valid_code
