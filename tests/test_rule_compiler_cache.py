"""Tests for bounded rule compiler attempts.

Verifies:
- convert_rule_to_code() respects max_total_attempts parameter
- Returns failure status on exhaustion instead of looping forever
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
