"""Shared test fixtures: fake LLM clients and helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, overload

import pytest

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    LLMMessage,
    LLMResponseEnvelope,
    TruncationError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RULES_PATH = FIXTURES_DIR / "mini_rules.json"

ActionResponse = str | dict[str, object]
ScriptedResponse = ActionResponse | Exception


class FakeLLMClient(BaseLLMClient):
    """Fake LLM client that returns pre-scripted responses.

    Responses are popped in order from the `responses` list. If a response is a
    TruncationError instance, it is raised instead.
    """

    def __init__(self, responses: Sequence[ScriptedResponse] | None = None) -> None:
        """Initialize the client with responses returned in FIFO order."""
        super().__init__(model_name="fake-model", role="test")
        self.responses = list(responses or [])
        self.prompts_seen: list[str] = []
        self.raw_completion_available = True
        self._call_count = 0

    @property
    def provider_name(self) -> str:
        """Fake provider name used in test assertions."""
        return "fake"

    def _call_api(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[LLMResponseEnvelope, LLMCallMetrics]:
        """Reject direct API calls because tests script `generate` responses."""
        del messages, is_continuation, continuation_depth, disable_thinking
        raise NotImplementedError("FakeLLMClient does not make API calls")

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[False] = False,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[True] = True,
    ) -> dict[str, object]: ...

    def generate(
        self, prompt: str, xml_tag: str | None = None, return_dict: bool = False
    ) -> ActionResponse:
        """Return the next scripted response while preserving observable call facts."""
        del xml_tag
        self.prompts_seen.append(prompt)
        self.last_raw_completion = None
        if not self.responses:
            raise RuntimeError("FakeLLMClient: no more scripted responses")
        resp = self.responses.pop(0)
        self._call_count += 1

        if isinstance(resp, Exception):
            if isinstance(resp, TruncationError):
                self._append_call_metric("length")
            raise resp

        if self.raw_completion_available:
            self.last_raw_completion = (
                resp if isinstance(resp, str) else json.dumps(resp, sort_keys=True)
            )
        self._append_call_metric("stop")
        if return_dict and isinstance(resp, str):
            parsed_response = json.loads(resp)
            if not isinstance(parsed_response, dict):
                raise TypeError("FakeLLMClient JSON response must decode to an object")
            result: ActionResponse = parsed_response
        else:
            result = resp
        self.generate_metrics.append(
            GenerateMetrics(
                total_calls=1,
                continuation_count=0,
                total_prompt_tokens=100,
                total_output_tokens=50,
                total_reasoning_tokens=30,
                total_answer_tokens=20,
                total_duration_seconds=0.1,
                success=True,
            )
        )
        return result

    def _append_call_metric(self, finish_reason: str) -> None:
        """Record one provider call exposed by the scripted client boundary."""
        self.call_metrics.append(
            LLMCallMetrics(
                model_name=self.model_name,
                role=self.role,
                prompt_tokens=100,
                output_tokens=50,
                reasoning_tokens=30,
                answer_tokens=20,
                duration_seconds=0.1,
                throughput_tokens_per_sec=500.0,
                finish_reason=finish_reason,
                has_reasoning=True,
                timestamp=float(self._call_count),
                provider=self.provider_name,
            )
        )

    def reset_usage_stats(self) -> None:
        """Verify reset usage stats."""
        self.call_metrics.clear()
        self.generate_metrics.clear()

    def get_usage_stats(self) -> dict[str, object]:
        """Return provider-neutral usage restored by production continuation code."""
        return super().get_usage_stats()


def make_action_response(card_str: str, **overrides: object) -> dict[str, object]:
    """Build a valid action response dict like the LLM would return."""
    resp = {
        "card": card_str,
        "reasoning_summary": overrides.pop("reasoning", "test reasoning"),
        "tentative_rule": overrides.pop("tentative_rule", ""),
        "confidence_level": overrides.pop("confidence_level", 3),
        "guess_rule": overrides.pop("guess_rule", False),
    }
    resp.update(overrides)
    return resp


@pytest.fixture
def mini_rules() -> dict[str, object]:
    """Load the 2-rule mini fixture library."""
    with open(MINI_RULES_PATH) as f:
        return json.load(f)


@pytest.fixture
def even_rule() -> Rule:
    """Rule: only even ranks."""
    return Rule(
        description="Only cards with an even rank (2,4,6,8,10,12).",
        code="return card.rank % 2 == 0",
    )


@pytest.fixture
def red_rule() -> Rule:
    """Rule: only red cards."""
    return Rule(
        description="Only red cards (hearts or diamonds).",
        code='return card.color == "red"',
    )


@pytest.fixture
def sample_hand() -> list[Card]:
    """A deterministic 4-card hand for testing."""
    return [
        Card(2, Suit.HEARTS),
        Card(5, Suit.SPADES),
        Card(7, Suit.DIAMONDS),
        Card(10, Suit.CLUBS),
    ]
