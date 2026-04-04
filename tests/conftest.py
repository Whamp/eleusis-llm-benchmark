"""Shared test fixtures: fake LLM clients and helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.llm.base import GenerateMetrics, LLMCallMetrics

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RULES_PATH = FIXTURES_DIR / "mini_rules.json"


class FakeLLMClient:
    """Fake LLM client that returns pre-scripted responses.

    Responses are popped in order from the `responses` list.
    If a response is a TruncationError instance, it is raised instead.
    """

    def __init__(self, responses: list | None = None):
        self.responses: list = list(responses or [])
        self.call_metrics: list[LLMCallMetrics] = []
        self.generate_metrics: list[GenerateMetrics] = []
        self.fallback_clients: list = []
        self.model_name = "fake-model"
        self.provider_name = "fake"
        self.role = "test"
        self._call_count = 0

    def generate(self, prompt: str, xml_tag: str | None = None, return_dict: bool = False):
        if not self.responses:
            raise RuntimeError("FakeLLMClient: no more scripted responses")
        resp = self.responses.pop(0)
        self._call_count += 1

        # Allow raising exceptions from scripts
        if isinstance(resp, Exception):
            raise resp

        # Record a minimal generate metric
        self.generate_metrics.append(GenerateMetrics(
            total_calls=1,
            continuation_count=0,
            total_prompt_tokens=100,
            total_output_tokens=50,
            total_reasoning_tokens=30,
            total_answer_tokens=20,
            total_duration_seconds=0.1,
            success=True,
        ))

        if return_dict and isinstance(resp, str):
            return json.loads(resp)
        return resp

    def reset_usage_stats(self):
        self.call_metrics.clear()
        self.generate_metrics.clear()

    def get_usage_stats(self):
        return {"total_calls": self._call_count}


def make_action_response(card_str: str, **overrides) -> dict:
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
def mini_rules():
    """Load the 2-rule mini fixture library."""
    with open(MINI_RULES_PATH) as f:
        return json.load(f)


@pytest.fixture
def even_rule():
    """Rule: only even ranks."""
    return Rule(
        description="Only cards with an even rank (2,4,6,8,10,12).",
        code="return card.rank % 2 == 0",
    )


@pytest.fixture
def red_rule():
    """Rule: only red cards."""
    return Rule(
        description="Only red cards (hearts or diamonds).",
        code='return card.color == "red"',
    )


@pytest.fixture
def sample_hand():
    """A deterministic 4-card hand for testing."""
    return [
        Card(2, Suit.HEARTS),
        Card(5, Suit.SPADES),
        Card(7, Suit.DIAMONDS),
        Card(10, Suit.CLUBS),
    ]
