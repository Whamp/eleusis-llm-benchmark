"""Contract tests for reasoning evidence captured by OpenAICompatClient.

Native reasoning streams (DeepSeek-style reasoning_content, OpenRouter-style
reasoning) must land in LLMCallMetrics.reasoning_text so Model Attempts carry
the trace, and token accounting must split reasoning from answer tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from eleusis.llm.openai_compat import OpenAICompatClient


@dataclass
class _FakeDelta:
    """One streamed delta with optional native reasoning fields."""

    content: str | None = None
    reasoning_content: str | None = None
    reasoning: str | None = None


@dataclass
class _FakeChoice:
    delta: _FakeDelta
    finish_reason: str | None = None


@dataclass
class _FakeUsageDetails:
    reasoning_tokens: int | None = None


@dataclass
class _FakeUsage:
    prompt_tokens: int = 50
    completion_tokens: int = 100
    completion_tokens_details: _FakeUsageDetails | None = None


@dataclass
class _FakeChunk:
    choices: list[_FakeChoice] = field(default_factory=list)
    usage: Any = None


def _make_client(reasoning_format: str = "reasoning_content") -> OpenAICompatClient:
    """Build a compat client aimed at a stub transport."""
    client = OpenAICompatClient(
        base_url="http://test.local/v1",
        api_key="test-key",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512,
        reasoning_format=reasoning_format,
        role="test",
    )
    return client


def _install_stream(client: OpenAICompatClient, chunks: list[_FakeChunk]) -> None:
    """Serve one canned stream from the client's chat-completions endpoint."""

    def create(**_kwargs: object) -> list[_FakeChunk]:
        return chunks

    client.client.chat.completions.create = create  # ty: ignore[invalid-assignment]


def test_reasoning_content_deltas_flow_to_metrics_reasoning_text() -> None:
    """DeepSeek-style reasoning_content deltas become attempt evidence."""
    client = _make_client()
    _install_stream(
        client,
        [
            _FakeChunk(
                choices=[_FakeChoice(delta=_FakeDelta(reasoning_content="step one,"))]
            ),
            _FakeChunk(
                choices=[_FakeChoice(delta=_FakeDelta(reasoning_content=" step two"))]
            ),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="7H"))]),
            _FakeChunk(usage=_FakeUsage(completion_tokens=40)),
        ],
    )

    choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert choice.message.content == "7H"
    assert metrics.reasoning_text == "step one, step two"
    assert metrics.has_reasoning is True
    assert metrics.reasoning_tokens > 0
    assert metrics.answer_tokens + metrics.reasoning_tokens == metrics.output_tokens


def test_reasoning_deltas_flow_to_metrics_reasoning_text() -> None:
    """OpenRouter-style reasoning deltas become attempt evidence."""
    client = _make_client()
    _install_stream(
        client,
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(reasoning="thinking"))]),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="AS"))]),
            _FakeChunk(usage=_FakeUsage(completion_tokens=20)),
        ],
    )

    _choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert metrics.reasoning_text == "thinking"
    assert metrics.has_reasoning is True


def test_provider_reported_reasoning_tokens_preferred_over_estimate() -> None:
    """Explicit usage reasoning tokens win over the subtraction estimate."""
    client = _make_client()
    _install_stream(
        client,
        [
            _FakeChunk(
                choices=[_FakeChoice(delta=_FakeDelta(reasoning_content="mull"))]
            ),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="7H"))]),
            _FakeChunk(
                usage=_FakeUsage(
                    completion_tokens=100,
                    completion_tokens_details=_FakeUsageDetails(reasoning_tokens=17),
                )
            ),
        ],
    )

    _choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert metrics.reasoning_tokens == 17
    assert metrics.answer_tokens == 83
    assert metrics.output_tokens == 100


def test_no_reasoning_leaves_reasoning_text_absent() -> None:
    """Plain content streams record no reasoning, not empty-string evidence."""
    client = _make_client()
    _install_stream(
        client,
        [
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="7H"))]),
            _FakeChunk(usage=_FakeUsage(completion_tokens=10)),
        ],
    )

    _choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert metrics.reasoning_text is None
    assert metrics.has_reasoning is False
    assert metrics.reasoning_tokens == 0


def test_reasoning_never_fabricated_for_none_format() -> None:
    """A none reasoning_format keeps zero reasoning even if a stray tag appears."""
    client = _make_client(reasoning_format="none")
    _install_stream(
        client,
        [
            _FakeChunk(
                choices=[
                    _FakeChoice(delta=_FakeDelta(reasoning_content="should not count"))
                ]
            ),
            _FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="7H"))]),
            _FakeChunk(usage=_FakeUsage(completion_tokens=10)),
        ],
    )

    _choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert metrics.reasoning_tokens == 0
    assert metrics.answer_tokens == 10


if __name__ == "__main__":
    pytest.main([__file__])
