"""Tests for OpenAICompatClient reasoning-effort and extra-body routing options."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from eleusis.benchmark_config import ModelConfig
from eleusis.llm.client_factory import create_client_from_config
from eleusis.llm.openai_compat import OpenAICompatClient


@dataclass
class _FakeDelta:
    """One streamed delta with visible content only."""

    content: str | None = None


@dataclass
class _FakeChoice:
    """One streamed choice carrying a delta."""

    delta: _FakeDelta
    finish_reason: str | None = None


@dataclass
class _FakeChunk:
    """One streamed chunk exposing choices."""

    choices: list[_FakeChoice] = field(default_factory=list)
    usage: Any = None


@dataclass
class _FakeCreate:
    """Records create() kwargs and returns one canned content chunk."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: object) -> list[_FakeChunk]:
        """Capture the request kwargs and emit a single-content stream."""
        self.kwargs = dict(kwargs)
        return [_FakeChunk(choices=[_FakeChoice(delta=_FakeDelta(content="hi"))])]


def _make_client(
    reasoning_effort: str | None = None,
    extra_body: dict[str, object] | None = None,
) -> OpenAICompatClient:
    """Build an OpenAICompatClient from an inline config with overrides."""
    values: dict[str, Any] = {
        "provider": "openai_compat",
        "model_id": "test-model",
        "base_url": "http://test.local/v1",
    }
    if reasoning_effort is not None:
        values["reasoning_effort"] = reasoning_effort
    if extra_body is not None:
        values["extra_body"] = extra_body
    config = cast(ModelConfig, values)
    client = create_client_from_config(config, role="test")
    assert isinstance(client, OpenAICompatClient)
    return client


def _call_and_capture(client: OpenAICompatClient) -> dict[str, Any]:
    """Run one API call through a fake transport and return its request kwargs."""
    fake = _FakeCreate()
    client.client.chat.completions.create = fake  # ty: ignore[invalid-assignment]
    result = client._call_api(
        [{"role": "user", "content": "hello"}],
    )
    assert result[0].message.content == "hi"
    return fake.kwargs


def test_factory_defaults_omit_reasoning_effort_and_extra_body() -> None:
    """Without new config keys the client sends neither option."""
    client = _make_client()
    assert client.reasoning_effort is None
    assert client.extra_body is None
    kwargs = _call_and_capture(client)
    assert kwargs.get("reasoning_effort") is None
    assert "extra_body" not in kwargs or kwargs.get("extra_body") is None


def test_factory_forwards_reasoning_effort_to_request() -> None:
    """Configured reasoning_effort reaches the provider request."""
    client = _make_client(reasoning_effort="medium")
    assert client.reasoning_effort == "medium"
    kwargs = _call_and_capture(client)
    assert kwargs["reasoning_effort"] == "medium"


def test_factory_forwards_extra_body_provider_pin() -> None:
    """Configured extra_body routing pin reaches the provider request."""
    pin: dict[str, object] = {
        "provider": {"order": ["Novita"], "allow_fallbacks": False}
    }
    client = _make_client(extra_body=pin)
    assert client.extra_body == pin
    kwargs = _call_and_capture(client)
    assert kwargs["extra_body"] == pin


def test_reasoning_effort_and_extra_body_combine_in_one_request() -> None:
    """Both options appear together in a single provider request."""
    pin: dict[str, object] = {
        "provider": {"order": ["AkashML"], "allow_fallbacks": False}
    }
    client = _make_client(reasoning_effort="high", extra_body=pin)
    kwargs = _call_and_capture(client)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == pin
