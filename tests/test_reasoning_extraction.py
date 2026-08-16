"""Tests for extracting reasoning traces from providers that hide them.

Two capture modes, both landing in Model Attempt evidence:

- Native summary capture: subscription requests ask for detailed reasoning
  summaries and accumulate the streamed summary deltas.
- deep_think extraction (the scratchpad-tool technique): native thinking is
  disabled and a deep_think function tool gives the model a private
  scratchpad, so its chain-of-thought arrives as visible tool-call arguments.

Contract cards:

C1 Subscription mode with reasoning enabled sends summary="detailed" and
   captures streamed reasoning summary deltas as metrics.reasoning_text.
C2 deep_think mode runs a tool loop: effort none plus the deep_think tool
   and a scratchpad instruction; think-call arguments become that round's
   reasoning_text; the recorded tool output is fed back; the final text
   round returns the answer; intermediate rounds land in call_metrics as
   continuations.
C3 deep_think rounds are capped; the final round is sent without tools to
   force a text answer.
C4 A deep_think call to any other tool is rejected with a clear error.
C5 LLMScientist aggregates an attempt's provider-call reasoning traces into
   the Model Attempt record; reasoning_text is absent-safe for historical
   records and continuation snapshots.
C6 models.yaml entries select deep_think extraction through create_client.
C7 Runaway tool-argument streams (the backend rejects max_output_tokens, so
   degenerate whitespace loops never end server-side) are aborted at client
   bounds; the deep_think loop salvages the partial reasoning, records it,
   and continues with a synthesized tool exchange.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from eleusis.benchmark_config import ModelConfig
from eleusis.llm.base import LLMCallMetrics
from eleusis.llm.client_factory import create_client
from eleusis.llm.openai_client import (
    DEEP_THINK_TOOL,
    OpenAIClient,
    ReasoningExtraction,
)
from eleusis.llm.pi_auth import PiCodexAuth
from eleusis.player import LLMScientist
from eleusis.round_record import _ModelAttemptRecord


def _subscription_client(
    reasoning_extraction: ReasoningExtraction | None = None,
) -> OpenAIClient:
    """Build a subscription-mode client with unreachable credentials."""
    return OpenAIClient(
        model_name="gpt-5.6-luna",
        api_key=None,
        codex_auth=PiCodexAuth(auth_path=Path("/nonexistent")),
        reasoning_effort="medium",
        reasoning_extraction=reasoning_extraction,
    )


@dataclass
class _FakeSummaryDeltaEvent:
    """One response.reasoning_summary_text.delta stream event."""

    type: str = "response.reasoning_summary_text.delta"
    delta: str = ""


@dataclass
class _FakeTextPart:
    """One output_text content part."""

    type: str = "output_text"
    text: str = "2H"


@dataclass
class _FakeMessageItem:
    """One message output item."""

    type: str = "message"
    content: list[_FakeTextPart] = field(default_factory=list)


@dataclass
class _FakeFunctionCallItem:
    """One function_call output item carrying tool arguments."""

    type: str = "function_call"
    name: str = "deep_think"
    arguments: str = "{}"
    call_id: str = "call_1"


@dataclass
class _FakeItemDoneEvent:
    """One response.output_item.done stream event."""

    type: str = "response.output_item.done"
    item: Any = None


@dataclass
class _FakeUsage:
    """Usage block of a completed response."""

    input_tokens: int = 10
    output_tokens: int = 5
    output_tokens_details: Any = None


@dataclass
class _FakeCompletedResponse:
    """The response embedded in response.completed."""

    status: str = "completed"
    incomplete_details: Any = None
    usage: _FakeUsage | None = None


@dataclass
class _FakeCompletedEvent:
    """One response.completed stream event."""

    type: str = "response.completed"
    response: _FakeCompletedResponse = field(default_factory=_FakeCompletedResponse)


def _think_round(
    thoughts: str, call_id: str = "call_1", usage: _FakeUsage | None = None
) -> list[Any]:
    """Stream events for one round that only calls deep_think."""
    return [
        _FakeItemDoneEvent(
            item=_FakeFunctionCallItem(
                name="deep_think",
                arguments=json.dumps({"thoughts": thoughts}),
                call_id=call_id,
            )
        ),
        _FakeCompletedEvent(response=_FakeCompletedResponse(usage=usage)),
    ]


def _text_round(text: str, usage: _FakeUsage | None = None) -> list[Any]:
    """Stream events for one round that answers with plain text."""
    return [
        _FakeItemDoneEvent(item=_FakeMessageItem(content=[_FakeTextPart(text=text)])),
        _FakeCompletedEvent(response=_FakeCompletedResponse(usage=usage)),
    ]


def test_subscription_mode_captures_native_reasoning_summaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C1: summary=detailed is requested and deltas land in reasoning_text."""
    client = _subscription_client()
    captured: dict[str, Any] = {}

    def fake_create(**kwargs: object) -> list[Any]:
        captured.update(kwargs)
        return [
            _FakeSummaryDeltaEvent(delta="Inferring the rule: "),
            _FakeSummaryDeltaEvent(delta="primes so far."),
            _FakeItemDoneEvent(
                item=_FakeMessageItem(content=[_FakeTextPart(text="3H")])
            ),
            _FakeCompletedEvent(response=_FakeCompletedResponse(usage=_FakeUsage())),
        ]

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    content = client.generate("Pick a card.")

    assert content == "3H"
    assert captured["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert client.call_metrics[-1].reasoning_text == (
        "Inferring the rule: primes so far."
    )


def test_deep_think_loop_captures_tool_argument_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C2: think arguments become reasoning evidence across the loop."""
    client = _subscription_client(reasoning_extraction="deep_think")
    calls: list[dict[str, Any]] = []
    rounds = iter([_think_round("chunk one"), _text_round("<ACTION>3H</ACTION>")])

    def fake_create(**kwargs: object) -> list[Any]:
        calls.append(kwargs)
        return next(rounds)

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    content = client.generate("Pick a card.")

    assert content == "<ACTION>3H</ACTION>"
    assert len(calls) == 2

    first = calls[0]
    assert first["reasoning"] == {"effort": "none"}
    assert first["tools"] == [DEEP_THINK_TOOL]
    first_input = first["input"]
    assert first_input[0]["role"] == "developer"
    assert "scratchpad" in first_input[0]["content"][0]["text"].lower()

    second_input = calls[1]["input"]
    kinds = [item.get("type") for item in second_input]
    assert "function_call" in kinds and "function_call_output" in kinds

    assert len(client.call_metrics) == 2
    trace_round, final_round = client.call_metrics
    assert trace_round.reasoning_text == "chunk one"
    assert trace_round.is_continuation is False
    assert final_round.reasoning_text is None
    assert final_round.is_continuation is True
    assert final_round.continuation_depth == 1


def test_deep_think_round_cap_forces_final_answer_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: after the cap the last round is sent without tools."""
    client = _subscription_client(reasoning_extraction="deep_think")
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: object) -> list[Any]:
        calls.append(kwargs)
        # Think on every round until tools are removed, then answer.
        if kwargs.get("tools"):
            return _think_round(f"chunk {len(calls)}")
        return _text_round("<ACTION>9S</ACTION>")

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    content = client.generate("Pick a card.")

    assert content == "<ACTION>9S</ACTION>"
    assert len(calls) == OpenAIClient.MAX_DEEP_THINK_ROUNDS + 1
    assert calls[-1].get("tools") in (None, [])


def test_deep_think_rejects_unexpected_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C4: a call to any other tool fails loudly instead of looping."""
    client = _subscription_client(reasoning_extraction="deep_think")

    def fake_create(**kwargs: object) -> list[Any]:
        return [
            _FakeItemDoneEvent(
                item=_FakeFunctionCallItem(
                    name="lookup_cards", arguments="{}", call_id="call_9"
                )
            ),
            _FakeCompletedEvent(),
        ]

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    with pytest.raises(RuntimeError, match="lookup_cards"):
        client.generate("Pick a card.")


def _attempt_metrics(reasoning_text: str | None) -> LLMCallMetrics:
    """One synthetic provider-call metric."""
    return LLMCallMetrics(
        model_name="gpt-5.6-luna",
        role="player",
        prompt_tokens=1,
        output_tokens=1,
        reasoning_tokens=0,
        answer_tokens=1,
        duration_seconds=0.01,
        throughput_tokens_per_sec=100.0,
        finish_reason="stop",
        has_reasoning=False,
        timestamp=time.time(),
        is_continuation=False,
        continuation_depth=0,
        provider="openai",
        reasoning_text=reasoning_text,
    )


def test_player_aggregates_attempt_reasoning_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C5: Model Attempt reasoning_text joins the attempt's provider calls."""
    client = _subscription_client(reasoning_extraction="deep_think")
    scientist = LLMScientist("scientist", client)

    client.call_metrics.append(_attempt_metrics("chunk one"))
    client.call_metrics.append(_attempt_metrics(None))
    client.call_metrics.append(_attempt_metrics("chunk two"))

    scientist._record_model_attempt(
        attempt_number=1,
        prompt="prompt",
        interpretation="usable_action",
        retry_cause=None,
        started_at=time.time(),
        call_metrics_before=0,
    )
    record = scientist.last_model_attempts[-1]
    assert record["reasoning_text"] == "chunk one\n\nchunk two"


def test_model_attempt_record_reasoning_text_is_defaulted() -> None:
    """C5b: historical attempts without reasoning_text still decode."""
    legacy = {
        "attempt_number": 1,
        "prompt": "prompt",
        "raw_completion": None,
        "structured_completion": None,
        "interpretation": "usable_action",
        "retry_cause": None,
        "started_at": 1.0,
        "duration_seconds": 1.0,
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "finish_reason": "stop",
        "token_metrics": {
            "prompt_tokens": 1,
            "output_tokens": 1,
            "reasoning_tokens": 0,
            "answer_tokens": 1,
        },
        "provider_calls": [],
    }
    record = _ModelAttemptRecord.model_validate(legacy)
    assert record.reasoning_text is None


def test_call_metrics_snapshot_preserves_reasoning_text() -> None:
    """C5c: continuation snapshots round-trip the reasoning trace."""
    from eleusis.llm.base import _restore_call_metrics

    metric = _attempt_metrics("trace text")
    restored = _restore_call_metrics(
        {
            "model_name": metric.model_name,
            "role": metric.role,
            "prompt_tokens": metric.prompt_tokens,
            "output_tokens": metric.output_tokens,
            "reasoning_tokens": metric.reasoning_tokens,
            "answer_tokens": metric.answer_tokens,
            "duration_seconds": metric.duration_seconds,
            "throughput_tokens_per_sec": metric.throughput_tokens_per_sec,
            "finish_reason": metric.finish_reason,
            "has_reasoning": metric.has_reasoning,
            "timestamp": metric.timestamp,
            "is_continuation": metric.is_continuation,
            "continuation_depth": metric.continuation_depth,
            "provider": metric.provider,
            "cost_usd": metric.cost_usd,
        }
    )
    assert restored.reasoning_text is None  # absent key defaults safely


@dataclass
class _FakeArgsDeltaEvent:
    """One response.function_call_arguments.delta stream event."""

    type: str = "response.function_call_arguments.delta"
    delta: str = " "


@dataclass
class _ClosableStream:
    """Fake stream that records close() calls."""

    events: list[Any]
    closed: bool = False

    def __iter__(self) -> _ClosableStream:
        return self

    def __next__(self) -> object:
        if not self.events:
            raise StopIteration
        return self.events.pop(0)

    def close(self) -> None:
        self.closed = True


def test_runaway_tool_arguments_abort_and_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-looping argument deltas abort; the loop continues."""
    client = _subscription_client(reasoning_extraction="deep_think")
    streams: list[_ClosableStream] = [
        _ClosableStream(events=[_FakeArgsDeltaEvent(delta=" ")] * 600),
        _ClosableStream(events=_think_round("bounded chunk")),
        _ClosableStream(events=_text_round("ok")),
    ]

    def fake_create(**kwargs: object) -> _ClosableStream:
        return streams.pop(0)

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    runaway_stream = streams[0]
    content = client.generate("Pick a card.")

    assert content == "ok"
    assert runaway_stream.closed is True
    assert client.call_metrics[0].reasoning_text == "bounded chunk"


def test_runaway_tool_arguments_salvage_partial_thoughts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7c: partial thoughts are salvaged, recorded, loop continues."""
    client = _subscription_client(reasoning_extraction="deep_think")
    salvaged_stream = _ClosableStream(
        events=[
            _FakeArgsDeltaEvent(delta='{"thoughts":"partial thinking'),
            *([_FakeArgsDeltaEvent(delta=" ")] * 600),
        ]
    )
    streams: list[_ClosableStream] = [
        salvaged_stream,
        _ClosableStream(events=_text_round("ok")),
    ]
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: object) -> _ClosableStream:
        calls.append(kwargs)
        return streams.pop(0)

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    content = client.generate("Pick a card.")

    assert content == "ok"
    assert client.call_metrics[-1].reasoning_text == "partial thinking"
    kinds = [item.get("type") for item in calls[1]["input"]]
    assert "function_call" in kinds and "function_call_output" in kinds


def test_runaway_tool_arguments_raise_outside_deep_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C7b: without the deep_think loop, a runaway stream surfaces its error."""
    from eleusis.llm.openai_client import CodexRunawayToolArgumentsError

    client = _subscription_client()

    monkeypatch.setattr(
        client.client.responses,
        "create",
        lambda **kwargs: _ClosableStream(
            events=[_FakeArgsDeltaEvent(delta="x" * 100)] * 1000
        ),
        raising=True,
    )
    with pytest.raises(CodexRunawayToolArgumentsError):
        client.generate("Pick a card.")


def test_factory_wires_deep_think_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6: models.yaml reasoning_extraction selects the scratchpad loop."""
    import eleusis.llm.client_factory as factory

    config: ModelConfig = {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "reasoning_effort": "none",
        "auth": "pi-codex",
        "reasoning_extraction": "deep_think",
    }
    monkeypatch.setattr(factory, "load_model_config", lambda _key: config)
    client = create_client("gpt-5.6-luna-deepthink")
    assert isinstance(client, OpenAIClient)
    assert client.reasoning_extraction == "deep_think"


def test_deep_think_round_tokens_count_as_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C8: a scratchpad round's output tokens are reasoning tokens.

    The whole output of a thinking round is the externalized chain of
    thought, so its provider-reported output tokens are classified as
    reasoning, not answer tokens.
    """
    client = _subscription_client(reasoning_extraction="deep_think")
    think_usage = _FakeUsage(input_tokens=10, output_tokens=120)
    answer_usage = _FakeUsage(input_tokens=20, output_tokens=30)
    streams: list[_ClosableStream] = [
        _ClosableStream(
            events=_think_round("weighing suits and ranks", usage=think_usage)
        ),
        _ClosableStream(events=_text_round("Play 4C", usage=answer_usage)),
    ]
    monkeypatch.setattr(
        client.client.responses,
        "create",
        lambda **kwargs: streams.pop(0),
        raising=True,
    )

    content = client.generate("Pick a card.")

    assert content == "Play 4C"
    think_metrics, answer_metrics = client.call_metrics
    assert think_metrics.output_tokens == 120
    assert think_metrics.reasoning_tokens == 120
    assert think_metrics.answer_tokens == 0
    assert answer_metrics.reasoning_tokens == 0
    assert answer_metrics.answer_tokens == 30
