"""Tests for ChatGPT-subscription (pi-backed Codex) authentication.

Contract cards:

C1 PiCodexAuth injects the Authorization bearer token and the
   chatgpt-account-id header on every request, re-reading pi's auth.json
   each time so refreshed tokens are picked up without rebuilding clients.
C2 PiCodexAuth raises a clear PiCodexAuthError when auth.json is missing,
   has no usable openai-codex entry, or the token has expired.
C3 OpenAIClient in subscription mode sends store=False, consumes the
   streamed events, and assembles a response-shaped payload so the existing
   choice/metrics parsing keeps working.
C4 create_client builds that subscription-mode client when a models.yaml
   entry sets auth: pi-codex, honoring its reasoning_effort.
C5 Subscription mode sends typed content blocks (input_text/output_text)
   because the Codex backend rejects plain string content; API-key mode
   keeps sending plain strings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from eleusis.llm.base import LLMMessage
from eleusis.llm.client_factory import create_client
from eleusis.llm.openai_client import OpenAIClient
from eleusis.llm.pi_auth import CODEX_BACKEND_BASE_URL, PiCodexAuth, PiCodexAuthError


def _write_auth_file(
    path: Path,
    access_token: str = "token-1",
    expires_in_seconds: float = 3600.0,
    account_id: str = "acct-123",
) -> None:
    """Write an auth.json with an openai-codex entry like pi's."""
    entry = {
        "access": access_token,
        "accountId": account_id,
        "expires": (time.time() + expires_in_seconds) * 1000,
        "refresh": "refresh-token",
        "type": "oauth",
    }
    path.write_text(json.dumps({"openai-codex": entry}))


def test_pi_codex_auth_injects_headers_and_rereads_each_request(tmp_path: Path) -> None:
    """C1: both subscription headers present and refreshed per request."""
    auth_path = tmp_path / "auth.json"
    _write_auth_file(auth_path, access_token="token-1")
    captured: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(dict(request.headers))
        return httpx.Response(200, json={})

    auth = PiCodexAuth(auth_path=auth_path)
    with httpx.Client(transport=httpx.MockTransport(handler), auth=auth) as http:
        http.get("https://example.test/api")
        _write_auth_file(auth_path, access_token="token-2")
        http.get("https://example.test/api")

    assert captured[0]["authorization"] == "Bearer token-1"
    assert captured[0]["chatgpt-account-id"] == "acct-123"
    assert captured[1]["authorization"] == "Bearer token-2"


def test_pi_codex_auth_rejects_missing_entry_and_expired_token(
    tmp_path: Path,
) -> None:
    """C2: unusable auth.json surfaces as a clear PiCodexAuthError."""
    missing = tmp_path / "missing.json"
    with pytest.raises(PiCodexAuthError, match="not found"):
        PiCodexAuth(auth_path=missing)._load_credentials()

    no_entry = tmp_path / "no-entry.json"
    no_entry.write_text(json.dumps({"other-provider": {}}))
    with pytest.raises(PiCodexAuthError, match="openai-codex"):
        PiCodexAuth(auth_path=no_entry)._load_credentials()

    expired = tmp_path / "expired.json"
    _write_auth_file(expired, expires_in_seconds=-1.0)
    with pytest.raises(PiCodexAuthError, match="expired"):
        PiCodexAuth(auth_path=expired)._load_credentials()


@dataclass
class _FakeUsage:
    """Minimal usage shape used by OpenAIClient metric extraction."""

    input_tokens: int = 10
    output_tokens: int = 4
    output_tokens_details: Any = None


@dataclass
class _FakeSummaryBlock:
    """One reasoning summary block."""

    text: str = "thinking"


@dataclass
class _FakeReasoningItem:
    """One reasoning output item from a stream event."""

    type: str = "reasoning"
    summary: list[_FakeSummaryBlock] = field(default_factory=list)


@dataclass
class _FakeTextPart:
    """One output_text content part from a stream event."""

    type: str = "output_text"
    text: str = "2H"


@dataclass
class _FakeMessageItem:
    """One message output item from a stream event."""

    type: str = "message"
    content: list[_FakeTextPart] = field(default_factory=list)


@dataclass
class _FakeItemDoneEvent:
    """One response.output_item.done stream event."""

    type: str = "response.output_item.done"
    item: Any = None


@dataclass
class _FakeIncompleteDetails:
    """Why a stream response stopped early."""

    reason: str = "max_output_tokens"


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


def test_openai_client_subscription_mode_streams_without_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C3: stream events accumulate, truncation flags, store stays off."""
    client = OpenAIClient(
        model_name="gpt-5.6-luna",
        api_key=None,
        codex_auth=PiCodexAuth(auth_path=Path("/nonexistent")),
        reasoning_effort="medium",
    )
    assert str(client.client.base_url).rstrip("/") == CODEX_BACKEND_BASE_URL

    captured: dict[str, Any] = {}

    def fake_create(**kwargs: object) -> list[Any]:
        captured.update(kwargs)
        return [
            _FakeItemDoneEvent(
                item=_FakeMessageItem(content=[_FakeTextPart(text="2H")])
            ),
            _FakeItemDoneEvent(
                item=_FakeReasoningItem(summary=[_FakeSummaryBlock(text="reasoned")])
            ),
            _FakeCompletedEvent(
                response=_FakeCompletedResponse(
                    status="incomplete",
                    incomplete_details=_FakeIncompleteDetails(
                        reason="max_output_tokens"
                    ),
                    usage=_FakeUsage(input_tokens=11, output_tokens=6),
                )
            ),
        ]

    monkeypatch.setattr(client.client.responses, "create", fake_create, raising=True)
    payload = client._create_response([], "medium")
    choice, metrics = client._call_api([{"role": "user", "content": "pick"}])

    assert captured["store"] is False
    assert captured["stream"] is True
    assert captured["reasoning"] == {"effort": "medium", "summary": "detailed"}
    assert payload.output_text == "2H"
    assert choice.message.content == "2H"
    assert choice.message.reasoning == "reasoned"
    assert choice.finish_reason == "length"
    assert metrics.prompt_tokens == 11
    assert metrics.output_tokens == 6
    assert metrics.provider == "openai"


def test_factory_builds_subscription_client_from_models_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C4: auth: pi-codex selects subscription mode with config effort."""
    import eleusis.llm.client_factory as factory

    config = {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "reasoning_effort": "medium",
        "auth": "pi-codex",
    }
    monkeypatch.setattr(factory, "load_model_config", lambda _key: config)
    client = create_client("gpt-5.6-luna-medium")
    assert isinstance(client, OpenAIClient)
    assert client.codex_auth is not None
    assert client.reasoning_effort == "medium"


def test_subscription_mode_sends_typed_content_blocks() -> None:
    """C5: codex input uses typed blocks; plain-string mode is unchanged."""
    subscription = OpenAIClient(
        model_name="gpt-5.6-luna",
        api_key=None,
        codex_auth=PiCodexAuth(auth_path=Path("/nonexistent")),
    )
    messages: list[LLMMessage] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    sub_input = cast(list[dict[str, Any]], subscription._build_response_input(messages))
    assert sub_input[0] == {
        "role": "developer",
        "content": [{"type": "input_text", "text": "sys"}],
    }
    assert sub_input[1]["content"] == [{"type": "input_text", "text": "hi"}]
    assert sub_input[2]["role"] == "assistant"
    assert sub_input[2]["content"] == [{"type": "input_text", "text": "hello"}]

    api_key_client = OpenAIClient(model_name="gpt-5.2", api_key="k")
    plain_input = cast(
        list[dict[str, Any]], api_key_client._build_response_input(messages)
    )
    assert plain_input[0] == {"role": "developer", "content": "sys"}
