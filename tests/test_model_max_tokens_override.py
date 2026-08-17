"""Contract tests for models.yaml max_tokens overrides in create_client."""

from __future__ import annotations

from typing import Any, cast

import pytest

from eleusis.benchmark_config import ModelConfig
from eleusis.llm import client_factory, create_client
from eleusis.llm.openai_compat import OpenAICompatClient


def _patch_model_registry(
    monkeypatch: pytest.MonkeyPatch, model_config: dict[str, Any]
) -> None:
    """Point models.yaml loading at one canned model entry."""
    monkeypatch.setattr(
        client_factory,
        "load_model_config",
        lambda _key: cast(ModelConfig, model_config),
    )


def test_model_max_tokens_overrides_run_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A models.yaml max_tokens entry wins over the run-level allowance.

    Heavy-thinking models need a larger output budget than the global default;
    the model entry is the stronger, model-specific setting.
    """
    _patch_model_registry(
        monkeypatch,
        {
            "provider": "openai_compat",
            "model_id": "deepseek-test",
            "base_url": "http://test.local/v1",
            "max_tokens": 32768,
        },
    )

    client = create_client("deepseek-test", max_tokens=16384, role="test")

    assert isinstance(client, OpenAICompatClient)
    assert client.max_tokens == 32768


def test_model_without_override_keeps_run_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Models without a max_tokens entry inherit the passed run allowance."""
    _patch_model_registry(
        monkeypatch,
        {
            "provider": "openai_compat",
            "model_id": "plain-model",
            "base_url": "http://test.local/v1",
        },
    )

    client = create_client("plain-model", max_tokens=16384, role="test")

    assert isinstance(client, OpenAICompatClient)
    assert client.max_tokens == 16384


if __name__ == "__main__":
    pytest.main([__file__])
