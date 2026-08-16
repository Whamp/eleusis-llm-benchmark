"""LLM client and player components for Eleusis."""

import logging
import os
import re
from pathlib import Path

import yaml

from eleusis.benchmark_config import (
    ModelConfig,
    parse_model_registry,
)
from eleusis.llm.anthropic import AnthropicClient
from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    TruncationError,
)
from eleusis.llm.google import GoogleClient
from eleusis.llm.huggingface import HuggingFaceClient
from eleusis.llm.openai_client import OpenAIClient
from eleusis.llm.openai_compat import OpenAICompatClient
from eleusis.llm.pi_auth import PiCodexAuth
from eleusis.llm.xai import XAIClient
from eleusis.player import LLMScientist  # Re-export for backward compat

logger = logging.getLogger(__name__)

__all__ = [
    "AnthropicClient",
    "BaseLLMClient",
    "GenerateMetrics",
    "GoogleClient",
    "HuggingFaceClient",
    "LLMCallMetrics",
    "LLMScientist",
    "OpenAIClient",
    "OpenAICompatClient",
    "TruncationError",
    "XAIClient",
    "create_client",
    "create_client_from_config",
    "load_model_config",
]


def _resolve_env_vars(value: str) -> str:
    """Resolve ${ENV_VAR} references in a string."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        env_val = os.getenv(var_name)
        if env_val is None:
            logger.warning(f"Environment variable {var_name} not set")
            return match.group(0)  # Leave as-is
        return env_val

    return re.sub(r"\$\{(\w+)\}", _replace, value)


def find_models_yaml() -> Path:
    """Find models.yaml in project root."""
    # Try common locations
    candidates = [
        Path("models.yaml"),
        Path(__file__).parent.parent.parent.parent / "models.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("models.yaml not found")


def load_model_config(model_key: str) -> ModelConfig:
    """Load model configuration from models.yaml."""
    models_path = find_models_yaml()
    with open(models_path) as f:
        all_models = parse_model_registry(yaml.safe_load(f))

    if model_key not in all_models:
        available = list(all_models.keys())
        raise ValueError(
            f"Model '{model_key}' not found in models.yaml. Available: {available}"
        )

    return all_models[model_key]


def create_client(
    model_key: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    seed: int | None = None,
) -> BaseLLMClient:
    """Create an LLM client based on model key from models.yaml.

    Args:
        model_key: Key referencing a model in models.yaml.
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        role: Role identifier for metrics
        seed: Random seed for reproducibility

    Returns:
        Configured LLM client instance
    """
    config = load_model_config(model_key)
    provider = config["provider"]
    model_id = config["model_id"]

    match provider:
        case "anthropic":
            reasoning_budget = config.get("reasoning_budget", 8192)
            return AnthropicClient(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                reasoning_budget=reasoning_budget,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case "openai":
            reasoning_effort = config.get("reasoning_effort", "medium")
            codex_auth = PiCodexAuth() if config.get("auth") == "pi-codex" else None
            return OpenAIClient(
                api_key=os.getenv("OPENAI_API_KEY") if codex_auth is None else None,
                reasoning_effort=reasoning_effort,
                codex_auth=codex_auth,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case "google":
            thinking_level = config.get("thinking_level", "high")
            return GoogleClient(
                api_key=os.getenv("GOOGLE_API_KEY"),
                thinking_level=thinking_level,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case "xai":
            base_url = config.get("base_url")
            if not base_url:
                raise ValueError(f"Model {model_key!r} requires an xAI base_url")
            return XAIClient(
                api_key=os.getenv("XAI_API_KEY"),
                base_url=base_url,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case "huggingface":
            hf_provider = config.get("hf_provider")
            reasoning_format = config.get("reasoning_format", "separate_field")
            return HuggingFaceClient(
                api_key=os.getenv("HF_TOKEN"),
                hf_provider=hf_provider,
                reasoning_format=reasoning_format,
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case "openai_compat":
            base_url = config.get("base_url")
            if not base_url:
                raise ValueError(
                    "openai_compat provider requires 'base_url' in model config"
                )
            api_key = _resolve_env_vars(config.get("api_key", "sk-no-key-required"))
            reasoning_format = config.get("reasoning_format", "reasoning_content")
            timeout = config.get("timeout", 600)
            return OpenAICompatClient(
                base_url=base_url,
                api_key=api_key,
                reasoning_format=reasoning_format,
                timeout=timeout,
                reasoning_effort=config.get("reasoning_effort"),
                extra_body=config.get("extra_body"),
                model_name=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                role=role,
                seed=seed,
            )

        case _:
            raise ValueError(f"Unknown provider: {provider}")


def create_client_from_config(
    config: ModelConfig,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    seed: int | None = None,
) -> BaseLLMClient:
    """Create an LLM client from inline config dict.

    Args:
        config: Dict with provider, model_id, and provider-specific options
        temperature: Sampling temperature (can be overridden by config)
        max_tokens: Maximum tokens to generate
        role: Role identifier for metrics
        seed: Random seed for reproducibility

    Returns:
        Configured LLM client instance
    """
    provider = config["provider"]
    model_id = config["model_id"]
    temperature = config.get("temperature", temperature)

    if provider == "huggingface":
        hf_provider = config.get("hf_provider")  # None if not specified
        reasoning_format = config.get("reasoning_format", "separate_field")
        return HuggingFaceClient(
            api_key=os.getenv("HF_TOKEN"),
            hf_provider=hf_provider,
            reasoning_format=reasoning_format,
            model_name=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

    if provider == "openai_compat":
        base_url = config.get("base_url")
        if not base_url:
            raise ValueError("openai_compat provider requires 'base_url' in config")
        api_key = _resolve_env_vars(config.get("api_key", "sk-no-key-required"))
        reasoning_format = config.get("reasoning_format", "reasoning_content")
        timeout = config.get("timeout", 600)
        return OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            reasoning_format=reasoning_format,
            timeout=timeout,
            reasoning_effort=config.get("reasoning_effort"),
            extra_body=config.get("extra_body"),
            model_name=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

    # Could extend to other providers if needed
    raise ValueError(
        "create_client_from_config only supports huggingface and openai_compat, got:"
        f" {provider}"
    )
