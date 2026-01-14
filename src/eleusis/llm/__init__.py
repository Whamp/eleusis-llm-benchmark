"""LLM client and player components for Eleusis."""

import logging
import os

from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
)
from eleusis.llm.huggingface import HuggingFaceClient
from eleusis.llm.openrouter import OpenRouterClient
from eleusis.player import LLMScientist  # Re-export for backward compat


logger = logging.getLogger(__name__)

__all__ = [
    "BaseLLMClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "LLMCallMetrics",
    "GenerateMetrics",
    "LLMScientist",
    "create_client",
]


def create_client(
    model_spec: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    seed: int | None = None,
) -> BaseLLMClient:
    """Create an LLM client based on model specification.

    Args:
        model_spec: Model specification with optional provider prefix:
            - "openrouter:model-name" → OpenRouterClient
            - "hf:model-name" → HuggingFaceClient
            - "model-name" (no prefix) → HuggingFaceClient
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        role: Role identifier for metrics
        seed: Random seed for reproducibility

    Returns:
        Configured LLM client instance
    """
    if ":" in model_spec:
        provider, model_name = model_spec.split(":", 1)
    else:
        provider = "hf"
        model_name = model_spec

    common_kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "role": role,
        "seed": seed,
    }

    if provider == "openrouter":
        client = OpenRouterClient(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            **common_kwargs,
        )
    elif provider in ("hf", "huggingface"):
        client = HuggingFaceClient(
            api_key=os.getenv("HF_TOKEN"),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return client
