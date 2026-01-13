"""LLM client and player components for Eleusis."""

import logging
import os

from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    ModelCapabilities,
    detect_reasoning_model_type,
    probe_model_capabilities,
)
from eleusis.llm.huggingface import HuggingFaceClient
from eleusis.llm.openrouter import OpenRouterClient
from eleusis.llm.player import LLMScientist

logger = logging.getLogger(__name__)

__all__ = [
    "BaseLLMClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "LLMCallMetrics",
    "GenerateMetrics",
    "ModelCapabilities",
    "LLMScientist",
    "create_client",
    "detect_reasoning_model_type",
    "probe_model_capabilities",
]


def create_client(
    model_spec: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    role: str = "unknown",
    max_continuation_attempts: int = 3,
    seed: int | None = None,
    run_probe: bool = False,
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
        max_continuation_attempts: Max continuation attempts for truncated responses
        seed: Random seed for reproducibility
        run_probe: If True, run capability probe on client before returning

    Returns:
        Configured LLM client instance (with capabilities set if run_probe=True)
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
        "max_continuation_attempts": max_continuation_attempts,
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

    if run_probe:
        logger.info(f"Running capability probe for {model_spec}...")
        client.capabilities = probe_model_capabilities(client)
        logger.info(f"Capabilities: reasoning_format={client.capabilities.reasoning_format}, "
                    f"supports_disable_thinking={client.capabilities.supports_disable_thinking}")

    return client
