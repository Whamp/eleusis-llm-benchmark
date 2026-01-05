"""LLM provider implementations and factory."""

from eleusis.providers.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    detect_reasoning_model_type,
)
from eleusis.providers.huggingface import HuggingFaceClient
from eleusis.providers.openrouter import OpenRouterClient


def create_client(model_spec: str, **kwargs) -> BaseLLMClient:
    """Factory function to create an LLM client based on model specification.

    Args:
        model_spec: Model specification with optional provider prefix.
            - "openrouter:model-name" → OpenRouterClient
            - "hf:model-name" → HuggingFaceClient
            - "model-name" (no prefix) → HuggingFaceClient (backward compatibility)
        **kwargs: Additional arguments passed to the client constructor

    Returns:
        Configured LLM client instance

    Examples:
        >>> client = create_client("hf:meta-llama/Llama-3.3-70B-Instruct", temperature=0.7)
        >>> client = create_client("openrouter:anthropic/claude-3-sonnet", temperature=0.7)
        >>> client = create_client("meta-llama/Llama-3.3-70B-Instruct")  # defaults to HF
    """
    if model_spec.startswith("openrouter:"):
        model_name = model_spec[11:]  # Remove "openrouter:" prefix
        return OpenRouterClient(model_name, **kwargs)
    elif model_spec.startswith("hf:"):
        model_name = model_spec[3:]  # Remove "hf:" prefix
        return HuggingFaceClient(model_name, **kwargs)
    else:
        # Default to HuggingFace for backward compatibility
        return HuggingFaceClient(model_spec, **kwargs)


__all__ = [
    "BaseLLMClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "LLMCallMetrics",
    "GenerateMetrics",
    "create_client",
    "detect_reasoning_model_type",
]
