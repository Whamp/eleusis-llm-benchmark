"""LLM client implementations - backward compatibility module.

This module re-exports from the new providers package for backward compatibility.
New code should import from eleusis.providers directly.
"""

# Re-export everything from providers for backward compatibility
from eleusis.providers import (
    BaseLLMClient,
    GenerateMetrics,
    HuggingFaceClient,
    LLMCallMetrics,
    OpenRouterClient,
    create_client,
    detect_reasoning_model_type,
)

__all__ = [
    "BaseLLMClient",
    "HuggingFaceClient",
    "OpenRouterClient",
    "LLMCallMetrics",
    "GenerateMetrics",
    "create_client",
    "detect_reasoning_model_type",
]
