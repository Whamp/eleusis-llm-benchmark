"""Utility functions shared across the Eleusis codebase."""

import re


def model_spec_to_display_name(model_spec: str) -> str:
    """Convert model spec to readable display name.

    Examples:
        "openrouter:anthropic/claude-3.5-haiku" -> "Claude 3.5 Haiku"
        "hf:meta-llama/Llama-3.3-70B" -> "Llama 3.3 70b"
    """
    # Remove provider prefix
    if ":" in model_spec:
        _, model_name = model_spec.split(":", 1)
    else:
        model_name = model_spec

    # Extract last part after /
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    # Clean up common suffixes and format
    model_name = model_name.replace("-", " ").replace("_", " ")
    model_name = re.sub(r'\s+', ' ', model_name).strip()
    return model_name.title()
