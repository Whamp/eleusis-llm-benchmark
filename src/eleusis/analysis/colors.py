"""Color scheme utilities for analysis plots."""

import logging
from pathlib import Path

import yaml
from typing_extensions import TypedDict

from eleusis.benchmark_config import ModelConfig, parse_model_registry

logger = logging.getLogger(__name__)

# Fallback color for unknown models
DEFAULT_COLOR = "#888888"

# Providers that indicate open-source models
OPEN_PROVIDERS = {"huggingface", "openai_compat"}


class ModelMetadata(TypedDict):
    """Display and licensing metadata for one configured model."""

    color: str
    is_open: bool
    provider: str


def _load_model_registry() -> dict[str, ModelConfig]:
    """Load and validate models.yaml for analysis display helpers."""
    models_path = Path(__file__).parent.parent.parent.parent / "models.yaml"
    if not models_path.exists():
        logger.warning(f"models.yaml not found at {models_path}")
        return {}
    with models_path.open() as models_file:
        return parse_model_registry(yaml.safe_load(models_file))


def load_model_metadata() -> dict[str, ModelMetadata]:
    """Load model metadata from models.yaml.

    Returns dict mapping model_key to {color, is_open, provider}.
    """
    models = _load_model_registry()
    metadata: dict[str, ModelMetadata] = {}
    for model_key, config_value in models.items():
        provider_value = config_value.get("provider", "")
        provider = provider_value if isinstance(provider_value, str) else ""
        color_value = config_value.get("color", DEFAULT_COLOR)
        color = color_value if isinstance(color_value, str) else DEFAULT_COLOR
        metadata[model_key] = {
            "color": color,
            "is_open": provider in OPEN_PROVIDERS,
            "provider": provider,
        }
    return metadata


def load_model_colors() -> dict[str, str]:
    """Load model colors from models.yaml."""
    models = _load_model_registry()
    colors: dict[str, str] = {}
    for model_key, config_value in models.items():
        color = config_value.get("color")
        if isinstance(color, str):
            colors[model_key] = color
    return colors


def normalize_model_name(name: str) -> str:
    """Normalize a model name for matching."""
    return name.lower().strip().replace(" ", "-").replace("_", "-")


def resolve_model_metadata(
    model_name: str,
    metadata: dict[str, ModelMetadata],
) -> ModelMetadata:
    """Resolve fuzzy model-name matches to display and licensing metadata."""
    normalized_name = normalize_model_name(model_name)
    for key, model_metadata in metadata.items():
        normalized_key = normalize_model_name(key)
        if (
            normalized_key == normalized_name
            or normalized_key in normalized_name
            or normalized_name in normalized_key
        ):
            return model_metadata
    return {"color": DEFAULT_COLOR, "is_open": False, "provider": "unknown"}


def get_model_color(model_name: str, model_colors: dict[str, str]) -> str:
    """Get color for a model name with fuzzy matching.

    Tries exact match first, then normalized match, then substring match.
    """
    # Exact match
    if model_name in model_colors:
        return model_colors[model_name]

    # Normalized exact match
    normalized = normalize_model_name(model_name)
    for key, color in model_colors.items():
        if normalize_model_name(key) == normalized:
            return color

    # Substring match (model name contains key or vice versa)
    for key, color in model_colors.items():
        norm_key = normalize_model_name(key)
        if norm_key in normalized or normalized in norm_key:
            return color

    logger.debug(f"No color found for model '{model_name}', using default")
    return DEFAULT_COLOR


def get_color_map(
    model_names: list[str], model_colors: dict[str, str]
) -> dict[str, str]:
    """Build color map for a list of model names."""
    return {name: get_model_color(name, model_colors) for name in model_names}
