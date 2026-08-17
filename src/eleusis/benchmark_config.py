"""Validated configuration types for benchmark and model YAML files."""

from typing import Literal, NotRequired

from pydantic import TypeAdapter
from typing_extensions import TypedDict

HuggingFaceProvider = Literal[
    "black-forest-labs",
    "cerebras",
    "clarifai",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "ovhcloud",
    "publicai",
    "replicate",
    "sambanova",
    "scaleway",
    "together",
    "wavespeed",
    "zai-org",
]

OpenAIReasoningEffort = Literal["high", "low", "medium", "minimal", "none", "xhigh"]

ProviderName = Literal[
    "anthropic",
    "google",
    "huggingface",
    "openai",
    "openai_compat",
    "xai",
]


class ModelConfig(TypedDict):
    """Configuration for one model provider adapter."""

    provider: ProviderName
    model_id: str
    auth: NotRequired[Literal["pi-codex"]]
    api_key: NotRequired[str]
    base_url: NotRequired[str]
    color: NotRequired[str]
    hf_provider: NotRequired[HuggingFaceProvider]
    reasoning_budget: NotRequired[int]
    reasoning_effort: NotRequired[OpenAIReasoningEffort]
    reasoning_extraction: NotRequired[Literal["deep_think"]]
    reasoning_format: NotRequired[str]
    extra_body: NotRequired[dict[str, object]]
    thinking_level: NotRequired[str]
    temperature: NotRequired[float]
    timeout: NotRequired[int]
    max_tokens: NotRequired[int]


class RuleCompilerConfig(ModelConfig):
    """Model configuration plus rule-equivalence simulation settings."""

    backup_providers: NotRequired[list[ModelConfig]]
    max_retries: NotRequired[int]
    num_simulations: NotRequired[int]
    simulation_seed: NotRequired[int | None]
    turns_per_simulation: NotRequired[int]


class GameConfig(TypedDict):
    """Gameplay and benchmark-round settings."""

    num_rules: NotRequired[int]
    num_rounds_per_rule: NotRequired[int]
    num_rounds: NotRequired[int]
    max_turns: int
    hand_size: int
    wrong_guess_penalty: int
    seed: int | None
    batch_round_offset: NotRequired[int | None]
    pause_after_turn: NotRequired[bool]
    shadow_mode: NotRequired[Literal["disabled", "offline", "online"]]


class LLMConfig(TypedDict):
    """Generation settings shared by benchmark model calls."""

    max_tokens: int
    max_llm_retries: int
    temperature: float
    seed: int | None


class RulesConfig(TypedDict):
    """Rule-library selection settings."""

    library_path: str | None
    selection: Literal["random", "sequential"]
    index: int


class BenchmarkConfig(TypedDict):
    """Complete validated benchmark configuration."""

    model: NotRequired[str]
    game: GameConfig
    llm: LLMConfig
    rule_compiler: RuleCompilerConfig
    rules: RulesConfig
    suite: NotRequired[str]


_BENCHMARK_CONFIG_ADAPTER = TypeAdapter(BenchmarkConfig)
_MODEL_REGISTRY_ADAPTER = TypeAdapter(dict[str, ModelConfig])


def parse_benchmark_config(value: object) -> BenchmarkConfig:
    """Validate and return one complete benchmark configuration."""
    return _BENCHMARK_CONFIG_ADAPTER.validate_python(value)


def parse_model_registry(value: object) -> dict[str, ModelConfig]:
    """Validate and return model configurations keyed by CLI model name."""
    return _MODEL_REGISTRY_ADAPTER.validate_python(value)
