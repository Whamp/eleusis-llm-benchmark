"""Validated persisted-result types for evaluation checkpoints and reports."""

from typing import Literal, NotRequired

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from eleusis.game.rule_library import RuleLibraryEntry


class ConsumedRule(TypedDict):
    """Rule consumed by an evaluation and its completed round count."""

    name: str | None
    description: str
    code: str
    rounds_completed: int


class RuleFactoryCheckpoint(TypedDict):
    """Rule-library cursor needed to resume sequential selection."""

    selection: str
    current_index: int


class CurrentRuleCheckpoint(TypedDict):
    """Current secret rule and its position within a repeated-round batch."""

    description: str | None
    code: str | None
    rounds_used_in_batch: NotRequired[int]
    num_rounds_per_rule: NotRequired[int]


class EvaluationCheckpoint(TypedDict):
    """Mutable resume state saved after each completed round."""

    completed_rounds: int
    total_rounds: int
    rule_factory_state: RuleFactoryCheckpoint
    current_rule: CurrentRuleCheckpoint | None
    rules_consumed: list[ConsumedRule]
    rules_library: list[RuleLibraryEntry]


class EvaluationStatistics(TypedDict):
    """Aggregated scores, token counts, retries, and final averages."""

    total_score: int
    successful_rounds: int
    failed_rounds: int
    total_turns: int
    total_failed_guesses: int
    total_output_tokens: NotRequired[int]
    total_reasoning_tokens: NotRequired[int]
    total_answer_tokens: NotRequired[int]
    total_wall_clock_seconds: NotRequired[float]
    total_retries: NotRequired[int]
    retry_by_cause: NotRequired[dict[str, int]]
    success_rate: NotRequired[float]
    average_score: NotRequired[float]
    average_turns: NotRequired[float]
    average_turns_when_successful: NotRequired[float]
    average_failed_guesses: NotRequired[float]


class GuessAttempt(TypedDict, total=False):
    """Formal or shadow rule guess attached to one Turn."""

    version: Literal[1]
    kind: Literal["formal"]
    guess: str
    correct: bool
    reasoning: str
    guessed_code: str | None
    node_count: int | None
    cyclomatic_complexity: int | None
    compilation: dict[str, object]
    equivalence: dict[str, object]
    shadow: bool
    evaluated: bool


class ModelAttemptTokenMetrics(TypedDict):
    """Provider-neutral token facts observed for one Model Attempt."""

    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int
    answer_tokens: int


class ProviderCallRecord(TypedDict):
    """One observable provider call nested within a Model Attempt."""

    call_number: int
    provider: str
    model: str
    timestamp: float
    duration_seconds: float
    finish_reason: str
    is_continuation: bool
    continuation_depth: int
    token_metrics: ModelAttemptTokenMetrics


class ModelAttemptRecord(TypedDict):
    """One prompt submission and its provider-neutral interpretation evidence."""

    attempt_number: int
    prompt: str
    raw_completion: str | None
    structured_completion: dict[str, object] | None
    interpretation: Literal[
        "usable_action",
        "card_parse_error",
        "truncated",
        "structured_response_parse_error",
        "provider_error",
    ]
    retry_cause: str | None
    started_at: float
    duration_seconds: float
    provider: str
    model: str
    finish_reason: str | None
    token_metrics: ModelAttemptTokenMetrics
    provider_calls: list[ProviderCallRecord]


class TurnRecord(TypedDict):
    """Serializable player state, action, and metrics for one turn."""

    turn_number: int
    player: str
    mainline_state: str
    hand: list[str]
    llm_response: dict[str, object]
    model_attempts: NotRequired[list[ModelAttemptRecord]]
    confidence_level_raw: object
    confidence_level: int | None
    schema_errors: list[str]
    action_result: dict[str, object]
    guess_attempt: GuessAttempt | None
    tokens: dict[str, int]
    retry_count: int
    retry_causes: list[dict[str, object]]
    error: dict[str, str | None] | None


class SavedRound(TypedDict):
    """Serializable subset of one runner result stored in results.json."""

    round_number: int
    rule_name: str | None
    batch_round_index: int
    turn_count: int
    rule_description: str
    rule_code: str
    success: bool
    score: int
    floored_score: int
    no_stakes_score: int
    first_correct_turn: int | None
    first_formal_correct_turn: NotRequired[int | None]
    first_shadow_correct_turn: NotRequired[int | None]
    failed_guesses: int
    game_over_reason: str
    llm_usage: dict[str, dict[str, object]]
    turns: list[TurnRecord]
    wall_clock_seconds: float


class EvaluationMetadata(TypedDict):
    """Self-describing configuration captured with each evaluation."""

    num_rules: int
    num_rounds_per_rule: int
    rule_compiler: str
    rule_compiler_provider: str
    rule_compiler_model_id: str
    rule_compiler_reasoning_format: str
    rule_compiler_temperature: float
    rule_compiler_max_retries: int
    rule_compiler_num_simulations: int
    rule_compiler_turns_per_simulation: int
    rule_compiler_simulation_seed: int | None
    player: str
    player_model: str
    hand_size: int
    max_turns: int
    wrong_guess_penalty: int
    seed: int | None
    llm_max_tokens: int
    llm_temperature: float
    llm_seed: int | None
    llm_max_retries: int
    batch_round_offset: int | None
    suite: str | None


class EvaluationResults(TypedDict):
    """Complete persisted evaluation document."""

    timestamp: str
    folder_name: str
    config: EvaluationMetadata
    rounds: list[SavedRound]
    statistics: EvaluationStatistics
    checkpoint: EvaluationCheckpoint


_EVALUATION_RESULTS_ADAPTER = TypeAdapter(EvaluationResults)


def parse_evaluation_results(value: object) -> EvaluationResults:
    """Validate and return a persisted evaluation results document."""
    return _EVALUATION_RESULTS_ADAPTER.validate_python(value)
