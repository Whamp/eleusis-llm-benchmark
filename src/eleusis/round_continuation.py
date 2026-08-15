"""Versioned snapshots for continuing one active Eleusis Round."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from eleusis.evaluation_results import TurnRecord
from eleusis.game.engine import GameEngine, Rule
from eleusis.game.rule_library import RuleMetadata
from eleusis.game.state import GameState
from eleusis.game.validator import RuleValidator
from eleusis.llm.base import BaseLLMClient
from eleusis.player import LLMScientist
from eleusis.round_execution import ActionErrorHandler, RoundRuntime

__all__ = [
    "ROUND_CONTINUATION_VERSION",
    "RestoredRoundContinuation",
    "RoundContinuationIncompatibilityError",
    "capture_round_continuation",
    "restore_round_continuation",
    "validate_round_continuation_document",
]

ROUND_CONTINUATION_VERSION = 1


class RoundContinuationIncompatibilityError(ValueError):
    """Raised when an active Round continuation cannot be restored exactly."""


_JSON_VALUE_ADAPTER = TypeAdapter(JsonValue, config=ConfigDict(strict=True))


class _StrictContinuationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _CardSnapshot(_StrictContinuationModel):
    rank: int = Field(ge=1, le=13)
    suit: Literal["hearts", "diamonds", "clubs", "spades"]


class _SidelineSnapshot(_StrictContinuationModel):
    mainline_index: int = Field(ge=0)
    cards: list[_CardSnapshot]


class _PlayerSnapshot(_StrictContinuationModel):
    name: str
    hand: list[_CardSnapshot]
    score: int


class _FailedGuessSnapshot(_StrictContinuationModel):
    player: str
    guess: str


class _GameStateSnapshot(_StrictContinuationModel):
    mainline: list[_CardSnapshot]
    sidelines: list[_SidelineSnapshot]
    deck: list[_CardSnapshot]
    player: _PlayerSnapshot
    failed_rule_guesses: list[_FailedGuessSnapshot]
    turn_number: int = Field(ge=1)
    game_over: bool
    winner: str | None

    @model_validator(mode="after")
    def _validate_sideline_positions(self) -> _GameStateSnapshot:
        indexes = [sideline.mainline_index for sideline in self.sidelines]
        if indexes != sorted(set(indexes)):
            raise ValueError("sideline positions must be unique and ordered")
        if indexes and indexes[-1] >= len(self.mainline):
            raise ValueError("sideline position must identify a mainline Card")
        return self


class _RuleMetadataSnapshot(_StrictContinuationModel):
    name: str | None
    description: str
    code: str


class _RuleSnapshot(_StrictContinuationModel):
    description: str
    code: str
    metadata: _RuleMetadataSnapshot | None


class _CodeComplexitySnapshot(_StrictContinuationModel):
    node_count: int
    cyclomatic: int


class _RuleComparisonSnapshot(_StrictContinuationModel):
    simulation_comparisons: int = Field(ge=0)
    simulation_mismatches: int = Field(ge=0)
    simulation_duration_seconds: float = Field(ge=0)
    guessed_code: str | None
    complexity_metrics: _CodeComplexitySnapshot | None
    compilation_status: str
    compilation_attempts: int = Field(ge=0)


class _ValidatorCacheKeySnapshot(_StrictContinuationModel):
    actual_rule_code: str
    guessed_rule_description: str
    num_simulations: int = Field(ge=0)
    turns_per_simulation: int = Field(ge=0)
    simulation_seed: int


class _ValidatorCacheEntrySnapshot(_StrictContinuationModel):
    key: _ValidatorCacheKeySnapshot
    correct: bool
    reasoning: str
    metadata: _RuleComparisonSnapshot


class _EngineSnapshot(_StrictContinuationModel):
    rule_guessed: bool
    winning_turn: int | None
    failed_guess_count: int = Field(ge=0)
    hand_size: int = Field(ge=0)
    wrong_guess_penalty: int = Field(ge=0)
    num_simulations: int = Field(ge=0)
    turns_per_simulation: int = Field(ge=0)
    simulation_seed: int
    compiler_max_retries: int = Field(ge=0)
    validator_cache: list[_ValidatorCacheEntrySnapshot]

    @model_validator(mode="after")
    def _validate_winning_turn(self) -> _EngineSnapshot:
        if self.rule_guessed != (self.winning_turn is not None):
            raise ValueError("rule_guessed and winning_turn must agree")
        return self


class _PlayHistorySnapshot(_StrictContinuationModel):
    card: str
    accepted: bool
    reasoning_summary: str


class _ScientistSnapshot(_StrictContinuationModel):
    name: str
    max_retries: int = Field(ge=0)
    max_turns: int = Field(ge=0)
    play_history: list[_PlayHistorySnapshot]
    rng_state: list[object]


class _CallMetricsSnapshot(_StrictContinuationModel):
    model_name: str
    role: str
    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    throughput_tokens_per_sec: float = Field(ge=0)
    finish_reason: str
    has_reasoning: bool
    timestamp: float
    is_continuation: bool
    continuation_depth: int = Field(ge=0)
    provider: str
    cost_usd: float | None


class _GenerateMetricsSnapshot(_StrictContinuationModel):
    total_calls: int = Field(ge=0)
    continuation_count: int = Field(ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    total_reasoning_tokens: int = Field(ge=0)
    total_answer_tokens: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    success: bool


class _CompileCacheKeySnapshot(_StrictContinuationModel):
    rule_text: str
    max_total_attempts: int = Field(ge=1)


class _CompileResultSnapshot(_StrictContinuationModel):
    code: str | None
    status: str
    attempts: int = Field(ge=0)
    sleep_cycles: int = Field(ge=0)
    provider_used: str | None


class _CompileCacheEntrySnapshot(_StrictContinuationModel):
    key: _CompileCacheKeySnapshot
    value: _CompileResultSnapshot


class _ClientSnapshot(_StrictContinuationModel):
    call_metrics: list[_CallMetricsSnapshot]
    generate_metrics: list[_GenerateMetricsSnapshot]
    compile_cache: list[_CompileCacheEntrySnapshot]
    fallback_clients: list[_ClientSnapshot]


class _ClientsSnapshot(_StrictContinuationModel):
    scientist: _ClientSnapshot
    compiler: _ClientSnapshot


class _RoundSnapshot(_StrictContinuationModel):
    number: int = Field(ge=1)
    player_name: str
    max_turns: int = Field(ge=1)
    shadow_mode: str
    elapsed_seconds: float = Field(ge=0)


class _RoundContinuationDocument(_StrictContinuationModel):
    version: Literal[1]
    next_turn_index: int = Field(ge=0)
    turn_records: list[TurnRecord]
    round: _RoundSnapshot
    rule: _RuleSnapshot
    game_state: _GameStateSnapshot
    engine: _EngineSnapshot
    scientist: _ScientistSnapshot
    clients: _ClientsSnapshot

    @model_validator(mode="after")
    def _validate_round_relationships(self) -> _RoundContinuationDocument:
        if self.next_turn_index != len(self.turn_records):
            raise ValueError("next_turn_index must equal the completed Turn count")
        turn_numbers = [turn["turn_number"] for turn in self.turn_records]
        if turn_numbers != list(range(1, self.next_turn_index + 1)):
            raise ValueError("completed Turn numbers must be contiguous from one")
        if self.game_state.player.name != self.round.player_name:
            raise ValueError("game-state player must match the Round player")
        if self.scientist.name != self.round.player_name:
            raise ValueError("scientist must match the Round player")
        return self


@dataclass(frozen=True)
class RestoredRoundContinuation:
    """Fresh runtime and completed Turns decoded from one continuation snapshot."""

    runtime: RoundRuntime
    turn_records: list[TurnRecord]
    next_turn_index: int


def _validate_round_continuation(
    payload: object,
) -> _RoundContinuationDocument:
    """Validate a decoded continuation and normalize all nested domain data."""
    if not isinstance(payload, Mapping):
        raise RoundContinuationIncompatibilityError(
            "Round continuation incompatible: document must be an object"
        )
    version = payload.get("version")
    if version != ROUND_CONTINUATION_VERSION:
        raise RoundContinuationIncompatibilityError(
            "Round continuation incompatible: unsupported version "
            f"{version!r}; expected {ROUND_CONTINUATION_VERSION}"
        )
    try:
        _JSON_VALUE_ADAPTER.validate_python(payload)
        return _RoundContinuationDocument.model_validate(payload)
    except ValidationError as error:
        raise RoundContinuationIncompatibilityError(
            f"Round continuation incompatible: malformed version {version}: {error}"
        ) from error


def validate_round_continuation_document(payload: object) -> dict[str, object]:
    """Validate and normalize one active Round continuation document."""
    return cast(
        dict[str, object],
        _validate_round_continuation(payload).model_dump(mode="json"),
    )


def capture_round_continuation(
    runtime: RoundRuntime,
    turn_records: list[TurnRecord],
    *,
    next_turn_index: int,
) -> dict[str, object]:
    """Capture one JSON-compatible active Round continuation.

    Args:
        runtime: Live Round collaborators after setup or a completed Turn.
        turn_records: Ordered records for every completed Turn.
        next_turn_index: Zero-based index of the next Turn to execute.

    Returns:
        A strictly validated version-one continuation document.

    Raises:
        RoundContinuationIncompatibilityError: If live state violates the contract.
    """
    elapsed_seconds = max(0.0, time.time() - runtime.start_time)
    payload: dict[str, object] = {
        "version": ROUND_CONTINUATION_VERSION,
        "next_turn_index": next_turn_index,
        "turn_records": [dict(turn) for turn in turn_records],
        "round": {
            "number": runtime.round_number,
            "player_name": runtime.player_name,
            "max_turns": runtime.max_turns,
            "shadow_mode": runtime.shadow_mode,
            "elapsed_seconds": elapsed_seconds,
        },
        "rule": {
            "description": runtime.rule.description(),
            "code": runtime.rule.get_code(),
            "metadata": (
                dict(runtime.rule_metadata)
                if runtime.rule_metadata is not None
                else None
            ),
        },
        "game_state": runtime.game_state.snapshot_game_continuation(),
        "engine": runtime.engine.snapshot_engine_continuation(),
        "scientist": runtime.scientist.snapshot_scientist_continuation(),
        "clients": {
            "scientist": runtime.scientist_client.snapshot_client_continuation(),
            "compiler": runtime.rule_compiler_client.snapshot_client_continuation(),
        },
    }
    return cast(
        dict[str, object],
        _validate_round_continuation(payload).model_dump(mode="json"),
    )


def restore_round_continuation(
    payload: object,
    *,
    scientist_client: BaseLLMClient,
    rule_compiler_client: BaseLLMClient,
    handle_action_error: ActionErrorHandler,
    pause_after_turn: bool = False,
    results_folder: str | None = None,
) -> RestoredRoundContinuation:
    """Restore fresh Round objects from a validated continuation document.

    Provider clients are supplied fresh from fixed Benchmark Run settings. Their
    provider-neutral accounting and compiler cache state are restored, while SDK
    objects, credentials, sockets, headers, and wire payloads never enter the document.

    Args:
        payload: Decoded JSON continuation document.
        scientist_client: Fresh client for benchmark-model calls.
        rule_compiler_client: Fresh client for rule-compilation calls.
        handle_action_error: Existing Round fallback boundary.
        pause_after_turn: Operational pause setting for resumed execution.
        results_folder: Operational output folder for the latest prompt.

    Returns:
        Fresh runtime objects, completed Turn records, and the next Turn index.

    Raises:
        RoundContinuationIncompatibilityError: If exact restoration is unsupported.
    """
    document = _validate_round_continuation(payload)
    try:
        scientist_client.restore_client_continuation(
            document.clients.scientist.model_dump(mode="python")
        )
        rule_compiler_client.restore_client_continuation(
            document.clients.compiler.model_dump(mode="python")
        )
        rule = Rule(document.rule.description, document.rule.code)
        game_state = GameState.restore_game_continuation(
            document.game_state.model_dump(mode="python")
        )
        validator = RuleValidator()
        validator.restore_validator_cache(
            [
                entry.model_dump(mode="python")
                for entry in document.engine.validator_cache
            ]
        )
        engine = GameEngine.restore_engine_continuation(
            document.engine.model_dump(mode="python"),
            game_state=game_state,
            rule=rule,
            rule_compiler_client=rule_compiler_client,
            rule_validator=validator,
        )
        scientist = LLMScientist.restore_scientist_continuation(
            document.scientist.model_dump(mode="python"),
            llm_client=scientist_client,
            engine=engine,
        )
        metadata = (
            cast(RuleMetadata, document.rule.metadata.model_dump(mode="python"))
            if document.rule.metadata is not None
            else None
        )
        runtime = RoundRuntime(
            round_number=document.round.number,
            start_time=time.time() - document.round.elapsed_seconds,
            rule=rule,
            rule_metadata=metadata,
            engine=engine,
            game_state=game_state,
            scientist=scientist,
            scientist_client=scientist_client,
            rule_compiler_client=rule_compiler_client,
            player_name=document.round.player_name,
            max_turns=document.round.max_turns,
            shadow_mode=document.round.shadow_mode,
            pause_after_turn=pause_after_turn,
            results_folder=results_folder,
            handle_action_error=handle_action_error,
        )
        turn_records = cast(
            list[TurnRecord],
            document.model_dump(mode="python")["turn_records"],
        )
    except (KeyError, SyntaxError, TypeError, ValueError) as error:
        raise RoundContinuationIncompatibilityError(
            "Round continuation incompatible: validated data could not restore "
            f"fresh runtime objects: {error}"
        ) from error
    return RestoredRoundContinuation(
        runtime=runtime,
        turn_records=turn_records,
        next_turn_index=document.next_turn_index,
    )
