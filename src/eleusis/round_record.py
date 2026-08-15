"""Strict versioned scientific record for one Eleusis Round."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)

from eleusis.evaluation_results import TurnRecord
from eleusis.game.cards import Card
from eleusis.round_continuation import validate_round_continuation_document

if TYPE_CHECKING:
    from eleusis.round_execution import RoundRuntime

ROUND_RECORD_VERSION = 1


class RoundRecordValidationError(ValueError):
    """Raised when proposed authoritative Round facts are inconsistent."""


class _StrictRoundRecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _CardRecord(_StrictRoundRecordModel):
    rank: int = Field(ge=1, le=13)
    suit: Literal["hearts", "diamonds", "clubs", "spades"]


class _BoardPositionRecord(_StrictRoundRecordModel):
    card: _CardRecord
    rejected_cards: list[_CardRecord]


class _VisibleStateRecord(_StrictRoundRecordModel):
    mainline: list[_BoardPositionRecord]
    hand: list[_CardRecord]
    turn_number: int = Field(ge=1)
    failed_rule_guesses: list[dict[str, str]]


class _ModelAttemptRecord(_StrictRoundRecordModel):
    attempt_number: int = Field(ge=1)
    prompt: str
    structured_completion: dict[str, JsonValue]
    interpretation: Literal["usable_action"]
    provider: str
    model: str
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    answer_tokens: int = Field(ge=0)


class _FinalDecisionRecord(_StrictRoundRecordModel):
    origin: Literal["model_attempt", "fallback"]
    selected_card: _CardRecord


class _CardOutcomeRecord(_StrictRoundRecordModel):
    accepted: bool
    replacement_draw: _CardRecord | None


class _RoundTurnRecord(_StrictRoundRecordModel):
    turn_number: int = Field(ge=1)
    pre_decision_state: _VisibleStateRecord
    model_attempts: list[_ModelAttemptRecord]
    final_decision: _FinalDecisionRecord
    card_outcome: _CardOutcomeRecord
    post_card_state: _VisibleStateRecord
    guess_attempt: JsonValue | None


class _SecretRuleRecord(_StrictRoundRecordModel):
    name: str | None
    description: str
    code: str


class _RoundSettingsRecord(_StrictRoundRecordModel):
    effective_game_seed: int
    effective_round_seed: int
    batch_round_index: int = Field(ge=0)
    hand_size: int = Field(ge=1)
    max_turns: int = Field(ge=1)
    wrong_guess_penalty: int = Field(ge=0)
    shadow_mode: str


class _TerminalOutcomeRecord(_StrictRoundRecordModel):
    kind: Literal["turn_limit", "correct_formal_guess"]


class _RoundRecordDocument(_StrictRoundRecordModel):
    version: Literal[1]
    run_id: str
    scheduled_round_number: int = Field(ge=1)
    secret_rule: _SecretRuleRecord
    settings: _RoundSettingsRecord
    model_identity: dict[str, JsonValue]
    compiler_identity: dict[str, JsonValue]
    prompt_identities: list[dict[str, JsonValue]]
    source_provenance: dict[str, JsonValue]
    turns: list[_RoundTurnRecord]
    terminal_outcome: _TerminalOutcomeRecord | None

    @model_validator(mode="after")
    def _validate_round_lifecycle(self) -> _RoundRecordDocument:
        turn_numbers = [turn.turn_number for turn in self.turns]
        if turn_numbers != list(range(1, len(self.turns) + 1)):
            raise ValueError("Round Turn numbers must be contiguous from one")
        for previous, current in zip(self.turns, self.turns[1:], strict=False):
            if previous.post_card_state != current.pre_decision_state:
                raise ValueError("Round Turn state continuity is broken")
        if self.terminal_outcome is None:
            return self
        if not self.turns:
            raise ValueError("Terminal Outcome requires at least one completed Turn")
        if (
            self.terminal_outcome.kind == "turn_limit"
            and len(self.turns) != self.settings.max_turns
        ):
            raise ValueError("Turn-limit outcome requires exactly max_turns Turns")
        return self


def _validated_round_record(payload: object) -> _RoundRecordDocument:
    """Decode one strict Round Record with a searchable domain error."""
    try:
        return _RoundRecordDocument.model_validate(payload)
    except ValidationError as error:
        raise RoundRecordValidationError(f"Round Record invalid: {error}") from error


def validate_round_record_document(payload: object) -> dict[str, object]:
    """Validate and normalize a version-one active or completed Round Record."""
    return cast(
        dict[str, object],
        _validated_round_record(payload).model_dump(mode="json"),
    )


def _canonical_card(payload: Mapping[str, object]) -> dict[str, int | str]:
    """Normalize one continuation Card through the owned Card codec."""
    return Card.from_canonical_card_data(payload).to_canonical_card_data()


def _game_state(continuation: Mapping[str, object]) -> Mapping[str, object]:
    """Return the validated game-state section of a continuation document."""
    game_state = continuation.get("game_state")
    if not isinstance(game_state, Mapping):
        raise RoundRecordValidationError(
            "Round Record transition invalid: continuation game_state is missing"
        )
    return game_state


def _visible_state(continuation: Mapping[str, object]) -> dict[str, object]:
    """Project model-visible structured state from hidden continuation state."""
    state = _game_state(continuation)
    mainline_payload = cast(list[Mapping[str, object]], state["mainline"])
    sidelines_payload = cast(list[Mapping[str, object]], state["sidelines"])
    rejected_by_index = {
        cast(int, sideline["mainline_index"]): [
            _canonical_card(card)
            for card in cast(list[Mapping[str, object]], sideline["cards"])
        ]
        for sideline in sidelines_payload
    }
    player = cast(Mapping[str, object], state["player"])
    return {
        "mainline": [
            {
                "card": _canonical_card(card),
                "rejected_cards": rejected_by_index.get(index, []),
            }
            for index, card in enumerate(mainline_payload)
        ],
        "hand": [
            _canonical_card(card)
            for card in cast(list[Mapping[str, object]], player["hand"])
        ],
        "turn_number": cast(int, state["turn_number"]),
        "failed_rule_guesses": [
            dict(guess)
            for guess in cast(
                list[Mapping[str, str]],
                state["failed_rule_guesses"],
            )
        ],
    }


def _card_key(card: Mapping[str, object]) -> tuple[int, str]:
    """Return a hashable canonical Card identity preserving multiplicity counts."""
    return cast(int, card["rank"]), cast(str, card["suit"])


def _all_card_counts(continuation: Mapping[str, object]) -> Counter[tuple[int, str]]:
    """Count Cards across every visible and hidden gameplay collection."""
    state = _game_state(continuation)
    player = cast(Mapping[str, object], state["player"])
    cards = list(cast(list[Mapping[str, object]], state["mainline"]))
    cards.extend(cast(list[Mapping[str, object]], state["deck"]))
    cards.extend(cast(list[Mapping[str, object]], player["hand"]))
    for sideline in cast(list[Mapping[str, object]], state["sidelines"]):
        cards.extend(cast(list[Mapping[str, object]], sideline["cards"]))
    return Counter(_card_key(card) for card in cards)


def _selected_card(
    turn_record: TurnRecord,
    pre_state: Mapping[str, object],
) -> dict[str, int | str]:
    """Resolve the played Card against the exact ordered pre-decision hand."""
    card_text = turn_record["action_result"].get("card")
    if not isinstance(card_text, str):
        raise RoundRecordValidationError(
            "Round Record transition invalid: completed Turn has no played Card"
        )
    hand = cast(list[Mapping[str, object]], pre_state["hand"])
    for card_payload in hand:
        card = Card.from_canonical_card_data(card_payload)
        if str(card) == card_text:
            return card.to_canonical_card_data()
    raise RoundRecordValidationError(
        "Round Record transition invalid: selected Card was not in the "
        "pre-decision hand"
    )


def _replacement_draw(
    selected_card: Mapping[str, object],
    pre_state: Mapping[str, object],
    post_state: Mapping[str, object],
) -> dict[str, int | str] | None:
    """Validate ordered hand mutation and return the replacement draw when present."""
    expected_hand = [
        dict(card) for card in cast(list[Mapping[str, object]], pre_state["hand"])
    ]
    selected_key = _card_key(selected_card)
    selected_index = next(
        (
            index
            for index, card in enumerate(expected_hand)
            if _card_key(card) == selected_key
        ),
        None,
    )
    if selected_index is None:
        raise RoundRecordValidationError(
            "Round Record transition invalid: selected Card membership changed"
        )
    expected_hand.pop(selected_index)
    post_hand = [
        dict(card) for card in cast(list[Mapping[str, object]], post_state["hand"])
    ]
    if post_hand == expected_hand:
        return None
    if len(post_hand) == len(expected_hand) + 1 and post_hand[:-1] == expected_hand:
        return _canonical_card(post_hand[-1])
    raise RoundRecordValidationError(
        "Round Record transition invalid: replacement draw broke hand order or "
        "multiplicity"
    )


def _copy_board_position(position: Mapping[str, object]) -> dict[str, object]:
    """Copy one canonical mainline position for transition comparison."""
    card = cast(Mapping[str, object], position["card"])
    rejected_cards = cast(
        list[Mapping[str, object]],
        position["rejected_cards"],
    )
    return {
        "card": dict(card),
        "rejected_cards": [dict(rejected) for rejected in rejected_cards],
    }


def _validate_board_transition(
    selected_card: Mapping[str, object],
    accepted: bool,
    pre_state: Mapping[str, object],
    post_state: Mapping[str, object],
) -> None:
    """Validate accepted-mainline or rejected-sideline placement semantics."""
    pre_mainline = cast(list[Mapping[str, object]], pre_state["mainline"])
    post_mainline = cast(list[Mapping[str, object]], post_state["mainline"])
    expected = [_copy_board_position(position) for position in pre_mainline]
    if accepted:
        expected.append({"card": dict(selected_card), "rejected_cards": []})
    elif expected:
        cast(list[dict[str, object]], expected[-1]["rejected_cards"]).append(
            dict(selected_card)
        )
    if expected != post_mainline:
        raise RoundRecordValidationError(
            "Round Record transition invalid: Card outcome does not match board state"
        )


def _model_attempt(
    runtime: RoundRuntime,
    turn_record: TurnRecord,
) -> dict[str, object]:
    """Build the successful model attempt visible at the current tracer seam."""
    prompt = runtime.scientist.last_prompt
    if prompt is None:
        raise RoundRecordValidationError(
            "Round Record transition invalid: successful Model Attempt has no prompt"
        )
    response = turn_record["llm_response"]
    return {
        "attempt_number": turn_record["retry_count"] + 1,
        "prompt": prompt,
        "structured_completion": dict(response),
        "interpretation": "usable_action",
        "provider": runtime.scientist_client.provider_name,
        "model": runtime.scientist_client.model_name,
        "output_tokens": turn_record["tokens"]["output_tokens"],
        "reasoning_tokens": turn_record["tokens"]["reasoning_tokens"],
        "answer_tokens": turn_record["tokens"]["answer_tokens"],
    }


def create_active_round_record(
    manifest: Mapping[str, object],
    runtime: RoundRuntime,
    *,
    effective_round_seed: int,
    batch_round_index: int,
) -> dict[str, object]:
    """Create the empty active Round Record committed with initial setup."""
    schedule = cast(list[Mapping[str, object]], manifest["schedule"])
    scheduled = schedule[runtime.round_number - 1]
    expected_code = scheduled.get("rule_code")
    if expected_code is not None and expected_code != runtime.rule.get_code():
        raise RoundRecordValidationError(
            "Round Record setup invalid: secret rule differs from the immutable "
            "schedule"
        )
    effective_settings = cast(Mapping[str, object], manifest["effective_settings"])
    payload = {
        "version": ROUND_RECORD_VERSION,
        "run_id": manifest["run_id"],
        "scheduled_round_number": runtime.round_number,
        "secret_rule": {
            "name": (
                runtime.rule_metadata.get("name")
                if runtime.rule_metadata is not None
                else scheduled.get("rule_name")
            ),
            "description": runtime.rule.description(),
            "code": runtime.rule.get_code(),
        },
        "settings": {
            "effective_game_seed": effective_settings["game_seed"],
            "effective_round_seed": effective_round_seed,
            "batch_round_index": batch_round_index,
            "hand_size": runtime.engine.hand_size,
            "max_turns": runtime.max_turns,
            "wrong_guess_penalty": runtime.engine.wrong_guess_penalty,
            "shadow_mode": runtime.shadow_mode,
        },
        "model_identity": manifest["model_identity"],
        "compiler_identity": manifest["compiler_identity"],
        "prompt_identities": manifest["prompt_identities"],
        "source_provenance": manifest["source_provenance"],
        "turns": [],
        "terminal_outcome": None,
    }
    return validate_round_record_document(payload)


def append_round_record_turn(
    record_payload: object,
    previous_continuation_payload: object,
    next_continuation_payload: object,
    turn_record: TurnRecord,
    runtime: RoundRuntime,
) -> dict[str, object]:
    """Validate and append one complete proposed Turn transition."""
    record = _validated_round_record(record_payload)
    if record.terminal_outcome is not None:
        raise RoundRecordValidationError(
            "Round Record transition invalid: completed records are immutable"
        )
    previous = validate_round_continuation_document(previous_continuation_payload)
    current = validate_round_continuation_document(next_continuation_payload)
    expected_turn = len(record.turns) + 1
    if turn_record["turn_number"] != expected_turn:
        raise RoundRecordValidationError(
            "Round Record transition invalid: Turn ordering is not contiguous"
        )
    pre_state = _visible_state(previous)
    post_state = _visible_state(current)
    if (
        record.turns
        and record.turns[-1].post_card_state.model_dump(mode="json") != pre_state
    ):
        raise RoundRecordValidationError(
            "Round Record transition invalid: pre-decision state breaks continuity"
        )
    selected = _selected_card(turn_record, pre_state)
    accepted_value = turn_record["action_result"].get("accepted")
    if not isinstance(accepted_value, bool):
        raise RoundRecordValidationError(
            "Round Record transition invalid: card acceptance verdict is missing"
        )
    replacement_draw = _replacement_draw(selected, pre_state, post_state)
    _validate_board_transition(selected, accepted_value, pre_state, post_state)
    if _all_card_counts(previous) != _all_card_counts(current):
        raise RoundRecordValidationError(
            "Round Record transition invalid: Card multiplicity was not conserved"
        )
    payload = record.model_dump(mode="json")
    turns = cast(list[dict[str, object]], payload["turns"])
    turns.append(
        {
            "turn_number": expected_turn,
            "pre_decision_state": pre_state,
            "model_attempts": [_model_attempt(runtime, turn_record)],
            "final_decision": {
                "origin": (
                    "model_attempt"
                    if turn_record["error"] is None
                    and bool(turn_record["llm_response"])
                    else "fallback"
                ),
                "selected_card": selected,
            },
            "card_outcome": {
                "accepted": accepted_value,
                "replacement_draw": replacement_draw,
            },
            "post_card_state": post_state,
            "guess_attempt": turn_record["guess_attempt"],
        }
    )
    return validate_round_record_document(payload)


def complete_round_record(
    record_payload: object,
    *,
    game_over_reason: str,
) -> dict[str, object]:
    """Validate and finalize an immutable terminal Round Record."""
    record = _validated_round_record(record_payload)
    if record.terminal_outcome is not None:
        raise RoundRecordValidationError(
            "Round Record completion invalid: record already has a Terminal Outcome"
        )
    outcome_by_reason = {
        "max_turns": "turn_limit",
        "correct_guess": "correct_formal_guess",
    }
    outcome = outcome_by_reason.get(game_over_reason)
    if outcome is None:
        raise RoundRecordValidationError(
            "Round Record completion invalid: unsupported game-over reason "
            f"{game_over_reason!r}"
        )
    payload = record.model_dump(mode="json")
    payload["terminal_outcome"] = {"kind": outcome}
    return validate_round_record_document(payload)
