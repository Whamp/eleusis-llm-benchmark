"""Post-hoc Shadow Guess evaluation from immutable structured Round facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from eleusis.game.cards import Card
from eleusis.game.engine import Rule
from eleusis.game.validator import RuleValidator
from eleusis.llm.base import BaseLLMClient
from eleusis.round_record import validate_round_record_document

SHADOW_VERDICT_VERSION = 1


class ShadowVerdictValidationError(ValueError):
    """Raised when Shadow Guess evaluation evidence is malformed."""


class _StrictShadowVerdictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _ShadowVerdictSettings(_StrictShadowVerdictModel):
    num_simulations: int = Field(ge=0)
    turns_per_simulation: int = Field(ge=0)
    simulation_seed: int
    compiler_max_retries: int = Field(ge=0)


class _ShadowVerdictResult(_StrictShadowVerdictModel):
    correct: bool
    reasoning: str


class _ShadowVerdictDocument(_StrictShadowVerdictModel):
    version: Literal[1]
    verdict_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_id: str
    judge_identity: dict[str, JsonValue] = Field(min_length=1)
    behavior_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    settings: _ShadowVerdictSettings
    verdict: _ShadowVerdictResult
    evidence: dict[str, JsonValue]


@dataclass(frozen=True)
class ShadowEvaluatorState:
    """Canonical post-card board supplied to one offline Shadow Guess judge."""

    mainline: tuple[Card, ...]
    rejected_cards_by_position: tuple[tuple[Card, ...], ...]

    def to_canonical_board_data(self) -> list[dict[str, object]]:
        """Encode ordered mainline Cards with rejected Cards attached by position."""
        return [
            {
                "card": card.to_canonical_card_data(),
                "rejected_cards": [
                    rejected.to_canonical_card_data() for rejected in rejected_cards
                ],
            }
            for card, rejected_cards in zip(
                self.mainline,
                self.rejected_cards_by_position,
                strict=True,
            )
        ]


def _decode_canonical_card(payload: object) -> Card:
    """Decode one canonical rank-and-suit Card from Shadow evaluator evidence."""
    if not isinstance(payload, Mapping):
        raise ShadowVerdictValidationError(
            "Shadow evaluator state invalid: Card must be an object"
        )
    try:
        return Card.from_canonical_card_data(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ShadowVerdictValidationError(
            f"Shadow evaluator state invalid: canonical Card rejected: {error}"
        ) from error


def decode_shadow_evaluator_state(
    turn: Mapping[str, object],
) -> ShadowEvaluatorState:
    """Decode exact post-card board state for offline Shadow Guess evaluation."""
    post_card_state = turn.get("post_card_state")
    if not isinstance(post_card_state, Mapping):
        raise ShadowVerdictValidationError(
            "Shadow evaluator state invalid: post_card_state is missing"
        )
    mainline_payload = post_card_state.get("mainline")
    if not isinstance(mainline_payload, list):
        raise ShadowVerdictValidationError(
            "Shadow evaluator state invalid: mainline is missing"
        )

    mainline: list[Card] = []
    rejected_cards_by_position: list[tuple[Card, ...]] = []
    for position_payload in mainline_payload:
        if not isinstance(position_payload, Mapping):
            raise ShadowVerdictValidationError(
                "Shadow evaluator state invalid: mainline position must be an object"
            )
        rejected_payload = position_payload.get("rejected_cards")
        if not isinstance(rejected_payload, list):
            raise ShadowVerdictValidationError(
                "Shadow evaluator state invalid: rejected_cards is missing"
            )
        mainline.append(_decode_canonical_card(position_payload.get("card")))
        rejected_cards_by_position.append(
            tuple(_decode_canonical_card(card) for card in rejected_payload)
        )

    return ShadowEvaluatorState(
        mainline=tuple(mainline),
        rejected_cards_by_position=tuple(rejected_cards_by_position),
    )


def validate_shadow_verdict_document(payload: object) -> dict[str, object]:
    """Validate and normalize one immutable version-one Shadow Verdict sidecar."""
    try:
        document = _ShadowVerdictDocument.model_validate(payload)
    except ValidationError as error:
        raise ShadowVerdictValidationError(
            f"Shadow Verdict invalid: {error}"
        ) from error
    return cast(dict[str, object], document.model_dump(mode="json"))


def _find_shadow_proposal_turn(
    record: Mapping[str, object],
    proposal_id: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Find exactly one Shadow Guess proposal and its containing Turn."""
    matches: list[tuple[Mapping[str, object], Mapping[str, object]]] = []
    for turn in cast(list[Mapping[str, object]], record["turns"]):
        guess = turn.get("guess_attempt")
        if (
            isinstance(guess, Mapping)
            and guess.get("kind") == "shadow"
            and guess.get("proposal_id") == proposal_id
        ):
            matches.append((turn, guess))
    if len(matches) != 1:
        raise ShadowVerdictValidationError(
            "Shadow Verdict proposal unavailable: expected one immutable proposal "
            f"for {proposal_id!r}, found {len(matches)}"
        )
    return matches[0]


def _shadow_verdict_evidence(
    evaluator_state: ShadowEvaluatorState,
    metadata: Mapping[str, object],
) -> dict[str, JsonValue]:
    """Project provider-neutral compiler and simulation evidence for one verdict."""
    complexity = metadata.get("complexity_metrics")
    complexity_values = complexity if isinstance(complexity, Mapping) else {}
    return {
        "evaluator_state": cast(JsonValue, evaluator_state.to_canonical_board_data()),
        "guessed_code": cast(str | None, metadata.get("guessed_code")),
        "node_count": cast(int | None, complexity_values.get("node_count")),
        "cyclomatic_complexity": cast(
            int | None,
            complexity_values.get("cyclomatic"),
        ),
        "compilation": {
            "status": cast(str, metadata["compilation_status"]),
            "attempt_count": cast(int, metadata["compilation_attempts"]),
            "cache_hit": cast(bool, metadata["compilation_cache_hit"]),
            "artifact_provider": cast(
                str | None,
                metadata["compilation_provider"],
            ),
        },
        "equivalence": {
            "cache_hit": cast(bool, metadata["equivalence_cache_hit"]),
            "comparisons": cast(int, metadata["simulation_comparisons"]),
            "mismatches": cast(int, metadata["simulation_mismatches"]),
            "duration_seconds": cast(
                float,
                metadata["simulation_duration_seconds"],
            ),
        },
    }


def _shadow_verdict_id(payload: Mapping[str, object]) -> str:
    """Derive a stable content identity for one immutable Shadow Verdict."""
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate_shadow_guess(
    round_record: Mapping[str, object],
    proposal_id: str,
    rule_compiler_client: BaseLLMClient,
    *,
    judge_identity: Mapping[str, JsonValue],
    behavior_fingerprint: str,
    settings: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one immutable Shadow Guess proposal into a versioned sidecar."""
    record = validate_round_record_document(round_record)
    if record["terminal_outcome"] is None:
        raise ShadowVerdictValidationError(
            "Shadow Verdict evaluation rejected: Round Record is still active"
        )
    turn, proposal = _find_shadow_proposal_turn(record, proposal_id)
    evaluator_state = decode_shadow_evaluator_state(turn)
    validated_settings = _ShadowVerdictSettings.model_validate(settings)
    secret_rule = cast(Mapping[str, object], record["secret_rule"])
    actual_rule = Rule(
        description=cast(str, secret_rule["description"]),
        code=cast(str, secret_rule["code"]),
    )
    correct, reasoning, metadata = RuleValidator().compare_rules(
        actual_rule=actual_rule,
        guessed_rule_desc=cast(str, proposal["guess"]),
        current_mainline=list(evaluator_state.mainline),
        rule_compiler_client=rule_compiler_client,
        num_simulations=validated_settings.num_simulations,
        turns_per_simulation=validated_settings.turns_per_simulation,
        simulation_seed=validated_settings.simulation_seed,
        compiler_max_retries=validated_settings.compiler_max_retries,
    )
    identity_payload: dict[str, object] = {
        "version": SHADOW_VERDICT_VERSION,
        "proposal_id": proposal_id,
        "judge_identity": dict(judge_identity),
        "behavior_fingerprint": behavior_fingerprint,
        "settings": validated_settings.model_dump(mode="json"),
    }
    return validate_shadow_verdict_document(
        {
            **identity_payload,
            "verdict_id": _shadow_verdict_id(identity_payload),
            "verdict": {"correct": correct, "reasoning": reasoning},
            "evidence": _shadow_verdict_evidence(evaluator_state, metadata),
        }
    )
