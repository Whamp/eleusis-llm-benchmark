"""Compatibility views for historical JSON-only Benchmark Runs."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from eleusis.analysis.legacy_records import LegacyRecord

_CARD_TEXT = r"(?:A|[2-9]|10|J|Q|K)[♥♦♣♠]"
_COMPACT_BOARD_PATTERN = re.compile(
    rf"^{_CARD_TEXT}(?: (?:{_CARD_TEXT}|\[{_CARD_TEXT}\]))*$"
)


class HistoricalRunCompatibilityError(ValueError):
    """Raised when a result artifact has no safe analysis compatibility view."""


@dataclass(frozen=True)
class MigrationDiagnostic:
    """Explain one unavailable or compatibility-derived historical field."""

    code: str
    path: str
    message: str

    def to_document(self) -> dict[str, str]:
        """Encode one migration diagnostic for in-memory report metadata."""
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class AnalysisRunArtifact:
    """One selected result artifact and its explicit analysis compatibility policy."""

    analysis_document: LegacyRecord | None
    source_format: Literal[
        "historical_json",
        "strict_round_record_export",
        "unsupported_export",
    ]
    artifact_source: Literal["json", "sqlite"]
    is_partial: bool
    diagnostics: tuple[MigrationDiagnostic, ...]
    unavailable_fields: tuple[str, ...]
    export_is_current: bool | None


def _historical_guess(turn: Mapping[str, object]) -> Mapping[str, object] | None:
    """Return a historical Guess Attempt when its stored shape is usable."""
    guess = turn.get("guess_attempt")
    return guess if isinstance(guess, Mapping) else None


def _historical_round_derivation(
    round_data: LegacyRecord,
    config: Mapping[str, object],
) -> dict[str, object] | None:
    """Derive score facts only when the historical Turn evidence is sufficient."""
    turns_value = round_data.get("turns")
    max_turns = config.get("max_turns")
    wrong_guess_penalty = config.get("wrong_guess_penalty")
    if (
        not isinstance(turns_value, list)
        or not isinstance(max_turns, int)
        or not isinstance(wrong_guess_penalty, int)
        or not all(
            isinstance(turn, Mapping)
            and isinstance(turn.get("turn_number"), int)
            and "guess_attempt" in turn
            for turn in turns_value
        )
    ):
        return None
    turns = cast(list[Mapping[str, object]], turns_value)
    formal_guesses = [
        guess
        for turn in turns
        if (guess := _historical_guess(turn)) is not None
        and guess.get("shadow", False) is not True
        and isinstance(guess.get("correct"), bool)
    ]
    failed_guesses = sum(guess["correct"] is False for guess in formal_guesses)
    successful_turns = [
        cast(int, turn["turn_number"])
        for turn in turns
        if isinstance(turn.get("turn_number"), int)
        and (guess := _historical_guess(turn)) is not None
        and guess.get("correct") is True
    ]
    formal_success = any(guess["correct"] is True for guess in formal_guesses)
    turn_count = len(turns)
    score = -wrong_guess_penalty * failed_guesses
    if formal_success:
        score = max_turns - turn_count + 1 - wrong_guess_penalty * failed_guesses
    first_correct_turn = min(successful_turns) if successful_turns else None
    no_stakes_score = (
        max_turns - first_correct_turn + 1 if first_correct_turn is not None else 0
    )
    return {
        "turn_count": turn_count,
        "success": formal_success,
        "score": score,
        "floored_score": max(0, score),
        "no_stakes_score": no_stakes_score,
        "first_correct_turn": first_correct_turn,
        "failed_guesses": failed_guesses,
    }


def _recompute_historical_turn_usage(round_data: LegacyRecord) -> bool:
    """Replace imported player token totals when every Turn exposes token facts."""
    turns = round_data.get("turns")
    usage = round_data.get("llm_usage")
    if not isinstance(turns, list) or not isinstance(usage, dict):
        return False
    player_usage = usage.get("player")
    if not isinstance(player_usage, dict):
        return False
    token_documents = [
        turn.get("tokens") for turn in turns if isinstance(turn, Mapping)
    ]
    if len(token_documents) != len(turns) or not all(
        isinstance(tokens, Mapping) for tokens in token_documents
    ):
        return False
    typed_tokens = cast(list[Mapping[str, object]], token_documents)
    recomputed = False
    for token_name in ("output_tokens", "reasoning_tokens", "answer_tokens"):
        values = [tokens.get(token_name) for tokens in typed_tokens]
        if all(isinstance(value, int) for value in values):
            player_usage[token_name] = sum(cast(list[int], values))
            recomputed = True
    return recomputed


def recompute_analysis_statistics(document: LegacyRecord) -> bool:
    """Replace imported run aggregates when all completed Round facts are available."""
    rounds = document.get("rounds")
    if not isinstance(rounds, list) or not all(
        isinstance(round_data, Mapping) for round_data in rounds
    ):
        return False
    required = ("score", "success", "turn_count", "failed_guesses")
    if not all(all(name in round_data for name in required) for round_data in rounds):
        return False
    statistics = document.get("statistics")
    if not isinstance(statistics, dict):
        statistics = {}
        document["statistics"] = statistics
    typed_rounds = cast(list[Mapping[str, object]], rounds)
    successful = sum(round_data["success"] is True for round_data in typed_rounds)
    statistics.update(
        {
            "total_score": sum(cast(int, item["score"]) for item in typed_rounds),
            "successful_rounds": successful,
            "failed_rounds": len(typed_rounds) - successful,
            "total_turns": sum(cast(int, item["turn_count"]) for item in typed_rounds),
            "total_failed_guesses": sum(
                cast(int, item["failed_guesses"]) for item in typed_rounds
            ),
        }
    )
    return True


def _diagnose_historical_turn(
    turn: Mapping[str, object],
    *,
    round_index: int,
    turn_index: int,
    diagnostics: list[MigrationDiagnostic],
    unavailable_fields: set[str],
) -> None:
    """Mark unavailable facts for one historical Turn without reconstructing them."""
    turn_path = f"rounds[{round_index}].turns[{turn_index}]"
    unavailable_fields.add(f"{turn_path}.pre_decision_state")
    for fact_name in ("model_attempts", "provider_calls", "replacement_draw"):
        unavailable_fields.add(f"{turn_path}.{fact_name}")
    if "tokens" not in turn:
        unavailable_fields.add(f"{turn_path}.tokens")
    compact_state = turn.get("mainline_state")
    compact_state_is_valid = isinstance(
        compact_state, str
    ) and _COMPACT_BOARD_PATTERN.fullmatch(compact_state)
    if not compact_state_is_valid:
        diagnostics.append(
            MigrationDiagnostic(
                code="historical_compact_state_invalid",
                path=f"{turn_path}.mainline_state",
                message=(
                    "Historical compact board state is malformed; structured state "
                    "remains unavailable."
                ),
            )
        )


def _diagnose_historical_round(
    round_data: Mapping[str, object],
    *,
    round_index: int,
    diagnostics: list[MigrationDiagnostic],
    unavailable_fields: set[str],
) -> None:
    """Collect unavailable fields and compact-state diagnostics for one Round."""
    for optional_name in ("wall_clock_seconds", "rule_name"):
        if optional_name not in round_data:
            unavailable_fields.add(f"rounds[{round_index}].{optional_name}")
    turns = round_data.get("turns")
    if not isinstance(turns, list):
        return
    for turn_index, turn in enumerate(turns):
        if isinstance(turn, Mapping):
            _diagnose_historical_turn(
                turn,
                round_index=round_index,
                turn_index=turn_index,
                diagnostics=diagnostics,
                unavailable_fields=unavailable_fields,
            )


def _diagnose_historical_turn_states(
    document: LegacyRecord,
    diagnostics: list[MigrationDiagnostic],
    unavailable_fields: set[str],
) -> None:
    """Mark compact presentation state unavailable instead of reconstructing a board."""
    rounds = document.get("rounds")
    if not isinstance(rounds, list):
        return
    for round_index, round_data in enumerate(rounds):
        if isinstance(round_data, Mapping):
            _diagnose_historical_round(
                round_data,
                round_index=round_index,
                diagnostics=diagnostics,
                unavailable_fields=unavailable_fields,
            )


def decode_historical_run(document_value: object) -> AnalysisRunArtifact:
    """Decode one unversioned result document as an explicitly partial view."""
    if not isinstance(document_value, dict):
        raise HistoricalRunCompatibilityError(
            "Historical Run compatibility failed: results.json must contain an object"
        )
    document = cast(LegacyRecord, copy.deepcopy(document_value))
    config = document.get("config")
    rounds = document.get("rounds")
    if not isinstance(config, Mapping) or not isinstance(rounds, list):
        raise HistoricalRunCompatibilityError(
            "Historical Run compatibility failed: config and rounds are required"
        )
    diagnostics = [
        MigrationDiagnostic(
            code="historical_round_record_partial",
            path="round_record",
            message=(
                "Historical JSON has no strict Round Record provenance or hidden "
                "continuation state."
            ),
        )
    ]
    unavailable_fields = {
        "round_record.run_id",
        "round_record.structured_states",
        "round_record.prompt_identities",
        "round_record.source_provenance",
        "round_checkpoint",
    }
    recomputed = False
    for round_data in rounds:
        if not isinstance(round_data, dict):
            continue
        derived = _historical_round_derivation(round_data, config)
        if derived is not None:
            round_data.update(derived)
            recomputed = True
        recomputed = _recompute_historical_turn_usage(round_data) or recomputed
    recomputed = recompute_analysis_statistics(document) or recomputed
    if recomputed:
        diagnostics.append(
            MigrationDiagnostic(
                code="historical_derived_values_recomputed",
                path="derived",
                message=(
                    "Historical persisted aggregates were ignored where Turn facts "
                    "allowed deterministic recomputation."
                ),
            )
        )
    _diagnose_historical_turn_states(document, diagnostics, unavailable_fields)
    compatibility = {
        "source_format": "historical_json",
        "partial": True,
        "unavailable_fields": sorted(unavailable_fields),
        "diagnostics": [item.to_document() for item in diagnostics],
    }
    document["_analysis_compatibility"] = compatibility
    return AnalysisRunArtifact(
        analysis_document=document,
        source_format="historical_json",
        artifact_source="json",
        is_partial=True,
        diagnostics=tuple(diagnostics),
        unavailable_fields=tuple(sorted(unavailable_fields)),
        export_is_current=None,
    )
