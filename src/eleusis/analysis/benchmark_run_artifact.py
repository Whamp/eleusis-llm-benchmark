"""Select authoritative Benchmark Run artifacts for analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from eleusis.analysis.historical_run import (
    AnalysisRunArtifact,
    HistoricalRunCompatibilityError,
    MigrationDiagnostic,
    decode_historical_run,
    recompute_analysis_statistics,
)
from eleusis.analysis.legacy_records import LegacyRecord
from eleusis.benchmark_run_manifest import (
    BENCHMARK_RUN_EXPORT_VERSION,
    ROUND_RECORD_VERSION,
    validate_benchmark_run_manifest_document,
)
from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
)
from eleusis.round_record import (
    RoundRecordValidationError,
    validate_round_record_document,
)


def _strict_card_text(card: Mapping[str, object]) -> str:
    """Render one validated canonical Card for legacy report compatibility."""
    rank = cast(int, card["rank"])
    suit = cast(str, card["suit"])
    rank_text = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(rank, str(rank))
    suit_text = {
        "hearts": "♥",
        "diamonds": "♦",
        "clubs": "♣",
        "spades": "♠",
    }[suit]
    return f"{rank_text}{suit_text}"


def _strict_board_text(state: Mapping[str, object]) -> str:
    """Render validated structured state without parsing historical display text."""
    positions = cast(list[Mapping[str, object]], state["mainline"])
    parts: list[str] = []
    for position in positions:
        parts.append(_strict_card_text(cast(Mapping[str, object], position["card"])))
        parts.extend(
            f"[{_strict_card_text(card)}]"
            for card in cast(
                list[Mapping[str, object]],
                position["rejected_cards"],
            )
        )
    return " ".join(parts)


def _strict_guess_legacy_view(guess_value: object) -> LegacyRecord | None:
    """Project one validated Guess Attempt into fields used by existing reports."""
    if not isinstance(guess_value, Mapping):
        return None
    guess = dict(guess_value)
    if guess.get("kind") == "formal":
        guess["shadow"] = False
        return cast(LegacyRecord, guess)
    online = guess.get("online_evaluation")
    guess["shadow"] = True
    guess["evaluated"] = isinstance(online, Mapping)
    if isinstance(online, Mapping):
        guess.update(online)
    return cast(LegacyRecord, guess)


def _strict_turn_legacy_view(turn: Mapping[str, object], player: str) -> LegacyRecord:
    """Project one strict Turn into the stable fields consumed by reports."""
    attempts = cast(list[Mapping[str, object]], turn["model_attempts"])
    usable = [
        attempt for attempt in attempts if attempt["interpretation"] == "usable_action"
    ]
    structured = usable[-1].get("structured_completion") if usable else None
    llm_response = dict(structured) if isinstance(structured, Mapping) else {}
    tokens = {
        token_name: sum(
            cast(
                int,
                cast(Mapping[str, object], attempt["token_metrics"])[token_name],
            )
            for attempt in attempts
        )
        for token_name in ("output_tokens", "reasoning_tokens", "answer_tokens")
    }
    pre_state = cast(Mapping[str, object], turn["pre_decision_state"])
    hand = cast(list[Mapping[str, object]], pre_state["hand"])
    card_outcome = cast(Mapping[str, object], turn["card_outcome"])
    return {
        "turn_number": turn["turn_number"],
        "player": player,
        "mainline_state": _strict_board_text(pre_state),
        "hand": [_strict_card_text(card) for card in hand],
        "llm_response": llm_response,
        "action_result": {"accepted": card_outcome["accepted"]},
        "guess_attempt": _strict_guess_legacy_view(turn.get("guess_attempt")),
        "tokens": tokens,
        "retry_count": max(0, len(attempts) - 1),
        "retry_causes": [],
    }


def _strict_config_legacy_view(manifest: Mapping[str, object]) -> LegacyRecord:
    """Project immutable strict settings into existing report configuration fields."""
    scientific = cast(Mapping[str, object], manifest["scientific_config"])
    game = cast(Mapping[str, object], scientific["game"])
    model = cast(Mapping[str, object], manifest["model_identity"])
    compiler = cast(Mapping[str, object], manifest["compiler_identity"])
    effective = cast(Mapping[str, object], manifest["effective_settings"])
    return {
        "player": model["display_name"],
        "player_model": model["model_key"],
        "rule_compiler": compiler["display_name"],
        "rule_compiler_provider": compiler["provider"],
        "rule_compiler_model_id": compiler["model_id"],
        "max_turns": effective["max_turns"],
        "wrong_guess_penalty": effective["wrong_guess_penalty"],
        "hand_size": effective["hand_size"],
        "seed": effective["game_seed"],
        "num_rounds_per_rule": game.get("num_rounds_per_rule", 1),
        "num_rules": game.get(
            "num_rules", len(cast(list[object], manifest["schedule"]))
        ),
    }


def _strict_round_legacy_view(
    record: Mapping[str, object],
    player: str,
) -> LegacyRecord:
    """Project a validated completed Round while recomputing every derived value."""
    derived = BenchmarkRunStore.derive_round_values(record)
    secret_rule = cast(Mapping[str, object], record["secret_rule"])
    settings = cast(Mapping[str, object], record["settings"])
    outcome = cast(Mapping[str, object], record["terminal_outcome"])
    turns = cast(list[Mapping[str, object]], record["turns"])
    usage = cast(Mapping[str, object], derived["usage"])
    game_over_reason = {
        "turn_limit": "max_turns",
        "correct_formal_guess": "correct_guess",
        "abandoned": "abandoned",
        "impossible_to_continue": "impossible_to_continue",
    }[cast(str, outcome["kind"])]
    return {
        "round_number": record["scheduled_round_number"],
        "rule_name": secret_rule["name"],
        "batch_round_index": settings["batch_round_index"],
        "rule_description": secret_rule["description"],
        "rule_code": secret_rule["code"],
        **derived,
        "success": outcome["kind"] == "correct_formal_guess",
        "game_over_reason": game_over_reason,
        "llm_usage": {
            "player": {
                "prompt_tokens": usage["prompt_tokens"],
                "output_tokens": usage["output_tokens"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "answer_tokens": usage["answer_tokens"],
            }
        },
        "turns": [_strict_turn_legacy_view(turn, player) for turn in turns],
        "wall_clock_seconds": usage["duration_seconds"],
    }


def _strict_analysis_document(
    manifest: Mapping[str, object],
    records: list[dict[str, object]],
) -> LegacyRecord:
    """Build the existing report view from strict manifest and Round facts."""
    config = _strict_config_legacy_view(manifest)
    player = cast(str, config["player"])
    rounds = [_strict_round_legacy_view(record, player) for record in records]
    schedule = cast(list[Mapping[str, object]], manifest["schedule"])
    rules_library: list[LegacyRecord] = []
    seen_rules: set[tuple[object, object]] = set()
    for scheduled in schedule:
        identity = (scheduled.get("rule_name"), scheduled.get("rule_code"))
        if identity in seen_rules:
            continue
        seen_rules.add(identity)
        rules_library.append(
            {
                "name": scheduled.get("rule_name"),
                "description": scheduled.get("rule_description"),
                "code": scheduled.get("rule_code"),
            }
        )
    consumed_rules = []
    for rule in rules_library:
        rule_code = rule.get("code")
        consumed_rules.append(
            {
                **rule,
                "rounds_completed": sum(
                    cast(Mapping[str, object], record["secret_rule"])["code"]
                    == rule_code
                    for record in records
                ),
            }
        )
    document: LegacyRecord = {
        "config": config,
        "rounds": rounds,
        "statistics": {},
        "checkpoint": {
            "rules_library": rules_library,
            "rules_consumed": consumed_rules,
        },
    }
    recompute_analysis_statistics(document)
    return document


def _unsupported_artifact(
    diagnostic: MigrationDiagnostic,
) -> AnalysisRunArtifact:
    """Return an explicit no-view policy for an unsupported versioned artifact."""
    return AnalysisRunArtifact(
        analysis_document=None,
        source_format="unsupported_export",
        artifact_source="json",
        is_partial=True,
        diagnostics=(diagnostic,),
        unavailable_fields=("analysis_document",),
        export_is_current=None,
    )


def _strict_artifact(
    manifest: Mapping[str, object],
    records: list[dict[str, object]],
    *,
    artifact_source: Literal["json", "sqlite"],
    export_is_current: bool | None,
) -> AnalysisRunArtifact:
    """Build a strict analysis artifact and describe coexisting export freshness."""
    diagnostics: list[MigrationDiagnostic] = []
    if artifact_source == "sqlite" and export_is_current is False:
        diagnostics.append(
            MigrationDiagnostic(
                code="strict_export_stale",
                path="results.json.watermark",
                message="SQLite is authoritative; the coexisting JSON export is stale.",
            )
        )
    document = _strict_analysis_document(manifest, records)
    document["_analysis_compatibility"] = {
        "source_format": "strict_round_record_export",
        "partial": False,
        "artifact_source": artifact_source,
        "export_is_current": export_is_current,
        "unavailable_fields": [],
        "diagnostics": [item.to_document() for item in diagnostics],
    }
    return AnalysisRunArtifact(
        analysis_document=document,
        source_format="strict_round_record_export",
        artifact_source=artifact_source,
        is_partial=False,
        diagnostics=tuple(diagnostics),
        unavailable_fields=(),
        export_is_current=export_is_current,
    )


def _decode_strict_export(document: Mapping[str, object]) -> AnalysisRunArtifact:
    """Apply explicit compatibility policy to one portable versioned export."""
    export_version = document.get("version")
    if export_version != BENCHMARK_RUN_EXPORT_VERSION:
        return _unsupported_artifact(
            MigrationDiagnostic(
                code="unsupported_benchmark_run_export_version",
                path="version",
                message=(
                    "Benchmark Run export version is unsupported: found "
                    f"{export_version!r}, expected {BENCHMARK_RUN_EXPORT_VERSION}."
                ),
            )
        )
    records_value = document.get("completed_round_records")
    if not isinstance(records_value, list):
        return _unsupported_artifact(
            MigrationDiagnostic(
                code="invalid_benchmark_run_export",
                path="completed_round_records",
                message="Benchmark Run export has no completed Round Record list.",
            )
        )
    records: list[dict[str, object]] = []
    for index, record_value in enumerate(records_value):
        record_version = (
            record_value.get("version") if isinstance(record_value, Mapping) else None
        )
        if record_version != ROUND_RECORD_VERSION:
            return _unsupported_artifact(
                MigrationDiagnostic(
                    code="unsupported_completed_round_record_version",
                    path=f"completed_round_records[{index}].version",
                    message=(
                        "Completed Round Record version is unsupported: "
                        f"found {record_version!r}, expected {ROUND_RECORD_VERSION}."
                    ),
                )
            )
        try:
            records.append(validate_round_record_document(record_value))
        except RoundRecordValidationError as error:
            return _unsupported_artifact(
                MigrationDiagnostic(
                    code="invalid_completed_round_record",
                    path=f"completed_round_records[{index}]",
                    message=str(error),
                )
            )
    try:
        manifest = validate_benchmark_run_manifest_document(document.get("run"))
    except ValueError as error:
        return _unsupported_artifact(
            MigrationDiagnostic(
                code="invalid_benchmark_run_manifest",
                path="run",
                message=str(error),
            )
        )
    return _strict_artifact(
        manifest,
        records,
        artifact_source="json",
        export_is_current=None,
    )


def read_analysis_run_artifact(run_folder: Path) -> AnalysisRunArtifact:
    """Select SQLite before JSON and decode one Run without rewriting artifacts."""
    database_path = run_folder / BENCHMARK_RUN_DATABASE_NAME
    if database_path.is_file():
        run_store = BenchmarkRunStore(run_folder)
        status = run_store.read_export_status()
        return _strict_artifact(
            run_store.read_manifest(),
            run_store.read_completed_rounds(),
            artifact_source="sqlite",
            export_is_current=status.is_current,
        )
    results_path = run_folder / "results.json"
    if not results_path.is_file():
        raise HistoricalRunCompatibilityError(
            "Benchmark Run analysis artifact unavailable: "
            f"{results_path} does not exist"
        )
    try:
        document = json.loads(results_path.read_text())
    except json.JSONDecodeError as error:
        raise HistoricalRunCompatibilityError(
            f"Benchmark Run analysis artifact is invalid JSON: {error}"
        ) from error
    if isinstance(document, Mapping) and "version" in document:
        return _decode_strict_export(document)
    return decode_historical_run(document)
