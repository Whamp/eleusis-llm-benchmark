"""Historical JSON compatibility and authoritative artifact selection."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from eleusis import evaluation_orchestrator, evaluation_startup, runner
from eleusis.analysis.benchmark_run_artifact import read_analysis_run_artifact
from eleusis.analysis.historical_run import HistoricalRunCompatibilityError
from eleusis.analysis.loader import load_results
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.benchmark_run_store import (
    BENCHMARK_RUN_DATABASE_NAME,
    BenchmarkRunStore,
)
from eleusis.evaluation_startup import resolve_evaluation_startup
from eleusis.evaluation_state import EvaluationState, _initialize_fresh_state
from eleusis.game.metrics import code_complexity
from eleusis.game.rule_library import RuleLibraryEntry
from scripts.status_report import load_worker_results
from tests.conftest import FakeLLMClient, ScriptedResponse, make_action_response
from tests.test_benchmark_run_schedule import (
    _configure_two_round_schedule,
    _first_dealt_card,
)
from tests.test_resume import _make_checkpoint

FIXTURES = Path(__file__).parent / "fixtures"


def _resume_args(run_folder: Path) -> Namespace:
    """Build CLI arguments for one attempted historical Run resume."""
    return Namespace(
        config="config.yaml",
        resume=str(run_folder),
        model=None,
        num_rules=None,
        rule_index=None,
        max_turns=None,
        tag=None,
        batch_round_offset=None,
        suite=None,
    )


def test_json_only_run_resume_is_refused_without_creating_sqlite(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Refuse implicit legacy import before preflight or filesystem mutation."""
    run_folder = tmp_path / "solo_evaluation_legacy"
    run_folder.mkdir()
    results_path = run_folder / "results.json"
    results_path.write_text(json.dumps(_make_checkpoint()))
    original_bytes = results_path.read_bytes()
    preflight_models: list[str] = []
    monkeypatch.setattr(
        evaluation_startup,
        "preflight_check",
        lambda model: preflight_models.append(model),
    )

    with caplog.at_level(logging.ERROR):
        startup = resolve_evaluation_startup(_resume_args(run_folder))

    assert startup is None
    assert preflight_models == []
    assert results_path.read_bytes() == original_bytes
    assert not (run_folder / BENCHMARK_RUN_DATABASE_NAME).exists()
    assert "Historical JSON-only Benchmark Run cannot be resumed" in caplog.text
    assert "Start a new Run; legacy import is not implemented" in caplog.text


@pytest.mark.parametrize(
    "fixture_name",
    [
        "historical_complete_results.json",
        "historical_missing_optional_results.json",
    ],
)
def test_historical_formats_load_as_explicit_partial_views_without_rewrite(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    """Keep supported old result shapes readable without inventing strict facts."""
    run_folder = tmp_path / "solo_evaluation_historical"
    run_folder.mkdir()
    results_path = run_folder / "results.json"
    shutil.copyfile(FIXTURES / fixture_name, results_path)
    original_bytes = results_path.read_bytes()

    artifact = read_analysis_run_artifact(run_folder)

    assert artifact.source_format == "historical_json"
    assert artifact.artifact_source == "json"
    assert artifact.is_partial is True
    assert artifact.analysis_document is not None
    assert "round_record.run_id" in artifact.unavailable_fields
    assert "round_record.structured_states" in artifact.unavailable_fields
    assert any(
        diagnostic.code == "historical_round_record_partial"
        for diagnostic in artifact.diagnostics
    )
    if fixture_name == "historical_missing_optional_results.json":
        assert "rounds[0].wall_clock_seconds" in artifact.unavailable_fields
        assert "rounds[0].turns[0].tokens" in artifact.unavailable_fields
    assert results_path.read_bytes() == original_bytes
    assert not (run_folder / BENCHMARK_RUN_DATABASE_NAME).exists()


def test_invalid_historical_compact_state_is_diagnosed_without_reconstruction(
    tmp_path: Path,
) -> None:
    """Expose malformed display state as unavailable instead of guessing a board."""
    run_folder = tmp_path / "solo_evaluation_invalid"
    run_folder.mkdir()
    shutil.copyfile(
        FIXTURES / "historical_invalid_compact_state_results.json",
        run_folder / "results.json",
    )

    artifact = read_analysis_run_artifact(run_folder)

    diagnostic = next(
        item
        for item in artifact.diagnostics
        if item.code == "historical_compact_state_invalid"
    )
    assert diagnostic.path == "rounds[0].turns[0].mainline_state"
    assert "rounds[0].turns[0].pre_decision_state" in artifact.unavailable_fields


def test_historical_aggregates_are_recomputed_from_available_turn_facts(
    tmp_path: Path,
) -> None:
    """Ignore stale imported scores and token totals when Turn facts suffice."""
    run_folder = tmp_path / "solo_evaluation_stale"
    run_folder.mkdir()
    shutil.copyfile(
        FIXTURES / "historical_stale_aggregates_results.json",
        run_folder / "results.json",
    )

    artifact = read_analysis_run_artifact(run_folder)

    assert artifact.analysis_document is not None
    round_data = artifact.analysis_document["rounds"][0]
    assert round_data["turn_count"] == 2
    assert round_data["success"] is True
    assert round_data["score"] == 4
    assert round_data["floored_score"] == 4
    assert round_data["first_correct_turn"] == 2
    assert round_data["failed_guesses"] == 0
    assert round_data["llm_usage"]["player"]["output_tokens"] == 12
    assert artifact.analysis_document["statistics"]["total_score"] == 4
    assert any(
        diagnostic.code == "historical_derived_values_recomputed"
        for diagnostic in artifact.diagnostics
    )


def test_existing_analysis_loader_marks_historical_views_as_partial(
    tmp_path: Path,
) -> None:
    """Let current reports identify compatibility data without changing their API."""
    run_folder = tmp_path / "solo_evaluation_historical_complete"
    run_folder.mkdir()
    shutil.copyfile(
        FIXTURES / "historical_complete_results.json",
        run_folder / "results.json",
    )

    results, folder_names = load_results(tmp_path)

    assert folder_names == [run_folder.name]
    assert len(results) == 1
    compatibility = results[0]["_analysis_compatibility"]
    assert compatibility["source_format"] == "historical_json"
    assert compatibility["partial"] is True


def _complete_one_strict_round(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    scientist_responses: Callable[
        [BenchmarkConfig, list[RuleLibraryEntry]], list[ScriptedResponse]
    ]
    | None = None,
    max_llm_retries: int = 1,
) -> EvaluationState:
    """Create one completed strict Round through production orchestration."""
    monkeypatch.chdir(tmp_path)
    startup, config, _config_path, rules = _configure_two_round_schedule(tmp_path)
    startup.num_rules = 1
    startup.num_rounds = 1
    startup.config["game"].update({"num_rules": 1, "num_rounds": 1})
    startup.game_config.update({"num_rules": 1, "num_rounds": 1})
    startup.config["llm"]["max_llm_retries"] = max_llm_retries
    state = _initialize_fresh_state(startup)
    assert state.run_store is not None
    responses = (
        scientist_responses(config, rules)
        if scientist_responses is not None
        else [make_action_response(_first_dealt_card(config, rules[0]))]
    )
    scientist = FakeLLMClient(responses)
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), scientist),
    )
    evaluation_orchestrator._run_evaluation_round(state, 1)
    return state


def _completed_strict_store(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> BenchmarkRunStore:
    """Return the narrowed store of one completed strict Round."""
    state = _complete_one_strict_round(monkeypatch, tmp_path)
    assert state.run_store is not None
    return state.run_store


def test_sqlite_is_authoritative_when_the_coexisting_export_is_stale(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read strict records from SQLite and report, but do not trust, stale JSON."""
    run_store = _completed_strict_store(monkeypatch, tmp_path)
    stale_export = json.loads(run_store.export_path.read_text())
    stale_export["watermark"] = -1
    run_store.export_path.write_text(json.dumps(stale_export))
    stale_bytes = run_store.export_path.read_bytes()

    artifact = read_analysis_run_artifact(run_store.run_folder)

    assert artifact.artifact_source == "sqlite"
    assert artifact.source_format == "strict_round_record_export"
    assert artifact.is_partial is False
    assert artifact.export_is_current is False
    assert artifact.analysis_document is not None
    assert len(artifact.analysis_document["rounds"]) == 1
    assert artifact.analysis_document["_analysis_compatibility"] == {
        "source_format": "strict_round_record_export",
        "partial": False,
        "artifact_source": "sqlite",
        "export_is_current": False,
        "unavailable_fields": [],
        "diagnostics": [
            {
                "code": "strict_export_stale",
                "path": "results.json.watermark",
                "message": (
                    "SQLite is authoritative; the coexisting JSON export is stale."
                ),
            }
        ],
    }
    assert run_store.export_path.read_bytes() == stale_bytes
    loaded, _folder_names = load_results(run_store.run_folder.parent)
    assert loaded[0]["_analysis_compatibility"]["artifact_source"] == "sqlite"
    assert loaded[0]["_analysis_compatibility"]["export_is_current"] is False
    status_results = load_worker_results([run_store.run_folder])
    assert len(status_results) == 1
    assert status_results[0]["rounds"][0]["score"] == 0


def test_portable_strict_export_loads_without_its_sqlite_database(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decode known completed records through strict validation for interchange."""
    run_store = _completed_strict_store(monkeypatch, tmp_path)
    portable_folder = tmp_path / "portable" / "solo_evaluation_portable"
    portable_folder.mkdir(parents=True)
    shutil.copyfile(run_store.export_path, portable_folder / "results.json")

    artifact = read_analysis_run_artifact(portable_folder)

    assert artifact.artifact_source == "json"
    assert artifact.source_format == "strict_round_record_export"
    assert artifact.is_partial is False
    assert artifact.export_is_current is None
    assert artifact.analysis_document is not None
    assert artifact.analysis_document["rounds"][0]["score"] == 0


@pytest.mark.parametrize(
    ("document", "diagnostic_code", "diagnostic_path"),
    [
        (
            {"version": 99},
            "unsupported_benchmark_run_export_version",
            "version",
        ),
        (
            {
                "version": 1,
                "run": {},
                "completed_round_records": [{"version": 99}],
                "shadow_verdicts": [],
                "derived": {"rounds": [], "summary": {}},
                "watermark": 1,
            },
            "unsupported_completed_round_record_version",
            "completed_round_records[0].version",
        ),
    ],
)
def test_unknown_completed_record_and_export_versions_have_explicit_policy(
    document: dict[str, object],
    diagnostic_code: str,
    diagnostic_path: str,
    tmp_path: Path,
) -> None:
    """Return diagnostics for unknown historical versions without lax validation."""
    run_folder = tmp_path / "solo_evaluation_unknown_version"
    run_folder.mkdir()
    (run_folder / "results.json").write_text(json.dumps(document))

    artifact = read_analysis_run_artifact(run_folder)

    assert artifact.is_partial is True
    assert artifact.analysis_document is None
    assert any(
        item.code == diagnostic_code and item.path == diagnostic_path
        for item in artifact.diagnostics
    )


def test_defective_sqlite_run_is_skipped_as_incompatible(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One unreadable store must not abort reporting over healthy sibling Runs."""
    good_store = _completed_strict_store(monkeypatch, tmp_path)
    broken_folder = good_store.run_folder.parent / "solo_evaluation_broken"
    broken_folder.mkdir()
    sqlite3.connect(broken_folder / BENCHMARK_RUN_DATABASE_NAME).close()

    with pytest.raises(HistoricalRunCompatibilityError, match="schema defect"):
        read_analysis_run_artifact(broken_folder)

    _loaded, folder_names = load_results(good_store.run_folder.parent)
    assert folder_names == [good_store.run_folder.name]
    status_results = load_worker_results([broken_folder, good_store.run_folder])
    assert len(status_results) == 1
    good_view = cast(dict[str, object], status_results[0])
    compatibility = cast(dict[str, object], good_view["_analysis_compatibility"])
    assert compatibility["artifact_source"] == "sqlite"


def test_retry_exhausted_fallback_keeps_every_attempt_cause(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Count all exhausted attempts as retries instead of one fewer."""
    state = _complete_one_strict_round(
        monkeypatch,
        tmp_path,
        scientist_responses=lambda _config, _rules: [
            make_action_response("BANANA"),
            make_action_response("BANANA"),
            make_action_response("BANANA"),
        ],
        max_llm_retries=3,
    )
    assert state.run_store is not None

    summary = state.run_store.read_derived_summary()

    assert summary["total_retries"] == 3
    assert summary["retry_by_cause"] == {"card_parse_error": 3}
    artifact = read_analysis_run_artifact(state.run_store.run_folder)
    assert artifact.analysis_document is not None
    turn = artifact.analysis_document["rounds"][0]["turns"][0]
    assert turn["retry_count"] == 3
    assert turn["retry_causes"] == [
        {"attempt": 1, "cause": "card_parse_error"},
        {"attempt": 2, "cause": "card_parse_error"},
        {"attempt": 3, "cause": "card_parse_error"},
    ]


def test_retry_causes_survive_strict_persistence_and_reports(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    tmp_path: Path,
) -> None:
    """Keep retry counts and causes in derived summaries, logs, and legacy views."""
    state = _complete_one_strict_round(
        monkeypatch,
        tmp_path,
        scientist_responses=lambda config, rules: [
            make_action_response("BANANA"),
            make_action_response(_first_dealt_card(config, rules[0])),
        ],
    )
    assert state.run_store is not None

    with caplog.at_level(logging.INFO, logger="eleusis.evaluation_orchestrator"):
        evaluation_orchestrator._finalize_evaluation(state)

    summary = state.run_store.read_derived_summary()
    assert summary["total_retries"] == 1
    assert summary["retry_by_cause"] == {"card_parse_error": 1}
    artifact = read_analysis_run_artifact(state.run_store.run_folder)
    assert artifact.analysis_document is not None
    turn = artifact.analysis_document["rounds"][0]["turns"][0]
    assert turn["retry_count"] == 1
    assert turn["retry_causes"] == [{"attempt": 1, "cause": "card_parse_error"}]
    assert "Total LLM retries: 1" in caplog.text
    assert "card_parse_error: 1" in caplog.text


def test_strict_analysis_view_computes_rule_complexity_metrics(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Strict views carry static rule complexity like legacy runs did.

    Legacy results.json embedded node_count/cyclomatic_complexity per rule in
    checkpoint.rules_library. Strict Round Records only persist the rule code,
    so the analysis view must compute those metrics from the code. Without
    them, complexity analysis crashes on a missing complexity_bin column.
    """
    store = _completed_strict_store(monkeypatch, tmp_path)
    artifact = read_analysis_run_artifact(store.run_folder)
    assert artifact.analysis_document is not None
    library = artifact.analysis_document["checkpoint"]["rules_library"]
    assert library, "expected at least one scheduled rule"
    for entry in library:
        code = entry["code"]
        assert isinstance(code, str)
        expected = code_complexity(code)
        assert entry["node_count"] == expected["node_count"]
        assert entry["cyclomatic_complexity"] == expected["cyclomatic"]
