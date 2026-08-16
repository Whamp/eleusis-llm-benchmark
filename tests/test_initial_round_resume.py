"""Initial Round Checkpoint resume through Benchmark Run orchestration."""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml
from pytest import MonkeyPatch

from eleusis import (
    benchmark_run_manifest,
    evaluation_orchestrator,
    evaluation_startup,
    runner,
)
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.benchmark_run_store import BenchmarkRunStore, BenchmarkRunStoreError
from eleusis.evaluation_startup import resolve_evaluation_startup
from eleusis.evaluation_state import (
    _initialize_fresh_state,
    initialize_evaluation_state,
)
from eleusis.game.cards import Card
from eleusis.game.rule_library import RuleLibraryEntry
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup


class _StopAfterInitialCheckpointError(RuntimeError):
    """Simulate process loss on the first provider call after setup committed."""


def _resume_args(config_path: Path, run_folder: Path) -> Namespace:
    """Build the real CLI argument shape for one new-format resume."""
    return Namespace(
        config=str(config_path),
        resume=str(run_folder),
        model=None,
        num_rules=None,
        rule_index=None,
        max_turns=None,
        tag=None,
        batch_round_offset=None,
        suite=None,
    )


def _write_resume_config(path: Path, config: object) -> None:
    """Write a production-loadable YAML configuration for a resume process."""
    path.write_text(yaml.safe_dump(config))


def _create_interrupted_initial_round(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> tuple[BenchmarkConfig, Path, dict[str, object]]:
    """Create one Run that stops after setup and before its first Model Attempt."""
    monkeypatch.chdir(tmp_path)
    startup = _startup()
    startup.config["game"]["max_turns"] = 1
    startup.config["game"]["seed"] = None
    startup.game_config["max_turns"] = 1
    startup.game_config["seed"] = None
    rule_entry: RuleLibraryEntry = {
        "name": "only_red",
        "description": "Only red cards.",
        "code": "return card.color == 'red'",
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": [rule_entry]}))
    startup.config["rules"]["library_path"] = str(rules_path)
    startup.rules_config["library_path"] = str(rules_path)
    config_before_run = copy.deepcopy(startup.config)

    state = _initialize_fresh_state(startup)
    run_store = state.run_store
    assert run_store is not None
    original_execute_round_turns = runner.execute_round_turns
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), FakeLLMClient()),
    )

    def stop_before_first_model_attempt(_runtime: object) -> object:
        raise _StopAfterInitialCheckpointError("process stopped")

    monkeypatch.setattr(
        runner,
        "execute_round_turns",
        stop_before_first_model_attempt,
    )
    with pytest.raises(_StopAfterInitialCheckpointError, match="process stopped"):
        evaluation_orchestrator._run_evaluation_round(state, 1)
    monkeypatch.setattr(runner, "execute_round_turns", original_execute_round_turns)

    active = run_store.read_active_round(1)
    assert active is not None
    assert active.record["turns"] == []
    assert active.continuation["next_turn_index"] == 0
    run_folder = tmp_path / "results" / state.folder_name
    return config_before_run, run_folder, active.continuation


def test_fresh_dirty_run_warns_with_persisted_source_fingerprint(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fresh Run creation permits dirty source with an auditable warning."""
    monkeypatch.chdir(tmp_path)
    startup = _startup()
    Path("rules.json").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "only_red",
                        "description": "Only red cards.",
                        "code": "return card.color == 'red'",
                    }
                ]
            }
        )
    )
    fingerprint = "d" * 64
    monkeypatch.setattr(
        benchmark_run_manifest,
        "capture_source_provenance",
        lambda: {
            "revision": "dirty-test-revision",
            "dirty": True,
            "files": [],
            "fingerprint": fingerprint,
        },
    )

    with caplog.at_level(logging.WARNING):
        state = initialize_evaluation_state(startup)

    assert state is not None
    assert state.run_store is not None
    persisted = state.run_store.read_manifest()["source_provenance"]
    assert isinstance(persisted, dict)
    assert persisted["fingerprint"] == fingerprint
    assert "dirty source provenance" in caplog.text
    assert fingerprint in caplog.text


def test_resume_restores_initial_checkpoint_without_round_setup(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume the exact initial deal and first prompt through orchestration."""
    config, run_folder, initial_continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    config_path = tmp_path / "resume-config.yaml"
    _write_resume_config(config_path, config)
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)

    resumed_startup = resolve_evaluation_startup(_resume_args(config_path, run_folder))
    assert resumed_startup is not None
    resumed_state = initialize_evaluation_state(resumed_startup)
    assert resumed_state is not None
    assert resumed_state.start_round == 1
    assert resumed_state.run_store is not None

    game_state = initial_continuation["game_state"]
    assert isinstance(game_state, dict)
    player = game_state["player"]
    assert isinstance(player, dict)
    hand = player["hand"]
    assert isinstance(hand, list)
    selected_card = Card.from_canonical_card_data(hand[0])
    resumed_scientist = FakeLLMClient([make_action_response(str(selected_card))])
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), resumed_scientist),
    )

    def reject_repeated_setup(_request: object) -> object:
        raise AssertionError("resume repeated normal Round setup")

    monkeypatch.setattr(runner, "_prepare_round_runtime", reject_repeated_setup)

    evaluation_orchestrator._run_evaluation_round(resumed_state, 1)

    assert len(resumed_scientist.prompts_seen) == 1
    completed = resumed_state.run_store.read_completed_round(1)
    turns = completed["turns"]
    assert isinstance(turns, list)
    assert len(turns) == 1
    first_turn = turns[0]
    assert isinstance(first_turn, dict)
    assert first_turn["pre_decision_state"] == {
        "mainline": [
            {"card": card, "rejected_cards": []} for card in game_state["mainline"]
        ],
        "hand": hand,
        "turn_number": 1,
        "failed_rule_guesses": [],
    }
    attempts = first_turn["model_attempts"]
    assert isinstance(attempts, list)
    assert attempts[0]["prompt"] == resumed_scientist.prompts_seen[0]


def test_resume_initial_checkpoint_across_fresh_process(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fresh Python process resumes the first Turn without setup."""
    config, run_folder, continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    config_path = tmp_path / "fresh-process-config.yaml"
    _write_resume_config(config_path, config)
    game_state = continuation["game_state"]
    assert isinstance(game_state, dict)
    player = game_state["player"]
    assert isinstance(player, dict)
    hand = player["hand"]
    assert isinstance(hand, list)
    selected_card = str(Card.from_canonical_card_data(hand[0]))
    assert isinstance(config, dict)
    run_manifest = BenchmarkRunStore(run_folder).read_manifest()
    effective_settings = run_manifest["effective_settings"]
    assert isinstance(effective_settings, dict)
    control_config = copy.deepcopy(config)
    control_game_config = control_config["game"]
    assert isinstance(control_game_config, dict)
    control_game_config["seed"] = effective_settings["game_seed"]
    control_scientist = FakeLLMClient([make_action_response(selected_card)])
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), control_scientist),
    )
    runner.play_round(
        control_config,
        1,
        start_rule_index=0,
        batch_round_index=0,
    )
    expected_first_prompt = control_scientist.prompts_seen[0]
    input_path = tmp_path / "resume-input.json"
    output_path = tmp_path / "resume-output.json"
    input_path.write_text(
        json.dumps(
            {
                "working_directory": str(tmp_path),
                "config_path": str(config_path),
                "run_folder": str(run_folder),
                "selected_card": selected_card,
            }
        )
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.initial_round_resume_subprocess",
            str(input_path),
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )

    result = json.loads(output_path.read_text())
    assert result["prompts"] == [expected_first_prompt]
    assert result["record"]["turns"][0]["pre_decision_state"]["hand"] == hand


def test_resume_rejects_changed_scientific_setting_with_field_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A changed max Turns setting fails before opening the active checkpoint."""
    config, run_folder, _continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    config_path = tmp_path / "changed-config.yaml"
    _write_resume_config(config_path, config)
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)
    args = _resume_args(config_path, run_folder)
    args.max_turns = 2

    startup = resolve_evaluation_startup(args)

    assert startup is None
    assert (
        "Benchmark Run resume incompatible: scientific_config.game.max_turns changed"
        in caplog.text
    )


def test_resume_rejects_changed_configured_seed_with_field_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A changed configured seed fails before restoring the effective seed."""
    config, run_folder, _continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    assert isinstance(config, dict)
    game = config["game"]
    assert isinstance(game, dict)
    game["seed"] = 42
    config_path = tmp_path / "changed-seed-config.yaml"
    _write_resume_config(config_path, config)
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)

    startup = resolve_evaluation_startup(_resume_args(config_path, run_folder))

    assert startup is None
    assert (
        "Benchmark Run resume incompatible: scientific_config.game.seed changed"
        in caplog.text
    )


def test_resume_allows_operational_pause_change(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Changing pause behavior does not alter the scientific manifest."""
    config, run_folder, _continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    assert isinstance(config, dict)
    game = config["game"]
    assert isinstance(game, dict)
    game["pause_after_turn"] = True
    config_path = tmp_path / "operational-config.yaml"
    _write_resume_config(config_path, config)
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)

    startup = resolve_evaluation_startup(_resume_args(config_path, run_folder))

    assert startup is not None
    assert startup.game_config["pause_after_turn"] is True


def test_resume_rejects_continuation_round_identity_before_model_attempt(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Active-row identity mismatch fails before restoring a scientist client."""
    config, run_folder, _continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    config_path = tmp_path / "identity-config.yaml"
    _write_resume_config(config_path, config)
    database_path = run_folder / "benchmark_run.sqlite3"
    import sqlite3

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            "SELECT continuation_document FROM rounds WHERE round_number = 1"
        ).fetchone()
        assert row is not None
        continuation = json.loads(row[0])
        continuation["round"]["number"] = 99
        connection.execute(
            "UPDATE rounds SET continuation_document = ? WHERE round_number = 1",
            (json.dumps(continuation),),
        )
    row_before = database_path.read_bytes()
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)
    startup = resolve_evaluation_startup(_resume_args(config_path, run_folder))
    assert startup is not None

    with pytest.raises(BenchmarkRunStoreError, match="Round identity"):
        initialize_evaluation_state(startup)

    assert database_path.read_bytes() == row_before


def test_malformed_sqlite_resume_reports_actionable_schema_error(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A database without the Run table fails through the resume diagnostic."""
    run_folder = tmp_path / "malformed-run"
    run_folder.mkdir()
    import sqlite3

    sqlite3.connect(run_folder / "benchmark_run.sqlite3").close()
    config_path = tmp_path / "unused-config.yaml"
    config_path.write_text("{}")

    startup = resolve_evaluation_startup(_resume_args(config_path, run_folder))

    assert startup is None
    assert str(run_folder / "benchmark_run.sqlite3") in caplog.text
    assert "schema defect" in caplog.text
    assert "benchmark_run" in caplog.text


def test_active_checkpoint_column_version_fails_closed(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The SQL checkpoint version is checked before decoding continuation data."""
    _config, run_folder, _continuation = _create_interrupted_initial_round(
        monkeypatch,
        tmp_path,
    )
    import sqlite3

    with sqlite3.connect(run_folder / "benchmark_run.sqlite3") as connection:
        connection.execute(
            "UPDATE rounds SET checkpoint_version = 999 WHERE round_number = 1"
        )

    from eleusis.benchmark_run_store import BenchmarkRunStore

    store = BenchmarkRunStore(run_folder)
    with pytest.raises(
        BenchmarkRunStoreError,
        match=(
            "Benchmark Run active checkpoint incompatible: found version 999, "
            "expected 1"
        ),
    ):
        store.read_active_round(1)
