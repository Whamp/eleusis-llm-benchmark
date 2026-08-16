"""Complete ordered Benchmark Run schedules through authoritative SQLite state."""

from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import cast

import pytest
import yaml
from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator, evaluation_startup, runner
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.benchmark_run_store import BenchmarkRunStore
from eleusis.evaluation_startup import EvaluationStartup, resolve_evaluation_startup
from eleusis.evaluation_state import (
    _initialize_fresh_state,
    initialize_evaluation_state,
)
from eleusis.game.cards import Deck
from eleusis.game.rule_library import RuleLibraryEntry
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup


def _resume_args(config_path: Path, run_folder: Path) -> Namespace:
    """Build production CLI arguments for one authoritative Run resume."""
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


def _configure_two_round_schedule(
    root: Path,
) -> tuple[EvaluationStartup, BenchmarkConfig, Path, list[RuleLibraryEntry]]:
    """Create two deterministic one-Turn Rounds with different secret rules."""
    startup = _startup()
    startup.num_rules = 2
    startup.num_rounds = 2
    startup.config["game"].update(
        {"num_rules": 2, "num_rounds": 2, "num_rounds_per_rule": 1, "max_turns": 1}
    )
    startup.game_config.update(
        {"num_rules": 2, "num_rounds": 2, "num_rounds_per_rule": 1, "max_turns": 1}
    )
    rules: list[RuleLibraryEntry] = [
        {
            "name": "only_red",
            "description": "Only red cards.",
            "code": "return card.color == 'red'",
        },
        {
            "name": "only_even",
            "description": "Only even ranks.",
            "code": "return card.rank % 2 == 0",
        },
    ]
    rules_path = root / "rules.json"
    rules_path.write_text(json.dumps({"rules": rules}))
    startup.config["rules"]["library_path"] = str(rules_path)
    startup.rules_config["library_path"] = str(rules_path)
    config = copy.deepcopy(startup.config)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return startup, config, config_path, rules


def _first_dealt_card(
    config: BenchmarkConfig,
    rule: RuleLibraryEntry,
    *,
    batch_round_index: int = 0,
) -> str:
    """Return the first deterministic hand card for one scheduled Round."""
    rule_hash = int(hashlib.md5(rule["code"].encode()).hexdigest(), 16) & 0xFFFFFFFF
    game_seed = config["game"]["seed"]
    assert isinstance(game_seed, int)
    round_seed = (game_seed + rule_hash + batch_round_index) & 0xFFFFFFFF
    deck = Deck()
    deck.shuffle(seed=round_seed)
    return str(deck.draw())


def test_fresh_orchestration_runs_and_finalizes_complete_persisted_schedule(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run every scheduled Round through the public Benchmark Run entry point."""
    monkeypatch.chdir(tmp_path)
    startup, config, _config_path, rules = _configure_two_round_schedule(tmp_path)
    scientists = iter(
        FakeLLMClient([make_action_response(_first_dealt_card(config, rule))])
        for rule in rules
    )
    monkeypatch.setattr(
        evaluation_orchestrator,
        "resolve_evaluation_startup",
        lambda _args: startup,
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), next(scientists)),
    )

    evaluation_orchestrator.run_evaluation(Namespace())

    run_folder = tmp_path / "results" / "solo_evaluation_20260815_000000_test"
    run_store = BenchmarkRunStore(run_folder)
    assert run_store.read_progress().is_complete is True
    assert [
        record["scheduled_round_number"] for record in run_store.read_completed_rounds()
    ] == [1, 2]
    exported = json.loads(run_store.export_path.read_text())
    assert exported["derived"]["summary"]["completed_rounds"] == 2

    run_store.export_path.unlink()
    monkeypatch.setattr(
        evaluation_orchestrator,
        "_run_evaluation_round",
        lambda *_args: (_ for _ in ()).throw(AssertionError("Round replayed")),
    )
    evaluation_orchestrator.run_evaluation(Namespace())
    regenerated = json.loads(run_store.export_path.read_text())
    assert regenerated == exported


def test_resume_after_completed_round_runs_next_persisted_schedule_entry(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resume Round 2 when Round 1 is terminal and no active checkpoint remains."""
    monkeypatch.chdir(tmp_path)
    startup, config, config_path, rules = _configure_two_round_schedule(tmp_path)
    state = _initialize_fresh_state(startup)
    run_store = state.run_store
    assert run_store is not None
    first_scientist = FakeLLMClient(
        [make_action_response(_first_dealt_card(config, rules[0]))]
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), first_scientist),
    )

    evaluation_orchestrator._run_evaluation_round(state, 1)

    assert run_store.read_active_round(1) is None
    assert [
        record["scheduled_round_number"] for record in run_store.read_completed_rounds()
    ] == [1]
    progress = run_store.read_progress()
    assert progress.completed_rounds == 1
    assert progress.active_round_number is None
    assert progress.committed_turns == 0
    assert progress.next_round_number == 2
    assert progress.total_rounds == 2
    assert progress.is_complete is False
    first_export = json.loads(run_store.export_path.read_text())
    first_watermark = first_export["watermark"]
    assert len(first_export["completed_round_records"]) == 1
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)
    resumed_startup = resolve_evaluation_startup(
        _resume_args(config_path, run_store.run_folder)
    )
    assert resumed_startup is not None

    resumed_state = initialize_evaluation_state(resumed_startup)

    assert resumed_state is not None
    assert resumed_state.start_round == 2
    second_scientist = FakeLLMClient(
        [make_action_response(_first_dealt_card(config, rules[1]))]
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), second_scientist),
    )
    evaluation_orchestrator._run_evaluation_round(resumed_state, 2)

    records = run_store.read_completed_rounds()
    assert [record["scheduled_round_number"] for record in records] == [1, 2]
    secret_rules = [
        cast(dict[str, object], record["secret_rule"]) for record in records
    ]
    assert [secret_rule["name"] for secret_rule in secret_rules] == [
        "only_red",
        "only_even",
    ]
    completed_progress = run_store.read_progress()
    assert completed_progress.completed_rounds == 2
    assert completed_progress.next_round_number is None
    assert completed_progress.is_complete is True
    final_export = json.loads(run_store.export_path.read_text())
    assert final_export["watermark"] > first_watermark
    assert len(final_export["completed_round_records"]) == 2
    assert final_export["derived"]["summary"] == {
        "completed_rounds": 2,
        "successful_rounds": 0,
        "failed_rounds": 2,
        "success_rate": 0.0,
        "total_score": 0,
        "average_score": 0.0,
        "total_turns": 2,
        "average_turns": 1.0,
        "average_turns_when_successful": 0.0,
        "total_failed_guesses": 0,
        "average_failed_guesses": 0.0,
        "schema_compliance_rate": 1.0,
        "total_retries": 0,
        "retry_by_cause": {},
        "usage": {
            "model_attempt_count": 2,
            "provider_call_count": 2,
            "prompt_tokens": 200,
            "output_tokens": 100,
            "reasoning_tokens": 60,
            "answer_tokens": 40,
            "duration_seconds": 0.2,
            "throughput_tokens_per_second": 500.0,
            "cost": {"usd": None, "pricing_version": None},
        },
    }


def test_random_rule_selection_is_fixed_in_complete_round_batches(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve random rule choices into a concrete immutable schedule up front."""
    monkeypatch.chdir(tmp_path)
    startup, _config, _config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.num_rounds_per_rule = 2
    startup.num_rounds = 4
    startup.rules_config["selection"] = "random"
    startup.config["rules"]["selection"] = "random"
    startup.game_config["num_rounds_per_rule"] = 2
    startup.game_config["num_rounds"] = 4
    startup.config["game"]["num_rounds_per_rule"] = 2
    startup.config["game"]["num_rounds"] = 4

    state = _initialize_fresh_state(startup)

    assert state.run_store is not None
    manifest = state.run_store.read_manifest()
    schedule = manifest["schedule"]
    assert isinstance(schedule, list)
    scheduled_rules = [
        (entry["rule_name"], entry["rule_description"], entry["rule_code"])
        for entry in schedule
    ]
    assert all(
        all(isinstance(value, str) for value in rule) for rule in scheduled_rules
    )
    assert scheduled_rules[0] == scheduled_rules[1]
    assert scheduled_rules[2] == scheduled_rules[3]
    assert [entry["batch_round_index"] for entry in schedule] == [0, 1, 0, 1]


def test_named_suite_schedule_preserves_rule_order_and_batch_indices(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist named-suite cases exactly instead of reselecting them at runtime."""
    monkeypatch.chdir(tmp_path)
    startup, _config, _config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.suite_name = "focused_suite"
    startup.config["suite"] = "focused_suite"
    startup.suite_cases = [("only_even", 2), ("only_red", 0)]

    state = _initialize_fresh_state(startup)

    assert state.run_store is not None
    manifest = state.run_store.read_manifest()
    assert manifest["schedule"] == [
        {
            "round_number": 1,
            "rule_name": "only_even",
            "rule_description": "Only even ranks.",
            "rule_code": "return card.rank % 2 == 0",
            "batch_round_index": 2,
        },
        {
            "round_number": 2,
            "rule_name": "only_red",
            "rule_description": "Only red cards.",
            "rule_code": "return card.color == 'red'",
            "batch_round_index": 0,
        },
    ]


def test_bare_resume_restores_cli_created_suite_from_manifest(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitted suite CLI input does not conflict with its stored fixed value."""
    monkeypatch.chdir(tmp_path)
    startup, _config, config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.suite_name = "focused_suite"
    startup.config["suite"] = "focused_suite"
    startup.suite_cases = [("only_even", 2), ("only_red", 0)]
    state = _initialize_fresh_state(startup)
    assert state.run_store is not None
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)

    resumed = resolve_evaluation_startup(
        _resume_args(config_path, state.run_store.run_folder)
    )

    assert resumed is not None
    assert resumed.suite_name == "focused_suite"
    assert resumed.suite_cases == [("only_even", 2), ("only_red", 0)]


def test_bare_resume_restores_cli_batch_offset_from_manifest(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitted worker offset does not conflict with its stored fixed value."""
    monkeypatch.chdir(tmp_path)
    startup, _config, config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.game_config["batch_round_offset"] = 3
    startup.config["game"]["batch_round_offset"] = 3
    startup.game_config["num_rounds_per_rule"] = 1
    startup.config["game"]["num_rounds_per_rule"] = 1
    state = _initialize_fresh_state(startup)
    assert state.run_store is not None
    monkeypatch.setattr(evaluation_startup, "preflight_check", lambda _model: None)

    resumed = resolve_evaluation_startup(
        _resume_args(config_path, state.run_store.run_folder)
    )

    assert resumed is not None
    assert resumed.game_config["batch_round_offset"] == 3
    assert resumed.run_manifest is not None
    schedule = cast(list[dict[str, object]], resumed.run_manifest["schedule"])
    assert {entry["batch_round_index"] for entry in schedule} == {3}


def test_live_progress_command_reads_active_round_from_sqlite(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report an initial checkpoint even before results.json can contain a Round."""
    monkeypatch.chdir(tmp_path)
    startup, _config, _config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.output_tag = "w0_qwen"
    state = _initialize_fresh_state(startup)
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), FakeLLMClient()),
    )

    def stop_after_initial_checkpoint(_runtime: object) -> object:
        raise RuntimeError("stop after initial checkpoint")

    monkeypatch.setattr(runner, "execute_round_turns", stop_after_initial_checkpoint)
    with pytest.raises(RuntimeError, match="stop after initial checkpoint"):
        evaluation_orchestrator._run_evaluation_round(state, 1)

    from scripts import check_progress

    monkeypatch.setattr(
        "sys.argv", ["check_progress.py", "--pattern", "solo_evaluation_*"]
    )
    check_progress.main()

    output = capsys.readouterr().out
    assert "0/2" in output
    assert "active Round 1, 0 committed Turns" in output


def test_parallel_workers_keep_separate_stores_offsets_and_effective_seeds(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Worker offsets remain isolated in per-folder stores and Round Records."""
    monkeypatch.chdir(tmp_path)
    worker_states = []
    worker_configs: list[BenchmarkConfig] = []
    worker_rules: list[list[RuleLibraryEntry]] = []
    for worker_index in (0, 1):
        worker_root = tmp_path / f"worker-{worker_index}"
        worker_root.mkdir()
        startup, config, _config_path, rules = _configure_two_round_schedule(
            worker_root
        )
        startup.output_tag = f"w{worker_index}"
        startup.game_config["batch_round_offset"] = worker_index
        startup.config["game"]["batch_round_offset"] = worker_index
        state = _initialize_fresh_state(startup)
        worker_states.append(state)
        worker_configs.append(config)
        worker_rules.append(rules)

    first_store = worker_states[0].run_store
    second_store = worker_states[1].run_store
    assert first_store is not None
    assert second_store is not None
    assert first_store.database_path != second_store.database_path
    first_schedule = first_store.read_manifest()["schedule"]
    second_schedule = second_store.read_manifest()["schedule"]
    assert isinstance(first_schedule, list)
    assert isinstance(second_schedule, list)
    assert first_schedule[0]["batch_round_index"] == 0
    assert second_schedule[0]["batch_round_index"] == 1

    for worker_index, state in enumerate(worker_states):
        scientist = FakeLLMClient(
            [
                make_action_response(
                    _first_dealt_card(
                        worker_configs[worker_index],
                        worker_rules[worker_index][0],
                        batch_round_index=worker_index,
                    )
                )
            ]
        )
        monkeypatch.setattr(
            runner,
            "_create_round_clients",
            lambda _config, _player_name, client=scientist: (
                FakeLLMClient(),
                client,
            ),
        )
        evaluation_orchestrator._run_evaluation_round(state, 1)

    first_record = first_store.read_completed_round(1)
    second_record = second_store.read_completed_round(1)
    first_settings = first_record["settings"]
    second_settings = second_record["settings"]
    assert isinstance(first_settings, dict)
    assert isinstance(second_settings, dict)
    assert first_settings["batch_round_index"] == 0
    assert second_settings["batch_round_index"] == 1
    assert (
        second_settings["effective_round_seed"]
        == first_settings["effective_round_seed"] + 1
    )


def test_watch_mode_refreshes_until_interrupted(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--watch redraws the live report and exits cleanly on Ctrl+C."""
    monkeypatch.chdir(tmp_path)
    startup, _config, _config_path, _rules = _configure_two_round_schedule(tmp_path)
    startup.output_tag = "w0_qwen"
    state = _initialize_fresh_state(startup)
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), FakeLLMClient()),
    )

    def stop_after_initial_checkpoint(_runtime: object) -> object:
        raise RuntimeError("stop after initial checkpoint")

    monkeypatch.setattr(runner, "execute_round_turns", stop_after_initial_checkpoint)
    with pytest.raises(RuntimeError, match="stop after initial checkpoint"):
        evaluation_orchestrator._run_evaluation_round(state, 1)

    from scripts import check_progress

    monkeypatch.setattr(
        "sys.argv",
        ["check_progress.py", "--pattern", "solo_evaluation_*", "--watch"],
    )

    refreshes: list[float] = []

    def fake_sleep(interval: float) -> None:
        refreshes.append(interval)
        raise KeyboardInterrupt

    monkeypatch.setattr(check_progress.time, "sleep", fake_sleep)
    check_progress.main()

    output = capsys.readouterr().out
    assert "0/2" in output
    assert "active Round 1, 0 committed Turns" in output
    assert len(refreshes) == 1
