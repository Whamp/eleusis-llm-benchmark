"""Completed play-only Turn resume through Benchmark Run orchestration."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import yaml
from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator, runner
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.benchmark_run_store import BenchmarkRunStore
from eleusis.evaluation_results import TurnRecord
from eleusis.evaluation_startup import EvaluationStartup
from eleusis.evaluation_state import _initialize_fresh_state
from eleusis.game.rule_library import RuleLibraryEntry
from eleusis.llm.base import TruncationError
from eleusis.round_execution import RoundRuntime
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup

_SELECTED_CARDS = ["8♥", "K♣", "5♦"]


class _TurnBoundaryInterruptionError(RuntimeError):
    """Simulate process loss immediately before or after a Turn commit."""


def _configure_three_turn_run(
    root: Path,
) -> tuple[EvaluationStartup, BenchmarkConfig, Path]:
    """Create deterministic accepted, rejected, accepted play-only inputs."""
    startup = _startup()
    startup.config["game"]["max_turns"] = 3
    startup.game_config["max_turns"] = 3
    rule_entry: RuleLibraryEntry = {
        "name": "even_rank",
        "description": "Only even ranks.",
        "code": "return card.rank % 2 == 0",
    }
    rules_path = root / "rules.json"
    rules_path.write_text(json.dumps({"rules": [rule_entry]}))
    startup.config["rules"]["library_path"] = str(rules_path)
    startup.rules_config["library_path"] = str(rules_path)
    config = copy.deepcopy(startup.config)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return startup, config, config_path


def _configure_retry_exhaustion_run(
    root: Path,
) -> tuple[EvaluationStartup, BenchmarkConfig, Path]:
    """Create a two-Turn Run whose model retries deterministically exhaust."""
    startup, _config, config_path = _configure_three_turn_run(root)
    startup.config["game"]["max_turns"] = 2
    startup.game_config["max_turns"] = 2
    startup.config["llm"]["max_llm_retries"] = 3
    config = copy.deepcopy(startup.config)
    config_path.write_text(yaml.safe_dump(config))
    return startup, config, config_path


def _run_uninterrupted_control(
    monkeypatch: MonkeyPatch,
    root: Path,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    """Run the deterministic three-Turn control through orchestration."""
    root.mkdir()
    monkeypatch.chdir(root)
    startup, _config, _config_path = _configure_three_turn_run(root)
    state = _initialize_fresh_state(startup)
    assert state.run_store is not None
    scientist = FakeLLMClient([make_action_response(card) for card in _SELECTED_CARDS])
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), scientist),
    )

    evaluation_orchestrator._run_evaluation_round(state, 1)

    completed = state.run_store.read_completed_round(1)
    exported = json.loads(state.run_store.export_path.read_text())
    derived = cast(dict[str, object], exported["derived"])
    return completed, derived, scientist.prompts_seen


def _create_interrupted_run(
    monkeypatch: MonkeyPatch,
    root: Path,
    *,
    boundary: str,
    turn_number: int,
) -> tuple[Path, Path, BenchmarkRunStore, FakeLLMClient]:
    """Stop one deterministic Run at the requested transactional boundary."""
    root.mkdir()
    monkeypatch.chdir(root)
    startup, _config, config_path = _configure_three_turn_run(root)
    state = _initialize_fresh_state(startup)
    run_store = state.run_store
    assert run_store is not None
    scientist = FakeLLMClient([make_action_response(card) for card in _SELECTED_CARDS])
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), scientist),
    )
    commit_completed_turn = run_store.commit_completed_turn

    def interrupt_commit(
        runtime: RoundRuntime,
        turns: list[TurnRecord],
    ) -> None:
        if boundary == "before" and len(turns) == turn_number:
            raise _TurnBoundaryInterruptionError("stopped before completed Turn commit")
        commit_completed_turn(runtime, turns)
        if boundary == "after" and len(turns) == turn_number:
            raise _TurnBoundaryInterruptionError("stopped after completed Turn commit")

    monkeypatch.setattr(run_store, "commit_completed_turn", interrupt_commit)
    with pytest.raises(_TurnBoundaryInterruptionError, match=f"stopped {boundary}"):
        evaluation_orchestrator._run_evaluation_round(state, 1)
    return run_store.run_folder, config_path, run_store, scientist


def _resume_in_fresh_process(
    root: Path,
    run_folder: Path,
    config_path: Path,
    selected_cards: list[str],
    *,
    truncation_count: int = 0,
) -> dict[str, object]:
    """Resume an active Round using the production startup and runner path."""
    input_path = root / "resume-input.json"
    output_path = root / "resume-output.json"
    input_path.write_text(
        json.dumps(
            {
                "working_directory": str(root),
                "config_path": str(config_path),
                "run_folder": str(run_folder),
                "selected_cards": selected_cards,
                "truncation_count": truncation_count,
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.completed_turn_resume_subprocess",
            str(input_path),
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    return cast(dict[str, object], json.loads(output_path.read_text()))


def _scientific_round_facts(record: dict[str, object]) -> dict[str, object]:
    """Exclude per-Run identity and nondeterministic attempt timing."""
    facts = copy.deepcopy(record)
    facts.pop("run_id", None)
    turns = cast(list[dict[str, object]], facts["turns"])
    for turn in turns:
        attempts = cast(list[dict[str, object]], turn["model_attempts"])
        for attempt in attempts:
            attempt.pop("started_at", None)
            attempt.pop("duration_seconds", None)
            provider_calls = cast(
                list[dict[str, object]],
                attempt["provider_calls"],
            )
            for provider_call in provider_calls:
                provider_call.pop("timestamp", None)
                provider_call.pop("duration_seconds", None)
    return facts


@pytest.mark.parametrize("committed_turns", [1, 2])
def test_resume_after_committed_turns_matches_uninterrupted_control(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    committed_turns: int,
) -> None:
    """Resume exactly after accepted and rejected card checkpoints."""
    control, control_derived, control_prompts = _run_uninterrupted_control(
        monkeypatch,
        tmp_path / "control",
    )
    interrupted_root = tmp_path / f"interrupted-{committed_turns}"
    run_folder, config_path, run_store, interrupted_scientist = _create_interrupted_run(
        monkeypatch,
        interrupted_root,
        boundary="after",
        turn_number=committed_turns,
    )
    active = run_store.read_active_round(1)
    assert active is not None
    assert active.continuation["next_turn_index"] == committed_turns
    assert len(cast(list[object], active.record["turns"])) == committed_turns
    assert len(interrupted_scientist.prompts_seen) == committed_turns

    resumed = _resume_in_fresh_process(
        interrupted_root,
        run_folder,
        config_path,
        _SELECTED_CARDS[committed_turns:],
    )

    resumed_record = cast(dict[str, object], resumed["record"])
    assert resumed["prompts"] == control_prompts[committed_turns:]
    assert _scientific_round_facts(resumed_record) == _scientific_round_facts(control)
    assert resumed["derived"] == control_derived
    turns = cast(list[dict[str, object]], resumed_record["turns"])
    first_outcome = cast(dict[str, object], turns[0]["card_outcome"])
    second_outcome = cast(dict[str, object], turns[1]["card_outcome"])
    assert first_outcome["accepted"] is True
    assert second_outcome["accepted"] is False
    first_post = cast(dict[str, object], turns[0]["post_card_state"])
    second_post = cast(dict[str, object], turns[1]["post_card_state"])
    assert len(cast(list[object], first_post["mainline"])) == 2
    second_mainline = cast(list[dict[str, object]], second_post["mainline"])
    assert second_mainline[-1]["rejected_cards"] == [{"rank": 13, "suit": "clubs"}]


def test_retry_exhaustion_resume_preserves_fallback_and_attempt_accounting(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A committed retry-exhaustion Turn resumes as the same terminal trajectory."""
    control_root = tmp_path / "fallback-control"
    control_root.mkdir()
    monkeypatch.chdir(control_root)
    control_startup, _control_config, _control_path = _configure_retry_exhaustion_run(
        control_root
    )
    control_state = _initialize_fresh_state(control_startup)
    assert control_state.run_store is not None
    control_scientist = FakeLLMClient(
        [TruncationError("truncated")] * 6,
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), control_scientist),
    )
    evaluation_orchestrator._run_evaluation_round(control_state, 1)
    control_record = control_state.run_store.read_completed_round(1)
    control_export = json.loads(control_state.run_store.export_path.read_text())
    control_derived = cast(dict[str, object], control_export["derived"])
    control_round_derived = cast(
        list[dict[str, object]],
        control_derived["rounds"],
    )[0]
    assert control_round_derived["usage"] == {
        "model_attempt_count": 6,
        "provider_call_count": 6,
        "prompt_tokens": 600,
        "output_tokens": 300,
        "reasoning_tokens": 180,
        "answer_tokens": 120,
        "duration_seconds": pytest.approx(0.6),
        "throughput_tokens_per_second": pytest.approx(500.0),
        "cost": {"usd": None, "pricing_version": None},
    }

    interrupted_root = tmp_path / "fallback-interrupted"
    interrupted_root.mkdir()
    monkeypatch.chdir(interrupted_root)
    interrupted_startup, _interrupted_config, interrupted_config_path = (
        _configure_retry_exhaustion_run(interrupted_root)
    )
    interrupted_state = _initialize_fresh_state(interrupted_startup)
    interrupted_store = interrupted_state.run_store
    assert interrupted_store is not None
    interrupted_scientist = FakeLLMClient(
        [TruncationError("truncated")] * 3,
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), interrupted_scientist),
    )
    commit_completed_turn = interrupted_store.commit_completed_turn

    def interrupt_after_fallback(
        runtime: RoundRuntime,
        turns: list[TurnRecord],
    ) -> None:
        commit_completed_turn(runtime, turns)
        raise _TurnBoundaryInterruptionError("stopped after fallback Turn commit")

    monkeypatch.setattr(
        interrupted_store,
        "commit_completed_turn",
        interrupt_after_fallback,
    )
    with pytest.raises(
        _TurnBoundaryInterruptionError,
        match="stopped after fallback Turn commit",
    ):
        evaluation_orchestrator._run_evaluation_round(interrupted_state, 1)

    active = interrupted_store.read_active_round(1)
    assert active is not None
    active_turns = cast(list[dict[str, object]], active.record["turns"])
    attempts = cast(list[dict[str, object]], active_turns[0]["model_attempts"])
    assert len(attempts) == 3
    assert [attempt["attempt_number"] for attempt in attempts] == [1, 2, 3]
    assert len({cast(str, attempt["prompt"]) for attempt in attempts}) == 2
    decision = cast(dict[str, object], active_turns[0]["final_decision"])
    assert decision["origin"] == "fallback"
    assert decision["cause"] == "retry_exhausted"

    resumed = _resume_in_fresh_process(
        interrupted_root,
        interrupted_store.run_folder,
        interrupted_config_path,
        [],
        truncation_count=3,
    )

    resumed_record = cast(dict[str, object], resumed["record"])
    assert _scientific_round_facts(resumed_record) == _scientific_round_facts(
        control_record
    )
    resumed_turns = cast(list[dict[str, object]], resumed_record["turns"])
    assert len(resumed_turns) == 2
    second_attempts = cast(
        list[dict[str, object]],
        resumed_turns[1]["model_attempts"],
    )
    assert len(second_attempts) == 3
    second_decision = cast(dict[str, object], resumed_turns[1]["final_decision"])
    assert second_decision["origin"] == "fallback"
    assert second_decision["cause"] == "retry_exhausted"
    assert resumed["derived"] == control_derived


def test_failure_before_turn_commit_repeats_turn_without_advancing_checkpoint(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A proposed Turn remains absent when its checkpoint transaction never begins."""
    control, control_derived, control_prompts = _run_uninterrupted_control(
        monkeypatch,
        tmp_path / "control",
    )
    interrupted_root = tmp_path / "before-commit"
    run_folder, config_path, run_store, interrupted_scientist = _create_interrupted_run(
        monkeypatch,
        interrupted_root,
        boundary="before",
        turn_number=1,
    )
    active = run_store.read_active_round(1)
    assert active is not None
    assert active.record["turns"] == []
    assert active.continuation["next_turn_index"] == 0
    assert interrupted_scientist.prompts_seen == [control_prompts[0]]

    resumed = _resume_in_fresh_process(
        interrupted_root,
        run_folder,
        config_path,
        _SELECTED_CARDS,
    )

    resumed_record = cast(dict[str, object], resumed["record"])
    assert resumed["prompts"] == control_prompts
    assert _scientific_round_facts(resumed_record) == _scientific_round_facts(control)
    assert resumed["derived"] == control_derived
