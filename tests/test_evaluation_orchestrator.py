"""Integration tests for single-model evaluation round orchestration."""

from argparse import Namespace
from pathlib import Path

from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator
from eleusis.benchmark_config import BenchmarkConfig
from eleusis.evaluation_startup import EvaluationStartup
from eleusis.evaluation_state import EvaluationState, _new_evaluation_results
from eleusis.game.rule_library import RuleLibraryEntry
from eleusis.runner import RoundResult


def _startup() -> EvaluationStartup:
    config: BenchmarkConfig = {
        "model": "test-model",
        "game": {
            "num_rules": 1,
            "num_rounds_per_rule": 1,
            "num_rounds": 1,
            "max_turns": 3,
            "hand_size": 2,
            "wrong_guess_penalty": 1,
            "seed": 7,
        },
        "llm": {
            "max_tokens": 100,
            "max_llm_retries": 1,
            "temperature": 0.0,
            "seed": 7,
        },
        "rule_compiler": {
            "provider": "openai_compat",
            "model_id": "compiler",
            "base_url": "http://localhost:3000/v1",
        },
        "rules": {
            "library_path": "rules.json",
            "selection": "sequential",
            "index": 0,
        },
    }
    return EvaluationStartup(
        args=Namespace(),
        checkpoint=None,
        config=config,
        player_model="test-model",
        player_display_name="Test Model",
        rule_compiler_display_name="Compiler",
        num_rounds_per_rule=1,
        suite_name=None,
        suite_cases=None,
        game_config=config["game"],
        rules_config=config["rules"],
        output_tag="test",
        timestamp="20260815_000000",
        log_file="logs/test.log",
        num_rounds=1,
        num_rules=1,
    )


def _round_result() -> RoundResult:
    return {
        "round_number": 1,
        "turn_count": 2,
        "rule_description": "Only red cards.",
        "rule_code": "return card.color == 'red'",
        "rule_metadata": {
            "name": "only_red",
            "description": "Only red cards.",
            "code": "return card.color == 'red'",
        },
        "success": True,
        "score": 2,
        "floored_score": 2,
        "no_stakes_score": 2,
        "first_correct_turn": 2,
        "first_formal_correct_turn": 2,
        "first_shadow_correct_turn": None,
        "failed_guesses": 0,
        "game_over_reason": "correct_guess",
        "schema_compliance_rate": 1.0,
        "llm_usage": {
            "rule_compiler": {},
            "player": {
                "output_tokens": 10,
                "reasoning_tokens": 6,
                "answer_tokens": 4,
            },
        },
        "turns": [],
        "wall_clock_seconds": 1.5,
    }


def test_round_updates_cursor_statistics_checkpoint_and_save(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist one completed round through every orchestration state transition."""
    startup = _startup()
    rule_library: list[RuleLibraryEntry] = [
        {
            "name": "only_red",
            "description": "Only red cards.",
            "code": "return card.color == 'red'",
        }
    ]
    state = EvaluationState(
        startup=startup,
        results=_new_evaluation_results(startup, "folder", rule_library, 0),
        folder_name="folder",
        start_round=1,
        current_rule=None,
        current_rule_name=None,
        rule_factory_index=0,
        checkpoint_rules_library=None,
        all_rules_library=rule_library,
        rule_name_to_index={"only_red": 0},
    )
    saved: list[str] = []
    monkeypatch.setattr(
        evaluation_orchestrator,
        "play_round",
        lambda **_kwargs: _round_result(),
    )
    monkeypatch.setattr(
        evaluation_orchestrator,
        "save_evaluation_results",
        lambda _results, folder: saved.append(folder) or tmp_path / "results.json",
    )

    evaluation_orchestrator._run_evaluation_round(state, 1)

    assert state.current_rule_name == "only_red"
    assert state.rule_factory_index == 1
    assert state.results["statistics"]["total_score"] == 2
    assert state.results["statistics"]["total_output_tokens"] == 10
    assert state.results["checkpoint"]["completed_rounds"] == 1
    assert state.results["checkpoint"]["rules_consumed"][0]["name"] == "only_red"
    assert saved == ["folder"]
