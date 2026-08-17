"""Shadow Guess proposal and post-hoc verdict lifecycle tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from eleusis import evaluation_orchestrator, runner
from eleusis.benchmark_run_store import BenchmarkRunStore
from eleusis.evaluation_state import _initialize_fresh_state
from eleusis.game.cards import Card, Suit
from eleusis.game.rule_library import RuleLibraryEntry
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_evaluation_orchestrator import _startup

_SELECTED_CARDS = ["8♥", "K♣", "5♦"]


def _complete_offline_shadow_round(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> tuple[BenchmarkRunStore, dict[str, object]]:
    """Run one deterministic Round with a multi-card board and Shadow Guesses."""
    monkeypatch.chdir(tmp_path)
    startup = _startup()
    startup.config["game"]["shadow_mode"] = "offline"
    startup.game_config["shadow_mode"] = "offline"
    rule_entry: RuleLibraryEntry = {
        "name": "even_rank",
        "description": "Only even ranks.",
        "code": "return card.rank % 2 == 0",
    }
    Path("rules.json").write_text(json.dumps({"rules": [rule_entry]}))
    state = _initialize_fresh_state(startup)
    assert state.run_store is not None
    scientist = FakeLLMClient(
        [
            make_action_response(
                card,
                tentative_rule="Only even ranks.",
                confidence_level=8 if index == 3 else 3,
            )
            for index, card in enumerate(_SELECTED_CARDS, start=1)
        ]
    )
    monkeypatch.setattr(
        runner,
        "_create_round_clients",
        lambda _config, _player_name: (FakeLLMClient(), scientist),
    )

    evaluation_orchestrator._run_evaluation_round(state, 1)
    return state.run_store, state.run_store.read_completed_round(1)


def test_offline_shadow_proposal_is_immutable_round_evidence(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record a Shadow Guess without changing gameplay, score, or termination."""
    run_store, completed = _complete_offline_shadow_round(monkeypatch, tmp_path)
    turns = cast(list[dict[str, object]], completed["turns"])
    proposal = cast(dict[str, object], turns[-1]["guess_attempt"])
    assert proposal == {
        "version": 1,
        "kind": "shadow",
        "proposal_id": "round:1:turn:3:shadow",
        "guess": "Only even ranks.",
        "online_evaluation": None,
    }
    assert turns[-1]["post_card_state"] == {
        "mainline": [
            {
                "card": {"rank": 10, "suit": "hearts"},
                "rejected_cards": [],
            },
            {
                "card": {"rank": 8, "suit": "hearts"},
                "rejected_cards": [
                    {"rank": 13, "suit": "clubs"},
                    {"rank": 5, "suit": "diamonds"},
                ],
            },
        ],
        "hand": [
            {"rank": 1, "suit": "spades"},
            {"rank": 1, "suit": "hearts"},
        ],
        "turn_number": 3,
        "failed_rule_guesses": [],
    }
    assert completed["terminal_outcome"] == {"kind": "turn_limit"}
    exported = json.loads(run_store.export_path.read_text())
    derived_round = cast(list[dict[str, object]], exported["derived"]["rounds"])[0]
    assert derived_round["score"] == 0
    assert derived_round["failed_guesses"] == 0


def test_shadow_evaluator_state_uses_canonical_board_attachments(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decode the exact post-card mainline and rejected Cards without display text."""
    from eleusis.shadow_verdict import decode_shadow_evaluator_state

    _run_store, completed = _complete_offline_shadow_round(monkeypatch, tmp_path)
    turns = cast(list[dict[str, object]], completed["turns"])

    evaluator_state = decode_shadow_evaluator_state(turns[-1])

    assert evaluator_state.mainline == (
        Card(10, Suit.HEARTS),
        Card(8, Suit.HEARTS),
    )
    assert evaluator_state.rejected_cards_by_position == (
        (),
        (Card(13, Suit.CLUBS), Card(5, Suit.DIAMONDS)),
    )


def test_offline_shadow_verdicts_are_immutable_exported_sidecars(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Append judge verdicts without updating the completed Round Record."""
    from scripts.evaluate_shadows import evaluate_and_store_shadow_verdicts

    run_store, completed = _complete_offline_shadow_round(monkeypatch, tmp_path)
    completed_before = json.dumps(completed, sort_keys=True)
    turns = cast(list[dict[str, object]], completed["turns"])
    proposal = cast(dict[str, object], turns[-1]["guess_attempt"])
    proposal_id = cast(str, proposal["proposal_id"])
    compiler = FakeLLMClient()
    compiler.convert_rule_to_code = MagicMock(
        return_value={
            "code": "return card.rank % 2 == 0",
            "status": "success",
            "attempts": 1,
            "sleep_cycles": 0,
            "provider_used": "fake/judge-v1",
            "cache_hit": False,
        }
    )
    settings = {
        "num_simulations": 1,
        "turns_per_simulation": 2,
        "simulation_seed": 19,
        "compiler_max_retries": 0,
        "compiler_temperature": 0.25,
        "llm_max_tokens": 4096,
        "llm_seed": 23,
    }

    added_verdicts = evaluate_and_store_shadow_verdicts(
        run_store,
        [completed],
        compiler,
        judge_identity={"provider": "fake", "model": "judge-v1"},
        behavior_fingerprint="a" * 64,
        settings=settings,
    )
    assert len(added_verdicts) == 1
    first_verdict = added_verdicts[0]

    assert first_verdict["version"] == 1
    assert first_verdict["proposal_id"] == proposal_id
    assert len(cast(str, first_verdict["verdict_id"])) == 64
    assert first_verdict["judge_identity"] == {
        "provider": "fake",
        "model": "judge-v1",
    }
    assert first_verdict["behavior_fingerprint"] == "a" * 64
    assert first_verdict["settings"] == settings
    verdict_result = cast(dict[str, object], first_verdict["verdict"])
    assert verdict_result["correct"] is True
    evidence = cast(dict[str, object], first_verdict["evidence"])
    post_card_state = cast(dict[str, object], turns[-1]["post_card_state"])
    assert evidence["evaluator_state"] == post_card_state["mainline"]
    assert (
        json.dumps(run_store.read_completed_round(1), sort_keys=True)
        == completed_before
    )
    assert run_store.read_shadow_verdicts() == [first_verdict]

    repeated_verdicts = evaluate_and_store_shadow_verdicts(
        run_store,
        [completed],
        compiler,
        judge_identity={"provider": "fake", "model": "judge-v1"},
        behavior_fingerprint="a" * 64,
        settings=settings,
    )
    assert repeated_verdicts == []
    assert compiler.convert_rule_to_code.call_count == 1

    added_verdicts = evaluate_and_store_shadow_verdicts(
        run_store,
        [completed],
        compiler,
        judge_identity={"provider": "fake", "model": "judge-v2"},
        behavior_fingerprint="b" * 64,
        settings=settings,
    )
    assert len(added_verdicts) == 1
    second_verdict = added_verdicts[0]

    stored_verdicts = run_store.read_shadow_verdicts()
    assert stored_verdicts == [first_verdict, second_verdict]
    assert stored_verdicts[0] == first_verdict
    assert (
        json.dumps(run_store.read_completed_round(1), sort_keys=True)
        == completed_before
    )
    exported = json.loads(run_store.export_path.read_text())
    assert exported["completed_round_records"] == [completed]
    assert exported["shadow_verdicts"] == stored_verdicts
    assert "continuation" not in exported


def test_shadow_verdict_identity_includes_all_judge_client_settings(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every setting configuring the judge client changes verdict identity."""
    from eleusis.shadow_verdict import evaluate_shadow_guess

    _run_store, completed = _complete_offline_shadow_round(monkeypatch, tmp_path)
    turns = cast(list[dict[str, object]], completed["turns"])
    proposal = cast(dict[str, object], turns[-1]["guess_attempt"])
    compiler = FakeLLMClient()
    compiler.convert_rule_to_code = MagicMock(
        return_value={
            "code": "return card.rank % 2 == 0",
            "status": "success",
            "attempts": 1,
            "sleep_cycles": 0,
            "provider_used": "fake/judge-v1",
            "cache_hit": False,
        }
    )
    base_settings = {
        "num_simulations": 1,
        "turns_per_simulation": 2,
        "simulation_seed": 19,
        "compiler_max_retries": 0,
        "compiler_temperature": 0.25,
        "llm_max_tokens": 4096,
        "llm_seed": 23,
    }
    verdict_ids = set()
    for field, value in (
        ("compiler_temperature", 0.75),
        ("llm_max_tokens", 8192),
        ("llm_seed", 99),
    ):
        settings = {**base_settings, field: value}
        verdict = evaluate_shadow_guess(
            completed,
            cast(str, proposal["proposal_id"]),
            compiler,
            judge_identity={"provider": "fake", "model": "judge-v1"},
            behavior_fingerprint="a" * 64,
            settings=settings,
        )
        assert verdict["settings"] == settings
        verdict_ids.add(verdict["verdict_id"])

    assert len(verdict_ids) == 3


def _store_one_offline_verdict(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> BenchmarkRunStore:
    """Complete one offline-shadow Round and append its verdict sidecar."""
    from scripts.evaluate_shadows import evaluate_and_store_shadow_verdicts

    run_store, completed = _complete_offline_shadow_round(monkeypatch, tmp_path)
    compiler = FakeLLMClient()
    compiler.convert_rule_to_code = MagicMock(
        return_value={
            "code": "return card.rank % 2 == 0",
            "status": "success",
            "attempts": 1,
            "sleep_cycles": 0,
            "provider_used": "fake/judge-v1",
            "cache_hit": False,
        }
    )
    evaluate_and_store_shadow_verdicts(
        run_store,
        [completed],
        compiler,
        judge_identity={"provider": "fake", "model": "judge-v1"},
        behavior_fingerprint="a" * 64,
        settings={
            "num_simulations": 1,
            "turns_per_simulation": 2,
            "simulation_seed": 19,
            "compiler_max_retries": 0,
            "compiler_temperature": 0.25,
            "llm_max_tokens": 4096,
            "llm_seed": 23,
        },
    )
    return run_store


def _shadow_guesses(
    analysis_document: dict[str, object],
) -> list[dict[str, object]]:
    return [
        cast(dict[str, object], turn["guess_attempt"])
        for round_data in cast(list[dict[str, object]], analysis_document["rounds"])
        for turn in cast(list[dict[str, object]], round_data["turns"])
        if isinstance(turn.get("guess_attempt"), dict)
        and cast(dict[str, object], turn["guess_attempt"]).get("shadow")
    ]


def test_analysis_views_join_offline_shadow_verdicts(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Offline verdict sidecars must reach report guesses in both read paths.

    Strict Rounds persist offline Shadow Guesses unevaluated and store verdicts
    as immutable sidecars. Analysis views that never join them make every
    strict-run shadow look unevaluated, which empties early-correct-turn,
    no-stakes-shadow, and complexity-ratio analyses.
    """
    from eleusis.analysis.benchmark_run_artifact import read_analysis_run_artifact

    run_store = _store_one_offline_verdict(monkeypatch, tmp_path)

    artifact = read_analysis_run_artifact(run_store.run_folder)
    assert artifact.analysis_document is not None
    shadows = _shadow_guesses(artifact.analysis_document)
    assert shadows
    assert all(shadow.get("evaluated") for shadow in shadows)
    assert all(shadow.get("correct") is True for shadow in shadows)
    assert all(isinstance(shadow.get("node_count"), int) for shadow in shadows)

    portable = tmp_path / "portable_export_only"
    shutil.copytree(run_store.run_folder, portable)
    (portable / "benchmark_run.sqlite3").unlink()
    json_artifact = read_analysis_run_artifact(portable)
    assert json_artifact.analysis_document is not None
    json_shadows = _shadow_guesses(json_artifact.analysis_document)
    assert json_shadows
    assert all(shadow.get("evaluated") for shadow in json_shadows)
    assert all(shadow.get("correct") is True for shadow in json_shadows)
