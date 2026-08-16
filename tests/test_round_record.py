"""Focused strict-transition tests for authoritative Round Records."""

import copy
import json
from typing import cast

import pytest

from eleusis.evaluation_results import TurnRecord
from eleusis.llm.base import TruncationError
from eleusis.round_continuation import capture_round_continuation
from eleusis.round_execution import RoundRuntime, execute_round_turn
from eleusis.round_record import (
    RoundRecordValidationError,
    append_round_record_turn,
    complete_round_record,
    create_active_round_record,
    validate_round_record_document,
)
from tests.conftest import FakeLLMClient, make_action_response
from tests.test_round_continuation import _build_round_runtime


def _manifest(runtime: RoundRuntime) -> dict[str, object]:
    """Build the minimal fixed Run manifest consumed by the Round codec."""
    schedule = [
        {
            "round_number": number,
            "rule_name": None,
            "rule_description": runtime.rule.description(),
            "rule_code": runtime.rule.get_code(),
            "batch_round_index": 0,
        }
        for number in range(1, runtime.round_number + 1)
    ]
    return {
        "version": 1,
        "run_id": "run-record-tests",
        "versions": {
            "database": 1,
            "manifest": 1,
            "round_record": 1,
            "round_checkpoint": 1,
            "export": 1,
        },
        "schedule": schedule,
        "effective_settings": {
            "configured_game_seed": 8675309,
            "game_seed": 8675309,
            "hand_size": 12,
            "max_turns": 40,
            "wrong_guess_penalty": 3,
            "shadow_mode": "disabled",
            "llm_max_tokens": 4096,
            "llm_temperature": 0.25,
            "llm_seed": 17,
            "llm_max_retries": 2,
            "rule_selection": "sequential",
        },
        "scientific_config": {
            "llm": {
                "max_tokens": 4096,
                "temperature": 0.25,
                "seed": 17,
                "max_llm_retries": 2,
            },
            "model": "fake",
            "game": {
                "num_rules": 1,
                "num_rounds_per_rule": 1,
                "max_turns": 40,
                "hand_size": 12,
                "wrong_guess_penalty": 3,
                "seed": 8675309,
                "batch_round_offset": None,
                "shadow_mode": "disabled",
            },
            "rule_compiler": {
                "provider": "fake",
                "model_id": "fake-compiler",
                "temperature": 0.5,
                "max_retries": 2,
                "num_simulations": 7,
                "turns_per_simulation": 9,
                "simulation_seed": 41,
            },
            "rules": {"selection": "sequential", "index": 0},
            "suite": None,
        },
        "model_identity": {"model_key": "fake", "display_name": "Fake"},
        "compiler_identity": {
            "provider": "fake",
            "model_id": "fake-compiler",
            "display_name": "Fake Compiler",
        },
        "prompt_identities": [{"name": "action", "sha256": "digest"}],
        "source_provenance": {
            "revision": "revision",
            "dirty": False,
            "files": [],
            "fingerprint": "fingerprint",
        },
    }


def _execute_play_only_turn(
    runtime: RoundRuntime,
    turn_index: int,
) -> TurnRecord:
    """Execute one valid scripted card play without a Guess Attempt."""
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    client.responses.append(make_action_response(str(selected)))
    turn_record, _result = execute_round_turn(runtime, turn_index)
    return turn_record


def test_round_record_preserves_model_attempt_outcomes_and_provider_calls() -> None:
    """Each changed prompt submission retains its provider-neutral evidence."""
    runtime = _build_round_runtime()
    runtime.scientist.max_retries = 5
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    client.responses.extend(
        [
            make_action_response("not-a-card"),
            "not structured JSON",
            TruncationError("truncated"),
            RuntimeError("provider unavailable"),
            make_action_response(str(selected)),
        ]
    )
    before = capture_round_continuation(runtime, [], next_turn_index=0)

    turn_record, _result = execute_round_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_turn = append_round_record_turn(
        active,
        before,
        after,
        turn_record,
        runtime,
    )

    turn = cast(list[dict[str, object]], with_turn["turns"])[0]
    attempts = cast(list[dict[str, object]], turn["model_attempts"])
    assert [attempt["attempt_number"] for attempt in attempts] == [1, 2, 3, 4, 5]
    assert [attempt["interpretation"] for attempt in attempts] == [
        "card_parse_error",
        "structured_response_parse_error",
        "truncated",
        "provider_error",
        "usable_action",
    ]
    assert len({cast(str, attempt["prompt"]) for attempt in attempts}) == 4
    assert attempts[0]["raw_completion"] is not None
    assert attempts[1]["raw_completion"] == "not structured JSON"
    assert attempts[3]["raw_completion"] is None
    assert attempts[4]["structured_completion"] == make_action_response(str(selected))
    assert attempts[0]["retry_cause"] == "card_parse_error"
    assert attempts[4]["retry_cause"] is None
    provider_calls = cast(list[dict[str, object]], attempts[0]["provider_calls"])
    assert len(provider_calls) == 1
    assert provider_calls[0]["provider"] == "fake"
    assert provider_calls[0]["model"] == "fake-model"
    assert provider_calls[0]["finish_reason"] == "stop"
    assert "cost_usd" not in provider_calls[0]
    decision = cast(dict[str, object], turn["final_decision"])
    assert decision == {
        "origin": "model_attempt",
        "selected_card": selected.to_canonical_card_data(),
        "model_attempt_number": 5,
    }


def test_round_record_represents_unavailable_raw_completion_explicitly() -> None:
    """A usable Model Attempt can honestly report unavailable raw text."""
    runtime = _build_round_runtime()
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    client.raw_completion_available = False
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    client.responses.append(make_action_response(str(selected)))
    before = capture_round_continuation(runtime, [], next_turn_index=0)

    turn_record, _result = execute_round_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_turn = append_round_record_turn(
        active,
        before,
        after,
        turn_record,
        runtime,
    )

    turn = cast(list[dict[str, object]], with_turn["turns"])[0]
    attempt = cast(list[dict[str, object]], turn["model_attempts"])[0]
    assert attempt["interpretation"] == "usable_action"
    assert attempt["raw_completion"] is None
    assert attempt["structured_completion"] == make_action_response(str(selected))


def test_round_record_preserves_formal_guess_evaluator_evidence() -> None:
    """A Formal Guess retains compiler provenance and equivalence evidence."""
    runtime = _build_round_runtime()
    compiler = runtime.rule_compiler_client
    scientist = runtime.scientist_client
    assert isinstance(compiler, FakeLLMClient)
    assert isinstance(scientist, FakeLLMClient)
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    compiler.responses.append("return card.rank % 2 == 0")
    precompiled = compiler.convert_rule_to_code("Only even ranks.", max_retries=2)
    assert precompiled["cache_hit"] is False
    scientist.responses.append(
        make_action_response(
            str(selected),
            tentative_rule="Only even ranks.",
            confidence_level=5,
            guess_rule=True,
        )
    )
    before = capture_round_continuation(runtime, [], next_turn_index=0)

    turn_record, result = execute_round_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_turn = append_round_record_turn(
        active,
        before,
        after,
        turn_record,
        runtime,
    )

    assert result["correct"] is True
    assert len(compiler.prompts_seen) == 1
    turn = cast(list[dict[str, object]], with_turn["turns"])[0]
    formal_guess = cast(dict[str, object], turn["guess_attempt"])
    equivalence = cast(dict[str, object], formal_guess["equivalence"])
    assert formal_guess == {
        "version": 1,
        "kind": "formal",
        "guess": "Only even ranks.",
        "correct": True,
        "reasoning": "Rules appear equivalent: 3276 comparisons, all matched",
        "guessed_code": "return card.rank % 2 == 0",
        "node_count": 12,
        "cyclomatic_complexity": 1,
        "compilation": {
            "status": "success",
            "attempt_count": 1,
            "cache_hit": True,
            "artifact_provider": "fake/fake-model",
            "rule_compilation_attempts": None,
        },
        "equivalence": {
            "num_simulations": 7,
            "turns_per_simulation": 9,
            "simulation_seed": 41,
            "cache_hit": False,
            "comparisons": 3276,
            "mismatches": 0,
            "duration_seconds": equivalence["duration_seconds"],
        },
    }


def test_round_record_rejects_incomplete_formal_guess_evidence() -> None:
    """A Formal Guess cannot omit required equivalence evidence."""
    runtime = _build_round_runtime()
    compiler = runtime.rule_compiler_client
    scientist = runtime.scientist_client
    assert isinstance(compiler, FakeLLMClient)
    assert isinstance(scientist, FakeLLMClient)
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    compiler.responses.append("return card.rank % 2 == 0")
    scientist.responses.append(
        make_action_response(
            str(selected),
            tentative_rule="Only even ranks.",
            confidence_level=5,
            guess_rule=True,
        )
    )
    before = capture_round_continuation(runtime, [], next_turn_index=0)
    turn_record, _result = execute_round_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_turn = append_round_record_turn(
        active,
        before,
        after,
        turn_record,
        runtime,
    )
    invalid = copy.deepcopy(with_turn)
    turn = cast(list[dict[str, object]], invalid["turns"])[0]
    formal_guess = cast(dict[str, object], turn["guess_attempt"])
    formal_guess.pop("equivalence")

    with pytest.raises(
        RoundRecordValidationError,
        match="Formal Guess evidence is invalid",
    ):
        validate_round_record_document(invalid)


@pytest.mark.parametrize(
    ("game_over_reason", "outcome_kind"),
    [
        ("abandoned", "abandoned"),
        ("impossible_to_continue", "impossible_to_continue"),
    ],
)
def test_round_record_distinguishes_explicit_non_gameplay_outcomes(
    game_over_reason: str,
    outcome_kind: str,
) -> None:
    """Explicit stop decisions are terminal but process interruption is not."""
    runtime = _build_round_runtime()
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )

    completed = complete_round_record(active, game_over_reason=game_over_reason)

    assert completed["terminal_outcome"] == {"kind": outcome_kind}
    assert active["terminal_outcome"] is None


def test_round_record_rejects_non_contiguous_turn_number() -> None:
    """A proposed Turn must immediately follow the active Round Record."""
    runtime = _build_round_runtime()
    before = capture_round_continuation(runtime, [], next_turn_index=0)
    turn_record = _execute_play_only_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    turn_record["turn_number"] = 2

    with pytest.raises(
        RoundRecordValidationError,
        match="Turn ordering is not contiguous",
    ):
        append_round_record_turn(active, before, after, turn_record, runtime)


def test_round_record_rejects_selected_card_outside_pre_decision_hand() -> None:
    """A proposed Turn cannot select a Card absent from its observation."""
    runtime = _build_round_runtime()
    before = capture_round_continuation(runtime, [], next_turn_index=0)
    turn_record = _execute_play_only_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    invalid_turn = copy.deepcopy(turn_record)
    invalid_turn["action_result"]["card"] = "not-a-card"

    with pytest.raises(
        RoundRecordValidationError,
        match="selected Card was not in the pre-decision hand",
    ):
        append_round_record_turn(active, before, after, invalid_turn, runtime)


def test_round_record_rejects_card_multiplicity_change() -> None:
    """A proposed Turn cannot create or lose a Card in hidden continuation state."""
    runtime = _build_round_runtime()
    before = capture_round_continuation(runtime, [], next_turn_index=0)
    turn_record = _execute_play_only_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    invalid_after = copy.deepcopy(after)
    game_state = cast(dict[str, object], invalid_after["game_state"])
    deck = cast(list[object], game_state["deck"])
    deck.pop()

    with pytest.raises(
        RoundRecordValidationError,
        match="Card multiplicity was not conserved",
    ):
        append_round_record_turn(active, before, invalid_after, turn_record, runtime)


def test_round_record_two_turn_prompts_match_pre_decision_turn_numbers() -> None:
    """Each prompt and structured observation identify the same Turn."""
    runtime = _build_round_runtime()
    before_first = capture_round_continuation(runtime, [], next_turn_index=0)
    first_turn = _execute_play_only_turn(runtime, 0)
    after_first = capture_round_continuation(runtime, [first_turn], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_first = append_round_record_turn(
        active,
        before_first,
        after_first,
        first_turn,
        runtime,
    )
    second_turn = _execute_play_only_turn(runtime, 1)
    after_second = capture_round_continuation(
        runtime,
        [first_turn, second_turn],
        next_turn_index=2,
    )

    with_second = append_round_record_turn(
        with_first,
        after_first,
        after_second,
        second_turn,
        runtime,
    )

    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    persisted_turns = cast(list[dict[str, object]], with_second["turns"])
    assert "Turn: 1 / 40" in client.prompts_seen[0]
    assert "Turn: 2 / 40" in client.prompts_seen[1]
    assert [
        cast(dict[str, object], turn["pre_decision_state"])["turn_number"]
        for turn in persisted_turns
    ] == [1, 2]


def test_round_record_persists_effective_settings_and_schema_errors() -> None:
    """Strict Round encoding retains credential-free scientific settings."""
    runtime = _build_round_runtime()
    before = capture_round_continuation(runtime, [], next_turn_index=0)
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    selected = runtime.game_state.player.hand.get_all_cards()[0]
    client.responses.append(
        make_action_response(str(selected), confidence_level="not-a-number")
    )
    turn_record, _result = execute_round_turn(runtime, 0)
    after = capture_round_continuation(runtime, [turn_record], next_turn_index=1)
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )

    with_turn = append_round_record_turn(active, before, after, turn_record, runtime)
    round_tripped = validate_round_record_document(
        copy.deepcopy(json.loads(json.dumps(with_turn)))
    )

    settings = cast(dict[str, object], round_tripped["settings"])
    scientific_config = cast(
        dict[str, object],
        _manifest(runtime)["scientific_config"],
    )
    assert settings["llm"] == scientific_config["llm"]
    assert settings["rule_compiler"] == scientific_config["rule_compiler"]
    turn = cast(list[dict[str, object]], round_tripped["turns"])[0]
    assert turn["schema_errors"] == ["confidence_type"]
    invalid = copy.deepcopy(round_tripped)
    cast(dict[str, object], invalid["settings"])["unknown_setting"] = True
    with pytest.raises(RoundRecordValidationError, match="extra_forbidden"):
        validate_round_record_document(invalid)


def test_round_record_rejects_state_discontinuity_between_turns() -> None:
    """Each Turn observation must equal the preceding post-card state."""
    runtime = _build_round_runtime()
    before_first = capture_round_continuation(runtime, [], next_turn_index=0)
    first_turn = _execute_play_only_turn(runtime, 0)
    after_first = capture_round_continuation(
        runtime,
        [first_turn],
        next_turn_index=1,
    )
    active = create_active_round_record(
        _manifest(runtime),
        runtime,
        effective_round_seed=8675309,
        batch_round_index=0,
    )
    with_first = append_round_record_turn(
        active,
        before_first,
        after_first,
        first_turn,
        runtime,
    )
    second_turn = _execute_play_only_turn(runtime, 1)
    after_second = capture_round_continuation(
        runtime,
        [first_turn, second_turn],
        next_turn_index=2,
    )
    discontinuous_before = copy.deepcopy(after_first)
    game_state = cast(dict[str, object], discontinuous_before["game_state"])
    failed_guesses = cast(list[object], game_state["failed_rule_guesses"])
    failed_guesses.append({"player": runtime.player_name, "guess": "invented"})

    with pytest.raises(
        RoundRecordValidationError,
        match="pre-decision state breaks continuity",
    ):
        append_round_record_turn(
            with_first,
            discontinuous_before,
            after_second,
            second_turn,
            runtime,
        )
