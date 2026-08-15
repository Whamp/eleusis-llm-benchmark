"""Focused strict-transition tests for authoritative Round Records."""

import copy
from typing import cast

import pytest

from eleusis.evaluation_results import TurnRecord
from eleusis.llm.base import TruncationError
from eleusis.round_continuation import capture_round_continuation
from eleusis.round_execution import RoundRuntime, execute_round_turn
from eleusis.round_record import (
    RoundRecordValidationError,
    append_round_record_turn,
    create_active_round_record,
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
        "run_id": "run-record-tests",
        "schedule": schedule,
        "effective_settings": {"game_seed": 8675309},
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
    deck.append({"rank": 1, "suit": "spades"})

    with pytest.raises(
        RoundRecordValidationError,
        match="Card multiplicity was not conserved",
    ):
        append_round_record_turn(active, before, invalid_after, turn_record, runtime)


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
