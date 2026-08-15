"""Focused strict-transition tests for authoritative Round Records."""

import copy
from typing import cast

import pytest

from eleusis.evaluation_results import TurnRecord
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
