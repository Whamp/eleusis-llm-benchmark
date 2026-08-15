"""Round runtime continuation contracts across a fresh Python process."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from eleusis.game.cards import Card, Deck, Hand, Suit
from eleusis.game.engine import GameEngine, PlayCardAction, Rule
from eleusis.game.state import GameState
from eleusis.game.validator import RuleValidator
from eleusis.llm.base import BaseLLMClient, GenerateMetrics, LLMCallMetrics
from eleusis.player import LLMScientist
from eleusis.round_continuation import (
    RoundContinuationIncompatibilityError,
    capture_round_continuation,
    restore_round_continuation,
)
from eleusis.round_execution import RoundRuntime, execute_round_turn
from tests.conftest import FakeLLMClient, make_action_response


class _AccountingFakeLLMClient(FakeLLMClient):
    """Fake provider using production usage-accounting aggregation."""

    def get_usage_stats(self) -> dict[str, object]:
        """Aggregate restored provider-neutral metrics through the base client."""
        return BaseLLMClient.get_usage_stats(self)


def _unexpected_action_error(
    error: Exception,
    scientist: LLMScientist,
    game_state: GameState,
) -> tuple[PlayCardAction, dict[str, str | None]]:
    """Fail if a valid scripted response unexpectedly reaches fallback handling."""
    del scientist, game_state
    raise AssertionError("Round continuation test used action fallback") from error


def _build_round_runtime() -> RoundRuntime:
    """Build a deterministic no-network Round runtime for continuation tests."""
    rule = Rule("Only even ranks.", "return card.rank % 2 == 0")
    state = GameState("Continuation Scientist")
    compiler = FakeLLMClient()
    scientist_client = FakeLLMClient()
    validator = RuleValidator()
    engine = GameEngine(
        state,
        rule,
        rule_compiler_client=compiler,
        rule_validator=validator,
        hand_size=12,
        wrong_guess_penalty=3,
        num_simulations=7,
        turns_per_simulation=9,
        simulation_seed=41,
        compiler_max_retries=2,
    )
    engine.setup_game(round_seed=8675309)
    scientist = LLMScientist(
        "Continuation Scientist",
        scientist_client,
        max_retries=2,
        engine=engine,
        max_turns=40,
        rng=random.Random(8675309),
    )
    return RoundRuntime(
        round_number=3,
        start_time=time.time(),
        rule=rule,
        rule_metadata=None,
        engine=engine,
        game_state=state,
        scientist=scientist,
        scientist_client=scientist_client,
        rule_compiler_client=compiler,
        player_name="Continuation Scientist",
        max_turns=40,
        shadow_mode="disabled",
        pause_after_turn=False,
        results_folder=None,
        handle_action_error=_unexpected_action_error,
    )


def _choose_card(runtime: RoundRuntime, *, accepted: bool) -> Card:
    """Choose one current hand Card with the requested secret-rule outcome."""
    return next(
        card
        for card in runtime.game_state.player.hand.get_all_cards()
        if runtime.engine.evaluate_card(card) is accepted
    )


def _scientific_continuation(payload: dict[str, object]) -> dict[str, object]:
    """Remove elapsed wall time before exact continuation comparisons."""
    comparable = dict(payload)
    round_payload = dict(cast(Mapping[str, object], comparable["round"]))
    round_payload.pop("elapsed_seconds")
    comparable["round"] = round_payload
    return comparable


@pytest.mark.parametrize("accepted", [True, False], ids=["accepted", "rejected"])
def test_round_continuation_restores_same_next_prompt_in_subprocess(
    tmp_path: Path,
    accepted: bool,
) -> None:
    """Accepted and rejected plays restore exact next state and model prompt."""
    runtime = _build_round_runtime()
    played_card = _choose_card(runtime, accepted=accepted)
    client = runtime.scientist_client
    assert isinstance(client, FakeLLMClient)
    client.responses.append(make_action_response(str(played_card)))
    turn_record, play_result = execute_round_turn(runtime, 0)
    assert play_result["accepted"] is accepted

    continuation = capture_round_continuation(
        runtime,
        [turn_record],
        next_turn_index=1,
    )
    canonical_cards = continuation["game_state"]
    encoded = json.dumps(canonical_cards)
    assert "symbol" not in encoded
    assert "color" not in encoded
    assert "physical" not in encoded

    next_card = runtime.game_state.player.hand.get_all_cards()[0]
    next_response = make_action_response(str(next_card))
    client.responses.append(next_response)
    runtime.scientist.get_action(runtime.game_state)
    expected_prompt = client.prompts_seen[-1]

    input_path = tmp_path / "continuation-input.json"
    output_path = tmp_path / "continuation-output.json"
    input_path.write_text(
        json.dumps(
            {
                "continuation": continuation,
                "next_response": next_response,
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.round_continuation_subprocess",
            str(input_path),
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    result = json.loads(output_path.read_text())

    assert result["next_prompt"] == expected_prompt
    assert _scientific_continuation(result["recaptured"]) == (
        _scientific_continuation(continuation)
    )


def test_round_continuation_preserves_hidden_runtime_state() -> None:
    """Round-trip preserves duplicates, caches, accounting, history, and counters."""
    runtime = _build_round_runtime()
    duplicate = Card(7, Suit.CLUBS)
    other = Card(12, Suit.HEARTS)
    runtime.game_state.deck = Deck.restore_deck_cards(
        [
            other.to_canonical_card_data(),
            duplicate.to_canonical_card_data(),
            duplicate.to_canonical_card_data(),
        ]
    )
    runtime.game_state.player.hand = Hand.restore_hand_cards(
        [
            duplicate.to_canonical_card_data(),
            duplicate.to_canonical_card_data(),
            other.to_canonical_card_data(),
        ]
    )
    runtime.game_state.player.score = 17
    runtime.game_state.record_failed_guess("all clubs")
    runtime.game_state.turn_number = 4
    runtime.game_state.game_over = True
    runtime.game_state.winner = None
    runtime.engine.failed_guess_count = 1
    runtime.scientist_client.api_key = "DO-NOT-PERSIST-API-KEY"
    assert isinstance(runtime.scientist_client, FakeLLMClient)
    runtime.scientist_client.responses.append({"raw_wire_payload": "DO-NOT-PERSIST"})
    runtime.scientist.record_play("7♣", accepted=False, reasoning_summary="probe")
    runtime.scientist.rng.choice([1, 2, 3])

    metric = LLMCallMetrics(
        model_name="fake-model",
        role="test",
        prompt_tokens=11,
        output_tokens=7,
        reasoning_tokens=3,
        answer_tokens=4,
        duration_seconds=0.5,
        throughput_tokens_per_sec=14.0,
        finish_reason="stop",
        has_reasoning=True,
        timestamp=123.0,
        provider="fake",
        cost_usd=0.25,
    )
    generated = GenerateMetrics(
        total_calls=1,
        continuation_count=0,
        total_prompt_tokens=11,
        total_output_tokens=7,
        total_reasoning_tokens=3,
        total_answer_tokens=4,
        total_duration_seconds=0.5,
        success=True,
    )
    runtime.scientist_client.call_metrics.append(metric)
    runtime.scientist_client.generate_metrics.append(generated)
    compiler_fallback = FakeLLMClient()
    compiler_fallback.call_metrics.append(metric)
    runtime.rule_compiler_client.fallback_clients.append(compiler_fallback)
    runtime.rule_compiler_client.restore_client_continuation(
        {
            "call_metrics": [],
            "generate_metrics": [],
            "compile_cache": [
                {
                    "key": {
                        "rule_text": "all clubs",
                        "max_total_attempts": 5,
                    },
                    "value": {
                        "code": "return card.suit.suit_name == 'clubs'",
                        "status": "success",
                        "attempts": 1,
                        "sleep_cycles": 0,
                        "provider_used": "fake/fake-model",
                    },
                }
            ],
            "fallback_clients": [compiler_fallback.snapshot_client_continuation()],
        }
    )
    validator = runtime.engine.rule_validator
    assert validator is not None
    validator.restore_validator_cache(
        [
            {
                "key": {
                    "actual_rule_code": runtime.rule.get_code(),
                    "guessed_rule_description": "all clubs",
                    "num_simulations": 7,
                    "turns_per_simulation": 9,
                    "simulation_seed": 41,
                },
                "correct": False,
                "reasoning": "cached mismatch",
                "metadata": {
                    "simulation_comparisons": 3,
                    "simulation_mismatches": 1,
                    "simulation_duration_seconds": 0.125,
                    "guessed_code": "return card.suit.suit_name == 'clubs'",
                    "complexity_metrics": None,
                    "compilation_status": "success",
                    "compilation_attempts": 1,
                },
            }
        ]
    )

    continuation = capture_round_continuation(
        runtime,
        [],
        next_turn_index=0,
    )
    encoded_continuation = json.dumps(continuation)
    assert "DO-NOT-PERSIST-API-KEY" not in encoded_continuation
    assert "DO-NOT-PERSIST" not in encoded_continuation
    restored_compiler = FakeLLMClient()
    restored_compiler.fallback_clients.append(FakeLLMClient())
    restored = restore_round_continuation(
        json.loads(json.dumps(continuation)),
        scientist_client=_AccountingFakeLLMClient(),
        rule_compiler_client=restored_compiler,
        handle_action_error=_unexpected_action_error,
    )
    recaptured = capture_round_continuation(
        restored.runtime,
        restored.turn_records,
        next_turn_index=restored.next_turn_index,
    )

    assert _scientific_continuation(recaptured) == _scientific_continuation(
        continuation
    )
    restored_usage = restored.runtime.scientist_client.get_usage_stats()
    assert restored_usage["prompt_tokens"] == 11
    assert restored_usage["output_tokens"] == 7
    assert restored_usage["call_count"] == 1
    assert restored_usage["cost_usd"] == pytest.approx(0.25)
    assert restored.runtime.scientist_client.call_metrics == [metric]
    assert restored.runtime.scientist_client.generate_metrics == [generated]
    assert restored.runtime.rule_compiler_client.fallback_clients[0].call_metrics == [
        metric
    ]


def test_round_continuation_rejects_incompatible_runtime_serialization() -> None:
    """Active RNG serialization refuses a different Python runtime contract."""
    runtime = _build_round_runtime()
    continuation = capture_round_continuation(
        runtime,
        [],
        next_turn_index=0,
    )
    runtime_identity = cast(dict[str, object], continuation["runtime"])
    runtime_identity["python_version"] = "0.0"

    with pytest.raises(
        RoundContinuationIncompatibilityError,
        match=r"Round continuation incompatible: runtime\.python_version changed",
    ):
        restore_round_continuation(
            continuation,
            scientist_client=FakeLLMClient(),
            rule_compiler_client=FakeLLMClient(),
            handle_action_error=_unexpected_action_error,
        )


def test_round_continuation_rejects_unknown_and_malformed_versions() -> None:
    """Continuation decoding fails closed with one searchable domain error."""
    runtime = _build_round_runtime()
    continuation = capture_round_continuation(
        runtime,
        [],
        next_turn_index=0,
    )

    invalid_rule = copy.deepcopy(continuation)
    invalid_rule_payload = cast(dict[str, object], invalid_rule["rule"])
    invalid_rule_payload["code"] = "return ("
    for payload in (
        {**continuation, "version": 999},
        {**continuation, "game_state": {"mainline": "not-cards"}},
        invalid_rule,
    ):
        with pytest.raises(
            RoundContinuationIncompatibilityError,
            match="Round continuation incompatible:",
        ):
            restore_round_continuation(
                payload,
                scientist_client=FakeLLMClient(),
                rule_compiler_client=FakeLLMClient(),
                handle_action_error=_unexpected_action_error,
            )
