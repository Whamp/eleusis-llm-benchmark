#!/usr/bin/env python3
"""PROTOTYPE — prove whether an Eleusis Round can resume after a completed Turn.

Question: Can the current Python runtime be reduced to structured JSON, restored in a
fresh process, and then expose the same next prompt and produce the same next game
transition as uninterrupted execution?

This deliberately reaches into private collection and cache fields to discover the
minimum production seams. It is evidence, not production code.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast, overload

from pydantic import TypeAdapter

from eleusis.evaluation_results import TurnRecord
from eleusis.game.cards import Card, Suit
from eleusis.game.engine import GameEngine, PlayCardAction, Rule
from eleusis.game.state import GameState, Mainline, Sideline
from eleusis.game.validator import RuleComparisonMetadata, RuleValidator
from eleusis.llm.base import (
    BaseLLMClient,
    GenerateMetrics,
    LLMCallMetrics,
    LLMMessage,
    LLMResponseEnvelope,
    RuleCompileResult,
    TruncationError,
)
from eleusis.player import LLMScientist, PlayHistoryEntry
from eleusis.round_execution import RoundRuntime, _execute_turn

PROTOTYPE_ROUND_SEED = 0xE1E0515
PROTOTYPE_MAX_TURNS = 8
PROTOTYPE_PLAYER = "Checkpoint Prototype"
PROTOTYPE_RULE_DESCRIPTION = "Only cards with an even rank are accepted."
PROTOTYPE_RULE_CODE = "return card.rank % 2 == 0"
PROTOTYPE_GUESS_TEXT = "Only even ranks"

JsonObject = dict[str, Any]
PrototypeResponse = dict[str, object] | Exception


class PrototypeLLMClient(BaseLLMClient):
    """Return queued benchmark decisions without making provider calls."""

    def __init__(self, responses: list[PrototypeResponse] | None = None) -> None:
        """Initialize a no-network client with queued prototype responses."""
        super().__init__(model_name="prototype-model", role="prototype")
        self.responses = list(responses or [])
        self.prompts_seen: list[str] = []
        self.prototype_call_count = 0

    @property
    def provider_name(self) -> str:
        """Identify this no-network prototype provider."""
        return "prototype"

    def _call_api(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[LLMResponseEnvelope, LLMCallMetrics]:
        """Reject provider calls because this prototype supplies queued decisions."""
        del messages, is_continuation, continuation_depth, disable_thinking
        raise NotImplementedError("Round resume prototype never calls a provider")

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[False] = False,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[True] = True,
    ) -> dict[str, object]: ...

    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: bool = False,
    ) -> str | dict[str, object]:
        """Return one queued response while retaining prompt and usage evidence."""
        del xml_tag
        self.prompts_seen.append(prompt)
        if not self.responses:
            raise RuntimeError("Round resume prototype has no queued model response")
        response = self.responses.pop(0)
        self.prototype_call_count += 1
        if isinstance(response, Exception):
            raise response
        self.generate_metrics.append(
            GenerateMetrics(
                total_calls=1,
                continuation_count=0,
                total_prompt_tokens=100,
                total_output_tokens=50,
                total_reasoning_tokens=30,
                total_answer_tokens=20,
                total_duration_seconds=0.1,
                success=True,
            )
        )
        if return_dict:
            return dict(response)
        return json.dumps(response, sort_keys=True)

    def get_usage_stats(self) -> dict[str, object]:
        """Expose the restored call count used by final Round accounting."""
        return {"total_calls": self.prototype_call_count}


def card_to_checkpoint(card: Card) -> JsonObject:
    """Encode a Card without display-only color or symbol fields."""
    return {"rank": card.rank, "suit": card.suit.suit_name}


def card_from_checkpoint(payload: JsonObject) -> Card:
    """Decode one rank-and-suit Card from the prototype checkpoint."""
    suit_name = str(payload["suit"])
    suit = next(candidate for candidate in Suit if candidate.suit_name == suit_name)
    return Card(rank=int(payload["rank"]), suit=suit)


def random_state_to_json(value: object) -> object:
    """Convert nested random.Random state tuples into JSON arrays."""
    if isinstance(value, tuple):
        return [random_state_to_json(item) for item in value]
    return value


def random_state_from_json(value: object) -> object:
    """Restore nested JSON arrays to tuples accepted by random.Random.setstate."""
    if isinstance(value, list):
        return tuple(random_state_from_json(item) for item in value)
    return value


def prototype_action_response(card: Card, turn_number: int) -> dict[str, object]:
    """Build one valid benchmark-model action for a Card currently in hand."""
    return {
        "card": str(card),
        "reasoning_summary": f"prototype decision for turn {turn_number}",
        "tentative_rule": "",
        "confidence_level": 3,
        "guess_rule": False,
    }


def prototype_fallback_action(
    error: Exception,
    scientist: LLMScientist,
    game_state: GameState,
) -> tuple[PlayCardAction, dict[str, str | None]]:
    """Mirror the runner fallback for an unexpected player exception."""
    card = scientist.rng.choice(game_state.player.hand.get_all_cards())
    return PlayCardAction(card), {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "fallback_card": str(card),
    }


def build_initial_runtime() -> RoundRuntime:
    """Construct an actual game runtime without network-backed model clients."""
    rule = Rule(PROTOTYPE_RULE_DESCRIPTION, PROTOTYPE_RULE_CODE)
    game_state = GameState(PROTOTYPE_PLAYER)
    compiler = PrototypeLLMClient()
    scientist_client = PrototypeLLMClient()
    validator = RuleValidator()
    engine = GameEngine(
        game_state,
        rule,
        rule_compiler_client=compiler,
        rule_validator=validator,
        hand_size=6,
        wrong_guess_penalty=3,
        num_simulations=7,
        turns_per_simulation=9,
        simulation_seed=41,
        compiler_max_retries=2,
    )
    engine.setup_game(round_seed=PROTOTYPE_ROUND_SEED)
    scientist = LLMScientist(
        PROTOTYPE_PLAYER,
        scientist_client,
        max_retries=2,
        engine=engine,
        max_turns=PROTOTYPE_MAX_TURNS,
        rng=random.Random(PROTOTYPE_ROUND_SEED),
    )
    return RoundRuntime(
        round_number=1,
        start_time=time.time(),
        rule=rule,
        rule_metadata=None,
        engine=engine,
        game_state=game_state,
        scientist=scientist,
        scientist_client=scientist_client,
        rule_compiler_client=compiler,
        player_name=PROTOTYPE_PLAYER,
        max_turns=PROTOTYPE_MAX_TURNS,
        shadow_mode="disabled",
        pause_after_turn=False,
        results_folder=None,
        handle_action_error=prototype_fallback_action,
    )


def choose_hand_card(runtime: RoundRuntime, *, accepted: bool) -> Card:
    """Choose a current hand Card with the requested secret-rule verdict."""
    cards = runtime.game_state.player.hand.get_all_cards()
    return next(
        card for card in cards if runtime.engine.evaluate_card(card) is accepted
    )


def execute_scripted_turn(
    runtime: RoundRuntime,
    turn_index: int,
    *,
    accepted: bool,
) -> TurnRecord:
    """Execute one real turn with a queued valid decision."""
    card = choose_hand_card(runtime, accepted=accepted)
    client = runtime.scientist_client
    assert isinstance(client, PrototypeLLMClient)
    client.responses.append(prototype_action_response(card, turn_index + 1))
    turn_record, _ = _execute_turn(runtime, turn_index)
    return turn_record


def seed_cache_state(runtime: RoundRuntime) -> None:
    """Place serializable cache entries in otherwise unexercised continuation state."""
    runtime.rule_compiler_client._compile_cache["cached prototype rule", 10] = {
        "code": "return True",
        "status": "success",
        "attempts": 1,
        "sleep_cycles": 0,
        "provider_used": "prototype/prototype-model",
    }
    validator = runtime.engine.rule_validator
    assert validator is not None
    validator._shadow_cache[PROTOTYPE_RULE_CODE, PROTOTYPE_GUESS_TEXT, 7, 9, 41] = (
        True,
        "cached prototype equivalence",
        {
            "simulation_comparisons": 63,
            "simulation_mismatches": 0,
            "simulation_duration_seconds": 0.01,
            "guessed_code": PROTOTYPE_RULE_CODE,
            "complexity_metrics": None,
            "compilation_status": "success",
            "compilation_attempts": 1,
        },
    )


def snapshot_client(client: BaseLLMClient) -> JsonObject:
    """Capture provider-neutral usage needed by final accounting."""
    call_count = (
        client.prototype_call_count
        if isinstance(client, PrototypeLLMClient)
        else len(client.generate_metrics)
    )
    return {
        "call_count": call_count,
        "generate_metrics": [asdict(metric) for metric in client.generate_metrics],
    }


def restore_client(payload: JsonObject) -> PrototypeLLMClient:
    """Create a fresh client carrying prior usage but no pending network request."""
    client = PrototypeLLMClient()
    client.prototype_call_count = int(payload["call_count"])
    client.generate_metrics = [
        GenerateMetrics(**metric) for metric in payload["generate_metrics"]
    ]
    return client


def snapshot_runtime(runtime: RoundRuntime, turns: list[TurnRecord]) -> JsonObject:
    """Serialize the minimum observed runtime state after completed Turns."""
    state = runtime.game_state
    validator = runtime.engine.rule_validator
    assert validator is not None
    compiler_cache = [
        {"key": [rule_text, max_attempts], "value": value}
        for (rule_text, max_attempts), value in (
            runtime.rule_compiler_client._compile_cache.items()
        )
    ]
    validator_cache = [
        {
            "key": list(key),
            "correct": value[0],
            "reasoning": value[1],
            "metadata": value[2],
        }
        for key, value in validator._shadow_cache.items()
    ]
    return {
        "schema": "round-resume-prototype-v1",
        "completed_turns": len(turns),
        "turn_records": turns,
        "round": {
            "number": runtime.round_number,
            "player_name": runtime.player_name,
            "max_turns": runtime.max_turns,
            "shadow_mode": runtime.shadow_mode,
            "elapsed_seconds": time.time() - runtime.start_time,
        },
        "rule": {
            "description": runtime.rule.description(),
            "code": runtime.rule.get_code(),
        },
        "game_state": {
            "mainline": [card_to_checkpoint(card) for card in state.mainline.get_all()],
            "sidelines": [
                {
                    "mainline_index": index,
                    "cards": [
                        card_to_checkpoint(card) for card in sideline.get_cards()
                    ],
                }
                for index, sideline in sorted(state.sidelines.items())
            ],
            "deck": [card_to_checkpoint(card) for card in state.deck._cards],
            "player": {
                "name": state.player.name,
                "hand": [
                    card_to_checkpoint(card)
                    for card in state.player.hand.get_all_cards()
                ],
                "score": state.player.score,
            },
            "failed_rule_guesses": state.failed_rule_guesses,
            "turn_number": state.turn_number,
            "game_over": state.game_over,
            "winner": state.winner,
        },
        "engine": {
            "rule_guessed": runtime.engine.rule_guessed,
            "winning_turn": runtime.engine.winning_turn,
            "failed_guess_count": runtime.engine.failed_guess_count,
            "hand_size": runtime.engine.hand_size,
            "wrong_guess_penalty": runtime.engine.wrong_guess_penalty,
            "num_simulations": runtime.engine.num_simulations,
            "turns_per_simulation": runtime.engine.turns_per_simulation,
            "simulation_seed": runtime.engine.simulation_seed,
            "compiler_max_retries": runtime.engine.compiler_max_retries,
            "validator_cache": validator_cache,
        },
        "scientist": {
            "name": runtime.scientist.name,
            "max_retries": runtime.scientist.max_retries,
            "max_turns": runtime.scientist.max_turns,
            "play_history": runtime.scientist.play_history,
            "rng_state": random_state_to_json(runtime.scientist.rng.getstate()),
        },
        "clients": {
            "scientist": snapshot_client(runtime.scientist_client),
            "compiler": snapshot_client(runtime.rule_compiler_client),
            "compiler_cache": compiler_cache,
        },
    }


def restore_runtime(checkpoint: JsonObject) -> tuple[RoundRuntime, list[TurnRecord]]:
    """Construct fresh runtime objects from structured prototype checkpoint data."""
    round_payload = checkpoint["round"]
    rule_payload = checkpoint["rule"]
    state_payload = checkpoint["game_state"]
    engine_payload = checkpoint["engine"]
    scientist_payload = checkpoint["scientist"]
    clients_payload = checkpoint["clients"]

    rule = Rule(str(rule_payload["description"]), str(rule_payload["code"]))
    game_state = GameState(str(round_payload["player_name"]))
    game_state.mainline = Mainline()
    for card_payload in state_payload["mainline"]:
        game_state.mainline.add_card(card_from_checkpoint(card_payload))
    game_state.sidelines = {}
    for sideline_payload in state_payload["sidelines"]:
        index = int(sideline_payload["mainline_index"])
        sideline = Sideline(index)
        for card_payload in sideline_payload["cards"]:
            sideline.add_card(card_from_checkpoint(card_payload))
        game_state.sidelines[index] = sideline
    game_state.deck._cards = deque(
        card_from_checkpoint(card_payload) for card_payload in state_payload["deck"]
    )
    game_state.player.hand.clear()
    for card_payload in state_payload["player"]["hand"]:
        game_state.player.hand.add_card(card_from_checkpoint(card_payload))
    game_state.player.score = int(state_payload["player"]["score"])
    game_state.failed_rule_guesses = [
        dict(guess) for guess in state_payload["failed_rule_guesses"]
    ]
    game_state.turn_number = int(state_payload["turn_number"])
    game_state.game_over = bool(state_payload["game_over"])
    winner = state_payload["winner"]
    game_state.winner = str(winner) if winner is not None else None

    compiler = restore_client(clients_payload["compiler"])
    for cache_entry in clients_payload["compiler_cache"]:
        key_payload = cache_entry["key"]
        cache_key = (str(key_payload[0]), int(key_payload[1]))
        cache_value = cast(RuleCompileResult, dict(cache_entry["value"]))
        compiler._compile_cache[cache_key] = cache_value
    scientist_client = restore_client(clients_payload["scientist"])
    validator = RuleValidator()
    for cache_entry in engine_payload["validator_cache"]:
        key_payload = cache_entry["key"]
        cache_key = (
            str(key_payload[0]),
            str(key_payload[1]),
            int(key_payload[2]),
            int(key_payload[3]),
            int(key_payload[4]),
        )
        metadata = cast(
            RuleComparisonMetadata,
            dict(cache_entry["metadata"]),
        )
        validator._shadow_cache[cache_key] = (
            bool(cache_entry["correct"]),
            str(cache_entry["reasoning"]),
            metadata,
        )
    engine = GameEngine(
        game_state,
        rule,
        rule_compiler_client=compiler,
        rule_validator=validator,
        hand_size=int(engine_payload["hand_size"]),
        wrong_guess_penalty=int(engine_payload["wrong_guess_penalty"]),
        num_simulations=int(engine_payload["num_simulations"]),
        turns_per_simulation=int(engine_payload["turns_per_simulation"]),
        simulation_seed=int(engine_payload["simulation_seed"]),
        compiler_max_retries=int(engine_payload["compiler_max_retries"]),
    )
    engine.rule_guessed = bool(engine_payload["rule_guessed"])
    winning_turn = engine_payload["winning_turn"]
    engine.winning_turn = int(winning_turn) if winning_turn is not None else None
    engine.failed_guess_count = int(engine_payload["failed_guess_count"])

    rng = random.Random()
    rng_state = cast(
        tuple[Any, ...],
        random_state_from_json(scientist_payload["rng_state"]),
    )
    rng.setstate(rng_state)
    scientist = LLMScientist(
        str(scientist_payload["name"]),
        scientist_client,
        max_retries=int(scientist_payload["max_retries"]),
        engine=engine,
        max_turns=int(scientist_payload["max_turns"]),
        rng=rng,
    )
    scientist.play_history = TypeAdapter(list[PlayHistoryEntry]).validate_python(
        scientist_payload["play_history"]
    )
    runtime = RoundRuntime(
        round_number=int(round_payload["number"]),
        start_time=time.time() - float(round_payload["elapsed_seconds"]),
        rule=rule,
        rule_metadata=None,
        engine=engine,
        game_state=game_state,
        scientist=scientist,
        scientist_client=scientist_client,
        rule_compiler_client=compiler,
        player_name=str(round_payload["player_name"]),
        max_turns=int(round_payload["max_turns"]),
        shadow_mode=str(round_payload["shadow_mode"]),
        pause_after_turn=False,
        results_folder=None,
        handle_action_error=prototype_fallback_action,
    )
    turns = TypeAdapter(list[TurnRecord]).validate_python(checkpoint["turn_records"])
    return runtime, turns


def observable_runtime(runtime: RoundRuntime) -> JsonObject:
    """Expose all continuation state whose equality matters to the prototype."""
    state = runtime.game_state
    validator = runtime.engine.rule_validator
    assert validator is not None
    return {
        "mainline": [card_to_checkpoint(card) for card in state.mainline.get_all()],
        "sidelines": [
            {
                "mainline_index": index,
                "cards": [card_to_checkpoint(card) for card in sideline.get_cards()],
            }
            for index, sideline in sorted(state.sidelines.items())
        ],
        "deck": [card_to_checkpoint(card) for card in state.deck._cards],
        "hand": [
            card_to_checkpoint(card) for card in state.player.hand.get_all_cards()
        ],
        "failed_rule_guesses": state.failed_rule_guesses,
        "turn_number": state.turn_number,
        "game_over": state.game_over,
        "winner": state.winner,
        "engine": {
            "rule_guessed": runtime.engine.rule_guessed,
            "winning_turn": runtime.engine.winning_turn,
            "failed_guess_count": runtime.engine.failed_guess_count,
        },
        "play_history": runtime.scientist.play_history,
        "rng_state": random_state_to_json(runtime.scientist.rng.getstate()),
        "scientist_usage": runtime.scientist_client.get_usage_stats(),
        "compiler_usage": runtime.rule_compiler_client.get_usage_stats(),
        "compiler_cache": [
            [list(key), value]
            for key, value in sorted(
                runtime.rule_compiler_client._compile_cache.items()
            )
        ],
        "validator_cache": [
            [list(key), list(value)]
            for key, value in sorted(validator._shadow_cache.items())
        ],
    }


def continue_runtime(runtime: RoundRuntime, turn_index: int, mode: str) -> JsonObject:
    """Execute the first post-checkpoint Turn and return comparison evidence."""
    client = runtime.scientist_client
    assert isinstance(client, PrototypeLLMClient)
    if mode in {"scripted", "formal_guess"}:
        card = runtime.game_state.player.hand.get_all_cards()[0]
        response = prototype_action_response(card, turn_index + 1)
        if mode == "formal_guess":
            response["tentative_rule"] = PROTOTYPE_GUESS_TEXT
            response["confidence_level"] = 5
            response["guess_rule"] = True
        client.responses.append(response)
    elif mode == "fallback":
        client.responses.extend(
            TruncationError("prototype truncation")
            for _ in range(runtime.scientist.max_retries)
        )
    else:
        raise ValueError(f"Unknown prototype continuation mode: {mode}")
    turn_record, _ = _execute_turn(runtime, turn_index)
    return {
        "prompt": client.prompts_seen[-1],
        "turn_record": turn_record,
        "runtime": observable_runtime(runtime),
    }


def prepare_checkpoint(completed_turns: int) -> tuple[RoundRuntime, JsonObject]:
    """Build an uninterrupted runtime and snapshot it at the requested boundary."""
    runtime = build_initial_runtime()
    turns: list[TurnRecord] = []
    for turn_index in range(completed_turns):
        turns.append(
            execute_scripted_turn(
                runtime,
                turn_index,
                accepted=turn_index % 2 == 1,
            )
        )
    if completed_turns:
        runtime.game_state.record_failed_guess("prototype failed formal guess")
        runtime.engine.failed_guess_count = 1
    seed_cache_state(runtime)
    return runtime, snapshot_runtime(runtime, turns)


def child_resume(checkpoint_path: Path, mode: str) -> None:
    """Restore and continue in a fresh process, emitting one JSON result."""
    checkpoint = json.loads(checkpoint_path.read_text())
    runtime, turns = restore_runtime(checkpoint)
    result = continue_runtime(runtime, len(turns), mode)
    print(json.dumps(result, sort_keys=True))


def run_cross_process_scenario(completed_turns: int, mode: str) -> JsonObject:
    """Compare uninterrupted and fresh-process continuation at one checkpoint."""
    uninterrupted_runtime, checkpoint = prepare_checkpoint(completed_turns)
    serialized = json.dumps(checkpoint, sort_keys=True)
    with tempfile.TemporaryDirectory(prefix="eleusis-round-resume-prototype-") as tmp:
        checkpoint_path = Path(tmp) / "checkpoint.json"
        checkpoint_path.write_text(serialized)
        child = subprocess.run(
            [
                sys.executable,
                __file__,
                "--child-resume",
                str(checkpoint_path),
                "--mode",
                mode,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    uninterrupted = continue_runtime(uninterrupted_runtime, completed_turns, mode)
    resumed = json.loads(child.stdout)
    if uninterrupted != resumed:
        raise AssertionError(
            f"Round resume prototype mismatch for turns={completed_turns}, mode={mode}"
        )
    guess_attempt = uninterrupted["turn_record"]["guess_attempt"]
    return {
        "completed_turns": completed_turns,
        "mode": mode,
        "checkpoint_bytes": len(serialized.encode()),
        "next_card": uninterrupted["turn_record"]["action_result"]["card"],
        "next_card_accepted": uninterrupted["turn_record"]["action_result"]["accepted"],
        "guess_correct": (
            guess_attempt.get("correct") if isinstance(guess_attempt, dict) else None
        ),
        "round_terminal": uninterrupted["runtime"]["engine"]["rule_guessed"],
        "next_prompt_bytes": len(uninterrupted["prompt"].encode()),
        "result": "identical",
    }


def run_prototype() -> None:
    """Exercise initial, normal, and fallback restoration boundaries."""
    scenarios = [
        run_cross_process_scenario(completed_turns=0, mode="scripted"),
        run_cross_process_scenario(completed_turns=2, mode="scripted"),
        run_cross_process_scenario(completed_turns=2, mode="fallback"),
        run_cross_process_scenario(completed_turns=2, mode="formal_guess"),
    ]
    print("PROTOTYPE QUESTION")
    print(
        "Can a completed-Turn Eleusis runtime cross structured JSON and process "
        "boundaries without changing its next prompt or transition?"
    )
    print("\nSCENARIO RESULTS")
    print(json.dumps(scenarios, indent=2, sort_keys=True))
    print("\nVERDICT")
    print(
        "YES, with a bounded restoration seam. The required checkpoint state is "
        "structured and small, but production code needs owned restore methods for "
        "Deck, GameState, GameEngine, LLMScientist, and provider-neutral usage/cache "
        "state instead of this prototype's private-field access."
    )


def parse_args() -> argparse.Namespace:
    """Parse normal and child-process prototype modes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-resume", type=Path)
    parser.add_argument(
        "--mode",
        choices=["scripted", "fallback", "formal_guess"],
        default="scripted",
    )
    return parser.parse_args()


def main() -> None:
    """Run the public prototype or its fresh-process continuation half."""
    args = parse_args()
    if args.child_resume:
        child_resume(args.child_resume, args.mode)
        return
    run_prototype()


if __name__ == "__main__":
    main()
