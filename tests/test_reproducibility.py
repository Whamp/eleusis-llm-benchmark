"""Tests for end-to-end deterministic reproducibility.

Verifies:
- Two identical seeded runs with fake clients produce byte-equivalent results
- Fallback-heavy runs (all retries exhausted) remain deterministic
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Literal, cast
from unittest.mock import patch

from eleusis.benchmark_config import BenchmarkConfig
from eleusis.evaluation_results import TurnRecord
from eleusis.llm.base import TruncationError
from eleusis.runner import RoundResult, play_round
from tests.conftest import FakeLLMClient, make_action_response

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RULES_PATH = FIXTURES_DIR / "mini_rules.json"


def _make_config(
    max_turns: int = 5,
    shadow_mode: Literal["disabled", "offline", "online"] = "offline",
) -> BenchmarkConfig:
    """Build a minimal config for play_round testing."""
    return {
        "model": "fake-model",
        "game": {
            "num_rules": 1,
            "num_rounds_per_rule": 1,
            "max_turns": max_turns,
            "hand_size": 4,
            "wrong_guess_penalty": 2,
            "seed": 42,
            "shadow_mode": shadow_mode,
        },
        "llm": {
            "max_tokens": 1024,
            "max_llm_retries": 2,
            "temperature": 0.0,
            "seed": 42,
        },
        "rule_compiler": {
            "provider": "huggingface",
            "model_id": "fake-model",
            "max_retries": 1,
            "num_simulations": 5,
            "turns_per_simulation": 5,
            "simulation_seed": 42,
        },
        "rules": {
            "library_path": str(MINI_RULES_PATH),
            "selection": "sequential",
            "index": 0,
        },
    }


def _stable_client(n_turns: int) -> FakeLLMClient:
    """Client that returns n_turns of scripted valid responses."""
    responses = []
    for _ in range(n_turns):
        resp = make_action_response(
            "2H",
            tentative_rule="Only even ranks",
            confidence_level=3,
            guess_rule=False,
        )
        responses.append(json.dumps(resp))
    return FakeLLMClient(responses)


def _normalize_result(result: RoundResult) -> dict[str, object]:
    """Strip Round and Model Attempt wall-clock timing for comparison."""
    normalized_result = copy.deepcopy(dict(result))
    normalized_result.pop("wall_clock_seconds", None)
    turns = cast(list[TurnRecord], normalized_result["turns"])
    for turn in turns:
        for attempt in turn.get("model_attempts", []):
            attempt_payload = cast(dict[str, object], attempt)
            attempt_payload.pop("started_at", None)
            attempt_payload.pop("duration_seconds", None)
            for provider_call in attempt["provider_calls"]:
                provider_call_payload = cast(dict[str, object], provider_call)
                provider_call_payload.pop("timestamp", None)
                provider_call_payload.pop("duration_seconds", None)
    return normalized_result


class TestSeededReproducibility:
    """Two identical seeded runs produce byte-equivalent results."""

    def test_identical_seeded_runs_match(self) -> None:
        """Check two play_round calls with the same config and fake responses.

        They must produce identical results after excluding wall-clock time.
        """
        config = _make_config(max_turns=3)

        results = []
        for _ in range(2):
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch(
                    "eleusis.runner.create_client_from_config", return_value=compiler
                ),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(
            results[1], sort_keys=True
        )

    def test_different_batch_indices_differ(self) -> None:
        """Use different deck shuffles for different batch round indices."""
        config = _make_config(max_turns=3)

        round_results = []
        for batch_idx in [0, 1]:
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch(
                    "eleusis.runner.create_client_from_config", return_value=compiler
                ),
            ):
                result = play_round(config, round_number=1, batch_round_index=batch_idx)

            round_results.append(result)

        # Different batch indices should produce different hands (different deck
        # shuffle)
        hands_0 = [t["hand"] for t in round_results[0]["turns"]]
        hands_1 = [t["hand"] for t in round_results[1]["turns"]]
        assert hands_0 != hands_1, (
            "Different batch_round_index must produce different shuffles"
        )

    def test_same_batch_index_same_shuffle(self) -> None:
        """Same rule + same batch_round_index = identical deck shuffle."""
        config = _make_config(max_turns=3)

        round_results = []
        for _ in range(2):
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch(
                    "eleusis.runner.create_client_from_config", return_value=compiler
                ),
            ):
                result = play_round(config, round_number=1, batch_round_index=0)

            round_results.append(result)

        hands_0 = [t["hand"] for t in round_results[0]["turns"]]
        hands_1 = [t["hand"] for t in round_results[1]["turns"]]
        assert hands_0 == hands_1


class TestFallbackDeterminism:
    """Fallback-heavy runs (all retries exhausted) remain deterministic."""

    def test_all_fallback_runs_match(self) -> None:
        """Check deterministic runs when every turn exhausts retries.

        Two runs with the same seed must still produce identical results.
        """
        config = _make_config(max_turns=3)
        max_retries = config["llm"]["max_llm_retries"]

        results = []
        for _ in range(2):
            # Every attempt raises TruncationError → fallback on every turn
            errors = [TruncationError("truncated")] * (max_retries * 3)
            client = FakeLLMClient(errors)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch(
                    "eleusis.runner.create_client_from_config", return_value=compiler
                ),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(
            results[1], sort_keys=True
        )

    def test_mixed_success_and_fallback_deterministic(self) -> None:
        """Check deterministic runs mixing successful calls and fallbacks.

        The mixed runs must produce identical results when seeded identically.
        """
        config = _make_config(max_turns=3)
        max_retries = config["llm"]["max_llm_retries"]

        def _make_mixed_client() -> FakeLLMClient:
            """Turn 1: success, Turn 2: all retries fail, Turn 3: success."""
            responses = []
            # Turn 1: success
            responses.append(
                json.dumps(
                    make_action_response(
                        "2H",
                        tentative_rule="even ranks",
                        confidence_level=5,
                        guess_rule=False,
                    )
                )
            )
            # Turn 2: all retries fail
            for _ in range(max_retries):
                responses.append(TruncationError("truncated"))
            # Turn 3: success
            responses.append(
                json.dumps(
                    make_action_response(
                        "2H",
                        tentative_rule="even ranks",
                        confidence_level=3,
                        guess_rule=False,
                    )
                )
            )
            return FakeLLMClient(responses)

        results = []
        for _ in range(2):
            client = _make_mixed_client()
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch(
                    "eleusis.runner.create_client_from_config", return_value=compiler
                ),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(
            results[1], sort_keys=True
        )

    def test_fallback_cards_come_from_hand(self) -> None:
        """Fallback cards must be actual cards from the player's hand."""
        config = _make_config(max_turns=3)
        max_retries = config["llm"]["max_llm_retries"]

        errors = [TruncationError("truncated")] * (max_retries * 3)
        client = FakeLLMClient(errors)
        compiler = FakeLLMClient()

        with (
            patch("eleusis.runner.create_client", return_value=client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler),
        ):
            result = play_round(config, round_number=1)

        for turn in result["turns"]:
            played = turn["action_result"]["card"]
            hand = turn["hand"]
            assert played in hand, f"Fallback card {played} not in hand {hand}"
