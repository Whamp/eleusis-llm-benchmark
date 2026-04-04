"""Tests for end-to-end deterministic reproducibility.

Verifies:
- Two identical seeded runs with fake clients produce byte-equivalent results
- Fallback-heavy runs (all retries exhausted) remain deterministic
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from eleusis.llm.base import TruncationError
from eleusis.runner import play_round
from tests.conftest import FakeLLMClient, make_action_response

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RULES_PATH = FIXTURES_DIR / "mini_rules.json"


def _make_config(max_turns: int = 5, shadow_mode: str = "offline") -> dict:
    """Build a minimal config for play_round testing."""
    return {
        "model": "fake-model",
        "game": {
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
            "provider": "fake",
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


def _normalize_result(result: dict) -> dict:
    """Strip non-deterministic fields (wall_clock_seconds) for comparison."""
    result = dict(result)
    result.pop("wall_clock_seconds", None)
    return result


class TestSeededReproducibility:
    """Two identical seeded runs produce byte-equivalent results."""

    def test_identical_seeded_runs_match(self):
        """Two play_round calls with same config + same fake responses
        must produce identical result dicts (excluding wall_clock_seconds)."""
        config = _make_config(max_turns=3)

        results = []
        for _ in range(2):
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch("eleusis.runner.create_client_from_config", return_value=compiler),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(results[1], sort_keys=True)

    def test_different_batch_indices_differ(self):
        """Same rule with different batch_round_index should produce different deck shuffles."""
        config = _make_config(max_turns=3)

        round_results = []
        for batch_idx in [0, 1]:
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch("eleusis.runner.create_client_from_config", return_value=compiler),
            ):
                result = play_round(config, round_number=1, batch_round_index=batch_idx)

            round_results.append(result)

        # Different batch indices should produce different hands (different deck shuffle)
        hands_0 = [t["hand"] for t in round_results[0]["turns"]]
        hands_1 = [t["hand"] for t in round_results[1]["turns"]]
        assert hands_0 != hands_1, "Different batch_round_index must produce different shuffles"

    def test_same_batch_index_same_shuffle(self):
        """Same rule + same batch_round_index = identical deck shuffle."""
        config = _make_config(max_turns=3)

        round_results = []
        for _ in range(2):
            client = _stable_client(3)
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch("eleusis.runner.create_client_from_config", return_value=compiler),
            ):
                result = play_round(config, round_number=1, batch_round_index=0)

            round_results.append(result)

        hands_0 = [t["hand"] for t in round_results[0]["turns"]]
        hands_1 = [t["hand"] for t in round_results[1]["turns"]]
        assert hands_0 == hands_1


class TestFallbackDeterminism:
    """Fallback-heavy runs (all retries exhausted) remain deterministic."""

    def test_all_fallback_runs_match(self):
        """When every turn exhausts retries and falls back, two runs
        with the same seed must still produce identical results."""
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
                patch("eleusis.runner.create_client_from_config", return_value=compiler),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(results[1], sort_keys=True)

    def test_mixed_success_and_fallback_deterministic(self):
        """Runs with a mix of successful LLM calls and fallbacks
        must produce identical results when seeded identically."""
        config = _make_config(max_turns=3)
        max_retries = config["llm"]["max_llm_retries"]

        def _make_mixed_client():
            """Turn 1: success, Turn 2: all retries fail, Turn 3: success."""
            responses = []
            # Turn 1: success
            responses.append(json.dumps(make_action_response(
                "2H", tentative_rule="even ranks", confidence_level=5, guess_rule=False,
            )))
            # Turn 2: all retries fail
            for _ in range(max_retries):
                responses.append(TruncationError("truncated"))
            # Turn 3: success
            responses.append(json.dumps(make_action_response(
                "2H", tentative_rule="even ranks", confidence_level=3, guess_rule=False,
            )))
            return FakeLLMClient(responses)

        results = []
        for _ in range(2):
            client = _make_mixed_client()
            compiler = FakeLLMClient()

            with (
                patch("eleusis.runner.create_client", return_value=client),
                patch("eleusis.runner.create_client_from_config", return_value=compiler),
            ):
                result = play_round(config, round_number=1)

            results.append(_normalize_result(result))

        assert json.dumps(results[0], sort_keys=True) == json.dumps(results[1], sort_keys=True)

    def test_fallback_cards_come_from_hand(self):
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
            assert played in hand, (
                f"Fallback card {played} not in hand {hand}"
            )
