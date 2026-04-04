"""Tests for shadow evaluation modes.

Verifies:
- offline mode: runner records tentative_rule + confidence but skips engine.evaluate_rule()
- disabled mode: runner records no shadow guess_attempt entries
- online mode: runner performs shadow evaluation (existing behavior)
- offline analysis script: produces shadow metrics from saved results
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule
from eleusis.runner import play_round
from tests.conftest import FakeLLMClient, make_action_response

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MINI_RULES_PATH = FIXTURES_DIR / "mini_rules.json"


def _make_config(shadow_mode: str = "offline", max_turns: int = 3) -> dict:
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
            "max_llm_retries": 1,
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


def _fake_client_with_plays(n_turns: int, confidence: int = 8) -> FakeLLMClient:
    """Create a FakeLLMClient that plays cards with high confidence, never guessing."""
    # We'll generate enough responses. The actual card played doesn't matter
    # since the game engine handles acceptance logic.
    responses = []
    for _ in range(n_turns):
        # Use a placeholder card — LLMScientist.get_action will pick from hand
        resp = make_action_response(
            "2H",  # placeholder, will be adjusted by scientist
            tentative_rule="Only even ranks",
            confidence_level=confidence,
            guess_rule=False,
        )
        responses.append(json.dumps(resp))
    return FakeLLMClient(responses)


class TestOfflineMode:
    """shadow_mode=offline records tentative_rule + confidence but skips evaluate_rule."""

    def test_no_shadow_evaluation_in_game_loop(self):
        """In offline mode, engine.evaluate_rule() is never called for shadow."""
        config = _make_config(shadow_mode="offline", max_turns=3)
        n_turns = config["game"]["max_turns"]

        player_client = _fake_client_with_plays(n_turns, confidence=8)
        compiler_client = FakeLLMClient()

        with (
            patch("eleusis.runner.create_client", return_value=player_client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler_client),
        ):
            # Spy on GameEngine.evaluate_rule
            with patch(
                "eleusis.runner.GameEngine.evaluate_rule",
                return_value=(False, "mocked", {}),
            ) as mock_eval:
                play_round(config, round_number=1)

        mock_eval.assert_not_called()

    def test_tentative_rule_data_recorded(self):
        """Offline mode records confidence and tentative rule in turn data."""
        config = _make_config(shadow_mode="offline", max_turns=2)

        player_client = _fake_client_with_plays(2, confidence=8)
        compiler_client = FakeLLMClient()

        with (
            patch("eleusis.runner.create_client", return_value=player_client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler_client),
        ):
            result = play_round(config, round_number=1)

        # Every turn should have confidence_level recorded
        for turn in result["turns"]:
            assert "confidence_level" in turn
            assert "confidence_level_raw" in turn

    def test_shadow_guess_attempt_has_evaluated_false(self):
        """Offline shadow entries have evaluated=False marker."""
        config = _make_config(shadow_mode="offline", max_turns=2)

        player_client = _fake_client_with_plays(2, confidence=8)
        compiler_client = FakeLLMClient()

        with (
            patch("eleusis.runner.create_client", return_value=player_client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler_client),
        ):
            result = play_round(config, round_number=1)

        shadow_turns = [
            t for t in result["turns"]
            if t.get("guess_attempt") and t["guess_attempt"].get("shadow")
        ]
        # With confidence=8 (>= 5 threshold), at least some turns should have shadow entries
        assert len(shadow_turns) > 0
        for t in shadow_turns:
            assert t["guess_attempt"]["evaluated"] is False
            assert "guess" in t["guess_attempt"]


class TestDisabledMode:
    """shadow_mode=disabled records no shadow guess_attempt entries."""

    def test_no_shadow_entries(self):
        """Disabled mode produces no shadow guess_attempt regardless of confidence."""
        config = _make_config(shadow_mode="disabled", max_turns=2)

        player_client = _fake_client_with_plays(2, confidence=9)
        compiler_client = FakeLLMClient()

        with (
            patch("eleusis.runner.create_client", return_value=player_client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler_client),
        ):
            with patch(
                "eleusis.runner.GameEngine.evaluate_rule",
                return_value=(False, "mocked", {}),
            ) as mock_eval:
                result = play_round(config, round_number=1)

        mock_eval.assert_not_called()
        shadow_turns = [
            t for t in result["turns"]
            if t.get("guess_attempt") and t["guess_attempt"].get("shadow")
        ]
        assert len(shadow_turns) == 0


class TestOnlineMode:
    """shadow_mode=online preserves existing shadow evaluation behavior."""

    def test_evaluate_rule_called_for_high_confidence(self):
        """Online mode calls engine.evaluate_rule() for high-confidence tentative rules."""
        config = _make_config(shadow_mode="online", max_turns=2)

        player_client = _fake_client_with_plays(2, confidence=8)
        compiler_client = FakeLLMClient()
        # Need compiler to return code for shadow evaluation
        compiler_client.convert_rule_to_code = MagicMock(return_value={
            "code": "return card.rank % 2 == 0",
            "status": "success",
            "attempts": 1,
            "sleep_cycles": 0,
            "provider_used": "fake/fake-model",
        })

        with (
            patch("eleusis.runner.create_client", return_value=player_client),
            patch("eleusis.runner.create_client_from_config", return_value=compiler_client),
        ):
            result = play_round(config, round_number=1)

        # With online mode and high confidence, shadow turns should have been evaluated
        shadow_turns = [
            t for t in result["turns"]
            if t.get("guess_attempt") and t["guess_attempt"].get("shadow")
        ]
        assert len(shadow_turns) > 0
        for t in shadow_turns:
            # Online mode should have full evaluation results
            assert "correct" in t["guess_attempt"]
            assert isinstance(t["guess_attempt"]["correct"], bool)


class TestOfflineAnalysisScript:
    """Offline analysis script produces shadow metrics from saved results."""

    def test_evaluate_shadows_offline(self):
        """evaluate_shadow_turns takes unevaluated shadow entries and evaluates them."""
        from scripts.evaluate_shadows import evaluate_shadow_turns

        turns = [
            {
                "turn_number": 1,
                "guess_attempt": {
                    "guess": "Only even ranks",
                    "shadow": True,
                    "evaluated": False,
                },
                "confidence_level": 8,
                "confidence_level_raw": 8,
            },
            {
                "turn_number": 2,
                "guess_attempt": None,
                "confidence_level": 3,
                "confidence_level_raw": 3,
            },
            {
                "turn_number": 3,
                "guess_attempt": {
                    "guess": "Red cards only",
                    "shadow": True,
                    "evaluated": False,
                },
                "confidence_level": 7,
                "confidence_level_raw": 7,
            },
        ]

        rule = Rule("Only even ranks.", "return card.rank % 2 == 0")

        compiler = FakeLLMClient()
        compiler.convert_rule_to_code = MagicMock(side_effect=[
            {
                "code": "return card.rank % 2 == 0",
                "status": "success",
                "attempts": 1,
                "sleep_cycles": 0,
                "provider_used": "fake/fake-model",
            },
            {
                "code": 'return card.color == "red"',
                "status": "success",
                "attempts": 1,
                "sleep_cycles": 0,
                "provider_used": "fake/fake-model",
            },
        ])

        mainline = [Card(4, Suit.HEARTS)]

        augmented = evaluate_shadow_turns(
            turns=turns,
            actual_rule=rule,
            mainline=mainline,
            rule_compiler_client=compiler,
            num_simulations=10,
            turns_per_simulation=10,
            simulation_seed=42,
        )

        # Turn 1 should be evaluated (was unevaluated shadow)
        assert augmented[0]["guess_attempt"]["evaluated"] is True
        assert "correct" in augmented[0]["guess_attempt"]
        assert isinstance(augmented[0]["guess_attempt"]["correct"], bool)

        # Turn 2 should be unchanged (no guess)
        assert augmented[1]["guess_attempt"] is None

        # Turn 3 should be evaluated
        assert augmented[2]["guess_attempt"]["evaluated"] is True
        assert "correct" in augmented[2]["guess_attempt"]
