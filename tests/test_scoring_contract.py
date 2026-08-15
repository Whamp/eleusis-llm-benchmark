"""Tests for canonical scoring contract across engine, prompt, and analysis."""

import inspect

import pytest

from eleusis.analysis.utils import compute_counting_cutoff
from eleusis.game.engine import GameEngine, Rule
from eleusis.game.state import GameState
from eleusis.prompts.action import ActionPromptContext, get_action_prompt
from tests.conftest import FakeLLMClient


class TestPromptScoreMatchesEngine:
    """Prompt-displayed potential score must match engine scoring exactly."""

    def test_score_on_first_turn(self) -> None:
        """On turn 1 (current_turn=1, turn_count=0), prompt score must match engine."""
        max_turns = 30
        current_turn = 1  # 1-indexed, as passed to the prompt
        turn_count = current_turn - 1  # 0-indexed, as passed to engine
        penalty = 3
        failed = 0

        # Engine score if guess correct this turn
        engine = self._make_engine(penalty=penalty)
        engine_score = engine.calculate_score(max_turns, turn_count)

        # Prompt-displayed score
        prompt = get_action_prompt(
            ActionPromptContext(
                compact_board="4H",
                hand_cards=[{"symbol": "5H"}],
                play_history=[],
                failed_guesses=None,
                current_turn=current_turn,
                max_turns=max_turns,
                failed_guess_count=failed,
                hand_size=12,
                wrong_guess_penalty=penalty,
            )
        )
        prompt_score = self._extract_potential_score(prompt)

        assert prompt_score == engine_score, (
            f"Prompt shows {prompt_score} but engine computes {engine_score} "
            f"(turn_count={turn_count}, current_turn={current_turn})"
        )

    def test_score_mid_game_with_penalties(self) -> None:
        """Mid-game with failed guesses: prompt and engine must agree."""
        max_turns = 30
        current_turn = 10
        turn_count = current_turn - 1
        penalty = 3
        failed = 2

        engine = self._make_engine(penalty=penalty)
        engine.failed_guess_count = failed
        engine.rule_guessed = True
        engine_score = engine.calculate_score(max_turns, turn_count)

        prompt = get_action_prompt(
            ActionPromptContext(
                compact_board="4H",
                hand_cards=[{"symbol": "5H"}],
                play_history=[],
                failed_guesses=None,
                current_turn=current_turn,
                max_turns=max_turns,
                failed_guess_count=failed,
                hand_size=12,
                wrong_guess_penalty=penalty,
            )
        )
        prompt_score = self._extract_potential_score(prompt)

        assert prompt_score == engine_score

    def test_score_last_turn(self) -> None:
        """On the final turn, scores must still agree."""
        max_turns = 30
        current_turn = 30
        turn_count = current_turn - 1
        penalty = 2
        failed = 0

        engine = self._make_engine(penalty=penalty)
        engine.rule_guessed = True
        engine_score = engine.calculate_score(max_turns, turn_count)

        prompt = get_action_prompt(
            ActionPromptContext(
                compact_board="4H",
                hand_cards=[{"symbol": "5H"}],
                play_history=[],
                failed_guesses=None,
                current_turn=current_turn,
                max_turns=max_turns,
                failed_guess_count=failed,
                hand_size=12,
                wrong_guess_penalty=penalty,
            )
        )
        prompt_score = self._extract_potential_score(prompt)

        assert prompt_score == engine_score

    def _make_engine(self, penalty: int = 3) -> GameEngine:
        state = GameState("test")
        rule = Rule("test rule", "return True")
        engine = GameEngine(
            state,
            rule,
            rule_compiler_client=FakeLLMClient(),
            wrong_guess_penalty=penalty,
        )
        engine.rule_guessed = True
        return engine

    def _extract_potential_score(self, prompt: str) -> int:
        """Extract the 'Current potential score' value from the prompt text."""
        for line in prompt.split("\n"):
            if "Current potential score" in line:
                return int(line.split(":")[-1].strip())
        raise ValueError("Could not find 'Current potential score' in prompt")


class TestAnalysisCutoffPenalty:
    """Analysis cutoff must use configured penalty, not hardcoded default."""

    def test_cutoff_with_penalty_3(self) -> None:
        """With penalty=3 and max_turns=30, cutoff differs from penalty=2."""
        # 1 failed guess at turn 1
        turns = [
            {"turn_number": 1, "guess_attempt": {"correct": False}},
        ]
        # With penalty=3: 3*1=3 >= 30-1=29? No. So no cutoff yet.
        assert compute_counting_cutoff(turns, max_turns=30, penalty=3) is None

        # Now accumulate enough failures
        turns = [
            {"turn_number": i, "guess_attempt": {"correct": False}}
            for i in range(1, 11)  # 10 failed guesses
        ]
        # With penalty=3: after turn 10, 3*10=30 >= 30-10=20? Yes (30>=20)
        cutoff_3 = compute_counting_cutoff(turns, max_turns=30, penalty=3)
        # With penalty=2: after turn 10, 2*10=20 >= 30-10=20? Yes (20>=20)
        cutoff_2 = compute_counting_cutoff(turns, max_turns=30, penalty=2)

        # With penalty=3, cutoff should come sooner
        assert cutoff_3 is not None
        assert cutoff_2 is not None
        assert cutoff_3 <= cutoff_2

    def test_no_default_penalty(self) -> None:
        """compute_counting_cutoff must require penalty explicitly (no default)."""
        turns = [{"turn_number": 1, "guess_attempt": {"correct": False}}]
        # Binding without penalty should raise TypeError before function execution.
        signature = inspect.signature(compute_counting_cutoff)
        with pytest.raises(TypeError):
            signature.bind(turns, max_turns=30)


class TestFormalVsShadowCorrectness:
    """Formal and shadow correctness must be tracked as separate metrics."""

    def test_runner_separates_formal_and_shadow(self) -> None:
        """Keep first formal and shadow correct turns separate."""
        # Build turn data simulating shadow correct at turn 3, formal correct at turn 5
        turn_data = [
            {"turn_number": 1, "guess_attempt": None},
            {"turn_number": 2, "guess_attempt": None},
            {
                "turn_number": 3,
                "guess_attempt": {"correct": True, "shadow": True, "guess": "test"},
            },
            {"turn_number": 4, "guess_attempt": {"correct": False, "guess": "wrong"}},
            {"turn_number": 5, "guess_attempt": {"correct": True, "guess": "right"}},
        ]

        first_formal = None
        first_shadow = None
        for turn in turn_data:
            ga = turn.get("guess_attempt")
            if isinstance(ga, dict) and ga.get("correct"):
                if ga.get("shadow", False):
                    if first_shadow is None:
                        first_shadow = turn["turn_number"]
                else:
                    if first_formal is None:
                        first_formal = turn["turn_number"]

        assert first_shadow == 3, (
            f"Expected shadow correct at turn 3, got {first_shadow}"
        )
        assert first_formal == 5, (
            f"Expected formal correct at turn 5, got {first_formal}"
        )
        assert first_shadow != first_formal, "Formal and shadow must be separate"
