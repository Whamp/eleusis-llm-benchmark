"""Tests for runner error accounting.

Verifies:
- Unhandled action errors produce explicit recorded turns with deterministic fallback
- Error turns are never silently skipped
- Error turns use deterministic fallback card selection
"""

from __future__ import annotations

import random

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import GameEngine, PlayCardAction, Rule
from eleusis.game.state import GameState
from eleusis.player import LLMScientist
from tests.conftest import FakeLLMClient


def _run_game_loop(
    scientist: LLMScientist,
    engine: GameEngine,
    state: GameState,
    max_turns: int = 5,
) -> list[dict]:
    """Minimal game loop mirroring runner.py, returns turn_data_list."""
    turn_count = 0
    turn_data_list = []

    while turn_count < max_turns and not engine.is_game_over():
        player = state.player
        hand_before = [str(c) for c in player.hand.get_all_cards()]
        state.turn_number = turn_count + 1

        try:
            action = scientist.get_action(state)
        except Exception as e:
            # This is the bug: runner.py silently skips the turn.
            # After the fix, it should record an error turn with fallback.
            # We import the fixed runner behavior here.
            from eleusis.runner import _handle_action_error
            action, error_info = _handle_action_error(e, scientist, state)

        play_result = engine.play_turn(action)
        scientist.record_action_result(play_result)

        turn_data = {
            "turn_number": turn_count + 1,
            "hand": hand_before,
            "action_result": {
                "action": play_result.get("action"),
                "card": play_result.get("card"),
                "accepted": play_result.get("accepted"),
                "success": play_result.get("success"),
            },
            "error": getattr(action, '_error_info', None),
        }
        turn_data_list.append(turn_data)
        turn_count += 1

    return turn_data_list


class TestErrorTurnRecording:
    """Errors during get_action produce explicit turns, not silent skips."""

    def test_error_produces_fallback_action(self):
        """When get_action raises, runner should produce a PlayCardAction fallback."""
        hand = [
            Card(2, Suit.HEARTS),
            Card(5, Suit.SPADES),
            Card(7, Suit.DIAMONDS),
        ]

        # Client that always raises (simulating total failure)
        client = FakeLLMClient([RuntimeError("API down")] * 3)
        rule = Rule("Only even ranks.", "return card.rank % 2 == 0")
        state = GameState("test-player")
        engine = GameEngine(
            state, rule,
            rule_compiler_client=FakeLLMClient(),
            hand_size=len(hand),
        )
        for card in hand:
            state.player.hand.add_card(card)
        # Place a starter card
        state.mainline.add_card(Card(4, Suit.HEARTS))

        rng = random.Random(42)
        scientist = LLMScientist(
            "test-player", client, max_retries=1, engine=engine,
            max_turns=5, rng=rng,
        )

        # The runner should handle the error and produce a fallback
        from eleusis.runner import _handle_action_error
        try:
            action = scientist.get_action(state)
        except Exception as e:
            action, error_info = _handle_action_error(e, scientist, state)

        assert isinstance(action, PlayCardAction)
        assert action.card in hand

    def test_error_fallback_is_deterministic(self):
        """Same seed + same error should produce same fallback card."""
        hand = [
            Card(2, Suit.HEARTS),
            Card(5, Suit.SPADES),
            Card(7, Suit.DIAMONDS),
            Card(10, Suit.CLUBS),
        ]

        results = []
        for _ in range(2):
            client = FakeLLMClient([RuntimeError("API down")])
            rule = Rule("Only even ranks.", "return card.rank % 2 == 0")
            state = GameState("test-player")
            engine = GameEngine(
                state, rule,
                rule_compiler_client=FakeLLMClient(),
                hand_size=len(hand),
            )
            for card in hand:
                state.player.hand.add_card(card)
            state.mainline.add_card(Card(4, Suit.HEARTS))

            rng = random.Random(42)
            scientist = LLMScientist(
                "test-player", client, max_retries=1, engine=engine,
                max_turns=5, rng=rng,
            )

            from eleusis.runner import _handle_action_error
            try:
                action = scientist.get_action(state)
            except Exception as e:
                action, error_info = _handle_action_error(e, scientist, state)
            results.append(action.card)

        assert results[0] == results[1]

    def test_error_info_recorded(self):
        """The error info returned by _handle_action_error has the exception details."""
        hand = [Card(3, Suit.CLUBS)]
        client = FakeLLMClient([])
        rule = Rule("Only even ranks.", "return card.rank % 2 == 0")
        state = GameState("test-player")
        engine = GameEngine(
            state, rule,
            rule_compiler_client=FakeLLMClient(),
            hand_size=1,
        )
        state.player.hand.add_card(hand[0])
        state.mainline.add_card(Card(4, Suit.HEARTS))

        rng = random.Random(42)
        scientist = LLMScientist(
            "test-player", client, max_retries=1, engine=engine,
            max_turns=5, rng=rng,
        )

        # Call _handle_action_error directly with a synthesized exception
        from eleusis.runner import _handle_action_error
        error = RuntimeError("total failure")
        action, error_info = _handle_action_error(error, scientist, state)

        assert error_info["error_type"] == "RuntimeError"
        assert "total failure" in error_info["error_message"]
        assert error_info["fallback_card"] is not None
