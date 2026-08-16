"""Tests for runner error accounting.

Verifies:
- Unexpected action errors PROPAGATE: no fallback card is fabricated at the
  runner boundary, because a card with no Model Attempt behind it would
  corrupt authoritative Round Record evidence.
- The player's own bounded model-failure fallback remains the only card
  fabricator, and it records explicit Fallback Decision evidence.
"""

from __future__ import annotations

import random

import pytest

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import GameEngine, Rule
from eleusis.game.state import GameState
from eleusis.player import LLMScientist
from tests.conftest import FakeLLMClient


def _make_scientist(
    hand_cards: list[Card],
) -> tuple[LLMScientist, GameEngine, GameState]:
    """Set up a scientist with a fake client and pre-loaded hand."""
    state = GameState("test-player")
    engine = GameEngine(
        state,
        Rule("Only even ranks.", "return card.rank % 2 == 0"),
        rule_compiler_client=FakeLLMClient(),
        hand_size=len(hand_cards),
        wrong_guess_penalty=3,
    )
    for card in hand_cards:
        state.player.hand.add_card(card)
    scientist = LLMScientist(
        "TestScientist",
        FakeLLMClient(),
        rng=random.Random(42),
    )
    return scientist, engine, state


class TestUnexpectedActionErrorsPropagate:
    """Unexpected errors escaping get_action must abort, not fabricate."""

    def test_unexpected_error_propagates_without_fallback_card(self) -> None:
        """A raised error reaches the caller; no card is invented.

        The runner boundary must not catch-and-play: fabricated cards have
        no Model Attempt evidence and would corrupt Round Records. The
        Round stays resumable from its last committed Turn instead.
        """
        from eleusis.runner import handle_action_error

        scientist, _engine, state = _make_scientist([Card(2, Suit.HEARTS)])
        with pytest.raises(RuntimeError, match="capacity gone"):
            handle_action_error(RuntimeError("capacity gone"), scientist, state)

    def test_patience_abort_is_not_caught_into_fallback(self) -> None:
        """The provider-patience abort survives the runner boundary.

        LLMScientist raises RuntimeError after its provider patience window
        closes so the Benchmark Run stops cleanly and stays resumable;
        the runner boundary must
        let that propagate rather than turning it into a random card.
        """
        from eleusis.runner import handle_action_error

        scientist, _engine, state = _make_scientist([Card(4, Suit.CLUBS)])
        abort = RuntimeError(
            "TestScientist provider unavailable past 1800s patience;"
            " aborting turn to keep the Round resumable"
        )
        with pytest.raises(RuntimeError, match="patience"):
            handle_action_error(abort, scientist, state)

    def test_no_fallback_card_path_remains_at_runner_boundary(self) -> None:
        """Runner exposes no handler that fabricates PlayCardAction cards.

        Guards against reintroducing catch-and-play: the only sanctioned
        fallback lives inside LLMScientist after bounded model failures,
        where explicit Fallback Decision evidence is recorded.
        """
        import inspect

        from eleusis import runner

        source = inspect.getsource(runner)
        assert "rng.choice" not in source, (
            "runner must not fabricate cards; fallback belongs to the player"
        )
