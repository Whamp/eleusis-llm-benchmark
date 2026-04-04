"""Tests for player turn-state correctness.

Verifies:
- last_action_response is cleared at the start of each _select_move call
- fallback card selection uses seeded RNG, not global random
- deterministic fallback behavior across identical seeded runs
"""

from __future__ import annotations

import random

from eleusis.game.cards import Card
from eleusis.game.engine import GameEngine, Rule
from eleusis.game.state import GameState
from eleusis.llm.base import TruncationError
from eleusis.player import LLMScientist
from tests.conftest import FakeLLMClient, make_action_response


def _make_scientist(
    fake_client: FakeLLMClient,
    hand_cards: list[Card],
    rule: Rule | None = None,
    max_retries: int = 3,
    rng: random.Random | None = None,
) -> tuple[LLMScientist, GameEngine, GameState]:
    """Set up a scientist with a fake client and pre-loaded hand."""
    rule = rule or Rule("Only even ranks.", "return card.rank % 2 == 0")
    state = GameState("test-player")
    engine = GameEngine(
        state,
        rule,
        rule_compiler_client=FakeLLMClient(),
        hand_size=len(hand_cards),
        wrong_guess_penalty=3,
    )
    # Manually set up hand instead of engine.setup_game()
    for card in hand_cards:
        state.player.hand.add_card(card)

    scientist = LLMScientist(
        "test-player",
        fake_client,
        max_retries=max_retries,
        engine=engine,
        max_turns=40,
        rng=rng,
    )
    return scientist, engine, state


class TestStaleActionResponse:
    """last_action_response must be None after failed retries."""

    def test_cleared_on_entry(self, sample_hand):
        """After all retries fail, last_action_response should be None."""
        # All attempts raise TruncationError
        errors = [TruncationError("truncated")] * 3
        client = FakeLLMClient(errors)

        rng = random.Random(42)
        scientist, engine, state = _make_scientist(
            client, sample_hand, max_retries=3, rng=rng,
        )

        # Seed a stale value
        scientist.last_action_response = {"card": "stale", "reasoning_summary": "old"}

        scientist.get_action(state)

        # Should be cleared, not stale
        assert scientist.last_action_response is None

    def test_set_on_success(self, sample_hand):
        """On a successful parse, last_action_response is the new response."""
        resp = make_action_response("2♥")
        client = FakeLLMClient([resp])

        rng = random.Random(42)
        scientist, _, state = _make_scientist(client, sample_hand, rng=rng)
        scientist.last_action_response = {"card": "stale"}

        scientist.get_action(state)

        assert scientist.last_action_response is not None
        assert scientist.last_action_response["card"] == "2♥"


class TestSeededFallback:
    """Fallback card must come from a seeded RNG, not global random."""

    def test_deterministic_fallback(self, sample_hand):
        """Two runs with same seed must pick the same fallback card."""
        errors = [TruncationError("truncated")] * 3

        results = []
        for _ in range(2):
            client = FakeLLMClient(list(errors))
            rng = random.Random(99)
            scientist, _, state = _make_scientist(
                client, sample_hand, max_retries=3, rng=rng,
            )
            action = scientist.get_action(state)
            results.append(action.card)

        assert results[0] == results[1], "Same seed must produce same fallback"

    def test_different_seeds_can_differ(self, sample_hand):
        """Different seeds should (generally) produce different fallback cards.

        With 4 cards in hand, probability of same card with different seeds is 1/4.
        We try a few seed pairs to find one that differs.
        """
        errors = [TruncationError("truncated")] * 3

        found_difference = False
        for seed_a, seed_b in [(1, 1000), (2, 2000), (3, 3000)]:
            cards = []
            for seed in [seed_a, seed_b]:
                client = FakeLLMClient(list(errors))
                rng = random.Random(seed)
                scientist, _, state = _make_scientist(
                    client, sample_hand, max_retries=3, rng=rng,
                )
                action = scientist.get_action(state)
                cards.append(action.card)
            if cards[0] != cards[1]:
                found_difference = True
                break

        assert found_difference, "Different seeds should produce different fallbacks"
