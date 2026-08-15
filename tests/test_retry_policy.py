"""Tests for cause-specific retry recovery in player.py.

Verifies:
- Truncation retries inject a distinct "output ONLY the ACTION XML" instruction
- Parse-error retries inject a distinct card-format correction instruction
- Retry causes are tagged and recorded in last_retry_causes
- max_retries is respected (no infinite loop)
"""

from __future__ import annotations

import random

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import GameEngine, PlayCardAction, Rule
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
    for card in hand_cards:
        state.player.hand.add_card(card)

    scientist = LLMScientist(
        "test-player",
        fake_client,
        max_retries=max_retries,
        engine=engine,
        max_turns=40,
        rng=rng or random.Random(42),
    )
    return scientist, engine, state


SAMPLE_HAND = [
    Card(2, Suit.HEARTS),
    Card(5, Suit.SPADES),
    Card(7, Suit.DIAMONDS),
    Card(10, Suit.CLUBS),
]


class TestTruncationRetry:
    """Truncation retries use a distinct recovery instruction."""

    def test_truncation_retry_uses_specific_prompt(self) -> None:
        """Check that a TruncationError retry prompt contains a specific instruction.

        an instruction about outputting ONLY the ACTION XML block.
        """
        # First call: truncation error, second call: success
        client = FakeLLMClient(
            [
                TruncationError("truncated"),
                make_action_response("2♥"),
            ]
        )

        scientist, _, state = _make_scientist(client, SAMPLE_HAND, max_retries=3)
        scientist.get_action(state)

        assert len(client.prompts_seen) == 2
        retry_prompt = client.prompts_seen[1]
        # Should contain truncation-specific instruction, not just generic hint
        assert "output" in retry_prompt.lower() and "only" in retry_prompt.lower()
        assert "ACTION" in retry_prompt

    def test_truncation_cause_recorded(self) -> None:
        """TruncationError retries are tagged as 'max_token_reached'."""
        client = FakeLLMClient(
            [
                TruncationError("truncated"),
                make_action_response("2♥"),
            ]
        )

        scientist, _, state = _make_scientist(client, SAMPLE_HAND, max_retries=3)
        scientist.get_action(state)

        assert scientist.last_retry_count == 1
        assert len(scientist.last_retry_causes) == 1
        assert scientist.last_retry_causes[0]["cause"] == "max_token_reached"


class TestCardParseErrorRetry:
    """Parse-error retries use a distinct card-format correction."""

    def test_parse_error_retry_uses_specific_prompt(self) -> None:
        """Check that a card parse retry prompt contains a format instruction.

        card-format correction instructions.
        """
        # First call: returns unparseable card, second call: success
        client = FakeLLMClient(
            [
                make_action_response("invalid_card_xyz"),
                make_action_response("2♥"),
            ]
        )

        scientist, _, state = _make_scientist(client, SAMPLE_HAND, max_retries=3)
        scientist.get_action(state)

        assert len(client.prompts_seen) == 2
        retry_prompt = client.prompts_seen[1]
        # Should contain card-format-specific instructions
        assert "♥" in retry_prompt or "♠" in retry_prompt or "♦" in retry_prompt

    def test_parse_error_cause_recorded(self) -> None:
        """Card parse errors are tagged as 'card_parse_error'."""
        client = FakeLLMClient(
            [
                make_action_response("not_a_card"),
                make_action_response("2♥"),
            ]
        )

        scientist, _, state = _make_scientist(client, SAMPLE_HAND, max_retries=3)
        scientist.get_action(state)

        assert scientist.last_retry_count == 1
        assert len(scientist.last_retry_causes) == 1
        assert scientist.last_retry_causes[0]["cause"] == "card_parse_error"


class TestDistinctRetryInstructions:
    """Truncation and parse-error retries use DIFFERENT instructions."""

    def test_truncation_and_parse_error_prompts_differ(self) -> None:
        """The retry instruction for truncation must differ from parse error."""
        # Run 1: truncation then success
        client1 = FakeLLMClient(
            [
                TruncationError("truncated"),
                make_action_response("2♥"),
            ]
        )
        scientist1, _, state1 = _make_scientist(client1, SAMPLE_HAND, max_retries=3)
        scientist1.get_action(state1)
        truncation_retry_prompt = client1.prompts_seen[1]

        # Run 2: parse error then success
        client2 = FakeLLMClient(
            [
                make_action_response("bad_card"),
                make_action_response("2♥"),
            ]
        )
        scientist2, _, state2 = _make_scientist(client2, SAMPLE_HAND, max_retries=3)
        scientist2.get_action(state2)
        parse_error_retry_prompt = client2.prompts_seen[1]

        # The hints appended should be different
        assert truncation_retry_prompt != parse_error_retry_prompt


class TestMaxRetriesRespected:
    """max_retries bounds the number of LLM attempts."""

    def test_max_retries_respected(self) -> None:
        """After max_retries failed attempts, falls back to random card."""
        client = FakeLLMClient(
            [
                TruncationError("truncated"),
                TruncationError("truncated"),
                TruncationError("truncated"),
            ]
        )

        scientist, _, state = _make_scientist(
            client,
            SAMPLE_HAND,
            max_retries=3,
            rng=random.Random(42),
        )
        action = scientist.get_action(state)

        # Should have used all retries and fallen back
        assert isinstance(action, PlayCardAction)
        assert scientist.last_retry_count == 3
        assert len(scientist.last_retry_causes) == 3
        # The action should still be valid (random fallback)
        assert action.card in SAMPLE_HAND

    def test_no_extra_calls_beyond_max_retries(self) -> None:
        """Exactly max_retries calls are made, no more."""
        client = FakeLLMClient(
            [
                TruncationError("t1"),
                TruncationError("t2"),
            ]
        )

        scientist, _, state = _make_scientist(
            client,
            SAMPLE_HAND,
            max_retries=2,
            rng=random.Random(42),
        )
        scientist.get_action(state)

        assert client._call_count == 2
