"""LLM-based player for Eleusis card game."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from eleusis.game.cards import Card, Suit
from eleusis.llm.base import TruncationError

__all__ = ["LLMScientist"]

if TYPE_CHECKING:
    from eleusis.game.engine import Action, GameEngine
    from eleusis.game.state import GameState
    from eleusis.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMScientist:
    """Scientist player using LLM for decision making."""

    def __init__(
        self,
        name: str,
        llm_client: BaseLLMClient,
        max_retries: int = 3,
        engine: GameEngine | None = None,
        max_turns: int = 40,
    ) -> None:
        """Initialize scientist with name and LLM client."""
        self.name = name
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.engine = engine
        self.max_turns = max_turns
        self.play_history: list[dict] = []
        self.last_action_response: dict | None = None
        # Retry tracking (reset each turn)
        self.last_retry_count: int = 0
        self.last_retry_causes: list[dict] = []

    def get_action(self, game_state: GameState) -> Action:
        """Get an action for the current game state."""
        return self._select_move(game_state)

    def _select_move(self, game_state: GameState) -> Action:
        """Select a card to play using LLM."""
        from eleusis.game.engine import PlayCardAction
        from eleusis.prompts import get_action_prompt

        # Reset retry tracking for this turn
        self.last_retry_count = 0
        self.last_retry_causes = []

        REASONING_HINT = "\n\nIMPORTANT: DO NOT REASON TOO LONG ABOUT THIS."

        player = game_state.player
        hand_cards = player.hand.get_all_cards()

        if not hand_cards:
            logger.error("Empty hand - should not happen!")
            return PlayCardAction(random.choice(hand_cards)) if hand_cards else None

        hand_dicts = [c.to_dict() for c in hand_cards]
        compact_board = game_state.to_compact_string()
        deck_remaining = game_state.deck.remaining_count()
        failed_guesses = game_state.failed_rule_guesses
        current_turn = game_state.turn_number
        failed_guess_count = self.engine.failed_guess_count if self.engine else 0

        base_prompt = get_action_prompt(
            compact_board=compact_board,
            hand_cards=hand_dicts,
            deck_remaining=deck_remaining,
            play_history=self.play_history,
            failed_guesses=failed_guesses,
            current_turn=current_turn,
            max_turns=self.max_turns,
            failed_guess_count=failed_guess_count,
        )

        for attempt in range(self.max_retries):
            cause = None
            try:
                # Add hint on retries (attempt >= 1)
                prompt = base_prompt + REASONING_HINT if attempt > 0 else base_prompt

                response = self.llm_client.generate(prompt, xml_tag="ACTION", return_dict=True)
                self.last_action_response = response

                card_value = response.get("card", "").strip()
                card = self._parse_card(card_value, hand_cards)

                if card:
                    logger.info(f"{self.name} plays {card}")
                    tentative = response.get("tentative_rule", "")
                    if tentative:
                        logger.debug(f"{self.name}'s tentative rule: {tentative}")
                    return PlayCardAction(card)

                # Card parsing failed
                cause = "card_parse_error"
                logger.warning(f"{self.name} attempt {attempt + 1}: {cause} - card='{card_value}'")

            except TruncationError as e:
                cause = "max_token_reached"
                logger.warning(f"{self.name} attempt {attempt + 1}: {cause} - {e}")

            except Exception as e:
                cause = "other_error"
                logger.warning(
                    f"{self.name} attempt {attempt + 1}: {cause} - {type(e).__name__}: {e}"
                )

            # Track this failed attempt
            if cause:
                self.last_retry_count = attempt + 1
                self.last_retry_causes.append({"attempt": attempt + 1, "cause": cause})

        logger.warning(
            f"{self.name} using random fallback after {self.max_retries} failed attempts"
        )
        return PlayCardAction(random.choice(hand_cards))

    def record_action_result(self, result: dict) -> None:
        """Record the result of an action in play history."""
        if result.get("success") and "card" in result:
            reasoning = ""
            if self.last_action_response:
                reasoning = self.last_action_response.get("reasoning_summary", "")
            self.record_play(
                card_str=result["card"],
                accepted=result.get("accepted", False),
                reasoning_summary=reasoning,
            )

    def _parse_card(self, card_value: str, hand_cards: list[Card]) -> Card | None:
        """Parse card value string to Card object from hand."""
        suit_map = {
            "♥": Suit.HEARTS,
            "♦": Suit.DIAMONDS,
            "♣": Suit.CLUBS,
            "♠": Suit.SPADES,
            "hearts": Suit.HEARTS,
            "diamonds": Suit.DIAMONDS,
            "clubs": Suit.CLUBS,
            "spades": Suit.SPADES,
        }
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}

        if not card_value:
            return None

        card_value = card_value.strip()

        suit = None
        rank_str = card_value

        for symbol, s in suit_map.items():
            if symbol in card_value:
                suit = s
                rank_str = card_value.replace(symbol, "").strip()
                break

        if not suit:
            return None

        rank_str = rank_str.upper()
        if rank_str in rank_map:
            rank = rank_map[rank_str]
        else:
            try:
                rank = int(rank_str)
            except ValueError:
                return None

        for card in hand_cards:
            if card.rank == rank and card.suit == suit:
                return card

        return None

    def record_play(self, card_str: str, accepted: bool, reasoning_summary: str = "") -> None:
        """Record a play attempt in history."""
        self.play_history.append({
            "card": card_str,
            "accepted": accepted,
            "reasoning_summary": reasoning_summary,
        })
