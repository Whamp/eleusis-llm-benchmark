"""Player classes for Eleusis card game."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eleusis.cards import Card, Suit

__all__ = ["LLMScientist"]

if TYPE_CHECKING:
    from eleusis.game_engine import Action
    from eleusis.game_state import GameState, PlayerState
    from eleusis.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMScientist:
    """Base Scientist player using LLM for decision making."""

    def __init__(self, name: str, llm_client: BaseLLMClient, max_retries: int = 3) -> None:
        """Initialize scientist with name and LLM client."""
        self.name = name
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.play_history: list[dict] = []
        self.last_action_response: dict | None = None

    def get_action(self, game_state: GameState) -> Action:
        """Get an action for the current game state."""
        current_player = game_state.get_current_player()
        return self._select_move(game_state, current_player)

    def _select_move(self, game_state: GameState, current_player: PlayerState) -> Action:
        """Select a move (to be overridden by subclass)."""
        raise NotImplementedError("Subclasses must implement _select_move")

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

        # Find suit symbol
        suit = None
        rank_str = card_value

        for symbol, s in suit_map.items():
            if symbol in card_value:
                suit = s
                rank_str = card_value.replace(symbol, "").strip()
                break

        if not suit:
            return None

        # Parse rank
        rank_str = rank_str.upper()
        if rank_str in rank_map:
            rank = rank_map[rank_str]
        else:
            try:
                rank = int(rank_str)
            except ValueError:
                return None

        # Find matching card in hand
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
