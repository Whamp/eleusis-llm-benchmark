"""Player classes for Eleusis card game."""

import logging
from eleusis.cards import Card, Suit

logger = logging.getLogger(__name__)


class LLMScientist:
    """Base Scientist player using LLM for decision making."""

    def __init__(self, name: str, llm_client, max_retries: int = 3):
        """Initialize scientist with name and LLM client."""
        self.name = name
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.play_history: list[dict] = []
        self.last_action_response: dict | None = None

    def select_move(self, game_state, current_player):
        """Select a move. Must be overridden by subclass."""
        return self._select_move(game_state, current_player)

    def _select_move(self, game_state, current_player):
        """Select a move (to be overridden)."""
        raise NotImplementedError("Subclasses must implement _select_move")

    def _parse_card(self, card_value: str, hand_cards: list[Card]) -> Card | None:
        """Parse card value string to Card object from hand."""
        # Handle Unicode suit symbols
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

        # Handle J/Q/K/A values
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}

        if not card_value:
            return None

        # Try to extract rank and suit
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

    def record_play(self, card: Card, accepted: bool, reasoning_summary: str = ""):
        """Record a play attempt in history."""
        self.play_history.append({
            "card": str(card),
            "accepted": accepted,
            "reasoning_summary": reasoning_summary,
        })
