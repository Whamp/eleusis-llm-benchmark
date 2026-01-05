"""Card system for Eleusis: Card, Deck, and Hand classes."""

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum

__all__ = ["Suit", "Card", "Deck", "Hand"]


class Suit(Enum):
    """Card suits with symbols."""

    HEARTS = ("hearts", "♥", "red")
    DIAMONDS = ("diamonds", "♦", "red")
    CLUBS = ("clubs", "♣", "black")
    SPADES = ("spades", "♠", "black")

    def __init__(self, name: str, symbol: str, color: str):
        self.suit_name = name
        self.symbol = symbol
        self.color = color


@dataclass(frozen=True)
class Card:
    """Immutable playing card with rank and suit."""

    rank: int  # 1 (Ace) through 13 (King)
    suit: Suit

    @property
    def color(self) -> str:
        """Color of the card (red or black)."""
        return self.suit.color

    @property
    def rank_name(self) -> str:
        """Human-readable rank name."""
        rank_names = {
            1: "A",
            11: "J",
            12: "Q",
            13: "K",
        }
        return rank_names.get(self.rank, str(self.rank))

    def __str__(self) -> str:
        """String representation like 5♥ or K♠."""
        return f"{self.rank_name}{self.suit.symbol}"

    def to_dict(self) -> dict[str, int | str]:
        """Convert card to dictionary for JSON serialization."""
        return {
            "rank": self.rank,
            "suit": self.suit.suit_name,
            "color": self.color,
            "symbol": str(self),
        }

NUM_DECKS = 2
NUM_CARDS = 13
class Deck:
    """Double deck of 104 cards (2 standard 52-card decks)."""

    def __init__(self) -> None:
        self._cards: deque[Card] = deque()
        self._initialize()

    def _initialize(self) -> None:
        """Create 104 cards (2 of each rank-suit combination)."""
        for _ in range(NUM_DECKS):
            for suit in Suit:
                for rank in range(1, NUM_CARDS + 1):
                    self._cards.append(Card(rank, suit))

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        cards_list = list(self._cards)
        random.shuffle(cards_list)
        self._cards = deque(cards_list)

    def draw(self) -> Card:
        """Draw a card from the top of the deck."""
        if self.is_empty():
            raise ValueError("Cannot draw from empty deck")
        return self._cards.popleft()

    def is_empty(self) -> bool:
        """Check if deck is empty."""
        return len(self._cards) == 0

    def remaining_count(self) -> int:
        """Get number of cards remaining in deck."""
        return len(self._cards)

    def add_cards(self, cards: list[Card]) -> None:
        """Add cards back to the deck (for reshuffling discarded cards)."""
        self._cards.extend(cards)


class Hand:
    """Player's hand of cards."""

    def __init__(self) -> None:
        self._cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        """Add a card to the hand."""
        self._cards.append(card)

    def remove_card(self, card: Card) -> None:
        """Remove a specific card from the hand."""
        if card not in self._cards:
            raise ValueError(f"Card {card} not in hand")
        self._cards.remove(card)

    def size(self) -> int:
        """Get number of cards in hand."""
        return len(self._cards)

    def contains(self, card: Card) -> bool:
        """Check if hand contains a specific card."""
        return card in self._cards

    def get_all_cards(self) -> list[Card]:
        """Get all cards in hand (returns a copy)."""
        return list(self._cards)

    def clear(self) -> None:
        """Remove all cards from hand."""
        self._cards.clear()

    def to_dict(self) -> list[dict[str, int | str]]:
        """Convert hand to list of card dictionaries for JSON serialization."""
        return [card.to_dict() for card in self._cards]
