"""Card system for Eleusis: Card, Deck, and Hand classes."""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

__all__ = ["Card", "Deck", "Hand", "Suit"]


class Suit(Enum):
    """Card suits with symbols."""

    HEARTS = ("hearts", "♥", "red")
    DIAMONDS = ("diamonds", "♦", "red")
    CLUBS = ("clubs", "♣", "black")
    SPADES = ("spades", "♠", "black")

    def __init__(self, name: str, symbol: str, color: str) -> None:
        """Initialize a suit's machine name, display symbol, and color."""
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
        """Convert card to dictionary for presentation-oriented JSON."""
        return {
            "rank": self.rank,
            "suit": self.suit.suit_name,
            "color": self.color,
            "symbol": str(self),
        }

    def to_canonical_card_data(self) -> dict[str, int | str]:
        """Encode the authoritative rank and suit without presentation fields."""
        return {"rank": self.rank, "suit": self.suit.suit_name}

    @classmethod
    def from_canonical_card_data(cls, payload: Mapping[str, object]) -> Card:
        """Decode one strictly canonical rank-and-suit Card."""
        if set(payload) != {"rank", "suit"}:
            raise ValueError("Canonical Card data requires only rank and suit")
        rank = payload["rank"]
        suit_name = payload["suit"]
        if type(rank) is not int:
            raise TypeError("Canonical Card rank must be an integer")
        if not 1 <= rank <= NUM_CARDS:
            raise ValueError("Canonical Card rank must be from 1 through 13")
        if not isinstance(suit_name, str):
            raise TypeError("Canonical Card suit must be a suit name")
        suit = next(
            (candidate for candidate in Suit if candidate.suit_name == suit_name),
            None,
        )
        if suit is None:
            raise ValueError(f"Canonical Card suit is unknown: {suit_name}")
        return cls(rank=rank, suit=suit)


NUM_DECKS = 2
NUM_CARDS = 13


class Deck:
    """Double deck of 104 cards (2 standard 52-card decks)."""

    def __init__(self) -> None:
        """Initialize a complete ordered double deck before shuffling."""
        self._cards: deque[Card] = deque()
        self._initialize()

    def _initialize(self) -> None:
        """Create 104 cards (2 of each rank-suit combination)."""
        for _ in range(NUM_DECKS):
            for suit in Suit:
                for rank in range(1, NUM_CARDS + 1):
                    self._cards.append(Card(rank, suit))

    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the deck in place.

        Args:
            seed: Random seed for reproducible shuffling. If None, uses global RNG.
        """
        cards_list = list(self._cards)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(cards_list)
        else:
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

    def snapshot_deck_cards(self) -> list[dict[str, int | str]]:
        """Capture the remaining deck in exact draw order using canonical Cards."""
        return [card.to_canonical_card_data() for card in self._cards]

    @classmethod
    def restore_deck_cards(cls, payloads: Sequence[Mapping[str, object]]) -> Deck:
        """Restore a remaining deck without initializing or shuffling it again."""
        deck = cls()
        deck._cards = deque(Card.from_canonical_card_data(item) for item in payloads)
        return deck


class Hand:
    """Player's hand of cards."""

    def __init__(self) -> None:
        """Initialize an empty player hand."""
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
        """Convert hand to presentation-oriented card dictionaries."""
        return [card.to_dict() for card in self._cards]

    def snapshot_hand_cards(self) -> list[dict[str, int | str]]:
        """Capture the hand in exact order using canonical Cards."""
        return [card.to_canonical_card_data() for card in self._cards]

    @classmethod
    def restore_hand_cards(cls, payloads: Sequence[Mapping[str, object]]) -> Hand:
        """Restore hand order and duplicate-card multiplicity."""
        hand = cls()
        hand._cards = [Card.from_canonical_card_data(item) for item in payloads]
        return hand
