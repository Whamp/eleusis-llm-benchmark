"""Game state management for Eleusis."""

import json
from dataclasses import dataclass, field

from eleusis.game.cards import Card, Deck, Hand

__all__ = ["GameState", "Mainline", "PlayerState", "Sideline"]


class Mainline:
    """Ordered sequence of accepted cards."""

    def __init__(self) -> None:
        """Initialize an empty accepted-card mainline."""
        self._cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        """Add a card to the end of the mainline."""
        self._cards.append(card)

    def get_last(self) -> Card | None:
        """Get the last card in the mainline."""
        return self._cards[-1] if self._cards else None

    def get_all(self) -> list[Card]:
        """Get all cards in the mainline."""
        return list(self._cards)

    def size(self) -> int:
        """Get number of cards in mainline."""
        return len(self._cards)

    def to_dict(self) -> list[dict[str, int | str]]:
        """Convert mainline to list of card dictionaries."""
        return [card.to_dict() for card in self._cards]

    def to_str(self) -> str:
        """Convert mainline to a compact string representation."""
        return " ".join(str(card) for card in self._cards)


class Sideline:
    """Rejected cards below a specific mainline position."""

    def __init__(self, mainline_index: int) -> None:
        """Create sideline for a mainline position."""
        self.mainline_index = mainline_index
        self._cards: list[Card] = []

    def add_card(self, card: Card) -> None:
        """Add a rejected card to this sideline."""
        self._cards.append(card)

    def get_cards(self) -> list[Card]:
        """Get all cards in this sideline."""
        return list(self._cards)

    def to_dict(self) -> dict[str, object]:
        """Convert sideline to dictionary."""
        return {
            "mainline_index": self.mainline_index,
            "cards": [c.to_dict() for c in self._cards],
        }


@dataclass
class PlayerState:
    """State for a single player."""

    name: str
    hand: Hand = field(default_factory=Hand)
    score: int = 0

    def to_dict(self, reveal_hand: bool = False) -> dict[str, object]:
        """Convert player state to dictionary."""
        result = {
            "name": self.name,
            "hand_size": self.hand.size(),
            "score": self.score,
        }
        if reveal_hand:
            result["hand"] = self.hand.to_dict()
        return result


class GameState:
    """Complete game state for solo mode."""

    def __init__(self, player_name: str) -> None:
        """Initialize game state with player name."""
        self.mainline = Mainline()
        self.sidelines: dict[int, Sideline] = {}
        self.deck = Deck()
        self._player = PlayerState(name=player_name)
        self.failed_rule_guesses: list[dict[str, str]] = []
        self.turn_number = 1
        self.game_over = False
        self.winner: str | None = None

    @property
    def player(self) -> PlayerState:
        """Solo player associated with this game state."""
        return self._player

    def add_sideline_card(self, card: Card) -> None:
        """Add a rejected card to the sideline for the current mainline position."""
        mainline_index = self.mainline.size() - 1
        if mainline_index not in self.sidelines:
            self.sidelines[mainline_index] = Sideline(mainline_index)
        self.sidelines[mainline_index].add_card(card)

    def record_failed_guess(self, guess_text: str) -> None:
        """Record a failed rule guess."""
        self.failed_rule_guesses.append(
            {"player": self._player.name, "guess": guess_text}
        )

    def to_compact_string(self) -> str:
        """Generate a compact mainline string with bracketed rejected cards."""
        if self.mainline.size() == 0:
            return "(empty)"

        result = []
        mainline_cards = self.mainline.get_all()

        for i, card in enumerate(mainline_cards):
            result.append(str(card))
            if i in self.sidelines:
                rejected = self.sidelines[i].get_cards()
                for r_card in rejected:
                    result.append(f"[{r_card!s}]")

        # Add any trailing rejected cards (after the last mainline card)
        if len(mainline_cards) in self.sidelines:
            rejected = self.sidelines[len(mainline_cards)].get_cards()
            for r_card in rejected:
                result.append(f"[{r_card!s}]")

        return " ".join(result)

    def to_json(self) -> str:
        """Serialize game state to JSON for LLM consumption."""
        state_dict = {
            "mainline": self.mainline.to_dict(),
            "sidelines": {
                str(idx): sideline.to_dict()["cards"]
                for idx, sideline in self.sidelines.items()
            },
            "player": self._player.to_dict(reveal_hand=True),
            "deck_remaining": self.deck.remaining_count(),
            "turn": self.turn_number,
        }
        return json.dumps(state_dict, indent=2)

    def to_dict(self) -> dict[str, object]:
        """Convert game state to dictionary."""
        return {
            "mainline": self.mainline.to_dict(),
            "sidelines": {
                str(idx): sideline.to_dict()["cards"]
                for idx, sideline in self.sidelines.items()
            },
            "player": self._player.to_dict(),
            "deck_remaining": self.deck.remaining_count(),
            "turn": self.turn_number,
            "game_over": self.game_over,
            "winner": self.winner,
        }
