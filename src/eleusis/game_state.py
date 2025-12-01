"""Game state management for Eleusis."""

import json
from dataclasses import dataclass, field

from eleusis.cards import Card, Deck, Hand


class Mainline:
    """Ordered sequence of accepted cards."""

    def __init__(self) -> None:
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

    def to_dict(self) -> dict:
        """Convert sideline to dictionary."""
        return {"mainline_index": self.mainline_index, "cards": [c.to_dict() for c in self._cards]}


@dataclass
class PlayerState:
    """State for a single player."""

    name: str
    hand: Hand = field(default_factory=Hand)
    score: int = 0
    is_rule_maker: bool = False

    def to_dict(self, reveal_hand: bool = False) -> dict:
        """Convert player state to dictionary."""
        result = {
            "name": self.name,
            "hand_size": self.hand.size(),
            "score": self.score,
            "is_rule_maker": self.is_rule_maker,
        }
        if reveal_hand:
            result["hand"] = self.hand.to_dict()
        return result


class GameState:
    """Complete game state visible to all players."""

    def __init__(self, player_names: list[str], rule_maker_index: int = 0) -> None:
        """Initialize game state with player names."""
        if len(player_names) != 4:
            raise ValueError("Game requires exactly 4 players")

        self.mainline = Mainline()
        self.sidelines: dict[int, Sideline] = {}
        self.deck = Deck()
        self.players: list[PlayerState] = []

        for i, name in enumerate(player_names):
            is_rule_maker = i == rule_maker_index
            self.players.append(PlayerState(name=name, is_rule_maker=is_rule_maker))

        self.current_turn_index = (rule_maker_index + 1) % 4
        self.round_number = 1
        self.game_over = False
        self.winner: str | None = None

    def get_current_player(self) -> PlayerState:
        """Get the player whose turn it is."""
        return self.players[self.current_turn_index]

    def get_rule_maker(self) -> PlayerState:
        """Get the rule-maker player."""
        for player in self.players:
            if player.is_rule_maker:
                return player
        raise ValueError("No rule-maker found")

    def get_scientists(self) -> list[PlayerState]:
        """Get all scientist players (non-rule-makers)."""
        return [p for p in self.players if not p.is_rule_maker]

    def advance_turn(self) -> None:
        """Move to next player's turn (skip rule-maker)."""
        rule_maker_index = next(i for i, p in enumerate(self.players) if p.is_rule_maker)

        self.current_turn_index = (self.current_turn_index + 1) % 4
        if self.current_turn_index == rule_maker_index:
            self.current_turn_index = (self.current_turn_index + 1) % 4

    def add_sideline_card(self, card: Card) -> None:
        """Add a rejected card to the sideline for the current mainline position."""
        mainline_index = self.mainline.size() - 1
        if mainline_index not in self.sidelines:
            self.sidelines[mainline_index] = Sideline(mainline_index)
        self.sidelines[mainline_index].add_card(card)

    def to_compact_string(self) -> str:
        """Generate compact string representation of mainline with rejected cards in brackets."""
        if self.mainline.size() == 0:
            return "(empty)"

        result = []
        mainline_cards = self.mainline.get_all()

        for i, card in enumerate(mainline_cards):
            # Add the accepted card
            result.append(str(card))

            # Add rejected cards (sideline) AFTER this position
            if i in self.sidelines:
                rejected = self.sidelines[i].get_cards()
                for r_card in rejected:
                    result.append(f"[{str(r_card)}]")

        # Add any trailing rejected cards (after the last mainline card)
        if len(mainline_cards) in self.sidelines:
            rejected = self.sidelines[len(mainline_cards)].get_cards()
            for r_card in rejected:
                result.append(f"[{str(r_card)}]")

        return " ".join(result)

    def to_json(self, current_player_name: str | None = None) -> str:
        """Serialize game state to JSON for LLM consumption."""
        state_dict = {
            "mainline": self.mainline.to_dict(),
            "sidelines": {
                str(idx): sideline.to_dict()["cards"]
                for idx, sideline in self.sidelines.items()
            },
            "players": {},
            "deck_remaining": self.deck.remaining_count(),
            "current_turn": self.get_current_player().name,
            "round": self.round_number,
        }

        # Add player info (reveal hand only for current player)
        for player in self.players:
            reveal = current_player_name is not None and player.name == current_player_name
            state_dict["players"][player.name] = player.to_dict(reveal_hand=reveal)

        return json.dumps(state_dict, indent=2)

    def to_dict(self) -> dict:
        """Convert game state to dictionary."""
        return {
            "mainline": self.mainline.to_dict(),
            "sidelines": {
                str(idx): sideline.to_dict()["cards"]
                for idx, sideline in self.sidelines.items()
            },
            "players": {p.name: p.to_dict() for p in self.players},
            "deck_remaining": self.deck.remaining_count(),
            "current_turn": self.get_current_player().name,
            "round": self.round_number,
            "game_over": self.game_over,
            "winner": self.winner,
        }
