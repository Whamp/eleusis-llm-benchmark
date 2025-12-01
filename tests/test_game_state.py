"""Unit tests for game_state module."""

import json

import pytest

from eleusis.cards import Card, Suit
from eleusis.game_state import GameState, Mainline, PlayerState, Sideline


class TestMainline:
    """Tests for Mainline class."""

    def test_mainline_initialization(self) -> None:
        """Test empty mainline."""
        mainline = Mainline()
        assert mainline.size() == 0
        assert mainline.get_last() is None

    def test_mainline_add_card(self) -> None:
        """Test adding cards to mainline."""
        mainline = Mainline()
        card1 = Card(5, Suit.HEARTS)
        card2 = Card(7, Suit.SPADES)

        mainline.add_card(card1)
        assert mainline.size() == 1
        assert mainline.get_last() == card1

        mainline.add_card(card2)
        assert mainline.size() == 2
        assert mainline.get_last() == card2

    def test_mainline_get_all(self) -> None:
        """Test getting all cards."""
        mainline = Mainline()
        card1 = Card(5, Suit.HEARTS)
        card2 = Card(7, Suit.SPADES)
        mainline.add_card(card1)
        mainline.add_card(card2)

        cards = mainline.get_all()
        assert len(cards) == 2
        assert cards[0] == card1
        assert cards[1] == card2


class TestSideline:
    """Tests for Sideline class."""

    def test_sideline_initialization(self) -> None:
        """Test sideline creation."""
        sideline = Sideline(mainline_index=2)
        assert sideline.mainline_index == 2
        assert len(sideline.get_cards()) == 0

    def test_sideline_add_card(self) -> None:
        """Test adding rejected cards."""
        sideline = Sideline(mainline_index=1)
        card = Card(3, Suit.DIAMONDS)
        sideline.add_card(card)
        assert len(sideline.get_cards()) == 1
        assert sideline.get_cards()[0] == card


class TestPlayerState:
    """Tests for PlayerState class."""

    def test_player_initialization(self) -> None:
        """Test player creation."""
        player = PlayerState(name="Alice")
        assert player.name == "Alice"
        assert player.score == 0
        assert not player.is_rule_maker
        assert player.hand.size() == 0

    def test_player_to_dict(self) -> None:
        """Test player serialization."""
        player = PlayerState(name="Alice", score=5, is_rule_maker=True)
        player.hand.add_card(Card(5, Suit.HEARTS))

        # Without revealing hand
        player_dict = player.to_dict(reveal_hand=False)
        assert player_dict["name"] == "Alice"
        assert player_dict["hand_size"] == 1
        assert player_dict["score"] == 5
        assert player_dict["is_rule_maker"]
        assert "hand" not in player_dict

        # With revealing hand
        player_dict = player.to_dict(reveal_hand=True)
        assert "hand" in player_dict
        assert len(player_dict["hand"]) == 1


class TestGameState:
    """Tests for GameState class."""

    def test_game_state_initialization(self) -> None:
        """Test game state creation."""
        players = ["Alice", "Bob", "Charlie", "Dave"]
        state = GameState(players, rule_maker_index=0)

        assert len(state.players) == 4
        assert state.players[0].name == "Alice"
        assert state.players[0].is_rule_maker
        assert state.current_turn_index == 1  # Skip rule-maker
        assert state.round_number == 1

    def test_game_state_invalid_player_count(self) -> None:
        """Test that wrong number of players raises error."""
        with pytest.raises(ValueError):
            GameState(["Alice", "Bob", "Charlie"])

    def test_get_current_player(self) -> None:
        """Test getting current player."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        assert state.get_current_player().name == "Bob"

    def test_get_rule_maker(self) -> None:
        """Test getting rule-maker."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=2)
        assert state.get_rule_maker().name == "Charlie"

    def test_get_scientists(self) -> None:
        """Test getting scientists."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        scientists = state.get_scientists()
        assert len(scientists) == 3
        assert state.players[0] not in scientists

    def test_advance_turn(self) -> None:
        """Test turn advancement."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        assert state.current_turn_index == 1  # Bob

        state.advance_turn()
        assert state.current_turn_index == 2  # Charlie

        state.advance_turn()
        assert state.current_turn_index == 3  # Dave

        state.advance_turn()
        assert state.current_turn_index == 1  # Back to Bob, skipping Alice

    def test_add_sideline_card(self) -> None:
        """Test adding cards to sideline."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"])
        state.mainline.add_card(Card(5, Suit.HEARTS))

        card = Card(3, Suit.DIAMONDS)
        state.add_sideline_card(card)

        assert 0 in state.sidelines
        assert card in state.sidelines[0].get_cards()

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        state.mainline.add_card(Card(5, Suit.HEARTS))

        json_str = state.to_json(current_player_name="Bob")
        data = json.loads(json_str)

        assert "mainline" in data
        assert "sidelines" in data
        assert "players" in data
        assert "deck_remaining" in data
        assert data["deck_remaining"] == 104
        assert data["current_turn"] == "Bob"

        # Check hand is revealed for Bob but not others
        assert "hand" in data["players"]["Bob"]
        assert "hand" not in data["players"]["Alice"]
