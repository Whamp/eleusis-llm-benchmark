"""Unit tests for game_engine module."""

import pytest

from eleusis.cards import Card, Suit
from eleusis.game_engine import GameEngine, NoPlayAction, PlayCardAction, Rule
from eleusis.game_state import GameState


def AlwaysAcceptRule() -> Rule:
    """Test rule that accepts all cards."""
    return Rule("Accept all cards", "return True")


def EvenRankRule() -> Rule:
    """Test rule that accepts only even ranks."""
    return Rule("Only even ranks are accepted", "return card.rank % 2 == 0")


class TestGameEngine:
    """Tests for GameEngine class."""

    def test_game_setup(self) -> None:
        """Test game setup with dealing and starter card."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, game_master=None)

        engine.setup_game()

        # Check all players have 12 cards each
        for player in state.players:
            assert player.hand.size() == 12

        # Check starter card is placed
        assert state.mainline.size() == 1

    def test_play_card_accepted(self) -> None:
        """Test playing a card that is accepted."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, game_master=None)
        engine.setup_game()

        current_player = state.get_current_player()
        initial_hand_size = current_player.hand.size()
        card = current_player.hand.get_all_cards()[0]

        result = engine.play_turn(PlayCardAction(card))

        assert result["success"]
        assert result["accepted"]
        assert current_player.hand.size() == initial_hand_size - 1
        assert state.mainline.size() == 2  # Starter + played card

    def test_play_card_rejected(self) -> None:
        """Test playing a card that is rejected."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = EvenRankRule()
        engine = GameEngine(state, rule, game_master=None)

        # Manually set up for testing
        state.mainline.add_card(Card(2, Suit.HEARTS))  # Starter
        current_player = state.get_current_player()
        odd_card = Card(3, Suit.DIAMONDS)
        current_player.hand.add_card(odd_card)
        initial_hand_size = current_player.hand.size()

        result = engine.play_turn(PlayCardAction(odd_card))

        assert result["success"]
        assert not result["accepted"]
        # Hand size increases by 1 (removed 1 card, drew 2 penalty cards)
        assert current_player.hand.size() == initial_hand_size + 1
        assert state.mainline.size() == 1  # Only starter
        assert 0 in state.sidelines  # Card in sideline

    def test_no_play_correct(self) -> None:
        """Test correct no-play declaration."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = EvenRankRule()
        engine = GameEngine(state, rule, game_master=None)

        state.mainline.add_card(Card(2, Suit.HEARTS))
        current_player = state.get_current_player()

        # Give player only odd cards
        current_player.hand.add_card(Card(3, Suit.DIAMONDS))
        current_player.hand.add_card(Card(5, Suit.CLUBS))
        initial_hand_size = current_player.hand.size()

        result = engine.play_turn(NoPlayAction())

        assert result["success"]
        assert result["correct"]
        # All cards discarded, draw max(0, initial_hand_size - 4) new cards
        expected_hand_size = max(0, initial_hand_size - 4)
        assert current_player.hand.size() == expected_hand_size

    def test_no_play_incorrect(self) -> None:
        """Test incorrect no-play declaration."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = EvenRankRule()
        engine = GameEngine(state, rule, game_master=None)

        state.mainline.add_card(Card(2, Suit.HEARTS))
        current_player = state.get_current_player()

        # Give player an even card (legal play exists)
        even_card = Card(4, Suit.SPADES)
        current_player.hand.add_card(even_card)
        current_player.hand.add_card(Card(3, Suit.DIAMONDS))
        initial_hand_size = current_player.hand.size()

        result = engine.play_turn(NoPlayAction())

        assert result["success"]
        assert not result["correct"]
        # Hand size increases by 3 (removed 1 card, drew 4 penalty cards)
        assert current_player.hand.size() == initial_hand_size + 3
        assert state.mainline.size() == 2  # Starter + forced card

    def test_calculate_scores(self) -> None:
        """Test score calculation."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, game_master=None)

        # Set up hand sizes
        state.players[0].hand.add_card(Card(2, Suit.HEARTS))
        state.players[0].hand.add_card(Card(3, Suit.HEARTS))  # 2 cards
        state.players[1].hand.add_card(Card(4, Suit.HEARTS))  # 1 card
        # state.players[2] has 0 cards

        scores = engine.calculate_scores()

        # Players get their hand size
        assert scores[state.players[0].name] == 2
        assert scores[state.players[1].name] == 1
        assert scores[state.players[2].name] == 0

    def test_is_game_over_deck_empty(self) -> None:
        """Test game over when deck is empty."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = EvenRankRule()
        engine = GameEngine(state, rule, game_master=None)

        # Empty the deck
        while not state.deck.is_empty():
            state.deck.draw()

        # Give all players only odd cards (no legal plays)
        for player in state.players:
            player.hand.add_card(Card(3, Suit.HEARTS))

        assert engine.is_game_over()

    def test_turn_advancement(self) -> None:
        """Test that turns advance correctly."""
        state = GameState(["Alice", "Bob", "Charlie"])
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule, game_master=None)
        engine.setup_game()

        initial_player = state.get_current_player()
        card = initial_player.hand.get_all_cards()[0]
        engine.play_turn(PlayCardAction(card))

        # Turn should have advanced
        assert state.get_current_player() != initial_player
