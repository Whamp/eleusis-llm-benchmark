"""Unit tests for game_engine module."""

import pytest

from eleusis.cards import Card, Suit
from eleusis.game_engine import GameEngine, NoPlayAction, PlayCardAction, Rule
from eleusis.game_state import GameState


class AlwaysAcceptRule(Rule):
    """Test rule that accepts all cards."""

    def evaluate(self, card: Card, mainline: list[Card]) -> bool:
        return True

    def description(self) -> str:
        return "Accept all cards"


class EvenRankRule(Rule):
    """Test rule that accepts only even ranks."""

    def evaluate(self, card: Card, mainline: list[Card]) -> bool:
        return card.is_even

    def description(self) -> str:
        return "Only even ranks are accepted"


class TestGameEngine:
    """Tests for GameEngine class."""

    def test_game_setup(self) -> None:
        """Test game setup with dealing and starter card."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule)

        engine.setup_game()

        # Check scientists have 12 cards each
        scientists = state.get_scientists()
        for scientist in scientists:
            assert scientist.hand.size() == 12

        # Check rule-maker has no cards
        assert state.get_rule_maker().hand.size() == 0

        # Check starter card is placed
        assert state.mainline.size() == 1

    def test_play_card_accepted(self) -> None:
        """Test playing a card that is accepted."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule)
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
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = EvenRankRule()
        engine = GameEngine(state, rule)

        # Manually set up for testing
        state.mainline.add_card(Card(2, Suit.HEARTS))  # Starter
        current_player = state.get_current_player()
        odd_card = Card(3, Suit.DIAMONDS)
        current_player.hand.add_card(odd_card)
        initial_hand_size = current_player.hand.size()

        result = engine.play_turn(PlayCardAction(odd_card))

        assert result["success"]
        assert not result["accepted"]
        # Hand size same (removed card, drew card)
        assert current_player.hand.size() == initial_hand_size
        assert state.mainline.size() == 1  # Only starter
        assert 0 in state.sidelines  # Card in sideline

    def test_no_play_correct(self) -> None:
        """Test correct no-play declaration."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = EvenRankRule()
        engine = GameEngine(state, rule)

        state.mainline.add_card(Card(2, Suit.HEARTS))
        current_player = state.get_current_player()

        # Give player only odd cards
        current_player.hand.add_card(Card(3, Suit.DIAMONDS))
        current_player.hand.add_card(Card(5, Suit.CLUBS))
        initial_hand_size = current_player.hand.size()

        result = engine.play_turn(NoPlayAction())

        assert result["success"]
        assert result["correct"]
        assert current_player.hand.size() == initial_hand_size - 1  # One card discarded

    def test_no_play_incorrect(self) -> None:
        """Test incorrect no-play declaration."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = EvenRankRule()
        engine = GameEngine(state, rule)

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
        # Hand size same (removed card, drew penalty)
        assert current_player.hand.size() == initial_hand_size
        assert state.mainline.size() == 2  # Starter + forced card

    def test_calculate_scores(self) -> None:
        """Test score calculation."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule)

        # Set up hand sizes
        scientists = state.get_scientists()
        scientists[0].hand.add_card(Card(2, Suit.HEARTS))
        scientists[0].hand.add_card(Card(3, Suit.HEARTS))  # 2 cards
        scientists[1].hand.add_card(Card(4, Suit.HEARTS))  # 1 card
        # scientists[2] has 0 cards

        scores = engine.calculate_scores()

        # Scientists get their hand size
        assert scores[scientists[0].name] == 2
        assert scores[scientists[1].name] == 1
        assert scores[scientists[2].name] == 0

        # Rule-maker gets second-lowest score (sorted: [0, 1, 2] -> second is 1)
        rule_maker = state.get_rule_maker()
        assert scores[rule_maker.name] == 1

    def test_is_game_over_deck_empty(self) -> None:
        """Test game over when deck is empty."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = EvenRankRule()
        engine = GameEngine(state, rule)

        # Empty the deck
        while not state.deck.is_empty():
            state.deck.draw()

        # Give scientists only odd cards (no legal plays)
        for scientist in state.get_scientists():
            scientist.hand.add_card(Card(3, Suit.HEARTS))

        assert engine.is_game_over()

    def test_turn_advancement(self) -> None:
        """Test that turns advance correctly."""
        state = GameState(["Alice", "Bob", "Charlie", "Dave"], rule_maker_index=0)
        rule = AlwaysAcceptRule()
        engine = GameEngine(state, rule)
        engine.setup_game()

        initial_player = state.get_current_player()
        card = initial_player.hand.get_all_cards()[0]
        engine.play_turn(PlayCardAction(card))

        # Turn should have advanced
        assert state.get_current_player() != initial_player
