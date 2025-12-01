"""Unit tests for cards module."""

import pytest

from eleusis.cards import Card, Deck, Hand, Suit


class TestCard:
    """Tests for Card class."""

    def test_card_creation(self) -> None:
        """Test creating a valid card."""
        card = Card(5, Suit.HEARTS)
        assert card.rank == 5
        assert card.suit == Suit.HEARTS

    def test_card_invalid_rank(self) -> None:
        """Test that invalid ranks raise error."""
        with pytest.raises(ValueError):
            Card(0, Suit.HEARTS)
        with pytest.raises(ValueError):
            Card(14, Suit.SPADES)

    def test_card_color(self) -> None:
        """Test card color property."""
        assert Card(5, Suit.HEARTS).color == "red"
        assert Card(5, Suit.DIAMONDS).color == "red"
        assert Card(5, Suit.CLUBS).color == "black"
        assert Card(5, Suit.SPADES).color == "black"

    def test_card_parity(self) -> None:
        """Test even/odd properties."""
        even_card = Card(2, Suit.HEARTS)
        odd_card = Card(3, Suit.HEARTS)
        assert even_card.is_even
        assert not even_card.is_odd
        assert odd_card.is_odd
        assert not odd_card.is_even

    def test_card_string_representation(self) -> None:
        """Test string representation."""
        assert str(Card(5, Suit.HEARTS)) == "5♥"
        assert str(Card(1, Suit.SPADES)) == "A♠"
        assert str(Card(13, Suit.CLUBS)) == "K♣"
        assert str(Card(12, Suit.DIAMONDS)) == "Q♦"
        assert str(Card(11, Suit.HEARTS)) == "J♥"

    def test_card_to_dict(self) -> None:
        """Test dictionary serialization."""
        card = Card(7, Suit.DIAMONDS)
        card_dict = card.to_dict()
        assert card_dict["rank"] == 7
        assert card_dict["suit"] == "diamonds"
        assert card_dict["color"] == "red"
        assert card_dict["symbol"] == "7♦"


class TestDeck:
    """Tests for Deck class."""

    def test_deck_initialization(self) -> None:
        """Test deck has 104 cards."""
        deck = Deck()
        assert deck.remaining_count() == 104

    def test_deck_draw(self) -> None:
        """Test drawing cards from deck."""
        deck = Deck()
        card1 = deck.draw()
        assert isinstance(card1, Card)
        assert deck.remaining_count() == 103

    def test_deck_empty(self) -> None:
        """Test deck emptying."""
        deck = Deck()
        for _ in range(104):
            deck.draw()
        assert deck.is_empty()
        with pytest.raises(ValueError):
            deck.draw()

    def test_deck_shuffle(self) -> None:
        """Test deck shuffle changes order."""
        deck1 = Deck()
        deck2 = Deck()
        deck2.shuffle()

        # Draw a few cards and verify they're likely different
        cards1 = [deck1.draw() for _ in range(10)]
        cards2 = [deck2.draw() for _ in range(10)]

        # Not a perfect test, but highly likely to be different after shuffle
        assert cards1 != cards2

    def test_deck_add_cards(self) -> None:
        """Test adding cards back to deck."""
        deck = Deck()
        card1 = deck.draw()
        card2 = deck.draw()
        assert deck.remaining_count() == 102

        deck.add_cards([card1, card2])
        assert deck.remaining_count() == 104


class TestHand:
    """Tests for Hand class."""

    def test_hand_initialization(self) -> None:
        """Test empty hand."""
        hand = Hand()
        assert hand.size() == 0

    def test_hand_add_card(self) -> None:
        """Test adding cards to hand."""
        hand = Hand()
        card = Card(5, Suit.HEARTS)
        hand.add_card(card)
        assert hand.size() == 1
        assert hand.contains(card)

    def test_hand_remove_card(self) -> None:
        """Test removing cards from hand."""
        hand = Hand()
        card = Card(5, Suit.HEARTS)
        hand.add_card(card)
        hand.remove_card(card)
        assert hand.size() == 0
        assert not hand.contains(card)

    def test_hand_remove_nonexistent_card(self) -> None:
        """Test removing card not in hand raises error."""
        hand = Hand()
        card = Card(5, Suit.HEARTS)
        with pytest.raises(ValueError):
            hand.remove_card(card)

    def test_hand_get_all_cards(self) -> None:
        """Test getting all cards."""
        hand = Hand()
        card1 = Card(5, Suit.HEARTS)
        card2 = Card(7, Suit.SPADES)
        hand.add_card(card1)
        hand.add_card(card2)

        cards = hand.get_all_cards()
        assert len(cards) == 2
        assert card1 in cards
        assert card2 in cards

    def test_hand_clear(self) -> None:
        """Test clearing hand."""
        hand = Hand()
        hand.add_card(Card(5, Suit.HEARTS))
        hand.add_card(Card(7, Suit.SPADES))
        hand.clear()
        assert hand.size() == 0
