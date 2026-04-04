"""Tests for scripts/evaluate_shadows.py."""

import pytest

from eleusis.game.cards import Card, Suit

# Import the private helper from the script
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "evaluate_shadows",
    Path(__file__).resolve().parent.parent / "scripts" / "evaluate_shadows.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_parse_card = _mod._parse_card


class TestParseCardUnicodeSuits:
    """_parse_card should handle Unicode suit symbols from Card.__str__()."""

    def test_heart_unicode(self):
        assert _parse_card("5♥") == Card(5, Suit.HEARTS)

    def test_spade_unicode(self):
        assert _parse_card("K♠") == Card(13, Suit.SPADES)

    def test_diamond_unicode(self):
        assert _parse_card("10♦") == Card(10, Suit.DIAMONDS)

    def test_club_unicode(self):
        assert _parse_card("A♣") == Card(1, Suit.CLUBS)
