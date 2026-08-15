"""Tests for scripts/evaluate_shadows.py."""

# Import the private helper from the script
import importlib.util
from pathlib import Path

from eleusis.game.cards import Card, Suit

_spec = importlib.util.spec_from_file_location(
    "evaluate_shadows",
    Path(__file__).resolve().parent.parent / "scripts" / "evaluate_shadows.py",
)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Could not load scripts/evaluate_shadows.py for tests")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_parse_card = _mod._parse_card


class TestParseCardUnicodeSuits:
    """_parse_card should handle Unicode suit symbols from Card.__str__()."""

    def test_heart_unicode(self) -> None:
        """Verify heart unicode."""
        assert _parse_card("5♥") == Card(5, Suit.HEARTS)

    def test_spade_unicode(self) -> None:
        """Verify spade unicode."""
        assert _parse_card("K♠") == Card(13, Suit.SPADES)

    def test_diamond_unicode(self) -> None:
        """Verify diamond unicode."""
        assert _parse_card("10♦") == Card(10, Suit.DIAMONDS)

    def test_club_unicode(self) -> None:
        """Verify club unicode."""
        assert _parse_card("A♣") == Card(1, Suit.CLUBS)
