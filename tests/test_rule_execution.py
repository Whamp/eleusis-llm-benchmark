"""Tests for generated rule code-object compilation and execution."""

from eleusis.game import Card, Rule, Suit


def test_rule_compiles_multistatement_body_without_exec() -> None:
    """Execute statements, comprehensions, and allowlisted built-ins in a rule body."""
    rule = Rule(
        "Accept a red card above both recent ranks.",
        """recent_ranks = [played.rank for played in mainline[-2:]]
threshold = max(recent_ranks) if recent_ranks else 0
return card.color == "red" and card.rank > threshold""",
    )
    mainline = [Card(4, Suit.CLUBS), Card(7, Suit.SPADES)]

    assert rule.evaluate(Card(8, Suit.HEARTS), mainline) is True
    assert rule.evaluate(Card(6, Suit.DIAMONDS), mainline) is False
    assert rule.evaluate(Card(10, Suit.CLUBS), mainline) is False
