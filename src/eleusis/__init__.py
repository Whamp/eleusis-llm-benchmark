"""Eleusis LLM Benchmark - Pattern Discovery Card Game."""

__version__ = "0.1.0"

# Re-export main components for convenience
from eleusis.game import (
    Card,
    Deck,
    GameEngine,
    GameState,
    GuessRuleAction,
    Hand,
    PlayCardAction,
    Rule,
    RuleFactory,
    RuleValidator,
    Suit,
)
from eleusis.llm import create_client
from eleusis.player import LLMScientist
from eleusis.runner import play_round

__all__ = [
    # Version
    "__version__",
    # Game
    "Card",
    "Deck",
    "Hand",
    "Suit",
    "GameState",
    "GameEngine",
    "Rule",
    "PlayCardAction",
    "GuessRuleAction",
    "RuleFactory",
    "RuleValidator",
    # LLM
    "create_client",
    "LLMScientist",
    # Runner
    "play_round",
]
