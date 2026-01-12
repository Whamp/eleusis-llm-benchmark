"""Game rules prompts for Eleusis."""

from pathlib import Path

import yaml

__all__ = ["get_game_rules", "get_eleusis_rules"]


def _load_game_config() -> dict:
    """Load game config from config.yaml."""
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("game")


def get_eleusis_rules() -> str:
    """Generate ELEUSIS game rules explanation for LLM context."""
    return """=== ELEUSIS GAME RULES ===

This is a card game where a secret rule determines which cards may be played.

## Overview

A **Rule-maker** invents a secret rule governing which cards may be accepted.
Players attempt to deduce the rule by playing cards and observing which are accepted or rejected.

## Components

- **Cards:** 2 standard 52-card decks shuffled together (104 cards total)
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of acceptance
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after

The entire layout (mainline and all sidelines) is visible to all players at all times.

---

## The Secret Rule

The secret rule is deterministic and decides whether a newly played card is **accepted** or **rejected**.

**The rule must:**
- Depend only on information visible in the mainline: the candidate card and/or any
  previously accepted mainline cards (their suits, colors, ranks, parity, positions, etc.)
- For the first card played, evaluate based solely on that card's properties (since
  no mainline exists yet)
- Give a unique, unambiguous answer (accepted or rejected) for every possible card in every
  possible mainline state

**The rule must NOT:**
- Reference sideline cards (rejected cards)
- Depend on hidden information (cards in the deck, cards in players' hands)
- Depend on player identity, turn order, or game history other than the mainline
- Include randomness or subjective judgment

These examples illustrate appropriate rule complexity:
- "The card must be a different color than the last mainline card." (Alternating red/black)
- "The card must be a heart or a spade."
- "The card must share either the suit or the color with the last mainline card, but not both."
- "The card's rank must have a different parity (odd/even) than the last mainline card."
- "The card's rank must differ from the last mainline card's rank by exactly 1 or 2
  (with Ace and King not considered adjacent)."
- "If the last mainline card is red, play a card with rank ≤ 7. If black, play a card
  with rank ≥ 7."

=== END OF THE ELEUSIS GAME RULES ===
"""


def get_game_rules() -> str:
    """Generate rules for pattern discovery game."""
    game_config = _load_game_config()

    hand_size = game_config.get('hand_size', 12)
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)
    max_turns = game_config.get('max_turns', 40)

    return f"""=== PATTERN DISCOVERY CARD GAME RULES ===

## Overview

This is a single-player pattern discovery card game. A hidden rule determines which cards
are accepted or rejected. Your goal is to discover the rule as efficiently as possible.

## Components

- **Cards:** 2 standard 52-card decks shuffled together (104 cards total)
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of acceptance
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after

The entire layout (mainline and all sidelines) is visible at all times.

---

## Setup

### Secret Rule

A deterministic rule decides whether a newly played card is **accepted** or **rejected**.

**The rule:**
- Depends only on information visible in the mainline: the candidate card and/or any
  previously accepted mainline cards (their suits, colors, ranks, parity, positions, etc.)
- For the first card played, evaluates based solely on that card's properties (since
  no mainline exists yet)
- Gives a unique, unambiguous answer (accepted/rejected) for every possible card in every
  possible mainline state

**The rule does NOT:**
- Reference sideline cards (rejected cards)
- Depend on hidden information (cards in the deck, cards in your hand)
- Include randomness or subjective judgment

Examples of rules:
- "The card must be a different color than the last mainline card." (Alternating red/black)
- "The card must be a heart or a spade."
- "The card must share either the suit or the color with the last mainline card, but not both."
- "The card's rank must have a different parity (odd/even) than the last mainline card."
- "The card's rank must differ from the last mainline card's rank by exactly 1 or 2."
- "If the last mainline card is red, play a card with rank ≤ 7. If black, play a card with rank ≥ 7."

### Initial Deal

You start with **{hand_size} cards** in your hand. This hand size remains constant throughout
the game - you always draw 1 card after playing.

A starter card that satisfies the rule is placed on the mainline to begin the game.

---

## Turn Structure

On each turn, you must:
1. **Play a card** from your hand
2. Receive feedback (accepted or rejected)
3. **Draw 1 card** from the deck (maintaining constant hand size)
4. Optionally **guess the rule** (see below)

### Playing a Card

1. Select one card from your hand to play
2. Place it to the right of the current last mainline card
3. Receive the judgment:

   **If Accepted:**
    - The card remains in place as the new last mainline card
    - You draw 1 card from the deck

   **If Rejected:**
    - The card is moved to a sideline column directly below the last mainline card
    - You draw 1 card from the deck

Your hand size stays constant at {hand_size} cards (unless the deck runs out).

---

## Guessing the Rule

At any point during your turn (after playing a card), you may attempt to state the rule.

The referee judges whether your stated rule is **equivalent** to the secret rule. Two
rules are equivalent if and only if they produce identical accepted/rejected judgments
for every possible (card, mainline-state) pair.

Note: The wording does not need to match exactly. Only logical equivalence matters.

- **Correct guess:** The game ends immediately - you win!
- **Incorrect guess:** The guess is recorded as wrong. A **penalty of {wrong_guess_penalty} points**
  is deducted from your final score. Play continues.

---

## End of Game

The game ends when:
1. You **correctly guess the rule**, or
2. You reach the **maximum of {max_turns} turns** without guessing correctly

---

## Scoring

Your score is calculated as follows:

    Score = {max_turns} - turn_number - ({wrong_guess_penalty} × number_of_failed_guesses)

Score can be negative.
In particular if you didn't guess correctly by turn {max_turns}, you get - ({wrong_guess_penalty} × number_of_failed_guesses)

**Higher scores are better.**
The goal is to discover the rule quickly with few failed guesses.
=== END OF GAME RULES ===
"""
