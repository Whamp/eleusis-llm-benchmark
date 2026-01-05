"""Prompt templates for solo pattern discovery card game."""

from pathlib import Path
import yaml

# Import common utilities from original prompts
from eleusis.prompts import get_continuation_prompt, get_rule_compilation_prompt


def _load_game_config() -> dict:
    """Load game config from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("game")


def get_solo_rules() -> str:
    """Generate rules for solo pattern discovery game."""
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


def get_solo_action_selection_prompt(
    compact_board: str,
    hand_cards: list[dict],
    deck_remaining: int,
    play_history: list[dict],
    failed_guesses: list[dict] | None = None,
    current_turn: int = 1,
    max_turns: int = 40,
    failed_guess_count: int = 0,
) -> str:
    """Generate prompt for LLM to select a move in solo mode."""
    hand_str = ", ".join([c["symbol"] for c in hand_cards])

    # Calculate current score
    game_config = _load_game_config()
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)
    current_score = max_turns - current_turn - (wrong_guess_penalty * failed_guess_count)

    # Format play history
    history_str = ""
    if play_history:
        history_str = "\n\nYOUR PREVIOUS ATTEMPTS:\n"
        for entry in play_history[-15:]:  # Show last 15 attempts
            card = entry.get("card", "N/A")
            reasoning_summary = entry.get("reasoning_summary", "")
            accepted = entry.get("accepted")

            if accepted is not None:
                result = "✓ ACCEPTED" if accepted else "✗ REJECTED"
                history_str += f"- {card}: {result}\n"
                if reasoning_summary:
                    history_str += f"  Your reasoning: {reasoning_summary}\n"

    # Format failed guesses history
    failed_guesses_str = ""
    if failed_guesses:
        failed_guesses_str = "\n\nYOUR FAILED RULE GUESSES:\n"
        for entry in failed_guesses:
            guess = entry.get("guess", "")
            failed_guesses_str += f"- \"{guess}\"\n"
        failed_guesses_str += "\nALL OF THESE GUESSES WERE INCORRECT. DO NOT REPEAT THEM.\n"

    return f"""
**YOU ARE PLAYING A PATTERN DISCOVERY CARD GAME**

{get_solo_rules()}

=== YOUR TASK: CHOOSE YOUR NEXT ACTION ===

You are trying to discover the hidden rule that determines which cards are accepted.

CURRENT TURN: {current_turn} / {max_turns}
CURRENT PENALTY ({failed_guess_count} wrong guesses): {wrong_guess_penalty * failed_guess_count}
CURRENT SCORE: {max_turns} - {current_turn} - ({wrong_guess_penalty} × {failed_guess_count}) = {current_score}

CURRENT BOARD (mainline + rejected cards in brackets):
{compact_board}

YOUR HAND: {hand_str}

DECK REMAINING: {deck_remaining} cards

YOUR PLAY HISTORY:
{history_str}

YOUR PREVIOUS FAILED RULE GUESSES, IF ANY:
{failed_guesses_str}

---

YOUR ACTION:

You must select a card from your hand to play. After playing, you can optionally guess the rule.

OUTPUT FORMAT:
You can freely reason step by step about the pattern you observe.
Then your response should end with your final decision wrapped in XML tags.

<your reasoning about the pattern>

<ACTION>
{{
    "reasoning_summary": "Brief summary of your analysis and why you're playing this card",
    "card": "5♥" (the card symbol from your hand),
    "tentative_rule": "Your current best guess about the rule (always provide this, must be unequivocal)",
    "confidence_level": 0-10 (your confidence in the tentative_rule, 0=no clue, 10=certain),
    "guess_rule": true or false (whether to officially guess the rule this turn)
}}
</ACTION>

IMPORTANT GUIDANCE:

1. **Card Selection**: You MUST select a card from your hand. The value should be the exact
   symbol (e.g., "5♥", "K♠", "A♦").

2. **Tentative Rule**: ALWAYS provide your current hypothesis about the rule, even if you're
   uncertain. This must be clear and unequivocal (no "maybe" or "possibly"). If you set guess_rule to false,
   this guess will NOT be evaluated - it's just for your own tracking.

3. **Confidence Level**: Rate 0-10 how confident you are (0=completely uncertain, 10=absolutely sure).

4. **Guessing**: Set "guess_rule" to true ONLY when you're confident enough to officially guess.
   - If CORRECT: You win immediately!
   - If INCORRECT: You lose {_load_game_config().get('wrong_guess_penalty', 3)} points from your final score
   - Consider the trade-off: guessing early but wrong is costly, guessing late reduces your score

Example:
<ACTION>
{{
    "reasoning_summary": "I see red and black cards alternating. My 3♥ is red, last card was black.",
    "card": "3♥",
    "tentative_rule": "Cards must alternate between red and black colors",
    "confidence_level": 8,
    "guess_rule": false
}}
</ACTION>
"""
