"""Game rules prompts for Eleusis."""

__all__ = ["get_game_rules"]


def get_game_rules(
    hand_size: int,
    wrong_guess_penalty: int,
    max_turns: int,
) -> str:
    """Generate rules for pattern discovery game."""
    return f"""## RULES OF THE GAME
### Overview
This is a single-player game. 
A hidden rule created by the game master determines which cards are accepted or rejected. 
Your goal is to play cards and discover the rule as efficiently as possible.

The game uses 2 standard 52-card decks shuffled together (104 cards total):
- Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13. Number cards are 1-10, face cards are 11-13.
- Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

The playing area consists of:
- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of acceptance
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after

### Setup

#### Secret Rule

A deterministic secret rule is chosen by the game master. 
The rule decides unambiguously whether a newly played card is **accepted** or **rejected**, for every possible card in every possible mainline state.
The rule is simple enough to be described in a single sentence.
The rule depends only on information visible in the mainline: the candidate card and/or any previously accepted mainline cards and their properties (their suits, colors, ranks, parity, positions, etc.)
The rule is deterministic, objective, does not reference sideline rejected cards or hidden information (cards in the deck, cards in your hand...)

Examples of rules:
- The card must be a different color than the last mainline card.
- The card must be a heart or a spade.
- The card must share either the suit or the color with the last mainline card, but not both.
- The card's rank must differ from the last mainline card's rank by exactly 1 or 2.

#### Initial Deal

You start with **{hand_size} cards** in your hand. 
This hand size remains constant throughout the game, you always draw 1 card after playing.
A starter card that satisfies the rule is placed on the mainline to begin the game.

### Turn Structure

On each turn, you must:
- Play a card from your hand, you then receive feedback (accepted or rejected)
- Optionally try to guess the rule, in order to end the round and score (see below)
- You draw a new card from the deck to maintain your hand size

#### Playing a Card

Once you played a card from your hand to play, you will receive the feedback from the game master, based on the secret rule.
- **If Rejected:** the card goes to a sideline column directly below the last mainline card;
- **If Accepted:** the card becomes the new last mainline card;

### Guessing the Rule

When playing a card, you may attempt to state the rule you believe governs acceptance/rejection.
The game master judges whether your stated rule is **equivalent** to the secret rule. 
The wording does not need to match exactly. Only logical equivalence matters.
- **Incorrect guess:** The guess is recorded as wrong and **penalty of {wrong_guess_penalty} points** will later be applied to your final score. Play continues.
- **Correct guess:** The game ends immediately, you score based on the number of turns taken and wrong guesses.

### End Game & Scoring

The game ends when:
1. You **correctly guess the rule**, or
2. You reach the maximum of {max_turns} turns without guessing correctly

Your score is calculated as follows: 
Score = ({max_turns} - current_turn) - ({wrong_guess_penalty} × number_of_failed_guesses)

Score can be negative. **Higher scores are better**. The goal is to discover the rule quickly with few failed guesses.
---
"""
