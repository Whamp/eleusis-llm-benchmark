# PATTERN DISCOVERY CARD GAME

You are playing a single-player card game where the goal is to discover a secret rule that determines which cards are accepted or rejected.
This is your turn to play. Your task is to select a card from your hand to play, and optionally try to guess the secret rule.
Your score will depend on how many turns it takes you to correctly identify the rule.
Below you will find the rules of the game, the current game state, your play history, and a description of what you are expected to do.

## RULES OF THE GAME
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

You start with **12 cards** in your hand. 
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
- **Incorrect guess:** The guess is recorded as wrong and **penalty of 2 points** will later be applied to your final score. Play continues.
- **Correct guess:** The game ends immediately, you score based on the number of turns taken and wrong guesses.

### End Game & Scoring

The game ends when:
1. You **correctly guess the rule**, or
2. You reach the maximum of 30 turns without guessing correctly

Your score is calculated as follows: 
Score = (30 - current_turn) - (2 × number_of_failed_guesses)

Score can be negative. **Higher scores are better**. The goal is to discover the rule quickly with few failed guesses.
---


## YOUR TASK: CHOOSE YOUR NEXT ACTION

As a player, this is your turn to play and you must simultaneously:
1. Select a card from your hand to play.
2. Optionally, make a guess about the hidden rule.

### Current game state

Turn: 30 / 30
Penalty so far for 9 wrong guesses: 18
Current potential score if you guess the rule correctly this turn: -18

#### Mainline & Sidelines

This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after :

K♠ [8♦] K♣ J♥ [7♠] J♠ 10♦ [K♠] [7♠] 10♣ 7♣ [8♦] [4♠] [A♣] [6♠] [6♦] [9♠] [9♥] [3♣] [4♥] [A♠] [Q♦] [5♦] [2♠] [6♦] [A♥] [3♥] 7♥ 5♥ [4♦]

#### Your hand

3♦, A♣, 4♣, 9♥, 3♠, 3♦, Q♠, 3♥, 9♦, J♦, 7♥, 8♠

#### Your recent play history

Your last 3 turns (the card you played, the outcome and your reasoning summary):
- 7♥: ✓ ACCEPTED
  Your reasoning: Playing 7♥ to test if a 7 is needed (same rank as last card 7♣) at this odd-length mainline position - if accepted, confirms the rank-matching pattern; if rejected, reveals a suit/color constraint I haven't identified.
- 5♥: ✓ ACCEPTED
  Your reasoning: Mainline shows clear alternating pattern of match/lower based on card count; with 8 cards (even), next should be strictly lower than 7, so testing 5♥ to confirm pattern before final guess.
- 4♦: ✗ REJECTED
  Your reasoning: Pattern clearly alternates between matching rank (odd mainline count) and strictly lower rank (even count), but all my phrasings were rejected; trying a cleaner formulation and playing 4♦ which should be rejected since we need a 5.


#### All your previous failed rule guesses (if any)

- "The card must be a face card (Jack, Queen, or King)"
- "The card's rank must differ by at most 2 from the rank of the last mainline card"
- "The card's rank must be the same as or at most 2 less than the last mainline card's rank (no increases allowed)"
- "The card must have the same rank as the last mainline card when the mainline has an odd number of cards, or a strictly lower rank when the mainline has an even number of cards"
- "Cards must alternate between matching the previous card's rank and being strictly lower than the previous rank, starting with a match after the starter card"
- "If the two most recent mainline cards have the same rank, the next card must be strictly lower than that rank. Otherwise, the next card must have the same rank as the most recent mainline card."
- "Match the last card's rank to complete a pair, unless the last two cards already have the same rank, in which case play a strictly lower card to start a new pair"
- "The card must alternate between matching the previous card's rank with a different suit (even positions) and being strictly lower in rank (odd positions after the starter)"
- "Acceptance alternates: when the mainline has an odd number of cards, play must match that rank; when even, play must be strictly lower"

All these guesses were incorrect.


### Task description and formatting instructions

You must select a card from your hand to play and optionally decide to try to guess the rule.

You will be asked to provide the following in your response:
- A short one sentence summary of your reasoning, explaining your thought process about the pattern so far and why you are playing the selected card.
- The card you are playing from your hand (must be one of the cards listed in your hand).
- Your current best guess about the hidden rule (must be clear and unequivocal).
- Your confidence level in your tentative guess rule on a scale of 0-10 (for instance 7 means "70% confident : there 70% probability to be correct")
- Whether you want to officially guess the rule this turn (true or false). Only set to true if you want to make an official guess with your tentative rule. Otherwise, your tentative rule is just for your own tracking and will not be evaluated this turn.

Format your response as follows:
<ACTION>
{
    "reasoning_summary": "One sentence summary of your analysis and why you're playing this card",
    "card": "5♥" (the card symbol from your hand, must be the exact symbol e.g., "5♥", "K♠", "A♦",...)
    "tentative_rule": "Your current best guess about the rule",
    "confidence_level": 0-10,
    "guess_rule": true or false (whether to officially try to guess the rule this turn)
}
</ACTION>

Always provide your current best hypothesis as a tentative rule, even if you're uncertain. 
If you set guess_rule to false, this tentative rule will not be evaluated, it's just for your own tracking.
Set "guess_rule" to true only when you want to officially try to guess the rule.
   - If correct, you score and the round ends immediately.
   - If incorrect, you will lose 2 points from your final score, and the round continues.

#### Example:
<ACTION>
{
    "reasoning_summary": "I see red and black cards alternating. My 3♥ is red, last card was black.",
    "card": "3♥",
    "tentative_rule": "Cards must alternate between red and black colors",
    "confidence_level": 8,
    "guess_rule": false
}
</ACTION>
