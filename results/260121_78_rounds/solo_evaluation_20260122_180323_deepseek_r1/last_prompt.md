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
Penalty so far for 16 wrong guesses: 32
Current potential score if you guess the rule correctly this turn: -32

#### Mainline & Sidelines

This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after :

K♠ [8♦] K♣ 6♠ [4♥] [7♠] [4♠] [J♠] [K♠] [A♣] [J♥] [7♠] 6♦ 5♥ [4♦] 5♦ 6♦ [9♥] [10♦] [3♦] [A♠] [A♥] [9♠] [Q♦] [3♦] [3♥] [8♦] [7♣] [7♥] [3♣]

#### Your hand

A♣, 10♣, 4♣, 9♥, 2♠, 3♠, Q♠, 3♥, 9♦, J♦, 7♥, 8♠

#### Your recent play history

Your last 3 turns (the card you played, the outcome and your reasoning summary):
- 7♣: ✗ REJECTED
  Your reasoning: No 5,6, or K in hand; playing 7♣ to confirm rejection while guessing the rule based on accepted cards requiring 5/6/K with color and suit conditions.
- 7♥: ✗ REJECTED
  Your reasoning: Playing 7♥ to test rejection of non-5/6/K cards, confirming the rule requires these ranks, while exploring color consistency.
- 3♣: ✗ REJECTED
  Your reasoning: Playing 3♣ to test rejection of a non-5/6/K card while guessing the rule, which requires 5/6/K with Kings needing the last card to be King and 5/6 needing same rank or color as last.


#### All your previous failed rule guesses (if any)

- "The card must be the same color as the last mainline card"
- "The card must be a King or a Six."
- "The card must be a black King or a black Six."
- "The card must be the same color as the last mainline card and have a different suit."
- "The card must be a King of a suit different from the last mainline card's suit, or a Six of any suit."
- "If the last mainline card is a six, the card must be a five or a six; otherwise, it must be the same color as the last mainline card."
- "The card must be a King, Six, or Five that has not appeared in the mainline."
- "The card must be a King, Six, or Five."
- "The card must be a Five, Six, or King. Additionally, if it is a King, its suit must not match any suit from a King previously in the mainline."
- "The card must be a Five, Six, or King. If the card is a King, the last mainline card must also be a King."
- "The card must be a Five, Six, or King; additionally, if the last mainline card is a King, the played card must be a King or Six; if the last is a Six, the played card must be a Five or Six; if the last is a Five, the played card must be a Five or Six."
- "The card is accepted if its rank is already in the mainline (with Kings requiring a suit different from all previous Kings) or if its rank is not in the mainline and it is a Five or Six."
- "The card must be a Five, Six, or King; additionally, if the last mainline card is a Six, the card must be a Five or Six (Kings are not allowed in this case)."
- "The card must be a Five, Six, or King. If the card is a King, the last mainline card must also be King; for Fives and Sixes, there are no additional conditions."
- "The card must be a Five, Six, or King; additionally, if it is a King or Five, it must be the same color as the last mainline card, and for Kings, the suit must be unique among all Kings in the mainline."
- "The card must be a Five, Six, or King. If the card is a King, the last mainline card must also be a King. If the card is a Five or Six, it must have either the same rank or the same color as the last mainline card."

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
