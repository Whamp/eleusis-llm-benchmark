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
- Ranks: Ace = 1 (low), 2-10, Jack = 11, Queen = 12, King = 13. Number cards are 1-10, face cards are 11-13.
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
Score = (30 - current_turn) - (2 x number_of_failed_guesses)

Score can be negative. **Higher scores are better**. The goal is to discover the rule quickly with few failed guesses.
---


## YOUR TASK: CHOOSE YOUR NEXT ACTION

As a player, this is your turn to play and you must simultaneously:
1. Select a card from your hand to play.
2. Optionally, make a guess about the hidden rule.

### Current game state

Turn: 26 / 30
Penalty so far for 17 wrong guesses: 34
Current potential score if you guess the rule correctly this turn: -29

#### Mainline & Sidelines

This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after :

5♠ [3♥] 3♠ 10♣ [2♦] K♣ K♥ [Q♦] [J♠] 8♥ 10♦ 2♦ 10♠ 10♠ 9♣ J♣ 2♥ 5♥ 8♣ 5♣ A♦ 4♦ K♠ [J♦] K♠ J♦

#### Your hand

A♥, 6♠, 7♥, 4♥, 8♦, K♥, 4♥, 3♣, 9♥, 5♦, 6♦, 4♣

#### Your recent play history

Your last 3 turns (the card you played, the outcome and your reasoning summary):
- J♦: ✗ REJECTED
- K♠: ✓ ACCEPTED
  Your reasoning: The accepted sequence appears to be same-suit pairs, and since the last mainline card K♠ is the first card of a new spade pair, I will play the other K♠ to complete a same-suit pair, as equal-rank spades were accepted before.
- J♦: ✓ ACCEPTED
  Your reasoning: The accepted cards form same-suit pairs with Fibonacci rank differences, and each new pair must use a different suit from the previous pair; since the last K♠ completed a spade pair, J♦ should start a new diamond pair.


#### All your previous failed rule guesses (if any)

- "The card must be black."
- "The card must be the same color as the last mainline card."
- "The card must share either the same color or the same rank as the last mainline card."
- "A card is accepted if it is black or has the same rank as the last mainline card."
- "Accepted cards must follow the repeating suit pattern ♠♠♣♣♥♥♦♦ (two of each suit in the order spades, clubs, hearts, diamonds), so after ♠♠♣♣♥ the next accepted card must be a heart."
- "Cards are accepted in same-suit pairs; within each pair, the two ranks must differ by a prime number, and after a pair is complete any card can start the next pair."
- "Cards are accepted in same-suit pairs, with suits cycling in the order ♠♣♥♦, and the absolute rank difference within each pair must be the Fibonacci sequence 2, 3, 5, 8, ..."
- "Cards are accepted in same-suit pairs in suit order ♠♣♥♦; the first card of each new pair may be any card of the next suit, and the matching second card must be the same suit with rank difference 2 for spades, 3 for clubs, 5 for hearts, and 8 for diamonds."
- "A card is accepted if its suit is the next suit in the repeating mainline pattern ♠♠♣♣♥♥♦♦ and its rank is a Fibonacci number (1,2,3,5,8,13) or a multiple of 5 (5,10)."
- "Cards are accepted in same-suit pairs with suits cycling spades, clubs, hearts, diamonds; the first card of each pair is any card of the required suit, and the second must be the same suit with a cyclic rank-distance from the first equal to the repeating Fibonacci list 2, 3, 5, 8, 13 (so 10♠ then 10♠ is the 13-distance case)."
- "Cards are accepted in same-suit pairs with suits repeating ♠♠,♣♣,♥♥,♦♦; the first card of each pair may be any rank of the required suit, and the second card must be the same suit with a rank difference from the first that is a Fibonacci number modulo 13 (so equal ranks count as 13 apart)."
- "Cards are accepted in same-suit pairs with suits cycling ♠♠,♣♣,♥♥,♦♦; the first card of each pair can be any card of the required suit, and the second must be the same suit with an absolute rank difference from the first that is 0 or a Fibonacci number (1,2,3,5,8,13)."
- "Cards are accepted in same-suit pairs, with suits repeating as ♠♠,♣♣,♥♥,♦♦; the first card of each pair may be any card of that suit, and the second must be the same suit with an absolute rank difference from the first equal to the repeating sequence 2,3,5,8,13,2,3,... (with equal ranks counted as a difference of 13)."
- "Cards are accepted in pairs; the first card of each pair can be any card, and the second card must have the same suit as the first with a rank difference of 0, 1, 2, 3, 5, 8, or 13."
- "Cards are accepted in same-suit pairs: any card can start a pair, but the next accepted card must have the same suit as that pair; rank and suit order are irrelevant."
- "Cards are accepted in same-suit pairs; the first card of each pair can be any card, and the second card must be the same suit with a rank difference that is a Fibonacci number (1,2,3,5,8,13, treating equal ranks as 13 apart)."
- "Cards must be played in same-suit pairs: any card can start a pair, the second card must be the same suit with a rank difference of 0, 1, 2, 3, 5, 8, or 13 from the first, and each new pair must be a different suit from the previous pair."

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


Your last response hit the output token limit. Output ONLY the <ACTION> XML block with no reasoning.