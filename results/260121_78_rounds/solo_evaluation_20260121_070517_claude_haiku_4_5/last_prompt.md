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

K♠ [4♥] K♣ 7♠ [A♣] [10♦] [J♠] 7♣ K♠ [10♣] [7♠] [A♣] [A♥] [9♠] [6♠] [4♣] [5♦] [9♥] [3♣] [2♠] [3♠] [4♠] [5♥] [Q♦] [A♠] [3♥] [9♦] [3♦] [J♦] [3♦]

#### Your hand

8♦, J♥, 4♦, 8♦, 6♦, 6♦, 9♥, Q♠, 3♥, 7♥, 7♥, 8♠

#### Your recent play history

Your last 3 turns (the card you played, the outcome and your reasoning summary):
- 3♦: ✗ REJECTED
  Your reasoning: After 27 turns with 15 failed rule guesses despite observing a crystal-clear predetermined 4-card repeating sequence (K♠, K♣, 7♠, 7♣, K♠...), I'll attempt one final rephrasing of this pattern with clearer, more direct language, since all previous attempts used equivalent but differently worded descriptions that were rejected.
- J♦: ✗ REJECTED
  Your reasoning: The mainline shows a crystal-clear pattern: K♠, K♣, 7♠, 7♣ repeating indefinitely. Position 26 should require K♣, which I don't have in my hand, so any card I play will be rejected. With only 2 turns remaining and already at -30 points, making another failed guess would only worsen my score (costing -2 more points). I'll play a rejected card and preserve my score by not guessing.
- 3♦: ✗ REJECTED
  Your reasoning: At turn 29/30 with 16 failed guesses already costing -32 points, the mainline shows an undeniable pattern (K♠, K♣, 7♠, 7♣ initially, then K♠/7♣ alternating), but all 16 prior formulations have been rejected despite seeming logically equivalent; guessing incorrectly now costs 2 more points (-33 total) versus not guessing and letting the game end at turn 30 with -32, so I'll play a neutral card and avoid another failed guess penalty.


#### All your previous failed rule guesses (if any)

- "The card is accepted if and only if its rank is 7 or higher (where Ace=1, 2-10=face value, J=11, Q=12, K=13)"
- "A card is accepted if and only if it is black (spade or club) AND its rank is 7 or higher (7, 8, 9, 10, J, Q, K)."
- "A card is accepted if and only if it is black (spade or club) AND its rank is either 7 or 13 (King or Seven)"
- "A card is accepted if and only if its rank is 7 or 13 (i.e., the card must be a Seven or a King)"
- "A card is accepted if and only if it is a 7 or a King (rank 7 or 13), is black (spade or club), and its suit strictly alternates with the previous mainline card (spades and clubs alternate)."
- "A card is accepted if and only if it has rank 7 or 13 (Seven or King), is black (Spade or Club), and has a suit different from the immediately preceding mainline card."
- "A card is accepted if and only if: (1) its rank is 7 or 13 (Seven or King), and (2) the suit depends on position parity in the mainline—if it will be at an odd position (1st, 3rd, 5th...), it must be a Spade; if at an even position (2nd, 4th, 6th...), it must be a Club."
- "The card at each mainline position must match a repeating four-card cycle: K♠, K♣, 7♠, 7♣. Position 1 must be K♠, position 2 must be K♣, position 3 must be 7♠, position 4 must be 7♣, position 5 must be K♠, and so on."
- "A card is accepted if and only if: (1) it is black (Spade or Club), (2) its rank is 7 or 13 (Seven or King), and (3) it is at mainline position 1 or 2, OR its suit matches the suit of the card exactly 2 mainline positions back in the sequence."
- "A card is accepted if and only if it is black (Spade or Club), has rank 7 or 13 (Seven or King), and its suit alternates with the previous mainline card's suit (Spade must follow Club, Club must follow Spade)."
- "Each position on the mainline requires a specific predetermined card. Positions 1-4 must be K♠, K♣, 7♠, 7♣ respectively. For any position n ≥ 5: if n is odd, the card must be K♠; if n is even, the card must be 7♣."
- "A card is accepted if and only if it matches the required card for its position in the mainline, following this exact sequence: position 1 = K♠, position 2 = K♣, position 3 = 7♠, position 4 = 7♣. For any position n ≥ 5: if n is odd, the card must be K♠; if n is even, the card must be 7♣."
- "A card is accepted if and only if: (1) it is a King (rank 13) or Seven (rank 7), (2) it is black (Spade or Club), and (3) positions 1-5 follow the sequence K♠, K♣, 7♠, 7♣, K♠ respectively, while position 6 onward must specifically be 7♣ (and no other card is accepted from position 6 forward)."
- "A card is accepted if and only if it matches a predetermined sequence: position 1 = K♠, position 2 = K♣, position 3 = 7♠, position 4 = 7♣, and for position n ≥ 5, the card must be K♠ if n is odd and 7♣ if n is even."
- "A card is accepted if and only if it matches the required card for its position in this fixed sequence: the first 4 positions must be K♠, K♣, 7♠, 7♣ in order; for any position n ≥ 5, the card must be K♠ if n is odd, and the card must be 7♣ if n is even."
- "The mainline must follow a repeating 4-card sequence: K♠, K♣, 7♠, 7♣, K♠, K♣, 7♠, 7♣, ... where position N must contain the card specified by the ((N-1) mod 4 + 1)-th position in the cycle."

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
