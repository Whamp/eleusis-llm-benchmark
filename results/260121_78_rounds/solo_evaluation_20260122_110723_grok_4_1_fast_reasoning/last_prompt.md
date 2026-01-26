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
Penalty so far for 18 wrong guesses: 36
Current potential score if you guess the rule correctly this turn: -36

#### Mainline & Sidelines

This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after :

K♠ [8♦] K♣ 7♠ [4♠] [J♥] [J♠] 7♣ 6♠ [A♣] [A♠] [10♣] 6♦ A♣ A♥ 3♦ [6♦] [5♥] 3♣ 2♠ [3♠] [4♣] [3♦] [Q♠] [7♠] [9♠] [K♠] [4♥] [4♦] [3♥]

#### Your hand

10♦, 5♦, 8♦, 9♥, 9♥, Q♦, 3♥, 9♦, 7♥, J♦, 7♥, 8♠

#### Your recent play history

Your last 3 turns (the card you played, the outcome and your reasoning summary):
- 4♥: ✗ REJECTED
  Your reasoning: The secret rule requires same rank as last (any suit) when mainline length before play is odd to complete a pair, and a previously unseen rank with suit next in the cycle ♥→♦→♣→♠→♥ after the last card's suit when even to start a new pair, perfectly matching all accepts (including new starter suits) and no even-length rejects because correct suits were played, while explaining failures of rank-only guesses; playing 4♥ expects rejection as rank 4≠2 on odd length.
- 4♦: ✗ REJECTED
  Your reasoning: The secret rule builds exact pairs of each rank on the mainline: accept a card matching the rank with odd count (currently only rank 2 appears once) if any such rank exists, or a never-before-seen rank if all ranks have even counts; previous guesses failed as they tied to position/length/last card specifically rather than general count parity invariant; playing 4♦ (new rank) to test but expects reject on current odd-count state.
- 3♥: ✗ REJECTED
  Your reasoning: Previous parity guesses failed because they assumed exactly one odd-count rank or tied to length/last card, but the rule uses global counts: accept if played rank has odd count (matching any such) or all counts even and played rank new (count 0); playing 3♥ (rank 3 has even count 2, while 2 has odd 1) expects rejection confirming need to match the odd rank.


#### All your previous failed rule guesses (if any)

- "The card must be black (clubs or spades) and have an odd rank."
- "The card must be a club if the previous mainline card is a spade, and a spade if the previous mainline card is a club."
- "If the previous mainline card is a spade, the next card must be a club of the same rank; if the previous mainline card is a club, the next card must be any spade."
- "If the last mainline card is a spade of rank R, the next card must be a club of exactly rank R; if the last mainline card is a club of rank R, the next card must be a spade of rank strictly less than R."
- "If the last mainline card is a spade of rank R, the next card must be a club of exactly rank R; if the last mainline card is a club of rank R, the next card must be a spade of rank less than or equal to R."
- "If the last mainline card is a spade, the new card must have exactly the same rank regardless of suit; otherwise, the new card must have a strictly lower rank than the last mainline card regardless of suit."
- "If the last mainline card is a spade, the new card must have exactly the same rank regardless of suit; otherwise, the new card must have a rank less than or equal to that of the last mainline card regardless of suit."
- "If the previous mainline card is a spade, the new card must have exactly the same rank regardless of suit; otherwise (after a non-spade), the new card must have a rank strictly lower than the rank of the most recent spade on the mainline regardless of suit."
- "If the number of cards currently on the mainline is odd, the played card must have exactly the same rank as the last mainline card; if even, any card is accepted."
- "If the mainline has at least two cards and the last two mainline cards have the same rank, any card is accepted; otherwise, only a card with the same rank as the last mainline card is accepted."
- "If the current number of cards on the mainline (before playing) is odd, the played card must have exactly the same rank as the last mainline card; if even, any card is accepted."
- "If the number of cards on the mainline before playing is odd, the played card must have the same rank as the last mainline card; if even, the played card must have a rank that does not yet appear anywhere on the mainline."
- "If the number of cards on the mainline before playing is odd, the played card must have the same rank as but different suit from the last mainline card; if even, any card is accepted."
- "If the last mainline card's rank appears exactly once in the mainline, the played card must have the same rank; otherwise, the played card must have a rank that has not yet appeared anywhere in the mainline."
- "If the last mainline card's rank appears exactly once in the mainline, the played card must have the same rank; otherwise, the played card must have a different rank from the last mainline card."
- "If the mainline has an odd number of cards before the play, the card must match the rank of the last mainline card (any suit); if even, the card must have a rank not previously appearing on the mainline and a suit that is the successor to the last mainline card's suit in the cycle Hearts → Diamonds → Clubs → Spades → Hearts."
- "The played card is accepted if it matches the rank that appears an odd number of times on the mainline when exactly one such rank exists, or if all ranks appear an even number of times and the played rank has never appeared on the mainline."
- "The played card is accepted if its rank currently has an odd number of appearances in the mainline or if every rank in the mainline has an even number of appearances and the played rank has zero appearances in the mainline."

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
