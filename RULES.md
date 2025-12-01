# Easy Eleusis

This is a set of rules for "Easy Eleusis", a simplified version of the card game Eleusis.

## Introduction

The game is played with 4 players, using 2 standard decks of 52 cards:
    - Ace = 1 (low), J = 11, Q = 12, K = 13.
    - 4 suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black).

In each round, one player acts as **Rule-maker** (dealer), the others are **Scientists** (ordinary players).

The Rule-maker chooses a secret rule for forming a sequence of cards.
Scientists hold hands of cards and take turns playing cards to a common layout.
The Rule-maker announces whether each played card is in (accepted) or out (rejected) according to the secret rule.
Scientists try to deduce the secret rule from the pattern of accepted and rejected cards.
Goal for Scientists: get rid of their cards and/or correctly state the secret rule.

## Setup

### Secret Rule
Before dealing, the Rule-maker chooses and writes down a deterministic rule that decides whether a newly played card is in or out, given the current layout. 
The rule must depend only on information visible in the layout (current last mainline card and/or earlier cards; card suits, colours, ranks, parity, etc.)
It may not depend on hidden information (e.g. unknown cards in deck/hand) or on player identity or order.
For every possible situation and card, the rule must give a unique answer: in or out.

### Dealing and starting layout

12 cards are dealt to each Scientist, the Rule-Maker receives no cards.

The Rule-maker draws cards from the top of the pile until finding one that is in under the rule and reveal this card face up as the starter.

Then players take turns playing until the end of the round.

Cards will be played by Scientists into a **mainline** and **sidelines**.
- Mainline: horizontal row of all cards that have been judged in, in order of acceptance.
- Sidelines: for any out card, place it in a vertical column below the mainline card it was attempted after.

## Turn structure

On their turn, scientists must choose exactly one action:
A. Play a card, or
B. Declare “no play”.

Then the turn passes to the next Scientist in order.

### A. Play a card

1. Player chooses one card from their hand that they think is **in** under the secret rule.
2. They place it face up to the **right** of the current last mainline card (a tentative mainline extension).
3. Rule-Maker announces:
    - If **“In” (accepted):**
        - Card stays in place as the new last mainline card.
        - Player **do not** draw; their hand size decreases by 1.
        - Player can now optionally attempt to guess the rule (see below).
    - If **“Out” (rejected):**
        - Move the card from the mainline position to a **sideline below** the last mainline card.
        - Player **draw 1 card** from the top of the deck into their hand.

Then turn ends.

### B. Declare “No play”

Player uses this if they believe **no card in their hand** would be accepted if played now.

1. They reveal their entire hand to the dealer and all players.
2. Dealer checks if any card in their hand would be **in** if played now:
    - **Correct no-play (they truly had no legal card):**
        - Player choose **one** card from their hand.
        - Place it directly below the last mainline card as a **sideline card** (it is, by definition, “out”).
        - Remove that card from their hand (hand size −1).
        - Mainline does **not** change.
        - Player can now optionally attempt to guess the rule (see below).
    - **Incorrect no-play (they did have at least one legal card):**
        - Dealer selects one of the legal cards and places it as the new last **mainline** card (an automatic correct play).
        - Player **draw 1 penalty card** from the deck into their hand.

Then their turn ends.

### Guessing the Rule

- **Timing:** Immediately after a player either
    - successfully plays an **in** card to the mainline, or
    - makes a **correct no-play**,
      player may optionally attempt to **state the secret rule**.
- They write down a verbal description of the rule
- Dealer decides whether their stated rule is **equivalent** to the real rule (i.e. would classify every possible card in every possible situation the same way).

Outcome:
- **If the guess is correct:** the **round ends immediately**.
- **If the guess is wrong:**
    - Dealer says it is wrong.
    - Player **draws 1 card** from the deck.
    - Play continues with the next player.

Special case:
- If a player’s hand ever reaches **0 cards**, that player **must immediately guess** the rule.
    - If correct, the round ends (they have both gone out and solved the rule).
    - If wrong, they draw 1 card and continue playing with that new card.


## Ending a Round and scoring

A round ends when either:

1. **A player correctly guesses the rule**, or
2. **The deck is exhausted** (no more cards to draw).

Scoring:
1. At round end, each non-dealer player scores **1 point per card in hand** (fewer is better).
2. A player who **correctly guessed the rule** scores **−3 extra points** (a bonus; lower total is better).
3. Dealer’s score for the hand = the **second lowest** player score.
4. After each player has been Dealer the same number of times, **lowest total score wins**.