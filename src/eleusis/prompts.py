"""Prompt templates for LLM interactions in Eleusis."""

# Common game rules explanation used across prompts
ELEUSIS_RULES = """=== ELEUSIS GAME RULES ===

We play a simplified version of the card game Eleusis.

## Overview

One player (the **Rule-maker**) invents a secret rule governing which cards may be
played. The other players (**Scientists**) take turns playing cards, learning from each
acceptance or rejection, and trying to deduce the rule.

## Components

- **Players:** 4 (one Rule-maker, three Scientists)
- **Cards:** 2 standard 52-card decks shuffled together into a single 104-card draw pile
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Game Structure

A full game consists of several rounds, so that each player serves as Rule-maker
the same number of times. The player with the lowest total score at the end wins.

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of
  acceptance.
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after.

The entire layout (mainline and all sidelines) is visible to all players at all times.

---

## Setup

### 1. Choose Turn Order

Randomly determine a starting Scientist. Play proceeds clockwise from that player,
skipping the Rule-maker.

### 2. Create the Secret Rule

Before dealing, the Rule-maker privately writes down a deterministic rule that decides
whether a newly played card is **in** (accepted) or **out** (rejected).

**The rule must:**
- Depend only on information visible in the mainline: the candidate card and/or any
  previously accepted mainline cards (their suits, colors, ranks, parity, positions, etc.)
- For the first card played, evaluate based solely on that card's properties (since
  no mainline exists yet)
- Give a unique, unambiguous answer (in or out) for every possible card in every
  possible mainline state

**The rule must NOT:**
- Reference sideline cards (rejected cards)
- Depend on hidden information (cards in the deck, cards in players' hands)
- Depend on player identity, turn order, or game history other than the mainline
- Include randomness or subjective judgment

See "Example Rules" at the end of this document for guidance on appropriate complexity.

### 3. Deal Hands

Each Scientist is dealt **12 cards**. The Rule-maker receives no cards.

### 4. Place the Starter Card

The Rule-maker draws cards one at a time from the top of the deck until finding a card
that satisfies the rule when evaluated as a first card (with an empty mainline). This
card is placed face-up as the first mainline card. All other drawn cards are shuffled
back into the deck.

---

## Turn Structure

On their turn, a Scientist must choose exactly one action:
- **A. Play a card**, or
- **B. Declare "no play"**

After resolving the action, play passes to the next Scientist clockwise.

### A. Play a Card

1. The Scientist selects one card from their hand that they believe is **in** under the secret rule.
2. They place it face-up to the right of the current last mainline card.
3. The Rule-maker announces the judgment:

   **If "In" (accepted):**
    - The card remains in place as the new last mainline card.
    - The Scientist does **not** draw. Their hand size decreases by 1.
    - The Scientist may now optionally **attempt to guess the rule** (see below).

   **If "Out" (rejected):**
    - The card is moved to a sideline column directly below the last mainline card.
    - The Scientist **draws 1 card** from the deck.

The turn then ends.

### B. Declare "No Play"

A Scientist uses this if they believe **none** of their cards would be accepted.

1. The Scientist reveals their entire hand to all players.
2. The Rule-maker checks whether any card in the hand would be **in** if played now:

   **If correct (no legal card exists):**
    - The Scientist chooses one card from their hand.
    - That card is placed in a sideline below the last mainline card.
    - The Scientist does **not** draw. Their hand size decreases by 1.
    - The Scientist may now optionally **attempt to guess the rule** (see below).

   **If incorrect (at least one legal card exists):**
    - The Rule-maker selects one of the legal cards and places it as the new last mainline card.
    - The Scientist **draws 1 penalty card** from the deck.

The turn then ends.

---

## Guessing the Rule

### When You May Guess

Immediately after:
- Successfully playing an **in** card, or
- Making a **correct no-play**

the Scientist may optionally attempt to state the secret rule.

### How to Guess

The Scientist writes down (or states) a verbal description of the rule.

The Rule-maker judges whether the stated rule is **equivalent** to the secret rule. Two
rules are equivalent if and only if they produce identical in/out judgments for every
possible (card, mainline-state) pair.

Note: The wording does not need to match exactly. Only logical equivalence matters.

### Outcome

- **Correct guess:** The round ends immediately.
- **Incorrect guess:** The Rule-maker announces the guess is wrong (without explaining
  why). The Scientist **draws 1 card** from the deck. Play continues with the next
  Scientist.

### Mandatory Guess at Zero Cards

If a Scientist's hand reaches **0 cards**, they **must** immediately guess the rule:
- If correct, the round ends.
- If incorrect, they draw 1 card and continue playing.

---

## Deck Exhaustion

If a player must draw but the deck is empty:
- They do not draw (no penalty beyond failing to reduce hand size).
- Play continues.

If the deck is empty **and** no player can make a legal play or correct no-play, the
round ends immediately.

---

## End of Round

A round ends when:
1. A Scientist **correctly guesses the rule**, or
2. The **deck is exhausted** and play cannot continue.

---

## Scoring

At the end of each round:

| Player | Score |
|--------|-------|
| Each Scientist | **+1 point per card remaining in hand** |
| Scientist who guessed correctly | **−3 bonus points** (added to their hand score, can
result in negative) |
| Rule-maker | Score equal to the **second-lowest** Scientist score for that round |

**Lower scores are better.**
After all rounds completed (each player having been Rule-maker the same number of
times), the player with the lowest total score wins.

---

## Example Rules

These examples illustrate appropriate rule complexity:

### Simple Rules
- "The card must be a different color than the last mainline card." (Alternating red/black)
- "The card must be a heart or a spade."
- "The card must have an even rank."

### Medium Rules
- "The card's rank must be higher than the last mainline card's rank. Any card may follow a King."
- "The card must share either the suit or the color with the last mainline card, but not both."
- "The card's rank must have a different parity (odd/even) than the last mainline card."

### Harder Rules (use sparingly)
- "The card's rank must differ from the last mainline card's rank by exactly 1 or 2
  (with Ace and King not considered adjacent)."
- "If the last mainline card is red, play a card with rank ≤ 7. If black, play a card
  with rank ≥ 7."

=== END OF THE ELEUSIS GAME RULES ===
"""



def get_rule_generation_prompt() -> str:
    """Generate prompt for LLM to create a game rule."""
    return f"""{ELEUSIS_RULES}


**YOU PLAY AS THE RULE-MAKER**

=== YOUR TASK: CREATE A SECRET RULE ===

You are the Rule-maker. Create a rule determining which cards are
IN (accepted) or OUT (rejected).

RULE CONSTRAINTS:
1. DETERMINISTIC: Same card + same mainline → always same result
2. CAN depend on:
   - Candidate card properties (rank, even/odd, face/pip, suit, color, etc.) 
   - Previously ACCEPTED mainline cards (their properties and positions)
3. CANNOT depend on:
   - Rejected cards (cards in brackets)
   - Hidden information (deck, players' hands)
   - Player identity, turn order, or randomness
4. MUST work with EMPTY mainline (any first card must have a valid answer IN or OUT)

COMPLEXITY:
- Simple: "Alternating colors", "Even ranks only"
- Medium: "Rank higher than previous", "Same suit or same color, not both"
- IMPORTANT Avoid overly complex unsolvable rules

**IMPORTANT Guidance for Rule-makers:** 
Aim for rules that are deducible within 15-25 plays. 
At a given point in the game, about 20-40% of possible cards should be legal plays.
Avoid rules so complex that random guessing is the only viable strategy.
Since your score equals the second-lowest Scientist score, you benefit when Scientists
can make progress. 
An unsolvable rule leads to high hand counts for everyone — including
you.
Avoid rules that depend on complex sequences or deep history; or has a singular behavior for the first card, etc.
CHOOSE A SIMPLE RULE THAT ALLOWS SCIENTISTS TO LEARN AND IMPROVE OVER TIME.

OUTPUT FORMAT:
Think through your rule if needed, once you are done, wrap your rule in XML tags:

<RULE>
  <DESCRIPTION>Natural language description of the rule (1-2 sentences)</DESCRIPTION>
  <CODE>
# Python code that implements the rule
# Available: card.rank (1-13), card.color ("red"/"black"), card.is_even, card.is_odd
#            card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
#            mainline: list of Card objects
# Must handle empty mainline (first card)
if not mainline:
    return True  # or False depending on first card rule, e.g., card.is_even
# Your logic here
return True/False
  </CODE>
</RULE>

Examples:
<RULE>
  <DESCRIPTION>Cards must alternate between red and black colors.</DESCRIPTION>
  <CODE>
if not mainline:
    return True
last_card = mainline[-1]
return card.color != last_card.color
  </CODE>
</RULE>

<RULE>
  <DESCRIPTION>Only cards with even ranks (2,4,6,8,10,12) are accepted.</DESCRIPTION>
  <CODE>
return card.is_even
  </CODE>
</RULE>

<RULE>
  <DESCRIPTION>If the last mainline card is red, play rank ≤7. If black, play rank ≥7.</DESCRIPTION>
  <CODE>
if not mainline:
    return True
last_card = mainline[-1]
if last_card.color == "red":
    return card.rank <= 7
else:
    return card.rank >= 7
  </CODE>
</RULE>

IMPORTANT:
- Code must be deterministic (same inputs → same output)
- No imports, no file I/O, no external calls
- Only use: len, sum, min, max, any, all, abs
- Return True (accepted) or False (rejected)
"""



def get_move_selection_prompt(
    compact_board: str, hand_cards: list[dict], deck_remaining: int, play_history: list[dict]
) -> str:
    """Generate prompt for LLM to select a move as Scientist."""
    hand_str = ", ".join([c["symbol"] for c in hand_cards])

    # Format play history
    history_str = ""
    if play_history:
        history_str = "\n\nYOUR PREVIOUS ATTEMPTS:\n"
        for entry in play_history[-10:]:  # Show last 10 attempts
            action_type = entry.get("action", "unknown")
            card = entry.get("card", "N/A")
            reasoning = entry.get("reasoning", "")
            accepted = entry.get("accepted")

            if accepted is not None:
                result = "✓ ACCEPTED" if accepted else "✗ REJECTED"
                history_str += f"- {card}: {result}\n"
                if reasoning:
                    history_str += f"  Your reasoning: {reasoning}\n"
            elif action_type == "no_play":
                correct = entry.get("correct", False)
                result = "✓ CORRECT" if correct else "✗ INCORRECT"
                history_str += f"- NO-PLAY: {result}\n"
                if reasoning:
                    history_str += f"  Your reasoning: {reasoning}\n"

    return f"""{ELEUSIS_RULES}

**YOU PLAY AS A SCIENTIST**

=== YOUR TASK: CHOOSE YOUR MOVE ===

You are a Scientist trying to deduce the secret rule.

CURRENT BOARD (mainline + sideline cards in brackets at the position they have been
played and rejected):
{compact_board}

YOUR HAND: {hand_str}
DECK REMAINING: {deck_remaining} cards
{history_str}
YOUR OPTIONS:
1. PLAY a card: Specify the card from your hand (e.g., "5♥")
2. NO-PLAY: Use the value "no_play" if you believe no card in your hand will be accepted

OUTPUT FORMAT:
Think through the pattern, then wrap your decision in XML tags.

IMPORTANT ABOUT GUESSING:
- You must ALWAYS provide a "tentative_rule" describing your current belief about the
  secret rule, even if you're not confident. This helps track your evolving understanding.
- Set "guess_rule_if_accepted" to true ONLY when you're confident in your tentative_rule.
- If your action succeeds (card accepted or correct no-play) AND guess_rule_if_accepted
  is true, your tentative_rule will be officially submitted to the referee.
- **If your guess is CORRECT: You win the round immediately!**
- **If your guess is INCORRECT: You draw 1 penalty card and continue playing.**
- Since incorrect guesses have a penalty, only guess when you're reasonably confident.

<ACTION>
{{
    "reasoning": "Your analysis of the pattern and why you're playing this card/no-play",
    "action": "5♥" or "no_play",
    "tentative_rule": "Your current best guess about the rule (always provide this)",
    "guess_rule_if_accepted": true or false (whether to officially guess if accepted)
}}
</ACTION>

Example:
<ACTION>
{{
    "reasoning": "I see red and black cards alternating. My 3♥ is red, last card was black.",
    "action": "3♥",
    "tentative_rule": "Cards must alternate between red and black colors",
    "guess_rule_if_accepted": false
}}
</ACTION>
"""



def get_referee_comparison_prompt(secret_rule: str, guessed_rule: str, mainline:str) -> str:
    """Generate prompt for referee to compare two rules for equivalence."""
    return f"""{ELEUSIS_RULES}

**YOU PLAY AS THE REFEREE**

=== YOUR TASK: DETERMINE RULE EQUIVALENCE ===

A Scientist is attempting to guess the secret rule. Your job is to determine if their
guess is equivalent to the actual secret rule.

**EQUIVALENCE DEFINITION:**
Two rules are said equivalent if *for that game and the present state of the mainline* 
both rules would produce identical IN/OUT judgments from now on to the end of the round,
for every possible card played in every possible future mainline state.

That means that the Scientist's might look more restrictive because the mainline has
already been partially built, but if both rules would accept and reject the same cards
from this point onward, they are equivalent.

Example: If the secret rule is "All cards should be of the same color" and the
mainline started with "red", then the guessed rule "Cards must be red" would be
equivalent because both rules would accept only red cards from that point onward.

Equivalent rules may use different wording but must have identical behavior
- Example: "Red cards only" ≡ "Card must be Hearts or Diamonds"
- Example: "Even ranks" ≡ "Rank is 2, 4, 6, 8, 10, or 12"


**ACTUAL SECRET RULE:**
{secret_rule}

**SCIENTIST'S GUESSED RULE:**
{guessed_rule}

**CURRENT MAINLINE STATE:**
{mainline}

OUTPUT FORMAT:
Think through your analysis carefully, then wrap your verdict in XML tags:
<VERDICT>
{{
    "equivalent": true or false,
    "reasoning": "Detailed explanation of why the rules are or are not equivalent"
}}
</VERDICT>
"""



def get_rule_evaluation_prompt(
    rule_text: str, card: dict, mainline_compact: str
) -> str:
    """Generate prompt for LLM to evaluate if a card follows their rule."""
    return f"""{ELEUSIS_RULES}

**YOU PLAY AS THE RULE-MAKER**

=== EVALUATE CARD AGAINST YOUR RULE ===

You created this rule:
{rule_text}

Current mainline: {mainline_compact}

Card to evaluate: {card['symbol']} (rank={card['rank']}, suit={card['suit']}, color={card['color']})

Is this card IN (accepted) or OUT (rejected) according to YOUR rule?

OUTPUT FORMAT:
Think through how your rule applies, then wrap your answer in XML tags:

<EVALUATION>
{{
    "result": "in" or "out",
    "reasoning": "Brief explanation"
}}
</EVALUATION>
"""
