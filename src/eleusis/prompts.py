"""Prompt templates for LLM interactions in Eleusis."""

from pathlib import Path
import yaml


# ================================================================================================

def get_continuation_prompt(xml_tag: str) -> str:
    """Get prompt for completing truncated structured response."""
    return f"""Please continue and COMPLETE your response now.
    DO NOT REASON ABOUT IT FURTHER, just provide the missing content.
    You MUST start your response immediately with the <{xml_tag}> tag.
    You MUST finish with a properly closed </{xml_tag}> tag containing valid JSON.
    - Include the complete JSON object in the XML tags
    - Ensure all JSON braces and brackets are properly closed
"""

# Common game rules explanation used across prompts
def _load_game_config() -> dict:
    """Load game config from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["game"]


def get_eleusis_rules() -> str:
    """Generate ELEUSIS game rules using values from config.yaml"""
    game_config = _load_game_config()

    cards_per_scientist = game_config['cards_per_scientist']
    card_reject_penalty = game_config['card_reject_penalty']
    no_play_correct_reduction = game_config['no_play_correct_reduction']
    no_play_incorrect_penalty = game_config['no_play_incorrect_penalty']
    wrong_guess_penalty = game_config['wrong_guess_penalty']
    correct_guess_bonus = game_config['correct_guess_bonus']

    return f"""=== ELEUSIS GAME RULES ===

This is simplified version of the card game Eleusis.

## Overview

A **Rule-maker** invents a secret rule governing which cards may be
played. The players (**Scientists**) take turns playing cards, learning from each
acceptance or rejection, and trying to deduce the rule.

## Components

- **Players:** typically 2 to 4 scientists
- **Cards:** 2 standard 52-card decks shuffled together into a single 104-card draw pile
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Game Structure

A full game consists of several rounds, so that the rule changes at every round. 
The player with the *lowest* total score at the end wins.

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of
  acceptance.
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after.

The entire layout (mainline and all sidelines) is visible to all players at all times.

---

## Setup

### Secret rule

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

These examples illustrate appropriate rule complexity:
- "The card must be a different color than the last mainline card." (Alternating red/black)
- "The card must be a heart or a spade."
- "The card must share either the suit or the color with the last mainline card, but not both."
- "The card's rank must have a different parity (odd/even) than the last mainline card."
- "The card's rank must differ from the last mainline card's rank by exactly 1 or 2
  (with Ace and King not considered adjacent)."
- "If the last mainline card is red, play a card with rank ≤ 7. If black, play a card
  with rank ≥ 7."


### Deal Hands

Each Scientist player is dealt **{cards_per_scientist} cards**. The Rule-maker receives no cards.

The Rule-maker draws cards one at a time from the top of the deck until finding a card
that satisfies the rule when evaluated as a first card (with an empty mainline). 

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
    - The Scientist **draws {card_reject_penalty} card** from the deck.

The turn then ends.

### B. Declare "No Play"

A Scientist uses this if they believe **none** of their cards would be accepted.

1. The Scientist reveals their entire hand to all players.
2. The Rule-maker checks whether any card in the hand would be **in** if played now:

   **If correct (no legal card exists):**
    - The Scientist chooses one card from their hand.
    - That card is placed in a sideline below the last mainline card.
    - The Scientist receives N-{no_play_correct_reduction} new cards from the deck, where N is their hand size
    - All the old cards are discarded.
    - The Scientist may now optionally **attempt to guess the rule** (see below).

   **If incorrect (at least one legal card exists):**
    - The Rule-maker selects one of the legal cards and places it as the new last mainline card.
    - The Scientist **draws {no_play_incorrect_penalty} penalty card** from the deck.

The turn then ends.

---

## Guessing the Rule

Immediately after successfully playing an **in** card, or making a correct no-play, the Scientist may optionally attempt to state the secret rule.

The Rule-maker judges whether the stated rule is **equivalent** to the secret rule. Two
rules are equivalent if and only if they produce identical in/out judgments for every
possible (card, mainline-state) pair.

Note: The wording does not need to match exactly. Only logical equivalence matters.

- **Correct guess:** The round ends immediately.
- **Incorrect guess:** The Rule-maker announces the guess is wrong (without explaining
  why). The Scientist **draws {wrong_guess_penalty} cards** from the deck. Play continues with the next
  Scientist.


## End of Round

A round ends when:
1. A Scientist **correctly guesses the rule**, or
2. The **deck is exhausted** and play cannot continue.

---

## Scoring

- Each get +1 point per card remaining in hand
- If a scientist guessed correctly, they receive **{correct_guess_bonus} bonus points** (added to their hand score, can
result in negative) |

**Lower scores are better.**
After all rounds completed (each player having been Rule-maker the same number of
times), the player with the lowest total score wins.

=== END OF THE ELEUSIS GAME RULES ===
"""



# ================================================================================================

def get_rule_compilation_prompt(rule_text: str) -> str:
    """Generate prompt for LLM to convert a game rule into Python code."""

    return f"""

    === YOUR TASK : CONVERT A RULE ABOUT CARDS INTO PYTHON CODE ===

    The context is a game of ELEUSIS, where a secret rule determines whether a played card is
    accepted (in) or rejected (out) based on the current mainline of accepted cards.

    {get_eleusis_rules()}

    === YOUR TASK : CONVERT THE FOLLOWING RULE INTO PYTHON CODE ===

    Rule: {rule_text}

    CRITICAL: Generate ONLY the function body code, NOT a complete function definition.
    Do NOT start with "def", do NOT define a new function.
    We will wrap your code in a function automatically.

    The code should:
    - Use available properties: card.rank (1-13), card.color ("red"/"black")
    - Use card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
    - Have access to mainline: list of Card objects
    - Handle empty mainline (first card) with: if not mainline:
    - Return True (accepted) or False (rejected)

    RESPONSE FORMAT (function body only, enclosed in <CODE> tags):

    <CODE>
    # Python code that implements the rule
    # Available: card.rank (1-13), card.color ("red"/"black"),
    #            card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
    #            mainline: list of Card objects
    # Must handle empty mainline (first card)
    if not mainline:
        # Your logic here
        return True/False
    # Your logic here
    return True/False
    </CODE>

    Example: 
    Rule is "Cards must alternate between red and black colors."
    <CODE>
    if not mainline:
        return True
    last_card = mainline[-1]
    return card.color != last_card.color
    </CODE>

    Example: 
    Rule is "Only cards with even ranks (2,4,6,8,10,12) are accepted."
    <CODE>
    return card.rank % 2 == 0
    </CODE> 
    """

# ================================================================================================


def get_library_generation_prompt(num_rules: int = 20) -> str:
    """Generate prompt for LLM to create multiple rules at once."""
    return f"""

**YOU PLAY AS THE RULE-MAKER IN A CARD GAME OF ELEUSIS**

{get_eleusis_rules()}

=== YOUR TASK: CREATE A LIBRARY OF {num_rules} SECRET RULES ===

=== WHAT IS A RULE AND HOW TO CREATE ONE === 

A rule is a deterministic function that decides whether a newly played card is
**in** (accepted) or **out** (rejected) based on the current mainline state.

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

**IMPORTANT Guidance for Rule-makers:**
Aim for rules that are deducible within 15-25 plays.
At a given point in the game, about 20-40% of possible cards should be legal plays.
Avoid rules so complex that random guessing is the only viable strategy.
Since your score equals the second-lowest Scientist score, you benefit when Scientists
can make progress.
An unsolvable rule leads to high hand counts for everyone — including
you.
Avoid rules that depend on complex sequences or deep history; or has a singular
behavior for the first card, etc.


OUTPUT FORMAT:
You should both output the description of the rule and the Python code implementing it.
Wrap your entire response in XML tags as shown below

CRITICAL: Generate ONLY the function body code, NOT a complete function definition.
Do NOT start with "def", do NOT define a new function.
We will wrap your code in a function automatically.

The code should:
- Use available properties: card.rank (1-13), card.color ("red"/"black")
- Use card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
- Have access to mainline: list of Card objects
- Handle empty mainline (first card) with: if not mainline:
- Return True (accepted) or False (rejected)


<RULE>
  <NAME>Rule Name Here</NAME>
  <DESCRIPTION>Natural language description of the rule (1-2 sentences)</DESCRIPTION>
    <CODE>
    # Python code that implements the rule
    # Available: card.rank (1-13), card.color ("red"/"black"),
    #            card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
    #            mainline: list of Card objects
    # Must handle empty mainline (first card)
    if not mainline:
        # Your logic here
        return True/False
    # Your logic here
    return True/False
    </CODE>
</RULE>

EXAMPLE:
<RULE>
  <NAME>Alternating Colors</NAME> 
  <DESCRIPTION>Cards must alternate between red and black colors.</DESCRIPTION>
  <CODE>
if not mainline:
    return True
last_card = mainline[-1]
return card.color != last_card.color
  </CODE>
</RULE>



Generate {num_rules} different rules for the Eleusis card game. Each rule should be:
- DETERMINISTIC (same inputs always give same output)
- PLAYABLE (not too complex, learnable in 15-25 plays)
- DIVERSE (cover different types of patterns)

COMPLEXITY MIX:
- {num_rules // 3} Simple rules (e.g., "Even ranks only", "Red cards only", "Alternating colors", "Rank higher than previous")
- {num_rules // 3} Medium rules (e.g., "Rank difference of 2 from previous", "Color alternates every two cards")
- {num_rules - 2 * (num_rules // 3)} Harder rules (e.g., "Fibonacci sequence of ranks", "Prime number ranks only", "Suits in a specific repeating order")

Generate {num_rules} unique, interesting, playable rules now.
Do not overcomplicate the rules, a rule impossible to guess will not be fun, and will be rejected by the rule-maker.

Answer below by providing all rules, each wrapped in <RULE> XML tags as shown above.
"""


# ================================================================================================


def get_action_selection_prompt(
    compact_board: str,
    hand_cards: list[dict],
    deck_remaining: int,
    play_history: list[dict],
    failed_guesses: list[dict] | None = None,
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
            reasoning_summary = entry.get("reasoning_summary", "")
            accepted = entry.get("accepted")

            if accepted is not None:
                result = "✓ ACCEPTED" if accepted else "✗ REJECTED"
                history_str += f"- {card}: {result}\n"
                if reasoning_summary:
                    history_str += f"  Your reasoning: {reasoning_summary}\n"
            elif action_type == "no_play":
                correct = entry.get("correct", False)
                result = "✓ CORRECT" if correct else "✗ INCORRECT"
                history_str += f"- NO-PLAY: {result}\n"
                if reasoning_summary:
                    history_str += f"  Your reasoning: {reasoning_summary}\n"

    # Format failed guesses history
    failed_guesses_str = ""
    if failed_guesses:
        failed_guesses_str = "\n\nFAILED RULE GUESSES (by all players):\n"
        for entry in failed_guesses:
            player = entry.get("player", "Unknown")
            guess = entry.get("guess", "")
            failed_guesses_str += f"- {player}: \"{guess}\"\n"

    return f"""
    
**YOU PLAY AS A SCIENTIST IN A GAME OF ELEUSIS, YOUR TASK IS TO SELECT YOUR NEXT ACTION**

{get_eleusis_rules()}

=== YOUR TASK: CHOOSE YOUR ACTION ===

You are a Scientist trying to deduce the secret rule.

CURRENT BOARD (mainline + sideline cards in brackets at the position they have been
played and rejected):
{compact_board}

FAILED RULE GUESSES SO FAR (IF ANY):
{failed_guesses_str}
ALL OF THOSE GUESSES WERE INCORRECT, THEY DO NOT MATCH THE SECRET RULE.

YOUR HAND: {hand_str}

DECK REMAINING: {deck_remaining} cards

YOUR PLAY HISTORY:
{history_str}

YOUR OPTIONS:
1. PLAY a card: Specify the card from your hand (e.g., "5♥")
2. NO-PLAY: Use the value "no_play" if you believe no card in your hand will be accepted

OUTPUT FORMAT:
You can freely reason step by step about this case in your response.
Then your response should end with your final decision wrapped in XML tags.

<your reasoning about the situation>
<ACTION>
{{
    "reasoning_summary": "A summary of your analysis of the pattern and why you're playing this card/no-play",
    "action": "5♥" or "no_play",
    "tentative_rule": "Your current best guess about the rule (always provide this, it has to be unequivocal)",
    "confidence_level": 0-10 (your confidence in the tentative_rule, 0=lowest, you have no clue, 10=maximum, you are 100% sure),
    "guess_rule_if_accepted": true or false (whether to officially guess if accepted)
}}
</ACTION>

IMPORTANT ABOUT GUESSING:
- You must ALWAYS provide a "tentative_rule" describing your current belief about the
  secret rule, even if you're not confident. This helps track your evolving understanding.
  This rule has to be unequivocal (no "maybe", etc.)
- Provide a "confidence_level" from 0-10 indicating how confident you are in your tentative_rule.
  (0 = no clue, 10 = 100% sure). This has no direct gameplay effect but helps you reflect on your certainty.
- Set "guess_rule_if_accepted" to true ONLY when you're confident in your tentative_rule.
- If your action succeeds (card accepted or correct no-play) AND guess_rule_if_accepted
  is true, your tentative_rule will be officially submitted to the referee.
- **If your guess is CORRECT: You win the round immediately!**
- **If your guess is INCORRECT: You draw 1 penalty card and continue playing.**
- Since incorrect guesses have a penalty, only guess when you're reasonably confident.


Example:
<ACTION>
{{
    "reasoning_summary": "I see red and black cards alternating. My 3♥ is red, last card was black.",
    "action": "3♥",
    "tentative_rule": "Cards must alternate between red and black colors",
    "confidence_level": 7,
    "guess_rule_if_accepted": false
}}
</ACTION>
"""
