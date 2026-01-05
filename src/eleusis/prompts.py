"""Prompt templates for LLM interactions in Eleusis."""


# ================================================================================================

def get_continuation_prompt(xml_tag: str, force_answer: bool = False) -> str:
    """Get prompt for completing truncated structured response.

    Args:
        xml_tag: The XML tag to use for the response
        force_answer: If True, use stronger instruction to prevent further reasoning
    """
    if force_answer:
        # Stronger prompt for reasoning models that keep thinking
        return f"""STOP. Output ONLY the final answer now.
DO NOT think further. DO NOT reason. DO NOT use <think> tags.
Immediately output the <{xml_tag}> tag with valid JSON inside, then close with </{xml_tag}>.
Start your response with: <{xml_tag}>"""
    else:
        return f"""Please continue and COMPLETE your response now.
DO NOT REASON ABOUT IT FURTHER, just provide the missing content.
You MUST start your response immediately with the <{xml_tag}> tag.
You MUST finish with a properly closed </{xml_tag}> tag containing valid JSON.
- Include the complete JSON object in the XML tags
- Ensure all JSON braces and brackets are properly closed
"""


def get_eleusis_rules() -> str:
    """Generate ELEUSIS game rules explanation for LLM context."""
    return """=== ELEUSIS GAME RULES ===

This is a card game where a secret rule determines which cards may be played.

## Overview

A **Rule-maker** invents a secret rule governing which cards may be accepted.
Players attempt to deduce the rule by playing cards and observing which are accepted or rejected.

## Components

- **Cards:** 2 standard 52-card decks shuffled together (104 cards total)
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of acceptance
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after

The entire layout (mainline and all sidelines) is visible to all players at all times.

---

## The Secret Rule

The secret rule is deterministic and decides whether a newly played card is **accepted** or **rejected**.

**The rule must:**
- Depend only on information visible in the mainline: the candidate card and/or any
  previously accepted mainline cards (their suits, colors, ranks, parity, positions, etc.)
- For the first card played, evaluate based solely on that card's properties (since
  no mainline exists yet)
- Give a unique, unambiguous answer (accepted or rejected) for every possible card in every
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


def get_card_evaluation_prompt(rule_code: str) -> str:
    """Generate prompt for LLM to evaluate if a card satisfies a rule."""
    return f"""You are evaluating whether a card satisfies a rule in the Eleusis card game.

The rule is implemented as:
```python
{rule_code}
```

Given a card and the current mainline, determine if the card should be ACCEPTED or REJECTED.
Respond with only "ACCEPTED" or "REJECTED"."""
