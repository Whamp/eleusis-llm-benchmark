"""Rule compilation prompts for Eleusis."""

from eleusis.prompts.game_rules import get_eleusis_rules

__all__ = ["get_rule_compile_prompt", "get_library_generation_prompt"]


def get_rule_compile_prompt(rule_text: str) -> str:
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
