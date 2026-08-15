"""Rule compilation prompts for Eleusis."""

__all__ = ["get_rule_compile_prompt"]


def get_rule_compile_prompt(rule_text: str) -> str:
    """Generate prompt for LLM to convert a rule into Python code with a nickname."""
    return f"""

# YOUR TASK: CONVERT A RULE INTO PYTHON CODE AND GIVE IT A NICKNAME

You are given a natural language description of a rule for the card game "Eleusis".

Your task is to:
1. Create a short snake_case nickname for the rule (e.g., "alternating_colors", "even_ranks_only")
2. Convert the rule into Python code for automatic evaluation.

## Game summary
The game Eleusis involves playing cards from a standard deck.

The game uses 2 standard 52-card decks shuffled together (104 cards total):
- Ranks: Ace = 1 (low), 2-10, Jack = 11, Queen = 12, King = 13. Number cards are 1-10, face cards are 11-13.
- Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

The dealer has created a hidden rule that determines whether played cards are accepted or rejected.
Cards are played one by one by the other players
A card is placed into a "mainline" if they are accepted by the rule, or into "sidelines" if they are rejected.
The rule determines whether a newly played card is accepted or rejected based on the current state of the mainline.
The mainline is a sequence of previously accepted cards, in the order they were played.
The players have to discover the hidden rule by playing cards and observing which ones are accepted or rejected.
When evaluating a new card, only the properties of the card and the mainline are relevant.

## YOUR TASK : converting a rule into Python code for automatic evaluation

### Rule to convert

The rule to convert is provided below.

{rule_text}

### Nickname requirements

- The nickname should be a concise, descriptive identifier for the rule.
- Use only lowercase letters and underscores to separate words (snake_case).
- Avoid using special characters, spaces, or numbers in the nickname.

### Code requirements

CRITICAL: Generate ONLY the function body code, NOT a complete function definition.
Do NOT start with "def", do NOT define a new function.
We will wrap your code in a function automatically.

The code should:
- Use available properties: card.rank (1-13), card.color ("red"/"black")
- Use card.suit.suit_name ("hearts", "diamonds", "clubs", "spades")
- Have access to mainline: list of Card objects
- Handle empty mainline (first card) with: if not mainline:
- Return True (accepted) or False (rejected)

SANDBOX RESTRICTIONS - Only these operations are available:
- Types: bool, int, str, list, tuple, set, dict
- Iteration: range(), reversed(), sorted(), enumerate(), zip()
- Aggregation: len(), sum(), min(), max(), abs(), any(), all()
- Math: round(), divmod()
- List/string methods work normally: .index(), slicing, etc.
- Standard arithmetic and comparisons: +, -, *, /, //, %, ==, !=, <, >, <=, >=
- Logical operators: and, or, not, in

DO NOT USE (will cause errors):
- import statements (no math, collections, etc.)
- File operations, eval, exec, or any external calls

Common patterns that work:
- Prime ranks: primes = {{2, 3, 5, 7, 11, 13}}; card.rank in primes
- Suit values: suit_val = {{"hearts":1, "diamonds":2, "clubs":3, "spades":4}}[card.suit.suit_name]
- Floor division: x // 2 instead of math.floor(x/2)

If the rule is about a property that every card must have (e.g., color, suit, rank), you can directly check that property.
In that case, the first card (when mainline is empty) must also satisfy the property.

## RESPONSE FORMAT

Provide your response in this exact XML format:

<NAME>snake_case_nickname</NAME>
<CODE>
# Your Python code here
</CODE>

## EXAMPLES

Rule: "Cards must alternate between red and black colors."
<NAME>alternating_colors</NAME>
<CODE>
if not mainline:
    return True
return card.color != mainline[-1].color
</CODE>

Rule: "Only cards with even ranks (2,4,6,8,10,12) are accepted."
<NAME>even_ranks_only</NAME>
<CODE>
return card.rank % 2 == 0
</CODE>

Rule: "Each card must have a rank greater than or equal to the previous card."
<NAME>rank_increasing</NAME>
<CODE>
if not mainline:
    return True
return card.rank >= mainline[-1].rank
</CODE>

Now convert the rule above. Provide ONLY the <NAME> and <CODE> tags, nothing else.
"""
