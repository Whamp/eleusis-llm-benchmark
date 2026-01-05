"""Prompt templates for LLM interactions in Eleusis."""

from pathlib import Path
import yaml

__all__ = [
    "get_continuation_prompt",
    "get_rule_compilation_prompt",
    "get_library_generation_prompt",
    "get_card_evaluation_prompt",
    "get_solo_rules",
    "get_solo_action_selection_prompt",
]


def _load_game_config() -> dict:
    """Load game config from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("game")


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


# ================================================================================================
# Solo Mode Prompts
# ================================================================================================


def get_solo_rules() -> str:
    """Generate rules for solo pattern discovery game."""
    game_config = _load_game_config()

    hand_size = game_config.get('hand_size', 12)
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)
    max_turns = game_config.get('max_turns', 40)

    return f"""=== PATTERN DISCOVERY CARD GAME RULES ===

## Overview

This is a single-player pattern discovery card game. A hidden rule determines which cards
are accepted or rejected. Your goal is to discover the rule as efficiently as possible.

## Components

- **Cards:** 2 standard 52-card decks shuffled together (104 cards total)
    - Ranks: Ace = 1 (low), 2–10, Jack = 11, Queen = 12, King = 13
    - Suits: Hearts ♥️ (red), Diamonds ♦️ (red), Clubs ♣️ (black), Spades ♠️ (black)

## Layout

The playing area consists of:

- **Mainline:** A horizontal row of accepted cards, ordered left-to-right by time of acceptance
- **Sidelines:** Vertical columns beneath mainline cards. When a card is rejected, it is
  placed in a column below the mainline card it was played after

The entire layout (mainline and all sidelines) is visible at all times.

---

## Setup

### Secret Rule

A deterministic rule decides whether a newly played card is **accepted** or **rejected**.

**The rule:**
- Depends only on information visible in the mainline: the candidate card and/or any
  previously accepted mainline cards (their suits, colors, ranks, parity, positions, etc.)
- For the first card played, evaluates based solely on that card's properties (since
  no mainline exists yet)
- Gives a unique, unambiguous answer (accepted/rejected) for every possible card in every
  possible mainline state

**The rule does NOT:**
- Reference sideline cards (rejected cards)
- Depend on hidden information (cards in the deck, cards in your hand)
- Include randomness or subjective judgment

Examples of rules:
- "The card must be a different color than the last mainline card." (Alternating red/black)
- "The card must be a heart or a spade."
- "The card must share either the suit or the color with the last mainline card, but not both."
- "The card's rank must have a different parity (odd/even) than the last mainline card."
- "The card's rank must differ from the last mainline card's rank by exactly 1 or 2."
- "If the last mainline card is red, play a card with rank ≤ 7. If black, play a card with rank ≥ 7."

### Initial Deal

You start with **{hand_size} cards** in your hand. This hand size remains constant throughout
the game - you always draw 1 card after playing.

A starter card that satisfies the rule is placed on the mainline to begin the game.

---

## Turn Structure

On each turn, you must:
1. **Play a card** from your hand
2. Receive feedback (accepted or rejected)
3. **Draw 1 card** from the deck (maintaining constant hand size)
4. Optionally **guess the rule** (see below)

### Playing a Card

1. Select one card from your hand to play
2. Place it to the right of the current last mainline card
3. Receive the judgment:

   **If Accepted:**
    - The card remains in place as the new last mainline card
    - You draw 1 card from the deck

   **If Rejected:**
    - The card is moved to a sideline column directly below the last mainline card
    - You draw 1 card from the deck

Your hand size stays constant at {hand_size} cards (unless the deck runs out).

---

## Guessing the Rule

At any point during your turn (after playing a card), you may attempt to state the rule.

The referee judges whether your stated rule is **equivalent** to the secret rule. Two
rules are equivalent if and only if they produce identical accepted/rejected judgments
for every possible (card, mainline-state) pair.

Note: The wording does not need to match exactly. Only logical equivalence matters.

- **Correct guess:** The game ends immediately - you win!
- **Incorrect guess:** The guess is recorded as wrong. A **penalty of {wrong_guess_penalty} points**
  is deducted from your final score. Play continues.

---

## End of Game

The game ends when:
1. You **correctly guess the rule**, or
2. You reach the **maximum of {max_turns} turns** without guessing correctly

---

## Scoring

Your score is calculated as follows:

    Score = {max_turns} - turn_number - ({wrong_guess_penalty} × number_of_failed_guesses)

Score can be negative.
In particular if you didn't guess correctly by turn {max_turns}, you get - ({wrong_guess_penalty} × number_of_failed_guesses)

**Higher scores are better.**
The goal is to discover the rule quickly with few failed guesses.
=== END OF GAME RULES ===
"""


def get_solo_action_selection_prompt(
    compact_board: str,
    hand_cards: list[dict],
    deck_remaining: int,
    play_history: list[dict],
    failed_guesses: list[dict] | None = None,
    current_turn: int = 1,
    max_turns: int = 40,
    failed_guess_count: int = 0,
) -> str:
    """Generate prompt for LLM to select a move in solo mode."""
    hand_str = ", ".join([c["symbol"] for c in hand_cards])

    # Calculate current score
    game_config = _load_game_config()
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)
    current_score = max_turns - current_turn - (wrong_guess_penalty * failed_guess_count)

    # Format play history
    history_str = ""
    if play_history:
        history_str = "\n\nYOUR PREVIOUS ATTEMPTS:\n"
        for entry in play_history[-15:]:  # Show last 15 attempts
            card = entry.get("card", "N/A")
            reasoning_summary = entry.get("reasoning_summary", "")
            accepted = entry.get("accepted")

            if accepted is not None:
                result = "✓ ACCEPTED" if accepted else "✗ REJECTED"
                history_str += f"- {card}: {result}\n"
                if reasoning_summary:
                    history_str += f"  Your reasoning: {reasoning_summary}\n"

    # Format failed guesses history
    failed_guesses_str = ""
    if failed_guesses:
        failed_guesses_str = "\n\nYOUR FAILED RULE GUESSES:\n"
        for entry in failed_guesses:
            guess = entry.get("guess", "")
            failed_guesses_str += f"- \"{guess}\"\n"
        failed_guesses_str += "\nALL OF THESE GUESSES WERE INCORRECT. DO NOT REPEAT THEM.\n"

    return f"""
**YOU ARE PLAYING A PATTERN DISCOVERY CARD GAME**

{get_solo_rules()}

=== YOUR TASK: CHOOSE YOUR NEXT ACTION ===

You are trying to discover the hidden rule that determines which cards are accepted.

CURRENT TURN: {current_turn} / {max_turns}
CURRENT PENALTY ({failed_guess_count} wrong guesses): {wrong_guess_penalty * failed_guess_count}
CURRENT SCORE: {max_turns} - {current_turn} - ({wrong_guess_penalty} × {failed_guess_count}) = {current_score}

CURRENT BOARD (mainline + rejected cards in brackets):
{compact_board}

YOUR HAND: {hand_str}

DECK REMAINING: {deck_remaining} cards

YOUR PLAY HISTORY:
{history_str}

YOUR PREVIOUS FAILED RULE GUESSES, IF ANY:
{failed_guesses_str}

---

YOUR ACTION:

You must select a card from your hand to play. After playing, you can optionally guess the rule.

OUTPUT FORMAT:
You can freely reason step by step about the pattern you observe.
Then your response should end with your final decision wrapped in XML tags.

<your reasoning about the pattern>

<ACTION>
{{
    "reasoning_summary": "Brief summary of your analysis and why you're playing this card",
    "card": "5♥" (the card symbol from your hand),
    "tentative_rule": "Your current best guess about the rule (always provide this, must be unequivocal)",
    "confidence_level": 0-10 (your confidence in the tentative_rule, 0=no clue, 10=certain),
    "guess_rule": true or false (whether to officially guess the rule this turn)
}}
</ACTION>

IMPORTANT GUIDANCE:

1. **Card Selection**: You MUST select a card from your hand. The value should be the exact
   symbol (e.g., "5♥", "K♠", "A♦").

2. **Tentative Rule**: ALWAYS provide your current hypothesis about the rule, even if you're
   uncertain. This must be clear and unequivocal (no "maybe" or "possibly"). If you set guess_rule to false,
   this guess will NOT be evaluated - it's just for your own tracking.

3. **Confidence Level**: Rate 0-10 how confident you are (0=completely uncertain, 10=absolutely sure).

4. **Guessing**: Set "guess_rule" to true ONLY when you're confident enough to officially guess.
   - If CORRECT: You win immediately!
   - If INCORRECT: You lose {wrong_guess_penalty} points from your final score
   - Consider the trade-off: guessing early but wrong is costly, guessing late reduces your score

Example:
<ACTION>
{{
    "reasoning_summary": "I see red and black cards alternating. My 3♥ is red, last card was black.",
    "card": "3♥",
    "tentative_rule": "Cards must alternate between red and black colors",
    "confidence_level": 8,
    "guess_rule": false
}}
</ACTION>
"""
