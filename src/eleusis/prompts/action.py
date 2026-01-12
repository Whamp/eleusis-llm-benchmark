"""Action selection prompt for Eleusis."""

from eleusis.prompts.game_rules import _load_game_config, get_game_rules

__all__ = ["get_action_prompt"]


def get_action_prompt(
    compact_board: str,
    hand_cards: list[dict],
    deck_remaining: int,
    play_history: list[dict],
    failed_guesses: list[dict] | None = None,
    current_turn: int = 1,
    max_turns: int = 40,
    failed_guess_count: int = 0,
) -> str:
    """Generate prompt for LLM to select a move."""
    hand_str = ", ".join([c["symbol"] for c in hand_cards])

    game_config = _load_game_config()
    wrong_guess_penalty = game_config.get('wrong_guess_penalty', 3)
    current_score = max_turns - current_turn - (wrong_guess_penalty * failed_guess_count)

    # Format play history
    history_str = ""
    if play_history:
        history_str = "\n\nYOUR PREVIOUS ATTEMPTS:\n"
        for entry in play_history[-15:]:
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

{get_game_rules()}

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
