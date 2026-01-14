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
        history_str = "\n\nYOUR LAST 10 ATTEMPTS:\n"
        for entry in play_history[-10:]:
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
# PATTERN DISCOVERY CARD GAME

You are playing a single-player card game where your goal is to discover a hidden rule
that determines which cards are accepted or rejected.
This is your turn to play. Your task is to select a card from your hand to play, and optionally guess the hidden rule.

## RULES OF THE GAME

{get_game_rules()}

## YOUR TASK: CHOOSE YOUR ACTION

As a player, this is your turn to play and you must:
1. Select a card from your hand to play.
2. Optionally, make a guess about the hidden rule.

### CURRENT GAME STATE

Turn: {current_turn} / {max_turns}
Your current penalty: ({failed_guess_count} wrong guesses): {wrong_guess_penalty * failed_guess_count}
Your current score: {max_turns} - {current_turn} - ({wrong_guess_penalty} × {failed_guess_count}) = {current_score}
Remaining in the deck: {deck_remaining} cards

#### CURRENT BOARD
This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after.

{compact_board}

#### YOUR HAND

{hand_str}

#### YOUR PLAY HISTORY
{history_str}

#### YOUR PREVIOUS FAILED RULE GUESSES (IF ANY)
{failed_guesses_str}

#### YOUR ACTION:

You must select a card from your hand to play. After playing, you can optionally guess the rule.

### RESPONSE & OUTPUT FORMAT
You can freely reason step by step about the pattern you observe.
Then your response should end with your final decision wrapped in XML tags as shown below.

You will be asked to provide the following in your response:
- A reasoning summary explaining your thought process about the pattern so far and why you are playing the selected card.
- The card you are playing from your hand (must be one of the cards listed in your hand).
- Your current best guess about the hidden rule (must be clear and unequivocal).
- Your confidence level in your tentative rule on a scale of 0-10 (1="10% confident / 10% probability to be correct", 3="30% confident / 30% probability to be correct", 7="70% confident / 70% probability to be correct", 10="certain to be correct").
- Whether you want to officially guess the rule this turn (true or false). Only set to true if you are confident enough to make an official guess. If you set to false, your guess will not be evaluated this turn and will have no consequence.

Format your response as follows:
<ACTION>
{{
    "reasoning_summary": "Brief summary of your analysis and why you're playing this card",
    "card": "5♥" (the card symbol from your hand),
    "tentative_rule": "Your current best guess about the rule (always provide this, must be unequivocal)",
    "confidence_level": 0-10 (your confidence in the tentative_rule, 0="no clue", 3="30% confident", 7="70% confident", 10="certain"),
    "guess_rule": true or false (whether to officially guess the rule this turn)
}}
</ACTION>


#### IMPORTANT GUIDANCE

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

#### Example:
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
