"""Action selection prompt for Eleusis."""

from eleusis.prompts.game_rules import get_game_rules

__all__ = ["get_action_prompt"]


def get_action_prompt(
    compact_board: str,
    hand_cards: list[dict],
    play_history: list[dict],
    failed_guesses: list[dict] | None,
    current_turn: int,
    max_turns: int,
    failed_guess_count: int,
    hand_size: int,
    wrong_guess_penalty: int,
) -> str:
    """Generate prompt for LLM to select a move."""
    hand_str = ", ".join([c["symbol"] for c in hand_cards])

    # Format play history
    history_str = ""
    if play_history:
        history_str = "Your last 3 turns (the card you played, the outcome and your reasoning summary):\n"
        for entry in play_history[-3:]:
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
        failed_guesses_str = "\n"
        for entry in failed_guesses:
            guess = entry.get("guess", "")
            failed_guesses_str += f"- \"{guess}\"\n"
        failed_guesses_str += "\nAll these guesses were incorrect.\n"

    return f"""# PATTERN DISCOVERY CARD GAME

You are playing a single-player card game where the goal is to discover a secret rule that determines which cards are accepted or rejected.
This is your turn to play. Your task is to select a card from your hand to play, and optionally try to guess the secret rule.
Your score will depend on how many turns it takes you to correctly identify the rule.
Below you will find the rules of the game, the current game state, your play history, and a description of what you are expected to do.

{get_game_rules(hand_size=hand_size, wrong_guess_penalty=wrong_guess_penalty, max_turns=max_turns)}

## YOUR TASK: CHOOSE YOUR NEXT ACTION

As a player, this is your turn to play and you must simultaneously:
1. Select a card from your hand to play.
2. Optionally, make a guess about the hidden rule.

### Current game state

Turn: {current_turn} / {max_turns}
Penalty so far for {failed_guess_count} wrong guesses: {wrong_guess_penalty * failed_guess_count}
Current potential score if you guess the rule correctly this turn: {max_turns - current_turn - (wrong_guess_penalty * failed_guess_count)}

#### Mainline & Sidelines

This is the mainline and sidelines so far. Rejected cards are shown in brackets after the mainline card they were played after :

{compact_board}

#### Your hand

{hand_str}

#### Your recent play history

{history_str}

#### All your previous failed rule guesses (if any)
{failed_guesses_str}

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
{{
    "reasoning_summary": "One sentence summary of your analysis and why you're playing this card",
    "card": "5♥" (the card symbol from your hand, must be the exact symbol e.g., "5♥", "K♠", "A♦",...)
    "tentative_rule": "Your current best guess about the rule",
    "confidence_level": 0-10,
    "guess_rule": true or false (whether to officially try to guess the rule this turn)
}}
</ACTION>

Always provide your current best hypothesis as a tentative rule, even if you're uncertain. 
If you set guess_rule to false, this tentative rule will not be evaluated, it's just for your own tracking.
Set "guess_rule" to true only when you want to officially try to guess the rule.
   - If correct, you score and the round ends immediately.
   - If incorrect, you will lose {wrong_guess_penalty} points from your final score, and the round continues.

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
