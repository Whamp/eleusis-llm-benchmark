"""Prompt experiment harness — turn-prefix replay for A/B prompt comparison.

This module enables comparing different prompt templates by replaying the same
turn-prefix states from a reference benchmark run. Each prompt variant generates
a prompt from the same game state, and the LLM response is recorded.

IMPORTANT: This is turn-prefix comparison only. After the first turn where a
variant's chosen action differs from the reference, subsequent game states would
diverge in a real game. This harness does NOT simulate the diverged game — it
replays the *original* turn states regardless. Results are valid for comparing
prompt quality on identical inputs, not for predicting full-game outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

__all__ = [
    "TurnState",
    "extract_turn_states",
    "run_prompt_experiment",
]

logger = logging.getLogger(__name__)


@dataclass
class TurnState:
    """Immutable snapshot of the game state at the start of a turn.

    Extracted from a reference run's results. Contains everything needed
    to generate a prompt for this turn.
    """

    turn_number: int
    mainline_state: str
    hand: list[str]
    play_history: list[dict] = field(default_factory=list)
    failed_guesses: list[dict] | None = None
    failed_guess_count: int = 0
    max_turns: int = 40
    hand_size: int = 12
    wrong_guess_penalty: int = 2


def extract_turn_states(results_data: dict, round_index: int = 0) -> list[TurnState]:
    """Extract turn-prefix states from a results data structure.

    Args:
        results_data: Parsed results JSON (with 'rounds' and 'config' keys).
        round_index: Which round to extract turns from (0-indexed).

    Returns:
        List of TurnState objects, one per turn in the round.
    """
    round_data = results_data["rounds"][round_index]
    config = results_data.get("config", {})
    game_cfg = config.get("game", {})

    max_turns = game_cfg.get("max_turns", 40)
    hand_size = game_cfg.get("hand_size", 12)
    wrong_guess_penalty = game_cfg.get("wrong_guess_penalty", 2)

    turns = round_data["turns"]
    states = []
    play_history: list[dict] = []
    failed_guesses: list[dict] = []
    failed_guess_count = 0

    for turn in turns:
        state = TurnState(
            turn_number=turn["turn_number"],
            mainline_state=turn["mainline_state"],
            hand=list(turn["hand"]),
            play_history=list(play_history),
            failed_guesses=list(failed_guesses) if failed_guesses else None,
            failed_guess_count=failed_guess_count,
            max_turns=max_turns,
            hand_size=hand_size,
            wrong_guess_penalty=wrong_guess_penalty,
        )
        states.append(state)

        # Build history for subsequent turns from this turn's results
        action_result = turn.get("action_result", {})
        llm_response = turn.get("llm_response", {})
        if action_result.get("success") and "card" in action_result:
            play_history.append({
                "card": action_result["card"],
                "accepted": action_result.get("accepted", False),
                "reasoning_summary": llm_response.get("reasoning_summary", ""),
            })

        # Track failed guesses
        guess = turn.get("guess_attempt")
        if guess and not guess.get("correct", False) and not guess.get("shadow", False):
            failed_guesses.append({"guess": guess.get("guess", "")})
            failed_guess_count += 1

    return states


# Type alias for prompt variant functions
PromptVariantFn = Callable[[TurnState], str]

# Type alias for generate functions
GenerateFn = Callable[[str], dict]


def run_prompt_experiment(
    turn_states: list[TurnState],
    variants: dict[str, PromptVariantFn],
    generate_fn: GenerateFn,
) -> list[dict]:
    """Run prompt variants against turn-prefix states and record outputs.

    For each turn state, each variant function generates a prompt, which is
    passed to generate_fn. The prompt and response are recorded side by side.

    NOTE: This is turn-prefix comparison. After the first divergent action,
    real game states would differ. Results are for prompt quality comparison,
    not full-game prediction.

    Args:
        turn_states: List of TurnState snapshots from a reference run.
        variants: Dict mapping variant name to prompt-generating function.
        generate_fn: Function that takes a prompt string and returns a
            parsed action response dict.

    Returns:
        List of per-turn result dicts with variant outputs for comparison.
    """
    results = []

    for state in turn_states:
        turn_result = {
            "turn_number": state.turn_number,
            "mainline_state": state.mainline_state,
            "hand": state.hand,
            "variant_outputs": {},
        }

        for variant_name, variant_fn in variants.items():
            prompt = variant_fn(state)
            try:
                response = generate_fn(prompt)
                turn_result["variant_outputs"][variant_name] = {
                    "prompt": prompt,
                    "response": response,
                    "error": None,
                }
            except Exception as e:
                logger.warning(
                    "Variant %r failed on turn %d: %s",
                    variant_name, state.turn_number, e,
                )
                turn_result["variant_outputs"][variant_name] = {
                    "prompt": prompt,
                    "response": None,
                    "error": str(e),
                }

        results.append(turn_result)

    return results
