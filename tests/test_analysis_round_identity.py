"""Analysis rounds from different run folders must stay distinct."""

from __future__ import annotations

from typing import Any

from eleusis.analysis.loader import build_rounds_dataframe, build_turns_dataframe
from eleusis.analysis.no_stakes import (
    compute_first_correct_turn,
    compute_no_stakes_scores,
)


def _turn(
    number: int,
    *,
    guess: bool = False,
    correct: bool | None = None,
    shadow: bool = False,
) -> dict[str, Any]:
    """Build one legacy analysis turn dict."""
    turn: dict[str, Any] = {
        "turn_number": number,
        "llm_response": {
            "confidence_level": 5,
            "guess_rule": guess,
            "tentative_rule": "even ranks" if guess else "",
            "reasoning_summary": "test reasoning",
        },
        "action_result": {"accepted": True},
        "tokens": {"output_tokens": 10},
    }
    if guess:
        turn["guess_attempt"] = {
            "shadow": shadow,
            "correct": correct,
            "node_count": 7,
            "cyclomatic_complexity": 2,
        }
    return turn


def _run_doc(folder: str, rule: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one legacy result document for a single-round run folder."""
    return {
        "_folder": folder,
        "config": {
            "player": "model-x",
            "player_model": "model-x",
            "game": {"max_turns": 30, "wrong_guess_penalty": 2},
        },
        "rounds": [
            {
                "round_number": 1,
                "success": True,
                "score": 26.0,
                "turn_count": len(turns),
                "failed_guesses": 0,
                "game_over_reason": "correct",
                "rule_description": rule,
                "llm_usage": {
                    "player": {
                        "output_tokens": 10 * len(turns),
                        "reasoning_tokens": 0,
                        "answer_tokens": 10 * len(turns),
                    }
                },
                "turns": turns,
            }
        ],
    }


def test_first_correct_and_no_stakes_stay_per_run() -> None:
    """Two folders sharing model and round number must not merge rounds.

    Parallel workers each write their own run folder with per-worker
    round_number restarting at 1. Grouping analyses by (model, round_number)
    collapses every worker into one round, so the fastest worker's first
    correct turn contaminates all others' no-stakes scores.
    """
    fast_win = _run_doc(
        "run_fast",
        "rule A",
        [
            _turn(1),
            _turn(2),
            _turn(3),
            _turn(4),
            _turn(5, guess=True, correct=True),
        ],
    )
    slow_win = _run_doc(
        "run_slow",
        "rule B",
        [
            _turn(1),
            _turn(2),
            _turn(3, guess=True, correct=False),
            _turn(4),
            _turn(5),
            _turn(6),
            _turn(7),
            _turn(8),
            _turn(9),
            _turn(10, guess=True, correct=True),
        ],
    )

    results = [fast_win, slow_win]
    df_turns = build_turns_dataframe(results)
    df_rounds = build_rounds_dataframe(results, {})

    first_correct = compute_first_correct_turn(df_turns)
    assert len(first_correct) == 2
    by_run = dict(
        zip(first_correct["run"], first_correct["first_correct_turn"], strict=True)
    )
    assert by_run == {"run_fast": 5, "run_slow": 10}

    no_stakes = compute_no_stakes_scores(df_rounds, first_correct)
    scores = dict(zip(no_stakes["run"], no_stakes["no_stakes_score"], strict=True))
    # max_turns 30: fast = 30 - 5 + 1 = 26; slow = 30 - 10 + 1 = 21.
    assert scores == {"run_fast": 26.0, "run_slow": 21.0}
