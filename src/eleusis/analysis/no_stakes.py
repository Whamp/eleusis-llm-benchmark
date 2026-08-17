"""No-stakes score computation utilities."""

import pandas as pd


def compute_first_correct_turn(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Find first turn with correct guess per round, split by formal vs shadow.

    Rounds are identified by model, run folder, and round number because
    parallel workers reuse round numbers across separate run folders.

    Returns DataFrame with columns: model, run, round_number,
        first_correct_turn, first_formal_correct_turn, first_shadow_correct_turn
    """
    correct_guesses = df_turns[df_turns["guess_correct"] == True].copy()  # ruff: ignore[true-false-comparison]

    if correct_guesses.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "model",
                    "run",
                    "round_number",
                    "first_correct_turn",
                    "first_formal_correct_turn",
                    "first_shadow_correct_turn",
                ]
            )
        )

    # Split by shadow vs formal
    shadow = correct_guesses[correct_guesses["is_shadow"] == True]  # ruff: ignore[true-false-comparison]
    formal = correct_guesses[correct_guesses["is_shadow"] != True]  # ruff: ignore[true-false-comparison]

    # First correct of any kind
    first_any = (
        correct_guesses.groupby(["model", "run", "round_number"])["turn_number"]
        .min()
        .reset_index()
        .rename(columns={"turn_number": "first_correct_turn"})
    )

    # First formal correct
    first_formal = (
        (
            formal.groupby(["model", "run", "round_number"])["turn_number"]
            .min()
            .reset_index()
            .rename(columns={"turn_number": "first_formal_correct_turn"})
        )
        if not formal.empty
        else pd.DataFrame(
            columns=pd.Index(
                ["model", "run", "round_number", "first_formal_correct_turn"]
            )
        )
    )

    # First shadow correct
    first_shadow = (
        (
            shadow.groupby(["model", "run", "round_number"])["turn_number"]
            .min()
            .reset_index()
            .rename(columns={"turn_number": "first_shadow_correct_turn"})
        )
        if not shadow.empty
        else pd.DataFrame(
            columns=pd.Index(
                ["model", "run", "round_number", "first_shadow_correct_turn"]
            )
        )
    )

    # Merge all three
    result = first_any
    result = result.merge(first_formal, on=["model", "run", "round_number"], how="left")
    result = result.merge(first_shadow, on=["model", "run", "round_number"], how="left")

    return result


def compute_no_stakes_scores(
    df_rounds: pd.DataFrame, df_first_correct: pd.DataFrame
) -> pd.DataFrame:
    """Compute no-stakes score for each round.

    no_stakes = max_turns - (first_correct_turn - 1)
    The -1 converts the 1-indexed turn number to the scoring turn count.
    For successful rounds, no-stakes score equals the score plus twice the failed
    guesses plus the early correct turns.
    """
    # Merge rounds with first correct turn
    df = df_rounds.merge(
        df_first_correct, on=["model", "run", "round_number"], how="left"
    )

    # Compute no-stakes score
    # turn_number is 1-indexed, but scoring uses turn_count which is turn_number-1
    # So: no_stakes = max_turns - (first_correct_turn - 1) = max_turns -
    # first_correct_turn + 1
    # If no correct turn found, no_stakes = 0
    df["no_stakes_score"] = df.apply(
        lambda row: (
            row["max_turns"] - row["first_correct_turn"] + 1
            if pd.notna(row["first_correct_turn"])
            else 0
        ),
        axis=1,
    )

    # Compute improvement over actual score
    df["score_improvement"] = df["no_stakes_score"] - df["score"]

    return df
