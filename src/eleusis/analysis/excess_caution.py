"""Analysis of excess caution: turns where model had correct answer but didn't guess."""

import logging
from pathlib import Path

import pandas as pd

from .excess_caution_distribution import plot_excess_caution
from .excess_caution_relationships import (
    plot_caution_vs_failed_guesses,
    plot_score_vs_recklessness,
)
from .utils import TeeWriter

logger = logging.getLogger(__name__)


def compute_early_correct_turns(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-win correct-shadow streaks for successful rounds.

    Returns DataFrame with columns: model, round_number, early_correct_turns
    """
    rows = []

    # Only look at successful rounds
    successful_rounds = df_turns[df_turns["round_success"]]

    for (model, round_num), group in successful_rounds.groupby(
        ["model", "round_number"]
    ):
        # Sort by turn number
        turns = group.sort_values("turn_number")

        # Find the winning turn (official guess that was correct)
        winning_turns = turns[(turns["guess_rule"]) & (turns["guess_correct"])]

        if winning_turns.empty:
            continue

        winning_turn_num = winning_turns["turn_number"].iloc[0]

        prior_turns = turns[turns["turn_number"] < winning_turn_num].sort_values(
            "turn_number", ascending=False
        )

        # Count consecutive correct shadow guesses
        early_correct = 0
        for _, turn in prior_turns.iterrows():
            if turn["is_shadow"] and turn["guess_correct"]:
                early_correct += 1
            else:
                break  # Stop at first non-correct or non-shadow turn

        rows.append(
            {
                "model": model,
                "round_number": round_num,
                "early_correct_turns": early_correct,
                "winning_turn": winning_turn_num,
            }
        )

    return pd.DataFrame(rows)


def compute_excess_caution_stats(df_early: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model statistics for early correct turns."""
    if df_early.empty:
        return pd.DataFrame()

    stats = (
        df_early.groupby("model")
        .agg(
            successful_rounds=("early_correct_turns", "count"),
            mean_early_correct=("early_correct_turns", "mean"),
            median_early_correct=("early_correct_turns", "median"),
            max_early_correct=("early_correct_turns", "max"),
            total_early_correct=("early_correct_turns", "sum"),
        )
        .reset_index()
    )

    # Compute proportion of rounds with any early correct turns
    rounds_with_early = (
        df_early[df_early["early_correct_turns"] > 0].groupby("model").size()
    )
    stats = stats.merge(
        rounds_with_early.rename("rounds_with_early").reset_index(),
        on="model",
        how="left",
    )
    stats["rounds_with_early"] = stats["rounds_with_early"].fillna(0).astype(int)
    stats["pct_rounds_with_early"] = (
        stats["rounds_with_early"] / stats["successful_rounds"] * 100
    )

    return stats.sort_values("mean_early_correct", ascending=False)


def analyze_excess_caution(
    df_turns: pd.DataFrame,
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
) -> None:
    """Run excess caution analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("EXCESS CAUTION ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Compute early correct turns for each successful round
    df_early = compute_early_correct_turns(df_turns)

    if df_early.empty:
        tee.write("No successful rounds with shadow evaluations found.\n")
        return

    # Compute and display statistics
    stats = compute_excess_caution_stats(df_early)

    tee.write(
        "Early Correct Turns: consecutive correct shadow guesses before winning guess\n"
    )
    tee.write(
        "(Only counts successful rounds where model eventually guessed correctly)\n\n"
    )

    # Format stats for display
    display_cols = [
        "model",
        "successful_rounds",
        "mean_early_correct",
        "median_early_correct",
        "pct_rounds_with_early",
    ]
    stats_display = stats[display_cols].copy()
    stats_display.columns = ["Model", "Rounds", "Mean", "Median", "% with Early"]
    stats_display["Mean"] = stats_display["Mean"].round(2)
    stats_display["Median"] = stats_display["Median"].round(1)
    stats_display["% with Early"] = stats_display["% with Early"].round(1)

    tee.write(stats_display.to_string(index=False) + "\n\n")

    # Overall summary
    total_early = df_early["early_correct_turns"].sum()
    total_rounds = len(df_early)
    overall_mean = df_early["early_correct_turns"].mean()
    overall_median = df_early["early_correct_turns"].median()
    rounds_with_early = (df_early["early_correct_turns"] > 0).sum()

    tee.write(
        f"Overall: {total_early} total early correct turns across {total_rounds}"
        " successful rounds\n"
    )
    tee.write(f"Mean: {overall_mean:.2f}, Median: {overall_median:.1f}\n")
    tee.write(
        f"{rounds_with_early}/{total_rounds}"
        f" ({100 * rounds_with_early / total_rounds:.1f}%) rounds had at least 1 early"
        " correct turn\n\n"
    )

    # Generate box plot
    png_path, json_path = plot_excess_caution(df_early, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate caution vs failed guesses scatter plot
    png_path, json_path = plot_caution_vs_failed_guesses(
        df_early, df_rounds, model_colors, output_folder
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate score vs recklessness index scatter plot
    png_path, json_path = plot_score_vs_recklessness(
        df_early, df_rounds, model_colors, output_folder
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
