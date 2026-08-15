"""Analysis of reckless guessing: patterns of consecutive wrong guesses."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import (
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def _count_round_double_downs(turns: pd.DataFrame) -> tuple[int, int]:
    """Count wrong guesses and immediate follow-up guesses in one round."""
    turn_rows = turns.sort_values("turn_number").to_dict(orient="records")
    wrong_guesses = 0
    next_turn_guesses = 0
    for index, turn in enumerate(turn_rows):
        is_wrong_formal_guess = (
            turn.get("guess_rule")
            and not turn.get("guess_correct")
            and not turn.get("is_shadow")
        )
        if not is_wrong_formal_guess:
            continue
        wrong_guesses += 1
        if index + 1 >= len(turn_rows):
            continue
        next_turn = turn_rows[index + 1]
        if next_turn.get("guess_rule") and not next_turn.get("is_shadow"):
            next_turn_guesses += 1
    return wrong_guesses, next_turn_guesses


def compute_double_down_rate(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute rate of guessing again after a wrong guess.

    Only counts turns before the counting cutoff (where score could still be positive).

    Returns model, wrong-guesses, next-turn-guesses, and double-down-rate columns.
    """
    # Filter to turns that count for analysis (before cutoff)
    df_counting = df_turns[df_turns["counts_for_analysis"] == True]  # ruff: ignore[true-false-comparison]

    rows = []

    for model, model_turns in df_counting.groupby("model"):
        wrong_guesses = 0
        next_turn_guesses = 0

        for _round_num, round_turns in model_turns.groupby("round_number"):
            round_wrong, round_follow_ups = _count_round_double_downs(round_turns)
            wrong_guesses += round_wrong
            next_turn_guesses += round_follow_ups

        rate = next_turn_guesses / wrong_guesses if wrong_guesses > 0 else 0
        rows.append(
            {
                "model": model,
                "wrong_guesses": wrong_guesses,
                "next_turn_guesses": next_turn_guesses,
                "double_down_rate": rate,
            }
        )

    return pd.DataFrame(rows).sort_values("double_down_rate", ascending=False)


def compute_wrong_guess_streaks(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute streaks of consecutive wrong guesses per round.

    Only counts turns before the counting cutoff (where score could still be positive).

    A streak requires consecutive turns with guesses. Streak ends when:
    - Correct guess
    - Turn without a guess (model steps back)
    - Round ends or cutoff reached

    Returns DataFrame with columns: model, round_number, streak_length
    """
    # Filter to turns that count for analysis (before cutoff)
    df_counting = df_turns[df_turns["counts_for_analysis"] == True]  # ruff: ignore[true-false-comparison]

    rows = []

    for (model, round_num), round_turns in df_counting.groupby(
        ["model", "round_number"]
    ):
        turns = round_turns.sort_values("turn_number")

        current_streak = 0
        for turn in turns.itertuples():
            # Only count non-shadow guesses
            made_guess = turn.guess_rule and not turn.is_shadow

            if made_guess:
                if not turn.guess_correct:
                    current_streak += 1
                else:
                    # Correct guess ends streak
                    if current_streak > 0:
                        rows.append(
                            {
                                "model": model,
                                "round_number": round_num,
                                "streak_length": current_streak,
                                "ended_by": "correct_guess",
                            }
                        )
                    current_streak = 0
            else:
                # No guess this turn - streak is broken (model stepped back)
                if current_streak > 0:
                    rows.append(
                        {
                            "model": model,
                            "round_number": round_num,
                            "streak_length": current_streak,
                            "ended_by": "stepped_back",
                        }
                    )
                current_streak = 0

        # If counting period ended with ongoing streak
        if current_streak > 0:
            rows.append(
                {
                    "model": model,
                    "round_number": round_num,
                    "streak_length": current_streak,
                    "ended_by": "cutoff_or_round_end",
                }
            )

    return pd.DataFrame(rows)


def plot_reckless_guessing(
    df_double_down: pd.DataFrame,
    df_streaks: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate bar chart of double-down rate.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax1 = plt.subplots(figsize=(10, 7))

    model_metadata = load_model_metadata()

    # Sort models by double-down rate (most reckless first)
    models = df_double_down["model"].tolist()

    # Prepare JSON export data
    plot_data = {
        "models": [],
        "metadata": {
            "description": (
                "Reckless guessing behavior: double-down rate and wrong guess streaks"
            ),
            "left_panel": "double_down_rate",
            "right_panel": "streak_length_distribution",
        },
    }

    # Left panel: Double-down rate horizontal bar chart
    colors = []
    is_open_list = []

    for model_name in models:
        colors.append(get_model_color(model_name, model_colors))
        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open = metadata["is_open"]
        provider = metadata["provider"]
        is_open_list.append(is_open)

        # Gather data for JSON
        model_row = df_double_down[df_double_down["model"] == model_name].iloc[0]
        model_streaks = df_streaks[df_streaks["model"] == model_name]["streak_length"]
        plot_data["models"].append(
            {
                "name": model_name,
                "color": colors[-1],
                "is_open": is_open,
                "provider": provider,
                "double_down_rate": float(model_row["double_down_rate"]),
                "wrong_guesses": int(model_row["wrong_guesses"]),
                "next_turn_guesses": int(model_row["next_turn_guesses"]),
                "streaks": model_streaks.tolist(),
                "mean_streak": (
                    float(model_streaks.mean()) if len(model_streaks) > 0 else 0
                ),
                "max_streak": int(model_streaks.max()) if len(model_streaks) > 0 else 0,
            }
        )

    y_pos = range(len(models))
    rates = df_double_down["double_down_rate"].values

    # Draw bars with open/closed distinction
    for i, (_model, rate, color, is_open) in enumerate(
        zip(models, rates, colors, is_open_list, strict=False)
    ):
        if is_open:
            ax1.barh(i, rate, color="white", edgecolor=color, linewidth=2, height=0.7)
        else:
            ax1.barh(i, rate, color=color, alpha=0.7, edgecolor=color, height=0.7)

        # Add percentage label
        ax1.text(rate + 0.02, i, f"{rate * 100:.0f}%", va="center", fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Double-Down Rate", fontsize=11)
    ax1.set_title(
        "After Wrong Guess: % Guessing Again Next Turn", fontsize=11, fontweight="bold"
    )
    ax1.invert_yaxis()  # Most reckless at top

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, edgecolor="gray", label="Closed model"),
        Patch(facecolor="white", edgecolor="gray", linewidth=2, label="Open model"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add note
    ax1.text(
        0.02,
        -0.08,
        "Higher = more reckless (keeps guessing after failures)",
        transform=ax1.transAxes,
        fontsize=8,
        ha="left",
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    png_path = output_folder / "reckless_guessing.png"
    json_path = output_folder / "reckless_guessing.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_reckless_guessing(
    df_turns: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
) -> None:
    """Run reckless guessing analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("RECKLESS GUESSING ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Compute metrics
    df_double_down = compute_double_down_rate(df_turns)
    df_streaks = compute_wrong_guess_streaks(df_turns)

    if df_double_down.empty or df_double_down["wrong_guesses"].sum() == 0:
        tee.write("No wrong guesses found in data.\n")
        return

    # Display double-down rate
    tee.write(
        "Double-Down Rate: After a wrong guess, % of next turns with another guess\n"
    )
    tee.write("(Only counts official guesses, not shadow/tentative guesses)\n\n")

    display_df = df_double_down.copy()
    display_df["double_down_rate"] = (display_df["double_down_rate"] * 100).round(1)
    display_df.columns = [
        "Model",
        "Wrong Guesses",
        "Next Turn Guesses",
        "Double-Down %",
    ]
    tee.write(display_df.to_string(index=False) + "\n\n")

    # Display streak statistics
    if not df_streaks.empty:
        tee.write("Wrong Guess Streak Statistics:\n")
        streak_stats = (
            df_streaks.groupby("model")
            .agg(
                num_streaks=("streak_length", "count"),
                mean_length=("streak_length", "mean"),
                max_length=("streak_length", "max"),
                total_wrong=("streak_length", "sum"),
            )
            .reset_index()
        )
        streak_stats["mean_length"] = streak_stats["mean_length"].round(2)
        streak_stats.columns = [
            "Model",
            "Streaks",
            "Mean Length",
            "Max Length",
            "Total Wrong",
        ]

        # Sort by same order as double_down
        streak_stats = (
            streak_stats.set_index("Model").loc[df_double_down["model"]].reset_index()
        )
        tee.write(streak_stats.to_string(index=False) + "\n\n")

        # Highlight worst streaks
        max_streak = df_streaks["streak_length"].max()
        worst = df_streaks[df_streaks["streak_length"] == max_streak]
        tee.write(f"Longest streak: {max_streak} consecutive wrong guesses\n")
        for _, row in worst.iterrows():
            tee.write(f"  - {row['model']} in round {row['round_number']}\n")
        tee.write("\n")

    # Generate plot
    png_path, json_path = plot_reckless_guessing(
        df_double_down, df_streaks, model_colors, output_folder
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
