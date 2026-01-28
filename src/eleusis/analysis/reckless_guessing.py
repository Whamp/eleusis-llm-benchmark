"""Analysis of reckless guessing: patterns of consecutive wrong guesses."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_double_down_rate(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute rate of guessing again after a wrong guess.

    Returns DataFrame with columns: model, wrong_guesses, next_turn_guesses, double_down_rate
    """
    rows = []

    for model, model_turns in df_turns.groupby("model"):
        wrong_guesses = 0
        next_turn_guesses = 0

        for round_num, round_turns in model_turns.groupby("round_number"):
            turns = round_turns.sort_values("turn_number")
            turn_list = list(turns.itertuples())

            for i, turn in enumerate(turn_list):
                # Check if this turn had a wrong guess (non-shadow)
                if turn.guess_rule and turn.guess_correct == False and not turn.is_shadow:
                    wrong_guesses += 1

                    # Check if next turn exists and has a guess
                    if i + 1 < len(turn_list):
                        next_turn = turn_list[i + 1]
                        if next_turn.guess_rule and not next_turn.is_shadow:
                            next_turn_guesses += 1

        rate = next_turn_guesses / wrong_guesses if wrong_guesses > 0 else 0
        rows.append({
            "model": model,
            "wrong_guesses": wrong_guesses,
            "next_turn_guesses": next_turn_guesses,
            "double_down_rate": rate,
        })

    return pd.DataFrame(rows).sort_values("double_down_rate", ascending=False)


def compute_wrong_guess_streaks(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute streaks of consecutive wrong guesses per round.

    A streak requires consecutive turns with guesses. Streak ends when:
    - Correct guess
    - Turn without a guess (model steps back)
    - Round ends

    Returns DataFrame with columns: model, round_number, streak_length
    """
    rows = []

    for (model, round_num), round_turns in df_turns.groupby(["model", "round_number"]):
        turns = round_turns.sort_values("turn_number")

        current_streak = 0
        for turn in turns.itertuples():
            # Only count non-shadow guesses
            made_guess = turn.guess_rule and not turn.is_shadow

            if made_guess:
                if turn.guess_correct == False:
                    current_streak += 1
                else:
                    # Correct guess ends streak
                    if current_streak > 0:
                        rows.append({
                            "model": model,
                            "round_number": round_num,
                            "streak_length": current_streak,
                            "ended_by": "correct_guess",
                        })
                    current_streak = 0
            else:
                # No guess this turn - streak is broken (model stepped back)
                if current_streak > 0:
                    rows.append({
                        "model": model,
                        "round_number": round_num,
                        "streak_length": current_streak,
                        "ended_by": "stepped_back",
                    })
                current_streak = 0

        # If round ended with ongoing streak
        if current_streak > 0:
            rows.append({
                "model": model,
                "round_number": round_num,
                "streak_length": current_streak,
                "ended_by": "round_end",
            })

    return pd.DataFrame(rows)


def plot_reckless_guessing(
    df_double_down: pd.DataFrame,
    df_streaks: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate two-panel plot of reckless guessing behavior.

    Left: Double-down rate bar chart
    Right: Streak length box plot

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    model_metadata = load_model_metadata()

    # Sort models by double-down rate (most reckless first)
    models = df_double_down["model"].tolist()

    # Prepare JSON export data
    plot_data = {
        "models": [],
        "metadata": {
            "description": "Reckless guessing behavior: double-down rate and wrong guess streaks",
            "left_panel": "double_down_rate",
            "right_panel": "streak_length_distribution",
        }
    }

    # Left panel: Double-down rate horizontal bar chart
    colors = []
    is_open_list = []

    for model_name in models:
        colors.append(get_model_color(model_name, model_colors))
        normalized_name = normalize_model_name(model_name)
        is_open = False
        provider = "unknown"
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            if norm_key == normalized_name or norm_key in normalized_name or normalized_name in norm_key:
                is_open = meta["is_open"]
                provider = meta["provider"]
                break
        is_open_list.append(is_open)

        # Gather data for JSON
        model_row = df_double_down[df_double_down["model"] == model_name].iloc[0]
        model_streaks = df_streaks[df_streaks["model"] == model_name]["streak_length"]
        plot_data["models"].append({
            "name": model_name,
            "color": colors[-1],
            "is_open": is_open,
            "provider": provider,
            "double_down_rate": float(model_row["double_down_rate"]),
            "wrong_guesses": int(model_row["wrong_guesses"]),
            "next_turn_guesses": int(model_row["next_turn_guesses"]),
            "streaks": model_streaks.tolist(),
            "mean_streak": float(model_streaks.mean()) if len(model_streaks) > 0 else 0,
            "max_streak": int(model_streaks.max()) if len(model_streaks) > 0 else 0,
        })

    y_pos = range(len(models))
    rates = df_double_down["double_down_rate"].values

    # Draw bars with open/closed distinction
    for i, (model, rate, color, is_open) in enumerate(zip(models, rates, colors, is_open_list)):
        if is_open:
            ax1.barh(i, rate, color="white", edgecolor=color, linewidth=2, height=0.7)
        else:
            ax1.barh(i, rate, color=color, alpha=0.7, edgecolor=color, height=0.7)

        # Add percentage label
        ax1.text(rate + 0.02, i, f"{rate*100:.0f}%", va="center", fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9)
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Double-Down Rate", fontsize=11)
    ax1.set_title("After Wrong Guess: % Guessing Again Next Turn", fontsize=11, fontweight="bold")
    ax1.invert_yaxis()  # Most reckless at top

    # Add note
    ax1.text(
        0.02, -0.08,
        "Higher = more reckless (keeps guessing after failures)",
        transform=ax1.transAxes, fontsize=8, ha="left",
        style="italic", color="gray"
    )

    # Right panel: Mean streak length bar chart
    mean_streaks = []
    for model_name in models:
        model_streaks = df_streaks[df_streaks["model"] == model_name]["streak_length"]
        mean_streaks.append(model_streaks.mean() if len(model_streaks) > 0 else 0)

    # Draw bars with open/closed distinction
    for i, (model, mean_streak, color, is_open) in enumerate(zip(models, mean_streaks, colors, is_open_list)):
        if is_open:
            ax2.barh(i, mean_streak, color="white", edgecolor=color, linewidth=2, height=0.7)
        else:
            ax2.barh(i, mean_streak, color=color, alpha=0.7, edgecolor=color, height=0.7)

        # Add value label
        ax2.text(mean_streak + 0.2, i, f"{mean_streak:.1f}", va="center", fontsize=9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel("Mean Consecutive Wrong Guesses", fontsize=11)
    ax2.set_title("Average Wrong Guess Streak Length", fontsize=11, fontweight="bold")
    ax2.invert_yaxis()  # Match left panel order

    # Add note
    ax2.text(
        0.02, -0.08,
        "Longer = more persistent reckless guessing",
        transform=ax2.transAxes, fontsize=8, ha="left",
        style="italic", color="gray"
    )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, edgecolor="gray", label="Closed model"),
        Patch(facecolor="white", edgecolor="gray", linewidth=2, label="Open model"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()

    # Save outputs
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
):
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
    tee.write("Double-Down Rate: After a wrong guess, % of next turns with another guess\n")
    tee.write("(Only counts official guesses, not shadow/tentative guesses)\n\n")

    display_df = df_double_down.copy()
    display_df["double_down_rate"] = (display_df["double_down_rate"] * 100).round(1)
    display_df.columns = ["Model", "Wrong Guesses", "Next Turn Guesses", "Double-Down %"]
    tee.write(display_df.to_string(index=False) + "\n\n")

    # Display streak statistics
    if not df_streaks.empty:
        tee.write("Wrong Guess Streak Statistics:\n")
        streak_stats = df_streaks.groupby("model").agg(
            num_streaks=("streak_length", "count"),
            mean_length=("streak_length", "mean"),
            max_length=("streak_length", "max"),
            total_wrong=("streak_length", "sum"),
        ).reset_index()
        streak_stats["mean_length"] = streak_stats["mean_length"].round(2)
        streak_stats.columns = ["Model", "Streaks", "Mean Length", "Max Length", "Total Wrong"]

        # Sort by same order as double_down
        streak_stats = streak_stats.set_index("Model").loc[df_double_down["model"]].reset_index()
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
