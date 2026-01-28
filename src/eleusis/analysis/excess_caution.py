"""Analysis of excess caution: turns where model had correct answer but didn't guess."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_early_correct_turns(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Compute consecutive correct shadow guesses before winning turn for each successful round.

    Returns DataFrame with columns: model, round_number, early_correct_turns
    """
    rows = []

    # Only look at successful rounds
    successful_rounds = df_turns[df_turns["round_success"] == True]

    for (model, round_num), group in successful_rounds.groupby(["model", "round_number"]):
        # Sort by turn number
        turns = group.sort_values("turn_number")

        # Find the winning turn (official guess that was correct)
        winning_turns = turns[(turns["guess_rule"] == True) & (turns["guess_correct"] == True)]

        if winning_turns.empty:
            continue

        winning_turn_num = winning_turns["turn_number"].iloc[0]

        # Get turns before the winning turn, in reverse order
        prior_turns = turns[turns["turn_number"] < winning_turn_num].sort_values(
            "turn_number", ascending=False
        )

        # Count consecutive correct shadow guesses
        early_correct = 0
        for _, turn in prior_turns.iterrows():
            if turn["is_shadow"] == True and turn["guess_correct"] == True:
                early_correct += 1
            else:
                break  # Stop at first non-correct or non-shadow turn

        rows.append({
            "model": model,
            "round_number": round_num,
            "early_correct_turns": early_correct,
            "winning_turn": winning_turn_num,
        })

    return pd.DataFrame(rows)


def compute_excess_caution_stats(df_early: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model statistics for early correct turns."""
    if df_early.empty:
        return pd.DataFrame()

    stats = df_early.groupby("model").agg(
        successful_rounds=("early_correct_turns", "count"),
        mean_early_correct=("early_correct_turns", "mean"),
        median_early_correct=("early_correct_turns", "median"),
        max_early_correct=("early_correct_turns", "max"),
        total_early_correct=("early_correct_turns", "sum"),
    ).reset_index()

    # Compute proportion of rounds with any early correct turns
    rounds_with_early = df_early[df_early["early_correct_turns"] > 0].groupby("model").size()
    stats = stats.merge(
        rounds_with_early.rename("rounds_with_early").reset_index(),
        on="model", how="left"
    )
    stats["rounds_with_early"] = stats["rounds_with_early"].fillna(0).astype(int)
    stats["pct_rounds_with_early"] = stats["rounds_with_early"] / stats["successful_rounds"] * 100

    return stats.sort_values("mean_early_correct", ascending=False)


def plot_excess_caution(
    df_early: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate box plot of early correct turns per model.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Sort models by median early correct turns
    model_medians = df_early.groupby("model")["early_correct_turns"].median().sort_values(ascending=False)
    models = model_medians.index.tolist()

    # Prepare data for box plot and JSON
    box_data = []
    colors = []
    is_open_list = []
    plot_data = {
        "models": [],
        "metadata": {
            "description": "Early correct turns: consecutive shadow guesses that were correct before the winning guess",
            "y_axis": "early_correct_turns",
        }
    }

    for model_name in models:
        model_data = df_early[df_early["model"] == model_name]["early_correct_turns"]
        box_data.append(model_data.values)
        colors.append(get_model_color(model_name, model_colors))

        # Determine if open model
        is_open = False
        provider = "unknown"
        normalized_name = normalize_model_name(model_name)
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            if norm_key == normalized_name or norm_key in normalized_name or normalized_name in norm_key:
                is_open = meta["is_open"]
                provider = meta["provider"]
                break
        is_open_list.append(is_open)

        # Store data for JSON
        plot_data["models"].append({
            "name": model_name,
            "color": colors[-1],
            "is_open": is_open,
            "provider": provider,
            "values": model_data.tolist(),
            "mean": float(model_data.mean()),
            "median": float(model_data.median()),
            "std": float(model_data.std()) if len(model_data) > 1 else 0.0,
            "count": len(model_data),
        })

    # Create box plot
    bp = ax.boxplot(
        box_data,
        labels=models,
        patch_artist=True,
        widths=0.6,
    )

    # Style boxes with model colors and open/closed distinction
    for i, (patch, color, is_open) in enumerate(zip(bp["boxes"], colors, is_open_list)):
        if is_open:
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)

    # Style medians
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    # Style whiskers and caps
    for whisker in bp["whiskers"]:
        whisker.set_color("gray")
        whisker.set_linewidth(1.5)
    for cap in bp["caps"]:
        cap.set_color("gray")
        cap.set_linewidth(1.5)

    ax.set_ylabel("Early Correct Turns", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title("Excess Caution: Turns with Correct Answer Before Guessing", fontsize=13, fontweight="bold")

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=9)

    # Add legend for open/closed distinction
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, edgecolor="gray", label="Closed model"),
        Patch(facecolor="white", edgecolor="gray", linewidth=2, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Add note
    ax.text(
        0.02, 0.98,
        "Higher = more cautious (delayed guessing despite having correct answer)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        style="italic", color="gray"
    )

    plt.tight_layout()

    # Save outputs
    png_path = output_folder / "excess_caution.png"
    json_path = output_folder / "excess_caution.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def plot_caution_vs_failed_guesses(
    df_early: pd.DataFrame,
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate scatter plot of average caution vs average failed guesses.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Compute per-model metrics
    # Use counting_failed_guesses (only guesses before score was guaranteed <= 0)
    caution_by_model = df_early.groupby("model")["early_correct_turns"].mean()
    failed_by_model = df_rounds.groupby("model")["counting_failed_guesses"].mean()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "avg_failed_guesses",
            "y_axis": "avg_early_correct_turns",
            "description": "Trade-off between caution (delaying correct guesses) and recklessness (failed guesses)"
        }
    }

    for model_name in caution_by_model.index:
        if model_name not in failed_by_model.index:
            continue

        x = failed_by_model[model_name]
        y = caution_by_model[model_name]
        color = get_model_color(model_name, model_colors)

        # Determine if open model
        is_open = False
        provider = "unknown"
        normalized_name = normalize_model_name(model_name)
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            if norm_key == normalized_name or norm_key in normalized_name or normalized_name in norm_key:
                is_open = meta["is_open"]
                provider = meta["provider"]
                break

        # Plot point: open circle for open models, filled for closed
        if is_open:
            ax.scatter(x, y, c="none", edgecolors=color, s=150, linewidths=2.5, zorder=3)
        else:
            ax.scatter(x, y, c=color, s=150, alpha=0.9, zorder=3)

        # Add label
        ax.annotate(
            model_name, (x, y),
            xytext=(8, 4), textcoords="offset points", fontsize=9,
            ha="left", va="bottom"
        )

        # Store data for JSON
        plot_data["models"].append({
            "name": model_name,
            "avg_early_correct_turns": float(y),
            "avg_failed_guesses": float(x),
            "color": color,
            "is_open": is_open,
            "provider": provider,
        })

    ax.set_xlabel("Average Failed Guesses per Round", fontsize=11)
    ax.set_ylabel("Average Early Correct Turns", fontsize=11)
    ax.set_title("Caution vs Recklessness Trade-off", fontsize=13, fontweight="bold")

    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        0.02, 0.98, "Cautious\n(delays correct guesses)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        style="italic", color="gray"
    )
    ax.text(
        0.98, 0.02, "Reckless\n(many failed guesses)",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", color="gray"
    )

    # Add legend for open/closed distinction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Closed model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="gray",
               markeredgewidth=2, markersize=10, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Save outputs
    png_path = output_folder / "caution_vs_failed_guesses.png"
    json_path = output_folder / "caution_vs_failed_guesses.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_excess_caution(
    df_turns: pd.DataFrame,
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
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

    tee.write("Early Correct Turns: consecutive correct shadow guesses before winning guess\n")
    tee.write("(Only counts successful rounds where model eventually guessed correctly)\n\n")

    # Format stats for display
    display_cols = [
        "model", "successful_rounds", "mean_early_correct", "median_early_correct",
        "pct_rounds_with_early"
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

    tee.write(f"Overall: {total_early} total early correct turns across {total_rounds} successful rounds\n")
    tee.write(f"Mean: {overall_mean:.2f}, Median: {overall_median:.1f}\n")
    tee.write(f"{rounds_with_early}/{total_rounds} ({100*rounds_with_early/total_rounds:.1f}%) rounds had at least 1 early correct turn\n\n")

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
