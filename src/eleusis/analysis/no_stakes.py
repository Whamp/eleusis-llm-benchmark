"""No-stakes score analysis: what if guessing was systematic and wrong guesses had no penalty."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_first_correct_turn(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Find first turn with correct guess (shadow or official) per round.

    Returns DataFrame with columns: model, round_number, first_correct_turn
    """
    # Filter to only turns with a correct guess
    correct_guesses = df_turns[df_turns["guess_correct"].fillna(False)].copy()

    if correct_guesses.empty:
        return pd.DataFrame(columns=["model", "round_number", "first_correct_turn"])

    # Find minimum turn number per (model, round_number)
    first_correct = (
        correct_guesses.groupby(["model", "round_number"])["turn_number"]
        .min()
        .reset_index()
    )
    first_correct.columns = ["model", "round_number", "first_correct_turn"]

    return first_correct


def compute_no_stakes_scores(
    df_rounds: pd.DataFrame, df_first_correct: pd.DataFrame
) -> pd.DataFrame:
    """Compute no-stakes score for each round.

    no_stakes = max_turns - (first_correct_turn - 1)
    The -1 accounts for turn_number being 1-indexed while scoring uses 0-indexed turn_count.
    This makes: no_stakes = score + 2*failed_guesses + early_correct_turns (for successful rounds)
    """
    # Merge rounds with first correct turn
    df = df_rounds.merge(
        df_first_correct, on=["model", "round_number"], how="left"
    )

    # Compute no-stakes score
    # turn_number is 1-indexed, but scoring uses turn_count which is turn_number-1
    # So: no_stakes = max_turns - (first_correct_turn - 1) = max_turns - first_correct_turn + 1
    # If no correct turn found, no_stakes = 0
    df["no_stakes_score"] = df.apply(
        lambda row: row["max_turns"] - row["first_correct_turn"] + 1
        if pd.notna(row["first_correct_turn"])
        else 0,
        axis=1,
    )

    # Compute improvement over actual score
    df["score_improvement"] = df["no_stakes_score"] - df["score"]

    return df


def compute_no_stakes_stats(df_no_stakes: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model aggregated stats."""
    stats = (
        df_no_stakes.groupby("model")
        .agg(
            rounds=("round_number", "count"),
            avg_score=("score", "mean"),
            avg_no_stakes_score=("no_stakes_score", "mean"),
            avg_improvement=("score_improvement", "mean"),
            total_failed_guesses=("failed_guesses", "sum"),
        )
        .reset_index()
    )

    # Sort by avg_no_stakes_score descending
    stats = stats.sort_values("avg_no_stakes_score", ascending=False)

    return stats


def plot_no_stakes_comparison(
    df_stats: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Grouped bar chart: actual vs no-stakes score per model."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    model_metadata = load_model_metadata()

    # Sort models by no-stakes score
    models = df_stats["model"].tolist()
    x = range(len(models))
    bar_width = 0.35

    # Collect data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "description": (
                "No-stakes analysis: scores if guessing was systematic "
                "and wrong guesses had no penalty"
            ),
            "y_axis": "average_score",
        },
    }

    actual_scores = df_stats["avg_score"].tolist()
    no_stakes_scores = df_stats["avg_no_stakes_score"].tolist()
    colors = []
    is_open_list = []

    for model_name in models:
        color = get_model_color(model_name, model_colors)
        colors.append(color)

        # Determine if open model
        is_open = False
        provider = "unknown"
        normalized_name = normalize_model_name(model_name)
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            name_match = (
                norm_key == normalized_name
                or norm_key in normalized_name
                or normalized_name in norm_key
            )
            if name_match:
                is_open = meta["is_open"]
                provider = meta["provider"]
                break
        is_open_list.append(is_open)

        model_row = df_stats[df_stats["model"] == model_name].iloc[0]
        plot_data["models"].append({
            "name": model_name,
            "color": color,
            "is_open": is_open,
            "provider": provider,
            "avg_score": float(model_row["avg_score"]),
            "avg_no_stakes_score": float(model_row["avg_no_stakes_score"]),
            "avg_improvement": float(model_row["avg_improvement"]),
            "rounds": int(model_row["rounds"]),
            "total_failed_guesses": int(model_row["total_failed_guesses"]),
        })

    # Draw bars
    bars_actual = ax.bar(
        [xi - bar_width / 2 for xi in x],
        actual_scores,
        bar_width,
        label="Actual Score",
        alpha=0.8,
    )
    bars_no_stakes = ax.bar(
        [xi + bar_width / 2 for xi in x],
        no_stakes_scores,
        bar_width,
        label="No-Stakes Score",
        alpha=0.8,
    )

    # Color bars by model
    for i, (bar_a, bar_ns, color, is_open) in enumerate(
        zip(bars_actual, bars_no_stakes, colors, is_open_list)
    ):
        if is_open:
            bar_a.set_facecolor("white")
            bar_a.set_edgecolor(color)
            bar_a.set_linewidth(2)
            bar_ns.set_facecolor("white")
            bar_ns.set_edgecolor(color)
            bar_ns.set_linewidth(2)
            bar_ns.set_hatch("///")
        else:
            bar_a.set_facecolor(color)
            bar_ns.set_facecolor(color)
            bar_ns.set_alpha(0.5)

    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(
        "Actual vs No-Stakes Score\n(No penalty for wrong guesses, score at first correct answer)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)

    # Add improvement annotation
    ax.text(
        0.02, 0.98,
        "No-stakes = score + 2*failed_guesses + early_correct (systematic guessing, no penalty)",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
        style="italic", color="gray",
    )

    plt.tight_layout()

    # Save outputs
    png_path = output_folder / "no_stakes.png"
    json_path = output_folder / "no_stakes.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_no_stakes(
    df_rounds: pd.DataFrame,
    df_turns: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
    """Main entry point for no-stakes analysis."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("NO STAKES ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    tee.write("No-stakes scoring: what if guessing was systematic ")
    tee.write("and wrong guesses had no penalty?\n")
    tee.write("  For successful rounds: no_stakes = score + 2*failed_guesses + early_correct\n")
    tee.write("  For unsuccessful rounds with correct shadow: score at first correct turn\n")
    tee.write("  Rounds with no correct answer get score 0\n\n")

    # Find first correct turn per round
    df_first_correct = compute_first_correct_turn(df_turns)

    if df_first_correct.empty:
        tee.write("No correct guesses found in any rounds.\n")
        return

    # Compute no-stakes scores
    df_no_stakes = compute_no_stakes_scores(df_rounds, df_first_correct)

    # Compute per-model stats
    stats = compute_no_stakes_stats(df_no_stakes)

    # Display stats
    display_cols = ["model", "rounds", "avg_score", "avg_no_stakes_score", "avg_improvement"]
    stats_display = stats[display_cols].copy()
    stats_display.columns = ["Model", "Rounds", "Actual", "No-Stakes", "Improvement"]
    stats_display["Actual"] = stats_display["Actual"].round(2)
    stats_display["No-Stakes"] = stats_display["No-Stakes"].round(2)
    stats_display["Improvement"] = stats_display["Improvement"].round(2)

    tee.write(stats_display.to_string(index=False) + "\n\n")

    # Overall summary
    overall_actual = df_no_stakes["score"].mean()
    overall_no_stakes = df_no_stakes["no_stakes_score"].mean()
    overall_improvement = df_no_stakes["score_improvement"].mean()
    rounds_improved = (df_no_stakes["score_improvement"] > 0).sum()
    total_rounds = len(df_no_stakes)

    tee.write(f"Overall: Actual avg {overall_actual:.2f}, No-stakes avg {overall_no_stakes:.2f}\n")
    tee.write(f"Average improvement: +{overall_improvement:.2f} points per round\n")
    pct_improved = 100 * rounds_improved / total_rounds
    tee.write(
        f"{rounds_improved}/{total_rounds} ({pct_improved:.1f}%) "
        "rounds would have higher score\n\n"
    )

    # Generate plot
    png_path, json_path = plot_no_stakes_comparison(stats, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
