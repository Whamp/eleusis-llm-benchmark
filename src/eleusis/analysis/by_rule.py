"""By-rule analysis showing score distribution across models."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_color_map
from .utils import TeeWriter, setup_matplotlib_style

logger = logging.getLogger(__name__)


def plot_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_path: Path,
):
    """Generate scatter plot showing scores per rule, colored by model."""
    setup_matplotlib_style()

    # Get unique rules and models
    rules = df_rounds["rule_description"].unique()
    models = df_rounds["model"].unique()
    color_map = get_color_map(models.tolist(), model_colors)

    # Create rule index mapping (sorted by mean score descending)
    rule_means = df_rounds.groupby("rule_description")["score"].mean().sort_values(ascending=False)
    rule_order = {rule: i for i, rule in enumerate(rule_means.index)}

    # Figure size based on number of rules
    fig_height = max(6, len(rules) * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Plot each model's scores
    for model in models:
        model_data = df_rounds[df_rounds["model"] == model]
        y_positions = model_data["rule_description"].map(rule_order)

        # Add small jitter to avoid overlapping points
        jitter = np.random.uniform(-0.15, 0.15, len(model_data))

        ax.scatter(
            model_data["score"],
            y_positions + jitter,
            c=color_map[model],
            s=60,
            alpha=0.7,
            label=model,
            edgecolors="white",
            linewidths=0.5,
        )

    # Configure axes
    ax.set_yticks(range(len(rule_means)))
    ax.set_yticklabels(rule_means.index, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_ylabel("Rule")
    ax.set_title("Score by Rule (colored by model)")

    # Add vertical line at score=0
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Legend outside plot
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    # Invert y-axis so highest scoring rules are at top
    ax.invert_yaxis()

    # Save with bbox_inches="tight" to handle long labels
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {output_path}")


def analyze_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
    """Run by-rule analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BY-RULE ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Summary stats per rule
    rule_stats = df_rounds.groupby("rule_description").agg(
        count=("score", "count"),
        avg_score=("score", "mean"),
        std_score=("score", "std"),
        success_rate=("success", "mean"),
    ).sort_values("avg_score", ascending=False).reset_index()

    tee.write("Score by rule (sorted by avg_score):\n")
    tee.write(rule_stats.to_string(index=False) + "\n\n")

    # Generate plot
    plot_path = output_folder / "by_rule.png"
    plot_by_rule(df_rounds, model_colors, plot_path)
    tee.write(f"Saved: {plot_path}\n")
