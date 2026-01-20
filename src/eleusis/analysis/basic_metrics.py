"""Basic model comparison metrics and plots."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import get_color_map
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_basic_metrics(df_rounds: pd.DataFrame) -> pd.DataFrame:
    """Compute basic comparison metrics per model."""
    # Basic aggregations
    metrics = df_rounds.groupby("model").agg(
        rounds_played=("success", "count"),
        total_score=("score", "sum"),
        avg_score=("score", "mean"),
        total_turns=("turn_count", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        total_wall_clock=("wall_clock_seconds", "sum"),
        avg_failed_guesses=("failed_guesses", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()

    # Derived metrics
    metrics["avg_output_tokens_per_turn"] = metrics["total_output_tokens"] / metrics["total_turns"]
    metrics["wall_clock_per_turn"] = metrics["total_wall_clock"] / metrics["total_turns"]

    # Variance analysis: intra-rule vs inter-rule
    # Intra-rule variance: average of variance within same rule
    intra_var = df_rounds.groupby(["model", "rule_description"])["score"].var()
    intra_var_by_model = intra_var.groupby("model").mean()
    metrics = metrics.merge(
        intra_var_by_model.rename("intra_rule_variance").reset_index(),
        on="model", how="left"
    )

    # Inter-rule variance: variance of per-rule mean scores
    rule_means = df_rounds.groupby(["model", "rule_description"])["score"].mean()
    inter_var_by_model = rule_means.groupby("model").var()
    metrics = metrics.merge(
        inter_var_by_model.rename("inter_rule_variance").reset_index(),
        on="model", how="left"
    )

    # Variance ratio (higher = more rule-dependent, lower = more random/model-dependent)
    metrics["variance_ratio"] = metrics["intra_rule_variance"] / metrics["inter_rule_variance"]

    return metrics.sort_values("avg_score", ascending=False)


def plot_basic_metrics(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_path: Path
):
    """Generate basic metrics charts."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort by avg_score for consistent ordering
    metrics = metrics.sort_values("avg_score", ascending=True)  # ascending for horizontal bar
    color_map = get_color_map(metrics["model"].tolist(), model_colors)
    colors = [color_map[m] for m in metrics["model"]]

    # Chart 1: Horizontal bar - avg_score per model
    ax = axes[0]
    ax.barh(metrics["model"], metrics["avg_score"], color=colors)
    ax.set_xlabel("Average Score")
    ax.set_title("Average Score by Model")
    for i, (model, score) in enumerate(zip(metrics["model"], metrics["avg_score"])):
        ax.text(score + 0.5, i, f"{score:.1f}", va="center", fontsize=9)

    # Chart 2: Scatter - avg_score vs avg_output_tokens_per_turn
    ax = axes[1]
    for model, row in metrics.iterrows():
        ax.scatter(
            metrics.loc[model, "avg_output_tokens_per_turn"],
            metrics.loc[model, "avg_score"],
            c=color_map[metrics.loc[model, "model"]],
            s=100, alpha=0.8
        )
    # Re-scatter with labels
    ax.clear()
    for _, row in metrics.iterrows():
        ax.scatter(
            row["avg_output_tokens_per_turn"], row["avg_score"],
            c=color_map[row["model"]], s=100, alpha=0.8, label=row["model"]
        )
        ax.annotate(
            row["model"], (row["avg_output_tokens_per_turn"], row["avg_score"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    ax.set_xlabel("Avg Output Tokens per Turn")
    ax.set_ylabel("Avg Score")
    ax.set_title("Score vs Token Usage")

    # Chart 3: Scatter - avg_score vs avg_failed_guesses
    ax = axes[2]
    for _, row in metrics.iterrows():
        ax.scatter(
            row["avg_failed_guesses"], row["avg_score"],
            c=color_map[row["model"]], s=100, alpha=0.8, label=row["model"]
        )
        ax.annotate(
            row["model"], (row["avg_failed_guesses"], row["avg_score"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8
        )
    ax.set_xlabel("Avg Failed Guesses")
    ax.set_ylabel("Avg Score")
    ax.set_title("Score vs Failed Guesses")

    save_figure(fig, output_path)


def analyze_basic_metrics(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
    """Run basic metrics analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BASIC MODEL COMPARISON\n")
    tee.write("=" * 60 + "\n\n")

    metrics = compute_basic_metrics(df_rounds)

    # Write to summary
    tee.write(metrics.to_string(index=False) + "\n\n")

    # Save CSV
    csv_path = output_folder / "basic_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    tee.write(f"Saved: {csv_path}\n")

    # Generate plots
    plot_path = output_folder / "basic_metrics.png"
    plot_basic_metrics(metrics, model_colors, plot_path)
    tee.write(f"Saved: {plot_path}\n")
