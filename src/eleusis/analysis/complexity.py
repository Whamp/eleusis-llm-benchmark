"""Complexity analysis for rule difficulty."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_color_map
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def find_optimal_k(df: pd.DataFrame) -> tuple[float, float]:
    """Find optimal K for aggregated complexity = cyclomatic + K * node_count.

    Returns (optimal_k, correlation).
    """
    df = df.dropna(subset=["cyclomatic_complexity", "node_count", "relative_score"])
    if len(df) < 3:
        return 0.1, 0.0

    best_k = 0.1
    best_corr = 0.0

    for k in np.arange(0.01, 1.01, 0.01):
        complexity = df["cyclomatic_complexity"] + k * df["node_count"]
        corr = complexity.corr(df["relative_score"])
        if not np.isnan(corr) and abs(corr) > abs(best_corr):
            best_corr = corr
            best_k = k

    return best_k, best_corr


def compute_complexity_metrics(df_rounds: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Compute complexity metrics including relative score and aggregated complexity.

    Returns (df_with_metrics, optimal_k, correlation).
    """
    df = df_rounds.copy()

    # Compute model average scores
    model_avg = df.groupby("model")["score"].mean()
    df["model_avg_score"] = df["model"].map(model_avg)

    # Relative score: how well did this round perform vs model's average
    df["relative_score"] = df["score"] / df["model_avg_score"]

    # Find optimal K for aggregated complexity
    optimal_k, correlation = find_optimal_k(df)

    # Compute aggregated complexity
    df["aggregated_complexity"] = (
        df["cyclomatic_complexity"] + optimal_k * df["node_count"]
    )

    # Create complexity bins (5 quantiles)
    df_valid = df.dropna(subset=["aggregated_complexity"])
    if len(df_valid) > 0:
        try:
            df.loc[df_valid.index, "complexity_bin"] = pd.qcut(
                df_valid["aggregated_complexity"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
                duplicates="drop"
            )
        except ValueError:
            # Not enough unique values for 5 bins
            df["complexity_bin"] = pd.cut(
                df["aggregated_complexity"], bins=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
            )

    return df, optimal_k, correlation


def plot_complexity_analysis(
    df: pd.DataFrame,
    model_colors: dict[str, str],
    optimal_k: float,
    output_path: Path,
):
    """Generate complexity analysis charts."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    df_valid = df.dropna(subset=["aggregated_complexity", "relative_score"])
    color_map = get_color_map(df["model"].unique().tolist(), model_colors)

    # Chart 1: Scatter - relative_score vs aggregated_complexity (colored by model)
    ax = axes[0]
    for model in df_valid["model"].unique():
        model_df = df_valid[df_valid["model"] == model]
        ax.scatter(
            model_df["aggregated_complexity"], model_df["relative_score"],
            c=color_map[model], alpha=0.6, s=50, label=model
        )
    ax.set_xlabel(f"Aggregated Complexity (cyclomatic + {optimal_k:.2f} * node_count)")
    ax.set_ylabel("Relative Score (score / model_avg)")
    ax.set_title("Relative Score vs Rule Complexity")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # Chart 2: Scatter - avg_score vs acceptance_rate (colored by aggregated_complexity)
    ax = axes[1]
    rule_stats = df_valid.groupby("rule_description").agg(
        avg_score=("score", "mean"),
        avg_acceptance_rate=("avg_acceptance_rate", "first"),
        aggregated_complexity=("aggregated_complexity", "first"),
    ).reset_index().dropna()

    if len(rule_stats) > 0:
        scatter = ax.scatter(
            rule_stats["avg_acceptance_rate"], rule_stats["avg_score"],
            c=rule_stats["aggregated_complexity"], cmap="viridis", alpha=0.7, s=60
        )
        plt.colorbar(scatter, ax=ax, label="Aggregated Complexity")
    ax.set_xlabel("Rule Acceptance Rate")
    ax.set_ylabel("Average Score")
    ax.set_title("Score vs Selectivity (color = complexity)")

    # Chart 3: Heatmap - Model x Complexity bins showing avg relative_score
    ax = axes[2]
    df_binned = df.dropna(subset=["complexity_bin", "relative_score"])
    if len(df_binned) > 0:
        heatmap = df_binned.pivot_table(
            values="relative_score",
            index="model",
            columns="complexity_bin",
            aggfunc="mean",
            observed=True
        )
        if not heatmap.empty:
            im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.5)
            ax.set_xticks(range(len(heatmap.columns)))
            ax.set_xticklabels(heatmap.columns)
            ax.set_yticks(range(len(heatmap.index)))
            ax.set_yticklabels(heatmap.index)
            ax.set_xlabel("Complexity Bin (Q1=lowest)")
            ax.set_title("Model x Complexity: Relative Score")
            plt.colorbar(im, ax=ax, label="Avg Relative Score")

            # Annotate cells
            for i in range(len(heatmap.index)):
                for j in range(len(heatmap.columns)):
                    val = heatmap.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)

    save_figure(fig, output_path)


def analyze_complexity(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
    """Run complexity analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("COMPLEXITY ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    df, optimal_k, correlation = compute_complexity_metrics(df_rounds)

    tee.write(f"Optimal K for aggregated complexity: {optimal_k:.2f}\n")
    tee.write(f"  Formula: complexity = cyclomatic + {optimal_k:.2f} * node_count\n")
    tee.write(f"  Correlation with relative_score: {correlation:.3f}\n\n")

    # Summary stats by complexity bin
    df_binned = df.dropna(subset=["complexity_bin"])
    if len(df_binned) > 0:
        bin_stats = df_binned.groupby("complexity_bin", observed=True).agg(
            count=("score", "count"),
            avg_score=("score", "mean"),
            avg_relative_score=("relative_score", "mean"),
            success_rate=("success", "mean"),
        ).reset_index()
        tee.write("Score by complexity bin:\n")
        tee.write(bin_stats.to_string(index=False) + "\n\n")

    # Generate plots
    plot_path = output_folder / "complexity_analysis.png"
    plot_complexity_analysis(df, model_colors, optimal_k, plot_path)
    tee.write(f"Saved: {plot_path}\n")

    return df  # Return enriched dataframe for potential reuse
