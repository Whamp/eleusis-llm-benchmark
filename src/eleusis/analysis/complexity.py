"""Complexity analysis for rule difficulty."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def find_optimal_k(df_rules: pd.DataFrame) -> tuple[float, float]:
    """Find optimal K for aggregated complexity = cyclomatic + K * node_count.

    Args:
        df_rules: DataFrame with one row per rule, containing cyclomatic_complexity,
                  node_count, and success_rate columns.

    Returns (optimal_k, correlation). Targets strongest negative correlation
    (higher complexity -> lower success rate).
    """
    df = df_rules.dropna(subset=["cyclomatic_complexity", "node_count", "success_rate"])
    if len(df) < 3:
        return 0.1, 0.0

    best_k = 0.1
    best_corr = 0.0

    for k in np.arange(0.01, 1.01, 0.01):
        complexity = df["cyclomatic_complexity"] + k * df["node_count"]
        corr = complexity.corr(df["success_rate"])
        # Target strongest negative correlation
        if not np.isnan(corr) and corr < best_corr:
            best_corr = corr
            best_k = k

    return best_k, best_corr


def compute_complexity_metrics(df_rounds: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Compute complexity metrics including per-rule success rate and aggregated complexity.

    Returns (df_with_metrics, optimal_k, correlation).
    """
    df = df_rounds.copy()

    # Compute per-rule success rate (times found / times played, across all models)
    rule_stats = df.groupby("rule_description").agg(
        times_played=("success", "count"),
        times_found=("success", "sum"),
        cyclomatic_complexity=("cyclomatic_complexity", "first"),
        node_count=("node_count", "first"),
    ).reset_index()
    rule_stats["success_rate"] = rule_stats["times_found"] / rule_stats["times_played"]

    # Find optimal K using per-rule data
    optimal_k, correlation = find_optimal_k(rule_stats)

    # Merge success_rate back to rounds dataframe
    df = df.merge(
        rule_stats[["rule_description", "success_rate"]],
        on="rule_description",
        how="left"
    )

    # Compute aggregated complexity
    df["aggregated_complexity"] = (
        df["cyclomatic_complexity"] + optimal_k * df["node_count"]
    )

    # Create complexity bins (4 quartiles based on rule complexity)
    # Use rule-level data to define bins, then map back
    rule_stats["aggregated_complexity"] = (
        rule_stats["cyclomatic_complexity"] + optimal_k * rule_stats["node_count"]
    )
    rule_valid = rule_stats.dropna(subset=["aggregated_complexity"])
    if len(rule_valid) > 0:
        try:
            rule_stats.loc[rule_valid.index, "complexity_bin"] = pd.qcut(
                rule_valid["aggregated_complexity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop"
            )
        except ValueError:
            rule_stats["complexity_bin"] = pd.cut(
                rule_stats["aggregated_complexity"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"]
            )
        # Merge bins back to rounds
        df = df.merge(
            rule_stats[["rule_description", "complexity_bin"]],
            on="rule_description",
            how="left",
            suffixes=("_old", "")
        )
        if "complexity_bin_old" in df.columns:
            df = df.drop(columns=["complexity_bin_old"])

    return df, optimal_k, correlation


def plot_complexity_analysis(
    df: pd.DataFrame,
    optimal_k: float,
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate complexity analysis heatmap.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()

    df_binned = df.dropna(subset=["complexity_bin", "success_rate"])

    # Prepare JSON data
    plot_data = {
        "models": [],
        "quartiles": ["Q1", "Q2", "Q3", "Q4"],
        "metadata": {
            "optimal_k": optimal_k,
            "complexity_formula": f"cyclomatic + {optimal_k:.2f} * node_count",
            "value": "success_rate",
            "description": "Success rate = per-model success rate for rules in each quartile.",
        }
    }

    if len(df_binned) == 0:
        # Empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        png_path = output_folder / "complexity_analysis.png"
        json_path = output_folder / "complexity_analysis.json"
        save_figure(fig, png_path)
        with open(json_path, "w") as f:
            json.dump(plot_data, f, indent=2)
        return png_path, json_path

    # Create pivot table for heatmap: per-model success rate by complexity bin
    heatmap = df_binned.pivot_table(
        values="success",
        index="model",
        columns="complexity_bin",
        aggfunc="mean",
        observed=True
    )

    # Order models by average score (best on top)
    model_avg_scores = df_binned.groupby("model")["score"].mean().sort_values(ascending=False)
    heatmap = heatmap.reindex(model_avg_scores.index)

    if heatmap.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        png_path = output_folder / "complexity_analysis.png"
        json_path = output_folder / "complexity_analysis.json"
        save_figure(fig, png_path)
        with open(json_path, "w") as f:
            json.dump(plot_data, f, indent=2)
        return png_path, json_path

    # Ensure quartile columns are in order
    quartile_order = ["Q1", "Q2", "Q3", "Q4"]
    heatmap = heatmap.reindex(columns=[q for q in quartile_order if q in heatmap.columns])

    # Create figure
    fig_height = max(4, len(heatmap.index) * 0.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns, fontsize=10)
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=10)
    ax.set_xlabel("Complexity Quartile (Q1 = easiest)", fontsize=11)
    ax.set_title("Model Success Rate by Rule Complexity", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)

    # Annotate cells with values
    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            val = heatmap.values[i, j]
            if not np.isnan(val):
                # Choose text color based on value
                text_color = "white" if val < 0.3 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    # Build JSON data
    for model in heatmap.index:
        model_data = {
            "name": model,
            "quartile_scores": {}
        }
        for q in heatmap.columns:
            val = heatmap.loc[model, q]
            model_data["quartile_scores"][q] = float(val) if not np.isnan(val) else None
        plot_data["models"].append(model_data)

    # Save outputs
    png_path = output_folder / "complexity_analysis.png"
    json_path = output_folder / "complexity_analysis.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


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
    tee.write(f"  Correlation with success_rate: {correlation:.3f}\n\n")

    # Summary stats by complexity bin
    df_binned = df.dropna(subset=["complexity_bin"])
    if len(df_binned) > 0:
        bin_stats = df_binned.groupby("complexity_bin", observed=True).agg(
            count=("score", "count"),
            avg_score=("score", "mean"),
            success_rate=("success", "mean"),
        ).reset_index()
        tee.write("Stats by complexity quartile:\n")
        tee.write(bin_stats.to_string(index=False) + "\n\n")

    # Generate plot
    png_path, json_path = plot_complexity_analysis(df, optimal_k, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    return df  # Return enriched dataframe for potential reuse
