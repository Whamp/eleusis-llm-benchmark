"""Basic model comparison metrics and plots."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .basic_metric_confidence import (
    plot_calibration_curves,
    plot_confidence_distribution,
)
from .basic_metric_performance import (
    plot_overall_performance,
    plot_score_vs_failed_guesses,
)
from .colors import (
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_basic_metrics(
    df_rounds: pd.DataFrame, df_turns: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute basic comparison metrics per model."""
    # Add floored_score column (score capped at minimum 0)
    df_rounds = df_rounds.copy()
    df_rounds["floored_score"] = df_rounds["score"].clip(lower=0)

    # Basic aggregations
    # Use counting_ columns (only count metrics before score was guaranteed <= 0)
    metrics = (
        df_rounds.groupby("model")
        .agg(
            rounds_played=("success", "count"),
            total_score=("score", "sum"),
            avg_score=("score", "mean"),
            total_floored_score=("floored_score", "sum"),
            avg_floored_score=("floored_score", "mean"),
            total_turns=("counting_turn_count", "sum"),
            total_output_tokens=("output_tokens", "sum"),
            total_wall_clock=("wall_clock_seconds", "sum"),
            avg_failed_guesses=("counting_failed_guesses", "mean"),
            success_rate=("counting_success", "mean"),
        )
        .reset_index()
    )

    # Compute counting output tokens from turns data (only turns before cutoff)
    if df_turns is not None and "output_tokens" in df_turns.columns:
        counting_turns = df_turns[df_turns["counts_for_analysis"] == True]  # ruff: ignore[true-false-comparison]
        counting_tokens = (
            counting_turns.groupby("model")["output_tokens"].sum().reset_index()
        )
        counting_tokens.columns = ["model", "counting_output_tokens"]
        metrics = metrics.merge(counting_tokens, on="model", how="left")
        metrics["counting_output_tokens"] = metrics["counting_output_tokens"].fillna(0)
    else:
        # Fallback to total if turns data not available
        metrics["counting_output_tokens"] = metrics["total_output_tokens"]

    # Compute no_stakes_score if turns data is available
    if df_turns is not None:
        from .no_stakes import compute_first_correct_turn, compute_no_stakes_scores

        df_first_correct = compute_first_correct_turn(df_turns)
        if not df_first_correct.empty:
            df_no_stakes = compute_no_stakes_scores(df_rounds, df_first_correct)
            no_stakes_stats = (
                df_no_stakes.groupby("model")
                .agg(
                    total_no_stakes_score=("no_stakes_score", "sum"),
                    avg_no_stakes_score=("no_stakes_score", "mean"),
                )
                .reset_index()
            )
            metrics = metrics.merge(no_stakes_stats, on="model", how="left")
        else:
            metrics["total_no_stakes_score"] = 0
            metrics["avg_no_stakes_score"] = 0.0
    else:
        metrics["total_no_stakes_score"] = 0
        metrics["avg_no_stakes_score"] = 0.0

    # Derived metrics (using counting turns/tokens for fair comparison)
    metrics["avg_output_tokens_per_turn"] = (
        metrics["counting_output_tokens"] / metrics["total_turns"]
    )
    metrics["wall_clock_per_turn"] = (
        metrics["total_wall_clock"] / metrics["total_turns"]
    )

    # Variance analysis: intra-rule vs inter-rule (using floored_score)
    # Intra-rule variance: average of variance within same rule
    intra_var = df_rounds.groupby(["model", "rule_description"])["floored_score"].var()
    intra_var_by_model = intra_var.groupby("model").mean()
    metrics = metrics.merge(
        intra_var_by_model.rename("intra_rule_variance").reset_index(),
        on="model",
        how="left",
    )

    # Inter-rule variance: variance of per-rule mean scores
    rule_means = df_rounds.groupby(["model", "rule_description"])[
        "floored_score"
    ].mean()
    inter_var_by_model = rule_means.groupby("model").var()
    metrics = metrics.merge(
        inter_var_by_model.rename("inter_rule_variance").reset_index(),
        on="model",
        how="left",
    )

    # Variance ratio (higher = more rule-dependent, lower = more random/model-dependent)
    metrics["variance_ratio"] = (
        metrics["intra_rule_variance"] / metrics["inter_rule_variance"]
    )

    return metrics.sort_values("avg_floored_score", ascending=False)


def plot_score_stack(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Stacked bar chart showing floored score and no-stakes delta.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    model_metadata = load_model_metadata()

    # Sort by no_stakes_score descending
    metrics_sorted = metrics.sort_values("avg_no_stakes_score", ascending=False)
    models = metrics_sorted["model"].tolist()
    x = range(len(models))

    # Compute the two components
    floored_scores = metrics_sorted["avg_floored_score"].tolist()
    no_stakes_scores = metrics_sorted["avg_no_stakes_score"].tolist()

    # Delta for stacking
    no_stakes_delta = [
        n - f for n, f in zip(no_stakes_scores, floored_scores, strict=False)
    ]

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "description": "Stacked score breakdown: floored score + no-stakes gain",
            "y_axis": "average_score",
            "components": ["floored_score", "no_stakes_delta"],
        },
    }

    # Collect colors and open/closed status
    colors = []
    is_open_list = []
    for model_name in models:
        color = get_model_color(model_name, model_colors)
        colors.append(color)

        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open = metadata["is_open"]
        provider = metadata["provider"]
        is_open_list.append(is_open)

        # Store data for JSON
        idx = models.index(model_name)
        plot_data["models"].append(
            {
                "name": model_name,
                "color": color,
                "is_open": is_open,
                "provider": provider,
                "avg_floored_score": float(floored_scores[idx]),
                "avg_no_stakes_score": float(no_stakes_scores[idx]),
                "no_stakes_delta": float(no_stakes_delta[idx]),
            }
        )

    # Draw stacked bars
    bar_width = 0.6

    # Bottom layer: floored score (always >= 0)
    ax.bar(
        x,
        floored_scores,
        bar_width,
        label="Floored Score",
        color="steelblue",
        alpha=0.9,
    )

    # Top layer: no-stakes delta (gain from no-penalty guessing)
    ax.bar(
        x,
        no_stakes_delta,
        bar_width,
        bottom=floored_scores,
        label="No-Stakes Gain",
        color="mediumseagreen",
        alpha=0.9,
    )

    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(
        "Score Breakdown: Floored Score + No-Stakes Gain",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    png_path = output_folder / "score_stack.png"
    json_path = output_folder / "score_stack.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_basic_metrics(
    df_rounds: pd.DataFrame,
    df_turns: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
) -> None:
    """Run basic metrics analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BASIC MODEL COMPARISON\n")
    tee.write("=" * 60 + "\n\n")

    metrics = compute_basic_metrics(df_rounds, df_turns)

    tee.write(metrics.to_string(index=False) + "\n\n")

    csv_path = output_folder / "basic_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    tee.write(f"Saved: {csv_path}\n")

    # Generate overall performance plot
    png_path, json_path = plot_overall_performance(metrics, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate score vs failed guesses plot
    png_path, json_path = plot_score_vs_failed_guesses(
        metrics, model_colors, output_folder
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate calibration curves plot
    png_path, json_path = plot_calibration_curves(df_turns, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate confidence distribution plot
    png_path, json_path = plot_confidence_distribution(
        df_turns, model_colors, output_folder
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate score stack plot (raw -> floored -> no-stakes)
    png_path, json_path = plot_score_stack(metrics, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
