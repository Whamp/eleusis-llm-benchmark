"""Basic model comparison metrics and plots."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_basic_metrics(df_rounds: pd.DataFrame, df_turns: pd.DataFrame = None) -> pd.DataFrame:
    """Compute basic comparison metrics per model."""
    # Add floored_score column (score capped at minimum 0)
    df_rounds = df_rounds.copy()
    df_rounds["floored_score"] = df_rounds["score"].clip(lower=0)

    # Basic aggregations
    metrics = df_rounds.groupby("model").agg(
        rounds_played=("success", "count"),
        total_score=("score", "sum"),
        avg_score=("score", "mean"),
        total_floored_score=("floored_score", "sum"),
        avg_floored_score=("floored_score", "mean"),
        total_turns=("turn_count", "sum"),
        total_output_tokens=("output_tokens", "sum"),
        total_wall_clock=("wall_clock_seconds", "sum"),
        avg_failed_guesses=("failed_guesses", "mean"),
        success_rate=("success", "mean"),
    ).reset_index()

    # Compute no_stakes_score if turns data is available
    if df_turns is not None:
        from .no_stakes import compute_first_correct_turn, compute_no_stakes_scores
        df_first_correct = compute_first_correct_turn(df_turns)
        if not df_first_correct.empty:
            df_no_stakes = compute_no_stakes_scores(df_rounds, df_first_correct)
            no_stakes_stats = df_no_stakes.groupby("model").agg(
                total_no_stakes_score=("no_stakes_score", "sum"),
                avg_no_stakes_score=("no_stakes_score", "mean"),
            ).reset_index()
            metrics = metrics.merge(no_stakes_stats, on="model", how="left")
        else:
            metrics["total_no_stakes_score"] = 0
            metrics["avg_no_stakes_score"] = 0.0
    else:
        metrics["total_no_stakes_score"] = 0
        metrics["avg_no_stakes_score"] = 0.0

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


def plot_overall_performance(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate overall performance scatter plot with open/closed model distinction.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {"models": [], "metadata": {"x_axis": "avg_output_tokens_per_turn", "y_axis": "avg_score"}}

    for _, row in metrics.iterrows():
        model_name = row["model"]
        x = row["avg_output_tokens_per_turn"]
        y = row["avg_score"]
        color = get_model_color(model_name, model_colors)

        # Determine if open model (using normalized matching)
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
            "avg_score": float(y),
            "avg_output_tokens_per_turn": float(x),
            "color": color,
            "is_open": is_open,
            "provider": provider,
        })

    ax.set_xlabel("Average Output Tokens per Turn", fontsize=11)
    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_title("Overall Performance: Score vs Token Usage", fontsize=13, fontweight="bold")

    # Add legend for open/closed distinction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Closed model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="gray",
               markeredgewidth=2, markersize=10, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Save outputs
    png_path = output_folder / "overall_performance.png"
    json_path = output_folder / "overall_performance.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def plot_score_vs_failed_guesses(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate scatter plot of avg_score vs avg_failed_guesses with open/closed distinction.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {"models": [], "metadata": {"x_axis": "avg_failed_guesses", "y_axis": "avg_score"}}

    for _, row in metrics.iterrows():
        model_name = row["model"]
        x = row["avg_failed_guesses"]
        y = row["avg_score"]
        color = get_model_color(model_name, model_colors)

        # Determine if open model (using normalized matching)
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
            "avg_score": float(y),
            "avg_failed_guesses": float(x),
            "color": color,
            "is_open": is_open,
            "provider": provider,
        })

    ax.set_xlabel("Average Failed Guesses per Round", fontsize=11)
    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_title("Score vs Failed Guesses", fontsize=13, fontweight="bold")

    # Add legend for open/closed distinction
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Closed model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="gray",
               markeredgewidth=2, markersize=10, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    # Save outputs
    png_path = output_folder / "score_vs_failed_guesses.png"
    json_path = output_folder / "score_vs_failed_guesses.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def plot_calibration_curves(
    df_turns: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate calibration curves for all models as line chart.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "confidence_level",
            "y_axis": "actual_success_rate",
            "description": "Calibration curve showing confidence vs actual success rate for rule guesses"
        }
    }

    models = df_turns["model"].unique()

    for model_name in sorted(models):
        model_turns = df_turns[df_turns["model"] == model_name]

        # Get guesses with correctness result and confidence
        guesses = model_turns[model_turns["guess_correct"].notna()].copy()
        guesses = guesses.dropna(subset=["confidence_level"])

        if len(guesses) == 0:
            continue

        # Cast confidence_level to int to avoid floating point groupby issues
        guesses["confidence_level"] = guesses["confidence_level"].astype(int)

        # Filter to confidence range 5-10 (low confidence guesses are rare and noisy)
        guesses = guesses[guesses["confidence_level"].between(5, 10)]

        if len(guesses) == 0:
            continue

        # Bin by confidence level
        cal = guesses.groupby("confidence_level").agg(
            accuracy=("guess_correct", "mean"),
            count=("guess_correct", "count")
        ).reset_index()

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

        # Line style: dashed for open models, solid for closed
        linestyle = "--" if is_open else "-"

        # Plot line
        ax.plot(
            cal["confidence_level"], cal["accuracy"],
            color=color, linestyle=linestyle, linewidth=2,
            marker="o", markersize=6, label=model_name, alpha=0.8
        )

        # Store data for JSON
        plot_data["models"].append({
            "name": model_name,
            "color": color,
            "is_open": is_open,
            "provider": provider,
            "calibration_points": [
                {
                    "confidence_level": int(row["confidence_level"]),
                    "actual_success_rate": float(row["accuracy"]),
                    "sample_count": int(row["count"])
                }
                for _, row in cal.iterrows()
            ]
        })

    # Perfect calibration line: confidence/10 = expected accuracy
    x = np.linspace(0, 10, 100)
    ax.plot(x, x / 10, "k--", alpha=0.4, linewidth=1.5, label="Perfect calibration")

    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Actual Success Rate", fontsize=11)
    ax.set_title("Calibration Curves: Confidence vs Success Rate", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 10.)
    ax.set_ylim(0, 1.0)

    # Legend with open/closed distinction note
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add note about line styles
    ax.text(
        0.98, 0.02, "Solid = Closed model, Dashed = Open model",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        style="italic", color="gray"
    )

    # Save outputs
    png_path = output_folder / "calibration_curves.png"
    json_path = output_folder / "calibration_curves.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def plot_confidence_distribution(
    df_turns: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate confidence level distribution when models choose to guess.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "confidence_level",
            "y_axis": "proportion",
            "description": "Distribution of confidence levels when models choose to guess"
        }
    }

    models = df_turns["model"].unique()
    confidence_levels = list(range(5, 11))  # 5-10

    for model_name in sorted(models):
        model_turns = df_turns[df_turns["model"] == model_name]

        # Filter to actual guesses (when model chose to guess)
        guesses = model_turns[model_turns["guess_rule"].eq(True)].copy()
        guesses = guesses.dropna(subset=["confidence_level"])

        if len(guesses) == 0:
            continue

        # Cast to int and filter to valid range
        guesses["confidence_level"] = guesses["confidence_level"].astype(int)
        guesses = guesses[guesses["confidence_level"].between(5, 10)]

        if len(guesses) == 0:
            continue

        # Compute distribution (proportion at each confidence level)
        dist = guesses["confidence_level"].value_counts(normalize=True).sort_index()

        # Ensure all confidence levels are present (fill missing with 0)
        dist = dist.reindex(confidence_levels, fill_value=0.0)

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

        # Line style: dashed for open models, solid for closed
        linestyle = "--" if is_open else "-"

        # Plot line
        ax.plot(
            dist.index, dist.values,
            color=color, linestyle=linestyle, linewidth=2,
            marker="o", markersize=6, label=model_name, alpha=0.8
        )

        # Store data for JSON
        total_guesses = len(guesses)
        plot_data["models"].append({
            "name": model_name,
            "color": color,
            "is_open": is_open,
            "provider": provider,
            "total_guesses": total_guesses,
            "distribution": [
                {
                    "confidence_level": int(level),
                    "proportion": float(dist[level]),
                    "count": int(dist[level] * total_guesses)
                }
                for level in confidence_levels
            ]
        })

    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Proportion of Guesses", fontsize=11)
    ax.set_title("Confidence Distribution When Guessing", fontsize=13, fontweight="bold")
    ax.set_xlim(4.5, 10.5)
    ax.set_ylim(0, None)

    # Legend
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add note about line styles
    ax.text(
        0.98, 0.98, "Solid = Closed model, Dashed = Open model",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        style="italic", color="gray"
    )

    # Save outputs
    png_path = output_folder / "confidence_distribution.png"
    json_path = output_folder / "confidence_distribution.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def plot_score_stack(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Stacked bar chart showing raw score, floored delta, and no-stakes delta.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Sort by no_stakes_score descending
    metrics_sorted = metrics.sort_values("avg_no_stakes_score", ascending=False)
    models = metrics_sorted["model"].tolist()
    x = range(len(models))

    # Compute the three components
    raw_scores = metrics_sorted["avg_score"].tolist()
    floored_scores = metrics_sorted["avg_floored_score"].tolist()
    no_stakes_scores = metrics_sorted["avg_no_stakes_score"].tolist()

    # Deltas for stacking
    floored_delta = [f - r for f, r in zip(floored_scores, raw_scores)]
    no_stakes_delta = [n - f for n, f in zip(no_stakes_scores, floored_scores)]

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "description": "Stacked score breakdown: raw score + flooring gain + no-stakes gain",
            "y_axis": "average_score",
            "components": ["raw_score", "floored_delta", "no_stakes_delta"],
        },
    }

    # Collect colors and open/closed status
    colors = []
    is_open_list = []
    for model_name in models:
        color = get_model_color(model_name, model_colors)
        colors.append(color)

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
        idx = models.index(model_name)
        plot_data["models"].append({
            "name": model_name,
            "color": color,
            "is_open": is_open,
            "provider": provider,
            "avg_score": float(raw_scores[idx]),
            "avg_floored_score": float(floored_scores[idx]),
            "avg_no_stakes_score": float(no_stakes_scores[idx]),
            "floored_delta": float(floored_delta[idx]),
            "no_stakes_delta": float(no_stakes_delta[idx]),
        })

    # Draw stacked bars
    bar_width = 0.6

    # Bottom layer: raw score (can be negative)
    bars_raw = ax.bar(x, raw_scores, bar_width, label="Raw Score", color="steelblue", alpha=0.9)

    # Middle layer: floored delta (gain from flooring negatives)
    bars_floored = ax.bar(
        x, floored_delta, bar_width, bottom=raw_scores,
        label="Flooring Gain", color="coral", alpha=0.9
    )

    # Top layer: no-stakes delta (gain from no-penalty guessing)
    bars_no_stakes = ax.bar(
        x, no_stakes_delta, bar_width, bottom=floored_scores,
        label="No-Stakes Gain", color="mediumseagreen", alpha=0.9
    )

    ax.set_ylabel("Average Score", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(
        "Score Breakdown: Raw → Floored → No-Stakes",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Save outputs
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
):
    """Run basic metrics analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BASIC MODEL COMPARISON\n")
    tee.write("=" * 60 + "\n\n")

    metrics = compute_basic_metrics(df_rounds, df_turns)

    # Write to summary
    tee.write(metrics.to_string(index=False) + "\n\n")

    # Save CSV
    csv_path = output_folder / "basic_metrics.csv"
    metrics.to_csv(csv_path, index=False)
    tee.write(f"Saved: {csv_path}\n")

    # Generate overall performance plot
    png_path, json_path = plot_overall_performance(metrics, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate score vs failed guesses plot
    png_path, json_path = plot_score_vs_failed_guesses(metrics, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate calibration curves plot
    png_path, json_path = plot_calibration_curves(df_turns, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate confidence distribution plot
    png_path, json_path = plot_confidence_distribution(df_turns, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Generate score stack plot (raw -> floored -> no-stakes)
    png_path, json_path = plot_score_stack(metrics, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
