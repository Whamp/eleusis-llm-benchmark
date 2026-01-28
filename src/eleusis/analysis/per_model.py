"""Per-model detailed analysis reports."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_model_color
from .utils import save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Convert model name to safe filename."""
    return name.lower().replace(" ", "_").replace(".", "_").replace("-", "_")


def plot_play_history(ax: plt.Axes, model_rounds: pd.DataFrame, model_color: str):
    """Plot floored score by rule index showing variance across seeds."""
    # Group by rule_description to get rule index (1-indexed)
    rules = model_rounds["rule_description"].unique()
    rule_to_idx = {r: i + 1 for i, r in enumerate(rules)}
    model_rounds = model_rounds.copy()
    model_rounds["rule_idx"] = model_rounds["rule_description"].map(rule_to_idx)
    model_rounds["floored_score"] = model_rounds["score"].clip(lower=0)

    # Scatter plot with jitter
    jitter = np.random.uniform(-0.2, 0.2, len(model_rounds))
    ax.scatter(
        model_rounds["rule_idx"] + jitter, model_rounds["floored_score"],
        c=model_color, alpha=0.6, s=40
    )

    ax.set_xlabel("Rule Index")
    ax.set_ylabel("Floored Score")
    ax.set_title("Floored Score by Rule")
    ax.set_xticks(range(1, len(rules) + 1))


def plot_confidence_distribution(ax: plt.Axes, model_turns: pd.DataFrame, model_color: str):
    """Plot overlaid histograms of confidence for real vs shadow guesses."""
    # Real guesses: guess_rule is True
    real_guesses = model_turns[model_turns["guess_rule"].eq(True)]
    # Shadow guesses: guess_rule is False but has guess_attempt (is_shadow is True)
    shadow_guesses = model_turns[model_turns["is_shadow"].eq(True)]

    real_conf = real_guesses["confidence_level"].dropna()
    shadow_conf = shadow_guesses["confidence_level"].dropna()

    bins = range(0, 12)
    ax.hist(
        real_conf, bins=bins, histtype="step", linewidth=2, linestyle="-",
        color=model_color, label=f"Real guesses (n={len(real_conf)})", density=True
    )
    ax.hist(
        shadow_conf, bins=bins, histtype="step", linewidth=2, linestyle="--",
        color="gray", label=f"Shadow guesses (n={len(shadow_conf)})", density=True
    )

    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Real vs Shadow Guesses")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, 10.5)


def plot_calibration_curve(ax: plt.Axes, model_turns: pd.DataFrame, model_color: str):
    """Plot confidence vs actual success rate with perfect calibration line."""
    # Combine real and shadow guesses that have a correctness result
    guesses = model_turns[model_turns["guess_correct"].notna()].copy()
    guesses = guesses.dropna(subset=["confidence_level"])

    if len(guesses) == 0:
        ax.text(0.5, 0.5, "No guess data", ha="center", va="center", transform=ax.transAxes)
        return

    # Round confidence to integer and filter out values < 5
    guesses["confidence_level"] = guesses["confidence_level"].round().astype(int)
    guesses = guesses[guesses["confidence_level"] >= 5]

    if len(guesses) == 0:
        ax.text(0.5, 0.5, "No guess data >= 5", ha="center", va="center", transform=ax.transAxes)
        return

    # Bin by confidence level
    cal = guesses.groupby("confidence_level").agg(
        accuracy=("guess_correct", "mean"),
        count=("guess_correct", "count")
    ).reset_index()

    # Bar plot of actual accuracy
    ax.bar(cal["confidence_level"], cal["accuracy"], alpha=0.7, color=model_color)

    # Perfect calibration line: confidence/10 = expected accuracy
    x = np.linspace(5, 10, 100)
    ax.plot(x, x / 10, "k--", alpha=0.5, label="Perfect calibration")

    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Actual Success Rate")
    ax.set_title("Calibration: Confidence vs Success")
    ax.set_xlim(4.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    # Annotate sample sizes
    for _, row in cal.iterrows():
        ax.annotate(
            f"n={int(row['count'])}",
            (row["confidence_level"], row["accuracy"] + 0.05),
            ha="center", fontsize=8
        )


def plot_complexity_scatter(
    ax: plt.Axes,
    model_turns: pd.DataFrame,
    rules_lib: dict,
    model_color: str,
    optimal_k: float,
):
    """Plot tentative complexity vs actual complexity."""
    # Get turns with tentative complexity
    df = model_turns[
        model_turns["tentative_node_count"].notna() &
        model_turns["tentative_cyclomatic"].notna()
    ].copy()

    if len(df) == 0:
        ax.text(0.5, 0.5, "No complexity data", ha="center", va="center", transform=ax.transAxes)
        return

    # Compute tentative aggregated complexity
    df["tentative_complexity"] = (
        df["tentative_cyclomatic"] + optimal_k * df["tentative_node_count"]
    )

    # Get actual complexity from rules_lib
    def get_actual_complexity(rule_desc):
        rule = rules_lib.get(rule_desc, {})
        cc = rule.get("cyclomatic_complexity")
        nc = rule.get("node_count")
        if cc is not None and nc is not None:
            return cc + optimal_k * nc
        return None

    df["actual_complexity"] = df["actual_rule"].apply(get_actual_complexity)
    df = df.dropna(subset=["actual_complexity"])

    if len(df) == 0:
        ax.text(0.5, 0.5, "No matching rules", ha="center", va="center", transform=ax.transAxes)
        return

    # Single color scatter plot (no correctness distinction)
    ax.scatter(
        df["actual_complexity"], df["tentative_complexity"],
        c=model_color, alpha=0.3, s=20
    )

    # Perfect match line
    ax.plot([0, 60], [0, 60], "k--", alpha=0.5)

    ax.set_xlabel("Actual Rule Complexity")
    ax.set_ylabel("Tentative Rule Complexity")
    ax.set_title(f"Tentative vs Actual Complexity (k={optimal_k:.2f})")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)


def generate_per_model_report(
    model: str,
    model_rounds: pd.DataFrame,
    model_turns: pd.DataFrame,
    rules_lib: dict,
    model_colors: dict[str, str],
    output_folder: Path,
    optimal_k: float = 0.1,
):
    """Generate report for a single model."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Analysis: {model}", fontsize=14, fontweight="bold")

    model_color = get_model_color(model, model_colors)

    # 5.1 Play History
    plot_play_history(axes[0, 0], model_rounds, model_color)

    # 5.2 Confidence Distribution
    plot_confidence_distribution(axes[0, 1], model_turns, model_color)

    # 5.3 Calibration Curve
    plot_calibration_curve(axes[1, 0], model_turns, model_color)

    # 5.4 Tentative vs Actual Complexity
    plot_complexity_scatter(axes[1, 1], model_turns, rules_lib, model_color, optimal_k)

    # Save
    filename = "model_" + sanitize_filename(model) + ".png"
    output_path = output_folder / filename
    save_figure(fig, output_path)
    return output_path


def generate_per_model_reports(
    df_rounds: pd.DataFrame,
    df_turns: pd.DataFrame,
    rules_lib: dict,
    model_colors: dict[str, str],
    output_folder: Path,
    optimal_k: float = 0.1,
) -> list[Path]:
    """Generate reports for all models."""
    models = df_rounds["model"].unique()
    output_paths = []

    for model in models:
        model_rounds = df_rounds[df_rounds["model"] == model]
        model_turns = df_turns[df_turns["model"] == model]

        path = generate_per_model_report(
            model, model_rounds, model_turns, rules_lib,
            model_colors, output_folder, optimal_k
        )
        output_paths.append(path)

    return output_paths
