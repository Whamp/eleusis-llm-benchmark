"""Confidence calibration and distribution plots."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .colors import (
    ModelMetadata,
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def _plot_model_calibration(
    ax: Axes,
    model_name: str,
    model_turns: pd.DataFrame,
    model_colors: dict[str, str],
    model_metadata: dict[str, ModelMetadata],
) -> dict[str, object] | None:
    """Plot and serialize one model's confidence calibration series."""
    guesses = model_turns[model_turns["guess_correct"].notna()].copy()
    guesses = guesses.dropna(subset=["confidence_level"])
    if guesses.empty:
        return None
    guesses["confidence_level"] = guesses["confidence_level"].astype(int)
    guesses = guesses[guesses["confidence_level"].between(5, 10)]
    if guesses.empty:
        return None
    calibration = (
        guesses.groupby("confidence_level")
        .agg(accuracy=("guess_correct", "mean"), count=("guess_correct", "count"))
        .reset_index()
    )
    color = get_model_color(model_name, model_colors)
    metadata = resolve_model_metadata(model_name, model_metadata)
    ax.plot(
        calibration["confidence_level"],
        calibration["accuracy"],
        color=color,
        linestyle="--" if metadata["is_open"] else "-",
        linewidth=2,
        marker="o",
        markersize=6,
        label=model_name,
        alpha=0.8,
    )
    return {
        "name": model_name,
        "color": color,
        "is_open": metadata["is_open"],
        "provider": metadata["provider"],
        "calibration_points": [
            {
                "confidence_level": int(row["confidence_level"]),
                "actual_success_rate": float(row["accuracy"]),
                "sample_count": int(row["count"]),
            }
            for _, row in calibration.iterrows()
        ],
    }


def plot_calibration_curves(
    df_turns: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate calibration curves for all models as line chart.

    Only includes turns before the counting cutoff (where score could still be
    positive).

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Filter to turns that count for analysis (before cutoff)
    df_counting = df_turns[df_turns["counts_for_analysis"] == True]  # ruff: ignore[true-false-comparison]

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "confidence_level",
            "y_axis": "actual_success_rate",
            "description": (
                "Calibration curve showing confidence vs actual success rate for rule"
                " guesses (counting turns only)"
            ),
        },
    }

    for model_name in sorted(df_counting["model"].unique()):
        model_data = _plot_model_calibration(
            ax,
            model_name,
            df_counting[df_counting["model"] == model_name],
            model_colors,
            model_metadata,
        )
        if model_data:
            plot_data["models"].append(model_data)

    # Perfect calibration line: confidence/10 = expected accuracy
    x = np.linspace(0, 10, 100)
    ax.plot(x, x / 10, "k--", alpha=0.4, linewidth=1.5, label="Perfect calibration")

    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Actual Success Rate", fontsize=11)
    ax.set_title(
        "Calibration Curves: Confidence vs Success Rate", fontsize=13, fontweight="bold"
    )
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 1.0)

    # Legend with open/closed distinction note
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add note about line styles
    ax.text(
        0.98,
        0.02,
        "Solid = Closed model, Dashed = Open model",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )

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
    """Generate a guess-rate-by-confidence plot.

    Shows the proportion of times each model guessed at each confidence level.

    Only includes turns before the counting cutoff (where score could still be
    positive).

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Filter to turns that count for analysis (before cutoff)
    df_counting = df_turns[df_turns["counts_for_analysis"] == True]  # ruff: ignore[true-false-comparison]

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "confidence_level",
            "y_axis": "guess_rate",
            "description": "Guess rate by confidence level (counting turns only)",
        },
    }

    models = df_counting["model"].unique()
    confidence_levels = list(range(5, 11))  # 5-10

    for model_name in sorted(models):
        model_turns = df_counting[df_counting["model"] == model_name]

        turns_with_conf = model_turns.dropna(subset=["confidence_level"]).copy()
        turns_with_conf["confidence_level"] = turns_with_conf[
            "confidence_level"
        ].astype(int)
        turns_with_conf = turns_with_conf[
            turns_with_conf["confidence_level"].between(5, 10)
        ]

        if len(turns_with_conf) == 0:
            continue

        # Compute guess rate at each confidence level
        # Use fill_value=1.0 for missing levels (no data means we assume they would
        # guess)
        guess_rate = turns_with_conf.groupby("confidence_level")["guess_rule"].mean()
        guess_rate = guess_rate.reindex(confidence_levels, fill_value=1.0)

        # Also compute counts for JSON export
        total_turns_per_level = turns_with_conf.groupby("confidence_level").size()
        guesses_only = turns_with_conf[turns_with_conf["guess_rule"] == True]  # ruff: ignore[true-false-comparison]
        guess_count_per_level = guesses_only.groupby("confidence_level").size()

        color = get_model_color(model_name, model_colors)

        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open = metadata["is_open"]
        provider = metadata["provider"]

        # Line style: dashed for open models, solid for closed
        linestyle = "--" if is_open else "-"

        # Plot line with markers
        ax.plot(
            guess_rate.index,
            guess_rate.values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=6,
            label=model_name,
            alpha=0.8,
        )

        # Store data for JSON
        plot_data["models"].append(
            {
                "name": model_name,
                "color": color,
                "is_open": is_open,
                "provider": provider,
                "distribution": [
                    {
                        "confidence_level": int(level),
                        "guess_rate": float(guess_rate[level]),
                        "total_turns": int(total_turns_per_level.get(level, 0)),
                        "guess_count": int(guess_count_per_level.get(level, 0)),
                    }
                    for level in confidence_levels
                ],
            }
        )

    ax.set_xlabel("Confidence Level", fontsize=11)
    ax.set_ylabel("Guess Rate", fontsize=11)
    ax.set_title("Guess Rate by Confidence Level", fontsize=13, fontweight="bold")
    ax.set_xlim(4.5, 10.5)
    ax.set_ylim(0, 1.05)

    # Legend
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add note about line styles
    ax.text(
        0.98,
        0.98,
        "Solid = Closed model, Dashed = Open model",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        style="italic",
        color="gray",
    )

    png_path = output_folder / "guess_rate.png"
    json_path = output_folder / "guess_rate.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path
