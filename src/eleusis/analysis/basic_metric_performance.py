"""Overall score and efficiency performance plots."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import (
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def plot_overall_performance(
    metrics: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate overall performance scatter plot with open/closed model distinction.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "avg_output_tokens_per_turn",
            "y_axis": "avg_floored_score",
        },
    }

    for _, row in metrics.iterrows():
        model_name = row["model"]
        x = row["avg_output_tokens_per_turn"]
        y = row["avg_floored_score"]
        color = get_model_color(model_name, model_colors)

        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open = metadata["is_open"]
        provider = metadata["provider"]

        # Plot point: open circle for open models, filled for closed
        if is_open:
            ax.scatter(
                x, y, c="none", edgecolors=color, s=150, linewidths=2.5, zorder=3
            )
        else:
            ax.scatter(x, y, c=color, s=150, alpha=0.9, zorder=3)

        # Add label
        ax.annotate(
            model_name,
            (x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        # Store data for JSON
        plot_data["models"].append(
            {
                "name": model_name,
                "avg_floored_score": float(y),
                "avg_output_tokens_per_turn": float(x),
                "color": color,
                "is_open": is_open,
                "provider": provider,
            }
        )

    ax.set_xlabel("Average Output Tokens per Turn", fontsize=11)
    ax.set_ylabel("Average Floored Score", fontsize=11)
    ax.set_title(
        "Overall Performance: Floored Score vs Token Usage",
        fontsize=13,
        fontweight="bold",
    )

    # Add legend for open/closed distinction
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Closed model",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=2,
            markersize=10,
            label="Open model",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

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
    """Generate a score-versus-failed-guesses scatter plot.

    Distinguishes open and closed models.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {"x_axis": "avg_failed_guesses", "y_axis": "avg_floored_score"},
    }

    for _, row in metrics.iterrows():
        model_name = row["model"]
        x = row["avg_failed_guesses"]
        y = row["avg_floored_score"]
        color = get_model_color(model_name, model_colors)

        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open = metadata["is_open"]
        provider = metadata["provider"]

        # Plot point: open circle for open models, filled for closed
        if is_open:
            ax.scatter(
                x, y, c="none", edgecolors=color, s=150, linewidths=2.5, zorder=3
            )
        else:
            ax.scatter(x, y, c=color, s=150, alpha=0.9, zorder=3)

        # Add label
        ax.annotate(
            model_name,
            (x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
        )

        # Store data for JSON
        plot_data["models"].append(
            {
                "name": model_name,
                "avg_floored_score": float(y),
                "avg_failed_guesses": float(x),
                "color": color,
                "is_open": is_open,
                "provider": provider,
            }
        )

    ax.set_xlabel("Average Failed Guesses per Round", fontsize=11)
    ax.set_ylabel("Average Floored Score", fontsize=11)
    ax.set_title("Floored Score vs Failed Guesses", fontsize=13, fontweight="bold")

    # Add legend for open/closed distinction
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label="Closed model",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="gray",
            markeredgewidth=2,
            markersize=10,
            label="Open model",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10)

    png_path = output_folder / "score_vs_failed_guesses.png"
    json_path = output_folder / "score_vs_failed_guesses.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path
