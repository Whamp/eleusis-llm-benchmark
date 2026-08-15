"""Score, caution, and failed-guess relationship plots."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .colors import (
    ModelMetadata,
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def _plot_model_point(
    ax: Axes,
    model_name: str,
    x: float,
    y: float,
    color: str,
    is_open: bool,
) -> None:
    """Plot and label one open- or closed-model scatter point."""
    if is_open:
        ax.scatter(x, y, c="none", edgecolors=color, s=150, linewidths=2.5, zorder=3)
    else:
        ax.scatter(x, y, c=color, s=150, alpha=0.9, zorder=3)
    ax.annotate(
        model_name,
        (x, y),
        xytext=(8, 4),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )


def _add_model_openness_legend(ax: Axes) -> None:
    """Add the shared open-versus-closed model legend."""
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
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)


def _plot_caution_models(
    ax: Axes,
    caution_by_model: pd.Series,
    failed_by_model: pd.Series,
    model_colors: dict[str, str],
    model_metadata: dict[str, ModelMetadata],
) -> list[dict[str, object]]:
    """Plot and serialize caution-versus-failed-guesses model points."""
    records: list[dict[str, object]] = []
    for model_name in caution_by_model.index:
        if model_name not in failed_by_model.index:
            continue
        x = float(failed_by_model[model_name])
        y = float(caution_by_model[model_name])
        color = get_model_color(model_name, model_colors)
        metadata = resolve_model_metadata(model_name, model_metadata)
        _plot_model_point(ax, model_name, x, y, color, metadata["is_open"])
        records.append(
            {
                "name": model_name,
                "avg_early_correct_turns": y,
                "avg_failed_guesses": x,
                "color": color,
                "is_open": metadata["is_open"],
                "provider": metadata["provider"],
            }
        )
    return records


def plot_caution_vs_failed_guesses(
    df_early: pd.DataFrame,
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate scatter plot of average caution vs average failed guesses.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Compute per-model metrics
    # Use counting_failed_guesses (only guesses before score was guaranteed <= 0)
    caution_by_model = df_early.groupby("model")["early_correct_turns"].mean()
    failed_by_model = df_rounds.groupby("model")["counting_failed_guesses"].mean()

    # Prepare data for JSON export
    plot_data = {
        "models": _plot_caution_models(
            ax,
            caution_by_model,
            failed_by_model,
            model_colors,
            model_metadata,
        ),
        "metadata": {
            "x_axis": "avg_failed_guesses",
            "y_axis": "avg_early_correct_turns",
            "description": (
                "Trade-off between caution (delaying correct guesses) and recklessness"
                " (failed guesses)"
            ),
        },
    }

    ax.set_xlabel("Average Failed Guesses per Round", fontsize=11)
    ax.set_ylabel("Average Early Correct Turns", fontsize=11)
    ax.set_title("Caution vs Recklessness Trade-off", fontsize=13, fontweight="bold")

    # Add quadrant labels
    ax.get_xlim()
    ax.get_ylim()
    ax.text(
        0.02,
        0.98,
        "Cautious\n(delays correct guesses)",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        style="italic",
        color="gray",
    )
    ax.text(
        0.98,
        0.02,
        "Reckless\n(many failed guesses)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )

    _add_model_openness_legend(ax)

    png_path = output_folder / "caution_vs_failed_guesses.png"
    json_path = output_folder / "caution_vs_failed_guesses.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def _plot_recklessness_models(
    ax: Axes,
    caution_by_model: pd.Series,
    failed_by_model: pd.Series,
    score_by_model: pd.Series,
    model_colors: dict[str, str],
    model_metadata: dict[str, ModelMetadata],
) -> list[dict[str, object]]:
    """Plot and serialize score-versus-recklessness model points."""
    records: list[dict[str, object]] = []
    for model_name in caution_by_model.index:
        if (
            model_name not in failed_by_model.index
            or model_name not in score_by_model.index
        ):
            continue
        caution = float(caution_by_model[model_name])
        failed = float(failed_by_model[model_name])
        recklessness = 2 * failed - caution
        score = float(score_by_model[model_name])
        color = get_model_color(model_name, model_colors)
        metadata = resolve_model_metadata(model_name, model_metadata)
        _plot_model_point(
            ax, model_name, recklessness, score, color, metadata["is_open"]
        )
        records.append(
            {
                "name": model_name,
                "avg_floored_score": score,
                "recklessness_index": recklessness,
                "avg_failed_guesses": failed,
                "avg_caution": caution,
                "color": color,
                "is_open": metadata["is_open"],
                "provider": metadata["provider"],
            }
        )
    return records


def plot_score_vs_recklessness(
    df_early: pd.DataFrame,
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate a score-versus-recklessness scatter plot.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    model_metadata = load_model_metadata()

    # Compute per-model metrics
    caution_by_model = df_early.groupby("model")["early_correct_turns"].mean()
    failed_by_model = df_rounds.groupby("model")["counting_failed_guesses"].mean()

    # Compute floored score per model
    df_rounds_copy = df_rounds.copy()
    df_rounds_copy["floored_score"] = df_rounds_copy["score"].clip(lower=0)
    score_by_model = df_rounds_copy.groupby("model")["floored_score"].mean()

    # Prepare data for JSON export
    plot_data = {
        "models": _plot_recklessness_models(
            ax,
            caution_by_model,
            failed_by_model,
            score_by_model,
            model_colors,
            model_metadata,
        ),
        "metadata": {
            "x_axis": "recklessness_index",
            "y_axis": "avg_floored_score",
            "description": "Recklessness index = 2 * avg_failed_guesses - avg_caution",
        },
    }

    ax.set_xlabel("Recklessness Index (2 x Failed Guesses - Caution)", fontsize=11)
    ax.set_ylabel("Average Floored Score", fontsize=11)
    ax.set_title("Score vs Recklessness Index", fontsize=13, fontweight="bold")

    # Add quadrant labels
    ax.text(
        0.02,
        0.02,
        "Conservative\n(cautious, few failed guesses)",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        style="italic",
        color="gray",
    )
    ax.text(
        0.98,
        0.02,
        "Reckless\n(many failed guesses, low caution)",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        style="italic",
        color="gray",
    )

    _add_model_openness_legend(ax)

    png_path = output_folder / "score_vs_recklessness.png"
    json_path = output_folder / "score_vs_recklessness.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path
