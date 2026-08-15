"""Distribution plots for early correct shadow guesses."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch

from .colors import (
    ModelMetadata,
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .utils import save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def _prepare_excess_caution_models(
    df_early: pd.DataFrame,
    models: list[str],
    model_colors: dict[str, str],
    model_metadata: dict[str, ModelMetadata],
) -> tuple[list[np.ndarray], list[str], list[bool], list[dict[str, object]]]:
    """Prepare boxplot values, styles, and JSON records for each model."""
    box_data: list[np.ndarray] = []
    colors: list[str] = []
    is_open_list: list[bool] = []
    records: list[dict[str, object]] = []
    for model_name in models:
        model_data = df_early[df_early["model"] == model_name]["early_correct_turns"]
        box_data.append(model_data.values)
        color = get_model_color(model_name, model_colors)
        colors.append(color)
        metadata = resolve_model_metadata(model_name, model_metadata)
        is_open_list.append(metadata["is_open"])
        records.append(
            {
                "name": model_name,
                "color": color,
                "is_open": metadata["is_open"],
                "provider": metadata["provider"],
                "values": model_data.tolist(),
                "mean": float(model_data.mean()),
                "median": float(model_data.median()),
                "std": float(model_data.std()) if len(model_data) > 1 else 0.0,
                "count": len(model_data),
            }
        )
    return box_data, colors, is_open_list, records


def _style_excess_caution_boxes(
    boxes: list[PathPatch],
    medians: list[Line2D],
    whiskers: list[Line2D],
    caps: list[Line2D],
    colors: list[str],
    is_open_list: list[bool],
) -> None:
    """Apply model and licensing styles to boxplot artists."""
    for patch, color, is_open in zip(boxes, colors, is_open_list, strict=False):
        patch.set_edgecolor(color)
        if is_open:
            patch.set_facecolor("white")
            patch.set_linewidth(2)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    for median in medians:
        median.set_color("black")
        median.set_linewidth(2)
    for line in [*whiskers, *caps]:
        line.set_color("gray")
        line.set_linewidth(1.5)


def plot_excess_caution(
    df_early: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Generate box plot of early correct turns per model.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    model_metadata = load_model_metadata()

    # Sort models by median early correct turns
    model_medians = df_early.groupby("model")["early_correct_turns"].median()
    models = sorted(
        model_medians.index.tolist(),
        key=lambda model_name: model_medians[model_name],
        reverse=True,
    )

    box_data, colors, is_open_list, model_records = _prepare_excess_caution_models(
        df_early, models, model_colors, model_metadata
    )
    plot_data = {
        "models": model_records,
        "metadata": {
            "description": (
                "Early correct turns: consecutive shadow guesses that were correct"
                " before the winning guess"
            ),
            "y_axis": "early_correct_turns",
        },
    }

    bp = ax.boxplot(
        box_data,
        tick_labels=models,
        patch_artist=True,
        widths=0.6,
    )

    _style_excess_caution_boxes(
        bp["boxes"],
        bp["medians"],
        bp["whiskers"],
        bp["caps"],
        colors,
        is_open_list,
    )

    ax.set_ylabel("Early Correct Turns", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(
        "Excess Caution: Turns with Correct Answer Before Guessing",
        fontsize=13,
        fontweight="bold",
    )

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=9)

    # Add legend for open/closed distinction
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, edgecolor="gray", label="Closed model"),
        Patch(facecolor="white", edgecolor="gray", linewidth=2, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Add note
    ax.text(
        0.02,
        0.98,
        "Higher = more cautious (delayed guessing despite having correct answer)",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="top",
        style="italic",
        color="gray",
    )

    plt.tight_layout()

    png_path = output_folder / "excess_caution.png"
    json_path = output_folder / "excess_caution.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path
