"""Analysis of complexity ratio: tentative rule complexity vs actual rule complexity."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .colors import (
    get_model_color,
    load_model_metadata,
    resolve_model_metadata,
)
from .legacy_records import RuleLookup
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComplexityRatioPlotSeries:
    """Aligned model values and display styles for a complexity-ratio plot."""

    models: list[str]
    medians: np.ndarray
    q25s: np.ndarray
    q75s: np.ndarray
    counts: np.ndarray
    colors: list[str]
    is_open: list[bool]


def compute_complexity_ratios(
    df_turns: pd.DataFrame,
    rules_lib: RuleLookup,
    optimal_k: float,
) -> pd.DataFrame:
    """Compute complexity ratio for each turn with confidence >= 5.

    Complexity ratio = (tentative rule complexity) / (actual rule complexity)

    Returns DataFrame with columns: model, round_number, turn_number,
    tentative_complexity, actual_complexity, complexity_ratio
    """
    # Filter to turns with tentative complexity data and confidence >= 5
    df = df_turns[
        df_turns["tentative_node_count"].notna()
        & df_turns["tentative_cyclomatic"].notna()
        & (df_turns["confidence_level"] >= 5)
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Compute tentative aggregated complexity
    df["tentative_complexity"] = (
        df["tentative_cyclomatic"] + optimal_k * df["tentative_node_count"]
    )

    def get_actual_complexity(rule_desc: str) -> float | None:
        rule = rules_lib.get(rule_desc, {})
        cc = rule.get("cyclomatic_complexity")
        nc = rule.get("node_count")
        if cc is not None and nc is not None:
            return cc + optimal_k * nc
        return None

    df["actual_complexity"] = df["actual_rule"].apply(get_actual_complexity)
    df = df.dropna(subset=["actual_complexity"])

    if df.empty:
        return pd.DataFrame()

    # Compute complexity ratio (avoid division by zero)
    df["complexity_ratio"] = df["tentative_complexity"] / df[
        "actual_complexity"
    ].replace(0, np.nan)
    df = df.dropna(subset=["complexity_ratio"])

    return df[
        [
            "model",
            "round_number",
            "turn_number",
            "confidence_level",
            "tentative_complexity",
            "actual_complexity",
            "complexity_ratio",
        ]
    ]


def compute_model_complexity_stats(df_ratios: pd.DataFrame) -> pd.DataFrame:
    """Compute median and quartiles of complexity ratio per model.

    Returns DataFrame with columns: model, median_ratio, q25, q75, count
    """
    if df_ratios.empty:
        return pd.DataFrame()

    stats = (
        df_ratios.groupby("model")
        .agg(
            median_ratio=("complexity_ratio", "median"),
            q25=("complexity_ratio", lambda x: x.quantile(0.25)),
            q75=("complexity_ratio", lambda x: x.quantile(0.75)),
            count=("complexity_ratio", "count"),
        )
        .reset_index()
    )

    # Sort by median_ratio descending
    stats = stats.sort_values("median_ratio", ascending=False)

    return stats


def _build_complexity_ratio_series(
    df_stats: pd.DataFrame,
    model_colors: dict[str, str],
) -> ComplexityRatioPlotSeries:
    """Build aligned complexity-ratio values and model styles."""
    model_metadata = load_model_metadata()
    models = df_stats["model"].tolist()
    colors = [get_model_color(model, model_colors) for model in models]
    is_open = [
        resolve_model_metadata(model, model_metadata)["is_open"] for model in models
    ]
    return ComplexityRatioPlotSeries(
        models=models,
        medians=df_stats["median_ratio"].values,
        q25s=df_stats["q25"].values,
        q75s=df_stats["q75"].values,
        counts=df_stats["count"].values,
        colors=colors,
        is_open=is_open,
    )


def _draw_complexity_ratio_series(
    ax: Axes,
    series: ComplexityRatioPlotSeries,
) -> None:
    """Draw model medians and asymmetric quartile error bars."""
    values = zip(
        series.medians,
        series.q25s,
        series.q75s,
        series.colors,
        series.is_open,
        series.counts,
        strict=False,
    )
    for index, (median, q25, q75, color, is_open, count) in enumerate(values):
        ax.errorbar(
            median,
            index,
            xerr=[[median - q25], [q75 - median]],
            fmt="s" if is_open else "o",
            markersize=10,
            markerfacecolor="white" if is_open else color,
            markeredgecolor=color,
            markeredgewidth=2,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )
        ax.text(q75 + 0.08, index, f"{median:.2f} (n={count})", va="center", fontsize=9)


def _add_complexity_ratio_legend(ax: Axes) -> None:
    """Add open- and closed-model marker definitions."""
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=10,
            label="Closed model",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="gray",
            markeredgewidth=2,
            markersize=10,
            label="Open model",
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)


def _serialize_complexity_ratio_series(
    series: ComplexityRatioPlotSeries,
) -> dict[str, object]:
    """Build the JSON-ready complexity-ratio plot payload."""
    return {
        "metadata": {
            "description": (
                "Complexity ratio: tentative rule complexity / actual rule complexity"
            ),
            "confidence_threshold": 5,
            "interpretation": (
                "Ratio > 1 means model overcomplicates, < 1 means oversimplifies"
            ),
        },
        "models": [
            {
                "name": model,
                "color": series.colors[index],
                "is_open": series.is_open[index],
                "median_ratio": float(series.medians[index]),
                "q25": float(series.q25s[index]),
                "q75": float(series.q75s[index]),
                "count": int(series.counts[index]),
            }
            for index, model in enumerate(series.models)
        ],
    }


def plot_complexity_ratio(
    df_stats: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate model complexity-ratio medians with quartile error bars."""
    setup_matplotlib_style()
    figure, axes = plt.subplots(figsize=(10, 7))
    series = _build_complexity_ratio_series(df_stats, model_colors)
    _draw_complexity_ratio_series(axes, series)

    axes.set_yticks(np.arange(len(series.models)))
    axes.set_yticklabels(series.models, fontsize=9)
    axes.set_xlabel("Complexity Ratio (Tentative / Actual)", fontsize=11)
    axes.set_title(
        "Rule Complexity Ratio by Model\n(Tentative rules with confidence ≥ 5)",
        fontsize=11,
        fontweight="bold",
    )
    axes.invert_yaxis()
    axes.axvline(
        x=1.0,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label="Ratio = 1 (exact match)",
    )
    _add_complexity_ratio_legend(axes)
    axes.text(
        0.02,
        -0.08,
        ">1: Overcomplicates  |  <1: Oversimplifies  |  =1: Matches complexity",
        transform=axes.transAxes,
        fontsize=8,
        ha="left",
        style="italic",
        color="gray",
    )
    plt.tight_layout()

    png_path = output_folder / "complexity_ratio.png"
    json_path = output_folder / "complexity_ratio.json"
    save_figure(figure, png_path)
    with json_path.open("w") as output_file:
        json.dump(_serialize_complexity_ratio_series(series), output_file, indent=2)
    logger.info(f"Saved: {json_path}")
    return png_path, json_path


def analyze_complexity_ratio(
    df_turns: pd.DataFrame,
    rules_lib: RuleLookup,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
    optimal_k: float,
) -> None:
    """Run complexity ratio analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("COMPLEXITY RATIO ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Compute ratios for all turns with confidence >= 5
    df_ratios = compute_complexity_ratios(df_turns, rules_lib, optimal_k)

    if df_ratios.empty:
        tee.write("No tentative rules with confidence >= 5 found.\n")
        return

    tee.write(f"Analyzed {len(df_ratios)} tentative rules with confidence >= 5\n")
    tee.write(f"Using optimal k = {optimal_k:.3f} for aggregated complexity\n\n")

    # Compute per-model statistics
    df_stats = compute_model_complexity_stats(df_ratios)

    if df_stats.empty:
        tee.write("Could not compute model statistics.\n")
        return

    # Display statistics
    tee.write("Complexity Ratio by Model:\n")
    tee.write("(Ratio = Tentative Complexity / Actual Complexity)\n\n")

    display_df = df_stats.copy()
    display_df["median_ratio"] = display_df["median_ratio"].round(3)
    display_df["q25"] = display_df["q25"].round(3)
    display_df["q75"] = display_df["q75"].round(3)
    display_df.columns = ["Model", "Median", "Q25", "Q75", "Count"]
    tee.write(display_df.to_string(index=False) + "\n\n")

    # Interpretation
    tee.write("Interpretation:\n")
    tee.write("  - Ratio > 1: Model tends to overcomplicate rules\n")
    tee.write("  - Ratio < 1: Model tends to oversimplify rules\n")
    tee.write("  - Ratio ≈ 1: Model matches actual rule complexity\n\n")

    # Find extremes
    most_overcomplicating = df_stats.iloc[0]
    least_overcomplicating = df_stats.iloc[-1]

    high_model = most_overcomplicating["model"]
    high_ratio = most_overcomplicating["median_ratio"]
    low_model = least_overcomplicating["model"]
    low_ratio = least_overcomplicating["median_ratio"]
    tee.write(f"Highest median: {high_model} ({high_ratio:.3f})\n")
    tee.write(f"Lowest median: {low_model} ({low_ratio:.3f})\n\n")

    # Generate plot
    png_path, json_path = plot_complexity_ratio(df_stats, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
