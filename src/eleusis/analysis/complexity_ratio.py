"""Analysis of complexity ratio: tentative rule complexity vs actual rule complexity."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def compute_complexity_ratios(
    df_turns: pd.DataFrame,
    rules_lib: dict,
    optimal_k: float,
) -> pd.DataFrame:
    """Compute complexity ratio for each turn with confidence >= 5.

    Complexity ratio = (tentative rule complexity) / (actual rule complexity)

    Returns DataFrame with columns: model, round_number, turn_number,
    tentative_complexity, actual_complexity, complexity_ratio
    """
    # Filter to turns with tentative complexity data and confidence >= 5
    df = df_turns[
        df_turns["tentative_node_count"].notna() &
        df_turns["tentative_cyclomatic"].notna() &
        (df_turns["confidence_level"] >= 5)
    ].copy()

    if df.empty:
        return pd.DataFrame()

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

    if df.empty:
        return pd.DataFrame()

    # Compute complexity ratio (avoid division by zero)
    df["complexity_ratio"] = df["tentative_complexity"] / df["actual_complexity"].replace(0, np.nan)
    df = df.dropna(subset=["complexity_ratio"])

    return df[["model", "round_number", "turn_number", "confidence_level",
               "tentative_complexity", "actual_complexity", "complexity_ratio"]]


def compute_model_complexity_stats(df_ratios: pd.DataFrame) -> pd.DataFrame:
    """Compute median and quartiles of complexity ratio per model.

    Returns DataFrame with columns: model, median_ratio, q25, q75, count
    """
    if df_ratios.empty:
        return pd.DataFrame()

    stats = df_ratios.groupby("model").agg(
        median_ratio=("complexity_ratio", "median"),
        q25=("complexity_ratio", lambda x: x.quantile(0.25)),
        q75=("complexity_ratio", lambda x: x.quantile(0.75)),
        count=("complexity_ratio", "count"),
    ).reset_index()

    # Sort by median_ratio descending
    stats = stats.sort_values("median_ratio", ascending=False)

    return stats


def plot_complexity_ratio(
    df_stats: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
) -> tuple[Path, Path]:
    """Generate bar chart with error bars for complexity ratio.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    model_metadata = load_model_metadata()

    models = df_stats["model"].tolist()
    medians = df_stats["median_ratio"].values
    q25s = df_stats["q25"].values
    q75s = df_stats["q75"].values
    counts = df_stats["count"].values

    # Prepare colors and open/closed status
    colors = []
    is_open_list = []

    for model_name in models:
        colors.append(get_model_color(model_name, model_colors))
        normalized_name = normalize_model_name(model_name)
        is_open = False
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            matches = (norm_key == normalized_name or
                       norm_key in normalized_name or
                       normalized_name in norm_key)
            if matches:
                is_open = meta["is_open"]
                break
        is_open_list.append(is_open)

    y_pos = np.arange(len(models))

    # Draw markers with error bars (asymmetric for quartiles)
    for i, (model, median, q25, q75, color, is_open, count) in enumerate(
        zip(models, medians, q25s, q75s, colors, is_open_list, counts)
    ):
        marker = "o" if not is_open else "s"  # circle for closed, square for open
        facecolor = color if not is_open else "white"
        # Asymmetric error bars: [[lower_errors], [upper_errors]]
        xerr = [[median - q25], [q75 - median]]
        ax.errorbar(
            median, i, xerr=xerr,
            fmt=marker, markersize=10, markerfacecolor=facecolor,
            markeredgecolor=color, markeredgewidth=2,
            ecolor=color, elinewidth=1.5, capsize=4, capthick=1.5
        )

        # Add ratio label
        label_x = q75 + 0.08
        ax.text(label_x, i, f"{median:.2f} (n={count})", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Complexity Ratio (Tentative / Actual)", fontsize=11)
    ax.set_title("Rule Complexity Ratio by Model\n(Tentative rules with confidence ≥ 5)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    # Add vertical line at ratio = 1 (perfect match)
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.7, label="Ratio = 1 (exact match)")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="gray", markersize=10, label="Closed model"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="gray", markeredgewidth=2, markersize=10, label="Open model"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add interpretation note
    ax.text(
        0.02, -0.08,
        ">1: Overcomplicates  |  <1: Oversimplifies  |  =1: Matches complexity",
        transform=ax.transAxes, fontsize=8, ha="left",
        style="italic", color="gray"
    )

    plt.tight_layout()

    # Prepare JSON export data
    plot_data = {
        "metadata": {
            "description": "Complexity ratio: tentative rule complexity / actual rule complexity",
            "confidence_threshold": 5,
            "interpretation": "Ratio > 1 means model overcomplicates, < 1 means oversimplifies",
        },
        "models": []
    }

    for i, model in enumerate(models):
        plot_data["models"].append({
            "name": model,
            "color": colors[i],
            "is_open": is_open_list[i],
            "median_ratio": float(medians[i]),
            "q25": float(q25s[i]),
            "q75": float(q75s[i]),
            "count": int(counts[i]),
        })

    # Save outputs
    png_path = output_folder / "complexity_ratio.png"
    json_path = output_folder / "complexity_ratio.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_complexity_ratio(
    df_turns: pd.DataFrame,
    rules_lib: dict,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
    optimal_k: float,
):
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

    high_model = most_overcomplicating['model']
    high_ratio = most_overcomplicating['median_ratio']
    low_model = least_overcomplicating['model']
    low_ratio = least_overcomplicating['median_ratio']
    tee.write(f"Highest median: {high_model} ({high_ratio:.3f})\n")
    tee.write(f"Lowest median: {low_model} ({low_ratio:.3f})\n\n")

    # Generate plot
    png_path, json_path = plot_complexity_ratio(df_stats, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
