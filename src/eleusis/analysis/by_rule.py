"""By-rule analysis showing score distribution across models."""

import json
import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .colors import get_color_map
from .legacy_records import LegacyRecord, RuleLookup
from .utils import TeeWriter, setup_matplotlib_style

logger = logging.getLogger(__name__)

# Default K for aggregated complexity if not computed
DEFAULT_K = 0.1


def _rule_average_score(rule_data: LegacyRecord) -> float:
    """Return one rule's average floored score for deterministic sorting."""
    score = rule_data.get("avg_floored_score", 0)
    return float(score) if isinstance(score, (int, float, str)) else 0.0


def compute_rule_complexity(
    rules_lib: RuleLookup, optimal_k: float = DEFAULT_K
) -> RuleLookup:
    """Compute aggregated complexity for each rule.

    Returns dict mapping rule_description to {name, complexity}.
    """
    rule_info = {}
    for desc, rule in rules_lib.items():
        cc = rule.get("cyclomatic_complexity")
        nc = rule.get("node_count")
        name = rule.get("name", desc[:30])

        complexity = cc + optimal_k * nc if cc is not None and nc is not None else None

        rule_info[desc] = {
            "name": name,
            "complexity": complexity,
            "cyclomatic_complexity": cc,
            "node_count": nc,
        }
    return rule_info


def _build_rule_plot_records(
    df_rounds: pd.DataFrame, rule_info: RuleLookup
) -> list[LegacyRecord]:
    """Aggregate score, success, and complexity values for each observed rule."""
    average_scores = df_rounds.groupby("rule_description")["floored_score"].mean()
    success_rates = df_rounds.groupby("rule_description")["counting_success"].mean()
    records: list[LegacyRecord] = []
    for description in df_rounds["rule_description"].unique():
        info = rule_info.get(description, {})
        records.append(
            {
                "description": description,
                "name": info.get("name", description[:30]),
                "complexity": info.get("complexity"),
                "cyclomatic_complexity": info.get("cyclomatic_complexity"),
                "node_count": info.get("node_count"),
                "avg_floored_score": average_scores.get(description, 0),
                "success_rate": success_rates.get(description, 0),
            }
        )
    records.sort(key=_rule_average_score, reverse=True)
    return records


def _plot_rule_complexity_cells(ax: Axes, rule_data: list[LegacyRecord]) -> None:
    """Draw the complexity heatmap cells aligned with rule rows."""
    complexities = [record.get("complexity") for record in rule_data]
    valid = [value for value in complexities if isinstance(value, (int, float))]
    minimum, maximum = (min(valid), max(valid)) if valid else (0, 1)
    color_map = plt.get_cmap("YlOrRd")
    normalization = mcolors.Normalize(vmin=minimum, vmax=maximum)
    for index, complexity in enumerate(complexities):
        if isinstance(complexity, (int, float)):
            color = color_map(normalization(complexity))
            text_color = "white" if normalization(complexity) > 0.5 else "black"
            text = f"{complexity:.1f}"
        else:
            color, text_color, text = "lightgray", "black", "?"
        ax.barh(index, 1, color=color, height=0.8)
        ax.text(
            0.5,
            index,
            text,
            ha="center",
            va="center",
            fontsize=8,
            color=text_color,
            fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_yticks(range(len(rule_data)))
    ax.set_yticklabels([record["name"] for record in rule_data], fontsize=9)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_frame_on(False)


def _plot_rule_model_scores(
    ax: Axes,
    df_rounds: pd.DataFrame,
    models: list[str],
    rule_order: dict[str, int],
    color_map: dict[str, str],
) -> None:
    """Draw jittered model score points for each rule row."""
    for model in models:
        model_data = df_rounds[df_rounds["model"] == model]
        positions = model_data["rule_description"].map(rule_order)
        jitter = np.random.uniform(-0.15, 0.15, len(model_data))
        ax.scatter(
            model_data["floored_score"],
            positions + jitter,
            c=color_map[model],
            s=60,
            alpha=0.7,
            label=model,
            edgecolors="white",
            linewidths=0.5,
        )


def _serialize_rule_scores(
    rule_data: list[LegacyRecord],
    df_rounds: pd.DataFrame,
    models: list[str],
) -> list[LegacyRecord]:
    """Build per-rule JSON records including scores grouped by model."""
    records: list[LegacyRecord] = []
    for rule in rule_data:
        description = rule["description"]
        rule_rounds = df_rounds[df_rounds["rule_description"] == description]
        scores_by_model = {
            model: rule_rounds[rule_rounds["model"] == model]["floored_score"].tolist()
            for model in models
        }
        records.append(
            {
                "name": rule["name"],
                "description": description,
                "cyclomatic_complexity": rule["cyclomatic_complexity"],
                "node_count": rule["node_count"],
                "aggregated_complexity": rule["complexity"],
                "avg_floored_score": rule["avg_floored_score"],
                "success_rate": rule["success_rate"],
                "floored_scores_by_model": scores_by_model,
            }
        )
    return records


def plot_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    rules_lib: RuleLookup,
    output_folder: Path,
    optimal_k: float = DEFAULT_K,
) -> tuple[Path, Path]:
    """Generate scores per rule with an aligned complexity heatmap."""
    setup_matplotlib_style()
    df_rounds = df_rounds.copy()
    df_rounds["floored_score"] = df_rounds["score"].clip(lower=0)
    rule_data = _build_rule_plot_records(
        df_rounds, compute_rule_complexity(rules_lib, optimal_k)
    )
    rule_order = {record["description"]: i for i, record in enumerate(rule_data)}
    models = df_rounds["model"].unique().tolist()
    color_map = get_color_map(models, model_colors)

    figure_height = max(6, len(rule_data) * 0.35)
    figure, (complexity_axes, score_axes) = plt.subplots(
        1,
        2,
        figsize=(12, figure_height),
        sharey=True,
        gridspec_kw={"width_ratios": [0.12, 1], "wspace": 0.01},
    )
    _plot_rule_complexity_cells(complexity_axes, rule_data)
    _plot_rule_model_scores(score_axes, df_rounds, models, rule_order, color_map)
    score_axes.set_xlabel("Floored Score", fontsize=11)
    score_axes.set_title(
        "Floored Score by Rule (ordered by avg, best at top)", fontsize=12
    )
    score_axes.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.9
    )

    plot_data = {
        "rules": _serialize_rule_scores(rule_data, df_rounds, models),
        "models": models,
        "metadata": {
            "optimal_k": optimal_k,
            "complexity_formula": f"cyclomatic + {optimal_k:.2f} * node_count",
            "ordering": "descending_avg_score",
        },
    }
    png_path = output_folder / "by_rule.png"
    json_path = output_folder / "by_rule.json"
    figure.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    logger.info(f"Saved: {png_path}")
    with json_path.open("w") as output_file:
        json.dump(plot_data, output_file, indent=2)
    logger.info(f"Saved: {json_path}")
    return png_path, json_path


def analyze_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    rules_lib: RuleLookup,
    output_folder: Path,
    tee: TeeWriter,
    optimal_k: float = DEFAULT_K,
) -> None:
    """Run by-rule analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BY-RULE ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Summary stats per rule (using floored_score and counting_success)
    df_rounds = df_rounds.copy()
    df_rounds["floored_score"] = df_rounds["score"].clip(lower=0)
    rule_stats = (
        df_rounds.groupby("rule_description")
        .agg(
            count=("floored_score", "count"),
            avg_floored_score=("floored_score", "mean"),
            std_floored_score=("floored_score", "std"),
            success_rate=("counting_success", "mean"),
        )
        .sort_values("avg_floored_score", ascending=False)
        .reset_index()
    )

    tee.write("Score by rule (sorted by avg_floored_score):\n")
    tee.write(rule_stats.to_string(index=False) + "\n\n")

    # Generate plot
    png_path, json_path = plot_by_rule(
        df_rounds, model_colors, rules_lib, output_folder, optimal_k
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
