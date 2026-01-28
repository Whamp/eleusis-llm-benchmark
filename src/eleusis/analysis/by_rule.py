"""By-rule analysis showing score distribution across models."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .colors import get_color_map
from .utils import TeeWriter, setup_matplotlib_style

logger = logging.getLogger(__name__)

# Default K for aggregated complexity if not computed
DEFAULT_K = 0.1


def compute_rule_complexity(rules_lib: dict, optimal_k: float = DEFAULT_K) -> dict[str, dict]:
    """Compute aggregated complexity for each rule.

    Returns dict mapping rule_description to {name, complexity}.
    """
    rule_info = {}
    for desc, rule in rules_lib.items():
        cc = rule.get("cyclomatic_complexity")
        nc = rule.get("node_count")
        name = rule.get("name", desc[:30])

        if cc is not None and nc is not None:
            complexity = cc + optimal_k * nc
        else:
            complexity = None

        rule_info[desc] = {
            "name": name,
            "complexity": complexity,
            "cyclomatic_complexity": cc,
            "node_count": nc,
        }
    return rule_info


def plot_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    rules_lib: dict,
    output_folder: Path,
    optimal_k: float = DEFAULT_K,
) -> tuple[Path, Path]:
    """Generate scatter plot showing scores per rule with complexity heatmap.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()

    # Get rule complexity info
    rule_info = compute_rule_complexity(rules_lib, optimal_k)

    # Get unique rules present in data
    rules_in_data = df_rounds["rule_description"].unique()

    # Compute average score per rule
    rule_avg_scores = df_rounds.groupby("rule_description")["score"].mean()

    # Compute success rate per rule
    rule_success_rates = df_rounds.groupby("rule_description")["success"].mean()

    # Build list of (description, name, complexity, avg_score, success_rate) for rules in data
    rule_data = []
    for desc in rules_in_data:
        info = rule_info.get(desc, {"name": desc[:30], "complexity": None, "cyclomatic_complexity": None, "node_count": None})
        rule_data.append({
            "description": desc,
            "name": info["name"],
            "complexity": info["complexity"],
            "cyclomatic_complexity": info.get("cyclomatic_complexity"),
            "node_count": info.get("node_count"),
            "avg_score": rule_avg_scores.get(desc, 0),
            "success_rate": rule_success_rates.get(desc, 0),
        })

    # Sort by average score (highest first)
    rule_data.sort(key=lambda x: x["avg_score"], reverse=True)

    # Create mapping from description to y-position
    rule_order = {r["description"]: i for i, r in enumerate(rule_data)}
    rule_names = [r["name"] for r in rule_data]
    complexities = [r["complexity"] for r in rule_data]

    # Get models and colors
    models = df_rounds["model"].unique()
    color_map = get_color_map(models.tolist(), model_colors)

    # Figure with two subplots sharing y-axis for perfect alignment
    fig_height = max(6, len(rule_data) * 0.35)
    fig, (ax_heat, ax_scatter) = plt.subplots(
        1, 2, figsize=(12, fig_height), sharey=True,
        gridspec_kw={"width_ratios": [0.12, 1], "wspace": 0.01}
    )

    # --- Complexity heatmap (left) ---
    valid_complexities = [c for c in complexities if c is not None]
    if valid_complexities:
        vmin, vmax = min(valid_complexities), max(valid_complexities)
    else:
        vmin, vmax = 0, 1

    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, comp in enumerate(complexities):
        if comp is not None:
            color = cmap(norm(comp))
            # Choose text color based on background brightness
            text_color = "white" if norm(comp) > 0.5 else "black"
            comp_text = f"{comp:.1f}"
        else:
            color = "lightgray"
            text_color = "black"
            comp_text = "?"
        ax_heat.barh(i, 1, color=color, height=0.8)
        ax_heat.text(0.5, i, comp_text, ha="center", va="center",
                     fontsize=8, color=text_color, fontweight="bold")

    ax_heat.set_xlim(0, 1)
    ax_heat.set_yticks(range(len(rule_data)))
    ax_heat.set_yticklabels(rule_names, fontsize=9)
    ax_heat.set_xticks([])
    ax_heat.invert_yaxis()
    ax_heat.set_frame_on(False)

    # --- Scatter plot (right) ---
    for model in models:
        model_data = df_rounds[df_rounds["model"] == model]
        y_positions = model_data["rule_description"].map(rule_order)

        # Add small jitter to avoid overlapping points
        jitter = np.random.uniform(-0.15, 0.15, len(model_data))

        ax_scatter.scatter(
            model_data["score"],
            y_positions + jitter,
            c=color_map[model],
            s=60,
            alpha=0.7,
            label=model,
            edgecolors="white",
            linewidths=0.5,
        )

    ax_scatter.set_xlabel("Score", fontsize=11)
    ax_scatter.set_title("Score by Rule (ordered by avg score, best at top)", fontsize=12)

    # Add vertical line at score=0
    ax_scatter.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Legend outside plot
    ax_scatter.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    # --- Prepare JSON data ---
    # Build scores_by_model for each rule
    rules_json = []
    for r in rule_data:
        desc = r["description"]
        rule_df = df_rounds[df_rounds["rule_description"] == desc]
        scores_by_model = {}
        for model in models:
            model_scores = rule_df[rule_df["model"] == model]["score"].tolist()
            scores_by_model[model] = model_scores
        rules_json.append({
            "name": r["name"],
            "description": r["description"],
            "cyclomatic_complexity": r["cyclomatic_complexity"],
            "node_count": r["node_count"],
            "aggregated_complexity": r["complexity"],
            "avg_score": r["avg_score"],
            "success_rate": r["success_rate"],
            "scores_by_model": scores_by_model,
        })

    plot_data = {
        "rules": rules_json,
        "models": list(models),
        "metadata": {
            "optimal_k": optimal_k,
            "complexity_formula": f"cyclomatic + {optimal_k:.2f} * node_count",
            "ordering": "descending_avg_score",
        }
    }

    # Save outputs
    png_path = output_folder / "by_rule.png"
    json_path = output_folder / "by_rule.json"

    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {png_path}")

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_by_rule(
    df_rounds: pd.DataFrame,
    model_colors: dict[str, str],
    rules_lib: dict,
    output_folder: Path,
    tee: TeeWriter,
    optimal_k: float = DEFAULT_K,
):
    """Run by-rule analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("BY-RULE ANALYSIS\n")
    tee.write("=" * 60 + "\n\n")

    # Summary stats per rule
    rule_stats = df_rounds.groupby("rule_description").agg(
        count=("score", "count"),
        avg_score=("score", "mean"),
        std_score=("score", "std"),
        success_rate=("success", "mean"),
    ).sort_values("avg_score", ascending=False).reset_index()

    tee.write("Score by rule (sorted by avg_score):\n")
    tee.write(rule_stats.to_string(index=False) + "\n\n")

    # Generate plot
    png_path, json_path = plot_by_rule(
        df_rounds, model_colors, rules_lib, output_folder, optimal_k
    )
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")
