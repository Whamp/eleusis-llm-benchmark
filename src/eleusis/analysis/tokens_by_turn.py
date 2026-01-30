"""Analysis of output tokens by turn index."""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .colors import get_model_color, load_model_metadata, normalize_model_name
from .utils import TeeWriter, save_figure, setup_matplotlib_style

logger = logging.getLogger(__name__)


def plot_tokens_by_turn(
    df_turns: pd.DataFrame, model_colors: dict[str, str], output_folder: Path
) -> tuple[Path, Path]:
    """Plot average output tokens as a function of turn index for each model.

    Returns (png_path, json_path).
    """
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Load metadata for open/closed distinction
    model_metadata = load_model_metadata()

    # Prepare data for JSON export
    plot_data = {
        "models": [],
        "metadata": {
            "x_axis": "turn_number",
            "y_axis": "avg_output_tokens",
            "description": "Average output tokens by turn index",
        },
    }

    models = sorted(df_turns["model"].unique())

    for model_name in models:
        model_turns = df_turns[df_turns["model"] == model_name]

        # Group by turn number and compute mean output tokens
        tokens_by_turn = model_turns.groupby("turn_number")["output_tokens"].agg(
            ["mean", "count"]
        ).reset_index()
        tokens_by_turn.columns = ["turn_number", "avg_tokens", "sample_count"]

        if len(tokens_by_turn) == 0:
            continue

        color = get_model_color(model_name, model_colors)

        # Determine if open model
        is_open = False
        provider = "unknown"
        normalized_name = normalize_model_name(model_name)
        for key, meta in model_metadata.items():
            norm_key = normalize_model_name(key)
            match = (norm_key == normalized_name or norm_key in normalized_name
                     or normalized_name in norm_key)
            if match:
                is_open = meta["is_open"]
                provider = meta["provider"]
                break

        # Line style: dashed for open models, solid for closed
        linestyle = "--" if is_open else "-"

        # Plot line
        ax.plot(
            tokens_by_turn["turn_number"],
            tokens_by_turn["avg_tokens"],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=4,
            label=model_name,
            alpha=0.8,
        )

        # Store data for JSON
        plot_data["models"].append({
            "name": model_name,
            "color": color,
            "is_open": is_open,
            "provider": provider,
            "tokens_by_turn": [
                {
                    "turn_number": int(row["turn_number"]),
                    "avg_output_tokens": float(row["avg_tokens"]),
                    "sample_count": int(row["sample_count"]),
                }
                for _, row in tokens_by_turn.iterrows()
            ],
        })

    ax.set_xlabel("Turn Number", fontsize=11)
    ax.set_ylabel("Average Output Tokens", fontsize=11)
    ax.set_title("Output Tokens by Turn Index", fontsize=13, fontweight="bold")

    # Legend
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Add note about line styles
    ax.text(
        0.98, 0.98, "Solid = Closed model, Dashed = Open model",
        transform=ax.transAxes, fontsize=8, ha="right", va="top",
        style="italic", color="gray",
    )

    # Save outputs
    png_path = output_folder / "tokens_by_turn.png"
    json_path = output_folder / "tokens_by_turn.json"

    save_figure(fig, png_path)

    with open(json_path, "w") as f:
        json.dump(plot_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    return png_path, json_path


def analyze_tokens_by_turn(
    df_turns: pd.DataFrame,
    model_colors: dict[str, str],
    output_folder: Path,
    tee: TeeWriter,
):
    """Run tokens by turn analysis and save outputs."""
    tee.write("\n" + "=" * 60 + "\n")
    tee.write("OUTPUT TOKENS BY TURN\n")
    tee.write("=" * 60 + "\n\n")

    # Generate the plot
    png_path, json_path = plot_tokens_by_turn(df_turns, model_colors, output_folder)
    tee.write(f"Saved: {png_path}\n")
    tee.write(f"Saved: {json_path}\n")

    # Print summary statistics
    models = sorted(df_turns["model"].unique())
    tee.write("\nTokens trend summary (early vs late turns):\n")

    for model_name in models:
        model_turns = df_turns[df_turns["model"] == model_name]
        max_turn = model_turns["turn_number"].max()

        # Compare early turns (1-5) vs late turns (last 5)
        early = model_turns[model_turns["turn_number"] <= 5]["output_tokens"].mean()
        late_start = max(6, max_turn - 4)
        late = model_turns[model_turns["turn_number"] >= late_start]["output_tokens"].mean()

        if pd.notna(early) and pd.notna(late) and early > 0:
            change_pct = ((late - early) / early) * 100
            tee.write(f"  {model_name}: early={early:.0f}, late={late:.0f} ({change_pct:+.1f}%)\n")
