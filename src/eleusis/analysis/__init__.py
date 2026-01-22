"""Analysis module for Eleusis LLM benchmark results."""

import logging
from pathlib import Path

from .basic_metrics import analyze_basic_metrics
from .by_rule import analyze_by_rule
from .colors import load_model_colors
from .complexity import analyze_complexity
from .loader import build_rounds_dataframe, build_rules_lookup, build_turns_dataframe, load_results
from .per_model import generate_per_model_reports
from .utils import TeeWriter

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_folder",
    "load_results",
    "build_rules_lookup",
    "build_rounds_dataframe",
    "build_turns_dataframe",
    "load_model_colors",
]


def analyze_folder(folder: Path):
    """Main entry point - produces all outputs within folder.

    Args:
        folder: Path to results folder containing solo_evaluation_* subfolders.
                All outputs will be saved directly into this folder.
    """
    # Setup output
    tee = TeeWriter(folder / "summary.txt")

    def out(text: str):
        tee.write(text + "\n")

    out("=" * 60)
    out("ELEUSIS RESULTS ANALYSIS")
    out("=" * 60)
    out(f"\nAnalyzing: {folder}")

    # Load data
    out("\nLoading results...")
    results, folder_names = load_results(folder)
    if not results:
        out("No results found!")
        tee.close()
        return

    out(f"Loaded {len(results)} evaluation runs:")
    for name in folder_names:
        out(f"  - {name}")

    # Build lookup tables
    rules_lib = build_rules_lookup(results)
    out(f"\nExtracted {len(rules_lib)} unique rules from results files")

    # Build DataFrames
    df_rounds = build_rounds_dataframe(results, rules_lib)
    df_turns = build_turns_dataframe(results)
    out(f"Built DataFrames: {len(df_rounds)} rounds, {len(df_turns)} turns")

    # Load model colors
    model_colors = load_model_colors()
    out(f"Loaded colors for {len(model_colors)} models")

    # Run analyses
    analyze_basic_metrics(df_rounds, df_turns, model_colors, folder, tee)
    analyze_by_rule(df_rounds, model_colors, rules_lib, folder, tee)

    # Complexity analysis returns enriched dataframe with optimal_k
    df_enriched = analyze_complexity(df_rounds, model_colors, folder, tee)

    # Extract optimal_k for per-model reports (from complexity analysis)
    # We need to re-compute since analyze_complexity doesn't return it
    from .complexity import find_optimal_k

    df_temp = df_enriched.copy()
    if "model_avg_score" not in df_temp.columns:
        model_avg = df_temp.groupby("model")["score"].mean()
        df_temp["model_avg_score"] = df_temp["model"].map(model_avg)
        df_temp["relative_score"] = df_temp["score"] / df_temp["model_avg_score"]
    optimal_k, _ = find_optimal_k(df_temp)

    # Per-model reports
    out("\n" + "=" * 60)
    out("PER-MODEL REPORTS")
    out("=" * 60 + "\n")
    paths = generate_per_model_reports(
        df_rounds, df_turns, rules_lib, model_colors, folder, optimal_k
    )
    for path in paths:
        out(f"Saved: {path}")

    # Final summary
    out("\n" + "=" * 60)
    out(f"Analysis complete! All outputs saved to: {folder}")
    out("=" * 60)

    tee.close()
