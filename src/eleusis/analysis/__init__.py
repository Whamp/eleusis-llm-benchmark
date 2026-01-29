"""Analysis module for Eleusis LLM benchmark results."""

import logging
from pathlib import Path

from .basic_metrics import analyze_basic_metrics
from .by_rule import analyze_by_rule
from .colors import load_model_colors
from .complexity import analyze_complexity
from .complexity_ratio import analyze_complexity_ratio
from .excess_caution import analyze_excess_caution
from .loader import build_rounds_dataframe, build_rules_lookup, build_turns_dataframe, load_results
from .per_model import generate_per_model_reports
from .reckless_guessing import analyze_reckless_guessing
from .tokens_by_turn import analyze_tokens_by_turn
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

    # Complexity analysis first to compute optimal_k
    df_enriched = analyze_complexity(df_rounds, model_colors, folder, tee)

    # Extract optimal_k (need to re-compute since analyze_complexity doesn't return it)
    from .complexity import find_optimal_k

    rule_stats = df_enriched.groupby("rule_description").agg(
        times_played=("success", "count"),
        times_found=("success", "sum"),
        cyclomatic_complexity=("cyclomatic_complexity", "first"),
        node_count=("node_count", "first"),
    ).reset_index()
    rule_stats["success_rate"] = rule_stats["times_found"] / rule_stats["times_played"]
    optimal_k, _ = find_optimal_k(rule_stats)

    # By-rule analysis uses optimal_k
    analyze_by_rule(df_rounds, model_colors, rules_lib, folder, tee, optimal_k)
    analyze_excess_caution(df_turns, df_rounds, model_colors, folder, tee)
    analyze_reckless_guessing(df_turns, model_colors, folder, tee)
    analyze_complexity_ratio(df_turns, rules_lib, model_colors, folder, tee, optimal_k)
    analyze_tokens_by_turn(df_turns, model_colors, folder, tee)

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
