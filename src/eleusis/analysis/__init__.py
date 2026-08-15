"""Analysis and report generation for Eleusis benchmark results."""

from eleusis.analysis.benchmark_report import analyze_folder
from eleusis.analysis.colors import load_model_colors
from eleusis.analysis.loader import (
    build_rounds_dataframe,
    build_rules_lookup,
    build_turns_dataframe,
    load_results,
)

__all__ = [
    "analyze_folder",
    "build_rounds_dataframe",
    "build_rules_lookup",
    "build_turns_dataframe",
    "load_model_colors",
    "load_results",
]
