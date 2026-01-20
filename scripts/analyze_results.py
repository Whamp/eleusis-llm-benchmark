"""Analyze evaluation results."""

import argparse
import logging
from pathlib import Path

from eleusis.analysis import analyze_folder

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Eleusis evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  uv run python scripts/analyze_results.py results/20260119_2x7_runs_5_models

Outputs (saved in the input folder):
  - summary.txt           Text summary of all analyses
  - basic_metrics.csv     Per-model metrics table
  - basic_metrics.png     Basic comparison charts
  - complexity_analysis.png  Complexity vs performance charts
  - <model_name>.png      Per-model detailed report
""",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Results folder containing solo_evaluation_* subfolders",
    )
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: Folder not found: {args.folder}")
        return 1

    if not args.folder.is_dir():
        print(f"Error: Not a directory: {args.folder}")
        return 1

    analyze_folder(args.folder)
    return 0


if __name__ == "__main__":
    exit(main())
