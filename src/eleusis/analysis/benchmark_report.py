"""Analysis module for Eleusis LLM benchmark results."""

import logging
from pathlib import Path

import pandas as pd

from .basic_metrics import analyze_basic_metrics
from .by_rule import analyze_by_rule
from .colors import load_model_colors
from .complexity import analyze_complexity
from .complexity_ratio import analyze_complexity_ratio, compute_complexity_ratios
from .excess_caution import analyze_excess_caution
from .legacy_records import LegacyRecord, LegacyResults
from .loader import (
    build_rounds_dataframe,
    build_rules_lookup,
    build_turns_dataframe,
    load_results,
)
from .per_model import generate_per_model_reports
from .reckless_guessing import analyze_reckless_guessing
from .tokens_by_turn import analyze_tokens_by_turn
from .utils import TeeWriter

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_folder",
    "build_rounds_dataframe",
    "build_rules_lookup",
    "build_turns_dataframe",
    "load_model_colors",
    "load_results",
]


def analyze_folder(folder: Path) -> None:
    """Main entry point - produces all outputs within folder.

    Args:
        folder: Path to results folder containing solo_evaluation_* subfolders.
                All outputs will be saved directly into this folder.
    """
    tee = TeeWriter(folder / "summary.txt")

    def out(text: str) -> None:
        tee.write(text + "\n")

    out("=" * 60)
    out("ELEUSIS RESULTS ANALYSIS")
    out("=" * 60)
    out(f"\nAnalyzing: {folder}")

    out("\nLoading results...")
    results, folder_names = load_results(folder)
    if not results:
        out("No results found!")
        tee.close()
        return

    out(f"Loaded {len(results)} evaluation runs:")
    for name in folder_names:
        out(f"  - {name}")

    rules_lib = build_rules_lookup(results)
    out(f"\nExtracted {len(rules_lib)} unique rules from results files")

    df_rounds = build_rounds_dataframe(results, rules_lib)
    df_turns = build_turns_dataframe(results)
    out(f"Built DataFrames: {len(df_rounds)} rounds, {len(df_turns)} turns")

    model_colors = load_model_colors()
    out(f"Loaded colors for {len(model_colors)} models")

    analyze_basic_metrics(df_rounds, df_turns, model_colors, folder, tee)

    # Complexity analysis first to compute optimal_k
    df_enriched = analyze_complexity(df_rounds, model_colors, folder, tee)

    # Extract optimal_k (need to re-compute since analyze_complexity doesn't return it)
    from .complexity import find_optimal_k

    rule_stats = (
        df_enriched.groupby("rule_description")
        .agg(
            times_played=("success", "count"),
            times_found=("success", "sum"),
            cyclomatic_complexity=("cyclomatic_complexity", "first"),
            node_count=("node_count", "first"),
        )
        .reset_index()
    )
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

    # Generate HTML report
    _generate_html_report(results, df_rounds, df_turns, folder)


def _compute_report_optimal_k(df_rounds: pd.DataFrame) -> float:
    """Fit the complexity aggregation coefficient used by report ratios."""
    from .complexity import find_optimal_k

    rule_stats = (
        df_rounds.groupby("rule_description")
        .agg(
            times_played=("success", "count"),
            times_found=("success", "sum"),
            cyclomatic_complexity=("cyclomatic_complexity", "first"),
            node_count=("node_count", "first"),
        )
        .reset_index()
    )
    rule_stats["success_rate"] = rule_stats["times_found"] / rule_stats["times_played"]
    optimal_k, _correlation = find_optimal_k(rule_stats)
    return optimal_k


def _build_report_model_data(
    results: LegacyResults,
    df_rounds: pd.DataFrame,
    df_turns: pd.DataFrame,
) -> list[LegacyRecord]:
    """Aggregate model score, behavior, token, and complexity report records."""
    rounds = df_rounds.copy()
    rounds["floored_score"] = rounds["score"].clip(lower=0)
    metrics = (
        rounds.groupby("model")
        .agg(
            rounds_played=("success", "count"),
            avg_floored_score=("floored_score", "mean"),
            success_rate=("counting_success", "mean"),
            double_down_rate=(
                "counting_failed_guesses",
                lambda values: (values >= 2).sum() / len(values) * 100,
            ),
        )
        .reset_index()
    )
    tokens_per_turn = df_turns.groupby("model")["output_tokens"].mean()
    ratios = compute_complexity_ratios(
        df_turns,
        build_rules_lookup(results),
        _compute_report_optimal_k(rounds),
    )
    complexity_ratio = (
        ratios.groupby("model")["complexity_ratio"].mean()
        if not ratios.empty
        else pd.Series(dtype=float)
    )
    models_data: list[LegacyRecord] = []
    for _, row in metrics.iterrows():
        model_name = row["model"]
        models_data.append(
            {
                "name": model_name,
                "avg_score": row["avg_floored_score"],
                "success_rate": row["success_rate"],
                "rounds": int(row["rounds_played"]),
                "is_in_progress": "in-progress" in model_name.lower(),
                "double_down_rate": row["double_down_rate"],
                "tokens_per_turn": tokens_per_turn.get(model_name, 0),
                "complexity_ratio": complexity_ratio.get(model_name, 1.0),
            }
        )
    models_data.sort(key=lambda record: record["avg_score"], reverse=True)
    return models_data


def _generate_html_report(
    results: LegacyResults,
    df_rounds: pd.DataFrame,
    df_turns: pd.DataFrame,
    folder: Path,
) -> None:
    """Generate the standalone HTML report from aggregated model data."""
    models_data = _build_report_model_data(results, df_rounds, df_turns)
    total_rules = len(df_rounds["rule_description"].unique())
    html_content = _build_html_template(models_data, total_rules, folder.name)
    output_path = folder / "report.html"
    output_path.write_text(html_content)
    logger.info(f"Saved: {output_path}")


def _render_html_template(values: list[str]) -> str:
    """Insert preformatted report values into the static HTML resource."""
    template_path = Path(__file__).with_name("benchmark_report.html")
    html = template_path.read_text()
    for index, value in enumerate(values):
        html = html.replace(f"@@REPORT_VALUE_{index}@@", value)
    return html


def _build_report_table_rows(models_data: list[LegacyRecord]) -> str:
    """Build sortable leaderboard table rows for report models."""
    rows = ""
    for rank, model in enumerate(models_data, 1):
        css_class = "in-progress" if model["is_in_progress"] else ""
        tokens = model.get("tokens_per_turn", 0)
        double_down = model.get("double_down_rate", 0)
        complexity = model.get("complexity_ratio", 1.0)
        rows += (
            f'                <tr class="{css_class}"'
            f' data-model="{model["name"].replace(chr(34), chr(39))}"'
            f' data-rank="{rank}"'
            f' data-score="{model["avg_score"]:.2f}"'
            f' data-success="{model["success_rate"]:.2f}"'
            f' data-rounds="{model["rounds"]}" data-tokens="{tokens:.0f}"'
            f' data-dd="{double_down:.1f}"'
            f' data-cr="{complexity:.2f}"><td>{rank}</td>'
            f"<td>{model['name']}</td><td>{model['avg_score']:.2f}</td>"
            f"<td>{model['success_rate']:.0%}</td><td>{tokens:,.0f}</td>"
            f"<td>{double_down:.1f}%</td><td>{complexity:.2f}</td></tr>\n"
        )
    return rows


def _report_reference_model(
    models_data: list[LegacyRecord],
) -> tuple[int | None, LegacyRecord | None]:
    """Choose the in-progress model, or the leading model when none is active."""
    for rank, model in enumerate(models_data, 1):
        if model["is_in_progress"]:
            return rank, model
    return None, models_data[0] if models_data else None


def _complexity_label(ratio: float) -> str:
    """Describe a model's tentative-to-actual complexity ratio."""
    if ratio > 1.5:
        return "overcomplication"
    if ratio > 1.0:
        return "slight overcomplication"
    if ratio > 0.8:
        return "balanced"
    return "simplification"


def _token_usage_label(tokens_per_turn: float) -> str:
    """Describe average generated tokens per turn."""
    if tokens_per_turn < 4000:
        return "efficient"
    if tokens_per_turn < 8000:
        return "moderate"
    return "high"


def _double_down_label(rate: float) -> str:
    """Describe the model's immediate repeat-guess rate."""
    if rate < 15:
        return "very cautious"
    if rate < 30:
        return "cautious"
    return "aggressive"


def _build_model_chart_markup(models_data: list[LegacyRecord]) -> tuple[str, str]:
    """Build per-model chart cards and matching lightbox markup."""
    cards = ""
    lightboxes = ""
    for model in models_data:
        safe_name = (
            model["name"].lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        )
        filename = f"model_{safe_name}.png"
        cards += (
            '            <div class="chart-card">\n'
            f"                <h3>{model['name']} ({model['avg_score']:.1f})</h3>\n"
            f'                <a href="#lb-model-{safe_name[:30]}">'
            f'<img src="{filename}" alt="{model["name"]}"></a>\n'
            "            </div>\n"
        )
        lightboxes += (
            f'    <div id="lb-model-{safe_name[:30]}" class="lightbox" '
            'onclick="history.back(); return false;">\n'
            '        <span class="lightbox-close">&times;</span>\n'
            f'        <img src="{filename}" alt="{model["name"]}">\n'
            "    </div>\n"
        )
    return cards, lightboxes


def _build_model_javascript_data(models_data: list[LegacyRecord]) -> str:
    """Build model records consumed by report-side filtering JavaScript."""
    records = ""
    for rank, model in enumerate(models_data, 1):
        safe_name = model["name"].replace('"', '\\"')
        tokens = model.get("tokens_per_turn", 0)
        double_down = model.get("double_down_rate", 0)
        complexity = model.get("complexity_ratio", 1.0)
        records += (
            f'            "{safe_name}": {{ rank: {rank}, avg_score:'
            f" {model['avg_score']:.4f},"
            f" success_rate: {model['success_rate']:.4f}, rounds: {model['rounds']},"
            f" is_in_progress: {str(model['is_in_progress']).lower()}, tokens:"
            f" {tokens:.0f}, double_down: {double_down:.2f}, complexity:"
            f" {complexity:.2f} }},\n"
        )
    return records


def _build_html_template(
    models_data: list[LegacyRecord], total_rules: int, folder_name: str
) -> str:
    """Render the standalone benchmark report HTML document."""
    in_progress_rank, reference_model = _report_reference_model(models_data)
    rank_display = in_progress_rank if in_progress_rank else (1 if models_data else 0)
    average_score = reference_model["avg_score"] if reference_model else 0
    success_rate = reference_model["success_rate"] if reference_model else 0
    tokens_per_turn = (
        reference_model.get("tokens_per_turn", 0) if reference_model else 0
    )
    double_down_rate = (
        reference_model.get("double_down_rate", 0) if reference_model else 0
    )
    complexity_ratio = (
        reference_model.get("complexity_ratio", 1.0) if reference_model else 1.0
    )
    model_cards, model_lightboxes = _build_model_chart_markup(models_data)
    highlight = " highlight" if in_progress_rank else ""
    return _render_html_template(
        [
            folder_name,
            folder_name,
            str(len(models_data)),
            str(total_rules),
            str(total_rules),
            str(total_rules),
            highlight,
            str(rank_display),
            str(len(models_data)),
            highlight,
            format(average_score, ".2f"),
            format(success_rate, ".0%"),
            format(tokens_per_turn, ",.0f"),
            _token_usage_label(tokens_per_turn),
            format(double_down_rate, ".1f"),
            _double_down_label(double_down_rate),
            format(complexity_ratio, ".2f"),
            _complexity_label(complexity_ratio),
            _build_report_table_rows(models_data),
            str(len(models_data)),
            model_cards,
            model_lightboxes,
            str(len(models_data)),
            str(total_rules),
            _build_model_javascript_data(models_data),
            models_data[0]["name"].replace(chr(34), chr(39)) if models_data else "",
        ]
    )
