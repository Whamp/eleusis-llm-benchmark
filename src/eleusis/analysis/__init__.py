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

    # Generate HTML report
    _generate_html_report(results, df_rounds, df_turns, folder)


def _generate_html_report(results, df_rounds, df_turns, folder):
    """Generate HTML report with all charts and metrics."""
    import json
    from pathlib import Path
    from .complexity import find_optimal_k

    # Compute optimal_k first (needed for complexity ratio)
    rule_stats = df_rounds.groupby("rule_description").agg(
        times_played=("success", "count"),
        times_found=("success", "sum"),
        cyclomatic_complexity=("cyclomatic_complexity", "first"),
        node_count=("node_count", "first"),
    ).reset_index()
    rule_stats["success_rate"] = rule_stats["times_found"] / rule_stats["times_played"]
    optimal_k, _ = find_optimal_k(rule_stats)

    # Compute basic metrics from rounds
    df_rounds = df_rounds.copy()
    df_rounds["floored_score"] = df_rounds["score"].clip(lower=0)

    metrics = df_rounds.groupby("model").agg(
        rounds_played=("success", "count"),
        avg_floored_score=("floored_score", "mean"),
        success_rate=("counting_success", "mean"),
        double_down_rate=("counting_failed_guesses", lambda x: (x >= 2).sum() / len(x) * 100),
    ).reset_index()

    # Compute tokens per turn from turns data
    tokens_per_turn = df_turns.groupby("model")["output_tokens"].mean()

    # Compute complexity ratio per model
    rules_lib = {}  # Build rules lookup for complexity computation
    for result in results:
        checkpoint = result.get("checkpoint", {})
        for rule in checkpoint.get("rules_library", []):
            desc = rule.get("description")
            if desc and desc not in rules_lib:
                rules_lib[desc] = rule

    df_complex = df_turns[
        df_turns["tentative_node_count"].notna() &
        df_turns["tentative_cyclomatic"].notna() &
        (df_turns["confidence_level"] >= 5)
    ].copy()

    if not df_complex.empty:
        df_complex["tentative_complexity"] = df_complex["tentative_cyclomatic"] + optimal_k * df_complex["tentative_node_count"]

        def get_actual_complexity(rule_desc):
            rule = rules_lib.get(rule_desc, {})
            cc = rule.get("cyclomatic_complexity")
            nc = rule.get("node_count")
            if cc is not None and nc is not None:
                return cc + optimal_k * nc
            return None

        df_complex["actual_complexity"] = df_complex["actual_rule"].apply(get_actual_complexity)
        df_complex = df_complex[df_complex["actual_complexity"].notna() & (df_complex["actual_complexity"] > 0)]
        df_complex["complexity_ratio"] = df_complex["tentative_complexity"] / df_complex["actual_complexity"]
        complexity_ratio = df_complex.groupby("model")["complexity_ratio"].mean()
    else:
        complexity_ratio = pd.Series(dtype=float)

    # Build model data with all metrics
    models_data = []
    for _, row in metrics.iterrows():
        model_name = row["model"]
        is_in_progress = "in-progress" in model_name.lower()
        cr = complexity_ratio.get(model_name, 1.0)
        models_data.append({
            "name": model_name,
            "avg_score": row["avg_floored_score"],
            "success_rate": row["success_rate"],
            "rounds": int(row["rounds_played"]),
            "is_in_progress": is_in_progress,
            "double_down_rate": row["double_down_rate"],
            "tokens_per_turn": tokens_per_turn.get(model_name, 0),
            "complexity_ratio": cr,
        })

    models_data.sort(key=lambda x: x["avg_score"], reverse=True)
    total_rules = len(df_rounds["rule_description"].unique())

    html_content = _build_html_template(models_data, total_rules, folder.name)

    output_path = folder / "report.html"
    with open(output_path, "w") as f:
        f.write(html_content)

    logger.info(f"Saved: {output_path}")


def _build_html_template(models_data, total_rules, folder_name):
    """Build HTML template string with refined dark theme."""

    # Build ranking rows with data attributes
    rows_html = ""
    for i, m in enumerate(models_data, 1):
        css_class = 'in-progress' if m["is_in_progress"] else ''
        tokens = m.get("tokens_per_turn", 0)
        dd_rate = m.get("double_down_rate", 0)
        cr = m.get("complexity_ratio", 1.0)
        rows_html += f'                <tr class="{css_class}" data-model="{m["name"].replace(chr(34), chr(39))}" data-rank="{i}" data-score="{m["avg_score"]:.2f}" data-success="{m["success_rate"]:.2f}" data-rounds="{m["rounds"]}" data-tokens="{tokens:.0f}" data-dd="{dd_rate:.1f}" data-cr="{cr:.2f}"><td>{i}</td><td>{m["name"]}</td><td>{m["avg_score"]:.2f}</td><td>{m["success_rate"]:.0%}</td><td>{tokens:,.0f}</td><td>{dd_rate:.1f}%</td><td>{cr:.2f}</td></tr>\n'

    # Find in-progress model and its rank
    in_progress_rank = None
    ref_model = None
    for i, m in enumerate(models_data, 1):
        if m["is_in_progress"]:
            in_progress_rank = i
            ref_model = m
            break
    if not ref_model and models_data:
        ref_model = models_data[0]

    rank_display = in_progress_rank if in_progress_rank else (1 if models_data else 0)
    avg_score = ref_model["avg_score"] if ref_model else 0
    success_rate = ref_model["success_rate"] if ref_model else 0
    tokens_per_turn = ref_model.get("tokens_per_turn", 0) if ref_model else 0
    double_down_rate = ref_model.get("double_down_rate", 0) if ref_model else 0
    complexity_ratio = ref_model.get("complexity_ratio", 1.0) if ref_model else 1.0

    # Determine labels
    if complexity_ratio > 1.5:
        complexity_label = "overcomplication"
    elif complexity_ratio > 1.0:
        complexity_label = "slight overcomplication"
    elif complexity_ratio > 0.8:
        complexity_label = "balanced"
    else:
        complexity_label = "simplification"

    if tokens_per_turn < 4000:
        tokens_label = "efficient"
    elif tokens_per_turn < 8000:
        tokens_label = "moderate"
    else:
        tokens_label = "high"

    if double_down_rate < 15:
        dd_label = "very cautious"
    elif double_down_rate < 30:
        dd_label = "cautious"
    else:
        dd_label = "aggressive"

    # Build per-model chart cards
    per_model_cards = ""
    for m in models_data:
        safe_name = m["name"].lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        filename = f"model_{safe_name}.png"
        per_model_cards += f'''            <div class="chart-card">
                <h3>{m["name"]} ({m["avg_score"]:.1f})</h3>
                <a href="#lb-model-{safe_name[:30]}"><img src="{filename}" alt="{m["name"]}"></a>
            </div>
'''

    # Build per-model lightboxes
    per_model_lightboxes = ""
    for m in models_data:
        safe_name = m["name"].lower().replace(" ", "_").replace(".", "_").replace("-", "_")
        filename = f"model_{safe_name}.png"
        per_model_lightboxes += f'''    <div id="lb-model-{safe_name[:30]}" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="{filename}" alt="{m["name"]}">
    </div>
'''

    # Build JavaScript data map
    model_data_js = ""
    for i, m in enumerate(models_data, 1):
        safe_name = m["name"].replace('"', '\\"')
        tokens = m.get("tokens_per_turn", 0)
        dd = m.get("double_down_rate", 0)
        cr = m.get("complexity_ratio", 1.0)
        model_data_js += f'            "{safe_name}": {{ rank: {i}, avg_score: {m["avg_score"]:.4f}, success_rate: {m["success_rate"]:.4f}, rounds: {m["rounds"]}, is_in_progress: {str(m["is_in_progress"]).lower()}, tokens: {tokens:.0f}, double_down: {dd:.2f}, complexity: {cr:.2f} }},\n'

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eleusis Benchmark Report - {folder_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border-subtle: #30363d;
            --border-accent: #3d444d;
            --text-primary: #f0f6fc;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-gold: #d4a853;
            --accent-gold-dim: #a88a4a;
            --accent-sage: #7ee787;
            --accent-rose: #f08888;
            --accent-blue: #79c0ff;
            --accent-purple: #d2a8ff;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 40px 24px; }}
        
        /* Header - Editorial Style */
        header {{
            position: relative;
            padding: 60px 0 40px;
            margin-bottom: 50px;
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-gold) 0%, var(--accent-gold-dim) 50%, transparent 100%);
        }}
        
        .header-kicker {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            color: var(--accent-gold);
            margin-bottom: 12px;
        }}
        
        h1 {{
            font-family: 'Crimson Pro', Georgia, serif;
            font-size: 3.2rem;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--text-primary);
            margin-bottom: 16px;
            line-height: 1.1;
        }}
        
        .subtitle {{
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}
        
        .subtitle strong {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        /* Summary Grid - Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 60px;
        }}
        
        .metric-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 24px;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--border-subtle);
            transition: background 0.3s ease;
        }}
        
        .metric-card:hover {{
            border-color: var(--border-accent);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }}
        
        .metric-card:hover::before {{
            background: var(--accent-gold);
        }}
        
        .metric-card.highlight::before {{
            background: var(--accent-gold);
        }}
        
        .metric-card.highlight {{
            background: var(--bg-tertiary);
            border-color: var(--accent-gold-dim);
        }}
        
        .metric-label {{
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        
        .metric-value {{
            font-family: 'Crimson Pro', Georgia, serif;
            font-size: 2.4rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }}
        
        .metric-sub {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}
        
        /* Section Headers */
        h2 {{
            font-family: 'Crimson Pro', Georgia, serif;
            font-size: 1.6rem;
            font-weight: 500;
            color: var(--text-primary);
            margin: 60px 0 24px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
            letter-spacing: -0.01em;
        }}
        
        /* Table - Minimal Dark */
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            margin-bottom: 20px;
        }}
        
        th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            padding: 16px;
            text-align: left;
            font-weight: 500;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-bottom: 1px solid var(--border-subtle);
        }}
        
        td {{
            padding: 14px 16px;
            border-bottom: 1px solid var(--border-subtle);
            font-size: 0.95rem;
            color: var(--text-secondary);
        }}
        
        td:first-child {{
            font-weight: 500;
            color: var(--accent-gold);
        }}
        
        td:nth-child(2) {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        tr.active {{
            background: var(--bg-tertiary);
        }}
        
        tr.active td:first-child {{
            border-left: 3px solid var(--accent-gold);
            padding-left: 13px;
        }}
        
        tr.in-progress {{
            background: var(--bg-tertiary);
        }}
        
        tr.in-progress td:nth-child(2) {{
            color: var(--accent-sage);
        }}
        
        tr.in-progress.active td:first-child {{
            border-left-color: var(--accent-sage);
        }}
        
        #ranking-table tbody tr {{
            cursor: pointer;
            transition: background 0.2s ease;
        }}
        
        /* Chart Grid */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
        }}
        
        .chart-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-subtle);
            transition: all 0.3s ease;
        }}
        
        .chart-card:hover {{
            border-color: var(--border-accent);
            box-shadow: 0 4px 20px rgba(0,0,0,0.25);
        }}
        
        .chart-card h3 {{
            background: var(--bg-tertiary);
            padding: 16px 20px;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
            border-bottom: 1px solid var(--border-subtle);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        
        .chart-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: zoom-in;
            transition: transform 0.3s ease;
        }}
        
        .chart-card:hover img {{
            transform: scale(1.01);
        }}
        
        .chart-card a {{
            display: block;
            text-decoration: none;
        }}
        
        /* Lightbox */
        .lightbox {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(13, 17, 23, 0.98);
            backdrop-filter: blur(8px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        }}
        
        .lightbox:target {{
            display: flex;
        }}
        
        .lightbox img {{
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }}
        
        .lightbox-close {{
            position: absolute;
            top: 30px;
            right: 40px;
            color: var(--text-secondary);
            font-size: 32px;
            cursor: pointer;
            opacity: 0.7;
            z-index: 1001;
            width: 48px;
            height: 48px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            transition: all 0.2s ease;
        }}
        
        .lightbox-close:hover {{
            opacity: 1;
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }}
        
        /* Footer */
        footer {{
            margin-top: 80px;
            padding: 30px 0;
            text-align: center;
            color: var(--text-muted);
            border-top: 1px solid var(--border-subtle);
            font-size: 0.85rem;
        }}
        
        /* Animations */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .metric-card {{
            animation: fadeInUp 0.5s ease forwards;
        }}
        
        .metric-card:nth-child(1) {{ animation-delay: 0.05s; }}
        .metric-card:nth-child(2) {{ animation-delay: 0.1s; }}
        .metric-card:nth-child(3) {{ animation-delay: 0.15s; }}
        .metric-card:nth-child(4) {{ animation-delay: 0.2s; }}
        .metric-card:nth-child(5) {{ animation-delay: 0.25s; }}
        .metric-card:nth-child(6) {{ animation-delay: 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-kicker">Eleusis LLM Benchmark</div>
            <h1>Benchmark Report</h1>
            <div class="subtitle"><strong>{folder_name}</strong> · {len(models_data)} models · {total_rules} rules</div>
        </header>

        <div class="summary-grid">
            <div class="metric-card">
                <div class="metric-label">Rules Completed</div>
                <div class="metric-value" id="card-progress">{total_rules}</div>
                <div class="metric-sub">of {total_rules} total</div>
            </div>
            <div class="metric-card{' highlight' if in_progress_rank else ''}" id="rank-card">
                <div class="metric-label">Current Rank</div>
                <div class="metric-value" id="card-rank">#{rank_display}</div>
                <div class="metric-sub">of {len(models_data)} models</div>
            </div>
            <div class="metric-card{' highlight' if in_progress_rank else ''}" id="score-card">
                <div class="metric-label">Avg Floored Score</div>
                <div class="metric-value" id="card-score">{avg_score:.2f}</div>
                <div class="metric-sub" id="card-success">{success_rate:.0%} success rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Tokens Per Turn</div>
                <div class="metric-value" id="card-tokens">{tokens_per_turn:,.0f}</div>
                <div class="metric-sub">{tokens_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Double-Down Rate</div>
                <div class="metric-value" id="card-dd">{double_down_rate:.1f}%</div>
                <div class="metric-sub">{dd_label}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Complexity Ratio</div>
                <div class="metric-value" id="card-complexity">{complexity_ratio:.2f}</div>
                <div class="metric-sub">{complexity_label}</div>
            </div>
        </div>

        <h2>Performance Ranking</h2>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 16px;">Click a row to view detailed metrics</p>
        <table id="ranking-table">
            <thead>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Avg Score</th>
                <th>Success Rate</th>
                <th>Tokens Per Turn</th>
                <th>Double-Down Rate</th>
                <th>Complexity Ratio</th>
            </tr>
            </thead>
            <tbody>
{rows_html}            </tbody>
        </table>

        <h2>Analysis Charts</h2>
        <div class="chart-grid">
            <div class="chart-card">
                <h3>Overall Performance</h3>
                <a href="#lb-overall"><img src="overall_performance.png" alt="Overall Performance"></a>
            </div>
            <div class="chart-card">
                <h3>Score vs Failed Guesses</h3>
                <a href="#lb-failed"><img src="score_vs_failed_guesses.png" alt="Score vs Failed Guesses"></a>
            </div>
            <div class="chart-card">
                <h3>Score Breakdown</h3>
                <a href="#lb-stack"><img src="score_stack.png" alt="Score Stack"></a>
            </div>
            <div class="chart-card">
                <h3>By Rule Analysis</h3>
                <a href="#lb-byrule"><img src="by_rule.png" alt="By Rule"></a>
            </div>
            <div class="chart-card">
                <h3>Calibration Curves</h3>
                <a href="#lb-calibration"><img src="calibration_curves.png" alt="Calibration"></a>
            </div>
            <div class="chart-card">
                <h3>Guess Rate by Confidence</h3>
                <a href="#lb-guessrate"><img src="guess_rate.png" alt="Guess Rate"></a>
            </div>
            <div class="chart-card">
                <h3>Complexity Analysis</h3>
                <a href="#lb-complexity"><img src="complexity_analysis.png" alt="Complexity"></a>
            </div>
            <div class="chart-card">
                <h3>Complexity Ratio</h3>
                <a href="#lb-ratio"><img src="complexity_ratio.png" alt="Complexity Ratio"></a>
            </div>
            <div class="chart-card">
                <h3>Excess Caution</h3>
                <a href="#lb-excess"><img src="excess_caution.png" alt="Excess Caution"></a>
            </div>
            <div class="chart-card">
                <h3>Reckless Guessing</h3>
                <a href="#lb-reckless"><img src="reckless_guessing.png" alt="Reckless Guessing"></a>
            </div>
            <div class="chart-card">
                <h3>Tokens by Turn</h3>
                <a href="#lb-tokens"><img src="tokens_by_turn.png" alt="Tokens by Turn"></a>
            </div>
        </div>

        <h2>Per-Model Reports ({len(models_data)} models)</h2>
        <div class="chart-grid">
{per_model_cards}        </div>

        <footer>
            Generated by Eleusis Benchmark
        </footer>
    </div>
    <div id="lb-overall" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="overall_performance.png" alt="Overall Performance">
    </div>
    <div id="lb-failed" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="score_vs_failed_guesses.png" alt="Score vs Failed Guesses">
    </div>
    <div id="lb-stack" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="score_stack.png" alt="Score Stack">
    </div>
    <div id="lb-byrule" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="by_rule.png" alt="By Rule">
    </div>
    <div id="lb-calibration" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="calibration_curves.png" alt="Calibration">
    </div>
    <div id="lb-guessrate" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="guess_rate.png" alt="Guess Rate">
    </div>
    <div id="lb-complexity" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="complexity_analysis.png" alt="Complexity">
    </div>
    <div id="lb-ratio" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="complexity_ratio.png" alt="Complexity Ratio">
    </div>
    <div id="lb-excess" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="excess_caution.png" alt="Excess Caution">
    </div>
    <div id="lb-reckless" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="reckless_guessing.png" alt="Reckless Guessing">
    </div>
    <div id="lb-tokens" class="lightbox" onclick="history.back(); return false;">
        <span class="lightbox-close">&times;</span>
        <img src="tokens_by_turn.png" alt="Tokens by Turn">
    </div>
{per_model_lightboxes}
    <script>
        // Model data for interactive cards
        const totalModels = {len(models_data)};
        const totalRules = {total_rules};
        const modelDataMap = {{
{model_data_js}        }};

        // Initialize with first model (or in-progress model)
        let initialModel = modelDataMap["{models_data[0]["name"].replace(chr(34), chr(39)) if models_data else ''}"];
        for (const [name, data] of Object.entries(modelDataMap)) {{
            if (data.is_in_progress) {{
                initialModel = data;
                // Highlight the row
                const row = document.querySelector(`tr[data-model="${{name}}"]`);
                if (row) row.classList.add('active');
                break;
            }}
        }}

        // Add click handlers to table rows
        document.querySelectorAll('#ranking-table tbody tr').forEach(row => {{
            row.addEventListener('click', () => {{
                const modelName = row.dataset.model;
                const modelData = modelDataMap[modelName];
                if (modelData) {{
                    updateCards(modelData);
                    // Remove active class from all rows
                    document.querySelectorAll('#ranking-table tbody tr').forEach(r => r.classList.remove('active'));
                    // Add active class to clicked row
                    row.classList.add('active');
                }}
            }});
        }});

        function updateCards(data) {{
            document.getElementById('card-rank').textContent = '#' + data.rank;
            document.getElementById('card-score').textContent = data.avg_score.toFixed(2);
            document.getElementById('card-success').textContent = (data.success_rate * 100).toFixed(0) + '% success rate';
            document.getElementById('card-progress').textContent = totalRules;
            
            // Update tokens with label
            const tokens = data.tokens;
            document.getElementById('card-tokens').textContent = tokens.toLocaleString();
            let tokensLabel = 'efficient';
            if (tokens >= 8000) tokensLabel = 'high';
            else if (tokens >= 4000) tokensLabel = 'moderate';
            const tokensSub = document.querySelector('#card-tokens').parentElement.querySelector('.metric-sub');
            if (tokensSub) tokensSub.textContent = tokensLabel;
            
            // Update double-down with label
            const ddRate = data.double_down;
            document.getElementById('card-dd').textContent = ddRate.toFixed(1) + '%';
            let ddLabel = 'very cautious';
            if (ddRate >= 30) ddLabel = 'aggressive';
            else if (ddRate >= 15) ddLabel = 'cautious';
            const ddSub = document.querySelector('#card-dd').parentElement.querySelector('.metric-sub');
            if (ddSub) ddSub.textContent = ddLabel;
            
            // Update complexity ratio with label
            const complexity = data.complexity;
            document.getElementById('card-complexity').textContent = complexity.toFixed(2);
            let crLabel = 'balanced';
            if (complexity > 1.5) crLabel = 'overcomplication';
            else if (complexity > 1.0) crLabel = 'slight overcomplication';
            else if (complexity < 0.8) crLabel = 'simplification';
            const crSub = document.querySelector('#card-complexity').parentElement.querySelector('.metric-sub');
            if (crSub) crSub.textContent = crLabel;
            
            // Add highlight effect
            document.querySelectorAll('.metric-card').forEach(card => {{
                card.style.transform = 'scale(1.02)';
                card.style.transition = 'transform 0.2s';
                setTimeout(() => {{
                    card.style.transform = 'scale(1)';
                }}, 200);
            }});
        }}
    </script>
</body>
</html>'''

    return html
