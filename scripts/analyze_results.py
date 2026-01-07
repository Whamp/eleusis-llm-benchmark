"""Analyze evaluation results across multiple LLM runs."""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> list[dict]:
    """Load all results.json files from results directory."""
    results = []
    for folder in results_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("solo_evaluation_"):
            results_file = folder / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["_folder"] = folder.name
                    results.append(data)
                    logger.info(f"Loaded: {folder.name}")
    return results


def load_rules_library(rules_path: Path) -> dict[str, dict]:
    """Load rules.json and index by description for lookup."""
    with open(rules_path) as f:
        data = json.load(f)
    # Index by description for easy lookup
    return {r["description"]: r for r in data["rules"]}


def build_rounds_dataframe(results: list[dict], rules_lib: dict) -> pd.DataFrame:
    """Build DataFrame with one row per round."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        model_spec = result["config"]["player_model"]
        for round_data in result["rounds"]:
            rule_desc = round_data["rule_description"]
            rule_info = rules_lib.get(rule_desc, {})

            rows.append({
                "model": model,
                "model_spec": model_spec,
                "round_number": round_data["round_number"],
                "success": round_data["success"],
                "score": round_data["score"],
                "turn_count": round_data["turn_count"],
                "failed_guesses": round_data["failed_guesses"],
                "game_over_reason": round_data["game_over_reason"],
                "rule_description": rule_desc,
                "player_tokens": round_data["llm_usage"]["player"]["total_tokens"],
                "wall_clock_seconds": round_data.get("wall_clock_seconds", 0),
                # Rule complexity metrics
                "cyclomatic_complexity": rule_info.get("cyclomatic_complexity"),
                "node_count": rule_info.get("node_count"),
                "avg_acceptance_rate": rule_info.get("avg_acceptance_rate"),
            })
    return pd.DataFrame(rows)


def build_turns_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build DataFrame with one row per turn (for calibration analysis)."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        for round_data in result["rounds"]:
            for turn in round_data["turns"]:
                llm_resp = turn.get("llm_response", {})
                guess_attempt = turn.get("guess_attempt")

                rows.append({
                    "model": model,
                    "round_number": round_data["round_number"],
                    "turn_number": turn["turn_number"],
                    "confidence_level": llm_resp.get("confidence_level"),
                    "guess_rule": llm_resp.get("guess_rule", False),
                    "guess_correct": guess_attempt["correct"] if guess_attempt else None,
                    "card_accepted": turn.get("action_result", {}).get("accepted"),
                    "tentative_rule": llm_resp.get("tentative_rule"),
                    "actual_rule": round_data["rule_description"],
                    "round_success": round_data["success"],
                })
    return pd.DataFrame(rows)


# =============================================================================
# Analysis 1: Basic Model Comparison
# =============================================================================

def analyze_basic_metrics(df_rounds: pd.DataFrame) -> pd.DataFrame:
    """Compute basic comparison metrics per model."""
    metrics = df_rounds.groupby("model").agg(
        rounds_played=("success", "count"),
        success_rate=("success", "mean"),
        avg_score=("score", "mean"),
        avg_turns=("turn_count", "mean"),
        avg_failed_guesses=("failed_guesses", "mean"),
        total_tokens=("player_tokens", "sum"),
        total_score=("score", "sum"),
    ).reset_index()

    # Token efficiency: score per 1K tokens
    metrics["token_efficiency"] = (metrics["total_score"] / metrics["total_tokens"]) * 1000

    # Average turns when successful
    successful = df_rounds[df_rounds["success"]]
    avg_turns_success = successful.groupby("model")["turn_count"].mean()
    metrics = metrics.merge(
        avg_turns_success.rename("avg_turns_success").reset_index(),
        on="model",
        how="left"
    )

    return metrics.sort_values("success_rate", ascending=False)


def plot_basic_metrics(metrics: pd.DataFrame, output_dir: Path):
    """Generate bar charts for basic metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Success rate
    ax = axes[0, 0]
    ax.barh(metrics["model"], metrics["success_rate"])
    ax.set_xlabel("Success Rate")
    ax.set_title("Success Rate by Model")
    ax.set_xlim(0, 1)
    for i, v in enumerate(metrics["success_rate"]):
        ax.text(v + 0.02, i, f"{v:.1%}", va="center")

    # Average score
    ax = axes[0, 1]
    ax.barh(metrics["model"], metrics["avg_score"])
    ax.set_xlabel("Average Score")
    ax.set_title("Average Score by Model")

    # Average turns (all vs successful)
    ax = axes[1, 0]
    x = range(len(metrics))
    width = 0.35
    ax.barh([i - width/2 for i in x], metrics["avg_turns"], width, label="All rounds")
    ax.barh([i + width/2 for i in x], metrics["avg_turns_success"], width, label="Successful only")
    ax.set_yticks(x)
    ax.set_yticklabels(metrics["model"])
    ax.set_xlabel("Average Turns")
    ax.set_title("Average Turns to Complete")
    ax.legend()

    # Token efficiency
    ax = axes[1, 1]
    ax.barh(metrics["model"], metrics["token_efficiency"])
    ax.set_xlabel("Score per 1K Tokens")
    ax.set_title("Token Efficiency")

    plt.tight_layout()
    plt.savefig(output_dir / "basic_metrics.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'basic_metrics.png'}")


# =============================================================================
# Analysis 2: Confidence Calibration
# =============================================================================

def analyze_confidence_calibration(df_turns: pd.DataFrame) -> dict:
    """Analyze confidence vs actual correctness when guessing."""
    # Filter to turns where a guess was made
    guesses = df_turns[df_turns["guess_rule"] == True].copy()
    if guesses.empty:
        return {"error": "No guesses found in data"}

    guesses = guesses.dropna(subset=["confidence_level", "guess_correct"])

    # Bin confidence levels
    bins = [0, 4, 7, 11]  # 0-3 (low), 4-6 (medium), 7-10 (high)
    labels = ["Low (0-3)", "Medium (4-6)", "High (7-10)"]
    guesses["confidence_bin"] = pd.cut(
        guesses["confidence_level"], bins=bins, labels=labels, right=False
    )

    # Calibration: accuracy per confidence bin
    calibration = guesses.groupby("confidence_bin", observed=True).agg(
        accuracy=("guess_correct", "mean"),
        count=("guess_correct", "count"),
    ).reset_index()

    # Overconfidence: mean confidence when wrong vs when correct
    correct_guesses = guesses[guesses["guess_correct"] == True]
    wrong_guesses = guesses[guesses["guess_correct"] == False]

    stats = {
        "calibration": calibration,
        "mean_confidence_when_correct": correct_guesses["confidence_level"].mean(),
        "mean_confidence_when_wrong": wrong_guesses["confidence_level"].mean(),
        "total_guesses": len(guesses),
        "correct_guesses": len(correct_guesses),
        "wrong_guesses": len(wrong_guesses),
    }

    # Per-model breakdown
    per_model = guesses.groupby("model").agg(
        total_guesses=("guess_correct", "count"),
        correct_guesses=("guess_correct", "sum"),
        mean_confidence=("confidence_level", "mean"),
    ).reset_index()
    per_model["accuracy"] = per_model["correct_guesses"] / per_model["total_guesses"]
    stats["per_model"] = per_model

    return stats


def plot_confidence_calibration(df_turns: pd.DataFrame, output_dir: Path):
    """Plot confidence calibration curves."""
    guesses = df_turns[df_turns["guess_rule"] == True].copy()
    guesses = guesses.dropna(subset=["confidence_level", "guess_correct"])

    if guesses.empty:
        logger.warning("No guess data for calibration plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve: confidence vs accuracy
    ax = axes[0]
    bins = list(range(0, 12))
    guesses["conf_bin"] = pd.cut(guesses["confidence_level"], bins=bins, labels=bins[:-1])
    cal = guesses.groupby("conf_bin", observed=True).agg(
        accuracy=("guess_correct", "mean"),
        count=("guess_correct", "count")
    ).reset_index()
    cal["conf_bin"] = cal["conf_bin"].astype(int)

    ax.bar(cal["conf_bin"], cal["accuracy"], alpha=0.7, label="Actual accuracy")
    ax.plot([0, 10], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Actual Accuracy")
    ax.set_title("Confidence Calibration")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.legend()

    # Add count annotations
    for _, row in cal.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["conf_bin"], row["accuracy"] + 0.05),
                    ha="center", fontsize=8)

    # Distribution of confidence when correct vs wrong
    ax = axes[1]
    correct = guesses[guesses["guess_correct"] == True]["confidence_level"]
    wrong = guesses[guesses["guess_correct"] == False]["confidence_level"]

    ax.hist(correct, bins=range(0, 12), alpha=0.6, label=f"Correct (n={len(correct)})", density=True)
    ax.hist(wrong, bins=range(0, 12), alpha=0.6, label=f"Wrong (n={len(wrong)})", density=True)
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution: Correct vs Wrong Guesses")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_calibration.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'confidence_calibration.png'}")


# =============================================================================
# Analysis 3: Success vs Rule Complexity
# =============================================================================

def analyze_complexity(df_rounds: pd.DataFrame) -> dict:
    """Analyze success rate by rule complexity."""
    # Filter to rounds with complexity data
    df = df_rounds.dropna(subset=["cyclomatic_complexity"])

    # Success rate by cyclomatic complexity
    by_cyclomatic = df.groupby("cyclomatic_complexity").agg(
        success_rate=("success", "mean"),
        count=("success", "count"),
        avg_score=("score", "mean"),
    ).reset_index()

    # Success rate by node_count bins
    df_nodes = df.dropna(subset=["node_count"])
    df_nodes["node_count_bin"] = pd.cut(
        df_nodes["node_count"],
        bins=[0, 15, 25, 35, 100],
        labels=["1-15", "16-25", "26-35", "36+"]
    )
    by_node_count = df_nodes.groupby("node_count_bin", observed=True).agg(
        success_rate=("success", "mean"),
        count=("success", "count"),
        avg_score=("score", "mean"),
    ).reset_index()

    # Success rate by acceptance rate bins
    df["acceptance_bin"] = pd.cut(
        df["avg_acceptance_rate"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    )
    by_acceptance = df.groupby("acceptance_bin", observed=True).agg(
        success_rate=("success", "mean"),
        count=("success", "count"),
    ).reset_index()

    # Model x cyclomatic complexity heatmap
    heatmap_cyclomatic = df.pivot_table(
        values="success",
        index="model",
        columns="cyclomatic_complexity",
        aggfunc="mean"
    )

    # Model x node_count_bin heatmap
    heatmap_nodes = df_nodes.pivot_table(
        values="success",
        index="model",
        columns="node_count_bin",
        aggfunc="mean"
    )

    return {
        "by_cyclomatic": by_cyclomatic,
        "by_node_count": by_node_count,
        "by_acceptance": by_acceptance,
        "heatmap_cyclomatic": heatmap_cyclomatic,
        "heatmap_nodes": heatmap_nodes,
    }


def plot_complexity(df_rounds: pd.DataFrame, output_dir: Path):
    """Plot success rate vs rule complexity."""
    df = df_rounds.dropna(subset=["cyclomatic_complexity"])

    # Create node_count bins
    df_nodes = df.dropna(subset=["node_count"]).copy()
    df_nodes["node_count_bin"] = pd.cut(
        df_nodes["node_count"],
        bins=[0, 15, 25, 35, 100],
        labels=["1-15", "16-25", "26-35", "36+"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Cyclomatic complexity
    # Success rate by cyclomatic complexity
    ax = axes[0, 0]
    by_cc = df.groupby("cyclomatic_complexity").agg(
        success_rate=("success", "mean"),
        count=("success", "count"),
    ).reset_index()
    ax.bar(by_cc["cyclomatic_complexity"], by_cc["success_rate"])
    ax.set_xlabel("Cyclomatic Complexity")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Cyclomatic Complexity")
    ax.set_ylim(0, 1)
    for i, row in by_cc.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["cyclomatic_complexity"], row["success_rate"] + 0.02),
                    ha="center", fontsize=9)

    # Success rate vs acceptance rate (scatter, colored by cyclomatic)
    ax = axes[0, 1]
    rule_stats = df.groupby("rule_description").agg(
        success_rate=("success", "mean"),
        avg_acceptance_rate=("avg_acceptance_rate", "first"),
        cyclomatic_complexity=("cyclomatic_complexity", "first"),
    ).reset_index()
    scatter = ax.scatter(
        rule_stats["avg_acceptance_rate"],
        rule_stats["success_rate"],
        c=rule_stats["cyclomatic_complexity"],
        cmap="viridis",
        alpha=0.7,
        s=50
    )
    ax.set_xlabel("Rule Acceptance Rate")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success vs Selectivity (color=cyclomatic)")
    plt.colorbar(scatter, ax=ax, label="Cyclomatic Complexity")

    # Model x cyclomatic complexity heatmap
    ax = axes[0, 2]
    heatmap = df.pivot_table(
        values="success",
        index="model",
        columns="cyclomatic_complexity",
        aggfunc="mean"
    )
    im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns.astype(int))
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlabel("Cyclomatic Complexity")
    ax.set_title("Model × Cyclomatic Complexity")
    plt.colorbar(im, ax=ax, label="Success Rate")
    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            val = heatmap.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9)

    # Row 2: Node count
    # Success rate by node_count bins
    ax = axes[1, 0]
    by_nc = df_nodes.groupby("node_count_bin", observed=True).agg(
        success_rate=("success", "mean"),
        count=("success", "count"),
    ).reset_index()
    ax.bar(range(len(by_nc)), by_nc["success_rate"])
    ax.set_xticks(range(len(by_nc)))
    ax.set_xticklabels(by_nc["node_count_bin"])
    ax.set_xlabel("Node Count (AST size)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate by Node Count")
    ax.set_ylim(0, 1)
    for i, row in by_nc.iterrows():
        ax.annotate(f"n={int(row['count'])}", (i, row["success_rate"] + 0.02),
                    ha="center", fontsize=9)

    # Success rate vs acceptance rate (scatter, colored by node_count)
    ax = axes[1, 1]
    rule_stats_nc = df_nodes.groupby("rule_description").agg(
        success_rate=("success", "mean"),
        avg_acceptance_rate=("avg_acceptance_rate", "first"),
        node_count=("node_count", "first"),
    ).reset_index()
    scatter = ax.scatter(
        rule_stats_nc["avg_acceptance_rate"],
        rule_stats_nc["success_rate"],
        c=rule_stats_nc["node_count"],
        cmap="plasma",
        alpha=0.7,
        s=50
    )
    ax.set_xlabel("Rule Acceptance Rate")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success vs Selectivity (color=node count)")
    plt.colorbar(scatter, ax=ax, label="Node Count")

    # Model x node_count_bin heatmap
    ax = axes[1, 2]
    heatmap_nc = df_nodes.pivot_table(
        values="success",
        index="model",
        columns="node_count_bin",
        aggfunc="mean"
    )
    im = ax.imshow(heatmap_nc.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_nc.columns)))
    ax.set_xticklabels(heatmap_nc.columns)
    ax.set_yticks(range(len(heatmap_nc.index)))
    ax.set_yticklabels(heatmap_nc.index)
    ax.set_xlabel("Node Count (AST size)")
    ax.set_title("Model × Node Count")
    plt.colorbar(im, ax=ax, label="Success Rate")
    for i in range(len(heatmap_nc.index)):
        for j in range(len(heatmap_nc.columns)):
            val = heatmap_nc.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "complexity_analysis.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'complexity_analysis.png'}")


# =============================================================================
# Analysis 4: Learning Curves
# =============================================================================

def plot_learning_curves(df_turns: pd.DataFrame, output_dir: Path):
    """Plot metrics over turn progression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confidence trajectory by turn number
    ax = axes[0]
    conf_by_turn = df_turns.groupby(["model", "turn_number"])["confidence_level"].mean().reset_index()
    for model in conf_by_turn["model"].unique():
        data = conf_by_turn[conf_by_turn["model"] == model]
        ax.plot(data["turn_number"], data["confidence_level"], label=model, alpha=0.7)
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Average Confidence Level")
    ax.set_title("Confidence Trajectory Over Turns")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(1, 40)

    # Card acceptance rate by turn number
    ax = axes[1]
    accept_by_turn = df_turns.groupby("turn_number")["card_accepted"].mean().reset_index()
    ax.plot(accept_by_turn["turn_number"], accept_by_turn["card_accepted"])
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Card Acceptance Rate")
    ax.set_title("Card Acceptance Rate Over Turns (All Models)")
    ax.set_xlim(1, 40)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {output_dir / 'learning_curves.png'}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Eleusis evaluation results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing evaluation results")
    parser.add_argument("--rules-file", type=str, default="rules.json",
                        help="Path to rules library")
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                        help="Output directory for plots and tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rules_path = Path(args.rules_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("Loading results...")
    results = load_results(results_dir)
    if not results:
        logger.error("No results found!")
        return

    logger.info(f"Loaded {len(results)} evaluation runs")

    rules_lib = load_rules_library(rules_path)
    logger.info(f"Loaded {len(rules_lib)} rules from library")

    # Build DataFrames
    df_rounds = build_rounds_dataframe(results, rules_lib)
    df_turns = build_turns_dataframe(results)
    logger.info(f"Built DataFrames: {len(df_rounds)} rounds, {len(df_turns)} turns")

    # Analysis 1: Basic metrics
    logger.info("=" * 60)
    logger.info("BASIC MODEL COMPARISON")
    logger.info("=" * 60)
    metrics = analyze_basic_metrics(df_rounds)
    print("\n")
    print(metrics.to_string(index=False))
    print("\n")
    metrics.to_csv(output_dir / "basic_metrics.csv", index=False)
    plot_basic_metrics(metrics, output_dir)

    # Analysis 2: Confidence calibration
    logger.info("=" * 60)
    logger.info("CONFIDENCE CALIBRATION")
    logger.info("=" * 60)
    cal_stats = analyze_confidence_calibration(df_turns)
    if "error" not in cal_stats:
        print(f"\nTotal guesses: {cal_stats['total_guesses']}")
        print(f"Correct: {cal_stats['correct_guesses']}, Wrong: {cal_stats['wrong_guesses']}")
        print(f"Mean confidence when correct: {cal_stats['mean_confidence_when_correct']:.1f}")
        print(f"Mean confidence when wrong: {cal_stats['mean_confidence_when_wrong']:.1f}")
        print("\nCalibration by confidence bin:")
        print(cal_stats["calibration"].to_string(index=False))
        print("\nPer-model breakdown:")
        print(cal_stats["per_model"].to_string(index=False))
        plot_confidence_calibration(df_turns, output_dir)
    else:
        logger.warning(cal_stats["error"])

    # Analysis 3: Complexity analysis
    logger.info("=" * 60)
    logger.info("RULE COMPLEXITY ANALYSIS")
    logger.info("=" * 60)
    complexity_stats = analyze_complexity(df_rounds)
    print("\nSuccess rate by cyclomatic complexity:")
    print(complexity_stats["by_cyclomatic"].to_string(index=False))
    print("\nSuccess rate by node count (AST size):")
    print(complexity_stats["by_node_count"].to_string(index=False))
    print("\nSuccess rate by acceptance rate:")
    print(complexity_stats["by_acceptance"].to_string(index=False))
    plot_complexity(df_rounds, output_dir)

    # Analysis 4: Learning curves
    logger.info("=" * 60)
    logger.info("LEARNING CURVES")
    logger.info("=" * 60)
    plot_learning_curves(df_turns, output_dir)

    logger.info("=" * 60)
    logger.info(f"Analysis complete! Outputs saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
