"""Analyze evaluation results across multiple LLM runs."""

import argparse
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class TeeWriter:
    """Write to both a file and stdout."""

    def __init__(self, file_path: Path):
        self.file = open(file_path, "w")
        self.buffer = io.StringIO()

    def write(self, text: str):
        self.file.write(text)
        self.buffer.write(text)
        print(text, end="")

    def close(self):
        self.file.close()


def load_results(results_dir: Path) -> tuple[list[dict], list[str]]:
    """Load all results.json files from results directory. Returns (results, folder_names)."""
    results = []
    folder_names = []
    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("solo_evaluation_"):
            results_file = folder / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    data["_folder"] = folder.name
                    results.append(data)
                    folder_names.append(folder.name)
                    logger.info(f"Loaded: {folder.name}")
    return results, folder_names


def build_rules_lookup(results: list[dict]) -> dict[str, dict]:
    """Build rules lookup from embedded rules_library in results files."""
    rules_lookup = {}
    for result in results:
        # rules_library is nested inside checkpoint
        checkpoint = result.get("checkpoint", {})
        rules_lib = checkpoint.get("rules_library", [])
        for rule in rules_lib:
            desc = rule.get("description")
            if desc and desc not in rules_lookup:
                rules_lookup[desc] = rule
    return rules_lookup


def build_rounds_dataframe(results: list[dict], rules_lib: dict) -> pd.DataFrame:
    """Build DataFrame with one row per round."""
    rows = []
    for result in results:
        model = result["config"]["player"]
        model_spec = result["config"]["player_model"]
        for round_data in result["rounds"]:
            rule_desc = round_data["rule_description"]
            rule_info = rules_lib.get(rule_desc, {})
            player_usage = round_data["llm_usage"]["player"]

            # Handle both old (total_tokens) and new (output_tokens) formats
            output_tokens = player_usage.get("output_tokens", player_usage.get("total_tokens", 0))
            reasoning_tokens = player_usage.get("reasoning_tokens", 0) or 0
            answer_tokens = player_usage.get("answer_tokens", output_tokens - reasoning_tokens)

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
                # Token metrics (normalized)
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "answer_tokens": answer_tokens,
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
        # Token metrics (normalized)
        total_output_tokens=("output_tokens", "sum"),
        total_reasoning_tokens=("reasoning_tokens", "sum"),
        total_answer_tokens=("answer_tokens", "sum"),
        total_turns=("turn_count", "sum"),
    ).reset_index()

    # Output tokens per turn
    metrics["output_tokens_per_turn"] = metrics["total_output_tokens"] / metrics["total_turns"]

    # Reasoning ratio: what fraction of output is reasoning
    metrics["reasoning_ratio"] = metrics["total_reasoning_tokens"] / metrics["total_output_tokens"]

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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

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

    # Average failed guesses
    ax = axes[0, 2]
    ax.barh(metrics["model"], metrics["avg_failed_guesses"])
    ax.set_xlabel("Average Failed Guesses")
    ax.set_title("Average Failed Guesses by Model")
    for i, v in enumerate(metrics["avg_failed_guesses"]):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center")

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

    # Output tokens per turn
    ax = axes[1, 1]
    ax.barh(metrics["model"], metrics["output_tokens_per_turn"])
    ax.set_xlabel("Output Tokens per Turn")
    ax.set_title("Output Tokens per Turn by Model")
    for i, v in enumerate(metrics["output_tokens_per_turn"]):
        ax.text(v + 10, i, f"{v:.0f}", va="center")

    # Reasoning ratio
    ax = axes[1, 2]
    ax.barh(metrics["model"], metrics["reasoning_ratio"])
    ax.set_xlabel("Reasoning Ratio")
    ax.set_title("Reasoning Tokens / Output Tokens")
    ax.set_xlim(0, 1)
    for i, v in enumerate(metrics["reasoning_ratio"]):
        ax.text(v + 0.02, i, f"{v:.1%}", va="center")

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
    """Analyze score by rule complexity."""
    # Filter to rounds with complexity data
    df = df_rounds.dropna(subset=["cyclomatic_complexity"])

    # Score by cyclomatic complexity (use actual values, not bins)
    by_cyclomatic = df.groupby("cyclomatic_complexity").agg(
        avg_score=("score", "mean"),
        count=("score", "count"),
        success_rate=("success", "mean"),
    ).reset_index()

    # Score by node_count (use actual values for small datasets)
    df_nodes = df.dropna(subset=["node_count"])
    by_node_count = df_nodes.groupby("node_count").agg(
        avg_score=("score", "mean"),
        count=("score", "count"),
        success_rate=("success", "mean"),
    ).reset_index()

    # Score by acceptance rate bins
    df_accept = df.dropna(subset=["avg_acceptance_rate"]).copy()
    df_accept["acceptance_bin"] = pd.cut(
        df_accept["avg_acceptance_rate"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    )
    by_acceptance = df_accept.groupby("acceptance_bin", observed=True).agg(
        avg_score=("score", "mean"),
        count=("score", "count"),
    ).reset_index()

    # Model x cyclomatic complexity heatmap (score)
    heatmap_cyclomatic = df.pivot_table(
        values="score",
        index="model",
        columns="cyclomatic_complexity",
        aggfunc="mean"
    )

    # Model x node_count heatmap (score)
    heatmap_nodes = df_nodes.pivot_table(
        values="score",
        index="model",
        columns="node_count",
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
    """Plot score vs rule complexity."""
    df = df_rounds.dropna(subset=["cyclomatic_complexity"])
    df_nodes = df.dropna(subset=["node_count"]).copy()

    # Compute score range for consistent y-axis
    score_max = df["score"].max() * 1.1 if not df.empty else 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Cyclomatic complexity
    # Score by cyclomatic complexity
    ax = axes[0, 0]
    by_cc = df.groupby("cyclomatic_complexity").agg(
        avg_score=("score", "mean"),
        count=("score", "count"),
    ).reset_index()
    ax.bar(by_cc["cyclomatic_complexity"], by_cc["avg_score"])
    ax.set_xlabel("Cyclomatic Complexity")
    ax.set_ylabel("Average Score")
    ax.set_title("Score by Cyclomatic Complexity")
    ax.set_ylim(0, score_max)
    for _, row in by_cc.iterrows():
        ax.annotate(f"n={int(row['count'])}", (row["cyclomatic_complexity"], row["avg_score"] + score_max * 0.02),
                    ha="center", fontsize=9)

    # Score vs acceptance rate (scatter, colored by cyclomatic)
    ax = axes[0, 1]
    rule_stats = df.groupby("rule_description").agg(
        avg_score=("score", "mean"),
        avg_acceptance_rate=("avg_acceptance_rate", "first"),
        cyclomatic_complexity=("cyclomatic_complexity", "first"),
    ).reset_index()
    scatter = ax.scatter(
        rule_stats["avg_acceptance_rate"],
        rule_stats["avg_score"],
        c=rule_stats["cyclomatic_complexity"],
        cmap="viridis",
        alpha=0.7,
        s=50
    )
    ax.set_xlabel("Rule Acceptance Rate")
    ax.set_ylabel("Average Score")
    ax.set_title("Score vs Selectivity (color=cyclomatic)")
    plt.colorbar(scatter, ax=ax, label="Cyclomatic Complexity")

    # Model x cyclomatic complexity heatmap (score)
    ax = axes[0, 2]
    heatmap = df.pivot_table(
        values="score",
        index="model",
        columns="cyclomatic_complexity",
        aggfunc="mean"
    )
    im = ax.imshow(heatmap.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=score_max)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(heatmap.columns.astype(int))
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    ax.set_xlabel("Cyclomatic Complexity")
    ax.set_title("Model × Cyclomatic Complexity")
    plt.colorbar(im, ax=ax, label="Avg Score")
    for i in range(len(heatmap.index)):
        for j in range(len(heatmap.columns)):
            val = heatmap.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

    # Row 2: Node count (using actual values, not bins)
    ax = axes[1, 0]
    by_nc = df_nodes.groupby("node_count").agg(
        avg_score=("score", "mean"),
        count=("score", "count"),
    ).reset_index()
    ax.bar(by_nc["node_count"].astype(str), by_nc["avg_score"])
    ax.set_xlabel("Node Count (AST size)")
    ax.set_ylabel("Average Score")
    ax.set_title("Score by Node Count")
    ax.set_ylim(0, score_max)
    for i, row in enumerate(by_nc.itertuples()):
        ax.annotate(f"n={int(row.count)}", (i, row.avg_score + score_max * 0.02),
                    ha="center", fontsize=9)

    # Score vs acceptance rate (scatter, colored by node_count)
    ax = axes[1, 1]
    rule_stats_nc = df_nodes.groupby("rule_description").agg(
        avg_score=("score", "mean"),
        avg_acceptance_rate=("avg_acceptance_rate", "first"),
        node_count=("node_count", "first"),
    ).reset_index()
    scatter = ax.scatter(
        rule_stats_nc["avg_acceptance_rate"],
        rule_stats_nc["avg_score"],
        c=rule_stats_nc["node_count"],
        cmap="plasma",
        alpha=0.7,
        s=50
    )
    ax.set_xlabel("Rule Acceptance Rate")
    ax.set_ylabel("Average Score")
    ax.set_title("Score vs Selectivity (color=node count)")
    plt.colorbar(scatter, ax=ax, label="Node Count")

    # Model x node_count heatmap (using actual values, score)
    ax = axes[1, 2]
    heatmap_nc = df_nodes.pivot_table(
        values="score",
        index="model",
        columns="node_count",
        aggfunc="mean"
    )
    im = ax.imshow(heatmap_nc.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=score_max)
    ax.set_xticks(range(len(heatmap_nc.columns)))
    ax.set_xticklabels(heatmap_nc.columns.astype(int))
    ax.set_yticks(range(len(heatmap_nc.index)))
    ax.set_yticklabels(heatmap_nc.index)
    ax.set_xlabel("Node Count (AST size)")
    ax.set_title("Model × Node Count")
    plt.colorbar(im, ax=ax, label="Avg Score")
    for i in range(len(heatmap_nc.index)):
        for j in range(len(heatmap_nc.columns)):
            val = heatmap_nc.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

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
    parser.add_argument("--output-dir", type=str, default="results/analysis",
                        help="Base output directory (timestamped subfolder will be created)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up tee writer for text output
    tee = TeeWriter(output_dir / "summary.txt")

    def out(text: str):
        """Write to both file and stdout."""
        tee.write(text + "\n")

    # Load data
    out("=" * 60)
    out("Loading results...")
    results, folder_names = load_results(results_dir)
    if not results:
        out("No results found!")
        tee.close()
        return

    out(f"Loaded {len(results)} evaluation runs")

    # Document source folders
    out("\n" + "=" * 60)
    out("SOURCE FOLDERS")
    out("=" * 60)
    for folder in folder_names:
        out(f"  - {folder}")

    # Build rules lookup from embedded rules_library in results files
    rules_lib = build_rules_lookup(results)
    out(f"\nExtracted {len(rules_lib)} unique rules from results files")

    # Build DataFrames
    df_rounds = build_rounds_dataframe(results, rules_lib)
    df_turns = build_turns_dataframe(results)
    out(f"Built DataFrames: {len(df_rounds)} rounds, {len(df_turns)} turns")

    # Analysis 1: Basic metrics
    out("\n" + "=" * 60)
    out("BASIC MODEL COMPARISON")
    out("=" * 60)
    metrics = analyze_basic_metrics(df_rounds)
    out("")
    out(metrics.to_string(index=False))
    out("")
    metrics.to_csv(output_dir / "basic_metrics.csv", index=False)
    plot_basic_metrics(metrics, output_dir)
    out(f"Saved: {output_dir / 'basic_metrics.png'}")

    # Analysis 2: Confidence calibration
    out("\n" + "=" * 60)
    out("CONFIDENCE CALIBRATION")
    out("=" * 60)
    cal_stats = analyze_confidence_calibration(df_turns)
    if "error" not in cal_stats:
        out(f"\nTotal guesses: {cal_stats['total_guesses']}")
        out(f"Correct: {cal_stats['correct_guesses']}, Wrong: {cal_stats['wrong_guesses']}")
        out(f"Mean confidence when correct: {cal_stats['mean_confidence_when_correct']:.1f}")
        out(f"Mean confidence when wrong: {cal_stats['mean_confidence_when_wrong']:.1f}")
        out("\nCalibration by confidence bin:")
        out(cal_stats["calibration"].to_string(index=False))
        out("\nPer-model breakdown:")
        out(cal_stats["per_model"].to_string(index=False))
        plot_confidence_calibration(df_turns, output_dir)
        out(f"\nSaved: {output_dir / 'confidence_calibration.png'}")
    else:
        out(f"Warning: {cal_stats['error']}")

    # Analysis 3: Complexity analysis
    out("\n" + "=" * 60)
    out("RULE COMPLEXITY ANALYSIS")
    out("=" * 60)
    complexity_stats = analyze_complexity(df_rounds)
    out("\nScore by cyclomatic complexity:")
    out(complexity_stats["by_cyclomatic"].to_string(index=False))
    out("\nScore by node count (AST size):")
    out(complexity_stats["by_node_count"].to_string(index=False))
    out("\nScore by acceptance rate:")
    out(complexity_stats["by_acceptance"].to_string(index=False))
    plot_complexity(df_rounds, output_dir)
    out(f"\nSaved: {output_dir / 'complexity_analysis.png'}")

    # Analysis 4: Learning curves
    out("\n" + "=" * 60)
    out("LEARNING CURVES")
    out("=" * 60)
    plot_learning_curves(df_turns, output_dir)
    out(f"Saved: {output_dir / 'learning_curves.png'}")

    out("\n" + "=" * 60)
    out(f"Analysis complete! Outputs saved to: {output_dir}")
    out("=" * 60)

    tee.close()


if __name__ == "__main__":
    main()
