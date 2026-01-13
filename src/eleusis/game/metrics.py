"""Rule evaluation metrics for Eleusis."""

import ast
import logging
import random

from eleusis.game.cards import Card, Suit
from eleusis.game.engine import Rule

__all__ = ["code_complexity", "RuleEvaluator"]

logger = logging.getLogger(__name__)


def code_complexity(code: str) -> dict:
    """Return AST node count and cyclomatic complexity for Python code."""
    tree = ast.parse(code)

    node_count = 0
    cyclomatic = 1  # base complexity

    for node in ast.walk(tree):
        node_count += 1

        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            cyclomatic += 1
        elif isinstance(node, ast.BoolOp):
            # each 'and'/'or' adds (n-1) decision points
            cyclomatic += len(node.values) - 1

    return {
        "node_count": node_count,
        "cyclomatic": cyclomatic,
    }


class RuleEvaluator:
    """Evaluates rules by simulating random card plays."""

    def __init__(
        self,
        num_simulations: int = 10,
        plays_per_simulation: int = 50,
    ) -> None:
        """Initialize evaluator with simulation parameters."""
        self.num_simulations = num_simulations
        self.plays_per_simulation = plays_per_simulation
        self._all_cards = [Card(rank, suit) for rank in range(1, 14) for suit in Suit]

    def _simulate_random_plays(self, rule: Rule) -> dict:
        """Simulate random card plays and return statistics."""
        total_plays = 0
        total_accepted = 0
        mainline = []

        for _ in range(self.plays_per_simulation):
            card = random.choice(self._all_cards)
            accepted = rule.evaluate(card, mainline)

            total_plays += 1
            if accepted:
                total_accepted += 1
                mainline.append(card)

        acceptance_rate = total_accepted / total_plays if total_plays > 0 else 0.0

        return {
            "total_plays": total_plays,
            "total_accepted": total_accepted,
            "acceptance_rate": acceptance_rate,
            "mainline_length": len(mainline),
        }

    def evaluate(self, rule: Rule) -> dict:
        """Evaluate a rule and return acceptance rate and complexity metrics.

        Returns:
            Dict with avg_acceptance_rate, node_count, cyclomatic_complexity
        """
        # Run multiple simulations
        sim_results = []
        for sim_num in range(self.num_simulations):
            logger.debug(f"  Simulation {sim_num + 1}/{self.num_simulations}")
            result = self._simulate_random_plays(rule)
            sim_results.append(result)

        # Compute averages
        avg_acceptance_rate = (
            sum(r["acceptance_rate"] for r in sim_results) / self.num_simulations
        )
        avg_mainline_length = (
            sum(r["mainline_length"] for r in sim_results) / self.num_simulations
        )

        logger.debug(f"  Acceptance rate: {avg_acceptance_rate:.1%}")
        logger.debug(f"  Avg mainline length: {avg_mainline_length:.1f}")

        # Compute code complexity
        complexity = code_complexity(rule.get_code())
        logger.debug(
            f"  Complexity: nodes={complexity['node_count']}, "
            f"cyclomatic={complexity['cyclomatic']}"
        )

        return {
            "avg_acceptance_rate": avg_acceptance_rate,
            "node_count": complexity["node_count"],
            "cyclomatic_complexity": complexity["cyclomatic"],
        }
