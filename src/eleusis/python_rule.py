"""Python code-based rule implementation for Eleusis."""

import logging
import textwrap
from typing import Callable

from eleusis.cards import Card
from eleusis.game_engine import Rule

logger = logging.getLogger(__name__)


class PythonRule(Rule):
    """Rule implemented as executable Python code."""

    def __init__(self, description: str, code: str) -> None:
        """Initialize Python rule.

        Args:
            description: Natural language description of the rule
            code: Python code implementing the evaluate logic
        """
        self.rule_description = description
        self.code = code
        self._eval_function: Callable[[Card, list[Card]], bool] | None = None
        self._compile_error: str | None = None

        try:
            self._eval_function = self._create_eval_function(code)
            logger.debug(f"Successfully compiled Python rule: {description}")
        except SyntaxError as e:
            self._compile_error = str(e)
            logger.error(f"Syntax error in Python rule: {e}")
            logger.error(f"Code:\n{code}")

    def _create_eval_function(self, code: str) -> Callable[[Card, list[Card]], bool]:
        """Create evaluation function from code string with safe execution environment."""

        # VALIDATION: Warn if code looks like a complete function definition
        first_line = code.strip().split('\n')[0].strip()
        if first_line.startswith('def '):
            logger.error(
                "Generated code appears to be a complete function definition!"
            )
            logger.error(f"First line: {first_line}")
            logger.error(
                "This will create a nested function that never gets called. "
                "Expected function body only, not 'def ...'."
            )
            logger.error("Rule will likely fail all evaluations (return None/False)")

        # Check if code has any return statements
        if 'return' not in code:
            logger.warning(
                "Generated code has no 'return' statement - will implicitly return None"
            )

        # Debug print function that logs to logger
        def debug_print(*args):
            """Print function for debugging rule execution."""
            logger.debug(f"  [Rule Debug] {' '.join(str(arg) for arg in args)}")

        # Define safe globals with restricted builtins
        safe_globals = {
            "__builtins__": {
                "len": len,
                "sum": sum,
                "abs": abs,
                "min": min,
                "max": max,
                "any": any,
                "all": all,
                "print": debug_print,  # Add print for debugging
            },
            "Card": Card,
        }

        # Wrap code in function definition
        full_code = f"""
def evaluate_rule(card, mainline):
{textwrap.indent(code, '    ')}
"""

        # Execute code to define the function
        local_namespace = {}
        exec(full_code, safe_globals, local_namespace)

        return local_namespace["evaluate_rule"]

    def evaluate(self, card: Card, mainline: list[Card]) -> bool:
        """Evaluate if card is IN according to the Python rule."""
        if self._compile_error:
            logger.warning(f"Cannot evaluate - compilation failed: {self._compile_error}")
            return False

        try:
            # Add debug context for troubleshooting
            logger.debug(f"Evaluating {card} with mainline of {len(mainline)} cards")
            logger.debug(f"  card.rank={card.rank}, card.color={card.color}")

            result = self._eval_function(card, mainline)
            rule_desc_short = self.rule_description[:100]
            logger.debug(
                f"Evaluating {card} against Python rule : {bool(result)} "
                f"for rule '{rule_desc_short}...'"
            )

            # Log detailed info if result seems unexpected
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"  result={result}, type={type(result)}")

            return bool(result)
        except Exception as e:
            logger.error(f"Runtime error in Python rule: {e}")
            logger.error(f"  Card: {card}, card.rank={card.rank}, card.color={card.color}")
            logger.error(f"  Mainline length: {len(mainline)}")
            if mainline:
                logger.error(f"  Last mainline card: {mainline[-1]}")
            return False

    def description(self) -> str:
        """Return the rule description."""
        return self.rule_description

    def get_code(self) -> str:
        """Return the Python code implementing this rule."""
        return self.code
