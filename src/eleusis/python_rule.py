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
            result = self._eval_function(card, mainline)
            return bool(result)
        except Exception as e:
            logger.error(f"Runtime error in Python rule: {e}")
            logger.error(f"Card: {card}, Mainline: {[str(c) for c in mainline]}")
            return False

    def description(self) -> str:
        """Return the rule description."""
        return self.rule_description

    def get_code(self) -> str:
        """Return the Python code implementing this rule."""
        return self.code
