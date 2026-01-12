"""Prompt templates for LLM interactions in Eleusis."""

from eleusis.prompts.action import get_action_prompt
from eleusis.prompts.compile import get_rule_compile_prompt
from eleusis.prompts.game_rules import get_game_rules

__all__ = [
    "get_action_prompt",
    "get_game_rules",
    "get_rule_compile_prompt",
]
