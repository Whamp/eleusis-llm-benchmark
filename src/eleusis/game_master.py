"""GameMaster handles LLM-based rule operations."""

import logging

from eleusis.llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class GameMaster:
    """Manages rule creation, conversion, and comparison using LLM."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize game master with LLM client."""
        self.llm_client = llm_client

    def convert_rule_to_code(self, rule_description: str) -> str:
        """Convert natural language rule to Python code."""
        from eleusis.prompts import get_rule_compilation_prompt

        logger.debug(f"Converting rule to code: {rule_description}")

        prompt = get_rule_compilation_prompt(rule_description)
        code = self.llm_client.generate(prompt, xml_tag="CODE")
        logger.debug(f"Converted code:\n{code}")
        return code

