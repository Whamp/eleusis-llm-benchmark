"""GameMaster handles LLM-based rule operations."""

import json
import logging

from eleusis.llm_client import HuggingFaceClient

logger = logging.getLogger(__name__)


class GameMaster:
    """Manages rule creation, conversion, and comparison using LLM."""

    def __init__(self, llm_client: HuggingFaceClient, max_retry_attempts: int = 3):
        """Initialize game master with LLM client."""
        self.llm_client = llm_client
        self.max_retry_attempts = max_retry_attempts

    def create_rule(self):
        """Generate a new rule with valid Python code (retries on failure)."""
        from eleusis.game_engine import Rule
        from eleusis.prompts import get_rule_generation_prompt

        last_error = None

        for attempt in range(self.max_retry_attempts):
            logger.info(f"Rule generation attempt {attempt + 1}/{self.max_retry_attempts}")

            prompt = get_rule_generation_prompt()
            response = self.llm_client.generate(prompt)

            # Extract <RULE> tags
            description, code = self._extract_rule_from_response(response)

            if not description or not code:
                last_error = "Failed to extract description and code from LLM response"
                logger.warning(last_error)
                continue

            # Try to compile code into Rule
            logger.info(f"Generated rule: {description}")
            logger.debug(f"Generated code:\n{code}")

            rule = Rule(description, code)
            logger.info("✓ Rule compiled successfully")
            return rule

        # All attempts failed
        error_msg = (
            f"Failed to generate valid rule after {self.max_retry_attempts} attempts. "
            f"Last error: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def _extract_rule_from_response(self, response: str) -> tuple[str, str]:
        """Extract rule description and code from LLM response."""
        import re

        # Extract <RULE> content
        rule_match = re.search(r"<RULE>(.*?)</RULE>", response, re.DOTALL | re.IGNORECASE)
        if not rule_match:
            logger.error("No <RULE> tags found in response")
            return "", ""

        rule_content = rule_match.group(1).strip()

        # Extract description and code from JSON
        try:
            rule_data = json.loads(rule_content)
            description = rule_data.get("description", "")
            code = rule_data.get("code", "")
            return description, code
        except json.JSONDecodeError:
            # Fallback: try to extract from text format
            desc_match = re.search(r'"description":\s*"([^"]+)"', rule_content)
            code_match = re.search(r'"code":\s*"([^"]+)"', rule_content)

            if desc_match and code_match:
                return desc_match.group(1), code_match.group(1)

            logger.error("Failed to parse rule JSON")
            return "", ""

    def convert_rule_to_code(self, rule_description: str) -> str:
        """Convert natural language rule to Python code."""
        from eleusis.prompts import get_rule_compilation_prompt

        logger.debug(f"Converting rule to code: {rule_description}")

        prompt = get_rule_compilation_prompt(rule_description)
        response = self.llm_client.generate(prompt)

        # Extract <CODE> tags
        import re

        code_match = re.search(r"<CODE>(.*?)</CODE>", response, re.DOTALL | re.IGNORECASE)
        if not code_match:
            error_msg = "Failed to extract code from <CODE> tags in LLM response"
            logger.error(error_msg)
            logger.error(f"Response was: {response}")
            raise ValueError(error_msg)

        code = code_match.group(1).strip()
        logger.debug(f"Converted code:\n{code}")
        return code

    def compare_rules_with_llm(
        self, rule1_desc: str, rule2_desc: str, mainline: str
    ) -> tuple[bool, str]:
        """Compare two rules using LLM reasoning (non-authoritative)."""
        from eleusis.prompts import get_referee_comparison_prompt

        logger.debug("Comparing rules with LLM (non-authoritative)")

        prompt = get_referee_comparison_prompt(rule1_desc, rule2_desc, mainline)
        response = self.llm_client.generate(prompt)

        # Extract <VERDICT> tags
        import re

        verdict_match = re.search(r"<VERDICT>(.*?)</VERDICT>", response, re.DOTALL | re.IGNORECASE)
        if not verdict_match:
            error_msg = "Failed to extract verdict from <VERDICT> tags"
            logger.error(error_msg)
            raise ValueError(error_msg)

        verdict_content = verdict_match.group(1).strip()

        # Parse JSON
        verdict_data = json.loads(verdict_content)
        equivalent = verdict_data.get("equivalent", False)
        reasoning = verdict_data.get("reasoning", "")

        return equivalent, reasoning
