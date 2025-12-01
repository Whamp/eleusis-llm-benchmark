"""LLM client implementations for Hugging Face Inference Providers."""

import json
import logging
import os
from typing import Any

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """Client for Hugging Face Inference Providers API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        """Initialize HuggingFace client using Inference Providers."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not provided and not found in environment")

        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize InferenceClient with billing to HuggingFace
        os.environ["HF_TOKEN"] = self.api_key
        self.client = InferenceClient(bill_to="huggingface")

    def generate(self, prompt: str, max_tokens: int = 8192) -> str:
        """Generate text completion from prompt using chat completions API."""
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )

                response_text = completion.choices[0].message.content
                logger.debug(f"LLM reasoning: {completion.choices[0].message.reasoning}...")
                logger.debug(f"LLM response: {response_text}")
                return response_text

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def generate_structured(
        self, prompt: str, max_tokens: int = 8192, xml_tag: str | None = None
    ) -> dict[str, Any]:
        """Generate structured JSON response.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum tokens in response
            xml_tag: If provided, extract JSON from <TAG>...</TAG> (e.g., "ACTION", "GUESS")
        """
        response_text = self.generate(prompt, max_tokens)

        # Try to extract JSON from response
        try:
            import re

            json_text = None

            # First try XML tags if specified
            if xml_tag:
                pattern = f"<{xml_tag}>(.*?)</{xml_tag}>"
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    json_text = match.group(1).strip()

            # Fall back to code blocks
            if not json_text:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()

            # Last resort: use whole response
            if not json_text:
                json_text = response_text.strip()

            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")


class RefereeClient:
    """Client for referee LLM using HuggingFace Inference Providers."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        """Initialize referee client with HuggingFace API."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN not provided and not found in environment")

        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize InferenceClient with billing to HuggingFace
        os.environ["HF_TOKEN"] = self.api_key
        self.client = InferenceClient(bill_to="huggingface")

    def check_rule_equivalence(self, rule1: str, rule2: str) -> tuple[bool, str]:
        """Check if two rules are logically equivalent."""
        from eleusis.prompts import get_referee_comparison_prompt

        prompt = get_referee_comparison_prompt(rule1, rule2)

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response_text = completion.choices[0].message.content
        logger.debug(f"Referee response: {response_text}")

        # Parse JSON from VERDICT tags
        try:
            import re

            # Try to extract from <VERDICT> tags
            pattern = r"<VERDICT>(.*?)</VERDICT>"
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                json_text = match.group(1).strip()
            # Fall back to code blocks
            elif "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)
            return result["equivalent"], result["reasoning"]
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse referee response: {response_text}")
            raise ValueError(f"Invalid referee response: {e}")

    def evaluate_rule_on_card(self, rule_text: str, card: dict, mainline: list[dict]) -> bool:
        """Ask referee to evaluate if a card is IN or OUT according to a rule."""
        mainline_str = ", ".join([c["symbol"] for c in mainline]) if mainline else "empty"

        prompt = f"""You are evaluating a card according to a rule in the game Eleusis.

Rule: {rule_text}

Current mainline: {mainline_str}
Card to evaluate: {card['symbol']}

Determine if this card is IN (accepted) or OUT (rejected) according to the rule.

Respond with JSON:
{{
    "result": "in" or "out",
    "reasoning": "Brief explanation"
}}"""

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        response_text = completion.choices[0].message.content

        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text.strip()

            result = json.loads(json_text)
            return result["result"].lower() == "in"
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse evaluation response: {response_text}")
            raise ValueError(f"Invalid evaluation response: {e}")
