"""LLM client implementations for Hugging Face Inference Providers."""

import json
import logging
import os
import re
import time
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
        max_tokens: int = 0
    ) -> None:
        """Initialize HuggingFace client using Inference Providers."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens

        os.environ["HF_TOKEN"] = self.api_key
        self.client = InferenceClient(bill_to="huggingface")

    def _call_api_with_retry(self, messages: list[dict]):
        """Core API call with retry logic and reasoning extraction."""
        logger.debug(
            f"Calling API with retry {self.max_tokens} tokens and the following messages:\n"
            f"{messages}"
        )
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                logger.debug(f"LLM response:\n{completion.choices[0]}")
                return completion.choices[0]

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")


    def generate(self, prompt: str) -> str:
        """Generate text completion from prompt."""
        messages = [{"role": "user", "content": prompt}]
        response = self._call_api_with_retry(messages)
        return response.message.content


    def generate_structured(
        self, prompt: str, xml_tag: str | None = None
    ) -> dict:
        """Generate structured JSON response with automatic continuation on truncation."""

        # Initial generation
        messages = [{"role": "user", "content": prompt}]
        response = self._call_api_with_retry(messages)
        response_message = response.message

        logger.info(f"Finish reason: {response.finish_reason}")

        # Check for truncation
        if response.finish_reason == "length":
            logger.warning(f"Response was truncated, attempting continuation.")
            return self._continue_and_parse(messages, response_message, xml_tag)

        return self._parse_structured_response(response_message.content, xml_tag)


    def _continue_and_parse(
        self,
        messages: list[dict],
        partial_response_message,
        xml_tag: str | None
    ) -> dict[str, Any]:
        """Continue truncated response using multi-turn conversation."""
        from eleusis.prompts import get_continuation_prompt

        # Add incomplete message and Request continuation
        messages.append({
            "role": "assistant",
            "reasoning": partial_response_message.reasoning,
            "content": partial_response_message.content,
        })
        tag_name = xml_tag or 'RESPONSE'
        messages.append({"role": "user", "content": get_continuation_prompt(tag_name)})
        continuation_response = self._call_api_with_retry(messages)
        continuation_message = continuation_response.message

        # Try parsing continuation alone first
        try:
            response = self.    _parse_structured_response(continuation_message.content, xml_tag)
            logger.warning("Successfully parsed structured response from continuation alone.")
            return response
        except ValueError:
            logger.error("Failed to parse structured response, continuation alone failed.")
            return {}


    def _extract_content_from_response(
        self,
        response_text: str,
        xml_tags: list[str] | None = None,
        try_code_blocks: bool = True,
    ) -> str:
        """Extract content from XML tags or markdown code blocks."""
        extracted = None

        # Try XML tags first
        if xml_tags:
            for tag in xml_tags:
                pattern = f"<{tag}>(.*?)</{tag}>"
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    break

        # Fallback to code blocks if requested and XML extraction failed
        if not extracted and try_code_blocks:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                extracted = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                extracted = response_text[start:end].strip()

        return extracted or response_text.strip()


    def _parse_structured_response(
        self, response_text: str, xml_tag: str | None = None
    ) -> dict[str, Any]:
        """Parse structured JSON response from text."""
        xml_tags = [xml_tag] if xml_tag else None
        json_text = self._extract_content_from_response(
            response_text, xml_tags, try_code_blocks=True
        )

        try:
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")



    def check_rule_equivalence(
        self, rule1: str, rule2: str, mainline: str
    ) -> tuple[bool, str]:
        """Check if two rules are logically equivalent."""
        from eleusis.prompts import get_referee_comparison_prompt

        prompt = get_referee_comparison_prompt(rule1, rule2, mainline)
        response_text = self.generate(prompt)

        json_text = self._extract_content_from_response(
            response_text, ["VERDICT"], try_code_blocks=True
        )
        result = json.loads(json_text)
        return result["equivalent"], result["reasoning"]


    def convert_rule_to_code(self, rule_text: str) -> str | None:
        """Convert natural language rule to Python code."""
        from eleusis.prompts import get_rule_compilation_prompt

        prompt = get_rule_compilation_prompt(rule_text)
        response_text = self.generate(prompt)

        code = self._extract_content_from_response(response_text, ["CODE"], try_code_blocks=False)
        return code


    def evaluate_rule_on_card(self, rule_text: str, card: dict, mainline: list[dict]) -> bool:
        """Evaluate if card is IN or OUT according to rule."""
        from eleusis.prompts import get_card_evaluation_prompt

        prompt = get_card_evaluation_prompt(rule_text, card, mainline)
        response_text = self.generate(prompt)

        json_text = self._extract_content_from_response(
            response_text, xml_tags=None, try_code_blocks=True
        )
        result = json.loads(json_text)
        return result["result"].lower() == "in"