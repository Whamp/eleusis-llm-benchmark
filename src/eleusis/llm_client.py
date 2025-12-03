"""LLM client implementations for Hugging Face Inference Providers."""

import json
import logging
import os
import re
import time

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
        logger.debug(f"PROMPT:\n{messages[-1]['content']}")
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


    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: bool = False
    ) -> str | dict:
        """Generate text or structured response with automatic continuation.

        Args:
            prompt: The prompt string to send to LLM
            xml_tag: Optional XML tag name to extract content from
            return_dict: If True, parse extracted/raw content as JSON

        Returns:
            str: Raw or XML-extracted text (if return_dict=False)
            dict: Parsed JSON object (if return_dict=True)
        """

        # Make initial API call
        messages = [{"role": "user", "content": prompt}]
        response = self._call_api_with_retry(messages)
        response_message = response.message

        logger.info(f"Finish reason: {response.finish_reason}")

        # Handle truncation by continuing
        if response.finish_reason == "length":
            logger.warning("Response was truncated, attempting continuation.")
            return self._continue_response(messages, response_message, xml_tag, return_dict)

        # No truncation - process response
        content = response_message.content

        # Extract from XML tag if specified
        if xml_tag:
            content = self._extract_content_from_response(content, [xml_tag], try_code_blocks=True)

        # Parse as JSON if requested
        if return_dict:
            return json.loads(content)

        return content




    def _continue_response(
        self,
        messages: list[dict],
        partial_response_message,
        xml_tag: str | None,
        return_dict: bool
    ) -> str | dict:
        """Continue truncated response and return in requested format."""
        from eleusis.prompts import get_continuation_prompt

        # Append incomplete assistant message
        messages.append({
            "role": "assistant",
            "reasoning": partial_response_message.reasoning,
            "content": partial_response_message.content,
        })

        # Request continuation
        tag_name = xml_tag if xml_tag else "RESPONSE"
        messages.append({"role": "user", "content": get_continuation_prompt(tag_name)})

        # Get continuation
        continuation_response = self._call_api_with_retry(messages)
        continuation_message = continuation_response.message

        # Recursively handle further truncation
        if continuation_response.finish_reason == "length":
            logger.warning("Continuation was also truncated, continuing again...")
            return self._continue_response(messages, continuation_message, xml_tag, return_dict)

        # Combine partial + continuation
        combined_content = partial_response_message.content + continuation_message.content

        # Extract from XML tag if specified
        if xml_tag:
            combined_content = self._extract_content_from_response(
                combined_content, [xml_tag], try_code_blocks=True
            )

        # Parse as JSON if requested
        if return_dict:
            return json.loads(combined_content)

        return combined_content


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





    def check_rule_equivalence(
        self, rule1: str, rule2: str, mainline: str
    ) -> tuple[bool, str]:
        """Check if two rules are logically equivalent."""
        from eleusis.prompts import get_referee_comparison_prompt

        prompt = get_referee_comparison_prompt(rule1, rule2, mainline)
        result = self.generate(prompt, xml_tag="VERDICT", return_dict=True)
        return result["equivalent"], result["reasoning"]


    def convert_rule_to_code(self, rule_text: str) -> str | None:
        """Convert natural language rule to Python code."""
        from eleusis.prompts import get_rule_compilation_prompt

        prompt = get_rule_compilation_prompt(rule_text)
        return self.generate(prompt, xml_tag="CODE")


    def evaluate_rule_on_card(self, rule_text: str, card: dict, mainline: list[dict]) -> bool:
        """Evaluate if card is IN or OUT according to rule."""
        from eleusis.prompts import get_card_evaluation_prompt

        prompt = get_card_evaluation_prompt(rule_text, card, mainline)
        result = self.generate(prompt, return_dict=True)
        return result["result"].lower() == "in"