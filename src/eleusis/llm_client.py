"""LLM client implementations for Hugging Face Inference Providers."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    model_name: str
    role: str  # "game_master" or player display name
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str
    has_reasoning: bool
    timestamp: float


class HuggingFaceClient:
    """Client for Hugging Face Inference Providers API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 0,
        role: str = "unknown"
    ) -> None:
        """Initialize HuggingFace client using Inference Providers."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.role = role
        self.call_metrics: list[LLMCallMetrics] = []

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
                # Start timing
                start_time = time.time()

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                # End timing
                end_time = time.time()
                choice = completion.choices[0]

                # Extract and track metrics
                if hasattr(completion, 'usage') and completion.usage:
                    usage = completion.usage
                    duration = end_time - start_time

                    metrics = LLMCallMetrics(
                        model_name=self.model_name,
                        role=self.role,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                        duration_seconds=duration,
                        throughput_tokens_per_sec=usage.completion_tokens / duration if duration > 0 else 0,
                        finish_reason=choice.finish_reason,
                        has_reasoning=hasattr(choice.message, 'reasoning') and choice.message.reasoning is not None and choice.message.reasoning != "",
                        timestamp=start_time,
                    )

                    self.call_metrics.append(metrics)
                    logger.debug(f"Metrics: {usage.completion_tokens} tokens in {duration:.2f}s ({metrics.throughput_tokens_per_sec:.2f} tok/s)")
                else:
                    logger.warning("No usage data available from LLM response")

                logger.debug(f"LLM response:\n{choice}")
                return choice

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

    def get_usage_stats(self) -> dict:
        """Get aggregated usage statistics for this client."""
        if not self.call_metrics:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "duration_seconds": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "call_count": 0,
            }

        # Aggregate metrics
        total_prompt = sum(m.prompt_tokens for m in self.call_metrics)
        total_completion = sum(m.completion_tokens for m in self.call_metrics)
        total_tokens = sum(m.total_tokens for m in self.call_metrics)
        total_duration = sum(m.duration_seconds for m in self.call_metrics)

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "duration_seconds": round(total_duration, 2),
            "throughput_tokens_per_sec": round(total_completion / total_duration if total_duration > 0 else 0, 2),
            "call_count": len(self.call_metrics),
        }

    def reset_usage_stats(self) -> None:
        """Reset call metrics (called at start of each round)."""
        self.call_metrics = []