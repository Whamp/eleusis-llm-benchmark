"""OpenAI GPT API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

from openai import OpenAI

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class OpenAIMessage:
    """Message wrapper for OpenAI responses."""
    content: str
    reasoning: str | None = None


@dataclass
class OpenAIChoice:
    """Choice wrapper for OpenAI responses."""
    message: OpenAIMessage
    finish_reason: str


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT API with reasoning effort support."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_effort: str = "medium",
    ) -> None:
        """Initialize OpenAI client."""
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.reasoning_effort = reasoning_effort
        self.client = OpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[OpenAIChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling OpenAI API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Convert messages format for Responses API
                # Responses API uses 'input' instead of 'messages' and 'developer' role
                input_messages = []
                for msg in messages:
                    role = msg["role"]
                    if role == "system":
                        role = "developer"
                    input_messages.append({"role": role, "content": msg["content"]})

                # Set reasoning effort
                effort = "none" if disable_thinking else self.reasoning_effort

                api_kwargs = {
                    "model": self.model_name,
                    "input": input_messages,
                    "reasoning": {"effort": effort},
                }

                # Add temperature for non-reasoning calls
                if disable_thinking:
                    api_kwargs["temperature"] = self.temperature

                response = self.client.responses.create(**api_kwargs)

                end_time = time.time()

                # Extract content from response
                text_content = ""
                reasoning_content = ""
                for item in response.output:
                    if hasattr(item, "content"):
                        for content_block in item.content:
                            if hasattr(content_block, "text"):
                                text_content += content_block.text
                    # Check for reasoning in the item
                    if hasattr(item, "type") and item.type == "reasoning":
                        if hasattr(item, "summary"):
                            for summary_block in item.summary:
                                if hasattr(summary_block, "text"):
                                    reasoning_content += summary_block.text

                # Determine finish reason
                finish_reason = "stop"
                if hasattr(response, "status") and response.status == "incomplete":
                    if hasattr(response, "incomplete_details"):
                        if response.incomplete_details.reason == "max_output_tokens":
                            finish_reason = "length"

                choice = OpenAIChoice(
                    message=OpenAIMessage(
                        content=text_content,
                        reasoning=reasoning_content if reasoning_content else None,
                    ),
                    finish_reason=finish_reason,
                )

                metrics = self._extract_metrics(
                    response, choice, start_time, end_time,
                    is_continuation, continuation_depth
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        response,
        choice: OpenAIChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response."""
        duration = end_time - start_time

        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = None

        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.input_tokens or 0
            completion_tokens = response.usage.output_tokens or 0
            if hasattr(response.usage, "output_tokens_details"):
                details = response.usage.output_tokens_details
                if hasattr(details, "reasoning_tokens"):
                    reasoning_tokens = details.reasoning_tokens

        total_tokens = prompt_tokens + completion_tokens
        has_reasoning = reasoning_tokens is not None and reasoning_tokens > 0

        # Estimate reasoning tokens from content if not provided
        if not has_reasoning and choice.message.reasoning:
            has_reasoning = True
            word_count = len(choice.message.reasoning.split())
            reasoning_tokens = int(word_count * 1.3)

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=completion_tokens / duration if duration > 0 else 0,
            finish_reason=choice.finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            reasoning_tokens=reasoning_tokens,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

        logger.debug(
            f"Metrics: {completion_tokens} tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
