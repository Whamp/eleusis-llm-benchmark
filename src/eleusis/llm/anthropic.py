"""Anthropic Claude API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

import anthropic

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class AnthropicMessage:
    """Message wrapper for Anthropic responses."""
    content: str
    reasoning: str | None = None


@dataclass
class AnthropicChoice:
    """Choice wrapper for Anthropic responses."""
    message: AnthropicMessage
    finish_reason: str


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API with extended thinking support."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_budget: int = 2048,
    ) -> None:
        """Initialize Anthropic client."""
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.reasoning_budget = reasoning_budget
        self.client = anthropic.Anthropic(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[AnthropicChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling Anthropic API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build thinking config
                if disable_thinking:
                    thinking_config = {"type": "disabled"}
                else:
                    thinking_config = {
                        "type": "enabled",
                        "budget_tokens": self.reasoning_budget,
                    }

                # Anthropic uses temperature 1.0 for thinking models
                api_kwargs = {
                    "model": self.model_name,
                    "max_tokens": self.max_tokens,
                    "messages": messages,
                    "thinking": thinking_config,
                }

                # Temperature must be 1.0 when thinking is enabled
                if not disable_thinking:
                    api_kwargs["temperature"] = 1.0
                else:
                    api_kwargs["temperature"] = self.temperature

                response = self.client.messages.create(**api_kwargs)

                end_time = time.time()

                # Extract content and reasoning from response
                text_content = ""
                reasoning_content = ""
                for block in response.content:
                    if block.type == "thinking":
                        reasoning_content = block.thinking
                    elif block.type == "text":
                        text_content = block.text

                choice = AnthropicChoice(
                    message=AnthropicMessage(
                        content=text_content,
                        reasoning=reasoning_content if reasoning_content else None,
                    ),
                    finish_reason=response.stop_reason or "end_turn",
                )

                metrics = self._extract_metrics(
                    response, choice, start_time, end_time,
                    is_continuation, continuation_depth
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except anthropic.APIError as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        response,
        choice: AnthropicChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response."""
        duration = end_time - start_time

        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        # Anthropic provides cache_read_input_tokens and cache_creation_input_tokens
        # but we focus on core metrics here

        has_reasoning = choice.message.reasoning is not None
        reasoning_tokens = None
        if has_reasoning and choice.message.reasoning:
            # Estimate reasoning tokens from text
            word_count = len(choice.message.reasoning.split())
            reasoning_tokens = int(word_count * 1.3)

        # Map Anthropic stop reasons to standard finish reasons
        finish_reason = choice.finish_reason
        if finish_reason == "end_turn":
            finish_reason = "stop"
        elif finish_reason == "max_tokens":
            finish_reason = "length"

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=completion_tokens / duration if duration > 0 else 0,
            finish_reason=finish_reason,
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
