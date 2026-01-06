"""OpenRouter API client implementation."""

import logging
import os
import time

from openai import OpenAI

from eleusis.providers.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseLLMClient):
    """Client for OpenRouter API (OpenAI-compatible)."""

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        max_continuation_attempts: int = 3,
        referer: str = "eleusis-benchmark",
        seed: int | None = None,
    ) -> None:
        """Initialize OpenRouter client.

        Args:
            model_name: OpenRouter model identifier (e.g., "anthropic/claude-3-sonnet")
            api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
            temperature: Sampling temperature
            max_retries: Max retry attempts on failure
            max_tokens: Maximum tokens to generate
            role: Role identifier for metrics
            max_continuation_attempts: Max continuation attempts for truncated responses
            referer: HTTP Referer header for OpenRouter
            seed: Random seed for reproducibility (not guaranteed by all models)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            max_continuation_attempts=max_continuation_attempts,
            seed=seed,
        )

        self.referer = referer
        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            default_headers={"HTTP-Referer": self.referer},
        )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling OpenRouter API with {self.max_tokens} tokens, messages:\n{messages}"
        )
        logger.debug(f"PROMPT:\n{messages[-1]['content']}")

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build API call kwargs
                api_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                # Add seed for reproducibility if provided
                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                # Try to disable thinking for reasoning models on continuation
                extra_body = {}
                if disable_thinking and self.reasoning_model_type:
                    logger.info("Attempting to disable thinking for continuation")
                    # Different models may support different parameters
                    if self.reasoning_model_type == "deepseek-r1":
                        # DeepSeek supports thinking parameter
                        extra_body["thinking"] = {"type": "disabled"}
                    elif self.reasoning_model_type in ("qwen-thinking", "gpt-oss"):
                        # Try generic thinking disable
                        extra_body["thinking"] = {"type": "disabled"}

                if extra_body:
                    api_kwargs["extra_body"] = extra_body

                completion = self.client.chat.completions.create(**api_kwargs)

                end_time = time.time()
                choice = completion.choices[0]

                # Extract metrics
                metrics = self._extract_metrics(
                    completion, choice, start_time, end_time,
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
        completion,
        choice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response."""
        duration = end_time - start_time

        # Default values
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = None

        if hasattr(completion, 'usage') and completion.usage:
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or 0

            # Check for reasoning tokens (DeepSeek R1 includes this)
            if hasattr(usage, 'reasoning_tokens'):
                reasoning_tokens = usage.reasoning_tokens

        # Check for reasoning content in message (DeepSeek R1 style)
        has_reasoning = False
        if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
            has_reasoning = True
        elif hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            has_reasoning = True
        # Also check for <think> tags in content (Qwen style)
        elif choice.message.content and "<think>" in choice.message.content:
            has_reasoning = True

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=completion_tokens / duration if duration > 0 else 0,
            finish_reason=choice.finish_reason or "unknown",
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
