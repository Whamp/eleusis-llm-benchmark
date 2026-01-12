"""HuggingFace Inference Providers client implementation."""

import logging
import os
import time

from huggingface_hub import InferenceClient

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


class HuggingFaceClient(BaseLLMClient):
    """Client for Hugging Face Inference Providers API."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        max_continuation_attempts: int = 3,
        seed: int | None = None,
    ) -> None:
        """Initialize HuggingFace client using Inference Providers."""
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("HF_TOKEN"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            max_continuation_attempts=max_continuation_attempts,
            seed=seed,
        )

        if self.api_key:
            os.environ["HF_TOKEN"] = self.api_key
        self.client = InferenceClient(bill_to="huggingface")

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling HF API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                api_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                if disable_thinking and self.reasoning_model_type:
                    if self.reasoning_model_type in ("gpt-oss", "deepseek-r1"):
                        logger.info("Attempting to disable thinking for continuation")

                completion = self.client.chat.completions.create(**api_kwargs)

                end_time = time.time()
                choice = completion.choices[0]

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
        from eleusis.llm.base import estimate_reasoning_tokens

        duration = end_time - start_time

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = None

        if hasattr(completion, 'usage') and completion.usage:
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            if hasattr(usage, 'reasoning_tokens'):
                reasoning_tokens = usage.reasoning_tokens

        has_reasoning = False
        if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            has_reasoning = True
            if reasoning_tokens is None:
                reasoning_tokens = estimate_reasoning_tokens(choice.message.reasoning)
        elif choice.message.content and "<think>" in choice.message.content:
            has_reasoning = True
            if reasoning_tokens is None:
                reasoning_tokens = estimate_reasoning_tokens(choice.message.content)

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
