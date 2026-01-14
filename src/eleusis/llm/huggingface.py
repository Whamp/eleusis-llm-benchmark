"""HuggingFace Inference Providers client implementation."""

import logging
import os
import time
from dataclasses import dataclass

from huggingface_hub import InferenceClient

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics


@dataclass
class StreamedMessage:
    """Message wrapper for streaming responses."""
    content: str
    reasoning: str | None = None


@dataclass
class StreamedChoice:
    """Choice wrapper for streaming responses."""
    message: StreamedMessage
    finish_reason: str

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
        seed: int | None = None,
        stream: bool = True,
        hf_provider: str | None = None,
        reasoning_format: str = "separate_field",
    ) -> None:
        """Initialize HuggingFace client using Inference Providers.

        Args:
            hf_provider: Inference provider to use (e.g., "together", "novita").
                        If None, uses HuggingFace's default routing.
            reasoning_format: How reasoning is provided by the model:
                            "think_tags" - reasoning in <think>...</think> tags in content
                            "separate_field" - reasoning in separate API field
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("HF_TOKEN"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.stream = stream
        self.hf_provider = hf_provider
        self.reasoning_format = reasoning_format

        if self.api_key:
            os.environ["HF_TOKEN"] = self.api_key

        # Initialize client with provider if specified
        client_kwargs = {"bill_to": "huggingface"}
        if self.hf_provider:
            client_kwargs["provider"] = self.hf_provider
        self.client = InferenceClient(**client_kwargs)

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
        if self.stream:
            return self._call_api_streaming(
                messages, is_continuation, continuation_depth, disable_thinking
            )
        return self._call_api_non_streaming(
            messages, is_continuation, continuation_depth, disable_thinking
        )

    def _call_api_streaming(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[StreamedChoice, LLMCallMetrics]:
        """Make a streaming API call."""
        logger.debug(
            f"Calling HF API (streaming) with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                api_kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": True,
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                # HuggingFace Inference API doesn't support disable_thinking
                if disable_thinking:
                    logger.debug("disable_thinking not supported by HF API")

                stream = self.client.chat.completions.create(**api_kwargs)

                content = ""
                reasoning = ""
                finish_reason = "stop"
                chars_since_dot = 0
                dots_printed = 0
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            content += delta.content
                            chars_since_dot += len(delta.content)
                            # Print a dot every ~100 tokens (~400 chars)
                            if chars_since_dot >= 400:
                                print(".", end="", flush=True)
                                dots_printed += 1
                                chars_since_dot = 0
                        # Capture reasoning field if present (GPT-OSS, etc.)
                        if hasattr(delta, 'reasoning') and delta.reasoning:
                            reasoning += delta.reasoning
                        if chunk.choices[0].finish_reason:
                            finish_reason = chunk.choices[0].finish_reason
                if dots_printed > 0:
                    print()  # Newline after dots

                end_time = time.time()

                choice = StreamedChoice(
                    message=StreamedMessage(
                        content=content,
                        reasoning=reasoning if reasoning else None
                    ),
                    finish_reason=finish_reason,
                )

                metrics = self._extract_metrics_streaming(
                    content, choice, start_time, end_time,
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

    def _call_api_non_streaming(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a non-streaming API call."""
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

                # HuggingFace Inference API doesn't support disable_thinking
                if disable_thinking:
                    logger.debug("disable_thinking not supported by HF API")

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
        if self.reasoning_format == "separate_field":
            # Check for reasoning in separate API field
            if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                has_reasoning = True
                if reasoning_tokens is None:
                    reasoning_tokens = estimate_reasoning_tokens(choice.message.reasoning)
        elif self.reasoning_format == "think_tags":
            # Check for <think>...</think> tags in content
            if choice.message.content and ("<think>" in choice.message.content or "</think>" in choice.message.content):
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

    def _extract_metrics_streaming(
        self,
        content: str,
        choice: StreamedChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from streaming response (estimates token counts)."""
        from eleusis.llm.base import estimate_reasoning_tokens

        duration = end_time - start_time

        # Streaming doesn't provide token counts, so we estimate
        completion_tokens = int(len(content.split()) * 1.3)

        has_reasoning = False
        reasoning_tokens = None

        if self.reasoning_format == "separate_field":
            # Check for reasoning in separate API field
            if choice.message.reasoning:
                has_reasoning = True
                reasoning_tokens = estimate_reasoning_tokens(choice.message.reasoning)
        elif self.reasoning_format == "think_tags":
            # Check for <think>...</think> tags in content
            if content and ("<think>" in content or "</think>" in content):
                has_reasoning = True
                reasoning_tokens = estimate_reasoning_tokens(content)

        metrics = LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=0,  # Unknown for streaming
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,  # Best estimate
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
            f"Metrics (streaming, estimated): {completion_tokens} tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
