"""HuggingFace Inference Providers client implementation."""

import logging
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

from huggingface_hub import (
    ChatCompletionOutput,
    ChatCompletionOutputComplete,
    ChatCompletionStreamOutput,
    ChatCompletionStreamOutputUsage,
    InferenceClient,
)
from huggingface_hub.errors import HfHubHTTPError, InferenceTimeoutError

from eleusis.benchmark_config import HuggingFaceProvider
from eleusis.llm.base import BaseLLMClient, LLMCallMetrics, LLMMessage
from eleusis.llm.huggingface_metrics import (
    HuggingFaceMetricInput,
    calculate_huggingface_metrics,
)


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


@dataclass
class HuggingFaceStreamResult:
    """Content, reasoning, finish state, and usage consumed from one stream."""

    content: str
    reasoning: str
    finish_reason: str
    usage: ChatCompletionStreamOutputUsage | None


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
        hf_provider: HuggingFaceProvider | None = None,
        reasoning_format: str = "separate_field",
        timeout: int = 300,
    ) -> None:
        """Initialize HuggingFace client using Inference Providers.

        Args:
            model_name: Hugging Face model ID.
            api_key: Hugging Face token; defaults to `HF_TOKEN`.
            temperature: Sampling temperature.
            max_retries: Maximum provider call attempts.
            max_tokens: Maximum generated tokens.
            role: Metrics label for this client.
            seed: Optional generation seed.
            stream: Whether to use streaming chat completions.
            hf_provider: Inference provider to use (e.g., "together", "novita").
                        If None, uses HuggingFace's default routing.
            reasoning_format: How reasoning is provided by the model:
                            "think_tags" - reasoning in content <think> tags
                            "separate_field" - reasoning in separate API field
            timeout: Request timeout in seconds (default 300s / 5 minutes).
            Billing defaults to the authenticated user's personal account. Set
            HF_BILL_TO to bill calls to a Hugging Face organization instead.
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
        self.timeout = timeout

        if self.api_key:
            os.environ["HF_TOKEN"] = self.api_key

        self.client = InferenceClient(
            api_key=self.api_key,
            timeout=self.timeout,
            provider=self.hf_provider,
            bill_to=os.getenv("HF_BILL_TO"),
        )

    @property
    def provider_name(self) -> str:
        """Provider name used in metrics and logs."""
        return "huggingface"

    def _call_api(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[StreamedChoice, LLMCallMetrics]:
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
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[StreamedChoice, LLMCallMetrics]:
        """Make a streaming API call."""
        logger.debug(
            f"Calling HF API (streaming) with {self.max_tokens} tokens,"
            f" messages:\n{messages}"
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
                    "stream_options": {"include_usage": True},
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                # HuggingFace Inference API doesn't support disable_thinking
                if disable_thinking:
                    logger.debug("disable_thinking not supported by HF API")

                stream = self.client.chat.completions.create(**api_kwargs)

                stream_result = self._consume_completion_stream(stream)
                end_time = time.time()
                choice = StreamedChoice(
                    message=StreamedMessage(
                        content=stream_result.content,
                        reasoning=stream_result.reasoning or None,
                    ),
                    finish_reason=stream_result.finish_reason,
                )

                metrics = self._extract_metrics_streaming(
                    stream_result.content,
                    choice,
                    start_time,
                    end_time,
                    is_continuation,
                    continuation_depth,
                    stream_result.usage,
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except (HfHubHTTPError, InferenceTimeoutError) as e:
                logger.warning(
                    f"{self.model_name} Attempt {attempt + 1}/{self.max_retries}"
                    f" failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    @staticmethod
    def _consume_completion_stream(
        stream: Iterable[ChatCompletionStreamOutput],
    ) -> HuggingFaceStreamResult:
        """Consume one Hugging Face stream while preserving progress output."""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        usage = None
        chars_since_dot = 0
        dots_printed = 0
        for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)
                chars_since_dot += len(delta.content)
                if chars_since_dot >= 400:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    dots_printed += 1
                    chars_since_dot = 0
            reasoning_parts.extend(HuggingFaceClient._extract_delta_reasoning(delta))
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if dots_printed:
            sys.stdout.write("\n")
        reasoning = "".join(reasoning_parts)
        if reasoning:
            logger.debug(
                f"[HF stream] Reasoning content captured ({len(reasoning)} chars):"
            )
            logger.debug(f"[HF stream] Reasoning preview: {reasoning[:500]}...")
        return HuggingFaceStreamResult(
            content="".join(content_parts),
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _extract_delta_reasoning(delta: object) -> list[str]:
        """Return nonstandard reasoning text exposed by inference providers."""
        values = [
            getattr(delta, "reasoning", None),
            getattr(delta, "reasoning_content", None),
        ]
        return [value for value in values if isinstance(value, str)]

    def _call_api_non_streaming(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[StreamedChoice, LLMCallMetrics]:
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
                    completion,
                    choice,
                    start_time,
                    end_time,
                    is_continuation,
                    continuation_depth,
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

            except (HfHubHTTPError, InferenceTimeoutError) as e:
                logger.warning(
                    f"{self.model_name} Attempt {attempt + 1}/{self.max_retries}"
                    f" failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def _extract_metrics(
        self,
        completion: ChatCompletionOutput,
        choice: ChatCompletionOutputComplete,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract normalized metrics from a non-streaming response."""
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens or 0 if usage else 0
        completion_tokens = usage.completion_tokens or 0 if usage else 0
        reasoning = getattr(choice.message, "reasoning", None)
        if not isinstance(reasoning, str):
            reasoning = getattr(choice.message, "reasoning_content", None)
        return calculate_huggingface_metrics(
            HuggingFaceMetricInput(
                model_name=self.model_name,
                role=self.role,
                reasoning_format=self.reasoning_format,
                content=choice.message.content or "",
                reasoning=reasoning if isinstance(reasoning, str) else None,
                finish_reason=choice.finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                start_time=start_time,
                end_time=end_time,
                is_continuation=is_continuation,
                continuation_depth=continuation_depth,
            )
        )

    def _extract_metrics_streaming(
        self,
        content: str,
        choice: StreamedChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
        usage: ChatCompletionStreamOutputUsage | None = None,
    ) -> LLMCallMetrics:
        """Extract normalized metrics from a streaming response."""
        prompt_tokens = usage.prompt_tokens or 0 if usage else 0
        if usage:
            completion_tokens = usage.completion_tokens or 0
            estimated = False
        else:
            completion_tokens = int(len(content.split()) * 1.3)
            estimated = True
        return calculate_huggingface_metrics(
            HuggingFaceMetricInput(
                model_name=self.model_name,
                role=self.role,
                reasoning_format=self.reasoning_format,
                content=content,
                reasoning=choice.message.reasoning,
                finish_reason=choice.finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                start_time=start_time,
                end_time=end_time,
                is_continuation=is_continuation,
                continuation_depth=continuation_depth,
                estimated=estimated,
            )
        )
