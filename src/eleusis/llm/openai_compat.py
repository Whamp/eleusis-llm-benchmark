"""OpenAI-compatible API client for self-hosted models (SGLang, vLLM, etc.)."""

import logging
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

from openai import APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

from eleusis.llm.base import (
    BaseLLMClient,
    LLMCallMetrics,
    LLMMessage,
    ProviderRejectionError,
    estimate_reasoning_tokens,
)
from eleusis.llm.openai_messages import build_openai_chat_messages

logger = logging.getLogger(__name__)


@dataclass
class CompatMessage:
    """Message wrapper for OpenAI-compat responses."""

    content: str
    reasoning: str | None = None


@dataclass
class CompatChoice:
    """Choice wrapper for OpenAI-compat responses."""

    message: CompatMessage
    finish_reason: str


@dataclass
class CompatStreamResult:
    """Content, reasoning, finish state, and usage consumed from one stream."""

    content: str
    reasoning: str
    finish_reason: str
    usage: CompletionUsage | None


class OpenAICompatClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs (SGLang, vLLM, etc.).

    Supports models that use reasoning_content field (Qwen3 thinking) or <think> tags in
    content (DeepSeek R1 style).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_format: str = "reasoning_content",
        timeout: int = 600,
        reasoning_effort: str | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> None:
        """Initialize OpenAI-compatible client.

        Args:
            model_name: Model ID as served by the endpoint.
            base_url: Base URL of the OpenAI-compatible API (e.g. http://host:30000/v1).
            api_key: API key (most self-hosted servers accept any string).
            temperature: Sampling temperature.
            max_retries: Maximum endpoint call attempts.
            max_tokens: Maximum generated tokens.
            role: Metrics label for this client.
            seed: Optional generation seed.
            reasoning_format: How reasoning appears in responses:
                "reasoning_content" - separate reasoning_content field (Qwen3)
                "think_tags" - <think>...</think> in content (DeepSeek R1)
                "none" - no reasoning content
            timeout: Request timeout in seconds.
            reasoning_effort: Optional reasoning-effort level forwarded on every
                request (e.g. "low"/"medium"/"high" for OpenRouter models).
            extra_body: Optional request-body fields merged into every call, used
                for gateway routing options such as OpenRouter provider pinning.
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key or "sk-no-key-required",
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.base_url = base_url
        self.reasoning_format = reasoning_format
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def provider_name(self) -> str:
        """Provider name used in metrics and logs."""
        return "openai_compat"

    def _call_api(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[CompatChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling OpenAI-compat API at {self.base_url} with {self.max_tokens}"
            " tokens"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                chat_messages = build_openai_chat_messages(messages)
                request_kwargs: dict[str, object] = {}
                if self.reasoning_effort is not None:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
                if self.extra_body is not None:
                    request_kwargs["extra_body"] = self.extra_body
                stream = self.client.chat.completions.create(  # ty: ignore[no-matching-overload]
                    model=self.model_name,
                    messages=chat_messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                    seed=self.seed,
                    # Optional gateway params are merged conditionally above, so
                    # the overload cannot see their static types.
                    **request_kwargs,
                )

                stream_result = self._consume_completion_stream(stream)
                end_time = time.time()
                choice = CompatChoice(
                    message=CompatMessage(
                        content=stream_result.content,
                        reasoning=stream_result.reasoning or None,
                    ),
                    finish_reason=stream_result.finish_reason,
                )

                metrics = self._extract_metrics(
                    stream_result.content,
                    choice,
                    start_time,
                    end_time,
                    is_continuation,
                    continuation_depth,
                    stream_result.usage,
                )

                return choice, metrics

            except APIStatusError as e:
                # Permanent 4xx refusals (moderation, invalid request, auth)
                # cannot succeed on an identical retry: fail immediately with
                # the typed rejection. 5xx and 429 stay on the capacity path.
                if 400 <= e.status_code < 500 and e.status_code not in (408, 429):
                    raise ProviderRejectionError(
                        f"{self.model_name} provider rejected the request"
                        f" (HTTP {e.status_code}): {e}"
                    ) from e
                raise
            except APIError as e:
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
        stream: Iterable[ChatCompletionChunk],
    ) -> CompatStreamResult:
        """Consume one completion stream while preserving progress output."""
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
            reasoning_parts.extend(OpenAICompatClient._extract_delta_reasoning(delta))
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        if dots_printed:
            sys.stdout.write("\n")
        reasoning = "".join(reasoning_parts)
        if reasoning:
            logger.debug(
                f"[compat] Reasoning captured ({len(reasoning)} chars): "
                f"{reasoning[:200]}..."
            )
        return CompatStreamResult(
            content="".join(content_parts),
            reasoning=reasoning,
            finish_reason=finish_reason,
            usage=usage,
        )

    @staticmethod
    def _extract_delta_reasoning(delta: object) -> list[str]:
        """Return nonstandard reasoning text exposed by compatible servers."""
        values = [
            getattr(delta, "reasoning_content", None),
            getattr(delta, "reasoning", None),
        ]
        return [value for value in values if isinstance(value, str)]

    def _extract_metrics(
        self,
        content: str,
        choice: CompatChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
        usage: object | None = None,
    ) -> LLMCallMetrics:
        """Extract metrics from streaming response with normalized token fields."""
        duration = end_time - start_time

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            api_completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            logger.debug(
                f"[compat] API usage: prompt_tokens={prompt_tokens}, "
                f"completion_tokens={api_completion_tokens}"
            )
        else:
            prompt_tokens = 0
            content_word_count = len(content.split())
            api_completion_tokens = int(content_word_count * 1.3)
            logger.debug(
                f"[compat] Estimating tokens from content: {api_completion_tokens}"
            )

        has_reasoning = False
        reasoning_tokens = 0
        output_tokens = api_completion_tokens

        if self.reasoning_format == "reasoning_content":
            # Separate field (Qwen3 via SGLang/vLLM)
            if choice.message.reasoning:
                has_reasoning = True
                reasoning_text = choice.message.reasoning
                reasoning_word_count = len(reasoning_text.split())
                logger.debug(f"[compat] Reasoning field: {reasoning_word_count} words")

            # Estimate answer tokens from visible content
            content_word_count = len(content.split())
            answer_tokens = int(content_word_count * 1.3)
            reasoning_tokens = max(0, output_tokens - answer_tokens)

        elif self.reasoning_format == "think_tags":
            # Inline <think>...</think> tags
            has_think_tags = "<think>" in content or "</think>" in content
            if has_think_tags:
                has_reasoning = True
                reasoning_tokens = estimate_reasoning_tokens(content) or 0
            answer_tokens = max(0, output_tokens - reasoning_tokens)

        else:
            # No reasoning
            answer_tokens = output_tokens
            reasoning_tokens = 0

        logger.debug(
            f"[compat] FINAL: prompt={prompt_tokens}, output={output_tokens} "
            f"(answer={answer_tokens} + reasoning={reasoning_tokens})"
        )

        return LLMCallMetrics(
            model_name=self.model_name,
            role=self.role,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0,
            finish_reason=choice.finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )
