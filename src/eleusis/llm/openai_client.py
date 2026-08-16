"""OpenAI GPT API client implementation."""

import logging
import os
import time
from dataclasses import dataclass, field

import httpx
from openai import APIError, OpenAI
from openai._streaming import Stream
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseStreamEvent,
)
from openai.types.responses.response_usage import ResponseUsage

from eleusis.benchmark_config import OpenAIReasoningEffort
from eleusis.llm.base import BaseLLMClient, LLMCallMetrics, LLMMessage
from eleusis.llm.pi_auth import CODEX_BACKEND_BASE_URL, PiCodexAuth

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


@dataclass
class _CodexSummaryBlock:
    """One reasoning summary block from a Codex stream."""

    text: str


@dataclass
class _CodexReasoningItem:
    """One reasoning output item from a Codex stream."""

    type: str
    summary: list[_CodexSummaryBlock]


@dataclass
class _CodexTokenDetails:
    """Token breakdown from a Codex stream response."""

    reasoning_tokens: int


@dataclass
class _CodexUsage:
    """Usage numbers from a Codex stream response."""

    input_tokens: int
    output_tokens: int
    output_tokens_details: _CodexTokenDetails | None = None


@dataclass
class _CodexIncompleteDetails:
    """Why a Codex stream response stopped early."""

    reason: str


@dataclass
class CodexStreamPayload:
    """Response-shaped payload accumulated from Codex stream events.

    The Codex subscription backend delivers message text and reasoning only
    as stream events; its response.completed event carries empty output, so
    the client assembles this payload for the shared choice/metrics parsing.
    """

    output_text: str = ""
    output: list[_CodexReasoningItem] = field(default_factory=list)
    status: str = "completed"
    incomplete_details: _CodexIncompleteDetails | None = None
    usage: _CodexUsage | None = None


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT API with reasoning effort support.

    Note: This client uses the Responses API which does not support a seed
    parameter. The seed is stored but not passed to the API. The older Chat
    Completions API had seed support but it's being deprecated.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        reasoning_effort: OpenAIReasoningEffort = "medium",
        codex_auth: PiCodexAuth | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            model_name: Model ID to request.
            api_key: Platform API key; unused in subscription mode.
            temperature: Sampling temperature.
            max_retries: Maximum endpoint call attempts.
            max_tokens: Maximum generated tokens.
            role: Metrics label for this client.
            seed: Stored for reproducibility metadata (Responses API has no seed).
            reasoning_effort: Reasoning-effort level sent to the API.
            codex_auth: When set, use ChatGPT-subscription credentials from
                pi's auth.json against the Codex backend instead of a platform
                API key.
        """
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
        self.codex_auth = codex_auth
        if codex_auth is None:
            self.client = OpenAI(api_key=self.api_key)
        else:
            # The subscription backend requires per-request OAuth headers;
            # api_key only satisfies the SDK's constructor requirement.
            self.client = OpenAI(
                api_key="pi-codex-subscription",
                base_url=CODEX_BACKEND_BASE_URL,
                http_client=httpx.Client(auth=codex_auth, timeout=600.0),
            )

    @property
    def provider_name(self) -> str:
        """Provider name used in metrics and logs."""
        return "openai"

    def _call_api(
        self,
        messages: list[LLMMessage],
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

                input_messages = self._build_response_input(messages)
                response = self._create_response(input_messages, disable_thinking)
                end_time = time.time()
                choice = self._parse_response_choice(response)

                metrics = self._extract_metrics(
                    response,
                    choice,
                    start_time,
                    end_time,
                    is_continuation,
                    continuation_depth,
                )

                logger.debug(f"LLM response:\n{choice}")
                return choice, metrics

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

    def _build_response_input(self, messages: list[LLMMessage]) -> ResponseInputParam:
        """Convert provider-neutral messages to Responses API input messages."""
        input_messages: ResponseInputParam = []
        for message in messages:
            role = message["role"]
            mapped_role = "developer" if role == "system" else role
            if self.codex_auth is None:
                input_messages.append(
                    {"role": mapped_role, "content": message["content"]}
                )
            else:
                # The Codex subscription backend rejects plain string content,
                # so every message becomes a typed input_text block. The
                # benchmark player only sends user/developer prompts (history
                # is embedded in the prompt text), so assistant input blocks
                # are not needed here.
                input_messages.append(
                    EasyInputMessageParam(
                        role=mapped_role,
                        content=[
                            ResponseInputTextParam(
                                type="input_text",
                                text=message["content"],
                            )
                        ],
                    )
                )
        return input_messages

    def _create_response(
        self,
        input_messages: ResponseInputParam,
        disable_thinking: bool,
    ) -> Response | CodexStreamPayload:
        """Issue one Responses API request with the configured reasoning mode."""
        effort = "none" if disable_thinking else self.reasoning_effort
        if self.codex_auth is not None:
            # The Codex subscription backend only supports streaming, refuses
            # server-side storage, and delivers output only as stream events.
            stream = self.client.responses.create(
                model=self.model_name,
                input=input_messages,
                reasoning={"effort": effort},
                stream=True,
                store=False,
            )
            return self._consume_codex_stream(stream)
        if disable_thinking:
            return self.client.responses.create(
                model=self.model_name,
                input=input_messages,
                reasoning={"effort": effort},
                temperature=self.temperature,
            )
        return self.client.responses.create(
            model=self.model_name,
            input=input_messages,
            reasoning={"effort": effort},
        )

    def _consume_codex_stream(
        self, stream: Stream[ResponseStreamEvent]
    ) -> CodexStreamPayload:
        """Accumulate one Codex subscription stream into a response payload.

        Message text and reasoning arrive only as stream events; the
        response.completed event carries the terminal status and usage.
        """
        payload = CodexStreamPayload()
        for event in stream:
            if event.type == "response.output_item.done":
                self._apply_codex_output_item(payload, event.item)
            elif event.type == "response.completed":
                response = event.response
                payload.status = response.status or "unknown"
                if response.incomplete_details is not None:
                    payload.incomplete_details = _CodexIncompleteDetails(
                        reason=response.incomplete_details.reason or ""
                    )
                if response.usage is not None:
                    payload.usage = self._codex_usage(response.usage)
                return payload
            elif event.type == "response.failed":
                raise RuntimeError(
                    f"Codex subscription response failed: {event.response.error}"
                )
        raise RuntimeError("Codex subscription stream ended without response.completed")

    @staticmethod
    def _apply_codex_output_item(
        payload: CodexStreamPayload, item: ResponseOutputItem
    ) -> None:
        """Fold one completed output item into the accumulating payload."""
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    payload.output_text += part.text
        elif item.type == "reasoning":
            payload.output.append(
                _CodexReasoningItem(
                    type="reasoning",
                    summary=[
                        _CodexSummaryBlock(text=block.text) for block in item.summary
                    ],
                )
            )

    @staticmethod
    def _codex_usage(usage: ResponseUsage) -> _CodexUsage:
        """Convert one completed-event usage object to the payload shape."""
        details = usage.output_tokens_details
        reasoning_tokens = (
            details.reasoning_tokens
            if details is not None and details.reasoning_tokens is not None
            else 0
        )
        return _CodexUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            output_tokens_details=(
                _CodexTokenDetails(reasoning_tokens=reasoning_tokens)
                if details is not None
                else None
            ),
        )

    @staticmethod
    def _parse_response_choice(
        response: Response | CodexStreamPayload,
    ) -> OpenAIChoice:
        """Convert one Responses API result to the provider-neutral choice shape."""
        text_content = response.output_text or ""
        reasoning_parts: list[str] = []
        for item in response.output:
            if item.type != "reasoning":
                continue
            for summary_block in item.summary:
                reasoning_parts.append(summary_block.text)
        finish_reason = "stop"
        if (
            response.status == "incomplete"
            and response.incomplete_details
            and response.incomplete_details.reason == "max_output_tokens"
        ):
            finish_reason = "length"
        reasoning_content = "".join(reasoning_parts)
        logger.debug(f"Response output_text: {text_content[:200]}...")
        return OpenAIChoice(
            message=OpenAIMessage(
                content=text_content,
                reasoning=reasoning_content or None,
            ),
            finish_reason=finish_reason,
        )

    def _extract_metrics(
        self,
        response: Response | CodexStreamPayload,
        choice: OpenAIChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response with normalized token fields.

        OpenAI's output_tokens includes reasoning tokens, so:
        - output_tokens = API output_tokens
        - reasoning_tokens = API reasoning_tokens
        - answer_tokens = output_tokens - reasoning_tokens
        """
        duration = end_time - start_time

        api_input_tokens = 0
        api_output_tokens = 0
        api_reasoning_tokens = None

        if hasattr(response, "usage") and response.usage:
            api_input_tokens = response.usage.input_tokens or 0
            api_output_tokens = response.usage.output_tokens or 0
            logger.debug(
                f"[OpenAI] RAW API usage: input_tokens={api_input_tokens},"
                f" output_tokens={api_output_tokens}"
            )

            if hasattr(response.usage, "output_tokens_details"):
                details = response.usage.output_tokens_details
                logger.debug(f"[OpenAI] output_tokens_details: {details}")
                if hasattr(details, "reasoning_tokens"):
                    api_reasoning_tokens = details.reasoning_tokens
                    logger.debug(
                        f"[OpenAI] RAW API reasoning_tokens={api_reasoning_tokens}"
                    )
        else:
            logger.debug("[OpenAI] No usage data in response")

        reasoning_text = choice.message.reasoning
        reasoning_word_count = len(reasoning_text.split()) if reasoning_text else 0
        logger.debug(
            f"[OpenAI] Reasoning content present: {reasoning_text is not None},"
            f" word_count={reasoning_word_count}"
        )
        if reasoning_text:
            logger.debug(f"[OpenAI] Reasoning preview: {reasoning_text[:200]}...")

        prompt_tokens = api_input_tokens
        output_tokens = api_output_tokens
        reasoning_tokens = api_reasoning_tokens or 0
        has_reasoning = reasoning_tokens > 0

        # Estimate reasoning tokens from content if not provided by API
        if not has_reasoning and reasoning_text:
            has_reasoning = True
            reasoning_tokens = int(reasoning_word_count * 1.3)
            logger.debug(
                f"[OpenAI] ESTIMATED reasoning_tokens: {reasoning_word_count} words x"
                f" 1.3 = {reasoning_tokens}"
            )
        elif api_reasoning_tokens:
            logger.debug(
                "[OpenAI] Using NATIVE reasoning_tokens from API:"
                f" {api_reasoning_tokens}"
            )

        answer_tokens = max(0, output_tokens - reasoning_tokens)

        logger.debug(
            f"[OpenAI] FINAL token counts: prompt={prompt_tokens},"
            f" output={output_tokens} (answer={answer_tokens} +"
            f" reasoning={reasoning_tokens})"
        )

        metrics = LLMCallMetrics(
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

        logger.debug(
            f"[OpenAI] Metrics summary: {output_tokens} output tokens in"
            f" {duration:.2f}s ({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
