"""OpenAI GPT API client implementation."""

import json
import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import Any, Literal, cast

import httpx
from openai import APIError, OpenAI
from openai._streaming import Stream
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseFunctionToolCall,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseReasoningItem,
    ResponseStreamEvent,
)
from openai.types.responses.response_usage import ResponseUsage

from eleusis.benchmark_config import OpenAIReasoningEffort
from eleusis.llm.base import (
    BaseLLMClient,
    LLMCallMetrics,
    LLMMessage,
    ProviderUnavailableError,
)
from eleusis.llm.pi_auth import CODEX_BACKEND_BASE_URL, PiCodexAuth

logger = logging.getLogger(__name__)

ReasoningExtraction = Literal["deep_think"]


class CodexRunawayToolArgumentsError(RuntimeError):
    """A streamed tool-call argument stream degenerated into a generation loop.

    The Codex subscription backend rejects max_output_tokens, so a model
    that externalizes long reasoning into tool arguments can occasionally
    loop without bound after finishing its thought text. The consumer
    aborts such streams; the deep_think loop salvages the partial argument
    text and continues with a fresh round.
    """

    def __init__(self, message: str, *, partial_arguments: str, call_id: str) -> None:
        """Store the aborted stream's partial tool-call payload."""
        super().__init__(message)
        self.partial_arguments = partial_arguments
        self.call_id = call_id


# Scratchpad tool used to externalize reasoning the provider hides: native
# thinking is disabled and the model reasons by calling deep_think, whose
# arguments carry the chain-of-thought in plain sight.
DEEP_THINK_TOOL: dict[str, object] = {
    "type": "function",
    "name": "deep_think",
    "description": (
        "Private scratchpad for your internal reasoning before the final"
        " answer. It will not be made visible to the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {"thoughts": {"type": "string"}},
        "required": ["thoughts"],
        "additionalProperties": False,
    },
}

DEEP_THINK_INSTRUCTION = (
    "Solve the user task exactly and follow its output format. Use the"
    " deep_think tool as a private scratchpad for your reasoning before"
    " answering. It will not be made visible to the user."
)


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
class _CodexFunctionCallItem:
    """One function-call output item from a Codex stream."""

    type: str
    name: str
    arguments: str
    call_id: str


@dataclass
class CodexStreamPayload:
    """Response-shaped payload accumulated from Codex stream events.

    The Codex subscription backend delivers message text and reasoning only
    as stream events; its response.completed event carries empty output, so
    the client assembles this payload for the shared choice/metrics parsing.
    """

    output_text: str = ""
    output: list[Any] = field(default_factory=list)
    reasoning_summary_text: str = ""
    status: str = "completed"
    incomplete_details: _CodexIncompleteDetails | None = None
    usage: _CodexUsage | None = None


def _salvage_deep_think_thoughts(partial_arguments: str) -> str | None:
    """Recover the thoughts text from an unterminated deep_think argument.

    Degenerate streams end mid-JSON after real reasoning text; keep that
    text and drop the trailing whitespace loop it degenerated into.
    """
    match = re.search(r'"thoughts"\s*:\s*"(.*)$', partial_arguments, re.DOTALL)
    if match is None:
        return None
    text = match.group(1)
    for escaped, literal in (
        ("\\n", "\n"),
        ("\\t", "\t"),
        ('\\"', '"'),
        ("\\\\", "\\"),
    ):
        text = text.replace(escaped, literal)
    text = text.rstrip(" \t\r\n")
    return text or None


def _close_quietly(stream: Stream[ResponseStreamEvent]) -> None:
    """Abort an SSE stream without letting close errors mask the abort."""
    try:
        stream.close()
    except Exception:  # best-effort abort
        logger.debug("Codex stream close during abort failed", exc_info=True)


def _deep_think_thoughts(arguments: str) -> str:
    """Extract the thoughts payload from one deep_think tool-call argument."""
    try:
        thoughts = json.loads(arguments).get("thoughts")
    except (json.JSONDecodeError, AttributeError):
        thoughts = None
    if not isinstance(thoughts, str) or not thoughts:
        return arguments
    return thoughts


def _payload_function_calls(
    payload: Response | CodexStreamPayload,
) -> list[tuple[str, str, str]]:
    """Return (name, arguments, call_id) for each function-call output item."""
    calls: list[tuple[str, str, str]] = []
    for item in payload.output:
        if isinstance(item, _CodexFunctionCallItem):
            calls.append((item.name, item.arguments, item.call_id))
        elif getattr(item, "type", None) == "function_call":
            sdk_call = cast("ResponseFunctionToolCall", item)
            calls.append((sdk_call.name, sdk_call.arguments, sdk_call.call_id))
    return calls


def _payload_reasoning_summaries(
    payload: Response | CodexStreamPayload,
) -> list[str]:
    """Return reasoning summary texts carried by the payload's items."""
    summaries: list[str] = []
    for item in payload.output:
        if isinstance(item, _CodexReasoningItem):
            summaries.extend(block.text for block in item.summary)
        elif getattr(item, "type", None) == "reasoning":
            sdk_item = cast("ResponseReasoningItem", item)
            summaries.extend(block.text for block in sdk_item.summary)
    return summaries


@dataclass
class _CodexArgumentRun:
    """Bounds state for one stream's tool-call argument deltas."""

    chars: int = 0
    whitespace_run: int = 0
    arguments: str = ""
    call_id: str = ""

    def observe(self, delta: str) -> None:
        """Fold one argument delta into the running totals."""
        self.chars += len(delta)
        self.arguments += delta
        self.whitespace_run = self.whitespace_run + 1 if not delta.strip() else 0


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT API with reasoning effort support.

    Note: This client uses the Responses API which does not support a seed
    parameter. The seed is stored but not passed to the API. The older Chat
    Completions API had seed support but it's being deprecated.
    """

    # Scratchpad rounds before the final no-tools round forces a text answer.
    MAX_DEEP_THINK_ROUNDS = 4

    # Client-side bounds for streamed tool-call arguments: the backend
    # rejects max_output_tokens, so runaway argument generation must be
    # aborted locally. Generous enough for long legitimate reasoning traces.
    MAX_TOOL_ARGUMENT_CHARS = 65_536
    MAX_CONSECUTIVE_WHITESPACE_DELTAS = 512

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
        reasoning_extraction: ReasoningExtraction | None = None,
        stream_deadline_seconds: float = 300.0,
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
            reasoning_extraction: When "deep_think", disable native thinking
                and externalize reasoning through the deep_think scratchpad
                tool so hidden chain-of-thought is captured as evidence.
            stream_deadline_seconds: Wall-clock bound on consuming one
                streaming response. Capacity-starved backends can trickle
                keepalive bytes forever, which resets per-read timeouts; only
                a deadline bounds such attempts.
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
        self.stream_deadline_seconds = stream_deadline_seconds
        self.codex_auth = codex_auth
        self.reasoning_extraction = reasoning_extraction
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
        """Make one API call, or the deep_think scratchpad loop."""
        if self.reasoning_extraction == "deep_think" and not disable_thinking:
            return self._call_api_deep_think(
                messages,
                is_continuation=is_continuation,
                continuation_depth=continuation_depth,
            )
        input_messages = self._build_response_input(messages)
        effort = "none" if disable_thinking else self.reasoning_effort
        _, choice, metrics = self._request_once(
            input_messages,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            effort=effort,
        )
        return choice, metrics

    def _request_once(
        self,
        input_messages: ResponseInputParam,
        *,
        is_continuation: bool,
        continuation_depth: int,
        effort: OpenAIReasoningEffort,
        tools: list[dict[str, object]] | None = None,
    ) -> tuple[Response | CodexStreamPayload, OpenAIChoice, LLMCallMetrics]:
        """Issue one Responses API request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self._create_response(
                    input_messages,
                    effort,
                    tools=tools,
                )
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
                return response, choice, metrics

            except APIError as e:
                logger.warning(
                    f"{self.model_name} Attempt {attempt + 1}/{self.max_retries}"
                    f" failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise
            except (httpx.HTTPError, ProviderUnavailableError) as e:
                # Transport drops and stalled streams are provider-capacity
                # failures: back off long enough for congestion to clear.
                logger.warning(
                    f"{self.model_name} Attempt {attempt + 1}/{self.max_retries}"
                    f" provider unavailable: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(min(15 * 2**attempt, 120))
                else:
                    raise ProviderUnavailableError(
                        f"{self.model_name} provider unavailable after"
                        f" {self.max_retries} attempts: {e}"
                    ) from e

        raise RuntimeError("Max retries exceeded")

    def _call_api_deep_think(
        self,
        messages: list[LLMMessage],
        *,
        is_continuation: bool,
        continuation_depth: int,
    ) -> tuple[OpenAIChoice, LLMCallMetrics]:
        """Run the deep_think scratchpad loop for one generation.

        Each round issues one provider call. Rounds that only reason append
        their own metrics to call_metrics (marked as continuations); the
        final text round's metrics are returned for BaseLLMClient.generate
        to append, so every round lands in call_metrics exactly once.
        """
        history = self._build_response_input(
            [{"role": "system", "content": DEEP_THINK_INSTRUCTION}, *messages]
        )
        # Reasoning chunks salvaged from aborted runaway streams; attached to
        # the next round's metrics so every trace character lands in exactly
        # one provider-call record.
        pending_salvage: list[str] = []
        for round_index in range(self.MAX_DEEP_THINK_ROUNDS + 1):
            final_round = round_index == self.MAX_DEEP_THINK_ROUNDS
            tools = None if final_round else [DEEP_THINK_TOOL]
            try:
                payload, choice, metrics = self._request_once(
                    history,
                    is_continuation=is_continuation or round_index > 0,
                    continuation_depth=continuation_depth + round_index,
                    effort="none",
                    tools=tools,
                )
            except CodexRunawayToolArgumentsError as error:
                # The model finished its thought but degenerated instead of
                # closing the argument; keep the reasoning and continue with
                # a fresh round rather than re-rolling the same request.
                salvaged = _salvage_deep_think_thoughts(error.partial_arguments)
                if salvaged:
                    pending_salvage.append(salvaged)
                history.append(
                    {
                        "type": "function_call",
                        "call_id": error.call_id,
                        "name": "deep_think",
                        "arguments": error.partial_arguments,
                    }
                )
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": error.call_id,
                        "output": json.dumps({"recorded": True}),
                    }
                )
                continue
            think_calls = _payload_function_calls(payload)
            if not think_calls:
                if pending_salvage:
                    metrics.reasoning_text = "\n\n".join(pending_salvage)
                return choice, metrics
            if final_round:
                raise RuntimeError(
                    "deep_think extraction exceeded"
                    f" {self.MAX_DEEP_THINK_ROUNDS} scratchpad rounds without"
                    " a final answer"
                )
            unexpected = [name for name, _, _ in think_calls if name != "deep_think"]
            if unexpected:
                raise RuntimeError(
                    "deep_think extraction got unexpected tool call:"
                    f" {', '.join(unexpected)}"
                )
            round_chunks = [
                _deep_think_thoughts(arguments) for _, arguments, _ in think_calls
            ]
            metrics.reasoning_text = "\n\n".join(pending_salvage + round_chunks)
            pending_salvage.clear()
            # The round's entire provider-reported output is the externalized
            # chain of thought: count it as reasoning, not answer tokens.
            metrics.reasoning_tokens = metrics.output_tokens
            metrics.answer_tokens = 0
            self.call_metrics.append(metrics)
            for name, arguments, call_id in think_calls:
                history.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"recorded": True}),
                    }
                )
        raise RuntimeError("unreachable deep_think round bound")

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
        effort: OpenAIReasoningEffort,
        tools: list[dict[str, object]] | None = None,
    ) -> Response | CodexStreamPayload:
        """Issue one Responses API request with the requested reasoning mode."""
        if self.codex_auth is not None:
            # The Codex subscription backend only supports streaming, refuses
            # server-side storage, and delivers output only as stream events.
            # Reasoning summaries arrive only as streamed deltas, so any
            # effort that reasons also asks for a detailed summary.
            reasoning: dict[str, object] = {"effort": effort}
            if effort != "none":
                reasoning["summary"] = "detailed"
            kwargs: dict[str, object] = {}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            stream = self.client.responses.create(  # ty: ignore[no-matching-overload]
                model=self.model_name,
                input=input_messages,
                reasoning=reasoning,
                stream=True,
                store=False,
                **kwargs,
            )
            return self._consume_codex_stream(stream)
        return self.client.responses.create(
            model=self.model_name,
            input=input_messages,
            reasoning={"effort": effort},
        )

    _STREAM_DONE = object()

    def _deadline_bounded(self, stream: Stream[ResponseStreamEvent]) -> Iterator[Any]:
        """Yield stream events while enforcing the wall-clock deadline.

        A pump thread feeds a queue so a stalled stream cannot block past the
        deadline even when no events ever arrive. The pump is a daemon: an
        abandoned blocked read dies with the process.
        """
        queue: SimpleQueue[Any] = SimpleQueue()

        def pump() -> None:
            try:
                for event in stream:
                    queue.put(event)
            except BaseException as error:  # ruff: ignore[blind-except]
                # Pump failures surface to the consumer through the queue.
                queue.put(error)
            else:
                queue.put(self._STREAM_DONE)

        threading.Thread(target=pump, daemon=True, name="codex-stream-deadline").start()
        deadline = time.monotonic() + self.stream_deadline_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderUnavailableError(
                    "provider stream exceeded"
                    f" {self.stream_deadline_seconds:.0f}s wall-clock deadline"
                )
            try:
                item = queue.get(timeout=remaining)
            except Empty:
                raise ProviderUnavailableError(
                    "provider stream exceeded"
                    f" {self.stream_deadline_seconds:.0f}s wall-clock deadline"
                ) from None
            if item is self._STREAM_DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _consume_codex_stream(
        self, stream: Stream[ResponseStreamEvent]
    ) -> CodexStreamPayload:
        """Accumulate one Codex subscription stream into a response payload.

        Message text and reasoning arrive only as stream events; the
        response.completed event carries the terminal status and usage.
        """
        payload = CodexStreamPayload()
        argument_run = _CodexArgumentRun()
        for event in self._deadline_bounded(stream):
            if event.type == "response.output_item.done":
                self._apply_codex_output_item(payload, event.item)
            elif event.type == "response.output_item.added":
                item = event.item
                if getattr(item, "type", None) == "function_call":
                    added_call = cast("ResponseFunctionToolCall", item)
                    argument_run.call_id = added_call.call_id or ""
            elif event.type == "response.reasoning_summary_text.delta":
                payload.reasoning_summary_text += event.delta
            elif event.type == "response.function_call_arguments.delta":
                argument_run.observe(event.delta)
                if self._codex_argument_runaway(argument_run):
                    _close_quietly(stream)
                    raise self._runaway_error(argument_run)
            elif event.type == "response.completed":
                return self._complete_codex_payload(payload, event.response)
            elif event.type == "response.failed":
                raise RuntimeError(
                    f"Codex subscription response failed: {event.response.error}"
                )
        raise RuntimeError("Codex subscription stream ended without response.completed")

    @staticmethod
    def _runaway_error(run: _CodexArgumentRun) -> CodexRunawayToolArgumentsError:
        """Build the abort error carrying the partial tool-call payload."""
        return CodexRunawayToolArgumentsError(
            "Codex tool-call arguments exceeded client bounds"
            f" ({run.chars} chars, whitespace run {run.whitespace_run});"
            " aborting the stream",
            partial_arguments=run.arguments,
            call_id=run.call_id or f"call_salvaged_{int(time.time() * 1000)}",
        )

    @staticmethod
    def _complete_codex_payload(
        payload: CodexStreamPayload, response: object
    ) -> CodexStreamPayload:
        """Fold one response.completed event into the final payload."""
        payload.status = getattr(response, "status", None) or "unknown"
        incomplete = getattr(response, "incomplete_details", None)
        if incomplete is not None:
            payload.incomplete_details = _CodexIncompleteDetails(
                reason=getattr(incomplete, "reason", None) or ""
            )
        usage = getattr(response, "usage", None)
        if usage is not None:
            payload.usage = OpenAIClient._codex_usage(cast("ResponseUsage", usage))
        return payload

    @classmethod
    def _codex_argument_runaway(cls, run: _CodexArgumentRun) -> bool:
        """Whether streamed tool arguments crossed the runaway bounds."""
        return (
            run.chars >= cls.MAX_TOOL_ARGUMENT_CHARS
            or run.whitespace_run >= cls.MAX_CONSECUTIVE_WHITESPACE_DELTAS
        )

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
        elif item.type == "function_call":
            payload.output.append(
                _CodexFunctionCallItem(
                    type="function_call",
                    name=item.name,
                    arguments=item.arguments,
                    call_id=item.call_id,
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
        reasoning_parts = _payload_reasoning_summaries(response)
        # Subscription streams may deliver summaries only as deltas rather
        # than completed reasoning items; fall back to the streamed text.
        streamed_summary = getattr(response, "reasoning_summary_text", "")
        finish_reason = "stop"
        if (
            response.status == "incomplete"
            and response.incomplete_details
            and response.incomplete_details.reason == "max_output_tokens"
        ):
            finish_reason = "length"
        reasoning_content = "".join(reasoning_parts) or streamed_summary
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
            reasoning_text=choice.message.reasoning,
        )

        logger.debug(
            f"[OpenAI] Metrics summary: {output_tokens} output tokens in"
            f" {duration:.2f}s ({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
