"""OpenAI GPT API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

from openai import APIError, OpenAI
from openai.types.responses import Response, ResponseInputParam

from eleusis.benchmark_config import OpenAIReasoningEffort
from eleusis.llm.base import BaseLLMClient, LLMCallMetrics, LLMMessage

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

    @staticmethod
    def _build_response_input(messages: list[LLMMessage]) -> ResponseInputParam:
        """Convert provider-neutral messages to Responses API input messages."""
        input_messages: ResponseInputParam = []
        for message in messages:
            role = message["role"]
            input_messages.append(
                {
                    "role": "developer" if role == "system" else role,
                    "content": message["content"],
                }
            )
        return input_messages

    def _create_response(
        self,
        input_messages: ResponseInputParam,
        disable_thinking: bool,
    ) -> Response:
        """Issue one Responses API request with the configured reasoning mode."""
        effort = "none" if disable_thinking else self.reasoning_effort
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

    @staticmethod
    def _parse_response_choice(response: Response) -> OpenAIChoice:
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
        response: Response,
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
