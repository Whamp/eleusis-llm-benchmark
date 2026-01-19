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
    """Client for Anthropic Claude API with extended thinking support.

    Note: Anthropic API does not support a seed parameter. The seed is stored
    but not passed to the API. Even at temperature=0, outputs may vary due to
    GPU non-determinism.
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
                logger.warning(f"{self.model_name} Attempt {attempt + 1}/{self.max_retries} failed: {e}")
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
        """Extract metrics from API response with normalized token fields.

        Anthropic's output_tokens INCLUDES thinking tokens (thinking may be summarized
        in the returned block, but token count reflects actual usage). So:
        - output_tokens = API output_tokens (ground truth)
        - answer_tokens = estimated from visible content text
        - reasoning_tokens = output_tokens - answer_tokens
        """
        duration = end_time - start_time

        # --- RAW API VALUES ---
        api_input_tokens = response.usage.input_tokens
        api_output_tokens = response.usage.output_tokens
        logger.debug(f"[Anthropic] RAW API usage: input_tokens={api_input_tokens}, output_tokens={api_output_tokens}")

        # Check for any additional usage fields
        if hasattr(response.usage, 'cache_read_input_tokens'):
            logger.debug(f"[Anthropic] Cache tokens: read={getattr(response.usage, 'cache_read_input_tokens', None)}, "
                        f"creation={getattr(response.usage, 'cache_creation_input_tokens', None)}")

        # --- CONTENT ANALYSIS ---
        has_reasoning = choice.message.reasoning is not None
        reasoning_text = choice.message.reasoning if has_reasoning else None
        reasoning_word_count = len(reasoning_text.split()) if reasoning_text else 0
        logger.debug(f"[Anthropic] Reasoning block present: {has_reasoning}, word_count={reasoning_word_count} (may be summarized)")
        if reasoning_text:
            logger.debug(f"[Anthropic] Reasoning preview: {reasoning_text[:200]}...")

        content_text = choice.message.content or ""
        content_word_count = len(content_text.split())
        logger.debug(f"[Anthropic] Content text: {content_word_count} words")

        # --- COMPUTED VALUES ---
        # output_tokens from API is the ground truth (includes both thinking + answer)
        prompt_tokens = api_input_tokens
        output_tokens = api_output_tokens

        # Estimate answer tokens from visible content
        answer_tokens = int(content_word_count * 1.3)
        logger.debug(f"[Anthropic] ESTIMATED answer_tokens: {content_word_count} words × 1.3 = {answer_tokens}")

        # Reasoning tokens = total - answer (clamped to 0)
        reasoning_tokens = max(0, output_tokens - answer_tokens)
        logger.debug(f"[Anthropic] COMPUTED reasoning_tokens: {output_tokens} - {answer_tokens} = {reasoning_tokens}")

        logger.debug(f"[Anthropic] FINAL token counts: prompt={prompt_tokens}, "
                    f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})")

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
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=answer_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=output_tokens / duration if duration > 0 else 0,
            finish_reason=finish_reason,
            has_reasoning=has_reasoning,
            timestamp=start_time,
            is_continuation=is_continuation,
            continuation_depth=continuation_depth,
            provider=self.provider_name,
        )

        logger.debug(
            f"[Anthropic] Metrics summary: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
