"""xAI Grok API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

import httpx
from openai import OpenAI

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class XAIMessage:
    """Message wrapper for xAI responses."""
    content: str
    reasoning: str | None = None


@dataclass
class XAIChoice:
    """Choice wrapper for xAI responses."""
    message: XAIMessage
    finish_reason: str


class XAIClient(BaseLLMClient):
    """Client for xAI Grok API (OpenAI-compatible)."""

    XAI_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
    ) -> None:
        """Initialize xAI client."""
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("XAI_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        # xAI API can take longer for reasoning, use extended timeout
        self.client = OpenAI(
            base_url=self.XAI_BASE_URL,
            api_key=self.api_key,
            timeout=httpx.Timeout(3600.0),
        )

    @property
    def provider_name(self) -> str:
        return "xai"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[XAIChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling xAI API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Grok-4 always reasons at max capacity, no effort parameter
                # For force-answer, we switch to the non-reasoning model variant
                model = self.model_name
                if disable_thinking and "reasoning" in model:
                    model = model.replace("reasoning", "non-reasoning")
                    logger.info(f"Force-answer: switching to {model}")

                api_kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                if self.seed is not None:
                    api_kwargs["seed"] = self.seed

                completion = self.client.chat.completions.create(**api_kwargs)

                end_time = time.time()
                choice = completion.choices[0]

                # Extract content
                text_content = choice.message.content or ""
                reasoning_content = None

                # Check for reasoning in additional_kwargs (LangChain style)
                # or reasoning_content field
                if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
                    reasoning_content = choice.message.reasoning_content

                wrapped_choice = XAIChoice(
                    message=XAIMessage(
                        content=text_content,
                        reasoning=reasoning_content,
                    ),
                    finish_reason=choice.finish_reason or "stop",
                )

                metrics = self._extract_metrics(
                    completion, wrapped_choice, start_time, end_time,
                    is_continuation, continuation_depth
                )

                logger.debug(f"LLM response:\n{wrapped_choice}")
                return wrapped_choice, metrics

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
        choice: XAIChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response with normalized token fields.

        xAI follows OpenAI pattern - completion_tokens includes reasoning, so:
        - output_tokens = API completion_tokens
        - reasoning_tokens = API reasoning_tokens (if available)
        - answer_tokens = output_tokens - reasoning_tokens
        """
        duration = end_time - start_time

        # --- RAW API VALUES ---
        api_prompt_tokens = 0
        api_completion_tokens = 0
        api_reasoning_tokens = None

        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage
            api_prompt_tokens = usage.prompt_tokens or 0
            api_completion_tokens = usage.completion_tokens or 0
            logger.debug(f"[xAI] RAW API usage: prompt_tokens={api_prompt_tokens}, completion_tokens={api_completion_tokens}")

            # Check for reasoning tokens in usage details
            if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
                details = usage.completion_tokens_details
                logger.debug(f"[xAI] completion_tokens_details: {details}")
                if hasattr(details, "reasoning_tokens"):
                    api_reasoning_tokens = details.reasoning_tokens
                    logger.debug(f"[xAI] RAW API reasoning_tokens={api_reasoning_tokens}")
        else:
            logger.debug("[xAI] No usage data in completion")

        # --- REASONING CONTENT ---
        reasoning_text = choice.message.reasoning
        reasoning_word_count = len(reasoning_text.split()) if reasoning_text else 0
        logger.debug(f"[xAI] Reasoning content present: {reasoning_text is not None}, word_count={reasoning_word_count}")
        if reasoning_text:
            logger.debug(f"[xAI] Reasoning preview: {reasoning_text[:200]}...")

        # --- COMPUTED VALUES ---
        prompt_tokens = api_prompt_tokens
        output_tokens = api_completion_tokens
        reasoning_tokens = api_reasoning_tokens or 0
        has_reasoning = reasoning_tokens > 0

        # Estimate reasoning tokens from content if not provided but reasoning exists
        if not has_reasoning and reasoning_text:
            has_reasoning = True
            reasoning_tokens = int(reasoning_word_count * 1.3)
            logger.debug(f"[xAI] ESTIMATED reasoning_tokens: {reasoning_word_count} words × 1.3 = {reasoning_tokens}")
        elif api_reasoning_tokens:
            logger.debug(f"[xAI] Using NATIVE reasoning_tokens from API: {api_reasoning_tokens}")

        answer_tokens = max(0, output_tokens - reasoning_tokens)

        logger.debug(f"[xAI] FINAL token counts: prompt={prompt_tokens}, "
                    f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})")

        # Map finish reasons
        finish_reason = choice.finish_reason
        if finish_reason == "max_tokens":
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
            f"[xAI] Metrics summary: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
