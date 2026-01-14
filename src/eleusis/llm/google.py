"""Google Gemini API client implementation."""

import logging
import os
import time
from dataclasses import dataclass

from google import genai
from google.genai import types

from eleusis.llm.base import BaseLLMClient, LLMCallMetrics

logger = logging.getLogger(__name__)


@dataclass
class GeminiMessage:
    """Message wrapper for Gemini responses."""
    content: str
    reasoning: str | None = None


@dataclass
class GeminiChoice:
    """Choice wrapper for Gemini responses."""
    message: GeminiMessage
    finish_reason: str


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini API with thinking level support."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        seed: int | None = None,
        thinking_level: str = "high",
    ) -> None:
        """Initialize Google Gemini client."""
        super().__init__(
            model_name=model_name,
            api_key=api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_retries=max_retries,
            max_tokens=max_tokens,
            role=role,
            seed=seed,
        )

        self.thinking_level = thinking_level
        # Set API key in environment for google-genai
        if self.api_key:
            os.environ["GOOGLE_API_KEY"] = self.api_key
        self.client = genai.Client()

    @property
    def provider_name(self) -> str:
        return "google"

    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[GeminiChoice, LLMCallMetrics]:
        """Make a single API call with retry logic."""
        logger.debug(
            f"Calling Gemini API with {self.max_tokens} tokens, messages:\n{messages}"
        )

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build content from messages
                # Gemini uses a different format - convert from OpenAI-style messages
                contents = []
                system_instruction = None
                for msg in messages:
                    if msg["role"] == "system":
                        system_instruction = msg["content"]
                    elif msg["role"] == "user":
                        contents.append(types.Content(
                            role="user",
                            parts=[types.Part(text=msg["content"])]
                        ))
                    elif msg["role"] == "assistant":
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part(text=msg["content"])]
                        ))

                # Thinking level: low for force-answer, otherwise configured level
                # Note: Gemini 3 Pro cannot fully disable thinking
                level = "low" if disable_thinking else self.thinking_level

                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level=level),
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                if system_instruction:
                    config.system_instruction = system_instruction

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                end_time = time.time()

                # Extract content and reasoning
                text_content = response.text if hasattr(response, "text") else ""
                reasoning_content = None

                # Check for thinking content in candidates
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "content") and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, "thought") and part.thought:
                                reasoning_content = part.text

                # Determine finish reason
                finish_reason = "stop"
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "finish_reason"):
                        fr = candidate.finish_reason
                        if fr == "MAX_TOKENS":
                            finish_reason = "length"
                        elif fr == "STOP":
                            finish_reason = "stop"

                choice = GeminiChoice(
                    message=GeminiMessage(
                        content=text_content,
                        reasoning=reasoning_content,
                    ),
                    finish_reason=finish_reason,
                )

                metrics = self._extract_metrics(
                    response, choice, start_time, end_time,
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
        response,
        choice: GeminiChoice,
        start_time: float,
        end_time: float,
        is_continuation: bool,
        continuation_depth: int,
    ) -> LLMCallMetrics:
        """Extract metrics from API response."""
        duration = end_time - start_time

        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
            completion_tokens = getattr(metadata, "candidates_token_count", 0) or 0
            if hasattr(metadata, "thoughts_token_count"):
                reasoning_tokens = metadata.thoughts_token_count

        total_tokens = prompt_tokens + completion_tokens
        has_reasoning = reasoning_tokens is not None and reasoning_tokens > 0

        # Estimate reasoning tokens from content if not provided
        if not has_reasoning and choice.message.reasoning:
            has_reasoning = True
            word_count = len(choice.message.reasoning.split())
            reasoning_tokens = int(word_count * 1.3)

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
