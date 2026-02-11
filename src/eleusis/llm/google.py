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
        self.client = genai.Client(
            http_options=types.HttpOptions(timeout=600_000),  # 10min in ms
        )

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

                config_kwargs = {
                    "thinking_config": types.ThinkingConfig(thinking_level=level),
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.seed is not None:
                    config_kwargs["seed"] = self.seed

                config = types.GenerateContentConfig(**config_kwargs)

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
                logger.warning(f"{self.model_name} Attempt {attempt + 1}/{self.max_retries} failed: {e}")
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
        """Extract metrics from API response with normalized token fields.

        Google's candidates_token_count excludes thinking tokens, so:
        - answer_tokens = API candidates_token_count
        - reasoning_tokens = API thoughts_token_count
        - output_tokens = answer_tokens + reasoning_tokens
        """
        duration = end_time - start_time

        # --- RAW API VALUES ---
        api_prompt_tokens = 0
        api_candidates_tokens = 0
        api_thoughts_tokens = None

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata = response.usage_metadata
            api_prompt_tokens = getattr(metadata, "prompt_token_count", 0) or 0
            api_candidates_tokens = getattr(metadata, "candidates_token_count", 0) or 0
            api_thoughts_tokens = getattr(metadata, "thoughts_token_count", None)

            logger.debug(f"[Google] RAW API usage_metadata: prompt_token_count={api_prompt_tokens}, "
                        f"candidates_token_count={api_candidates_tokens}, thoughts_token_count={api_thoughts_tokens}")

            # Log any other metadata fields
            for attr in ['total_token_count', 'cached_content_token_count']:
                if hasattr(metadata, attr):
                    logger.debug(f"[Google] Additional metadata: {attr}={getattr(metadata, attr)}")
        else:
            logger.debug("[Google] No usage_metadata in response")

        # --- REASONING CONTENT ---
        reasoning_text = choice.message.reasoning
        reasoning_word_count = len(reasoning_text.split()) if reasoning_text else 0
        logger.debug(f"[Google] Reasoning content present: {reasoning_text is not None}, word_count={reasoning_word_count}")
        if reasoning_text:
            logger.debug(f"[Google] Reasoning preview: {reasoning_text[:200]}...")

        # --- COMPUTED VALUES ---
        prompt_tokens = api_prompt_tokens
        answer_tokens = api_candidates_tokens
        reasoning_tokens = api_thoughts_tokens or 0
        has_reasoning = reasoning_tokens > 0

        # Estimate reasoning tokens from content if not provided by API
        if not has_reasoning and reasoning_text:
            has_reasoning = True
            reasoning_tokens = int(reasoning_word_count * 1.3)
            logger.debug(f"[Google] ESTIMATED reasoning_tokens: {reasoning_word_count} words × 1.3 = {reasoning_tokens}")
        elif api_thoughts_tokens:
            logger.debug(f"[Google] Using NATIVE thoughts_token_count from API: {api_thoughts_tokens}")

        output_tokens = answer_tokens + reasoning_tokens

        logger.debug(f"[Google] FINAL token counts: prompt={prompt_tokens}, "
                    f"output={output_tokens} (answer={answer_tokens} + reasoning={reasoning_tokens})")

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
            f"[Google] Metrics summary: {output_tokens} output tokens in {duration:.2f}s "
            f"({metrics.throughput_tokens_per_sec:.2f} tok/s)"
        )

        return metrics
