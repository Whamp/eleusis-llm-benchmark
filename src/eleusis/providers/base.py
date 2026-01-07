"""Base classes and metrics for LLM providers."""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call."""
    model_name: str
    role: str  # "game_master" or player display name
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str
    has_reasoning: bool
    timestamp: float
    # Optional fields
    reasoning_tokens: int | None = None  # If available from API or estimated
    is_continuation: bool = False  # Was this a continuation call?
    continuation_depth: int = 0  # 0=initial, 1=first continuation, etc.
    provider: str = "unknown"
    cost_usd: float | None = None  # Cost in USD (OpenRouter only)


@dataclass
class GenerateMetrics:
    """Metrics for a single generate() call (may include multiple API calls)."""
    total_calls: int  # Number of API calls for this generate()
    continuation_count: int  # How many continuations were needed
    total_prompt_tokens: int
    total_completion_tokens: int
    total_reasoning_tokens: int | None
    total_duration_seconds: float
    success: bool  # Did we get valid response?


def detect_reasoning_model_type(model_name: str) -> str | None:
    """Detect reasoning model type from model name.

    Returns:
        - "gpt-oss": Uses .reasoning field on message
        - "qwen-thinking": Uses <think> tags in content
        - "deepseek-r1": Uses reasoning_content field
        - None: Standard model (no special reasoning)
    """
    model_lower = model_name.lower()

    if "gpt-oss" in model_lower:
        return "gpt-oss"
    elif "qwen" in model_lower and "thinking" in model_lower:
        return "qwen-thinking"
    elif "deepseek" in model_lower and "r1" in model_lower:
        return "deepseek-r1"
    elif "kimi" in model_lower and "thinking" in model_lower:
        return "qwen-thinking"  # Kimi uses similar format

    return None


def estimate_reasoning_tokens(content: str) -> int | None:
    """Estimate reasoning tokens from <think> blocks or raw reasoning text.

    Uses a simple heuristic: ~1.3 tokens per word (conservative estimate).
    Returns None if no reasoning content found.
    """
    if not content:
        return None

    # Try to extract <think> block content
    match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if match:
        thinking_text = match.group(1)
    else:
        # Assume entire content is reasoning (for .reasoning field)
        thinking_text = content

    if not thinking_text.strip():
        return None

    # Heuristic: ~1.3 tokens per word (covers subword tokenization)
    word_count = len(thinking_text.split())
    return int(word_count * 1.3)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        max_tokens: int = 4096,
        role: str = "unknown",
        max_continuation_attempts: int = 3,
        seed: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.role = role
        self.max_continuation_attempts = max_continuation_attempts
        self.seed = seed
        self.call_metrics: list[LLMCallMetrics] = []
        self.generate_metrics: list[GenerateMetrics] = []
        self.reasoning_model_type = detect_reasoning_model_type(model_name)

        if self.reasoning_model_type:
            logger.info(f"Detected reasoning model type: {self.reasoning_model_type}")

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'huggingface', 'openrouter')."""
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a single API call and return response + metrics.

        Args:
            messages: Chat messages to send
            is_continuation: Whether this is a continuation call
            continuation_depth: How many continuations deep we are
            disable_thinking: Try to disable reasoning/thinking if possible

        Returns:
            Tuple of (API response choice, metrics for this call)
        """
        pass

    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: bool = False
    ) -> str | dict:
        """Generate text or structured response with automatic continuation.

        Args:
            prompt: The prompt string to send to LLM
            xml_tag: Optional XML tag name to extract content from
            return_dict: If True, parse extracted/raw content as JSON

        Returns:
            str: Raw or XML-extracted text (if return_dict=False)
            dict: Parsed JSON object (if return_dict=True)
        """
        start_time = time.time()
        calls_in_generate = []

        # Make initial API call
        messages = [{"role": "user", "content": prompt}]
        response, metrics = self._call_api(messages, is_continuation=False, continuation_depth=0)
        calls_in_generate.append(metrics)
        self.call_metrics.append(metrics)

        logger.info(f"Finish reason: {metrics.finish_reason}")

        # Handle truncation by continuing
        if metrics.finish_reason == "length":
            logger.warning("Response was truncated, attempting continuation.")
            content, continuation_metrics = self._continue_response(
                messages, response.message, xml_tag, return_dict, depth=1
            )
            calls_in_generate.extend(continuation_metrics)
        else:
            # No truncation - process response
            content = response.message.content

            # Extract from XML tag if specified
            if xml_tag:
                content = self._extract_content_from_response(content, [xml_tag], try_code_blocks=True)

            # Parse as JSON if requested
            if return_dict:
                content = json.loads(content)

        # Record generate-level metrics
        total_duration = time.time() - start_time
        gen_metrics = GenerateMetrics(
            total_calls=len(calls_in_generate),
            continuation_count=len(calls_in_generate) - 1,
            total_prompt_tokens=sum(m.prompt_tokens for m in calls_in_generate),
            total_completion_tokens=sum(m.completion_tokens for m in calls_in_generate),
            total_reasoning_tokens=sum(m.reasoning_tokens or 0 for m in calls_in_generate) if any(m.reasoning_tokens for m in calls_in_generate) else None,
            total_duration_seconds=total_duration,
            success=True,
        )
        self.generate_metrics.append(gen_metrics)

        return content

    def _continue_response(
        self,
        messages: list[dict],
        partial_response_message,
        xml_tag: str | None,
        return_dict: bool,
        depth: int = 1,
    ) -> tuple[str | dict, list[LLMCallMetrics]]:
        """Continue truncated response with escalating strategies."""
        from eleusis.prompts import get_continuation_prompt

        continuation_metrics = []

        # Build message with partial response
        assistant_msg = {
            "role": "assistant",
            "content": partial_response_message.content,
        }
        # Preserve reasoning if available (for gpt-oss style)
        if hasattr(partial_response_message, 'reasoning') and partial_response_message.reasoning:
            assistant_msg["reasoning"] = partial_response_message.reasoning

        messages.append(assistant_msg)

        # Escalating strategy based on depth
        disable_thinking = depth >= 2  # Try to disable thinking on 2nd+ continuation

        # Request continuation
        tag_name = xml_tag if xml_tag else "RESPONSE"
        continuation_prompt = get_continuation_prompt(tag_name, force_answer=(depth >= 2))

        # For <think> tag models on deeper continuations, try prefilling
        if depth >= 2 and self.reasoning_model_type in ("qwen-thinking", "deepseek-r1"):
            # Add a hint to skip thinking
            continuation_prompt = f"</think>\n{continuation_prompt}"

        messages.append({"role": "user", "content": continuation_prompt})

        # Check if we've exceeded max attempts
        if depth > self.max_continuation_attempts:
            logger.error(f"Max continuation attempts ({self.max_continuation_attempts}) exceeded")
            # Return what we have
            combined_content = partial_response_message.content
            if xml_tag:
                combined_content = self._extract_content_from_response(
                    combined_content, [xml_tag], try_code_blocks=True
                )
            if return_dict:
                return json.loads(combined_content), continuation_metrics
            return combined_content, continuation_metrics

        # Get continuation
        response, metrics = self._call_api(
            messages,
            is_continuation=True,
            continuation_depth=depth,
            disable_thinking=disable_thinking,
        )
        continuation_metrics.append(metrics)
        self.call_metrics.append(metrics)

        # Recursively handle further truncation
        if metrics.finish_reason == "length":
            logger.warning(f"Continuation {depth} was also truncated, continuing again...")
            content, more_metrics = self._continue_response(
                messages, response.message, xml_tag, return_dict, depth=depth + 1
            )
            continuation_metrics.extend(more_metrics)
            return content, continuation_metrics

        # Combine partial + continuation
        combined_content = partial_response_message.content + response.message.content

        # Extract from XML tag if specified
        if xml_tag:
            combined_content = self._extract_content_from_response(
                combined_content, [xml_tag], try_code_blocks=True
            )

        # Parse as JSON if requested
        if return_dict:
            return json.loads(combined_content), continuation_metrics

        return combined_content, continuation_metrics

    def _extract_content_from_response(
        self,
        response_text: str,
        xml_tags: list[str] | None = None,
        try_code_blocks: bool = True,
    ) -> str:
        """Extract content from XML tags or markdown code blocks."""
        extracted = None

        # Try XML tags first
        if xml_tags:
            for tag in xml_tags:
                pattern = f"<{tag}>(.*?)</{tag}>"
                match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    break

        # Fallback to code blocks if requested and XML extraction failed
        if not extracted and try_code_blocks:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                extracted = response_text[start:end].strip()
            elif "```" in response_text:
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                extracted = response_text[start:end].strip()

        return extracted or response_text.strip()

    def convert_rule_to_code(self, rule_text: str) -> str | None:
        """Convert natural language rule to Python code."""
        from eleusis.prompts import get_rule_compilation_prompt

        prompt = get_rule_compilation_prompt(rule_text)
        return self.generate(prompt, xml_tag="CODE")

    def evaluate_rule_on_card(self, rule_text: str, card: dict, mainline: list[dict]) -> bool:
        """Evaluate if card is IN or OUT according to rule."""
        from eleusis.prompts import get_card_evaluation_prompt

        prompt = get_card_evaluation_prompt(rule_text, card, mainline)
        result = self.generate(prompt, return_dict=True)
        return result["result"].lower() == "in"

    def get_usage_stats(self) -> dict:
        """Get aggregated usage statistics for this client."""
        if not self.call_metrics:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": None,
                "duration_seconds": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "call_count": 0,
                "continuation_calls": 0,
                "calls_requiring_continuation": 0,
                "provider": self.provider_name,
            }

        # Aggregate metrics
        total_prompt = sum(m.prompt_tokens for m in self.call_metrics)
        total_completion = sum(m.completion_tokens for m in self.call_metrics)
        total_tokens = sum(m.total_tokens for m in self.call_metrics)
        total_reasoning = sum(m.reasoning_tokens or 0 for m in self.call_metrics)
        total_cost = sum(m.cost_usd for m in self.call_metrics if m.cost_usd is not None)
        total_duration = sum(m.duration_seconds for m in self.call_metrics)
        continuation_calls = sum(1 for m in self.call_metrics if m.is_continuation)
        calls_requiring_continuation = sum(
            1 for gm in self.generate_metrics if gm.continuation_count > 0
        )

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "reasoning_tokens": total_reasoning if total_reasoning > 0 else None,
            "cost_usd": round(total_cost, 6) if total_cost > 0 else None,
            "duration_seconds": round(total_duration, 2),
            "throughput_tokens_per_sec": round(total_completion / total_duration if total_duration > 0 else 0, 2),
            "call_count": len(self.call_metrics),
            "continuation_calls": continuation_calls,
            "calls_requiring_continuation": calls_requiring_continuation,
            "provider": self.provider_name,
        }

    def get_detailed_metrics(self) -> dict:
        """Get detailed per-call metrics for analysis."""
        return {
            "calls": [
                {
                    "model_name": m.model_name,
                    "role": m.role,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": m.completion_tokens,
                    "reasoning_tokens": m.reasoning_tokens,
                    "duration_seconds": round(m.duration_seconds, 3),
                    "finish_reason": m.finish_reason,
                    "is_continuation": m.is_continuation,
                    "continuation_depth": m.continuation_depth,
                    "timestamp": m.timestamp,
                }
                for m in self.call_metrics
            ],
            "generates": [
                {
                    "total_calls": g.total_calls,
                    "continuation_count": g.continuation_count,
                    "total_tokens": g.total_prompt_tokens + g.total_completion_tokens,
                    "reasoning_tokens": g.total_reasoning_tokens,
                    "duration_seconds": round(g.total_duration_seconds, 3),
                }
                for g in self.generate_metrics
            ],
        }

    def reset_usage_stats(self) -> None:
        """Reset call metrics (called at start of each round)."""
        self.call_metrics = []
        self.generate_metrics = []
