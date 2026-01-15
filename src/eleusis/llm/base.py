"""Base classes and metrics for LLM providers."""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM API call.

    Token fields are normalized across providers:
    - output_tokens: Total output (reasoning + answer)
    - reasoning_tokens: Chain-of-thought/thinking tokens
    - answer_tokens: Non-reasoning response tokens

    Invariant: output_tokens = reasoning_tokens + answer_tokens
    """
    model_name: str
    role: str
    prompt_tokens: int
    output_tokens: int      # Total output (reasoning + answer)
    reasoning_tokens: int   # CoT/thinking tokens (0 if none)
    answer_tokens: int      # Non-reasoning output tokens
    duration_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str
    has_reasoning: bool
    timestamp: float
    is_continuation: bool = False
    continuation_depth: int = 0
    provider: str = "unknown"
    cost_usd: float | None = None


@dataclass
class GenerateMetrics:
    """Metrics for a single generate() call (may include multiple API calls)."""
    total_calls: int
    continuation_count: int
    total_prompt_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int
    total_answer_tokens: int
    total_duration_seconds: float
    success: bool


def estimate_reasoning_tokens(content: str) -> int | None:
    """Estimate reasoning tokens from <think> blocks or raw reasoning text."""
    if not content:
        return None

    # Try standard <think>...</think> format first
    match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if match:
        thinking_text = match.group(1)
    elif "</think>" in content:
        # Qwen3 Thinking format: may only have </think> without opening tag
        # Everything before </think> is reasoning content
        thinking_text = content.split("</think>", 1)[0]
    else:
        thinking_text = content

    if not thinking_text.strip():
        return None

    word_count = len(thinking_text.split())
    return int(word_count * 1.3)


# Force-answer prompt for truncated responses
FORCE_ANSWER_PROMPT = """STOP reasoning. Output your answer NOW.
You have already done your thinking. Do not think further.
Immediately output the answer in the requested format."""


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
        seed: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.role = role
        self.seed = seed
        self.call_metrics: list[LLMCallMetrics] = []
        self.generate_metrics: list[GenerateMetrics] = []

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    @abstractmethod
    def _call_api(
        self,
        messages: list[dict],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[object, LLMCallMetrics]:
        """Make a single API call and return response + metrics."""
        pass

    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: bool = False
    ) -> str | dict:
        """Generate text or structured response with single force-answer attempt on truncation."""
        start_time = time.time()
        calls_in_generate = []

        messages = [{"role": "user", "content": prompt}]
        response, metrics = self._call_api(messages)
        calls_in_generate.append(metrics)
        self.call_metrics.append(metrics)

        logger.info(f"Finish reason: {metrics.finish_reason}")

        if metrics.finish_reason == "length":
            logger.warning(f"{self.model_name} Response truncated, attempting force-answer")
            content, force_metrics = self._force_answer(
                messages, response.message, metrics, xml_tag, return_dict
            )
            calls_in_generate.append(force_metrics)
        else:
            content = response.message.content

            if xml_tag:
                content = self._extract_content_from_response(content, [xml_tag], try_code_blocks=True)

            if return_dict:
                logger.debug(f"Parsing JSON from extracted content:\n{content[:500]}")
                content = json.loads(content)

        total_duration = time.time() - start_time
        gen_metrics = GenerateMetrics(
            total_calls=len(calls_in_generate),
            continuation_count=len(calls_in_generate) - 1,
            total_prompt_tokens=sum(m.prompt_tokens for m in calls_in_generate),
            total_output_tokens=sum(m.output_tokens for m in calls_in_generate),
            total_reasoning_tokens=sum(m.reasoning_tokens for m in calls_in_generate),
            total_answer_tokens=sum(m.answer_tokens for m in calls_in_generate),
            total_duration_seconds=total_duration,
            success=True,
        )
        self.generate_metrics.append(gen_metrics)

        return content

    def _force_answer(
        self,
        messages: list[dict],
        partial_response_message,
        initial_metrics: LLMCallMetrics,
        xml_tag: str | None,
        return_dict: bool,
    ) -> tuple[str | dict, LLMCallMetrics]:
        """Single force-answer attempt. Raises RuntimeError if still truncated."""
        # Build assistant message with truncated content
        assistant_content = partial_response_message.content

        # Detect if reasoning is separate (field-based or token-counted)
        has_separate_reasoning = (
            (hasattr(partial_response_message, 'reasoning') and partial_response_message.reasoning)
            or (hasattr(partial_response_message, 'reasoning_content') and partial_response_message.reasoning_content)
            or (initial_metrics.reasoning_tokens and initial_metrics.reasoning_tokens > 0)
        )

        # If reasoning is NOT separate → assume inline (Qwen-style), close think block
        if not has_separate_reasoning and "</think>" not in assistant_content:
            assistant_content += "</think>"

        assistant_msg = {"role": "assistant", "content": assistant_content}
        if hasattr(partial_response_message, 'reasoning') and partial_response_message.reasoning:
            assistant_msg["reasoning"] = partial_response_message.reasoning

        messages.append(assistant_msg)
        messages.append({"role": "user", "content": FORCE_ANSWER_PROMPT})

        response, metrics = self._call_api(messages, disable_thinking=True)
        self.call_metrics.append(metrics)

        if metrics.finish_reason == "length":
            raise RuntimeError("Force-answer attempt still truncated")

        # Combine original + force-answer content, then extract
        combined_content = partial_response_message.content + response.message.content

        if xml_tag:
            combined_content = self._extract_content_from_response(
                combined_content, [xml_tag], try_code_blocks=True
            )

        if return_dict:
            combined_content = json.loads(combined_content)

        return combined_content, metrics

    def _extract_content_from_response(
        self,
        response_text: str,
        xml_tags: list[str] | None = None,
        try_code_blocks: bool = True,
    ) -> str:
        """Extract content from XML tags or markdown code blocks."""
        # Strip thinking content if present (Qwen3 Thinking format)
        # This prevents tags mentioned in reasoning from being matched
        if "</think>" in response_text:
            response_text = response_text.split("</think>", 1)[-1]

        extracted = None

        if xml_tags:
            logger.debug(f"Response text length: {len(response_text)}, contains '<ACTION>': {'<ACTION>' in response_text}")
            for tag in xml_tags:
                pattern = f"<{tag}>(.*?)</{tag}>"
                # Use findall and take the last match - avoids false matches when
                # the LLM mentions the tag in its reasoning before the actual tag
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                logger.debug(f"Searching for <{tag}>: found {len(matches)} match(es)")
                if matches:
                    extracted = matches[-1].strip()
                    logger.debug(f"Extracted {len(extracted)} chars from <{tag}> (using last match)")
                    break

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
        from eleusis.prompts import get_rule_compile_prompt

        prompt = get_rule_compile_prompt(rule_text)
        return self.generate(prompt, xml_tag="CODE")

    def get_usage_stats(self) -> dict:
        """Get aggregated usage statistics for this client.

        Returns normalized token fields:
        - output_tokens: Total output (reasoning + answer)
        - reasoning_tokens: CoT/thinking tokens
        - answer_tokens: Non-reasoning output
        """
        if not self.call_metrics:
            return {
                "prompt_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "answer_tokens": 0,
                "cost_usd": None,
                "duration_seconds": 0.0,
                "throughput_tokens_per_sec": 0.0,
                "call_count": 0,
                "continuation_calls": 0,
                "calls_requiring_continuation": 0,
                "provider": self.provider_name,
            }

        total_prompt = sum(m.prompt_tokens for m in self.call_metrics)
        total_output = sum(m.output_tokens for m in self.call_metrics)
        total_reasoning = sum(m.reasoning_tokens for m in self.call_metrics)
        total_answer = sum(m.answer_tokens for m in self.call_metrics)
        total_cost = sum(m.cost_usd for m in self.call_metrics if m.cost_usd is not None)
        total_duration = sum(m.duration_seconds for m in self.call_metrics)
        continuation_calls = sum(1 for m in self.call_metrics if m.is_continuation)
        calls_requiring_continuation = sum(
            1 for gm in self.generate_metrics if gm.continuation_count > 0
        )

        return {
            "prompt_tokens": total_prompt,
            "output_tokens": total_output,
            "reasoning_tokens": total_reasoning,
            "answer_tokens": total_answer,
            "cost_usd": round(total_cost, 6) if total_cost > 0 else None,
            "duration_seconds": round(total_duration, 2),
            "throughput_tokens_per_sec": round(total_output / total_duration if total_duration > 0 else 0, 2),
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
                    "output_tokens": m.output_tokens,
                    "reasoning_tokens": m.reasoning_tokens,
                    "answer_tokens": m.answer_tokens,
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
                    "output_tokens": g.total_output_tokens,
                    "reasoning_tokens": g.total_reasoning_tokens,
                    "answer_tokens": g.total_answer_tokens,
                    "duration_seconds": round(g.total_duration_seconds, 3),
                }
                for g in self.generate_metrics
            ],
        }

    def reset_usage_stats(self) -> None:
        """Reset call metrics (called at start of each round)."""
        self.call_metrics = []
        self.generate_metrics = []
