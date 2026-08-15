"""Normalized token accounting for Hugging Face inference responses."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from eleusis.llm.base import LLMCallMetrics, estimate_reasoning_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuggingFaceMetricInput:
    """Provider response values needed to calculate normalized call metrics."""

    model_name: str
    role: str
    reasoning_format: str
    content: str
    reasoning: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    start_time: float
    end_time: float
    is_continuation: bool
    continuation_depth: int
    estimated: bool = False


@dataclass(frozen=True)
class HuggingFaceTokenBreakdown:
    """Normalized output, reasoning, and answer token counts."""

    output_tokens: int
    reasoning_tokens: int
    answer_tokens: int
    has_reasoning: bool


def _think_tag_word_count(content: str) -> int:
    """Count words inside the leading think-tag region when present."""
    if "</think>" not in content:
        return 0
    think_content = content.split("</think>", 1)[0]
    if "<think>" in think_content:
        think_content = think_content.split("<think>", 1)[1]
    return len(think_content.split())


def _calculate_token_breakdown(
    content: str,
    reasoning: str | None,
    reasoning_format: str,
    completion_tokens: int,
) -> HuggingFaceTokenBreakdown:
    """Normalize provider token counts for the configured reasoning representation."""
    if reasoning_format == "separate_field":
        answer_tokens = int(len(content.split()) * 1.3)
        return HuggingFaceTokenBreakdown(
            output_tokens=completion_tokens,
            reasoning_tokens=max(0, completion_tokens - answer_tokens),
            answer_tokens=answer_tokens,
            has_reasoning=bool(reasoning),
        )
    if reasoning_format == "think_tags":
        has_reasoning = "<think>" in content or "</think>" in content
        reasoning_tokens = (
            estimate_reasoning_tokens(content) or 0 if has_reasoning else 0
        )
        logger.debug(
            "[HF metrics] Think content: ~%s words",
            _think_tag_word_count(content),
        )
        return HuggingFaceTokenBreakdown(
            output_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            answer_tokens=max(0, completion_tokens - reasoning_tokens),
            has_reasoning=has_reasoning,
        )
    return HuggingFaceTokenBreakdown(
        output_tokens=completion_tokens,
        reasoning_tokens=0,
        answer_tokens=completion_tokens,
        has_reasoning=False,
    )


def calculate_huggingface_metrics(
    metric_input: HuggingFaceMetricInput,
) -> LLMCallMetrics:
    """Calculate normalized metrics from one Hugging Face response."""
    duration = metric_input.end_time - metric_input.start_time
    breakdown = _calculate_token_breakdown(
        metric_input.content,
        metric_input.reasoning,
        metric_input.reasoning_format,
        metric_input.completion_tokens,
    )
    metrics = LLMCallMetrics(
        model_name=metric_input.model_name,
        role=metric_input.role,
        prompt_tokens=metric_input.prompt_tokens,
        output_tokens=breakdown.output_tokens,
        reasoning_tokens=breakdown.reasoning_tokens,
        answer_tokens=breakdown.answer_tokens,
        duration_seconds=duration,
        throughput_tokens_per_sec=(
            breakdown.output_tokens / duration if duration > 0 else 0
        ),
        finish_reason=metric_input.finish_reason,
        has_reasoning=breakdown.has_reasoning,
        timestamp=metric_input.start_time,
        is_continuation=metric_input.is_continuation,
        continuation_depth=metric_input.continuation_depth,
        provider="huggingface",
    )
    estimated_label = " (ESTIMATED)" if metric_input.estimated else ""
    logger.debug(
        "[HF metrics] Summary%s: %s output tokens in %.2fs (%.2f tok/s)",
        estimated_label,
        breakdown.output_tokens,
        duration,
        metrics.throughput_tokens_per_sec,
    )
    return metrics
