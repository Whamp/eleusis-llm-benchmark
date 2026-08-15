"""Base classes and metrics for LLM providers."""

import json
import logging
import re
import textwrap
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol, TypedDict, cast, overload

logger = logging.getLogger(__name__)


class LLMMessage(TypedDict):
    """Provider-neutral chat message passed into one LLM adapter call."""

    role: Literal["assistant", "system", "user"]
    content: str


class ResponseMessage(Protocol):
    """Text-bearing message returned by a provider response."""

    @property
    def content(self) -> str:
        """Visible response text."""
        ...


class LLMResponseEnvelope(Protocol):
    """Provider response containing one text-bearing message."""

    @property
    def message(self) -> ResponseMessage:
        """Provider response message."""
        ...


class RuleCompileResult(TypedDict):
    """Outcome of compiling a natural-language rule to executable code."""

    code: str | None
    status: str
    attempts: int
    sleep_cycles: int
    provider_used: str | None


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
    output_tokens: int  # Total output (reasoning + answer)
    reasoning_tokens: int  # CoT/thinking tokens (0 if none)
    answer_tokens: int  # Non-reasoning output tokens
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


def _restore_call_metrics(payload: Mapping[str, object]) -> LLMCallMetrics:
    """Build one call metric from validated provider-neutral data."""
    return LLMCallMetrics(
        model_name=cast(str, payload["model_name"]),
        role=cast(str, payload["role"]),
        prompt_tokens=cast(int, payload["prompt_tokens"]),
        output_tokens=cast(int, payload["output_tokens"]),
        reasoning_tokens=cast(int, payload["reasoning_tokens"]),
        answer_tokens=cast(int, payload["answer_tokens"]),
        duration_seconds=cast(float, payload["duration_seconds"]),
        throughput_tokens_per_sec=cast(
            float,
            payload["throughput_tokens_per_sec"],
        ),
        finish_reason=cast(str, payload["finish_reason"]),
        has_reasoning=cast(bool, payload["has_reasoning"]),
        timestamp=cast(float, payload["timestamp"]),
        is_continuation=cast(bool, payload["is_continuation"]),
        continuation_depth=cast(int, payload["continuation_depth"]),
        provider=cast(str, payload["provider"]),
        cost_usd=cast(float | None, payload["cost_usd"]),
    )


def _restore_generate_metrics(payload: Mapping[str, object]) -> GenerateMetrics:
    """Build one generate metric from validated provider-neutral data."""
    return GenerateMetrics(
        total_calls=cast(int, payload["total_calls"]),
        continuation_count=cast(int, payload["continuation_count"]),
        total_prompt_tokens=cast(int, payload["total_prompt_tokens"]),
        total_output_tokens=cast(int, payload["total_output_tokens"]),
        total_reasoning_tokens=cast(int, payload["total_reasoning_tokens"]),
        total_answer_tokens=cast(int, payload["total_answer_tokens"]),
        total_duration_seconds=cast(float, payload["total_duration_seconds"]),
        success=cast(bool, payload["success"]),
    )


def estimate_reasoning_tokens(content: str) -> int | None:
    """Estimate reasoning tokens from <think> blocks or raw reasoning text."""
    if not content:
        return None

    # Try standard <think>...</think> format first
    match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
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


class TruncationError(Exception):
    """Raised when LLM response is truncated due to max tokens."""


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
        """Initialize shared provider settings and metric stores."""
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.role = role
        self.seed = seed
        self.call_metrics: list[LLMCallMetrics] = []
        self.generate_metrics: list[GenerateMetrics] = []
        self.fallback_clients: list[BaseLLMClient] = []
        self._compile_cache: dict[tuple[str, int], RuleCompileResult] = {}

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name."""

    @abstractmethod
    def _call_api(
        self,
        messages: list[LLMMessage],
        is_continuation: bool = False,
        continuation_depth: int = 0,
        disable_thinking: bool = False,
    ) -> tuple[LLMResponseEnvelope, LLMCallMetrics]:
        """Make a single API call and return response + metrics."""

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[False] = False,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        xml_tag: str | None = None,
        return_dict: Literal[True] = True,
    ) -> dict[str, object]: ...

    def generate(
        self, prompt: str, xml_tag: str | None = None, return_dict: bool = False
    ) -> str | dict[str, object]:
        """Generate text or a parsed JSON object from one provider response."""
        start_time = time.time()
        calls_in_generate = []

        messages: list[LLMMessage] = [{"role": "user", "content": prompt}]
        response, metrics = self._call_api(messages)
        calls_in_generate.append(metrics)
        self.call_metrics.append(metrics)

        logger.info(f"Finish reason: {metrics.finish_reason}")

        if metrics.finish_reason == "length":
            logger.warning(f"{self.model_name} Response truncated (max tokens reached)")
            raise TruncationError(
                f"Response truncated after {metrics.output_tokens} tokens"
            )

        content = response.message.content

        if xml_tag:
            content = self._extract_content_from_response(
                content, [xml_tag], try_code_blocks=True
            )

        structured_content: dict[str, object] | None = None
        if return_dict:
            logger.debug(f"Parsing JSON from extracted content:\n{content[:500]}")
            parsed_content = json.loads(content)
            if not isinstance(parsed_content, dict):
                raise TypeError("LLM structured response must decode to a JSON object")
            structured_content = parsed_content

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

        return (
            structured_content
            if return_dict and structured_content is not None
            else content
        )

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
            logger.debug(
                f"Response text length: {len(response_text)}, contains '<ACTION>':"
                f" {'<ACTION>' in response_text}"
            )
            for tag in xml_tags:
                pattern = f"<{tag}>(.*?)</{tag}>"
                # Use findall and take the last match - avoids false matches when
                # the LLM mentions the tag in its reasoning before the actual tag
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
                logger.debug(f"Searching for <{tag}>: found {len(matches)} match(es)")
                if matches:
                    extracted = matches[-1].strip()
                    logger.debug(
                        f"Extracted {len(extracted)} chars from <{tag}> (using last"
                        " match)"
                    )
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

    def convert_rule_to_code(
        self,
        rule_text: str,
        max_retries: int = 1,
        fallback_clients: list["BaseLLMClient"] | None = None,
        max_total_attempts: int = 5,
    ) -> RuleCompileResult:
        """Convert a natural-language rule using retries and fallback providers."""
        from eleusis.llm.rule_compilation import RuleCompilationCoordinator

        fallbacks = (
            fallback_clients if fallback_clients is not None else self.fallback_clients
        )
        coordinator = RuleCompilationCoordinator(
            primary_client=self,
            fallback_clients=fallbacks,
            compile_cache=self._compile_cache,
            validate_code_syntax=self._validate_code_syntax,
            max_retries=max_retries,
            max_total_attempts=max_total_attempts,
        )
        return coordinator.compile(rule_text)

    def _validate_code_syntax(self, code: str) -> bool:
        """Check if code compiles without syntax errors.

        The code is a function body (not a full function definition), so we wrap it in a
        function before compiling to allow return statements.
        """
        # Wrap in function definition like Rule._compile_code() does
        full_code = f"def _validate(card, mainline):\n{textwrap.indent(code, '    ')}"
        try:
            compile(full_code, "<string>", "exec")
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            return False

    def get_usage_stats(self) -> dict[str, object]:
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
        total_cost = sum(
            m.cost_usd for m in self.call_metrics if m.cost_usd is not None
        )
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
            "throughput_tokens_per_sec": round(
                total_output / total_duration if total_duration > 0 else 0, 2
            ),
            "call_count": len(self.call_metrics),
            "continuation_calls": continuation_calls,
            "calls_requiring_continuation": calls_requiring_continuation,
            "provider": self.provider_name,
        }

    def get_detailed_metrics(self) -> dict[str, object]:
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

    def snapshot_client_continuation(self) -> dict[str, object]:
        """Capture provider-neutral usage and reusable rule compilations."""
        return {
            "call_metrics": [asdict(metric) for metric in self.call_metrics],
            "generate_metrics": [asdict(metric) for metric in self.generate_metrics],
            "compile_cache": [
                {
                    "key": {
                        "rule_text": rule_text,
                        "max_total_attempts": max_total_attempts,
                    },
                    "value": dict(value),
                }
                for (rule_text, max_total_attempts), value in sorted(
                    self._compile_cache.items()
                )
            ],
            "fallback_clients": [
                client.snapshot_client_continuation()
                for client in self.fallback_clients
            ],
        }

    def restore_client_continuation(
        self,
        payload: Mapping[str, object],
    ) -> None:
        """Restore provider-neutral accounting and cache state on a fresh client."""
        call_payloads = cast(
            Sequence[Mapping[str, object]],
            payload["call_metrics"],
        )
        generate_payloads = cast(
            Sequence[Mapping[str, object]],
            payload["generate_metrics"],
        )
        self.call_metrics = [_restore_call_metrics(metric) for metric in call_payloads]
        self.generate_metrics = [
            _restore_generate_metrics(metric) for metric in generate_payloads
        ]
        restored_cache: dict[tuple[str, int], RuleCompileResult] = {}
        cache_payloads = cast(
            Sequence[Mapping[str, object]],
            payload["compile_cache"],
        )
        for entry in cache_payloads:
            key = cast(Mapping[str, object], entry["key"])
            restored_cache[
                cast(str, key["rule_text"]),
                cast(int, key["max_total_attempts"]),
            ] = cast(
                RuleCompileResult,
                dict(cast(Mapping[str, object], entry["value"])),
            )
        self._compile_cache = restored_cache

        fallback_payloads = cast(
            Sequence[Mapping[str, object]],
            payload["fallback_clients"],
        )
        if len(fallback_payloads) != len(self.fallback_clients):
            raise ValueError(
                "Client continuation fallback count mismatch: "
                f"snapshot has {len(fallback_payloads)}, "
                f"fresh client has {len(self.fallback_clients)}"
            )
        for client, fallback_payload in zip(
            self.fallback_clients,
            fallback_payloads,
            strict=True,
        ):
            client.restore_client_continuation(fallback_payload)

    def clear_compile_cache(self) -> None:
        """Clear the compile cache, forcing re-compilation on next call."""
        self._compile_cache.clear()

    def reset_usage_stats(self) -> None:
        """Reset call metrics (called at start of each round)."""
        self.call_metrics = []
        self.generate_metrics = []
