"""Base classes and metrics for LLM providers."""

import json
import logging
import re
import textwrap
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


class TruncationError(Exception):
    """Raised when LLM response is truncated due to max tokens."""
    pass


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
        self.fallback_clients: list["BaseLLMClient"] = []
        self._compile_cache: dict[str, dict] = {}

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
            logger.warning(f"{self.model_name} Response truncated (max tokens reached)")
            raise TruncationError(f"Response truncated after {metrics.output_tokens} tokens")

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

    def convert_rule_to_code(
        self,
        rule_text: str,
        max_retries: int = 1,
        fallback_clients: list["BaseLLMClient"] | None = None,
        max_total_attempts: int = 5,
    ) -> dict:
        """Convert natural language rule to Python code with fallback and exponential backoff.

        Tries this client first, then each fallback client in order.
        If all clients fail, sleeps with exponential backoff (1m -> 2m -> 4m -> ...
        capped at 15m) before retrying the full sequence.

        Stops after max_total_attempts total generate() calls across all clients
        and sleep cycles. Returns status="exhausted" if the cap is reached.

        Returns dict with:
        - code: The generated code (None on failure)
        - status: "success", "retry_success", "exhausted", "no_code_returned", or "syntax_error"
        - attempts: Number of attempts made
        - sleep_cycles: Number of sleep cycles before success
        - provider_used: Which provider ultimately succeeded (None on exhaustion)
        """
        from eleusis.prompts import get_rule_compile_prompt

        # Check compile cache — keyed on rule text only (same client = same config)
        cache_key = rule_text
        if cache_key in self._compile_cache:
            logger.debug(f"Compile cache hit for rule: {rule_text[:60]}")
            return self._compile_cache[cache_key]

        prompt = get_rule_compile_prompt(rule_text)
        all_clients = [self] + (fallback_clients if fallback_clients is not None else self.fallback_clients)
        sleep_cycle = 0
        total_attempts = 0
        base_sleep = 60  # 1 minute initial
        max_sleep = 15 * 60  # 15 minute cap

        while True:
            for client_idx, client in enumerate(all_clients):
                client_label = f"{client.provider_name}/{client.model_name}"
                if client_idx > 0:
                    logger.info(f"Trying fallback provider: {client_label}")

                code = None
                for attempt in range(max_retries + 1):
                    if total_attempts >= max_total_attempts:
                        logger.warning(
                            f"Rule compiler exhausted after {total_attempts} total attempts"
                        )
                        return {
                            "code": None,
                            "status": "exhausted",
                            "attempts": total_attempts,
                            "sleep_cycles": sleep_cycle,
                            "provider_used": None,
                        }

                    try:
                        code = client.generate(prompt, xml_tag="CODE")
                        total_attempts += 1
                    except Exception as e:
                        total_attempts += 1
                        logger.warning(f"[{client_label}] Code generation attempt {attempt + 1} failed: {e}")
                        code = None

                    if code and self._validate_code_syntax(code):
                        status = "success" if attempt == 0 and sleep_cycle == 0 and client_idx == 0 else "retry_success"
                        result = {
                            "code": code,
                            "status": status,
                            "attempts": total_attempts,
                            "sleep_cycles": sleep_cycle,
                            "provider_used": client_label,
                        }
                        self._compile_cache[cache_key] = result
                        return result

                    if attempt < max_retries:
                        logger.info(f"[{client_label}] Compilation attempt {attempt + 1} failed, retrying...")

                logger.warning(f"[{client_label}] All {max_retries + 1} attempts failed.")

            # All clients exhausted — check total attempt cap before sleeping
            if total_attempts >= max_total_attempts:
                logger.warning(
                    f"Rule compiler exhausted after {total_attempts} total attempts"
                )
                result = {
                    "code": None,
                    "status": "exhausted",
                    "attempts": total_attempts,
                    "sleep_cycles": sleep_cycle,
                    "provider_used": None,
                }
                self._compile_cache[cache_key] = result
                return result

            # All clients exhausted — exponential backoff
            sleep_cycle += 1
            sleep_secs = min(base_sleep * (2 ** (sleep_cycle - 1)), max_sleep)
            sleep_mins = sleep_secs / 60
            logger.warning(
                f"All providers failed. Sleeping {sleep_mins:.0f}m before retry "
                f"(cycle {sleep_cycle})..."
            )
            print(
                f"[Rule Compiler] All providers failed. "
                f"Sleeping {sleep_mins:.0f}m before retry (cycle {sleep_cycle})..."
            )
            time.sleep(sleep_secs)

    def _validate_code_syntax(self, code: str) -> bool:
        """Check if code compiles without syntax errors.

        The code is a function body (not a full function definition), so we wrap
        it in a function before compiling to allow return statements.
        """
        # Wrap in function definition like Rule._compile_code() does
        full_code = f"def _validate(card, mainline):\n{textwrap.indent(code, '    ')}"
        try:
            compile(full_code, "<string>", "exec")
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            return False

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

    def clear_compile_cache(self) -> None:
        """Clear the compile cache, forcing re-compilation on next call."""
        self._compile_cache.clear()

    def reset_usage_stats(self) -> None:
        """Reset call metrics (called at start of each round)."""
        self.call_metrics = []
        self.generate_metrics = []
