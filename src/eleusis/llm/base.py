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
    """Metrics for a single LLM API call."""
    model_name: str
    role: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float
    throughput_tokens_per_sec: float
    finish_reason: str
    has_reasoning: bool
    timestamp: float
    reasoning_tokens: int | None = None
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
    total_completion_tokens: int
    total_reasoning_tokens: int | None
    total_duration_seconds: float
    success: bool


@dataclass
class ModelCapabilities:
    """Runtime-detected model capabilities."""
    # Reasoning format detection
    has_reasoning: bool = False
    # Possible values:
    #   "field:reasoning_content" - GPT-OSS style separate field
    #   "field:reasoning" - Kimi/Claude style separate field
    #   "tags:think" - <think>...</think> tags in content (DeepSeek, Qwen)
    #   "hidden" - reasoning tokens counted but content not exposed (encrypted)
    #   None - no reasoning detected
    reasoning_format: str | None = None
    thinking_tags_malformed: bool = False  # True if model omits opening <think> tag (Qwen behavior)

    # API support
    supports_disable_thinking: bool = False

    # Metadata from probe
    probe_latency_seconds: float | None = None
    probe_response_preview: str | None = None

    # Provider info
    provider: str = "unknown"
    model_name: str = ""


def detect_reasoning_model_type(model_name: str) -> str | None:
    """Detect reasoning model type from model name."""
    model_lower = model_name.lower()

    if "gpt-oss" in model_lower:
        return "gpt-oss"
    elif "qwen" in model_lower and "thinking" in model_lower:
        return "qwen-thinking"
    elif "deepseek" in model_lower and "r1" in model_lower:
        return "deepseek-r1"
    elif "kimi" in model_lower and "thinking" in model_lower:
        return "qwen-thinking"

    return None


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


# Probe prompt designed to elicit reasoning from thinking models
PROBE_PROMPT = "What is the capital of the state containing Dallas?"


def probe_model_capabilities(client: "BaseLLMClient") -> ModelCapabilities:
    """Make a test call to detect model capabilities.

    Args:
        client: An LLM client instance to probe

    Returns:
        ModelCapabilities with detected reasoning format and features

    Raises:
        Exception: If API call fails (connectivity issues, invalid credentials, etc.)
    """
    caps = ModelCapabilities(
        provider=client.provider_name,
        model_name=client.model_name,
    )

    start_time = time.time()

    # Make API call to inspect response structure
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    response, metrics = client._call_api(messages)

    caps.probe_latency_seconds = time.time() - start_time

    # Extract message from response
    message = response.message
    content = message.content if hasattr(message, 'content') else ""
    caps.probe_response_preview = content[:200] if content else ""

    # Detect reasoning format from response structure
    # Priority: explicit fields first, then inline tags

    # Debug: log available fields on message object
    msg_fields = [attr for attr in dir(message) if not attr.startswith('_')]
    logger.debug(f"Message object fields: {msg_fields}")
    if hasattr(message, 'reasoning'):
        logger.debug(f"reasoning field value: {repr(message.reasoning)[:100]}")
    if hasattr(message, 'reasoning_content'):
        logger.debug(f"reasoning_content field value: {repr(message.reasoning_content)[:100]}")

    # Check for reasoning_content field (GPT-OSS style)
    if hasattr(message, 'reasoning_content') and message.reasoning_content:
        caps.has_reasoning = True
        caps.reasoning_format = "field:reasoning_content"
        # OpenRouter supports disable_thinking for these models
        caps.supports_disable_thinking = client.provider_name == "openrouter"
        logger.info("Detected reasoning format: field:reasoning_content")

    # Check for reasoning field (Claude style)
    elif hasattr(message, 'reasoning') and message.reasoning:
        caps.has_reasoning = True
        caps.reasoning_format = "field:reasoning"
        caps.supports_disable_thinking = client.provider_name == "openrouter"
        logger.info("Detected reasoning format: field:reasoning")

    # Check for think tags in content (DeepSeek, Qwen, Kimi)
    elif content:
        if "<think>" in content and "</think>" in content:
            caps.has_reasoning = True
            caps.reasoning_format = "tags:think"
            caps.supports_disable_thinking = client.provider_name == "openrouter"
            logger.info("Detected reasoning format: tags:think")
        elif "</think>" in content and "<think>" not in content:
            # Qwen malformed tags (missing opening tag)
            caps.has_reasoning = True
            caps.reasoning_format = "tags:think"
            caps.thinking_tags_malformed = True
            caps.supports_disable_thinking = client.provider_name == "openrouter"
            logger.info("Detected reasoning format: tags:think (malformed)")

    # Check for reasoning tokens in metrics (hidden/encrypted reasoning)
    if not caps.has_reasoning and metrics.reasoning_tokens and metrics.reasoning_tokens > 0:
        caps.has_reasoning = True
        caps.reasoning_format = "hidden"  # Reasoning exists but content not exposed
        logger.info(f"Detected reasoning via token count: {metrics.reasoning_tokens} tokens")

    # Log result and compare with string-based detection
    string_based_type = detect_reasoning_model_type(client.model_name)
    if caps.has_reasoning:
        if string_based_type and caps.reasoning_format:
            logger.debug(f"Probe confirmed reasoning (string hint: {string_based_type})")
    else:
        if string_based_type:
            logger.warning(
                f"String-based hints '{string_based_type}' but probe found no reasoning. "
                "May need longer prompt or reasoning not exposed."
            )
        else:
            logger.info("No reasoning format detected (non-reasoning model)")

    return caps


def _get_continuation_prompt(xml_tag: str, force_answer: bool = False) -> str:
    """Get prompt for completing truncated structured response."""
    if force_answer:
        return f"""STOP. Output ONLY the final answer now.
DO NOT think further. DO NOT reason. DO NOT use <think> tags.
Immediately output the <{xml_tag}> tag with valid JSON inside, then close with </{xml_tag}>.
Start your response with: <{xml_tag}>"""
    else:
        return f"""Please continue and COMPLETE your response now.
DO NOT REASON ABOUT IT FURTHER, just provide the missing content.
You MUST start your response immediately with the <{xml_tag}> tag.
You MUST finish with a properly closed </{xml_tag}> tag containing valid JSON.
- Include the complete JSON object in the XML tags
- Ensure all JSON braces and brackets are properly closed
"""


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
        self.capabilities: ModelCapabilities | None = None  # Set by probe_model_capabilities()

        if self.reasoning_model_type:
            logger.info(f"Detected reasoning model type: {self.reasoning_model_type}")

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
        """Generate text or structured response with automatic continuation."""
        start_time = time.time()
        calls_in_generate = []

        messages = [{"role": "user", "content": prompt}]
        response, metrics = self._call_api(messages, is_continuation=False, continuation_depth=0)
        calls_in_generate.append(metrics)
        self.call_metrics.append(metrics)

        logger.info(f"Finish reason: {metrics.finish_reason}")

        if metrics.finish_reason == "length":
            logger.warning("Response was truncated, attempting continuation.")
            content, continuation_metrics = self._continue_response(
                messages, response.message, xml_tag, return_dict, depth=1
            )
            calls_in_generate.extend(continuation_metrics)
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
        continuation_metrics = []

        assistant_msg = {
            "role": "assistant",
            "content": partial_response_message.content,
        }
        if hasattr(partial_response_message, 'reasoning') and partial_response_message.reasoning:
            assistant_msg["reasoning"] = partial_response_message.reasoning

        messages.append(assistant_msg)

        disable_thinking = depth >= 2

        tag_name = xml_tag if xml_tag else "RESPONSE"
        continuation_prompt = _get_continuation_prompt(tag_name, force_answer=(depth >= 2))

        # Inject </think> tag for models that use think tags, to close any open reasoning block
        uses_think_tags = (
            (self.capabilities and self.capabilities.reasoning_format == "tags:think")
            or self.reasoning_model_type in ("qwen-thinking", "deepseek-r1")  # Fallback if no probe
        )
        if depth >= 2 and uses_think_tags:
            continuation_prompt = f"</think>\n{continuation_prompt}"

        messages.append({"role": "user", "content": continuation_prompt})

        if depth > self.max_continuation_attempts:
            logger.error(f"Max continuation attempts ({self.max_continuation_attempts}) exceeded")
            combined_content = partial_response_message.content
            if xml_tag:
                combined_content = self._extract_content_from_response(
                    combined_content, [xml_tag], try_code_blocks=True
                )
            if return_dict:
                return json.loads(combined_content), continuation_metrics
            return combined_content, continuation_metrics

        response, metrics = self._call_api(
            messages,
            is_continuation=True,
            continuation_depth=depth,
            disable_thinking=disable_thinking,
        )
        continuation_metrics.append(metrics)
        self.call_metrics.append(metrics)

        if metrics.finish_reason == "length":
            logger.warning(f"Continuation {depth} was also truncated, continuing again...")
            content, more_metrics = self._continue_response(
                messages, response.message, xml_tag, return_dict, depth=depth + 1
            )
            continuation_metrics.extend(more_metrics)
            return content, continuation_metrics

        combined_content = partial_response_message.content + response.message.content

        if xml_tag:
            combined_content = self._extract_content_from_response(
                combined_content, [xml_tag], try_code_blocks=True
            )

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
