"""Rule compiler retry, fallback, cache, and backoff coordination."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from eleusis.prompts import get_rule_compile_prompt

if TYPE_CHECKING:
    from eleusis.llm.base import BaseLLMClient, RuleCompileResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleCompilationAttempt:
    """A completed compilation result and whether it belongs in the cache."""

    result: RuleCompileResult
    cacheable: bool


class RuleCompilationCoordinator:
    """Coordinate rule compilation across retries and fallback providers."""

    def __init__(
        self,
        primary_client: BaseLLMClient,
        fallback_clients: list[BaseLLMClient],
        compile_cache: dict[tuple[str, int], RuleCompileResult],
        validate_code_syntax: Callable[[str], bool],
        max_retries: int,
        max_total_attempts: int,
    ) -> None:
        """Capture one compilation request's clients, limits, and shared cache."""
        self.primary_client = primary_client
        self.clients = [primary_client, *fallback_clients]
        self.compile_cache = compile_cache
        self.validate_code_syntax = validate_code_syntax
        self.max_retries = max_retries
        self.max_total_attempts = max_total_attempts
        self.total_attempts = 0
        self.sleep_cycle = 0

    def compile(self, rule_text: str) -> RuleCompileResult:
        """Compile one rule through configured clients, retries, and backoff."""
        cache_key = (rule_text, self.max_total_attempts)
        if cached := self.compile_cache.get(cache_key):
            logger.debug(f"Compile cache hit for rule: {rule_text[:60]}")
            return cached

        prompt = get_rule_compile_prompt(rule_text)
        while True:
            for client_index, client in enumerate(self.clients):
                attempt = self._try_client(client, client_index, prompt)
                if attempt:
                    if attempt.cacheable:
                        self.compile_cache[cache_key] = attempt.result
                    return attempt.result
            if self.total_attempts >= self.max_total_attempts:
                result = self._exhausted_result()
                self.compile_cache[cache_key] = result
                return result
            self._sleep_before_next_cycle()

    def _try_client(
        self,
        client: BaseLLMClient,
        client_index: int,
        prompt: str,
    ) -> RuleCompilationAttempt | None:
        """Try one compiler client up to its per-cycle retry limit."""
        client_label = f"{client.provider_name}/{client.model_name}"
        if client_index > 0:
            logger.info(f"Trying fallback provider: {client_label}")
        for attempt_index in range(self.max_retries + 1):
            if self.total_attempts >= self.max_total_attempts:
                logger.warning(
                    "Rule compiler exhausted after "
                    f"{self.total_attempts} total attempts"
                )
                return RuleCompilationAttempt(self._exhausted_result(), cacheable=False)
            code = self._generate_code(client, client_label, prompt, attempt_index)
            if code and self.validate_code_syntax(code):
                status = (
                    "success"
                    if attempt_index == 0
                    and self.sleep_cycle == 0
                    and client_index == 0
                    else "retry_success"
                )
                return RuleCompilationAttempt(
                    {
                        "code": code,
                        "status": status,
                        "attempts": self.total_attempts,
                        "sleep_cycles": self.sleep_cycle,
                        "provider_used": client_label,
                    },
                    cacheable=True,
                )
            if attempt_index < self.max_retries:
                logger.info(
                    f"[{client_label}] Compilation attempt {attempt_index + 1} failed,"
                    " retrying..."
                )
        logger.warning(f"[{client_label}] All {self.max_retries + 1} attempts failed.")
        return None

    def _generate_code(
        self,
        client: BaseLLMClient,
        client_label: str,
        prompt: str,
        attempt_index: int,
    ) -> str | None:
        """Generate one candidate while converting provider failures to retries."""
        try:
            code = client.generate(prompt, xml_tag="CODE")
        # Provider SDKs expose unrelated exception hierarchies at this boundary.
        except Exception as error:  # ruff: ignore[blind-except]
            logger.warning(
                f"[{client_label}] Code generation attempt {attempt_index + 1} failed:"
                f" {error}"
            )
            code = None
        self.total_attempts += 1
        return code

    def _exhausted_result(self) -> RuleCompileResult:
        """Build the canonical exhausted compilation result."""
        return {
            "code": None,
            "status": "exhausted",
            "attempts": self.total_attempts,
            "sleep_cycles": self.sleep_cycle,
            "provider_used": None,
        }

    def _sleep_before_next_cycle(self) -> None:
        """Sleep with capped exponential backoff before another provider cycle."""
        self.sleep_cycle += 1
        sleep_seconds = min(60 * (2 ** (self.sleep_cycle - 1)), 15 * 60)
        logger.warning(
            f"All providers failed. Sleeping {sleep_seconds / 60:.0f}m before retry "
            f"(cycle {self.sleep_cycle})..."
        )
        time.sleep(sleep_seconds)
