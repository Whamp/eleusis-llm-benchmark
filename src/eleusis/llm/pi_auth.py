"""Subscription authentication backed by pi's auth.json.

pi (the coding agent harness this benchmark's operator uses) keeps an
OAuth entry named ``openai-codex`` in ``~/.pi/agent/auth.json``. That entry
holds a ChatGPT-subscription bearer token plus the account ID required by
the Codex backend at https://chatgpt.com/backend-api/codex.

Reading the file fresh on every request means a token refreshed by pi is
picked up mid-run without rebuilding LLM clients.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_PI_AUTH_PATH = Path("~/.pi/agent/auth.json").expanduser()
CODEX_BACKEND_BASE_URL = "https://chatgpt.com/backend-api/codex"


class PiCodexAuthError(RuntimeError):
    """Raised when pi's auth.json cannot supply usable Codex credentials."""


@dataclass(frozen=True)
class CodexCredentials:
    """One snapshot of the subscription credentials."""

    access_token: str
    account_id: str


class PiCodexAuth(httpx.Auth):
    """Injects ChatGPT-subscription credentials from pi's auth.json.

    The file is re-read on every request so tokens refreshed by pi are
    used immediately. Requires the ``openai`` SDK client to be pointed at
    ``CODEX_BACKEND_BASE_URL``.
    """

    def __init__(
        self,
        auth_path: Path = DEFAULT_PI_AUTH_PATH,
        now: Callable[[], float] = time.time,
    ) -> None:
        """Initialize with an auth.json path and injectable clock."""
        self.auth_path = auth_path
        self._now = now

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """Attach subscription headers to one outgoing request."""
        credentials = self._load_credentials()
        request.headers["Authorization"] = f"Bearer {credentials.access_token}"
        request.headers["chatgpt-account-id"] = credentials.account_id
        yield request

    def _load_credentials(self) -> CodexCredentials:
        """Read and validate the openai-codex entry from auth.json."""
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PiCodexAuthError(
                f"pi auth file not found at {self.auth_path}; log in with pi"
                " before using pi-codex authentication"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PiCodexAuthError(
                f"pi auth file at {self.auth_path} is not valid JSON: {exc}"
            ) from exc

        entry = data.get("openai-codex") if isinstance(data, dict) else None
        if not isinstance(entry, dict):
            raise PiCodexAuthError(
                f"no usable 'openai-codex' entry in {self.auth_path};"
                " log in with pi before using pi-codex authentication"
            )
        access_token = entry.get("access")
        account_id = entry.get("accountId")
        if not isinstance(access_token, str) or not access_token:
            raise PiCodexAuthError(
                f"'openai-codex' entry in {self.auth_path} has no access token"
            )
        if not isinstance(account_id, str) or not account_id:
            raise PiCodexAuthError(
                f"'openai-codex' entry in {self.auth_path} has no accountId"
            )

        expires_ms = entry.get("expires")
        expired = (
            isinstance(expires_ms, (int, float))
            and expires_ms > 0
            and expires_ms / 1000.0 <= self._now()
        )
        if expired:
            raise PiCodexAuthError(
                "openai-codex token in "
                f"{self.auth_path} has expired; refresh it by using pi"
                " (any authenticated command) and retry"
            )

        return CodexCredentials(access_token=access_token, account_id=account_id)


__all__ = [
    "CODEX_BACKEND_BASE_URL",
    "CodexCredentials",
    "PiCodexAuth",
    "PiCodexAuthError",
]
