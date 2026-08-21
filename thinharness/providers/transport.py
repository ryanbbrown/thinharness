"""Shared provider HTTP transport and retry policy."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import random
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from ..tools.base import Json

logger = logging.getLogger("thinharness.providers")
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429})
_MAX_RETRY_DELAY = 60.0

class ProviderError(RuntimeError):
    """Raised when a provider request fails."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _validate_retry_settings(request_retries: int, request_retry_backoff: float) -> None:
    """Validate public provider retry settings."""
    if not isinstance(request_retries, int) or isinstance(request_retries, bool) or not 0 <= request_retries <= 10:
        raise ValueError("request_retries must be between 0 and 10")
    if (
        not isinstance(request_retry_backoff, (int, float))
        or isinstance(request_retry_backoff, bool)
        or not math.isfinite(request_retry_backoff)
        or request_retry_backoff < 0
    ):
        raise ValueError("request_retry_backoff must be a finite number >= 0")


def _is_retryable_status(status_code: int) -> bool:
    """Return whether an HTTP response status is transient."""
    return status_code in _RETRYABLE_HTTP_STATUSES or 500 <= status_code <= 599


def _retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse a standard Retry-After delay or HTTP date."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            seconds = (retry_at - (now or datetime.now(UTC))).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _retry_delay(base: float, retry_index: int, retry_after: str | None = None) -> float:
    """Return a capped exponential delay with positive jitter."""
    exponential = base * (2 ** retry_index)
    jittered = exponential + random.uniform(0, exponential * 0.25)
    provider_delay = _retry_after_seconds(retry_after) or 0.0
    return min(max(jittered, provider_delay), _MAX_RETRY_DELAY)


class Provider:
    """Provider transport, auth, and endpoint configuration."""

    name = "provider"
    api_key_env = ""
    default_base_url = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        request_retries: int = 3,
        request_retry_backoff: float = 1.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        _validate_retry_settings(request_retries, request_retry_backoff)
        self.api_key = api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self.request_retries = request_retries
        self.request_retry_backoff = float(request_retry_backoff)
        self._http_client = http_client
        self._owns_client = http_client is None

    def headers(self) -> Json:
        """Return auth headers for this provider."""
        if not self.api_key:
            raise ProviderError(f"{self.api_key_env} is required for {self.name}")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _client(self) -> httpx.AsyncClient:
        """Return the shared async HTTP client for this provider."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def aclose(self) -> None:
        """Close this provider's owned HTTP client."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def post_json(self, path: str, payload: Json) -> Json:
        """POST JSON to this provider with bounded transient-failure retries."""
        url = f"{self.base_url}{path}"
        headers = self.headers()
        client = self._client()
        for attempt in range(self.request_retries + 1):
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if attempt == self.request_retries or not _is_retryable_status(status_code):
                    raise ProviderError(
                        f"provider error {status_code}: {exc.response.text}",
                        status_code=status_code,
                    ) from exc
                await exc.response.aread()
                delay = _retry_delay(self.request_retry_backoff, attempt, exc.response.headers.get("Retry-After"))
                logger.info(
                    "Retrying %s provider request after HTTP %s (retry %s/%s in %.3fs)",
                    self.name,
                    status_code,
                    attempt + 1,
                    self.request_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                if attempt == self.request_retries:
                    raise ProviderError(f"provider request failed: {exc}") from exc
                delay = _retry_delay(self.request_retry_backoff, attempt)
                logger.info(
                    "Retrying %s provider request after %s (retry %s/%s in %.3fs)",
                    self.name,
                    type(exc).__name__,
                    attempt + 1,
                    self.request_retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            except httpx.HTTPError as exc:
                raise ProviderError(f"provider request failed: {exc}") from exc
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderError(f"provider returned invalid JSON: {exc}") from exc
        raise AssertionError("unreachable provider retry loop exit")
