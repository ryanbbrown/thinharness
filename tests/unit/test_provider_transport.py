from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from fakes import echo_tool

from thinharness import (
    Harness,
    HarnessConfig,
    OpenAIProvider,
    OpenAIResponsesModel,
)
from thinharness.providers import ProviderError
from thinharness.providers.transport import _retry_after_seconds, _retry_delay


async def test_provider_wraps_transport_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", request_retries=0, http_client=client)
        with pytest.raises(ProviderError, match="provider request failed") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code is None
    assert calls == 1


async def test_provider_wraps_http_status_errors() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="rate limited", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", request_retries=0, http_client=client)
        with pytest.raises(ProviderError, match="provider error 429: rate limited") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code == 429
    assert calls == 1


async def test_provider_wraps_invalid_json() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", base_url="http://example.invalid", request_retries=3, http_client=client)
        with pytest.raises(ProviderError, match="invalid JSON") as exc_info:
            await provider.post_json("/responses", {})
    assert exc_info.value.status_code is None
    assert calls == 1


@pytest.mark.parametrize("status_code", [408, 409, 425, 429, 500, 599])
async def test_provider_retries_transient_statuses(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, text="temporary", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, end: end)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(
            api_key="key",
            base_url="https://example.test",
            request_retries=1,
            request_retry_backoff=2,
            http_client=client,
        )
        assert await provider.post_json("/responses", {"secret": "value"}) == {"ok": True}

    assert calls == 2
    assert sleeps == [2.5]


@pytest.mark.parametrize("status_code", [400, 401, 404])
async def test_provider_does_not_retry_permanent_statuses(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, text="permanent", request=request)

    async def fail_sleep(_delay: float) -> None:
        raise AssertionError("permanent failure must not sleep")

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fail_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=3, http_client=client)
        with pytest.raises(ProviderError) as exc_info:
            await provider.post_json("/responses", {})

    assert exc_info.value.status_code == status_code
    assert calls == 1


async def test_provider_retries_transport_error_and_raises_final_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="first", request=request)
        raise httpx.ConnectError("offline", request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, _end: 0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0.25, http_client=client)
        with pytest.raises(ProviderError, match="provider request failed: offline") as exc_info:
            await provider.post_json("/responses", {})

    assert exc_info.value.status_code is None
    assert calls == 2
    assert sleeps == [0.25]


def test_provider_retry_settings_validate_direct_construction() -> None:
    for kwargs in (
        {"request_retries": -1},
        {"request_retries": 11},
        {"request_retry_backoff": -0.1},
        {"request_retry_backoff": float("inf")},
        {"request_retry_backoff": float("nan")},
    ):
        with pytest.raises(ValueError):
            OpenAIProvider(api_key="key", **kwargs)
    assert OpenAIProvider(api_key="key", request_retries=0, request_retry_backoff=0).request_retries == 0
    assert OpenAIProvider(api_key="key", request_retries=10).request_retries == 10


def test_retry_after_parsing_and_delay_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert _retry_after_seconds("3.5", now=now) == 3.5
    assert _retry_after_seconds("Thu, 01 Jan 2026 00:00:05 GMT", now=now) == 5
    assert _retry_after_seconds("Thu, 01 Jan 2026 00:00:05", now=now) == 5
    for value in ("garbage", "-1", "nan", "inf", "Wed, 31 Feb 2026 00:00:00 GMT"):
        assert _retry_after_seconds(value, now=now) is None

    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, _end: 0.0)
    assert _retry_delay(2, 1, "1") == 4
    assert _retry_delay(0, 0, "5") == 5
    assert _retry_delay(100, 1) == 60


async def test_provider_retry_does_not_duplicate_tool_continuation_state(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        if len(payloads) == 1:
            return httpx.Response(200, json={
                "id": "resp_1",
                "output": [{"type": "function_call", "call_id": "call_1", "name": "echo", "arguments": '{"value":"ok"}'}],
            }, request=request)
        if len(payloads) == 2:
            return httpx.Response(503, text="temporary", request=request)
        return httpx.Response(200, json={"id": "resp_2", "output_text": "done"}, request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            "test-model",
            provider=OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0, http_client=client),
        )
        result = await Harness(HarnessConfig(root=tmp_path), model=model, tools=[echo_tool()]).run("go")

    assert result.text == "done"
    assert payloads[1] == payloads[2]
    assert "previous_response_id" not in payloads[1]
    assert payloads[1]["store"] is False
    assert [item["type"] for item in payloads[1]["input"]] == ["message", "function_call", "function_call_output"]
    assert result.resume_state is not None
    assert [entry["role"] for entry in result.resume_state["entries"]] == ["user", "assistant", "tool", "assistant"]


async def test_provider_cancellation_during_backoff_stops_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, request=request)

    async def cancel_sleep(_delay: float) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", cancel_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=3, request_retry_backoff=0, http_client=client)
        with pytest.raises(asyncio.CancelledError):
            await provider.post_json("/responses", {})
    assert calls == 1


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda request: httpx.UnsupportedProtocol("unsupported", request=request),
        lambda request: httpx.LocalProtocolError("invalid local request", request=request),
        lambda request: httpx.ProxyError("invalid proxy", request=request),
        lambda request: httpx.TooManyRedirects("redirect loop", request=request),
    ],
)
async def test_provider_does_not_retry_other_http_errors(monkeypatch: pytest.MonkeyPatch, error_factory) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_factory(request)

    async def fail_sleep(_delay: float) -> None:
        raise AssertionError("non-retryable failure must not sleep")

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fail_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=3, http_client=client)
        with pytest.raises(ProviderError, match="provider request failed"):
            await provider.post_json("/responses", {})
    assert calls == 1


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda request: httpx.ConnectError("offline", request=request),
        lambda request: httpx.ReadTimeout("timed out", request=request),
        lambda request: httpx.RemoteProtocolError("server disconnected", request=request),
    ],
)
async def test_provider_network_error_then_success_retries(monkeypatch: pytest.MonkeyPatch, error_factory) -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error_factory(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, _end: 0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0.5, http_client=client)
        assert await provider.post_json("/responses", {}) == {"ok": True}
    assert calls == 2
    assert sleeps == [0.5]


async def test_provider_retryable_status_exhaustion_uses_final_response(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text=f"failure-{calls}", request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=2, request_retry_backoff=0, http_client=client)
        with pytest.raises(ProviderError, match="provider error 503: failure-3") as exc_info:
            await provider.post_json("/responses", {})
    assert calls == 3
    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("header_kind", ["numeric", "future", "past"])
async def test_provider_response_retry_after_controls_sleep(
    monkeypatch: pytest.MonkeyPatch,
    header_kind: str,
) -> None:
    calls = 0
    sleeps: list[float] = []
    if header_kind == "numeric":
        retry_after = "5"
    elif header_kind == "future":
        retry_after = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    else:
        retry_after = format_datetime(datetime.now(UTC) - timedelta(seconds=30), usegmt=True)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": retry_after}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, _end: 0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=2, http_client=client)
        assert await provider.post_json("/responses", {}) == {"ok": True}

    if header_kind == "numeric":
        assert sleeps == [5.0]
    elif header_kind == "future":
        assert 20 <= sleeps[0] <= 30
    else:
        assert sleeps == [2.0]


async def test_provider_retries_identical_requests_and_safe_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshots: list[tuple[str, bytes, tuple[tuple[bytes, bytes], ...]]] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        snapshots.append((str(request.url), request.content, tuple(request.headers.raw)))
        if len(snapshots) == 1:
            return httpx.Response(503, text="sensitive-response", request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("thinharness.providers.transport.random.uniform", lambda _start, _end: 0.0)
    caplog.set_level("INFO", logger="thinharness.providers")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0, http_client=client)
        assert await provider.post_json("/responses", {"secret": "sensitive-payload"}) == {"ok": True}

    assert snapshots[0] == snapshots[1]
    assert sleeps == [0.0]
    assert "HTTP 503" in caplog.text
    assert "retry 1/1" in caplog.text
    assert "sensitive-payload" not in caplog.text
    assert "sensitive-response" not in caplog.text
    assert {record.name for record in caplog.records} == {"thinharness.providers"}


async def test_provider_recovered_retries_reuse_shared_client(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls % 2:
            return httpx.Response(429, content=b"rate-limit-body", request=request)
        return httpx.Response(200, json={"call": calls}, request=request)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("thinharness.providers.transport.asyncio.sleep", no_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=1, request_retry_backoff=0, http_client=client)
        results = [await provider.post_json("/responses", {"index": index}) for index in range(3)]
    assert results == [{"call": 2}, {"call": 4}, {"call": 6}]
    assert calls == 6


async def test_provider_cancellation_inside_request_stops_attempts() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise asyncio.CancelledError()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAIProvider(api_key="key", request_retries=3, request_retry_backoff=0, http_client=client)
        with pytest.raises(asyncio.CancelledError):
            await provider.post_json("/responses", {})
    assert calls == 1
