from __future__ import annotations

import pytest

from thinharness import (
    HarnessConfig,
    ModelNotice,
    parse_model_ref,
)
from thinharness.providers import (
    ModelSettings,
    TokenUsage,
    extract_finish_reason,
    extract_token_usage,
    render_model_notices,
)


def test_model_refs_require_provider_prefix() -> None:
    assert parse_model_ref("openai:gpt-4.1-mini") == ("openai", "gpt-4.1-mini")
    assert parse_model_ref("anthropic:claude-3-5-haiku-latest") == ("anthropic", "claude-3-5-haiku-latest")
    with pytest.raises(ValueError):
        parse_model_ref("gpt-4.1-mini")


def test_model_notice_rendering_is_deterministic() -> None:
    first = ModelNotice(kind="limit_warning", content="Final request.", limit_kind="model_requests", remaining=1)
    second = ModelNotice(kind="limit_warning", content="One tool call remains.", limit_kind="tool_calls", remaining=1)

    assert render_model_notices(None) == ""
    assert render_model_notices([first]) == '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'
    assert render_model_notices([first, second]) == (
        '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'
        "\n\n"
        '<harness_notice kind="limit_warning">\nOne tool call remains.\n</harness_notice>'
    )


def test_model_settings_and_harness_config_validate_max_tokens() -> None:
    with pytest.raises(ValueError):
        ModelSettings(max_tokens=0)
    with pytest.raises(ValueError):
        ModelSettings(max_tokens=-1)
    with pytest.raises(ValueError):
        HarnessConfig(max_tokens=0)
    with pytest.raises(ValueError):
        HarnessConfig(max_tokens=-1)


def test_extract_token_usage_tolerates_partial_and_missing_usage() -> None:
    assert extract_token_usage({}) is None
    assert extract_token_usage({"usage": {"input_tokens": 9}}) == TokenUsage(input_tokens=9, output_tokens=None)
    assert extract_token_usage({"usage": {"completion_tokens": 3}}) == TokenUsage(input_tokens=None, output_tokens=3)


def test_extract_token_usage_carries_cached_input_breakdowns() -> None:
    assert extract_token_usage({
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": {"cached_tokens": 12},
            "output_tokens": 5,
        },
    }) == TokenUsage(input_tokens=20, output_tokens=5, cached_tokens=12)
    assert extract_token_usage({
        "usage": {
            "prompt_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 11},
            "completion_tokens": 5,
        },
    }) == TokenUsage(input_tokens=20, output_tokens=5, cached_tokens=11)
    assert extract_token_usage({
        "usage": {
            "input_tokens": 20,
            "cache_read_input_tokens": 10,
            "output_tokens": 5,
        },
    }) == TokenUsage(input_tokens=20, output_tokens=5, cached_tokens=10)


def test_extract_finish_reason_precedence() -> None:
    assert extract_finish_reason({"finish_reason": "length"}) == "length"
    assert extract_finish_reason({"stop_reason": "end_turn", "finish_reason": "length"}) == "end_turn"
    assert extract_finish_reason({"finish_reason": "length", "choices": [{"finish_reason": "stop"}]}) == "length"
    assert extract_finish_reason({"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}) == "stop"
    assert extract_finish_reason({}) is None


def test_harness_retry_settings_validate_and_reach_all_inferred_providers() -> None:
    with pytest.raises(ValueError):
        HarnessConfig(request_retries=-1)
    with pytest.raises(ValueError):
        HarnessConfig(request_retries=11)
    with pytest.raises(ValueError):
        HarnessConfig(request_retry_backoff=-0.1)
    with pytest.raises(ValueError):
        HarnessConfig(request_retry_backoff=float("inf"))
    with pytest.raises(ValueError):
        HarnessConfig(request_retry_backoff=float("nan"))
    for provider_name in ("openai", "anthropic", "openrouter"):
        from thinharness.providers import infer_model

        model = infer_model(f"{provider_name}:test", request_retries=2, request_retry_backoff=0.5)
        assert model.provider.request_retries == 2
        assert model.provider.request_retry_backoff == 0.5
