from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.longmemeval_v2.orchestrate import (  # noqa: E402
    normalized_query_usage,
    summarize_pairs,
    token_cost,
)


def test_normalized_query_usage_preserves_cache_reasoning_requests_and_tools() -> None:
    attempts = [
        {
            "usage": {
                "requests": 2,
                "input_tokens": 100,
                "cached_input_tokens": 60,
                "output_tokens": 20,
                "reasoning_output_tokens": 15,
                "cache_write_input_tokens": 0,
            },
            "tool_call_count": 3,
        },
        {
            "usage": {
                "model_requests": 1,
                "input_tokens": 50,
                "cached_tokens": 10,
                "output_tokens": 5,
                "tool_calls": 2,
            },
            "tool_call_count": None,
        },
    ]

    assert normalized_query_usage(attempts) == {
        "requests": 3,
        "input_tokens": 150,
        "cached_input_tokens": 70,
        "ordinary_input_tokens": 80,
        "cache_write_input_tokens": 0,
        "output_tokens": 25,
        "reasoning_output_tokens": 15,
        "tool_calls": 5,
    }


def test_token_cost_uses_cached_rate_only_for_reported_cached_tokens() -> None:
    cost = token_cost(
        input_tokens=1_000_000,
        cached_tokens=750_000,
        output_tokens=100_000,
        input_rate=0.20,
        cached_rate=0.02,
        output_rate=1.20,
    )

    assert cost == 0.185


def test_pair_summary_separates_deterministic_and_llm_judged_results() -> None:
    receipts = [
        {
            "question_id": "det",
            "harness": "native",
            "domain": "web",
            "question_type": "static-environment",
            "scoring_path": "deterministic",
            "score": 0.0,
        },
        {
            "question_id": "det",
            "harness": "thinharness",
            "domain": "web",
            "question_type": "static-environment",
            "scoring_path": "deterministic",
            "score": 1.0,
        },
        {
            "question_id": "judge",
            "harness": "native",
            "domain": "enterprise",
            "question_type": "procedure-abs",
            "scoring_path": "llm_judged",
            "score": 1.0,
        },
        {
            "question_id": "judge",
            "harness": "thinharness",
            "domain": "enterprise",
            "question_type": "procedure-abs",
            "scoring_path": "llm_judged",
            "score": 0.0,
        },
    ]

    summary = summarize_pairs(receipts)["summaries"]

    assert summary["all"] == {
        "pair_count": 2,
        "native_correct": 1.0,
        "thinharness_correct": 1.0,
        "mean_paired_difference": 0.0,
        "thin_win_native_loss": 1,
        "native_win_thin_loss": 1,
    }
    assert summary["deterministic"]["mean_paired_difference"] == 1.0
    assert summary["llm_judged"]["mean_paired_difference"] == -1.0
