#!/usr/bin/env python3
"""Run one frozen LongMemEval-V2 cell through one query harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

READER_PROVIDER = {
    "order": ["Parasail"],
    "allow_fallbacks": False,
    "require_parameters": True,
    "data_collection": "deny",
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ensure_direct_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not injected")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not injected")
    configured_base = os.environ.pop("OPENAI_BASE_URL", None)
    if configured_base and configured_base.rstrip("/") != "https://api.openai.com/v1":
        print("Ignoring OPENAI_BASE_URL so GPT calls use the direct OpenAI API.", flush=True)


def common_harness_argv(
    *,
    domain: str,
    runtime_dir: Path,
    data_root: Path,
    memory_config_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        "evaluation.harness",
        "--domain",
        domain,
        "--questions-path",
        str(runtime_dir / "questions.json"),
        "--haystack-path",
        str(runtime_dir / "haystack.json"),
        "--trajectories-path",
        str(data_root / "trajectories.jsonl"),
        "--memory-config-path",
        str(memory_config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        "qwen/qwen3.5-9b",
        "--base-url",
        "https://openrouter.ai/api/v1",
        "--api-key-env",
        "OPENROUTER_API_KEY",
        "--reader-provider-json",
        json.dumps(READER_PROVIDER, separators=(",", ":")),
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--max-completion-tokens",
        "20000",
        "--memory-context-max-tokens",
        "200000",
        "--reader-max-concurrent-requests",
        "1",
        "--prompt-build-max-workers",
        "1",
        "--evaluator-model",
        "gpt-5.2",
        "--evaluator-api-key-env",
        "OPENAI_API_KEY",
        "--evaluator-reasoning-effort",
        "medium",
        "--evaluator-max-completion-tokens",
        "4096",
    ]


def run_native(args: argparse.Namespace) -> None:
    from evaluation.run_eval import main as run_eval_main

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluation.run_eval",
            "--data-root",
            str(args.data_root),
            "--domain",
            args.domain,
            "--tier",
            "small",
            "--method",
            "agentrunbook_c_v2",
            "--output-dir",
            str(args.output_dir),
            "--question-ids",
            args.question_id,
            "--reader-model",
            "qwen/qwen3.5-9b",
            "--reader-base-url",
            "https://openrouter.ai/api/v1",
            "--reader-api-key-env",
            "OPENROUTER_API_KEY",
            "--reader-provider-json",
            json.dumps(READER_PROVIDER, separators=(",", ":")),
            "--reader-max-concurrent-requests",
            "1",
            "--openai-sdk-model",
            "gpt-5.6-luna",
            "--openai-sdk-reasoning-effort",
            "xhigh",
            "--openai-sdk-max-retries",
            "3",
            "--openai-sdk-api-key-env",
            "OPENAI_API_KEY",
            "--openai-sdk-max-turns",
            "30",
            "--prompt-build-max-workers",
            "1",
            "--no-enable-online-learning",
            "--evaluator-model",
            "gpt-5.2",
            "--evaluator-api-key-env",
            "OPENAI_API_KEY",
            "--evaluator-reasoning-effort",
            "medium",
        ]
        run_eval_main()
    finally:
        sys.argv = old_argv


def run_thinharness(args: argparse.Namespace) -> None:
    from data.public_data import materialize_runtime_haystack, materialize_runtime_questions
    from evaluation.harness import main as harness_main

    from benchmarks.longmemeval_v2 import memory as _memory_registration  # noqa: F401

    runtime_dir = args.output_dir / "runtime_inputs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    questions = materialize_runtime_questions(
        data_root=args.data_root,
        domain=args.domain,
        question_ids=[args.question_id],
        limit=None,
        output_path=runtime_dir / "questions.json",
    )
    materialize_runtime_haystack(
        data_root=args.data_root,
        tier="small",
        selected_questions=questions,
        output_path=runtime_dir / "haystack.json",
    )
    memory_config_path = runtime_dir / "memory_config.json"
    write_json(
        memory_config_path,
        {
            "memory_type": "thinharness",
            "memory_params": {
                "model": "openai:gpt-5.6-luna",
                "base_url": None,
                "api_key_env": "OPENAI_API_KEY",
                "timeout_seconds": 1200.0,
                "max_retries": 3,
                "max_model_requests": 30,
                "max_tool_calls": 128,
                "output_retries": 1,
                "builtin_tools": ["read", "search", "jsonl_search", "list", "glob"],
                "output_mode": "native",
                "reasoning_effort": "xhigh",
                "extra_body": {},
                "query_trace_dir": str((args.output_dir / "query_traces").resolve()),
            },
        },
    )
    old_argv = sys.argv
    try:
        sys.argv = common_harness_argv(
            domain=args.domain,
            runtime_dir=runtime_dir,
            data_root=args.data_root,
            memory_config_path=memory_config_path,
            output_dir=args.output_dir,
        )
        harness_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--harness", choices=("native", "thinharness"), required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--domain", choices=("web", "enterprise"), required=True)
    args = parser.parse_args()
    args.official_root = args.official_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    ensure_direct_openai()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "cell_metadata.json",
        {
            "harness": args.harness,
            "question_id": args.question_id,
            "domain": args.domain,
            "query_model": "gpt-5.6-luna",
            "query_reasoning_effort": "xhigh",
            "query_api": "https://api.openai.com/v1",
            "online_learning": False,
            "reader_model": "qwen/qwen3.5-9b",
            "reader_api": "https://openrouter.ai/api/v1",
            "reader_provider": READER_PROVIDER,
            "evaluator_model": "gpt-5.2",
            "evaluator_api": "https://api.openai.com/v1",
            "api_key_env_names": ["OPENAI_API_KEY", "OPENROUTER_API_KEY"],
        },
    )
    if args.harness == "native":
        run_native(args)
    else:
        run_thinharness(args)


if __name__ == "__main__":
    main()
