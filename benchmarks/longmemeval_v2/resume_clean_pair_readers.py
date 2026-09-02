#!/usr/bin/env python3
"""Resume only the four failed readers from frozen clean-pair prompt artifacts."""

from __future__ import annotations

import argparse
import asyncio
import email.utils
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.longmemeval_v2.orchestrate import build_receipt, sha256_file, summarize_pairs, write_json
from benchmarks.longmemeval_v2.run_cell import READER_PROVIDER
from benchmarks.longmemeval_v2.run_clean_pair import (
    TARGET_CELLS,
    compile_results,
    validate_completed,
    write_results,
)

RECOVERY_CELLS = (
    "03-eaba5c44-native",
    "05-b54161f8-thinharness",
    "06-11cc7ac2-native",
    "10-af2ebaed-thinharness",
)
FALLBACK_DELAYS_SECONDS = (5.0, 10.0, 20.0, 40.0, 80.0, 120.0)
SAFE_HEADER_NAMES = {
    "cf-ray",
    "retry-after",
    "retry-after-ms",
    "server",
    "x-request-id",
}
SAFE_HEADER_PREFIXES = ("x-openrouter-", "x-ratelimit-")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        handle.flush()


def write_jsonl_record(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_retry_after(headers: Any, *, now: float | None = None) -> float | None:
    if headers is None:
        return None
    retry_ms = headers.get("retry-after-ms")
    try:
        value = float(retry_ms) / 1000.0
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        pass
    retry_after = headers.get("retry-after")
    try:
        value = float(retry_after)
        return value if math.isfinite(value) and value > 0 else None
    except (TypeError, ValueError):
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(retry_after)
    except (TypeError, ValueError, OverflowError):
        return None
    timestamp = time.time() if now is None else now
    value = parsed.timestamp() - timestamp
    return value if math.isfinite(value) and value > 0 else None


def safe_headers(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    return {
        str(name).lower(): str(value)
        for name, value in headers.items()
        if str(name).lower() in SAFE_HEADER_NAMES
        or str(name).lower().startswith(SAFE_HEADER_PREFIXES)
    }


def error_body(error: Exception) -> Any:
    body = getattr(error, "body", None)
    if isinstance(body, (dict, list, str, int, float, bool)) or body is None:
        return body
    return str(body)


def is_temporary(error: Exception) -> bool:
    if type(error).__name__ in {"APIConnectionError", "APITimeoutError"}:
        return True
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        return False
    if status_code >= 500 or status_code in {408, 409}:
        return True
    if status_code != 429:
        return False
    text = json.dumps(error_body(error), ensure_ascii=False).lower()
    permanent_markers = ("insufficient_quota", "billing", "payment required", "credit")
    return not any(marker in text for marker in permanent_markers)


def fallback_delay(failure_number: int) -> float:
    return FALLBACK_DELAYS_SECONDS[min(failure_number - 1, len(FALLBACK_DELAYS_SECONDS) - 1)]


def reader_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="qwen/qwen3.5-9b",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        api_key_file=None,
        reader_provider_json=json.dumps(READER_PROVIDER, separators=(",", ":")),
        max_completion_tokens=20_000,
        timeout_seconds=43_200.0,
        reasoning_effort=None,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=None,
        top_k=20,
        repetition_penalty=None,
        reader_enable_thinking=True,
        evaluator_model="gpt-5.2",
        evaluator_base_url=None,
        evaluator_api_key_env="OPENAI_API_KEY",
        evaluator_api_key_file=None,
        evaluator_reasoning_effort="medium",
        evaluator_max_completion_tokens=4096,
        evaluator_timeout_seconds=43_200.0,
    )


def frozen_request_config(harness: Any, args: argparse.Namespace) -> dict[str, Any]:
    request = harness.build_reader_request(args, [{"role": "user", "content": "<preserved prompt>"}])
    request["messages"] = "preserved prompt_rows.jsonl messages"
    return request


async def call_reader_with_backoff(
    harness: Any,
    *,
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    ledger_path: Path,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=os.environ[args.api_key_env],
        max_retries=0,
    )
    failure_number = 0
    try:
        while True:
            attempt_number = failure_number + 1
            started = time.perf_counter()
            started_at = utc_now()
            try:
                response = await client.chat.completions.create(
                    **harness.build_reader_request(args, messages)
                )
                text = harness.extract_text_from_response_message(response.choices[0].message)
                if not text:
                    raise RuntimeError("Reader returned empty text")
                response_extra = getattr(response, "model_extra", None)
                metadata = {
                    "response_id": getattr(response, "id", None),
                    "response_model": getattr(response, "model", None),
                    "provider": (
                        response_extra.get("provider") if isinstance(response_extra, dict) else None
                    ),
                    "recovery_transport_attempt": attempt_number,
                }
                output = {
                    "response_raw": text,
                    "response_parsed_boxed": harness.extract_boxed_answer(text),
                    "is_unknown": harness.is_unknown(harness.extract_boxed_answer(text)),
                    "usage": harness.extract_usage_dict(response),
                    "reader_metadata": metadata,
                }
                append_jsonl(
                    ledger_path,
                    {
                        "attempt": attempt_number,
                        "started_at_utc": started_at,
                        "completed_at_utc": utc_now(),
                        "duration_seconds": time.perf_counter() - started,
                        "outcome": "success",
                        "response_id": metadata["response_id"],
                        "response_model": metadata["response_model"],
                        "provider": metadata["provider"],
                        "usage": output["usage"],
                    },
                )
                return output
            except Exception as error:
                failure_number += 1
                response = getattr(error, "response", None)
                headers = getattr(response, "headers", None)
                server_delay = parse_retry_after(headers)
                delay = server_delay if server_delay is not None else fallback_delay(failure_number)
                temporary = is_temporary(error)
                append_jsonl(
                    ledger_path,
                    {
                        "attempt": attempt_number,
                        "started_at_utc": started_at,
                        "completed_at_utc": utc_now(),
                        "duration_seconds": time.perf_counter() - started,
                        "outcome": "temporary_error" if temporary else "permanent_error",
                        "error_type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None),
                        "body": error_body(error),
                        "response_headers": safe_headers(headers),
                        "retry_after_seconds": server_delay,
                        "next_delay_seconds": delay if temporary else None,
                        "delay_source": "response_header" if server_delay is not None else "deterministic_backoff",
                    },
                )
                if not temporary:
                    raise
                print(
                    f"READER_RETRY attempt={attempt_number} delay={delay:.3f}s "
                    f"status={getattr(error, 'status_code', None)}",
                    flush=True,
                )
                await asyncio.sleep(delay)
    finally:
        await client.close()


def make_record(harness: Any, prompt_row: dict[str, Any], output: dict[str, Any], score_bool: bool) -> dict[str, Any]:
    return {
        "index": prompt_row["index"],
        "stream_index": prompt_row["stream_index"],
        "question_id": prompt_row["question_id"],
        "question_type": prompt_row["question_type"],
        "category": prompt_row["category"],
        "is_abstention_problem": prompt_row["is_abstention_problem"],
        "eval_function": prompt_row["eval_function"],
        "question_text": prompt_row["question_text"],
        "question_image": prompt_row["question_image"],
        "haystack_ids": prompt_row["haystack_ids"],
        "memory_context": prompt_row["memory_context"],
        "memory_query_duration_seconds": prompt_row["memory_query_duration_seconds"],
        "memory_post_query_duration_seconds": prompt_row["memory_post_query_duration_seconds"],
        "memory_post_query_metadata": prompt_row["memory_post_query_metadata"],
        "memory_context_original_token_count": prompt_row["memory_context_original_token_count"],
        "memory_context_token_count": prompt_row["memory_context_token_count"],
        "memory_context_was_truncated": prompt_row["memory_context_was_truncated"],
        "prompt_messages": prompt_row["prompt_messages"],
        "answer_gold": prompt_row["answer_gold"],
        "response_raw": output["response_raw"],
        "response_parsed_boxed": output["response_parsed_boxed"],
        "is_unknown": output["is_unknown"],
        "score": 1.0 if score_bool else 0.0,
        "score_bool": score_bool,
        "usage": output["usage"],
        "reader_metadata": output["reader_metadata"],
        "timestamp_utc": harness.utc_now_iso(),
    }


def write_aggregate(harness: Any, cell_root: Path, record: dict[str, Any]) -> None:
    aggregate = harness.aggregate_metrics([record])
    aggregate["tokens"] = {
        "prompt_tokens": record["usage"]["prompt_tokens"],
        "completion_tokens": record["usage"]["completion_tokens"],
        "total_tokens": record["usage"]["prompt_tokens"] + record["usage"]["completion_tokens"],
        "avg_prompt_tokens": record["usage"]["prompt_tokens"],
        "avg_completion_tokens": record["usage"]["completion_tokens"],
        "avg_total_tokens": record["usage"]["prompt_tokens"] + record["usage"]["completion_tokens"],
    }
    aggregate["memory_context"] = {
        "avg_original_tokens": record["memory_context_original_token_count"],
        "avg_final_tokens": record["memory_context_token_count"],
        "num_truncated_sequences": int(record["memory_context_was_truncated"]),
    }
    duration = float(record["memory_query_duration_seconds"])
    post_duration = float(record["memory_post_query_duration_seconds"])
    aggregate["memory_query"] = {
        "avg_seconds": duration,
        "p50_seconds": duration,
        "p95_seconds": duration,
        "max_seconds": duration,
        "total_seconds": duration,
    }
    aggregate["memory_post_query"] = {
        "avg_seconds": post_duration,
        "p50_seconds": post_duration,
        "p95_seconds": post_duration,
        "max_seconds": post_duration,
        "total_seconds": post_duration,
    }
    aggregate["completed_at_utc"] = harness.utc_now_iso()
    aggregate["shared_haystack"] = True
    aggregate["shared_haystack_ids"] = record["haystack_ids"]
    write_json(cell_root / "aggregated_metrics.json", aggregate)


def original_retry_evidence(args: argparse.Namespace) -> dict[str, Any]:
    log_root = args.run_root / "logs"
    errors: list[dict[str, Any]] = []
    for cell_name in RECOVERY_CELLS:
        log_path = log_root / f"{cell_name}.log"
        text = log_path.read_text(encoding="utf-8")
        matching = [line for line in text.splitlines() if "openai.RateLimitError: Error code: 429" in line]
        if len(matching) != 1:
            raise RuntimeError(f"Expected one final preserved 429 in {log_path}")
        errors.append(
            {
                "cell": cell_name,
                "log": str(log_path),
                "log_sha256": sha256_file(log_path),
                "final_error": matching[0],
            }
        )
    return {
        "sdk": "openai 3.7.0",
        "client_max_retries": 10,
        "http_attempts_per_failed_reader": 11,
        "reader_requests_concurrent": 1,
        "cell_processes_concurrent": 1,
        "route": READER_PROVIDER,
        "fallback_delay_policy_seconds": {
            "base_before_jitter": [0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            "jitter_multiplier": "0.75 <= multiplier <= 1.0",
            "total_if_no_retry_after": "41.625 <= seconds <= 55.5",
        },
        "retry_after_policy": "Honor retry-after-ms or Retry-After when greater than zero and at most 120 seconds.",
        "observed_headers": "The original runner did not persist response headers, so actual Retry-After and provider headers are unavailable.",
        "observed_body": {
            "provider_name": "Parasail",
            "is_byok": False,
            "limit_source": "upstream_provider_shared_pool",
            "message": "qwen/qwen3.5-9b is temporarily rate-limited upstream",
        },
        "origin": "OpenRouter returned the HTTP response and identified Parasail as the rate-limited upstream provider.",
        "errors": errors,
    }


def preflight(args: argparse.Namespace, harness: Any) -> dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    manifest = read_json(args.evidence_root / "raw_artifact_manifest.json")["hashes"]
    cells = []
    for cell_name in RECOVERY_CELLS:
        cell_root = args.run_root / "cells" / cell_name
        prompt_path = cell_root / "prompt_rows.jsonl"
        relative = str(prompt_path.relative_to(args.run_root))
        if manifest.get(relative) != sha256_file(prompt_path):
            raise RuntimeError(f"Preserved prompt hash differs: {cell_name}")
        attempts = list(cell_root.glob("query_traces/*/attempt_*/summary.json"))
        if len(attempts) != 1 or attempts[0].parent.name != "attempt_001":
            raise RuntimeError(f"Query attempt integrity differs: {cell_name}")
        receipt = read_json(cell_root / "cell_receipt.json")
        if receipt["final_outcome"] != "infrastructure_error" or receipt["score"] is not None:
            raise RuntimeError(f"Cell is not an original unscored failure: {cell_name}")
        if (cell_root / "reader_outputs.jsonl").exists() or (cell_root / "per_question.jsonl").exists():
            raise RuntimeError(f"Cell unexpectedly has reader/scoring output: {cell_name}")
        cells.append(
            {
                "cell": cell_name,
                "prompt_rows_sha256": sha256_file(prompt_path),
                "query_summary": str(attempts[0].relative_to(args.run_root)),
                "query_summary_sha256": sha256_file(attempts[0]),
                "original_failure_receipt_sha256": sha256_file(cell_root / "cell_receipt.json"),
            }
        )
    request_config = frozen_request_config(harness, reader_args())
    return {
        "schema_version": 1,
        "checked_at_utc": utc_now(),
        "purpose": "Resume only reader and required evaluator stages; do not execute query harnesses.",
        "source_hashes": {
            "recovery_runner": sha256_file(Path(__file__)),
            "official_harness": sha256_file(args.official_root / "evaluation/harness.py"),
            "official_evaluator": sha256_file(args.official_root / "evaluation/qa_eval_metrics.py"),
        },
        "cells": cells,
        "reader_request": request_config,
        "reader_request_sha256": hashlib.sha256(
            json.dumps(request_config, sort_keys=True).encode()
        ).hexdigest(),
        "reader_transport": {
            "sdk_internal_retries": 0,
            "concurrency": 1,
            "retry_after": "honor any positive Retry-After or retry-after-ms value",
            "fallback_delays_seconds": list(FALLBACK_DELAYS_SECONDS),
            "fallback_cap_seconds": FALLBACK_DELAYS_SECONDS[-1],
            "temporary_errors": "retry without a fixed attempt limit; preserve each attempt",
            "permanent_errors": "fail closed without changing route or model",
        },
        "evaluator": {
            "model": "gpt-5.2",
            "api": "https://api.openai.com/v1",
            "reasoning_effort": "medium",
            "max_completion_tokens": 4096,
        },
        "original_reader_failures": original_retry_evidence(args),
        "secret_boundary": "API-key values are read by clients from injected environment variables and are not persisted.",
    }


def recover_cell(args: argparse.Namespace, harness: Any, cell_name: str) -> None:
    cell_root = args.run_root / "cells" / cell_name
    recovery_root = cell_root / "reader_recovery"
    original_receipt = recovery_root / "original_failure_receipt.json"
    if not original_receipt.exists():
        recovery_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cell_root / "cell_receipt.json", original_receipt)
    prompt_row = json.loads((cell_root / "prompt_rows.jsonl").read_text(encoding="utf-8").splitlines()[0])
    output_path = recovery_root / "reader_output.json"
    ledger_path = recovery_root / "transport_attempts.jsonl"
    if output_path.exists():
        output = read_json(output_path)
        print(f"RECOVERY_CHECKPOINT reader {cell_name}", flush=True)
    else:
        output = asyncio.run(
            call_reader_with_backoff(
                harness,
                args=reader_args(),
                messages=prompt_row["messages"],
                ledger_path=ledger_path,
            )
        )
        write_json(output_path, output)
    evaluator_path = recovery_root / "evaluator_receipts.jsonl"
    record_path = recovery_root / "scored_record.json"
    if record_path.exists():
        record = read_json(record_path)
        print(f"RECOVERY_CHECKPOINT score {cell_name}", flush=True)
    else:
        score_row = {**prompt_row, **output}
        if evaluator_path.exists():
            evaluator_receipts = [
                json.loads(line)
                for line in evaluator_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(evaluator_receipts) != 1 or not prompt_row["eval_name"].startswith("llm_"):
                raise RuntimeError(f"Cannot resume evaluator checkpoint safely: {cell_name}")
            from evaluation.qa_eval_metrics import (  # pyright: ignore[reportMissingImports]
                _parse_llm_binary_judgement,
            )

            label, _reason = _parse_llm_binary_judgement(evaluator_receipts[0]["judgement"])
            score_bool = label == 1 and not output["is_unknown"]
        else:
            os.environ["LME_EVALUATOR_RECEIPTS_PATH"] = str(evaluator_path)
            score_bool, _, _ = harness.score_prediction(score_row, harness.make_eval_config(reader_args()))
        record = make_record(harness, prompt_row, output, score_bool)
        write_json(record_path, record)
    write_jsonl_record(
        cell_root / "reader_outputs.jsonl",
        {"question_id": prompt_row["question_id"], **output},
    )
    write_jsonl_record(cell_root / "per_question.jsonl", record)
    write_aggregate(harness, cell_root, record)

    selection = read_json(args.selection)
    question = next(row for row in selection["questions"] if row["question_id"] == prompt_row["question_id"])
    original = read_json(original_receipt)
    receipt = build_receipt(
        output_dir=cell_root,
        harness_name=original["harness"],
        question=question,
        process_returncode=0,
        process_log=Path(original["process_log"]),
        evaluator_receipts_path=evaluator_path,
    )
    receipt["cell_name"] = cell_name
    receipt["experiment_id"] = selection["experiment_id"]
    receipt["query_attempt_limit"] = 1
    receipt["output_retry_limit"] = 0
    receipt["reader_recovery"] = {
        "original_failure_receipt": str(original_receipt.relative_to(cell_root)),
        "original_failure_receipt_sha256": sha256_file(original_receipt),
        "transport_attempts": str(ledger_path.relative_to(cell_root)),
        "transport_attempt_count": len(ledger_path.read_text(encoding="utf-8").splitlines()),
        "query_rerun": False,
    }
    write_json(cell_root / "cell_receipt.json", receipt)
    append_jsonl(
        args.run_root / "reader_recovery_progress.jsonl",
        {
            "completed_at_utc": utc_now(),
            "cell": cell_name,
            "score": receipt["score"],
            "reader_cost_provider_reported": receipt["costs_usd"]["reader_provider_reported"],
            "evaluator_cost_api_equivalent": receipt["costs_usd"]["evaluator_api_equivalent"],
            "query_rerun": False,
            "receipt_sha256": sha256_file(cell_root / "cell_receipt.json"),
        },
    )
    print(f"RECOVERY_COMPLETE {cell_name} score={receipt['score']}", flush=True)


def finalize(args: argparse.Namespace) -> None:
    selection = read_json(args.selection)
    receipts = [
        read_json(args.run_root / "cells" / f"{question['order']:02d}-{question['question_id']}-{side}" / "cell_receipt.json")
        for question in selection["questions"]
        for side in ("native", "thinharness")
    ]
    if len(receipts) != TARGET_CELLS or any(row["final_outcome"] != "scored" for row in receipts):
        raise RuntimeError("All 20 query cells must now have final scores")
    costs = {
        "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in receipts),
        "reader_provider_reported": sum(float(row["costs_usd"]["reader_provider_reported"] or 0) for row in receipts),
        "reader_api_equivalent": sum(float(row["costs_usd"]["reader_api_equivalent"]) for row in receipts),
        "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in receipts),
    }
    costs["total_api_equivalent"] = (
        costs["query_api_equivalent"] + costs["reader_api_equivalent"] + costs["evaluator_api_equivalent"]
    )
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "status": "completed_after_reader_recovery",
        "completed_cells": TARGET_CELLS,
        "target_cells": TARGET_CELLS,
        "native_cells": 10,
        "thinharness_cells": 10,
        "query_cells_rerun": 0,
        "reader_cells_recovered": len(RECOVERY_CELLS),
        "cost_totals_usd": costs,
        **summarize_pairs(receipts),
    }
    write_json(args.evidence_root / "final_summary.json", final)
    write_json(
        args.evidence_root / "cell_receipts.json",
        [
            {key: value for key, value in receipt.items() if key not in {"artifact_hashes", "query_attempts", "response_raw"}}
            for receipt in receipts
        ],
    )
    results = compile_results(args.run_root, selection, receipts)
    write_json(args.evidence_root / "comparison.json", results)
    write_results(args.evidence_root / "RESULTS.md", results, final)
    preflight_receipt = read_json(args.evidence_root / "preflight.json")
    validation = validate_completed(args, selection, receipts, preflight_receipt)
    validation["reader_recovery"] = {
        "cells": list(RECOVERY_CELLS),
        "query_cells_rerun": 0,
        "all_twenty_query_summaries_still_attempt_001": True,
    }
    write_json(args.evidence_root / "validation.json", validation)
    write_json(args.run_root / "COMPLETED_AFTER_READER_RECOVERY.json", final)
    print(json.dumps(final, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--cost-estimate", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in ("repo_root", "official_root", "run_root", "evidence_root", "selection", "cost_estimate"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    import evaluation.harness as harness  # pyright: ignore[reportMissingImports]

    recovery_preflight = args.run_root / "reader_recovery_preflight.json"
    if not recovery_preflight.exists():
        receipt = preflight(args, harness)
        write_json(recovery_preflight, receipt)
        write_json(args.evidence_root / "reader_recovery_preflight.json", receipt)
    else:
        receipt = read_json(recovery_preflight)
        if [row["cell"] for row in receipt["cells"]] != list(RECOVERY_CELLS):
            raise RuntimeError("Recovery preflight cell identities differ")
    if args.preflight_only:
        print(json.dumps(receipt, indent=2), flush=True)
        return
    for index, cell_name in enumerate(RECOVERY_CELLS):
        recover_cell(args, harness, cell_name)
        if index + 1 < len(RECOVERY_CELLS):
            time.sleep(5.0)
    finalize(args)


if __name__ == "__main__":
    main()
