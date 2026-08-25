#!/usr/bin/env python3
"""Restart-safe paired LongMemEval-V2 wave orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OFFICIAL_REVISION = "2cc8c540bdb87fe6761629b585e727e1c4704520"
LUNA_INPUT_USD_PER_M = 0.20
LUNA_CACHED_INPUT_USD_PER_M = 0.02
LUNA_OUTPUT_USD_PER_M = 1.20
READER_INPUT_USD_PER_M = 0.10
READER_OUTPUT_USD_PER_M = 0.25
EVALUATOR_INPUT_USD_PER_M = 1.75
EVALUATOR_CACHED_INPUT_USD_PER_M = 0.175
EVALUATOR_OUTPUT_USD_PER_M = 14.0
CREDIT_PATTERNS = (
    "insufficient_quota",
    "billing_hard_limit_reached",
    "credit balance",
    "credits exhausted",
    "account_deactivated",
    "payment_required",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, excluded: set[str] | None = None) -> dict[str, str]:
    excluded = excluded or set()
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        hashes[relative] = sha256_file(path)
    return hashes


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def normalize_patch_blank_context(patch: bytes) -> bytes:
    return b"".join(
        b"\n" if line in {b" \n", b" "} else line
        for line in patch.splitlines(keepends=True)
    )


def preflight(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime

    if len(selection["questions"]) != 14:
        raise RuntimeError("Frozen selection does not contain 14 questions")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    official_revision = command_output(["git", "rev-parse", "HEAD"], cwd=args.official_root)
    if official_revision != OFFICIAL_REVISION:
        raise RuntimeError(f"Official harness revision differs: {official_revision}")
    patch_bytes = args.patch.read_bytes()
    official_diff = subprocess.check_output(
        ["git", "diff", "--", "evaluation/harness.py", "evaluation/run_eval.py", "evaluation/qa_eval_metrics.py"],
        cwd=args.official_root,
    )
    if normalize_patch_blank_context(official_diff) != patch_bytes:
        raise RuntimeError("Official harness instrumentation differs from the committed patch")
    repo_status = command_output(["git", "status", "--porcelain"], cwd=args.repo_root)
    if repo_status:
        raise RuntimeError("ThinHarness repository must be clean before paid execution")
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")
    if prepared["selection_sha256"] != sha256_file(args.selection):
        raise RuntimeError("Prepared data does not match the frozen selection")
    for domain in ("web", "enterprise"):
        data_root = Path(prepared["domains"][domain]["data_root"])
        if sha256_file(data_root / "trajectories.jsonl") != prepared["domains"][domain]["trajectories_jsonl_sha256"]:
            raise RuntimeError(f"Prepared {domain} trajectories hash differs")
    return {
        "checked_at_utc": utc_now(),
        "official_revision": official_revision,
        "official_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "thinharness_revision": command_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root),
        "selection_sha256": sha256_file(args.selection),
        "prepared_data_manifest_sha256": sha256_file(args.prepared_root / "prepared_data_manifest.json"),
        "ripgrep_runtime": verify_ripgrep_runtime(),
        "api_key_boundaries": {
            "OPENAI_API_KEY": "present, value not read or persisted",
            "OPENROUTER_API_KEY": "present, value not read or persisted",
        },
        "retry_policy": {
            "query_attempts_per_cell": 3,
            "native_query_http_retries_per_attempt": 0,
            "thinharness_query_http_retries_per_attempt": 0,
            "reader_openai_client_max_retries": 10,
            "evaluator_openai_client_max_retries": 10,
            "whole_cell_retries": 0,
            "rule": "No rerun after a cell receipt exists; retries are transport/protocol only.",
        },
    }


def iter_json_files(root: Path, name: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob(name)):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append({"path": str(path), "payload": payload})
    return rows


def query_attempts(output_dir: Path) -> list[dict[str, Any]]:
    attempts = []
    for item in iter_json_files(output_dir / "query_traces", "summary.json"):
        payload = item["payload"]
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            summary_fields = payload.get("summary_fields")
            if isinstance(summary_fields, dict):
                usage = summary_fields.get("usage")
        attempts.append(
            {
                "path": str(Path(item["path"]).relative_to(output_dir)),
                "status": payload.get("status_after"),
                "detail": payload.get("status_after_detail") or payload.get("runner_error_detail"),
                "duration_seconds": payload.get("duration_seconds"),
                "tool_call_count": payload.get("tool_call_count")
                or (payload.get("summary_fields") or {}).get("tool_call_count"),
                "usage": usage if isinstance(usage, dict) else {},
            }
        )
    return attempts


def normalized_query_usage(attempts: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "ordinary_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "tool_calls": 0,
    }
    for attempt in attempts:
        usage = attempt["usage"]
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        cached = int(usage.get("cached_input_tokens", usage.get("cached_tokens", 0)) or 0)
        totals["requests"] += int(usage.get("requests", usage.get("model_requests", 0)) or 0)
        totals["input_tokens"] += input_tokens
        totals["cached_input_tokens"] += cached
        totals["ordinary_input_tokens"] += int(
            usage.get("ordinary_input_tokens", max(input_tokens - cached, 0)) or 0
        )
        totals["cache_write_input_tokens"] += int(usage.get("cache_write_input_tokens", 0) or 0)
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        totals["reasoning_output_tokens"] += int(usage.get("reasoning_output_tokens", 0) or 0)
        totals["tool_calls"] += int(
            attempt.get("tool_call_count") or usage.get("tool_calls", 0) or 0
        )
    return totals


def token_cost(
    *,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    input_rate: float,
    cached_rate: float,
    output_rate: float,
) -> float:
    ordinary = max(input_tokens - cached_tokens, 0)
    return (
        ordinary * input_rate + cached_tokens * cached_rate + output_tokens * output_rate
    ) / 1_000_000


def evaluator_receipts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_receipt(
    *,
    output_dir: Path,
    harness_name: str,
    question: dict[str, Any],
    process_returncode: int,
    process_log: Path,
    evaluator_receipts_path: Path,
) -> dict[str, Any]:
    per_question_path = output_dir / "per_question.jsonl"
    records = []
    if per_question_path.exists():
        records = [json.loads(line) for line in per_question_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) > 1:
        raise RuntimeError(f"Cell has more than one scored record: {output_dir}")
    record = records[0] if records else {}
    attempts = query_attempts(output_dir)
    query_usage = normalized_query_usage(attempts)
    reader_usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    reader_details = reader_usage.get("details") if isinstance(reader_usage.get("details"), dict) else {}
    reader_cached = int(
        ((reader_details.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
        if isinstance(reader_details.get("prompt_tokens_details"), dict)
        else 0
    )
    judge_receipts = evaluator_receipts(evaluator_receipts_path)
    judge_input = judge_output = judge_cached = 0
    for receipt in judge_receipts:
        usage = receipt.get("usage") if isinstance(receipt.get("usage"), dict) else {}
        judge_input += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        judge_output += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
        if isinstance(details, dict):
            judge_cached += int(details.get("cached_tokens", 0) or 0)
    query_cost = token_cost(
        input_tokens=query_usage["input_tokens"],
        cached_tokens=query_usage["cached_input_tokens"],
        output_tokens=query_usage["output_tokens"],
        input_rate=LUNA_INPUT_USD_PER_M,
        cached_rate=LUNA_CACHED_INPUT_USD_PER_M,
        output_rate=LUNA_OUTPUT_USD_PER_M,
    )
    reader_cost_equivalent = token_cost(
        input_tokens=int(reader_usage.get("prompt_tokens", 0) or 0),
        cached_tokens=reader_cached,
        output_tokens=int(reader_usage.get("completion_tokens", 0) or 0),
        input_rate=READER_INPUT_USD_PER_M,
        cached_rate=READER_INPUT_USD_PER_M,
        output_rate=READER_OUTPUT_USD_PER_M,
    )
    evaluator_cost = token_cost(
        input_tokens=judge_input,
        cached_tokens=judge_cached,
        output_tokens=judge_output,
        input_rate=EVALUATOR_INPUT_USD_PER_M,
        cached_rate=EVALUATOR_CACHED_INPUT_USD_PER_M,
        output_rate=EVALUATOR_OUTPUT_USD_PER_M,
    )
    provider_reported_reader_cost = reader_details.get("cost")
    receipt = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "harness": harness_name,
        "question_id": question["question_id"],
        "domain": question["domain"],
        "question_type": question["question_type"],
        "has_question_image": question["has_question_image"],
        "eval_name": question["eval_name"],
        "scoring_path": "llm_judged" if question["eval_name"].startswith("llm_") else "deterministic",
        "process_returncode": process_returncode,
        "final_outcome": "scored" if record else "infrastructure_error",
        "answer_gold": record.get("answer_gold"),
        "response_raw": record.get("response_raw"),
        "response_parsed_boxed": record.get("response_parsed_boxed"),
        "score": record.get("score"),
        "score_bool": record.get("score_bool"),
        "is_unknown": record.get("is_unknown"),
        "memory_query_duration_seconds": record.get("memory_query_duration_seconds"),
        "memory_post_query_duration_seconds": record.get("memory_post_query_duration_seconds"),
        "memory_context_original_token_count": record.get("memory_context_original_token_count"),
        "memory_context_token_count": record.get("memory_context_token_count"),
        "query_attempts": attempts,
        "query_usage": query_usage,
        "reader_usage": reader_usage,
        "reader_metadata": record.get("reader_metadata"),
        "evaluator_receipts": judge_receipts,
        "costs_usd": {
            "query_api_equivalent": query_cost,
            "reader_provider_reported": provider_reported_reader_cost,
            "reader_api_equivalent": reader_cost_equivalent,
            "evaluator_api_equivalent": evaluator_cost,
            "total_api_equivalent": query_cost + reader_cost_equivalent + evaluator_cost,
        },
        "process_log": str(process_log),
    }
    receipt["artifact_hashes"] = hash_tree(
        output_dir,
        excluded={"cell_receipt.json"},
    )
    receipt["artifact_manifest_sha256"] = hashlib.sha256(
        json.dumps(receipt["artifact_hashes"], sort_keys=True).encode()
    ).hexdigest()
    write_json(output_dir / "cell_receipt.json", receipt)
    return receipt


def detect_credit_exhaustion(text: str, receipt: dict[str, Any]) -> bool:
    corpus = text.lower() + "\n" + json.dumps(receipt.get("query_attempts", [])).lower()
    return any(pattern in corpus for pattern in CREDIT_PATTERNS)


def run_subprocess(command: list[str], log_path: Path, cwd: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            captured.append(line)
        return process.wait(), "".join(captured)


def summarize_pairs(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    for receipt in receipts:
        by_question.setdefault(receipt["question_id"], {})[receipt["harness"]] = receipt
    pairs = []
    for question_id, pair in by_question.items():
        native = pair.get("native")
        thin = pair.get("thinharness")
        pairs.append(
            {
                "question_id": question_id,
                "domain": (native or thin or {}).get("domain"),
                "question_type": (native or thin or {}).get("question_type"),
                "scoring_path": (native or thin or {}).get("scoring_path"),
                "native_score": native.get("score") if native else None,
                "thinharness_score": thin.get("score") if thin else None,
                "paired_difference": (
                    float(thin["score"]) - float(native["score"])
                    if native and thin and native.get("score") is not None and thin.get("score") is not None
                    else None
                ),
            }
        )
    complete = [row for row in pairs if row["paired_difference"] is not None]
    summaries = {}
    for scoring_path in ("all", "deterministic", "llm_judged"):
        rows = complete if scoring_path == "all" else [row for row in complete if row["scoring_path"] == scoring_path]
        summaries[scoring_path] = {
            "pair_count": len(rows),
            "native_correct": sum(float(row["native_score"]) for row in rows),
            "thinharness_correct": sum(float(row["thinharness_score"]) for row in rows),
            "mean_paired_difference": (
                sum(float(row["paired_difference"]) for row in rows) / len(rows) if rows else None
            ),
            "thin_win_native_loss": sum(
                row["thinharness_score"] == 1.0 and row["native_score"] == 0.0 for row in rows
            ),
            "native_win_thin_loss": sum(
                row["native_score"] == 1.0 and row["thinharness_score"] == 0.0 for row in rows
            ),
        }
    return {"pairs": pairs, "summaries": summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    for field in ("repo_root", "official_root", "prepared_root", "selection", "patch", "run_root", "evidence_root"):
        setattr(args, field, getattr(args, field).resolve())
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    preflight_receipt = preflight(args, selection)
    write_json(args.run_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "preflight.json", preflight_receipt)
    prepared_manifest = read_json(args.prepared_root / "prepared_data_manifest.json")

    receipts: list[dict[str, Any]] = []
    progress_ledger = args.run_root / "progress.jsonl"
    credit_exhausted = False
    for question in selection["questions"]:
        for harness_name in ("native", "thinharness"):
            cell_name = f"{question['order']:02d}-{question['question_id']}-{harness_name}"
            output_dir = args.run_root / "cells" / cell_name
            receipt_path = output_dir / "cell_receipt.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                receipts.append(receipt)
                print(f"CHECKPOINT skip completed {cell_name}", flush=True)
                continue
            if credit_exhausted:
                break
            print(f"CELL_START {cell_name}", flush=True)
            evaluator_path = args.run_root / "evaluator_receipts" / f"{cell_name}.jsonl"
            evaluator_path.parent.mkdir(parents=True, exist_ok=True)
            os.environ["LME_EVALUATOR_RECEIPTS_PATH"] = str(evaluator_path)
            command = [
                sys.executable,
                "-m",
                "benchmarks.longmemeval_v2.run_cell",
                "--official-root",
                str(args.official_root),
                "--data-root",
                prepared_manifest["domains"][question["domain"]]["data_root"],
                "--output-dir",
                str(output_dir),
                "--harness",
                harness_name,
                "--question-id",
                question["question_id"],
                "--domain",
                question["domain"],
            ]
            process_log = args.run_root / "logs" / f"{cell_name}.log"
            returncode, output = run_subprocess(command, process_log, args.repo_root)
            receipt = build_receipt(
                output_dir=output_dir,
                harness_name=harness_name,
                question=question,
                process_returncode=returncode,
                process_log=process_log,
                evaluator_receipts_path=evaluator_path,
            )
            receipts.append(receipt)
            with progress_ledger.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps({
                    "completed_at_utc": receipt["completed_at_utc"],
                    "cell": cell_name,
                    "final_outcome": receipt["final_outcome"],
                    "score": receipt["score"],
                    "receipt_sha256": sha256_file(receipt_path),
                }, ensure_ascii=True) + "\n")
            write_json(
                args.run_root / "progress.json",
                {
                    "updated_at_utc": utc_now(),
                    "completed_cells": len(receipts),
                    "target_cells": 28,
                    "last_cell": cell_name,
                },
            )
            print(
                f"CELL_COMPLETE {cell_name} outcome={receipt['final_outcome']} score={receipt['score']}",
                flush=True,
            )
            if detect_credit_exhaustion(output, receipt):
                credit_exhausted = True
                print(f"CREDIT_EXHAUSTED after {cell_name}", flush=True)
                break
            if receipt["final_outcome"] != "scored":
                raise RuntimeError(f"Unrecoverable cell failure: {cell_name}; see {process_log}")
        if credit_exhausted:
            break

    pair_summary = summarize_pairs(receipts)
    totals = {
        "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in receipts),
        "reader_provider_reported": sum(
            float(row["costs_usd"]["reader_provider_reported"] or 0) for row in receipts
        ),
        "reader_api_equivalent": sum(float(row["costs_usd"]["reader_api_equivalent"]) for row in receipts),
        "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in receipts),
        "total_api_equivalent": sum(float(row["costs_usd"]["total_api_equivalent"]) for row in receipts),
    }
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "status": "credit_exhausted" if credit_exhausted else "completed",
        "completed_cells": len(receipts),
        "target_cells": 28,
        "native_cells": sum(row["harness"] == "native" for row in receipts),
        "thinharness_cells": sum(row["harness"] == "thinharness" for row in receipts),
        "cost_totals_usd": totals,
        **pair_summary,
    }
    marker_name = "CREDIT_EXHAUSTED.json" if credit_exhausted else "COMPLETED.json"
    write_json(args.run_root / marker_name, final)
    write_json(args.evidence_root / "final_summary.json", final)
    compact_receipts = [
        {key: value for key, value in receipt.items() if key not in {"response_raw", "artifact_hashes", "query_attempts"}}
        for receipt in receipts
    ]
    write_json(args.evidence_root / "cell_receipts.json", compact_receipts)
    write_json(
        args.evidence_root / "raw_artifact_manifest.json",
        {
            "run_root": str(args.run_root),
            "run_root_hashes": hash_tree(args.run_root),
        },
    )
    print(f"WAVE_COMPLETE status={final['status']} cells={len(receipts)}/28", flush=True)


if __name__ == "__main__":
    main()
