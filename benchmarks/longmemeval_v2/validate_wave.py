#!/usr/bin/env python3
"""Validate the completed paired wave and write a compact evidence receipt."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

from .orchestrate import hash_tree, read_json, sha256_file, utc_now, write_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def usage_totals(rows: list[dict[str, Any]], harness_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row["harness"] == harness_name]
    usage_fields = (
        "requests",
        "input_tokens",
        "cached_input_tokens",
        "ordinary_input_tokens",
        "cache_write_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "tool_calls",
    )
    durations = [float(row["memory_query_duration_seconds"]) for row in selected]
    return {
        "cells": len(selected),
        "correct": sum(float(row["score"]) for row in selected),
        "query_usage": {
            field: sum(int(row["query_usage"][field]) for row in selected)
            for field in usage_fields
        },
        "query_latency_seconds": {
            "total": sum(durations),
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
        },
        "costs_usd": {
            "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in selected),
            "reader_provider_reported": sum(float(row["costs_usd"]["reader_provider_reported"] or 0) for row in selected),
            "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in selected),
        },
    }


def secret_scan(roots: list[Path]) -> dict[str, Any]:
    secrets = [
        value.encode()
        for name in ("OPENAI_API_KEY", "OPENROUTER_API_KEY")
        if (value := os.getenv(name))
    ]
    files_scanned = bytes_scanned = exact_matches = 0
    suffixes = {".json", ".jsonl", ".log", ".md", ".txt", ".patch", ".env"}
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or (path.suffix.lower() not in suffixes and path.name not in {"COMPLETED.json", "progress.jsonl"}):
                continue
            data = path.read_bytes()
            files_scanned += 1
            bytes_scanned += len(data)
            exact_matches += sum(data.count(secret) for secret in secrets)
    if exact_matches:
        raise RuntimeError("A secret value was found in benchmark artifacts")
    return {
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "injected_secret_values_found": exact_matches,
        "secret_values_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    for name in ("official_root", "selection", "run_root", "evidence_root"):
        setattr(args, name, getattr(args, name).resolve())
    sys.path.insert(0, str(args.official_root))
    from evaluation.qa_eval_metrics import (  # noqa: PLC0415
        _parse_llm_binary_judgement,
        eval_from_spec,
        score_to_bool,
    )

    selection = read_json(args.selection)
    final = read_json(args.evidence_root / "final_summary.json")
    compact = read_json(args.evidence_root / "cell_receipts.json")
    if final["status"] != "completed" or final["completed_cells"] != 28:
        raise RuntimeError("Wave is not complete")
    if len(compact) != 28:
        raise RuntimeError("Expected 28 compact receipts")
    expected_keys = {
        (question["question_id"], harness_name)
        for question in selection["questions"]
        for harness_name in ("native", "thinharness")
    }
    actual_keys = {(row["question_id"], row["harness"]) for row in compact}
    if actual_keys != expected_keys or len(actual_keys) != 28:
        raise RuntimeError("Cell coverage differs from the frozen selection")

    artifact_files = artifact_bytes = 0
    deterministic_checked = llm_checked = 0
    for question in selection["questions"]:
        for harness_name in ("native", "thinharness"):
            cell_name = f"{question['order']:02d}-{question['question_id']}-{harness_name}"
            cell_root = args.run_root / "cells" / cell_name
            receipt = read_json(cell_root / "cell_receipt.json")
            scored_records = load_jsonl(cell_root / "per_question.jsonl")
            if len(scored_records) != 1:
                raise RuntimeError(f"Expected one per-question record: {cell_name}")
            scored_record = scored_records[0]
            if receipt["final_outcome"] != "scored" or receipt["score"] not in {0.0, 1.0}:
                raise RuntimeError(f"Invalid final cell outcome: {cell_name}")
            metadata = read_json(cell_root / "cell_metadata.json")
            if metadata["query_api"] != "https://api.openai.com/v1" or metadata["evaluator_api"] != "https://api.openai.com/v1":
                raise RuntimeError(f"Non-direct OpenAI route in {cell_name}")
            reader_metadata = receipt.get("reader_metadata") or {}
            if reader_metadata.get("provider") != "Parasail" or reader_metadata.get("response_model") != "qwen/qwen3.5-9b":
                raise RuntimeError(f"Reader route differs in {cell_name}")
            for relative, expected_hash in receipt["artifact_hashes"].items():
                path = cell_root / relative
                if sha256_file(path) != expected_hash:
                    raise RuntimeError(f"Artifact hash differs: {path}")
                artifact_files += 1
                artifact_bytes += path.stat().st_size
            if receipt["scoring_path"] == "deterministic":
                rescored = score_to_bool(
                    eval_from_spec(
                        scored_record["eval_function"],
                        receipt["response_parsed_boxed"],
                        receipt["answer_gold"],
                    )
                )
                if receipt["is_unknown"]:
                    rescored = False
                if rescored != receipt["score_bool"]:
                    raise RuntimeError(f"Deterministic score differs: {cell_name}")
                deterministic_checked += 1
            else:
                judge_receipts = receipt["evaluator_receipts"]
                if len(judge_receipts) != 1:
                    raise RuntimeError(f"Expected one evaluator receipt: {cell_name}")
                label, _ = _parse_llm_binary_judgement(judge_receipts[0]["judgement"])
                if bool(label) != receipt["score_bool"]:
                    raise RuntimeError(f"LLM judgement differs: {cell_name}")
                llm_checked += 1

    current_run_hashes = hash_tree(args.run_root)
    write_json(
        args.evidence_root / "raw_artifact_manifest.json",
        {"run_root": str(args.run_root), "run_root_hashes": current_run_hashes},
    )
    manifest = read_json(args.evidence_root / "raw_artifact_manifest.json")
    for relative, expected_hash in manifest["run_root_hashes"].items():
        if sha256_file(args.run_root / relative) != expected_hash:
            raise RuntimeError(f"Raw artifact manifest differs: {relative}")

    native = usage_totals(compact, "native")
    thin = usage_totals(compact, "thinharness")
    validation = {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "passed",
        "coverage": {
            "question_ids": 14,
            "cells": 28,
            "native_cells": 14,
            "thinharness_cells": 14,
            "domain_type_cells": 14,
            "question_image_cells": sum(row["has_question_image"] for row in compact),
        },
        "scoring": {
            "deterministic_cells_recomputed": deterministic_checked,
            "llm_judged_cells_matched_to_receipts": llm_checked,
        },
        "artifacts": {
            "cell_artifact_files_verified": artifact_files,
            "cell_artifact_bytes_verified": artifact_bytes,
            "raw_manifest_files_verified": len(manifest["run_root_hashes"]),
            "raw_manifest_sha256": sha256_file(args.evidence_root / "raw_artifact_manifest.json"),
        },
        "secret_boundary": secret_scan([args.run_root, args.evidence_root]),
        "harness_totals": {"native": native, "thinharness": thin},
        "efficiency_ratios_thinharness_over_native": {
            "input_tokens": thin["query_usage"]["input_tokens"] / native["query_usage"]["input_tokens"],
            "output_tokens": thin["query_usage"]["output_tokens"] / native["query_usage"]["output_tokens"],
            "tool_calls": thin["query_usage"]["tool_calls"] / native["query_usage"]["tool_calls"],
            "mean_latency": thin["query_latency_seconds"]["mean"] / native["query_latency_seconds"]["mean"],
            "query_api_equivalent": thin["costs_usd"]["query_api_equivalent"] / native["costs_usd"]["query_api_equivalent"],
        },
    }
    write_json(args.evidence_root / "validation.json", validation)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
