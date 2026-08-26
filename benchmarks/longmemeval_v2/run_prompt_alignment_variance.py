#!/usr/bin/env python3
"""Run the two-cell native-prompt alignment variance check."""

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

from benchmarks.longmemeval_v2.orchestrate import (
    OFFICIAL_REVISION,
    build_receipt,
    hash_tree,
    normalize_patch_blank_context,
    run_subprocess,
    sha256_file,
    write_json,
)
from benchmarks.longmemeval_v2.prompt_alignment import build_aligned_query_prompt
from benchmarks.longmemeval_v2.run_cell import thinharness_memory_params
from benchmarks.longmemeval_v2.run_prompt_alignment import (
    _question_text,
    command_output,
    configure_ripgrep,
    extract_native_query_prompt,
)
from benchmarks.longmemeval_v2.run_rg_diagnostics import search_diagnostics
from benchmarks.longmemeval_v2.validate_wave import secret_scan

FROZEN_COMMIT = "57013717f0c8343cd3f626153df5bd26c4aa0b93"
EXPECTED_IDS = ["f61a096f", "b82d0dd6"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_digest_at_commit(repo_root: Path, commit: str, prefix: str) -> str:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", commit, prefix], cwd=repo_root, text=True
    )
    rows: list[tuple[str, str]] = []
    for line in output.splitlines():
        _metadata, path = line.split("\t", 1)
        content = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo_root)
        rows.append((path, hashlib.sha256(content).hexdigest()))
    serialized = "".join(f"{digest}  {path}\n" for path, digest in rows).encode()
    return hashlib.sha256(serialized).hexdigest()


def validate_frozen_source(repo_root: Path, selection: dict[str, Any]) -> dict[str, str]:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", FROZEN_COMMIT, "HEAD"],
        cwd=repo_root,
        check=False,
    ).returncode:
        raise RuntimeError(f"Current revision does not descend from {FROZEN_COMMIT}")
    current_hashes: dict[str, str] = {}
    for relative, expected in selection["frozen_source_sha256"].items():
        path = repo_root / relative
        actual = sha256_file(path)
        frozen = hashlib.sha256(
            subprocess.check_output(["git", "show", f"{FROZEN_COMMIT}:{relative}"], cwd=repo_root)
        ).hexdigest()
        if actual != expected or frozen != expected:
            raise RuntimeError(f"Frozen source differs: {relative}")
        current_hashes[relative] = actual
    tree_digest = tree_digest_at_commit(repo_root, "HEAD", "thinharness/")
    frozen_tree_digest = tree_digest_at_commit(repo_root, FROZEN_COMMIT, "thinharness/")
    expected_tree_digest = selection["thinharness_python_tree_sha256"]
    if tree_digest != expected_tree_digest or frozen_tree_digest != expected_tree_digest:
        raise RuntimeError("ThinHarness Python source tree differs from commit 5701371")
    current_hashes["thinharness/"] = tree_digest
    return current_hashes


def normalize_memory_config(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    params = normalized["memory_params"]
    for key in ("workspace_dir", "trajectories_root_dir", "query_trace_dir"):
        params[key] = f"<{key}>"
    return normalized


def expected_memory_config() -> dict[str, Any]:
    return normalize_memory_config(
        {
            "memory_type": "thinharness",
            "memory_params": thinharness_memory_params(Path("/frozen/output"), Path("/frozen/data")),
        }
    )


def preflight(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    if [row["question_id"] for row in selection.get("questions", [])] != EXPECTED_IDS:
        raise RuntimeError("Variance-check selection must be exactly f61a096f then b82d0dd6")
    if selection["frozen_source_commit"] != FROZEN_COMMIT:
        raise RuntimeError("Frozen source commit differs")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    if command_output(["git", "status", "--porcelain"], cwd=args.repo_root):
        raise RuntimeError("ThinHarness repository must be clean before paid execution")

    source_hashes = validate_frozen_source(args.repo_root, selection)
    if sha256_file(args.selection) != args.selection_sha256:
        raise RuntimeError("Variance-check selection hash differs from the launch command")
    if sha256_file(args.alignment_selection) != selection["prompt_alignment_selection_sha256"]:
        raise RuntimeError("Prompt-alignment selection differs")
    if sha256_file(args.alignment_validation) != selection["prompt_alignment_validation_sha256"]:
        raise RuntimeError("Prompt-alignment validation differs")
    if sha256_file(args.alignment_receipts) != selection["prompt_alignment_cell_receipts_sha256"]:
        raise RuntimeError("Prompt-alignment compact receipts differ")

    official_revision = command_output(["git", "rev-parse", "HEAD"], cwd=args.official_root)
    if official_revision != OFFICIAL_REVISION or official_revision != selection["official_harness_revision"]:
        raise RuntimeError(f"Official harness revision differs: {official_revision}")
    patch_bytes = args.patch.read_bytes()
    official_diff = subprocess.check_output(
        ["git", "diff", "--", "evaluation/harness.py", "evaluation/run_eval.py", "evaluation/qa_eval_metrics.py"],
        cwd=args.official_root,
    )
    if normalize_patch_blank_context(official_diff) != patch_bytes:
        raise RuntimeError("Official harness instrumentation differs from the frozen patch")
    if hashlib.sha256(patch_bytes).hexdigest() != selection["official_patch_sha256"]:
        raise RuntimeError("Official patch hash differs")

    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")
    if sha256_file(args.prepared_root / "prepared_data_manifest.json") != selection["prepared_data_manifest_sha256"]:
        raise RuntimeError("Prepared-data manifest differs")
    if prepared["source_questions_sha256"] != selection["source_questions_sha256"]:
        raise RuntimeError("Source questions hash differs")
    for domain, expected in selection["prepared_trajectories_sha256"].items():
        data_path = Path(prepared["domains"][domain]["data_root"]) / "trajectories.jsonl"
        if sha256_file(data_path) != expected:
            raise RuntimeError(f"Prepared {domain} trajectories differ")

    native_prompt = extract_native_query_prompt(args.official_root / "memory_modules" / "agentrunbook_c.py")
    native_instruction_path = (
        args.official_root / "memory_modules" / "assets" / "agentrunbook_c" / "INSTRUCTION.md"
    )
    native_instruction = native_instruction_path.read_text(encoding="utf-8")
    if hashlib.sha256(native_prompt.encode()).hexdigest() != selection["native_query_prompt_sha256"]:
        raise RuntimeError("Native query prompt differs")
    if hashlib.sha256(native_instruction.encode()).hexdigest() != selection["native_instruction_sha256"]:
        raise RuntimeError("Native instruction differs")

    original_cells = args.alignment_run_root / "cells"
    for question in selection["questions"]:
        old_name = (
            f"aligned-prompt-{question['aligned_attempt_1_order']:02d}-"
            f"{question['question_id']}-thinharness"
        )
        old_root = original_cells / old_name
        if sha256_file(old_root / "cell_receipt.json") != question["aligned_attempt_1_cell_receipt_sha256"]:
            raise RuntimeError(f"Aligned attempt-1 receipt differs: {question['question_id']}")
        if sha256_file(old_root / "runtime_inputs" / "questions.json") != question["runtime_questions_sha256"]:
            raise RuntimeError(f"Aligned attempt-1 task reference differs: {question['question_id']}")
        if sha256_file(old_root / "runtime_inputs" / "haystack.json") != question["runtime_haystack_sha256"]:
            raise RuntimeError(f"Aligned attempt-1 haystack reference differs: {question['question_id']}")
        old_config = read_json(old_root / "runtime_inputs" / "memory_config.json")
        if normalize_memory_config(old_config) != expected_memory_config():
            raise RuntimeError(f"Aligned attempt-1 memory config differs: {question['question_id']}")

    from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime

    ripgrep = verify_ripgrep_runtime()
    if Path(ripgrep["rg_path"]).resolve() != args.rg_bin:
        raise RuntimeError("Ripgrep path differs")
    return {
        "schema_version": 1,
        "checked_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "selection_sha256": sha256_file(args.selection),
        "launch_selection_sha256": args.selection_sha256,
        "thinharness_revision": command_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root),
        "frozen_source_commit": FROZEN_COMMIT,
        "frozen_source_sha256": source_hashes,
        "official_revision": official_revision,
        "official_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "prepared_data_manifest_sha256": sha256_file(args.prepared_root / "prepared_data_manifest.json"),
        "prompt_identity": {
            "native_query_prompt_sha256": selection["native_query_prompt_sha256"],
            "native_instruction_sha256": selection["native_instruction_sha256"],
            "aligned_prompt_template_sha256": selection["aligned_prompt_template_sha256"],
            "prompt_diff_sha256": selection["prompt_diff_sha256"],
        },
        "runtime_config": expected_memory_config(),
        "retry_policy": {
            "whole_cell_retries": 0,
            "query_http_retries": 0,
            "required_observed_query_attempts_per_cell": 1,
            "required_observed_output_retries_per_cell": 0,
            "frozen_max_attempts_guardrail": 3,
            "frozen_output_retries_guardrail": 1,
            "rule": "Never relaunch a question after output starts; fail validation unless exactly one query attempt completed.",
        },
        "concurrency": 1,
        "ripgrep_runtime": ripgrep,
        "api_key_boundaries": {
            "OPENAI_API_KEY": "present; value not read or persisted",
            "OPENROUTER_API_KEY": "present; value not read or persisted",
        },
    }


def load_memory_context(cell_root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in (cell_root / "prompt_rows.jsonl").read_text().splitlines() if line]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one prompt row: {cell_root}")
    context = rows[0].get("memory_context")
    if not isinstance(context, list):
        raise RuntimeError(f"Missing memory context: {cell_root}")
    return context


def compact_metrics(receipt: dict[str, Any], cell_root: Path) -> dict[str, Any]:
    diagnostics = search_diagnostics(cell_root, "thinharness" if receipt["harness"] == "thinharness" else "native")
    return {
        "score": receipt["score"],
        "answer_gold": receipt["answer_gold"],
        "response_parsed_boxed": receipt["response_parsed_boxed"],
        "response_raw": receipt["response_raw"],
        "memory_context": load_memory_context(cell_root),
        "requests": receipt["query_usage"]["requests"],
        "tool_calls": receipt["query_usage"]["tool_calls"],
        "input_tokens": receipt["query_usage"]["input_tokens"],
        "cached_input_tokens": receipt["query_usage"]["cached_input_tokens"],
        "ordinary_input_tokens": receipt["query_usage"]["ordinary_input_tokens"],
        "cache_write_input_tokens": receipt["query_usage"]["cache_write_input_tokens"],
        "output_tokens": receipt["query_usage"]["output_tokens"],
        "reasoning_output_tokens": receipt["query_usage"]["reasoning_output_tokens"],
        "reader_input_tokens": int(receipt["reader_usage"].get("prompt_tokens", 0) or 0),
        "reader_output_tokens": int(receipt["reader_usage"].get("completion_tokens", 0) or 0),
        "query_latency_seconds": receipt["memory_query_duration_seconds"],
        "search_failures": diagnostics["failed_search_calls"],
        "query_api_equivalent_usd": receipt["costs_usd"]["query_api_equivalent"],
        "reader_provider_reported_usd": receipt["costs_usd"]["reader_provider_reported"],
        "reader_api_equivalent_usd": receipt["costs_usd"]["reader_api_equivalent"],
        "evaluator_api_equivalent_usd": receipt["costs_usd"]["evaluator_api_equivalent"],
        "total_api_equivalent_usd": receipt["costs_usd"]["total_api_equivalent"],
        "scoring_path": receipt["scoring_path"],
    }


def compile_comparison(args: argparse.Namespace, selection: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    replicate_by_id = {row["question_id"]: row for row in receipts}
    rows: list[dict[str, Any]] = []
    for question in selection["questions"]:
        question_id = question["question_id"]
        aligned_name = (
            f"aligned-prompt-{question['aligned_attempt_1_order']:02d}-{question_id}-thinharness"
        )
        native_name = f"{question['original_native_wave_order']:02d}-{question_id}-native"
        replicate_name = f"variance-replicate-2-{question['order']:02d}-{question_id}-thinharness"
        aligned_root = args.alignment_run_root / "cells" / aligned_name
        native_root = args.original_run_root / "cells" / native_name
        replicate_root = args.run_root / "cells" / replicate_name
        aligned_receipt = read_json(aligned_root / "cell_receipt.json")
        native_receipt = read_json(native_root / "cell_receipt.json")
        values = {
            "original_native": compact_metrics(native_receipt, native_root),
            "aligned_attempt_1": compact_metrics(aligned_receipt, aligned_root),
            "aligned_replicate_2": compact_metrics(replicate_by_id[question_id], replicate_root),
        }
        delta_fields = (
            "requests",
            "tool_calls",
            "input_tokens",
            "cached_input_tokens",
            "ordinary_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "query_latency_seconds",
            "search_failures",
            "query_api_equivalent_usd",
            "reader_api_equivalent_usd",
            "evaluator_api_equivalent_usd",
            "total_api_equivalent_usd",
        )
        values["replicate_2_minus_attempt_1"] = {
            field: float(values["aligned_replicate_2"][field] or 0)
            - float(values["aligned_attempt_1"][field] or 0)
            for field in delta_fields
        }
        values["replicate_2_minus_native"] = {
            field: float(values["aligned_replicate_2"][field] or 0)
            - float(values["original_native"][field] or 0)
            for field in delta_fields
        }
        rows.append({"question_id": question_id, **values})
    recovered = sum(row["aligned_replicate_2"]["score"] == 1.0 for row in rows)
    return {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "interpretation": "Two new stochastic ThinHarness replicates; they do not replace prior cells.",
        "recovered": recovered,
        "target": 2,
        "rows": rows,
    }


def validate_completed(
    args: argparse.Namespace,
    selection: dict[str, Any],
    receipts: list[dict[str, Any]],
    preflight_receipt: dict[str, Any],
) -> dict[str, Any]:
    if len(receipts) != 2 or [row["question_id"] for row in receipts] != EXPECTED_IDS:
        raise RuntimeError("Completed cells differ from the frozen two-question selection")
    native_prompt = extract_native_query_prompt(args.official_root / "memory_modules" / "agentrunbook_c.py")
    native_instruction = (
        args.official_root / "memory_modules" / "assets" / "agentrunbook_c" / "INSTRUCTION.md"
    ).read_text(encoding="utf-8")
    artifact_files = 0
    artifact_bytes = 0
    prompt_hashes: dict[str, str] = {}
    response_receipts_checked = 0
    for question, receipt in zip(selection["questions"], receipts, strict=True):
        question_id = question["question_id"]
        cell_name = f"variance-replicate-2-{question['order']:02d}-{question_id}-thinharness"
        cell_root = args.run_root / "cells" / cell_name
        if receipt["final_outcome"] != "scored" or receipt["score"] not in {0.0, 1.0}:
            raise RuntimeError(f"Cell did not produce one final score: {cell_name}")
        if receipt["replicate_label"] != selection["experiment_id"] or receipt["replicate_number"] != 2:
            raise RuntimeError(f"Replicate identity differs: {cell_name}")
        if len(receipt["query_attempts"]) != 1:
            raise RuntimeError(f"Expected exactly one query attempt: {cell_name}")
        attempt_usage = receipt["query_attempts"][0]["usage"]
        if int(attempt_usage.get("output_retries", 0) or 0) != 0:
            raise RuntimeError(f"Output retry occurred: {cell_name}")
        if attempt_usage.get("tool_retries") not in ({}, None):
            raise RuntimeError(f"Tool retry occurred: {cell_name}")
        if receipt["query_usage"]["requests"] != int(attempt_usage.get("response_count", 0)):
            raise RuntimeError(f"Response receipt count differs: {cell_name}")
        response_receipts_checked += receipt["query_usage"]["requests"]

        metadata = read_json(cell_root / "cell_metadata.json")
        if metadata["query_model"] != selection["query_model"] or metadata["query_reasoning_effort"] != "xhigh":
            raise RuntimeError(f"Query model config differs: {cell_name}")
        if metadata["query_api"] != selection["query_api"]:
            raise RuntimeError(f"Query route differs: {cell_name}")
        if metadata["ripgrep_runtime"]["jsonl_search"]["status"] != "passed":
            raise RuntimeError(f"Ripgrep probe differs: {cell_name}")
        if normalize_memory_config(read_json(cell_root / "runtime_inputs" / "memory_config.json")) != expected_memory_config():
            raise RuntimeError(f"Memory config differs: {cell_name}")
        if sha256_file(cell_root / "runtime_inputs" / "questions.json") != question["runtime_questions_sha256"]:
            raise RuntimeError(f"Task reference differs: {cell_name}")
        if sha256_file(cell_root / "runtime_inputs" / "haystack.json") != question["runtime_haystack_sha256"]:
            raise RuntimeError(f"Haystack reference differs: {cell_name}")

        runtime_questions = read_json(cell_root / "runtime_inputs" / "questions.json")
        expected_prompt = build_aligned_query_prompt(
            _question_text(runtime_questions[0]["question"]),
            native_query_prompt=native_prompt,
            native_instruction=native_instruction,
        )
        prompt_paths = sorted(cell_root.glob("query_traces/*/attempt_*/prompt.md"))
        if len(prompt_paths) != 1 or prompt_paths[0].read_text(encoding="utf-8") != expected_prompt:
            raise RuntimeError(f"Rendered prompt differs: {cell_name}")
        prompt_hash = sha256_file(prompt_paths[0])
        if prompt_hash != question["rendered_prompt_sha256"]:
            raise RuntimeError(f"Rendered prompt hash differs from attempt 1: {cell_name}")
        prompt_hashes[question_id] = prompt_hash

        reader = receipt.get("reader_metadata") or {}
        if reader.get("provider") != selection["reader_provider"] or reader.get("response_model") != selection["reader_model"]:
            raise RuntimeError(f"Reader route differs: {cell_name}")
        expected_judges = 1 if question["eval_name"].startswith("llm_") else 0
        if len(receipt["evaluator_receipts"]) != expected_judges:
            raise RuntimeError(f"Evaluator policy differs: {cell_name}")
        for evaluator in receipt["evaluator_receipts"]:
            if not str(evaluator.get("response_model", "")).startswith(selection["evaluator_model"]):
                raise RuntimeError(f"Evaluator model differs: {cell_name}")

        for relative, expected_hash in receipt["artifact_hashes"].items():
            path = cell_root / relative
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Artifact hash differs: {path}")
            artifact_files += 1
            artifact_bytes += path.stat().st_size

    run_hashes = hash_tree(args.run_root)
    write_json(
        args.evidence_root / "raw_artifact_manifest.json",
        {"run_root": str(args.run_root), "run_root_hashes": run_hashes},
    )
    return {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "passed",
        "coverage": {"question_ids": EXPECTED_IDS, "cells": 2, "native_cells_run": 0, "concurrency": 1},
        "identity": {
            "selection_sha256": sha256_file(args.selection),
            "frozen_source_commit": FROZEN_COMMIT,
            "frozen_source_sha256": preflight_receipt["frozen_source_sha256"],
            "official_revision": preflight_receipt["official_revision"],
            "official_patch_sha256": preflight_receipt["official_patch_sha256"],
            "prepared_data_manifest_sha256": preflight_receipt["prepared_data_manifest_sha256"],
            "rendered_prompt_sha256_by_question": prompt_hashes,
        },
        "retries": {
            "whole_cell_retries": 0,
            "observed_query_attempts": 2,
            "observed_output_retries": 0,
            "query_http_retries_configured": 0,
        },
        "routes": {
            "query": selection["query_api"],
            "reader": selection["reader_api"],
            "reader_provider": selection["reader_provider"],
            "evaluator": selection["evaluator_api"],
        },
        "receipts": {"query_response_receipts_checked": response_receipts_checked, "scored_records": 2},
        "artifacts": {
            "cell_artifact_files_verified": artifact_files,
            "cell_artifact_bytes_verified": artifact_bytes,
            "raw_manifest_files": len(run_hashes),
        },
        "ripgrep_runtime": preflight_receipt["ripgrep_runtime"],
        "secret_boundary": secret_scan([args.run_root, args.evidence_root]),
    }


def write_results(path: Path, comparison: dict[str, Any], costs: dict[str, float]) -> None:
    lines = [
        "# LongMemEval native-prompt alignment variance check",
        "",
        "Two additional ThinHarness stochastic replicates. No native cells were run.",
        "",
        "| Question | Evidence | Score | Requests | Tools | Input | Output | Reasoning | Search failures | Latency | Query cost | Total cost |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        for label, key in (
            ("native", "original_native"),
            ("aligned attempt 1", "aligned_attempt_1"),
            ("aligned replicate 2", "aligned_replicate_2"),
        ):
            value = row[key]
            lines.append(
                f"| {row['question_id']} | {label} | {value['score']:.0f} | {value['requests']} | "
                f"{value['tool_calls']} | {value['input_tokens']:,} | {value['output_tokens']:,} | "
                f"{value['reasoning_output_tokens']:,} | {value['search_failures']} | "
                f"{value['query_latency_seconds']:.2f}s | {value['query_api_equivalent_usd']:.8f} USD | "
                f"{value['total_api_equivalent_usd']:.8f} USD |"
            )
    lines.extend(
        [
            "",
            f"Recovered: {comparison['recovered']}/2.",
            f"New query API-equivalent cost: {costs['query_api_equivalent']:.8f} USD.",
            f"New reader API-equivalent cost: {costs['reader_api_equivalent']:.8f} USD.",
            f"New evaluator API-equivalent cost: {costs['evaluator_api_equivalent']:.8f} USD.",
            f"New total API-equivalent cost: {costs['total_api_equivalent']:.8f} USD.",
            "",
            "These two outcomes estimate stochastic variance only. They do not replace prior cells or establish a stable regression rate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--alignment-selection", type=Path, required=True)
    parser.add_argument("--alignment-validation", type=Path, required=True)
    parser.add_argument("--alignment-receipts", type=Path, required=True)
    parser.add_argument("--alignment-run-root", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--rg-bin", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    for field in (
        "repo_root",
        "official_root",
        "prepared_root",
        "selection",
        "alignment_selection",
        "alignment_validation",
        "alignment_receipts",
        "alignment_run_root",
        "original_run_root",
        "patch",
        "rg_bin",
        "run_root",
        "evidence_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    configure_ripgrep(args.rg_bin)
    args.run_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    preflight_receipt = preflight(args, selection)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    write_json(args.run_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "preflight.json", preflight_receipt)
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")

    receipts: list[dict[str, Any]] = []
    progress_path = args.run_root / "progress.jsonl"
    for question in selection["questions"]:
        cell_name = f"variance-replicate-2-{question['order']:02d}-{question['question_id']}-thinharness"
        output_dir = args.run_root / "cells" / cell_name
        receipt_path = output_dir / "cell_receipt.json"
        if receipt_path.exists():
            receipts.append(read_json(receipt_path))
            print(f"CHECKPOINT skip completed {cell_name}", flush=True)
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"Partial cell exists without receipt; zero-retry policy forbids relaunch: {cell_name}")
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
            prepared["domains"][question["domain"]]["data_root"],
            "--output-dir",
            str(output_dir),
            "--harness",
            "thinharness",
            "--question-id",
            question["question_id"],
            "--domain",
            question["domain"],
        ]
        process_log = args.run_root / "logs" / f"{cell_name}.log"
        returncode, _output = run_subprocess(command, process_log, args.repo_root)
        receipt = build_receipt(
            output_dir=output_dir,
            harness_name="thinharness",
            question=question,
            process_returncode=returncode,
            process_log=process_log,
            evaluator_receipts_path=evaluator_path,
        )
        receipt["replicate_label"] = selection["experiment_id"]
        receipt["replicate_kind"] = "native_prompt_alignment_variance_check"
        receipt["replicate_number"] = 2
        receipt["replaces_original_cell"] = False
        write_json(receipt_path, receipt)
        receipts.append(receipt)
        with progress_path.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "completed_at_utc": receipt["completed_at_utc"],
                        "cell": cell_name,
                        "score": receipt["score"],
                        "final_outcome": receipt["final_outcome"],
                        "receipt_sha256": sha256_file(receipt_path),
                    }
                )
                + "\n"
            )
        print(f"CELL_COMPLETE {cell_name} outcome={receipt['final_outcome']} score={receipt['score']}", flush=True)
        if receipt["final_outcome"] != "scored":
            raise RuntimeError(f"Cell failed and zero-retry policy stops execution: {cell_name}")
        if len(receipt["query_attempts"]) != 1:
            raise RuntimeError(f"Cell used more than one query attempt: {cell_name}")

    comparison = compile_comparison(args, selection, receipts)
    costs = {
        "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in receipts),
        "reader_provider_reported": sum(float(row["costs_usd"]["reader_provider_reported"] or 0) for row in receipts),
        "reader_api_equivalent": sum(float(row["costs_usd"]["reader_api_equivalent"]) for row in receipts),
        "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in receipts),
        "total_api_equivalent": sum(float(row["costs_usd"]["total_api_equivalent"]) for row in receipts),
    }
    compact_receipts = [
        {key: value for key, value in receipt.items() if key not in {"artifact_hashes", "query_attempts", "response_raw"}}
        for receipt in receipts
    ]
    write_json(args.evidence_root / "cell_receipts.json", compact_receipts)
    write_json(args.evidence_root / "comparison.json", comparison)
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "status": "completed",
        "completed_cells": 2,
        "target_cells": 2,
        "native_cells_run": 0,
        "recovered": comparison["recovered"],
        "cost_totals_usd": costs,
    }
    write_json(args.evidence_root / "final_summary.json", final)
    write_results(args.evidence_root / "RESULTS.md", comparison, costs)
    write_json(args.run_root / "COMPLETED.json", final)
    validation = validate_completed(args, selection, receipts, preflight_receipt)
    write_json(args.evidence_root / "validation.json", validation)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
