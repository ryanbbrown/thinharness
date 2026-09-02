#!/usr/bin/env python3
"""Run the frozen clean ten-question native versus ThinHarness comparison."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.longmemeval_v2.orchestrate import (
    OFFICIAL_REVISION,
    build_receipt,
    detect_credit_exhaustion,
    hash_tree,
    normalize_patch_blank_context,
    run_subprocess,
    sha256_file,
    summarize_pairs,
    write_json,
)
from benchmarks.longmemeval_v2.prompt_alignment import build_aligned_query_prompt, prompt_diff_artifact
from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime
from benchmarks.longmemeval_v2.run_prompt_alignment import configure_ripgrep, data_access_audit
from benchmarks.longmemeval_v2.validate_wave import secret_scan

TARGET_QUESTIONS = 10
TARGET_CELLS = 20
QUERY_ATTEMPTS = 1
OUTPUT_RETRIES = 0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def extract_native_query_prompt(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "DEFAULT_QUERY_PROMPT" for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"Could not extract DEFAULT_QUERY_PROMPT from {path}")


def native_ripgrep_probe(run_root: Path) -> dict[str, Any]:
    probe_root = run_root / "native_rg_preflight"
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_file = probe_root / "probe.txt"
    probe_file.write_text("native-probe-hit\n", encoding="utf-8")
    script = "set -euo pipefail; command -v rg; rg --fixed-strings --count native-probe-hit probe.txt"
    process = subprocess.run(
        ["/bin/bash", "-c", script],
        cwd=probe_root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if process.returncode != 0 or "native-probe-hit" in process.stderr or not process.stdout.rstrip().endswith("1"):
        raise RuntimeError(f"Native shell ripgrep probe failed: {process.stderr.strip()}")
    resolved = process.stdout.splitlines()[0]
    return {
        "runtime": "native shell",
        "rg_path": str(Path(resolved).resolve()),
        "command_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "match_count": 1,
    }


def validate_selection(selection: dict[str, Any]) -> None:
    questions = selection.get("questions", [])
    if len(questions) != TARGET_QUESTIONS or len({row["question_id"] for row in questions}) != TARGET_QUESTIONS:
        raise RuntimeError("Clean-pair selection must contain ten unique questions")
    if {row["question_type"] for row in questions} != {
        "static-environment",
        "static-environment-abs",
        "dynamic-environment",
        "dynamic-environment-abs",
        "procedure",
        "procedure-abs",
        "errors-gotchas",
    }:
        raise RuntimeError("Clean-pair selection does not cover all seven question types")
    if sum(row["domain"] == "web" for row in questions) != 5 or sum(
        row["domain"] == "enterprise" for row in questions
    ) != 5:
        raise RuntimeError("Clean-pair selection must be balanced by domain")
    if sum(not row["eval_name"].startswith("llm_") for row in questions) != 5:
        raise RuntimeError("Clean-pair selection must have five deterministic questions")
    image_domains = [row["domain"] for row in questions if row["has_question_image"]]
    if image_domains != ["web", "enterprise"]:
        raise RuntimeError("Clean-pair selection must cover both image paths")


def source_hashes(repo_root: Path, official_root: Path) -> dict[str, str]:
    paths = {
        "thinharness_memory": repo_root / "benchmarks/longmemeval_v2/memory.py",
        "thinharness_corpus": repo_root / "benchmarks/longmemeval_v2/corpus.py",
        "thinharness_prompt_alignment": repo_root / "benchmarks/longmemeval_v2/prompt_alignment.py",
        "thinharness_filesystem_tools": repo_root / "thinharness/tools/filesystem.py",
        "thinharness_openai_provider": repo_root / "thinharness/providers/openai.py",
        "official_agentrunbook": official_root / "memory_modules/agentrunbook_c.py",
        "official_agentrunbook_v2": official_root / "memory_modules/agentrunbook_c_v2.py",
        "official_openai_agents_runner": official_root / "memory_modules/oai_agents_sdk.py",
        "official_instruction": official_root / "memory_modules/assets/agentrunbook_c/INSTRUCTION.md",
        "official_inspector": official_root / "memory_modules/assets/agentrunbook_c/scripts/inspect_trajectory.py",
        "official_harness": official_root / "evaluation/harness.py",
        "official_evaluator": official_root / "evaluation/qa_eval_metrics.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def preflight(
    args: argparse.Namespace,
    selection: dict[str, Any],
    estimate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    validate_selection(selection)
    if config["selection"]["sha256"] != sha256_file(args.selection):
        raise RuntimeError("Frozen config selection hash differs")
    if config["selection"]["ids"] != [row["question_id"] for row in selection["questions"]]:
        raise RuntimeError("Frozen config question order differs")
    if config["prompt"]["diff_sha256"] != sha256_file(args.prompt_diff):
        raise RuntimeError("Frozen config prompt-diff hash differs")
    if config["cost"]["estimate_sha256"] != sha256_file(args.cost_estimate):
        raise RuntimeError("Frozen config cost-estimate hash differs")
    runtime_modules = (
        "openai",
        "agents",
        "PIL",
        "transformers",
        "evaluation.harness",
        "evaluation.run_eval",
        "memory_modules.agentrunbook_c_v2",
    )
    if str(args.official_root) not in sys.path:
        sys.path.insert(0, str(args.official_root))
    runtime_imports = {}
    for module_name in runtime_modules:
        module = importlib.import_module(module_name)
        runtime_imports[module_name] = str(getattr(module, "__version__", "imported"))

    if config["query"] != {
        "model": "gpt-5.6-luna",
        "reasoning_effort": "xhigh",
        "api": "https://api.openai.com/v1",
        "online_learning": False,
        "attempts_per_harness_question": QUERY_ATTEMPTS,
        "http_retries": 0,
        "output_retries": OUTPUT_RETRIES,
        "max_model_requests_or_turns": 30,
        "execution_order": "native then thinharness for each selected question",
    }:
        raise RuntimeError("Frozen query configuration differs")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    if command_output(["git", "status", "--porcelain"], cwd=args.repo_root):
        raise RuntimeError("ThinHarness repository must be clean before paid execution")
    official_revision = command_output(["git", "rev-parse", "HEAD"], cwd=args.official_root)
    if config["harnesses"]["official_revision"] != OFFICIAL_REVISION:
        raise RuntimeError("Frozen official revision differs")
    if official_revision != OFFICIAL_REVISION:
        raise RuntimeError(f"Official harness revision differs: {official_revision}")
    official_diff = subprocess.check_output(
        ["git", "diff", "--", "evaluation/harness.py", "evaluation/run_eval.py", "evaluation/qa_eval_metrics.py"],
        cwd=args.official_root,
    )
    patch_bytes = args.patch.read_bytes()
    if normalize_patch_blank_context(official_diff) != patch_bytes:
        raise RuntimeError("Official instrumentation differs from the frozen patch")
    if config["harnesses"]["official_patch_sha256"] != hashlib.sha256(patch_bytes).hexdigest():
        raise RuntimeError("Frozen official patch hash differs")
    if sha256_file(args.official_root / "uv.lock") != config["official_runtime"]["uv_lock_sha256"]:
        raise RuntimeError("Frozen official runtime lock differs")

    current_sources = {
        relative: sha256_file(args.repo_root / relative) for relative in config["source_hashes"]
    }
    if current_sources != config["source_hashes"]:
        raise RuntimeError("Frozen ThinHarness source hashes differ")
    current_official_sources = {
        relative: sha256_file(args.official_root / relative) for relative in config["official_source_hashes"]
    }
    if current_official_sources != config["official_source_hashes"]:
        raise RuntimeError("Frozen official source hashes differ")

    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")
    if prepared["selection_sha256"] != sha256_file(args.selection):
        raise RuntimeError("Prepared data does not match the frozen selection")
    if selection["source_hashes"]["questions.jsonl"] != prepared["source_questions_sha256"]:
        raise RuntimeError("Prepared questions hash differs")
    for domain in ("web", "enterprise"):
        domain_root = Path(prepared["domains"][domain]["data_root"])
        if sha256_file(domain_root / "trajectories.jsonl") != prepared["domains"][domain]["trajectories_jsonl_sha256"]:
            raise RuntimeError(f"Prepared {domain} corpus differs")

    native_prompt = extract_native_query_prompt(args.official_root / "memory_modules/agentrunbook_c.py")
    native_instruction = (
        args.official_root / "memory_modules/assets/agentrunbook_c/INSTRUCTION.md"
    ).read_text(encoding="utf-8")
    generated_diff = prompt_diff_artifact(native_prompt, native_instruction)
    frozen_diff = read_json(args.prompt_diff)
    if generated_diff != frozen_diff:
        raise RuntimeError("Frozen prompt diff differs from the official sources")

    for item in estimate["source_artifacts"].values():
        if sha256_file(args.repo_root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Cost-estimate source differs: {item['path']}")
    if estimate["estimated_hard_cap_usage_usd"] > estimate["hard_cap_usd"]:
        raise RuntimeError("Conservative estimate exceeds the hard cap")

    thin_rg = verify_ripgrep_runtime()
    native_rg = native_ripgrep_probe(args.run_root)
    expected_rg = args.rg_bin.resolve()
    if Path(thin_rg["rg_path"]).resolve() != expected_rg or Path(native_rg["rg_path"]).resolve() != expected_rg:
        raise RuntimeError("The two harnesses did not select the same frozen ripgrep binary")

    return {
        "schema_version": 1,
        "checked_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "selection_sha256": sha256_file(args.selection),
        "config_sha256": sha256_file(args.config),
        "cost_estimate_sha256": sha256_file(args.cost_estimate),
        "prompt_diff_sha256": sha256_file(args.prompt_diff),
        "prepared_data_manifest_sha256": sha256_file(args.prepared_root / "prepared_data_manifest.json"),
        "thinharness_revision": command_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root),
        "official_revision": official_revision,
        "official_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "source_hashes": source_hashes(args.repo_root, args.official_root),
        "runtime_imports": runtime_imports,
        "prompt_equivalence": generated_diff,
        "ripgrep": {"native": native_rg, "thinharness": thin_rg},
        "cost": {
            "estimate_usd": estimate["estimated_hard_cap_usage_usd"],
            "hard_cap_usd": estimate["hard_cap_usd"],
            "per_cell_reserve_usd": estimate["per_cell_reserve_usd"],
        },
        "query_policy": {
            "attempts_per_harness_question": QUERY_ATTEMPTS,
            "output_retries": OUTPUT_RETRIES,
            "query_http_retries": 0,
            "max_model_requests": 30,
            "native_max_turns": 30,
            "thinharness_max_tool_calls": 128,
        },
        "models": {
            "query": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
            "reader": "qwen/qwen3.5-9b via OpenRouter Parasail",
            "evaluator": "gpt-5.2 medium via direct OpenAI",
        },
        "api_key_boundaries": {
            "OPENAI_API_KEY": "present; value not read or persisted",
            "OPENROUTER_API_KEY": "present; value not read or persisted",
        },
    }


def cell_sequence(selection: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    cells = []
    for question in selection["questions"]:
        for harness in ("native", "thinharness"):
            name = f"{question['order']:02d}-{question['question_id']}-{harness}"
            cells.append((question, harness, name))
    return cells


def projected_total(
    receipts: list[dict[str, Any]],
    pending: list[tuple[dict[str, Any], str, str]],
    reserves: dict[str, float],
) -> float:
    spent = sum(float(row["costs_usd"]["total_api_equivalent"]) for row in receipts)
    return spent + sum(float(reserves[harness]) for _question, harness, _name in pending)


def trace_metrics(run_root: Path, cell_name: str, harness: str) -> dict[str, Any]:
    cell_root = run_root / "cells" / cell_name
    result_chars = 0
    search_calls = 0
    failed_tool_calls = 0
    valid_spans = invalid_spans = span_states = 0
    accessed_paths: set[str] = set()
    if harness == "native":
        for path in cell_root.glob("query_traces/*/attempt_*/events.json"):
            events = read_json(path)
            for call in events.get("tool_calls", []):
                response = call.get("tool_response", {})
                result_chars += len(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
                command = str(call.get("command", ""))
                if re.search(r"(?:^|[;&|\s])rg(?:\s|$)", command) or "inspect_trajectory.py" in command and "--match" in command:
                    search_calls += 1
                response_text = json.dumps(response)
                if call.get("timed_out") or call.get("returncode") not in (None, 0) or "command not found" in response_text:
                    failed_tool_calls += 1
                accessed_paths.update(re.findall(r"trajectories/[A-Za-z0-9_-]+/(?:trajectory\.json|screenshots/\S+)", command))
        audit: dict[str, Any] = {"accessed_paths": sorted(accessed_paths)}
    else:
        for path in cell_root.glob("query_traces/*/attempt_*/tool_calls.json"):
            calls = read_json(path)
            for call in calls:
                name = call.get("call", {}).get("name")
                if name in {"search", "jsonl_search"}:
                    search_calls += 1
                result = call.get("result", {})
                if not result.get("ok", False):
                    failed_tool_calls += 1
                output = call.get("output")
                if not isinstance(output, str):
                    output = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                result_chars += len(output)
        audit = data_access_audit(cell_root)

    summaries = sorted(cell_root.glob("query_traces/*/attempt_*/summary.json"))
    for path in summaries:
        summary = read_json(path)
        valid = summary.get("trajectory_spans_valid", [])
        invalid = summary.get("trajectory_spans_invalid", [])
        valid_spans += len(valid)
        invalid_spans += len(invalid)
        for span in valid:
            span_states += int(span["end_state_index"]) - int(span["start_state_index"]) + 1
    return {
        "search_calls": search_calls,
        "failed_tool_calls": failed_tool_calls,
        "tool_result_chars": result_chars,
        "valid_spans": valid_spans,
        "invalid_spans": invalid_spans,
        "span_states": span_states,
        "retrieval_audit": audit,
    }


def wilson(successes: float, total: int) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def bootstrap_cost_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or any(row["native_query_cost_usd"] <= 0 for row in rows):
        return {"ratio": None, "interval_95": None}
    ratio = sum(row["thinharness_query_cost_usd"] for row in rows) / sum(
        row["native_query_cost_usd"] for row in rows
    )
    rng = random.Random(20260831)
    samples = []
    for _ in range(100_000):
        drawn = [rows[rng.randrange(len(rows))] for _item in rows]
        native = sum(row["native_query_cost_usd"] for row in drawn)
        thin = sum(row["thinharness_query_cost_usd"] for row in drawn)
        if native > 0:
            samples.append(thin / native)
    samples.sort()
    return {
        "ratio": ratio,
        "method": "100,000 paired bootstrap resamples with frozen seed 20260831",
        "interval_95": [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]],
    }


def compile_results(
    run_root: Path,
    selection: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {(row["question_id"], row["harness"]): row for row in receipts}
    rows = []
    for question in selection["questions"]:
        question_id = question["question_id"]
        native = by_key.get((question_id, "native"))
        thin = by_key.get((question_id, "thinharness"))
        row: dict[str, Any] = {**question}
        for harness, receipt in (("native", native), ("thinharness", thin)):
            if receipt is None:
                row[harness] = None
                continue
            name = receipt["cell_name"]
            trace = trace_metrics(run_root, name, harness)
            row[harness] = {
                "score": receipt["score"],
                "outcome": receipt["final_outcome"],
                "query_usage": receipt["query_usage"],
                "query_cost_usd": receipt["costs_usd"]["query_api_equivalent"],
                "reader_cost_usd": receipt["costs_usd"]["reader_api_equivalent"],
                "evaluator_cost_usd": receipt["costs_usd"]["evaluator_api_equivalent"],
                "total_cost_usd": receipt["costs_usd"]["total_api_equivalent"],
                "latency_seconds": receipt["memory_query_duration_seconds"],
                "evidence_context_tokens": receipt["memory_context_token_count"],
                **trace,
            }
        if native is not None and thin is not None:
            row["native_query_cost_usd"] = native["costs_usd"]["query_api_equivalent"]
            row["thinharness_query_cost_usd"] = thin["costs_usd"]["query_api_equivalent"]
        rows.append(row)

    complete = [row for row in rows if row.get("native") and row.get("thinharness")]
    harness_totals: dict[str, Any] = {}
    for harness in ("native", "thinharness"):
        values = [row[harness] for row in complete]
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
        score = sum(float(value["score"] or 0) for value in values)
        harness_totals[harness] = {
            "cells": len(values),
            "correct": score,
            "accuracy": score / len(values) if values else None,
            "accuracy_wilson_95": wilson(score, len(values)),
            "query_usage": {
                field: sum(int(value["query_usage"][field]) for value in values) for field in usage_fields
            },
            "query_cost_usd": sum(float(value["query_cost_usd"]) for value in values),
            "reader_cost_usd": sum(float(value["reader_cost_usd"]) for value in values),
            "evaluator_cost_usd": sum(float(value["evaluator_cost_usd"]) for value in values),
            "total_cost_usd": sum(float(value["total_cost_usd"]) for value in values),
            "query_latency_seconds": sum(float(value["latency_seconds"] or 0) for value in values),
            "search_calls": sum(int(value["search_calls"]) for value in values),
            "failed_tool_calls": sum(int(value["failed_tool_calls"]) for value in values),
            "tool_result_chars": sum(int(value["tool_result_chars"]) for value in values),
            "valid_spans": sum(int(value["valid_spans"]) for value in values),
            "invalid_spans": sum(int(value["invalid_spans"]) for value in values),
            "span_states": sum(int(value["span_states"]) for value in values),
            "evidence_context_tokens": sum(int(value["evidence_context_tokens"] or 0) for value in values),
        }
    paired_rows = [row for row in complete if "native_query_cost_usd" in row]
    return {
        "schema_version": 1,
        "compiled_at_utc": utc_now(),
        "rows": rows,
        "totals": harness_totals,
        "query_cost_ratio": bootstrap_cost_ratio(paired_rows),
        "paired_scores": summarize_pairs(receipts),
    }


def validate_completed(
    args: argparse.Namespace,
    selection: dict[str, Any],
    receipts: list[dict[str, Any]],
    preflight_receipt: dict[str, Any],
) -> dict[str, Any]:
    if len(receipts) != TARGET_CELLS:
        raise RuntimeError(f"Expected {TARGET_CELLS} receipts, got {len(receipts)}")
    if len({(row["question_id"], row["harness"]) for row in receipts}) != TARGET_CELLS:
        raise RuntimeError("Cell identities are not unique")
    attempts = list(args.run_root.glob("cells/*/query_traces/*/attempt_*/summary.json"))
    if len(attempts) != TARGET_CELLS or any("attempt_001" not in str(path) for path in attempts):
        raise RuntimeError("Each query cell must preserve exactly one attempt_001 summary")
    if list(args.run_root.glob("cells/*/query_traces/*/attempt_002")):
        raise RuntimeError("Unexpected second query attempt")

    native_prompt = extract_native_query_prompt(args.official_root / "memory_modules/agentrunbook_c.py")
    instruction_path = args.official_root / "memory_modules/assets/agentrunbook_c/INSTRUCTION.md"
    native_instruction = instruction_path.read_text(encoding="utf-8")
    prompt_checks = 0
    artifact_files = 0
    artifact_bytes = 0
    for receipt in receipts:
        cell_root = args.run_root / "cells" / receipt["cell_name"]
        for relative, expected_hash in receipt["artifact_hashes"].items():
            path = cell_root / relative
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Cell artifact hash differs: {path}")
            artifact_files += 1
            artifact_bytes += path.stat().st_size
        if receipt["harness"] == "thinharness":
            runtime_questions = read_json(cell_root / "runtime_inputs/questions.json")
            question_text = runtime_questions[0]["question"]
            if isinstance(question_text, list):
                question_text = " ".join(str(item) for item in question_text)
            expected = build_aligned_query_prompt(
                str(question_text),
                native_query_prompt=native_prompt,
                native_instruction=native_instruction,
            )
            prompt_paths = list(cell_root.glob("query_traces/*/attempt_001/prompt.md"))
            if len(prompt_paths) != 1 or prompt_paths[0].read_text(encoding="utf-8") != expected:
                raise RuntimeError(f"Thin prompt differs: {receipt['cell_name']}")
            prompt_checks += 1
        else:
            sandbox_instructions = list(cell_root.glob("query_traces/*/attempt_001/sandbox/INSTRUCTION.md"))
            if len(sandbox_instructions) != 1 or sandbox_instructions[0].read_text(encoding="utf-8") != native_instruction:
                raise RuntimeError(f"Native instruction differs: {receipt['cell_name']}")
            prompt_checks += 1
        expected_evaluators = (
            1 if receipt["scoring_path"] == "llm_judged" and receipt["final_outcome"] == "scored" else 0
        )
        if len(receipt["evaluator_receipts"]) != expected_evaluators:
            raise RuntimeError(f"Evaluator receipt count differs: {receipt['cell_name']}")

    run_hashes = hash_tree(args.run_root)
    write_json(args.evidence_root / "raw_artifact_manifest.json", {"run_root": str(args.run_root), "hashes": run_hashes})
    total_cost = sum(float(row["costs_usd"]["total_api_equivalent"]) for row in receipts)
    if total_cost > float(read_json(args.cost_estimate)["hard_cap_usd"]):
        raise RuntimeError("Completed spend exceeded the hard cap")
    return {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "passed",
        "coverage": {
            "questions": TARGET_QUESTIONS,
            "cells": TARGET_CELLS,
            "native": sum(row["harness"] == "native" for row in receipts),
            "thinharness": sum(row["harness"] == "thinharness" for row in receipts),
            "deterministic": sum(row["scoring_path"] == "deterministic" for row in receipts),
            "llm_judged": sum(row["scoring_path"] == "llm_judged" for row in receipts),
        },
        "attempts": {"query_summaries": len(attempts), "all_attempt_001": True, "second_attempts": 0},
        "prompt_checks": prompt_checks,
        "preflight_hash": sha256_file(args.evidence_root / "preflight.json"),
        "source_hashes": preflight_receipt["source_hashes"],
        "artifact_files": artifact_files,
        "artifact_bytes": artifact_bytes,
        "run_manifest_files": len(run_hashes),
        "cost_usd": total_cost,
        "hard_cap_usd": read_json(args.cost_estimate)["hard_cap_usd"],
        "secret_boundary": secret_scan([args.run_root, args.evidence_root]),
    }


def write_results(path: Path, results: dict[str, Any], final: dict[str, Any]) -> None:
    totals = results["totals"]
    native = totals["native"]
    thin = totals["thinharness"]
    ratio = results["query_cost_ratio"]
    if ratio["ratio"] is None:
        ratio_text = "ratio unavailable because no query usage was recorded"
    else:
        ratio_text = (
            f"ratio {ratio['ratio']:.3f}x "
            f"(paired bootstrap 95% interval {ratio['interval_95'][0]:.3f}x to {ratio['interval_95'][1]:.3f}x)"
        )
    lines = [
        "# LongMemEval clean paired comparison",
        "",
        f"Fresh paired accuracy: native {native['correct']:.0f}/10; ThinHarness {thin['correct']:.0f}/10.",
        f"Fresh query API-equivalent cost: native {native['query_cost_usd']:.8f} USD; "
        f"ThinHarness {thin['query_cost_usd']:.8f} USD; {ratio_text}.",
        "",
        "| Question | Type | Native score | Thin score | Native query cost | Thin query cost | Native / Thin input | Native / Thin tools |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results["rows"]:
        native_row = row["native"]
        thin_row = row["thinharness"]
        lines.append(
            f"| {row['question_id']} | {row['question_type']} | {native_row['score']} | {thin_row['score']} | "
            f"{native_row['query_cost_usd']:.8f} USD | {thin_row['query_cost_usd']:.8f} USD | "
            f"{native_row['query_usage']['input_tokens']:,} / {thin_row['query_usage']['input_tokens']:,} | "
            f"{native_row['query_usage']['tool_calls']} / {thin_row['query_usage']['tool_calls']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate telemetry",
            "",
            "| Metric | Native | ThinHarness |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, key in (
        ("Input tokens", "input_tokens"),
        ("Cached input tokens", "cached_input_tokens"),
        ("Ordinary input tokens", "ordinary_input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Reasoning tokens", "reasoning_output_tokens"),
        ("Requests", "requests"),
        ("Tool calls", "tool_calls"),
    ):
        lines.append(f"| {label} | {native['query_usage'][key]:,} | {thin['query_usage'][key]:,} |")
    for label, key in (
        ("Search calls", "search_calls"),
        ("Failed tool calls", "failed_tool_calls"),
        ("Tool-result characters", "tool_result_chars"),
        ("Query wall time", "query_latency_seconds"),
        ("Evidence context tokens", "evidence_context_tokens"),
        ("Valid evidence spans", "valid_spans"),
        ("Invalid evidence spans", "invalid_spans"),
        ("Evidence states", "span_states"),
    ):
        lines.append(f"| {label} | {native[key]:,.2f} | {thin[key]:,.2f} |")
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"- Query: {final['cost_totals_usd']['query_api_equivalent']:.8f} USD API-equivalent.",
            f"- Reader: {final['cost_totals_usd']['reader_api_equivalent']:.8f} USD API-equivalent; "
            f"{final['cost_totals_usd']['reader_provider_reported']:.8f} USD provider-reported.",
            f"- Evaluator: {final['cost_totals_usd']['evaluator_api_equivalent']:.8f} USD API-equivalent.",
            f"- Total: {final['cost_totals_usd']['total_api_equivalent']:.8f} USD API-equivalent, below the 2 USD cap.",
            "- OpenAI responses report tokens rather than billed dollars; query and evaluator costs use the frozen rates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cost-estimate", type=Path, required=True)
    parser.add_argument("--prompt-diff", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--rg-bin", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for name in (
        "repo_root",
        "official_root",
        "prepared_root",
        "selection",
        "config",
        "cost_estimate",
        "prompt_diff",
        "patch",
        "rg_bin",
        "run_root",
        "evidence_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    configure_ripgrep(args.rg_bin)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    config = read_json(args.config)
    estimate = read_json(args.cost_estimate)
    preflight_receipt = preflight(args, selection, estimate, config)
    write_json(args.run_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "preflight.json", preflight_receipt)
    if args.preflight_only:
        print(json.dumps(preflight_receipt, indent=2), flush=True)
        return
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")

    receipts: list[dict[str, Any]] = []
    progress_path = args.run_root / "progress.jsonl"
    sequence = cell_sequence(selection)
    status = "completed"
    for index, (question, harness, cell_name) in enumerate(sequence):
        receipt_path = args.run_root / "cells" / cell_name / "cell_receipt.json"
        if receipt_path.exists():
            receipts.append(read_json(receipt_path))
            print(f"CHECKPOINT skip completed {cell_name}", flush=True)
            continue
        pending = sequence[index:]
        projected = projected_total(receipts, pending, estimate["per_cell_reserve_usd"])
        if projected > float(estimate["hard_cap_usd"]):
            status = "cost_cap_stop"
            print(f"COST_CAP_STOP projected={projected:.8f}", flush=True)
            break
        print(f"CELL_START {cell_name} projected_total={projected:.8f}", flush=True)
        output_dir = receipt_path.parent
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
            harness,
            "--question-id",
            question["question_id"],
            "--domain",
            question["domain"],
            "--query-attempts",
            str(QUERY_ATTEMPTS),
            "--output-retries",
            str(OUTPUT_RETRIES),
        ]
        process_log = args.run_root / "logs" / f"{cell_name}.log"
        returncode, output = run_subprocess(command, process_log, args.repo_root)
        receipt = build_receipt(
            output_dir=output_dir,
            harness_name=harness,
            question=question,
            process_returncode=returncode,
            process_log=process_log,
            evaluator_receipts_path=evaluator_path,
        )
        receipt["cell_name"] = cell_name
        receipt["experiment_id"] = selection["experiment_id"]
        receipt["query_attempt_limit"] = QUERY_ATTEMPTS
        receipt["output_retry_limit"] = OUTPUT_RETRIES
        write_json(receipt_path, receipt)
        receipts.append(receipt)
        with progress_path.open("a", encoding="utf-8") as ledger:
            ledger.write(
                json.dumps(
                    {
                        "completed_at_utc": receipt["completed_at_utc"],
                        "cell": cell_name,
                        "outcome": receipt["final_outcome"],
                        "score": receipt["score"],
                        "total_cost_usd": receipt["costs_usd"]["total_api_equivalent"],
                        "receipt_sha256": sha256_file(receipt_path),
                    }
                )
                + "\n"
            )
        write_json(
            args.run_root / "progress.json",
            {
                "updated_at_utc": utc_now(),
                "completed_cells": len(receipts),
                "target_cells": TARGET_CELLS,
                "last_cell": cell_name,
                "accumulated_cost_usd": sum(
                    float(row["costs_usd"]["total_api_equivalent"]) for row in receipts
                ),
            },
        )
        print(
            f"CELL_COMPLETE {cell_name} outcome={receipt['final_outcome']} score={receipt['score']} "
            f"cost={receipt['costs_usd']['total_api_equivalent']:.8f}",
            flush=True,
        )
        if receipt["final_outcome"] == "infrastructure_error" and receipt["query_usage"]["requests"] == 0:
            raise RuntimeError(f"Pre-query infrastructure failure: {cell_name}; see {process_log}")
        if detect_credit_exhaustion(output, receipt):
            status = "credit_exhausted"
            print(f"CREDIT_EXHAUSTED after {cell_name}", flush=True)
            break

    costs = {
        "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in receipts),
        "reader_provider_reported": sum(float(row["costs_usd"]["reader_provider_reported"] or 0) for row in receipts),
        "reader_api_equivalent": sum(float(row["costs_usd"]["reader_api_equivalent"]) for row in receipts),
        "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in receipts),
        "total_api_equivalent": sum(float(row["costs_usd"]["total_api_equivalent"]) for row in receipts),
    }
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "status": status,
        "completed_cells": len(receipts),
        "target_cells": TARGET_CELLS,
        "native_cells": sum(row["harness"] == "native" for row in receipts),
        "thinharness_cells": sum(row["harness"] == "thinharness" for row in receipts),
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
    marker = "COMPLETED.json" if status == "completed" else f"{status.upper()}.json"
    write_json(args.run_root / marker, final)
    if status == "completed" and len(receipts) == TARGET_CELLS:
        results = compile_results(args.run_root, selection, receipts)
        write_json(args.evidence_root / "comparison.json", results)
        write_results(args.evidence_root / "RESULTS.md", results, final)
        validation = validate_completed(args, selection, receipts, preflight_receipt)
        write_json(args.evidence_root / "validation.json", validation)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
