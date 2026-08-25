#!/usr/bin/env python3
"""Run five ThinHarness-only native-prompt alignment diagnostic replicates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
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
    write_json,
)
from benchmarks.longmemeval_v2.prompt_alignment import (
    build_aligned_query_prompt,
    prompt_diff_artifact,
)
from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime
from benchmarks.longmemeval_v2.run_rg_diagnostics import metrics, search_diagnostics
from benchmarks.longmemeval_v2.validate_wave import secret_scan

DATA_FORMS = (
    "global_jsonl",
    "per_trajectory_jsonl",
    "raw_trajectory_json",
    "summaries",
    "spill_artifacts",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def configure_ripgrep(rg_bin: Path) -> None:
    resolved = rg_bin.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Configured rg is not executable: {resolved}")
    os.environ["PATH"] = os.pathsep.join([str(resolved.parent), os.getenv("PATH", "")])
    selected = shutil.which("rg")
    if selected is None or Path(selected).resolve() != resolved:
        raise RuntimeError(f"Runtime did not select configured rg: expected {resolved}, got {selected}")


def extract_native_query_prompt(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "DEFAULT_QUERY_PROMPT" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            break
        return value
    raise RuntimeError(f"Could not extract DEFAULT_QUERY_PROMPT from {path}")


def _top_cost_delta_ids(receipts: list[dict[str, Any]]) -> list[str]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in receipts:
        pairs.setdefault(row["question_id"], {})[row["harness"]] = row
    deltas = []
    for question_id, pair in pairs.items():
        native = pair["native"]
        thin = pair["thinharness"]
        delta = float(thin["costs_usd"]["query_api_equivalent"]) - float(
            native["costs_usd"]["query_api_equivalent"]
        )
        if delta > 0:
            deltas.append((delta, question_id))
    return [question_id for _delta, question_id in sorted(deltas, reverse=True)[:5]]


def _validate_frozen_baselines(selection: dict[str, Any], receipts: list[dict[str, Any]]) -> None:
    pairs = {(row["question_id"], row["harness"]): row for row in receipts}
    selected_ids = [row["question_id"] for row in selection["questions"]]
    if selected_ids != _top_cost_delta_ids(receipts):
        raise RuntimeError("Prompt-alignment selection is not the top five positive query-cost deltas")
    usage_fields = (
        "input_tokens",
        "cached_input_tokens",
        "ordinary_input_tokens",
        "output_tokens",
        "tool_calls",
    )
    for question in selection["questions"]:
        question_id = question["question_id"]
        for harness, baseline_key in (("native", "original_native"), ("thinharness", "original_thinharness")):
            receipt = pairs[(question_id, harness)]
            baseline = question[baseline_key]
            expected = {
                "score": receipt["score"],
                "query_cost_usd": receipt["costs_usd"]["query_api_equivalent"],
                **{field: receipt["query_usage"][field] for field in usage_fields},
                "model_requests": receipt["query_usage"]["requests"],
                "latency_seconds": receipt["memory_query_duration_seconds"],
                "reader_context_tokens": receipt["memory_context_token_count"],
            }
            for field, value in expected.items():
                if baseline[field] != value:
                    raise RuntimeError(f"Frozen {harness} baseline differs for {question_id}: {field}")
        expected_delta = (
            float(question["original_thinharness"]["query_cost_usd"])
            - float(question["original_native"]["query_cost_usd"])
        )
        if abs(float(question["query_cost_delta_usd"]) - expected_delta) > 1e-12:
            raise RuntimeError(f"Frozen query-cost delta differs for {question_id}")


def preflight(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    if len(selection.get("questions", [])) != 5:
        raise RuntimeError("Prompt-alignment selection must contain exactly five questions")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    if command_output(["git", "status", "--porcelain"], cwd=args.repo_root):
        raise RuntimeError("ThinHarness repository must be clean before paid execution")
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
    if sha256_file(args.original_receipts) != selection["original_cell_receipts_sha256"]:
        raise RuntimeError("Original full-wave receipts differ from the frozen selection")
    if sha256_file(args.original_selection) != selection["original_selection_sha256"]:
        raise RuntimeError("Original full-wave selection differs")
    if sha256_file(args.prompt_diff) != selection["prompt_diff_sha256"]:
        raise RuntimeError("Prompt-diff artifact differs from the frozen selection")
    originals = read_json(args.original_receipts)
    _validate_frozen_baselines(selection, originals)
    original_selection = read_json(args.original_selection)
    original_by_id = {row["question_id"]: row for row in original_selection["questions"]}
    for question in selection["questions"]:
        original = original_by_id.get(question["question_id"])
        if original is None:
            raise RuntimeError(f"Selected question is absent from original wave: {question['question_id']}")
        for field in ("domain", "question_type", "environment", "has_question_image", "eval_name"):
            if question[field] != original[field]:
                raise RuntimeError(f"Selected metadata differs for {question['question_id']}: {field}")
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")
    if prepared["selection_sha256"] != selection["original_selection_sha256"]:
        raise RuntimeError("Prepared data does not match the original frozen selection")

    native_query_prompt = extract_native_query_prompt(args.official_root / "memory_modules" / "agentrunbook_c.py")
    native_instruction = (
        args.official_root / "memory_modules" / "assets" / "agentrunbook_c" / "INSTRUCTION.md"
    ).read_text(encoding="utf-8")
    diff = prompt_diff_artifact(native_query_prompt, native_instruction)
    if diff["native_query_prompt_sha256"] != selection["native_query_prompt_sha256"]:
        raise RuntimeError("Native query prompt hash differs")
    if diff["native_instruction_sha256"] != selection["native_instruction_sha256"]:
        raise RuntimeError("Native instruction hash differs")
    ripgrep = verify_ripgrep_runtime()
    if Path(ripgrep["rg_path"]).resolve() != args.rg_bin.resolve():
        raise RuntimeError("Ripgrep probe did not use the configured binary")
    return {
        "checked_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "thinharness_revision": command_output(["git", "rev-parse", "HEAD"], cwd=args.repo_root),
        "official_revision": official_revision,
        "official_patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "selection_sha256": sha256_file(args.selection),
        "original_cell_receipts_sha256": sha256_file(args.original_receipts),
        "prepared_data_manifest_sha256": sha256_file(args.prepared_root / "prepared_data_manifest.json"),
        "prompt_diff_sha256": sha256_file(args.prompt_diff),
        "prompt_equivalence": diff,
        "ripgrep_runtime": ripgrep,
        "api_key_boundaries": {
            "OPENAI_API_KEY": "present, value not persisted",
            "OPENROUTER_API_KEY": "present, value not persisted",
        },
        "execution_scope": "five new ThinHarness-only prompt-alignment replicates; no native cells",
        "reader_scope": (
            "The official reader runs once per cell because the retrieval module does not produce the benchmark answer; "
            "the evaluator runs only where the frozen scoring specification requires it."
        ),
    }


def classify_path(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    if normalized.startswith(".thinharness/outputs/") or "/.thinharness/outputs/" in normalized:
        return "spill_artifacts"
    if normalized in {
        "corpus/trajectory_manifest.jsonl",
        "corpus/states.jsonl",
        "corpus/actions.jsonl",
    }:
        return "global_jsonl"
    if re.fullmatch(r"corpus/trajectories/[^/]+/(?:states|actions)\.jsonl", normalized):
        return "per_trajectory_jsonl"
    if re.fullmatch(r"corpus/trajectories/[^/]+/trajectory\.json", normalized):
        return "raw_trajectory_json"
    if normalized in {
        "corpus/TRAJECTORY_SUMMARY_CONCISE.md",
        "corpus/TRAJECTORY_SUMMARY_FULL.md",
    }:
        return "summaries"
    return None


def broad_scope_forms(path: str) -> list[str]:
    normalized = path.rstrip("/")
    if normalized in {".", "corpus"}:
        return ["global_jsonl", "per_trajectory_jsonl", "raw_trajectory_json", "summaries"]
    if normalized == "corpus/trajectories":
        return ["per_trajectory_jsonl", "raw_trajectory_json"]
    return []


def visible_paths(content: str) -> list[str]:
    matches = re.findall(
        r"(?:^|[\s\"'=])(corpus/(?:TRAJECTORY_SUMMARY_(?:CONCISE|FULL)\.md|(?:trajectory_manifest|states|actions)\.jsonl|trajectories/[A-Za-z0-9_-]+/(?:trajectory\.json|states\.jsonl|actions\.jsonl)))",
        content,
        flags=re.MULTILINE,
    )
    return sorted(set(matches))


def data_access_audit(output_dir: Path) -> dict[str, Any]:
    path_calls: list[dict[str, Any]] = []
    direct_counts = {form: 0 for form in DATA_FORMS}
    successful_direct_counts = {form: 0 for form in DATA_FORMS}
    broad_scope_counts = {form: 0 for form in DATA_FORMS}
    visible_counts = {form: 0 for form in DATA_FORMS}
    failed_calls: list[dict[str, Any]] = []
    spills_created: set[str] = set()
    spills_read: set[str] = set()
    for tool_path in sorted(output_dir.glob("query_traces/*/attempt_*/tool_calls.json")):
        calls = read_json(tool_path)
        for index, item in enumerate(calls, start=1):
            call = item.get("call", {})
            result = item.get("result", {})
            try:
                arguments = json.loads(call.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            path = arguments.get("path") if isinstance(arguments.get("path"), str) else None
            form = classify_path(path) if path is not None else None
            if form is not None:
                direct_counts[form] += 1
                if result.get("ok"):
                    successful_direct_counts[form] += 1
                    if form == "spill_artifacts" and call.get("name") == "read":
                        spills_read.add(path)
            content = str(result.get("content", ""))
            requested_scope_forms = (
                broad_scope_forms(path)
                if path is not None and call.get("name") in {"search", "jsonl_search"}
                else []
            )
            scope_forms = requested_scope_forms if result.get("ok") or "timed out" in content.lower() else []
            for scope_form in scope_forms:
                broad_scope_counts[scope_form] += 1
            mentioned_paths = visible_paths(content)
            for mentioned in mentioned_paths:
                mentioned_form = classify_path(mentioned)
                if mentioned_form is not None:
                    visible_counts[mentioned_form] += 1
            metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            saved_to_display = metadata.get("saved_to_display")
            if isinstance(saved_to_display, str):
                spills_created.add(saved_to_display)
            record = {
                "attempt_tool_calls_path": str(tool_path.relative_to(output_dir)),
                "call_index": index,
                "tool": call.get("name"),
                "path": path,
                "path_form": form,
                "broad_scope_forms": scope_forms,
                "ok": bool(result.get("ok")),
                "visible_paths": mentioned_paths,
                "spill_saved_to": saved_to_display,
            }
            path_calls.append(record)
            if not result.get("ok"):
                failed_calls.append(
                    {
                        **record,
                        "arguments": arguments,
                        "error": content,
                    }
                )
    return {
        "definitions": {
            "direct_target_calls": "Calls whose path names this form directly, including rejected calls.",
            "successful_direct_target_calls": "Successful calls whose path names this form directly.",
            "broad_scope_scan_calls": "Completed or timed-out search/jsonl_search calls whose path recursively covers this form.",
            "model_visible_result_mentions": "Tool results returned to the model that name a file in this form.",
        },
        "forms": {
            form: {
                "direct_target_calls": direct_counts[form],
                "successful_direct_target_calls": successful_direct_counts[form],
                "broad_scope_scan_calls": broad_scope_counts[form],
                "model_visible_result_mentions": visible_counts[form],
                "accessed": bool(successful_direct_counts[form] or broad_scope_counts[form]),
                "model_visible": bool(successful_direct_counts[form] or visible_counts[form]),
            }
            for form in DATA_FORMS
        },
        "spill_artifacts_created": sorted(spills_created),
        "spill_artifacts_read": sorted(spills_read),
        "failed_tool_calls": failed_calls,
        "path_calls": path_calls,
    }


def final_request_input_tokens(output_dir: Path) -> int | None:
    response_paths = sorted(output_dir.glob("query_traces/*/attempt_*/responses.json"))
    if response_paths:
        responses = read_json(response_paths[-1])
        if responses:
            usage = responses[-1].get("usage") if isinstance(responses[-1], dict) else None
            if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
                return int(usage["input_tokens"])
    summary_paths = sorted(output_dir.glob("query_traces/*/attempt_*/summary.json"))
    if not summary_paths:
        return None
    summary = read_json(summary_paths[-1])
    usage = summary.get("usage") if isinstance(summary, dict) else None
    per_response = usage.get("raw_response_usage") if isinstance(usage, dict) else None
    if not isinstance(per_response, list) or not per_response:
        return None
    value = per_response[-1].get("input_tokens") if isinstance(per_response[-1], dict) else None
    return int(value) if isinstance(value, int) else None


def comparison_metrics(receipt: dict[str, Any], diagnostics: dict[str, int], output_dir: Path | None = None) -> dict[str, Any]:
    values = metrics(receipt, diagnostics)
    values["reader_context_tokens"] = receipt["memory_context_token_count"]
    values["final_request_input_tokens"] = final_request_input_tokens(output_dir) if output_dir is not None else None
    return values


def compile_comparison(
    args: argparse.Namespace,
    selection: dict[str, Any],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    original_receipts = read_json(args.original_receipts)
    originals = {(row["question_id"], row["harness"]): row for row in original_receipts}
    new_by_id = {row["question_id"]: row for row in receipts}
    rows = []
    for question in selection["questions"]:
        question_id = question["question_id"]
        original_order = question["original_wave_order"]
        native_dir = args.original_run_root / "cells" / f"{original_order:02d}-{question_id}-native"
        thin_dir = args.original_run_root / "cells" / f"{original_order:02d}-{question_id}-thinharness"
        new_dir = args.run_root / "cells" / f"aligned-prompt-{question['order']:02d}-{question_id}-thinharness"
        native = comparison_metrics(originals[(question_id, "native")], search_diagnostics(native_dir, "native"), native_dir)
        old_thin = comparison_metrics(originals[(question_id, "thinharness")], search_diagnostics(thin_dir, "thinharness"), thin_dir)
        aligned = comparison_metrics(new_by_id[question_id], search_diagnostics(new_dir, "thinharness"), new_dir)
        audit = data_access_audit(new_dir)
        rows.append(
            {
                "question_id": question_id,
                "domain": question["domain"],
                "question_type": question["question_type"],
                "has_question_image": question["has_question_image"],
                "selection_query_cost_delta_usd": question["query_cost_delta_usd"],
                "original_native": native,
                "original_thinharness": old_thin,
                "aligned_prompt_thinharness": aligned,
                "aligned_vs_original_thinharness": {
                    field + "_ratio": (
                        float(aligned[field]) / float(old_thin[field])
                        if aligned[field] is not None and old_thin[field] not in {None, 0}
                        else None
                    )
                    for field in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
                },
                "aligned_vs_original_native": {
                    field + "_ratio": (
                        float(aligned[field]) / float(native[field])
                        if aligned[field] is not None and native[field] not in {None, 0}
                        else None
                    )
                    for field in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
                },
                "data_access": audit,
            }
        )
    total_fields = ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd", "failed_search_calls")
    totals: dict[str, dict[str, float]] = {}
    for label in ("original_native", "original_thinharness", "aligned_prompt_thinharness"):
        totals[label] = {field: sum(float(row[label][field] or 0) for row in rows) for field in total_fields}
        totals[label]["correct"] = sum(float(row[label]["score"] or 0) for row in rows)
    for denominator in ("original_thinharness", "original_native"):
        totals[f"aligned_vs_{denominator}"] = {
            field + "_ratio": (
                totals["aligned_prompt_thinharness"][field] / totals[denominator][field]
                if totals[denominator][field] else None
            )
            for field in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
        }
    return {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "interpretation": "Five new ThinHarness stochastic replicates with only the query prompt aligned; not replacements or same-seed reruns.",
        "rows": rows,
        "totals": totals,
    }


def _question_text(question_payload: Any) -> str:
    if isinstance(question_payload, str):
        return question_payload
    if isinstance(question_payload, dict) and isinstance(question_payload.get("text"), str):
        return question_payload["text"]
    raise RuntimeError("Runtime question has no text")


def validate_completed(
    args: argparse.Namespace,
    selection: dict[str, Any],
    receipts: list[dict[str, Any]],
    preflight_receipt: dict[str, Any],
) -> dict[str, Any]:
    if len(receipts) != 5 or {row["question_id"] for row in receipts} != {
        row["question_id"] for row in selection["questions"]
    }:
        raise RuntimeError("Completed prompt-alignment cells do not match the frozen selection")
    native_query_prompt = extract_native_query_prompt(args.official_root / "memory_modules" / "agentrunbook_c.py")
    native_instruction = (
        args.official_root / "memory_modules" / "assets" / "agentrunbook_c" / "INSTRUCTION.md"
    ).read_text(encoding="utf-8")
    artifact_files = artifact_bytes = prompts_checked = deterministic_checked = llm_checked = 0
    prompt_hashes: dict[str, str] = {}
    receipt_by_id = {row["question_id"]: row for row in receipts}
    for question in selection["questions"]:
        question_id = question["question_id"]
        cell_name = f"aligned-prompt-{question['order']:02d}-{question_id}-thinharness"
        cell_root = args.run_root / "cells" / cell_name
        receipt = receipt_by_id[question_id]
        if receipt["final_outcome"] != "scored" or receipt["score"] not in {0.0, 1.0}:
            raise RuntimeError(f"Invalid final outcome: {cell_name}")
        if receipt.get("replicate_label") != selection["experiment_id"] or receipt.get("replaces_original_cell") is not False:
            raise RuntimeError(f"Replicate label differs: {cell_name}")
        metadata = read_json(cell_root / "cell_metadata.json")
        if metadata["harness"] != "thinharness" or metadata["query_api"] != "https://api.openai.com/v1":
            raise RuntimeError(f"Query route differs: {cell_name}")
        if metadata["ripgrep_runtime"]["status"] != "passed":
            raise RuntimeError(f"Ripgrep preflight differs: {cell_name}")
        reader_metadata = receipt.get("reader_metadata") or {}
        if reader_metadata.get("provider") != "Parasail" or reader_metadata.get("response_model") != "qwen/qwen3.5-9b":
            raise RuntimeError(f"Reader route differs: {cell_name}")
        runtime_questions = read_json(cell_root / "runtime_inputs" / "questions.json")
        if len(runtime_questions) != 1:
            raise RuntimeError(f"Runtime question count differs: {cell_name}")
        expected_prompt = build_aligned_query_prompt(
            _question_text(runtime_questions[0]["question"]),
            native_query_prompt=native_query_prompt,
            native_instruction=native_instruction,
        )
        prompt_paths = sorted(cell_root.glob("query_traces/*/attempt_*/prompt.md"))
        if not prompt_paths or any(path.read_text(encoding="utf-8") != expected_prompt for path in prompt_paths):
            raise RuntimeError(f"Rendered prompt differs from aligned source: {cell_name}")
        hashes = [sha256_file(path) for path in prompt_paths]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Rendered prompt attempts differ: {cell_name}")
        prompt_hashes[question_id] = hashes[0]
        prompts_checked += len(prompt_paths)
        for relative, expected_hash in receipt["artifact_hashes"].items():
            path = cell_root / relative
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Cell artifact hash differs: {path}")
            artifact_files += 1
            artifact_bytes += path.stat().st_size
        if receipt["scoring_path"] == "deterministic":
            deterministic_checked += 1
        else:
            if len(receipt["evaluator_receipts"]) != 1:
                raise RuntimeError(f"Expected one evaluator receipt: {cell_name}")
            llm_checked += 1

    run_hashes = hash_tree(args.run_root)
    write_json(args.evidence_root / "raw_artifact_manifest.json", {"run_root": str(args.run_root), "run_root_hashes": run_hashes})
    for relative, expected_hash in run_hashes.items():
        if sha256_file(args.run_root / relative) != expected_hash:
            raise RuntimeError(f"Raw artifact hash differs: {relative}")
    return {
        "schema_version": 1,
        "validated_at_utc": utc_now(),
        "status": "passed",
        "coverage": {
            "question_ids": 5,
            "cells": 5,
            "native_cells_run": 0,
            "thinharness_prompt_alignment_replicates": 5,
            "question_image_cells": sum(bool(row["has_question_image"]) for row in selection["questions"]),
        },
        "selection": {
            "selection_sha256": sha256_file(args.selection),
            "top_five_positive_cost_deltas_recomputed": True,
            "original_receipts_sha256": sha256_file(args.original_receipts),
        },
        "prompt_equivalence": {
            "prompt_diff_sha256": sha256_file(args.prompt_diff),
            "native_query_prompt_sha256": preflight_receipt["prompt_equivalence"]["native_query_prompt_sha256"],
            "native_instruction_sha256": preflight_receipt["prompt_equivalence"]["native_instruction_sha256"],
            "rendered_prompts_checked": prompts_checked,
            "rendered_prompt_sha256_by_question": prompt_hashes,
            "localized_replacement_count": len(preflight_receipt["prompt_equivalence"]["query_prompt_replacements"])
            + len(preflight_receipt["prompt_equivalence"]["instruction_replacements"]),
        },
        "scoring": {
            "deterministic_cells": deterministic_checked,
            "llm_judged_cells_with_receipts": llm_checked,
            "official_reader_required_cells": 5,
        },
        "artifacts": {
            "cell_artifact_files_verified": artifact_files,
            "cell_artifact_bytes_verified": artifact_bytes,
            "raw_manifest_files_verified": len(run_hashes),
            "raw_manifest_sha256": sha256_file(args.evidence_root / "raw_artifact_manifest.json"),
        },
        "secret_boundary": secret_scan([args.run_root, args.evidence_root]),
        "ripgrep_runtime": preflight_receipt["ripgrep_runtime"],
    }


def write_results(path: Path, comparison: dict[str, Any], costs: dict[str, float]) -> None:
    lines = [
        "# LongMemEval native-prompt alignment diagnostic",
        "",
        "Five new ThinHarness-only stochastic replicates. The query prompt is the only intentionally changed runtime variable.",
        "",
        "| Question | Evidence | Input | Tools | Failures | Latency | Query cost | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        for label, key in (
            ("original native", "original_native"),
            ("original ThinHarness", "original_thinharness"),
            ("aligned-prompt ThinHarness", "aligned_prompt_thinharness"),
        ):
            value = row[key]
            lines.append(
                f"| {row['question_id']} | {label} | {value['input_tokens']:,} | {value['tool_calls']} | "
                f"{value['failed_search_calls']} | {value['latency_seconds']:.2f}s | "
                f"${value['query_cost_usd']:.8f} | {value['score']:.0f} |"
            )
    totals = comparison["totals"]
    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Aligned/original ThinHarness query-cost ratio: {totals['aligned_vs_original_thinharness']['query_cost_usd_ratio']:.3f}x.",
            f"- Aligned/original native query-cost ratio: {totals['aligned_vs_original_native']['query_cost_usd_ratio']:.3f}x.",
            (
                f"- Correct: native {totals['original_native']['correct']:.0f}/5, "
                f"original ThinHarness {totals['original_thinharness']['correct']:.0f}/5, "
                f"aligned prompt {totals['aligned_prompt_thinharness']['correct']:.0f}/5."
            ),
            f"- New-cell query API-equivalent cost: ${costs['query_api_equivalent']:.8f}.",
            f"- New-cell total API-equivalent cost: ${costs['total_api_equivalent']:.8f}.",
            "- Scores are new stochastic outcomes and do not replace the paired wave.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--original-selection", type=Path, required=True)
    parser.add_argument("--original-receipts", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--prompt-diff", type=Path, required=True)
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
        "original_selection",
        "original_receipts",
        "original_run_root",
        "prompt_diff",
        "patch",
        "rg_bin",
        "run_root",
        "evidence_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    configure_ripgrep(args.rg_bin)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    preflight_receipt = preflight(args, selection)
    write_json(args.run_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "prompt_diff.json", preflight_receipt["prompt_equivalence"])
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")

    receipts: list[dict[str, Any]] = []
    progress_path = args.run_root / "progress.jsonl"
    credit_exhausted = False
    for question in selection["questions"]:
        cell_name = f"aligned-prompt-{question['order']:02d}-{question['question_id']}-thinharness"
        output_dir = args.run_root / "cells" / cell_name
        receipt_path = output_dir / "cell_receipt.json"
        if receipt_path.exists():
            receipts.append(read_json(receipt_path))
            print(f"CHECKPOINT skip completed {cell_name}", flush=True)
            continue
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
        returncode, output = run_subprocess(command, process_log, args.repo_root)
        receipt = build_receipt(
            output_dir=output_dir,
            harness_name="thinharness",
            question=question,
            process_returncode=returncode,
            process_log=process_log,
            evaluator_receipts_path=evaluator_path,
        )
        receipt["replicate_label"] = selection["experiment_id"]
        receipt["replicate_kind"] = "native_prompt_alignment"
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
        if detect_credit_exhaustion(output, receipt):
            credit_exhausted = True
            print(f"CREDIT_EXHAUSTED after {cell_name}", flush=True)
            break
        if receipt["final_outcome"] != "scored":
            raise RuntimeError(f"Unrecoverable cell failure: {cell_name}; see {process_log}")

    comparison = compile_comparison(args, selection, receipts) if len(receipts) == 5 else None
    compact_receipts = [
        {key: value for key, value in receipt.items() if key not in {"artifact_hashes", "query_attempts", "response_raw"}}
        for receipt in receipts
    ]
    write_json(args.evidence_root / "cell_receipts.json", compact_receipts)
    if comparison is not None:
        write_json(args.evidence_root / "comparison.json", comparison)
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
        "status": "credit_exhausted" if credit_exhausted else "completed",
        "completed_cells": len(receipts),
        "target_cells": 5,
        "native_cells_run": 0,
        "thinharness_prompt_alignment_cells_run": len(receipts),
        "reader_cells_run": len(receipts),
        "reader_necessity": "Required to convert retrieved memory context into the benchmark answer for scoring.",
        "cost_totals_usd": costs,
    }
    write_json(args.evidence_root / "final_summary.json", final)
    if comparison is not None:
        write_results(args.evidence_root / "RESULTS.md", comparison, costs)
    marker = "CREDIT_EXHAUSTED.json" if credit_exhausted else "COMPLETED.json"
    write_json(args.run_root / marker, final)
    if not credit_exhausted and len(receipts) == 5:
        validation = validate_completed(args, selection, receipts, preflight_receipt)
        write_json(args.evidence_root / "validation.json", validation)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
