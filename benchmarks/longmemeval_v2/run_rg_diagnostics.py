#!/usr/bin/env python3
"""Run four ThinHarness-only ripgrep diagnostic replicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.longmemeval_v2.orchestrate import (
    OFFICIAL_REVISION,
    build_receipt,
    detect_credit_exhaustion,
    normalize_patch_blank_context,
    run_subprocess,
    sha256_file,
    write_json,
)
from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    import subprocess

    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def configure_ripgrep(rg_bin: Path) -> None:
    resolved = rg_bin.expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Configured rg is not executable: {resolved}")
    os.environ["PATH"] = os.pathsep.join([str(resolved.parent), os.getenv("PATH", "")])
    selected = shutil.which("rg")
    if selected is None or Path(selected).resolve() != resolved:
        raise RuntimeError(f"Runtime did not select configured rg: expected {resolved}, got {selected}")


def preflight(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    if len(selection.get("questions", [])) != 4:
        raise RuntimeError("Diagnostic selection must contain exactly four questions")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    if command_output(["git", "status", "--porcelain"], cwd=args.repo_root):
        raise RuntimeError("ThinHarness repository must be clean before paid execution")
    official_revision = command_output(["git", "rev-parse", "HEAD"], cwd=args.official_root)
    if official_revision != OFFICIAL_REVISION:
        raise RuntimeError(f"Official harness revision differs: {official_revision}")
    patch_bytes = args.patch.read_bytes()
    official_diff = __import__("subprocess").check_output(
        ["git", "diff", "--", "evaluation/harness.py", "evaluation/run_eval.py", "evaluation/qa_eval_metrics.py"],
        cwd=args.official_root,
    )
    if normalize_patch_blank_context(official_diff) != patch_bytes:
        raise RuntimeError("Official harness instrumentation differs from the committed patch")
    if sha256_file(args.original_receipts) != selection["original_cell_receipts_sha256"]:
        raise RuntimeError("Original cell receipts differ from the diagnostic selection")
    original_selection_path = args.selection.parent / "selection.json"
    if sha256_file(original_selection_path) != selection["original_selection_sha256"]:
        raise RuntimeError("Original frozen selection differs")
    original_selection = read_json(original_selection_path)
    original_by_id = {row["question_id"]: row for row in original_selection["questions"]}
    for row in selection["questions"]:
        original = original_by_id.get(row["question_id"])
        if original is None:
            raise RuntimeError(f"Diagnostic question is absent from original selection: {row['question_id']}")
        for field in ("domain", "question_type", "has_question_image", "eval_name"):
            if row[field] != original[field]:
                raise RuntimeError(f"Diagnostic metadata differs for {row['question_id']}: {field}")
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")
    if prepared["selection_sha256"] != selection["original_selection_sha256"]:
        raise RuntimeError("Prepared data does not match the original frozen selection")
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
        "ripgrep_runtime": ripgrep,
        "api_key_boundaries": {
            "OPENAI_API_KEY": "present, value not read or persisted",
            "OPENROUTER_API_KEY": "present, value not read or persisted",
        },
        "execution_scope": "four new ThinHarness-only replicates; no native cells",
    }


def search_diagnostics(output_dir: Path, harness: str) -> dict[str, int]:
    failed_search_calls = 0
    rg_unavailable_events = 0
    search_calls = 0
    if harness == "thinharness":
        for path in output_dir.glob("query_traces/*/attempt_*/tool_calls.json"):
            for call in read_json(path):
                name = call.get("call", {}).get("name")
                if name not in {"search", "jsonl_search"}:
                    continue
                search_calls += 1
                result = call.get("result", {})
                if not result.get("ok"):
                    failed_search_calls += 1
                content = str(result.get("content", ""))
                if "No such file or directory: 'rg'" in content:
                    rg_unavailable_events += 1
    else:
        for path in output_dir.glob("query_traces/*/attempt_*/events.json"):
            payload = read_json(path)
            for call in payload.get("tool_calls", []):
                if call.get("tool") != "shell":
                    continue
                command = str(call.get("command", ""))
                response = call.get("tool_response", {})
                stderr = str(response.get("stderr", ""))
                if "rg " in command or command.startswith("rg"):
                    search_calls += 1
                if "rg: command not found" in stderr:
                    rg_unavailable_events += 1
                    failed_search_calls += 1
    return {
        "search_calls": search_calls,
        "failed_search_calls": failed_search_calls,
        "rg_unavailable_events": rg_unavailable_events,
    }


def metrics(receipt: dict[str, Any], diagnostics: dict[str, int]) -> dict[str, Any]:
    usage = receipt["query_usage"]
    return {
        "score": receipt["score"],
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "ordinary_input_tokens": usage["ordinary_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_output_tokens": usage["reasoning_output_tokens"],
        "model_requests": usage["requests"],
        "tool_calls": usage["tool_calls"],
        "latency_seconds": receipt["memory_query_duration_seconds"],
        "query_cost_usd": receipt["costs_usd"]["query_api_equivalent"],
        **diagnostics,
    }


def ratio(new: float | int | None, baseline: float | int | None) -> float | None:
    if new is None or baseline in (None, 0):
        return None
    return float(new) / float(baseline)


def compile_evidence(args: argparse.Namespace, selection: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    originals = read_json(args.original_receipts)
    original_by_pair = {(row["question_id"], row["harness"]): row for row in originals}
    replicate_by_id = {row["question_id"]: row for row in receipts}
    rows = []
    for question in selection["questions"]:
        question_id = question["question_id"]
        original_thin = original_by_pair[(question_id, "thinharness")]
        native = original_by_pair[(question_id, "native")]
        replicate = replicate_by_id[question_id]
        original_order = next(
            row["order"] for row in read_json(args.selection.parent / "selection.json")["questions"]
            if row["question_id"] == question_id
        )
        original_thin_dir = args.original_run_root / "cells" / f"{original_order:02d}-{question_id}-thinharness"
        native_dir = args.original_run_root / "cells" / f"{original_order:02d}-{question_id}-native"
        replicate_dir = args.run_root / "cells" / f"replicate-{question['order']:02d}-{question_id}-thinharness"
        thin_metrics = metrics(original_thin, search_diagnostics(original_thin_dir, "thinharness"))
        native_metrics = metrics(native, search_diagnostics(native_dir, "native"))
        replicate_metrics = metrics(replicate, search_diagnostics(replicate_dir, "thinharness"))
        rows.append({
            "question_id": question_id,
            "domain": question["domain"],
            "question_type": question["question_type"],
            "has_question_image": question["has_question_image"],
            "rationale": question["rationale"],
            "original_native": native_metrics,
            "original_thinharness": thin_metrics,
            "rg_replicate_thinharness": replicate_metrics,
            "replicate_vs_original_thinharness": {
                key + "_ratio": ratio(replicate_metrics[key], thin_metrics[key])
                for key in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
            },
            "replicate_vs_original_native": {
                key + "_ratio": ratio(replicate_metrics[key], native_metrics[key])
                for key in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
            },
        })
    total_fields = ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd", "failed_search_calls")
    totals: dict[str, dict[str, float]] = {}
    for label in ("original_native", "original_thinharness", "rg_replicate_thinharness"):
        totals[label] = {field: sum(float(row[label][field] or 0) for row in rows) for field in total_fields}
        totals[label]["correct"] = sum(float(row[label]["score"] or 0) for row in rows)
    totals["replicate_vs_original_thinharness"] = {
        field + "_ratio": ratio(totals["rg_replicate_thinharness"][field], totals["original_thinharness"][field])
        for field in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
    }
    totals["replicate_vs_original_native"] = {
        field + "_ratio": ratio(totals["rg_replicate_thinharness"][field], totals["original_native"][field])
        for field in ("input_tokens", "tool_calls", "latency_seconds", "query_cost_usd")
    }
    return {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "interpretation": "New ThinHarness stochastic replicates compared with preserved original cells; not replacements or paired same-seed reruns.",
        "rows": rows,
        "totals": totals,
    }


def write_results_markdown(path: Path, comparison: dict[str, Any], cost_totals: dict[str, float]) -> None:
    lines = [
        "# LongMemEval ripgrep diagnostic replicates",
        "",
        "These are four new ThinHarness-only stochastic replicates. They preserve and do not replace the original paired-wave cells.",
        "",
        "| Question | Original Thin input | Replicate input | Native input | Original Thin tools "
        "| Replicate tools | Search failures old/new | Score native/old/new |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        native = row["original_native"]
        old = row["original_thinharness"]
        new = row["rg_replicate_thinharness"]
        lines.append(
            f"| {row['question_id']} | {old['input_tokens']:,} | {new['input_tokens']:,} | {native['input_tokens']:,} "
            f"| {old['tool_calls']} | {new['tool_calls']} | {old['failed_search_calls']}/{new['failed_search_calls']} "
            f"| {native['score']:.0f}/{old['score']:.0f}/{new['score']:.0f} |"
        )
    totals = comparison["totals"]
    ratio_old = totals["replicate_vs_original_thinharness"]
    ratio_native = totals["replicate_vs_original_native"]
    lines.extend([
        "",
        "## Totals",
        "",
        f"- Replicate/original ThinHarness input ratio: {ratio_old['input_tokens_ratio']:.3f}x.",
        f"- Replicate/original ThinHarness query-cost ratio: {ratio_old['query_cost_usd_ratio']:.3f}x.",
        f"- Replicate/original native input ratio: {ratio_native['input_tokens_ratio']:.3f}x.",
        f"- Replicate/original native query-cost ratio: {ratio_native['query_cost_usd_ratio']:.3f}x.",
        f"- Replicate final-cell API-equivalent cost: ${cost_totals['total_api_equivalent']:.8f}.",
        "- Scores are stochastic outcomes. This diagnostic cannot attribute score changes only to ripgrep.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--original-receipts", type=Path, required=True)
    parser.add_argument("--original-run-root", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--rg-bin", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args()
    for field in (
        "repo_root", "official_root", "prepared_root", "selection", "original_receipts",
        "original_run_root", "patch", "rg_bin", "run_root", "evidence_root",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    configure_ripgrep(args.rg_bin)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    selection = read_json(args.selection)
    preflight_receipt = preflight(args, selection)
    write_json(args.run_root / "preflight.json", preflight_receipt)
    write_json(args.evidence_root / "preflight.json", preflight_receipt)
    prepared = read_json(args.prepared_root / "prepared_data_manifest.json")

    receipts: list[dict[str, Any]] = []
    progress_path = args.run_root / "progress.jsonl"
    credit_exhausted = False
    for question in selection["questions"]:
        cell_name = f"replicate-{question['order']:02d}-{question['question_id']}-thinharness"
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
            "--official-root", str(args.official_root),
            "--data-root", prepared["domains"][question["domain"]]["data_root"],
            "--output-dir", str(output_dir),
            "--harness", "thinharness",
            "--question-id", question["question_id"],
            "--domain", question["domain"],
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
        receipt["replaces_original_cell"] = False
        write_json(receipt_path, receipt)
        receipts.append(receipt)
        with progress_path.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps({
                "completed_at_utc": receipt["completed_at_utc"],
                "cell": cell_name,
                "score": receipt["score"],
                "final_outcome": receipt["final_outcome"],
                "receipt_sha256": sha256_file(receipt_path),
            }) + "\n")
        print(f"CELL_COMPLETE {cell_name} outcome={receipt['final_outcome']} score={receipt['score']}", flush=True)
        if detect_credit_exhaustion(output, receipt):
            credit_exhausted = True
            print(f"CREDIT_EXHAUSTED after {cell_name}", flush=True)
            break
        if receipt["final_outcome"] != "scored":
            raise RuntimeError(f"Unrecoverable cell failure: {cell_name}; see {process_log}")

    comparison = compile_evidence(args, selection, receipts) if len(receipts) == 4 else None
    compact_receipts = [
        {key: value for key, value in receipt.items() if key not in {"artifact_hashes", "query_attempts", "response_raw"}}
        for receipt in receipts
    ]
    write_json(args.evidence_root / "cell_receipts.json", compact_receipts)
    if comparison is not None:
        write_json(args.evidence_root / "comparison.json", comparison)
    cost_totals = {
        "query_api_equivalent": sum(float(row["costs_usd"]["query_api_equivalent"]) for row in receipts),
        "reader_provider_reported": sum(float(row["costs_usd"]["reader_provider_reported"] or 0) for row in receipts),
        "evaluator_api_equivalent": sum(float(row["costs_usd"]["evaluator_api_equivalent"]) for row in receipts),
    }
    cost_totals["total_api_equivalent"] = sum(
        float(row["costs_usd"]["total_api_equivalent"]) for row in receipts
    )
    final = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "experiment_id": selection["experiment_id"],
        "status": "credit_exhausted" if credit_exhausted else "completed",
        "completed_cells": len(receipts),
        "target_cells": 4,
        "native_cells_run": 0,
        "thinharness_replicate_cells_run": len(receipts),
        "cost_totals_usd": cost_totals,
    }
    write_json(args.evidence_root / "final_summary.json", final)
    if comparison is not None:
        write_results_markdown(args.evidence_root / "RESULTS.md", comparison, cost_totals)
    marker = "CREDIT_EXHAUSTED.json" if credit_exhausted else "COMPLETED.json"
    write_json(args.run_root / marker, final)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
