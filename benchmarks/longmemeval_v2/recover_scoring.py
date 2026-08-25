#!/usr/bin/env python3
"""Recover reader/scoring after a post-query infrastructure failure."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--cell-output", type=Path, required=True)
    parser.add_argument("--evaluator-receipts", type=Path, required=True)
    args = parser.parse_args()
    args.official_root = args.official_root.resolve()
    args.cell_output = args.cell_output.resolve()
    args.evaluator_receipts = args.evaluator_receipts.resolve()
    sys.path.insert(0, str(args.official_root))
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Required API keys were not injected")
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["LME_EVALUATOR_RECEIPTS_PATH"] = str(args.evaluator_receipts)
    args.evaluator_receipts.parent.mkdir(parents=True, exist_ok=True)

    from evaluation.harness import (  # noqa: PLC0415
        aggregate_metrics,
        generate_all_reader_outputs,
        make_eval_config,
        score_prediction,
        utc_now_iso,
    )

    per_question_path = args.cell_output / "per_question.jsonl"
    if per_question_path.exists():
        raise RuntimeError(f"Refusing to overwrite scored output: {per_question_path}")
    run_args = argparse.Namespace(**json.loads((args.cell_output / "run_args.json").read_text(encoding="utf-8")))
    prompt_rows = load_jsonl(args.cell_output / "prompt_rows.jsonl")
    if len(prompt_rows) != 1:
        raise RuntimeError("Scoring recovery supports exactly one prompt row")
    reader_outputs_path = args.cell_output / "reader_outputs.jsonl"
    reader_reused = reader_outputs_path.exists()
    if reader_reused:
        outputs_by_question_id = {
            row["question_id"]: {key: value for key, value in row.items() if key != "question_id"}
            for row in load_jsonl(reader_outputs_path)
        }
    else:
        outputs_by_question_id = asyncio.run(generate_all_reader_outputs(run_args, prompt_rows))
        with reader_outputs_path.open("w", encoding="utf-8") as handle:
            for row in prompt_rows:
                output = outputs_by_question_id[row["question_id"]]
                handle.write(json.dumps({"question_id": row["question_id"], **output}, ensure_ascii=True) + "\n")
                handle.flush()

    row = prompt_rows[0]
    output = outputs_by_question_id[row["question_id"]]
    scored_row = {
        **row,
        "response_raw": output["response_raw"],
        "response_parsed_boxed": output["response_parsed_boxed"],
        "is_unknown": output["is_unknown"],
        "usage": output["usage"],
        "reader_metadata": output["reader_metadata"],
    }
    score_bool, _, _ = score_prediction(scored_row, make_eval_config(run_args))
    record = {
        "index": scored_row["index"],
        "stream_index": scored_row["stream_index"],
        "question_id": scored_row["question_id"],
        "question_type": scored_row["question_type"],
        "category": scored_row["category"],
        "is_abstention_problem": scored_row["is_abstention_problem"],
        "eval_function": scored_row["eval_function"],
        "question_text": scored_row["question_text"],
        "question_image": scored_row["question_image"],
        "haystack_ids": scored_row["haystack_ids"],
        "memory_context": scored_row["memory_context"],
        "memory_query_duration_seconds": scored_row["memory_query_duration_seconds"],
        "memory_post_query_duration_seconds": scored_row["memory_post_query_duration_seconds"],
        "memory_post_query_metadata": scored_row["memory_post_query_metadata"],
        "memory_context_original_token_count": scored_row["memory_context_original_token_count"],
        "memory_context_token_count": scored_row["memory_context_token_count"],
        "memory_context_was_truncated": scored_row["memory_context_was_truncated"],
        "prompt_messages": scored_row["prompt_messages"],
        "answer_gold": scored_row["answer_gold"],
        "response_raw": scored_row["response_raw"],
        "response_parsed_boxed": scored_row["response_parsed_boxed"],
        "is_unknown": scored_row["is_unknown"],
        "score": 1.0 if score_bool else 0.0,
        "score_bool": score_bool,
        "usage": scored_row["usage"],
        "reader_metadata": scored_row["reader_metadata"],
        "timestamp_utc": utc_now_iso(),
    }
    per_question_path.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")
    aggregated = aggregate_metrics([record])
    aggregated["tokens"] = {
        "prompt_tokens": int(record["usage"].get("prompt_tokens", 0) or 0),
        "completion_tokens": int(record["usage"].get("completion_tokens", 0) or 0),
        "total_tokens": int(record["usage"].get("total_tokens", 0) or 0),
        "avg_prompt_tokens": int(record["usage"].get("prompt_tokens", 0) or 0),
        "avg_completion_tokens": int(record["usage"].get("completion_tokens", 0) or 0),
        "avg_total_tokens": int(record["usage"].get("total_tokens", 0) or 0),
    }
    aggregated["memory_context"] = {
        "avg_original_tokens": record["memory_context_original_token_count"],
        "avg_final_tokens": record["memory_context_token_count"],
        "num_truncated_sequences": int(record["memory_context_was_truncated"]),
    }
    for key, field in (
        ("memory_query", "memory_query_duration_seconds"),
        ("memory_post_query", "memory_post_query_duration_seconds"),
    ):
        duration = float(record[field])
        aggregated[key] = {
            "avg_seconds": duration,
            "p50_seconds": duration,
            "p95_seconds": duration,
            "max_seconds": duration,
            "total_seconds": duration,
        }
    aggregated["completed_at_utc"] = utc_now_iso()
    aggregated["shared_haystack"] = True
    aggregated["shared_haystack_ids"] = record["haystack_ids"]
    write_json(args.cell_output / "aggregated_metrics.json", aggregated)
    write_json(
        args.cell_output / "scoring_recovery.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "reason": "Evaluator receipt directory was missing after query and reader completed.",
            "query_rerun": False,
            "reader_reused": reader_reused,
            "reader_rerun": not reader_reused,
            "evaluator_rerun": True,
            "score": record["score"],
        },
    )
    print(f"Recovered score for {record['question_id']}: {record['score']}")


if __name__ == "__main__":
    main()
