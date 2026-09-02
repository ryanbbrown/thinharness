#!/usr/bin/env python3
"""Freeze the representative ten-question clean paired selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.longmemeval_v2.freeze_selection import DATASET_REVISION, sha256_file

DEFAULT_SEED = "thinharness-lmev2-luna-xhigh-clean-pair-v1"
SELECTION_SLOTS = (
    ("static-environment", "web", "webarena-reddit"),
    ("static-environment", "enterprise", "workarena"),
    ("static-environment-abs", "web", "webarena-cms"),
    ("dynamic-environment", "web", "webarena-onestopshop"),
    ("dynamic-environment", "enterprise", "workarena"),
    ("dynamic-environment-abs", "enterprise", "workarena"),
    ("procedure", "web", "webarena-cms"),
    ("procedure-abs", "enterprise", "workarena"),
    ("errors-gotchas", "web", "webarena-reddit"),
    ("errors-gotchas", "enterprise", "workarena"),
)


def stable_rank(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def build_selection(data_root: Path, seed: str) -> dict[str, Any]:
    questions_path = data_root / "questions.jsonl"
    haystack_path = data_root / "haystacks" / "lme_v2_small.json"
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line]
    if len(questions) != 451:
        raise RuntimeError(f"Expected 451 questions, found {len(questions)}")
    haystacks = json.loads(haystack_path.read_text(encoding="utf-8"))

    selected: list[dict[str, Any]] = []
    for question_type, domain, environment in SELECTION_SLOTS:
        candidates = [
            row
            for row in questions
            if row["question_type"] == question_type
            and row["domain"] == domain
            and row["environment"] == environment
        ]
        if not candidates:
            raise RuntimeError(f"No candidates for {question_type}/{domain}/{environment}")
        winner = min(candidates, key=lambda row: stable_rank(seed, row["id"]))
        if winner["id"] not in haystacks or len(haystacks[winner["id"]]) != 100:
            raise RuntimeError(f"Invalid Small haystack for {winner['id']}")
        selected.append(
            {
                "order": len(selected) + 1,
                "question_id": winner["id"],
                "domain": domain,
                "question_type": question_type,
                "environment": environment,
                "has_question_image": winner["image"] is not None,
                "eval_name": winner["eval_function"].split("|", 1)[0],
                "selection_rank_sha256": stable_rank(seed, winner["id"]),
            }
        )

    if len(selected) != 10 or len({row["question_id"] for row in selected}) != 10:
        raise RuntimeError("Selection must contain ten unique questions")
    if {row["question_type"] for row in selected} != {slot[0] for slot in SELECTION_SLOTS}:
        raise RuntimeError("Selection must cover all seven question types")
    if {domain: sum(row["domain"] == domain for row in selected) for domain in ("web", "enterprise")} != {
        "web": 5,
        "enterprise": 5,
    }:
        raise RuntimeError("Selection must contain five questions per domain")
    image_rows = [row for row in selected if row["has_question_image"]]
    if [(row["domain"], row["question_type"]) for row in image_rows] != [
        ("web", "errors-gotchas"),
        ("enterprise", "errors-gotchas"),
    ]:
        raise RuntimeError("Selection must exercise both question-image paths")
    deterministic = sum(not row["eval_name"].startswith("llm_") for row in selected)
    if deterministic != 5:
        raise RuntimeError("Selection must contain five deterministic and five LLM-judged questions")

    return {
        "schema_version": 1,
        "experiment_id": "luna-xhigh-clean-pair-v1",
        "seed": seed,
        "selection_algorithm": (
            "minimum SHA-256(seed + ':' + question_id) in each predeclared question-type/domain/environment slot"
        ),
        "selection_basis": (
            "Ten fixed slots cover all seven question types, five questions per domain, all three web environments, "
            "five deterministic and five LLM-judged questions, and both image paths. Prior costs, scores, traces, "
            "and outcomes are not inputs."
        ),
        "dataset_revision": DATASET_REVISION,
        "tier": "small",
        "source_hashes": {
            "questions.jsonl": sha256_file(questions_path),
            "haystacks/lme_v2_small.json": sha256_file(haystack_path),
        },
        "ordered_slots": [
            {"question_type": question_type, "domain": domain, "environment": environment}
            for question_type, domain, environment in SELECTION_SLOTS
        ],
        "questions": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    selection = build_selection(args.data_root.resolve(), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(selection['questions'])} questions at {args.output}")


if __name__ == "__main__":
    main()
