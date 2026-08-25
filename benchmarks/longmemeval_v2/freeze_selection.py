#!/usr/bin/env python3
"""Freeze the seeded LongMemEval-V2 paired-wave selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_REVISION = "f152293e235517d504809563c833d7190b8c713b"
DEFAULT_SEED = "thinharness-lmev2-luna-xhigh-wave1-v1"
DOMAINS = ("web", "enterprise")
QUESTION_TYPES = (
    "static-environment",
    "static-environment-abs",
    "dynamic-environment",
    "dynamic-environment-abs",
    "procedure",
    "procedure-abs",
    "errors-gotchas",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 451:
        raise RuntimeError(f"Expected 451 questions, found {len(rows)}")
    return rows


def stable_rank(seed: str, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def build_selection(data_root: Path, seed: str) -> dict[str, Any]:
    questions_path = data_root / "questions.jsonl"
    haystack_path = data_root / "haystacks" / "lme_v2_small.json"
    rows = load_questions(questions_path)
    haystacks = json.loads(haystack_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPES:
        for domain in DOMAINS:
            candidates = [
                row
                for row in rows
                if row["domain"] == domain and row["question_type"] == question_type
            ]
            if not candidates:
                raise RuntimeError(f"No candidates for {domain}/{question_type}")
            winner = min(candidates, key=lambda row: stable_rank(seed, row["id"]))
            if winner["id"] not in haystacks or len(haystacks[winner["id"]]) != 100:
                raise RuntimeError(f"Invalid Small haystack for {winner['id']}")
            selected.append(
                {
                    "order": len(selected) + 1,
                    "question_id": winner["id"],
                    "domain": domain,
                    "question_type": question_type,
                    "environment": winner["environment"],
                    "has_question_image": winner["image"] is not None,
                    "eval_name": winner["eval_function"].split("|", 1)[0],
                    "selection_rank_sha256": stable_rank(seed, winner["id"]),
                }
            )
    if len(selected) != 14 or len({row["question_id"] for row in selected}) != 14:
        raise RuntimeError("Selection must contain 14 unique questions")
    image_rows = [row for row in selected if row["has_question_image"]]
    if {(row["domain"], row["question_type"]) for row in image_rows} != {
        ("web", "errors-gotchas"),
        ("enterprise", "errors-gotchas"),
    }:
        raise RuntimeError("Selection must include one gotchas image question per domain")
    return {
        "schema_version": 1,
        "seed": seed,
        "selection_algorithm": "minimum SHA-256(seed + ':' + question_id) within each ordered stratum",
        "dataset_revision": DATASET_REVISION,
        "tier": "small",
        "source_hashes": {
            "questions.jsonl": sha256_file(questions_path),
            "haystacks/lme_v2_small.json": sha256_file(haystack_path),
        },
        "stratum_order": {
            "question_types": list(QUESTION_TYPES),
            "domains_within_type": list(DOMAINS),
        },
        "questions": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    payload = build_selection(args.data_root.resolve(), args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(payload['questions'])} questions at {args.output}")


if __name__ == "__main__":
    main()
