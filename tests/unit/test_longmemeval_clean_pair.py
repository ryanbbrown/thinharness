from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.longmemeval_v2.freeze_clean_pair_selection import (  # noqa: E402
    SELECTION_SLOTS,
    build_selection,
)
from benchmarks.longmemeval_v2.resume_clean_pair_readers import (  # noqa: E402
    fallback_delay,
    parse_retry_after,
    write_jsonl_record,
)
from benchmarks.longmemeval_v2.run_clean_pair import projected_total  # noqa: E402


def test_clean_pair_selection_covers_declared_representative_slots(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "haystacks").mkdir(parents=True)
    rows = []
    haystacks = {}
    expected_ids = []
    for index, (question_type, domain, environment) in enumerate(SELECTION_SLOTS):
        question_id = f"slot-{index}"
        expected_ids.append(question_id)
        rows.append(
            {
                "id": question_id,
                "domain": domain,
                "environment": environment,
                "question_type": question_type,
                "question": "Question",
                "image": "question.png" if question_type == "errors-gotchas" else None,
                "answer": "Answer",
                "eval_function": (
                    "llm_abstention_checker" if question_type.endswith("-abs") else
                    "llm_gotchas_checker" if question_type == "errors-gotchas" else
                    "norm_phrase_set_match"
                ),
            }
        )
        haystacks[question_id] = [f"trajectory-{item}" for item in range(100)]
    while len(rows) < 451:
        index = len(rows)
        question_id = f"filler-{index}"
        rows.append(
            {
                "id": question_id,
                "domain": "web",
                "environment": "unselected-environment",
                "question_type": "static-environment",
                "question": "Question",
                "image": None,
                "answer": "Answer",
                "eval_function": "norm_phrase_set_match",
            }
        )
        haystacks[question_id] = [f"trajectory-{item}" for item in range(100)]
    (data_root / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (data_root / "haystacks" / "lme_v2_small.json").write_text(json.dumps(haystacks), encoding="utf-8")

    selection = build_selection(data_root, "test-seed")

    assert [row["question_id"] for row in selection["questions"]] == expected_ids
    assert [(row["question_type"], row["domain"], row["environment"]) for row in selection["questions"]] == list(
        SELECTION_SLOTS
    )
    assert [row["domain"] for row in selection["questions"] if row["has_question_image"]] == [
        "web",
        "enterprise",
    ]


def test_cost_projection_replaces_completed_reserves_with_receipted_cost() -> None:
    receipts = [{"costs_usd": {"total_api_equivalent": 0.04}}]
    pending = [({}, "native", "n"), ({}, "thinharness", "t")]

    projected = projected_total(receipts, pending, {"native": 0.08, "thinharness": 0.10})

    assert projected == 0.22


def test_reader_recovery_honors_retry_after_milliseconds_before_seconds() -> None:
    assert parse_retry_after({"retry-after-ms": "2500", "retry-after": "9"}) == 2.5


def test_reader_recovery_uses_capped_deterministic_backoff_without_header() -> None:
    assert [fallback_delay(attempt) for attempt in range(1, 8)] == [5, 10, 20, 40, 80, 120, 120]


def test_reader_recovery_writes_one_valid_jsonl_record(tmp_path: Path) -> None:
    path = tmp_path / "record.jsonl"

    write_jsonl_record(path, {"question_id": "q1", "score": 1.0})

    assert path.read_text(encoding="utf-8").splitlines() == ['{"question_id": "q1", "score": 1.0}']
