from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.longmemeval_v2.freeze_selection import DOMAINS, QUESTION_TYPES, build_selection  # noqa: E402


def test_build_selection_covers_every_domain_type_cell_and_both_image_paths(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    (data_root / "haystacks").mkdir(parents=True)
    rows = []
    haystacks = {}
    for question_type in QUESTION_TYPES:
        for domain in DOMAINS:
            for candidate in range(2):
                question_id = f"{domain}-{question_type}-{candidate}"
                rows.append(
                    {
                        "id": question_id,
                        "domain": domain,
                        "environment": "test",
                        "question_type": question_type,
                        "question": "Question",
                        "image": "question.png" if question_type == "errors-gotchas" else None,
                        "answer": "Answer",
                        "eval_function": "norm_phrase_set_match|lower=true",
                    }
                )
                haystacks[question_id] = [f"trajectory-{index}" for index in range(100)]
    while len(rows) < 451:
        index = len(rows)
        question_id = f"filler-{index}"
        rows.append(
            {
                "id": question_id,
                "domain": "web",
                "environment": "test",
                "question_type": "static-environment",
                "question": "Question",
                "image": None,
                "answer": "Answer",
                "eval_function": "norm_phrase_set_match|lower=true",
            }
        )
        haystacks[question_id] = [f"trajectory-{item}" for item in range(100)]
    (data_root / "questions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (data_root / "haystacks" / "lme_v2_small.json").write_text(
        json.dumps(haystacks), encoding="utf-8"
    )

    selection = build_selection(data_root, "fixed-test-seed")

    selected = selection["questions"]
    assert len(selected) == 14
    assert [(row["question_type"], row["domain"]) for row in selected] == [
        (question_type, domain) for question_type in QUESTION_TYPES for domain in DOMAINS
    ]
    assert [(row["domain"], row["question_type"]) for row in selected if row["has_question_image"]] == [
        ("web", "errors-gotchas"),
        ("enterprise", "errors-gotchas"),
    ]
