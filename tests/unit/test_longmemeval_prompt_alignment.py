from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.longmemeval_v2.prompt_alignment import (  # noqa: E402
    INSTRUCTION_REPLACEMENTS,
    QUERY_PROMPT_REPLACEMENTS,
    build_aligned_query_prompt,
)
from benchmarks.longmemeval_v2.run_prompt_alignment import data_access_audit  # noqa: E402


def test_aligned_prompt_changes_only_declared_native_fragments() -> None:
    native_query = "\n".join(replacement.old for replacement in QUERY_PROMPT_REPLACEMENTS)
    native_instruction = "\n".join(replacement.old for replacement in INSTRUCTION_REPLACEMENTS)

    prompt = build_aligned_query_prompt(
        "Which control appears first?",
        native_query_prompt=native_query,
        native_instruction=native_instruction,
    )

    for replacement in (*QUERY_PROMPT_REPLACEMENTS, *INSTRUCTION_REPLACEMENTS):
        assert replacement.old not in prompt
        assert replacement.new in prompt
    assert prompt.endswith("# Question\n\nWhich control appears first?\n")


def test_aligned_prompt_rejects_changed_native_source() -> None:
    native_query = "\n".join(replacement.old for replacement in QUERY_PROMPT_REPLACEMENTS[:-1])
    native_instruction = "\n".join(replacement.old for replacement in INSTRUCTION_REPLACEMENTS)

    try:
        build_aligned_query_prompt(
            "Question",
            native_query_prompt=native_query,
            native_instruction=native_instruction,
        )
    except RuntimeError as exc:
        assert "Native query_prompt fragment count differs" in str(exc)
    else:
        raise AssertionError("changed native source was accepted")


def test_data_access_audit_distinguishes_direct_scans_visible_results_and_spills(tmp_path: Path) -> None:
    trace_dir = tmp_path / "query_traces" / "invocation" / "attempt_001"
    trace_dir.mkdir(parents=True)
    calls = [
        {
            "call": {
                "name": "search",
                "arguments": json.dumps({"query": "needle", "path": "corpus"}),
            },
            "result": {
                "ok": True,
                "content": "corpus/actions.jsonl\n  1: needle\n",
                "metadata": {
                    "saved_to_display": ".thinharness/outputs/search-result.txt",
                },
            },
        },
        {
            "call": {
                "name": "jsonl_search",
                "arguments": json.dumps(
                    {
                        "path": "corpus/trajectories/abc/states.jsonl",
                        "where": [],
                    }
                ),
            },
            "result": {
                "ok": True,
                "content": "corpus/trajectories/abc/states.jsonl\n  1: {}",
                "metadata": {},
            },
        },
        {
            "call": {
                "name": "read",
                "arguments": json.dumps({"path": ".thinharness/outputs/search-result.txt"}),
            },
            "result": {
                "ok": False,
                "content": "file not found",
                "metadata": {},
            },
        },
    ]
    (trace_dir / "tool_calls.json").write_text(json.dumps(calls), encoding="utf-8")

    audit = data_access_audit(tmp_path)

    assert audit["forms"]["global_jsonl"] == {
        "direct_target_calls": 0,
        "successful_direct_target_calls": 0,
        "broad_scope_scan_calls": 1,
        "model_visible_result_mentions": 1,
        "accessed": True,
        "model_visible": True,
    }
    assert audit["forms"]["per_trajectory_jsonl"]["direct_target_calls"] == 1
    assert audit["forms"]["raw_trajectory_json"]["broad_scope_scan_calls"] == 1
    assert audit["spill_artifacts_created"] == [".thinharness/outputs/search-result.txt"]
    assert audit["spill_artifacts_read"] == []
    assert len(audit["failed_tool_calls"]) == 1
