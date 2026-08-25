# ruff: noqa: E501

"""Build the ThinHarness query prompt from the current native V2 prompt."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptReplacement:
    scope: str
    old: str
    new: str
    reason: str
    expected_occurrences: int = 1


QUERY_PROMPT_REPLACEMENTS = (
    PromptReplacement(
        "query_prompt",
        "Read the local files in this directory, especially INSTRUCTION.md and question.json.",
        "Follow the aligned INSTRUCTION.md text below and read the question at the end of this prompt.",
        "ThinHarness receives the instruction and question inline instead of through sandbox files.",
    ),
    PromptReplacement(
        "query_prompt",
        "The local trajectories/ directory contains the current haystack for this evaluation item, and you must explore trajectories/ before returning your final result.",
        "The local corpus/ directory contains the current haystack for this evaluation item, and you must explore corpus/ before returning your final result.",
        "The unchanged ThinHarness corpus is rooted at corpus/.",
    ),
    PromptReplacement(
        "query_prompt",
        "If question.json refers to an image, view it carefully.",
        "If the question has an attached image, view it carefully.",
        "ThinHarness attaches the question image directly instead of naming it in question.json.",
    ),
    PromptReplacement(
        "query_prompt",
        "Write your final result to memory_module_output.json as valid JSON.",
        "Return your final result through the configured structured output as valid JSON.",
        "ThinHarness uses native structured output instead of a writable output file.",
    ),
    PromptReplacement(
        "query_prompt",
        "Use the local inspection helper under scripts/ when you need to inspect one trajectory, one state, one span, or match text within one trajectory quickly.",
        "Use read and search for bounded summary exploration and jsonl_search on one shortlisted trajectory when you need to inspect one state, one span, or matching text quickly.",
        "ThinHarness exposes read and jsonl_search instead of native shell and inspect_trajectory.py.",
    ),
)


INSTRUCTION_REPLACEMENTS = (
    PromptReplacement(
        "instruction",
        "The question is in `question.json`. You need to aggregate information from the local `trajectories/` directory.",
        "The question is at the end of this prompt. You need to aggregate information from the local `corpus/` directory.",
        "ThinHarness receives the question inline and keeps its unchanged corpus under corpus/.",
    ),
    PromptReplacement(
        "instruction",
        "Write your final result to `memory_module_output.json` as valid JSON with this exact schema:",
        "Return your final result through the configured structured output as valid JSON with this exact schema:",
        "ThinHarness uses native structured output and has no write tool in this experiment.",
    ),
    PromptReplacement(
        "instruction",
        "First, `trajectories/` is already organized in a fixed way.",
        "First, `corpus/` is already organized in a fixed way.",
        "The unchanged ThinHarness corpus root differs from the native sandbox root.",
    ),
    PromptReplacement(
        "instruction",
        "- `trajectories/<trajectory_id>/` contains one full session.\n- `trajectories/<trajectory_id>/trajectory.json` is the main file for that session.\n- `trajectories/<trajectory_id>/screenshots/` contains the screenshots referenced by that session.",
        "- `corpus/trajectories/<trajectory_id>/` contains one full session.\n- `corpus/trajectories/<trajectory_id>/trajectory.json` is the main file for that session.\n- Selected trajectory spans send their referenced screenshots to the downstream reader.",
        "ThinHarness keeps per-trajectory text below corpus/trajectories and does not expose trajectory screenshots through its query tools.",
    ),
    PromptReplacement(
        "instruction",
        "  - `text`: the main current state dump axtree\n  - `screenshot`: the screenshot for the current state corresponding to the axtree like `screenshots/0007.png`\n  - `thoughts`: short reasoning / note text\n  - `action`: next action taken after observing the state",
        "  - `accessibility_tree`: the main current state dump axtree\n  - `screenshot`: the source screenshot reference for the current state\n  - `thought`: short reasoning / note text\n  - `action`: next action taken after observing the state\n  - `action_annotated`: the action with its referenced AXTree element appended when available",
        "ThinHarness preserves the data but uses its existing normalized field names and action annotation.",
    ),
    PromptReplacement(
        "instruction",
        "Luckily I have rendered a summary of each of the existing trajectories in `trajectories/TRAJECTORY_SUMMARY_CONCISE.md` (This one has a quick high-level overview of each trajectory so you can get oriented fast and later you can use `trajectories/TRAJECTORY_SUMMARY_FULL.md` which has the detailed thought/action sequence for shortlist selection and exact verification).",
        "Luckily I have rendered a summary of each of the existing trajectories in `corpus/TRAJECTORY_SUMMARY_CONCISE.md` (This one has a quick high-level overview of each trajectory so you can get oriented fast and later you can use `corpus/TRAJECTORY_SUMMARY_FULL.md` which has the detailed thought/action sequence for shortlist selection and exact verification).",
        "ThinHarness stores the unchanged summary forms at the corpus root.",
    ),
    PromptReplacement(
        "instruction",
        "- If `question.json` contains an image, inspect that image.",
        "- If the question has an attached image, inspect that image.",
        "ThinHarness attaches the question image directly.",
    ),
    PromptReplacement(
        "instruction",
        "- Start from `trajectories/TRAJECTORY_SUMMARY_FULL.md` and shortlist only a few likely trajectories using the goal, start URL, action sequence, and final reward.",
        "- Start from `corpus/TRAJECTORY_SUMMARY_FULL.md` and shortlist only a few likely trajectories using the goal, start URL, action sequence, and final reward.",
        "ThinHarness stores the full summary at the corpus root.",
    ),
    PromptReplacement(
        "instruction",
        "- After shortlisting, do not read raw `trajectory.json` unless necessary. Prefer the helper script for quick inspection:\n  - `python scripts/inspect_trajectory.py <trajectory_id>` for a compact trajectory summary\n  - `python scripts/inspect_trajectory.py <trajectory_id> --state 7` for one exact state\n  - `python scripts/inspect_trajectory.py <trajectory_id> --span 6:8` for a short contiguous span\n  - `python scripts/inspect_trajectory.py <trajectory_id> --match \"Delete Review|Previous\"` to find matching states within one candidate trajectory\n- Use the helper only on shortlisted trajectories. It is for exact verification, not for broad rediscovery.\n  - IMPORTANT: avoid using rg or find over the raw `trajectory.json`. This is extremely time consuming and will give you too much context to consume, defeating the purpose of a fast retrieval.",
        "- After shortlisting, do not read raw `trajectory.json` unless necessary. Prefer focused ThinHarness tools for quick inspection:\n  - Use `read` with bounded `offset`, `limit`, and `max_chars`, and use `search` only within summary files for shortlist terms.\n  - Use `jsonl_search` on `corpus/trajectories/<trajectory_id>/states.jsonl` with a `where` filter for one exact state.\n  - Use `jsonl_search` on that per-trajectory states file with an `in` filter for a short contiguous span.\n  - Use `jsonl_search` with `field_searches` on that per-trajectory states file to match AXTree text within one candidate trajectory.\n- Use these tools only on shortlisted trajectories. They are for exact verification, not for broad rediscovery.\n  - IMPORTANT: avoid broad `search` or `jsonl_search` over raw trajectory files or global JSONL files. This is extremely time consuming and will give you too much context to consume, defeating the purpose of a fast retrieval.",
        "ThinHarness has typed read and jsonl_search tools instead of shell and inspect_trajectory.py; the native shortlist-first policy and stop rule remain unchanged.",
    ),
    PromptReplacement(
        "instruction",
        "- You may write scratch files in the current directory if needed.",
        "- Use bounded tool results as scratch context; no write tool is available.",
        "The fixed ThinHarness tool set has no write tool.",
    ),
)


REMAINING_DIFFERENCES = (
    {
        "difference": "ThinHarness system and filesystem-plugin instructions remain active in addition to the aligned query prompt.",
        "justification": "Changing the system prompt or plugin instructions would violate the prompt-only experiment scope.",
    },
    {
        "difference": "ThinHarness receives the question and native instruction inline instead of reading question.json and INSTRUCTION.md through a shell call.",
        "justification": "The fixed ThinHarness tool surface has no native sandbox instruction files or shell tool.",
    },
    {
        "difference": "ThinHarness uses read and jsonl_search examples instead of shell and inspect_trajectory.py examples.",
        "justification": "Only the existing ThinHarness tools can be named; tool schemas and implementations remain unchanged.",
    },
    {
        "difference": "ThinHarness returns the same JSON shape through native structured output instead of writing memory_module_output.json.",
        "justification": "The experiment must keep ThinHarness structured output and its fixed no-write tool set.",
    },
    {
        "difference": "Corpus paths and normalized state field names match the existing ThinHarness corpus.",
        "justification": "Corpus files and data exposure are fixed variables in this experiment.",
    },
    {
        "difference": "The question image remains attached directly to the ThinHarness request.",
        "justification": "Image delivery is explicitly held constant in this experiment.",
    },
)


def _apply_replacements(text: str, replacements: tuple[PromptReplacement, ...]) -> tuple[str, list[dict[str, Any]]]:
    aligned = text
    applied: list[dict[str, Any]] = []
    for replacement in replacements:
        count = aligned.count(replacement.old)
        if count != replacement.expected_occurrences:
            raise RuntimeError(
                f"Native {replacement.scope} fragment count differs: "
                f"expected {replacement.expected_occurrences}, got {count}: {replacement.old!r}"
            )
        aligned = aligned.replace(replacement.old, replacement.new)
        applied.append({**asdict(replacement), "source_occurrences": count})
    return aligned, applied


def align_native_sources(native_query_prompt: str, native_instruction: str) -> tuple[str, str]:
    aligned_query, _ = _apply_replacements(native_query_prompt, QUERY_PROMPT_REPLACEMENTS)
    aligned_instruction, _ = _apply_replacements(native_instruction, INSTRUCTION_REPLACEMENTS)
    return aligned_query, aligned_instruction


def build_aligned_query_prompt(
    query: str,
    *,
    native_query_prompt: str,
    native_instruction: str,
) -> str:
    aligned_query, aligned_instruction = align_native_sources(native_query_prompt, native_instruction)
    return (
        aligned_query.strip()
        + "\n\n"
        + aligned_instruction.strip()
        + "\n\n# Question\n\n"
        + query.strip()
        + "\n"
    )


def prompt_diff_artifact(native_query_prompt: str, native_instruction: str) -> dict[str, Any]:
    aligned_query, query_replacements = _apply_replacements(native_query_prompt, QUERY_PROMPT_REPLACEMENTS)
    aligned_instruction, instruction_replacements = _apply_replacements(native_instruction, INSTRUCTION_REPLACEMENTS)
    return {
        "schema_version": 1,
        "method": "Apply only the ordered localized replacements below to the current native V2 query prompt and INSTRUCTION.md.",
        "native_query_prompt_sha256": hashlib.sha256(native_query_prompt.encode()).hexdigest(),
        "native_instruction_sha256": hashlib.sha256(native_instruction.encode()).hexdigest(),
        "aligned_query_prompt_sha256": hashlib.sha256(aligned_query.encode()).hexdigest(),
        "aligned_instruction_sha256": hashlib.sha256(aligned_instruction.encode()).hexdigest(),
        "query_prompt_replacements": query_replacements,
        "instruction_replacements": instruction_replacements,
        "remaining_differences": list(REMAINING_DIFFERENCES),
        "unchanged_native_contracts": [
            "task overview and role",
            "output JSON field names and 20-state inclusive evidence limit",
            "quick triage before detailed trajectory inspection",
            "full-summary shortlist before exact verification",
            "direct lookup, comparison, procedure, image, contradiction, and uncertainty guidance",
            "small evidence package and stop-after-support rules",
            "final reminder ordering",
        ],
    }
