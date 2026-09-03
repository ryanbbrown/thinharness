"""Provider-neutral transcript and resume-state behavior."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from ..content import ImageBlock, NormalizedContent, TextBlock, content_from_json, content_to_json
from ..tools.base import Json, ToolResult
from ..types import HarnessError
from .base import Model, ModelSession, ModelToolCall, ModelTurn, ReasoningPart, ToolOutput


@dataclass
class AssistantEntry:
    """Provider-neutral assistant transcript entry."""

    text: str
    tool_calls: list[ModelToolCall]
    reasoning: list[ReasoningPart] = field(default_factory=list)


@dataclass
class UserEntry:
    """Provider-neutral user transcript entry."""

    content: NormalizedContent
    notice: bool = False


@dataclass
class ToolResultEntry:
    """Provider-neutral tool-result transcript entry."""

    call_id: str
    result: ToolResult
    wire_output: str | None = None

TranscriptEntry: TypeAlias = AssistantEntry | UserEntry | ToolResultEntry


_TRANSCRIPT_RESUME_KEYS = frozenset({"kind", "version", "origin_provider", "origin_model", "entries", "openai_items"})
_TRANSCRIPT_ENTRY_KEYS = {
    "assistant": frozenset({"role", "text", "tool_calls", "reasoning"}),
    "user": frozenset({"role", "content", "notice"}),
    "tool": frozenset({"role", "call_id", "ok", "content", "metadata", "wire_output"}),
}
_REASONING_PART_KEYS = frozenset({"text", "signature", "id", "provider_name", "provider_details"})
_TRANSCRIPT_VERSION = 5


def _validate_resume_state(state: dict[str, Any]) -> list[TranscriptEntry]:
    """Validate built-in provider resume state shape before any session mutation."""
    if not isinstance(state, dict):
        raise HarnessError("resume_from must be a dict")
    try:
        json.dumps(state)
    except (TypeError, ValueError) as exc:
        raise HarnessError("resume_from must be JSON-serializable") from exc
    if state.get("kind") != "transcript":
        raise HarnessError(f"resume_from kind {state.get('kind')!r} is not supported; regenerate resume_state")
    if state.get("version") != _TRANSCRIPT_VERSION:
        raise HarnessError(f"resume_from version {state.get('version')!r} is not supported; regenerate resume_state")
    unknown = set(state) - _TRANSCRIPT_RESUME_KEYS
    if unknown:
        raise HarnessError(f"resume_from has unknown keys: {sorted(unknown)!r}")
    for field_name, expected_type in {"origin_provider": str, "origin_model": str, "entries": list}.items():
        if field_name not in state:
            raise HarnessError(f"resume_from missing required field: {field_name!r}")
        if not isinstance(state[field_name], expected_type):
            raise HarnessError(f"resume_from field {field_name!r} has wrong type")
    if "openai_items" in state:
        if not isinstance(state["openai_items"], list):
            raise HarnessError("resume_from field 'openai_items' has wrong type")
        _validate_openai_items(state["openai_items"])
    return [_transcript_entry_from_dict(entry) for entry in state["entries"]]


def _validate_openai_items(items: list[Any]) -> None:
    if any(not isinstance(item, dict) or not isinstance(item.get("type"), str) for item in items):
        raise HarnessError("resume_from openai_items entries must be dicts with a string 'type'")
    unanswered: list[Any] = []
    for index, item in enumerate(items):
        item_type = item["type"]
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise HarnessError("resume_from openai_items function_call call_id must be a string")
            if call_id in unanswered:
                raise HarnessError("resume_from openai_items has duplicate unanswered function_call call_id")
            unanswered.append(call_id)
        elif item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                raise HarnessError("resume_from openai_items function_call_output call_id must be a string")
            if call_id not in unanswered:
                raise HarnessError("resume_from openai_items function_call_output has no unanswered function_call")
            unanswered.remove(call_id)
        elif item_type == "reasoning":
            if unanswered:
                raise HarnessError("resume_from openai_items has an unanswered function_call before reasoning")
            if index == len(items) - 1 or items[index + 1].get("type") in {"function_call_output"} or (
                items[index + 1].get("type") == "message" and items[index + 1].get("role") == "user"
            ):
                raise HarnessError("resume_from openai_items reasoning item has no valid following assistant item")
        elif item_type == "message" and item.get("role") == "user" and unanswered:
            raise HarnessError("resume_from openai_items has an unanswered function_call before a user message")


def _transcript_state(
    *,
    model: Model,
    origin_provider: str,
    entries: list[TranscriptEntry],
    openai_items: list[Json] | None = None,
) -> dict[str, Any]:
    """Return the neutral transcript resume envelope."""
    state = {
        "kind": "transcript",
        "version": _TRANSCRIPT_VERSION,
        "origin_provider": origin_provider,
        "origin_model": model.model,
        "entries": [_transcript_entry_to_dict(entry) for entry in entries],
    }
    if openai_items is not None:
        state["openai_items"] = copy.deepcopy(openai_items)
    return state


def _transcript_entry_to_dict(entry: TranscriptEntry) -> Json:
    if isinstance(entry, UserEntry):
        return {"role": "user", "content": content_to_json(entry.content), "notice": entry.notice}
    if isinstance(entry, ToolResultEntry):
        wire_output = entry.wire_output
        if wire_output == entry.result.to_json():
            wire_output = None
        return {"role": "tool", "call_id": entry.call_id, **entry.result.to_value(), "wire_output": wire_output}
    return {
        "role": "assistant",
        "text": entry.text,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in entry.tool_calls
        ],
        "reasoning": [_reasoning_part_to_dict(part) for part in entry.reasoning],
    }


def _reasoning_part_to_dict(part: ReasoningPart) -> Json:
    data: Json = {"text": part.text}
    if part.signature is not None:
        data["signature"] = part.signature
    if part.id is not None:
        data["id"] = part.id
    if part.provider_name is not None:
        data["provider_name"] = part.provider_name
    if part.provider_details is not None:
        data["provider_details"] = part.provider_details
    return data


def _transcript_entry_from_dict(value: Any) -> TranscriptEntry:
    if not isinstance(value, dict):
        raise HarnessError("resume_from entries must be dicts")
    role = value.get("role")
    if role not in _TRANSCRIPT_ENTRY_KEYS:
        raise HarnessError(f"resume_from entry role {role!r} is not supported")
    if set(value) != _TRANSCRIPT_ENTRY_KEYS[role]:
        raise HarnessError(f"resume_from entry {role!r} has wrong keys")
    if role == "user":
        if type(value["notice"]) is not bool:
            raise HarnessError("resume_from user entry has wrong type")
        try:
            content = content_from_json(value["content"], label="resume_from user content")
        except (TypeError, ValueError) as exc:
            raise HarnessError(str(exc)) from exc
        return UserEntry(content=content, notice=value["notice"])
    if role == "tool":
        if not isinstance(value["call_id"], str) or (value["wire_output"] is not None and not isinstance(value["wire_output"], str)):
            raise HarnessError("resume_from tool entry has wrong type")
        try:
            result = ToolResult.from_value({key: value[key] for key in ("ok", "content", "metadata")}, label="resume_from tool entry")
        except (TypeError, ValueError) as exc:
            raise HarnessError(str(exc)) from exc
        return ToolResultEntry(call_id=value["call_id"], result=result, wire_output=value["wire_output"])
    if not isinstance(value["text"], str) or not isinstance(value["tool_calls"], list) or not isinstance(value["reasoning"], list):
        raise HarnessError("resume_from assistant entry has wrong type")
    return AssistantEntry(
        text=value["text"],
        tool_calls=[_model_tool_call_from_dict(call) for call in value["tool_calls"]],
        reasoning=[_reasoning_part_from_dict(part) for part in value["reasoning"]],
    )


def _reasoning_part_from_dict(value: Any) -> ReasoningPart:
    if not isinstance(value, dict):
        raise HarnessError("resume_from reasoning part must be a dict")
    unknown = set(value) - _REASONING_PART_KEYS
    if unknown:
        raise HarnessError(f"resume_from reasoning part has unknown keys: {sorted(unknown)!r}")
    if not isinstance(value.get("text"), str):
        raise HarnessError("resume_from reasoning part text must be a string")
    for key in ("signature", "id", "provider_name"):
        if key in value and not isinstance(value[key], str):
            raise HarnessError(f"resume_from reasoning part {key!r} must be a string")
    return ReasoningPart(
        text=value["text"],
        signature=value.get("signature"),
        id=value.get("id"),
        provider_name=value.get("provider_name"),
        provider_details=value.get("provider_details"),
    )


def _model_tool_call_from_dict(value: Any) -> ModelToolCall:
    if not isinstance(value, dict) or set(value) != {"id", "name", "arguments"}:
        raise HarnessError("resume_from assistant tool call has wrong shape")
    if not isinstance(value["id"], str) or not isinstance(value["name"], str) or not isinstance(value["arguments"], str):
        raise HarnessError("resume_from assistant tool call has wrong type")
    return ModelToolCall(id=value["id"], name=value["name"], arguments=value["arguments"])


def _append_tool_results(transcript: list[TranscriptEntry], outputs: list[ToolOutput], notice_text: str) -> None:
    transcript.extend(
        ToolResultEntry(
            call_id=output.call_id,
            result=copy.deepcopy(output.result),
            wire_output=output.wire_output,
        )
        for output in outputs
    )
    if notice_text:
        transcript.append(UserEntry(content=(TextBlock(notice_text),), notice=True))


def session_image_blocks(session: ModelSession) -> list[ImageBlock]:
    """Return image blocks restored into a built-in model session."""
    transcript = getattr(session, "transcript", None)
    if not isinstance(transcript, list):
        return []
    images: list[ImageBlock] = []
    for entry in transcript:
        if isinstance(entry, UserEntry):
            images.extend(block for block in entry.content if isinstance(block, ImageBlock))
        elif isinstance(entry, ToolResultEntry):
            images.extend(block for block in entry.result.blocks if isinstance(block, ImageBlock))
    return images


def _append_assistant_turn(transcript: list[TranscriptEntry], turn: ModelTurn) -> None:
    transcript.append(
        AssistantEntry(
            text=turn.text,
            tool_calls=copy.deepcopy(turn.tool_calls),
            reasoning=copy.deepcopy(turn.reasoning),
        )
    )


def _validate_anthropic_tool_arguments(entries: list[TranscriptEntry]) -> None:
    for entry in entries:
        if isinstance(entry, AssistantEntry):
            for call in entry.tool_calls:
                try:
                    json.loads(call.arguments)
                except ValueError as exc:
                    raise HarnessError("resume_from assistant tool call arguments must be JSON for Anthropic resume") from exc


def _thinking_fallback(text: str) -> str:
    """Render reasoning text as a degraded cross-provider thinking block."""
    return f"<thinking>\n{text}\n</thinking>"
