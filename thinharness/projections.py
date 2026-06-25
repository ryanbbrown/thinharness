"""Shared projections from neutral model turns and transcript entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .events import StreamToolCall
from .providers import (
    AssistantEntry,
    ModelNotice,
    ModelTurn,
    ToolOutput,
    ToolResultEntry,
    TranscriptEntry,
    UserEntry,
    append_notices_to_text,
    render_model_notices,
)
from .types import Json

ModelRequestKind = Literal["start", "resume", "tool_outputs", "approval_resume", "correction", "output_retry_tool"]


@dataclass(frozen=True)
class ModelRequestDelta:
    """Exact model-visible input entries for one provider request."""

    kind: ModelRequestKind
    entries: list[TranscriptEntry]
    notices: list[ModelNotice] = field(default_factory=list)
    structured_output: str | None = None


def model_request_delta_from_prompt(
    *,
    kind: Literal["start", "resume", "correction"],
    prompt: str,
    notices: list[ModelNotice],
    structured_output: str | None,
) -> ModelRequestDelta:
    """Build a request delta for a user-text provider continuation."""
    return ModelRequestDelta(
        kind=kind,
        entries=[UserEntry(content=append_notices_to_text(prompt, notices))],
        notices=list(notices),
        structured_output=structured_output,
    )


def model_request_delta_from_tool_outputs(
    *,
    kind: Literal["tool_outputs", "approval_resume", "output_retry_tool"],
    outputs: list[ToolOutput],
    notices: list[ModelNotice],
    structured_output: str | None,
) -> ModelRequestDelta:
    """Build a request delta for a tool-output provider continuation."""
    entries: list[TranscriptEntry] = [
        ToolResultEntry(call_id=output.call_id, output=output.output)
        for output in outputs
    ]
    if notice_text := render_model_notices(notices):
        entries.append(UserEntry(content=notice_text, notice=True))
    return ModelRequestDelta(
        kind=kind,
        entries=entries,
        notices=list(notices),
        structured_output=structured_output,
    )


def trace_input_messages_from_entries(entries: list[TranscriptEntry]) -> list[Json]:
    """Project neutral transcript entries into OTel-style input messages."""
    messages: list[Json] = []
    for entry in entries:
        if isinstance(entry, UserEntry):
            messages.append({"role": "user", "parts": [{"type": "text", "content": entry.content}]})
        elif isinstance(entry, ToolResultEntry):
            messages.append({
                "role": "tool",
                "parts": [{
                    "type": "tool_result",
                    "id": entry.call_id,
                    "content": entry.output,
                }],
            })
        else:
            messages.extend(trace_output_messages_from_assistant(entry))
    return messages


def trace_output_messages_from_assistant(entry_or_turn: AssistantEntry | ModelTurn) -> list[Json]:
    """Project a neutral assistant entry or model turn into OTel-style output messages."""
    parts: list[Json] = []
    for reasoning in entry_or_turn.reasoning:
        # OTel GenAI `thinking` part carries only text; opaque signatures/blobs are never traced.
        if reasoning.text:
            parts.append({"type": "thinking", "content": reasoning.text})
    if entry_or_turn.text:
        parts.append({"type": "text", "content": entry_or_turn.text})
    for call in entry_or_turn.tool_calls:
        parts.append({
            "type": "tool_call",
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
        })
    return [{"role": "assistant", "parts": parts}]


def model_request_input_from_delta(delta: ModelRequestDelta) -> Json | None:
    """Return the trace display payload for one model-visible request delta."""
    if len(delta.entries) == 1 and isinstance(delta.entries[0], UserEntry):
        content = delta.entries[0].content
        if delta.kind in {"start", "resume"}:
            return {"prompt": content}
        if delta.kind == "correction":
            return {"correction": content}

    tool_outputs = [
        {"call_id": entry.call_id, "output": entry.output}
        for entry in delta.entries
        if isinstance(entry, ToolResultEntry)
    ]
    if tool_outputs and len(tool_outputs) == len(delta.entries):
        return {"tool_outputs": tool_outputs}
    if delta.entries:
        return {"messages": trace_input_messages_from_entries(delta.entries)}
    return None


def stream_tool_calls_from_assistant(entry_or_turn: AssistantEntry | ModelTurn) -> tuple[StreamToolCall, ...]:
    """Project assistant tool calls into public stream-event summaries."""
    return tuple(StreamToolCall(id=call.id, name=call.name) for call in entry_or_turn.tool_calls)
