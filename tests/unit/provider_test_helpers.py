from __future__ import annotations

from thinharness import ModelNotice, RequestConstants, ToolResult
from thinharness.providers import StructuredOutputRequest, ToolOutput


def tool_output(call_id: str, content: str) -> ToolOutput:
    """Return one normalized text tool output."""
    return ToolOutput(call_id, ToolResult(True, content))


def notice() -> ModelNotice:
    """Return a reusable test notice."""
    return ModelNotice(kind="limit_warning", content="Final request.", limit_kind="model_requests", remaining=1)


def notice_text() -> str:
    """Return rendered text for the reusable test notice."""
    return '<harness_notice kind="limit_warning">\nFinal request.\n</harness_notice>'


def constants(
    tools: list | None = None,
    *,
    instructions: str = "system",
    structured_output: StructuredOutputRequest | None = None,
) -> RequestConstants:
    """Return reusable per-run request constants."""
    return RequestConstants(instructions=instructions, tools=tools or [], structured_output=structured_output)


ECHO_TOOLS = [{"type": "function", "name": "echo", "description": "Echo", "parameters": {"type": "object", "properties": {}}}]
