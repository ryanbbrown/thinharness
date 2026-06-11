"""Shared leaf types for harness runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Json = dict[str, Any]
StopReason = Literal[
    "end_turn",
    "provider_error",
    "limit_reached",
    "error",
    "cancelled_by_hook",
    "cancelled",
    "output_validation_failed",
    "tool_retries_exceeded",
    "unexpected_model_behavior",
]
LimitNoticeKey = tuple[Literal["limit_warning"], Literal["model_requests", "tool_calls"], int]


@dataclass
class HarnessResult:
    """Final result returned by a harness run."""

    text: str
    output: Any | None = None
    responses: list[Json] = field(default_factory=list)
    tool_call_records: list[Json] = field(default_factory=list)
    usage: RunUsage = field(default_factory=lambda: RunUsage())
    stop_reason: StopReason = "end_turn"
    resume_state: dict[str, Any] | None = None


@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0
    output_retries: int = 0
    tool_retries: dict[str, int] = field(default_factory=dict)


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""


class UnexpectedModelBehavior(HarnessError):
    """Raised when the model returns an invalid tool/finalization pattern."""
