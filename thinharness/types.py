"""Shared leaf types for harness runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Json = dict[str, Any]
StopReason = Literal[
    "end_turn",
    "provider_error",
    "limit_reached",
    "approval_required",
    "error",
    "cancelled_by_hook",
    "cancelled",
    "output_validation_failed",
    "tool_retries_exceeded",
    "unexpected_model_behavior",
]
LimitNoticeKey = tuple[Literal["limit_warning"], Literal["model_requests", "tool_calls"], int]


@dataclass(frozen=True)
class PendingApproval:
    """One model-requested tool call awaiting a host decision."""

    call_id: str
    tool_name: str
    arguments: str


@dataclass(frozen=True)
class ApprovalDecision:
    """One host decision for a pending approval."""

    call_id: str
    approved: bool
    reason: str | None = None


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
    pending_approvals: list[PendingApproval] = field(default_factory=list)


@dataclass
class RunUsage:
    """Provider and tool usage for one harness run."""

    model_requests: int = 0
    tool_calls: int = 0
    cancelled_tool_calls: int = 0
    output_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_retries: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> Json:
        """Serialize run usage for a JSON envelope."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Any, *, label: str) -> RunUsage:
        """Validate and decode run usage from a JSON envelope.

        The four counter fields are required; token totals default to 0 so
        envelopes written before token accounting existed still resume; unknown
        keys are ignored.
        """
        if not isinstance(data, dict):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        usage = dict(data)
        retries = usage.get("tool_retries", {})
        if not isinstance(retries, dict) or not all(isinstance(key, str) and isinstance(val, int) for key, val in retries.items()):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        model_requests = usage.get("model_requests")
        tool_calls = usage.get("tool_calls")
        cancelled_tool_calls = usage.get("cancelled_tool_calls")
        output_retries = usage.get("output_retries")
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        counters = [model_requests, tool_calls, cancelled_tool_calls, output_retries, input_tokens, output_tokens]
        if not all(isinstance(value, int) for value in counters):
            raise HarnessError(f"{label} field 'usage' has wrong type")
        assert isinstance(model_requests, int)
        assert isinstance(tool_calls, int)
        assert isinstance(cancelled_tool_calls, int)
        assert isinstance(output_retries, int)
        assert isinstance(input_tokens, int)
        assert isinstance(output_tokens, int)
        return cls(
            model_requests=model_requests,
            tool_calls=tool_calls,
            cancelled_tool_calls=cancelled_tool_calls,
            output_retries=output_retries,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_retries=dict(retries),
        )


def limit_notice_keys_to_json(keys: set[LimitNoticeKey]) -> list[list[Any]]:
    """Encode emitted limit-notice keys for a JSON envelope."""
    return [list(key) for key in keys]


def limit_notice_keys_from_json(value: Any, *, label: str) -> set[LimitNoticeKey]:
    """Validate and decode emitted limit-notice keys from a JSON envelope."""
    if not isinstance(value, list):
        raise HarnessError(f"{label} field 'emitted_limit_warnings' has wrong type")
    keys: set[LimitNoticeKey] = set()
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or item[0] != "limit_warning"
            or item[1] not in {"model_requests", "tool_calls"}
            or not isinstance(item[2], int)
        ):
            raise HarnessError(f"{label} field 'emitted_limit_warnings' has wrong shape")
        keys.add((item[0], item[1], item[2]))
    return keys


class HarnessError(RuntimeError):
    """Raised when the harness cannot complete a run."""


class UnexpectedModelBehavior(HarnessError):
    """Raised when the model returns an invalid tool/finalization pattern."""
