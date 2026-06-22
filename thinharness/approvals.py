"""Human approval pause envelope helpers."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any

from .types import ApprovalDecision, HarnessError, Json, LimitNoticeKey, RunUsage

APPROVAL_ENVELOPE_KIND = "approval_pause"
APPROVAL_ENVELOPE_VERSION = 1

_ENVELOPE_KEYS = frozenset({
    "kind",
    "version",
    "provider_state",
    "batch",
    "approval_required_ids",
    "cancelled_background_task_ids",
    "ready_background_completion_messages",
    "usage",
    "responses",
    "tool_call_records",
    "emitted_limit_warnings",
    "metadata",
})
_BATCH_KEYS = frozenset({"id", "name", "arguments"})


@dataclass(frozen=True)
class ApprovalToolCall:
    """JSON-restored model tool call from an approval envelope."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ApprovalPause:
    """Validated approval pause envelope data.

    provider_state is the provider resume payload. Built-in providers store the
    full neutral transcript there; the approval envelope version is independent.
    """

    provider_state: Json
    batch: list[ApprovalToolCall]
    approval_required_ids: frozenset[str]
    cancelled_background_task_ids: list[str]
    ready_background_completion_messages: list[str]
    usage: RunUsage
    responses: list[Json]
    tool_call_records: list[Json]
    emitted_limit_warnings: set[LimitNoticeKey]
    metadata: Json


def build_approval_envelope(
    *,
    provider_state: Json,
    batch: list[Any],
    approval_required_ids: list[str],
    cancelled_background_task_ids: list[str],
    ready_background_completion_messages: list[str],
    usage: RunUsage,
    responses: list[Json],
    tool_call_records: list[Json],
    emitted_limit_warnings: set[LimitNoticeKey],
    metadata: Json,
) -> Json:
    """Build an isolated JSON approval pause envelope."""
    # Built-in provider_state contains the full neutral transcript. The outer
    # envelope version is unchanged because only the nested provider payload
    # changed; old nested payloads fail later when the provider resumes them.
    # Raw provider responses make the post-resume result a full logical-run result.
    # Both provider_state and responses make approval envelopes grow with run length.
    envelope: Json = {
        "kind": APPROVAL_ENVELOPE_KIND,
        "version": APPROVAL_ENVELOPE_VERSION,
        "provider_state": provider_state,
        "batch": [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in batch],
        "approval_required_ids": approval_required_ids,
        "cancelled_background_task_ids": cancelled_background_task_ids,
        "ready_background_completion_messages": ready_background_completion_messages,
        "usage": asdict(usage),
        "responses": responses,
        "tool_call_records": tool_call_records,
        "emitted_limit_warnings": list(emitted_limit_warnings),
        "metadata": metadata,
    }
    return json.loads(json.dumps(envelope))


def validate_approval_pause_state(state: dict[str, Any], *, label: str = "approval state") -> ApprovalPause:
    """Validate and normalize an approval pause envelope."""
    if not isinstance(state, dict):
        raise HarnessError(f"{label} must be a dict")
    if state.get("kind") != APPROVAL_ENVELOPE_KIND:
        raise HarnessError(f"{label} kind {state.get('kind')!r} does not match {APPROVAL_ENVELOPE_KIND!r}")
    if state.get("version") != APPROVAL_ENVELOPE_VERSION:
        raise HarnessError(f"{label} version {state.get('version')!r} is not supported")
    missing = set(_ENVELOPE_KEYS - set(state))
    missing.discard("ready_background_completion_messages")
    if missing:
        raise HarnessError(f"{label} missing required field: {sorted(missing)[0]!r}")
    unknown = set(state) - _ENVELOPE_KEYS
    if unknown:
        raise HarnessError(f"{label} has unknown keys: {sorted(unknown)!r}")
    try:
        isolated = json.loads(json.dumps(state))
    except (TypeError, ValueError) as exc:
        raise HarnessError(f"{label} must be JSON-serializable") from exc

    provider_state = _expect_dict(isolated["provider_state"], "provider_state", label)
    batch = _parse_batch(isolated["batch"], label)
    approval_required_ids = _parse_approval_ids(isolated["approval_required_ids"], batch, label)
    cancelled_ids = _parse_string_list(isolated["cancelled_background_task_ids"], "cancelled_background_task_ids", label)
    ready_messages = _parse_string_list(isolated.get("ready_background_completion_messages", []), "ready_background_completion_messages", label)
    responses = _parse_dict_list(isolated["responses"], "responses", label)
    records = _parse_dict_list(isolated["tool_call_records"], "tool_call_records", label)
    emitted = _parse_limit_notice_keys(isolated["emitted_limit_warnings"], label)
    metadata = _expect_dict(isolated["metadata"], "metadata", label)
    usage = _parse_usage(isolated["usage"], label)

    return ApprovalPause(
        provider_state=provider_state,
        batch=batch,
        approval_required_ids=approval_required_ids,
        cancelled_background_task_ids=cancelled_ids,
        ready_background_completion_messages=ready_messages,
        usage=usage,
        responses=responses,
        tool_call_records=records,
        emitted_limit_warnings=emitted,
        metadata=metadata,
    )


def validate_approval_decisions(decisions: list[ApprovalDecision], required_ids: frozenset[str]) -> dict[str, ApprovalDecision]:
    """Validate that host decisions exactly cover approval-required calls."""
    by_id: dict[str, ApprovalDecision] = {}
    for decision in decisions:
        if not isinstance(decision, ApprovalDecision):
            raise HarnessError("approval decisions must be ApprovalDecision objects")
        if type(decision.approved) is not bool:
            raise HarnessError(f"approval decision for call_id {decision.call_id!r} has non-bool approved value")
        if decision.reason is not None and not isinstance(decision.reason, str):
            raise HarnessError(f"approval decision for call_id {decision.call_id!r} has non-string reason")
        if decision.call_id in by_id:
            raise HarnessError(f"duplicate approval decision for call_id {decision.call_id!r}")
        by_id[decision.call_id] = decision
    unknown = set(by_id) - required_ids
    if unknown:
        raise HarnessError(f"unknown approval decision call_id {sorted(unknown)[0]!r}")
    missing = required_ids - set(by_id)
    if missing:
        raise HarnessError(f"missing approval decision for call_id {sorted(missing)[0]!r}")
    return by_id


def copy_restored_run_state(pause: ApprovalPause, run_metadata: Json | None) -> tuple[Json, RunUsage, list[Json], list[Json], set[LimitNoticeKey]]:
    """Return mutable copies of state restored for the resumed logical run."""
    metadata = copy.deepcopy(run_metadata if run_metadata is not None else pause.metadata)
    return (
        metadata,
        copy.deepcopy(pause.usage),
        copy.deepcopy(pause.responses),
        copy.deepcopy(pause.tool_call_records),
        set(pause.emitted_limit_warnings),
    )


def is_approval_pause_state(state: dict[str, Any] | None) -> bool:
    """Return whether a dict looks like a harness approval pause envelope."""
    return isinstance(state, dict) and state.get("kind") == APPROVAL_ENVELOPE_KIND


def _parse_batch(value: Any, label: str) -> list[ApprovalToolCall]:
    if not isinstance(value, list):
        raise HarnessError(f"{label} field 'batch' has wrong type")
    calls: list[ApprovalToolCall] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != _BATCH_KEYS:
            raise HarnessError(f"{label} field 'batch' has wrong shape")
        call_id = item["id"]
        name = item["name"]
        arguments = item["arguments"]
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, str):
            raise HarnessError(f"{label} field 'batch' has wrong type")
        if call_id in seen:
            raise HarnessError(f"{label} field 'batch' has duplicate call id {call_id!r}")
        seen.add(call_id)
        calls.append(ApprovalToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def _parse_approval_ids(value: Any, batch: list[ApprovalToolCall], label: str) -> frozenset[str]:
    ids = _parse_string_list(value, "approval_required_ids", label)
    if not ids:
        raise HarnessError(f"{label} field 'approval_required_ids' must not be empty")
    batch_ids = {call.id for call in batch}
    unknown = set(ids) - batch_ids
    if unknown:
        raise HarnessError(f"{label} field 'approval_required_ids' contains unknown id {sorted(unknown)[0]!r}")
    if len(set(ids)) != len(ids):
        raise HarnessError(f"{label} field 'approval_required_ids' contains duplicates")
    return frozenset(ids)


def _parse_usage(value: Any, label: str) -> RunUsage:
    usage = _expect_dict(value, "usage", label)
    retries = usage.get("tool_retries", {})
    if not isinstance(retries, dict) or not all(isinstance(key, str) and isinstance(val, int) for key, val in retries.items()):
        raise HarnessError(f"{label} field 'usage' has wrong type")
    fields = {
        "model_requests": usage.get("model_requests"),
        "tool_calls": usage.get("tool_calls"),
        "cancelled_tool_calls": usage.get("cancelled_tool_calls"),
        "output_retries": usage.get("output_retries"),
    }
    if not all(isinstance(value, int) for value in fields.values()):
        raise HarnessError(f"{label} field 'usage' has wrong type")
    model_requests = fields["model_requests"]
    tool_calls = fields["tool_calls"]
    cancelled_tool_calls = fields["cancelled_tool_calls"]
    output_retries = fields["output_retries"]
    assert isinstance(model_requests, int)
    assert isinstance(tool_calls, int)
    assert isinstance(cancelled_tool_calls, int)
    assert isinstance(output_retries, int)
    return RunUsage(
        model_requests=model_requests,
        tool_calls=tool_calls,
        cancelled_tool_calls=cancelled_tool_calls,
        output_retries=output_retries,
        tool_retries=dict(retries),
    )


def _parse_limit_notice_keys(value: Any, label: str) -> set[LimitNoticeKey]:
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


def _parse_string_list(value: Any, field_name: str, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise HarnessError(f"{label} field {field_name!r} has wrong type")
    return list(value)


def _parse_dict_list(value: Any, field_name: str, label: str) -> list[Json]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise HarnessError(f"{label} field {field_name!r} has wrong type")
    return [dict(item) for item in value]


def _expect_dict(value: Any, field_name: str, label: str) -> Json:
    if not isinstance(value, dict):
        raise HarnessError(f"{label} field {field_name!r} has wrong type")
    return dict(value)
