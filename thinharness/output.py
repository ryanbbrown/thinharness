"""Structured output helpers for harness runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError

from .providers import StructuredOutputRequest
from .tools import Json, _clean_schema, _inline_schema_refs

OutputMode = Literal["auto", "native", "tool", "prompted"]
ResolvedOutputMode = Literal["native", "tool", "prompted", "text"]
FINAL_RESULT_TOOL_NAME = "final_result"


@dataclass(frozen=True)
class NativeOutput:
    """Request provider-native JSON-schema structured output."""

    output_type: Any


@dataclass(frozen=True)
class PromptedOutput:
    """Request prompt-instruction structured output."""

    output_type: Any


@dataclass(frozen=True)
class ToolStructuredOutput:
    """Request tool-call structured output."""

    output_type: Any


@dataclass(frozen=True)
class TextOutput:
    """Request plain text output in HarnessResult.output."""

    output_type: Any = str


OutputSpec = Any | NativeOutput | PromptedOutput | ToolStructuredOutput | TextOutput


class OutputValidationError(ValueError):
    """Raised when a model response does not match the output schema."""


@dataclass(frozen=True)
class OutputSchema:
    """Validate, serialize, and describe one structured output request."""

    output_type: Any
    mode: ResolvedOutputMode
    adapter: TypeAdapter[Any]
    schema: Json
    argument_schema: Json
    wraps_value: bool
    name: str = FINAL_RESULT_TOOL_NAME

    @classmethod
    def build(cls, output_type: OutputSpec, mode: OutputMode) -> OutputSchema:
        """Build an output schema from a configured output type and mode."""
        output_type, mode = resolve_output_spec(output_type, mode)
        if output_type is str or mode == "text":
            adapter = TypeAdapter(str)
            return cls(
                output_type=str,
                mode="text",
                adapter=adapter,
                schema={"type": "string"},
                argument_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
                wraps_value=True,
            )
        if mode == "auto":
            raise ValueError("output mode must be resolved before building OutputSchema")
        adapter = TypeAdapter(output_type)
        schema = _inline_schema_refs(adapter.json_schema())
        _clean_schema(schema)
        _set_object_additional_properties(schema)
        argument_schema, wraps_value = _as_arguments_schema(schema)
        return cls(
            output_type=output_type,
            mode=mode,
            adapter=adapter,
            schema=schema,
            argument_schema=argument_schema,
            wraps_value=wraps_value,
        )

    def synthetic_tools(self) -> list[Json]:
        """Return synthetic provider-neutral tool schemas."""
        if self.mode != "tool":
            return []
        return [{
            "type": "function",
            "name": self.name,
            "description": "Submit the final structured answer.",
            "parameters": self.argument_schema,
        }]

    def structured_output_request(self) -> StructuredOutputRequest | None:
        """Return native structured-output metadata for providers."""
        if self.mode != "native":
            return None
        return StructuredOutputRequest(name=self.name, schema=self.schema, strict=_schema_is_strict_compatible(self.schema))

    def build_instructions(self) -> str:
        """Return prompted-mode schema instructions."""
        schema = json.dumps(self.schema, ensure_ascii=False, sort_keys=True)
        return (
            "Return the final answer as JSON that validates against this JSON Schema. "
            "Do not include explanatory text outside the JSON value.\n\n"
            f"{schema}"
        )

    def validate_tool_arguments(self, arguments: str) -> Any:
        """Validate final_result tool-call arguments."""
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            raise OutputValidationError(f"invalid JSON arguments: {exc}") from exc
        if not isinstance(parsed, dict):
            raise OutputValidationError("final_result arguments must be a JSON object")
        value = parsed.get("value") if self.wraps_value else parsed
        try:
            return self.adapter.validate_python(value)
        except ValidationError as exc:
            raise OutputValidationError(str(exc)) from exc

    def validate_text(self, text: str) -> Any:
        """Validate final response text."""
        if self.mode == "text":
            return text
        stripped = strip_markdown_fences(text)
        try:
            return self.adapter.validate_json(stripped)
        except ValidationError as exc:
            raise OutputValidationError(str(exc)) from exc

    def dump(self, value: Any) -> str:
        """Serialize a validated output value for text boundaries."""
        dumped = self.adapter.dump_python(value, mode="json")
        return json.dumps(dumped, ensure_ascii=False, default=str)


def strip_markdown_fences(text: str) -> str:
    """Strip one surrounding Markdown code fence if present."""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def resolve_output_spec(output_type: OutputSpec, mode: OutputMode) -> tuple[Any, ResolvedOutputMode | OutputMode]:
    """Resolve output marker wrappers into a type and mode."""
    if isinstance(output_type, NativeOutput):
        return output_type.output_type, "native"
    if isinstance(output_type, PromptedOutput):
        return output_type.output_type, "prompted"
    if isinstance(output_type, ToolStructuredOutput):
        return output_type.output_type, "tool"
    if isinstance(output_type, TextOutput):
        return output_type.output_type, "text"
    return output_type, mode


def _as_arguments_schema(schema: Json) -> tuple[Json, bool]:
    """Return a function-argument schema for the output schema."""
    if schema.get("type") == "object" and "anyOf" not in schema and "oneOf" not in schema:
        argument_schema = dict(schema)
        argument_schema.setdefault("additionalProperties", False)
        return argument_schema, False
    return {
        "type": "object",
        "properties": {"value": schema},
        "required": ["value"],
        "additionalProperties": False,
    }, True


def _schema_is_strict_compatible(schema: Json) -> bool:
    """Return whether a schema is simple enough for strict native output."""
    if not _object_nodes_are_strict(schema):
        return False
    return _contains_object_node(schema)


def _set_object_additional_properties(schema: Any) -> None:
    """Set additionalProperties false on every object node."""
    if isinstance(schema, list):
        for item in schema:
            _set_object_additional_properties(item)
        return
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    for value in schema.values():
        _set_object_additional_properties(value)


def _object_nodes_are_strict(schema: Any) -> bool:
    """Return whether every object node is compatible with strict JSON schema."""
    if isinstance(schema, list):
        return all(_object_nodes_are_strict(item) for item in schema)
    if not isinstance(schema, dict):
        return True
    if schema.get("type") == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or set(properties) != set(required or []):
            return False
        if schema.get("additionalProperties") is not False:
            return False
    return all(_object_nodes_are_strict(value) for value in schema.values())


def _contains_object_node(schema: Any) -> bool:
    """Return whether a schema contains an object node."""
    if isinstance(schema, list):
        return any(_contains_object_node(item) for item in schema)
    if not isinstance(schema, dict):
        return False
    if schema.get("type") == "object":
        return True
    return any(_contains_object_node(value) for value in schema.values())
