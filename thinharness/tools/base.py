"""Shared tool contracts, path policies, schemas, and invocation helpers."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeGuard, TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from ..types import Json

ToolBackgroundMode = Literal["never", "always", "model"]
ToolHandler = Callable[[Any], Any | Awaitable[Any]]
T = TypeVar("T", bound=BaseModel)

@dataclass(frozen=True)
class ToolSpec:
    """A JSON-schema-described callable exposed to the model."""

    name: str
    description: str
    parameters: Json | type[BaseModel]
    handler: ToolHandler
    sequential: bool = False
    metadata: Json = field(default_factory=dict)
    max_retries: int | None = None
    instructions: str | None = None
    background: ToolBackgroundMode = "never"

    def __post_init__(self) -> None:
        """Validate per-tool retry configuration."""
        if self.background not in {"never", "always", "model"}:
            raise ValueError(f"unknown background mode: {self.background}")
        if self.sequential and self.background != "never":
            raise ValueError("sequential tools cannot run in background")
        if self.max_retries is not None and self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def response_tool(self, *, include_background: bool = False) -> Json:
        """Return an OpenAI Responses API function tool definition."""
        parameters = copy.deepcopy(tool_parameters(self.parameters))
        if include_background:
            _add_background_parameter(parameters)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def parse_args(self, args: Json) -> Any:
        """Validate and coerce tool arguments when backed by a Pydantic model."""
        if _is_args_model(self.parameters):
            return self.parameters.model_validate(args)
        return args


@dataclass
class ToolResult:
    """Structured result returned by built-in tools."""

    ok: bool
    content: str
    metadata: Json = field(default_factory=dict)

    def as_json(self) -> str:
        """Serialize the tool result for a function_call_output item."""
        return json.dumps(
            {"ok": self.ok, "content": self.content, "metadata": self.metadata},
            ensure_ascii=False,
            default=str,
        )


class ModelRetry(Exception):
    """Raised by a tool handler to ask the model to try again with a hint message."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

@dataclass(frozen=True)
class AllowedPath:
    """One resolved path allowed by a workspace path policy."""

    path: Path
    exact: bool = False


class PathValidationError(ValueError):
    """Raised when a tool path or selector escapes its allowed policy."""


class ArgumentShapeError(ValueError):
    """Raised when model-provided tool arguments are not a JSON object."""


class PathPolicy:
    """Resolve workspace paths and enforce an allowlist under root."""

    def __init__(self, root: Path, allowed_paths: Sequence[str | Path] | None, label: str) -> None:
        self.root = root
        self.label = label
        raw_paths = list(allowed_paths) if allowed_paths is not None else ["."]
        if not raw_paths:
            raise ValueError(f"{label}_paths must not be empty")
        self.allowed_paths = [self._allowed_path(raw) for raw in raw_paths]

    def resolve(self, raw: str | Path) -> Path:
        """Resolve a raw tool path and require it to be allowed."""
        resolved = _resolve_under_root(self.root, raw)
        if not self.allows(resolved):
            raise PathValidationError(f"path is outside allowed {self.label} paths: {raw}")
        return resolved

    def allows(self, path: Path) -> bool:
        """Return whether a resolved path is allowed by this policy."""
        resolved = path.resolve()
        if not _is_relative_to(resolved, self.root):
            return False
        for allowed in self.allowed_paths:
            if allowed.exact:
                if resolved == allowed.path:
                    return True
            elif resolved == allowed.path or allowed.path in resolved.parents:
                return True
        return False

    def existing_search_roots(self) -> list[Path]:
        """Return existing allow roots for commands that accept search paths."""
        return [allowed.path for allowed in self.allowed_paths if allowed.path.exists()]

    def _allowed_path(self, raw: str | Path) -> AllowedPath:
        """Normalize a configured allow path under the workspace root."""
        resolved = contained_path(self.root, raw)
        return AllowedPath(resolved, exact=resolved.exists() and resolved.is_file())

class StrictArgs(BaseModel):
    """Base class for tool arguments."""

    model_config = ConfigDict(extra="forbid")

def _prepare_args(spec: ToolSpec, raw_args: str | Json) -> str | Any:
    """Parse and validate raw tool arguments."""
    try:
        args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            raise ArgumentShapeError("tool arguments must be a JSON object")
    except json.JSONDecodeError as exc:
        return _retry_envelope("InvalidArguments", f"invalid JSON arguments: {exc}")
    except ArgumentShapeError as exc:
        return _retry_envelope("InvalidArguments", str(exc))
    if _is_args_model(spec.parameters):
        try:
            return spec.parameters.model_validate(args)
        except ValidationError as exc:
            return _retry_envelope(
                "ValidationError",
                _format_validation_errors(exc),
                errors=cast(list[Json], exc.errors(include_url=False, include_context=False)),
            )
    return args


def _normalize_result(result: Any) -> str:
    """Normalize a tool handler result to a structured JSON envelope."""
    if isinstance(result, ToolResult):
        return result.as_json()
    if isinstance(result, str):
        return ToolResult(True, result).as_json()
    return ToolResult(True, json.dumps(result, indent=2, sort_keys=True, default=str)).as_json()


def _retry_envelope(error_type: str, message: str, *, errors: list[Json] | None = None) -> str:
    """Return a failed tool envelope that asks the model to retry."""
    metadata: Json = {"error_type": error_type, "retry": True}
    if errors is not None:
        metadata["errors"] = errors
    return ToolResult(False, message, metadata).as_json()


def _format_validation_errors(error: ValidationError) -> str:
    """Format Pydantic validation errors as compact retry guidance."""
    lines = ["Invalid arguments:"]
    for item in error.errors():
        location = _format_error_location(item.get("loc", ()))
        message = item.get("msg", "Invalid value")
        if item.get("type") != "missing" and "input" in item:
            message = f"{message} (got {_format_error_input(item['input'])})"
        lines.append(f"- {location}: {message}")
    return "\n".join(lines)


def _format_error_location(location: Any) -> str:
    """Format a Pydantic error location as a dotted path."""
    if not location:
        return "<root>"
    if not isinstance(location, tuple):
        location = (location,)
    return ".".join(str(part) for part in location)


def _format_error_input(value: Any) -> str:
    """Return a small display value for invalid Pydantic input."""
    if isinstance(value, str):
        return repr(value)
    return type(value).__name__


def call_tool(spec: ToolSpec, raw_args: str | Json) -> str:
    """Invoke a sync tool handler and normalize the result to structured JSON."""
    args = _prepare_args(spec, raw_args)
    if isinstance(args, str):
        return args
    try:
        result = spec.handler(args)
    except ModelRetry as exc:
        return _retry_envelope("ModelRetry", exc.message)
    except Exception as exc:
        if getattr(exc, "_thinharness_strict_hook", False):
            raise
        return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).as_json()
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if close is not None:
            close()
        return ToolResult(
            False,
            "async handler requires harness execution",
            {"error_type": "AsyncHandlerInSyncContext"},
        ).as_json()
    return _normalize_result(result)


async def _invoke_tool(spec: ToolSpec, raw_args: str | Json) -> str:
    """Invoke a tool handler without blocking the event loop."""
    args = _prepare_args(spec, raw_args)
    if isinstance(args, str):
        return args
    try:
        if _is_async_callable(spec.handler):
            result = await spec.handler(args)
        else:
            worker = asyncio.create_task(asyncio.to_thread(spec.handler, args))
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                await asyncio.gather(worker, return_exceptions=True)
                raise
        if inspect.isawaitable(result):
            result = await result
    except ModelRetry as exc:
        return _retry_envelope("ModelRetry", exc.message)
    except Exception as exc:
        if getattr(exc, "_thinharness_strict_hook", False):
            raise
        return ToolResult(False, f"{type(exc).__name__}: {exc}", {"error_type": type(exc).__name__}).as_json()
    return _normalize_result(result)


def _is_async_callable(handler: ToolHandler) -> bool:
    """Return whether a handler is natively async."""
    obj: Any = handler
    while isinstance(obj, partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj) or (callable(obj) and inspect.iscoroutinefunction(obj.__call__))

def contained_path(root: Path, raw: str | Path) -> Path:
    """Resolve a path and require it to remain inside root."""
    return _resolve_under_root(root, raw)


def _resolve_under_root(root: Path, raw: str | Path) -> Path:
    """Resolve raw under root and require the result to stay inside root."""
    path = Path(raw).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not _is_relative_to(resolved, root):
        raise PathValidationError(f"path escapes root: {raw}")
    return resolved


def _path_error(exc: PathValidationError) -> ToolResult:
    """Return a structured tool result for path policy failures."""
    return ToolResult(False, str(exc), {"error_type": "PathValidationError"})


def tool_parameters(parameters: Json | type[BaseModel]) -> Json:
    """Return provider-ready JSON schema for a manual schema or Pydantic model."""
    if not _is_args_model(parameters):
        return cast(Json, parameters)
    schema = parameters.model_json_schema()
    schema = _inline_schema_refs(schema)
    _clean_schema(schema)
    schema.setdefault("type", "object")
    schema.setdefault("additionalProperties", False)
    return schema


def _add_background_parameter(schema: Json) -> None:
    """Add the model-facing background opt-in parameter to a copied schema."""
    schema.setdefault("type", "object")
    properties = schema.setdefault("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("background-capable tool schemas must have object properties")
    if "_background" in properties:
        raise ValueError("tool schema already defines reserved _background argument")
    properties["_background"] = {
        "type": "boolean",
        "description": "Start this tool in the background and continue other work; omit or set false for normal synchronous execution.",
        "default": False,
    }


def coerce_args(args: T | Json, model: type[T]) -> T:
    """Validate dict args when a built-in tool is called directly."""
    return args if isinstance(args, model) else model.model_validate(args)


def _is_args_model(value: Any) -> TypeGuard[type[BaseModel]]:
    """Return whether value is a Pydantic args model class."""
    return isinstance(value, type) and issubclass(value, BaseModel)


def _inline_schema_refs(schema: Json) -> Json:
    """Inline simple local $defs references produced by Pydantic."""
    defs = schema.pop("$defs", {})

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            ref = value.pop("$ref", None)
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                merged = dict(defs.get(name, {}))
                merged.update(value)
                value = merged
            for key, item in list(value.items()):
                value[key] = visit(item)
        elif isinstance(value, list):
            value = [visit(item) for item in value]
        return value

    return visit(schema)


def _clean_schema(schema: Any) -> None:
    """Remove Pydantic-only decoration and simplify optional-null fields."""
    if isinstance(schema, list):
        for item in schema:
            _clean_schema(item)
        return
    if not isinstance(schema, dict):
        return
    schema.pop("title", None)
    if "anyOf" in schema:
        non_null = [item for item in schema["anyOf"] if item.get("type") != "null"]
        if len(non_null) == 1:
            replacement = dict(non_null[0])
            replacement.update({key: value for key, value in schema.items() if key not in {"anyOf", "default"}})
            schema.clear()
            schema.update(replacement)
    for value in schema.values():
        _clean_schema(value)

def _timeout_error_message(command_name: str, timeout: int) -> str:
    """Return a compact timeout failure message."""
    return f"{command_name} timed out after {timeout}s"

def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether path is inside root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
