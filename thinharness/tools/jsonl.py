"""JSONL search tool for structured line-delimited data."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from ..defaults import DEFAULT_JSONL_SEARCH_DESCRIPTION
from .base import (
    Json,
    PathPolicy,
    PathValidationError,
    StrictArgs,
    ToolResult,
    ToolSpec,
    _path_error,
    _timeout_error_message,
    coerce_args,
)
from .search_support import (
    _rg_error_message,
    _rg_partial_warning_metadata,
    parse_contained_rg_json,
    search_root_display_paths,
    validate_glob_selector,
)


class JsonlWhereFilter(StrictArgs):
    """One JSONL where filter."""

    field: str
    op: Literal["eq", "ne", "in", "contains", "regex", "exists"]
    value: str | None = None
    values: list[str] | None = None


class JsonlSearchArgs(StrictArgs):
    """Arguments for jsonl_search."""

    query: str = Field(default="", description="Optional ripgrep query. If omitted, scan all rows in scope.")
    path: str = Field(default=".", description="File, directory, or glob path to JSONL files. Directories are searched recursively.")
    fields: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict,
        description="Map of jq-style field path to max chars (0 = no truncation). If omitted, return the whole row.",
    )
    where: list[JsonlWhereFilter] = Field(default_factory=list, description="Filters AND-ed together.")
    max_files: int = Field(default=100, ge=1)
    max_matches_per_file: int = Field(default=25, ge=1)
    timeout: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


@dataclass
class _CandidateScan:
    """JSONL candidate rows plus scan metadata or a hard scan error."""

    candidates: Iterator[tuple[str, int, str]]
    metadata: Json
    error: ToolResult | None = None


class JsonlSearch:
    """Search JSONL files with optional ripgrep prefiltering and field projection."""

    def __init__(
        self,
        root: Path,
        read_policy: PathPolicy,
        *,
        max_tool_chars: int,
        rg_timeout: int,
        truncate: Callable[..., ToolResult],
    ) -> None:
        self.root = root
        self.read_policy = read_policy
        self.max_tool_chars = max_tool_chars
        self.rg_timeout = rg_timeout
        self._truncate = truncate

    def spec(self, *, instructions: str | None = None) -> ToolSpec:
        """Return the jsonl_search tool spec."""
        return ToolSpec(
            "jsonl_search",
            DEFAULT_JSONL_SEARCH_DESCRIPTION,
            JsonlSearchArgs,
            self.search,
            instructions=instructions,
        )

    def search(self, args: JsonlSearchArgs | Json) -> ToolResult:
        """Filter JSONL files with an optional ripgrep prefilter and structured field/where filtering."""
        args = coerce_args(args, JsonlSearchArgs)
        query = args.query
        path = args.path
        fields = args.fields
        where = [item.model_dump(exclude_none=True) for item in args.where]
        max_files = args.max_files
        max_matches_per_file = args.max_matches_per_file
        timeout = args.timeout or self.rg_timeout
        limit_chars = args.max_chars or self.max_tool_chars

        scan = self._candidates(query, path, timeout)
        if scan.error is not None:
            return scan.error

        shown: dict[str, list[tuple[int, Any]]] = {}
        row_counts: dict[str, int] = {}
        json_errors = 0
        total_files = 0
        total_rows = 0
        for path, line_number, line_text in scan.candidates:
            if not line_text.strip():
                continue
            try:
                row = json.loads(line_text)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            try:
                if not _where_passes(row, where):
                    continue
            except ValueError as exc:
                return ToolResult(False, f"invalid where filter: {exc}")
            if path not in row_counts:
                total_files += 1
                row_counts[path] = 0
            row_counts[path] += 1
            total_rows += 1
            if path in shown or len(shown) < max_files:
                shown.setdefault(path, [])
                if len(shown[path]) < max_matches_per_file:
                    shown[path].append((line_number, row))

        shown_files = sorted(shown.items())

        header = [
            "summary:",
            f"  query: {query or '(none)'}",
            f"  scope: path={path}",
        ]
        if where:
            header.append(f"  where: {_describe_where(where)}")
        if fields:
            header.append(f"  fields: {', '.join(fields)}")
        header.append(f"  files: {total_files} total, {len(shown_files)} shown")
        header.append(f"  rows_matched: {total_rows}")
        if json_errors:
            header.append(f"  json_parse_errors: {json_errors}")
        body = [""]
        for path, rows in shown_files:
            rows.sort(key=lambda lr: lr[0])
            body.append(path)
            for line_number, row in rows:
                try:
                    projected = _project_fields(row, fields) if fields else row
                except ValueError as exc:
                    return ToolResult(False, f"invalid field path: {exc}")
                body.append(f"  {line_number}: {json.dumps(projected, ensure_ascii=False, default=str)}")
            omitted_rows = row_counts[path] - len(rows)
            if omitted_rows:
                body.append(f"  ... {omitted_rows} more row(s)")
        if total_files > max_files:
            body.append(f"note: {total_files - max_files} more file(s) omitted")

        result = self._truncate("\n".join(header + body), prefix="jsonl_search", max_chars=limit_chars)
        result.metadata.update(scan.metadata)
        return result

    def _candidates(self, query: str, path: str, timeout: int) -> _CandidateScan:
        """Collect (path, line_number, line_text) tuples for jsonl_search."""
        if query:
            try:
                search_roots, glob_filter = self._query_scope(path)
            except PathValidationError as exc:
                return _CandidateScan(iter(()), {}, _path_error(exc))
            command = ["rg", "--json"]
            if glob_filter is not None:
                command.extend(["--glob", glob_filter])
            command.extend(["--", query, *search_roots])
            if not search_roots:
                return _CandidateScan(iter(()), {"returncode": 1, "cmd": command})
            try:
                proc = subprocess.run(
                    command,
                    cwd=self.root,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return _CandidateScan(iter(()), {}, ToolResult(False, _timeout_error_message("ripgrep", timeout), {"timeout": timeout, "cmd": command}))
            files = parse_contained_rg_json(proc.stdout or "", self.root, self.read_policy)
            if proc.returncode not in (0, 1) and not files:
                return _CandidateScan(
                    iter(()),
                    {},
                    ToolResult(False, _rg_error_message(proc.returncode, proc.stdout), {"returncode": proc.returncode, "cmd": command}),
                )
            metadata: Json = {"returncode": proc.returncode, "cmd": command}
            if proc.returncode not in (0, 1):
                metadata.update(_rg_partial_warning_metadata(proc.returncode, proc.stdout, include_match_events=False))
            rows = sorted(
                (
                    (file.path, match.line_number, match.line_text)
                    for file in files
                    if _is_jsonl_path(file.path)
                    for match in file.matches
                ),
                key=lambda item: (item[0], item[1]),
            )
            return _CandidateScan(iter(rows), metadata)
        try:
            paths = self._jsonl_paths(path)
        except PathValidationError as exc:
            return _CandidateScan(iter(()), {}, _path_error(exc))
        return _CandidateScan(self._iter_jsonl_rows(paths), {})

    def _query_scope(self, path: str) -> tuple[list[str], str | None]:
        """Return ripgrep roots and optional glob for a JSONL search path."""
        if path in {"", "."}:
            return search_root_display_paths(self.root, self.read_policy), "**/*.jsonl"
        if _has_glob(path):
            validate_glob_selector(path, field="path")
            return search_root_display_paths(self.root, self.read_policy), path
        resolved = self.read_policy.resolve(path)
        if not resolved.exists():
            return [], None
        display = _display_path(self.root, resolved)
        if resolved.is_file():
            return ([display], None) if _is_jsonl_path(display) else ([], None)
        return [display], "**/*.jsonl"

    def _iter_jsonl_rows(self, paths: list[Path]) -> Iterator[tuple[str, int, str]]:
        """Yield JSONL candidate rows without accumulating them."""
        for item in sorted(paths, key=lambda file_path: str(file_path)):
            resolved = item.resolve()
            if not self.read_policy.allows(resolved) or not item.is_file() or not _is_jsonl_path(item.name):
                continue
            rel = str(resolved.relative_to(self.root))
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line_text in enumerate(handle, start=1):
                    yield rel, line_number, line_text.rstrip("\n")

    def _jsonl_paths(self, path: str) -> list[Path]:
        """Return JSONL file paths for a file, directory, or glob selector."""
        if path in {"", "."}:
            paths: list[Path] = []
            for root in self.read_policy.existing_search_roots():
                if root.is_file():
                    if _is_jsonl_path(root.name):
                        paths.append(root)
                    continue
                paths.extend(item for item in root.rglob("*.jsonl") if item.is_file())
            return paths
        if _has_glob(path):
            validate_glob_selector(path, field="path")
            return [
                item
                for item in self.root.glob(path)
                if item.is_file() and _is_jsonl_path(item.name) and self.read_policy.allows(item.resolve())
            ]
        resolved = self.read_policy.resolve(path)
        if not resolved.exists():
            return []
        if resolved.is_file():
            if _is_jsonl_path(resolved.name):
                return [resolved]
            return []
        return [item for item in resolved.rglob("*.jsonl") if item.is_file()]


def _has_glob(path: str) -> bool:
    """Return whether a path value uses glob syntax."""
    return any(char in path for char in "*?[")


def _is_jsonl_path(path: str) -> bool:
    """Return whether a display path or filename points at a JSONL file."""
    return Path(path).suffix.lower() == ".jsonl"


def _display_path(root: Path, path: Path) -> str:
    """Return a workspace-relative display path when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


_MISSING = object()
_PATH_TOKEN = re.compile(
    r'\.?(?:([A-Za-z_][A-Za-z0-9_]*)|\[(-?\d+)\]|\["([^"]*)"\]|\[\'([^\']*)\'\])'
)


def _parse_jq_path(path: str) -> list[str | int]:
    """Parse a jq-compatible path: .foo, .foo.bar, .foo[0], .foo["weird key"]."""
    segments: list[str | int] = []
    i = 0
    while i < len(path):
        match = _PATH_TOKEN.match(path, i)
        if not match or match.end() == i:
            raise ValueError(f"invalid jq path: {path!r} at position {i}")
        ident, idx, qkey, sqkey = match.groups()
        if ident is not None:
            segments.append(ident)
        elif idx is not None:
            segments.append(int(idx))
        elif qkey is not None:
            segments.append(qkey)
        elif sqkey is not None:
            segments.append(sqkey)
        i = match.end()
    if not segments:
        raise ValueError(f"empty path: {path!r}")
    return segments


def _get_field_by_path(obj: Any, segments: list[str | int]) -> Any:
    """Walk a parsed path through obj; return _MISSING if it doesn't resolve."""
    cur: Any = obj
    for segment in segments:
        if isinstance(segment, int) and isinstance(cur, list):
            if -len(cur) <= segment < len(cur):
                cur = cur[segment]
                continue
            return _MISSING
        if isinstance(segment, str) and isinstance(cur, dict):
            if segment in cur:
                cur = cur[segment]
                continue
            return _MISSING
        return _MISSING
    return cur


def _apply_where_op(value: Any, op: str, target: Any, targets: Any) -> bool:
    """Evaluate one where operator against a resolved field value."""
    if op == "exists":
        return value is not _MISSING and value is not None
    if value is _MISSING or value is None:
        return False
    rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if op == "eq":
        return rendered == str(target)
    if op == "ne":
        return rendered != str(target)
    if op == "in":
        return rendered in [str(item) for item in (targets or [])]
    if op == "contains":
        return str(target) in rendered
    if op == "regex":
        return re.search(str(target), rendered) is not None
    raise ValueError(f"unknown op: {op!r}")


def _where_passes(row: Any, where: list[Json]) -> bool:
    """Return True if a row passes every where filter (AND)."""
    for filt in where:
        field = filt.get("field")
        op = filt.get("op")
        if not field or not op:
            raise ValueError(f"where filter missing field or op: {filt}")
        if op in {"eq", "ne", "contains", "regex"} and "value" not in filt:
            raise ValueError(f"op {op!r} requires 'value'")
        if op == "in" and "values" not in filt:
            raise ValueError("op 'in' requires 'values'")
        value = _get_field_by_path(row, _parse_jq_path(field))
        if not _apply_where_op(value, op, filt.get("value"), filt.get("values")):
            return False
    return True


def _project_fields(row: Any, fields: dict[str, Any]) -> Json:
    """Project requested fields with per-field max_chars truncation."""
    result: Json = {}
    for key, max_chars in fields.items():
        value = _get_field_by_path(row, _parse_jq_path(key))
        if value is _MISSING:
            result[key] = None
            continue
        limit = int(max_chars) if isinstance(max_chars, (int, float)) else 0
        if limit > 0:
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            result[key] = rendered if len(rendered) <= limit else rendered[:limit] + "…"
        else:
            result[key] = value
    return result


def _describe_where(where: list[Json]) -> str:
    """Format where filters for the summary header."""
    parts = []
    for filt in where:
        op = filt.get("op", "")
        if op == "in":
            parts.append(f"{filt.get('field')} in {filt.get('values')}")
        elif op == "exists":
            parts.append(f"{filt.get('field')} exists")
        else:
            parts.append(f"{filt.get('field')} {op} {filt.get('value')!r}")
    return "; ".join(parts)
