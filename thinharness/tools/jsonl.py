"""JSONL search tool for structured line-delimited data."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from .base import (
    Json,
    PathValidationError,
    StrictArgs,
    ToolResult,
    ToolSpec,
    _path_error,
    _rg_error_message,
    _timeout_error_message,
    coerce_args,
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
    path_glob: str = Field(default="**/*.jsonl", description="Glob filter; defaults to **/*.jsonl.")
    fields: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict,
        description="Map of jq-style field path to max chars (0 = no truncation). If omitted, return the whole row.",
    )
    where: list[JsonlWhereFilter] = Field(default_factory=list, description="Filters AND-ed together.")
    max_files: int = Field(default=10, ge=1)
    max_matches_per_file: int = Field(default=3, ge=1)
    timeout: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


class JsonlSearch:
    """Search JSONL files with optional ripgrep prefiltering and field projection."""

    def __init__(
        self,
        root: Path,
        *,
        max_tool_chars: int,
        rg_timeout: int,
        truncate: Callable[..., ToolResult],
        parse_rg_json: Callable[[str], list[Any]],
        path_allowed: Callable[[Path], bool],
        search_roots: Callable[[], list[str]],
    ) -> None:
        self.root = root
        self.max_tool_chars = max_tool_chars
        self.rg_timeout = rg_timeout
        self._truncate = truncate
        self._parse_rg_json = parse_rg_json
        self._path_allowed = path_allowed
        self._search_roots = search_roots

    def spec(self) -> ToolSpec:
        """Return the jsonl_search tool spec."""
        return ToolSpec(
            "jsonl_search",
            "Search JSONL files: optional ripgrep prefilter plus structured field/where filtering. Default scope is **/*.jsonl.",
            JsonlSearchArgs,
            self.search,
        )

    def search(self, args: JsonlSearchArgs | Json) -> ToolResult:
        """Filter JSONL files with an optional ripgrep prefilter and structured field/where filtering."""
        args = coerce_args(args, JsonlSearchArgs)
        query = args.query
        path_glob = args.path_glob
        try:
            validate_glob_selector(path_glob, field="path_glob")
        except PathValidationError as exc:
            return _path_error(exc)
        fields = args.fields
        where = [item.model_dump(exclude_none=True) for item in args.where]
        max_files = args.max_files
        max_matches_per_file = args.max_matches_per_file
        timeout = args.timeout or self.rg_timeout
        limit_chars = args.max_chars or self.max_tool_chars

        candidates, scan_error = self._candidates(query, path_glob, timeout)
        if scan_error is not None:
            return scan_error

        shown: dict[str, list[tuple[int, Any]]] = {}
        row_counts: dict[str, int] = {}
        json_errors = 0
        total_files = 0
        total_rows = 0
        for path, line_number, line_text in candidates:
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

        shown_files = list(shown.items())

        header = [
            "  summary:",
            f"    query: {query or '(none)'}",
            f"    scope: glob={path_glob}",
        ]
        if where:
            header.append(f"    where: {_describe_where(where)}")
        if fields:
            header.append(f"    fields: {', '.join(fields)}")
        header.append(f"    files: {total_files} total, {len(shown_files)} shown")
        header.append(f"    rows_matched: {total_rows}")
        if json_errors:
            header.append(f"    json_parse_errors: {json_errors}")
        body = [""]
        for path, rows in shown_files:
            rows.sort(key=lambda lr: lr[0])
            for line_number, row in rows:
                try:
                    projected = _project_fields(row, fields) if fields else row
                except ValueError as exc:
                    return ToolResult(False, f"invalid field path: {exc}")
                body.append(f"{path}:{line_number}: {json.dumps(projected, ensure_ascii=False, default=str)}")
            omitted_rows = row_counts[path] - len(rows)
            if omitted_rows:
                body.append(f"  ... {omitted_rows} more row(s) in {path}")
        if total_files > max_files:
            body.append(f"  note: {total_files - max_files} more file(s) omitted")

        return self._truncate("\n".join(header + body), prefix="jsonl_search", max_chars=limit_chars)

    def _candidates(self, query: str, path_glob: str, timeout: int) -> tuple[Iterator[tuple[str, int, str]], ToolResult | None]:
        """Collect (path, line_number, line_text) tuples for jsonl_search."""
        if query:
            search_roots = self._search_roots()
            command = ["rg", "--json", "--glob", path_glob, "--", query, *search_roots]
            if not search_roots:
                return iter(()), None
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
                return iter(()), ToolResult(False, _timeout_error_message("ripgrep", timeout), {"timeout": timeout, "cmd": command})
            if proc.returncode not in (0, 1):
                return iter(()), ToolResult(False, _rg_error_message(proc.returncode, proc.stdout), {"returncode": proc.returncode, "cmd": command})
            files = self._parse_rg_json(proc.stdout or "")
            return ((file.path, match.line_number, match.line_text) for file in files for match in file.matches), None
        return self._iter_jsonl_files(path_glob), None

    def _iter_jsonl_files(self, path_glob: str) -> Iterator[tuple[str, int, str]]:
        """Yield JSONL candidate rows without accumulating them."""
        for path in self.root.glob(path_glob):
            resolved = path.resolve()
            if not self._path_allowed(resolved) or not path.is_file():
                continue
            rel = str(resolved.relative_to(self.root))
            with resolved.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line_text in enumerate(handle, start=1):
                    yield rel, line_number, line_text.rstrip("\n")


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
