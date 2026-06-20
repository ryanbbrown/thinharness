"""JSONL search tool for structured line-delimited data."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
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
    op: Literal["eq", "ne", "in", "contains", "regex", "exists", "gt", "gte", "lt", "lte"]
    value: str | None = None
    values: list[str] | None = None
    type: Literal["number", "date"] | None = None


class JsonlFieldSearch(StrictArgs):
    """Search inside one string field on selected JSONL rows."""

    field: str = Field(min_length=1)
    query: str = Field(min_length=1)
    regex: bool = False
    case_sensitive: bool = False
    context_lines: int = Field(default=0, ge=0)
    max_matches: int = Field(default=20, ge=1)
    max_line_chars: int = Field(default=300, ge=1)


class JsonlSearchArgs(StrictArgs):
    """Arguments for jsonl_search."""

    query: str = Field(default="", description="Optional ripgrep query. If omitted, scan all rows in scope.")
    path: str = Field(default=".", description="File, directory, or glob path to JSONL files. Directories are searched recursively.")
    fields: dict[str, Annotated[int, Field(ge=0)]] = Field(
        default_factory=dict,
        description="Map of jq-style field path to max chars (0 = no truncation). If omitted, return the whole row.",
    )
    where: list[JsonlWhereFilter] = Field(default_factory=list, description="Filters AND-ed together.")
    field_searches: list[JsonlFieldSearch] = Field(
        default_factory=list,
        description="Search inside selected string fields and return matching internal field lines/snippets.",
    )
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


_RANGE_OPS = {"gt", "gte", "lt", "lte"}
_TYPED_EQUALITY_OPS = {"eq", "ne"}


@dataclass(frozen=True)
class _DateValue:
    """Parsed date-like value plus whether the source was date-only."""

    value: date | datetime
    date_only: bool


@dataclass(frozen=True)
class _CompiledWhere:
    """Pre-validated JSONL where filter."""

    field: str
    segments: list[str | int]
    op: str
    value: str | None
    values: list[str] | None
    compare_type: Literal["number", "date"] | None
    target: Any = None


@dataclass(frozen=True)
class _WhereResult:
    """One row's where result, including one-shot comparison warning state."""

    passed: bool
    compare_warning: bool = False


@dataclass(frozen=True)
class _CompiledFieldSearch:
    """Pre-validated field-level string search."""

    field: str
    occurrence: int
    segments: list[str | int]
    query: str
    pattern: re.Pattern[str] | None
    case_sensitive: bool
    context_lines: int
    max_matches: int
    max_line_chars: int
    show_query_label: bool


@dataclass(frozen=True)
class _RenderedFieldSearch:
    """Rendered field search lines for one selected row."""

    label: str
    lines: list[str]
    omitted_matches: int = 0
    note: str | None = None


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
        where_for_display = [item.model_dump(exclude_none=True) for item in args.where]
        try:
            where = _compile_where(args.where)
        except ValueError as exc:
            return ToolResult(False, f"invalid where filter: {exc}")
        try:
            field_searches = _compile_field_searches(args.field_searches)
        except re.error as exc:
            return ToolResult(False, f"invalid field_search regex: {exc}")
        except ValueError as exc:
            return ToolResult(False, f"invalid field path: {exc}")
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
        compare_warnings = 0
        total_files = 0
        total_rows = 0
        for file_path, line_number, line_text in scan.candidates:
            if not line_text.strip():
                continue
            try:
                row = json.loads(line_text)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            where_result = _where_passes(row, where)
            if where_result.compare_warning:
                compare_warnings += 1
            if not where_result.passed:
                continue
            if file_path not in row_counts:
                total_files += 1
                row_counts[file_path] = 0
            row_counts[file_path] += 1
            total_rows += 1
            if file_path in shown or len(shown) < max_files:
                shown.setdefault(file_path, [])
                if len(shown[file_path]) < max_matches_per_file:
                    shown[file_path].append((line_number, row))

        shown_files = sorted(shown.items())

        header = [
            "summary:",
            f"  query: {query or '(none)'}",
            f"  scope: path={path}",
        ]
        if where_for_display:
            header.append(f"  where: {_describe_where(where_for_display)}")
        if fields:
            header.append(f"  fields: {', '.join(fields)}")
        if field_searches:
            header.append(f"  field_searches: {', '.join(search.field for search in field_searches)}")
        header.append(f"  files: {total_files} total, {len(shown_files)} shown")
        header.append(f"  rows_matched: {total_rows}")
        if compare_warnings:
            header.append(f"  compare_warnings: {compare_warnings} row(s) had non-comparable values")
        if json_errors:
            header.append(f"  json_parse_errors: {json_errors}")
        body = [""]
        for file_path, rows in shown_files:
            rows.sort(key=lambda lr: lr[0])
            body.append(file_path)
            for line_number, row in rows:
                try:
                    projected = _project_fields(row, fields) if fields else ({} if field_searches else row)
                except ValueError as exc:
                    return ToolResult(False, f"invalid field path: {exc}")
                body.append(f"  {line_number}: {json.dumps(projected, ensure_ascii=False, default=str)}")
                if field_searches:
                    for rendered in _field_search_matches(row, field_searches, show_empty=not fields):
                        if rendered.lines:
                            body.append(f"    {rendered.label}:")
                            body.extend(f"      {line}" for line in rendered.lines)
                            if rendered.omitted_matches:
                                body.append(f"      ... {rendered.omitted_matches} more match(es)")
                        elif rendered.note:
                            body.append(f"    {rendered.label}: {rendered.note}")
            omitted_rows = row_counts[file_path] - len(rows)
            if omitted_rows:
                body.append(f"  ... {omitted_rows} more row(s)")
        if total_files > max_files:
            body.append(f"note: {total_files - max_files} more file(s) omitted")

        result = self._truncate("\n".join(header + body), prefix="jsonl_search", max_chars=limit_chars)
        result.metadata.update(scan.metadata)
        if compare_warnings:
            result.metadata["compare_warnings"] = compare_warnings
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
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
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


def _compile_where(where: list[JsonlWhereFilter]) -> list[_CompiledWhere]:
    """Validate where filters once and parse range targets before scanning."""
    compiled = []
    for filt in where:
        field = filt.field
        op = filt.op
        if not field or not op:
            raise ValueError(f"where filter missing field or op: {filt.model_dump(exclude_none=True)}")
        segments = _parse_jq_path(field)
        if op in _RANGE_OPS:
            if filt.value is None:
                raise ValueError(f"op {op!r} requires 'value'")
            if filt.value == "":
                raise ValueError(f"op {op!r} requires non-empty 'value'")
            if filt.type is None:
                raise ValueError(f"op {op!r} requires 'type'")
            if filt.values is not None:
                raise ValueError(f"op {op!r} does not accept 'values'")
            compiled.append(
                _CompiledWhere(
                    field=field,
                    segments=segments,
                    op=op,
                    value=filt.value,
                    values=None,
                    compare_type=filt.type,
                    target=_parse_range_target(filt.type, filt.value),
                )
            )
            continue
        if filt.type is not None and op not in _TYPED_EQUALITY_OPS:
            raise ValueError(f"op {op!r} does not accept 'type'")
        if op in {"eq", "ne", "contains", "regex"} and filt.value is None:
            raise ValueError(f"op {op!r} requires 'value'")
        typed_equality_target: Decimal | _DateValue | None = None
        if op in _TYPED_EQUALITY_OPS and filt.type is not None:
            if filt.value == "":
                raise ValueError(f"op {op!r} requires non-empty 'value'")
            if filt.values is not None:
                raise ValueError(f"op {op!r} does not accept 'values'")
            assert filt.value is not None
            typed_equality_target = _parse_range_target(filt.type, filt.value)
        if op == "in" and filt.values is None:
            raise ValueError("op 'in' requires 'values'")
        compiled.append(
            _CompiledWhere(
                field=field,
                segments=segments,
                op=op,
                value=filt.value,
                values=filt.values,
                compare_type=filt.type,
                target=typed_equality_target,
            )
        )
    return compiled


def _compile_field_searches(field_searches: list[JsonlFieldSearch]) -> list[_CompiledFieldSearch]:
    """Validate field searches once before scanning rows."""
    field_counts: dict[str, int] = {}
    for search in field_searches:
        field_counts[search.field] = field_counts.get(search.field, 0) + 1

    compiled = []
    field_occurrences: dict[str, int] = {}
    for search in field_searches:
        occurrence = field_occurrences.get(search.field, 0) + 1
        field_occurrences[search.field] = occurrence
        flags = 0 if search.case_sensitive else re.IGNORECASE
        compiled.append(
            _CompiledFieldSearch(
                field=search.field,
                occurrence=occurrence,
                segments=_parse_jq_path(search.field),
                query=search.query,
                pattern=re.compile(search.query, flags) if search.regex else None,
                case_sensitive=search.case_sensitive,
                context_lines=search.context_lines,
                max_matches=search.max_matches,
                max_line_chars=search.max_line_chars,
                show_query_label=field_counts[search.field] > 1,
            )
        )
    return compiled


def _field_search_matches(row: Any, searches: list[_CompiledFieldSearch], *, show_empty: bool) -> list[_RenderedFieldSearch]:
    """Return rendered snippet blocks for every matching field search on one row."""
    rendered = []
    for search in searches:
        value = _get_field_by_path(row, search.segments)
        label = _field_search_label(search)
        if value is _MISSING:
            if show_empty:
                rendered.append(_RenderedFieldSearch(label, [], note="none (missing field)"))
            continue
        if value is None:
            if show_empty:
                rendered.append(_RenderedFieldSearch(label, [], note="none (null field)"))
            continue
        if not isinstance(value, str):
            if show_empty:
                rendered.append(_RenderedFieldSearch(label, [], note="none (non-string field)"))
            continue
        lines = value.splitlines()
        matching_indexes = [index for index, line in enumerate(lines) if _field_line_matches(line, search)]
        if not matching_indexes:
            if show_empty:
                rendered.append(_RenderedFieldSearch(label, [], note="none"))
            continue

        displayed_matches = matching_indexes[: search.max_matches]
        ranges = _line_ranges_for_matches(displayed_matches, total_lines=len(lines), context_lines=search.context_lines)
        displayed_indexes = {index for start, end in ranges for index in range(start, end)}
        snippet_lines = [
            f"{index + 1}: {_truncate_internal_line(lines[index], search.max_line_chars)}"
            for start, end in ranges
            for index in range(start, end)
        ]
        omitted_matches = sum(1 for index in matching_indexes[search.max_matches :] if index not in displayed_indexes)
        rendered.append(_RenderedFieldSearch(label, snippet_lines, omitted_matches=omitted_matches))
    if any(item.lines for item in rendered):
        return [item for item in rendered if item.lines]
    return rendered


def _field_search_label(search: _CompiledFieldSearch) -> str:
    """Return a stable output label for a field search."""
    if search.show_query_label:
        return f"{search.field} matches #{search.occurrence} (query={search.query!r})"
    return f"{search.field} matches"


def _field_line_matches(line: str, search: _CompiledFieldSearch) -> bool:
    """Return whether one internal field line matches a compiled field search."""
    if search.pattern is not None:
        return search.pattern.search(line) is not None
    if search.case_sensitive:
        return search.query in line
    return search.query.casefold() in line.casefold()


def _line_ranges_for_matches(matches: list[int], *, total_lines: int, context_lines: int) -> list[tuple[int, int]]:
    """Build merged half-open line ranges around primary match indexes."""
    ranges: list[tuple[int, int]] = []
    for index in matches:
        start = max(0, index - context_lines)
        end = min(total_lines, index + context_lines + 1)
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    return ranges


def _truncate_internal_line(line: str, max_chars: int) -> str:
    """Truncate one rendered field line."""
    return line if len(line) <= max_chars else line[:max_chars] + "…"


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


def _where_passes(row: Any, where: list[_CompiledWhere]) -> _WhereResult:
    """Return whether a row passes every where filter (AND)."""
    for filt in where:
        value = _get_field_by_path(row, filt.segments)
        if filt.op in _RANGE_OPS:
            if filt.compare_type is None:
                raise ValueError(f"range op {filt.op!r} missing compare type")
            passed, non_comparable = _apply_range_op(value, filt.op, filt.compare_type, filt.target)
            if non_comparable:
                return _WhereResult(False, compare_warning=True)
            if not passed:
                return _WhereResult(False)
            continue
        if filt.op in _TYPED_EQUALITY_OPS and filt.compare_type is not None:
            passed, non_comparable = _apply_typed_equality_op(value, filt.op, filt.compare_type, filt.target)
            if non_comparable:
                return _WhereResult(False, compare_warning=True)
            if not passed:
                return _WhereResult(False)
            continue
        if not _apply_where_op(value, filt.op, filt.value, filt.values):
            return _WhereResult(False)
    return _WhereResult(True)


def _parse_range_target(compare_type: Literal["number", "date"], value: str) -> Decimal | _DateValue:
    """Parse a range filter target for its declared type."""
    if compare_type == "number":
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"number range value must be a finite number: {value!r}") from exc
        if parsed.is_nan() or parsed.is_infinite():
            raise ValueError(f"number range value must be a finite number: {value!r}")
        return parsed
    parsed_date = _parse_date_value(value)
    if parsed_date is None:
        raise ValueError(f"date range value must be ISO-like: {value!r}")
    return parsed_date


def _parse_date_value(value: Any) -> _DateValue | None:
    """Parse supported ISO-ish date strings."""
    if not isinstance(value, str):
        return None
    if _DATE_ONLY_RE.fullmatch(value):
        try:
            return _DateValue(date.fromisoformat(value), date_only=True)
        except ValueError:
            return None
    try:
        return _DateValue(datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")), date_only=False)
    except ValueError:
        return None


def _apply_range_op(value: Any, op: str, compare_type: Literal["number", "date"], target: Any) -> tuple[bool, bool]:
    """Evaluate one range operator, returning (passed, non_comparable)."""
    if compare_type == "number":
        comparable = _number_value(value)
        if comparable is None:
            return False, True
        return _compare_range(comparable, op, target), False
    if compare_type == "date":
        comparable = _parse_date_value(value)
        if comparable is None:
            return False, True
        compared = _compare_date_range(comparable, op, target)
        if compared is None:
            return False, True
        return compared, False
    raise ValueError(f"unknown range type: {compare_type!r}")


def _apply_typed_equality_op(value: Any, op: str, compare_type: Literal["number", "date"], target: Any) -> tuple[bool, bool]:
    """Evaluate a typed equality operator, returning (passed, non-comparable)."""
    if compare_type == "number":
        comparable = _number_value(value)
        if comparable is None:
            return False, True
        passed = comparable == target
    elif compare_type == "date":
        comparable = _parse_date_value(value)
        if comparable is None:
            return False, True
        compared = _compare_date_equality(comparable, target)
        if compared is None:
            return False, True
        passed = compared
    else:
        raise ValueError(f"unknown equality type: {compare_type!r}")
    return (not passed if op == "ne" else passed), False


def _number_value(value: Any) -> Decimal | None:
    """Return a comparable JSON number, excluding bool and non-finite float values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return Decimal(value) if isinstance(value, int) else Decimal(str(value))


def _compare_date_range(left: _DateValue, op: str, right: _DateValue) -> bool | None:
    """Compare parsed date values, returning None for aware/naive datetime mismatch."""
    if left.date_only or right.date_only:
        return _compare_range(_calendar_date(left.value), op, _calendar_date(right.value))

    if not isinstance(left.value, datetime) or not isinstance(right.value, datetime):
        return None
    if _datetime_is_aware(left.value) != _datetime_is_aware(right.value):
        return None
    return _compare_range(left.value, op, right.value)


def _compare_date_equality(left: _DateValue, right: _DateValue) -> bool | None:
    """Compare parsed date values, returning None for aware/naive datetime mismatch."""
    if left.date_only or right.date_only:
        return _calendar_date(left.value) == _calendar_date(right.value)
    if not isinstance(left.value, datetime) or not isinstance(right.value, datetime):
        return None
    if _datetime_is_aware(left.value) != _datetime_is_aware(right.value):
        return None
    return left.value == right.value


def _datetime_is_aware(value: datetime) -> bool:
    """Return whether a datetime has an effective timezone offset."""
    return value.utcoffset() is not None


def _calendar_date(value: date | datetime) -> date:
    """Return the written calendar date for a date or datetime value."""
    if isinstance(value, datetime):
        return value.date()
    return value


def _compare_range(left: Any, op: str, right: Any) -> bool:
    """Apply a range operator to already-comparable values."""
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    raise ValueError(f"unknown op: {op!r}")


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
        elif op in _RANGE_OPS or filt.get("type") is not None:
            parts.append(f"{filt.get('field')} {op} {filt.get('value')!r} ({filt.get('type')})")
        else:
            parts.append(f"{filt.get('field')} {op} {filt.get('value')!r}")
    return "; ".join(parts)
