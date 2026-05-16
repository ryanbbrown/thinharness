"""Filesystem and extension tools for the Responses harness."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import itertools
import json
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import Any, TypeGuard, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

Json = dict[str, Any]
ToolHandler = Callable[[Any], Any | Awaitable[Any]]
T = TypeVar("T", bound=BaseModel)
DEFAULT_SEARCH_LOW_PRIORITY_DIRS = [
    "example",
    "examples",
    "sample",
    "samples",
    "fixture",
    "fixtures",
    "mock",
    "mocks",
    "testdata",
    "vendor",
    "node_modules",
    "third_party",
]
DEFAULT_SEARCH_TEST_DIRS = ["test", "tests", "testing", "spec", "specs"]


# =============================================================================
# Tool data structures
# =============================================================================


@dataclass(frozen=True)
class ToolSpec:
    """A JSON-schema-described callable exposed to the model."""

    name: str
    description: str
    parameters: Json | type[BaseModel]
    handler: ToolHandler
    sequential: bool = False
    metadata: Json = field(default_factory=dict)

    def response_tool(self) -> Json:
        """Return an OpenAI Responses API function tool definition."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": tool_parameters(self.parameters),
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


@dataclass
class SearchMatch:
    """A single match extracted from rg --json output."""

    line_number: int
    line_text: str
    is_definition: bool


@dataclass
class SearchFile:
    """Aggregated search matches for one file."""

    path: str
    matches: list[SearchMatch]


@dataclass(frozen=True)
class AllowedPath:
    """One resolved path allowed by a workspace path policy."""

    path: Path
    exact: bool = False


class PathValidationError(ValueError):
    """Raised when a tool path or selector escapes its allowed policy."""


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


# =============================================================================
# Built-in tool argument models
# =============================================================================


class StrictArgs(BaseModel):
    """Base class for tool arguments."""

    model_config = ConfigDict(extra="forbid")


class ReadArgs(StrictArgs):
    """Arguments for read."""

    path: str
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=400, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


class WriteArgs(StrictArgs):
    """Arguments for write."""

    path: str
    content: str
    append: bool = False


class EditArgs(StrictArgs):
    """Arguments for edit."""

    path: str
    old_string: str
    new_string: str
    all: bool = False
    expected_replacements: int | None = Field(default=None, ge=1)


class SearchArgs(StrictArgs):
    """Arguments for search."""

    query: str = Field(description="Regex or literal search string.")
    path_glob: str = Field(default="", description="Optional glob filter such as **/*.py.")
    file_type: str = Field(default="", description="Optional ripgrep type such as py, rust, or js.")
    max_files: int = Field(default=10, ge=1)
    max_matches_per_file: int = Field(default=3, ge=1)
    max_line_chars: int | None = Field(default=None, ge=1, description="Search-only matched line preview cap. JSONL field output is controlled by fields.")
    timeout: int | None = Field(default=None, ge=1)
    max_chars: int | None = Field(default=None, ge=1)


class ListArgs(StrictArgs):
    """Arguments for list."""

    path: str = "."
    glob: str = Field(default="", description="Optional glob pattern relative to path.")
    recursive: bool = False
    max_results: int = Field(default=200, ge=1)


class GlobArgs(StrictArgs):
    """Arguments for glob."""

    pattern: str
    path: str = "."
    include_dirs: bool = False
    max_results: int = Field(default=200, ge=1)


# =============================================================================
# Built-in filesystem tools
# =============================================================================


class FileTools:
    """Small filesystem tool collection rooted at a workspace directory."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        output_dir: str | Path | None = None,
        max_read_chars: int = 40_000,
        max_read_bytes: int = 1_000_000,
        max_tool_chars: int = 40_000,
        max_search_line_chars: int = 180,
        rg_timeout: int = 30,
        search_exclude_globs: list[str] | None = None,
        search_low_priority_dirs: list[str] | None = None,
        search_test_dirs: list[str] | None = None,
        read_paths: Sequence[str | Path] | None = None,
        write_paths: Sequence[str | Path] | None = None,
    ) -> None:
        from .jsonl import JsonlSearch

        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = contained_path(self.root, output_dir or ".fsharness/outputs")
        self.read_policy = PathPolicy(self.root, read_paths, "read")
        self.write_policy = PathPolicy(self.root, write_paths, "write")
        self.max_read_chars = max_read_chars
        self.max_read_bytes = max_read_bytes
        self.max_tool_chars = max_tool_chars
        self.max_search_line_chars = max_search_line_chars
        self.rg_timeout = rg_timeout
        self.search_exclude_globs = list(search_exclude_globs or [])
        for exclude_glob in self.search_exclude_globs:
            validate_glob_selector(exclude_glob, field="search_exclude_globs", allow_negation=True)
        self.search_low_priority_dirs = {part.lower() for part in (search_low_priority_dirs or DEFAULT_SEARCH_LOW_PRIORITY_DIRS)}
        self.search_test_dirs = {part.lower() for part in (search_test_dirs or DEFAULT_SEARCH_TEST_DIRS)}
        self.jsonl = JsonlSearch(
            self.root,
            max_tool_chars=self.max_tool_chars,
            rg_timeout=self.rg_timeout,
            truncate=self._truncate,
            parse_rg_json=self._parse_contained_rg_json,
            path_allowed=self.read_policy.allows,
            search_roots=lambda: [self._display(path) for path in self.read_policy.existing_search_roots()],
        )

    # -------------------------------------------------------------------------
    # Tool schemas
    # -------------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        """Return built-in filesystem tool specs."""
        return [
            ToolSpec("read", "Read a UTF-8 text file with line numbers, offset, and limit.", ReadArgs, self.read),
            ToolSpec("write", "Create, overwrite, or append to a UTF-8 text file under the workspace root.", WriteArgs, self.write, sequential=True),
            ToolSpec("edit", "Replace exact text in a UTF-8 file. old_string must be unique unless all=true.", EditArgs, self.edit, sequential=True),
            ToolSpec("search", "Search code with ripgrep, then rank and format matches for agent follow-up reads.", SearchArgs, self.search),
            ToolSpec("list", "List a directory or glob files under the workspace root.", ListArgs, self.list_files),
            ToolSpec("glob", "Find files by glob pattern under the workspace root.", GlobArgs, self.glob),
            self.jsonl.spec(),
        ]

    # -------------------------------------------------------------------------
    # File read/write/edit tools
    # -------------------------------------------------------------------------

    def read(self, args: ReadArgs | Json) -> ToolResult:
        """Read a contained text file with line numbers."""
        args = coerce_args(args, ReadArgs)
        try:
            path = self.read_policy.resolve(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        if not path.exists():
            return ToolResult(False, f"file not found: {self._display(path)}", {"path": str(path)})
        if path.is_dir():
            return ToolResult(False, f"path is a directory: {self._display(path)}", {"path": str(path)})
        offset = args.offset
        limit = args.limit
        is_bounded_request = "offset" in args.model_fields_set or "limit" in args.model_fields_set
        size = path.stat().st_size
        if size > self.max_read_bytes and not is_bounded_request:
            return ToolResult(
                False,
                (
                    f"file is {size} bytes, over max_read_bytes={self.max_read_bytes}; "
                    "pass offset and limit to read a bounded range"
                ),
                {"path": str(path), "size_bytes": size, "max_read_bytes": self.max_read_bytes},
            )
        if size > self.max_read_bytes:
            selected, total_lines = self._read_large_range(path, offset, limit)
            note = f"read {len(selected)} line(s) from large file {self._display(path)} starting at line {offset}"
            if len(selected) == limit:
                note += " (more lines may be available; increase offset/limit)"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            selected = lines[offset - 1: offset - 1 + limit]
            total_lines = len(lines)
            note = f"read {len(selected)} of {total_lines} lines from {self._display(path)}"
            if offset + len(selected) - 1 < total_lines:
                note += " (more lines available; increase offset/limit)"
        limit_chars = min(args.max_chars or self.max_read_chars, self.max_read_chars)
        body = "\n".join(f"{i}\t{line}" for i, line in enumerate(selected, start=offset))
        result = self._truncate(f"{note}\n{body}" if body else note, prefix="read", max_chars=limit_chars)
        result.metadata.update({"path": str(path), "total_lines": total_lines, "returned_lines": len(selected), "size_bytes": size})
        return result

    def write(self, args: WriteArgs | Json) -> ToolResult:
        """Write a contained UTF-8 file."""
        args = coerce_args(args, WriteArgs)
        try:
            path = self.write_policy.resolve(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        content = args.content
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        action = "appended" if args.append else "wrote"
        return ToolResult(True, f"{action} {len(content.encode('utf-8'))} bytes to {self._display(path)}", {"path": str(path)})

    def edit(self, args: EditArgs | Json) -> ToolResult:
        """Replace exact text in a contained UTF-8 file."""
        args = coerce_args(args, EditArgs)
        try:
            path = self.write_policy.resolve(args.path)
        except PathValidationError as exc:
            return _path_error(exc)
        old = args.old_string
        new = args.new_string
        replace_all = args.all
        if not old:
            return ToolResult(False, "old_string must not be empty")
        if not path.exists():
            return ToolResult(False, f"file not found: {self._display(path)}", {"path": str(path)})
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(False, "old_string not found", {"path": str(path)})
        if args.expected_replacements is not None and count != args.expected_replacements:
            return ToolResult(False, f"expected {args.expected_replacements} replacement(s), found {count}", {"matches": count})
        if count > 1 and not replace_all:
            return ToolResult(False, f"old_string appears {count} times; add more context or set all=true", {"matches": count})
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        changed = count if replace_all else 1
        return ToolResult(True, f"replaced {changed} occurrence(s) in {self._display(path)}", {"path": str(path), "replacements": changed})

    # -------------------------------------------------------------------------
    # Search and listing tools
    # -------------------------------------------------------------------------

    def search(self, args: SearchArgs | Json) -> ToolResult:
        """Search code with pgr-style grouping, ranking, and output shaping."""
        args = coerce_args(args, SearchArgs)
        query = args.query
        if not query:
            return ToolResult(False, "query is required; pass a non-empty query string")
        path_glob = args.path_glob
        try:
            validate_glob_selector(path_glob, field="path_glob")
        except PathValidationError as exc:
            return _path_error(exc)
        file_type = args.file_type
        max_files = args.max_files
        max_matches_per_file = args.max_matches_per_file
        max_line_chars = args.max_line_chars or self.max_search_line_chars
        command = ["rg", "--json"]
        if path_glob:
            command.extend(["--glob", path_glob])
        for exclude_glob in self.search_exclude_globs:
            command.extend(["--glob", _exclude_glob(exclude_glob)])
        if file_type:
            command.extend(["--type", file_type])
        search_roots = self.read_policy.existing_search_roots()
        if not search_roots:
            command.extend(["--", query])
            return ToolResult(True, self._no_matches_message(query, path_glob, file_type), {"returncode": 1, "cmd": command, "matches": 0})
        command.extend(["--", query])
        command.extend(self._display(path) for path in search_roots)
        timeout = args.timeout or self.rg_timeout
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
            return ToolResult(False, _timeout_error_message("ripgrep", timeout), {"timeout": timeout, "cmd": command})
        if proc.returncode not in (0, 1):
            return ToolResult(False, _rg_error_message(proc.returncode, proc.stdout), {"returncode": proc.returncode, "cmd": command})
        files = self._parse_contained_rg_json(proc.stdout or "")
        if not files:
            return ToolResult(True, self._no_matches_message(query, path_glob, file_type), {"returncode": proc.returncode, "cmd": command, "matches": 0})
        files.sort(key=self._search_file_sort_key)
        total_files = len(files)
        shown_files = files[:max_files]
        content = self._format_search_output(query, path_glob, file_type, shown_files, total_files, max_files, max_matches_per_file, max_line_chars)
        result = self._truncate(content, prefix="search", max_chars=args.max_chars or self.max_tool_chars)
        result.metadata.update({"returncode": proc.returncode, "cmd": command, "cwd": str(self.root)})
        return result

    def list_files(self, args: ListArgs | Json) -> ToolResult:
        """List contained files and directories."""
        args = coerce_args(args, ListArgs)
        try:
            base = self.read_policy.resolve(args.path)
            validate_glob_selector(args.glob, field="glob")
        except PathValidationError as exc:
            return _path_error(exc)
        max_results = args.max_results
        pattern = args.glob
        if pattern:
            iterator = base.rglob(pattern) if args.recursive else base.glob(pattern)
        else:
            if not base.exists():
                return ToolResult(False, f"path not found: {self._display(base)}", {"path": str(base)})
            if base.is_file():
                iterator = iter([base])
            else:
                iterator = base.rglob("*") if args.recursive else base.iterdir()
        total, shown = self._bounded_paths(iterator, max_results, include_dirs=True)
        shown = sorted(shown, key=lambda p: (not p.is_dir(), str(p).lower()))
        lines = [("dir  " if p.is_dir() else "file ") + self._display(p) for p in shown]
        if total > len(shown):
            lines.append(f"... {total - len(shown)} more result(s) omitted")
        return ToolResult(True, "\n".join(lines) or "no files", {"total": total, "returned": len(shown)})

    def glob(self, args: GlobArgs | Json) -> ToolResult:
        """Glob for contained files and directories."""
        args = coerce_args(args, GlobArgs)
        try:
            base = self.read_policy.resolve(args.path)
            validate_glob_selector(args.pattern, field="pattern")
        except PathValidationError as exc:
            return _path_error(exc)
        max_results = args.max_results
        total, matches = self._newest_paths(base.glob(args.pattern), max_results, include_dirs=args.include_dirs)
        rows = [self._display(path) + ("/" if path.is_dir() else "") for path in matches]
        if total > max_results:
            rows.append(f"... {total - max_results} more result(s) omitted")
        return ToolResult(True, "\n".join(rows) or "no files", {"total": total, "returned": len(matches)})

    def jsonl_search(self, args: Json) -> ToolResult:
        """Delegate to the optional JSONL search tool."""
        return self.jsonl.search(args)

    # -------------------------------------------------------------------------
    # Search formatting helpers
    # -------------------------------------------------------------------------

    def _no_matches_message(self, query: str, path_glob: str, file_type: str) -> str:
        """Return a diagnostic empty-search message."""
        scope = _describe_search_scope(path_glob, file_type, self.search_exclude_globs)
        return (
            "No matches found.\n"
            f"  query: {query}\n"
            f"  scope: {scope}\n"
            "  hint: broaden the query, remove path_glob/file_type filters, or try a simpler symbol name."
        )

    @staticmethod
    def _parse_rg_json(stdout: str) -> list[SearchFile]:
        """Parse rg --json match output into files with matches."""
        file_map: dict[str, list[SearchMatch]] = {}
        file_order: list[str] = []
        for line in stdout.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("type") != "match":
                continue
            data = item.get("data") or {}
            raw_path = ((data.get("path") or {}).get("text") or "").strip()
            line_number = data.get("line_number")
            line_text = ((data.get("lines") or {}).get("text") or "").rstrip("\n")
            if not raw_path or not isinstance(line_number, int) or not line_text:
                continue
            path = raw_path.removeprefix("./")
            if path not in file_map:
                file_order.append(path)
            file_map.setdefault(path, []).append(SearchMatch(line_number, line_text, _is_definition(line_text)))
        return [SearchFile(path, file_map[path]) for path in file_order]

    def _parse_contained_rg_json(self, stdout: str) -> list[SearchFile]:
        """Parse rg output and drop matches outside the readable policy."""
        return [file for file in self._parse_rg_json(stdout) if self._search_file_allowed(file.path)]

    def _search_file_allowed(self, path: str) -> bool:
        """Return whether a search result path is readable."""
        try:
            resolved = _resolve_under_root(self.root, path)
        except PathValidationError:
            return False
        return self.read_policy.allows(resolved)

    def _search_file_sort_key(self, file: SearchFile) -> tuple[int, int, str]:
        """Sort definition matches first, then source before tests and low-priority paths."""
        has_definition = any(match.is_definition for match in file.matches)
        return (0 if has_definition else 1, self._file_priority(file.path), file.path)

    def _format_search_output(
        self,
        query: str,
        path_glob: str,
        file_type: str,
        files: list[SearchFile],
        total_files: int,
        max_files: int,
        max_matches_per_file: int,
        max_line_chars: int,
    ) -> str:
        """Format grouped and ranked search results for an agent."""
        source_count = sum(1 for file in files if self._file_priority(file.path) == 0)
        test_count = sum(1 for file in files if self._file_priority(file.path) == 1)
        low_priority_count = sum(1 for file in files if self._file_priority(file.path) > 1)
        definition_count = sum(1 for file in files if any(match.is_definition for match in file.matches))
        parts = [
            "  summary:\n"
            f"    query: {query}\n"
            f"    scope: {_describe_search_scope(path_glob, file_type, self.search_exclude_globs)}\n"
            f"    files: {total_files} total, {len(files)} shown\n"
            f"    buckets: {source_count} source, {test_count} test, {low_priority_count} low-priority\n"
            f"    definition_candidates: {definition_count}\n"
        ]
        if files and files[0].matches:
            parts[0] += f"    best_next_step: read {files[0].path} around line {files[0].matches[0].line_number}\n"
        for file in files:
            file.matches.sort(key=lambda match: (not match.is_definition, match.line_number))
            block = [file.path, f"  why: {self._file_reason(file)}"]
            for match in file.matches[:max_matches_per_file]:
                line = _truncate_line(match.line_text, max_line_chars)
                block.append(f"  {match.line_number}-{match.line_number}:")
                block.append(f"    {match.line_number}| {line}")
            parts.append("\n".join(block) + "\n")
        if total_files > max_files:
            parts.append(f"  note: truncated to top {max_files} files; refine the query or filters to narrow further.")
        return "\n".join(parts)

    def _file_reason(self, file: SearchFile) -> str:
        """Return why a file was ranked where it was."""
        kind = "definition" if any(match.is_definition for match in file.matches) else "reference"
        bucket = {0: "source", 1: "test"}.get(self._file_priority(file.path), "low-priority")
        return f"{kind}, {bucket}"

    def _file_priority(self, path: str) -> int:
        """Classify a path as source, test, or low-priority."""
        parts = path.replace("\\", "/").split("/")
        filename = parts[-1].lower() if parts else ""
        if any(part.lower() in self.search_low_priority_dirs for part in parts):
            return 2
        if any(part.lower() in self.search_test_dirs for part in parts[:-1]):
            return 1
        if "_test." in filename or filename.startswith("test_") or ".test." in filename or ".spec." in filename:
            return 1
        return 0

    # -------------------------------------------------------------------------
    # Shared file/output helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _read_large_range(path: Path, offset: int, limit: int) -> tuple[list[str], int | None]:
        """Stream a bounded line range from a large file."""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            selected = [
                line.rstrip("\n").rstrip("\r")
                for line in itertools.islice(handle, offset - 1, offset - 1 + limit)
            ]
        return selected, None

    def _display(self, path: Path) -> str:
        """Return a workspace-relative display path when possible."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _bounded_paths(self, paths: Any, max_results: int, *, include_dirs: bool) -> tuple[int, list[Path]]:
        """Collect at most max_results contained paths while counting all matches."""
        total = 0
        shown: list[Path] = []
        for path in paths:
            path = Path(path)
            if not _is_relative_to(path.resolve(), self.root):
                continue
            if not include_dirs and not path.is_file():
                continue
            total += 1
            if len(shown) < max_results:
                shown.append(path)
        return total, shown

    def _newest_paths(self, paths: Any, max_results: int, *, include_dirs: bool) -> tuple[int, list[Path]]:
        """Return the newest contained paths without materializing every match."""
        total = 0
        newest: list[tuple[float, str, Path]] = []
        for path in paths:
            path = Path(path)
            if not _is_relative_to(path.resolve(), self.root):
                continue
            if not include_dirs and not path.is_file():
                continue
            total += 1
            item = (path.stat().st_mtime if path.exists() else 0, str(path), path)
            if len(newest) < max_results:
                heapq.heappush(newest, item)
            elif item > newest[0]:
                heapq.heapreplace(newest, item)
        return total, [path for _, _, path in sorted(newest, reverse=True)]

    def _truncate(self, text: str, *, prefix: str, max_chars: int | None = None) -> ToolResult:
        """Truncate long tool output and spill the full content to disk."""
        limit = max_chars or self.max_tool_chars
        if len(text) <= limit:
            return ToolResult(True, text)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.output_dir / f"{prefix}-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.txt"
        artifact.write_text(text, encoding="utf-8")
        head = limit // 2
        tail = limit - head
        content = (
            f"[truncated {len(text)} chars to {limit}; full output saved to {self._display(artifact)}]\n"
            f"{text[:head]}\n...\n{text[-tail:]}"
        )
        return ToolResult(True, content, {"truncated": True, "saved_to": str(artifact), "chars": len(text)})


# =============================================================================
# Tool plumbing
# =============================================================================


def builtin_tools(root: str | Path = ".", **kwargs: Any) -> list[ToolSpec]:
    """Create the default filesystem tool set."""
    return FileTools(root, **kwargs).specs()


def _prepare_args(spec: ToolSpec, raw_args: str | Json) -> str | Any:
    """Parse and validate raw tool arguments."""
    try:
        args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as exc:
        return ToolResult(False, f"invalid JSON arguments: {exc}").as_json()
    if not isinstance(args, dict):
        return ToolResult(False, "tool arguments must be a JSON object").as_json()
    try:
        return spec.parse_args(args)
    except ValidationError as exc:
        return ToolResult(False, f"invalid arguments: {exc}").as_json()


def _normalize_result(result: Any) -> str:
    """Normalize a tool handler result to a structured JSON envelope."""
    if isinstance(result, ToolResult):
        return result.as_json()
    if isinstance(result, str):
        return ToolResult(True, result).as_json()
    return ToolResult(True, json.dumps(result, indent=2, sort_keys=True, default=str)).as_json()


def call_tool(spec: ToolSpec, raw_args: str | Json) -> str:
    """Invoke a sync tool handler and normalize the result to structured JSON."""
    args = _prepare_args(spec, raw_args)
    if isinstance(args, str):
        return args
    try:
        result = spec.handler(args)
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


# =============================================================================
# Path and schema helpers
# =============================================================================


def contained_path(root: Path, raw: str | Path) -> Path:
    """Resolve a path and require it to remain inside root."""
    return _resolve_under_root(root, raw)


def validate_glob_selector(pattern: str, *, field: str, allow_negation: bool = False) -> None:
    """Reject selector patterns that can address paths outside root."""
    if not pattern:
        return
    body = pattern[1:] if allow_negation and pattern.startswith("!") else pattern
    if "\x00" in body:
        raise PathValidationError(f"{field} must not contain NUL bytes: {pattern}")
    if Path(body).is_absolute() or PureWindowsPath(body).is_absolute():
        raise PathValidationError(f"{field} must be relative: {pattern}")
    parts = [part for part in body.replace("\\", "/").split("/") if part]
    if any(part == ".." for part in parts):
        raise PathValidationError(f"{field} must not contain '..': {pattern}")


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


# =============================================================================
# Search helpers
# =============================================================================


def _rg_error_message(returncode: int, output: str | None) -> str:
    """Return a compact ripgrep failure message."""
    details = (output or "").strip()
    suffix = f": {details[:400]}" if details else ""
    return f"ripgrep failed (rc={returncode}){suffix}"


def _timeout_error_message(command_name: str, timeout: int) -> str:
    """Return a compact timeout failure message."""
    return f"{command_name} timed out after {timeout}s"


def _exclude_glob(pattern: str) -> str:
    """Return a ripgrep exclusion glob."""
    return pattern if pattern.startswith("!") else f"!{pattern}"


def _describe_search_scope(path_glob: str, file_type: str, exclude_globs: list[str] | None = None) -> str:
    """Describe active search filters."""
    parts = []
    if path_glob:
        parts.append(f"glob={path_glob}")
    if file_type:
        parts.append(f"type={file_type}")
    for exclude_glob in exclude_globs or []:
        parts.append(f"exclude={exclude_glob}")
    return ", ".join(parts) if parts else "all files"


def _truncate_line(line: str, max_chars: int) -> str:
    """Truncate a matched line for compact search output."""
    return line if len(line) <= max_chars else f"{line[:max_chars]}..."


def _is_definition(content: str) -> bool:
    """Return whether a matched line looks like a code definition."""
    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith(("//", "#", "/*", "*")):
            continue
        if _matches_definition_prefix(trimmed):
            return True
    return False


def _matches_definition_prefix(trimmed: str) -> bool:
    """Return whether a line starts with a known definition-like prefix."""
    prefixes = (
        "fn ",
        "pub fn ",
        "pub(crate) fn ",
        "struct ",
        "pub struct ",
        "enum ",
        "pub enum ",
        "trait ",
        "pub trait ",
        "impl ",
        "impl<",
        "type ",
        "pub type ",
        "mod ",
        "pub mod ",
        "func ",
        "class ",
        "def ",
        "function ",
        "export ",
        "const ",
        "let ",
        "var ",
        "interface ",
        "module.exports",
        "union ",
        "typedef ",
    )
    return trimmed.startswith(prefixes)


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether path is inside root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
