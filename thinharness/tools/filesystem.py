"""Filesystem-backed built-in tools."""

from __future__ import annotations

import heapq
import itertools
import subprocess
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from ..defaults import (
    DEFAULT_EDIT_DESCRIPTION,
    DEFAULT_GLOB_DESCRIPTION,
    DEFAULT_LIST_DESCRIPTION,
    DEFAULT_READ_DESCRIPTION,
    DEFAULT_SEARCH_DESCRIPTION,
    DEFAULT_WRITE_DESCRIPTION,
)
from .base import (
    Json,
    PathPolicy,
    PathValidationError,
    StrictArgs,
    ToolResult,
    ToolSpec,
    _is_relative_to,
    _path_error,
    _timeout_error_message,
    coerce_args,
    contained_path,
)
from .search_support import (
    SearchFile,
    _rg_error_message,
    _rg_partial_warning_metadata,
    parse_contained_rg_json,
    search_root_display_paths,
    validate_glob_selector,
)


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
    max_files: int = Field(default=50, ge=1)
    max_matches_per_file: int = Field(default=10, ge=1)
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
        read_paths: Sequence[str | Path] | None = None,
        write_paths: Sequence[str | Path] | None = None,
    ) -> None:
        from .jsonl import JsonlSearch

        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = contained_path(self.root, output_dir or ".thinharness/outputs")
        self._spill_artifacts: set[Path] = set()
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
        self.jsonl = JsonlSearch(
            self.root,
            self.read_policy,
            max_tool_chars=self.max_tool_chars,
            rg_timeout=self.rg_timeout,
            truncate=self._truncate,
        )

    # -------------------------------------------------------------------------
    # Tool schemas
    # -------------------------------------------------------------------------

    def specs(self) -> list[ToolSpec]:
        """Return built-in filesystem tool specs."""
        return [
            ToolSpec("read", DEFAULT_READ_DESCRIPTION, ReadArgs, self.read),
            ToolSpec("write", DEFAULT_WRITE_DESCRIPTION, WriteArgs, self.write, sequential=True),
            ToolSpec("edit", DEFAULT_EDIT_DESCRIPTION, EditArgs, self.edit, sequential=True),
            ToolSpec("search", DEFAULT_SEARCH_DESCRIPTION, SearchArgs, self.search),
            ToolSpec("list", DEFAULT_LIST_DESCRIPTION, ListArgs, self.list_files),
            ToolSpec("glob", DEFAULT_GLOB_DESCRIPTION, GlobArgs, self.glob),
            self.jsonl.spec(),
        ]

    # -------------------------------------------------------------------------
    # File read/write/edit tools
    # -------------------------------------------------------------------------

    def read(self, args: ReadArgs | Json) -> ToolResult:
        """Read a contained text file with line numbers."""
        args = coerce_args(args, ReadArgs)
        try:
            path = self._resolve_read_path(args.path)
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
        """Search readable files and return compact grouped matches."""
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
        command.extend(search_root_display_paths(self.root, self.read_policy))
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
        files = parse_contained_rg_json(proc.stdout or "", self.root, self.read_policy)
        warning_metadata: Json = {}
        if proc.returncode not in (0, 1):
            if not files:
                return ToolResult(False, _rg_error_message(proc.returncode, proc.stdout), {"returncode": proc.returncode, "cmd": command})
            warning_metadata = _rg_partial_warning_metadata(proc.returncode, proc.stdout)
        if not files:
            return ToolResult(True, self._no_matches_message(query, path_glob, file_type), {"returncode": proc.returncode, "cmd": command, "matches": 0})
        files.sort(key=lambda file: file.path)
        for file in files:
            file.matches.sort(key=lambda match: match.line_number)
        total_files = len(files)
        total_matches = sum(len(file.matches) for file in files)
        shown_files = files[:max_files]
        content = self._format_search_output(
            query,
            path_glob,
            file_type,
            shown_files,
            total_files,
            total_matches,
            max_files,
            max_matches_per_file,
            max_line_chars,
        )
        result = self._truncate(content, prefix="search", max_chars=args.max_chars or self.max_tool_chars)
        result.metadata.update({"returncode": proc.returncode, "cmd": command, "cwd": str(self.root)})
        result.metadata.update(warning_metadata)
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
            "  hint: broaden the query, remove path_glob/file_type filters, or try simpler terms."
        )

    def _format_search_output(
        self,
        query: str,
        path_glob: str,
        file_type: str,
        files: list[SearchFile],
        total_files: int,
        total_matches: int,
        max_files: int,
        max_matches_per_file: int,
        max_line_chars: int,
    ) -> str:
        """Format grouped search results for document-friendly reading."""
        shown_matches = sum(min(len(file.matches), max_matches_per_file) for file in files)
        omitted_matches = total_matches - shown_matches
        parts = [
            "summary:\n"
            f"  query: {query}\n"
            f"  scope: {_describe_search_scope(path_glob, file_type, self.search_exclude_globs)}\n"
            f"  files: {total_files} total, {len(files)} shown\n"
            f"  matches: {shown_matches} shown, {omitted_matches} omitted\n"
        ]
        for file in files:
            block = [file.path]
            for match in file.matches[:max_matches_per_file]:
                line = _truncate_line(match.line_text, max_line_chars)
                block.append(f"  {match.line_number}: {line}")
            omitted = len(file.matches) - min(len(file.matches), max_matches_per_file)
            if omitted:
                block.append(f"  ... {omitted} more match(es)")
            parts.append("\n".join(block) + "\n")
        if total_files > max_files:
            parts.append(f"note: {total_files - max_files} more file(s) omitted")
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Shared file/output helpers
    # -------------------------------------------------------------------------

    def _resolve_read_path(self, raw: str | Path) -> Path:
        """Resolve a readable path, including generated spill artifacts."""
        try:
            return self.read_policy.resolve(raw)
        except PathValidationError as policy_error:
            resolved = contained_path(self.root, raw)
            if self._is_readable_spill_artifact(resolved):
                return resolved
            raise policy_error

    def _is_readable_spill_artifact(self, path: Path) -> bool:
        """Return whether path is an exact generated spill artifact."""
        resolved = path.resolve()
        output_dir = self.output_dir.resolve()
        return resolved in self._spill_artifacts and (resolved == output_dir or output_dir in resolved.parents)

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
        resolved_artifact = artifact.resolve()
        self._spill_artifacts.add(resolved_artifact)
        saved_to_display = self._display(resolved_artifact)
        head = limit // 2
        tail = limit - head
        content = (
            f"[truncated {len(text)} chars to {limit}; full output saved to {saved_to_display}]\n"
            f"Read the saved output with read(path=\"{saved_to_display}\", offset=1, limit=400), then continue with later offsets as needed.\n"
            f"{text[:head]}\n...\n{text[-tail:]}"
        )
        return ToolResult(
            True,
            content,
            {"truncated": True, "saved_to": str(resolved_artifact), "saved_to_display": saved_to_display, "chars": len(text)},
        )


# =============================================================================
# Tool plumbing
# =============================================================================


def builtin_tools(root: str | Path = ".", **kwargs: Any) -> list[ToolSpec]:
    """Create the default filesystem tool set."""
    return FileTools(root, **kwargs).specs()

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
    return ", ".join(parts) if parts else "all readable files"


def _truncate_line(line: str, max_chars: int) -> str:
    """Truncate a matched line for compact search output."""
    return line if len(line) <= max_chars else f"{line[:max_chars]}..."
