"""Filesystem and extension tools for the Responses harness."""

from __future__ import annotations

import glob as globlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Json = dict[str, Any]
ToolHandler = Callable[[Json], Any]


@dataclass(frozen=True)
class ToolSpec:
    """A JSON-schema-described callable exposed to the model."""

    name: str
    description: str
    parameters: Json
    handler: ToolHandler

    def response_tool(self) -> Json:
        """Return an OpenAI Responses API function tool definition."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


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


class FileTools:
    """Small filesystem tool collection rooted at a workspace directory."""

    def __init__(
        self,
        root: str | Path = ".",
        *,
        output_dir: str | Path | None = None,
        max_read_chars: int = 40_000,
        max_tool_chars: int = 40_000,
        rg_timeout: int = 30,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_dir = contained_path(self.root, output_dir or ".fsharness/outputs")
        self.max_read_chars = max_read_chars
        self.max_tool_chars = max_tool_chars
        self.rg_timeout = rg_timeout

    def specs(self) -> list[ToolSpec]:
        """Return built-in filesystem tool specs."""
        return [
            ToolSpec("read", "Read a UTF-8 text file with line numbers, offset, and limit.", {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 1, "default": 1},
                    "limit": {"type": "integer", "minimum": 1, "default": 400},
                    "max_chars": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            }, self.read),
            ToolSpec("write", "Create, overwrite, or append to a UTF-8 text file under the workspace root.", {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            }, self.write),
            ToolSpec("edit", "Replace exact text in a UTF-8 file. old_string must be unique unless all=true.", {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "all": {"type": "boolean", "default": False},
                    "expected_replacements": {"type": "integer", "minimum": 1},
                },
                "required": ["path", "old_string", "new_string"],
                "additionalProperties": False,
            }, self.edit),
            ToolSpec("search", "Search code with ripgrep, then rank and format matches for agent follow-up reads.", {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Regex or literal search string."},
                    "path_glob": {"type": "string", "description": "Optional glob filter such as **/*.py."},
                    "file_type": {"type": "string", "description": "Optional ripgrep type such as py, rust, or js."},
                    "max_files": {"type": "integer", "minimum": 1, "default": 10},
                    "max_matches_per_file": {"type": "integer", "minimum": 1, "default": 3},
                    "timeout": {"type": "integer", "minimum": 1},
                    "max_chars": {"type": "integer", "minimum": 1},
                },
                "required": ["query"],
                "additionalProperties": False,
            }, self.search),
            ToolSpec("list", "List a directory or glob files under the workspace root.", {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "glob": {"type": "string", "description": "Optional glob pattern relative to path."},
                    "recursive": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "minimum": 1, "default": 200},
                },
                "additionalProperties": False,
            }, self.list_files),
            ToolSpec("glob", "Find files by glob pattern under the workspace root.", {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "include_dirs": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "minimum": 1, "default": 200},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            }, self.glob),
            ToolSpec("jsonl_search", "Search JSONL files: optional ripgrep prefilter plus structured field/where filtering. Default scope is **/*.jsonl.", {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional ripgrep query. If omitted, scan all rows in scope."},
                    "path_glob": {"type": "string", "description": "Glob filter; defaults to **/*.jsonl.", "default": "**/*.jsonl"},
                    "fields": {
                        "type": "object",
                        "description": "Map of jq-style field path to max chars (0 = no truncation). If omitted, return the whole row.",
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    "where": {
                        "type": "array",
                        "description": "Filters AND-ed together. Each: {field, op, value | values}. Ops: eq, ne, in, contains, regex, exists.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "op": {"type": "string", "enum": ["eq", "ne", "in", "contains", "regex", "exists"]},
                                "value": {"type": "string"},
                                "values": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["field", "op"],
                            "additionalProperties": False,
                        },
                    },
                    "max_files": {"type": "integer", "minimum": 1, "default": 10},
                    "max_matches_per_file": {"type": "integer", "minimum": 1, "default": 3},
                    "timeout": {"type": "integer", "minimum": 1},
                    "max_chars": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            }, self.jsonl_search),
        ]

    def read(self, args: Json) -> ToolResult:
        """Read a contained text file with line numbers."""
        path = contained_path(self.root, args["path"])
        if not path.exists():
            return ToolResult(False, f"file not found: {self._display(path)}", {"path": str(path)})
        if path.is_dir():
            return ToolResult(False, f"path is a directory: {self._display(path)}", {"path": str(path)})
        offset = max(1, int(args.get("offset", 1)))
        limit = max(1, int(args.get("limit", 400)))
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        selected = lines[offset - 1: offset - 1 + limit]
        body = "\n".join(f"{i}\t{line}" for i, line in enumerate(selected, start=offset))
        note = f"read {len(selected)} of {len(lines)} lines from {self._display(path)}"
        if offset + len(selected) - 1 < len(lines):
            note += " (more lines available; increase offset/limit)"
        limit_chars = min(int(args.get("max_chars", self.max_read_chars)), self.max_read_chars)
        result = self._truncate(f"{note}\n{body}" if body else note, prefix="read", max_chars=limit_chars)
        result.metadata.update({"path": str(path), "total_lines": len(lines), "returned_lines": len(selected)})
        return result

    def write(self, args: Json) -> ToolResult:
        """Write a contained UTF-8 file."""
        path = contained_path(self.root, args["path"])
        content = str(args.get("content", ""))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.get("append") else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        action = "appended" if args.get("append") else "wrote"
        return ToolResult(True, f"{action} {len(content.encode('utf-8'))} bytes to {self._display(path)}", {"path": str(path)})

    def edit(self, args: Json) -> ToolResult:
        """Replace exact text in a contained UTF-8 file."""
        path = contained_path(self.root, args["path"])
        old = str(args["old_string"])
        new = str(args["new_string"])
        replace_all = bool(args.get("all", False))
        if not old:
            return ToolResult(False, "old_string must not be empty")
        if not path.exists():
            return ToolResult(False, f"file not found: {self._display(path)}", {"path": str(path)})
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(False, "old_string not found", {"path": str(path)})
        expected = args.get("expected_replacements")
        if expected is not None and count != int(expected):
            return ToolResult(False, f"expected {expected} replacement(s), found {count}", {"matches": count})
        if count > 1 and not replace_all:
            return ToolResult(False, f"old_string appears {count} times; add more context or set all=true", {"matches": count})
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        path.write_text(updated, encoding="utf-8")
        changed = count if replace_all else 1
        return ToolResult(True, f"replaced {changed} occurrence(s) in {self._display(path)}", {"path": str(path), "replacements": changed})

    def search(self, args: Json) -> ToolResult:
        """Search code with pgr-style grouping, ranking, and output shaping."""
        query = str(args.get("query") or "")
        if not query:
            return ToolResult(False, "query is required; pass a non-empty query string")
        path_glob = str(args.get("path_glob") or "")
        file_type = str(args.get("file_type") or "")
        max_files = max(1, int(args.get("max_files", 10)))
        max_matches_per_file = max(1, int(args.get("max_matches_per_file", 3)))
        command = ["rg", "--json"]
        if path_glob:
            command.extend(["--glob", path_glob])
        if file_type:
            command.extend(["--type", file_type])
        command.extend(["--", query, "."])
        proc = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(args.get("timeout", self.rg_timeout)),
            check=False,
        )
        if proc.returncode not in (0, 1) and not proc.stdout:
            return ToolResult(False, self._no_matches_message(query, path_glob, file_type), {"returncode": proc.returncode, "cmd": command})
        files = self._parse_rg_json(proc.stdout or "")
        if not files:
            return ToolResult(True, self._no_matches_message(query, path_glob, file_type), {"returncode": proc.returncode, "cmd": command, "matches": 0})
        files.sort(key=self._search_file_sort_key)
        total_files = len(files)
        shown_files = files[:max_files]
        content = self._format_search_output(query, path_glob, file_type, shown_files, total_files, max_files, max_matches_per_file)
        result = self._truncate(content, prefix="search", max_chars=int(args.get("max_chars", self.max_tool_chars)))
        result.metadata.update({"returncode": proc.returncode, "cmd": command, "cwd": str(self.root)})
        return result

    def list_files(self, args: Json) -> ToolResult:
        """List contained files and directories."""
        base = contained_path(self.root, args.get("path", "."))
        max_results = max(1, int(args.get("max_results", 200)))
        pattern = args.get("glob")
        if pattern:
            matches = [Path(p) for p in globlib.glob(str(base / str(pattern)), recursive=bool(args.get("recursive", False)))]
        else:
            if not base.exists():
                return ToolResult(False, f"path not found: {self._display(base)}", {"path": str(base)})
            if base.is_file():
                matches = [base]
            else:
                iterator = base.rglob("*") if args.get("recursive", False) else base.iterdir()
                matches = list(iterator)
        matches = [path for path in matches if _is_relative_to(path.resolve(), self.root)]
        matches = sorted(matches, key=lambda p: (not p.is_dir(), str(p).lower()))
        shown = matches[:max_results]
        lines = [("dir  " if p.is_dir() else "file ") + self._display(p) for p in shown]
        if len(matches) > len(shown):
            lines.append(f"... {len(matches) - len(shown)} more result(s) omitted")
        return ToolResult(True, "\n".join(lines) or "no files", {"total": len(matches), "returned": len(shown)})

    def glob(self, args: Json) -> ToolResult:
        """Glob for contained files and directories."""
        base = contained_path(self.root, args.get("path", "."))
        max_results = max(1, int(args.get("max_results", 200)))
        include_dirs = bool(args.get("include_dirs", False))
        matches = [Path(p) for p in globlib.glob(str(base / str(args["pattern"])), recursive=True)]
        matches = [path for path in matches if _is_relative_to(path.resolve(), self.root)]
        matches = [path for path in matches if include_dirs or path.is_file()]
        matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        rows = [self._display(path) + ("/" if path.is_dir() else "") for path in matches[:max_results]]
        if len(matches) > max_results:
            rows.append(f"... {len(matches) - max_results} more result(s) omitted")
        return ToolResult(True, "\n".join(rows) or "no files", {"total": len(matches), "returned": min(len(matches), max_results)})

    def jsonl_search(self, args: Json) -> ToolResult:
        """Filter JSONL files with an optional ripgrep prefilter and structured field/where filtering."""
        query = str(args.get("query") or "")
        path_glob = str(args.get("path_glob") or "**/*.jsonl")
        fields = args.get("fields") or {}
        where = args.get("where") or []
        max_files = max(1, int(args.get("max_files", 10)))
        max_matches_per_file = max(1, int(args.get("max_matches_per_file", 3)))
        timeout = int(args.get("timeout", self.rg_timeout))
        limit_chars = int(args.get("max_chars", self.max_tool_chars))

        if not isinstance(fields, dict):
            return ToolResult(False, "fields must be an object of {path: max_chars}")
        if not isinstance(where, list):
            return ToolResult(False, "where must be a list of filter objects")

        candidates, scan_error = self._jsonl_candidates(query, path_glob, timeout)
        if scan_error is not None:
            return scan_error

        matched: dict[str, list[tuple[int, Any]]] = {}
        json_errors = 0
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
            matched.setdefault(path, []).append((line_number, row))

        files = sorted(matched.items(), key=lambda kv: kv[0])
        total_files = len(files)
        total_rows = sum(len(rows) for _, rows in files)
        shown_files = files[:max_files]

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
            for line_number, row in rows[:max_matches_per_file]:
                try:
                    projected = _project_fields(row, fields) if fields else row
                except ValueError as exc:
                    return ToolResult(False, f"invalid field path: {exc}")
                body.append(f"{path}:{line_number}: {json.dumps(projected, ensure_ascii=False, default=str)}")
            if len(rows) > max_matches_per_file:
                body.append(f"  ... {len(rows) - max_matches_per_file} more row(s) in {path}")
        if total_files > max_files:
            body.append(f"  note: {total_files - max_files} more file(s) omitted")

        return self._truncate("\n".join(header + body), prefix="jsonl_search", max_chars=limit_chars)

    def _jsonl_candidates(self, query: str, path_glob: str, timeout: int) -> tuple[list[tuple[str, int, str]], ToolResult | None]:
        """Collect (path, line_number, line_text) tuples for jsonl_search."""
        if query:
            command = ["rg", "--json", "--glob", path_glob, "--", query, "."]
            proc = subprocess.run(
                command,
                cwd=self.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            if proc.returncode not in (0, 1) and not proc.stdout:
                return [], ToolResult(False, f"ripgrep failed (rc={proc.returncode}): {(proc.stdout or '').strip()[:400]}", {"cmd": command})
            files = self._parse_rg_json(proc.stdout or "")
            return [(file.path, match.line_number, match.line_text) for file in files for match in file.matches], None
        matches = [Path(p) for p in globlib.glob(str(self.root / path_glob), recursive=True)]
        matches = [path for path in matches if _is_relative_to(path.resolve(), self.root) and path.is_file()]
        candidates: list[tuple[str, int, str]] = []
        for path in sorted(matches):
            rel = str(path.relative_to(self.root))
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line_text in enumerate(handle, start=1):
                    candidates.append((rel, line_number, line_text.rstrip("\n")))
        return candidates, None

    @staticmethod
    def _no_matches_message(query: str, path_glob: str, file_type: str) -> str:
        """Return a diagnostic empty-search message."""
        scope = _describe_search_scope(path_glob, file_type)
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

    @staticmethod
    def _search_file_sort_key(file: SearchFile) -> tuple[int, int, str]:
        """Sort definition matches first, then source before tests and low-priority paths."""
        has_definition = any(match.is_definition for match in file.matches)
        return (0 if has_definition else 1, _file_priority(file.path), file.path)

    @staticmethod
    def _format_search_output(
        query: str,
        path_glob: str,
        file_type: str,
        files: list[SearchFile],
        total_files: int,
        max_files: int,
        max_matches_per_file: int,
    ) -> str:
        """Format grouped and ranked search results for an agent."""
        source_count = sum(1 for file in files if _file_priority(file.path) == 0)
        test_count = sum(1 for file in files if _file_priority(file.path) == 1)
        low_priority_count = sum(1 for file in files if _file_priority(file.path) > 1)
        definition_count = sum(1 for file in files if any(match.is_definition for match in file.matches))
        parts = [
            "  summary:\n"
            f"    query: {query}\n"
            f"    scope: {_describe_search_scope(path_glob, file_type)}\n"
            f"    files: {total_files} total, {len(files)} shown\n"
            f"    buckets: {source_count} source, {test_count} test, {low_priority_count} low-priority\n"
            f"    definition_candidates: {definition_count}\n"
        ]
        if files and files[0].matches:
            parts[0] += f"    best_next_step: read {files[0].path} around line {files[0].matches[0].line_number}\n"
        for file in files:
            file.matches.sort(key=lambda match: (not match.is_definition, match.line_number))
            block = [file.path, f"  why: {_file_reason(file)}"]
            for match in file.matches[:max_matches_per_file]:
                line = _truncate_line(match.line_text)
                block.append(f"  {match.line_number}-{match.line_number}:")
                block.append(f"    {match.line_number}| {line}")
            parts.append("\n".join(block) + "\n")
        if total_files > max_files:
            parts.append(f"  note: truncated to top {max_files} files; refine the query or filters to narrow further.")
        return "\n".join(parts)

    def _display(self, path: Path) -> str:
        """Return a workspace-relative display path when possible."""
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _truncate(self, text: str, *, prefix: str, max_chars: int | None = None) -> ToolResult:
        """Truncate long tool output and spill the full content to disk."""
        limit = max_chars or self.max_tool_chars
        if len(text) <= limit:
            return ToolResult(True, text)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.output_dir / f"{prefix}-{int(time.time() * 1000)}.txt"
        artifact.write_text(text, encoding="utf-8")
        head = limit // 2
        tail = limit - head
        content = (
            f"[truncated {len(text)} chars to {limit}; full output saved to {self._display(artifact)}]\n"
            f"{text[:head]}\n...\n{text[-tail:]}"
        )
        return ToolResult(True, content, {"truncated": True, "saved_to": str(artifact), "chars": len(text)})


def builtin_tools(root: str | Path = ".", **kwargs: Any) -> list[ToolSpec]:
    """Create the default filesystem tool set."""
    return FileTools(root, **kwargs).specs()


def call_tool(spec: ToolSpec, raw_args: str | Json) -> str:
    """Invoke a tool handler and normalize the result to a string."""
    args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args
    result = spec.handler(args)
    if isinstance(result, ToolResult):
        return result.as_json()
    if isinstance(result, str):
        return result
    return json.dumps(result, indent=2, sort_keys=True, default=str)


def contained_path(root: Path, raw: str | Path) -> Path:
    """Resolve a path and require it to remain inside root."""
    path = Path(raw).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes root: {raw}")
    return resolved


def object_schema(fields: dict[str, str], required: list[str] | None = None) -> Json:
    """Create a small JSON object schema from compact field type names."""
    props: Json = {}
    required = required or [name for name, typ in fields.items() if not typ.endswith("?")]
    for name, typ in fields.items():
        base = typ.rstrip("?")
        if base == "array":
            props[name] = {"type": "array", "items": {"type": "string"}}
        elif base == "integer":
            props[name] = {"type": "integer"}
        elif base == "boolean":
            props[name] = {"type": "boolean"}
        else:
            props[name] = {"type": "string"}
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _describe_search_scope(path_glob: str, file_type: str) -> str:
    """Describe active search filters."""
    parts = []
    if path_glob:
        parts.append(f"glob={path_glob}")
    if file_type:
        parts.append(f"type={file_type}")
    return ", ".join(parts) if parts else "all files"


def _file_reason(file: SearchFile) -> str:
    """Return why a file was ranked where it was."""
    kind = "definition" if any(match.is_definition for match in file.matches) else "reference"
    bucket = {0: "source", 1: "test"}.get(_file_priority(file.path), "low-priority")
    return f"{kind}, {bucket}"


def _truncate_line(line: str) -> str:
    """Truncate a matched line for compact search output."""
    return line if len(line) <= 180 else f"{line[:180]}..."


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


def _file_priority(path: str) -> int:
    """Classify a path as source, test, or low-priority."""
    low_dirs = {"example", "examples", "sample", "samples", "fixture", "fixtures", "mock", "mocks", "testdata", "vendor", "node_modules", "third_party"}
    test_dirs = {"test", "tests", "testing", "spec", "specs"}
    parts = path.replace("\\", "/").split("/")
    filename = parts[-1].lower() if parts else ""
    if any(part.lower() in low_dirs for part in parts):
        return 2
    if any(part.lower() in test_dirs for part in parts[:-1]):
        return 1
    if "_test." in filename or filename.startswith("test_") or ".test." in filename or ".spec." in filename:
        return 1
    return 0


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether path is inside root."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
