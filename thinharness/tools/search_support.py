"""Shared helpers for ripgrep-backed search tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .base import Json, PathPolicy, PathValidationError, contained_path


@dataclass
class SearchMatch:
    """A single match extracted from rg --json output."""

    line_number: int
    line_text: str


@dataclass
class SearchFile:
    """Aggregated search matches for one file."""

    path: str
    matches: list[SearchMatch]


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


def _rg_error_message(returncode: int, output: str | None) -> str:
    """Return a compact ripgrep failure message."""
    details = (output or "").strip()
    suffix = f": {details[:400]}" if details else ""
    return f"ripgrep failed (rc={returncode}){suffix}"


def _rg_partial_warning_metadata(returncode: int, output: str | None, *, include_match_events: bool = True) -> Json:
    """Return metadata for recoverable ripgrep partial output."""
    metadata: Json = {
        "returncode": returncode,
        "warning": f"ripgrep returned {returncode}; showing parsed partial matches",
    }
    excerpt = _compact_rg_excerpt(output, include_match_events=include_match_events)
    if excerpt:
        metadata["warning_excerpt"] = excerpt
    return metadata


def _compact_rg_excerpt(output: str | None, *, include_match_events: bool, limit: int = 400) -> str:
    """Return a compact excerpt from ripgrep output."""
    parts: list[str] = []
    for line in (output or "").splitlines():
        if not include_match_events and _is_rg_json_match_or_context(line):
            continue
        parts.append(line)
    return " ".join("\n".join(parts).strip().split())[:limit]


def _is_rg_json_match_or_context(line: str) -> bool:
    """Return whether an rg --json line carries matched or contextual text."""
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(item, dict) and item.get("type") in {"match", "context"}


def parse_contained_rg_json(stdout: str, root: Path, read_policy: PathPolicy) -> list[SearchFile]:
    """Parse rg output and drop matches outside the readable policy."""
    return [file for file in _parse_rg_json(stdout) if _search_file_allowed(file.path, root, read_policy)]


def search_root_display_paths(root: Path, read_policy: PathPolicy) -> list[str]:
    """Return readable search roots as workspace-relative display paths."""
    return [_display(root, path) for path in read_policy.existing_search_roots()]


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
        file_map.setdefault(path, []).append(SearchMatch(line_number, line_text))
    return [SearchFile(path, file_map[path]) for path in file_order]


def _search_file_allowed(path: str, root: Path, read_policy: PathPolicy) -> bool:
    """Return whether a search result path is readable."""
    try:
        resolved = contained_path(root, path)
    except PathValidationError:
        return False
    return read_policy.allows(resolved)


def _display(root: Path, path: Path) -> str:
    """Return a workspace-relative display path when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
