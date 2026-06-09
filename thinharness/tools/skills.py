"""Frontmatter-based skill discovery and tool adapters."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import Json, PathValidationError, ToolResult, ToolSpec, _path_error, _timeout_error_message, coerce_args, contained_path


@dataclass(frozen=True)
class Skill:
    """A discovered skill directory and its metadata."""

    name: str
    description: str
    root: Path
    skill_file: Path
    metadata: Json


class SkillArgs(BaseModel):
    """Base class for skill tool arguments."""

    model_config = ConfigDict(extra="forbid")


class SkillReadArgs(SkillArgs):
    """Arguments for skill_read."""

    skill_name: str
    path: str | None = None
    max_chars: int = Field(default=40_000, ge=1)


class SkillRunArgs(SkillArgs):
    """Arguments for skill_run."""

    skill_name: str
    script: str
    args: list[str] = Field(default_factory=list)
    timeout: int = Field(default=60, ge=1)
    max_chars: int = Field(default=40_000, ge=1)


class SkillRegistry:
    """Load skills from directories containing SKILL.md files."""

    def __init__(
        self,
        skills_dir: str | Path | Sequence[str | Path] | None,
        *,
        selected_skills: Sequence[str] | None = None,
    ) -> None:
        self.skills_dirs = _normalize_skill_dirs(skills_dir)
        self.selected_skills = list(selected_skills) if selected_skills is not None else None
        self._skills = self._select_skills(self._discover())

    @property
    def skills(self) -> dict[str, Skill]:
        """Return a copy of the discovered skills map."""
        return dict(self._skills)

    def prompt_summary(self) -> str:
        """Return a compact skill list for the system prompt."""
        if not self._skills:
            return ""
        lines = ["Available skills (call skill_read before using details):"]
        for skill in self._skills.values():
            desc = f" - {skill.description}" if skill.description else ""
            lines.append(f"- {skill.name}{desc}")
        return "\n".join(lines)

    def specs(self) -> list[ToolSpec]:
        """Return tool specs for reading and running skills."""
        if not self._skills:
            return []
        return [
            ToolSpec("skill_read", "Read a skill's SKILL.md or another contained file, with a file tree.", SkillReadArgs, self.skill_read),
            ToolSpec(
                "skill_run",
                "Run a script inside a skill directory with JSON-array args. No sandboxing is applied.",
                SkillRunArgs,
                self.skill_run,
                sequential=True,
            ),
        ]

    def skill_read(self, args: SkillReadArgs | Json) -> ToolResult:
        """Read a skill file and include the skill tree."""
        args = coerce_args(args, SkillReadArgs)
        skill = self._get(args.skill_name)
        rel = str(args.path or skill.skill_file.relative_to(skill.root))
        try:
            target = contained_path(skill.root, rel)
        except PathValidationError as exc:
            return _path_error(exc)
        if not target.exists() or target.is_dir():
            return ToolResult(False, f"file not found: {rel}")
        content = target.read_text(encoding="utf-8", errors="replace")
        body = f"# Skill: {skill.name}\nRoot: {skill.root}\n\n## Files\n{self._tree(skill.root)}\n\n## {rel}\n{content}"
        return self._truncate(body, args.max_chars)

    def skill_run(self, args: SkillRunArgs | Json) -> ToolResult:
        """Run a contained skill script."""
        args = coerce_args(args, SkillRunArgs)
        skill = self._get(args.skill_name)
        try:
            script = contained_path(skill.root, args.script)
        except PathValidationError as exc:
            return _path_error(exc)
        if not script.exists() or script.is_dir():
            return ToolResult(False, f"script not found: {args.script}")
        command = _skill_command(script, args.args)
        try:
            proc = subprocess.run(
                command,
                cwd=skill.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, _timeout_error_message("skill script", args.timeout), {"timeout": args.timeout, "cmd": command})
        output = f"exit_code: {proc.returncode}\n{proc.stdout or ''}".strip()
        result = self._truncate(output or "(empty)", args.max_chars)
        result.ok = proc.returncode == 0
        result.metadata.update({"returncode": proc.returncode, "cmd": command})
        return result

    def _discover(self) -> dict[str, Skill]:
        """Discover skill files from the configured directories."""
        found: dict[str, Skill] = {}
        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue
            files = [path for path in skills_dir.rglob("SKILL.md") if path.is_file()]
            files += [path for path in skills_dir.glob("*.md") if path.name != "SKILL.md"]
            for path in sorted(set(files)):
                metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
                default_name = path.parent.name if path.name == "SKILL.md" else path.stem
                name = str(metadata.get("name") or default_name).strip()
                if not name:
                    continue
                skill = Skill(
                    name=name,
                    description=str(metadata.get("description") or "").strip(),
                    root=path.parent.resolve(),
                    skill_file=path.resolve(),
                    metadata=metadata,
                )
                if name in found:
                    raise ValueError(f"duplicate skill name: {name} in {found[name].skill_file} and {skill.skill_file}")
                found[name] = skill
        return found

    def _select_skills(self, discovered: dict[str, Skill]) -> dict[str, Skill]:
        """Return discovered skills filtered by selected_skills."""
        if self.selected_skills is None:
            return discovered
        selected: dict[str, Skill] = {}
        for name in self.selected_skills:
            if name in selected:
                raise ValueError(f"duplicate selected skill: {name}")
            if name not in discovered:
                available = ", ".join(sorted(discovered)) or "none"
                raise ValueError(f"unknown selected skill: {name}; available: {available}")
            selected[name] = discovered[name]
        return selected

    def _get(self, name: str) -> Skill:
        """Look up a skill by name."""
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills)) or "none"
            raise ValueError(f"unknown skill: {name}; available: {available}") from exc

    @staticmethod
    def _tree(root: Path, limit: int = 200) -> str:
        """Return a compact recursive file tree."""
        lines: list[str] = []
        for path in root.rglob("*"):
            if any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
                continue
            if len(lines) >= limit:
                lines.append("... tree truncated")
                break
            suffix = "/" if path.is_dir() else ""
            lines.append(str(path.relative_to(root)) + suffix)
        return "\n".join(lines) or "(empty)"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> ToolResult:
        """Return text clipped to max_chars."""
        if len(text) <= max_chars:
            return ToolResult(True, text)
        head = max_chars // 2
        tail = max_chars - head
        return ToolResult(True, f"[truncated {len(text)} chars to {max_chars}]\n{text[:head]}\n...\n{text[-tail:]}", {"truncated": True, "chars": len(text)})


def parse_frontmatter(text: str) -> tuple[Json, str]:
    """Parse simple YAML-like frontmatter."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[text.find("\n", end + 1) + 1:]
    data: Json = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip('"\'')
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
        data[key.strip()] = parsed
    return data, body


def _normalize_skill_dirs(skills_dir: str | Path | Sequence[str | Path] | None) -> list[Path]:
    """Normalize one or more skill directories."""
    if skills_dir is None:
        return []
    if isinstance(skills_dir, str | Path):
        return [Path(skills_dir).expanduser().resolve()]
    return [Path(path).expanduser().resolve() for path in skills_dir]


def _skill_command(script: Path, args: Sequence[str]) -> list[str]:
    """Build the process argv for a skill script."""
    script_args = [str(arg) for arg in args]
    match script.suffix:
        case ".py":
            return ["uv", "run", str(script), *script_args]
        case ".sh" | ".bash":
            return ["bash", str(script), *script_args]
        case ".js" | ".mjs":
            return ["node", str(script), *script_args]
        case ".go":
            return ["go", "run", str(script), *script_args]
        case _:
            return [str(script), *script_args]
