"""Skills plugin."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..tools.skills import SkillRegistry
from .base import PluginBinding, PluginContext, PluginContribution

SkillToolName = Literal["skill_read", "skill_run"]
_VALID_TOOLS = ("skill_read", "skill_run")


@dataclass(frozen=True)
class _SkillsConfig:
    tools: tuple[SkillToolName, ...]
    registry: SkillRegistry
    contribution: PluginContribution


class _SkillsPluginMeta(type):
    """Keep the skills plugin name fixed on the class hierarchy."""

    def __setattr__(cls, attribute: str, value: object) -> None:
        if attribute == "name":
            raise AttributeError("SkillsPlugin.name is fixed to 'skills'")
        super().__setattr__(attribute, value)

    def __delattr__(cls, attribute: str) -> None:
        if attribute == "name":
            raise AttributeError("SkillsPlugin.name is fixed to 'skills'")
        super().__delattr__(attribute)


class SkillsPlugin(metaclass=_SkillsPluginMeta):
    """Expose one constructor-time skill catalog through selected tools."""

    name = "skills"
    _config: _SkillsConfig
    _frozen: bool

    def __init_subclass__(cls) -> None:
        """Reject subclasses that replace the fixed plugin name."""
        super().__init_subclass__()
        if "name" in cls.__dict__:
            raise TypeError("SkillsPlugin subclasses cannot override the fixed name 'skills'")

    def __setattr__(self, attribute: str, value: object) -> None:
        """Reject configuration changes after construction."""
        if attribute == "name":
            raise AttributeError("SkillsPlugin.name is fixed to 'skills'")
        if getattr(self, "_frozen", False):
            raise AttributeError("SkillsPlugin configuration is frozen")
        object.__setattr__(self, attribute, value)

    def __delattr__(self, attribute: str) -> None:
        """Reject configuration deletion after construction."""
        if attribute == "name":
            raise AttributeError("SkillsPlugin.name is fixed to 'skills'")
        if getattr(self, "_frozen", False):
            raise AttributeError("SkillsPlugin configuration is frozen")
        object.__delattr__(self, attribute)

    def __init__(
        self,
        skills_dir: str | Path | Sequence[str | Path],
        *,
        selected_skills: Sequence[str] | None = None,
        tools: Sequence[SkillToolName],
    ) -> None:
        if isinstance(skills_dir, (set, frozenset)):
            raise TypeError("SkillsPlugin skills_dir must be an ordered sequence, not a set")
        if isinstance(skills_dir, str | Path):
            directories: str | Path | tuple[str | Path, ...] = skills_dir
        else:
            directories = tuple(skills_dir)
            if not directories:
                raise ValueError("SkillsPlugin skills_dir must not be empty")
        if isinstance(tools, (set, frozenset)):
            raise TypeError("SkillsPlugin tools must be an ordered sequence, not a set")
        selected_tools = tuple(tools)
        if not selected_tools:
            raise ValueError("SkillsPlugin tools must not be empty")
        if len(set(selected_tools)) != len(selected_tools):
            raise ValueError("SkillsPlugin tools contains a duplicate name")
        unknown = next((name for name in selected_tools if name not in _VALID_TOOLS), None)
        if unknown is not None:
            available = ", ".join(_VALID_TOOLS)
            raise ValueError(f"unknown SkillsPlugin tool: {unknown}; available: {available}")

        registry = SkillRegistry(
            directories,
            selected_skills=tuple(selected_skills) if selected_skills is not None else None,
        )
        by_name = {spec.name: spec for spec in registry.specs()}
        specs = tuple(by_name[name] for name in selected_tools if name in by_name)
        instructions: tuple[str, ...] = ()
        if specs:
            summary = registry.prompt_summary(include_read_hint="skill_read" in selected_tools)
            if summary:
                instructions = (summary,)
        object.__setattr__(self, "_config", _SkillsConfig(
            tools=selected_tools,
            registry=registry,
            contribution=PluginContribution(tools=specs, instructions=instructions),
        ))
        object.__setattr__(self, "_frozen", True)

    @property
    def tools(self) -> tuple[SkillToolName, ...]:
        """Return the frozen selected tool names."""
        return self._config.tools

    @property
    def registry(self) -> SkillRegistry:
        """Return the shared constructor-time registry."""
        return self._config.registry

    def for_child(self) -> SkillsPlugin:
        """Reuse the shared frozen registry and catalog for a child binding."""
        return self

    def bind(self, context: PluginContext) -> PluginBinding:
        """Return the constructor-time contribution without I/O."""
        del context
        return PluginBinding(static=self._config.contribution)


__all__ = ["SkillsPlugin"]
