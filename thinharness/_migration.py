"""Helpers for fail-loud removal guards."""

from __future__ import annotations

from collections.abc import Mapping

REMOVED_HARNESS_CONFIG_FIELDS = {
    "builtin_tools": "plugin composition; use SubagentsPlugin for delegation",
    "subagents": "SubagentsPlugin(agents=[...])",
    "skills_dir": "SkillsPlugin",
    "selected_skills": "SkillsPlugin",
    "read_paths": "ParallelLlmPlugin",
    "write_paths": "ParallelLlmPlugin",
    "builtin_parallel_llm_model": "ParallelLlmPlugin",
    "builtin_parallel_llm_temperature": "ParallelLlmPlugin",
    "parallel_llm_max_prompts": "ParallelLlmPlugin",
}


def reject_removed_fields(data: object, *, owner: str, migrations: Mapping[str, str]) -> object:
    """Reject the first removed field with its migration target."""
    if not isinstance(data, dict):
        return data
    for field_name, migration in migrations.items():
        if field_name in data:
            raise ValueError(f"{owner}.{field_name} has been removed; use {migration}")
    return data


__all__ = ["REMOVED_HARNESS_CONFIG_FIELDS", "reject_removed_fields"]
