from __future__ import annotations

import ast
from pathlib import Path


def test_core_has_no_mcp_imports_or_lifecycle_state() -> None:
    """Core stays independent from MCP composition and lifecycle details."""
    core_path = Path(__file__).resolve().parents[2] / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_modules.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)

    assert not any(module.endswith(("tools.mcp", "plugins.mcp")) for module in imported_modules)
    for forbidden in ("MCPServer", "mcp_servers", "_mcp_", "_open_mcp_tools"):
        assert forbidden not in source


def test_core_has_no_skills_or_parallel_llm_implementation_details() -> None:
    """Core stays independent from skills and parallel LLM composition."""
    core_path = Path(__file__).resolve().parents[2] / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")

    forbidden = (
        "tools.skills",
        "SkillRegistry",
        "skills_dir",
        "selected_skills",
        "_skills_enabled",
        "prompt_summary",
        "tools.parallel_llm",
        "create_parallel_llm_tool",
        "builtin_parallel_llm",
        "parallel_llm_max_prompts",
    )
    for token in forbidden:
        assert token not in source

    migration_source = (core_path.parent / "_migration.py").read_text(encoding="utf-8")
    for field_name in (
        "skills_dir",
        "selected_skills",
        "read_paths",
        "write_paths",
        "builtin_parallel_llm_model",
        "builtin_parallel_llm_temperature",
        "parallel_llm_max_prompts",
    ):
        assert f'"{field_name}"' in migration_source
