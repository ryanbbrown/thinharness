from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from thinharness import PluginBinding


def test_private_plugin_runtime_contracts_stay_narrow() -> None:
    """The public binding excludes agent catalogs and the runtime scope stays private."""
    root = Path(__file__).resolve().parents[2]
    package_source = (root / "thinharness" / "__init__.py").read_text(encoding="utf-8")

    assert [field.name for field in fields(PluginBinding)] == ["static", "connect"]
    assert "_ToolRuntimeScope" not in package_source


def test_core_has_no_bash_imports_or_construction() -> None:
    """Core stays independent from Bash process behavior."""
    root = Path(__file__).resolve().parents[2]
    core_source = (root / "thinharness" / "core.py").read_text(encoding="utf-8")
    tools_init = (root / "thinharness" / "tools" / "__init__.py").read_text(encoding="utf-8")
    plugin_source = (root / "thinharness" / "plugins" / "bash.py").read_text(encoding="utf-8")

    assert "BashPlugin" not in core_source
    assert "plugins.bash" not in core_source
    assert "tools.bash" not in core_source
    assert "create_subprocess" not in core_source
    assert not (root / "thinharness" / "tools" / "bash.py").exists()
    assert "BashTool" not in tools_init
    assert "BashArgs" not in tools_init
    assert "class BashPlugin" in plugin_source
    assert "create_subprocess_exec" in plugin_source
    assert "def for_child" not in plugin_source
    assert "Executor" not in plugin_source


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


def test_core_has_no_subagent_feature_imports_or_identifiers() -> None:
    """Core depends only on the neutral child host and authoritative provenance."""
    root = Path(__file__).resolve().parents[2]
    core_path = root / "thinharness" / "core.py"
    source = core_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert not any(module.endswith(("subagents", "plugins.subagents")) for module in imported_modules)
    forbidden_names = {
        "SubAgentConfig",
        "SubagentsPlugin",
        "SubAgentArgs",
        "DEFAULT_SUBAGENT_NAME",
        "create_subagent_tool",
        "build_child_harness",
        "_select_builtin_tools",
        "builtin_tools",
        "subagent_hooks",
    }
    assert forbidden_names.isdisjoint(names | attributes)
    assert not (root / "thinharness" / "subagents.py").exists()
    assert "PluginContext(root=self.root, model=self.model, child_harnesses=child_harnesses)" in source
    assert "harness=self" not in source.split("PluginContext(", 1)[1].split(")", 1)[0]


def test_subagents_plugin_owns_configuration_and_delegation_tool() -> None:
    """Feature vocabulary and tool construction stay in the plugin module."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "thinharness" / "plugins" / "subagents.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    string_values = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert {"SubagentsPlugin", "SubAgentConfig", "SubAgentArgs"} <= classes
    assert "subagent" in string_values
    assert "register_delegation_tool" in source


def test_child_inheritance_and_delegation_detection_are_structural() -> None:
    """Automatic inheritance and tracing do not use plugin or tool name guesses."""
    root = Path(__file__).resolve().parents[2]
    children_source = (root / "thinharness" / "children.py").read_text(encoding="utf-8")
    execution_source = (root / "thinharness" / "tool_execution.py").read_text(encoding="utf-8")
    execution_tree = ast.parse(execution_source)

    assert "isinstance(plugin, ChildInheritablePlugin)" in children_source
    comparisons = [node for node in ast.walk(execution_tree) if isinstance(node, ast.Compare)]
    compared_strings = {
        comparator.value
        for node in comparisons
        for comparator in node.comparators
        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str)
    }
    assert "subagent" not in compared_strings
    assert "composition.delegation" in execution_source


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
