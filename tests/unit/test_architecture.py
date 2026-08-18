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
