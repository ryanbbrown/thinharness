from __future__ import annotations

import builtins

import pytest

from thinharness import MCPDependencyError, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP


async def test_construction_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing MCP servers is cheap, but connecting requires the extra."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Block imports from the optional mcp package."""
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("No module named 'mcp'")
        return real_import(name, globals, locals, fromlist, level)

    servers = [
        MCPServerStdio(command="x"),
        MCPServerSSE(url="http://localhost/sse"),
        MCPServerStreamableHTTP(url="http://localhost/mcp"),
    ]
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    for server in servers:
        with pytest.raises(MCPDependencyError, match="thinharness\\[mcp\\]"):
            await server.__aenter__()
