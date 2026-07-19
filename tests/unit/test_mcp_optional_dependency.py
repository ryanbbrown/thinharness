from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path

import pytest

from thinharness import MCPDependencyError, MCPServer, MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP


def _block_imports(monkeypatch: pytest.MonkeyPatch, blocked: set[str]) -> None:
    """Make imports of the given top-level packages fail."""
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Block imports from the optional packages."""
        if name.split(".")[0] in blocked:
            raise ImportError(f"No module named {name.split('.')[0]!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


async def test_construction_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing MCP servers is cheap, but connecting requires the extra."""
    _block_imports(monkeypatch, {"mcp", "fastmcp"})

    servers = [
        MCPServerStdio(command="x"),
        MCPServerSSE(url="http://localhost/sse"),
        MCPServerStreamableHTTP(url="http://localhost/mcp"),
        MCPServer(object()),
    ]

    for server in servers:
        with pytest.raises(MCPDependencyError, match="thinharness\\[mcp\\]"):
            await server.__aenter__()


async def test_missing_fastmcp_alone_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale environment with mcp but no fastmcp still gets the install hint."""
    _block_imports(monkeypatch, {"fastmcp"})
    server = MCPServerStdio(command="x")

    with pytest.raises(MCPDependencyError, match="thinharness\\[mcp\\]") as excinfo:
        await server.__aenter__()

    assert "fastmcp" in str(excinfo.value.__cause__)


_SUBPROCESS_TEMPLATE = """
import asyncio, sys

BLOCKED = {blocked!r}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"No module named {{name!r}}")

sys.meta_path.insert(0, Blocker())

import thinharness

servers = [
    thinharness.MCPServerStdio(command="x"),
    thinharness.MCPServerSSE(url="http://localhost/sse"),
    thinharness.MCPServerStreamableHTTP(url="http://localhost/mcp"),
    thinharness.MCPServer(object()),
]

async def main():
    for server in servers:
        try:
            await server.__aenter__()
        except thinharness.MCPDependencyError as exc:
            assert "thinharness[mcp]" in str(exc), str(exc)
        else:
            raise SystemExit(f"expected MCPDependencyError for {{type(server).__name__}}")
    print("OK")

asyncio.run(main())
"""


@pytest.mark.parametrize("blocked", [{"mcp"}, {"fastmcp"}, {"mcp", "fastmcp"}], ids=["mcp", "fastmcp", "both"])
def test_missing_dependency_in_fresh_interpreter(blocked: set[str]) -> None:
    """A fresh process without the optional packages imports thinharness and gets the hint."""
    code = _SUBPROCESS_TEMPLATE.format(blocked=blocked)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
