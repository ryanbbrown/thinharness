from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import Harness, HarnessConfig, Hook, MCPPlugin, MCPServerStdio, ModelToolCall, ModelTurn


class DeterministicSession:
    """Drive one MCP call without provider credentials."""

    async def start(self, prompt: str, constants: Any, **_kwargs: Any) -> ModelTurn:
        assert prompt == "multiply"
        assert [tool["name"] for tool in constants.tools] == ["multiply"]
        return ModelTurn(
            tool_calls=[ModelToolCall(id="multiply-1", name="multiply", arguments='{"left":6,"right":7}')],
            raw={"id": "start"},
        )

    async def continue_with_tools(self, outputs: list[Any], constants: Any, **_kwargs: Any) -> ModelTurn:
        del constants
        assert len(outputs) == 1
        assert "product=42" in outputs[0].output
        return ModelTurn(text="product=42 MCP_DONE", raw={"id": "done"})

    async def continue_with_user_text(self, text: str, constants: Any, **_kwargs: Any) -> ModelTurn:
        raise AssertionError(f"unexpected user continuation: {text!r}, {constants!r}")

    def dump_state(self) -> None:
        return None


class DeterministicModel:
    """Return the deterministic MCP journey session."""

    model = "deterministic:mcp"
    api_key = "unused"
    provider = SimpleNamespace(name="Deterministic")

    def new_session(self) -> DeterministicSession:
        return DeterministicSession()


def main() -> None:
    """Run local stdio discovery, execution, and cleanup end to end."""
    asyncio.run(_run())


async def _run() -> None:
    with TemporaryDirectory(prefix="thinharness-e2e-mcp-") as raw_root:
        root = Path(raw_root)
        server_path = root / "tiny_mcp_server.py"
        server_path.write_text(SERVER_CODE, encoding="utf-8")
        tool_names: list[str] = []
        server = MCPServerStdio(sys.executable, [str(server_path)])

        async with Harness(
            HarnessConfig(root=root, builtin_tools=[], max_model_requests=2, max_tool_calls=1),
            model=DeterministicModel(),
            plugins=[MCPPlugin(servers=[server])],
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        ) as harness:
            assert harness.tools == []
            result = await harness.run("multiply")
            assert [tool.name for tool in harness.tools] == ["multiply"]

        assert tool_names == ["multiply"]
        assert result.text == "product=42 MCP_DONE"
        print(f"PASS mcp_journey tools={tool_names} server={server.id}")


SERVER_CODE = """
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tiny-e2e")


@mcp.tool()
def multiply(left: int, right: int) -> str:
    return f"product={left * right}"


if __name__ == "__main__":
    mcp.run()
""".lstrip()


if __name__ == "__main__":
    main()
