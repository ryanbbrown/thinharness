from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, Hook, MCPServerStdio


MODEL = os.getenv("E2E_MCP_MODEL", "openrouter:anthropic/claude-sonnet-4.5")
SYSTEM_PROMPT = """You are an MCP test agent. Use the discovered MCP tool for arithmetic."""
PROMPT = """
Use the MCP multiply tool to multiply 6 by 7.
Your final answer must include "product=42" and end with MCP_DONE.
""".strip()


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-mcp-") as raw_root:
        root = Path(raw_root)
        server_path = root / "tiny_mcp_server.py"
        server_path.write_text(SERVER_CODE, encoding="utf-8")
        tool_names: list[str] = []

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=[],
                mcp_servers=[MCPServerStdio(sys.executable, [str(server_path)])],
                max_model_requests=6,
                max_tool_calls=3,
            ),
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        )

        # Run
        result = harness.run_sync(PROMPT)

        # Assertions
        assert tool_names == ["multiply"], f"expected MCP multiply call; saw {tool_names}"
        assert "product=42" in result.text
        assert "MCP_DONE" in result.text
        print(f"PASS mcp_journey model={MODEL} tools={tool_names}")


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


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP mcp_journey: CI is set")
        return True
    if importlib.util.find_spec("mcp") is None:
        print("SKIP mcp_journey: install MCP support with `uv sync --extra mcp`")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP mcp_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
