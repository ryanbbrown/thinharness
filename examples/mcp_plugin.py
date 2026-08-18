"""Connect one ordered group of MCP servers through MCPPlugin."""

import asyncio

from thinharness import Harness, HarnessConfig, MCPPlugin, MCPServerStdio


async def main() -> None:
    """Run an agent with tools discovered from a local MCP server."""
    async with Harness(
        HarnessConfig(root=".", model="openai:gpt-5.5", builtin_tools=[]),
        plugins=[
            MCPPlugin(
                servers=[
                    MCPServerStdio(
                        "python",
                        ["server.py"],
                        tool_prefix="docs",
                    )
                ]
            )
        ],
    ) as harness:
        result = await harness.run("Use the docs tools to answer the question.")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
