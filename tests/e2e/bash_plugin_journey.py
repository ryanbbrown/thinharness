from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import BashPlugin, Harness, HarnessConfig, Hook, ToolResult

MODEL = os.getenv("E2E_BASH_MODEL", "openai:gpt-5.2")
PROMPT = """
Use the bash tool exactly once. Run this exact command:

printf BEGIN; printf '%0200d' 0; printf END

Read the bounded tool output that you receive. Your final answer must be exactly:
BASH_PLUGIN_DONE BEGIN END
""".strip()


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-bash-") as raw_root:
        root = Path(raw_root)
        tool_results: list[ToolResult] = []

        def capture(ctx) -> None:
            if ctx.tool_name == "bash":
                tool_results.append(ctx.envelope)

        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                max_model_requests=5,
                max_tool_calls=2,
            ),
            plugins=[BashPlugin(max_output_bytes=64)],
            hooks=[Hook("after_tool_call", capture, tools=["bash"])],
        )

        result = harness.run_sync(PROMPT)

        assert len(tool_results) == 1, f"expected one Bash call, saw {len(tool_results)}"
        tool_result = tool_results[0]
        assert tool_result.ok is True
        assert tool_result.metadata["stdout_bytes"] == 208
        assert tool_result.metadata["stdout_truncated"] is True
        assert tool_result.content.startswith("stdout:\nBEGIN")
        assert "bytes omitted" in tool_result.content
        assert "END\nstderr:\n(no output)" in tool_result.content
        assert result.text.strip() == "BASH_PLUGIN_DONE BEGIN END"
        print(f"PASS bash_plugin_journey model={MODEL}")


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP bash_plugin_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP bash_plugin_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
