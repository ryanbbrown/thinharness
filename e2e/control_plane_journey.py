from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, HarnessError, Hook, ModelRetry, ToolSpec


MODEL = os.getenv("E2E_CONTROL_MODEL", "openrouter:google/gemini-2.5-flash")
SYSTEM_PROMPT = """You are a control-plane test agent. Follow tool-use instructions exactly."""
PROMPT = """
Call flaky_health_check once with {"reason": "control-plane"} before doing anything else.
Do not answer directly. The tool is expected to ask for a retry.
""".strip()


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-control-") as raw_root:
        root = Path(raw_root)
        events: list[str] = []

        def flaky_health_check(_args: dict) -> str:
            raise ModelRetry("retry requested by e2e control-plane script")

        def record(event: str):
            def handler(ctx):
                suffix = f":{getattr(ctx, 'tool_name', getattr(ctx, 'limit_kind', ctx.stop_reason if event == 'run_end' else ''))}"
                events.append(event + suffix)

            return handler

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=[],
                max_model_requests=4,
                max_tool_calls=2,
                tool_retries=0,
                tool_execution="sequential",
            ),
            tools=[
                ToolSpec(
                    "flaky_health_check",
                    "Always asks the model to retry, so retry limits can be validated.",
                    {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                    flaky_health_check,
                )
            ],
            hooks=[
                Hook("run_start", record("run_start")),
                Hook("user_prompt_submit", record("user_prompt_submit")),
                Hook("before_tool_call", record("before_tool_call")),
                Hook("after_tool_call", record("after_tool_call")),
                Hook("limit_reached", record("limit_reached")),
                Hook("run_end", record("run_end")),
            ],
        )

        # Run
        try:
            harness.run_sync(PROMPT)
        except HarnessError as exc:
            error = exc
        else:
            raise AssertionError("expected HarnessError from tool_retries=0")

        # Assertions
        assert "max_retries=0" in str(error)
        assert events[:4] == [
            "run_start:",
            "user_prompt_submit:",
            "before_tool_call:flaky_health_check",
            "after_tool_call:flaky_health_check",
        ], events
        assert "limit_reached:tool_retries" in events, events
        assert events[-1] == "run_end:tool_retries_exceeded", events
        print(f"PASS control_plane_journey model={MODEL} events={events}")


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP control_plane_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP control_plane_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
