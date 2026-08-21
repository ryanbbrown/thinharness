from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import (
    AfterSubagentRunContext,
    ChildHarnessRequest,
    Harness,
    HarnessConfig,
    HarnessError,
    Hook,
    PluginBinding,
    PluginContribution,
    SubAgentConfig,
    SubagentsPlugin,
    ToolResult,
    ToolSpec,
)

PARENT_MODEL = os.getenv("E2E_SUBAGENTS_MODEL", "anthropic:claude-sonnet-4-5-20250929")
CHILD_MODEL = os.getenv("E2E_SUBAGENTS_CHILD_MODEL", "openai:gpt-5-mini")
SYSTEM_PROMPT = """You are a delegation test parent. Follow the requested delegation steps exactly."""
PROMPT = """
Use the subagent tool exactly twice, in this order:
1. Call subagent with exactly one argument field: {"task":"Reply with exactly DEFAULT_CHILD_OK."}. The agent field must be absent; do not set it to "default".
2. Call subagent with agent="guard". Ask it to call attempt_nested once, then reply with exactly NAMED_CHILD_OK if that tool reports NESTED_BLOCKED.
Do not call attempt_nested yourself. After both calls, state both child markers and end with SUBAGENTS_JOURNEY_DONE.
""".strip()


class NestedAttemptPlugin:
    """Expose a child-inherited probe of the disabled child host."""

    name = "nested_attempt"

    def __init__(self, blocked: list[str]) -> None:
        self.blocked = blocked

    def for_child(self) -> NestedAttemptPlugin:
        return self

    def bind(self, context) -> PluginBinding:
        async def attempt_nested(_args) -> ToolResult:
            request = ChildHarnessRequest(
                agent_name="forbidden",
                agent_description="Forbidden nested child.",
                trace_agent_name="subagent.forbidden",
                task="This must not run.",
                tool_mode="explicit",
                system_prompt="This must not run.",
            )
            try:
                await context.child_harnesses.run(request)
            except HarnessError as exc:
                self.blocked.append(str(exc))
                return ToolResult(True, "NESTED_BLOCKED")
            raise AssertionError("child host allowed nested delegation")

        return PluginBinding(static=PluginContribution(tools=(
            ToolSpec("attempt_nested", "Verify that nested child creation is blocked.", {"type": "object", "properties": {}}, attempt_nested),
        )))


def main() -> None:
    """Run real default and named delegation, including the disabled child-host probe."""
    if _should_skip(PARENT_MODEL, CHILD_MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-subagents-") as raw_root:
        blocked: list[str] = []
        child_events: list[tuple[str, list[str], str]] = []

        def after_child(ctx) -> None:
            assert isinstance(ctx, AfterSubagentRunContext)
            assert ctx.result is not None
            child_events.append((ctx.agent, list(ctx.tools), ctx.result.text))

        probe = NestedAttemptPlugin(blocked)
        harness = Harness(
            HarnessConfig(
                root=Path(raw_root),
                model=PARENT_MODEL,
                system_prompt=SYSTEM_PROMPT,
                max_model_requests=12,
                max_tool_calls=4,
            ),
            plugins=[
                probe,
                SubagentsPlugin(agents=[
                    SubAgentConfig(
                        name="guard",
                        description="Tests that child delegation is disabled.",
                        system_prompt="Call attempt_nested once. If it returns NESTED_BLOCKED, reply with exactly NAMED_CHILD_OK.",
                        inherit_parent=True,
                        model=CHILD_MODEL,
                        max_model_requests=4,
                        max_tool_calls=1,
                    )
                ]),
            ],
            hooks=[Hook("after_subagent_run", after_child)],
        )

        result = harness.run_sync(PROMPT)
        delegation_records = [record for record in result.tool_call_records if record.get("call", {}).get("name") == "subagent"]
        metadata = [ToolResult.from_json(record["output"]).metadata for record in delegation_records]

        assert len(delegation_records) == 2, f"expected two delegations, got {len(delegation_records)}"
        assert [item["agent"] for item in metadata] == ["default", "guard"]
        assert [item["tool_mode"] for item in metadata] == ["inherited", "inherited"]
        assert all(item["tools"] == ["attempt_nested"] for item in metadata)
        assert [item["structured_output"] for item in metadata] == [False, False]
        assert blocked and "cannot create nested child harnesses" in blocked[0]
        assert [event[0] for event in child_events] == ["default", "guard"]
        assert "DEFAULT_CHILD_OK" in child_events[0][2]
        assert "NAMED_CHILD_OK" in child_events[1][2]
        assert "DEFAULT_CHILD_OK" in result.text
        assert "NAMED_CHILD_OK" in result.text
        assert "SUBAGENTS_JOURNEY_DONE" in result.text
        print(
            f"PASS subagents_journey parent={PARENT_MODEL} child={CHILD_MODEL} "
            f"agents={[item['agent'] for item in metadata]}"
        )


def _should_skip(*models: str) -> bool:
    if os.getenv("CI"):
        print("SKIP subagents_journey: CI is set")
        return True
    provider_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    missing = sorted({provider_env[model.split(":", 1)[0]] for model in models if not os.getenv(provider_env[model.split(":", 1)[0]])})
    if missing:
        print(f"SKIP subagents_journey: missing {', '.join(missing)}")
        return True
    return False


if __name__ == "__main__":
    main()
