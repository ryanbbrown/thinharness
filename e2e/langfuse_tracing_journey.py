from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, SubAgentConfig, TracingOptions, create_otlp_tracing

MODEL = os.getenv("E2E_LANGFUSE_TRACING_MODEL", "openrouter:anthropic/claude-haiku-4.5")
SYSTEM_PROMPT = "You are a tracing validation parent. Do your own parent checks, then delegate child file work to the named subagent."
PROMPT = """
Complete this tracing validation journey:

Parent steps:
1. Use list on the workspace root.
2. Read parent-input.md.
3. Use the subagent tool with agent="writer" for the child steps below.
4. After the subagent returns, write outputs/parent-summary.md with a concise parent summary that mentions the subagent result and parent-input.md.

Child steps for the writer subagent:
1. Create outputs/subagent-draft.md with title "Draft" and one sentence about Langfuse tracing.
2. Create outputs/subagent-notes.md with two short bullet notes.
3. Read outputs/subagent-draft.md back.
4. Edit outputs/subagent-draft.md so the title is "Revised Draft" and it mentions nested spans.
5. Return a concise summary to the parent.

The parent must perform the parent steps itself. End the final answer with LANGFUSE_TRACING_DONE.
""".strip()


def main() -> None:
    """Run a live Langfuse tracing journey."""
    if _should_skip(MODEL):
        return

    tracing = _create_langfuse_otlp_tracing()
    try:
        with TemporaryDirectory(prefix="thinharness-e2e-langfuse-") as raw_root:
            root = Path(raw_root)
            trace_dir = root / "traces"
            (root / "parent-input.md").write_text(
                "# Parent Input\n\nParent should inspect this file before and after subagent delegation.\n",
                encoding="utf-8",
            )
            harness = Harness(
                HarnessConfig(
                    root=root,
                    model=MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    builtin_tools=["list", "read", "write", "subagent"],
                    max_model_requests=40,
                    max_tool_calls=12,
                    local_trace_dir=trace_dir,
                    subagents=[
                        SubAgentConfig(
                            name="writer",
                            description="Creates and revises files for tracing validation.",
                            builtin_tools=["read", "write", "edit", "list"],
                            max_model_requests=20,
                            max_tool_calls=8,
                        )
                    ],
                ),
                tracing=[TracingOptions(
                    tracer=tracing.tracer,
                    agent_name="langfuse-parent",
                    capture_messages=True,
                    capture_tool_args=True,
                    capture_tool_results=True,
                )],
            )

            result = harness.run_sync(PROMPT)

            draft = root / "outputs" / "subagent-draft.md"
            notes = root / "outputs" / "subagent-notes.md"
            parent_summary = root / "outputs" / "parent-summary.md"
            assert draft.exists(), "subagent did not create outputs/subagent-draft.md"
            assert notes.exists(), "subagent did not create outputs/subagent-notes.md"
            assert parent_summary.exists(), "parent did not create outputs/parent-summary.md"
            assert "Revised Draft" in draft.read_text(encoding="utf-8")
            assert "nested spans" in draft.read_text(encoding="utf-8")
            summary_text = parent_summary.read_text(encoding="utf-8")
            assert "parent-input.md" in summary_text
            assert "subagent" in summary_text.lower()
            _assert_local_trace(trace_dir)
            assert "LANGFUSE_TRACING_DONE" in result.text
            print(f"PASS langfuse_tracing_journey model={MODEL} root={root}")
    finally:
        tracing.force_flush()
        tracing.shutdown()


def _should_skip(model: str) -> bool:
    """Return whether required live credentials are missing."""
    if os.getenv("CI"):
        print("SKIP langfuse_tracing_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    missing = [name for name in [env_name, "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"] if not os.getenv(name)]
    if missing:
        print(f"SKIP langfuse_tracing_journey: missing {', '.join(missing)}")
        return True
    return False


def _create_langfuse_otlp_tracing():
    """Create generic OTLP tracing configured for Langfuse."""
    public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
    secret_key = os.environ["LANGFUSE_SECRET_KEY"]
    host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return create_otlp_tracing(
        service_name="thinharness-langfuse-e2e",
        endpoint=host.rstrip("/") + "/api/public/otel/v1/traces",
        headers={
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        },
    )


def _assert_local_trace(trace_dir: Path) -> None:
    """Assert local trace JSONL was written alongside Langfuse export."""
    trace_files = list(trace_dir.rglob("*.jsonl"))
    assert trace_files, "local tracing did not create a trace JSONL file"
    records = []
    for path in trace_files:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    names = {record["name"] for record in records}
    assert "invoke_agent thinharness" in names
    assert "invoke_agent subagent.writer" in names
    assert "execute_tool subagent" in names
    serialized = json.dumps(records)
    assert "LANGFUSE_TRACING_DONE" in serialized
    assert "nested spans" in serialized


if __name__ == "__main__":
    main()
