from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, Hook

MODEL = os.getenv("E2E_WORKSPACE_MODEL", "openai:gpt-5.2")
SYSTEM_PROMPT = """You are an exacting workspace agent. Use tools when instructed and keep the final answer brief."""
PROMPT = """
Complete this workspace journey using tools, not memory:
1. Use list on the workspace root.
2. Use glob for **/*.txt.
3. Read docs/brief.txt.
4. Search for the exact word "latency".
5. Use jsonl_search on data/events.jsonl for rows where kind eq incident.
6. Write reports/summary.md with a placeholder line "owner=OWNER_PENDING".
7. Edit reports/summary.md so the owner line becomes "owner=ThinHarness".

The final reports/summary.md must contain these exact lines:
status=workspace-tools-complete
owner=ThinHarness
incident_count=2

End your final answer with WORKSPACE_TOOLS_DONE.
""".strip()


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-workspace-") as raw_root:
        root = Path(raw_root)
        _seed_workspace(root)
        tool_names: list[str] = []
        trace_dir = root / "traces"

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=["read", "write", "edit", "search", "list", "glob", "jsonl_search"],
                max_model_requests=30,
                max_tool_calls=12,
                local_trace_dir=trace_dir,
            ),
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        )

        # Run
        result = harness.run_sync(PROMPT)

        # Assertions
        summary = root / "reports" / "summary.md"
        assert summary.exists(), "agent did not create reports/summary.md"
        text = summary.read_text(encoding="utf-8")
        assert "status=workspace-tools-complete" in text
        assert "owner=ThinHarness" in text
        assert "incident_count=2" in text
        for name in ["list", "glob", "read", "search", "jsonl_search", "write", "edit"]:
            assert name in tool_names, f"expected tool call: {name}; saw {tool_names}"
        _assert_local_trace(trace_dir)
        assert "WORKSPACE_TOOLS_DONE" in result.text
        print(f"PASS workspace_tools_journey model={MODEL} tools={tool_names}")


def _seed_workspace(root: Path) -> None:
    (root / "docs").mkdir()
    (root / "data").mkdir()
    (root / "docs" / "brief.txt").write_text(
        "ThinHarness tracks small agent loops.\nLatency matters for every provider call.\n",
        encoding="utf-8",
    )
    (root / "docs" / "notes.txt").write_text("secondary note\n", encoding="utf-8")
    (root / "data" / "events.jsonl").write_text(
        '{"kind":"incident","id":"inc-001","severity":"high"}\n'
        '{"kind":"notice","id":"note-001","severity":"low"}\n'
        '{"kind":"incident","id":"inc-002","severity":"medium"}\n',
        encoding="utf-8",
    )


def _assert_local_trace(trace_dir: Path) -> None:
    trace_files = list(trace_dir.rglob("*.jsonl"))
    assert trace_files, "local tracing did not create a trace JSONL file"
    records = []
    for path in trace_files:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    names = {record["name"] for record in records}
    assert "invoke_agent thinharness" in names
    assert any(name.startswith("chat ") for name in names)
    assert "execute_tool read" in names
    serialized = json.dumps(records)
    assert "WORKSPACE_TOOLS_DONE" in serialized
    assert "owner=ThinHarness" in serialized


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP workspace_tools_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP workspace_tools_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
