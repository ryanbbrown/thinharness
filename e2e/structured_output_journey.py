from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, Hook


MODEL = os.getenv("E2E_STRUCTURED_MODEL", "openai:gpt-5-mini")
SYSTEM_PROMPT = """You are a structured-output extraction agent. Use tools before finalizing."""
PROMPT = """
Read inventory.txt, then return the requested structured object.
The sku is TH-001, the count is 7, and the status is ready.
""".strip()


class InventoryAnswer(BaseModel):
    sku: str
    count: int
    status: str
    source_file: str


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-structured-") as raw_root:
        root = Path(raw_root)
        (root / "inventory.txt").write_text("sku=TH-001\ncount=7\nstatus=ready\n", encoding="utf-8")
        tool_names: list[str] = []

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=["read"],
                output_type=InventoryAnswer,
                output_mode="native",
                max_model_requests=4,
                max_tool_calls=2,
            ),
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        )

        # Run
        result = harness.run_sync(PROMPT)

        # Assertions
        assert tool_names == ["read"], f"expected exactly one read call; saw {tool_names}"
        assert result.output == InventoryAnswer(sku="TH-001", count=7, status="ready", source_file="inventory.txt")
        assert result.stop_reason == "end_turn"
        print(f"PASS structured_output_journey model={MODEL} output={result.output.model_dump()}")


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP structured_output_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP structured_output_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
