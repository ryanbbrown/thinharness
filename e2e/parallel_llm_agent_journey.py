from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from thinharness import Harness, HarnessConfig, Hook, ParallelLlmTool

AGENT_MODEL = os.getenv("E2E_PARALLEL_AGENT_MODEL", "openai:gpt-5-mini")
CUSTOM_TOOL_MODEL = os.getenv("E2E_PARALLEL_AGENT_TOOL_MODEL", os.getenv("E2E_PARALLEL_OPENROUTER_MODEL", "openrouter:google/gemini-2.5-flash"))
SYSTEM_PROMPT = """You are a parallel LLM tool e2e agent. Use the requested tools exactly and keep the final answer brief."""
PROMPT = """
Complete this journey using tools, not memory:
1. Call parallel_llm with these exact arguments:
   {"prompts":["Reply with BUILTIN_ONE only.","Reply with BUILTIN_TWO only."],"system":"Reply with the requested token only.","max_concurrency":2}
2. Call parallel_json with these exact arguments:
   {
     "prompts": [
       "Return only this JSON object: {\"name\":\"red\",\"count\":1}",
       "Return only this JSON object: {\"name\":\"blue\",\"count\":2}"
     ],
     "system": "Return compact JSON only.",
     "max_concurrency": 2
   }

After both tool calls finish, answer with PARALLEL_LLM_AGENT_DONE.
""".strip()


class ColorCount(BaseModel):
    """Structured custom parallel output."""

    name: str
    count: int


def main() -> None:
    if _should_skip(AGENT_MODEL, CUSTOM_TOOL_MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-parallel-agent-") as raw_root:
        root = Path(raw_root)
        tool_names: list[str] = []

        custom_tool = ParallelLlmTool(
            name="parallel_json",
            description="Run independent prompts that each return one validated JSON object.",
            model=CUSTOM_TOOL_MODEL,
            root=root,
            max_prompts=4,
            max_attempts=1,
            output_type=ColorCount,
            output_mode="prompted",
        ).spec()

        harness = Harness(
            HarnessConfig(
                root=root,
                model=AGENT_MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=["parallel_llm"],
                max_model_requests=8,
                max_tool_calls=4,
                tool_retries=2,
            ),
            tools=[custom_tool],
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        )

        result = harness.run_sync(PROMPT)

        assert "parallel_llm" in tool_names, tool_names
        assert "parallel_json" in tool_names, tool_names
        assert "PARALLEL_LLM_AGENT_DONE" in result.text, result.text
        outputs = {record["call"]["name"]: json.loads(record["output"]) for record in result.tool_call_records}
        builtin_payload = json.loads(outputs["parallel_llm"]["content"])
        custom_payload = json.loads(outputs["parallel_json"]["content"])
        assert builtin_payload["total"] == 2, builtin_payload
        assert builtin_payload["succeeded"] == 2, builtin_payload
        assert custom_payload["total"] == 2, custom_payload
        assert custom_payload["succeeded"] == 2, custom_payload
        parsed_custom = [item["result"] for item in custom_payload["results"]]
        assert parsed_custom == [{"name": "red", "count": 1}, {"name": "blue", "count": 2}], parsed_custom
        print(f"PASS parallel_llm_agent_journey agent_model={AGENT_MODEL} custom_tool_model={CUSTOM_TOOL_MODEL} tools={tool_names}")


def _should_skip(*models: str) -> bool:
    if os.getenv("CI"):
        print("SKIP parallel_llm_agent_journey: CI is set")
        return True
    for model in models:
        provider = model.split(":", 1)[0]
        env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
        if not os.getenv(env_name):
            print(f"SKIP parallel_llm_agent_journey: {env_name} is not set for {model}")
            return True
    return False


if __name__ == "__main__":
    main()
