from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import ParallelLlmTool
from thinharness.tools.base import _invoke_tool

PROVIDER_MODELS = {
    "openai": os.getenv("E2E_PARALLEL_OPENAI_MODEL", "openai:gpt-5-mini"),
    "anthropic": os.getenv("E2E_PARALLEL_ANTHROPIC_MODEL", "anthropic:claude-haiku-4-5-20251001"),
    "openrouter": os.getenv("E2E_PARALLEL_OPENROUTER_MODEL", "openrouter:google/gemini-2.5-flash"),
}


async def main() -> None:
    if os.getenv("CI"):
        print("SKIP parallel_llm_tool_journey: CI is set")
        return

    ran: list[str] = []
    for provider, model in PROVIDER_MODELS.items():
        if _missing_key(provider):
            print(f"SKIP parallel_llm_tool_journey provider={provider}: {_env_name(provider)} is not set")
            continue
        await _run_provider(provider, model)
        ran.append(provider)

    if not ran:
        print("SKIP parallel_llm_tool_journey: no provider API keys are set")
        return
    print(f"PASS parallel_llm_tool_journey providers={ran}")


async def _run_provider(provider: str, model: str) -> None:
    """Run the standalone tool against one live provider."""
    with TemporaryDirectory(prefix=f"thinharness-e2e-parallel-tool-{provider}-") as raw_root:
        root = Path(raw_root)
        (root / "prompts.json").write_text(json.dumps([
            "Reply with TOOL_ALPHA only.",
            "Reply with TOOL_BETA only.",
        ]), encoding="utf-8")
        spec = ParallelLlmTool(
            name=f"{provider}_parallel_llm",
            model=model,
            root=root,
            read_paths=["prompts.json"],
            write_paths=["outputs"],
            max_prompts=4,
            max_attempts=1,
        ).spec()

        output = await _invoke_tool(spec, {
            "source": {"kind": "file", "path": "prompts.json"},
            "system": "Follow each prompt exactly. Do not add punctuation or explanation.",
            "output_file": "outputs/results.json",
            "max_concurrency": 2,
        })
        envelope = json.loads(output.to_json())
        assert envelope["ok"] is True, envelope
        summary = json.loads(envelope["content"])
        assert summary["total"] == 2, summary
        assert summary["succeeded"] == 2, summary
        assert summary["failed"] == 0, summary
        assert summary["model_requests"] == 2, summary
        file_payload = json.loads((root / "outputs" / "results.json").read_text(encoding="utf-8"))
        results = [item["result"].strip() for item in file_payload["results"]]
        assert "TOOL_ALPHA" in results[0], results
        assert "TOOL_BETA" in results[1], results
        assert file_payload["model_requests"] == 2, file_payload


def _env_name(provider: str) -> str:
    """Return the API key environment variable for a provider."""
    return {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]


def _missing_key(provider: str) -> bool:
    """Return whether the provider API key is missing."""
    return not os.getenv(_env_name(provider))


if __name__ == "__main__":
    asyncio.run(main())
