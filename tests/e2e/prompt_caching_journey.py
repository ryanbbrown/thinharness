from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thinharness import Harness, HarnessConfig

MODEL = os.getenv("E2E_PROMPT_CACHING_MODEL", "anthropic:claude-haiku-4-5")

# Anthropic's minimum cacheable prefix is model-dependent (up to 4096 tokens),
# so pad the system prompt well past the floor or nothing is ever cached.
SYSTEM_PROMPT = "You are a filesystem inspection agent. Use tools before answering.\n\n" + "\n".join(
    f"Reference note {i}: workspace files are plain UTF-8 text and paths are relative to the workspace root."
    for i in range(600)
)
PROMPT = "Read inventory.txt and report the sku value."


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-caching-") as raw_root:
        root = Path(raw_root)
        (root / "inventory.txt").write_text("sku=TH-001\n", encoding="utf-8")

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                builtin_tools=["read"],
                max_model_requests=4,
                max_tool_calls=2,
            )
        )

        # Run
        result = harness.run_sync(PROMPT)

        # Assertions
        assert result.usage.model_requests >= 2, f"expected a tool-use loop; got {result.usage.model_requests} request(s)"
        assert result.usage.cached_tokens > 0, f"expected cached input tokens on the follow-up request; usage={result.usage}"
        print(f"PASS prompt_caching_journey model={MODEL} cached_tokens={result.usage.cached_tokens}")


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP prompt_caching_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP prompt_caching_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
