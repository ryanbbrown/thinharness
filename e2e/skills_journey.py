from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thinharness import Harness, HarnessConfig, Hook

MODEL = os.getenv("E2E_SKILLS_MODEL", "anthropic:claude-sonnet-4-5-20250929")
SYSTEM_PROMPT = """You are an exacting skill-using agent. Read a skill before running any script from it."""
PROMPT = """
Use the configured skill named arithmetic-auditor:
1. Call skill_read for arithmetic-auditor.
2. Call skill_run for scripts/sum_numbers.py with args ["4", "9"].
3. Use the script result in your final answer.

Your final answer must include "total=13" and end with SKILLS_DONE.
""".strip()


def main() -> None:
    if _should_skip(MODEL):
        return

    with TemporaryDirectory(prefix="thinharness-e2e-skills-") as raw_root:
        root = Path(raw_root)
        skills_dir = root / "skills"
        _seed_skill(skills_dir)
        tool_names: list[str] = []

        # Config
        harness = Harness(
            HarnessConfig(
                root=root,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                skills_dir=skills_dir,
                selected_skills=["arithmetic-auditor"],
                builtin_tools=["skill_read", "skill_run"],
                max_model_requests=6,
                max_tool_calls=4,
            ),
            hooks=[Hook("before_tool_call", lambda ctx: tool_names.append(ctx.tool_name))],
        )

        # Run
        result = harness.run_sync(PROMPT)

        # Assertions
        assert tool_names[:2] == ["skill_read", "skill_run"], f"expected skill_read then skill_run; saw {tool_names}"
        assert "total=13" in result.text
        assert "SKILLS_DONE" in result.text
        print(f"PASS skills_journey model={MODEL} tools={tool_names}")


def _seed_skill(skills_dir: Path) -> None:
    skill = skills_dir / "arithmetic-auditor"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\n"
        "name: arithmetic-auditor\n"
        "description: Adds small integer lists and reports a total.\n"
        "---\n"
        "Always use scripts/sum_numbers.py for arithmetic totals. Report the result as total=<number>.\n",
        encoding="utf-8",
    )
    (scripts / "sum_numbers.py").write_text(
        "from __future__ import annotations\n\n"
        "import sys\n\n"
        "values = [int(value) for value in sys.argv[1:]]\n"
        "print(f'total={sum(values)}')\n",
        encoding="utf-8",
    )


def _should_skip(model: str) -> bool:
    if os.getenv("CI"):
        print("SKIP skills_journey: CI is set")
        return True
    provider = model.split(":", 1)[0]
    env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}[provider]
    if not os.getenv(env_name):
        print(f"SKIP skills_journey: {env_name} is not set")
        return True
    return False


if __name__ == "__main__":
    main()
