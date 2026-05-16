from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from thinharness import (
    SkillRegistry,
)


def test_skill_registry_reads_and_runs_skill(tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\nBody", encoding="utf-8")
    script = skill / "scripts" / "echo.py"
    script.write_text("import sys\nprint('hi', *sys.argv[1:])\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")
    assert "demo - Demo skill" in registry.prompt_summary()
    read = registry.skill_read({"skill_name": "demo"})
    assert read.ok
    assert "SKILL.md" in read.content
    run = registry.skill_run({"skill_name": "demo", "script": "scripts/echo.py", "args": ["there"]})
    assert run.ok
    assert "hi there" in run.content

def test_skill_run_timeout_returns_structured_result(tmp_path: Path, monkeypatch) -> None:
    skill = tmp_path / "skills" / "demo"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: demo\n---\nBody", encoding="utf-8")
    (skill / "scripts" / "slow.py").write_text("print('slow')\n", encoding="utf-8")
    registry = SkillRegistry(tmp_path / "skills")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(kwargs.get("args", "python"), timeout=1)

    monkeypatch.setattr("subprocess.run", timeout)
    result = registry.skill_run({"skill_name": "demo", "script": "scripts/slow.py", "timeout": 1})

    assert not result.ok
    assert result.content == "skill script timed out after 1s"
    assert result.metadata["timeout"] == 1

def test_skill_registry_aggregates_dirs_and_filters_selected_skills(tmp_path: Path) -> None:
    alpha = tmp_path / "a" / "alpha"
    beta = tmp_path / "b" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "SKILL.md").write_text("---\nname: alpha\ndescription: Alpha skill\n---\nAlpha", encoding="utf-8")
    (beta / "SKILL.md").write_text("---\nname: beta\ndescription: Beta skill\n---\nBeta", encoding="utf-8")

    registry = SkillRegistry([tmp_path / "a", tmp_path / "b"], selected_skills=["beta"])

    assert list(registry.skills) == ["beta"]
    assert "beta - Beta skill" in registry.prompt_summary()
    assert "alpha - Alpha skill" not in registry.prompt_summary()

def test_skill_registry_rejects_duplicate_skill_names(tmp_path: Path) -> None:
    first = tmp_path / "first" / "demo"
    second = tmp_path / "second" / "demo"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "SKILL.md").write_text("---\nname: demo\n---\nFirst", encoding="utf-8")
    (second / "SKILL.md").write_text("---\nname: demo\n---\nSecond", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate skill name: demo"):
        SkillRegistry([tmp_path / "first", tmp_path / "second"])
