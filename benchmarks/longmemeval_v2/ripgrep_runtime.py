"""Fail-fast ripgrep validation for the LongMemEval ThinHarness runtime."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from thinharness.tools.filesystem import FileTools


def verify_ripgrep_runtime() -> dict[str, Any]:
    """Prove that rg and ThinHarness JSONL search work in this process environment."""
    resolved = shutil.which("rg")
    if resolved is None:
        raise RuntimeError("LongMemEval ThinHarness runtime requires rg on PATH")
    rg_path = Path(resolved).resolve()
    version = subprocess.run(
        [str(rg_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]

    with tempfile.TemporaryDirectory(prefix="thinharness-lme-rg-") as temporary:
        root = Path(temporary)
        fixture = root / "probe.jsonl"
        fixture.write_text(
            json.dumps({"id": "probe-hit", "text": "runtime-ripgrep-needle"}) + "\n"
            + json.dumps({"id": "probe-miss", "text": "other"})
            + "\n",
            encoding="utf-8",
        )
        result = FileTools(root).jsonl_search(
            {
                "path": "probe.jsonl",
                "query": "runtime-ripgrep-needle",
                "fields": {"id": 0},
            }
        )
    if not result.ok:
        raise RuntimeError(f"ThinHarness jsonl_search ripgrep probe failed: {result.content}")
    if "rows_matched: 1" not in result.content or "probe-hit" not in result.content:
        raise RuntimeError("ThinHarness jsonl_search ripgrep probe returned unexpected output")
    command = result.metadata.get("cmd")
    if not isinstance(command, list) or not command or command[0] != "rg":
        raise RuntimeError("ThinHarness jsonl_search probe did not execute rg")

    return {
        "rg_path": str(rg_path),
        "rg_version": version,
        "path_contains_rg_parent": str(rg_path.parent) in os.getenv("PATH", "").split(os.pathsep),
        "jsonl_search": {
            "status": "passed",
            "command": command,
            "returncode": result.metadata.get("returncode"),
            "output_sha256": hashlib.sha256(result.content.encode()).hexdigest(),
            "matched_rows": 1,
        },
    }
