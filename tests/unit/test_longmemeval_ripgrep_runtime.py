from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.longmemeval_v2.ripgrep_runtime import verify_ripgrep_runtime  # noqa: E402


def test_ripgrep_runtime_proves_jsonl_search_uses_rg() -> None:
    receipt = verify_ripgrep_runtime()

    assert Path(receipt["rg_path"]).name == "rg"
    assert receipt["rg_version"].startswith("ripgrep ")
    assert receipt["jsonl_search"]["status"] == "passed"
    assert receipt["jsonl_search"]["command"][0] == "rg"
    assert receipt["jsonl_search"]["returncode"] == 0
    assert receipt["jsonl_search"]["matched_rows"] == 1


def test_ripgrep_runtime_fails_when_rg_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    with pytest.raises(RuntimeError, match="requires rg on PATH"):
        verify_ripgrep_runtime()
