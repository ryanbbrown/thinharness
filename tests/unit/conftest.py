from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_default_local_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests from writing default traces to the user home."""
    monkeypatch.setenv("THINHARNESS_DISABLE_LOCAL_TRACING", "1")
