from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fakes import (
    MultiCallClient,
    _fake_openai,
    slow_tool,
    tool_output,
)

from thinharness import (
    Harness,
    HarnessConfig,
    ToolSpec,
)
from thinharness.tools import FileTools


def test_tool_spec_sequential_default_and_not_in_schema() -> None:
    spec = ToolSpec("echo", "Echo", {"type": "object", "properties": {}}, lambda args: "ok")
    assert spec.sequential is False
    assert "sequential" not in spec.response_tool()
    flagged = ToolSpec("write_thing", "writes", {"type": "object", "properties": {}}, lambda args: "ok", sequential=True)
    assert flagged.sequential is True
    assert "sequential" not in flagged.response_tool()

def test_builtin_tools_mark_mutating_specs_sequential(tmp_path: Path) -> None:
    by_name = {spec.name: spec for spec in FileTools(tmp_path).specs()}
    assert by_name["read"].sequential is False
    assert by_name["search"].sequential is False
    assert by_name["list"].sequential is False
    assert by_name["glob"].sequential is False
    assert by_name["jsonl_search"].sequential is False
    assert by_name["write"].sequential is True
    assert by_name["edit"].sequential is True

def test_parallel_safe_batch_runs_concurrently(tmp_path: Path) -> None:
    delay = 0.2
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay)],
    )

    start = time.monotonic()
    result = harness.run("go")
    elapsed = time.monotonic() - start

    assert result.text == "done"
    assert elapsed < delay * 1.8, f"expected concurrent execution, elapsed={elapsed:.3f}s"
    assert len(client.payloads) == 2
    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == ["slow_a", "slow_b"]

def test_sequential_tool_forces_serial_batch(tmp_path: Path) -> None:
    delay = 0.2
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay, sequential=True)],
    )

    start = time.monotonic()
    result = harness.run("go")
    elapsed = time.monotonic() - start

    assert result.text == "done"
    assert elapsed >= delay * 1.9, f"expected serial execution, elapsed={elapsed:.3f}s"
    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]

def test_tool_execution_sequential_forces_serial_even_for_safe_tools(tmp_path: Path) -> None:
    delay = 0.15
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[], tool_execution="sequential"),
        model=_fake_openai(client),
        tools=[slow_tool("slow_a", delay), slow_tool("slow_b", delay)],
    )

    start = time.monotonic()
    harness.run("go")
    elapsed = time.monotonic() - start

    assert elapsed >= delay * 1.9

def test_parallel_batch_preserves_model_call_order(tmp_path: Path) -> None:
    client = MultiCallClient([("slow_first", "{}"), ("fast_second", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[slow_tool("slow_first", 0.2), slow_tool("fast_second", 0.01)],
    )

    harness.run("go")

    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == ["call_1", "call_2"]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == ["slow_first", "fast_second"]

def test_parallel_batch_continues_when_one_tool_errors(tmp_path: Path) -> None:
    client = MultiCallClient([("boom", "{}"), ("ok", "{}")])

    def boom(_args):
        raise RuntimeError("nope")

    boom_spec = ToolSpec("boom", "Always raises.", {"type": "object", "properties": {}}, boom)
    ok_spec = ToolSpec("ok", "Returns ok.", {"type": "object", "properties": {}}, lambda args: "ok")
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[boom_spec, ok_spec],
    )

    result = harness.run("go")

    assert result.text == "done"
    continuation_inputs = client.payloads[1]["input"]
    assert continuation_inputs[0]["call_id"] == "call_1"
    assert "RuntimeError" in continuation_inputs[0]["output"]
    assert continuation_inputs[1]["call_id"] == "call_2"
    assert tool_output(continuation_inputs[1]["output"])["content"] == "ok"

def test_parallel_batch_makes_one_provider_continuation(tmp_path: Path) -> None:
    client = MultiCallClient([("a", "{}"), ("b", "{}"), ("c", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[slow_tool("a", 0.01), slow_tool("b", 0.01), slow_tool("c", 0.01)],
    )

    harness.run("go")

    assert client.invocations == 2
    assert len(client.payloads[1]["input"]) == 3

def test_truncate_spill_files_do_not_collide_under_parallel_reads(tmp_path: Path) -> None:
    big = "x" * 200 + "\n" + "y" * 200 + "\n"
    (tmp_path / "a.txt").write_text(big, encoding="utf-8")
    (tmp_path / "b.txt").write_text(big, encoding="utf-8")
    client = MultiCallClient([("read", '{"path":"a.txt","max_chars":50}'), ("read", '{"path":"b.txt","max_chars":50}')])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", max_tool_chars=50, max_read_chars=50),
        model=_fake_openai(client),
    )

    harness.run("go")

    saved_paths = []
    for item in client.payloads[1]["input"]:
        body = json.loads(item["output"])
        assert body["metadata"]["truncated"] is True
        saved_paths.append(body["metadata"]["saved_to"])
    assert saved_paths[0] != saved_paths[1]
    assert all(Path(path).exists() for path in saved_paths)

def test_dict_tool_can_opt_into_sequential(tmp_path: Path) -> None:
    delay = 0.15
    sequential_dict_tool = {
        "name": "slow_b",
        "description": "Slow, sequential",
        "parameters": {"type": "object", "properties": {}},
        "handler": lambda args: (time.sleep(delay), "slow_b")[1],
        "sequential": True,
    }
    client = MultiCallClient([("slow_a", "{}"), ("slow_b", "{}")])
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[slow_tool("slow_a", delay), sequential_dict_tool],
    )

    start = time.monotonic()
    harness.run("go")
    elapsed = time.monotonic() - start

    assert elapsed >= delay * 1.9, f"expected serial execution, elapsed={elapsed:.3f}s"

def test_parallel_batch_with_more_calls_than_worker_cap(tmp_path: Path) -> None:
    batch = [(f"t{i}", "{}") for i in range(20)]
    client = MultiCallClient(batch)
    tools = [slow_tool(name, 0.01) for name, _ in batch]
    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=tools,
    )

    harness.run("go")

    continuation_inputs = client.payloads[1]["input"]
    assert [item["call_id"] for item in continuation_inputs] == [f"call_{i+1}" for i in range(20)]
    assert [tool_output(item["output"])["content"] for item in continuation_inputs] == [name for name, _ in batch]

def test_parallel_batch_tools_execute_in_separate_threads(tmp_path: Path) -> None:
    client = MultiCallClient([("track_a", "{}"), ("track_b", "{}")])
    seen_threads: list[int] = []
    barrier = threading.Barrier(2, timeout=2)

    def run_a(_args):
        seen_threads.append(threading.get_ident())
        barrier.wait()
        return "a"

    def run_b(_args):
        seen_threads.append(threading.get_ident())
        barrier.wait()
        return "b"

    harness = Harness(
        HarnessConfig(root=tmp_path, model="openai:test-model", builtin_tools=[]),
        model=_fake_openai(client),
        tools=[
            ToolSpec("track_a", "a", {"type": "object", "properties": {}}, run_a),
            ToolSpec("track_b", "b", {"type": "object", "properties": {}}, run_b),
        ],
    )

    harness.run("go")

    assert len(seen_threads) == 2
    assert seen_threads[0] != seen_threads[1]
