from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "site" / "examples.html"
LONG_PREVIEW_CHARS = 1200
WEB_RESEARCH_REPORT_META = {
    "slug": "web_research_report",
    "title": "Web Research Report Agent",
    "overview": (
        "A live Exa-backed research agent that plans a market landscape brief, searches and fetches web sources, "
        "creates parallel source notes, asks a citation critic to review the draft, and writes a source-grounded report."
    ),
    "expected_flow": [
        "Write a research plan before searching.",
        "Run targeted Exa searches and inspect the saved JSONL results.",
        "Select high-value sources, fetch full contents, and write source artifacts.",
        "Prepare full-record source-note prompts and summarize them with parallel_llm.",
        "Draft the report from source notes and run citation_critic against the evidence.",
        "Revise the draft and submit a structured receipt with open questions and next action.",
    ],
    "expected_output": (
        "A credible markdown market landscape report with executive summary, method, findings, "
        "source notes, risks, recommended next steps, and a source list."
    ),
}
AGENT_META = {"web_research_report": WEB_RESEARCH_REPORT_META}


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def to_text(value: Any) -> str:
    value = parse_jsonish(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def clean_system_text(text: str) -> str:
    return text.replace("\n\nNo skills are configured.", "").replace("No skills are configured.", "").rstrip()


def system_text_from_attr(value: Any) -> str:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        chunks = []
        for item in parsed:
            if isinstance(item, dict) and item.get("content"):
                chunks.append(str(item["content"]))
        if chunks:
            return clean_system_text("\n\n".join(chunks))
    if isinstance(parsed, dict) and parsed.get("content"):
        return clean_system_text(str(parsed["content"]))
    return clean_system_text(to_text(parsed))


def one_line(value: Any, limit: int = 220) -> str:
    text = " ".join(to_text(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def call_label(index: int, tool_name: str, count: int) -> str:
    return f"{index:02d} - {tool_name} #{count}"


def model_label(index: int, request_kind: str) -> str:
    return f"{index:02d} - model · {(request_kind or 'turn').replace('_', ' ')}"


def run_label(index: int, name: str) -> str:
    return f"{index:02d} - run · {name.removeprefix('invoke_agent ')}"


def normalized_call_id(call_id: Any) -> str:
    return str(call_id or "").lower()


def refresh_search_text(event: dict[str, Any]) -> None:
    searchable = {key: value for key, value in event.items() if key != "search_text"}
    event["search_text"] = to_text(searchable).lower()


def relabel_call_references(events: list[dict[str, Any]]) -> None:
    labels_by_call_id = {
        normalized_call_id(event.get("tool_call_id")): str(event.get("display_label") or "")
        for event in events
        if event.get("tool_call_id") and event.get("display_label")
    }

    for event in events:
        if event["kind"] in {"tool", "subagent"} and event.get("display_label"):
            event["tool_label"] = event["display_label"]

        for message in event.get("input_messages") or []:
            if not isinstance(message, dict):
                continue
            for part in message.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                label = labels_by_call_id.get(normalized_call_id(part.get("id")))
                if label:
                    part["label"] = label
                    part["display_label"] = label

        for call in event.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            label = labels_by_call_id.get(normalized_call_id(call.get("id")))
            if label:
                call["label"] = label
                call["display_label"] = label


def latest_trace_files(summary_path: Path, summary: dict[str, Any]) -> list[Path]:
    root = summary_path.parents[1]
    files = [Path(trace) if Path(trace).is_absolute() else root / trace for trace in summary.get("trace_files", [])]
    files = [path for path in files if path.exists()]
    if files:
        return files
    traces_root = root / "outputs" / "traces"
    traces = sorted(traces_root.glob("**/*.jsonl"), key=lambda path: path.stat().st_mtime)
    return traces[-1:] if traces else []


def display_trace_path(trace_path: Path, root: Path) -> str:
    try:
        return str(trace_path.relative_to(root))
    except ValueError:
        return str(trace_path)


def spec_audit_metadata() -> dict[str, dict[str, Any]]:
    return AGENT_META


def tool_result_parts(raw_result: Any) -> dict[str, Any]:
    parsed = parse_jsonish(raw_result)
    if not isinstance(parsed, dict) or "content" not in parsed:
        full = to_text(parsed)
        return {
            "ok": None,
            "preview": one_line(full, LONG_PREVIEW_CHARS),
            "full": full,
            "metadata": "",
            "error_type": "",
        }

    content = parse_jsonish(parsed.get("content"))
    full = to_text(content)
    metadata = parsed.get("metadata")
    return {
        "ok": parsed.get("ok"),
        "preview": one_line(full, LONG_PREVIEW_CHARS),
        "full": full,
        "metadata": to_text(metadata) if metadata else "",
        "error_type": str(metadata.get("error_type", "")) if isinstance(metadata, dict) else "",
    }


def format_message_part(part: dict[str, Any], call_labels: dict[str, str]) -> dict[str, Any]:
    part_type = str(part.get("type") or "text")
    if part_type == "text":
        return {"kind": "text", "text": str(part.get("content") or "")}
    if part_type == "tool_call":
        call_id = str(part.get("id") or "")
        args = parse_jsonish(part.get("arguments"))
        return {
            "kind": "tool_call",
            "id": call_id,
            "name": part.get("name"),
            "label": call_labels.get(call_id, str(part.get("name") or "")),
            "text": one_line(args),
            "body": to_text(args),
        }
    if part_type == "tool_result":
        call_id = str(part.get("id") or "")
        result = tool_result_parts(part.get("content"))
        return {
            "kind": "tool_result",
            "id": call_id,
            "label": call_labels.get(call_id, call_id),
            "ok": result["ok"],
            "text": result["preview"],
            "body": result["full"],
            "metadata": result["metadata"],
        }
    return {"kind": part_type, "text": to_text(part)}


def format_messages(raw_messages: Any, call_labels: dict[str, str]) -> list[dict[str, Any]]:
    messages = parse_jsonish(raw_messages)
    if not isinstance(messages, list):
        return []
    formatted: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            parts = [{"type": "text", "content": message.get("content", "")}]
        formatted.append(
            {
                "role": str(message.get("role") or "message"),
                "parts": [format_message_part(part, call_labels) for part in parts if isinstance(part, dict)],
            }
        )
    return formatted


def assistant_text_from_messages(messages: list[dict[str, Any]], completion: Any) -> str:
    chunks: list[str] = []
    if completion:
        chunks.append(str(completion))
    for message in messages:
        if message["role"] != "assistant":
            continue
        for part in message["parts"]:
            if part["kind"] == "text" and part.get("text"):
                chunks.append(str(part["text"]))
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = chunk.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return "\n\n".join(deduped)


def tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "assistant":
            continue
        for part in message["parts"]:
            if part["kind"] == "tool_call":
                calls.append(part)
    return calls


def final_result_from_tool_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for call in tool_calls:
        if call.get("name") != "final_result":
            continue
        parsed = parse_jsonish(call.get("body"))
        if isinstance(parsed, dict):
            return parsed
    return None


def event_from_span(span: dict[str, Any], trace_rel: str, index: int, call_labels: dict[str, str]) -> dict[str, Any]:
    attrs = span.get("attributes") or {}
    name = str(span.get("name") or "")
    base = {
        "index": index,
        "trace": trace_rel,
        "trace_id": span.get("trace_id"),
        "span_id": span.get("span_id"),
        "parent_id": span.get("parent_id"),
        "name": name,
        "started_at": span.get("started_at"),
        "duration_ms": span.get("duration_ms"),
    }

    if name.startswith("chat "):
        input_messages = format_messages(attrs.get("gen_ai.input.messages"), call_labels)
        output_messages = format_messages(attrs.get("gen_ai.output.messages"), call_labels)
        tool_calls = tool_calls_from_messages(output_messages)
        final_result = final_result_from_tool_calls(tool_calls)
        assistant_text = assistant_text_from_messages(output_messages, attrs.get("gen_ai.completion"))
        event = {
            **base,
            "kind": "chat",
            "model": attrs.get("gen_ai.request.model") or name.removeprefix("chat "),
            "request_kind": attrs.get("thinharness.model.request.kind") or "",
            "finish": attrs.get("gen_ai.response.finish_reasons") or [],
            "input_messages": input_messages,
            "assistant_text": assistant_text,
            "tool_calls": [call for call in tool_calls if call.get("name") != "final_result"],
            "final_result": final_result,
            "is_final": bool(attrs.get("gen_ai.output.finalized")),
            "usage": {
                "input": attrs.get("gen_ai.usage.input_tokens"),
                "output": attrs.get("gen_ai.usage.output_tokens"),
                "total": attrs.get("gen_ai.usage.total_tokens"),
            },
        }
    elif name.startswith("execute_tool "):
        tool_name = str(attrs.get("gen_ai.tool.name") or name.removeprefix("execute_tool "))
        args = parse_jsonish(attrs.get("gen_ai.tool.call.arguments"))
        result = tool_result_parts(attrs.get("gen_ai.tool.call.result"))
        is_subagent = tool_name == "subagent"
        event = {
            **base,
            "kind": "subagent" if is_subagent else "tool",
            "tool_name": tool_name,
            "tool_call_id": attrs.get("gen_ai.tool.call.id") or "",
            "tool_label": call_labels.get(str(attrs.get("gen_ai.tool.call.id") or ""), tool_name),
            "subagent_name": attrs.get("subagent.name") if is_subagent else "",
            "args": to_text(args),
            "args_preview": one_line(args),
            "result_preview": result["preview"],
            "result_full": result["full"],
            "result_ok": result["ok"],
            "metadata": result["metadata"],
            "error": bool(attrs.get("error.type") or span.get("exceptions") or result["ok"] is False),
            "error_type": attrs.get("error.type") or result["error_type"],
            "exceptions": to_text(span.get("exceptions")),
        }
    elif name.startswith("invoke_agent "):
        event = {
            **base,
            "kind": "agent",
            "conversation_id": attrs.get("gen_ai.conversation.id") or "",
            "prompt": to_text(attrs.get("langfuse.trace.input")),
            "output": to_text(attrs.get("langfuse.trace.output")),
            "system": system_text_from_attr(attrs.get("gen_ai.system_instructions")),
            "error": bool(attrs.get("error.type") or span.get("exceptions")),
            "exceptions": to_text(span.get("exceptions")),
        }
    else:
        event = {
            **base,
            "kind": "span",
            "title": name,
            "body": to_text(attrs),
            "error": bool(attrs.get("error.type") or span.get("exceptions")),
            "exceptions": to_text(span.get("exceptions")),
        }

    refresh_search_text(event)
    return event


def load_agents() -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    audit_metadata = spec_audit_metadata()
    for summary_path in sorted(EXAMPLES_ROOT.glob("*/outputs/run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        root = summary_path.parents[1]
        slug = str(summary.get("slug") or root.name)
        agent_meta = audit_metadata.get(slug, {})
        trace_paths = latest_trace_files(summary_path, summary)
        spans_by_trace: list[tuple[dict[str, Any], str]] = []
        call_labels: dict[str, str] = {}
        for trace_path in trace_paths:
            rel = display_trace_path(trace_path, root)
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                span = json.loads(line)
                spans_by_trace.append((span, rel))
        events: list[dict[str, Any]] = []
        for span, rel in spans_by_trace:
            events.append(event_from_span(span, rel, len(events) + 1, call_labels))
        events.sort(key=lambda event: (event.get("started_at") or 0, event["index"]))
        event_counts: dict[str, int] = {}
        for event_index, event in enumerate(events):
            if event["kind"] == "chat":
                event["display_label"] = model_label(event_index, str(event.get("request_kind") or "turn"))
            elif event["kind"] in {"tool", "subagent"}:
                tool_name = str(event.get("tool_name") or event["kind"])
                event_counts[tool_name] = event_counts.get(tool_name, 0) + 1
                event["display_label"] = call_label(event_index, tool_name, event_counts[tool_name])
                if event["kind"] == "subagent" and event.get("subagent_name"):
                    event["display_label"] += f" · {event['subagent_name']}"
            elif event["kind"] == "agent":
                event["display_label"] = run_label(event_index, str(event.get("name") or "run"))
            else:
                event["display_label"] = f"{event_index:02d} - {event.get('name') or event['kind']}"
        relabel_call_references(events)
        for event in events:
            refresh_search_text(event)
        system_prompt = next(
            (
                event["system"]
                for event in events
                if event["kind"] == "agent" and event.get("name") == "invoke_agent thinharness" and event.get("system")
            ),
            "",
        )
        validation = summary.get("validation") if isinstance(summary.get("validation"), dict) else {}
        receipt = summary.get("receipt") if isinstance(summary.get("receipt"), dict) else {}
        usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
        report_path = str(validation.get("report_path") or receipt.get("report_path") or "")
        report_file = root / report_path if report_path else None
        report_text = report_file.read_text(encoding="utf-8") if report_file and report_file.exists() else ""
        report = {
            "path": str(root.relative_to(REPO_ROOT) / report_path) if report_path else "",
            "word_count": len(re.findall(r"\S+", report_text)),
            "validated": bool(validation.get("validated")),
            "source_notes_succeeded": validation.get("source_notes_succeeded"),
            "source_count": receipt.get("source_count"),
            "model_requests": usage.get("model_requests"),
            "tool_calls": usage.get("tool_calls"),
        }
        summary_text = summary.get("summary")
        if not summary_text and receipt:
            summary_text = (
                f"Completed with {receipt.get('source_count', 'unknown')} cited sources and a "
                f"{report['word_count']:,}-word final report at {receipt.get('report_path', report_path)}."
            )
        agents.append(
            {
                "slug": slug,
                "title": summary.get("title") or agent_meta.get("title") or slug.replace("_", " ").title(),
                "overview": summary.get("overview") or agent_meta.get("overview", ""),
                "expected_flow": summary.get("expected_flow")
                or agent_meta.get("expected_flow", []),
                "expected_output": summary.get("expected_output")
                or agent_meta.get("expected_output", ""),
                "model": summary["model"],
                "stop_reason": summary["stop_reason"],
                "resume_stop_reason": summary.get("resume_stop_reason"),
                "summary": summary_text or "",
                "system_prompt": system_prompt,
                "tools": summary.get("tools", []),
                "trace_files": [display_trace_path(path, root) for path in trace_paths],
                "event_count": len(events),
                "events": events,
                "report": report,
            }
        )
    return agents


def render_html(agents: list[dict[str, Any]], *, template_path: Path | None = None) -> str:
    data = json.dumps({"agents": agents}, ensure_ascii=False)
    script_data = data.replace("</", "<\\/")
    if template_path and template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        pattern = re.compile(r'(<script id="trace-data" type="application/json">)(.*?)(</script>)', re.S)
        if pattern.search(template):
            return pattern.sub(lambda match: f"{match.group(1)}{script_data}{match.group(3)}", template, count=1)
    raise ValueError("examples template must contain <script id=\"trace-data\" type=\"application/json\">")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render example agent traces as readable example HTML.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    agents = load_agents()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(agents, template_path=args.output), encoding="utf-8")
    print(args.output)
    print(json.dumps({"agents": [agent["slug"] for agent in agents], "count": len(agents)}, indent=2))


if __name__ == "__main__":
    main()
