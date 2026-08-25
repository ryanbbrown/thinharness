import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLOBAL_AXTREE_PREVIEW_CHARS = 20_000
TRAJECTORY_SUMMARY_CONCISE_FILENAME = "TRAJECTORY_SUMMARY_CONCISE.md"
TRAJECTORY_SUMMARY_FULL_FILENAME = "TRAJECTORY_SUMMARY_FULL.md"

A11Y_LINE_RE = re.compile(r"^\s*\[([A-Za-z0-9_-]+)\]\s*(.+)$")
ACTION_OBJECT_ID_RE = re.compile(r"^[A-Za-z]*\d+[A-Za-z0-9_-]*$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_json(path: Path) -> Any:
    require(path.exists(), f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def goal_text(raw_goal: Any) -> str:
    if isinstance(raw_goal, str):
        stripped = raw_goal.strip()
        return stripped if stripped else "<goal not found>"
    if isinstance(raw_goal, list):
        parts = [part.strip() for part in raw_goal if isinstance(part, str) and part.strip()]
        if parts:
            return " ".join(parts)
    return "<goal not found>"


def _extract_object_lookup_from_tree(tree_text: str) -> dict[str, str]:
    object_lookup: dict[str, str] = {}
    for line in tree_text.splitlines():
        match = A11Y_LINE_RE.match(line)
        if not match:
            continue
        object_id = match.group(1)
        if object_id not in object_lookup:
            object_lookup[object_id] = line.strip()
    return object_lookup


def _action_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _extract_interacted_object_ids(action_text: str) -> list[str]:
    try:
        parsed = ast.parse(action_text, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(parsed, ast.Expression):
        return []
    call = parsed.body
    if not isinstance(call, ast.Call):
        return []
    name = _action_name(call.func)
    arg_indexes = [0, 1] if name == "drag_and_drop" else [0]
    object_ids: list[str] = []
    for index in arg_indexes:
        if index >= len(call.args):
            continue
        arg = call.args[index]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            continue
        object_id = arg.value.strip()
        if ACTION_OBJECT_ID_RE.match(object_id):
            object_ids.append(object_id)
    return object_ids


def annotate_action(action_text: str, object_lookup: dict[str, str]) -> str:
    """Annotate an action with the AXTree line(s) for the element id(s) it touches."""
    object_ids = _extract_interacted_object_ids(action_text)
    if not object_ids:
        return action_text
    details: list[str] = []
    seen_ids: set[str] = set()
    for object_id in object_ids:
        if object_id in seen_ids:
            continue
        seen_ids.add(object_id)
        detail = object_lookup.get(object_id)
        if detail:
            details.append(detail)
    if not details:
        return action_text
    return f"{action_text}  # {' | '.join(details)}"


def _optional_string(value: Any, *, field_name: str) -> str | None:
    require(value is None or isinstance(value, str), f"{field_name} must be string or null")
    return value


def _metadata_string(value: Any, *, fallback: str | None = None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _stable_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedTextTrajectory:
    trajectory_id: str
    trajectory: dict[str, Any]
    manifest_row: dict[str, Any]
    state_rows: tuple[dict[str, Any], ...]
    trajectory_state_rows: tuple[dict[str, Any], ...]
    action_rows: tuple[dict[str, Any], ...]
    fingerprint: str


def prepare_text_trajectory(trajectory: dict[str, object]) -> PreparedTextTrajectory:
    trajectory_id = trajectory.get("id")
    require(isinstance(trajectory_id, str) and trajectory_id, "trajectory id must be a non-empty string")

    public_states = trajectory.get("states")
    content = trajectory.get("content")
    metadata = trajectory.get("metadata")
    outcome = _optional_string(trajectory.get("outcome"), field_name=f"trajectory {trajectory_id} outcome")

    if isinstance(public_states, list) and public_states:
        domain = _metadata_string(trajectory.get("domain"))
        environment = _metadata_string(trajectory.get("environment"))
        goal = goal_text(trajectory.get("goal"))
        start_url_value = trajectory.get("start_url")
        require(
            isinstance(start_url_value, str) and start_url_value.strip(),
            f"trajectory start_url must be a non-empty string for {trajectory_id}",
        )
        raw_states = public_states
    else:
        require(
            isinstance(content, list) and content,
            f"trajectory content must be a non-empty list for {trajectory_id}",
        )
        require(
            isinstance(metadata, dict),
            f"trajectory metadata must be an object for {trajectory_id}",
        )
        domain = _metadata_string(trajectory.get("domain"), fallback=_metadata_string(metadata.get("domain")))
        environment = _metadata_string(
            trajectory.get("environment"),
            fallback=_metadata_string(metadata.get("environment")),
        )
        goal = goal_text(metadata.get("original_goal"))
        start_url_value = None
        raw_states = content

    simplified_states: list[dict[str, Any]] = []
    actions: list[str] = []
    known_object_lookup: dict[str, str] = {}
    for state_index, state in enumerate(raw_states):
        require(isinstance(state, dict), f"trajectory {trajectory_id} state {state_index} must be an object")
        url = state.get("url")
        action = state.get("action")
        thought = state.get("thought", state.get("thoughts"))
        screenshot = state.get("screenshot")
        observation = state.get("observation")
        if isinstance(observation, dict):
            text = observation.get("text")
        else:
            text = state.get("accessibility_tree", state.get("text"))
        require(
            isinstance(url, str) and url.strip(),
            f"trajectory {trajectory_id} state {state_index} missing url",
        )
        require(
            action is None or isinstance(action, str),
            f"trajectory {trajectory_id} state {state_index} action must be string or null",
        )
        require(
            thought is None or isinstance(thought, str),
            f"trajectory {trajectory_id} state {state_index} thought must be string or null",
        )
        require(
            isinstance(text, str),
            f"trajectory {trajectory_id} state {state_index} accessibility_tree/text must be string",
        )
        require(
            screenshot is None or isinstance(screenshot, str),
            f"trajectory {trajectory_id} state {state_index} screenshot must be string or null",
        )
        if start_url_value is None:
            start_url_value = url
        current_object_lookup = _extract_object_lookup_from_tree(text) if text else {}
        object_lookup = {**known_object_lookup, **current_object_lookup}
        action_annotated: str | None = None
        if isinstance(action, str) and action.strip():
            action_annotated = annotate_action(action.strip(), object_lookup)
            actions.append(action_annotated)
        known_object_lookup.update(current_object_lookup)
        step_value = state.get("step")
        original_state_index = state.get("state_index")
        simplified_states.append(
            {
                "state_index": state_index,
                "step": (
                    step_value
                    if isinstance(step_value, int) and not isinstance(step_value, bool)
                    else (
                        original_state_index
                        if isinstance(original_state_index, int)
                        and not isinstance(original_state_index, bool)
                        else state_index
                    )
                ),
                "url": url,
                "action": action,
                "action_annotated": action_annotated,
                "thought": thought,
                "accessibility_tree": text,
                "screenshot": screenshot,
            }
        )

    require(isinstance(start_url_value, str) and start_url_value.strip(), f"trajectory {trajectory_id} is missing a start url")
    simplified = {
        "id": trajectory_id,
        "domain": domain,
        "environment": environment,
        "goal": goal,
        "outcome": outcome,
        "start_url": start_url_value,
        "actions": actions,
        "states": simplified_states,
    }
    manifest_row = {
        "trajectory_id": trajectory_id,
        "domain": domain,
        "environment": environment,
        "outcome": outcome,
        "goal": goal,
        "start_url": start_url_value,
        "state_count": len(simplified_states),
        "action_count": len(actions),
    }
    state_rows: list[dict[str, Any]] = []
    trajectory_state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for state in simplified_states:
        full_state_row = {
            **manifest_row,
            "state_index": state["state_index"],
            "step": state["step"],
            "url": state["url"],
            "action": state["action"],
            "action_annotated": state["action_annotated"],
            "thought": state["thought"],
            "accessibility_tree": state["accessibility_tree"],
            "screenshot": state["screenshot"],
        }
        state_row = {
            **full_state_row,
            "accessibility_tree": full_state_row["accessibility_tree"][:GLOBAL_AXTREE_PREVIEW_CHARS],
        }
        trajectory_state_rows.append(full_state_row)
        state_rows.append(state_row)
        action = state["action"]
        if isinstance(action, str) and action.strip():
            action_rows.append(
                {
                    **manifest_row,
                    "state_index": state["state_index"],
                    "step": state["step"],
                    "url": state["url"],
                    "action": action,
                    "action_annotated": state["action_annotated"],
                }
            )

    return PreparedTextTrajectory(
        trajectory_id=trajectory_id,
        trajectory=simplified,
        manifest_row=manifest_row,
        state_rows=tuple(state_rows),
        trajectory_state_rows=tuple(trajectory_state_rows),
        action_rows=tuple(action_rows),
        fingerprint=_stable_fingerprint(simplified),
    )


def _markdown_table_cell(value: Any) -> str:
    return str(value).replace("\n", " ").strip().replace("|", "\\|")


def _render_concise_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Trajectory Summary Concise",
        "",
        f"Total trajectories: {len(records)}",
        "",
        "Sorted by start URL so similar surfaces appear near each other.",
        "",
        "| # | Trajectory ID | Goal | Outcome | States | Start URL |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(
            "| "
            + " | ".join(
                _markdown_table_cell(value)
                for value in [
                    index,
                    record["trajectory_id"],
                    record["goal"],
                    record["outcome"],
                    record["state_count"],
                    record["start_url"],
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_full_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Trajectory Summary Full",
        "",
        f"Total trajectories: {len(records)}",
        "",
        "Sorted by start URL so similar surfaces appear near each other.",
        "Action steps are annotated with the AXTree element they interact with.",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record['trajectory_id']}",
                "",
                f"Start URL: {record['start_url']}",
                "",
                f"Goal: {record['goal']}",
                "",
                "Action sequence:",
            ]
        )
        action_steps = record["action_steps"]
        if action_steps:
            for action_index, action_text in enumerate(action_steps, start=1):
                lines.append(f"{action_index}. {action_text}")
        else:
            lines.append("1. <no actions recorded>")
        lines.extend(["", f"Outcome: {record['outcome']}", ""])
    return "\n".join(lines).rstrip() + "\n"


class ThinHarnessCorpusBuilder:
    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.corpus_dir = self.workspace_dir / "corpus"
        self.trajectory_dir = self.corpus_dir / "trajectories"
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        for filename in ["trajectory_manifest.jsonl", "states.jsonl", "actions.jsonl"]:
            path = self.corpus_dir / filename
            if not path.exists():
                path.write_text("", encoding="utf-8")
        for filename, title in [
            (TRAJECTORY_SUMMARY_CONCISE_FILENAME, "# Trajectory Summary Concise\n\n"),
            (TRAJECTORY_SUMMARY_FULL_FILENAME, "# Trajectory Summary Full\n\n"),
        ]:
            summary_path = self.corpus_dir / filename
            if not summary_path.exists():
                summary_path.write_text(title, encoding="utf-8")

    def insert(self, trajectory: dict[str, object]) -> PreparedTextTrajectory:
        prepared = prepare_text_trajectory(trajectory)
        trajectory_dir = self.trajectory_dir / prepared.trajectory_id
        fingerprint_path = trajectory_dir / "fingerprint.txt"
        if fingerprint_path.exists():
            existing = fingerprint_path.read_text(encoding="utf-8").strip()
            require(
                existing == prepared.fingerprint,
                f"Existing text corpus trajectory differs for {prepared.trajectory_id}",
            )
            return prepared

        try:
            trajectory_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            if fingerprint_path.exists():
                existing = fingerprint_path.read_text(encoding="utf-8").strip()
                require(
                    existing == prepared.fingerprint,
                    f"Existing text corpus trajectory differs for {prepared.trajectory_id}",
                )
                return prepared
            raise RuntimeError(f"Trajectory dir exists without fingerprint: {trajectory_dir}") from exc
        save_json(trajectory_dir / "trajectory.json", prepared.trajectory)
        write_jsonl(trajectory_dir / "states.jsonl", list(prepared.trajectory_state_rows))
        write_jsonl(trajectory_dir / "actions.jsonl", list(prepared.action_rows))
        fingerprint_path.write_text(prepared.fingerprint + "\n", encoding="utf-8")

        self._append_jsonl(self.corpus_dir / "trajectory_manifest.jsonl", prepared.manifest_row)
        for row in prepared.state_rows:
            self._append_jsonl(self.corpus_dir / "states.jsonl", row)
        for row in prepared.action_rows:
            self._append_jsonl(self.corpus_dir / "actions.jsonl", row)
        return prepared

    def _append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    def rebuild_summaries(self) -> None:
        """Rebuild the concise and full markdown summaries from all inserted trajectories.

        Both files are sorted by start URL so similar surfaces cluster together, which
        helps the retrieval agent shortlist comparison candidates. Action sequences are
        already element-annotated in each trajectory.json, so this reads those directly.
        """
        records: list[dict[str, Any]] = []
        for trajectory_path in sorted(self.trajectory_dir.glob("*/trajectory.json")):
            trajectory = load_json(trajectory_path)
            require(isinstance(trajectory, dict), f"trajectory.json must be an object: {trajectory_path}")
            states = trajectory.get("states")
            actions = trajectory.get("actions")
            records.append(
                {
                    "trajectory_id": trajectory.get("id", trajectory_path.parent.name),
                    "start_url": trajectory.get("start_url", "<start url not found>"),
                    "goal": goal_text(trajectory.get("goal")),
                    "outcome": trajectory.get("outcome") or "<outcome not found>",
                    "action_steps": list(actions) if isinstance(actions, list) else [],
                    "state_count": len(states) if isinstance(states, list) else 0,
                }
            )
        records.sort(
            key=lambda record: (
                str(record["start_url"]).strip().lower(),
                str(record["goal"]).strip().lower(),
                str(record["trajectory_id"]).strip().lower(),
            )
        )
        (self.corpus_dir / TRAJECTORY_SUMMARY_CONCISE_FILENAME).write_text(
            _render_concise_markdown(records), encoding="utf-8"
        )
        (self.corpus_dir / TRAJECTORY_SUMMARY_FULL_FILENAME).write_text(
            _render_full_markdown(records), encoding="utf-8"
        )
