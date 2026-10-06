from typing import Any

import requests
import time

from langflow.custom import Component
from langflow.io import BoolInput, DataInput, IntInput, Output, StrInput
from langflow.schema import Data, Message


VISUAL_CONTRACT_VERSION = "workflow-ui/2"
TERMINAL_NODE_STATUSES = {"completed", "failed"}
SNAPSHOT_REQUEST_TIMEOUT_SECONDS = 60
CARD_TIME_NOTE = (
    "Langflow card time is component build time. "
    "Use `Duration ms` here for the real Data Collection node duration."
)


def coerce_wait_for_completion(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def format_node_status_text(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Node: {snapshot.get('node_name') or 'pending'}",
        f"Status: {snapshot.get('status') or 'pending'}",
        f"Duration ms: {snapshot.get('duration_ms') or 'pending'}",
    ]
    llm = snapshot.get("llm_summary") if isinstance(snapshot.get("llm_summary"), dict) else {}
    tools = snapshot.get("tool_summary") if isinstance(snapshot.get("tool_summary"), dict) else {}
    if llm:
        lines.append(f"LLM: {llm}")
    if tools:
        lines.append(f"Tools: {tools}")
    if snapshot.get("trace_url"):
        lines.append(f"LangSmith trace: {snapshot['trace_url']}")
    if snapshot.get("error"):
        lines.append(f"Error: {snapshot['error']}")
    return "\n".join(lines)


def _compact_json_text(value: Any, *, max_chars: int = 1800) -> str:
    if value in (None, {}, []):
        return "pending"
    try:
        import json

        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n... [truncated]"


def format_node_detail_message(snapshot: dict[str, Any]) -> str:
    if snapshot.get("node_detail_markdown"):
        text = str(snapshot["node_detail_markdown"])
        if CARD_TIME_NOTE in text:
            return text
        return text + f"\n\n> {CARD_TIME_NOTE}"
    lines = [
        f"## {snapshot.get('node_name') or 'pending'}",
        "",
        f"> {CARD_TIME_NOTE}",
        "",
        f"- Status: `{snapshot.get('status') or 'pending'}`",
        f"- Duration ms: `{snapshot.get('duration_ms') or 'pending'}`",
        f"- Started: `{snapshot.get('started_at') or 'pending'}`",
        f"- Completed: `{snapshot.get('completed_at') or 'pending'}`",
        f"- Node detail API: {snapshot.get('node_status_url') or 'pending'}",
        "",
        "### Input summary",
        f"```json\n{_compact_json_text(snapshot.get('input_summary'))}\n```",
        "",
        "### Output summary",
        f"```json\n{_compact_json_text(snapshot.get('output_summary'))}\n```",
        "",
        "### Tool summary",
        f"```json\n{_compact_json_text(snapshot.get('tool_summary'))}\n```",
        "",
        "### LLM summary",
        f"```json\n{_compact_json_text(snapshot.get('llm_summary'))}\n```",
    ]
    return "\n".join(lines)


def require_terminal_node_snapshot(
    snapshot: dict[str, Any],
    *,
    node_name: str,
    wait_for_completion: bool,
) -> None:
    if not wait_for_completion:
        return
    status = str(snapshot.get("status") or "").lower()
    if status in TERMINAL_NODE_STATUSES:
        return
    timeout_note = " after polling timed out" if snapshot.get("polling_timed_out") else ""
    raise TimeoutError(
        f"Data Collection node {node_name} is still {status or 'unknown'}{timeout_note}; "
        "Langflow will not mark this node complete until the real workflow node finishes."
    )


def fetch_node_snapshot(
    api_base_url: str,
    session_id: str,
    node_name: str,
    *,
    wait_for_completion: bool = True,
    timeout_seconds: int = 1800,
    poll_interval_seconds: int = 2,
    request_get=requests.get,
    sleep=time.sleep,
    status_callback=None,
) -> dict[str, Any]:
    base_url = str(api_base_url).rstrip("/")
    clean_session = str(session_id).strip()
    clean_node = str(node_name).strip()
    if not clean_session:
        raise ValueError("Session ID is required to inspect a workflow node.")
    if not clean_node:
        raise ValueError("Node name is required to inspect a workflow node.")
    endpoint = (
        f"{base_url}/runs/{clean_session}/nodes/{clean_node}"
        f"?visual_contract_version={VISUAL_CONTRACT_VERSION}"
    )
    deadline = time.monotonic() + max(0, int(timeout_seconds))
    poll_interval = max(1, int(poll_interval_seconds))
    snapshot: dict[str, Any] = {}
    while True:
        try:
            response = request_get(endpoint, timeout=SNAPSHOT_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            snapshot = response.json()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            snapshot = {
                "session_id": clean_session,
                "node_name": clean_node,
                "status": "polling",
                "duration_ms": None,
                "last_message": f"Snapshot request timed out; retrying. {exc}",
            }
            if status_callback:
                status_callback(snapshot)
            if time.monotonic() >= deadline:
                snapshot["polling_timed_out"] = True
                snapshot["status_text"] = format_node_status_text(snapshot)
                snapshot["node_detail_markdown"] = format_node_detail_message(snapshot)
                break
            sleep(poll_interval)
            continue
        snapshot["status_text"] = snapshot.get("status_text") or format_node_status_text(snapshot)
        snapshot["node_detail_markdown"] = snapshot.get("node_detail_markdown") or format_node_detail_message(snapshot)
        if status_callback:
            status_callback(snapshot)
        status = str(snapshot.get("status") or "").lower()
        if not wait_for_completion or status in TERMINAL_NODE_STATUSES:
            break
        if time.monotonic() >= deadline:
            snapshot["polling_timed_out"] = True
            snapshot["status_text"] = format_node_status_text(snapshot)
            snapshot["node_detail_markdown"] = format_node_detail_message(snapshot)
            break
        sleep(poll_interval)
    require_terminal_node_snapshot(
        snapshot,
        node_name=clean_node,
        wait_for_completion=bool(wait_for_completion),
    )
    return snapshot


def resolve_session_id(explicit_session_id: str, previous_node_status: Any = None) -> str:
    payload = getattr(previous_node_status, "data", previous_node_status)
    if isinstance(payload, dict):
        session_id = payload.get("session_id")
        if session_id:
            return str(session_id).strip()
    explicit = str(explicit_session_id or "").strip()
    if explicit:
        return explicit
    return ""


def resolve_api_base_url(explicit_api_base_url: str, previous_node_status: Any = None) -> str:
    explicit = str(explicit_api_base_url or "").strip().rstrip("/")
    payload = getattr(previous_node_status, "data", previous_node_status)
    if isinstance(payload, dict):
        direct = payload.get("api_base_url")
        if direct:
            return str(direct).strip().rstrip("/")

        for key in ("status_url", "events_url", "node_status_url"):
            url = str(payload.get(key) or "").strip()
            if "/runs/" in url:
                return url.split("/runs/", 1)[0].rstrip("/")
    if explicit:
        return explicit
    return ""


class DataCollectionWorkflowNodeInspector(Component):
    display_name = "Data Collection Workflow Node Inspector"
    description = "Inspect one mirrored Data Collection LangGraph node from the local demo API."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "circle-dot"
    name = "node_inspector"

    inputs = [
        StrInput(
            name="api_base_url",
            display_name="API Base URL",
            value="http://127.0.0.1:8010",
        ),
        StrInput(name="workflow_session_id", display_name="Session ID", value=""),
        StrInput(
            name="node_name",
            display_name="Node Name",
            value="task_intake_and_scope_planning",
        ),
        DataInput(
            name="previous_node_status",
            display_name="Previous Node Status",
            advanced=False,
            required=False,
        ),
        BoolInput(
            name="wait_for_completion",
            display_name="Wait For Completion",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="poll_interval_seconds",
            display_name="Poll Interval Seconds",
            value=2,
            advanced=True,
        ),
        IntInput(
            name="timeout_seconds",
            display_name="Timeout Seconds",
            value=1800,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Node Status", name="node_status", method="inspect_node"),
        Output(display_name="Node Details", name="node_details", method="inspect_node_message"),
    ]

    def _inspect_node_once(self) -> dict[str, Any]:
        api_base_url = resolve_api_base_url(self.api_base_url, self.previous_node_status)
        session_id = resolve_session_id(self.workflow_session_id, self.previous_node_status)
        cache_key = (
            api_base_url,
            session_id,
            str(self.node_name),
            coerce_wait_for_completion(self.wait_for_completion),
        )
        cached_key = getattr(self, "_workflow_node_snapshot_key", None)
        cached = getattr(self, "_workflow_node_snapshot", None)
        if cached_key == cache_key and isinstance(cached, dict):
            return cached
        self.status = f"Waiting for Data Collection node: {self.node_name}"

        def _update_status(snapshot: dict[str, Any]) -> None:
            duration = snapshot.get("duration_ms")
            duration_text = f"{duration} ms" if duration is not None else "pending"
            self.status = (
                f"Data Collection node {snapshot.get('node_name') or self.node_name}: "
                f"{snapshot.get('status') or 'pending'}; "
                f"real duration: {duration_text}; "
                f"last event: {snapshot.get('last_message') or 'pending'}"
            )

        snapshot = fetch_node_snapshot(
            api_base_url,
            session_id,
            self.node_name,
            wait_for_completion=coerce_wait_for_completion(self.wait_for_completion),
            timeout_seconds=int(self.timeout_seconds),
            poll_interval_seconds=int(self.poll_interval_seconds),
            status_callback=_update_status,
        )
        self._workflow_node_snapshot = snapshot
        self._workflow_node_snapshot_key = cache_key
        self.status = snapshot.get("node_detail_markdown") or snapshot.get("status_text") or format_node_status_text(snapshot)
        if snapshot.get("status") == "failed":
            raise RuntimeError(snapshot.get("status_text") or f"{self.node_name} failed")
        return snapshot

    def inspect_node(self) -> Data:
        snapshot = self._inspect_node_once()
        return Data(data=snapshot)

    def inspect_node_message(self) -> Message:
        snapshot = self._inspect_node_once()
        return Message(text=format_node_detail_message(snapshot), data=snapshot)
