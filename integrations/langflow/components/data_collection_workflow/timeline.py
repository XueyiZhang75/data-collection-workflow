from typing import Any

import requests
import time

from langflow.custom import Component
from langflow.io import BoolInput, DataInput, IntInput, Output, StrInput
from langflow.schema import Data, Message


VISUAL_CONTRACT_VERSION = "workflow-ui/2"
TERMINAL_RUN_STATUSES = {"completed", "failed"}
SNAPSHOT_REQUEST_TIMEOUT_SECONDS = 60


def coerce_wait_for_completion(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"", "false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _duration_text(duration_ms: Any) -> str:
    if duration_ms is None:
        return "pending"
    try:
        duration = float(duration_ms)
    except (TypeError, ValueError):
        return str(duration_ms)
    if duration >= 1000:
        return f"{duration / 1000:.1f} s"
    return f"{int(duration)} ms"


def format_timeline_text(snapshot: dict[str, Any]) -> str:
    progress = snapshot.get("node_progress")
    progress = progress if isinstance(progress, dict) else {}
    timeline = snapshot.get("node_timeline")
    timeline = timeline if isinstance(timeline, list) else []
    lines = [
        "# Data Collection Live Timeline / Real Node Durations",
        "",
        "> Langflow card time is not the real workflow duration. Use this timeline.",
        "",
        f"- Session: `{snapshot.get('session_id') or 'pending'}`",
        f"- Run status: `{snapshot.get('status') or 'pending'}`",
        f"- Current node: `{snapshot.get('current_node') or 'none'}`",
        (
            "- Progress: "
            f"`{progress.get('completed_count', 0)}/{progress.get('total_count', 0)} completed`"
        ),
        f"- Running: `{', '.join(progress.get('running_nodes') or []) or 'none'}`",
        f"- Failed: `{', '.join(progress.get('failed_nodes') or []) or 'none'}`",
        "",
        "| # | Node | Status | Real duration | Last message |",
        "|---:|---|---|---:|---|",
    ]
    for index, item in enumerate(timeline, start=1):
        if not isinstance(item, dict):
            continue
        node_name = item.get("node_name") or "unknown"
        status = item.get("status") or "pending"
        duration = _duration_text(item.get("duration_ms"))
        message = str(item.get("last_message") or "").replace("|", "\\|")
        lines.append(f"| {index} | {node_name} | {status} | {duration} | {message} |")
    return "\n".join(lines)


def fetch_timeline_snapshot(
    api_base_url: str,
    session_id: str,
    *,
    wait_for_completion: bool = False,
    timeout_seconds: int = 1800,
    poll_interval_seconds: int = 2,
    request_get=requests.get,
    sleep=time.sleep,
    status_callback=None,
) -> dict[str, Any]:
    base_url = str(api_base_url).rstrip("/")
    clean_session = str(session_id).strip()
    if not clean_session:
        raise ValueError("Session ID is required to load the Data Collection live timeline.")
    endpoint = f"{base_url}/runs/{clean_session}?visual_contract_version={VISUAL_CONTRACT_VERSION}"
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
                "status": "polling",
                "current_node": "unknown",
                "node_progress": {},
                "node_timeline": [],
                "last_message": f"Timeline request timed out; retrying. {exc}",
            }
        snapshot["timeline_text"] = format_timeline_text(snapshot)
        if status_callback:
            status_callback(snapshot)
        status = str(snapshot.get("status") or "").lower()
        if not wait_for_completion or status in TERMINAL_RUN_STATUSES:
            break
        if time.monotonic() >= deadline:
            snapshot["polling_timed_out"] = True
            snapshot["timeline_text"] = format_timeline_text(snapshot)
            break
        sleep(poll_interval)
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


class WorkflowTimeline(Component):
    display_name = "Data Collection Live Timeline / Real Node Durations"
    description = "Refresh real workflow status and node durations from the local Data Collection API."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "activity"
    name = "timeline"

    inputs = [
        StrInput(
            name="api_base_url",
            display_name="API Base URL",
            value="http://127.0.0.1:8010",
        ),
        StrInput(name="workflow_session_id", display_name="Session ID", value=""),
        DataInput(
            name="previous_node_status",
            display_name="Run Metadata",
            advanced=False,
            required=False,
        ),
        BoolInput(
            name="wait_for_completion",
            display_name="Wait For Completion",
            value=False,
            advanced=False,
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
        Output(display_name="Timeline Report", name="timeline_report", method="get_timeline_message"),
        Output(display_name="Timeline Data", name="timeline_data", method="get_timeline"),
    ]

    def _get_timeline_once(self) -> dict[str, Any]:
        api_base_url = resolve_api_base_url(self.api_base_url, self.previous_node_status)
        session_id = resolve_session_id(self.workflow_session_id, self.previous_node_status)
        wait = coerce_wait_for_completion(self.wait_for_completion)
        cache_key = (api_base_url, session_id, wait)
        cached_key = getattr(self, "_workflow_timeline_snapshot_key", None)
        cached = getattr(self, "_workflow_timeline_snapshot", None)
        if cached_key == cache_key and isinstance(cached, dict):
            return cached
        self.status = "Loading real Data Collection timeline"

        def _update_status(snapshot: dict[str, Any]) -> None:
            progress = snapshot.get("node_progress")
            progress = progress if isinstance(progress, dict) else {}
            self.status = (
                f"Real Data Collection status: {snapshot.get('status') or 'pending'}; "
                f"current node: {snapshot.get('current_node') or 'none'}; "
                f"completed: {progress.get('completed_count', 0)}/{progress.get('total_count', 0)}"
            )

        snapshot = fetch_timeline_snapshot(
            api_base_url,
            session_id,
            wait_for_completion=wait,
            timeout_seconds=int(self.timeout_seconds),
            poll_interval_seconds=int(self.poll_interval_seconds),
            status_callback=_update_status,
        )
        self._workflow_timeline_snapshot = snapshot
        self._workflow_timeline_snapshot_key = cache_key
        self.status = snapshot.get("timeline_text") or format_timeline_text(snapshot)
        return snapshot

    def get_timeline_message(self) -> Message:
        snapshot = self._get_timeline_once()
        return Message(text=snapshot.get("timeline_text") or format_timeline_text(snapshot), data=snapshot)

    def get_timeline(self) -> Data:
        snapshot = self._get_timeline_once()
        return Data(data=snapshot)
