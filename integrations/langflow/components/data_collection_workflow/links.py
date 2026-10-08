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
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


REPORT_KEYS = (
    "final_report_english",
    "report_bundle",
    "final_dataset_csv",
    "final_dataset_json",
    "final_package_json",
)


def format_deep_links_text(snapshot: dict[str, Any]) -> str:
    artifact_urls = snapshot.get("artifact_urls")
    artifact_urls = artifact_urls if isinstance(artifact_urls, dict) else {}
    progress = snapshot.get("node_progress")
    progress = progress if isinstance(progress, dict) else {}
    timeline = snapshot.get("node_timeline")
    timeline = timeline if isinstance(timeline, list) else []
    lines = [
        "Langflow card time is component build time; node durations below are real Data Collection Workflow durations.",
        "",
        f"Session: {snapshot.get('session_id') or 'pending'}",
        f"Status: {snapshot.get('status') or 'pending'}",
        f"Current node: {snapshot.get('current_node') or 'none'}",
        f"Session directory: {snapshot.get('session_dir') or 'pending'}",
        f"Generated config: {snapshot.get('generated_config_path') or 'pending'}",
        (
            "Date range: "
            f"{snapshot.get('normalized_start_date') or 'pending'} to "
            f"{snapshot.get('normalized_end_date') or 'pending'}"
        ),
        f"Quick test mode: {bool(snapshot.get('quick_test_mode'))}",
        (
            "Progress: "
            f"{progress.get('completed_count', 0)}/{progress.get('total_count', 0)} completed"
        ),
        f"Running: {', '.join(progress.get('running_nodes') or []) or 'none'}",
        f"Failed: {', '.join(progress.get('failed_nodes') or []) or 'none'}",
        f"LangSmith trace: {snapshot.get('langsmith_trace_url') or 'pending'}",
        f"Workflow console: {snapshot.get('workflow_console_url') or 'pending'}",
        f"Workflow visualization: {snapshot.get('workflow_visualization_url') or 'pending'}",
    ]
    if snapshot.get("result_location_text"):
        lines.append("Result locations:")
        lines.append(str(snapshot["result_location_text"]))
    visible_timeline = []
    for item in timeline:
        if not isinstance(item, dict):
            continue
        status = item.get("status") or "pending"
        if status in {"completed", "running", "failed"}:
            duration = item.get("duration_ms")
            suffix = f" ({duration} ms)" if duration else ""
            visible_timeline.append(f"- {item.get('node_name')}: {status}{suffix}")
    if visible_timeline:
        lines.append("Node timeline:")
        lines.extend(visible_timeline[-8:])
    for key in REPORT_KEYS:
        if artifact_urls.get(key):
            lines.append(f"{key}: {artifact_urls[key]}")
    return "\n".join(lines)


def require_terminal_run_snapshot(
    snapshot: dict[str, Any],
    *,
    session_id: str,
    wait_for_completion: bool,
) -> None:
    if not wait_for_completion:
        return
    status = str(snapshot.get("status") or "").lower()
    if status in TERMINAL_RUN_STATUSES:
        return
    timeout_note = " after polling timed out" if snapshot.get("polling_timed_out") else ""
    raise TimeoutError(
        f"Data Collection run {session_id} is still {status or 'unknown'}{timeout_note}; "
        "Langflow will not mark final results complete until the real workflow finishes."
    )


def fetch_deep_links(
    api_base_url: str,
    session_id: str,
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
    if not clean_session:
        raise ValueError("Session ID is required to build deep demo links.")
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
                "last_message": f"Run snapshot request timed out; retrying. {exc}",
            }
            snapshot["deep_links_text"] = format_deep_links_text(snapshot)
            if status_callback:
                status_callback(snapshot)
            if time.monotonic() >= deadline:
                snapshot["polling_timed_out"] = True
                snapshot["deep_links_text"] = format_deep_links_text(snapshot)
                break
            sleep(poll_interval)
            continue
        snapshot["deep_links_text"] = format_deep_links_text(snapshot)
        if status_callback:
            status_callback(snapshot)
        status = str(snapshot.get("status") or "").lower()
        if not wait_for_completion or status in TERMINAL_RUN_STATUSES:
            break
        if time.monotonic() >= deadline:
            snapshot["polling_timed_out"] = True
            snapshot["deep_links_text"] = format_deep_links_text(snapshot)
            break
        sleep(poll_interval)
    require_terminal_run_snapshot(
        snapshot,
        session_id=clean_session,
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


class WorkflowLinks(Component):
    display_name = "Data Collection Final Results - Run Full Workflow"
    description = "Click this node once to run the full Data Collection visual chain and return final results."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "external-link"
    name = "links"

    inputs = [
        StrInput(
            name="api_base_url",
            display_name="API Base URL",
            value="http://127.0.0.1:8010",
        ),
        StrInput(name="workflow_session_id", display_name="Session ID", value=""),
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
        Output(display_name="Final Results", name="final_results", method="get_links_message"),
        Output(display_name="Deep Links Data", name="deep_links", method="get_links"),
    ]

    def _get_links_once(self) -> dict[str, Any]:
        api_base_url = resolve_api_base_url(self.api_base_url, self.previous_node_status)
        session_id = resolve_session_id(self.workflow_session_id, self.previous_node_status)
        cache_key = (
            api_base_url,
            session_id,
            coerce_wait_for_completion(self.wait_for_completion),
        )
        cached_key = getattr(self, "_links_snapshot_key", None)
        cached = getattr(self, "_links_snapshot", None)
        if cached_key == cache_key and isinstance(cached, dict):
            return cached
        self.status = "Loading Data Collection Workflow progress"

        def _update_status(snapshot: dict[str, Any]) -> None:
            progress = snapshot.get("node_progress")
            progress = progress if isinstance(progress, dict) else {}
            self.status = (
                f"Data Collection run {snapshot.get('session_id') or 'pending'}: "
                f"{snapshot.get('status') or 'pending'}; "
                f"current node: {snapshot.get('current_node') or 'none'}; "
                f"completed: {progress.get('completed_count', 0)}/{progress.get('total_count', 0)}"
            )

        snapshot = fetch_deep_links(
            api_base_url,
            session_id,
            wait_for_completion=coerce_wait_for_completion(self.wait_for_completion),
            timeout_seconds=int(self.timeout_seconds),
            poll_interval_seconds=int(self.poll_interval_seconds),
            status_callback=_update_status,
        )
        self._links_snapshot = snapshot
        self._links_snapshot_key = cache_key
        self.status = snapshot.get("deep_links_text") or format_deep_links_text(snapshot)
        if snapshot.get("status") == "failed":
            raise RuntimeError(snapshot.get("error") or "Data Collection Workflow run failed.")
        return snapshot

    def get_links(self) -> Data:
        snapshot = self._get_links_once()
        return Data(data=snapshot)

    def get_links_message(self) -> Message:
        snapshot = self._get_links_once()
        return Message(text=snapshot.get("deep_links_text") or format_deep_links_text(snapshot), data=snapshot)
