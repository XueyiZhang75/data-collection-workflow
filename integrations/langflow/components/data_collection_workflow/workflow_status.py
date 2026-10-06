import time
from collections.abc import Callable
from typing import Any

import requests

from langflow.custom import Component
from langflow.io import BoolInput, IntInput, Output, StrInput
from langflow.schema import Data


TERMINAL_STATUSES = {"completed", "failed"}
VISUAL_CONTRACT_VERSION = "workflow-ui/2"


def _value_or_pending(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "pending"


def format_status_text(snapshot: dict[str, Any]) -> str:
    artifact_urls = snapshot.get("artifact_urls")
    artifact_keys = sorted(artifact_urls) if isinstance(artifact_urls, dict) else []
    lines = [
        f"Session: {_value_or_pending(snapshot.get('session_id'))}",
        f"Status: {_value_or_pending(snapshot.get('status'))}",
        f"Current node: {_value_or_pending(snapshot.get('current_node'))}",
        f"Live dashboard: {_value_or_pending(snapshot.get('live_dashboard_url'))}",
        f"Workflow console: {_value_or_pending(snapshot.get('workflow_console_url'))}",
        f"Workflow visualization: {_value_or_pending(snapshot.get('workflow_visualization_url'))}",
        f"Status API: {_value_or_pending(snapshot.get('status_url'))}",
    ]
    if artifact_keys:
        lines.append(f"Artifact URLs: {', '.join(artifact_keys)}")
    error = snapshot.get("error")
    if error:
        lines.append(f"Error: {error}")
    if snapshot.get("polling_timed_out"):
        lines.append("Polling: timed out before the run reached a terminal status")
    return "\n".join(lines)


def fetch_run_status(
    api_base_url: str,
    session_id: str,
    *,
    request_get: Callable[..., Any] = requests.get,
) -> dict[str, Any]:
    base_url = str(api_base_url).rstrip("/")
    cleaned_session_id = str(session_id).strip()
    if not cleaned_session_id:
        raise ValueError("Session ID is required to query workflow status.")
    response = request_get(
        f"{base_url}/runs/{cleaned_session_id}?visual_contract_version={VISUAL_CONTRACT_VERSION}",
        timeout=10,
    )
    response.raise_for_status()
    snapshot = response.json()
    snapshot["status_text"] = format_status_text(snapshot)
    return snapshot


class DataCollectionWorkflowStatus(Component):
    display_name = "Data Collection Workflow Status"
    description = "Query or poll a local Data Collection Workflow run through the demo API."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "activity"
    name = "workflow_status"

    inputs = [
        StrInput(
            name="api_base_url",
            display_name="API Base URL",
            value="http://127.0.0.1:8010",
        ),
        StrInput(name="session_id", display_name="Session ID", value=""),
        BoolInput(name="wait_until_finished", display_name="Wait Until Finished", value=False),
        IntInput(name="poll_interval_seconds", display_name="Poll Interval Seconds", value=5),
        IntInput(name="timeout_seconds", display_name="Timeout Seconds", value=120),
    ]

    outputs = [
        Output(display_name="Run Status", name="run_status", method="get_status"),
    ]

    def get_status(self) -> Data:
        poll_interval = max(1, int(self.poll_interval_seconds))
        timeout_seconds = max(1, int(self.timeout_seconds))
        snapshot = fetch_run_status(self.api_base_url, self.session_id)

        if bool(self.wait_until_finished) and snapshot.get("status") not in TERMINAL_STATUSES:
            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                time.sleep(poll_interval)
                snapshot = fetch_run_status(self.api_base_url, self.session_id)
                if snapshot.get("status") in TERMINAL_STATUSES:
                    break
            else:
                snapshot["polling_timed_out"] = True
                snapshot["status_text"] = format_status_text(snapshot)

        return Data(data=snapshot)
