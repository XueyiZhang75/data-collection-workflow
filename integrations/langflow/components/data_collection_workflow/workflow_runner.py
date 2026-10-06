from typing import Any

import requests

from langflow.custom import Component
from langflow.io import BoolInput, MultilineInput, Output, StrInput
from langflow.schema import Data, Message


VISUAL_CONTRACT_VERSION = "workflow-ui/2"


def build_run_payload(
    *,
    disease: str,
    location: str,
    start_date: str,
    end_date: str,
    session_id: str = "",
    provider: str,
    model: str,
    no_llm: bool,
    user_request: str = "",
    quick_test_mode: bool = False,
) -> dict:
    payload = {
        "disease": disease,
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "provider": provider,
        "model": model,
        "no_llm": bool(no_llm),
        "quick_test_mode": bool(quick_test_mode),
        "dashboard_enabled": False,
        "visual_contract_version": VISUAL_CONTRACT_VERSION,
    }
    clean_request = str(user_request or "").strip()
    if clean_request:
        payload["user_request"] = clean_request
    clean_session = str(session_id or "").strip()
    if clean_session:
        payload["session_id"] = clean_session
    return payload


def format_runner_message(snapshot: dict[str, Any]) -> str:
    lines = [
        "## Data Collection Workflow Run",
        "",
        f"- Session ID: `{snapshot.get('session_id') or 'pending'}`",
        f"- Status: `{snapshot.get('status') or 'pending'}`",
        f"- Current node: `{snapshot.get('current_node') or 'pending'}`",
        f"- Date range: `{snapshot.get('normalized_start_date') or 'pending'}` to `{snapshot.get('normalized_end_date') or 'pending'}`",
        f"- Quick test mode: `{bool(snapshot.get('quick_test_mode'))}`",
        f"- Session directory: `{snapshot.get('session_dir') or 'pending'}`",
        f"- Generated config: `{snapshot.get('generated_config_path') or 'pending'}`",
        f"- Status API: {snapshot.get('status_url') or 'pending'}",
    ]
    if snapshot.get("result_location_text"):
        lines.extend(["", "### Result locations", snapshot["result_location_text"]])
    if snapshot.get("error"):
        lines.extend(["", f"Error: `{snapshot['error']}`"])
    return "\n".join(lines)


class DataCollectionWorkflowRunner(Component):
    display_name = "Data Collection Workflow Runner"
    description = "Create or reuse the backend Data Collection Workflow run. Use the final results node to run the full visual chain."
    documentation = "https://docs.langflow.org/components-custom-components"
    icon = "workflow"
    name = "workflow_runner"

    inputs = [
        StrInput(
            name="api_base_url",
            display_name="API Base URL",
            value="http://127.0.0.1:8010",
        ),
        MultilineInput(
            name="user_request",
            display_name="User Request",
            value="",
            required=False,
        ),
        StrInput(name="disease", display_name="Disease / Virus", value="", required=True),
        StrInput(name="location", display_name="Location", value="", required=True),
        StrInput(name="start_date", display_name="Start Date", value="", required=True),
        StrInput(name="end_date", display_name="End Date", value="", required=True),
        StrInput(name="session_id", display_name="Session ID", value=""),
        StrInput(
            name="provider", display_name="LLM Provider", value="",
            info="anthropic or openai; blank uses LLM_PROVIDER from the server environment.",
        ),
        StrInput(
            name="model", display_name="LLM Model", value="",
            info="Enter your provider's exact model ID; blank uses LLM_MODEL from the server environment.",
        ),
        BoolInput(name="no_llm", display_name="Disable LLM", value=False),
        BoolInput(name="quick_test_mode", display_name="Quick Test Mode", value=False),
    ]

    outputs = [
        Output(display_name="Run Metadata", name="run_metadata", method="run_workflow"),
        Output(display_name="Run Summary", name="run_summary", method="run_workflow_message"),
    ]

    def _run_workflow_once(self) -> dict[str, Any]:
        base_url = str(self.api_base_url).rstrip("/")
        payload = build_run_payload(
            disease=self.disease,
            location=self.location,
            start_date=self.start_date,
            end_date=self.end_date,
            session_id=self.session_id,
            provider=self.provider,
            model=self.model,
            no_llm=bool(self.no_llm),
            user_request=self.user_request,
            quick_test_mode=bool(self.quick_test_mode),
        )
        cache_key = (base_url, tuple(sorted(payload.items())))
        cached_key = getattr(self, "_workflow_run_snapshot_key", None)
        cached = getattr(self, "_workflow_run_snapshot", None)
        if cached_key == cache_key and isinstance(cached, dict):
            return cached

        response = requests.post(f"{base_url}/runs", json=payload, timeout=30)
        response.raise_for_status()
        snapshot = response.json()
        self._workflow_run_snapshot = snapshot
        self._workflow_run_snapshot_key = cache_key
        self.status = format_runner_message(snapshot)
        return snapshot

    def run_workflow(self) -> Data:
        snapshot = self._run_workflow_once()
        return Data(data=snapshot)

    def run_workflow_message(self) -> Message:
        snapshot = self._run_workflow_once()
        return Message(text=format_runner_message(snapshot), data=snapshot)
