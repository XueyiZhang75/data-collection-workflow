"""Local Langflow demo adapter for the data collection workflow.

This module is intentionally thin: it starts the existing workflow runner in the
background and exposes read-only links to artifacts written by the normal run.
"""

from data_collection_workflow.environment import get_env

import argparse
import json
import re
import sys
import threading
import time
import zipfile
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .runtime_profile import load_project_env, resolve_llm_selection


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISUAL_CONTRACT_VERSION = "workflow-ui/2"

SUMMARY_ARTIFACT_KEYS = {
    "final_report_english",
    "report_bundle",
    "workflow_console_html",
    "workflow_console_summary_json",
    "workflow_visualization_index",
    "workflow_visualization_summary",
    "workflow_replay_notebook",
}

STANDARD_ARTIFACT_RELATIVE_PATHS = {
    "final_package_json": Path("collection") / "final_package.json",
    "final_dataset_csv": Path("collection") / "final_dataset.csv",
    "final_dataset_json": Path("collection") / "final_dataset.json",
    "final_dataset_post_review_json": Path("collection") / "final_dataset_post_review.json",
    "source_registry_json": Path("collection") / "source_registry.json",
}

ALLOWED_ARTIFACT_KEYS = SUMMARY_ARTIFACT_KEYS | set(STANDARD_ARTIFACT_RELATIVE_PATHS)

DEFAULT_LANGSMITH_PROJECT = "data-collection-workflow-demo"
DEFAULT_STUDIO_URL = "http://127.0.0.1:2024"


class ArtifactAccessError(ValueError):
    """Raised when a requested artifact key or path is not safe to serve."""


class RunStartError(RuntimeError):
    """Raised when the workflow cannot be started for a user-correctable reason."""


class LangflowDemoRunRequest(BaseModel):
    disease: str = Field(min_length=1)
    location: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str = Field(min_length=1)
    user_request: str | None = None
    session_id: str | None = None
    provider: str = ""
    model: str = ""
    no_llm: bool = False
    quick_test_mode: bool = False
    audit_mode: bool = False
    dashboard_enabled: bool = False
    dashboard_port: int = 8501
    visual_contract_version: str | None = None

    def with_resolved_session_id(self) -> "LangflowDemoRunRequest":
        start_date, end_date = normalize_date_range(self.start_date, self.end_date)
        session_id = normalize_session_id(
            self.session_id or generated_session_id(
                self.disease,
                self.location,
                start_date,
                end_date,
            )
        )
        return self.model_copy(
            update={
                "session_id": session_id,
                "start_date": start_date,
                "end_date": end_date,
            }
        )


class RunRegistry:
    """In-memory run registry for the local demo API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, dict[str, Any]] = {}

    def register(
        self,
        *,
        session_id: str,
        session_dir: str | Path,
        dashboard_url: str | None = None,
        generated_config_path: str | Path | None = None,
        dashboard: dict[str, Any] | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        record = {
            "session_id": session_id,
            "status": "queued",
            "session_dir": str(Path(session_dir)),
            "live_dashboard_url": dashboard_url,
            "dashboard": dashboard or {},
            "generated_config_path": str(generated_config_path) if generated_config_path else None,
            "request_metadata": request_metadata or {},
            "summary": {},
            "error": None,
            "created_at_utc": now,
            "updated_at_utc": now,
        }
        with self._lock:
            self._runs[session_id] = record
            return deepcopy(record)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._runs

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id not in self._runs:
                raise KeyError(session_id)
            return deepcopy(self._runs[session_id])

    def mark_running(self, session_id: str) -> None:
        self._update(session_id, status="running", error=None)

    def mark_completed(self, session_id: str, summary: dict[str, Any]) -> None:
        self._update(session_id, status="completed", summary=summary, error=None)

    def mark_failed(self, session_id: str, error: str) -> None:
        self._update(session_id, status="failed", error=error)

    def _update(self, session_id: str, **updates: Any) -> None:
        with self._lock:
            if session_id not in self._runs:
                raise KeyError(session_id)
            self._runs[session_id].update(updates)
            self._runs[session_id]["updated_at_utc"] = _utc_now()


def require_visual_contract_version(version: str | None) -> None:
    if version != VISUAL_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported or missing Langflow visual contract version. "
            f"Use visual_contract_version={VISUAL_CONTRACT_VERSION}."
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "workflow_run"


def generated_session_id(disease: str, location: str, start_date: str, end_date: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return f"{_slug(disease)}_{_slug(location)}_{_slug(start_date)}_{_slug(end_date)}_{stamp}"


def normalize_date_value(value: str, *, is_end: bool = False) -> str:
    """Normalize user-facing year/day input to an ISO date string."""

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Date value is required.")
    year_match = re.fullmatch(r"(\d{4})", raw)
    if year_match:
        year = int(year_match.group(1))
        return date(year, 12 if is_end else 1, 31 if is_end else 1).isoformat()
    date_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if not date_match:
        raise ValueError(
            f"Unsupported date value: {raw}. Use YYYY, YYYY-M-D, or YYYY-MM-DD."
        )
    year, month, day = (int(part) for part in date_match.groups())
    return date(year, month, day).isoformat()


def normalize_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    start = normalize_date_value(start_date, is_end=False)
    end = normalize_date_value(end_date, is_end=True)
    if start > end:
        raise ValueError(f"Start date {start} must be on or before end date {end}.")
    return start, end


def normalize_session_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return cleaned or "workflow_run"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _summary_for_session(session_dir: Path) -> dict[str, Any]:
    summary = _read_json(session_dir / "workflow_run_summary.json", {})
    return summary if isinstance(summary, dict) else {}


def _run_status_for_session(session_dir: Path) -> dict[str, Any]:
    status = _read_json(session_dir / "diagnostics" / "run_status.json", {})
    return status if isinstance(status, dict) else {}


def _node_status_for_session(session_dir: Path) -> dict[str, Any]:
    status = _read_json(session_dir / "diagnostics" / "node_status.json", {})
    return status if isinstance(status, dict) else {}


def _run_events_for_session(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "diagnostics" / "run_events.ndjson"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def workflow_node_order() -> list[str]:
    try:
        configured = _scripts_module("run_workflow")
        order = getattr(configured, "WORKFLOW_NODE_ORDER", None)
        if isinstance(order, list) and order:
            return [str(node) for node in order]
    except Exception:
        pass
    return [
        "task_intake_and_scope_planning",
        "disease_intelligence_builder",
        "profile_and_schema_setup",
        "executable_source_planning",
        "query_strategy_builder",
        "source_discovery",
        "source_dedup_and_registry",
        "source_screening",
        "source_critic_and_uncertainty_routing",
        "content_fetch_and_parse",
        "document_quality_check",
        "evidence_chunking_and_data_presence_flagging",
        "structured_extraction",
        "schema_validation_and_repair",
        "record_normalization",
        "record_linking",
        "cross_source_consistency_check",
        "quality_gate_routing",
        "human_review",
        "final_data_package_builder",
    ]


def _studio_url() -> str:
    return (get_env("LANGGRAPH_STUDIO_URL") or DEFAULT_STUDIO_URL).rstrip("/")


def _langsmith_project(run_status: dict[str, Any] | None = None) -> str:
    if isinstance(run_status, dict) and run_status.get("langsmith_project"):
        return str(run_status["langsmith_project"])
    return (get_env("LANGSMITH_PROJECT") or DEFAULT_LANGSMITH_PROJECT).strip()


def _langsmith_project_url(project_name: str) -> str | None:
    if not project_name:
        return None
    base = (
        get_env("LANGSMITH_APP_BASE_URL")
        or "https://smith.langchain.com"
    ).rstrip("/")
    return f"{base}/o/projects/p/{project_name}"


def resolve_langsmith_trace_url(
    *,
    session_id: str,
    trace_id: str | None,
    project_name: str | None,
    client_factory: Callable[[], Any] | None = None,
) -> str | None:
    """Resolve a LangSmith trace URL for a session without failing the demo API."""

    project = project_name or DEFAULT_LANGSMITH_PROJECT
    if not trace_id:
        return None
    try:
        if client_factory is None:
            from langsmith import Client

            client_factory = Client
        client = client_factory()
        runs = client.list_runs(
            project_name=project,
            trace_id=trace_id,
            is_root=True,
            limit=1,
        )
        run = next(iter(runs), None)
        if run is None:
            return None
        return client.get_run_url(run=run, project_name=project)
    except Exception:
        return None


def _candidate_path_from_summary(session_dir: Path, summary: dict[str, Any], key: str) -> Path | None:
    artifact_paths = summary.get("artifact_paths") or {}
    if not isinstance(artifact_paths, dict):
        return None
    raw = artifact_paths.get(key)
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _candidate_path_for_key(session_dir: Path, summary: dict[str, Any], key: str) -> Path | None:
    if key in STANDARD_ARTIFACT_RELATIVE_PATHS:
        path = session_dir / STANDARD_ARTIFACT_RELATIVE_PATHS[key]
        return path if path.exists() else None
    if key in SUMMARY_ARTIFACT_KEYS:
        return _candidate_path_from_summary(session_dir, summary, key)
    return None


def resolve_artifact_path(session_dir: str | Path, artifact_key: str) -> Path:
    """Resolve an allowed artifact key to a file inside the session directory."""

    root = Path(session_dir)
    if artifact_key not in ALLOWED_ARTIFACT_KEYS:
        raise ArtifactAccessError(f"Artifact key is not allowed: {artifact_key}")
    summary = _summary_for_session(root)
    path = _candidate_path_for_key(root, summary, artifact_key)
    if path is None or not path.exists() or not path.is_file():
        raise ArtifactAccessError(f"Artifact is not available: {artifact_key}")
    if not _is_relative_to(path, root):
        raise ArtifactAccessError(f"Artifact path escapes the session directory: {artifact_key}")
    return path.resolve()


def _allowed_artifact_paths(session_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in sorted(ALLOWED_ARTIFACT_KEYS):
        path = _candidate_path_for_key(session_dir, summary, key)
        if path and path.exists() and path.is_file() and _is_relative_to(path, session_dir):
            paths[key] = str(path.resolve())
    return paths


def _artifact_urls(
    *,
    session_id: str,
    api_base_url: str,
    artifact_paths: dict[str, str],
) -> dict[str, str]:
    base = api_base_url.rstrip("/")
    return {
        key: f"{base}/runs/{session_id}/artifacts/{key}"
        for key in sorted(artifact_paths)
    }


def _display_path(path: str | Path | None) -> str | None:
    if not path:
        return None
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return str(resolved).replace("\\", "/")


def format_result_location_text(snapshot: dict[str, Any]) -> str:
    artifact_urls = snapshot.get("artifact_urls")
    artifact_urls = artifact_urls if isinstance(artifact_urls, dict) else {}
    lines = [
        f"Session ID: {snapshot.get('session_id') or 'pending'}",
        f"Status: {snapshot.get('status') or 'pending'}",
        f"Session directory: {_display_path(snapshot.get('session_dir')) or 'pending'}",
        f"Generated config: {_display_path(snapshot.get('generated_config_path')) or 'pending'}",
        f"Status API: {snapshot.get('status_url') or 'pending'}",
    ]
    if snapshot.get("normalized_start_date") or snapshot.get("normalized_end_date"):
        lines.append(
            "Date range: "
            f"{snapshot.get('normalized_start_date') or 'pending'} to "
            f"{snapshot.get('normalized_end_date') or 'pending'}"
        )
    lines.append(f"Quick test mode: {bool(snapshot.get('quick_test_mode'))}")
    if snapshot.get("error"):
        lines.append(f"Error: {snapshot['error']}")

    unavailable = "not generated" if snapshot.get("status") in TERMINAL_STATUSES else "pending"
    for label, key in (
        ("Final report", "final_report_english"),
        ("Report bundle", "report_bundle"),
        ("Workflow console", "workflow_console_html"),
        ("Workflow visualization", "workflow_visualization_index"),
        ("Final dataset CSV", "final_dataset_csv"),
        ("Final dataset JSON", "final_dataset_json"),
        ("Final package JSON", "final_package_json"),
    ):
        lines.append(f"{label}: {artifact_urls.get(key) or unavailable}")
    return "\n".join(lines)


def _extract_llm_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            lowered = str(key).lower()
            if lowered in {"provider", "model"} and value:
                summary[lowered] = value
            elif lowered in {"llm_enabled", "enabled"}:
                summary["enabled"] = bool(value)
            elif lowered in {"llm_call_count", "call_count"}:
                summary["call_count"] = value
            elif lowered in {"llm_success_count", "success_count"}:
                summary["success_count"] = value
            elif lowered in {"llm_error_count", "error_count", "failure_count"}:
                summary["error_count"] = value
            elif lowered.startswith("llm_"):
                summary[lowered] = value
    if "enabled" not in summary and summary:
        summary["enabled"] = True
    return summary


def _extract_tool_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    providers: set[str] = set()
    counts: dict[str, Any] = {}
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        for key in ("provider", "search_provider", "fetch_provider"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                providers.add(value.strip())
        for key in (
            "result_count",
            "executed_query_count",
            "raw_search_result_count",
            "candidate_from_search_count",
            "fetch_request_count",
            "document_count",
        ):
            if key in payload:
                counts[key] = payload[key]
        if isinstance(payload.get("fetch_provider_counts"), dict):
            counts["fetch_provider_counts"] = payload["fetch_provider_counts"]
            providers.update(str(name) for name in payload["fetch_provider_counts"])
    summary: dict[str, Any] = {}
    if providers:
        summary["providers"] = sorted(providers)
    summary.update(counts)
    return summary


MAX_SNAPSHOT_DEPTH = 4
MAX_SNAPSHOT_DICT_ITEMS = 30
MAX_SNAPSHOT_LIST_ITEMS = 20
MAX_SNAPSHOT_STRING_CHARS = 700


def _compact_snapshot_value(value: Any, *, depth: int = 0) -> Any:
    """Keep Langflow node snapshots readable and cheap to serialize."""

    if depth >= MAX_SNAPSHOT_DEPTH:
        if isinstance(value, dict):
            return {"_truncated_depth": True, "type": "dict", "key_count": len(value)}
        if isinstance(value, list):
            return {"_truncated_depth": True, "type": "list", "item_count": len(value)}
        return value
    if isinstance(value, str):
        if len(value) <= MAX_SNAPSHOT_STRING_CHARS:
            return value
        omitted = len(value) - MAX_SNAPSHOT_STRING_CHARS
        return value[:MAX_SNAPSHOT_STRING_CHARS].rstrip() + f"\n... [truncated {omitted} chars]"
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:MAX_SNAPSHOT_DICT_ITEMS]:
            compacted[str(key)] = _compact_snapshot_value(item, depth=depth + 1)
        if len(items) > MAX_SNAPSHOT_DICT_ITEMS:
            compacted["_truncated_keys"] = len(items) - MAX_SNAPSHOT_DICT_ITEMS
        return compacted
    if isinstance(value, list):
        compacted_list = [
            _compact_snapshot_value(item, depth=depth + 1)
            for item in value[:MAX_SNAPSHOT_LIST_ITEMS]
        ]
        if len(value) > MAX_SNAPSHOT_LIST_ITEMS:
            compacted_list.append({"_truncated_items": len(value) - MAX_SNAPSHOT_LIST_ITEMS})
        return compacted_list
    return value


def apply_quick_test_mode(config: dict[str, Any]) -> dict[str, Any]:
    """Apply a small live-run budget for short Langflow smoke tests."""

    source_search = config.get("source_search")
    if isinstance(source_search, dict):
        source_search["max_queries"] = 2
        source_search["max_results_per_query"] = 3
        source_search["max_total_results"] = 6
        iterative = source_search.get("iterative")
        if isinstance(iterative, dict):
            iterative["max_iterations"] = 1
            iterative["max_queries_per_iteration"] = 2
            iterative["max_total_queries"] = 3
            iterative["max_total_results"] = 6
            iterative["stop_when_llm_says_sufficient"] = True

    content_fetch = config.get("content_fetch")
    if isinstance(content_fetch, dict):
        content_fetch["max_search_derived_sources"] = 5
        content_fetch["max_total_sources"] = 5
        external_fetch = content_fetch.get("external_fetch")
        if isinstance(external_fetch, dict):
            tavily_extract = external_fetch.get("tavily_extract")
            if isinstance(tavily_extract, dict):
                tavily_extract["chunks_per_source"] = 2
            adaptive_budget = external_fetch.get("adaptive_budget")
            if isinstance(adaptive_budget, dict):
                adaptive_budget["max_candidate_urls"] = 12
                adaptive_budget["max_fetch_urls"] = 5
                adaptive_budget["min_usable_documents"] = 3
                adaptive_budget["min_collection_sources"] = 2
                adaptive_budget["min_validation_sources"] = 1
                adaptive_budget["max_iterations"] = 1

    llm = config.get("llm")
    if isinstance(llm, dict):
        llm["max_chunks"] = 5
        llm["must_fetch_min_chunks_per_source"] = 2
        llm["official_extraction_max_chunks"] = 8
        for section_name, max_sources in (
            ("source_critic", 5),
            ("source_identity", 5),
            ("source_credibility", 3),
        ):
            section = llm.get(section_name)
            if isinstance(section, dict):
                section["max_sources"] = max_sources
    config["quick_test_mode"] = True
    return config


def _status_text_for_node(snapshot: dict[str, Any]) -> str:
    lines = [
        f"Node: {snapshot.get('node_name')}",
        f"Status: {snapshot.get('status') or 'pending'}",
        f"Duration ms: {snapshot.get('duration_ms') or 'pending'}",
    ]
    if snapshot.get("trace_url"):
        lines.append(f"LangSmith trace: {snapshot['trace_url']}")
    if snapshot.get("error"):
        lines.append(f"Error: {snapshot['error']}")
    return "\n".join(lines)


def _compact_json_text(value: Any, *, max_chars: int = 1800) -> str:
    if value in (None, {}, []):
        return "pending"
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 32].rstrip() + "\n... [truncated]"


def _markdown_code(value: Any, *, max_chars: int = 1800) -> str:
    return f"```json\n{_compact_json_text(value, max_chars=max_chars)}\n```"


def _recent_event_lines(events: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for event in events[-5:]:
        event_type = event.get("event_type") or "event"
        status = event.get("status") or "unknown"
        message = event.get("message") or ""
        duration = event.get("duration_ms")
        suffix = f", {duration} ms" if duration else ""
        lines.append(f"- `{event_type}` / `{status}`{suffix}: {message}")
    return lines or ["- pending"]


def format_node_detail_markdown(snapshot: dict[str, Any]) -> str:
    duration = snapshot.get("duration_ms")
    duration_text = f"{duration} ms" if duration is not None else "pending"
    lines = [
        f"## {snapshot.get('node_name') or 'pending'}",
        "",
        f"- Status: `{snapshot.get('status') or 'pending'}`",
        f"- Duration: `{duration_text}`",
        f"- Started: `{snapshot.get('started_at') or 'pending'}`",
        f"- Completed: `{snapshot.get('completed_at') or 'pending'}`",
        f"- Last message: {snapshot.get('last_message') or 'pending'}",
        f"- Node detail API: {snapshot.get('node_status_url') or 'pending'}",
    ]
    if snapshot.get("trace_url"):
        lines.append(f"- LangSmith trace: {snapshot['trace_url']}")
    if snapshot.get("error"):
        lines.append(f"- Error: `{snapshot['error']}`")

    lines.extend(
        [
            "",
            "### Input summary",
            _markdown_code(snapshot.get("input_summary")),
            "",
            "### Output summary",
            _markdown_code(snapshot.get("output_summary")),
            "",
            "### Last payload",
            _markdown_code(snapshot.get("last_payload")),
            "",
            "### Tool summary",
            _markdown_code(snapshot.get("tool_summary")),
            "",
            "### LLM summary",
            _markdown_code(snapshot.get("llm_summary")),
            "",
            "### Recent events",
            *_recent_event_lines(
                snapshot.get("recent_events")
                if isinstance(snapshot.get("recent_events"), list)
                else []
            ),
        ]
    )
    return "\n".join(lines)


def _node_progress_for_session(session_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    node_status = _node_status_for_session(session_dir)
    order = workflow_node_order()
    timeline: list[dict[str, Any]] = []
    counts = {"completed": 0, "running": 0, "failed": 0, "pending": 0}
    running_nodes: list[str] = []
    failed_nodes: list[str] = []
    for node_name in order:
        record = node_status.get(node_name) if isinstance(node_status, dict) else None
        record = record if isinstance(record, dict) else {}
        status = str(record.get("status") or "pending")
        if status not in counts:
            status = "pending"
        counts[status] += 1
        if status == "running":
            running_nodes.append(node_name)
        if status == "failed":
            failed_nodes.append(node_name)
        timeline.append(
            {
                "node_name": node_name,
                "status": status,
                "duration_ms": record.get("duration_ms"),
                "last_message": record.get("last_message"),
                "started_at": record.get("started_at_utc"),
                "completed_at": record.get("completed_at_utc"),
            }
        )
    progress = {
        "total_count": len(order),
        "completed_count": counts["completed"],
        "running_count": counts["running"],
        "pending_count": counts["pending"],
        "failed_count": counts["failed"],
        "running_nodes": running_nodes,
        "failed_nodes": failed_nodes,
    }
    return progress, timeline


def _first_event_payload(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for event in events:
        if event.get("event_type") == event_type and isinstance(event.get("payload"), dict):
            return event["payload"]
    return {}


def _last_event_payload(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("event_type") == event_type and isinstance(event.get("payload"), dict):
            return event["payload"]
    return {}


def build_node_snapshot(
    *,
    session_id: str,
    session_dir: str | Path,
    node_name: str,
    api_base_url: str,
) -> dict[str, Any]:
    root = Path(session_dir)
    node_status = _node_status_for_session(root)
    events = [
        event
        for event in _run_events_for_session(root)
        if event.get("node_name") == node_name
    ]
    node_record = node_status.get(node_name) if isinstance(node_status, dict) else None
    if not isinstance(node_record, dict):
        node_record = {"node_name": node_name, "status": "pending"}
    run_status = _run_status_for_session(root)
    summary = _summary_for_session(root)
    artifact_paths = _allowed_artifact_paths(root, summary)
    artifact_urls = _artifact_urls(
        session_id=session_id,
        api_base_url=api_base_url,
        artifact_paths=artifact_paths,
    )
    payloads = [
        event.get("payload")
        for event in events
        if isinstance(event.get("payload"), dict)
    ]
    if isinstance(node_record.get("last_payload"), dict):
        payloads.append(node_record["last_payload"])
    project = _langsmith_project(run_status)
    trace_id = run_status.get("trace_id")
    trace_url = resolve_langsmith_trace_url(
        session_id=session_id,
        trace_id=str(trace_id) if trace_id else None,
        project_name=project,
    )
    snapshot = {
        "session_id": session_id,
        "node_name": node_name,
        "status": node_record.get("status") or "pending",
        "duration_ms": node_record.get("duration_ms"),
        "started_at": node_record.get("started_at_utc"),
        "completed_at": node_record.get("completed_at_utc"),
        "last_message": node_record.get("last_message"),
        "input_summary": _compact_snapshot_value(_first_event_payload(events, "node_started")),
        "output_summary": _compact_snapshot_value(_last_event_payload(events, "node_completed")),
        "last_payload": _compact_snapshot_value(node_record.get("last_payload") or {}),
        "recent_events": _compact_snapshot_value(events[-10:]),
        "llm_summary": _extract_llm_summary(payloads),
        "tool_summary": _extract_tool_summary(events),
        "artifact_urls": artifact_urls,
        "trace_id": trace_id,
        "trace_url": trace_url,
        "node_status_url": f"{api_base_url.rstrip('/')}/runs/{session_id}/nodes/{node_name}",
        "error": node_record.get("error"),
    }
    snapshot["status_text"] = _status_text_for_node(snapshot)
    snapshot["node_detail_markdown"] = format_node_detail_markdown(snapshot)
    return snapshot


TERMINAL_STATUSES = {"completed", "failed"}


def _wait_until(
    snapshot_builder: Callable[[], dict[str, Any]],
    *,
    status_key: str = "status",
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    poll_interval = max(0.001, float(poll_interval_seconds))
    snapshot = snapshot_builder()
    while snapshot.get(status_key) not in TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(poll_interval)
        snapshot = snapshot_builder()
    if snapshot.get(status_key) not in TERMINAL_STATUSES:
        snapshot["polling_timed_out"] = True
        if "status_text" in snapshot:
            snapshot["status_text"] = _status_text_for_node(snapshot)
    return snapshot


def build_run_snapshot(
    *,
    session_id: str,
    session_dir: str | Path,
    api_base_url: str,
    status: str | None = None,
    dashboard_url: str | None = None,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    generated_config_path: str | Path | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(session_dir)
    loaded_summary = summary if isinstance(summary, dict) and summary else _summary_for_session(root)
    run_status = _run_status_for_session(root)
    resolved_status = status or run_status.get("status") or "unknown"
    node_progress, node_timeline = _node_progress_for_session(root)
    artifact_paths = _allowed_artifact_paths(root, loaded_summary)
    artifact_urls = _artifact_urls(
        session_id=session_id,
        api_base_url=api_base_url,
        artifact_paths=artifact_paths,
    )
    status_url = f"{api_base_url.rstrip().rstrip('/')}/runs/{session_id}"
    project = _langsmith_project(run_status)
    trace_id = run_status.get("trace_id")
    langsmith_trace_url = resolve_langsmith_trace_url(
        session_id=session_id,
        trace_id=str(trace_id) if trace_id else None,
        project_name=project,
    )
    request_metadata = request_metadata if isinstance(request_metadata, dict) else {}
    generated_config = generated_config_path
    if generated_config is None:
        candidate = PROJECT_ROOT / "outputs" / "generated_configs" / f"{session_id}.json"
        generated_config = candidate if candidate.exists() else None
    snapshot = {
        "session_id": session_id,
        "status": resolved_status,
        "current_node": run_status.get("current_node"),
        "session_dir": str(root),
        "generated_config_path": str(generated_config) if generated_config else None,
        "normalized_start_date": request_metadata.get("start_date"),
        "normalized_end_date": request_metadata.get("end_date"),
        "quick_test_mode": bool(request_metadata.get("quick_test_mode")),
        "status_url": status_url,
        "events_url": f"{api_base_url.rstrip().rstrip('/')}/runs/{session_id}/events",
        "studio_url": _studio_url(),
        "langsmith_project": project,
        "langsmith_project_url": _langsmith_project_url(project),
        "trace_id": trace_id,
        "langsmith_trace_url": langsmith_trace_url,
        "live_dashboard_url": dashboard_url,
        "workflow_console_url": artifact_urls.get("workflow_console_html"),
        "workflow_visualization_url": artifact_urls.get("workflow_visualization_index"),
        "node_progress": node_progress,
        "node_timeline": node_timeline,
        "artifact_paths": artifact_paths,
        "artifact_urls": artifact_urls,
        "summary": loaded_summary,
        "error": error,
    }
    snapshot["result_location_text"] = format_result_location_text(snapshot)
    return snapshot


def _scripts_module(name: str) -> Any:
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return __import__(name)


def _build_runner_args(
    config_path: Path,
    *,
    user_request: str | None = None,
    write_run_notebook: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config_path),
        enable_live_fetch=False,
        disable_live_fetch=False,
        enable_all_llm=False,
        disable_all_llm=False,
        provider=None,
        model=None,
        timeout_seconds=None,
        llm_max_chunks=None,
        output_dir=None,
        session_id=None,
        user_request=user_request,
        print_config_only=False,
        live_status=False,
        write_run_notebook=write_run_notebook,
    )


def start_data_collection_workflow_background(
    run_request: LangflowDemoRunRequest,
    registry: RunRegistry,
) -> None:
    """Prepare artifacts and start the existing configured runner in a thread."""

    provider, model = resolve_llm_selection(run_request.provider, run_request.model)
    if not run_request.no_llm:
        if provider not in {"anthropic", "openai"}:
            raise RunStartError("Unsupported LLM provider. Choose anthropic or openai.")
        if not model:
            raise RunStartError(
                "An LLM model ID is required. Enter a model in the visual workflow "
                "or set LLM_MODEL in the project .env file."
            )
    run_request = run_request.model_copy(update={"provider": provider, "model": model})

    interactive = _scripts_module("collect")
    configured = _scripts_module("run_workflow")

    missing = interactive._require_keys(  # noqa: SLF001
        provider=run_request.provider,
        llm_enabled=not run_request.no_llm,
    )
    if missing:
        raise RunStartError(
            "Missing required API key(s): " + ", ".join(str(name) for name in missing)
        )

    config = interactive._real_run_config(  # noqa: SLF001
        disease=run_request.disease,
        location=run_request.location,
        start_date=run_request.start_date,
        end_date=run_request.end_date,
        target_fields=list(interactive.DEFAULT_TARGET_FIELDS),
        session_id=str(run_request.session_id),
        provider=run_request.provider,
        model=run_request.model,
        output_dir=None,
        no_llm=run_request.no_llm,
        user_request=run_request.user_request,
        quick_test_mode=run_request.quick_test_mode,
        audit_mode=run_request.audit_mode,
    )
    if run_request.quick_test_mode:
        apply_quick_test_mode(config)
        config.setdefault("structured_task", {})["quick_test_mode"] = True
    generated_config = interactive._write_generated_config(  # noqa: SLF001
        config,
        str(run_request.session_id),
    )
    session_dir = interactive._session_dir_for_config(config)  # noqa: SLF001
    dashboard = {}
    if run_request.dashboard_enabled:
        dashboard = interactive._launch_live_dashboard(  # noqa: SLF001
            session_dir,
            preferred_port=run_request.dashboard_port,
        )
    dashboard_url = dashboard.get("url") if isinstance(dashboard, dict) else None
    registry.register(
        session_id=str(run_request.session_id),
        session_dir=session_dir,
        dashboard_url=dashboard_url,
        generated_config_path=generated_config,
        dashboard=dashboard if isinstance(dashboard, dict) else {},
        request_metadata=run_request.model_dump(),
    )

    def _worker() -> None:
        session_id = str(run_request.session_id)
        try:
            registry.mark_running(session_id)
            summary = configured.run_workflow(
                _build_runner_args(
                    generated_config,
                    user_request=run_request.user_request,
                    write_run_notebook=False,
                )
            )
            registry.mark_completed(session_id, summary)
        except Exception as exc:  # pragma: no cover - exercised in real demo runs.
            registry.mark_failed(session_id, f"{exc.__class__.__name__}: {exc}")

    thread = threading.Thread(
        target=_worker,
        name=f"workflow-langflow-{run_request.session_id}",
        daemon=True,
    )
    thread.start()


Runner = Callable[[LangflowDemoRunRequest, RunRegistry], None]


def _session_dir_from_record_or_default(record: dict[str, Any], session_id: str) -> Path:
    if record.get("session_dir"):
        return Path(str(record["session_dir"]))
    return PROJECT_ROOT / "outputs" / "sessions" / session_id


def create_app(
    *,
    runner: Runner | None = None,
    registry: RunRegistry | None = None,
    api_base_url: str | None = None,
) -> Any:
    """Create the optional FastAPI app used by the Langflow demo."""

    load_project_env()

    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse, Response
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise RuntimeError(
            "FastAPI is required for the Langflow demo API. "
            "Install with `python -m pip install -e .[langflow-demo]`."
        ) from exc

    app = FastAPI(title="Data Collection Langflow Demo API", version="0.1.0")
    run_registry = registry or RunRegistry()
    run_workflow = runner or start_data_collection_workflow_background

    def _base_url(request: Request) -> str:
        return (api_base_url or str(request.base_url)).rstrip("/")

    def _require_visual_contract(version: str | None) -> None:
        try:
            require_visual_contract_version(version)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "workflow-langflow-api"}

    @app.post("/runs")
    def create_run(payload: LangflowDemoRunRequest, request: Request) -> dict[str, Any]:
        _require_visual_contract(payload.visual_contract_version)
        try:
            run_request = payload.with_resolved_session_id()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        session_id = str(run_request.session_id)
        if run_registry.exists(session_id):
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
            return build_run_snapshot(
                session_id=session_id,
                session_dir=session_dir,
                api_base_url=_base_url(request),
                status=record.get("status"),
                dashboard_url=record.get("live_dashboard_url"),
                summary=record.get("summary"),
                error=record.get("error"),
                generated_config_path=record.get("generated_config_path"),
                request_metadata=record.get("request_metadata"),
            )
        try:
            run_workflow(run_request, run_registry)
            record = run_registry.get(session_id)
        except RunStartError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=500,
                detail="Runner did not register the requested session.",
            ) from exc
        session_dir = _session_dir_from_record_or_default(record, session_id)
        return build_run_snapshot(
            session_id=session_id,
            session_dir=session_dir,
            api_base_url=_base_url(request),
            status=record.get("status"),
            dashboard_url=record.get("live_dashboard_url"),
            summary=record.get("summary"),
            error=record.get("error"),
            generated_config_path=record.get("generated_config_path"),
            request_metadata=record.get("request_metadata"),
        )

    @app.get("/runs/{session_id}")
    def get_run(
        session_id: str,
        request: Request,
        visual_contract_version: str | None = None,
    ) -> dict[str, Any]:
        _require_visual_contract(visual_contract_version)
        try:
            record = run_registry.get(session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
            if not session_dir.exists():
                raise HTTPException(status_code=404, detail=f"Run not found: {session_id}") from None
            record = {
                "status": None,
                "session_dir": str(session_dir),
                "live_dashboard_url": None,
                "summary": {},
                "error": None,
            }
        return build_run_snapshot(
            session_id=session_id,
            session_dir=_session_dir_from_record_or_default(record, session_id),
            api_base_url=_base_url(request),
            status=record.get("status"),
            dashboard_url=record.get("live_dashboard_url"),
            summary=record.get("summary"),
            error=record.get("error"),
            generated_config_path=record.get("generated_config_path"),
            request_metadata=record.get("request_metadata"),
        )

    @app.get("/runs/{session_id}/wait")
    def wait_for_run(
        session_id: str,
        request: Request,
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 2.0,
        visual_contract_version: str | None = None,
    ) -> dict[str, Any]:
        _require_visual_contract(visual_contract_version)

        def _snapshot() -> dict[str, Any]:
            return get_run(
                session_id,
                request,
                visual_contract_version=visual_contract_version,
            )

        return _wait_until(
            _snapshot,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    @app.get("/runs/{session_id}/events")
    def get_run_events(session_id: str) -> dict[str, Any]:
        try:
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
            if not session_dir.exists():
                raise HTTPException(status_code=404, detail=f"Run not found: {session_id}") from None
        return {
            "session_id": session_id,
            "events": _run_events_for_session(session_dir),
        }

    @app.get("/runs/{session_id}/nodes/{node_name}")
    def get_run_node(
        session_id: str,
        node_name: str,
        request: Request,
        visual_contract_version: str | None = None,
    ) -> dict[str, Any]:
        _require_visual_contract(visual_contract_version)
        known_nodes = set(workflow_node_order())
        try:
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
            if not session_dir.exists():
                raise HTTPException(status_code=404, detail=f"Run not found: {session_id}") from None
        node_status = _node_status_for_session(session_dir)
        if node_name not in known_nodes and node_name not in node_status:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_name}")
        return build_node_snapshot(
            session_id=session_id,
            session_dir=session_dir,
            node_name=node_name,
            api_base_url=_base_url(request),
        )

    @app.get("/runs/{session_id}/nodes/{node_name}/wait")
    def wait_for_run_node(
        session_id: str,
        node_name: str,
        request: Request,
        timeout_seconds: float = 1800.0,
        poll_interval_seconds: float = 2.0,
        visual_contract_version: str | None = None,
    ) -> dict[str, Any]:
        _require_visual_contract(visual_contract_version)
        known_nodes = set(workflow_node_order())
        try:
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
            if not session_dir.exists():
                raise HTTPException(status_code=404, detail=f"Run not found: {session_id}") from None
        node_status = _node_status_for_session(session_dir)
        if node_name not in known_nodes and node_name not in node_status:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_name}")

        def _snapshot() -> dict[str, Any]:
            return build_node_snapshot(
                session_id=session_id,
                session_dir=session_dir,
                node_name=node_name,
                api_base_url=_base_url(request),
            )

        return _wait_until(
            _snapshot,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    @app.get("/runs/{session_id}/artifacts/{artifact_key}")
    def get_artifact(session_id: str, artifact_key: str) -> Any:
        try:
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
        try:
            path = resolve_artifact_path(session_dir, artifact_key)
        except ArtifactAccessError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    @app.get("/runs/{session_id}/artifacts/{group}/{filename}")
    def get_report_material(session_id: str, group: str, filename: str) -> Any:
        # Relative links in final_report.html resolve under artifacts/. Read
        # only its portable bundle, never arbitrary session files or folders.
        media_types = {".csv": "text/csv", ".json": "application/json", ".txt": "text/plain"}
        if (group not in {"data", "evidence"}
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename)
                or filename in {".", ".."}
                or Path(filename).suffix not in media_types):
            raise HTTPException(status_code=404, detail="Report material is not available")
        try:
            record = run_registry.get(session_id)
            session_dir = _session_dir_from_record_or_default(record, session_id)
        except KeyError:
            session_dir = PROJECT_ROOT / "outputs" / "sessions" / session_id
        try:
            bundle = resolve_artifact_path(session_dir, "report_bundle")
            with zipfile.ZipFile(bundle) as archive:
                content = archive.read(f"{group}/{filename}")
        except (ArtifactAccessError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise HTTPException(status_code=404, detail="Report material is not available") from exc
        return Response(content=content, media_type=media_types[Path(filename).suffix])

    return app


try:
    app = create_app()
except RuntimeError:
    app = None
