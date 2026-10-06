"""Runtime event logging for live workflow visibility.

This module is intentionally UI-agnostic. The configured runner writes a small
NDJSON event stream plus compact status snapshots; CLI, dashboard, notebooks,
and static reports can all consume those files without coupling themselves to
LangGraph internals.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:  # Rich is a core dependency, but keep import-time behavior defensive.
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - exercised only in broken installations.
    Console = None
    Live = None
    Panel = None
    Table = None


RunEventType = Literal[
    "run_started",
    "node_started",
    "node_completed",
    "node_failed",
    "custom_progress",
    "artifact_written",
    "run_completed",
]

SECRET_VALUE_PREFIXES = ("sk-", "sk_ant", "sk-ant", "tvly-", "tavily-")
SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "authorization",
    "bearer",
)
LARGE_TEXT_KEYS = {
    "clean_text",
    "raw_text",
    "full_text",
    "document_text",
    "html",
    "markdown",
    "page_content",
    "content",
}
DEFAULT_MAX_STRING_LENGTH = 500
DEFAULT_MAX_LIST_ITEMS = 20
DEFAULT_MAX_DICT_ITEMS = 80


@dataclass
class WorkflowRunEvent:
    """One append-only runtime event written to ``run_events.ndjson``."""

    event_id: str
    session_id: str
    sequence: int
    timestamp_utc: str
    event_type: RunEventType
    status: str | None = None
    node_name: str | None = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    duration_ms: int | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "event_type": self.event_type,
            "status": self.status,
            "node_name": self.node_name,
            "message": self.message,
            "payload": self.payload,
            "duration_ms": self.duration_ms,
            "artifact_paths": self.artifact_paths,
            "error": self.error,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in SECRET_KEY_FRAGMENTS)


def _looks_secret_value(value: str) -> bool:
    lowered = value.strip().lower()
    return any(lowered.startswith(prefix) for prefix in SECRET_VALUE_PREFIXES)


def _truncate_string(value: str, max_string_length: int) -> str:
    if len(value) <= max_string_length:
        return value
    return value[:max_string_length].rstrip() + "...[truncated]"


def sanitize_event_payload(
    value: Any,
    *,
    max_string_length: int = DEFAULT_MAX_STRING_LENGTH,
    max_list_items: int = DEFAULT_MAX_LIST_ITEMS,
    max_dict_items: int = DEFAULT_MAX_DICT_ITEMS,
    _key: str | None = None,
) -> Any:
    """Return a compact, JSON-safe, secret-redacted copy of an event payload."""

    if _key and _looks_secret_key(_key):
        return "***REDACTED***"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"error_type": value.__class__.__name__, "message": str(value)}
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            value = str(value)
    if isinstance(value, str):
        if _looks_secret_value(value):
            return "***REDACTED***"
        limit = 160 if (_key or "").lower() in LARGE_TEXT_KEYS else max_string_length
        return _truncate_string(value, limit)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        items = [
            sanitize_event_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            items.append({"truncated_items": len(value) - max_list_items})
        return items
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_dict_items:
                sanitized["truncated_keys"] = len(value) - max_dict_items
                break
            key_text = str(key)
            sanitized[key_text] = sanitize_event_payload(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                _key=key_text,
            )
        return sanitized
    return _truncate_string(str(value), max_string_length)


def summarize_state_update(update: Any) -> dict[str, Any]:
    """Produce a small status payload from a LangGraph node update."""

    if not isinstance(update, dict):
        return {"value": sanitize_event_payload(update)}
    summary: dict[str, Any] = {}
    for key, value in update.items():
        if key == "collection_trace":
            summary["collection_trace_count"] = len(value or []) if isinstance(value, list) else 0
            continue
        if key.endswith("_summary") and isinstance(value, dict):
            summary[key] = sanitize_event_payload(value)
            continue
        if isinstance(value, list):
            summary[f"{key}_count"] = len(value)
            continue
        if isinstance(value, dict):
            summary[f"{key}_keys"] = sorted(str(item) for item in value.keys())[:20]
            continue
        summary[key] = sanitize_event_payload(value)
    return summary


class RichRunStatusRenderer:
    """Small terminal renderer that degrades to no-op outside interactive TTYs."""

    def __init__(self, node_order: list[str] | None = None, *, enabled: bool = True):
        self.enabled = bool(enabled and sys.stderr.isatty() and Console and Live)
        self.node_order = list(node_order or [])
        self.node_status: dict[str, str] = {
            node: "pending" for node in self.node_order
        }
        self.current_node: str | None = None
        self.recent_events: deque[str] = deque(maxlen=8)
        self.console = Console(stderr=True) if self.enabled else None
        self.live = None

    def __enter__(self):
        if self.enabled:
            self.live = Live(self._render(), console=self.console, refresh_per_second=4)
            self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.live is not None:
            self.live.__exit__(exc_type, exc, tb)
        self.live = None
        return False

    def update(self, event: WorkflowRunEvent) -> None:
        if not self.enabled:
            return
        if event.node_name:
            if event.node_name not in self.node_status:
                self.node_status[event.node_name] = event.status or "running"
                self.node_order.append(event.node_name)
            if event.event_type == "node_started":
                self.node_status[event.node_name] = "running"
                self.current_node = event.node_name
            elif event.event_type == "node_completed":
                self.node_status[event.node_name] = "completed"
            elif event.event_type == "node_failed":
                self.node_status[event.node_name] = "failed"
        if event.event_type == "run_completed":
            self.current_node = None
        self.recent_events.append(
            f"{event.sequence:04d} {event.event_type}"
            + (f" [{event.node_name}]" if event.node_name else "")
            + (f": {event.message}" if event.message else "")
        )
        if self.live is not None:
            self.live.update(self._render())

    def _render(self):
        table = Table(title="Data Collection Workflow live status")
        table.add_column("Node")
        table.add_column("Status")
        for node in self.node_order:
            table.add_row(node, self.node_status.get(node, "pending"))
        recent = "\n".join(self.recent_events) or "waiting for events..."
        return Panel.fit(
            table,
            title=f"Current node: {self.current_node or 'none'}",
            subtitle=recent,
        )


class RunEventWriter:
    """Append run events and keep compact status snapshots up to date."""

    def __init__(
        self,
        *,
        session_dir: str | Path,
        session_id: str,
        node_order: list[str] | None = None,
        live_status: bool = False,
        resume: bool = False,
    ):
        self.session_dir = Path(session_dir)
        self.session_id = session_id
        self.diagnostics_dir = self.session_dir / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.diagnostics_dir / "run_events.ndjson"
        self.run_status_path = self.diagnostics_dir / "run_status.json"
        self.node_status_path = self.diagnostics_dir / "node_status.json"
        self.sequence = 0
        self.node_order = list(node_order or [])
        self.node_status: dict[str, dict[str, Any]] = {
            node: {"node_name": node, "status": "pending"} for node in self.node_order
        }
        self.run_status: dict[str, Any] = {
            "session_id": session_id,
            "status": "created",
            "current_node": None,
            "event_count": 0,
            "artifact_paths": {},
        }
        self._node_started_at: dict[str, float] = {}
        self._run_started_at: float | None = None
        self.renderer = RichRunStatusRenderer(self.node_order, enabled=live_status)
        if resume and self.events_path.exists():
            previous_events = [json.loads(line) for line in self.events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
            if any(event.get('session_id') != session_id for event in previous_events):
                raise ValueError('event log belongs to another session')
            self.sequence = max((int(event.get('sequence') or 0) for event in previous_events), default=0)
            if self.run_status_path.exists():
                self.run_status.update(json.loads(self.run_status_path.read_text(encoding='utf-8')))
            if self.node_status_path.exists():
                previous_nodes = json.loads(self.node_status_path.read_text(encoding='utf-8'))
                if isinstance(previous_nodes, dict):
                    self.node_status.update(previous_nodes.get('nodes') or previous_nodes)
        else:
            self.events_path.write_text('', encoding='utf-8')
        self._write_status_files()

    @property
    def artifact_paths(self) -> dict[str, str]:
        return {
            "run_events_ndjson": str(self.events_path),
            "run_status_json": str(self.run_status_path),
            "node_status_json": str(self.node_status_path),
        }

    def __enter__(self):
        self.renderer.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.renderer.__exit__(exc_type, exc, tb)
        return False

    def append_event(
        self,
        event_type: RunEventType,
        *,
        status: str | None = None,
        node_name: str | None = None,
        message: str = "",
        payload: Any | None = None,
        duration_ms: int | None = None,
        artifact_paths: dict[str, str | Path] | None = None,
        error: Any | None = None,
    ) -> WorkflowRunEvent:
        self.sequence += 1
        clean_artifacts = {
            str(key): str(value) for key, value in (artifact_paths or {}).items()
        }
        clean_error = sanitize_event_payload(error) if error is not None else None
        if isinstance(clean_error, str):
            clean_error = {"message": clean_error}
        event = WorkflowRunEvent(
            event_id=f"{self.session_id}-{self.sequence:06d}",
            session_id=self.session_id,
            sequence=self.sequence,
            timestamp_utc=_utc_now(),
            event_type=event_type,
            status=status,
            node_name=node_name,
            message=message,
            payload=sanitize_event_payload(payload or {}),
            duration_ms=duration_ms,
            artifact_paths=clean_artifacts,
            error=clean_error,
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._apply_event(event)
        self._write_status_files()
        self.renderer.update(event)
        return event

    def append_run_started(self, payload: Any | None = None) -> WorkflowRunEvent:
        self._run_started_at = time.perf_counter()
        return self.append_event(
            "run_started",
            status="running",
            message="Workflow run started.",
            payload=payload or {},
        )

    def append_node_started(
        self,
        node_name: str,
        payload: Any | None = None,
    ) -> WorkflowRunEvent:
        self._node_started_at[node_name] = time.perf_counter()
        return self.append_event(
            "node_started",
            status="running",
            node_name=node_name,
            message=f"{node_name} started.",
            payload=payload or {},
        )

    def append_node_completed(
        self,
        node_name: str,
        payload: Any | None = None,
        *,
        duration_ms: int | None = None,
    ) -> WorkflowRunEvent:
        if duration_ms is None:
            duration_ms = self._elapsed_ms(self._node_started_at.get(node_name))
        return self.append_event(
            "node_completed",
            status="completed",
            node_name=node_name,
            message=f"{node_name} completed.",
            payload=payload or {},
            duration_ms=duration_ms,
        )

    def append_node_failed(
        self,
        node_name: str | None,
        error: Any,
        payload: Any | None = None,
    ) -> WorkflowRunEvent:
        duration_ms = self._elapsed_ms(self._node_started_at.get(node_name or ""))
        return self.append_event(
            "node_failed",
            status="failed",
            node_name=node_name,
            message=f"{node_name or 'workflow node'} failed.",
            payload=payload or {},
            duration_ms=duration_ms,
            error=error,
        )

    def append_custom_progress(
        self,
        *,
        node_name: str | None,
        message: str,
        payload: Any | None = None,
        status: str = "running",
    ) -> WorkflowRunEvent:
        return self.append_event(
            "custom_progress",
            status=status,
            node_name=node_name,
            message=message,
            payload=payload or {},
        )

    def append_artifact_written(
        self,
        label: str,
        path: str | Path,
    ) -> WorkflowRunEvent:
        return self.append_event(
            "artifact_written",
            status=self.run_status.get("status"),
            message=f"Artifact written: {label}.",
            artifact_paths={label: path},
        )

    def append_run_completed(
        self,
        payload: Any | None = None,
        *,
        duration_ms: int | None = None,
    ) -> WorkflowRunEvent:
        if duration_ms is None:
            duration_ms = self._elapsed_ms(self._run_started_at)
        return self.append_event(
            "run_completed",
            status="completed",
            message="Workflow run completed.",
            payload=payload or {},
            duration_ms=duration_ms,
        )

    def mark_run_failed(self, error: Any, *, node_name: str | None = None) -> None:
        self.run_status["status"] = "failed"
        self.run_status["current_node"] = node_name
        self.run_status["completed_at_utc"] = _utc_now()
        self.run_status["error"] = sanitize_event_payload(error)
        self._write_status_files()

    def _apply_event(self, event: WorkflowRunEvent) -> None:
        self.run_status["event_count"] = self.sequence
        self.run_status["updated_at_utc"] = event.timestamp_utc
        if event.event_type == "run_started":
            self.run_status["status"] = "running"
            self.run_status["started_at_utc"] = event.timestamp_utc
            if isinstance(event.payload, dict):
                for key in ("trace_id", "langsmith_project", "langgraph_graph"):
                    if event.payload.get(key):
                        self.run_status[key] = event.payload[key]
        elif event.event_type == "node_started" and event.node_name:
            self.run_status["status"] = "running"
            self.run_status["current_node"] = event.node_name
            node = self._node(event.node_name)
            node.update(
                {
                    "status": "running",
                    "started_at_utc": event.timestamp_utc,
                    "last_message": event.message,
                    "last_payload": event.payload,
                }
            )
        elif event.event_type == "custom_progress" and event.node_name:
            node = self._node(event.node_name)
            node.update(
                {
                    "status": event.status or node.get("status") or "running",
                    "last_message": event.message,
                    "last_payload": event.payload,
                    "updated_at_utc": event.timestamp_utc,
                }
            )
        elif event.event_type == "node_completed" and event.node_name:
            node = self._node(event.node_name)
            node.update(
                {
                    "status": "completed",
                    "completed_at_utc": event.timestamp_utc,
                    "duration_ms": event.duration_ms,
                    "last_message": event.message,
                    "last_payload": event.payload,
                }
            )
        elif event.event_type == "node_failed":
            self.run_status["status"] = "failed"
            self.run_status["current_node"] = event.node_name
            self.run_status["error"] = event.error
            if event.node_name:
                node = self._node(event.node_name)
                node.update(
                    {
                        "status": "failed",
                        "completed_at_utc": event.timestamp_utc,
                        "duration_ms": event.duration_ms,
                        "last_message": event.message,
                        "last_payload": event.payload,
                        "error": event.error,
                    }
                )
        elif event.event_type == "artifact_written":
            self.run_status.setdefault("artifact_paths", {}).update(event.artifact_paths)
        elif event.event_type == "run_completed":
            self.run_status["status"] = "completed"
            self.run_status["current_node"] = None
            self.run_status["completed_at_utc"] = event.timestamp_utc
            self.run_status["duration_ms"] = event.duration_ms
            if event.payload:
                self.run_status["summary"] = event.payload

    def _node(self, node_name: str) -> dict[str, Any]:
        if node_name not in self.node_status:
            self.node_status[node_name] = {"node_name": node_name, "status": "pending"}
            self.node_order.append(node_name)
        return self.node_status[node_name]

    def _write_status_files(self) -> None:
        self.run_status_path.write_text(
            json.dumps(sanitize_event_payload(self.run_status), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.node_status_path.write_text(
            json.dumps(sanitize_event_payload(self.node_status), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _elapsed_ms(started_at: float | None) -> int | None:
        if started_at is None:
            return None
        return int((time.perf_counter() - started_at) * 1000)


def emit_workflow_progress(
    node_name: str,
    message: str,
    payload: dict[str, Any] | None = None,
    *,
    status: str = "running",
) -> None:
    """Emit LangGraph custom stream data when a node runs under streaming."""

    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    try:
        writer(
            {
                "node_name": node_name,
                "message": message,
                "payload": sanitize_event_payload(payload or {}),
                "status": status,
            }
        )
    except Exception:
        return
