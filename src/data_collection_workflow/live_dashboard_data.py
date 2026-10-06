"""Read-only data helpers for the live workflow dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_run_events(session_dir: str | Path) -> list[dict]:
    """Load valid NDJSON events, ignoring a trailing partial line during live writes."""

    path = Path(session_dir) / "diagnostics" / "run_events.ndjson"
    if not path.exists():
        return []
    events: list[dict] = []
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


def load_dashboard_snapshot(session_dir: str | Path) -> dict:
    """Return one compact snapshot consumed by Streamlit and tests."""

    root = Path(session_dir)
    diagnostics = root / "diagnostics"
    summary = _read_json(root / "workflow_run_summary.json", {})
    run_status = _read_json(diagnostics / "run_status.json", {})
    node_status = _read_json(diagnostics / "node_status.json", {})
    events = load_run_events(root)
    artifact_paths = {}
    if isinstance(summary, dict):
        artifact_paths.update(summary.get("artifact_paths") or {})
        for key in (
            "run_events_ndjson",
            "run_status_json",
            "node_status_json",
            "workflow_replay_notebook",
        ):
            if summary.get(key):
                artifact_paths[key] = summary[key]
    if isinstance(run_status, dict):
        artifact_paths.update(run_status.get("artifact_paths") or {})
    return {
        "session_dir": str(root),
        "events": events,
        "recent_events": events[-50:],
        "run_status": run_status if isinstance(run_status, dict) else {},
        "node_status": node_status if isinstance(node_status, dict) else {},
        "summary": summary if isinstance(summary, dict) else {},
        "artifact_paths": artifact_paths,
    }
