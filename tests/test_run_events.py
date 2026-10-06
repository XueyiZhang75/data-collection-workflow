from __future__ import annotations

import json
from pathlib import Path

from data_collection_workflow.run_events import (
    RunEventWriter,
    WorkflowRunEvent,
    sanitize_event_payload,
)


def _read_ndjson(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_workflow_run_event_serializes_stable_fields():
    event = WorkflowRunEvent(
        event_id="run_001-000001",
        session_id="run_001",
        sequence=1,
        timestamp_utc="2026-06-11T00:00:00+00:00",
        event_type="node_started",
        status="running",
        node_name="source_discovery",
        message="Source discovery started.",
        payload={"planned_query_count": 3},
    )

    payload = event.to_dict()

    assert payload == {
        "event_id": "run_001-000001",
        "session_id": "run_001",
        "sequence": 1,
        "timestamp_utc": "2026-06-11T00:00:00+00:00",
        "event_type": "node_started",
        "status": "running",
        "node_name": "source_discovery",
        "message": "Source discovery started.",
        "payload": {"planned_query_count": 3},
        "duration_ms": None,
        "artifact_paths": {},
        "error": None,
    }


def test_sanitize_event_payload_redacts_secrets_and_truncates_large_text():
    payload = {
        "api_key": "tvly-live-secret",
        "nested": {"token": "sk-ant-live-secret"},
        "safe": "visible",
        "document": {"clean_text": "A" * 800},
        "rows": [{"value": index} for index in range(40)],
    }

    sanitized = sanitize_event_payload(payload, max_string_length=120, max_list_items=5)
    serialized = json.dumps(sanitized, ensure_ascii=False)

    assert "tvly-live-secret" not in serialized
    assert "sk-ant-live-secret" not in serialized
    assert sanitized["api_key"] == "***REDACTED***"
    assert sanitized["nested"]["token"] == "***REDACTED***"
    assert sanitized["safe"] == "visible"
    assert sanitized["document"]["clean_text"].endswith("...[truncated]")
    assert len(sanitized["rows"]) == 6
    assert sanitized["rows"][-1] == {"truncated_items": 35}


def test_run_event_writer_appends_events_and_updates_status_files(tmp_path):
    writer = RunEventWriter(
        session_dir=tmp_path,
        session_id="pytest_session",
        node_order=["source_discovery"],
    )

    writer.append_run_started({"user_request": "collect data"})
    writer.append_node_started("source_discovery", {"api_key": "tvly-secret"})
    writer.append_node_completed(
        "source_discovery",
        {"source_discovery_summary": {"candidate_count": 2}},
        duration_ms=125,
    )
    writer.append_artifact_written("run_report", tmp_path / "workflow_run_report.md")
    writer.append_run_completed({"normalized_record_count": 1}, duration_ms=250)

    events = _read_ndjson(tmp_path / "diagnostics" / "run_events.ndjson")
    assert [event["event_type"] for event in events] == [
        "run_started",
        "node_started",
        "node_completed",
        "artifact_written",
        "run_completed",
    ]
    assert events[1]["payload"]["api_key"] == "***REDACTED***"

    run_status = json.loads(
        (tmp_path / "diagnostics" / "run_status.json").read_text(encoding="utf-8")
    )
    node_status = json.loads(
        (tmp_path / "diagnostics" / "node_status.json").read_text(encoding="utf-8")
    )

    assert run_status["status"] == "completed"
    assert run_status["session_id"] == "pytest_session"
    assert run_status["event_count"] == 5
    assert run_status["artifact_paths"]["run_report"].endswith("workflow_run_report.md")
    assert node_status["source_discovery"]["status"] == "completed"
    assert node_status["source_discovery"]["duration_ms"] == 125
