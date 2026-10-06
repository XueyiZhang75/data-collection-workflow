from __future__ import annotations

import json

from data_collection_workflow.live_dashboard_data import load_dashboard_snapshot, load_run_events


def test_load_run_events_tolerates_partial_ndjson(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "run_events.ndjson").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event_id": "s-000001",
                        "session_id": "s",
                        "sequence": 1,
                        "timestamp_utc": "2026-06-11T00:00:00+00:00",
                        "event_type": "run_started",
                        "status": "running",
                        "node_name": None,
                        "message": "run started",
                        "payload": {},
                        "duration_ms": None,
                        "artifact_paths": {},
                        "error": None,
                    }
                ),
                '{"event_id": "partial"',
            ]
        ),
        encoding="utf-8",
    )

    events = load_run_events(tmp_path)

    assert len(events) == 1
    assert events[0]["event_type"] == "run_started"


def test_load_dashboard_snapshot_combines_status_events_and_artifacts(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "run_events.ndjson").write_text("", encoding="utf-8")
    (diagnostics / "run_status.json").write_text(
        json.dumps({"status": "running", "current_node": "source_discovery"}),
        encoding="utf-8",
    )
    (diagnostics / "node_status.json").write_text(
        json.dumps({"source_discovery": {"status": "running"}}),
        encoding="utf-8",
    )
    (tmp_path / "workflow_run_summary.json").write_text(
        json.dumps(
            {
                "normalized_record_count": 3,
                "human_review_item_count": 2,
                "artifact_paths": {"workflow_console_html": "workflow_console/index.html"},
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_dashboard_snapshot(tmp_path)

    assert snapshot["run_status"]["status"] == "running"
    assert snapshot["node_status"]["source_discovery"]["status"] == "running"
    assert snapshot["summary"]["normalized_record_count"] == 3
    assert snapshot["artifact_paths"]["workflow_console_html"] == "workflow_console/index.html"
