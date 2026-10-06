from __future__ import annotations

import json

from data_collection_workflow.workflow_notebook import write_workflow_replay_notebook


def test_write_workflow_replay_notebook_creates_valid_ipynb(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "run_events.ndjson").write_text(
        json.dumps(
            {
                "event_id": "notebook-000001",
                "session_id": "notebook",
                "sequence": 1,
                "timestamp_utc": "2026-06-11T00:00:00+00:00",
                "event_type": "node_completed",
                "status": "completed",
                "node_name": "structured_extraction",
                "message": "Built 2 raw records.",
                "payload": {"raw_record_count": 2},
                "duration_ms": 42,
                "artifact_paths": {},
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (diagnostics / "run_status.json").write_text(
        json.dumps({"session_id": "notebook", "status": "completed"}),
        encoding="utf-8",
    )
    (tmp_path / "workflow_run_summary.json").write_text(
        json.dumps(
            {
                "session_id": "notebook",
                "normalized_record_count": 2,
                "artifact_paths": {"run_report": "workflow_run_report.md"},
            }
        ),
        encoding="utf-8",
    )

    path = write_workflow_replay_notebook(tmp_path)
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "workflow_replay_notebook.ipynb"
    assert notebook["nbformat"] == 4
    assert any("Workflow Replay" in cell["source"] for cell in notebook["cells"])
    assert any("structured_extraction" in cell["source"] for cell in notebook["cells"])
    assert any("workflow_run_report.md" in cell["source"] for cell in notebook["cells"])
