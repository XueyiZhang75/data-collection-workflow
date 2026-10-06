from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_workflow.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_workflow_under_test",
    RUNNER_PATH,
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


def _events(session_dir: Path) -> list[dict]:
    path = session_dir / "diagnostics" / "run_events.ndjson"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_graph_with_events_streams_tasks_and_accumulates_final_state(
    tmp_path,
    monkeypatch,
):
    class FakeGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            assert stream_mode == ["tasks", "updates", "custom"]
            assert version == "v2"
            assert initial_state["user_request"] == "collect test data"
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "source_discovery",
                    "input": {"api_key": "tvly-secret"},
                },
            }
            yield {
                "type": "updates",
                "data": {
                    "source_discovery": {
                        "source_candidates": [{"source_id": "src_1"}],
                        "source_discovery_summary": {"candidate_count": 1},
                    }
                },
            }
            yield {
                "type": "custom",
                "data": {
                    "node_name": "source_discovery",
                    "message": "query completed",
                    "payload": {"provider": "tavily", "result_count": 1},
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "source_discovery",
                    "result": {"source_discovery_summary": {"candidate_count": 1}},
                    "error": None,
                },
            }

    monkeypatch.setattr(runner, "build_graph", lambda: FakeGraph())

    result = runner._run_graph_with_events(
        {"user_request": "collect test data"},
        output_dir=tmp_path,
        session_id="streaming_pytest",
        live_status=False,
    )

    assert result["source_candidates"] == [{"source_id": "src_1"}]
    assert result["source_discovery_summary"] == {"candidate_count": 1}

    events = _events(tmp_path)
    assert [event["event_type"] for event in events] == [
        "run_started",
        "node_started",
        "custom_progress",
        "node_completed",
        "run_completed",
    ]
    assert "tvly-secret" not in (tmp_path / "diagnostics" / "run_events.ndjson").read_text(
        encoding="utf-8"
    )


def test_run_graph_with_events_records_failure_status(tmp_path, monkeypatch):
    class FailingGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "content_fetch_and_parse",
                    "input": {"source_count": 1},
                },
            }
            raise RuntimeError("fetch failed")

    monkeypatch.setattr(runner, "build_graph", lambda: FailingGraph())

    with pytest.raises(RuntimeError, match="fetch failed"):
        runner._run_graph_with_events(
            {"user_request": "collect test data"},
            output_dir=tmp_path,
            session_id="streaming_failure_pytest",
            live_status=False,
        )

    events = _events(tmp_path)
    assert events[-1]["event_type"] == "node_failed"
    assert events[-1]["node_name"] == "content_fetch_and_parse"
    assert events[-1]["error"]["message"] == "fetch failed"

    run_status = json.loads(
        (tmp_path / "diagnostics" / "run_status.json").read_text(encoding="utf-8")
    )
    assert run_status["status"] == "failed"
    assert run_status["current_node"] == "content_fetch_and_parse"


def test_run_graph_with_events_raises_when_task_event_contains_error(
    tmp_path,
    monkeypatch,
):
    class ErrorEventGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "structured_extraction",
                    "input": {"chunk_count": 1},
                },
            }
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "structured_extraction",
                    "result": None,
                    "error": {"message": "bad extraction"},
                },
            }

    monkeypatch.setattr(runner, "build_graph", lambda: ErrorEventGraph())

    with pytest.raises(RuntimeError, match="bad extraction"):
        runner._run_graph_with_events(
            {"user_request": "collect test data"},
            output_dir=tmp_path,
            session_id="streaming_error_event_pytest",
            live_status=False,
        )

    events = _events(tmp_path)
    assert [event["event_type"] for event in events] == [
        "run_started",
        "node_started",
        "node_failed",
    ]


def test_run_graph_with_events_passes_langsmith_trace_config_and_flushes(tmp_path, monkeypatch):
    captured = {}
    flushed = []

    class FakeGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            captured["config"] = config
            yield {
                "type": "tasks",
                "data": {
                    "id": "task-1",
                    "name": "source_discovery",
                    "input": {"query_count": 1},
                },
            }
            yield {
                "type": "updates",
                "data": {"source_discovery": {"source_candidates": []}},
            }
            yield {
                "type": "tasks",
                "data": {"id": "task-1", "name": "source_discovery", "result": {}, "error": None},
            }

    monkeypatch.setattr(runner, "build_graph", lambda: FakeGraph())
    monkeypatch.setattr(runner, "_flush_langsmith_tracers", lambda: flushed.append(True))

    runner._run_graph_with_events(
        {"user_request": "collect test data"},
        output_dir=tmp_path,
        session_id="trace_session",
        live_status=False,
    )

    config = captured["config"]
    assert config["run_name"] == "Data Collection Workflow run trace_session"
    assert "data-collection-workflow" in config["tags"]
    assert config["metadata"]["session_id"] == "trace_session"
    assert config["metadata"]["langgraph_graph"] == "data_collection_workflow"
    assert "trace_id" in config["metadata"]
    assert flushed == [True]

    run_status = json.loads(
        (tmp_path / "diagnostics" / "run_status.json").read_text(encoding="utf-8")
    )
    assert run_status["trace_id"] == config["metadata"]["trace_id"]
    assert run_status["langsmith_project"]


def test_run_graph_with_events_suppresses_external_langsmith_env_by_default(
    tmp_path,
    monkeypatch,
):
    captured_env = {}

    class FakeGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            captured_env["LANGSMITH_API_KEY"] = os.environ.get("LANGSMITH_API_KEY")
            captured_env["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
            captured_env["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING")
            captured_env["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2")
            yield {
                "type": "tasks",
                "data": {"id": "task-1", "name": "source_discovery", "input": {}},
            }
            yield {
                "type": "tasks",
                "data": {"id": "task-1", "name": "source_discovery", "result": {}, "error": None},
            }

    monkeypatch.setattr(runner, "build_graph", lambda: FakeGraph())
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-secret")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("ENABLE_LANGSMITH_TRACE", raising=False)

    runner._run_graph_with_events(
        {"user_request": "collect test data"},
        output_dir=tmp_path,
        session_id="no_external_trace_session",
        live_status=False,
    )

    assert captured_env == {
        "LANGSMITH_API_KEY": None,
        "LANGCHAIN_API_KEY": None,
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
    }
    assert os.environ["LANGSMITH_API_KEY"] == "ls-secret"
    assert os.environ["LANGCHAIN_API_KEY"] == "lc-secret"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"


def test_run_graph_with_events_flushes_langsmith_tracers_on_failure(tmp_path, monkeypatch):
    flushed = []

    class FailingGraph:
        def stream(self, initial_state, *, config=None, stream_mode, version):
            yield {
                "type": "tasks",
                "data": {"id": "task-1", "name": "source_discovery", "input": {}},
            }
            raise RuntimeError("planned failure")

    monkeypatch.setattr(runner, "build_graph", lambda: FailingGraph())
    monkeypatch.setattr(runner, "_flush_langsmith_tracers", lambda: flushed.append(True))

    with pytest.raises(RuntimeError, match="planned failure"):
        runner._run_graph_with_events(
            {"user_request": "collect test data"},
            output_dir=tmp_path,
            session_id="trace_failure_session",
            live_status=False,
        )

    assert flushed == [True]
