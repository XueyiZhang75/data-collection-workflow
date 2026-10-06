from __future__ import annotations

import json
import importlib.util
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_provider_configuration_visual_launcher_uses_environment(monkeypatch):
    module = _load_run_visual_workflow_module(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "account-specific-model")
    captured = []
    monkeypatch.setattr(module.start_langflow_demo, "main", lambda argv: captured.extend(argv) or 0)
    assert module.main([
        "--disease", "flu", "--location", "Virginia", "--start-date", "2024",
        "--end-date", "2024", "--session-id", "provider-test", "--user-request", "Collect records",
    ]) == 0
    assert captured[captured.index("--runner-provider") + 1] == "openai"
    assert captured[captured.index("--runner-model") + 1] == "account-specific-model"


def test_provider_configuration_missing_model_stops_before_background_setup(monkeypatch):
    from data_collection_workflow import langflow_demo as adapter
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(adapter, "_scripts_module", lambda name: pytest.fail("runner setup must not start"))
    request = adapter.LangflowDemoRunRequest(
        disease="flu", location="Virginia", start_date="2024", end_date="2024", model="",
    ).with_resolved_session_id()
    with pytest.raises(adapter.RunStartError, match="model|LLM_MODEL"):
        adapter.start_data_collection_workflow_background(request, adapter.RunRegistry())


def test_provider_configuration_flow_has_no_model_preset():
    project_root = Path(__file__).resolve().parents[1]
    flow = json.loads((project_root / "integrations/langflow/flows/workflow.json").read_text(encoding="utf-8"))
    runner = next(node for node in flow["data"]["nodes"] if node["data"]["type"] == "workflow_runner")
    template = runner["data"]["node"]["template"]
    assert template["provider"]["value"] == ""
    assert template["model"]["value"] == ""
    assert not template["model"].get("options")
    assert "model ID" in template["model"]["info"]
    component = project_root / "integrations/langflow/components/data_collection_workflow/workflow_runner.py"
    assert template["code"]["value"] == component.read_text(encoding="utf-8")


@pytest.mark.parametrize("provider,model,expected", [
    ("", "", ("openai", "environment-model")),
    ("openai", "custom/model-2026-10", ("openai", "custom/model-2026-10")),
    ("anthropic", "account-model", ("anthropic", "account-model")),
])
def test_provider_configuration_background_preserves_selection(monkeypatch, tmp_path, provider, model, expected):
    from data_collection_workflow import langflow_demo as adapter
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-credential-do-not-serialize")
    captured = {}
    def make_config(**kwargs):
        captured.update(kwargs)
        return {"llm": {"provider": kwargs["provider"], "model": kwargs["model"]}}
    interactive = SimpleNamespace(
        DEFAULT_TARGET_FIELDS=["cases"], _require_keys=lambda **kwargs: [],
        _real_run_config=make_config,
        _write_generated_config=lambda config, session_id: tmp_path / "config.json",
        _session_dir_for_config=lambda config: tmp_path / "session",
    )
    monkeypatch.setattr(adapter, "_scripts_module", lambda name: interactive if name == "collect" else SimpleNamespace())
    monkeypatch.setattr(adapter.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    request = adapter.LangflowDemoRunRequest(
        disease="flu", location="Virginia", start_date="2024", end_date="2024",
        session_id="provider-test", provider=provider, model=model,
    ).with_resolved_session_id()
    registry = adapter.RunRegistry()
    adapter.start_data_collection_workflow_background(request, registry)
    assert (captured["provider"], captured["model"]) == expected
    metadata = registry.get("provider-test")["request_metadata"]
    assert (metadata["provider"], metadata["model"]) == expected
    assert "synthetic-test-credential" not in json.dumps([captured, metadata])


def test_provider_configuration_unsupported_provider_stops_before_setup(monkeypatch):
    from data_collection_workflow import langflow_demo as adapter
    monkeypatch.setattr(adapter, "_scripts_module", lambda name: pytest.fail("runner setup must not start"))
    request = adapter.LangflowDemoRunRequest(
        disease="flu", location="Virginia", start_date="2024", end_date="2024",
        provider="unsupported-provider", model="custom-model",
    )
    with pytest.raises(adapter.RunStartError, match="anthropic or openai"):
        adapter.start_data_collection_workflow_background(request, adapter.RunRegistry())


@pytest.mark.parametrize("script", ["start_studio", "start_langflow_demo", "start_langflow_tracing"])
def test_provider_configuration_launchers_load_env_before_setup(monkeypatch, script):
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(f"{script}_env_test", project_root / "scripts" / f"{script}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []
    monkeypatch.setattr(module, "load_project_env", lambda: seen.append("dotenv"))
    def stop_setup(*args):
        assert seen == ["dotenv"]
        raise RuntimeError("offline setup boundary")
    monkeypatch.setattr(module, "_config_with_cli_overrides" if script == "start_studio" else "_require_optional_module", stop_setup)
    with pytest.raises(RuntimeError, match="offline setup boundary"):
        module.main([])


def test_provider_configuration_api_startup_loads_env_and_missing_model_is_english(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from data_collection_workflow import langflow_demo as adapter
    loaded = []
    monkeypatch.setattr(adapter, "load_project_env", lambda: loaded.append(True))
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setattr(adapter, "_scripts_module", lambda name: pytest.fail("runner setup must not start"))
    app = adapter.create_app()
    assert loaded == [True]
    response = TestClient(app).post("/runs", json={
        "disease": "flu", "location": "Virginia", "start_date": "2024", "end_date": "2024",
        "visual_contract_version": VISUAL_CONTRACT_VERSION,
    })
    assert response.status_code == 400
    assert "LLM model ID is required" in response.json()["detail"]
    assert response.json()["detail"].isascii()


def test_provider_configuration_studio_preview_allows_blank_model_but_launch_rejects(monkeypatch, capsys):
    project_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("studio_model_test", project_root / "scripts/start_studio.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_project_env", lambda: None)
    monkeypatch.setattr(module, "_config_with_cli_overrides", lambda args: (Path("config.json"), {"llm": {"provider": "openai", "model": ""}}))
    monkeypatch.setattr(module, "workflow_run_env_from_config", lambda config: {"ENABLE_LLM_SOURCE_IDENTITY": "true"})
    monkeypatch.setattr(module, "_print_launch_context", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: pytest.fail("server must not start"))
    assert module.main(["--print-config-only"]) == 0
    assert module.main([]) == 2
    error = capsys.readouterr().err
    assert "LLM model ID is required" in error
    assert error.isascii()

from data_collection_workflow.langflow_demo import (
    ArtifactAccessError,
    RunRegistry,
    VISUAL_CONTRACT_VERSION,
    apply_quick_test_mode,
    build_node_snapshot,
    build_run_snapshot,
    normalize_date_range,
    resolve_artifact_path,
    resolve_langsmith_trace_url,
)


def test_build_run_snapshot_exposes_allowed_artifact_urls(tmp_path):
    session_dir = tmp_path / "sessions" / "demo_session"
    console = session_dir / "workflow_console" / "data_collection_workflow_console.html"
    visualization = session_dir / "workflow_visualization" / "index.html"
    final_csv = session_dir / "collection" / "final_dataset.csv"
    final_json = session_dir / "collection" / "final_dataset.json"
    latest_console = tmp_path / "workflow_console" / "data_collection_workflow_console.html"
    for path in (console, visualization, final_csv, final_json, latest_console):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("demo", encoding="utf-8")

    (session_dir / "workflow_run_summary.json").write_text(
        json.dumps(
            {
                "normalized_record_count": 3,
                "human_review_item_count": 2,
                "artifact_paths": {
                    "workflow_console_html": str(console),
                    "workflow_visualization_index": str(visualization),
                    "latest_workflow_console_html": str(latest_console),
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_run_snapshot(
        session_id="demo_session",
        session_dir=session_dir,
        api_base_url="http://127.0.0.1:8010",
        status="completed",
        dashboard_url="http://localhost:8501",
    )

    assert snapshot["session_id"] == "demo_session"
    assert snapshot["status"] == "completed"
    assert snapshot["live_dashboard_url"] == "http://localhost:8501"
    assert snapshot["artifact_urls"]["workflow_console_html"].endswith(
        "/runs/demo_session/artifacts/workflow_console_html"
    )
    assert snapshot["artifact_urls"]["workflow_visualization_index"].endswith(
        "/runs/demo_session/artifacts/workflow_visualization_index"
    )
    assert snapshot["artifact_urls"]["final_dataset_csv"].endswith(
        "/runs/demo_session/artifacts/final_dataset_csv"
    )
    assert snapshot["artifact_urls"]["final_dataset_json"].endswith(
        "/runs/demo_session/artifacts/final_dataset_json"
    )
    assert "latest_workflow_console_html" not in snapshot["artifact_urls"]
    assert snapshot["summary"]["normalized_record_count"] == 3


def test_resolve_artifact_path_rejects_unknown_keys_and_escaped_paths(tmp_path):
    session_dir = tmp_path / "sessions" / "demo_session"
    session_dir.mkdir(parents=True)
    outside_file = tmp_path / "outside.html"
    outside_file.write_text("outside", encoding="utf-8")
    (session_dir / "workflow_run_summary.json").write_text(
        json.dumps({"artifact_paths": {"workflow_console_html": str(outside_file)}}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactAccessError):
        resolve_artifact_path(session_dir, "not_a_real_artifact")

    with pytest.raises(ArtifactAccessError):
        resolve_artifact_path(session_dir, "workflow_console_html")


def test_run_registry_tracks_background_status(tmp_path):
    registry = RunRegistry()
    session_dir = tmp_path / "sessions" / "demo_session"

    registry.register(
        session_id="demo_session",
        session_dir=session_dir,
        dashboard_url="http://localhost:8501",
    )
    registry.mark_running("demo_session")
    running = registry.get("demo_session")

    assert running["status"] == "running"
    assert running["session_dir"] == str(session_dir)
    assert running["live_dashboard_url"] == "http://localhost:8501"

    registry.mark_completed("demo_session", {"normalized_record_count": 1})
    completed = registry.get("demo_session")

    assert completed["status"] == "completed"
    assert completed["summary"]["normalized_record_count"] == 1


def test_fastapi_app_accepts_background_runner_when_fastapi_is_installed(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    def fake_runner(run_request, registry):
        session_dir = tmp_path / "sessions" / run_request.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "workflow_run_summary.json").write_text(
            json.dumps({"artifact_paths": {}}),
            encoding="utf-8",
        )
        registry.register(
            session_id=run_request.session_id,
            session_dir=session_dir,
            dashboard_url="http://localhost:8501",
        )
        registry.mark_completed(run_request.session_id, {"output_dir": str(session_dir)})

    app = create_app(runner=fake_runner, api_base_url="http://testserver")
    client = TestClient(app)

    response = client.post(
        "/runs",
        json={
            "disease": "hantavirus",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2026",
            "session_id": "demo_session",
            "no_llm": True,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo_session"
    assert payload["status_url"].endswith("/runs/demo_session")

    status = client.get(
        "/runs/demo_session",
        params={"visual_contract_version": VISUAL_CONTRACT_VERSION},
    )
    assert status.status_code == 200
    assert status.json()["status"] == "completed"


def test_fastapi_app_accepts_user_request_and_does_not_require_dashboard(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    captured = {}

    def fake_runner(run_request, registry):
        captured["request"] = run_request
        session_dir = tmp_path / "outputs" / "sessions" / run_request.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")
        registry.register(
            session_id=run_request.session_id,
            session_dir=session_dir,
            dashboard_url=None,
        )
        registry.mark_completed(run_request.session_id, {"output_dir": str(session_dir)})

    app = create_app(runner=fake_runner, api_base_url="http://testserver")
    client = TestClient(app)

    response = client.post(
        "/runs",
        json={
            "user_request": "Collect influenza hospitalization data for New York in 2024.",
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2025",
            "session_id": "demo_flu",
            "no_llm": True,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )

    assert response.status_code == 200
    assert captured["request"].user_request == (
        "Collect influenza hospitalization data for New York in 2024."
    )
    assert captured["request"].dashboard_enabled is False
    assert response.json()["live_dashboard_url"] is None


def test_langflow_demo_normalizes_day_level_dates_and_quick_mode_budget():
    assert normalize_date_range("2024-5-1", "2024-5-3") == (
        "2024-05-01",
        "2024-05-03",
    )
    assert normalize_date_range("2024", "2025") == (
        "2024-01-01",
        "2025-12-31",
    )

    config = {
        "source_search": {
            "max_queries": 8,
            "max_results_per_query": 8,
            "max_total_results": 64,
            "iterative": {
                "enabled": True,
                "max_iterations": 3,
                "max_queries_per_iteration": 4,
                "max_total_queries": 20,
                "max_total_results": 120,
            },
        },
        "content_fetch": {
            "max_search_derived_sources": 50,
            "max_total_sources": 50,
            "external_fetch": {
                "adaptive_budget": {
                    "max_candidate_urls": 120,
                    "max_fetch_urls": 50,
                    "min_usable_documents": 12,
                    "min_collection_sources": 6,
                    "min_validation_sources": 3,
                    "max_iterations": 3,
                }
            },
        },
        "llm": {
            "max_chunks": 30,
            "source_critic": {"max_sources": 30},
            "source_identity": {"max_sources": 30},
            "source_credibility": {"max_sources": 6},
        },
    }

    apply_quick_test_mode(config)

    assert config["source_search"]["max_queries"] == 2
    assert config["source_search"]["max_results_per_query"] == 3
    assert config["source_search"]["max_total_results"] == 6
    assert config["source_search"]["iterative"]["max_iterations"] == 1
    assert config["source_search"]["iterative"]["max_total_queries"] == 3
    assert config["content_fetch"]["max_search_derived_sources"] == 5
    assert config["content_fetch"]["max_total_sources"] == 5
    assert config["content_fetch"]["external_fetch"]["adaptive_budget"]["max_fetch_urls"] == 5
    assert config["llm"]["max_chunks"] == 5
    assert config["llm"]["source_critic"]["max_sources"] == 5


def test_fastapi_app_records_normalized_dates_quick_mode_and_result_locations(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    captured = {}

    def fake_runner(run_request, registry):
        captured["request"] = run_request
        session_dir = tmp_path / "outputs" / "sessions" / run_request.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        generated_config = tmp_path / "generated_configs" / f"{run_request.session_id}.json"
        generated_config.parent.mkdir(parents=True, exist_ok=True)
        generated_config.write_text("{}", encoding="utf-8")
        (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")
        registry.register(
            session_id=run_request.session_id,
            session_dir=session_dir,
            generated_config_path=generated_config,
            request_metadata=run_request.model_dump(),
        )
        registry.mark_completed(run_request.session_id, {"output_dir": str(session_dir)})

    app = create_app(runner=fake_runner, api_base_url="http://testserver")
    client = TestClient(app)

    response = client.post(
        "/runs",
        json={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-5-1",
            "end_date": "2024-5-3",
            "session_id": "demo_flu_days",
            "quick_test_mode": True,
            "no_llm": True,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert captured["request"].start_date == "2024-05-01"
    assert captured["request"].end_date == "2024-05-03"
    assert captured["request"].quick_test_mode is True
    assert body["normalized_start_date"] == "2024-05-01"
    assert body["normalized_end_date"] == "2024-05-03"
    assert body["quick_test_mode"] is True
    assert "outputs/sessions/demo_flu_days" in body["result_location_text"].replace("\\", "/")
    assert body["generated_config_path"].endswith("demo_flu_days.json")


def test_fastapi_app_returns_existing_run_for_duplicate_session(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    calls = []

    def fake_runner(run_request, registry):
        calls.append(run_request.session_id)
        session_dir = tmp_path / "sessions" / run_request.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "workflow_run_summary.json").write_text(
            json.dumps({"artifact_paths": {}}),
            encoding="utf-8",
        )
        registry.register(
            session_id=run_request.session_id,
            session_dir=session_dir,
            dashboard_url="http://localhost:8501",
        )
        registry.mark_completed(run_request.session_id, {"output_dir": str(session_dir)})

    app = create_app(runner=fake_runner, api_base_url="http://testserver")
    client = TestClient(app)
    payload = {
        "disease": "hantavirus",
        "location": "New York",
        "start_date": "2024",
        "end_date": "2026",
        "session_id": "demo_session",
        "no_llm": True,
        "visual_contract_version": VISUAL_CONTRACT_VERSION,
    }

    first = client.post("/runs", json=payload)
    second = client.post("/runs", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == ["demo_session"]
    assert second.json()["session_id"] == "demo_session"
    assert second.json()["status"] == "completed"


def test_fastapi_app_rejects_missing_or_old_visual_contract_version(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    app = create_app(runner=lambda run_request, registry: None, api_base_url="http://testserver")
    client = TestClient(app)
    base_payload = {
        "disease": "hantavirus",
        "location": "New York",
        "start_date": "2024",
        "end_date": "2026",
        "session_id": "demo_session",
        "no_llm": True,
    }

    missing = client.post("/runs", json=base_payload)
    old = client.post(
        "/runs",
        json={**base_payload, "visual_contract_version": "hdc-langflow-visual-v1"},
    )

    assert missing.status_code == 400
    assert old.status_code == 400
    assert VISUAL_CONTRACT_VERSION in missing.json()["detail"]
    assert VISUAL_CONTRACT_VERSION in old.json()["detail"]


def test_fastapi_app_rejects_old_visual_contract_for_run_and_node_snapshots(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    session_dir = tmp_path / "sessions" / "demo_session"
    _write_deep_run_files(session_dir)
    registry = RunRegistry()
    registry.register(session_id="demo_session", session_dir=session_dir)
    registry.mark_running("demo_session")

    app = create_app(
        registry=registry,
        runner=lambda run_request, registry: None,
        api_base_url="http://testserver",
    )
    client = TestClient(app)

    missing_run = client.get("/runs/demo_session")
    old_run = client.get(
        "/runs/demo_session",
        params={"visual_contract_version": "hdc-langflow-visual-v1"},
    )
    missing_node = client.get("/runs/demo_session/nodes/source_discovery")
    valid_node = client.get(
        "/runs/demo_session/nodes/source_discovery",
        params={"visual_contract_version": VISUAL_CONTRACT_VERSION},
    )

    assert missing_run.status_code == 400
    assert old_run.status_code == 400
    assert missing_node.status_code == 400
    assert valid_node.status_code == 200


def test_build_run_snapshot_includes_studio_and_langsmith_links(tmp_path, monkeypatch):
    from data_collection_workflow import langflow_demo

    session_dir = tmp_path / "sessions" / "demo_session"
    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")
    (diagnostics / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_node": "source_discovery",
                "trace_id": "trace-123",
                "langsmith_project": "data-collection-workflow-demo",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        langflow_demo,
        "resolve_langsmith_trace_url",
        lambda **kwargs: "https://smith.langchain.com/r/trace-123",
    )
    monkeypatch.setenv("LANGGRAPH_STUDIO_URL", "http://127.0.0.1:2024")

    snapshot = build_run_snapshot(
        session_id="demo_session",
        session_dir=session_dir,
        api_base_url="http://127.0.0.1:8010",
        dashboard_url="http://localhost:8501",
    )

    assert snapshot["studio_url"] == "http://127.0.0.1:2024"
    assert snapshot["langsmith_project"] == "data-collection-workflow-demo"
    assert snapshot["trace_id"] == "trace-123"
    assert snapshot["langsmith_trace_url"] == "https://smith.langchain.com/r/trace-123"


def test_build_run_snapshot_includes_node_progress_summary(tmp_path):
    session_dir = tmp_path / "sessions" / "demo_session"
    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")
    (diagnostics / "run_status.json").write_text(
        json.dumps({"status": "running", "current_node": "source_discovery"}),
        encoding="utf-8",
    )
    (diagnostics / "node_status.json").write_text(
        json.dumps(
            {
                "task_intake_and_scope_planning": {
                    "node_name": "task_intake_and_scope_planning",
                    "status": "completed",
                    "duration_ms": 5,
                    "last_message": "task completed",
                },
                "source_discovery": {
                    "node_name": "source_discovery",
                    "status": "running",
                    "last_message": "searching",
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = build_run_snapshot(
        session_id="demo_session",
        session_dir=session_dir,
        api_base_url="http://127.0.0.1:8010",
    )

    assert snapshot["node_progress"]["completed_count"] == 1
    assert snapshot["node_progress"]["running_nodes"] == ["source_discovery"]
    assert snapshot["node_progress"]["total_count"] >= 20
    assert snapshot["node_timeline"][0]["node_name"] == "task_intake_and_scope_planning"
    assert snapshot["node_timeline"][0]["status"] == "completed"


def test_resolve_langsmith_trace_url_uses_project_trace_and_client():
    class FakeRun:
        id = "run-1"

    class FakeClient:
        def list_runs(self, **kwargs):
            assert kwargs["project_name"] == "data-collection-workflow-demo"
            assert kwargs["trace_id"] == "trace-123"
            assert kwargs["is_root"] is True
            assert kwargs["limit"] == 1
            return iter([FakeRun()])

        def get_run_url(self, **kwargs):
            assert kwargs["run"].id == "run-1"
            assert kwargs["project_name"] == "data-collection-workflow-demo"
            return "https://smith.langchain.com/r/run-1"

    assert resolve_langsmith_trace_url(
        session_id="demo_session",
        trace_id="trace-123",
        project_name="data-collection-workflow-demo",
        client_factory=lambda: FakeClient(),
    ) == "https://smith.langchain.com/r/run-1"


def _write_deep_run_files(session_dir: Path) -> None:
    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_id": "demo-000001",
            "session_id": "demo_session",
            "sequence": 1,
            "timestamp_utc": "2026-06-12T00:00:00+00:00",
            "event_type": "node_started",
            "status": "running",
            "node_name": "source_discovery",
            "message": "source_discovery started.",
            "payload": {"input": {"query_count": 2}},
            "duration_ms": None,
            "artifact_paths": {},
            "error": None,
        },
        {
            "event_id": "demo-000002",
            "session_id": "demo_session",
            "sequence": 2,
            "timestamp_utc": "2026-06-12T00:00:01+00:00",
            "event_type": "custom_progress",
            "status": "running",
            "node_name": "source_discovery",
            "message": "query completed",
            "payload": {"provider": "tavily", "result_count": 4},
            "duration_ms": None,
            "artifact_paths": {},
            "error": None,
        },
        {
            "event_id": "demo-000003",
            "session_id": "demo_session",
            "sequence": 3,
            "timestamp_utc": "2026-06-12T00:00:02+00:00",
            "event_type": "node_completed",
            "status": "completed",
            "node_name": "source_discovery",
            "message": "source_discovery completed.",
            "payload": {"source_discovery_summary": {"candidate_count": 4}},
            "duration_ms": 250,
            "artifact_paths": {},
            "error": None,
        },
        {
            "event_id": "demo-000004",
            "session_id": "demo_session",
            "sequence": 4,
            "timestamp_utc": "2026-06-12T00:00:03+00:00",
            "event_type": "node_completed",
            "status": "completed",
            "node_name": "structured_extraction",
            "message": "structured_extraction completed.",
            "payload": {
                "llm_enabled": True,
                "llm_call_count": 2,
                "llm_success_count": 2,
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            },
            "duration_ms": 500,
            "artifact_paths": {},
            "error": None,
        },
    ]
    (diagnostics / "run_events.ndjson").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    (diagnostics / "node_status.json").write_text(
        json.dumps(
            {
                "source_discovery": {
                    "node_name": "source_discovery",
                    "status": "completed",
                    "duration_ms": 250,
                    "last_payload": {"source_discovery_summary": {"candidate_count": 4}},
                },
                "structured_extraction": {
                    "node_name": "structured_extraction",
                    "status": "completed",
                    "duration_ms": 500,
                    "last_payload": {
                        "llm_enabled": True,
                        "llm_call_count": 2,
                        "llm_success_count": 2,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (diagnostics / "run_status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_node": "structured_extraction",
                "trace_id": "trace-123",
                "langsmith_project": "data-collection-workflow-demo",
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")


def test_build_node_snapshot_exposes_status_events_tool_and_llm_summaries(tmp_path, monkeypatch):
    from data_collection_workflow import langflow_demo

    session_dir = tmp_path / "sessions" / "demo_session"
    _write_deep_run_files(session_dir)
    monkeypatch.setattr(
        langflow_demo,
        "resolve_langsmith_trace_url",
        lambda **kwargs: "https://smith.langchain.com/r/trace-123",
    )

    source = build_node_snapshot(
        session_id="demo_session",
        session_dir=session_dir,
        node_name="source_discovery",
        api_base_url="http://127.0.0.1:8010",
    )
    extraction = build_node_snapshot(
        session_id="demo_session",
        session_dir=session_dir,
        node_name="structured_extraction",
        api_base_url="http://127.0.0.1:8010",
    )

    assert source["status"] == "completed"
    assert source["duration_ms"] == 250
    assert source["input_summary"]["input"]["query_count"] == 2
    assert source["output_summary"]["source_discovery_summary"]["candidate_count"] == 4
    assert source["last_payload"]["source_discovery_summary"]["candidate_count"] == 4
    assert source["tool_summary"]["providers"] == ["tavily"]
    assert source["trace_url"] == "https://smith.langchain.com/r/trace-123"
    assert "Node: source_discovery" in source["status_text"]

    assert extraction["llm_summary"]["enabled"] is True
    assert extraction["llm_summary"]["call_count"] == 2
    assert extraction["llm_summary"]["provider"] == "anthropic"
    assert "## source_discovery" in source["node_detail_markdown"]
    assert "Input summary" in source["node_detail_markdown"]
    assert "Output summary" in source["node_detail_markdown"]
    assert "Tool summary" in source["node_detail_markdown"]


def test_build_node_snapshot_compacts_large_payloads_for_langflow(tmp_path):
    from data_collection_workflow.langflow_demo import build_node_snapshot

    session_dir = tmp_path / "sessions" / "large_session"
    diagnostics = session_dir / "diagnostics"
    diagnostics.mkdir(parents=True)
    huge_payload = {
        "query_count": 20,
        "long_text": "x" * 2500,
        "large_list": [{"rank": index, "text": "y" * 200} for index in range(60)],
        "deep": {"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
    }
    events = [
        {
            "event_type": "node_started",
            "node_name": "source_discovery",
            "payload": {"input": huge_payload},
        },
        {
            "event_type": "node_completed",
            "node_name": "source_discovery",
            "payload": {"output": huge_payload},
        },
    ]
    (diagnostics / "run_events.ndjson").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    (diagnostics / "node_status.json").write_text(
        json.dumps(
            {
                "source_discovery": {
                    "node_name": "source_discovery",
                    "status": "completed",
                    "duration_ms": 250,
                    "last_payload": huge_payload,
                }
            }
        ),
        encoding="utf-8",
    )
    (diagnostics / "run_status.json").write_text("{}", encoding="utf-8")
    (session_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")

    snapshot = build_node_snapshot(
        session_id="large_session",
        session_dir=session_dir,
        node_name="source_discovery",
        api_base_url="http://127.0.0.1:8010",
    )

    assert snapshot["last_payload"]["query_count"] == 20
    assert "... [truncated" in snapshot["last_payload"]["long_text"]
    assert len(snapshot["last_payload"]["large_list"]) < len(huge_payload["large_list"])
    assert "_truncated_items" in snapshot["last_payload"]["large_list"][-1]
    assert "_truncated_depth" in snapshot["last_payload"]["deep"]["a"]["b"]["c"]
    assert len(json.dumps(snapshot["last_payload"])) < 7000
    assert len(json.dumps(snapshot["recent_events"])) < 12000


def test_fastapi_app_exposes_events_and_node_snapshots(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow import langflow_demo
    from data_collection_workflow.langflow_demo import create_app

    session_dir = tmp_path / "sessions" / "demo_session"
    _write_deep_run_files(session_dir)
    monkeypatch.setattr(
        langflow_demo,
        "resolve_langsmith_trace_url",
        lambda **kwargs: "https://smith.langchain.com/r/trace-123",
    )

    registry = RunRegistry()
    registry.register(
        session_id="demo_session",
        session_dir=session_dir,
        dashboard_url="http://localhost:8501",
    )
    app = create_app(
        registry=registry,
        runner=lambda request, registry: None,
        api_base_url="http://testserver",
    )
    client = TestClient(app)

    events = client.get("/runs/demo_session/events")
    assert events.status_code == 200
    assert len(events.json()["events"]) == 4

    node = client.get(
        "/runs/demo_session/nodes/source_discovery",
        params={"visual_contract_version": VISUAL_CONTRACT_VERSION},
    )
    assert node.status_code == 200
    assert node.json()["tool_summary"]["providers"] == ["tavily"]

    missing = client.get(
        "/runs/demo_session/nodes/not_a_node",
        params={"visual_contract_version": VISUAL_CONTRACT_VERSION},
    )
    assert missing.status_code == 404


def test_fastapi_wait_endpoints_return_terminal_or_timeout_snapshots(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from data_collection_workflow.langflow_demo import create_app

    completed_dir = tmp_path / "sessions" / "completed_session"
    _write_deep_run_files(completed_dir)
    (completed_dir / "diagnostics" / "run_status.json").write_text(
        json.dumps({"status": "completed", "current_node": None}),
        encoding="utf-8",
    )

    pending_dir = tmp_path / "sessions" / "pending_session"
    (pending_dir / "diagnostics").mkdir(parents=True)
    (pending_dir / "workflow_run_summary.json").write_text("{}", encoding="utf-8")
    (pending_dir / "diagnostics" / "node_status.json").write_text("{}", encoding="utf-8")
    (pending_dir / "diagnostics" / "run_status.json").write_text(
        json.dumps({"status": "running", "current_node": "source_discovery"}),
        encoding="utf-8",
    )

    registry = RunRegistry()
    registry.register(session_id="completed_session", session_dir=completed_dir)
    registry.mark_completed("completed_session", {})
    registry.register(session_id="pending_session", session_dir=pending_dir)
    registry.mark_running("pending_session")

    app = create_app(
        registry=registry,
        runner=lambda request, registry: None,
        api_base_url="http://testserver",
    )
    client = TestClient(app)

    node = client.get(
        "/runs/completed_session/nodes/source_discovery/wait",
        params={
            "timeout_seconds": 0.01,
            "poll_interval_seconds": 0.001,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )
    assert node.status_code == 200
    assert node.json()["status"] == "completed"
    assert node.json().get("polling_timed_out") is not True

    run = client.get(
        "/runs/completed_session/wait",
        params={
            "timeout_seconds": 0.01,
            "poll_interval_seconds": 0.001,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )
    assert run.status_code == 200
    assert run.json()["status"] == "completed"

    timeout = client.get(
        "/runs/pending_session/nodes/source_discovery/wait",
        params={
            "timeout_seconds": 0.01,
            "poll_interval_seconds": 0.001,
            "visual_contract_version": VISUAL_CONTRACT_VERSION,
        },
    )
    assert timeout.status_code == 200
    assert timeout.json()["status"] == "pending"
    assert timeout.json()["polling_timed_out"] is True


def _load_langflow_status_component(project_root: Path):
    pytest.importorskip("langflow")
    module_path = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "workflow_status.py"
    )
    assert module_path.exists()
    spec = importlib.util.spec_from_file_location("workflow_status_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_langflow_status_component_formats_live_run_progress():
    project_root = Path(__file__).resolve().parents[1]
    module = _load_langflow_status_component(project_root)

    status_text = module.format_status_text(
        {
            "session_id": "demo_session",
            "status": "running",
            "current_node": "executable_source_planning",
            "status_url": "http://127.0.0.1:8010/runs/demo_session",
            "live_dashboard_url": "http://localhost:8502",
            "workflow_console_url": None,
            "workflow_visualization_url": None,
            "artifact_urls": {"final_dataset_csv": "http://example/final.csv"},
            "error": None,
        }
    )

    assert "Session: demo_session" in status_text
    assert "Status: running" in status_text
    assert "Current node: executable_source_planning" in status_text
    assert "Live dashboard: http://localhost:8502" in status_text
    assert "Workflow console: pending" in status_text
    assert "Workflow visualization: pending" in status_text
    assert "Artifact URLs: final_dataset_csv" in status_text


def test_langflow_status_component_fetches_status_snapshot(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    module = _load_langflow_status_component(project_root)

    requested_urls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "session_id": "demo_session",
                "status": "completed",
                "current_node": None,
                "status_url": "http://127.0.0.1:8010/runs/demo_session",
                "live_dashboard_url": "http://localhost:8502",
                "workflow_console_url": "http://127.0.0.1:8010/console",
                "workflow_visualization_url": "http://127.0.0.1:8010/viz",
                "artifact_urls": {},
                "error": None,
            }

    def fake_get(url, timeout):
        requested_urls.append((url, timeout))
        return FakeResponse()

    snapshot = module.fetch_run_status(
        "http://127.0.0.1:8010/",
        "demo_session",
        request_get=fake_get,
    )

    assert requested_urls == [
        (
            "http://127.0.0.1:8010/runs/demo_session"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            10,
        )
    ]
    assert snapshot["status"] == "completed"
    assert "Status: completed" in snapshot["status_text"]


def _load_langflow_runner_component(project_root: Path):
    pytest.importorskip("langflow")
    module_path = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "workflow_runner.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_runner_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_langflow_node_component(project_root: Path):
    pytest.importorskip("langflow")
    module_path = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "node_inspector.py"
    )
    spec = importlib.util.spec_from_file_location("workflow_node_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_langflow_deep_links_component(project_root: Path):
    pytest.importorskip("langflow")
    module_path = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "links.py"
    )
    spec = importlib.util.spec_from_file_location("links_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_langflow_live_timeline_component(project_root: Path):
    pytest.importorskip("langflow")
    module_path = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "timeline.py"
    )
    spec = importlib.util.spec_from_file_location("timeline_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_langflow_runner_builds_native_visual_run_payload():
    module = _load_langflow_runner_component(Path(__file__).resolve().parents[1])

    payload = module.build_run_payload(
        disease="FLU",
        location="New York",
        start_date="2024",
        end_date="2025",
        session_id="demo_session",
        provider="anthropic",
        model="claude-sonnet-4-6",
        no_llm=False,
        user_request="Find influenza hospitalization data in New York.",
        quick_test_mode=True,
    )

    assert payload["user_request"] == "Find influenza hospitalization data in New York."
    assert payload["session_id"] == "demo_session"
    assert payload["dashboard_enabled"] is False
    assert payload["quick_test_mode"] is True
    assert payload["visual_contract_version"] == VISUAL_CONTRACT_VERSION
    assert "dashboard_port" not in payload


def test_langflow_runner_cache_is_bound_to_current_payload(monkeypatch):
    module = _load_langflow_runner_component(Path(__file__).resolve().parents[1])
    component = module.DataCollectionWorkflowRunner()
    component.api_base_url = "http://127.0.0.1:8010"
    component.location = "New York"
    component.start_date = "2024"
    component.end_date = "2025"
    component.session_id = ""
    component.provider = "anthropic"
    component.model = "claude-sonnet-4-6"
    component.no_llm = False
    component.user_request = ""
    component.quick_test_mode = False
    posted = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": self.payload["disease"].lower(), "status": "running"}

    def fake_post(url, json, timeout):
        posted.append(json)
        return FakeResponse(json)

    monkeypatch.setattr(module.requests, "post", fake_post)

    component.disease = "FLU"
    first = component._run_workflow_once()
    component.disease = "COVID-19"
    second = component._run_workflow_once()

    assert first["session_id"] == "flu"
    assert second["session_id"] == "covid-19"
    assert [payload["disease"] for payload in posted] == ["FLU", "COVID-19"]


def test_langflow_node_inspector_polls_snapshot_endpoint_until_completed():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])
    requested = []
    statuses = iter(["running", "completed"])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            status = next(statuses)
            return {
                "node_name": "source_discovery",
                "status": status,
                "duration_ms": 42000 if status == "completed" else None,
                "node_detail_markdown": f"## source_discovery\n{status}",
            }

    def fake_get(url, timeout):
        requested.append((url, timeout))
        return FakeResponse()

    snapshot = module.fetch_node_snapshot(
        "http://127.0.0.1:8010/",
        "demo_session",
        "source_discovery",
        wait_for_completion=True,
        timeout_seconds=123,
        poll_interval_seconds=4,
        request_get=fake_get,
        sleep=lambda seconds: None,
    )

    assert snapshot["status"] == "completed"
    assert snapshot["duration_ms"] == 42000
    assert snapshot["node_detail_markdown"].startswith("## source_discovery")
    assert requested == [
        (
            "http://127.0.0.1:8010/runs/demo_session/nodes/source_discovery"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        ),
        (
            "http://127.0.0.1:8010/runs/demo_session/nodes/source_discovery"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        ),
    ]


def test_langflow_node_inspector_retries_transient_snapshot_timeout():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])
    requested = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "node_name": "evidence_chunking_and_data_presence_flagging",
                "status": "completed",
                "duration_ms": 3210,
            }

    def fake_get(url, timeout):
        requested.append((url, timeout))
        if len(requested) == 1:
            raise module.requests.exceptions.ReadTimeout("read timeout=10")
        return FakeResponse()

    snapshots_seen = []
    snapshot = module.fetch_node_snapshot(
        "http://127.0.0.1:8010/",
        "demo_session",
        "evidence_chunking_and_data_presence_flagging",
        wait_for_completion=True,
        timeout_seconds=30,
        poll_interval_seconds=1,
        request_get=fake_get,
        sleep=lambda seconds: None,
        status_callback=snapshots_seen.append,
    )

    assert snapshot["status"] == "completed"
    assert snapshot["duration_ms"] == 3210
    assert len(requested) == 2
    assert snapshots_seen[0]["status"] == "polling"
    assert "Snapshot request timed out" in snapshots_seen[0]["last_message"]
    assert requested[0][1] > 10


def test_langflow_node_inspector_rejects_nonterminal_wait_snapshot():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])

    assert module.coerce_wait_for_completion(None) is True
    assert module.coerce_wait_for_completion("") is True
    assert module.coerce_wait_for_completion("true") is True
    assert module.coerce_wait_for_completion(False) is False
    assert module.coerce_wait_for_completion("false") is False

    with pytest.raises(TimeoutError, match="source_discovery.*still pending"):
        module.require_terminal_node_snapshot(
            {"status": "pending", "polling_timed_out": True},
            node_name="source_discovery",
            wait_for_completion=True,
        )

    module.require_terminal_node_snapshot(
        {"status": "pending"},
        node_name="source_discovery",
        wait_for_completion=False,
    )


def test_langflow_node_inspector_times_out_without_returning_running_snapshot():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"node_name": "source_discovery", "status": "running"}

    with pytest.raises(TimeoutError, match="source_discovery.*still running"):
        module.fetch_node_snapshot(
            "http://127.0.0.1:8010/",
            "demo_session",
            "source_discovery",
            wait_for_completion=True,
            timeout_seconds=0,
            poll_interval_seconds=1,
            request_get=lambda url, timeout: FakeResponse(),
            sleep=lambda seconds: None,
        )


def test_langflow_node_inspector_prefers_upstream_api_base_url():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])

    upstream = {
        "session_id": "demo_session",
        "api_base_url": "http://127.0.0.1:8011",
        "status_url": "http://127.0.0.1:8011/runs/demo_session",
    }

    assert module.resolve_api_base_url("http://127.0.0.1:8010", upstream) == (
        "http://127.0.0.1:8011"
    )


def test_langflow_node_inspector_prefers_upstream_workflow_session_over_internal_uuid():
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])

    upstream = {"session_id": "flu_virginia_2024_10_01_2024_10_05_20260613_191836_utc"}
    langflow_internal_session = "aaa1d3ec-fec7-584b-8345-a9748690e16d"

    assert module.resolve_session_id(langflow_internal_session, upstream) == upstream["session_id"]


def test_langflow_node_inspector_cache_is_bound_to_current_session(monkeypatch):
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])
    component = module.DataCollectionWorkflowNodeInspector()
    component.api_base_url = "http://127.0.0.1:8010"
    component.node_name = "source_discovery"
    component.previous_node_status = None
    component.wait_for_completion = True
    component.timeout_seconds = 1800
    component.poll_interval_seconds = 1
    calls = []

    def fake_fetch(api_base_url, session_id, node_name, **kwargs):
        calls.append((api_base_url, session_id, node_name))
        return {
            "session_id": session_id,
            "node_name": node_name,
            "status": "completed",
            "duration_ms": 100,
        }

    monkeypatch.setattr(module, "fetch_node_snapshot", fake_fetch)

    component.workflow_session_id = "old_session"
    first = component._inspect_node_once()
    component.workflow_session_id = "new_session"
    second = component._inspect_node_once()

    assert first["session_id"] == "old_session"
    assert second["session_id"] == "new_session"
    assert [call[1] for call in calls] == ["old_session", "new_session"]


def test_langflow_node_inspector_missing_wait_input_still_waits(monkeypatch):
    module = _load_langflow_node_component(Path(__file__).resolve().parents[1])
    component = module.DataCollectionWorkflowNodeInspector()
    component.api_base_url = "http://127.0.0.1:8010"
    component.workflow_session_id = "demo_session"
    component.node_name = "source_discovery"
    component.previous_node_status = None
    component.wait_for_completion = None
    component.timeout_seconds = 1800
    component.poll_interval_seconds = 1
    seen = []

    def fake_fetch(api_base_url, session_id, node_name, **kwargs):
        seen.append(kwargs["wait_for_completion"])
        return {
            "session_id": session_id,
            "node_name": node_name,
            "status": "completed",
        }

    monkeypatch.setattr(module, "fetch_node_snapshot", fake_fetch)

    component._inspect_node_once()

    assert seen == [True]


def test_langflow_deep_links_polls_run_snapshot_until_completed():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    requested = []
    statuses = iter(["running", "completed"])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "session_id": "demo_session",
                "status": next(statuses),
                "artifact_urls": {},
            }

    def fake_get(url, timeout):
        requested.append((url, timeout))
        return FakeResponse()

    snapshot = module.fetch_deep_links(
        "http://127.0.0.1:8010/",
        "demo_session",
        wait_for_completion=True,
        timeout_seconds=456,
        poll_interval_seconds=5,
        request_get=fake_get,
        sleep=lambda seconds: None,
    )

    assert snapshot["status"] == "completed"
    assert requested == [
        (
            "http://127.0.0.1:8010/runs/demo_session"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        ),
        (
            "http://127.0.0.1:8010/runs/demo_session"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        ),
    ]


def test_langflow_deep_links_retries_transient_run_snapshot_timeout():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    requested = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "session_id": "demo_session",
                "status": "completed",
                "node_progress": {"completed_count": 20, "total_count": 20},
            }

    def fake_get(url, timeout):
        requested.append((url, timeout))
        if len(requested) == 1:
            raise module.requests.exceptions.ReadTimeout("read timeout=10")
        return FakeResponse()

    snapshots_seen = []
    snapshot = module.fetch_deep_links(
        "http://127.0.0.1:8010/",
        "demo_session",
        wait_for_completion=True,
        timeout_seconds=30,
        poll_interval_seconds=1,
        request_get=fake_get,
        sleep=lambda seconds: None,
        status_callback=snapshots_seen.append,
    )

    assert snapshot["status"] == "completed"
    assert len(requested) == 2
    assert snapshots_seen[0]["status"] == "polling"
    assert "Run snapshot request timed out" in snapshots_seen[0]["last_message"]
    assert requested[0][1] == module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS


def test_langflow_deep_links_rejects_nonterminal_wait_snapshot():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])

    assert module.coerce_wait_for_completion(None) is True
    assert module.coerce_wait_for_completion("") is True
    assert module.coerce_wait_for_completion("true") is True
    assert module.coerce_wait_for_completion(False) is False
    assert module.coerce_wait_for_completion("false") is False

    with pytest.raises(TimeoutError, match="demo_session.*still running"):
        module.require_terminal_run_snapshot(
            {"status": "running", "polling_timed_out": True},
            session_id="demo_session",
            wait_for_completion=True,
        )

    module.require_terminal_run_snapshot(
        {"status": "running"},
        session_id="demo_session",
        wait_for_completion=False,
    )


def test_langflow_deep_links_times_out_without_returning_running_snapshot():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"session_id": "demo_session", "status": "running", "artifact_urls": {}}

    with pytest.raises(TimeoutError, match="demo_session.*still running"):
        module.fetch_deep_links(
            "http://127.0.0.1:8010/",
            "demo_session",
            wait_for_completion=True,
            timeout_seconds=0,
            poll_interval_seconds=1,
            request_get=lambda url, timeout: FakeResponse(),
            sleep=lambda seconds: None,
        )


def test_langflow_final_results_cache_is_bound_to_current_session(monkeypatch):
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    component = module.WorkflowLinks()
    component.api_base_url = "http://127.0.0.1:8010"
    component.previous_node_status = None
    component.wait_for_completion = True
    component.timeout_seconds = 1800
    component.poll_interval_seconds = 1
    calls = []

    def fake_fetch(api_base_url, session_id, **kwargs):
        calls.append((api_base_url, session_id))
        return {
            "session_id": session_id,
            "status": "completed",
            "node_progress": {"completed_count": 20, "total_count": 20},
        }

    monkeypatch.setattr(module, "fetch_deep_links", fake_fetch)

    component.workflow_session_id = "old_session"
    first = component._get_links_once()
    component.workflow_session_id = "new_session"
    second = component._get_links_once()

    assert first["session_id"] == "old_session"
    assert second["session_id"] == "new_session"
    assert [call[1] for call in calls] == ["old_session", "new_session"]


def test_langflow_final_results_missing_wait_input_still_waits(monkeypatch):
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    component = module.WorkflowLinks()
    component.api_base_url = "http://127.0.0.1:8010"
    component.workflow_session_id = "demo_session"
    component.previous_node_status = None
    component.wait_for_completion = None
    component.timeout_seconds = 1800
    component.poll_interval_seconds = 1
    seen = []

    def fake_fetch(api_base_url, session_id, **kwargs):
        seen.append(kwargs["wait_for_completion"])
        return {"session_id": session_id, "status": "completed"}

    monkeypatch.setattr(module, "fetch_deep_links", fake_fetch)

    component._get_links_once()

    assert seen == [True]


def test_langflow_live_timeline_formats_real_node_durations():
    module = _load_langflow_live_timeline_component(Path(__file__).resolve().parents[1])

    text = module.format_timeline_text(
        {
            "session_id": "demo_session",
            "status": "running",
            "current_node": "source_discovery",
            "node_progress": {"completed_count": 5, "total_count": 20},
            "node_timeline": [
                {"node_name": "task_intake", "status": "completed", "duration_ms": 3},
                {"node_name": "source_discovery", "status": "running", "duration_ms": None},
                {"node_name": "final_data_package_builder", "status": "pending"},
            ],
        }
    )

    assert "Data Collection Live Timeline / Real Node Durations" in text
    assert "Langflow card time is not the real workflow duration" in text
    assert "Current node: `source_discovery`" in text
    assert "Progress: `5/20 completed`" in text
    assert "| task_intake | completed | 3 ms |" in text
    assert "| source_discovery | running | pending |" in text


def test_langflow_live_timeline_fetches_current_snapshot_without_waiting():
    module = _load_langflow_live_timeline_component(Path(__file__).resolve().parents[1])
    requested = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "session_id": "demo_session",
                "status": "running",
                "current_node": "source_discovery",
                "node_progress": {"completed_count": 5, "total_count": 20},
                "node_timeline": [
                    {"node_name": "source_discovery", "status": "running"},
                ],
            }

    def fake_get(url, timeout):
        requested.append((url, timeout))
        return FakeResponse()

    snapshot = module.fetch_timeline_snapshot(
        "http://127.0.0.1:8010/",
        "demo_session",
        wait_for_completion=False,
        timeout_seconds=0,
        request_get=fake_get,
        sleep=lambda seconds: None,
    )

    assert snapshot["status"] == "running"
    assert "source_discovery" in snapshot["timeline_text"]
    assert requested == [
        (
            "http://127.0.0.1:8010/runs/demo_session"
            f"?visual_contract_version={VISUAL_CONTRACT_VERSION}",
            module.SNAPSHOT_REQUEST_TIMEOUT_SECONDS,
        )
    ]


def test_langflow_deep_links_text_surfaces_current_progress():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])

    text = module.format_deep_links_text(
        {
            "session_id": "demo_session",
            "status": "running",
            "current_node": "source_discovery",
            "node_progress": {
                "completed_count": 5,
                "total_count": 20,
                "running_nodes": ["source_discovery"],
                "failed_count": 0,
                "pending_count": 14,
            },
            "node_timeline": [
                {
                    "node_name": "task_intake_and_scope_planning",
                    "status": "completed",
                    "duration_ms": 5,
                },
                {
                    "node_name": "source_discovery",
                    "status": "running",
                    "last_message": "searching",
                },
            ],
            "artifact_urls": {},
            "session_dir": "outputs/sessions/demo_session",
            "generated_config_path": "outputs/generated_configs/demo_session.json",
            "result_location_text": "Session directory: outputs/sessions/demo_session",
        }
    )

    assert "Current node: source_discovery" in text
    assert "Progress: 5/20 completed" in text
    assert "Running: source_discovery" in text
    assert "source_discovery: running" in text
    assert "outputs/sessions/demo_session" in text


def test_langflow_deep_links_prefers_upstream_api_base_url():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])

    upstream = {
        "session_id": "demo_session",
        "events_url": "http://127.0.0.1:8011/runs/demo_session/events",
    }

    assert module.resolve_api_base_url("http://127.0.0.1:8010", upstream) == (
        "http://127.0.0.1:8011"
    )


def test_langflow_deep_links_prefers_upstream_workflow_session_over_internal_uuid():
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])

    upstream = {"session_id": "flu_virginia_2024_10_01_2024_10_05_20260613_191836_utc"}
    langflow_internal_session = "aaa1d3ec-fec7-584b-8345-a9748690e16d"

    assert module.resolve_session_id(langflow_internal_session, upstream) == upstream["session_id"]


def test_langflow_deep_links_component_ignores_langflow_internal_session_id(monkeypatch):
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    captured: dict[str, str] = {}

    def fake_fetch(api_base_url, session_id, **kwargs):
        captured["session_id"] = session_id
        return {"session_id": session_id, "status": "completed", "node_progress": {}}

    monkeypatch.setattr(module, "fetch_deep_links", fake_fetch)

    component = module.WorkflowLinks()
    component.api_base_url = "http://fallback.local"
    component.workflow_session_id = ""
    component.session_id = "aaa1d3ec-fec7-584b-8345-a9748690e16d"
    component.previous_node_status = module.Data(data={"session_id": "real_workflow_session"})
    component.wait_for_completion = False
    component.timeout_seconds = 1
    component.poll_interval_seconds = 1

    component.get_links()

    assert captured["session_id"] == "real_workflow_session"


def test_langflow_deep_links_component_uses_workflow_session_id_fallback(monkeypatch):
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    captured: dict[str, str] = {}

    def fake_fetch(api_base_url, session_id, **kwargs):
        captured["session_id"] = session_id
        return {"session_id": session_id, "status": "completed", "node_progress": {}}

    monkeypatch.setattr(module, "fetch_deep_links", fake_fetch)

    component = module.WorkflowLinks()
    component.api_base_url = "http://fallback.local"
    component.workflow_session_id = "real_workflow_session"
    component.session_id = "aaa1d3ec-fec7-584b-8345-a9748690e16d"
    component.previous_node_status = None
    component.wait_for_completion = False
    component.timeout_seconds = 1
    component.poll_interval_seconds = 1

    component.get_links()

    assert captured["session_id"] == "real_workflow_session"


def test_langflow_demo_optional_extra_and_files_are_declared():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    extra = pyproject["project"]["optional-dependencies"]["langflow-demo"]
    assert "langflow" in extra
    assert "fastapi" in extra
    assert "uvicorn" in extra
    deep_extra = pyproject["project"]["optional-dependencies"]["langflow-deep-demo"]
    assert "langsmith" in deep_extra
    assert "langgraph-cli[inmem]" in deep_extra

    component = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "workflow_runner.py"
    )
    status_component = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "workflow_status.py"
    )
    init_file = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "__init__.py"
    )
    start_script = project_root / "scripts" / "start_langflow_demo.py"
    deep_start_script = project_root / "scripts" / "start_langflow_tracing.py"
    deep_links_component = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "links.py"
    )
    live_timeline_component = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "timeline.py"
    )
    node_component = (
        project_root
        / "integrations"
        / "langflow"
        / "components"
        / "data_collection_workflow"
        / "node_inspector.py"
    )
    deep_flow = (
        project_root
        / "integrations"
        / "langflow"
        / "flows"
        / "workflow.json"
    )

    assert component.exists()
    assert "DataCollectionWorkflowRunner" in component.read_text(encoding="utf-8")
    assert status_component.exists()
    assert "DataCollectionWorkflowStatus" in status_component.read_text(encoding="utf-8")
    assert "DataCollectionWorkflowStatus" in init_file.read_text(encoding="utf-8")
    assert deep_links_component.exists()
    assert "WorkflowLinks" in deep_links_component.read_text(encoding="utf-8")
    assert live_timeline_component.exists()
    assert "WorkflowTimeline" in live_timeline_component.read_text(encoding="utf-8")
    assert "WorkflowTimeline" in init_file.read_text(encoding="utf-8")
    assert node_component.exists()
    assert "DataCollectionWorkflowNodeInspector" in node_component.read_text(encoding="utf-8")
    assert "DataCollectionWorkflowNodeInspector" in init_file.read_text(encoding="utf-8")
    assert deep_start_script.exists()
    assert "LANGSMITH_API_KEY" in deep_start_script.read_text(encoding="utf-8")
    assert deep_flow.exists()
    flow_payload = json.loads(deep_flow.read_text(encoding="utf-8"))
    flow_text = json.dumps(flow_payload)
    assert flow_payload["name"] == "Data Collection Workflow"
    flow_data = flow_payload.get("data")
    assert isinstance(flow_data, dict)
    flow_nodes = flow_data.get("nodes") or []
    flow_edges = flow_data.get("edges") or []
    assert len(flow_nodes) >= 23
    inspector_node_names = [
        node["data"]["node"]["template"]["node_name"]["value"]
        for node in flow_nodes
        if node.get("data", {}).get("type") == "node_inspector"
    ]
    assert inspector_node_names == [
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
    assert len(flow_edges) == 22
    expected_chain = [
        "workflow_runner",
        *[
            f"workflow_node_{index:02d}_{node_name}"
            for index, node_name in enumerate(inspector_node_names, start=1)
        ],
        "workflow_final_results",
    ]
    actual_edge_pairs = [(edge["source"], edge["target"]) for edge in flow_edges]
    expected_chain_pairs = list(zip(expected_chain, expected_chain[1:]))
    assert actual_edge_pairs[: len(expected_chain_pairs)] == expected_chain_pairs
    assert ("workflow_runner", "timeline") in actual_edge_pairs
    inspector_nodes = [
        node
        for node in flow_nodes
        if node.get("data", {}).get("type") == "node_inspector"
    ]
    for node in inspector_nodes:
        node_data = node["data"]["node"]
        template = node_data["template"]
        outputs = node_data["outputs"]
        assert node_data["minimized"] is True
        assert template["api_base_url"]["show"] is False
        assert "session_id" not in template
        assert template["workflow_session_id"]["show"] is False
        assert template["node_name"]["show"] is False
        assert template["previous_node_status"]["show"] is True
        assert template["previous_node_status"]["advanced"] is False
        assert outputs[0]["display_name"] == "Node Status"
        assert outputs[0]["selected"] == "JSON"
        assert any(
            output["display_name"] == "Node Details" and output["selected"] == "Message"
            for output in outputs
        )
    node_by_id = {node["id"]: node for node in flow_nodes}
    for edge in flow_edges:
        target = node_by_id[edge["target"]]
        target_field = edge["data"]["targetHandle"]["fieldName"]
        target_template = target["data"]["node"]["template"]
        assert target_template[target_field]["show"] is True
    assert "task_intake_and_scope_planning" in flow_text
    assert "final_data_package_builder" in flow_text
    assert "Data Collection Final Results - Run Full Workflow" in flow_text
    assert "Data Collection Live Timeline / Real Node Durations" in flow_text
    assert VISUAL_CONTRACT_VERSION in flow_text
    assert "dashboard_port" not in flow_text
    assert "Data Collection Workflow Status" not in flow_text
    runner_nodes = [
        node for node in flow_nodes if node.get("data", {}).get("type") == "workflow_runner"
    ]
    assert len(runner_nodes) == 1
    runner_template = runner_nodes[0]["data"]["node"]["template"]
    assert runner_template["user_request"]["show"] is True
    assert runner_template["api_base_url"]["show"] is True
    final_result_nodes = [
        node for node in flow_nodes if node.get("id") == "workflow_final_results"
    ]
    assert len(final_result_nodes) == 1
    final_result_template = final_result_nodes[0]["data"]["node"]["template"]
    assert "session_id" not in final_result_template
    assert final_result_template["workflow_session_id"]["show"] is False
    assert final_result_template["wait_for_completion"]["value"] is True
    final_result_outputs = final_result_nodes[0]["data"]["node"]["outputs"]
    assert final_result_outputs[0]["display_name"] == "Final Results"
    assert final_result_outputs[0]["selected"] == "Message"
    live_timeline_nodes = [
        node for node in flow_nodes if node.get("id") == "timeline"
    ]
    assert len(live_timeline_nodes) == 1
    live_timeline_template = live_timeline_nodes[0]["data"]["node"]["template"]
    assert "session_id" not in live_timeline_template
    assert live_timeline_template["workflow_session_id"]["show"] is False
    assert live_timeline_template["previous_node_status"]["show"] is True
    assert live_timeline_template["wait_for_completion"]["value"] is False
    live_timeline_outputs = live_timeline_nodes[0]["data"]["node"]["outputs"]
    assert live_timeline_outputs[0]["display_name"] == "Timeline Report"
    assert live_timeline_outputs[0]["selected"] == "Message"
    assert start_script.exists()
    assert "LANGFLOW_COMPONENTS_PATH" in start_script.read_text(encoding="utf-8")
    assert "dashboard_port" not in start_script.read_text(encoding="utf-8")
    visual_script = project_root / "scripts" / "run_visual_workflow.py"
    assert visual_script.exists()
    assert "run_visual_workflow" in visual_script.read_text(encoding="utf-8")


def test_langflow_demo_is_documented():
    project_root = Path(__file__).resolve().parents[1]
    readme = (project_root / "README.md").read_text(encoding="utf-8")
    code_map = (project_root / "docs" / "code_map.md").read_text(encoding="utf-8")

    for interface in ("Langflow", "Studio", "dashboard"):
        assert interface in readme
        assert interface in code_map
    assert "optional" in readme.lower()
    assert "local API" in code_map
    assert "terminal runner" in code_map
    assert "Static visualizations" in code_map
    assert "standard graph" in code_map


def _load_start_langflow_demo_module(project_root: Path):
    module_path = project_root / "scripts" / "start_langflow_demo.py"
    spec = importlib.util.spec_from_file_location("start_langflow_demo_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_visual_workflow_module(project_root: Path):
    module_path = project_root / "scripts" / "run_visual_workflow.py"
    scripts_path = str(module_path.parent)
    spec = importlib.util.spec_from_file_location("run_visual_workflow_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, scripts_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts_path)
    return module


def test_run_visual_workflow_collects_inputs_without_quick_mode_prompt(monkeypatch):
    visual = _load_run_visual_workflow_module(Path(__file__).resolve().parents[1])
    prompts: list[str] = []
    answers = iter(["FLU", "VIRGINIA", "2024-10-1", "2024-10-5", "", ""])

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(visual.sys.stdin, "isatty", lambda: True)

    collected = visual._collect_inputs(visual.build_parser().parse_args([]))

    assert collected["disease"] == "FLU"
    assert collected["location"] == "VIRGINIA"
    assert collected["start_date"] == "2024-10-01"
    assert collected["end_date"] == "2024-10-05"
    assert str(collected["session_id"]).startswith("flu_virginia_2024_10_01_2024_10_05")
    assert collected["quick_test_mode"] is False
    assert not any("Quick Test Mode" in prompt for prompt in prompts)


def test_start_langflow_demo_waits_for_ready_and_patches_flow_payload():
    start_demo = _load_start_langflow_demo_module(Path(__file__).resolve().parents[1])

    attempts = []

    class FakeResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"status {self.status_code}")

    def fake_get(url, timeout):
        attempts.append((url, timeout))
        if len(attempts) == 1:
            raise OSError("not ready")
        return FakeResponse(200)

    assert start_demo._wait_for_http_ready(
        "http://127.0.0.1:7860/health_check",
        timeout_seconds=1,
        poll_interval_seconds=0.001,
        request_get=fake_get,
        sleep=lambda seconds: None,
    )
    assert len(attempts) == 2

    flow = start_demo._patched_flow_payload(
        api_base_url="http://127.0.0.1:8017",
        runner_defaults={
            "disease": "flu",
            "location": "New York",
            "start_date": "2024-05-01",
            "end_date": "2024-05-03",
            "quick_test_mode": True,
        },
    )
    flow_text = json.dumps(flow)
    assert "http://127.0.0.1:8017" in flow_text
    assert "2024-05-01" in flow_text
    assert "quick_test_mode" in flow_text


def test_start_langflow_demo_uses_session_specific_flow_id_and_name():
    start_demo = _load_start_langflow_demo_module(Path(__file__).resolve().parents[1])
    runner_defaults = {
        "session_id": "flu_virginia_2024_10_01_2024_10_05_20260613_185749_utc",
        "disease": "FLU",
        "location": "VIRGINIA",
        "start_date": "2024-10-01",
        "end_date": "2024-10-05",
    }

    flow_id = start_demo._flow_id_from_runner_defaults(runner_defaults)
    flow = start_demo._patched_flow_payload(
        api_base_url="http://127.0.0.1:8010",
        runner_defaults=runner_defaults,
        flow_id=flow_id,
    )
    flow_text = json.dumps(flow)

    assert flow_id != start_demo.VISUAL_FLOW_ID
    assert flow["id"] == flow_id
    assert flow["name"].startswith("Data Collection")
    assert "v2" not in flow["name"].lower()
    assert "v3" not in flow["name"].lower()
    assert flow["name"].endswith(runner_defaults["session_id"])
    assert "VIRGINIA" in flow_text
    assert "2024-10-01" in flow_text


def test_start_langflow_demo_uploads_flow_and_returns_open_url():
    start_demo = _load_start_langflow_demo_module(Path(__file__).resolve().parents[1])

    calls = []

    class FakeResponse:
        def __init__(self, status_code=201, payload=None):
            self.status_code = status_code
            self._payload = payload or [{"id": "flow-123"}]

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, timeout):
            calls.append(("GET", url, timeout))
            return FakeResponse(payload={"access_token": "demo-token"})

        def post(self, url, files, timeout):
            calls.append(("POST", url, files, timeout))
            return FakeResponse()

    result = start_demo._upload_flow_to_langflow(
        "http://127.0.0.1:7860",
        {"name": "Data Collection Visual", "data": {"nodes": [], "edges": []}},
        request_session_factory=FakeSession,
    )

    assert result["flow_id"] == "flow-123"
    assert result["flow_url"] == "http://127.0.0.1:7860/flow/flow-123"
    assert calls[0] == ("GET", "http://127.0.0.1:7860/api/v1/auto_login", 30)
    assert calls[1][0] == "POST"
    assert calls[1][1] == "http://127.0.0.1:7860/api/v1/flows/upload/"
