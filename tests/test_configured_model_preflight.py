"""Offline readiness ordering for budgeted evidence model preflight."""
from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.run_workflow as runner
from data_collection_workflow import llm_clients, document_acquisition
from data_collection_workflow.session_runtime import get_runtime


class GraphReached(Exception):
    pass


@pytest.fixture
def configured_run(monkeypatch, tmp_path):
    config = {
        "pipeline_mode": "evidence",
        "structured_task": {"disease": "mpox", "location": "United States",
                            "start_date": "2025", "end_date": "2025"},
        "llm": {"provider": "anthropic", "model": "claude-selected-for-run",
                "source_planning_enabled": True, "source_critic_enabled": False,
                "structured_extraction_enabled": False, "source_identity": {"enabled": False}},
        "output": {"session_id": "run", "write_latest_alias": False},
    }
    events = []
    monkeypatch.setattr(runner, "_config_with_cli_overrides", lambda args: (Path("unused.json"), config))
    monkeypatch.setattr(runner, "workflow_output_dir_from_config", lambda cfg: tmp_path / "run")

    def ready(policy, directory):
        events.append("browser_ocr")
        return {"ready": True}

    def preflight(settings=None):
        context = get_runtime()
        assert context is not None
        assert context.readiness["ready"] is True
        assert os.environ["LLM_PROVIDER"] == "anthropic"
        assert os.environ["LLM_MODEL"] == "claude-selected-for-run"
        events.append("model")
        return {"status": "ok", "provider": "anthropic", "model": "claude-selected-for-run"}

    def graph(*args, **kwargs):
        events.append("graph")
        raise GraphReached

    monkeypatch.setattr(document_acquisition, "preflight_acquisition", ready)
    monkeypatch.setattr(llm_clients, "preflight_llm_model", preflight)
    monkeypatch.setattr(runner, "_run_graph_with_events", graph)
    return config, events, Namespace(resume_session=None, live_status=False)


def test_evidence_runner_preflights_model_after_local_readiness_and_before_graph(configured_run):
    config, events, args = configured_run
    with pytest.raises(GraphReached):
        runner.run_workflow(args)
    assert events == ["browser_ocr", "model", "graph"]


def test_evidence_runner_rejects_unavailable_model_before_graph(configured_run, monkeypatch):
    config, events, args = configured_run

    def rejected(settings=None):
        events.append("model_failed")
        raise ValueError("selected model unavailable")

    monkeypatch.setattr(llm_clients, "preflight_llm_model", rejected)
    with pytest.raises(ValueError, match="selected model unavailable"):
        runner.run_workflow(args)
    assert events == ["browser_ocr", "model_failed"]


def test_evidence_runner_does_not_preflight_model_when_browser_ocr_readiness_fails(configured_run, monkeypatch):
    config, events, args = configured_run

    def not_ready(policy, directory):
        events.append("browser_ocr_failed")
        raise RuntimeError("browser missing")

    monkeypatch.setattr(document_acquisition, "preflight_acquisition", not_ready)
    with pytest.raises(RuntimeError, match="browser missing"):
        runner.run_workflow(args)
    assert events == ["browser_ocr_failed"]


def test_evidence_runner_skips_model_preflight_when_all_llm_stages_are_disabled(configured_run):
    config, events, args = configured_run
    config["llm"]["source_planning_enabled"] = False
    config["disease_intelligence"] = {"llm_enabled": False}
    with pytest.raises(GraphReached):
        runner.run_workflow(args)
    assert events == ["browser_ocr", "graph"]


@pytest.mark.parametrize("trace_setting", [None, "false", "true"])
@pytest.mark.parametrize("preflight_fails", [False, True])
def test_evidence_preflight_respects_explicit_trace_policy_and_restores_environment(
    configured_run, monkeypatch, trace_setting, preflight_fails,
):
    config, events, args = configured_run
    original = {
        "LANGSMITH_API_KEY": "dummy-langsmith-key", "LANGCHAIN_API_KEY": "dummy-langchain-key",
        "LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_TRACING": "true",
    }
    for name, value in original.items():
        monkeypatch.setenv(name, value)
    if trace_setting is None:
        monkeypatch.delenv("ENABLE_LANGSMITH_TRACE", raising=False)
    else:
        monkeypatch.setenv("ENABLE_LANGSMITH_TRACE", trace_setting)
    observed = {}

    def preflight():
        observed.update({name: os.environ.get(name) for name in original})
        if preflight_fails:
            raise ValueError("offline model rejection")
        return {"status": "ok"}

    monkeypatch.setattr(llm_clients, "preflight_llm_model", preflight)
    monkeypatch.setattr(runner, "_flush_langsmith_tracers", lambda: None)
    with pytest.raises(ValueError if preflight_fails else GraphReached):
        runner.run_workflow(args)
    expected = original if trace_setting == "true" else {
        "LANGSMITH_API_KEY": None, "LANGCHAIN_API_KEY": None,
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "LANGCHAIN_TRACING": "false",
    }
    assert observed == expected
    assert {name: os.environ.get(name) for name in original} == original
