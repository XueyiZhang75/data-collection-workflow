"""Offline model-selection regressions for interactive and configured runs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import scripts.collect as interactive
from data_collection_workflow.runtime_profile import temporary_workflow_env, workflow_run_env_from_config

_SETTING_KEYS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_THINKING",
    "LLM_EFFORT", "LLM_STRUCTURED_OUTPUT_METHOD",
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    for name in _SETTING_KEYS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("HDC_" + name, raising=False)
    monkeypatch.setattr(interactive, "PROJECT_ROOT", tmp_path)

    def forbid_live_call(*args, **kwargs):
        raise AssertionError("configuration preview must not run the workflow or preflight")

    monkeypatch.setattr(interactive, "run_workflow", forbid_live_call)
    monkeypatch.setattr(interactive.llm_clients, "preflight_llm_model", forbid_live_call)


def preview(capsys, *options):
    result = interactive.main([
        "--disease", "mpox", "--location", "United States",
        "--start-date", "2025", "--end-date", "2025",
        "--session-id", "model_settings_preview", "--print-config-only", *options,
    ])
    assert result == 0
    output = capsys.readouterr().out
    return json.loads(output.split("sanitized_config_json:", 1)[1])["config"]


def test_interactive_honors_environment_model_instead_of_parser_default(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "requested-model-from-shell")
    config = preview(capsys)
    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["model"] == "requested-model-from-shell"


def test_interactive_reads_project_dotenv_without_overriding_shell(monkeypatch, tmp_path, capsys):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=anthropic\nLLM_MODEL=claude-from-dotenv\n",
        encoding="utf-8",
    )
    assert preview(capsys)["llm"]["model"] == "claude-from-dotenv"
    monkeypatch.setenv("LLM_MODEL", "claude-from-shell")
    assert preview(capsys)["llm"]["model"] == "claude-from-shell"


def test_explicit_interactive_model_and_provider_win_over_environment(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    config = preview(capsys, "--provider", "anthropic", "--model", "claude-explicit")
    assert config["llm"]["provider"] == "anthropic"
    assert config["llm"]["model"] == "claude-explicit"


def test_interactive_preview_does_not_select_a_model_for_the_user(capsys):
    config = preview(capsys)
    assert config["llm"]["provider"] == "anthropic"
    assert config["llm"]["model"] == ""


def test_interactive_generation_options_reach_run_environment(monkeypatch, capsys):
    monkeypatch.setenv("LLM_THINKING", "disabled")
    monkeypatch.setenv("LLM_EFFORT", "low")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_prompt")
    config = preview(capsys, "--model", "claude-explicit", "--llm-thinking", "adaptive",
                     "--llm-effort", "high", "--llm-output-mode", "json_schema")
    assert config["llm"]["thinking"] == "adaptive"
    assert config["llm"]["effort"] == "high"
    assert config["llm"]["structured_output_method"] == "json_schema"
    updates = workflow_run_env_from_config(config)
    assert updates["LLM_THINKING"] == "adaptive"
    assert updates["LLM_EFFORT"] == "high"
    assert updates["LLM_STRUCTURED_OUTPUT_METHOD"] == "json_schema"


def test_interactive_generation_options_inherit_environment(monkeypatch, capsys):
    monkeypatch.setenv("LLM_THINKING", "disabled")
    monkeypatch.setenv("LLM_EFFORT", "medium")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "function_calling")
    config = preview(capsys)
    assert config["llm"]["thinking"] == "disabled"
    assert config["llm"]["effort"] == "medium"
    assert config["llm"]["structured_output_method"] == "function_calling"


def test_configured_generation_options_override_environment_and_restore_it(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "disabled")
    monkeypatch.setenv("LLM_EFFORT", "low")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_prompt")
    updates = workflow_run_env_from_config({"llm": {
        "thinking": "adaptive", "effort": "max", "structured_output_method": "json_schema",
    }})
    with temporary_workflow_env(updates):
        assert os.environ["LLM_THINKING"] == "adaptive"
        assert os.environ["LLM_EFFORT"] == "max"
        assert os.environ["LLM_STRUCTURED_OUTPUT_METHOD"] == "json_schema"
    assert os.environ["LLM_THINKING"] == "disabled"
    assert os.environ["LLM_EFFORT"] == "low"
    assert os.environ["LLM_STRUCTURED_OUTPUT_METHOD"] == "json_prompt"


def test_configured_generation_options_inherit_environment_when_absent(monkeypatch):
    monkeypatch.setenv("LLM_THINKING", "adaptive")
    monkeypatch.setenv("LLM_EFFORT", "xhigh")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_schema")
    updates = workflow_run_env_from_config({"llm": {}})
    assert updates["LLM_THINKING"] == "adaptive"
    assert updates["LLM_EFFORT"] == "xhigh"
    assert updates["LLM_STRUCTURED_OUTPUT_METHOD"] == "json_schema"


def test_generation_options_have_auto_defaults_and_do_not_leak():
    updates = workflow_run_env_from_config({"llm": {}})
    assert updates["LLM_THINKING"] == "auto"
    assert updates["LLM_EFFORT"] == ""
    assert updates["LLM_STRUCTURED_OUTPUT_METHOD"] == "auto"
    with temporary_workflow_env(updates):
        assert os.environ["LLM_STRUCTURED_OUTPUT_METHOD"] == "auto"
    assert "LLM_THINKING" not in os.environ
    assert "LLM_EFFORT" not in os.environ
    assert "LLM_STRUCTURED_OUTPUT_METHOD" not in os.environ


@pytest.mark.parametrize("env_name,first,second", [
    ("LLM_THINKING", "disabled", "adaptive"),
    ("LLM_EFFORT", "low", "high"),
    ("LLM_STRUCTURED_OUTPUT_METHOD", "json_prompt", "json_schema"),
])
def test_inherited_generation_option_changes_cannot_resume_same_session(
    monkeypatch, tmp_path, env_name, first, second,
):
    from data_collection_workflow.runtime_profile import load_workflow_run_config
    from data_collection_workflow.session_runtime import RunContext, ResumeMismatch

    config_path = tmp_path / "run.json"
    config_path.write_text('{"pipeline_mode": "evidence", "llm": {}}', encoding="utf-8")
    monkeypatch.setenv(env_name, first)
    config = load_workflow_run_config(config_path)
    RunContext(tmp_path / "session", config)
    monkeypatch.setenv(env_name, second)
    changed = load_workflow_run_config(config_path)
    with pytest.raises(ResumeMismatch, match="fingerprint mismatch"):
        RunContext(tmp_path / "session", changed, resume=True)


def test_explicit_config_generation_options_stay_fixed_when_environment_changes(monkeypatch, tmp_path):
    from data_collection_workflow.runtime_profile import load_workflow_run_config

    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps({"llm": {
        "thinking": "disabled", "effort": "low", "structured_output_method": "json_prompt",
    }}), encoding="utf-8")
    monkeypatch.setenv("LLM_THINKING", "adaptive")
    monkeypatch.setenv("LLM_EFFORT", "high")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_schema")
    config = load_workflow_run_config(config_path)
    updates = workflow_run_env_from_config(config)
    assert updates["LLM_THINKING"] == "disabled"
    assert updates["LLM_EFFORT"] == "low"
    assert updates["LLM_STRUCTURED_OUTPUT_METHOD"] == "json_prompt"


def test_legacy_preflight_uses_generated_generation_settings_and_token_limit(monkeypatch, capsys):
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setenv("LLM_MAX_TOKENS", "111")
    monkeypatch.setenv("LLM_THINKING", "disabled")
    monkeypatch.setenv("LLM_EFFORT", "low")
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_METHOD", "json_prompt")
    observed = {}

    def preflight(settings=None):
        observed.update({key: os.environ.get(key) for key in (
            "LLM_MODEL", "LLM_THINKING", "LLM_EFFORT",
            "LLM_STRUCTURED_OUTPUT_METHOD", "LLM_MAX_TOKENS",
        )})
        raise ValueError("stop after offline preflight inspection")

    monkeypatch.setattr(interactive.llm_clients, "preflight_llm_model", preflight)
    result = interactive.main([
        "--disease", "mpox", "--location", "United States",
        "--start-date", "2025", "--end-date", "2025", "--session-id", "preflight_settings",
        "--model", "claude-selected", "--llm-thinking", "adaptive",
        "--llm-effort", "high", "--llm-output-mode", "json_schema",
    ])
    assert result != 0
    assert observed == {
        "LLM_MODEL": "claude-selected", "LLM_THINKING": "adaptive",
        "LLM_EFFORT": "high", "LLM_STRUCTURED_OUTPUT_METHOD": "json_schema",
        "LLM_MAX_TOKENS": "8192",
    }
    assert os.environ["LLM_MAX_TOKENS"] == "111"
    assert os.environ["LLM_THINKING"] == "disabled"


def test_temporary_workflow_environment_restores_every_applied_key(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "25")
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    with temporary_workflow_env({"LLM_TIMEOUT_SECONDS": "100", "LLM_MAX_RETRIES": "0"}):
        assert os.environ["LLM_TIMEOUT_SECONDS"] == "100"
        assert os.environ["LLM_MAX_RETRIES"] == "0"
    assert os.environ["LLM_TIMEOUT_SECONDS"] == "25"
    assert "LLM_MAX_RETRIES" not in os.environ


@pytest.mark.parametrize("trace_setting", [None, "false", "true"])
@pytest.mark.parametrize("preflight_fails", [False, True])
def test_legacy_preflight_respects_explicit_trace_policy_and_restores_environment(
    monkeypatch, capsys, tmp_path, trace_setting, preflight_fails,
):
    import run_workflow as configured_runner

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
        return {"status": "ok", "provider": "anthropic", "model": "claude-selected"}

    monkeypatch.setattr(interactive.llm_clients, "preflight_llm_model", preflight)
    monkeypatch.setattr(configured_runner, "_flush_langsmith_tracers", lambda: None)
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_write_generated_config", lambda *args: tmp_path / "unused.json")
    monkeypatch.setattr(interactive, "run_workflow", lambda args: {})
    result = interactive.main([
        "--disease", "mpox", "--location", "United States",
        "--start-date", "2025", "--end-date", "2025", "--session-id", "trace_settings",
        "--model", "claude-selected", "--no-dashboard",
    ])
    assert result == (2 if preflight_fails else 0)
    expected = original if trace_setting == "true" else {
        "LANGSMITH_API_KEY": None, "LANGCHAIN_API_KEY": None,
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "LANGCHAIN_TRACING": "false",
    }
    assert observed == expected
    assert {name: os.environ.get(name) for name in original} == original


def test_explicit_null_effort_clears_inherited_effort_in_effective_settings(monkeypatch, tmp_path):
    from data_collection_workflow.runtime_profile import load_workflow_run_config

    config_path = tmp_path / "run.json"
    config_path.write_text('{"llm": {"effort": null}}', encoding="utf-8")
    monkeypatch.setenv("LLM_EFFORT", "high")
    config = load_workflow_run_config(config_path)
    assert config["llm"]["effort"] is None
    updates = workflow_run_env_from_config(config)
    assert updates["LLM_EFFORT"] == ""
    with temporary_workflow_env(updates):
        assert interactive.llm_clients.get_llm_settings()["effort"] is None
    assert os.environ["LLM_EFFORT"] == "high"


def test_explicit_null_effort_keeps_resume_fingerprint_independent_of_shell(monkeypatch, tmp_path):
    from data_collection_workflow.runtime_profile import load_workflow_run_config
    from data_collection_workflow.session_runtime import RunContext

    config_path = tmp_path / "run.json"
    config_path.write_text('{"pipeline_mode": "evidence", "llm": {"effort": null}}', encoding="utf-8")
    monkeypatch.setenv("LLM_EFFORT", "high")
    original = RunContext(tmp_path / "session", load_workflow_run_config(config_path))
    monkeypatch.setenv("LLM_EFFORT", "low")
    resumed = RunContext(tmp_path / "session", load_workflow_run_config(config_path), resume=True)
    assert resumed.fingerprint == original.fingerprint


@pytest.mark.parametrize("selection", ["process", "cli", "dotenv", "default"])
def test_evidence_startup_displays_effective_model_and_selection_source(monkeypatch, tmp_path, capsys, selection):
    old_model = "claude-3-5-sonnet-20241022"
    options = []
    expected_model = interactive.DEFAULT_MODEL
    expected_source = "project default"
    if selection in {"process", "cli"}:
        monkeypatch.setenv("LLM_MODEL", old_model)
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        expected_model, expected_source = old_model, "process environment LLM_MODEL"
    if selection == "cli":
        options = ["--model", "claude-explicit-selection", "--provider", "anthropic"]
        expected_model, expected_source = "claude-explicit-selection", "--model"
    if selection == "dotenv":
        (tmp_path / ".env").write_text("LLM_MODEL=claude-project-selection\nLLM_PROVIDER=anthropic\n", encoding="utf-8")
        expected_model, expected_source = "claude-project-selection", "project .env"
    captured_config = {}
    def preserve(config, session_id):
        captured_config.update(config)
        return tmp_path / "generated.json"
    def run(args):
        before_run = capsys.readouterr().out
        assert f"llm_model: {expected_model} (source: {expected_source}" in before_run
        assert "llm_provider: anthropic (source:" in before_run
        return {}
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_write_generated_config", preserve)
    monkeypatch.setattr(interactive, "run_workflow", run)
    result = interactive.main([
        "--pipeline-mode", "evidence", "--disease", "mpox",
        "--location", "Sierra Leone", "--start-date", "2025", "--end-date", "2025",
        "--session-id", "source_display", "--no-dashboard", *options,
    ])
    if selection == "default":
        assert result == 2
        assert not captured_config
        assert "--model" in capsys.readouterr().err
        return
    assert result == 0
    assert captured_config["llm"]["model"] == expected_model


def test_evidence_preflight_failure_returns_actionable_error_without_retry(monkeypatch, tmp_path, capsys):
    assert hasattr(interactive.llm_clients, "LLMModelPreflightError"), "Preflight needs its own ValueError-compatible error type"
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_write_generated_config", lambda *args: tmp_path / "generated.json")
    calls = []
    def unavailable(args):
        calls.append(args)
        raise interactive.llm_clients.LLMModelPreflightError("provider='anthropic', model='claude-3-5-sonnet-20241022': model not found")
    monkeypatch.setattr(interactive, "run_workflow", unavailable)
    result = interactive.main([
        "--pipeline-mode", "evidence", "--disease", "mpox",
        "--location", "Sierra Leone", "--start-date", "2025", "--end-date", "2025",
        "--session-id", "unavailable_model", "--no-dashboard",
    ])
    captured = capsys.readouterr()
    assert result == 2
    assert len(calls) == 1
    assert "claude-3-5-sonnet-20241022" in captured.err
    assert "--model" in captured.err
    assert "process environment LLM_MODEL" in captured.out
    assert "Traceback" not in captured.err


def test_evidence_other_value_error_still_propagates(monkeypatch, tmp_path):
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_write_generated_config", lambda *args: tmp_path / "generated.json")
    def unrelated_failure(args):
        raise ValueError("unrelated graph invariant")
    monkeypatch.setattr(interactive, "run_workflow", unrelated_failure)
    with pytest.raises(ValueError, match="unrelated graph invariant"):
        interactive.main([
            "--pipeline-mode", "evidence", "--disease", "mpox",
            "--location", "Sierra Leone", "--start-date", "2025", "--end-date", "2025",
            "--session-id", "graph_failure", "--no-dashboard", "--model", "offline-test-model",
        ])
