"""Public model selection is explicit, provider-scoped, and entirely offline here."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from data_collection_workflow import cli, runtime_profile as profile
from data_collection_workflow.session_runtime import fingerprint


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    for name in (
        "LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "LLM_THINKING", "LLM_EFFORT", "LLM_STRUCTURED_OUTPUT_METHOD",
    ):
        # Register restoration even when dotenv later creates a previously absent key.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(profile, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(profile, "DEFAULT_SEARCH_FIXTURE_PATH", tmp_path / "fixtures/search.json")
    monkeypatch.setattr(profile, "DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(profile, "DEFAULT_CONSOLE_OUTPUT_ROOT", tmp_path / "outputs/console")


def config_file(tmp_path, llm=None):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps({
        "structured_task": {"disease": "dengue", "location": "Example region",
                            "start_date": "2025-01-01", "end_date": "2025-12-31"},
        "llm": llm or {},
    }), encoding="utf-8")
    return path


def test_public_defaults_do_not_select_a_model():
    raw = profile.default_workflow_run_config()
    assert profile.DEFAULT_MODEL == ""
    assert raw["llm"]["provider"] == ""
    assert raw["llm"]["model"] == ""
    loaded = profile.load_workflow_run_config()
    assert loaded["llm"]["provider"] == "anthropic"
    assert loaded["llm"]["model"] == ""


@pytest.mark.parametrize("provider,model,env_provider,env_model,expected", [
    (None, None, None, None, ("anthropic", "")),
    (None, None, " OpenAI ", " user-model ", ("openai", "user-model")),
    (" OPENAI ", " custom-Future-ID ", "anthropic", "old-model", ("openai", "custom-Future-ID")),
    ("openai", "", "openai", "matching-model", ("openai", "matching-model")),
    ("openai", None, "anthropic", "wrong-provider-model", ("openai", "")),
    ("anthropic", None, "openai", "wrong-provider-model", ("anthropic", "")),
    ("openai", None, None, "unscoped-user-model", ("openai", "unscoped-user-model")),
])
def test_selection_normalizes_provider_and_only_inherits_matching_environment_model(
    monkeypatch, provider, model, env_provider, env_model, expected,
):
    if env_provider is not None:
        monkeypatch.setenv("LLM_PROVIDER", env_provider)
    if env_model is not None:
        monkeypatch.setenv("LLM_MODEL", env_model)
    assert profile.resolve_llm_selection(provider, model) == expected


def test_project_dotenv_is_loaded_but_shell_takes_precedence(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=openai\nLLM_MODEL=dotenv-model\nOPENAI_API_KEY=offline-placeholder\n",
        encoding="utf-8",
    )
    path = config_file(tmp_path)
    loaded = profile.load_workflow_run_config(path)
    assert (loaded["llm"]["provider"], loaded["llm"]["model"]) == ("openai", "dotenv-model")
    assert "offline-placeholder" not in json.dumps(loaded)
    assert profile.api_key_present("openai")
    monkeypatch.setenv("LLM_MODEL", "shell-model")
    assert profile.load_workflow_run_config(path)["llm"]["model"] == "shell-model"


def test_cli_config_environment_precedence_and_model_id_preservation(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    path = config_file(tmp_path, {"provider": "anthropic", "model": "config-model"})
    loaded = profile.load_workflow_run_config(path)
    assert (loaded["llm"]["provider"], loaded["llm"]["model"]) == ("anthropic", "config-model")
    overridden = cli._apply_cli_config_overrides(
        loaded, Namespace(provider="OPENAI", model="custom-Future-ID"),
    )
    assert (overridden["llm"]["provider"], overridden["llm"]["model"]) == ("openai", "custom-Future-ID")
    updates = profile.workflow_run_env_from_config(overridden)
    assert (updates["LLM_PROVIDER"], updates["LLM_MODEL"]) == ("openai", "custom-Future-ID")


@pytest.mark.parametrize("env_provider,expected", [("anthropic", ""), ("openai", "environment-model")])
def test_provider_switch_clears_previous_provider_model(monkeypatch, env_provider, expected):
    monkeypatch.setenv("LLM_PROVIDER", env_provider)
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    original = {"llm": {"provider": "anthropic", "model": "previous-provider-model"}}
    updated = profile.workflow_run_config_with_overrides(original, provider=" OpenAI ")
    assert updated["llm"]["provider"] == "openai"
    assert updated["llm"]["model"] == expected
    assert original["llm"]["model"] == "previous-provider-model"


def test_same_provider_override_keeps_configured_model():
    config = {"llm": {"provider": "openai", "model": "configured-model"}}
    assert profile.workflow_run_config_with_overrides(config, provider=" OPENAI ")["llm"]["model"] == "configured-model"


def test_missing_model_is_explicitly_cleared_in_runtime_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "old-provider-model")
    updates = profile.workflow_run_env_from_config({"llm": {"provider": "openai", "model": ""}})
    assert updates["LLM_MODEL"] == ""
    with profile.temporary_workflow_env(updates):
        from data_collection_workflow.llm_clients import get_llm_settings
        assert get_llm_settings()["model"] == ""


def test_environment_model_is_materialized_before_session_fingerprint(monkeypatch, tmp_path):
    path = config_file(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "first-model")
    first = profile.load_workflow_run_config(path)
    monkeypatch.setenv("LLM_MODEL", "second-model")
    second = profile.load_workflow_run_config(path)
    assert first["llm"]["model"] == "first-model"
    assert second["llm"]["model"] == "second-model"
    assert fingerprint(first) != fingerprint(second)
    assert profile.workflow_run_env_from_config(first)["LLM_MODEL"] == "first-model"


@pytest.mark.parametrize("identity_only", [False, True])
def test_cli_rejects_missing_model_before_runner(monkeypatch, tmp_path, capsys, identity_only):
    llm = {"provider": "openai", "model": "", "source_planning_enabled": not identity_only,
           "source_critic_enabled": False, "structured_extraction_enabled": False,
           "source_identity": {"enabled": identity_only}}
    path = config_file(tmp_path, llm)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["disease_intelligence"] = {"llm_enabled": False}
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(cli, "_call_configured_runner", lambda *a, **k: pytest.fail("runner must not start"))
    assert cli.main(["collect", "--config", str(path)]) == 3
    assert "model" in capsys.readouterr().err.lower()


def test_direct_runner_rejects_missing_model_before_runtime_or_external_work(monkeypatch, tmp_path):
    from scripts import run_workflow as runner
    from data_collection_workflow import session_runtime

    path = config_file(tmp_path, {"provider": "openai", "model": ""})
    config = profile.load_workflow_run_config(path)
    monkeypatch.setattr(runner, "_config_with_cli_overrides", lambda args: (path, config))
    monkeypatch.setattr(session_runtime, "initialize_universal_run", lambda *a, **k: pytest.fail("runtime must not start"))
    monkeypatch.setattr(runner, "_run_graph_with_events", lambda *a, **k: pytest.fail("graph must not start"))
    with pytest.raises(ValueError, match="[Mm]odel"):
        runner.run_workflow(Namespace(resume_session=None))


def test_preview_remains_available_without_a_model(tmp_path, capsys):
    path = config_file(tmp_path)
    assert cli.main(["collect", "--config", str(path), "--dry-run"]) == 0
    assert "graph_invoked: false" in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["offline", "fixture-search", "live-search"])
def test_generated_template_keeps_llm_optional(tmp_path, mode):
    path = tmp_path / "offline.jsonc"
    args = [
        "init-config", "--disease", "dengue", "--location", "Example region",
        "--start-date", "2025-01-01", "--end-date", "2025-12-31",
        "--mode", mode, "--output", str(path),
    ]
    if mode == "fixture-search":
        search_path = tmp_path / "search.json"
        search_path.write_text(json.dumps({"queries": []}), encoding="utf-8")
        args.extend(["--search-fixture-path", str(search_path)])
    assert cli.main(args) == 0
    config = profile.load_workflow_run_config(path)
    assert config["llm"]["model"] == ""
    updates = profile.workflow_run_env_from_config(config)
    assert not any(
        value == "true" for name, value in updates.items()
        if name.startswith("ENABLE_LLM_")
    )
