"""Behavioral checks for portable, provenance-aware settings inventories."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


@pytest.fixture
def build():
    def call(**kwargs):
        name = "data_collection_workflow.reporting.run_settings"
        assert importlib.util.find_spec(name) is not None, "The reusable settings inventory is not implemented"
        return importlib.import_module(name).build_run_settings(**kwargs)
    return call


def rows(payload):
    return {row["key"]: row for group in payload["groups"] for row in group["rows"]}


def leaves(value, prefix=""):
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict) and item:
            yield from leaves(item, name)
        else:
            yield name, item


def test_unrecorded_values_never_inherit_current_defaults(build):
    result = build()
    inventory = rows(result)
    assert inventory["source_search.max_queries"]["effective_display"] == "Not recorded"
    assert inventory["source_search.max_queries"]["value_status"] == "not_recorded"
    assert inventory["source_search.max_queries"]["current_default"] == 8
    assert inventory["source_search.max_queries"]["default_status"] == "current_reference_only"
    assert inventory["llm.model"]["effective_value"] is None
    assert result["counts"]["total"] == len(inventory)
    assert len(result["groups"]) == 11
    assert "mpox" not in json.dumps(result).lower()


def test_covers_complete_public_config_and_new_nested_controls(build):
    from data_collection_workflow.runtime_profile import default_workflow_run_config

    config = default_workflow_run_config()
    config["structured_task"] = {"disease": "Dengue", "location": "Peru", "start_date": "2021-01-01", "end_date": "2021-03-31"}
    config["new_stage"] = {"nested": {"cap": 19, "options": ["a", "b"]}, "empty": {}}
    result = build(config=config)
    inventory = rows(result)
    for key, value in leaves(config):
        assert key in inventory, key
        assert inventory[key]["configured_value"] == value, key
    assert inventory["new_stage.nested.cap"]["current_default_display"] == "No fixed default recorded"
    assert inventory["structured_task.disease"]["effective_value"] == "Dengue"
    assert inventory["structured_task.location"]["effective_value"] == "Peru"
    assert config["new_stage"]["nested"]["cap"] == 19


def test_null_false_zero_and_empty_have_distinct_recorded_meanings(build):
    inventory = rows(build(config={"new_stage": {"zero": 0, "disabled": False, "unset": None, "blank": ""}}))
    assert inventory["new_stage.zero"]["configured_display"] == "0"
    assert inventory["new_stage.disabled"]["configured_display"] == "Disabled"
    assert inventory["new_stage.unset"]["configured_display"] == "Unset (null)"
    assert inventory["new_stage.blank"]["configured_display"] == "Empty string"
    for key in ("zero", "disabled", "unset", "blank"):
        assert inventory[f"new_stage.{key}"]["value_status"] == "recorded_configuration"


def test_recorded_runtime_overrides_keep_configured_values_and_provenance(build):
    inventory = rows(build(
        config={"source_search": {"max_queries": 8}, "llm": {"model": "selected-model"}},
        summary={"runtime_profile": {"LLM_MODEL": "resolved-model", "SEARCH_MAX_QUERIES": "12"}},
        state={"source_search_execution_summary": {"max_queries": 16, "queries_executed": 9}},
    ))
    assert inventory["source_search.max_queries"]["configured_value"] == 8
    assert inventory["source_search.max_queries"]["effective_value"] == 16
    assert "source_search_execution_summary" in inventory["source_search.max_queries"]["basis"]
    assert inventory["llm.model"]["effective_value"] == "resolved-model"
    assert inventory["llm.model"]["value_status"] == "recorded_effective"
    assert "environment.LLM_MODEL" not in inventory
    assert "queries_executed" not in inventory


def test_nested_recorded_runtime_and_budget_limits_are_actual_not_usage(build):
    inventory = rows(build(
        config={"universal": {"budget_limits": {"search": 20}}},
        summary={"runtime_profile": {"llm": {"provider": "openai", "model": "custom-model"}}, "live_fetch_enabled": False},
        state={"run_budget_ledger": {"limits": {"search": 25, "http_requests": 0, "model:SourceIdentityAgentOutput": 7}, "used": {"search": 18}, "extraction_reserve": 4}},
    ))
    assert inventory["universal.budget_limits.search"]["configured_value"] == 20
    assert inventory["universal.budget_limits.search"]["effective_value"] == 25
    assert inventory["universal.budget_limits.http_requests"]["effective_value"] == 0
    assert inventory["universal.model_limits.SourceIdentityAgentOutput"]["effective_value"] == 7
    assert inventory["universal.extraction_reserve"]["effective_value"] == 4
    assert inventory["live_web.enabled"]["effective_value"] is False
    assert inventory["llm.provider"]["effective_value"] == "openai"
    assert not any("used" in key.split(".") for key in inventory)


def test_environment_and_cli_aliases_describe_one_canonical_setting(build):
    inventory = rows(build(config={"environment": {"LLM_MODEL": "env-model", "UNRELATED_SYSTEM_SETTING": "private-value"}, "cli": {"provider": "openai", "model": "cli-model", "dashboard_port": 9999}}))
    assert inventory["llm.model"]["configured_value"] == "cli-model"
    assert inventory["llm.model"]["environment_variables"] == ["LLM_MODEL"]
    assert "--model" in inventory["llm.model"]["cli_aliases"]
    assert "environment.LLM_MODEL" not in inventory
    assert "cli.model" not in inventory
    assert inventory["cli.dashboard_port"]["configured_value"] == 9999
    assert "private-value" not in json.dumps(inventory)


def test_inactive_modes_and_deprecated_settings_remain_explicit(build):
    inventory = rows(build(config={"pipeline_mode": "standard", "source_search": {"mode": "live"}, "human_review": {"enabled": False}, "llm": {"max_chunks": 12}}))
    assert "Inactive" in inventory["universal.budget_limits.search"]["applicability"]
    assert "Inactive" in inventory["source_search.fixture_path"]["applicability"]
    assert "Inactive" in inventory["human_review.apply_decisions"]["applicability"]
    assert "deprecated" in inventory["llm.max_chunks"]["applicability"].lower()
    assert inventory["llm.max_chunks"]["configured_value"] == 12
    assert "Reserved" in inventory["source_search.cache_enabled"]["applicability"]


def test_does_not_assume_missing_mode_or_switch_is_disabled(build):
    inventory = rows(build())
    assert "Inactive" not in inventory["source_search.fixture_path"]["applicability"]
    assert "Inactive" not in inventory["human_review.apply_decisions"]["applicability"]
    assert "Inactive" not in inventory["universal.budget_limits.search"]["applicability"]


def test_secrets_are_redacted_recursively_in_keys_lists_and_urls(build):
    config = {
        "llm": {"ApiKey": "secret-a", "max_tokens": 100},
        "extension": {"AUTHORIZATION": "secret-b", "nested": {"clientSecret": "secret-c"}, "entries": [{"token": "secret-d", "url": "https://user:secret-e@example.org/file?access_token=secret-f&month=2#token=secret-g"}]},
        "source_search": {"fixture_path": "https://user:secret-h@example.org/file?API_KEY=secret-i"},
        "environment": {"OPENAI_API_KEY": "secret-j", "SYSTEM_PASSWORD": "secret-k"},
    }
    result = build(config=config, summary={"runtime_profile": {"environment": {"TAVILY_API_KEY": "secret-l", "LLM_MODEL": "safe-model"}}})
    serialized = json.dumps(result)
    for suffix in "abcdefghijkl":
        assert f"secret-{suffix}" not in serialized
    assert "month=2" in serialized
    inventory = rows(result)
    assert inventory["llm.max_tokens"]["configured_value"] == 100
    assert inventory["llm.model"]["effective_value"] == "safe-model"
    assert inventory["credentials.OPENAI_API_KEY"]["value_status"] == "credential_redacted"
    assert config["llm"]["ApiKey"] == "secret-a"


def test_does_not_read_process_environment_or_dotenv(build, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Settings reports must use supplied recorded evidence only")

    from data_collection_workflow import runtime_profile
    monkeypatch.setattr(runtime_profile, "load_project_env", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(type(os.environ), "get", forbidden)
    inventory = rows(build(config={"llm": {"model": "recorded-model"}}))
    assert inventory["llm.model"]["effective_value"] == "recorded-model"


def test_absolute_local_paths_are_shortened_but_task_scope_is_preserved(build):
    inventory = rows(build(config={"output": {"run_output_root": "C:\\Users\\Somebody\\private\\outputs"}, "human_review": {"decisions_path": "/home/name/private/decisions.json"}, "workflow": {"seed_source_overlay_path": "configs/overlay.json"}, "structured_task": {"location": "Lima / Callao", "target_fields": ["cases_confirmed", "deaths"]}}))
    assert inventory["output.run_output_root"]["configured_display"] == "outputs"
    assert inventory["human_review.decisions_path"]["configured_display"] == "decisions.json"
    assert inventory["workflow.seed_source_overlay_path"]["configured_display"] == "configs/overlay.json"
    assert inventory["structured_task.location"]["configured_value"] == "Lima / Callao"


def test_packaged_defaults_work_without_repository_or_config_template(build, tmp_path):
    package = Path(__file__).resolve().parents[1] / "src" / "data_collection_workflow"
    archive = tmp_path / "installed_package.zip"
    with zipfile.ZipFile(archive, "w") as wheel:
        for path in package.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
                wheel.write(path, path.relative_to(package.parent).as_posix())
    script = (
        "import json,sys;sys.path.insert(0,sys.argv[1]);"
        "from data_collection_workflow.reporting.run_settings import build_run_settings;"
        "r=build_run_settings();print(json.dumps(r))"
    )
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(archive)], cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", check=True)
    inventory = rows(json.loads(completed.stdout))
    assert inventory["llm.extraction.scheduler.safety_max_calls"]["current_default"] == 2400
    assert inventory["structured_task.disease"]["effective_display"] == "Not recorded"


def test_persisted_configuration_sanitizer_preserves_scope_and_removes_nested_url_secrets():
    module = importlib.import_module("data_collection_workflow.reporting.run_settings")
    sanitize = getattr(module, "sanitize_configuration", None)
    assert callable(sanitize), "The live runner needs a public configuration sanitizer"
    config = {"structured_task": {"disease": "Cholera", "location": "Haiti"}, "PASSWORD": "do-not-save", "extension": {"url": "https://example.org/?next=https%3A%2F%2Fother.org%2Fdata%3Ftoken%3Dhidden-inner%26page%3D2", "AUTH": "do-not-save-auth"}}
    safe = sanitize(config)
    assert safe["structured_task"] == config["structured_task"]
    assert "do-not-save" not in json.dumps(safe)
    assert "hidden-inner" not in json.dumps(safe)
    assert safe["extension"]["url"].startswith("https://example.org/")
    assert config["PASSWORD"] == "do-not-save"


def test_new_uppercase_config_keys_are_preserved_but_runtime_system_keys_are_not(build):
    inventory = rows(build(config={"NEW_STAGE_CAP": 7}, summary={"runtime_profile": {"SYSTEM_OPTION": "never-export-system", "env": {"LLM_MODEL": "resolved-model"}}}))
    assert inventory["NEW_STAGE_CAP"]["configured_value"] == 7
    assert "never-export-system" not in json.dumps(inventory)


def test_live_runner_execution_options_and_whitelisted_env_are_recorded(build):
    inventory = rows(build(state={"execution_options": {"resume_session": "session-9", "write_run_notebook": False}, "runtime_profile": {"env": {"FETCH_PARSE_TABLES": "false", "LLM_EXTRACTION_MAX_CONCURRENCY": "0", "OPENAI_API_KEY": "secret-from-profile"}}}))
    assert inventory["cli.resume_session"]["effective_value"] == "session-9"
    assert inventory["cli.write_run_notebook"]["effective_value"] is False
    assert inventory["content_fetch.parse_tables"]["effective_value"] is False
    assert inventory["llm.extraction.scheduler.max_concurrency"]["effective_value"] == 0
    assert "secret-from-profile" not in json.dumps(inventory)


def test_token_limit_exception_never_overrides_other_secret_markers(build):
    result = build(config={"password.max_tokens": "hidden-password", "llm": {"max_tokens": 0}})
    assert "hidden-password" not in json.dumps(result)
    assert rows(result)["llm.max_tokens"]["configured_value"] == 0


def test_disabled_cli_alias_records_the_same_canonical_boolean(build):
    inventory = rows(build(config={"cli": {"no_live_fetch": True}}))
    assert inventory["live_web.enabled"]["configured_value"] is False
    assert "cli.no_live_fetch" not in inventory


def test_saved_run_summary_provider_and_model_are_recorded_without_a_config(build):
    inventory = rows(build(summary={"provider": "openai", "model": "account-model"}))
    assert inventory["llm.provider"]["effective_value"] == "openai"
    assert inventory["llm.model"]["effective_value"] == "account-model"
    assert inventory["llm.model"]["configured_display"] == "Not recorded"


def test_content_fixture_map_is_independent_of_preloaded_fixture_documents(build):
    inventory = rows(build(config={"workflow": {"use_fixture_documents": False}, "source_search": {"mode": "fixture"}, "content_fetch": {"content_fixture_map_path": "my-fixture-map.json"}}))
    assert "Inactive" not in inventory["content_fetch.content_fixture_map_path"]["applicability"]


def test_stage_controls_are_inactive_when_their_recorded_switch_is_disabled(build):
    inventory = rows(build(config={"source_search": {"enabled": False}, "llm": {"structured_extraction_enabled": False, "source_critic_enabled": False}}))
    for key in ("source_search.max_queries", "llm.extraction.scheduler.max_concurrency", "llm.source_critic.max_sources"):
        assert "Inactive" in inventory[key]["applicability"], key
