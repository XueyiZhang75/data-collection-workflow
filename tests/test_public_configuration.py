"""Public task configuration must not silently run a bundled case study."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from data_collection_workflow import cli, runtime_profile


def _validate(config):
    validator = getattr(runtime_profile, "validate_workflow_task", None)
    assert callable(validator), "Task validation must run before external work."
    return validator(config)


@pytest.mark.parametrize("factory", [runtime_profile.default_workflow_run_config,
                                     runtime_profile.load_workflow_run_config])
def test_default_configuration_does_not_supply_a_case_task_or_source_set(factory):
    config = factory()
    state = runtime_profile.workflow_initial_state_from_config(config)
    env = runtime_profile.workflow_run_env_from_config(config)
    assert state["user_request"] == ""
    assert state.get("structured_task", {}) == {}
    assert env["PIPELINE_MODE"] == "standard"
    assert env["COLLECTION_MODE"] == "standard"
    assert env["SEED_SOURCE_OVERLAY_PATH"] == ""
    assert env["SOURCE_ROLE_POLICY_OVERLAY_PATH"] == ""
    assert env["SOURCE_ID_ALLOWLIST"] == ""
    assert config["source_sets"]["source_id_allowlist_enabled"] is False


def test_direct_runtime_environment_has_no_implicit_source_allowlist():
    assert runtime_profile.workflow_run_env()["SOURCE_ID_ALLOWLIST"] == ""
    assert runtime_profile.workflow_run_env(source_id_allowlist_enabled=True)["SOURCE_ID_ALLOWLIST"] == ""
    assert runtime_profile.workflow_run_env(
        source_id_allowlist_enabled=True, source_id_allowlist=["source_chosen_by_user"]
    )["SOURCE_ID_ALLOWLIST"] == "source_chosen_by_user"


@pytest.mark.parametrize("config", [None, {}, "builtin", "template"])
def test_public_search_does_not_implicitly_mix_in_seed_catalog(config):
    if config is None:
        env = runtime_profile.workflow_run_env()
    else:
        if config == "builtin":
            config = runtime_profile.default_workflow_run_config()
        elif config == "template":
            config = runtime_profile.load_workflow_run_config()
        env = runtime_profile.workflow_run_env_from_config(config)
    assert env["SEARCH_COMBINE_WITH_SEED_CATALOG"] == "false"


def test_seed_catalog_merge_remains_available_when_explicitly_requested():
    env = runtime_profile.workflow_run_env_from_config(
        {"source_search": {"combine_with_seed_catalog": True}}
    )
    assert env["SEARCH_COMBINE_WITH_SEED_CATALOG"] == "true"


def test_free_text_only_config_is_rejected_instead_of_running_default_disease(tmp_path):
    request = "Collect measles case counts in Canada from 2022 to 2024."
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"user_request": request}), encoding="utf-8")
    config = runtime_profile.load_workflow_run_config(path)
    with pytest.raises(ValueError, match="structured_task.*disease.*location.*start_date.*end_date"):
        _validate(config)


def test_free_text_cli_override_cannot_bypass_required_structured_scope():
    request = "Collect measles case counts in Canada from 2022 to 2024."
    config = runtime_profile.workflow_run_config_with_overrides(
        runtime_profile.default_workflow_run_config(), user_request=request
    )
    with pytest.raises(ValueError, match="structured_task"):
        _validate(config)


@pytest.mark.parametrize("config", [{}, {"user_request": " \n "},
                                   {"structured_task": {}}, {"structured_task": None}])
def test_empty_task_is_rejected(config):
    with pytest.raises(ValueError, match="user_request|task"):
        _validate(config)


@pytest.mark.parametrize("field", ["disease", "location", "start_date", "end_date"])
@pytest.mark.parametrize("value", [None, "", "  "])
def test_explicit_structured_task_requires_every_scope_field_even_with_free_text(field, value):
    task = {"disease": "measles", "location": "Canada", "start_date": "2022", "end_date": "2024"}
    task[field] = value
    with pytest.raises(ValueError, match=field):
        _validate({"user_request": "Collect measles cases in Canada in 2022–2024.",
                   "structured_task": task})


@pytest.mark.parametrize("task", [[], "measles", 42])
def test_non_object_structured_task_is_rejected(task):
    with pytest.raises(ValueError, match="structured_task"):
        _validate({"user_request": "Collect measles cases in Canada in 2022–2024.",
                   "structured_task": task})


def test_complete_structured_task_can_run_without_free_text():
    from data_collection_workflow.nodes.task_scope import task_intake_and_scope_planning

    task = {"disease": "measles", "location": "Canada", "start_date": "2022", "end_date": "2024"}
    _validate({"structured_task": task})
    state = runtime_profile.workflow_initial_state_from_config({"structured_task": task})
    scope = task_intake_and_scope_planning(state)["collection_spec"]
    assert scope["disease"] == "measles"
    assert scope["geography"] == "Canada"
    assert scope["start_date"] == "2022"
    assert scope["end_date"] == "2024"
    assert state["user_request"] == ""


@pytest.mark.parametrize("task_text", ["", "Collect measles in Canada from 2022 to 2024."])
@pytest.mark.parametrize("mode", ["standard", "evidence"])
def test_collect_rejects_missing_structured_scope_before_provider_checks(tmp_path, monkeypatch, capsys, task_text, mode):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"user_request": task_text, "structured_task": {}, "pipeline_mode": mode}), encoding="utf-8")

    def unexpected_provider_check(*args, **kwargs):
        raise AssertionError("Task without structured scope reached provider setup.")

    monkeypatch.setattr(cli, "api_key_present", unexpected_provider_check)
    assert cli.main(["collect", "--config", str(path)]) == 2
    assert "task" in capsys.readouterr().err.lower()


@pytest.mark.parametrize("task_text", ["", "Collect measles in Canada from 2022 to 2024."])
@pytest.mark.parametrize("mode", ["standard", "evidence"])
def test_configured_runner_rejects_missing_scope_before_session_or_preflight(tmp_path, monkeypatch, task_text, mode):
    script = Path(runtime_profile.PROJECT_ROOT) / "scripts" / "run_workflow.py"
    spec = importlib.util.spec_from_file_location("public_configuration_runner", script)
    runner = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(runner)
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"user_request": task_text, "structured_task": {}, "pipeline_mode": mode}), encoding="utf-8")

    def unexpected_session_setup(*args, **kwargs):
        raise AssertionError("Task without structured scope reached session setup.")

    monkeypatch.setattr(runner, "workflow_output_dir_from_config", unexpected_session_setup)
    args = argparse.Namespace(config=str(path), enable_live_fetch=False, disable_live_fetch=False,
                              enable_all_llm=False, disable_all_llm=False, provider=None, model=None,
                              timeout_seconds=None, llm_max_chunks=None, output_dir=None,
                              session_id=None, user_request=None)
    with pytest.raises(ValueError, match="task|user_request"):
        runner.run_workflow(args)


def test_blank_template_can_still_be_previewed(capsys):
    assert cli.main(["collect", "--dry-run"]) == 0
    assert "graph_invoked: false" in capsys.readouterr().out
