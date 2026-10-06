"""Offline startup disclosure regressions; no graph/provider/network execution."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
import re

import pytest

from data_collection_workflow import session_runtime as runtime
from data_collection_workflow.runtime_profile import workflow_run_env_from_config
import scripts.run_workflow as runner
import scripts.collect as interactive


def _config(*, adaptive=True, max_bytes=12345):
    config = {
        "pipeline_mode": "evidence",
        "structured_task": {"disease": "measles", "location": "Canada", "start_date": "2024-01-01", "end_date": "2024-12-31"},
        "content_fetch": {"max_total_sources": 3, "max_search_derived_sources": 2},
        "source_search": {"iterative": {"enabled": True, "max_total_queries": 20},
                          "authority_gap_retry": {"enabled": False}},
        "llm": {"provider": "anthropic", "model": "offline-test-model"},
        "universal": {},
    }
    if max_bytes is not None:
        config["content_fetch"]["max_bytes"] = max_bytes
    if adaptive:
        config["universal"] = {
            "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 50},
            "budget_limits": {"source_targets": 200, "http_requests": 1000, "search": 7},
        }
    return config


def _snapshot(config):
    return {"limits": runtime.derive_budget_limits(config), "used": {},
            "extraction_reserve": 40, "model_default_limit": 30,
            "budget_revision": 0, "budget_amendments": []}


def _assert_number(text, label, value):
    assert re.search(r"\b" + re.escape(label) + r"\s*[:=]\s*" + str(value) + r"\b", text), text


def _display(config, capsys, *, origin="config file", snapshot=None, env=None):
    helper = getattr(runner, "_print_effective_acquisition_settings", None)
    assert callable(helper), "Ordinary startup must disclose effective download and acquisition budgets"
    snapshot = snapshot if snapshot is not None else _snapshot(config)
    env = env if env is not None else workflow_run_env_from_config(config)
    before = deepcopy((config, snapshot, env))
    helper(config, env_updates=env, budget_snapshot=snapshot, origin=origin)
    assert (config, snapshot, env) == before, "Rendering must not change any effective settings or ledger data"
    return capsys.readouterr().out


@pytest.mark.parametrize("max_bytes, expected", [(12345, 12345), (20_000_000, 20_000_000), (None, 1_000_000)])
def test_configured_display_uses_actual_download_limit_and_explicit_origin(capsys, max_bytes, expected):
    config = _config(max_bytes=max_bytes)
    # This lower-priority value must not replace the effective fetch environment.
    config["universal"]["acquisition"] = {"max_bytes": 99_999_999}
    text = _display(config, capsys, origin="config file")
    _assert_number(text, "max_bytes", expected)
    _assert_number(text, "source_targets", 200)
    _assert_number(text, "http_requests", 1000)
    _assert_number(text, "search", 7)
    assert "config file" in text
    assert "interactive preset" not in text
    assert "content_fetch.max_bytes" in text if max_bytes is not None else "default" in text
    assert "source_search" in text and "universal.budget_limits.search" in text


def test_unversioned_strict_display_preserves_small_source_caps(capsys):
    text = _display(_config(adaptive=False), capsys, origin="saved session config")
    assert "strict" in text
    _assert_number(text, "max_bytes", 12345)
    _assert_number(text, "fetch_ordinary", 2)
    assert "soft_source_target" not in text
    assert "source_targets" not in text
    assert "saved session config" in text


def test_zero_quick_budget_is_not_replaced_with_a_default(capsys):
    config = _config()
    config["universal"]["budget_policy"]["soft_source_target"] = 0
    config["universal"]["budget_limits"].update(source_targets=0, http_requests=0, search=0)
    text = _display(config, capsys, origin="interactive quick preset")
    for name in ("soft_source_target", "source_targets", "http_requests", "search"):
        _assert_number(text, name, 0)
    assert "interactive quick preset" in text


def test_amendment_display_uses_persisted_effective_caps_without_exposing_reason(capsys, tmp_path):
    config = _config()
    ledger = runtime.RunBudgetLedger(tmp_path / "operations.sqlite", runtime.derive_budget_limits(config), adaptive=True)
    ledger.amend_budget(amendment_id="increase-http", increases={"http_requests": 1500},
                        reason="SENTINEL_PRIVATE_REASON")
    snapshot = ledger.snapshot()
    before_bytes = (tmp_path / "operations.sqlite").read_bytes()
    text = _display(config, capsys, snapshot=snapshot, origin="saved session config")
    _assert_number(text, "http_requests", 1500)
    _assert_number(text, "source_targets", 200)
    _assert_number(text, "max_bytes", 12345)
    assert "amendment" in text and "revision" in text and "1" in text
    assert "SENTINEL_PRIVATE_REASON" not in text
    assert (tmp_path / "operations.sqlite").read_bytes() == before_bytes
    assert ledger.snapshot() == snapshot


def test_display_does_not_dump_config_environment_or_secret_values(capsys):
    config = _config()
    config["api_key"] = "SENTINEL_CONFIG_SECRET"
    env = workflow_run_env_from_config(config)
    env["ANTHROPIC_API_KEY"] = "SENTINEL_ENV_SECRET"
    text = _display(config, capsys, env=env)
    assert "SENTINEL_" not in text
    assert "api_key" not in text.lower()
    assert "ANTHROPIC_API_KEY" not in text


class _StopBeforePaidCall(Exception):
    pass


def _patch_runner_before_preflight(monkeypatch, tmp_path, capsys, captured, *, origin, context=None):
    monkeypatch.setattr(runner, "_config_with_cli_overrides", lambda args: (tmp_path / "settings.json", captured["config"]))
    monkeypatch.setattr(runner, "workflow_output_dir_from_config", lambda config: tmp_path / "synthetic-session")
    monkeypatch.setattr(runner, "workflow_initial_state_from_config", lambda config: {})
    monkeypatch.setattr(runner, "resolve_task_compatible_validation_records", lambda **kwargs: {})
    monkeypatch.setattr(runner, "_llm_enabled", lambda env: True)
    def initialize(config, *args, **kwargs):
        if context is not None:
            return context
        ledger = SimpleNamespace(snapshot=lambda: _snapshot(config), provider_status=lambda provider: None)
        return SimpleNamespace(ledger=ledger, activate=lambda: nullcontext())
    monkeypatch.setattr(runtime, "initialize_universal_run", initialize)
    def before_paid_call():
        text = capsys.readouterr().out
        _assert_number(text, "max_bytes", captured["expected_bytes"])
        _assert_number(text, "source_targets", captured["expected_targets"])
        _assert_number(text, "http_requests", captured["expected_http"])
        assert origin in text
        assert len(re.findall(r"\bmax_bytes\s*[:=]", text)) == 1
        raise _StopBeforePaidCall
    monkeypatch.setattr(runner, "_preflight_llm_with_trace_policy", before_paid_call)
    monkeypatch.setattr(runner, "_run_graph_with_events", lambda *a, **kw: pytest.fail("No graph execution in display regression"))


def test_normal_configured_runner_discloses_budget_before_paid_preflight(monkeypatch, tmp_path, capsys):
    captured = {"config": _config(), "expected_bytes": 12345, "expected_targets": 200, "expected_http": 1000}
    _patch_runner_before_preflight(monkeypatch, tmp_path, capsys, captured, origin="config file")
    args = SimpleNamespace(config=str(tmp_path / "settings.json"), resume_session=None, budget_amendment=None)
    with pytest.raises(_StopBeforePaidCall):
        runner.run_workflow(args)
    assert not (tmp_path / "synthetic-session").exists()


@pytest.mark.parametrize("quick", [False, True])
def test_normal_interactive_entry_discloses_preset_origin_through_shared_runner(monkeypatch, tmp_path, capsys, quick):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_configure_utf8_stdio", lambda: None)
    monkeypatch.setattr(interactive, "PROJECT_ROOT", tmp_path)
    captured = {"expected_bytes": 1_000_000 if quick else 20_000_000,
                "expected_targets": 5 if quick else 200, "expected_http": 25 if quick else 1000}
    def saved(config, session_id):
        captured["config"] = deepcopy(config)
        return tmp_path / "generated.json"
    monkeypatch.setattr(interactive, "_write_generated_config", saved)
    monkeypatch.setattr(interactive, "run_workflow", runner.run_workflow)
    _patch_runner_before_preflight(monkeypatch, tmp_path, capsys, captured,
                                   origin="interactive quick preset" if quick else "interactive preset")
    argv = ["--pipeline-mode", "evidence", "--disease", "example fever", "--location", "Exampleland",
            "--start-date", "2025-01-01", "--end-date", "2025-12-31", "--session-id", "offline-display",
            "--provider", "anthropic", "--model", "offline-test-model", "--no-dashboard", "--no-run-notebook", "--no-live-status"]
    if quick:
        argv.append("--quick-test-mode")
    with pytest.raises(_StopBeforePaidCall):
        interactive.main(argv)
    assert "acquisition_settings_origin" not in captured["config"]
    assert not (tmp_path / "generated.json").exists()


def test_configured_resume_discloses_amendment_after_acceptance_before_paid_call(monkeypatch, tmp_path, capsys):
    from data_collection_workflow import acquisition_budget
    config = _config()
    ledger = runtime.RunBudgetLedger(tmp_path / "operations.sqlite", runtime.derive_budget_limits(config), adaptive=True)
    context = SimpleNamespace(ledger=ledger, activate=lambda: nullcontext(), amend_budget=ledger.amend_budget)
    amendment = {"amendment_id": "new-http", "increases": {"http_requests": 1500}, "reason": "SENTINEL_PRIVATE_REASON"}
    monkeypatch.setattr(acquisition_budget, "load_budget_amendment", lambda path: amendment)
    captured = {"config": config, "expected_bytes": 12345, "expected_targets": 200, "expected_http": 1500}
    _patch_runner_before_preflight(monkeypatch, tmp_path, capsys, captured,
                                   origin="saved session config", context=context)
    args = SimpleNamespace(config=str(tmp_path / "settings.json"), resume_session="synthetic-session",
                           budget_amendment="not-read.json", acquisition_settings_origin="saved session config")
    before_config = deepcopy(config)
    with pytest.raises(_StopBeforePaidCall):
        runner.run_workflow(args)
    assert config == before_config
    assert ledger.snapshot()["budget_revision"] == 1
    assert ledger.snapshot()["used"] == {}


def test_exact_interactive_resume_uses_saved_origin_without_new_preset(monkeypatch, tmp_path, capsys):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)
    monkeypatch.setattr(interactive, "_require_keys", lambda **kwargs: [])
    monkeypatch.setattr(interactive, "_configure_utf8_stdio", lambda: None)
    config = _config()
    config["output"] = {"session_id": "saved-display", "run_output_root": str(tmp_path)}
    def loaded(args, explicit_options):
        args.provider = "anthropic"
        args.model = "offline-test-model"
        return config, {"session_id": "saved-display"}, tmp_path / "saved.json"
    monkeypatch.setattr(interactive, "_load_named_resume_config", loaded)
    monkeypatch.setattr(interactive, "_collect_inputs", lambda *a: pytest.fail("No prompts while resuming"))
    monkeypatch.setattr(interactive, "_write_generated_config", lambda *a: pytest.fail("No rewritten resume config"))
    monkeypatch.setattr(interactive, "run_workflow", runner.run_workflow)
    captured = {"config": config, "expected_bytes": 12345, "expected_targets": 200, "expected_http": 1000}
    _patch_runner_before_preflight(monkeypatch, tmp_path, capsys, captured, origin="saved session config")
    before_config = deepcopy(config)
    with pytest.raises(_StopBeforePaidCall):
        interactive.main(["--resume-session", "saved-display", "--no-dashboard", "--no-run-notebook", "--no-live-status"])
    assert config == before_config
