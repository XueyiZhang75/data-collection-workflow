from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import scripts.collect as interactive


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = [sys.executable, "scripts/collect.py"]


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    if extra:
        env.update(extra)
    return env


def _run(args: list[str], *, input_text: str = "", env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        SCRIPT + args,
        cwd=PROJECT_ROOT,
        env=env or _env(),
        input=input_text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=120,
    )


def _extract_preview_json(stdout: str) -> dict:
    marker = "sanitized_config_json:"
    assert marker in stdout
    text = stdout.split(marker, 1)[1].strip()
    return json.loads(text)


def test_interactive_help_hides_target_field_option_from_normal_users():
    result = _run(["--help"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--target-field" not in result.stdout
    assert "target fields" not in result.stdout.lower()


def test_interactive_script_generates_real_workflow_config_from_prompts_without_leaking_keys():
    env = _env(
        {
            "TAVILY_API_KEY": "tvly-test-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
        }
    )
    result = _run(
        ["--print-config-only"],
        input_text="\n".join(
            [
                "Orthoebolavirus zairense",
                "Democratic Republic of the Congo",
                "2022",
                "2026",
                "orthoebolavirus_drc_preview",
                "",
            ]
        )
        + "\n",
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "tvly-test-key" not in combined
    assert "sk-ant-test-key" not in combined

    preview = _extract_preview_json(result.stdout)
    assert preview["project_name"] == "data collection workflow"
    assert preview["api_keys"]["tavily_api_key_present"] is True
    assert preview["api_keys"]["llm_api_key_present"] is True

    config = preview["config"]
    assert config["pipeline_mode"] == "standard"
    assert config["structured_task"]["disease"] == "Orthoebolavirus zairense"
    assert config["structured_task"]["location"] == "Democratic Republic of the Congo"
    assert config["structured_task"]["start_date"] == "2022-01-01"
    assert config["structured_task"]["end_date"] == "2026-12-31"
    assert config["structured_task"]["target_fields"] == [
        "disease",
        "country",
        "subnational_location",
        "locality",
        "date_reported",
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
        "source_url",
        "source_type",
        "evidence_quote",
    ]
    assert config["workflow"]["collection_mode"] == "direct_collection"
    assert config["structured_task"]["collection_mode"] == "direct_collection"
    assert config["workflow"]["use_fixture_documents"] is False
    assert config["source_search"]["enabled"] is True
    assert config["source_search"]["mode"] == "live"
    assert config["source_search"]["provider"] == "tavily"
    assert config["source_search"]["combine_with_seed_catalog"] is True
    assert config["source_search"]["max_queries"] == 16
    assert config["source_search"]["max_results_per_query"] == 16
    assert config["source_search"]["max_total_results"] == 400
    assert config["source_search"]["iterative"]["enabled"] is True
    assert config["source_search"]["iterative"]["max_total_results"] == 260
    assert config["source_search"]["iterative"]["require_llm"] is True
    assert config["source_search"]["iterative"]["allow_deterministic_fallback"] is False
    assert config["source_search"]["authority_gap_retry"] == {
        "enabled": True,
        "max_queries": 24,
        "known_domain_max_queries": 14,
        "jurisdiction_max_queries": 16,
        "official_page_family_max_queries": 14,
        "result_budget": 120,
        "max_iterations": 1,
    }
    assert config["live_web"]["enabled"] is True
    assert config["content_fetch"]["fetch_search_derived_sources"] is True
    assert config["content_fetch"]["max_search_derived_sources"] == 50
    assert config["content_fetch"]["max_total_sources"] == 50
    assert config["content_fetch"]["allow_needs_review"] is True
    assert config["content_fetch"]["external_fetch"]["enabled"] is True
    assert config["content_fetch"]["external_fetch"]["provider_order"] == [
        "tavily_extract",
        "native_requests",
    ]
    assert config["llm"]["source_planning_enabled"] is True
    assert config["llm"]["source_critic_enabled"] is True
    assert config["llm"]["structured_extraction_enabled"] is True
    assert config["llm"]["max_chunks"] == 400
    assert config["llm"]["max_tokens"] == 8192
    assert config["llm"]["extraction"] == {
        "focused_recovery_reserved_calls": 120,
        "max_domain_call_share": 0.20,
        "high_value_source_min_spans": 2,
        "event_source_min_spans": 1,
        "scheduler": {
            "mode": "quality_adaptive",
            "max_concurrency": 4,
            "soft_checkpoint_calls": 400,
            "safety_max_calls": 2400,
            "rolling_yield_window": 40,
        },
    }
    assert config["llm"]["focused_recovery"] == {
        "max_calls": 120,
        "max_sources": 80,
    }
    assert config["llm"]["official_extraction_max_chunks"] == 120
    assert config["llm"]["source_identity"]["enabled"] is True
    assert config["llm"]["source_identity"]["require_llm"] is True
    assert config["llm"]["source_identity"]["allow_deterministic_fallback"] is False
    assert config["llm"]["source_credibility"]["enabled"] is True
    assert config["disease_intelligence"]["llm_enabled"] is True
    assert config["disease_intelligence"]["force_llm"] is True
    assert config["disease_intelligence"]["fallback_to_curated"] is True
    assert config["human_review"]["enabled"] is False
    assert config["validation"]["mode"] == "diagnostic_only"
    assert config["output"]["session_id"] == "orthoebolavirus_drc_preview"


def test_interactive_script_accepts_noninteractive_arguments_for_preview():
    result = _run(
        [
            "--disease",
            "COVID-19",
            "--location",
            "New York",
            "--start-date",
            "2024",
            "--end-date",
            "2024",
            "--target-field",
            "cases_confirmed",
            "--target-field",
            "deaths",
            "--session-id",
            "covid19_ny_preview",
            "--print-config-only",
        ],
        env=_env({"TAVILY_API_KEY": "tvly-test-key", "ANTHROPIC_API_KEY": "sk-ant-test-key"}),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    config = _extract_preview_json(result.stdout)["config"]
    assert config["structured_task"]["disease"] == "COVID-19"
    assert config["structured_task"]["target_fields"] == ["cases_confirmed", "deaths"]
    assert config["structured_task"]["start_date"] == "2024-01-01"
    assert config["structured_task"]["end_date"] == "2024-12-31"
    assert config["source_search"]["mode"] == "live"
    assert config["llm"]["provider"] == "anthropic"


def test_collect_inputs_uses_default_request_for_fully_specified_cli_even_when_tty(
    monkeypatch,
):
    def fail_prompt(*_args, **_kwargs):
        raise AssertionError("fully specified CLI arguments should not prompt")

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(interactive, "_prompt", fail_prompt)

    collected = interactive._collect_inputs(
        Namespace(
            disease="COVID-19",
            location="New York",
            start_date="2024-01-01",
            end_date="2024-01-07",
            target_field=[],
            session_id="covid19_ny_cli",
            user_request=None,
        )
    )

    assert collected["session_id"] == "covid19_ny_cli"
    assert collected["user_request"] == (
        "Collect COVID-19 cases, deaths, dates, locations, source URLs, "
        "source types, and evidence quotes for New York from 2024-01-01 to 2024-01-07."
    )


def test_configure_utf8_stdio_reconfigures_non_utf8_streams():
    class FakeStream:
        encoding = "cp1252"

        def __init__(self) -> None:
            self.calls: list[dict] = []

        def reconfigure(self, **kwargs) -> None:
            self.calls.append(kwargs)

    stdout = FakeStream()
    stderr = FakeStream()

    interactive._configure_utf8_stdio(stdout=stdout, stderr=stderr)

    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_interactive_script_normalizes_day_level_dates_and_question_mark_session_id_without_field_prompt():
    result = _run(
        ["--print-config-only"],
        input_text="\n".join(
            [
                "hantavirus",
                "America",
                "2024-5-1",
                "2024-5-3",
                "？",
                "",
            ]
        )
        + "\n",
        env=_env({"TAVILY_API_KEY": "tvly-test-key", "ANTHROPIC_API_KEY": "sk-ant-test-key"}),
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Start date (YYYY, YYYY-M-D, or YYYY-MM-DD)" in combined
    assert "Using safe session id:" in combined
    assert "Target fields" not in combined

    config = _extract_preview_json(result.stdout)["config"]
    assert config["structured_task"]["start_date"] == "2024-05-01"
    assert config["structured_task"]["end_date"] == "2024-05-03"
    assert config["structured_task"]["target_fields"] == [
        "disease",
        "country",
        "subnational_location",
        "locality",
        "date_reported",
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
        "source_url",
        "source_type",
        "evidence_quote",
    ]
    assert config["output"]["session_id"].startswith("hantavirus_america_2024_05_01_2024_05_03_")
    assert "？" not in config["output"]["session_id"]


def test_interactive_script_quick_test_mode_matches_visual_budget_controls():
    result = _run(
        [
            "--disease",
            "FLU",
            "--location",
            "Virginia",
            "--start-date",
            "2024-11-1",
            "--end-date",
            "2024-11-5",
            "--session-id",
            "flu_va_quick_preview",
            "--quick-test-mode",
            "--print-config-only",
        ],
        env=_env({"TAVILY_API_KEY": "tvly-test-key", "ANTHROPIC_API_KEY": "sk-ant-test-key"}),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    config = _extract_preview_json(result.stdout)["config"]
    assert config["structured_task"]["start_date"] == "2024-11-01"
    assert config["structured_task"]["end_date"] == "2024-11-05"
    assert config["structured_task"]["quick_test_mode"] is True
    assert config["quick_test_mode"] is True
    assert config["source_search"]["max_queries"] == 2
    assert config["content_fetch"]["max_total_sources"] == 5
    assert config["llm"]["max_chunks"] == 5


def test_interactive_script_defaults_to_direct_collection_mode():
    result = _run(
        [
            "--disease",
            "FLU",
            "--location",
            "Virginia",
            "--start-date",
            "2024-10-01",
            "--end-date",
            "2024-10-10",
            "--session-id",
            "flu_va_direct_preview",
            "--print-config-only",
        ],
        env=_env({"TAVILY_API_KEY": "tvly-test-key", "ANTHROPIC_API_KEY": "sk-ant-test-key"}),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    config = _extract_preview_json(result.stdout)["config"]
    assert config["workflow"]["collection_mode"] == "direct_collection"
    assert config["structured_task"]["collection_mode"] == "direct_collection"
    assert config["human_review"]["enabled"] is False
    assert config["validation"]["mode"] == "diagnostic_only"
    assert config["llm"]["source_critic"]["review_blocks_fetch"] is False


def test_interactive_script_audit_mode_preserves_legacy_standard_collection_mode():
    result = _run(
        [
            "--disease",
            "FLU",
            "--location",
            "Virginia",
            "--start-date",
            "2024-10-01",
            "--end-date",
            "2024-10-10",
            "--session-id",
            "flu_va_audit_preview",
            "--audit-mode",
            "--print-config-only",
        ],
        env=_env({"TAVILY_API_KEY": "tvly-test-key", "ANTHROPIC_API_KEY": "sk-ant-test-key"}),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    config = _extract_preview_json(result.stdout)["config"]
    assert config["workflow"]["collection_mode"] == "standard"
    assert config["structured_task"]["collection_mode"] == "standard"
    assert config["validation"]["mode"] == "live_cross_source"


def test_interactive_script_blocks_real_run_when_required_keys_are_missing():
    env = _env({"TAVILY_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    result = _run(
        [
            "--disease",
            "hantavirus",
            "--location",
            "New Mexico",
            "--start-date",
            "2020",
            "--end-date",
            "2026",
            "--session-id",
            "missing_key_run",
        ],
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Missing required API key" in combined
    assert "TAVILY_API_KEY" in combined
    assert "ANTHROPIC_API_KEY" in combined


def test_interactive_parser_writes_replay_notebook_by_default():
    args = interactive.build_parser().parse_args([])

    assert args.write_run_notebook is True


def test_interactive_runner_args_write_replay_notebook_by_default(tmp_path):
    args = Namespace(output_dir=None, live_status=True, write_run_notebook=True)

    runner_args = interactive._runner_args(tmp_path / "generated_config.json", args)

    assert runner_args.live_status is True
    assert runner_args.write_run_notebook is True


def test_interactive_dashboard_launcher_uses_streamlit_module(tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(interactive.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(interactive.subprocess, "Popen", fake_popen)

    launched = interactive._launch_live_dashboard(
        tmp_path / "sessions" / "session_a",
        preferred_port=8765,
    )

    assert launched["url"] == "http://localhost:8765"
    command, kwargs = calls[0]
    assert command[:4] == [
        sys.executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert "--session-dir" in command
    assert str(tmp_path / "sessions" / "session_a") in command
    assert kwargs["cwd"] == interactive.PROJECT_ROOT
