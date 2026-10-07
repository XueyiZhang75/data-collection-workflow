from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from synthetic_workflow_inputs import write_workflow_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, "-m", "data_collection_workflow.cli"]


def _env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    if extra:
        env.update(extra)
    return env


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        CLI + args,
        cwd=PROJECT_ROOT,
        env=env or _env(),
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_cli_help_lists_user_facing_subcommands():
    result = _run(["--help"])

    assert result.returncode == 0, result.stderr
    assert "data collection workflow" in result.stdout
    for command in ("collect", "validate-config", "inspect-run", "export", "review-summary", "init-config"):
        assert command in result.stdout


def test_validate_config_accepts_fixture_and_live_configs_without_secrets(tmp_path):
    fixture_config = write_workflow_config(tmp_path / "fixture", phase="review")
    live_config = write_workflow_config(tmp_path / "live")
    live = json.loads(live_config.read_text(encoding="utf-8"))
    live["live_web"]["enabled"] = True
    live["source_search"].update(enabled=True, mode="live", provider="tavily", fixture_path=None)
    live_config.write_text(json.dumps(live), encoding="utf-8")
    configs = [fixture_config, live_config]

    for config in configs:
        result = _run(["validate-config", "--config", str(config)])
        combined = result.stdout + result.stderr
        assert result.returncode == 0, combined
        assert "valid: true" in result.stdout.lower()
        assert "tvly-test-key" not in combined
        assert "sk-ant-test-key" not in combined


def test_collect_print_config_only_sanitizes_secret_values(tmp_path):
    config_path = write_workflow_config(tmp_path, phase="review")
    env = _env(
        {
            "TAVILY_API_KEY": "tvly-test-key",
            "ANTHROPIC_API_KEY": "sk-ant-test-key",
        }
    )

    result = _run(
        [
            "collect",
            "--config",
            str(config_path),
            "--print-config-only",
        ],
        env=env,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "api_key_present" in result.stdout
    assert "source_search_api_key_present" in result.stdout
    assert "tvly-test-key" not in combined
    assert "sk-ant-test-key" not in combined


def test_collect_dry_run_shows_structured_overrides_without_running_graph(tmp_path):
    config_path = write_workflow_config(tmp_path, phase="review")
    result = _run(
        [
            "collect",
            "--config",
            str(config_path),
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
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert '"disease": "COVID-19"' in result.stdout
    assert '"location": "New York"' in result.stdout
    assert '"target_fields"' in result.stdout
    assert "HDC workflow run completed" not in result.stdout
    assert not (tmp_path / "sessions").exists()


def test_configured_runner_case_study_real_mode_helper_accepts_cli_namespace_without_flag(monkeypatch):
    from scripts.run_workflow import _case_study_real_mode_enabled

    monkeypatch.delenv("RUN_MODE", raising=False)

    assert _case_study_real_mode_enabled(argparse.Namespace()) is False


def test_collect_offline_fixture_run_inspect_review_and_export(tmp_path):
    config_path = write_workflow_config(tmp_path, phase="review")
    session_id = "pytest_cli_temporary_fixture"
    output_root = tmp_path / "runs"
    export_dir = tmp_path / "exported"

    collect = _run(
        [
            "collect",
            "--config",
            str(config_path),
            "--session-id",
            session_id,
            "--output-dir",
            str(output_root),
            "--disable-all-llm",
        ]
    )
    assert collect.returncode == 0, collect.stderr
    assert "output_dir:" in collect.stdout

    session_dir = output_root / "sessions" / session_id
    assert (session_dir / "collection" / "final_package.json").exists()
    assert (session_dir / "workflow_console" / "data_collection_workflow_console.html").exists()

    inspect = _run(["inspect-run", "--session-dir", str(session_dir)])
    assert inspect.returncode == 0, inspect.stderr
    assert "final_dataset_count" in inspect.stdout
    assert "anomaly_count" in inspect.stdout
    assert "human_review_item_count" in inspect.stdout

    review = _run(["review-summary", "--session-dir", str(session_dir)])
    assert review.returncode == 0, review.stderr
    assert "decisions_applied_count" in review.stdout
    assert "audit_trail_count" in review.stdout

    export = _run(
        [
            "export",
            "--session-dir",
            str(session_dir),
            "--output-dir",
            str(export_dir),
            "--format",
            "both",
        ]
    )
    assert export.returncode == 0, export.stderr
    assert (export_dir / "final_dataset.json").exists()
    assert (export_dir / "final_dataset.csv").exists()
    assert (export_dir / "final_dataset_post_review.json").exists()
    assert (session_dir / "collection" / "final_package.json").exists()


def test_init_config_writes_safe_template_and_validate_config_accepts_it(tmp_path):
    write_workflow_config(tmp_path, phase="search")
    config_path = tmp_path / "generated_task_config.jsonc"

    init = _run(
        [
            "init-config",
            "--disease",
            "Example disease",
            "--location",
            "Example region",
            "--start-date",
            "2025",
            "--end-date",
            "2025",
            "--target-field",
            "cases_unspecified",
            "--target-field",
            "deaths",
            "--mode",
            "fixture-search",
            "--search-fixture-path",
            str(tmp_path / "search.json"),
            "--output",
            str(config_path),
        ]
    )

    assert init.returncode == 0, init.stderr
    text = config_path.read_text(encoding="utf-8")
    assert "structured_task" in text
    assert "source_search" in text
    assert "TAVILY_API_KEY" in text
    assert "tvly-" not in text
    assert "sk-ant-" not in text

    validate = _run(["validate-config", "--config", str(config_path)])
    assert validate.returncode == 0, validate.stderr
    assert "valid: true" in validate.stdout.lower()


def test_workflow_console_uses_generic_public_health_record_wording():
    text = (PROJECT_ROOT / "scripts" / "build_console.py").read_text(encoding="utf-8")

    assert "抽取 HantavirusRecord" not in text
    assert "PublicHealthRecord" in text or "generic public-health records" in text
    assert "anomaly_summary" in text
    assert "human_review_application_summary" in text
    assert "final_dataset_post_review" in text


def test_public_documentation_covers_configuration_runtime_and_results():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    configuration = (PROJECT_ROOT / "docs" / "configuration.md").read_text(encoding="utf-8")
    runtime = (PROJECT_ROOT / "docs" / "runtime.md").read_text(encoding="utf-8")
    code_map = (PROJECT_ROOT / "docs" / "code_map.md").read_text(encoding="utf-8")

    assert "data collection workflow" in readme.lower()
    for token in ("configs/workflow.jsonc", "TAVILY_API_KEY",
                  "ANTHROPIC_API_KEY", "scripts/collect.py", "result_manifest.json",
                  "review-summary", "--resume-session"):
        assert token in readme
    for token in ("structured_task.disease", "start_date", "end_date", "pipeline_mode",
                  "--budget-amendment", "--provider-resume"):
        assert token in configuration
    for token in ("Chromium", "Tesseract", "TESSDATA_DIR", "synthetic"):
        assert token in runtime
    for module in ("session_runtime.py", "document_acquisition.py", "evidence_qualification.py",
                   "workflow_recovery.py"):
        assert module in code_map
