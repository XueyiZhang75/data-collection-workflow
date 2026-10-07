"""The public CLI must never select disease demo assets implicitly."""
from __future__ import annotations

import pytest

from data_collection_workflow.runtime_profile import load_workflow_run_config
from test_cli_and_user_experience import _run


def _init_args(path, mode):
    return ["init-config", "--disease", "Example disease", "--location", "Example region",
            "--start-date", "2024", "--end-date", "2024", "--mode", mode,
            "--output", str(path)]


@pytest.mark.parametrize("mode", ["offline", "live-search"])
def test_init_config_does_not_attach_implicit_fixture_assets(tmp_path, mode):
    path = tmp_path / "task.jsonc"
    result = _run(_init_args(path, mode))
    assert result.returncode == 0, result.stderr
    config = load_workflow_run_config(path)
    assert config["source_search"]["fixture_path"] is None
    assert config["content_fetch"]["content_fixture_map_path"] is None
    assert config["human_review"]["decisions_path"] is None
    assert config["human_review"]["apply_decisions"] is False


def test_init_config_fixture_search_requires_explicit_search_input(tmp_path):
    path = tmp_path / "task.jsonc"
    result = _run(_init_args(path, "fixture-search"))
    assert result.returncode != 0
    assert "search-fixture-path" in result.stderr + result.stdout
    assert not path.exists()


def test_init_config_uses_explicit_fixture_paths_and_validates(tmp_path):
    path = tmp_path / "task.jsonc"
    search = tmp_path / "search.json"
    content = tmp_path / "content.json"
    decisions = tmp_path / "decisions.json"
    for target, text in ((search, '{"queries": []}'),
                         (content, '{"fixtures": []}'),
                         (decisions, '{"decisions": []}')):
        target.write_text(text, encoding="utf-8")
    result = _run(_init_args(path, "fixture-search") + [
        "--search-fixture-path", str(search), "--content-fixture-map-path", str(content),
        "--review-decisions-path", str(decisions)])
    assert result.returncode == 0, result.stderr
    config = load_workflow_run_config(path)
    assert config["source_search"]["fixture_path"] == str(search)
    assert config["content_fetch"]["content_fixture_map_path"] == str(content)
    assert config["human_review"]["decisions_path"] == str(decisions)
    assert config["human_review"]["apply_decisions"] is True
    validated = _run(["validate-config", "--config", str(path)])
    assert validated.returncode == 0, validated.stdout + validated.stderr


@pytest.mark.parametrize("mode", ["offline", "live-search"])
def test_non_fixture_modes_ignore_explicit_fixture_arguments(tmp_path, mode):
    path = tmp_path / "task.jsonc"
    result = _run(_init_args(path, mode) + ["--search-fixture-path", "unused-search.json",
        "--content-fixture-map-path", "unused-content.json", "--review-decisions-path", "unused-review.json"])
    assert result.returncode == 0, result.stderr
    config = load_workflow_run_config(path)
    assert config["source_search"]["fixture_path"] is None
    assert config["content_fetch"]["content_fixture_map_path"] is None
    assert config["human_review"]["decisions_path"] is None
    assert config["human_review"]["apply_decisions"] is False
