"""Saved evidence run configuration is immutable; all work stays in temporary roots."""
import json
import os
import pytest
import scripts.collect as interactive


def _config(disease='pertussis'):
    return {'pipeline_mode': 'evidence', 'structured_task': {'disease': disease}, 'output': {'session_id': 'existing'}}


def test_different_generated_evidence_config_cannot_overwrite_prior_run(tmp_path, monkeypatch):
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    saved = interactive._write_generated_config(_config(), 'existing')
    before = saved.read_bytes()
    with pytest.raises(FileExistsError, match='existing'):
        interactive._write_generated_config(_config('dengue'), 'existing')
    assert saved.read_bytes() == before


def test_identical_resume_config_is_reused_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    saved = interactive._write_generated_config(_config(), 'existing')
    os.utime(saved, (1234567890, 1234567890))
    before = saved.stat().st_mtime_ns
    assert interactive._write_generated_config(_config(), 'existing') == saved
    assert saved.stat().st_mtime_ns == before


def test_malformed_saved_config_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    saved = tmp_path / 'outputs/generated_configs/existing.json'
    saved.parent.mkdir(parents=True)
    saved.write_text('incomplete original config', encoding='utf-8')
    with pytest.raises((FileExistsError, ValueError)):
        interactive._write_generated_config(_config(), 'existing')
    assert saved.read_text(encoding='utf-8') == 'incomplete original config'


def test_cli_refuses_config_collision_before_launching_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(interactive, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(interactive, '_configure_utf8_stdio', lambda: None)
    monkeypatch.setattr(interactive, '_require_keys', lambda **kwargs: [])
    saved = interactive._write_generated_config(_config(), 'existing')
    before = saved.read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail('conflicting saved configuration must not launch a workflow or dashboard')
    monkeypatch.setattr(interactive, 'run_workflow', forbidden)
    monkeypatch.setattr(interactive, '_launch_live_dashboard', forbidden)
    result = interactive.main(['--pipeline-mode', 'evidence', '--disease', 'dengue',
        '--location', 'Brazil', '--start-date', '2024', '--end-date', '2024', '--session-id', 'existing', '--no-llm'])
    assert result == 2
    assert 'existing' in capsys.readouterr().err
    assert saved.read_bytes() == before
