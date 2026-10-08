"""Public execution/export contracts for the unified reading report."""
import argparse
import json
from pathlib import Path

from data_collection_workflow.result_manifest import build_result_manifest, write_universal_outputs, write_universal_run_outputs
from data_collection_workflow.task_result_report import build_task_result, write_task_result_artifacts


def empty_package():
    package = {key: [] for key in ('final_dataset', 'final_case_dataset', 'aggregate_dataset', 'candidate_records', 'context_records', 'source_registry')}
    package['result_manifest'] = build_result_manifest(package, {'structured_task': {
        'disease': 'measles', 'location': 'Canada', 'start_date': '2024-01-01', 'end_date': '2024-12-31',
        'target_fields': ['cases_confirmed', 'deaths'],
    }})
    return package


def test_task_answer_writer_keeps_json_without_a_second_reading_report(tmp_path):
    package = empty_package()
    result = build_task_result(package, package['result_manifest'])
    paths = write_task_result_artifacts(result, tmp_path)
    assert set(paths) == {'task_result_json'}
    assert json.loads(Path(paths['task_result_json']).read_text(encoding='utf-8')) == result
    assert not (tmp_path / 'task_result.md').exists()


def test_collection_export_does_not_create_a_duplicate_final_markdown(tmp_path):
    paths = write_universal_outputs(empty_package(), tmp_path)
    assert (tmp_path / 'final_dataset.csv').is_file()
    assert (tmp_path / 'final_package.json').is_file()
    assert not (tmp_path / 'final_report.md').exists()
    assert 'final_report.md' not in paths


def test_run_output_replaces_stale_reading_reports_but_preserves_diagnostics(tmp_path):
    stale = ['task_result.md', 'final_report.md', 'workflow_run_report.md', 'workflow_run_report_chinese.md', 'workflow_interpretive_report.md']
    for name in stale:
        (tmp_path / name).write_text('stale previous reading report', encoding='utf-8')
    diagnostic = tmp_path / 'diagnostics' / 'stage_notes.md'
    diagnostic.parent.mkdir()
    diagnostic.write_text('stage diagnostic', encoding='utf-8')
    summary = {'artifact_paths': {'run_report': str(tmp_path / 'workflow_run_report.md'), 'interpretive_report_english': str(tmp_path / 'workflow_interpretive_report.md')}}
    write_universal_run_outputs(empty_package(), summary, tmp_path)
    assert Path(summary['artifact_paths']['final_report_english']).name == 'final_report.html'
    assert Path(summary['artifact_paths']['report_bundle']).is_file()
    assert all(not (tmp_path / name).exists() for name in stale)
    assert diagnostic.read_text(encoding='utf-8') == 'stage diagnostic'
    assert not any('interpretive_report' in key or key == 'task_result_english' for key in summary['artifact_paths'])
    saved = json.loads((tmp_path / 'workflow_run_summary.json').read_text(encoding='utf-8'))
    assert saved['artifact_paths']['final_report_english'] == summary['artifact_paths']['final_report_english']
    from bs4 import BeautifulSoup
    console = BeautifulSoup((tmp_path / 'collection/workflow_console.html').read_text(encoding='utf-8'), 'html.parser')
    assert console.a['href'] == '../final_report.html'


def test_export_copies_one_portable_report_with_its_materials(tmp_path):
    from data_collection_workflow.cli import cmd_export
    source, target = tmp_path / 'session', tmp_path / 'share'
    source.mkdir()
    write_universal_run_outputs(empty_package(), {'artifact_paths': {}}, source)
    result = cmd_export(argparse.Namespace(session_dir=str(source), output_dir=str(target), format='both'))
    assert result == 0
    assert (target / 'final_report.html').is_file()
    assert (target / 'data' / 'source_catalog.json').is_file()
    assert (target / 'data' / 'run_settings.json').is_file()
    assert (target / 'session_report.zip').is_file()
    assert not list(target.glob('*.md'))


def test_public_report_entry_uses_html_and_retires_known_legacy_reading_files(tmp_path):
    from data_collection_workflow.reporting import write_final_reports

    write_universal_outputs(empty_package(), tmp_path / 'collection')
    (tmp_path / 'task_result.md').write_text('old answer', encoding='utf-8')
    (tmp_path / 'final_report.md').write_text('old summary', encoding='utf-8')
    paths = write_final_reports(tmp_path)
    assert Path(paths['english_report']).name == 'final_report.html'
    assert paths['english_report'] == paths['final_report_english']
    assert not list(tmp_path.glob('*.md'))


def test_latest_shortcut_supports_report_on_a_different_windows_drive(tmp_path, monkeypatch):
    from data_collection_workflow.reporting import output_contract
    from bs4 import BeautifulSoup

    report = tmp_path / 'session with space' / 'final_report.html'
    report.parent.mkdir()
    report.write_text('report', encoding='utf-8')
    def different_mount(*args):
        raise ValueError('path is on a different mount')
    monkeypatch.setattr(output_contract.os.path, 'relpath', different_mount)
    alias = tmp_path / 'latest.html'
    output_contract.write_report_shortcut(alias, report)
    soup = BeautifulSoup(alias.read_text(encoding='utf-8'), 'html.parser')
    assert soup.a['href'] == report.resolve().as_uri()
