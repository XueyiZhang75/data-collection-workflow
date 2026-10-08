"""Exports rebuild one report from current records and original session evidence."""
import argparse
import json
import zipfile

from data_collection_workflow.cli import cmd_export
from data_collection_workflow.reporting.run_settings import build_run_settings
from data_collection_workflow.reporting.unified_report import write_unified_report
from data_collection_workflow.result_manifest import build_result_manifest


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _package(value):
    quote = f'Brazil reported {value} confirmed dengue cases during 2024.'
    record = {'record_id': 'r', 'source_id': 's', 'source_url': 'https://example.org/report',
              'disease': 'dengue', 'country': 'Brazil', 'geographic_scope': 'Brazil',
              'geographic_scope_type': 'country', 'reporting_period': '2024',
              'metric_period_start': '2024-01-01', 'metric_period_end': '2024-12-31',
              'cases_confirmed': value, 'count_semantics': 'cumulative', 'evidence_quote': quote}
    record['evidence_qualification'] = {'status': 'qualified', 'product_kind': 'aggregate',
        'field_evidence': [{'field': field, 'value': record[field], 'supported': True,
                            'quote': quote, 'document_hash': 'saved123', 'locator': {'chunk_id': 'passage'}}
                           for field in ('disease', 'country', 'geographic_scope', 'reporting_period',
                                         'metric_period_start', 'metric_period_end', 'cases_confirmed')]}
    package = {'final_dataset': [record], 'aggregate_dataset': [record], 'final_case_dataset': [],
               'candidate_records': [], 'context_records': [],
               'source_registry': [{'source_id': 's', 'url': record['source_url'], 'title': 'Annual dengue report'}]}
    task = {'disease': 'dengue', 'location': 'Brazil', 'start_date': '2024-01-01',
            'end_date': '2024-12-31', 'target_fields': ['cases_confirmed']}
    package['result_manifest'] = build_result_manifest(package, {'structured_task': task})
    return package


def _source_session(path, value):
    package = _package(value)
    _json(path / 'collection/final_package.json', package)
    _json(path / 'collection/evidence_chunks.json', [{'chunk_id': 'passage', 'source_id': 's',
                                                     'text': package['final_dataset'][0]['evidence_quote']}])
    _json(path / 'acquisition/saved123.json', {'source_id': 's', 'content_hash': 'saved123',
                                             'clean_text': 'Original saved source evidence for export.',
                                             'content_readable': True})
    settings = build_run_settings(config={'llm': {'model': 'configured-model'}},
                                  state={'runtime_profile': {'env': {'LLM_MODEL': 'observed-model'}}})
    _json(path / 'data/run_settings.json', settings)
    return package, settings


def _export(source, destination):
    assert cmd_export(argparse.Namespace(session_dir=str(source), output_dir=str(destination), format='both')) == 0


def test_existing_report_is_rebuilt_when_current_package_values_change(tmp_path):
    source, target = tmp_path / 'session', tmp_path / 'export'
    old, settings = _source_session(source, 10)
    write_unified_report(source, old)
    current = _package(20)
    _json(source / 'collection/final_package.json', current)
    _export(source, target)
    snapshot = json.loads((target / 'data/report_snapshot.json').read_text())
    assert snapshot['results']['cases_confirmed']['value'] == 20
    assert json.loads((target / 'data/final_dataset.json').read_text())[0]['cases_confirmed'] == 20
    assert json.loads((target / 'final_dataset.json').read_text())[0]['cases_confirmed'] == 20
    assert json.loads((target / 'data/run_settings.json').read_text()) == settings
    with zipfile.ZipFile(target / 'session_report.zip') as bundle:
        assert json.loads(bundle.read('data/report_snapshot.json'))['results']['cases_confirmed']['value'] == 20
        assert json.loads(bundle.read('data/final_dataset.json'))[0]['cases_confirmed'] == 20
    # Export must not rewrite the source session's historical report.
    assert json.loads((source / 'data/report_snapshot.json').read_text())['results']['cases_confirmed']['value'] == 10


def test_old_session_without_html_preserves_saved_settings_and_evidence_on_export(tmp_path):
    source, target = tmp_path / 'old-session', tmp_path / 'export'
    _, settings = _source_session(source, 35)
    assert not (source / 'final_report.html').exists()
    _export(source, target)
    assert json.loads((target / 'data/run_settings.json').read_text()) == settings
    catalog = json.loads((target / 'data/source_catalog.json').read_text())
    assert catalog['sources'][0]['records_by_role']['qualified'] == ['r']
    assert catalog['sources'][0]['evidence_details'][0]['supported_fields']
    assert 'Original saved source evidence for export.' in (target / 'evidence/S001_saved_text.txt').read_text()
    with zipfile.ZipFile(target / 'session_report.zip') as bundle:
        assert 'evidence/S001_saved_text.txt' in bundle.namelist()
        assert json.loads(bundle.read('data/run_settings.json')) == settings


def test_export_retires_only_known_old_reading_files_in_a_reused_destination(tmp_path):
    source, target = tmp_path / 'session', tmp_path / 'export'
    _source_session(source, 5)
    target.mkdir()
    for name in ('final_report.md', 'task_result.md', 'workflow_interpretive_report.md'):
        (target / name).write_text('Old report', encoding='utf-8')
    (target / 'notes.md').write_text('User notes', encoding='utf-8')
    (target / 'diagnostics').mkdir()
    (target / 'diagnostics/stage.md').write_text('Stage diagnostic', encoding='utf-8')
    _export(source, target)
    assert sorted(path.name for path in target.glob('*.md')) == ['notes.md']
    assert (target / 'notes.md').read_text() == 'User notes'
    assert (target / 'diagnostics/stage.md').read_text() == 'Stage diagnostic'
