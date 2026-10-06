import csv
import json
import re

import pytest

from data_collection_workflow.result_manifest import build_result_manifest, write_universal_outputs


def package():
    row={'record_id':'a','cases_confirmed':4,'product_kind':'aggregate','evidence_qualification':{'status':'qualified'}}
    return {'pipeline_mode':'evidence','final_dataset':[row],'final_case_dataset':[],'aggregate_dataset':[row],'candidate_records':[{'record_id':'c'}],'source_registry':[],'context_records':[]}


def test_outputs_share_manifest_counts_and_no_patient_count_claim(tmp_path):
    p=package(); p['result_manifest']=build_result_manifest(p,{'human_review_enabled':False,'recovery_stop_reason':'no_progress'})
    paths=write_universal_outputs(p,tmp_path)
    manifest=json.loads((tmp_path/'result_manifest.json').read_text())
    exported=json.loads((tmp_path/'final_package.json').read_text())
    with (tmp_path/'aggregate_dataset.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    assert len(rows)==manifest['counts']['aggregate_observations']==1
    assert manifest['counts']['individual_case_records']==0
    assert manifest['data_availability']=='qualified_aggregates_without_individual_cases'
    assert manifest['quality_status']=='qualified_under_evidence_contract'
    assert exported['result_manifest']==manifest
    assert not (tmp_path/'final_report_chinese.md').exists()
    assert (tmp_path/'final_report.md').exists()


def test_manifest_contradiction_refuses_export(tmp_path):
    p=package(); p['result_manifest']=build_result_manifest(p,{})
    p['aggregate_dataset']=[]
    with pytest.raises(ValueError,match='manifest'):
        write_universal_outputs(p,tmp_path)


def test_candidates_only_is_not_technical_failure_or_qualified_data():
    p=package(); p['final_dataset']=[];p['aggregate_dataset']=[]
    manifest=build_result_manifest(p,{})
    assert manifest['technical_completion']=='completed'
    assert manifest['data_availability']=='candidates_only'
    assert manifest['release_status']=='not_evaluated'


def test_actual_finalization_builds_manifest_and_evidence_products(monkeypatch, tmp_path):
    import hashlib
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    text = 'Pertussis in Washington, United States: 12 confirmed cases during 2024.'
    state = {'normalized_records': [{'record_id': 'a', 'source_id': 's', 'chunk_id': 'c', 'disease': 'Pertussis', 'country': 'United States', 'subnational_location': 'Washington', 'reporting_period': '2024', 'cases_confirmed': 12}],
             'documents': [{'source_id': 's', 'clean_text': text, 'content_hash': hashlib.sha256(text.encode()).hexdigest()}],
             'evidence_chunks': [{'chunk_id': 'c', 'source_id': 's', 'text': text, 'char_start': 0, 'char_end': len(text)}]}
    result = final_data_package_builder(state)
    package = result['final_data_package']
    assert result['result_manifest'] == package['result_manifest']
    assert package['result_manifest']['counts']['qualified_observations'] == 1
    assert len(package['evidence_products']['aggregate_groups']) == 1
    assert package['result_manifest']['release_status'] == 'not_evaluated'
    from data_collection_workflow.export import export_final_data_package
    exported = export_final_data_package(package, tmp_path)
    assert exported['section_counts']['aggregate_dataset'] == 1


def mixed_finalized_package(monkeypatch):
    import hashlib
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    aggregate = 'Pertussis in Washington, United States: 12 confirmed cases during 2024.'
    patient = 'Patient A in Washington, United States had Pertussis during 2024.'
    docs = [{'source_id': str(i), 'clean_text': text, 'content_hash': hashlib.sha256(text.encode()).hexdigest()} for i, text in enumerate((aggregate, patient))]
    chunks = [{'chunk_id': str(i), 'source_id': str(i), 'text': text, 'char_start': 0, 'char_end': len(text)} for i, text in enumerate((aggregate, patient))]
    common = {'disease': 'Pertussis', 'country': 'United States', 'subnational_location': 'Washington', 'reporting_period': '2024'}
    rows = [dict(common, record_id='aggregate', source_id='0', chunk_id='0', cases_confirmed=12),
            dict(common, record_id='patient', source_id='1', chunk_id='1', workflow_case_label='Patient A', case_span_quote=patient),
            dict(common, record_id='candidate', source_id='0', chunk_id='0', cases_confirmed=19)]
    return final_data_package_builder({'normalized_records': rows, 'documents': docs, 'evidence_chunks': chunks, 'human_review_enabled': False})


def test_actual_case_aggregate_candidate_finalization_export_and_all_summary_aliases(monkeypatch, tmp_path):
    from data_collection_workflow.export import export_final_data_package
    result = mixed_finalized_package(monkeypatch)
    p = result['final_data_package']
    assert p['result_manifest']['counts']['qualified_observations'] == 2
    assert p['result_manifest']['counts']['individual_case_records'] == 1
    assert p['result_manifest']['counts']['aggregate_observations'] == 1
    assert p['result_manifest']['counts']['candidate_records'] == 1
    assert p['primary_case_dataset'] == p['final_case_dataset']
    assert result['finalization_summary']['final_dataset_count'] == 2
    assert p['run_quality_summary']['primary_case_dataset_eligible_count'] == 1
    exported = export_final_data_package(p, tmp_path / 'collection')
    assert exported['section_counts']['primary_case_dataset'] == 1
    assert exported['section_counts']['final_dataset'] == 2
    for name, count in [('final_dataset', 2), ('final_case_dataset', 1), ('primary_case_dataset', 1), ('aggregate_dataset', 1), ('candidate_records', 1), ('reviewable_dataset', 1)]:
        assert len(json.loads((tmp_path / 'collection' / (name + '.json')).read_text())) == count
        with (tmp_path / 'collection' / (name + '.csv')).open(encoding='utf8', newline='') as handle:
            assert len(list(csv.DictReader(handle))) == count
    saved = json.loads((tmp_path / 'collection' / 'final_package.json').read_text())
    assert saved['artifact_manifest']['section_counts']['primary_case_dataset'] == 1
    assert saved['result_manifest'] == p['result_manifest']


def test_runner_outputs_overwrite_stale_reports_facts_console_and_printed_counts(monkeypatch, tmp_path, capsys):
    from data_collection_workflow import result_manifest as module
    result = mixed_finalized_package(monkeypatch); p = result['final_data_package']
    summary = {'final_dataset_count': 0, 'final_case_dataset_count': 0, 'primary_case_dataset_eligible_count': 0,
               'run_quality_status': 'no_data', 'artifact_paths': {}}
    for key in ('run_report', 'stable_run_report', 'final_report_chinese', 'final_report_english', 'stable_final_report_chinese', 'interpretive_report_chinese', 'interpretive_report_english', 'stable_interpretive_report_english', 'final_report_facts', 'stable_final_report_facts', 'interpretive_report_summary', 'stable_interpretive_report_summary', 'workflow_console_html', 'latest_workflow_console_html', 'workflow_console_summary_json', 'latest_workflow_console_summary_json'):
        path = tmp_path / (key + ('.json' if 'facts' in key or 'summary' in key else '.html' if 'html' in key else '.md'))
        path.write_text('STALE NO DATA')
        summary['artifact_paths'][key] = str(path)
    assert hasattr(module, 'write_universal_run_outputs')
    module.write_universal_run_outputs(p, summary, tmp_path)
    assert summary['final_dataset_count'] == 2
    assert summary['final_record_count'] == 2
    assert summary['primary_case_dataset_eligible_count'] == 1
    for key, raw_path in summary['artifact_paths'].items():
        if key.startswith(('stable_', 'latest_', 'interpretive_', 'final_report_', 'workflow_console_', 'run_report')):
            assert 'STALE' not in __import__('pathlib').Path(raw_path).read_text(encoding='utf8')
    for key in ('final_report_facts', 'interpretive_report_summary', 'workflow_console_summary_json'):
        artifact = json.loads(__import__('pathlib').Path(summary['artifact_paths'][key]).read_text())
        assert artifact['result_manifest'] == p['result_manifest']
        assert artifact['final_dataset_count'] == 2
        assert artifact['release_status'] == 'not_evaluated'
    import scripts.run_workflow as runner
    monkeypatch.setattr(runner, '_config_with_cli_overrides', lambda args: (None, {}))
    monkeypatch.setattr(runner, '_llm_enabled', lambda env: False)
    monkeypatch.setattr(runner, 'run_workflow', lambda args: summary)
    monkeypatch.setattr('sys.argv', ['runner'])
    runner.main()
    output = capsys.readouterr().out
    assert 'final_dataset_count: 2' in output
    assert 'task_result_english:' in output


def test_manifest_alias_counts_match_standalone_and_repeated_runner_exports(monkeypatch, tmp_path):
    from data_collection_workflow.export import export_final_data_package
    from data_collection_workflow.result_manifest import write_universal_run_outputs
    result = mixed_finalized_package(monkeypatch); p = result['final_data_package']
    exported = export_final_data_package(p, tmp_path / 'collection')
    summary = {'artifact_paths': {'collection_manifest': exported}}
    write_universal_run_outputs(p, summary, tmp_path)
    saved = json.loads((tmp_path / 'collection' / 'final_package.json').read_text())
    assert saved['artifact_manifest']['section_counts']['primary_case_dataset'] == 1
    assert saved['artifact_manifest']['files']['claim_comparisons_json'] == exported['files']['claim_comparisons_json']
    for key in ('primary_case_dataset', 'reviewable_dataset', 'final_dataset_post_review', 'task_aware_observation_dataset'):
        assert saved['result_manifest']['dataset_counts'][key] == len(saved[key])
    standalone = package(); standalone['result_manifest'] = build_result_manifest(standalone, {})
    write_universal_outputs(standalone, tmp_path / 'standalone')
    assert standalone['result_manifest']['dataset_counts']['task_aware_observation_dataset'] == 1


def test_equal_counts_cannot_hide_different_final_product_rows(tmp_path):
    p = package(); p['result_manifest'] = build_result_manifest(p, {})
    p['final_dataset'] = [dict(p['final_dataset'][0], cases_confirmed=99)]
    with pytest.raises(ValueError, match='partition'):
        write_universal_outputs(p, tmp_path)


def test_shared_manifest_reports_unresolved_acquisition_without_hiding_qualified_data(tmp_path):
    p = package()
    docs = [
        {'source_id': 'ordinary', 'acquisition_status': 'budget_exhausted', 'acquisition_incomplete': True,
         'budget_exhausted_kind': 'fetch_ordinary', 'content_readable': False},
        {'source_id': 'pdf', 'content_hash': 'pdf-hash', 'acquisition_status': 'budget_exhausted',
         'acquisition_incomplete': True, 'budget_exhausted_kind': 'ocr', 'content_readable': True,
         'unprocessed_pages': [2, 4]},
        {'source_id': 'dashboard', 'acquisition_status': 'budget_exhausted', 'acquisition_incomplete': True,
         'budget_exhausted_kind': 'browser', 'content_readable': False},
    ]
    p['result_manifest'] = build_result_manifest(p, {'documents': docs, 'recovery_stop_reason': 'round_limit'})
    manifest = p['result_manifest']
    acquisition = manifest['acquisition']
    assert acquisition['status'] == 'partial'
    assert acquisition['budget_deferred_document_count'] == 3
    assert acquisition['partially_readable_document_count'] == 1
    assert acquisition['budget_exhausted_causes'] == {'fetch_ordinary': 1, 'ocr': 1, 'browser': 1}
    assert acquisition['unresolved_source_ids'] == ['dashboard', 'ordinary', 'pdf']
    assert acquisition['unprocessed_page_count'] == 2
    assert manifest['data_availability'] == 'qualified_aggregates_without_individual_cases'
    assert manifest['technical_completion'] == 'completed'
    assert manifest['recovery_stop_reason'] == 'round_limit'
    write_universal_outputs(p, tmp_path)
    for name, label in [('final_report.md', 'Acquisition incomplete'),
                        ('workflow_console.html', 'Acquisition incomplete')]:
        report = (tmp_path / name).read_text(encoding='utf-8')
        assert label in report
        assert 'fetch_ordinary' in report and 'ocr' in report and 'browser' in report
        assert 'dashboard' in report and 'pdf' in report
    summary = json.loads((tmp_path / 'workflow_console_summary.json').read_text())
    assert summary['acquisition'] == acquisition


def test_manifest_excludes_resolved_stub_and_completed_same_version_but_not_other_versions():
    p = package()
    stub = {'source_id': 's', 'acquisition_incomplete': True, 'budget_exhausted_kind': 'fetch_ordinary', 'content_readable': False}
    partial = {'source_id': 's', 'content_hash': 'same', 'acquisition_incomplete': True,
               'budget_exhausted_kind': 'ocr', 'content_readable': True, 'unprocessed_pages': [2]}
    complete = {'source_id': 's', 'content_hash': 'same', 'content_readable': True, 'acquisition_status': 'readable'}
    old_partial = {**partial, 'content_hash': 'old'}
    for docs in ([stub, partial, complete, old_partial], [complete, old_partial, stub, partial]):
        result = build_result_manifest(p, {'documents': docs})['acquisition']
        assert result['budget_deferred_document_count'] == 1
        assert result['budget_exhausted_causes'] == {'ocr': 1}
        assert result['unprocessed_page_count'] == 1


def test_serialized_manifest_outputs_use_english_for_partial_provider_and_acquisition_status(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    p = package()
    state = {
        'source_registry': [{'source_id': 'source', 'url': 'https://example.org/report'}],
        'documents': [{'source_id': 'source', 'acquisition_status': 'budget_exhausted',
                       'budget_exhausted_kind': 'ocr', 'content_readable': True,
                       'acquisition_incomplete': True, 'unprocessed_pages': [2]}],
        'run_budget_ledger': {'providers': {'openai': {'status': 'halted', 'reason': 'account_limit'}}},
    }
    p['result_manifest'] = build_result_manifest(p, state)
    output_dir = Path('collection')
    write_universal_outputs(p, output_dir)
    for path in output_dir.iterdir():
        if path.suffix in {'.md', '.html', '.json', '.csv'}:
            assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', path.read_text(encoding='utf-8')), path.name
    report = (output_dir / 'final_report.md').read_text(encoding='utf-8')
    assert 'Collection is partial (provider_account_limit)' in report
    assert 'Acquisition incomplete: 1 sources and 1 documents deferred by budget' in report
    assert '1 pages unprocessed' in report
    assert not (output_dir / 'final_report_chinese.md').exists()


def test_actual_runner_projection_has_one_english_headline_and_no_chinese_artifact_routes(tmp_path, monkeypatch):
    from pathlib import Path
    from data_collection_workflow.result_manifest import write_universal_run_outputs

    monkeypatch.chdir(tmp_path)
    p = package()
    p['result_manifest'] = build_result_manifest(p, {})
    legacy_path = Path('old_report.md')
    legacy_path.write_text('Old report', encoding='utf-8')
    p['artifact_manifest'] = {'files': {'final_report_chinese': str(legacy_path)}, 'section_counts': {}}
    summary = {'artifact_paths': {'final_report_chinese': str(legacy_path)}}
    write_universal_run_outputs(p, summary, Path('run'))
    assert set(summary['task_result_summary']) == {'headline', 'answer_status'}
    assert summary['task_result_summary']['headline']
    assert 'final_report_chinese' not in summary['artifact_paths']
    assert 'final_report_chinese' not in p['artifact_manifest']['files']
    assert 'Disease collection results' in legacy_path.read_text(encoding='utf-8')
    for path in Path('run').rglob('*'):
        if path.is_file() and path.suffix in {'.md', '.html', '.json', '.csv'}:
            text = path.read_text(encoding='utf-8')
            assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text), str(path)
