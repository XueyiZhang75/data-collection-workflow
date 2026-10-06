"""Source progress is evidence contribution, never an inflated coverage claim."""
import json
import re

from data_collection_workflow.result_manifest import build_result_manifest, write_universal_outputs


def _package():
    row = {'record_id': 'q', 'source_id': 'good', 'cases_confirmed': 3,
           'evidence_qualification': {'status': 'qualified'}}
    return {'final_dataset': [row], 'final_case_dataset': [], 'aggregate_dataset': [row],
            'candidate_records': [{'record_id': 'c', 'source_id': 'uncertain'}],
            'context_records': [], 'source_registry': []}


def _state():
    return {'source_registry': [
        {'source_id': 'good', 'url': 'https://example.org/data', 'target_fit_status': 'verified_target'},
        {'source_id': 'alias', 'url': 'https://example.org/data#table'},
        {'source_id': 'uncertain', 'url': 'https://example.org/candidate'},
        {'source_id': 'wrong', 'url': 'https://example.org/wrong', 'target_fit_status': 'temporal_mismatch'},
        {'source_id': 'failed', 'url': 'https://example.org/missing', 'target_fit_status': 'verified_target'},
        {'source_id': 'new', 'url': 'https://example.org/new'}],
        'documents': [
            {'source_id': 'alias', 'clean_text': 'source evidence', 'content_readable': True},
            {'source_id': 'uncertain', 'clean_text': 'candidate evidence', 'content_readable': True},
            {'source_id': 'wrong', 'clean_text': 'outside task', 'content_readable': True},
            {'source_id': 'failed', 'content_readable': False, 'fetch_status': 'fetch_failed'}]}


def test_discovered_sources_are_split_from_usable_evidence_and_aliases_collapse():
    package, state = _package(), _state()
    manifest = build_result_manifest(package, state)
    progress = manifest['source_progress']
    assert progress['discovered_unique_sources'] == 5
    assert progress['readable_sources'] == 3
    assert progress['qualified_evidence_sources'] == 1
    assert progress['candidate_only_sources'] == 1
    assert progress['without_extracted_evidence_sources'] == 3
    assert progress['fetch_failed_sources'] == 1
    assert progress['screening_mismatch_sources'] == 1
    assert progress['not_fetched_sources'] == 1
    assert progress['source_recall'] is None  # Unknown universe: no invented percentage.
    assert manifest['coverage_status'] == 'not_assessed'
    qualified = [row for row in progress['sources'] if row['evidence_status'] == 'qualified']
    assert qualified[0]['source_ids'] == ['alias', 'good']
    assert qualified[0]['qualified_record_ids'] == ['q']


def test_source_contribution_uses_bound_field_locators_not_unverified_primary_id():
    package, state = _package(), _state()
    record = package['final_dataset'][0]
    record['source_id'] = 'wrong'
    record['evidence_qualification']['field_evidence'] = [
        {'supported': True, 'locator': {'chunk_id': 'bound'}, 'field': 'cases_confirmed'},
        {'supported': False, 'locator': {'chunk_id': 'unbound'}, 'field': 'deaths'}]
    state['evidence_chunks'] = [{'chunk_id': 'bound', 'source_id': 'good'},
                                {'chunk_id': 'unbound', 'source_id': 'wrong'}]
    manifest = build_result_manifest(package, state)
    qualified = [row for row in manifest['source_progress']['sources'] if row['evidence_status'] == 'qualified']
    assert len(qualified) == 1 and qualified[0]['canonical_url'] == 'https://example.org/data'


def test_source_status_partitions_are_stable_and_unknown_sources_not_officially_promoted():
    package, state = _package(), _state()
    first = build_result_manifest(package, state)['source_progress']
    state['source_registry'].reverse()
    state['documents'].reverse()
    second = build_result_manifest(package, state)['source_progress']
    assert first == second
    assert first['discovered_unique_sources'] == sum(first[key] for key in (
        'qualified_evidence_sources', 'candidate_only_sources', 'context_only_sources',
        'without_extracted_evidence_sources'))
    missing = next(row for row in first['sources'] if row['source_ids'] == ['failed'])
    assert missing['evidence_status'] == 'none'


def test_source_progress_identical_in_package_reports_and_console(tmp_path):
    package, state = _package(), _state()
    package['result_manifest'] = build_result_manifest(package, state)
    write_universal_outputs(package, tmp_path)
    summary = json.loads((tmp_path / 'workflow_console_summary.json').read_text())
    assert summary['source_progress'] == package['result_manifest']['source_progress']
    for filename, phrase in [('final_report.md', 'Sources contributing qualified observations: 1'),
                             ('workflow_console.html', 'Sources contributing qualified observations: 1')]:
        assert phrase in (tmp_path / filename).read_text(encoding='utf-8')


def test_serialized_source_progress_reports_have_only_english_system_text(tmp_path):
    package, state = _package(), _state()
    package['source_coverage_audit'] = {'requirements': [
        {'requirement_id': 'cases', 'qualified_record_ids': ['q']}]}
    package['result_manifest'] = build_result_manifest(package, state)
    paths = write_universal_outputs(package, tmp_path)
    reports = list(tmp_path.glob('*.md')) + list(tmp_path.glob('*.html'))
    assert reports
    for path in reports:
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', text), path.name
        assert 'Sources contributing to requested metric/scope requirements: 1' in text
        assert 'Sources contributing qualified observations: 1' in text
    assert 'final_report_chinese.md' not in paths


def test_actual_official_alias_field_carries_acquisition_and_evidence():
    package = _package()
    package['source_registry'] = [{'source_id': 'preferred', 'url': 'https://example.org/report',
                                   'official_report_alias_source_ids': ['good']}]
    state = {'documents': [{'source_id': 'good', 'content_readable': True, 'clean_text': 'data'}]}
    progress = build_result_manifest(package, state)['source_progress']
    assert progress['qualified_evidence_sources'] == progress['readable_sources'] == 1
    assert progress['not_fetched_sources'] == 0
    assert progress['sources'][0]['source_ids'] == ['good', 'preferred']


def test_error_body_legacy_status_and_shell_never_count_as_readable():
    package = _package()
    state = {'source_registry': [{'source_id': key, 'url': 'https://example.org/' + key}
                                for key in ('blocked', 'failed', 'shell', 'wrong-task')],
             'documents': [
                 {'source_id': 'blocked', 'clean_text': 'Access denied', 'acquisition_status': 'blocked', 'parse_status': 'failed'},
                 {'source_id': 'failed', 'clean_text': 'upstream failed', 'fetch_status': 'failed'},
                 {'source_id': 'shell', 'clean_text': 'Loading data', 'is_shell': True},
                 {'source_id': 'wrong-task', 'clean_text': 'Other disease data', 'content_readable': True,
                  'quality_status': 'not_task_relevant'}]}
    progress = build_result_manifest(package, state)['source_progress']
    assert progress['readable_sources'] == 1  # Readability is separate from relevance.
    assert progress['fetch_failed_sources'] == 3


def test_qualified_related_observation_does_not_claim_requested_metric_coverage():
    package, state = _package(), _state()
    package['source_coverage_audit'] = {'requirements': [
        {'requirement_id': 'deaths', 'required_metric_fields': ['deaths'], 'qualified_record_ids': []}]}
    progress = build_result_manifest(package, state)['source_progress']
    assert progress['qualified_evidence_sources'] == 1
    assert progress['task_requirement_contributing_sources'] == 0
    package['source_coverage_audit']['requirements'][0]['qualified_record_ids'] = ['q']
    progress = build_result_manifest(package, state)['source_progress']
    assert progress['task_requirement_contributing_sources'] == 1
    contributing = next(row for row in progress['sources'] if row['qualified_record_ids'])
    assert contributing['coverage_requirement_ids'] == ['deaths']
