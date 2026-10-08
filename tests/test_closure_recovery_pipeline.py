"""Recovered outbreak totals retain review exclusions and honest time coverage."""
import csv
import hashlib
import json

from data_collection_workflow.evidence_qualification import qualified_coverage
from data_collection_workflow.export import write_csv_rows, write_json
from data_collection_workflow.run_quality_gates import apply_run_quality_gates
from data_collection_workflow.task_result_report import build_task_result


def saved_state():
    count = ('Since the first Cholera case was recorded on Jan. 9, Haiti has reported '
             '42 confirmed cases, including 3 deaths.')
    text = ('Published December 17, 2025.\n'
            'Haiti has declared the end of a Cholera outbreak in the country. '
            'The declaration on Tuesday meets international standards, the Health Minister told a ceremony. '
            + count)
    digest = hashlib.sha256(text.encode()).hexdigest()
    task = {'disease': 'Cholera', 'location': 'Haiti', 'start_date': '2025-01-01',
            'end_date': '2025-12-31', 'target_fields': ['cases_confirmed', 'deaths']}
    row = {'record_id': 'original', 'source_id': 'source', 'supporting_chunk_id': 'chunk',
           'disease': 'Cholera', 'country': 'Haiti', 'cases_confirmed': 42, 'deaths': 3,
           'count_semantics': 'cumulative', 'source_url': 'https://example.invalid/report',
           'field_provenance_json': {name: {'quote': count}
                                     for name in ('disease', 'country', 'cases_confirmed', 'deaths', 'count_semantics')}}
    return {'structured_task': task, 'normalized_records': [row],
            'documents': [{'source_id': 'source', 'content_hash': digest,
                           'text_hash': digest, 'clean_text': text}],
            'evidence_chunks': [{'source_id': 'source', 'chunk_id': 'chunk', 'document_hash': digest,
                                 'text': text, 'char_start': 0, 'char_end': len(text)}]}


def test_recovered_count_reaches_report_and_exports_without_annual_coverage(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    state = saved_state()
    result = apply_run_quality_gates(state)
    assert len(result['final_dataset']) == 1
    recovered = result['final_dataset'][0]
    assert recovered['recovered_from_record_id'] == 'original'
    assert recovered['outbreak_closure_date'] == '2025-12-16'
    assert not recovered.get('as_of_date') and not recovered.get('metric_period_end')
    output = build_task_result(result, {'task': state['structured_task']})
    for field, value in [('cases_confirmed', 42), ('deaths', 3)]:
        answer = output['answers'][field]
        assert answer['selected_result']['value'] == value
        assert answer['selected_result']['result_kind'] == 'completed_outbreak_total'
        assert answer['value'] is None
    requirement = {'requirement_id': 'annual', **{key: state['structured_task'][key]
                                                for key in ('disease', 'location', 'start_date', 'end_date')}}
    coverage = qualified_coverage([requirement], result['qualified_records'])
    assert not coverage['coverage_complete']
    destination = write_csv_rows(result['final_dataset'], tmp_path / 'final_dataset.csv')
    with destination.open(encoding='utf-8', newline='') as handle:
        row = next(csv.DictReader(handle))
    assert row['outbreak_closure_date'] == '2025-12-16'
    assert row['recovered_from_record_id'] == 'original'
    q = json.loads(row['evidence_qualification'])
    assert any(e['field'] == 'outbreak_closure_date' and e['supported'] for e in q['field_evidence'])
    path = write_json(result['final_dataset'], tmp_path / 'final_dataset.json')
    assert json.loads(path.read_text(encoding='utf-8')) == result['final_dataset']


def test_human_exclusion_applies_to_recovery_of_excluded_record(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    state = saved_state()
    assert len(apply_run_quality_gates(state)['final_dataset']) == 1
    state['records_excluded_by_human_review'] = [{'record_id': 'original'}]
    result = apply_run_quality_gates(state)
    assert result['final_dataset'] == []
    assert result['qualified_records'] == []
    assert result['qualified_aggregate_records'] == []


def test_requalification_does_not_duplicate_recovered_result(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    state = saved_state()
    first = apply_run_quality_gates(state)
    assert len(first['final_dataset']) == 1
    state['normalized_records'] += first['final_dataset']
    second = apply_run_quality_gates(state)
    assert [row['record_id'] for row in second['final_dataset']] == [row['record_id'] for row in first['final_dataset']]


def test_reviewed_recovery_is_revalidated_without_restoring_pre_review_value(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    state = saved_state()
    first = apply_run_quality_gates(state)
    reviewed = dict(first['final_dataset'][0], cases_confirmed=99)
    state['final_dataset_post_review'] = [reviewed]
    result = apply_run_quality_gates(state)
    assert result['final_dataset'] == []
    assert any(row['record_id'] == reviewed['record_id'] and row['cases_confirmed'] == 99
               for row in result['candidate_records'])
