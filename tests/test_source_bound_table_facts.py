"""Real parsed table cells retain metric meaning, including parenthetical splits."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
from data_collection_workflow.evidence_qualification import build_evidence_index, assess_record_evidence


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def source_row(tmp_path, label, value, **facts):
    html = f'<h1>Measles in Canada</h1><table><tr><th>Metric</th><th>Value</th></tr><tr><td>{label}</td><td>{value}</td></tr></table>'
    doc = parse_response(html.encode(), url='https://example.test/report', source_id='s', session_dir=tmp_path, content_type='text/html')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = next(c for c in state['evidence_chunks'] if c.get('row_id') is not None)
    row = {'record_id': 'r', 'source_id': 's', 'chunk_id': chunk['chunk_id'],
           'disease': 'Measles', 'country': 'Canada', **facts}
    return row, build_evidence_index(state)


def test_explicit_annual_key_value_row_supports_its_classified_case_counts(tmp_path):
    row, index = source_row(tmp_path, 'National Cases (Full Year 2025)',
        '5,355 cases (5,045 confirmed, 310 probable)',
        cases_confirmed=5045, cases_probable=310, reporting_period='Full Year 2025', count_semantics='annual')
    q = assess_record_evidence(row, contract={'start_date': '2025-01-01', 'end_date': '2025-12-31'}, evidence_index=index)
    assert q.status == 'qualified', q.reasons


@pytest.mark.parametrize('value,changes', [
    ('5,355 cases (5,045 confirmed, 310 probable)', {'cases_confirmed': 5355}),
    ('2 confirmed deaths', {'cases_confirmed': 2}),
    ('over 5,355 cases', {'cases_unspecified': 5355}),
])
def test_key_value_row_cannot_exchange_metric_or_precision(tmp_path, value, changes):
    row, index = source_row(tmp_path, 'National Cases (Full Year 2025)', value,
                            reporting_period='Full Year 2025', **changes)
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


def test_combined_prose_count_is_supported_only_as_combined(tmp_path):
    row, index = source_row(tmp_path, 'National Cases (Full Year 2025)',
        '1,069 confirmed and probable cases', cases_unspecified=1069,
        case_definition='confirmed and probable', reporting_period='Full Year 2025')
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    assert q.status == 'qualified', q.reasons


@pytest.mark.parametrize('value', [
    '10 malaria cases (4 confirmed, 6 probable)',
    '10 hospitals reporting cases (4 confirmed, 6 probable)',
    '4 confirmed and probable malaria cases',
])
def test_classification_shortcuts_cannot_borrow_wrong_disease_or_object(tmp_path, value):
    field = 'cases_unspecified' if 'and probable' in value else 'cases_confirmed'
    row, index = source_row(tmp_path, 'National Cases (Full Year 2025)', value,
                           reporting_period='Full Year 2025', **{field: 4})
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


def test_repeated_disease_quote_uses_verified_nearest_table_heading(tmp_path):
    html = '<h1>Measles in Canada during 2025</h1><h2>Measles</h2><table><tr><th>Province</th><th>Cases</th></tr><tr><td>Ontario</td><td>2</td></tr></table>'
    doc = parse_response(html.encode(), url='https://example.test/report', source_id='s', session_dir=tmp_path, content_type='text/html')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = next(c for c in state['evidence_chunks'] if c.get('row_id') is not None)
    row = {'record_id': 'r', 'source_id': 's', 'chunk_id': chunk['chunk_id'],
           'disease': 'Measles', 'country': 'Canada', 'subnational_location': 'Ontario',
           'cases_unspecified': 2, 'reporting_period': '2025',
           'field_provenance_json': {'disease': {'quote': 'Measles'}}}
    q = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    assert q.status == 'qualified', q.reasons
