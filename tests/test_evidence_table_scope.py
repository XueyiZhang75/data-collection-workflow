"""Temporal evidence must belong to the selected numerator's table axis."""
import pytest

from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
from data_collection_workflow.evidence_qualification import _local_span, assess_record_evidence, build_evidence_index


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def source_local_table(tmp_path, headers, row, *, heading=None, return_state=False):
    title = heading or 'Measles in Canada, report week 7 (February 9 to February 15, 2025)'
    html = '<h1>' + title + '</h1><table><tr>' + ''.join('<th>'+v+'</th>' for v in headers)
    html += '</tr><tr>' + ''.join('<td>'+v+'</td>' for v in row) + '</tr></table>'
    doc = parse_response(html.encode(), url='https://example.test/report', source_id='s',
                         session_dir=tmp_path, content_type='text/html')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = next(c for c in state['evidence_chunks'] if c.get('row_id') is not None)
    record = {'record_id': 'r', 'source_id': 's', 'supporting_chunk_id': chunk['chunk_id'],
              'evidence_quote': chunk['text']}
    local, _, _, error = _local_span(record, {'quote': chunk['text']}, build_evidence_index(state))
    assert not error
    assert headers[1] in local and row[-1] in local
    return (local, record, state) if return_state else local


def decide(name, value, text, record):
    from data_collection_workflow.evidence_table_scope import table_observation_scope_support
    return table_observation_scope_support(name, value, text, record)


@pytest.fixture
def weekly(tmp_path):
    return source_local_table(tmp_path,
        ['Province', 'New cases in week 7', 'Total cases in 2025', 'Week of last rash onset'],
        ['Manitoba', '0', '5', 'Week 5 (January 26 to February 1, 2025)'])


@pytest.mark.parametrize('field, value', [
    ('metric_period_start', '2025-02-09'), ('metric_period_end', '2025-02-15'),
    ('reporting_period', 'Week 7'),
    ('reporting_period', 'Week 7 (February 9 to February 15, 2025)'),
])
def test_selected_new_case_week_uses_only_matching_heading_dates(weekly, field, value):
    row = {'cases_unspecified': 0, 'source_column_label': 'New cases in week 7'}
    assert decide(field, value, weekly, row) is True


@pytest.mark.parametrize('field, value', [
    ('metric_period_start', '2025-01-26'), ('metric_period_end', '2025-02-01'),
    ('reporting_period', 'Week 5 (January 26 to February 1, 2025)'),
    ('as_of_date', '2025-02-15'), ('date_reported', '2025-02-15'),
])
def test_new_cases_cannot_borrow_last_onset_column_or_turn_period_end_into_cutoff(weekly, field, value):
    assert decide(field, value, weekly, {'cases_unspecified': 0}) is False


def test_numeric_scope_consistency_rejects_compatible_number_with_wrong_dates(weekly):
    row = {'cases_unspecified': 0, 'reporting_period': 'Week 7',
           'metric_period_start': '2025-01-26', 'metric_period_end': '2025-02-01'}
    assert decide('cases_unspecified', 0, weekly, row) is False
    row.update(metric_period_start='2025-02-09', metric_period_end='2025-02-15')
    assert decide('cases_unspecified', 0, weekly, row) is True


def test_row_week_outweighs_report_heading_without_calendar_inference(tmp_path):
    text = source_local_table(tmp_path,
        ['Epidemiological week of rash onset', 'Exposed outside Canada', 'Exposed in Canada', 'Unknown exposure'],
        ['1', '0', '5', '0'])
    row = {'cases_unspecified': 5, 'source_column_label': 'Exposed in Canada'}
    assert decide('reporting_period', 'Week 1', text, row) is True
    for field, value in [('reporting_period', 'Week 7'), ('reporting_period', '2025-W01'),
                         ('metric_period_start', '2025-02-09'), ('metric_period_end', '2025-02-15')]:
        assert decide(field, value, text, row) is False


def test_generic_case_column_cannot_acquire_report_week_as_observation_period(tmp_path):
    text = source_local_table(tmp_path, ['Province', 'Number of cases'], ['British Columbia', '2'])
    row = {'cases_unspecified': 2}
    assert decide('reporting_period', 'Week 7', text, row) is False
    assert decide('as_of_date', '2025-02-15', text, row) is False


def test_model_column_label_cannot_override_numeric_cell_header(weekly):
    row = {'cases_unspecified': 0, 'source_column_label': 'Total cases in 2025'}
    assert decide('metric_period_start', '2025-02-09', weekly, row) is False
    assert decide('cases_unspecified', 0, weekly, row) is False


def test_no_decision_for_non_table_or_table_without_week_axis():
    assert decide('metric_period_end', '2025-02-15', 'Canada reported 2 cases.', {'cases_unspecified': 2}) is None
    assert decide('reporting_period', '2025', 'Province | Cases\nOntario | 2', {'cases_unspecified': 2}) is None


def test_same_value_in_competing_case_columns_does_not_trust_model_choice(tmp_path):
    text = source_local_table(tmp_path, ['Province', 'New cases in week 7', 'Cases in week 5'], ['Ontario', '0', '0'])
    assert decide('reporting_period', 'Week 7', text, {'cases_unspecified': 0,
                  'source_column_label': 'New cases in week 7'}) is False


def weekly_observation(tmp_path):
    _, row, state = source_local_table(tmp_path,
        ['Province', 'New cases in week 7', 'Total cases in 2025', 'Week of last rash onset'],
        ['Manitoba', '0', '5', 'Week 5 (January 26 to February 1, 2025)'], return_state=True)
    row.update(disease='Measles', country='Canada', subnational_location='Manitoba',
               geographic_scope='Manitoba', geographic_scope_type='subnational',
               cases_unspecified=0, source_column_label='New cases in week 7',
               reporting_period='Week 7 (February 9 to February 15, 2025)',
               metric_period_start='2025-02-09', metric_period_end='2025-02-15')
    return row, build_evidence_index(state)


def test_corrected_weekly_observation_fully_qualifies_from_actual_parsed_source(tmp_path):
    row, index = weekly_observation(tmp_path)
    result = assess_record_evidence(row, contract={}, evidence_index=index)
    assert result.status == 'qualified', result.to_dict()
    assert any(item.field == 'cases_unspecified' and item.value == 0 and item.supported
               for item in result.field_evidence)


def test_direct_record_with_other_columns_week_remains_candidate(tmp_path):
    row, index = weekly_observation(tmp_path)
    row.update(reporting_period='Week 5 (January 26 to February 1, 2025)',
               metric_period_start='2025-01-26', metric_period_end='2025-02-01')
    result = assess_record_evidence(row, contract={}, evidence_index=index)
    assert result.status == 'candidate'
    assert any(item.field == 'cases_unspecified' and not item.supported for item in result.field_evidence)


def test_direct_record_cannot_override_true_numeric_header_with_model_label(tmp_path):
    row, index = weekly_observation(tmp_path)
    row['source_column_label'] = 'Total cases in 2025'
    result = assess_record_evidence(row, contract={}, evidence_index=index)
    assert result.status == 'candidate'
    assert any(item.field == 'cases_unspecified' and not item.supported for item in result.field_evidence)


def test_generic_case_table_without_observation_period_remains_candidate(tmp_path):
    _, row, state = source_local_table(tmp_path, ['Province', 'Number of cases'], ['British Columbia', '2'],
                                     return_state=True)
    row.update(disease='Measles', country='Canada', subnational_location='British Columbia',
               geographic_scope='British Columbia', geographic_scope_type='subnational', cases_unspecified=2)
    result = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    assert result.status == 'candidate'
    assert 'missing_observation_period' in result.reasons


@pytest.mark.parametrize('field', ['count_semantics', 'statistical_count_type'])
def test_new_case_semantics_bind_to_selected_header_only(weekly, field):
    assert decide(field, 'newly_reported', weekly, {'cases_unspecified': 0}) is True
    assert decide(field, 'cumulative', weekly, {'cases_unspecified': 0}) is False
    assert decide(field, 'newly_reported', weekly, {'cases_unspecified': 5}) is False


@pytest.mark.parametrize('field', ['count_semantics', 'statistical_count_type'])
def test_explicit_cumulative_column_supports_cumulative_without_week_axis(field):
    text = 'Province | New cases | Cumulative cases\nOntario | 2 | 12'
    assert decide(field, 'cumulative', text, {'cases_unspecified': 12}) is True
    assert decide(field, 'cumulative', text, {'cases_unspecified': 2}) is False
    assert decide(field, 'newly_reported', text, {'cases_unspecified': 12}) is False


@pytest.mark.parametrize('semantics', ['annual', 'cumulative', 'newly_reported'])
def test_generic_total_header_does_not_invent_count_semantics(semantics):
    text = 'Province | Total cases in 2025\nOntario | 12'
    assert decide('count_semantics', semantics, text, {'cases_unspecified': 12}) is False


def test_complex_key_value_table_defers_to_existing_assertion_validator():
    text = 'Measure | Value\nNational Cases (Full Year 2025) | 5,355 cases (5,045 confirmed, 310 probable)'
    row = {'cases_confirmed': 5045, 'cases_probable': 310, 'cases_unspecified': 5355}
    assert decide('count_semantics', 'annual', text, row) is None


def test_real_extraction_normalization_and_qualification_retains_new_case_semantics(tmp_path):
    from data_collection_workflow.config import load_llm_structured_extraction_policy
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output
    from data_collection_workflow.nodes.normalization import record_normalization

    _, _, state = source_local_table(tmp_path,
        ['Province', 'New cases in week 7', 'Total cases in 2025', 'Week of last rash onset'],
        ['Manitoba', '0', '5', 'Week 5 (January 26 to February 1, 2025)'], return_state=True)
    chunk = next(c for c in state['evidence_chunks'] if c.get('row_id') is not None)
    doc = state['documents'][0]
    output = _build_record_from_llm_output(
        LLMExtractedRecord(disease='Measles', country='Canada', subnational_location='Manitoba',
            geographic_scope='Manitoba', geographic_scope_type='subnational', cases_unspecified=0,
            reporting_period='Week 7 (February 9 to February 15, 2025)',
            metric_period_start='2025-02-09', metric_period_end='2025-02-15',
            count_semantics='newly_reported', statistical_count_type='newly_reported'),
        chunk, 1, LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {'disease_standard_name': 'Measles', 'is_hantavirus': False, 'documents_by_source_id': {'s': [doc]}})
    assert output is not None
    state['validated_records'] = [output.model_dump()]
    state.update(record_normalization(state))
    row = state['normalized_records'][0]
    assert row['count_semantics'] == row['statistical_count_type'] == 'newly_reported'
    result = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    assert result.status == 'qualified', result.to_dict()
