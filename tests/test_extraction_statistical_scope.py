"""Small source snippets protect statistical scope and numerator semantics."""
import pytest

from data_collection_workflow.config import load_llm_structured_extraction_policy, load_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy, StructuredExtractionPolicy
from data_collection_workflow.nodes import extraction
from data_collection_workflow.source_assertions import count_mentions


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def official(text):
    chunk = {'source_id': 's', 'chunk_id': 'c', 'text': text, 'source_type': 'WHO_DON'}
    context = {'disease_standard_name': 'Measles', 'is_hantavirus': False, 'disease_terms': ['measles'],
               'structured_task': {'disease': 'Measles', 'location': 'Canada'},
               'disease_intelligence': {'disease_standard_name': 'Measles', 'aliases': ['measles']}}
    record, diagnostic = extraction._official_outbreak_record_from_chunk(
        chunk, 1, StructuredExtractionPolicy(**load_structured_extraction_policy()), context)
    assert record is not None, diagnostic
    return record.model_dump()


def llm_record(text, facts, **metadata):
    chunk = {'source_id': 's', 'chunk_id': 'c', 'text': text, **metadata}
    record = extraction._build_record_from_llm_output(
        LLMExtractedRecord(disease='Measles', **facts), chunk, 1,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {'disease_standard_name': 'Measles', 'is_hantavirus': False})
    assert record is not None
    return record.model_dump()


def test_province_count_retains_province_as_statistical_scope():
    row = official('In 2025, 18 measles cases have been reported in Ontario, Canada.')
    assert row['country'] == 'Canada'
    assert row['subnational_location'] == row['geographic_scope'] == 'Ontario'
    assert row['geographic_scope_type'] == 'subnational'
    assert row['geography_inference_method'] == 'explicit_province_parent_hierarchy'


@pytest.mark.parametrize('phrase', ['confirmed and probable', 'probable and confirmed', 'confirmed/probable'])
def test_combined_case_total_is_not_a_confirmed_only_count(phrase):
    row = official(f'Canada reported 18 {phrase} cases of measles during 2025.')
    assert row['cases_unspecified'] == 18
    assert row['cases_confirmed'] is None
    assert row['cases_probable'] is None
    assert row['case_definition'] == phrase


def test_national_total_with_province_list_stays_national():
    row = official('Canada reported 18 confirmed and probable cases of measles from two provinces: Ontario and Quebec.')
    assert row['country'] == row['geographic_scope'] == 'Canada'
    assert row['subnational_location'] is None


def test_travel_origin_and_publisher_do_not_become_statistical_scope():
    for text in ('Canada reported 18 measles cases after travel from Ontario.',
                 'Ontario Health published a report: Canada reported 18 measles cases.'):
        row = official(text)
        assert row['geographic_scope'] == 'Canada'
        assert row['subnational_location'] is None


def test_death_row_does_not_duplicate_deaths_as_confirmed_cases():
    row = llm_record('Total Confirmed Deaths | 2 deaths, both infants',
                     {'cases_confirmed': 2, 'deaths': 2}, chunk_kind='metric_row')
    assert row['deaths'] == 2
    assert row['cases_confirmed'] is None


def test_explicit_case_and_death_counts_remain_distinct():
    row = llm_record('Canada reported 2 confirmed measles cases and 2 deaths.',
                     {'cases_confirmed': 2, 'deaths': 2})
    assert row['cases_confirmed'] == row['deaths'] == 2


def test_numeric_province_row_does_not_become_national_or_inherit_period():
    row = llm_record('British Columbia | 2',
                     {'country': 'Canada', 'geographic_scope': 'Canada',
                      'geographic_scope_type': 'country', 'cases_unspecified': 2},
                     chunk_kind='metric_row', row_id='1', table_id='table_1',
                     table_header='Province | Number of cases',
                     reporting_period_start='2025-02-09', reporting_period_end='2025-02-15')
    assert row['geographic_scope'] == 'British Columbia'
    assert row['geographic_scope_type'] == 'subnational'
    assert row['metric_period_start'] is None
    assert row['metric_period_end'] is None


def test_metric_previous_week_does_not_compute_unstated_calendar_dates():
    row = llm_record('Positive tests | 4',
                     {'metric_name': 'Positive tests', 'metric_value': 4,
                      'source_column_label': 'Previous week'},
                     chunk_kind='metric_row', row_id='1',
                     reporting_period_start='2025-02-09', reporting_period_end='2025-02-15')
    assert row['metric_period_start'] is None
    assert row['metric_period_end'] is None
    assert row['date_reported'] is None



def test_runtime_failure_is_exported_separately_from_valid_empty_output(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION', 'false')
    runtime = RunContext(tmp_path, {'universal': {'budget_limits': {'extraction': 5}, 'extraction_reserve': 0}})
    def fail():
        raise ValueError('invalid response')
    with pytest.raises(ValueError):
        runtime.call('extraction', {'chunk_id': 'failed'}, fail)
    runtime.call('extraction', {'chunk_id': 'empty'}, lambda: {'records': []})
    with runtime.activate():
        result = extraction.structured_extraction({'evidence_chunks': [], 'structured_task': {'disease': 'Measles'}})
    assert result['structured_extraction_summary']['failed_chunk_ids'] == ['failed']
    assert set(result['extraction_attempted_chunk_ids']) == {'failed', 'empty'}

def source_table_record(tmp_path, header, row, facts, *, title='Measles in Canada, report week 7 (February 9 to February 15, 2025)'):
    from data_collection_workflow.document_acquisition import parse_response
    from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
    html = '<h1>' + title + '</h1><table><tr>' + ''.join('<th>'+cell+'</th>' for cell in header) + '</tr><tr>' + ''.join('<td>'+cell+'</td>' for cell in row) + '</tr></table>'
    doc = parse_response(html.encode(), url='https://example.test/report', source_id='s', session_dir=tmp_path, content_type='text/html')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = next(c for c in state['evidence_chunks'] if c.get('row_id') is not None)
    record = extraction._build_record_from_llm_output(LLMExtractedRecord(disease='Measles', **facts),
        chunk, 1, LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {'disease_standard_name': 'Measles', 'is_hantavirus': False, 'documents_by_source_id': {'s': [doc]}})
    assert record is not None
    return record.model_dump()


def test_selected_new_case_column_does_not_use_last_onset_week(tmp_path):
    row = source_table_record(tmp_path,
        ['Province', 'New cases in week 7', 'Total cases in 2025', 'Week of last rash onset'],
        ['Manitoba', '0', '5', 'Week 5 (January 26 to February 1)'],
        {'country': 'Canada', 'cases_unspecified': 0, 'reporting_period': 'Week 5 (January 26 to February 1)',
         'metric_period_start': '2025-01-26', 'metric_period_end': '2025-02-01'})
    assert row['source_column_label'] == 'New cases in week 7'
    assert '7' in row['reporting_period']
    assert row['metric_period_start'] == '2025-02-09'
    assert row['metric_period_end'] == '2025-02-15'
    assert row['date_reported'] is None


def test_row_observation_week_wins_over_later_report_week_without_calendar_guess(tmp_path):
    row = source_table_record(tmp_path,
        ['Epidemiological week of rash onset', 'Exposed outside Canada', 'Exposed in Canada', 'Unknown exposure'],
        ['1', '0', '5', '0'],
        {'country': 'Canada', 'cases_unspecified': 5, 'reporting_period': 'Week 7',
         'metric_period_start': '2025-02-09', 'metric_period_end': '2025-02-15'})
    assert row['source_column_label'] == 'Exposed in Canada'
    assert row['reporting_period'].lower() == 'week 1'
    assert row['metric_period_start'] is None
    assert row['metric_period_end'] is None


def test_number_of_cases_column_does_not_acquire_report_period(tmp_path):
    row = source_table_record(tmp_path, ['Province', 'Number of cases'], ['British Columbia', '2'],
                              {'country': 'Canada', 'cases_unspecified': 2})
    assert row['source_column_label'] == 'Number of cases'
    assert row['reporting_period'] is None
    assert row['metric_period_end'] is None


def test_unverified_table_header_does_not_rebind_numeric_column():
    row = llm_record('Manitoba | 0 | 5', {'cases_unspecified': 0}, chunk_kind='metric_row',
                     table_header='Province | New cases in week 7 | Total cases')
    assert row['source_column_label'] is None
    assert row['reporting_period'] is None

@pytest.mark.parametrize('place', ['Quebec', 'Qu\u00e9bec'])
def test_explicit_province_spelling_is_retained(place):
    row = official(f'In 2025, 18 measles cases were reported in {place}, Canada.')
    assert row['subnational_location'] == row['geographic_scope'] == place
    assert row['country'] == 'Canada'

def test_province_rule_extraction_normalization_and_qualification(tmp_path):
    from data_collection_workflow.document_acquisition import parse_response
    from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
    from data_collection_workflow.nodes.normalization import record_normalization
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    text = '# Measles in Canada\n\nIn 2025, 18 measles cases have been reported in Ontario.'
    doc = parse_response(text.encode(), url='https://example.test/report', source_id='s', session_dir=tmp_path, content_type='text/plain')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = next(c for c in state['evidence_chunks'] if '18' in c['text'])
    record, diagnostic = extraction._official_outbreak_record_from_chunk(chunk, 1,
        StructuredExtractionPolicy(**load_structured_extraction_policy()),
        {'disease_standard_name': 'Measles', 'disease_terms': ['measles'], 'is_hantavirus': False, 'documents_by_source_id': {'s': [doc]}})
    assert record is not None, diagnostic
    state['validated_records'] = [record.model_dump()]
    state.update(record_normalization(state))
    assert len(state['normalized_records']) == 1
    normalized = state['normalized_records'][0]
    result = assess_record_evidence(normalized, contract={}, evidence_index=build_evidence_index(state))
    assert result.status == 'qualified', result.to_dict()
    assert normalized['geographic_scope'] == 'Ontario'
    assert normalized['country'] == 'Canada'

def test_parenthesized_case_breakdown_inherits_case_object_only():
    text = 'National Cases (Full Year 2025) | 23 cases (19 confirmed, 4 probable)'
    values = {(m['field'], m['value']) for m in count_mentions(text)}
    assert values == {('cases_unspecified', 23), ('cases_confirmed', 19), ('cases_probable', 4)}
    for mention in count_mentions(text):
        assert text[mention['char_start']:mention['char_end']] == mention['span']


@pytest.mark.parametrize('text', [
    '23 deaths (19 confirmed, 4 probable)',
    '23 tests (19 confirmed, 4 probable)',
    '23 cases (2 confirmed deaths, 4 probable deaths)',
])
def test_other_parenthetical_objects_do_not_become_case_breakdowns(text):
    values = {(m['field'], m['value']) for m in count_mentions(text)}
    assert ('cases_confirmed', 19) not in values
    assert ('cases_confirmed', 2) not in values
    assert ('cases_probable', 4) not in values


def test_institution_named_for_province_does_not_establish_provincial_scope():
    row = official('University of Alberta reported 18 measles cases in Canada during 2025.')
    assert row['geographic_scope'] == 'Canada'
    assert row['subnational_location'] is None

def test_first_case_event_does_not_become_one_case_month_total():
    text = 'In January 2025, Sierra Leone reported its first mpox case in eight years.'
    row = llm_record(text, {'country': 'Sierra Leone', 'cases_unspecified': 1,
                            'reporting_period': 'January 2025', 'count_semantics': 'monthly'})
    assert row['cases_unspecified'] is None
    assert row['metric_value'] is None


def test_explicit_single_case_count_is_not_removed_as_first_case_narrative():
    row = llm_record('Canada reported 1 measles case in January 2025.',
                     {'country': 'Canada', 'cases_unspecified': 1, 'reporting_period': 'January 2025'})
    assert row['cases_unspecified'] == 1

def test_other_sentence_province_cannot_override_selected_national_count():
    text = 'Canada reported 18 measles cases in 2025. Ontario reported 5 measles cases in 2025.'
    row = llm_record(text, {'country': 'Canada', 'geographic_scope': 'Canada',
                            'geographic_scope_type': 'country', 'cases_unspecified': 18})
    assert row['geographic_scope'] == 'Canada'
    assert row['subnational_location'] is None


def test_selected_province_count_can_bind_within_multi_sentence_chunk():
    text = 'Canada reported 18 measles cases in 2025. Ontario reported 5 measles cases in 2025.'
    row = llm_record(text, {'country': 'Canada', 'cases_unspecified': 5})
    assert row['geographic_scope'] == row['subnational_location'] == 'Ontario'

def test_combined_table_header_keeps_combined_numerator():
    data = extraction._extract_from_table_text('Province | Confirmed and probable cases\nOntario | 18',
        StructuredExtractionPolicy(**load_structured_extraction_policy()))
    assert data.get('cases_unspecified') == 18
    assert data.get('cases_confirmed') is None
    assert data.get('cases_probable') is None

def test_national_short_quote_does_not_borrow_scope_from_next_paragraph():
    text = 'Canada\n\nTotal cases: 23\n\nCases continue to rise in Ontario and Quebec.'
    row = llm_record(text, {'country': 'Canada', 'cases_unspecified': 23,
                            'geographic_scope': 'Canada', 'geographic_scope_type': 'country',
                            'evidence_quote': 'Canada Total cases: 23'})
    assert row['geographic_scope'] == 'Canada'
    assert row['subnational_location'] is None


def test_count_phrase_cannot_bridge_separate_paragraphs():
    assert not count_mentions('Total cases: 23\n\nCases continue to rise in Ontario.')


def test_province_list_does_not_select_its_first_member_as_scope():
    from data_collection_workflow.geography import explicit_statistical_scope
    assert not explicit_statistical_scope('18 measles cases were reported in Ontario and Quebec.')

def test_included_province_scope_does_not_replace_parent_national_total(tmp_path):
    from data_collection_workflow.document_acquisition import parse_response
    from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    text = 'In 2025 Canada reported 23 measles cases, including 7 in Ontario.'
    doc = parse_response(text.encode(), url='https://example.test/report', source_id='s', session_dir=tmp_path, content_type='text/plain')
    doc.update(document_id='d', quality_status='usable', extraction_readiness='ready')
    state = {'documents': [doc], 'structured_task': {'disease': 'Measles'}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    chunk = state['evidence_chunks'][0]
    record = extraction._build_record_from_llm_output(
        LLMExtractedRecord(disease='Measles', country='Canada', geographic_scope='Canada',
            geographic_scope_type='country', reporting_period='2025', cases_unspecified=23,
            evidence_quote=text), chunk, 1,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {'disease_standard_name': 'Measles', 'is_hantavirus': False})
    assert record.geographic_scope == 'Canada'
    assert record.subnational_location is None
    result = assess_record_evidence(record.model_dump(), contract={}, evidence_index=build_evidence_index(state))
    assert result.status == 'qualified', result.reasons


def test_bare_province_does_not_supply_an_unstated_parent_country():
    row = llm_record('Ontario reported 18 measles cases in 2025.',
                     {'country': None, 'subnational_location': 'Ontario', 'cases_unspecified': 18})
    assert row['country'] is None
    assert row['geographic_scope'] == 'Ontario'

def test_institution_location_cannot_replace_explicit_count_scope():
    row = llm_record('The National Microbiology Laboratory in Manitoba reported 18 measles cases in Canada during 2025.',
                     {'country': 'Canada', 'geographic_scope': 'Canada',
                      'geographic_scope_type': 'country', 'cases_unspecified': 18})
    assert row['geographic_scope'] == 'Canada'
    assert row['subnational_location'] is None
