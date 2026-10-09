"""Observation admission preserves source precision without enlarging the task answer."""
from copy import deepcopy
import hashlib

import pytest

from data_collection_workflow.evidence_qualification import (
    assess_record_evidence, build_evidence_index, qualify_records, qualified_coverage,
)
from data_collection_workflow.task_result_report import _full_period


YEAR = {'disease': 'Measles', 'location': 'Canada',
        'start_date': '2025-01-01', 'end_date': '2025-12-31'}


def observation(text, **values):
    doc = {'source_id': 's', 'clean_text': text,
           'content_hash': hashlib.sha256(text.encode()).hexdigest()}
    chunk = {'source_id': 's', 'chunk_id': 'c', 'text': text,
             'char_start': 0, 'char_end': len(text)}
    row = {'record_id': 'r', 'source_id': 's', 'chunk_id': 'c',
           'disease': 'Measles', 'country': 'Canada', **values}
    return row, build_evidence_index({'documents': [doc], 'evidence_chunks': [chunk]})


def test_combined_adverse_event_cases_do_not_become_disease_cases():
    row, index = observation('In 2025 Canada reported 23 confirmed and probable cases '
                            'of adverse events following measles vaccination.',
                            cases_unspecified=23, reporting_period='2025')
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_reported_month_keeps_month_precision_and_does_not_cover_month():
    row, index = observation('In January 2025, Canada reported 4 confirmed measles cases.',
                             cases_confirmed=4, date_reported='2025-01')
    original = deepcopy(row)
    groups = qualify_records([row], contract=YEAR, evidence_index=index)
    assert len(groups['qualified_records']) == 1, groups['record_qualifications']
    assert row == original
    admitted = groups['qualified_records'][0]
    assert admitted['date_reported'] == '2025-01'
    assert not _full_period(admitted, YEAR)
    assert not qualified_coverage([{'start_date': '2025-01-01', 'end_date': '2025-01-31'}], [admitted])['coverage_complete']


@pytest.mark.parametrize('value', ['2025-01-01', '2025-01-31', '2025-02', '2025-13'])
def test_month_source_cannot_supply_an_invented_day_or_different_month(value):
    row, index = observation('In January 2025, Canada reported 4 confirmed measles cases.',
                             cases_confirmed=4, date_reported=value)
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_as_of_month_is_accepted_without_creating_a_cutoff_day():
    row, index = observation('As of September 2025, Canada had 4 confirmed measles cases.',
                             cases_confirmed=4, as_of_date='2025-09')
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'qualified', result.reasons


def test_cross_year_observation_is_retained_as_overlapping_not_annual_total():
    text = 'Canada reported 4 confirmed measles cases from December 29, 2024 to January 4, 2025.'
    row, index = observation(text, cases_confirmed=4,
                             metric_period_start='2024-12-29', metric_period_end='2025-01-04')
    groups = qualify_records([row], contract=YEAR, evidence_index=index)
    assert len(groups['qualified_records']) == 1, groups['record_qualifications']
    admitted = groups['qualified_records'][0]
    assert admitted['metric_period_start'] == '2024-12-29'
    assert admitted['evidence_qualification']['task_temporal_relation'] == 'overlaps_task_boundary'
    assert not _full_period(admitted, YEAR)
    assert not qualified_coverage([YEAR], [admitted])['coverage_complete']


def test_nonoverlapping_year_is_still_rejected():
    row, index = observation('Canada reported 4 confirmed measles cases during 2024.',
                             cases_confirmed=4, reporting_period='2024')
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_unknown_date_is_not_described_as_known_out_of_range():
    row, index = observation('Canada reported 4 confirmed measles cases.', cases_confirmed=4)
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'candidate'
    assert 'start_date:unresolved_observation_period' in result.reasons
    assert 'start_date:contract_mismatch' not in result.reasons


def test_short_quote_is_resolved_inside_unique_record_sentence():
    own = 'In January 2025, Canada reported 4 confirmed measles cases.'
    row, index = observation('measles background. ' + own,
        cases_confirmed=4, date_reported='2025-01', case_span_quote=own,
        field_provenance_json={'disease': {'quote': 'measles'}})
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'qualified', result.reasons


def test_repeated_quote_without_unique_record_anchor_stays_ambiguous():
    row, index = observation('measles background. In January 2025, Canada reported 4 confirmed measles cases.',
        cases_confirmed=4, date_reported='2025-01',
        field_provenance_json={'disease': {'quote': 'measles'}})
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_unsupported_optional_pathogen_does_not_discard_independent_count():
    row, index = observation('Canada reported 4 confirmed measles cases during 2025.',
        cases_confirmed=4, reporting_period='2025', virus_or_syndrome='Unstated pathogen')
    original = deepcopy(row)
    groups = qualify_records([row], contract=YEAR, evidence_index=index)
    assert len(groups['qualified_records']) == 1, groups['record_qualifications']
    assert groups['candidate_records'][0]['record_id'] == 'r'
    repaired = groups['qualified_records'][0]
    assert repaired['recovered_from_record_id'] == 'r'
    assert not repaired.get('virus_or_syndrome')
    assert repaired['evidence_normalization_actions'][-1]['original_value'] == 'Unstated pathogen'
    assert row == original
    again = qualify_records(groups['candidate_records'] + groups['qualified_records'], contract=YEAR, evidence_index=index)
    assert len(again['qualified_records']) == 1


@pytest.mark.parametrize('change', [
    {'cases_confirmed': 5}, {'country': 'France'}, {'reporting_period': '2024'},
    {'count_semantics': 'cumulative'}, {'geographic_scope': 'France'},
])
def test_scope_count_or_semantics_errors_are_not_erased_to_pass(change):
    row, index = observation('Canada reported 4 confirmed measles cases during 2025.',
                             cases_confirmed=4, reporting_period='2025')
    row.update(change)
    assert not qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records']


def test_optional_field_with_forged_locator_is_not_silently_removed():
    row, index = observation('Canada reported 4 confirmed measles cases during 2025.',
        cases_confirmed=4, reporting_period='2025', virus_or_syndrome='Unstated pathogen',
        field_provenance_json={'virus_or_syndrome': {'quote': 'Measles', 'document_hash': 'forged'}})
    assert not qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records']


def test_province_count_with_country_heading_cannot_pass_as_national_scope():
    row, index = observation('Canada: Ontario reported 1,453 measles cases during 2025.',
        cases_unspecified=1453, reporting_period='2025', geographic_scope='Canada', geographic_scope_type='country')
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_explicit_province_has_parent_country_but_remains_provincial():
    row, index = observation('Ontario, Canada reported 1,453 measles cases during 2025.',
        cases_unspecified=1453, reporting_period='2025', subnational_location='Ontario',
        geographic_scope='Ontario', geographic_scope_type='subnational')
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'qualified', result.reasons
    from data_collection_workflow.task_result_report import _location_matches
    assert _location_matches(row, YEAR, exact=False)
    assert not _location_matches(row, YEAR, exact=True)


def test_national_count_from_named_provinces_remains_national():
    row, index = observation('Canada reported 1,453 measles cases during 2025 from Ontario, Alberta, and Quebec.',
        cases_unspecified=1453, reporting_period='2025', geographic_scope='Canada', geographic_scope_type='country')
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'qualified', result.reasons


def test_reporting_month_does_not_replace_explicit_prior_year_observation():
    row, index = observation('In January 2025, Canada reported 4 confirmed measles cases during 2024.',
        cases_confirmed=4, date_reported='2025-01')
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


def test_country_aliases_in_same_sentence_do_not_recurse():
    row, index = observation('United States (USA) reported 4 confirmed measles cases during 2025.',
        cases_confirmed=4, reporting_period='2025')
    row['country'] = 'United States of America'
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'qualified'


def test_neighboring_province_statistic_does_not_change_national_scope():
    row, index = observation('Canada reported 1,453 measles cases during 2025. Ontario reported 300 measles cases during 2025.',
        cases_unspecified=1453, reporting_period='2025', geographic_scope='Canada', geographic_scope_type='country')
    result = assess_record_evidence(row, contract=YEAR, evidence_index=index)
    assert result.status == 'qualified', result.reasons


@pytest.mark.parametrize('offsets', [
    {'char_start': 99999, 'char_end': 100006}, {'char_start': None}, {'char_end': 1},
])
def test_recovery_cannot_hide_invalid_explicit_quote_offsets(offsets):
    row, index = observation('Canada reported 4 confirmed measles cases during 2025.',
        cases_confirmed=4, reporting_period='2025', virus_or_syndrome='Unstated pathogen',
        field_provenance_json={'virus_or_syndrome': {'quote': 'measles', **offsets}})
    assert not qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records']


def test_bare_year_label_does_not_establish_a_full_year_answer():
    row, index = observation('Canada reported 500 measles cases in 2025.',
        cases_unspecified=500, reporting_period='2025', source_url='https://example.test/count')
    admitted = qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records'][0]
    assert not _full_period(admitted, YEAR)
    assert not qualified_coverage([YEAR], [admitted])['coverage_complete']
    from data_collection_workflow.task_result_report import build_task_result
    report = build_task_result({'structured_task': {**YEAR, 'target_fields': ['cases_unspecified']},
                                'final_dataset': [admitted]}, {'task': {**YEAR, 'target_fields': ['cases_unspecified']}})
    answer = report['answers']['cases_unspecified']
    assert answer['status'] == 'unconfirmed'
    assert answer['selected_result']['result_kind'] == 'reported_period'
    assert answer['selected_result']['period_end'] is None


def test_explicit_full_year_label_supports_annual_answer_without_exact_endpoint_fields():
    row, index = observation('For the full year 2025, Canada reported 500 measles cases.',
        cases_unspecified=500, reporting_period='2025', source_url='https://example.test/annual')
    admitted = qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records'][0]
    assert _full_period(admitted, YEAR)
    assert qualified_coverage([YEAR], [admitted])['coverage_complete']


@pytest.mark.parametrize('text,field', [
    ('Canada had 4 hospitals reporting confirmed measles cases during 2025.', 'cases_confirmed'),
    ('Canada had 4 schools with measles cases during 2025.', 'cases_unspecified'),
])
def test_counts_of_institutions_are_not_case_counts(text, field):
    row, index = observation(text, reporting_period='2025', **{field: 4})
    assert assess_record_evidence(row, contract=YEAR, evidence_index=index).status == 'candidate'


@pytest.mark.parametrize('suffix', ['in the annual vaccination campaign.', '; annual vaccination campaigns continued.'])
def test_unrelated_annual_activity_does_not_establish_annual_case_total(suffix):
    row, index = observation('Canada reported 4 confirmed measles cases during 2025 ' + suffix,
        cases_confirmed=4, reporting_period='2025')
    admitted = qualify_records([row], contract=YEAR, evidence_index=index)['qualified_records'][0]
    assert not _full_period(admitted, YEAR)
