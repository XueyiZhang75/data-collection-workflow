"""Regression probes for source-local qualification and comparable evidence products."""
import hashlib

import pytest

from data_collection_workflow.evidence_products import build_evidence_products
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index


def observation(text, *, source='s', **changes):
    row = dict(record_id=source, source_id=source, chunk_id=source, disease='Pertussis',
               country='United States', subnational_location='Washington',
               reporting_period='2024', cases_confirmed=12)
    row.update(changes)
    doc = dict(source_id=source, clean_text=text, content_hash=hashlib.sha256(text.encode()).hexdigest())
    chunk = dict(source_id=source, chunk_id=source, text=text, char_start=0, char_end=len(text))
    index = build_evidence_index(dict(documents=[doc], evidence_chunks=[chunk]))
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    row['evidence_qualification'] = q.to_dict()
    return row, index


def status(row):
    return row['evidence_qualification']['status']


@pytest.mark.parametrize('disease,country,place', [('Pertussis', 'United States', 'Washington'), ('Cholera', 'Haiti', 'Artibonite')])
@pytest.mark.parametrize('predicate', ['did not cause', 'may cause', 'could cause', 'possibly caused'])
def test_uncertain_or_negated_counts_remain_candidates(disease, country, place, predicate):
    row, index = observation(f'{disease} in {place}, {country} {predicate} 12 confirmed cases during 2024.',
                             disease=disease, country=country, subnational_location=place)
    assert status(row) == 'candidate'
    assert build_evidence_products([row], evidence_index=index)['aggregate_groups'] == []


def test_conditional_counts_remain_candidates():
    row, _ = observation('If Pertussis in Washington, United States caused 12 confirmed cases during 2024, an alert would be issued.')
    assert status(row) == 'candidate'


@pytest.mark.parametrize('other', ['His mother was aged 45.', 'Her father was aged 45.', 'A visitor was aged 45.'])
def test_unlabeled_other_person_cannot_supply_patient_age(other):
    text = 'Patient A in Washington, United States had Pertussis during 2024. ' + other
    row, _ = observation(text, cases_confirmed=None, workflow_case_label='Patient A', age='45', case_span_quote=text)
    assert status(row) == 'candidate'


@pytest.mark.parametrize('field', ['as_of_date', 'event_start_date', 'event_end_date', 'metric_period_start', 'metric_period_end'])
def test_invalid_calendar_dates_remain_candidates(field):
    row, _ = observation('Pertussis in Washington, United States caused 12 confirmed cases on 2024-02-31.',
                         reporting_period=None, **{field: '2024-02-31'})
    assert status(row) == 'candidate'


@pytest.mark.parametrize('field,label', [('date_onset', 'symptom onset on'), ('date_confirmation', 'confirmed on'), ('date_death', 'died on')])
def test_invalid_clinical_calendar_dates_remain_candidates(field, label):
    text = f'Patient A in Washington, United States had Pertussis during 2024; {label} 2024-02-31.'
    row, _ = observation(text, cases_confirmed=None, workflow_case_label='Patient A', case_span_quote=text, **{field: '2024-02-31'})
    assert status(row) == 'candidate'


def test_explicit_identity_quote_is_a_patient_span_without_redundant_metadata():
    row, index = observation('Patient A, aged 45, in Washington, United States had Pertussis during 2024.',
                             cases_confirmed=None, workflow_case_label='Patient A', age='45')
    assert status(row) == 'qualified'
    products = build_evidence_products([row], evidence_index=index)
    assert len(products['case_entities']) == 1
    assert products['excluded_observations'] == []


def test_explicit_unresolvable_patient_span_is_candidate():
    row, _ = observation('Patient A in Washington, United States had Pertussis during 2024.',
                         cases_confirmed=None, workflow_case_label='Patient A', case_span_quote='Patient Z')
    assert status(row) == 'candidate'


@pytest.mark.parametrize('rows', ['Oregon | 12\nWashington | 100', 'Washington | 100\nOregon | 12'])
def test_table_count_must_share_row_with_geography(rows):
    row, _ = observation('Pertussis cases in United States during 2024\nState | Confirmed cases\n' + rows)
    assert status(row) == 'candidate'


@pytest.mark.parametrize('rows', ['Influenza | 12\nPertussis | 100', 'Pertussis | 100\nInfluenza | 12'])
def test_table_count_must_share_row_with_disease(rows):
    row, _ = observation('Cases in Washington, United States during 2024\nDisease | Confirmed cases\n' + rows)
    assert status(row) == 'candidate'


@pytest.mark.parametrize('rows', ['Oregon | 100\nWashington | 12', 'Washington | 12\nOregon | 100'])
def test_supported_table_row_is_order_invariant(rows):
    row, _ = observation('Pertussis cases in United States during 2024\nState | Confirmed cases\n' + rows)
    assert status(row) == 'qualified'


@pytest.mark.parametrize('field,left,right', [('metric_name', 'deaths', 'hospitalizations'), ('metric_unit', 'persons', 'reports'), ('count_semantics', 'incident', 'cumulative')])
def test_distinct_metric_semantics_never_share_aggregate_group(field, left, right):
    rows, documents, chunks = [], [], {}
    for i, value in enumerate((left, right)):
        values = dict(cases_confirmed=None, metric_name='hospitalizations', metric_value=12, metric_unit='persons', count_semantics='incident')
        values[field] = value
        text = f"Pertussis in Washington, United States during 2024: 12 {values['metric_name']}; {values['metric_unit']}; {values['count_semantics']}."
        row, index = observation(text, source=str(i), **values)
        assert status(row) == 'qualified'
        rows.append(row)
        documents.extend(index['documents'])
        chunks.update(index['evidence_chunks'])
    index = dict(documents=documents, evidence_chunks=chunks)
    products = build_evidence_products(rows, evidence_index=index)
    assert len(products['aggregate_groups']) == 2
    assert products == build_evidence_products(list(reversed(rows)), evidence_index=index)


def test_positive_leap_day_and_affirmative_count_stay_qualified():
    row, _ = observation('Pertussis in Washington, United States caused 12 confirmed cases on 2024-02-29.',
                         reporting_period=None, as_of_date='2024-02-29')
    assert status(row) == 'qualified'

@pytest.mark.parametrize('field,value,label', [('hospitalized', True, 'not hospitalized: yes'), ('age', '45', 'not aged 45')])
def test_negated_clinical_assertions_remain_candidates(field, value, label):
    text = f'Patient A in Washington, United States had Pertussis during 2024 and was {label}.'
    row, _ = observation(text, cases_confirmed=None, workflow_case_label='Patient A', case_span_quote=text, **{field: value})
    assert status(row) == 'candidate'


def test_exact_model_counts_and_existing_unit_fields_support_proven_partition():
    from data_collection_workflow.models import PublicHealthRecord
    rows, documents, chunks = [], [], {}
    for member, count in [('Children', 3), ('Adults', 7)]:
        text = f'Pertussis in Washington, United States during 2024: {count} confirmed cases among {member}; persons; incident.'
        raw, index = observation(text, source=member, cases_confirmed=count, count_unit='persons',
                                 statistical_count_type='incident', count_semantics='incident', population_scope=member, case_definition='confirmed')
        row = PublicHealthRecord(**raw).model_dump()
        row['supporting_chunk_id'] = member
        row['evidence_qualification'] = assess_record_evidence(row, contract={}, evidence_index=index).to_dict()
        assert status(row) == 'qualified'
        rows.append(row)
        documents.extend(index['documents'])
        chunks.update(index['evidence_chunks'])
    declaration = dict(axis='population_scope', members=['Children', 'Adults'], target='All residents',
                       metric='cases_confirmed', mutually_exclusive=True, exhaustive=True)
    proof, index = observation('Pertussis in Washington, United States during 2024. Children and Adults are mutually exclusive and exhaustive for All residents; confirmed cases.',
                              source='proof', cases_confirmed=None, disjoint_partition=declaration)
    assert status(proof) == 'qualified'
    documents.extend(index['documents'])
    chunks.update(index['evidence_chunks'])
    out = build_evidence_products(rows + [proof], evidence_index=dict(documents=documents, evidence_chunks=chunks))
    assert len(out['derivations']) == 1
    assert out['derivations'][0]['value'] == 10
    assert len(out['derivations'][0]['parent_observation_ids']) == 2

@pytest.mark.parametrize('identity_label', ['study numbered', 'report ID', 'reference number'])
def test_numbered_identifier_is_not_observation_year(identity_label):
    row, _ = observation(f'Pertussis in Washington, United States caused 12 confirmed cases in a {identity_label} 2024; the cases occurred during 2023.')
    assert status(row) == 'candidate'


@pytest.mark.parametrize('origin_phrase', ['visitors from', 'people who travelled from', 'nationals of'])
def test_travel_or_nationality_country_is_not_event_country(origin_phrase):
    row, _ = observation(f'Pertussis in Washington, United States caused 12 confirmed cases during 2024 in {origin_phrase} Canada.', country='Canada')
    assert status(row) == 'candidate'


def test_explicit_event_country_and_period_survive_origin_and_identifier_mentions():
    row, _ = observation('Pertussis in Washington, United States caused 12 confirmed cases during 2024 among visitors from Canada; report ID 2023.')
    assert status(row) == 'qualified'


def test_repeated_country_with_explicit_event_role_is_qualified():
    row, _ = observation('Pertussis in Ontario, Canada caused 12 confirmed cases during 2024 among returning visitors from Canada.',
                         country='Canada', subnational_location='Ontario')
    assert status(row) == 'qualified'


def partition_observations(*, count_semantics='incident', proof_changes=None):
    rows, documents, chunks = [], [], {}
    for member, count in [('Children', 3), ('Adults', 7)]:
        text = f'Pertussis in Washington, United States during 2024: {count} confirmed cases among {member}; persons. Count semantics: {count_semantics}.'
        row, index = observation(text, source=member, cases_confirmed=count, count_unit='persons',
                                 count_semantics=count_semantics, population_scope=member, case_definition='confirmed')
        assert status(row) == 'qualified'
        rows.append(row)
        documents.extend(index['documents'])
        chunks.update(index['evidence_chunks'])
    fields = dict(disease='Pertussis', country='United States', subnational_location='Washington', reporting_period='2024')
    fields.update(proof_changes or {})
    declaration = dict(axis='population_scope', members=['Children', 'Adults'], target='All residents',
                       metric='cases_confirmed', mutually_exclusive=True, exhaustive=True)
    text = f"{fields['disease']} in {fields['subnational_location']}, {fields['country']} during {fields['reporting_period']}. Children and Adults are mutually exclusive and exhaustive for All residents; confirmed cases."
    proof, index = observation(text, source='proof', cases_confirmed=None, disjoint_partition=declaration, **fields)
    assert status(proof) == 'qualified'
    documents.extend(index['documents'])
    chunks.update(index['evidence_chunks'])
    return rows + [proof], dict(documents=documents, evidence_chunks=chunks)


@pytest.mark.parametrize('scope_change', [{'disease': 'Cholera'}, {'country': 'Canada'}, {'subnational_location': 'Oregon'}, {'reporting_period': '2023'}])
def test_inapplicable_partition_scope_never_derives(scope_change):
    rows, index = partition_observations(proof_changes=scope_change)
    assert build_evidence_products(rows, evidence_index=index)['derivations'] == []


@pytest.mark.parametrize('semantics', ['unknown', 'unspecified'])
def test_unknown_partition_count_semantics_never_derives(semantics):
    rows, index = partition_observations(count_semantics=semantics)
    assert build_evidence_products(rows, evidence_index=index)['derivations'] == []


def test_same_scope_known_semantics_partition_derives_once():
    rows, index = partition_observations()
    products = build_evidence_products(rows, evidence_index=index)
    assert [row['value'] for row in products['derivations']] == [10]
    assert products == build_evidence_products(rows + rows, evidence_index=index)


@pytest.mark.parametrize('period', ['2024-W01', '2024 Q1', 'January 2024', 'MMWR week 40, 2024', 'week ending November 2, 2024', '2024-01-01 to 2024-01-07'])
def test_source_exact_reporting_period_labels_remain_qualified(period):
    row, _ = observation(f'Pertussis in Washington, United States caused 12 confirmed cases during {period}.', reporting_period=period)
    assert status(row) == 'qualified'


@pytest.mark.parametrize('period', ['2024-02-31', '2024-13', '2024-W54', '2024-02-31 to 2024-03-07'])
def test_invalid_calendar_values_in_period_labels_remain_candidates(period):
    row, _ = observation(f'Pertussis in Washington, United States caused 12 confirmed cases during {period}.', reporting_period=period)
    assert status(row) == 'candidate'

@pytest.mark.parametrize('suffix', ['during May 2024', 'and no deaths during 2024', '; no deaths during 2024'])
def test_unrelated_negative_or_month_does_not_reject_affirmative_count(suffix):
    row, _ = observation(f'Pertussis in Washington, United States caused 12 confirmed cases {suffix}.')
    assert status(row) == 'qualified'


def test_other_table_column_negation_does_not_reject_case_count():
    row, _ = observation('Pertussis in United States during 2024\nState | Confirmed cases | Deaths\nWashington | 12 | no deaths')
    assert status(row) == 'qualified'


def test_other_clinical_predicate_negation_does_not_reject_age():
    text = 'Patient A, aged 45, in Washington, United States had Pertussis during 2024 but was not hospitalized.'
    row, _ = observation(text, cases_confirmed=None, workflow_case_label='Patient A', age='45', case_span_quote=text)
    assert status(row) == 'qualified'


@pytest.mark.parametrize('text', [
    'Pertussis in Washington, United States during 2024: 12 confirmed cases were not reported.',
    'If an outbreak occurred, Pertussis in Washington, United States would cause 12 confirmed cases during 2024.',
])
def test_post_count_negation_and_leading_condition_remain_candidates(text):
    row, _ = observation(text)
    assert status(row) == 'candidate'


def test_unknown_partition_unit_never_derives():
    rows, index = partition_observations()
    for row in rows[:-1]:
        row['count_unit'] = 'unknown'
        text = next(d['clean_text'] for d in index['documents'] if d['source_id'] == row['source_id'])
        replacement, replacement_index = observation(text + ' Unit: unknown.', source=row['source_id'], cases_confirmed=row['cases_confirmed'],
                                                    count_unit='unknown', count_semantics='incident', population_scope=row['population_scope'], case_definition='confirmed')
        assert status(replacement) == 'qualified'
        row.clear()
        row.update(replacement)
        index['documents'] = [d for d in index['documents'] if d['source_id'] != row['source_id']] + replacement_index['documents']
        index['evidence_chunks'].update(replacement_index['evidence_chunks'])
    assert build_evidence_products(rows, evidence_index=index)['derivations'] == []


def test_later_negative_clinical_sentence_does_not_reject_prior_age():
    text = 'Patient A, aged 45, in Washington, United States had Pertussis during 2024. He was not hospitalized.'
    row, _ = observation(text, cases_confirmed=None, workflow_case_label='Patient A', age='45', case_span_quote=text)
    assert status(row) == 'qualified'


@pytest.mark.parametrize('prefix', [
    'Pertussis in United States during 2024 could cause the following cases',
    'Pertussis in United States during 2024 may cause the following cases',
    'The following Pertussis counts in United States during 2024 were not reported',
    'Hypothetical Pertussis cases in United States during 2024',
])
def test_uncertain_table_context_cannot_assert_numeric_cells(prefix):
    row, index = observation(prefix + '\nState | Confirmed cases\nWashington | 12')
    assert status(row) == 'candidate'
    assert build_evidence_products([row], evidence_index=index)['aggregate_groups'] == []


def test_affirmative_may_table_context_and_other_negative_column_are_supported():
    row, index = observation('Pertussis cases in United States during May 2024\nState | Confirmed cases | Deaths\nWashington | 12 | no deaths')
    assert status(row) == 'qualified'
    assert len(build_evidence_products([row], evidence_index=index)['aggregate_groups']) == 1
