"""Source-local dates and counts must survive abbreviations without borrowing scope."""
import hashlib
import json

import pytest

from data_collection_workflow.evidence_qualification import (
    assess_record_evidence, build_evidence_index, qualify_records,
)


def fixture(text, **changes):
    digest = hashlib.sha256(text.encode()).hexdigest()
    doc = dict(source_id='s', clean_text=text, content_hash=digest)
    chunk = dict(source_id='s', chunk_id='c', text=text, char_start=0, char_end=len(text))
    row = dict(record_id='r', source_id='s', chunk_id='c', disease='Cholera',
               country='Haiti', cases_confirmed=42, as_of_date='2025-01-09')
    row.update(changes)
    return row, build_evidence_index(dict(documents=[doc], evidence_chunks=[chunk]))


@pytest.mark.parametrize('month', ['Jan.', 'Sept.', 'Dec.'])
def test_short_numeric_citation_expands_across_month_abbreviation(month):
    month_number = {'Jan.':'01', 'Sept.':'09', 'Dec.':'12'}[month]
    text = f'Cholera in Haiti as of {month} 9, 2025: 42 confirmed cases, including 3 deaths.'
    row, index = fixture(text, as_of_date=f'2025-{month_number}-09', deaths=3,
                         field_provenance_json={'cases_confirmed':{'quote':'42 confirmed cases'},
                                                'deaths':{'quote':'including 3 deaths'}})
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    assert q.status == 'qualified', q.reasons
    count = next(e for e in q.field_evidence if e.field == 'cases_confirmed')
    assert count.quote == text


def test_repeated_date_without_locator_does_not_pick_a_different_observation():
    text = ('Cholera in Haiti as of Jan. 9, 2025: 42 confirmed cases. '
            'Cholera in France as of Jan. 9, 2025: 100 confirmed cases.')
    row, index = fixture(text, field_provenance_json={'as_of_date':{'quote':'Jan. 9, 2025'}})
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    assert q.status == 'candidate'
    assert 'as_of_date:ambiguous_span' in q.reasons


def test_abbreviation_does_not_merge_neighboring_country_or_period():
    text = ('Cholera in Haiti as of Jan. 9, 2025: 42 confirmed cases. '
            'Cholera in France as of Feb. 9, 2025: 100 confirmed cases.')
    row, index = fixture(text, as_of_date='2025-02-09',
                         field_provenance_json={'cases_confirmed':{'quote':'42 confirmed cases'}})
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


@pytest.mark.parametrize('with_offsets', [False, True])
def test_publication_date_is_retained_as_metadata_not_observation_endpoint(with_offsets):
    text = 'Published December 17, 2025.\nCholera in Haiti during 2025: 42 confirmed cases.'
    citation = {'quote':'December 17, 2025'}
    if with_offsets:
        citation.update(char_start=text.index(citation['quote']),
                        char_end=text.index(citation['quote'])+len(citation['quote']))
    row, index = fixture(text, reporting_period='2025', as_of_date='2025-12-17',
                         field_provenance_json={'as_of_date':citation,
                                                'cases_confirmed':{'quote':'42 confirmed cases'}})
    groups = qualify_records([row], contract={}, evidence_index=index)
    assert len(groups['qualified_records']) == 1, groups['record_qualifications']
    fixed = groups['qualified_records'][0]
    assert fixed.get('as_of_date') is None
    assert fixed['as_of_date_raw'] == '2025-12-17'
    assert fixed['source_publication_date'] == '2025-12-17'
    assert fixed['evidence_normalization_actions'][0]['reason'] == 'publication_date_is_not_observation_date'
    assert row['as_of_date'] == '2025-12-17'


def test_publication_correction_does_not_make_undated_observation_qualify():
    text = 'Published December 17, 2025.\nCholera in Haiti: 42 confirmed cases.'
    row, index = fixture(text, as_of_date='2025-12-17',
                         field_provenance_json={'as_of_date':{'quote':'December 17, 2025'}})
    groups = qualify_records([row], contract={}, evidence_index=index)
    assert groups['qualified_records'] == []
    assert 'missing_observation_period' in groups['candidate_records'][0]['evidence_qualification']['reasons']


def test_publication_date_matching_valid_observation_is_not_removed():
    text = ('Published December 17, 2025.\n'
            'Cholera in Haiti as of December 17, 2025: 42 confirmed cases.')
    row, index = fixture(text, as_of_date='2025-12-17', field_provenance_json={
        'as_of_date':{'quote':'Cholera in Haiti as of December 17, 2025: 42 confirmed cases.'}})
    fixed = qualify_records([row], contract={}, evidence_index=index)['qualified_records'][0]
    assert fixed['as_of_date'] == '2025-12-17'
    assert not fixed.get('evidence_normalization_actions')


def test_unrelated_published_article_does_not_correct_another_chunk():
    text = 'Published December 17, 2025.\nCholera in Haiti during 2025: 42 confirmed cases.'
    row, index = fixture(text, as_of_date='2025-12-17', reporting_period='2025')
    chunk = index['evidence_chunks']['c']
    chunk['char_start'] = text.index('Cholera')
    chunk['text'] = text[chunk['char_start']:]
    groups = qualify_records([row], contract={}, evidence_index=index)
    assert groups['qualified_records'] == []
    assert groups['candidate_records'][0]['as_of_date'] == '2025-12-17'


def test_explicit_since_first_case_is_cumulative_including_attached_deaths():
    text = ('Since the first Cholera case was recorded on Jan. 9, 2025, '
            'Haiti has reported 42 confirmed cases, including 3 deaths.')
    row, index = fixture(text, as_of_date=None, reporting_period='Jan. 9, 2025', deaths=3,
                         count_semantics='cumulative', statistical_count_type='cumulative',
                         metric_name='cumulative confirmed Cholera cases', metric_value=42,
                         metric_unit='count')
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    assert q.status == 'qualified', q.reasons
    row.pop('cases_confirmed')
    q = assess_record_evidence(row, contract={'required_metric_fields':['cases_confirmed']}, evidence_index=index)
    assert q.status == 'qualified', q.reasons


@pytest.mark.parametrize('text', [
    'Since the first Cholera case in Haiti on Jan. 9, 2025, 42 new confirmed cases were reported today.',
    'Cholera in Haiti during 2025: 42 confirmed cases. Since Jan. 9, 2025, France reported 100 confirmed cases.',
])
def test_cumulative_qualifier_cannot_override_new_or_sibling_counts(text):
    row, index = fixture(text, as_of_date=None, reporting_period='2025', count_semantics='cumulative')
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


def test_curated_disease_alias_is_valid_for_redundant_syndrome_label():
    text = 'Whooping cough in Haiti during 2025: 42 confirmed cases.'
    row, index = fixture(text, disease='Pertussis', virus_or_syndrome='Whooping cough',
                         as_of_date=None, reporting_period='2025')
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'qualified'
    row['virus_or_syndrome'] = 'Pertussis'
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'qualified'
    row['virus_or_syndrome'] = 'Bordetella pertussis'
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


@pytest.mark.parametrize('predicate', ['may have caused', 'did not report', 'possibly reported'])
def test_month_abbreviation_cannot_hide_negative_or_uncertain_predicate(predicate):
    text = f'Cholera in Haiti {predicate}, as of Jan. 9, 2025, 42 confirmed cases.'
    row, index = fixture(text)
    assert assess_record_evidence(row, contract={}, evidence_index=index).status == 'candidate'


def test_valid_count_with_unsupported_date_identifies_temporal_dependency():
    text = 'Since the first Cholera case was recorded on Jan. 9, Haiti has reported 42 confirmed cases.'
    row, index = fixture(text, as_of_date=None, metric_period_start='2025-01-09')
    q = assess_record_evidence(row, contract={}, evidence_index=index)
    assert q.status == 'candidate'
    assert 'metric_period_start:unbound_field_value' in q.reasons
    assert 'cases_confirmed:unbound_observation_scope' in q.reasons


def test_corrected_provenance_preserves_serialized_input_and_is_idempotent():
    text = 'Published December 17, 2025.\nCholera in Haiti during 2025: 42 confirmed cases.'
    row, index = fixture(text, reporting_period='2025', as_of_date='2025-12-17',
                         field_provenance_json=json.dumps({'as_of_date':{'quote':'December 17, 2025'}}))
    first = qualify_records([row], contract={}, evidence_index=index)['qualified_records'][0]
    assert isinstance(first['field_provenance_json'], str)
    second = qualify_records([first], contract={}, evidence_index=index)['qualified_records'][0]
    assert first == second


def test_chunk_limit_prefers_real_sentence_boundary_to_month_abbreviation():
    from data_collection_workflow.evidence_chunking import exact_chunk_specs
    observation = 'Cholera in Haiti as of Jan. 9, 2025: 42 confirmed cases.'
    text = 'Background. ' * 72 + observation + ' Subsequent context.'
    chunks = list(exact_chunk_specs({'clean_text':text}))
    assert any(observation in chunk['text'] for chunk in chunks)
    assert all(text[c['char_start']:c['char_end']] == c['text'] for c in chunks)


@pytest.mark.parametrize('citation', [
    {'document_hash':'forged', 'quote':'December 17, 2025'},
    {'chunk_id':'missing', 'quote':'December 17, 2025'},
    {'quote':'invented publication quote', 'char_start':9999},
    {'quote':'December 17, 2025', 'char_start':9999},
    {'quote':'December 17, 2025', 'char_end':9999},
    {'quote':'December 17, 2025', 'basis':'inferred_from_task'},
    {'quote':'December 17, 2025', 'source':'search'},
    {'quote':'Cholera in Haiti during 2025: 42 confirmed cases.'},
])
def test_publication_normalization_cannot_erase_invalid_explicit_citation(citation):
    text = 'Published December 17, 2025.\nCholera in Haiti during 2025: 42 confirmed cases.'
    row, index = fixture(text, reporting_period='2025', as_of_date='2025-12-17',
                         field_provenance_json={'as_of_date':citation})
    original = assess_record_evidence(row, contract={}, evidence_index=index)
    assert original.status == 'candidate'
    groups = qualify_records([row], contract={}, evidence_index=index)
    assert not groups['qualified_records']
    retained = groups['candidate_records'][0]
    assert retained['as_of_date'] == '2025-12-17'
    assert retained['field_provenance_json']['as_of_date'] == citation
    assert not retained.get('evidence_normalization_actions')
