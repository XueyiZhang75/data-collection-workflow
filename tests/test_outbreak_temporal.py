import hashlib

import pytest


def fixture(*, publication='Published December 17, 2025.', weekday='Tuesday',
            closure='Haiti has declared the end of a Cholera outbreak in the country.',
            declaration=None, count=None):
    declaration = declaration or f'The declaration on {weekday} meets international standards, the Health Minister told a ceremony.'
    count = count or 'Since the first Cholera case was recorded on Jan. 9, Haiti has reported 42 confirmed cases, including 3 deaths.'
    text = '\n'.join((publication, closure, declaration, count))
    digest = hashlib.sha256(text.encode()).hexdigest()
    doc = {'source_id':'s', 'content_hash':digest, 'text_hash':digest, 'clean_text':text}
    chunk = {'source_id':'s', 'chunk_id':'c', 'document_hash':digest,
             'text':text, 'char_start':0, 'char_end':len(text)}
    row = {'record_id':'r', 'source_id':'s', 'supporting_chunk_id':'c', 'disease':'Cholera',
           'source_url':'https://example.invalid/report',
           'country':'Haiti', 'cases_confirmed':42, 'deaths':3,
           'field_provenance_json':{'cases_confirmed':{'quote':count}, 'deaths':{'quote':count}}}
    return row, {'documents':[doc], 'evidence_chunks':{'c':chunk}}


def resolve(row, index):
    from data_collection_workflow.outbreak_temporal import resolve_outbreak_closure_date
    return resolve_outbreak_closure_date(row, evidence_index=index)


def test_source_relative_weekday_retains_both_literal_anchors():
    row, index = fixture()
    result = resolve(row, index)
    assert result is not None
    assert result['value'] == '2025-12-16'
    assert result['field'] == 'outbreak_closure_date'
    assert result['basis'] == 'derived_relative_date'
    assert result['rule'] == 'previous_weekday_from_publication_date'
    assert result['publication_date'] == '2025-12-17'
    assert result['weekday'] == 'Tuesday'
    assert set(result['numeric_fields']) == {'cases_confirmed', 'deaths'}
    for anchor in result['anchors']:
        assert index['documents'][0]['clean_text'][anchor['char_start']:anchor['char_end']] == anchor['quote']
    assert not row.get('outbreak_closure_date')


@pytest.mark.parametrize('change', [
    {'publication':'Updated December 17, 2025.'},
    {'publication':'December 17, 2025.'},
    {'publication':'Published December 17, 2025. Published December 24, 2025.'},
    {'declaration':'The declaration might occur on Tuesday, the Health Minister said.'},
    {'declaration':'The declaration will occur on Tuesday, the Health Minister said.'},
    {'declaration':'The declaration on Tuesday was described by a blogger.'},
    {'declaration':'The declaration on Tuesday in a previous month meets standards, the Health Minister said.'},
    {'closure':'France has declared the end of a Cholera outbreak in the country.'},
    {'closure':'Haiti has not declared the end of a Cholera outbreak in the country.'},
    {'count':'Cholera in Haiti: 42 new confirmed cases today, including 3 deaths.'},
])
def test_unresolved_or_inapplicable_weekday_context_is_not_a_date(change):
    row, index = fixture(**change)
    assert resolve(row, index) is None


def test_tampered_hash_cannot_supply_relative_date():
    row, index = fixture()
    index['documents'][0]['clean_text'] += 'tampered'
    assert resolve(row, index) is None


def test_incorrect_count_cannot_borrow_closure_date():
    row, index = fixture()
    row['cases_confirmed'] = 99
    assert resolve(row, index) is None


def test_missing_country_and_year_are_not_filled_from_task():
    row, index = fixture()
    row.pop('country')
    assert resolve(row, index) is None


def test_source_local_recovery_preserves_candidate_and_unknown_statistical_period():
    from data_collection_workflow.evidence_qualification import qualify_records, qualified_coverage
    row, index = fixture()
    row.update(as_of_date='2025-12-17', event_start_date='2025-01-09')
    contract = {'disease':'Cholera', 'country':'Haiti', 'start_date':'2025-01-01', 'end_date':'2025-12-31'}
    groups = qualify_records([row], contract=contract, evidence_index=index)
    assert len(groups['candidate_records']) == 1
    assert len(groups['qualified_records']) == 1
    recovered = groups['qualified_records'][0]
    assert recovered['recovered_from_record_id'] == 'r'
    assert recovered['outbreak_closure_date'] == '2025-12-16'
    assert recovered.get('as_of_date') is None
    assert recovered.get('event_start_date') is None
    assert recovered.get('metric_period_start') is None
    assert recovered.get('metric_period_end') is None
    assert recovered['evidence_qualification']['temporal_basis'] == 'outbreak_closure_event'
    assert groups['candidate_records'][0]['event_start_date'] == '2025-01-09'
    assert row['event_start_date'] == '2025-01-09'
    assert not qualified_coverage([contract], [recovered])['coverage_complete']
    again = qualify_records(groups['candidate_records']+groups['qualified_records'], contract=contract, evidence_index=index)
    assert len(again['qualified_records']) == len(again['candidate_records']) == 1
    assert again['qualified_records'][0] == recovered


def test_derived_closure_anchors_and_date_are_revalidated():
    from copy import deepcopy
    from data_collection_workflow.evidence_qualification import assess_record_evidence, qualify_records
    row, index = fixture()
    recovered = qualify_records([row], contract={}, evidence_index=index)['qualified_records'][0]
    tampered = deepcopy(recovered)
    tampered['outbreak_closure_date'] = '2024-12-16'
    assert assess_record_evidence(tampered, contract={}, evidence_index=index).status == 'candidate'
    tampered = deepcopy(recovered)
    tampered['outbreak_closure_derivation']['anchors'][0]['char_start'] += 1
    assert assess_record_evidence(tampered, contract={}, evidence_index=index).status == 'candidate'


@pytest.mark.parametrize('declaration', [
    'The declaration on Tuesday last week meets standards, the Health Minister said.',
    'The declaration on Tuesday, December 7, 2021 meets standards, the Health Minister said.',
    'The anniversary of the declaration on Tuesday meets standards, the Health Minister said.',
    'A vaccine campaign announcement on Tuesday was made by the Health Minister.',
])
def test_other_temporal_roles_do_not_resolve(declaration):
    row, index = fixture(declaration=declaration)
    assert resolve(row, index) is None


def test_weekday_cannot_refer_to_a_different_declaration_in_same_chunk():
    closure = ('The Health Ministry declared the Cholera outbreak in Haiti over. '
               'The Health Ministry declared the measles outbreak in Haiti over after reviewing Cholera protocols.')
    row, index = fixture(closure=closure)
    assert resolve(row, index) is None


@pytest.mark.parametrize('suffix', ['on March 4, 2025', 'in March', 'in 2021'])
def test_explicit_time_in_closure_statement_is_not_overridden(suffix):
    row, index = fixture(closure=f'Haiti has declared the end of a Cholera outbreak in the country {suffix}.')
    assert resolve(row, index) is None


def test_news_metadata_and_agency_anniversary_do_not_change_declaration_role():
    row, index = fixture(publication='FREETOWN\nDecember 17, 2025\nPublish-\nDecember 17, 2025\nCollected image',
                        declaration='The declaration on Tuesday meets international standards, the Health Minister told a ceremony marking the second anniversary of the National Public Health Agency.')
    result = resolve(row, index)
    assert result is not None
    assert result['value'] == '2025-12-16'
    assert not result['quote'].startswith('FREETOWN')


@pytest.mark.parametrize('prefix', ['In 2021', '2021', 'In March', 'On March 4, 2025'])
def test_temporal_scope_on_wrapped_line_is_not_discarded_as_metadata(prefix):
    row, index = fixture(closure=prefix+'\nHaiti has declared the end of a Cholera outbreak in the country.')
    assert resolve(row, index) is None
