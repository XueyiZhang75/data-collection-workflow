"""Observation place and date roles cannot be borrowed from travel origins."""
import hashlib
import pytest
from data_collection_workflow.source_assertions import typed_date_support, observation_dates
from data_collection_workflow.evidence_qualification import assess_record_evidence, _role_bound_occurrence
from data_collection_workflow.nodes.extraction import _official_geography

@pytest.fixture(autouse=True)
def revision(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')

def assess_scope(body, headings=(), **fields):
    prefix='\n'.join(headings)
    text=(prefix+'\n' if prefix else '')+body
    digest=hashlib.sha256(text.encode()).hexdigest()
    spans=[]; offset=0
    for index, heading in enumerate(headings):
        spans.append(dict(role='heading',quote=heading,char_start=offset,char_end=offset+len(heading),heading_level=index+1,section_id='h'+str(index)))
        offset+=len(heading)+1
    document=dict(source_id='s',document_id='d',clean_text=text,content_hash=digest,text_hash=digest,parser_version='test/1',locator_spans=spans)
    chunk=dict(source_id='s',chunk_id='c',text=body,document_hash=digest,char_start=offset,char_end=len(text),bound_context_spans=spans)
    return assess_record_evidence(dict(record_id='r',source_id='s',supporting_chunk_id='c',**fields),contract={},evidence_index={'documents':[document],'evidence_chunks':{'c':chunk}})

@pytest.mark.parametrize('phrase', ['Depuis le 1er janvier 2025', 'Depuis 1 janvier 2025', 'Since January 1, 2025', 'From January 1, 2025', 'À partir du 1 janvier 2025'])
def test_start_date_is_not_asof(phrase):
    text=phrase+', 766 chikungunya cases were identified in France.'
    assert typed_date_support('as_of_date','2025-01-01',text) is False
    assert typed_date_support('metric_period_start','2025-01-01',text) is True
    assert typed_date_support('period_start_date','2025-01-01',text) is True
    result=observation_dates(text)
    assert not result.get('as_of_date')
    assert result.get('metric_period_start')=='2025-01-01'

@pytest.mark.parametrize('phrase', ['As of January 1, 2025', 'Au 1 janvier 2025', 'Le 1 janvier 2025'])
def test_asof_control(phrase):
    text=phrase+', France reported 76 measles cases.'
    assert typed_date_support('as_of_date','2025-01-01',text) is True
    assert observation_dates(text)['as_of_date']=='2025-01-01'

@pytest.mark.parametrize('phrase', ["Jusqu'au 1 janvier 2025", 'Until January 1, 2025', 'Through January 1, 2025'])
def test_explicit_end_not_start_or_asof(phrase):
    text=phrase+', France recorded 766 chikungunya cases.'
    assert not typed_date_support('as_of_date','2025-01-01',text)
    assert typed_date_support('metric_period_end','2025-01-01',text)
    assert observation_dates(text).get('metric_period_end')=='2025-01-01'

@pytest.mark.parametrize('origin', ['La Réunion','Canada','Brazil'])
@pytest.mark.parametrize('predicate', ['venaient de','provenaient de','étaient originaires de','returned from','originated in','were imported from'])
def test_origin_role_is_not_observation_location(origin,predicate):
    text=f'766 cases were identified in 2025, 97% of them {predicate} {origin}.'
    assert not _role_bound_occurrence('country',origin,text)

@pytest.mark.parametrize('disease,observed,origin,count', [('chikungunya','France','La Réunion',766),('measles','Canada','Brazil',181),('dengue','Peru','India',502)])
@pytest.mark.parametrize('phrase',['venaient de','originated in','returned from'])
def test_heading_cannot_promote_origin_to_count_scope(disease,observed,origin,count,phrase):
    body=f'During 2025, {count} {disease} cases were identified, 97% of them {phrase} {origin}.'
    headings=[f'{disease} surveillance in {origin}',f'Imported cases in {observed}']
    bad=assess_scope(body,headings,disease=disease,country=origin,reporting_period='2025',cases_unspecified=count)
    assert bad.status=='candidate',bad
    assert any(item.field=='cases_unspecified' and not item.supported for item in bad.field_evidence)
    good=assess_scope(body,headings[1:],disease=disease,country=observed,reporting_period='2025',cases_unspecified=count)
    assert good.status=='qualified',good.reasons

@pytest.mark.parametrize('disease,country,count',[('chikungunya','France',766),('measles','Canada',181)])
def test_true_observation_geo_remains_supported(disease,country,count):
    body=f'{country} reported {count} {disease} cases during 2025.'
    assert assess_scope(body,disease=disease,country=country,reporting_period='2025',cases_unspecified=count).status=='qualified'

@pytest.mark.parametrize('origin',['France','Canada','La Réunion'])
def test_rule_geo_does_not_resurrect_legacy_country_fallback(origin):
    result=_official_geography(f'766 cases in 2025, 97% of them venaient de {origin}.','')
    assert not result.get('country'),result
    assert not result.get('geographic_scope'),result

@pytest.mark.parametrize('disease,observed,origin,count',[('chikungunya','France','La Réunion',766),('measles','Canada','Brazil',181)])
def test_start_bound_count_correct_observation_country(disease,observed,origin,count):
    body=f'Depuis le 1er janvier 2025, {count} cas ont été identifiés, 97 % venaient de {origin}.'
    headings=[f'{disease} surveillance',f'{count} cas importés en {observed}']
    good=assess_scope(body,headings,disease=disease,country=observed,metric_period_start='2025-01-01',cases_unspecified=count)
    assert good.status=='qualified',good.reasons
    bad=assess_scope(body,headings,disease=disease,country=origin,as_of_date='2025-01-01',cases_unspecified=count)
    assert bad.status=='candidate'
    assert any(item.field=='as_of_date' and not item.supported for item in bad.field_evidence)
    assert any(item.field=='cases_unspecified' and not item.supported for item in bad.field_evidence)

@pytest.mark.parametrize('disease,observed,count',[('mpox','Sierra Leone',18),('measles','Canada',29)])
@pytest.mark.parametrize('date_clause',[
    'since the disease was confirmed on 13 January 2025',
    'since the outbreak began on 13 January 2025',
    'since the patient developed symptoms on 13 January 2025',
    'depuis que la maladie a été confirmée le 13 janvier 2025',
    'since the first patient died on 13 January 2025',
])
def test_clinical_event_date_is_not_count_asof(disease,observed,count,date_clause):
    text=f'{observed} reported {count} {disease} cases {date_clause}.'
    assert not typed_date_support('as_of_date','2025-01-13',text)
    assert not observation_dates(text).get('as_of_date')
    q=assess_scope(text,disease=disease,country=observed,as_of_date='2025-01-13',cases_unspecified=count)
    assert q.status=='candidate'
    assert any(f.field=='as_of_date' and not f.supported for f in q.field_evidence)

@pytest.mark.parametrize('clause',[
    'As of 13 January 2025, France reported 18 mpox cases.',
    'France reported 18 mpox cases on 13 January 2025.',
    'The outbreak began on 1 January 2025. As of 13 January 2025, France reported 18 mpox cases.',
    'France reported 18 mpox cases as of 13 January 2025, since the first patient died on 13 January 2025.',
])
def test_independent_current_count_date_after_clinical_event_is_preserved(clause):
    assert typed_date_support('as_of_date','2025-01-13',clause)

@pytest.mark.parametrize('text',[
    'As of May 1, 2025, France reported 18 confirmed mpox cases.',
    'On May 1, 2025, a total of 18 confirmed mpox cases were reported in France.',
    'Since the disease was confirmed on January 13, 2025, 18 mpox cases were reported in France as of February 18, 2025.',
    'As of February 18, 2025, France reported 18 mpox cases since the disease was confirmed on January 13, 2025.',
])
def test_cutoff_with_confirmed_word_and_cross_date_clauses(text):
    wanted='2025-05-01' if 'May' in text else '2025-02-18'
    assert typed_date_support('as_of_date',wanted,text)
    if 'January 13' in text:
        assert not typed_date_support('as_of_date','2025-01-13',text)
