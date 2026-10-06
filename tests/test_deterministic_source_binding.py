"""Actual evidence deterministic producers must preserve source-local facts.

All source bytes are synthetic minimal statements; no model response or extracted
answer is injected. The same fixtures exercise parser, chunking, schema repair,
normalization, and the final evidence assessor.
"""
import pytest
from data_collection_workflow.config import load_structured_extraction_policy
from data_collection_workflow.models import StructuredExtractionPolicy
from data_collection_workflow.nodes.content_processing import document_quality_check, evidence_chunking_and_data_presence_flagging
from data_collection_workflow.nodes.extraction import (structured_extraction, schema_validation_and_repair,
    _build_extraction_context, _official_outbreak_record_from_chunk)
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index

COUNTS=('cases_confirmed','cases_probable','cases_suspected','cases_unspecified','deaths','metric_value')

@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')
    monkeypatch.setenv('ENABLE_LANGSMITH_TRACE','false')


def _run(tmp_path,body,*,disease='Pertussis',country='France',official=False):
    doc=parse_response(body.encode(),url='https://authority.invalid/report',source_id='source',
        session_dir=tmp_path,content_type='text/html' if '<' in body else 'text/plain')
    doc.update(document_id='document',fetch_status='fetched',fetch_purpose='data_extraction',
        source_role_final='collection',source_type='national_public_health_agency',
        title=(doc.get('metadata') or {}).get('page_heading'))
    state={'documents':[doc],'source_registry':[{'source_id':'source','url':doc['url'],
        'source_role_final':'collection','fetch_purpose':'data_extraction','source_type':'national_public_health_agency'}],
        'structured_task':{'disease':disease,'location':country,'start_date':'2025-01-01',
                           'end_date':'2025-12-31','collection_mode':'direct_collection'}}
    state.update(document_quality_check(state))
    state.update(evidence_chunking_and_data_presence_flagging(state))
    if official:
        policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
        context=_build_extraction_context(state,policy);raw=[]
        for pos,chunk in enumerate(state['evidence_chunks'],1):
            if not chunk.get('contains_target_data') or not chunk.get('extraction_eligible_for_task_disease'):continue
            record,_=_official_outbreak_record_from_chunk(chunk,pos,policy,context)
            if record is not None:raw.append(record.model_dump())
        state['raw_records']=raw
    else:
        state.update(structured_extraction(state))
    state.update(schema_validation_and_repair(state));state.update(record_normalization(state))
    idx=build_evidence_index(state)
    contract={'disease':disease,'country':country,'start_date':'2025-01-01','end_date':'2025-12-31'}
    state['assessed_records']=[{**row,'evidence_qualification':assess_record_evidence(row,contract=contract,evidence_index=idx).to_dict()}
                               for row in state.get('normalized_records') or []]
    state['qualified_records']=[row for row in state['assessed_records'] if row['evidence_qualification']['status']=='qualified']
    return state


def _values(row):
    return [row.get(key) for key in COUNTS if row.get(key) is not None]


def _diagnostics(state):
    return {'raw':[{k:r.get(k) for k in ('country','reporting_period','date_reported','cases_unspecified','metric_value','metric_period_start','metric_period_end','evidence_quote')} for r in state.get('raw_records') or []],
            'rejected':[r.get('validation_errors') for r in state.get('rejected_records') or []],
            'candidates':[r['evidence_qualification']['reasons'] for r in state['assessed_records']]}


@pytest.mark.parametrize('disease,country', [('Pertussis','France'),('Dengue','Brazil')])
def test_public_rule_pairs_each_count_with_its_source_year(tmp_path,disease,country):
    body=(f'In 2024, a total of 17 {disease} cases were reported in {country}. '
          f'In 2025, a total of 23 {disease} cases were reported in {country}.')
    state=_run(tmp_path,body,disease=disease,country=country)
    assert any(23 in _values(row) for row in state['qualified_records']),_diagnostics(state)
    assert not any(17 in _values(row) for row in state['qualified_records'])
    assert not any(17 in _values(row) and str(row.get('metric_period_start') or '').startswith('2025')
                   for row in state.get('raw_records') or []),_diagnostics(state)


def test_rule_does_not_fill_day_precision_or_definition_from_task(tmp_path):
    state=_run(tmp_path,'In 2025, 23 Pertussis cases were reported in France.')
    counted=[row for row in state.get('raw_records') or [] if _values(row)]
    assert counted,_diagnostics(state)
    for row in counted:
        assert not row.get('date_reported'),_diagnostics(state)
        assert not row.get('metric_period_start') and not row.get('metric_period_end'),_diagnostics(state)
        assert not row.get('case_definition'),_diagnostics(state)
    assert state['qualified_records'],_diagnostics(state)


@pytest.mark.parametrize('disease,country,number',[('Pertussis','France',40),('Dengue','Brazil',17)])
def test_country_cardinality_is_never_extracted_as_case_count(tmp_path,disease,country,number):
    text=f'During 2025, {disease} in {country} spread across approximately {number} countries and includes travel-associated cases.'
    state=_run(tmp_path,text,disease=disease,country=country)
    assert not any(number in _values(row) for row in state.get('raw_records') or []),_diagnostics(state)


@pytest.mark.parametrize('literal,country',[('United States','United States of America'),('United States of America','United States')])
def test_source_country_alias_remains_valid_after_full_rule_chain(tmp_path,literal,country):
    text=f'In 2025, 23 Pertussis cases were reported in {literal}.'
    state=_run(tmp_path,text,country=country,official=True)
    assert any(23 in _values(row) for row in state['qualified_records']),_diagnostics(state)


@pytest.mark.parametrize('disease,country',[('Pertussis','France'),('Dengue','Brazil')])
def test_official_rule_source_heading_reaches_schema_and_assessor(tmp_path,disease,country):
    body=f'<h1>{disease} surveillance in {country}, 2025</h1><p>In 2025, a total of 23 cases were reported.</p>'
    state=_run(tmp_path,body,disease=disease,country=country,official=True)
    assert any(23 in _values(row) for row in state['qualified_records']),_diagnostics(state)


def test_source_heading_never_overrides_explicit_other_country(tmp_path):
    body='<h1>Pertussis surveillance in France, 2025</h1><p>In Brazil, 23 Pertussis cases were reported during 2025.</p>'
    state=_run(tmp_path,body,official=True)
    assert not state['qualified_records'],_diagnostics(state)
    assert not any(row.get('country')=='France' and 23 in _values(row) for row in state.get('raw_records') or [])


def test_source_heading_is_not_carried_through_conflicting_disease_section(tmp_path):
    body='<h1>Pertussis surveillance in France, 2025</h1><h2>Dengue in Brazil</h2><p>In 2025, a total of 23 cases were reported.</p>'
    state=_run(tmp_path,body,official=True)
    assert not state['qualified_records'],_diagnostics(state)


@pytest.mark.parametrize('tamper',['document_hash','span_quote','document_text'])
def test_official_rule_rejects_unverified_heading_metadata(tmp_path,tamper):
    from copy import deepcopy
    from data_collection_workflow.nodes.extraction import _verified_rule_headings
    state=_run(tmp_path,'<h1>Pertussis surveillance in France, 2025</h1><p>In 2025, a total of 23 cases were reported.</p>',official=True)
    chunk=deepcopy(next(c for c in state['evidence_chunks'] if '23 cases' in c['text']))
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    context=_build_extraction_context(deepcopy(state),policy)
    if tamper=='document_hash':
        chunk['document_hash']='0'*64
    elif tamper=='span_quote':
        chunk['bound_context_spans'][0]['quote']='Pertussis surveillance in France, 2024'
    else:
        context['documents_by_source_id']['source'][0]['clean_text']+=' altered'
    # An unverified convenience title cannot replace a failed exact source link.
    chunk['heading_context']='Pertussis surveillance in France, 2025'
    chunk['title']='Pertussis surveillance in France, 2025'
    assert _verified_rule_headings(chunk,context)==[]
    record,_=_official_outbreak_record_from_chunk(chunk,1,policy,context)
    assert record is None or record.country != 'France'
