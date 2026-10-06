"""Exercise the real extraction schema, prompt and record evidence boundary."""
import hashlib
import json
import pytest
from data_collection_workflow import llm_clients
from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.nodes import extraction
from data_collection_workflow.evidence_qualification import assess_record_evidence

QUOTE = 'Dengue in Brazil caused 12 confirmed cases during 2024.'


def _chunk(text=QUOTE, **extra):
    return dict(source_id='s', chunk_id='c', document_id='d', document_hash=hashlib.sha256(text.encode()).hexdigest(),
                source_url='https://example.invalid/report', source_type='official_public_health_agency',
                source_role_final='collection', text=text, char_start=0, char_end=len(text), **extra)


def _policy():
    return LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())


def _build(payload, chunk=None, context=None):
    return extraction._build_record_from_llm_output(LLMExtractedRecord(**payload), chunk or _chunk(), 1,
        _policy(), {}, context or {'disease_standard_name':'Dengue','is_hantavirus':False}).model_dump()


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')


@pytest.mark.parametrize('serialized', [False, True])
def test_schema_and_builder_preserve_field_quotes_from_model(serialized):
    entry={'cases_confirmed':{'quote':QUOTE,'chunk_id':'model_invented','document_hash':'model_invented'}}
    payload={'disease':'Dengue','country':'Brazil','reporting_period':'2024','cases_confirmed':12,
             'evidence_quote':QUOTE,'field_provenance_json':json.dumps(entry) if serialized else entry}
    parsed=LLMExtractedRecord(**payload).model_dump()
    assert parsed.get('evidence_quote')==QUOTE
    assert parsed.get('field_provenance_json')
    row=_build(payload)
    provenance=json.loads(row['field_provenance_json'])
    assert provenance['cases_confirmed']['quote']==QUOTE
    assert provenance['cases_confirmed']['chunk_id']=='c'
    assert provenance['cases_confirmed']['document_hash']==_chunk()['document_hash']
    assert provenance['disease']['quote']==QUOTE


def test_short_model_quote_survives_large_transport_chunk():
    text=QUOTE+'\n'+('Administrative information. '*60)
    chunk=_chunk(text)
    row=_build({'disease':'Dengue','country':'Brazil','reporting_period':'2024','cases_confirmed':12,
                'evidence_quote':QUOTE},chunk)
    assert row['evidence_quote']==QUOTE
    document={'source_id':'s','document_id':'d','content_hash':chunk['document_hash'],'text_hash':chunk['document_hash'],'clean_text':text}
    q=assess_record_evidence(row,contract={},evidence_index={'documents':[document],'evidence_chunks':{'c':chunk}})
    assert q.status=='qualified',q.to_dict()


def test_invalid_explicit_field_quote_does_not_fall_back_to_the_whole_chunk():
    row=_build({'disease':'Dengue','country':'Brazil','reporting_period':'2024','cases_confirmed':12,
                'field_provenance_json':{'cases_confirmed':{'quote':'12 was invented'}}})
    assert json.loads(row['field_provenance_json'])['cases_confirmed']['quote']=='12 was invented'
    chunk=_chunk()
    q=assess_record_evidence(row,contract={},evidence_index={'documents':[{'source_id':'s','document_id':'d',
       'content_hash':chunk['document_hash'],'text_hash':chunk['document_hash'],'clean_text':QUOTE}],'evidence_chunks':{'c':chunk}})
    assert q.status=='candidate'


def test_task_constraint_cannot_replace_explicit_or_missing_source_disease():
    assert _build({'disease':'Cholera','cases_confirmed':12})['disease']=='Cholera'
    assert not _build({'cases_confirmed':12})['disease']


def test_legacy_disease_standardization_is_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    assert _build({'disease':'Cholera','cases_confirmed':12})['disease']=='Dengue'


def test_evidence_prompt_exposes_source_bound_context_and_field_quote_contract():
    bound=[{'role':'heading','quote':'Dengue in Brazil during 2024','char_start':0,'char_end':28}]
    chunk=_chunk(bound_context_spans=bound)
    messages=llm_clients._build_llm_messages(chunk,_policy())
    system=messages[0]['content']; prompt=messages[1]['content']
    assert 'field_provenance_json' in system
    assert 'bound_context_spans' in prompt and 'Dengue in Brazil during 2024' in prompt
    assert 'Set disease to the active task disease' not in system


def test_row_batch_preserves_each_rows_context_and_locator():
    rows=[]
    for i in (1,2):
        rows.append(_chunk('Brazil | 12',row_id=str(i),table_id='t',row_quote='Brazil | 12',
            bound_context_spans=[{'role':'heading','quote':f'Report {2020+i}','char_start':10*i,'char_end':10*i+11}]))
        rows[-1].update(chunk_id=f'c{i}',char_start=100*i,char_end=100*i+11)
    batch=extraction._make_metric_row_batch_chunk(rows,batch_index=0)
    for i in (1,2):
        metadata=batch['row_metadata_by_id'][str(i)]
        assert metadata.get('bound_context_spans')==rows[i-1]['bound_context_spans']
        assert metadata.get('char_start')==100*i
    messages=llm_clients._build_llm_messages(batch,_policy())
    assert 'Report 2021' in messages[1]['content'] and 'Report 2022' in messages[1]['content']
    row=_build({'disease':'Dengue','country':'Brazil','cases_confirmed':12,'source_row_id':'2',
                'field_provenance_json':{'cases_confirmed':{'quote':'Brazil | 12'}}},batch)
    assert json.loads(row['field_provenance_json'])['cases_confirmed']['chunk_id']=='c2'


def test_semantic_guardrails_use_only_the_bound_batch_row():
    rows=[]
    for i, text in enumerate(('Emergency department visits percent | 8', 'Laboratory positivity percent | 12'),1):
        item=_chunk(text,row_id=str(i),table_id='t',row_quote=text,chunk_kind='metric_row')
        item['chunk_id']=f'c{i}'
        rows.append(item)
    batch=extraction._make_metric_row_batch_chunk(rows,batch_index=0)
    record=_build({'disease':'Dengue','source_row_id':'2','positivity_rate':12,
                   'evidence_quote':rows[1]['text']},batch)
    assert record['positivity_rate']==12
    assert record['metric_name']!='nssp_ed_visit_percent'
    assert record['metric_denominator']!='emergency_department_visits'
    assert record['supporting_chunk_id']=='c2'


def test_metric_row_groups_separate_content_versions():
    first=_chunk('Brazil | 12',row_id='1',table_id='t')
    second=_chunk('Brazil | 13',row_id='1',table_id='t')
    assert extraction._metric_row_key(first)!=extraction._metric_row_key(second)
    one=extraction._make_metric_row_batch_chunk([first],batch_index=1)
    two=extraction._make_metric_row_batch_chunk([second],batch_index=1)
    assert one['chunk_id']!=two['chunk_id']


def test_metric_batch_disambiguates_duplicate_row_identifiers():
    first=_chunk('Brazil | 12',row_id='1',table_id='t')
    second={**first,'chunk_id':'c2'}
    batch=extraction._make_metric_row_batch_chunk([first,second],batch_index=0)
    row_ids=[row['row_id'] for row in batch['metric_rows']]
    assert len(set(row_ids))==2
    assert set(batch['row_chunk_id_by_id'].values())=={'c','c2'}


def test_metric_category_is_not_invented_as_count_semantics():
    record=_build({'disease':'Dengue','cases_confirmed':12,
                   'metric_name':'confirmed_cases','metric_category':'case_count',
                   'metric_value':12,'metric_unit':'count'})
    assert record['metric_category']=='case_count'
    # A year in a quote does not establish that this is the complete annual total.
    assert record['count_semantics'] is None
    assert record['statistical_count_type'] is None


def test_explicit_source_supported_count_classification_is_preserved():
    text = 'Dengue in Brazil: annual total of 12 confirmed cases during 2024.'
    record = _build({'disease': 'Dengue', 'country': 'Brazil', 'reporting_period': '2024',
                     'cases_confirmed': 12, 'statistical_count_type': 'annual',
                     'evidence_quote': text}, _chunk(text))
    assert record['statistical_count_type'] == 'annual'
    chunk = _chunk(text)
    qualification = assess_record_evidence(record, contract={}, evidence_index={
        'documents': [{'source_id': 's', 'document_id': 'd', 'clean_text': text,
                       'content_hash': chunk['document_hash'], 'text_hash': chunk['document_hash']}],
        'evidence_chunks': {'c': chunk}})
    assert qualification.status == 'qualified', qualification.to_dict()
