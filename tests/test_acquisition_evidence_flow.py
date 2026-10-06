"""Whole positive and negative evidence flow; only the model transport is synthetic."""
import json
from copy import deepcopy

import pytest
from data_collection_workflow import llm_clients
from data_collection_workflow.models import Document, LLMExtractionOutput
from data_collection_workflow.nodes.content_processing import document_quality_check, evidence_chunking_and_data_presence_flagging
from data_collection_workflow.nodes.extraction import structured_extraction, schema_validation_and_repair
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.nodes.linking_validation import record_linking, cross_source_consistency_check, quality_gate_routing
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.session_runtime import RunContext


@pytest.mark.parametrize('language', ['english', 'french'])
@pytest.mark.parametrize('mutate_country', [False, True])
def test_grouped_count_survives_real_nodes_but_wrong_country_does_not(tmp_path, monkeypatch, language, mutate_country):
    for name,value in {
        'PIPELINE_MODE':'evidence', 'ENABLE_LLM_EXTRACTION':'true',
        'LLM_FALLBACK_TO_RULE_BASED':'false', 'LLM_PROVIDER':'openai', 'LLM_MODEL':'offline-response-fixture',
        'ENABLE_LLM_SOURCE_IDENTITY':'false', 'ENABLE_LLM_DISEASE_INTELLIGENCE':'false',
        'ENABLE_HUMAN_REVIEW':'false',
    }.items():
        monkeypatch.setenv(name,value)
    if language=='english':
        disease,country='pertussis','Canada'
        text='Canada reported 18 000 suspected pertussis cases during 2025.'
        facts={'disease':disease,'country':country,'reporting_period':'2025','cases_suspected':18000}
        field='cases_suspected'
    else:
        disease,country='chikungunya','France'
        text='France a signalé 32 456 cas confirmés de chikungunya du 1er janvier au 23 mai 2025.'
        facts={'disease':disease,'country':country,'reporting_period':'du 1er janvier au 23 mai 2025',
            'metric_period_start':'2025-01-01','metric_period_end':'2025-05-23','cases_confirmed':32456}
        field='cases_confirmed'
    if mutate_country:
        facts['country']='Brazil'
    response={'chunk_is_relevant':True,'records':[{**facts,'evidence_quote':text,
        'field_provenance_json':{name:{'quote':text} for name in facts}}]}
    invocations=[]
    class Model:
        model_name='offline-response-fixture'
        def with_structured_output(self,schema):
            assert schema is LLMExtractionOutput
            return self
        def invoke(self,messages,config=None):
            invocations.append(messages)
            chunk=messages[-1]['content'].split('Evidence chunk text:',1)[1]
            return deepcopy(response) if disease in chunk else {'chunk_is_relevant':False,'records':[]}
    monkeypatch.setattr(llm_clients,'build_chat_model',lambda *args,**kwargs:Model())
    runtime=RunContext(tmp_path, {'pipeline_mode':'evidence',
        'universal':{'budget_policy':{'version':2,'mode':'adaptive'},
                     'budget_limits':{'extraction':8},'extraction_reserve':0}})
    url='https://surveillance.example/report'
    source={'source_id':'s','url':url,'canonical_url':url,'source_type':'official_public_health_agency',
        'source_role':'data_source','source_role_final':'collection','status':'fetched',
        'fetch_purpose':'data_extraction','credibility_score':0.9,'credibility_level':'high'}
    parsed=parse_response(text.encode('utf-8'),url=url,source_id='s',session_dir=tmp_path)
    state={'structured_task':{'disease':disease,'location':country,'start_date':'2025-01-01','end_date':'2025-12-31'},
        'collection_spec':{'disease':disease,'geography':country,'start_date':'2025-01-01','end_date':'2025-12-31','target_population':'humans'},
        'source_registry':[source],'documents':[Document(**{**source,**parsed}).model_dump()],
        'human_review_queue':[],'collection_trace':[]}
    with runtime.activate():
        for node in (document_quality_check,evidence_chunking_and_data_presence_flagging,
                     structured_extraction,schema_validation_and_repair,record_normalization,
                     record_linking,cross_source_consistency_check,quality_gate_routing,final_data_package_builder):
            state.update(node(state))
    assert invocations, json.dumps({key:state.get(key) for key in ("llm_extraction_summary","extraction_summary","document_quality_summary","evidence_chunks")},ensure_ascii=False,default=str)
    package=state['final_data_package']
    rows=package['final_dataset']
    if mutate_country:
        assert not rows, rows
        assert package['candidate_records'] or package.get('rejected_records')
        return
    assert len(rows)==1, json.dumps([row.get('evidence_qualification',{}).get('reasons') for row in package.get('candidate_records',[])],ensure_ascii=False,default=str)
    row=rows[0]
    assert row[field]==facts[field]
    assert row['country']==country
    for evidence in row['evidence_qualification']['field_evidence']:
        assert evidence['supported']
        assert evidence['document_hash']==parsed['content_hash']
        location=evidence['locator']
        assert parsed['clean_text'][location['char_start']:location['char_end']]==evidence['quote']
    assert (tmp_path/parsed['raw_artifact_path']).read_bytes()==text.encode('utf-8')
    assert package['result_manifest']['counts']['qualified_observations']==1
