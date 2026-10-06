"""Stable evidence observation identities across independent extraction batches."""
from copy import deepcopy
import hashlib
import pytest
from data_collection_workflow.config import load_llm_structured_extraction_policy, load_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy, StructuredExtractionPolicy
from data_collection_workflow.nodes import extraction
from data_collection_workflow.workflow_recovery import RecoveryDelta, merge_recovery_delta, _repair_candidate_fields, RecoveryAction
from data_collection_workflow.session_runtime import fingerprint

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')

def build(text, *, chunk_id='chunk-a', start=0, index=1, disease='measles', country='Canada', count=12, **extra):
    chunk={'source_id':'source-a','chunk_id':chunk_id,'document_id':'doc-a','document_hash':'raw-hash',
        'text_hash':'text-hash','char_start':start,'char_end':start+len(text),'text':text,
        'source_url':'https://example.invalid/report','source_role_final':'collection'}
    data={'disease':disease,'country':country,'cases_confirmed':count,'reporting_period':'2025','evidence_quote':text,**extra}
    row=extraction._build_record_from_llm_output(LLMExtractedRecord(**data),chunk,index,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),{},
        {'disease_standard_name':disease,'is_hantavirus':False})
    assert row is not None
    return row.model_dump(),chunk

@pytest.mark.parametrize('disease,country',[('measles','Canada'),('chikungunya','France')])
def test_independent_batches_at_different_source_positions_cannot_reuse_index_id(disease,country):
    text=f'{country} reported 12 confirmed {disease} cases during 2025.'
    first,_=build(text,disease=disease,country=country)
    second,_=build(text,chunk_id='chunk-b',start=500,disease=disease,country=country)
    assert first['record_id']!=second['record_id']

def test_same_observation_id_does_not_depend_on_batch_index_or_iteration_order():
    text='Canada reported 12 confirmed measles cases during 2025.'
    first,_=build(text,index=1)
    repeated,_=build(text,index=91)
    assert first['record_id']==repeated['record_id']

def test_same_chunk_different_contradictory_claims_keep_separate_ids():
    text='Canada reported 12 or 13 confirmed measles cases during 2025.'
    first,_=build(text,count=12)
    second,_=build(text,count=13)
    assert first['record_id']!=second['record_id']

def test_same_chunk_two_patients_with_same_clinical_values_keep_separate_ids():
    a='Patient A in Canada had measles, fever and was aged 30 years during 2025.'
    b='Patient B in Canada had measles, fever and was aged 30 years during 2025.'
    first,_=build(a,count=None,workflow_case_label='Patient A',age='30',symptoms='fever')
    second,_=build(b,count=None,workflow_case_label='Patient B',age='30',symptoms='fever')
    assert first['record_id']!=second['record_id']

@pytest.mark.parametrize('builder_name',['_build_record_from_chunk','_official_outbreak_record_from_chunk'])
def test_deterministic_producers_also_ignore_loop_index(builder_name):
    text='Canada reported 12 confirmed measles cases during 2025.'
    _,chunk=build(text)
    chunk.update(fetch_purpose='data_extraction', contains_target_data=True, extraction_readiness='ready', task_relevance_status='target_disease_match', signal_strength='strong', data_types=['case_count'])
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    context=extraction._build_extraction_context({'structured_task':{'disease':'measles','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'}},policy)
    builder=getattr(extraction,builder_name)
    first=builder(chunk,1,policy,context)
    repeated=builder(chunk,9,policy,context)
    if isinstance(first,tuple):
        first,repeated=first[0],repeated[0]
    assert first is not None and repeated is not None
    assert first.record_id==repeated.record_id

@pytest.mark.parametrize('revision',['legacy_v1','standard'])
def test_legacy_builder_keeps_original_source_index_ids(monkeypatch,revision):
    monkeypatch.setenv('PIPELINE_MODE',revision)
    first,_=build('Canada reported 12 confirmed measles cases during 2025.',index=7)
    assert first['record_id']=='rec_source-a_007'

def test_cross_batch_merge_preserves_different_claims_but_deduplicates_same_observation():
    text='Canada reported 12 confirmed measles cases during 2025.'
    first,chunk=build(text)
    other,_=build(text,start=500,chunk_id='chunk-b')
    other['record_id']=first['record_id']  # Replay old producer collision.
    state={'raw_records':[first],'evidence_chunks':[chunk]}
    merged=merge_recovery_delta(state,RecoveryDelta(records=[other]))
    assert len(merged['raw_records'])==2
    assert len({r['record_id'] for r in merged['raw_records']})==2
    again=merge_recovery_delta(merged,RecoveryDelta(records=[other]))
    assert again['raw_records']==merged['raw_records']

def test_derived_annotations_do_not_create_duplicate_observation_on_merge():
    first,chunk=build('Canada reported 12 confirmed measles cases during 2025.')
    repeated=deepcopy(first)
    repeated.update(conflict_ids=['derived-a'],normalization_warnings=['derived'],requires_human_review=True)
    merged=merge_recovery_delta({'raw_records':[first],'evidence_chunks':[chunk]},RecoveryDelta(records=[repeated]))
    assert len(merged['raw_records'])==1
    assert merged['raw_records'][0]['record_id']==first['record_id']

def test_update_targets_matching_before_fingerprint_even_with_old_id_collision():
    first={'record_id':'old-collision','source_id':'s','supporting_chunk_id':'a','disease':'measles','cases_confirmed':12}
    second={**first,'supporting_chunk_id':'b','cases_confirmed':17}
    update={'record_id':'old-collision','before_fingerprint':fingerprint(second),'fields':{'country':'Canada'},'field_provenance':{}}
    merged=merge_recovery_delta({'raw_records':[first,second]},RecoveryDelta(record_updates=[update]))
    assert not merged['raw_records'][0].get('country')
    assert merged['raw_records'][1]['country']=='Canada'
    assert merged['raw_records'][1]['record_id']=='old-collision'

def test_ambiguous_old_id_never_selects_first_candidate_for_repair():
    rows=[{'record_id':'same','source_id':'s','supporting_chunk_id':chunk,'disease':'measles','cases_confirmed':12} for chunk in ('a','b')]
    update,reason=_repair_candidate_fields(RecoveryAction('repair_fields','same','action','fill missing scope'),{'raw_records':rows})
    assert update is None and reason=='ambiguous_record_identity'
