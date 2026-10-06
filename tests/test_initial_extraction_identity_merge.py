"""Repeated initial/focused observations are one fact, not new recovery gain."""
import pytest
from data_collection_workflow.config import load_structured_extraction_policy,load_llm_structured_extraction_policy
from data_collection_workflow.models import StructuredExtractionPolicy,LLMStructuredExtractionPolicy,LLMExtractionOutput
from data_collection_workflow.nodes import extraction


def run(monkeypatch,disease='mpox',country='Brazil',repeat=False):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LLM_HARD_PRIMARY_CALLS','3')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_ENABLED','true')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_MAX_CALLS','2')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_MAX_SOURCES','2')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_SPANS_PER_SOURCE','2')
    text=f'{disease} in {country} caused 12 confirmed cases during 2025.'
    chunk={'source_id':'s','chunk_id':'c','text':text,'char_start':0,'char_end':len(text),'document_hash':'a'*64,
        'title':f'{disease} outbreak report','source_url':'https://www.who.int/emergencies/disease-outbreak-news/item/local-test',
        'source_type':'official_public_health_agency','chunk_kind':'text','fetch_purpose':'data_extraction',
        'contains_target_data':True,'extraction_eligible_for_task_disease':True,
        'disease_relevance_status':'target_disease_match','source_role_final':'collection'}
    context={'disease_standard_name':disease,'disease_terms':[disease],
        'structured_task':{'disease':disease,'location':country},'guard_enabled':False}
    calls=[]
    def empty(chunk,policy):
        calls.append(chunk['chunk_id'])
        return LLMExtractionOutput(chunk_is_relevant=True,records=[])
    monkeypatch.setattr(extraction.llm_clients,'extract_chunk_with_llm',empty)
    monkeypatch.setattr(extraction.llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'offline-test'})
    records,stats=extraction._llm_extract_records_from_chunks([chunk,dict(chunk)] if repeat else [chunk],
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),
        StructuredExtractionPolicy(**load_structured_extraction_policy()),False,context)
    return records,stats,calls


@pytest.mark.parametrize('disease,country',[('mpox','Brazil'),('measles','Canada')])
def test_rule_observation_revisited_in_focused_recovery_is_not_added_again(monkeypatch,disease,country):
    records,stats,calls=run(monkeypatch,disease,country)
    assert records and calls
    assert len(records)==len({r.record_id for r in records})
    assert stats['deterministic_recovery_succeeded_count']==0
    assert stats['focused_recovery_valid_record_count']==0
    assert stats['focused_recovery_field_gain_count']==0


def test_duplicate_input_span_preserves_unique_initial_observations(monkeypatch):
    records,stats,_=run(monkeypatch,repeat=True)
    assert len(records)==1
    assert stats['valid_record_count']==stats['unique_valid_record_count']==1
    assert stats['official_outbreak_deterministic_record_count']==1


def test_different_complete_fact_payloads_are_preserved(monkeypatch):
    from data_collection_workflow.record_identity import record_identity_payload
    records,stats,_=run(monkeypatch,'dengue','Argentina')
    assert len(records)==2
    assert len({r.record_id for r in records})==2
    assert {r.cases_confirmed for r in records}=={12}
    assert {r.pathogen_or_syndrome for r in records}=={'Dengue',None}
    assert record_identity_payload(records[0])!=record_identity_payload(records[1])
    assert stats['official_outbreak_deterministic_record_count']==1
    assert stats['deterministic_recovery_succeeded_count']==1
    assert stats['focused_recovery_valid_record_count']==1
