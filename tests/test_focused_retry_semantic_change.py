"""Successful empty spans need substantive, locatable focused input changes."""
import copy
import pytest
from data_collection_workflow.config import load_llm_structured_extraction_policy,load_structured_extraction_policy
from data_collection_workflow.models import LLMExtractionOutput,LLMStructuredExtractionPolicy,StructuredExtractionPolicy
from data_collection_workflow.nodes import extraction as ex


def run(monkeypatch, *, disease='mpox',country='Brazil',quote=None,text=None,revision='evidence',error=False,queue_change=None):
    monkeypatch.setenv('PIPELINE_MODE',revision)
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_ENABLED','true')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_MAX_CALLS','2')
    monkeypatch.setenv('LLM_EXTRACTION_SAFETY_MAX_CALLS','8')
    monkeypatch.setenv('LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS','6')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_RESERVED_CALLS','2')
    raw=text or f'{disease} in {country}.\n  The patient had fever.\n Case details were reported in 2025.'
    chunk={'source_id':'s','chunk_id':'c','text':raw,'char_start':150,'char_end':150+len(raw),'document_hash':'a'*64,'title':f'{disease} report','source_url':'https://example.org/report','source_type':'peer_reviewed_literature','source_role_final':'collection','chunk_kind':'text','fetch_purpose':'data_extraction','contains_target_data':True,'extraction_eligible_for_task_disease':True,'disease_relevance_status':'target_disease_match'}
    context={'disease_standard_name':disease,'disease_terms':[disease],'structured_task':{'disease':disease,'location':country},'guard_enabled':False}
    calls=[]
    def answer(c,p):
        calls.append(copy.deepcopy(c))
        if error and len(calls)==1:raise RuntimeError('transient offline fixture failure')
        return LLMExtractionOutput(chunk_is_relevant=True,records=[])
    monkeypatch.setattr(ex.llm_clients,'extract_chunk_with_llm',answer)
    monkeypatch.setattr(ex.llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'offline'})
    monkeypatch.setattr(ex,'_rule_based_extract_records_from_chunks',lambda *a,**k:([],{}))
    monkeypatch.setattr(ex,'_classify_llm_empty_output_reason',lambda c:'llm_empty_strong_signal')
    queue=ex._focused_recovery_queue
    def focus(candidates,*args):
        rows=queue(candidates,*args)
        if quote is not None:
            rows=[dict(c,case_span_quote=quote) for c in rows]
        if queue_change:rows=[queue_change(dict(c)) for c in rows]
        return rows
    monkeypatch.setattr(ex,'_focused_recovery_queue',focus)
    monkeypatch.setattr(ex,'_chunk_has_strong_record_signal',lambda c:True)
    before=copy.deepcopy(chunk)
    records,stats=ex._llm_extract_records_from_chunks([chunk],LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),StructuredExtractionPolicy(**load_structured_extraction_policy()),False,context)
    assert chunk==before
    return calls,stats,raw


@pytest.mark.parametrize('disease,country',[('mpox','Brazil'),('measles','Canada'),('chikungunya','Réunion'),('unknown syndrome','Qeqertarsuaq')])
def test_completed_empty_full_span_is_not_paid_twice(monkeypatch,disease,country):
    calls,stats,_=run(monkeypatch,disease=disease,country=country)
    assert len(calls)==1
    assert stats['focused_recovery_call_count']==0
    assert stats['focused_retry_succeeded_count']==0
    assert stats['focused_recovery_status_by_source']['s']=='focused_retry_skipped_unchanged_empty'
    row=stats['llm_empty_output_diagnostics'][-1]
    assert row['status']=='skipped' and row['input_changed'] is False


def test_whitespace_only_quote_change_is_not_new_evidence(monkeypatch):
    raw='mpox in Brazil.\n\t  The patient had fever.\n Case details were reported in 2025.'
    calls,stats,_=run(monkeypatch,text=raw,quote=' '.join(raw.split()))
    assert len(calls)==1 and stats['focused_recovery_call_count']==0


@pytest.mark.parametrize('quote',['The patient had fever.','The patient had fever. Case details were reported in 2025.'])
def test_real_local_span_keeps_original_positions(monkeypatch,quote):
    calls,stats,raw=run(monkeypatch,quote=quote)
    assert len(calls)==2 and stats['focused_recovery_call_count']==1
    focused=calls[1]
    start,end=focused['case_span_start'],focused['case_span_end']
    assert raw[start:end]==focused['case_span_quote']
    assert ' '.join(raw[start:end].split())==quote
    assert focused['char_start']==150 and focused['char_end']==150+len(raw)
    assert stats['focused_retry_empty_count']==1 and stats['focused_retry_succeeded_count']==0


def test_unlocated_focus_is_not_a_new_paid_strategy(monkeypatch):
    calls,stats,_=run(monkeypatch,quote='A different patient died elsewhere.')
    assert len(calls)==1 and stats['focused_recovery_call_count']==0
    assert stats['focused_recovery_status_by_source']['s']=='focused_retry_skipped_unlocated_span'


def test_old_revision_keeps_existing_focused_behavior(monkeypatch):
    calls,stats,_=run(monkeypatch,revision='legacy')
    assert len(calls)==2 and stats['focused_recovery_call_count']==1


def test_failed_primary_is_not_classified_as_successful_empty(monkeypatch):
    calls,stats,_=run(monkeypatch,error=True)
    assert stats['llm_error_count']==1
    assert stats['llm_empty_output_count']==0
    assert 'focused_retry_skipped_unchanged_empty' not in stats['focused_recovery_status_by_source'].values()


def test_repeated_local_quote_requires_unambiguous_position(monkeypatch):
    raw='mpox in Brazil. The patient had fever. Later the patient had fever. The patient had fever.'
    calls,stats,_=run(monkeypatch,text=raw,quote='The patient had fever.')
    assert len(calls)==1
    assert stats['focused_recovery_status_by_source']['s']=='focused_retry_skipped_unlocated_span'


def test_valid_declared_position_disambiguates_repeated_local_quote(monkeypatch):
    raw='mpox in Brazil. The patient had fever. Later. The patient had fever.'
    quote='The patient had fever.';start=raw.rfind(quote)
    def select(c):
        return dict(c,case_span_start=start,case_span_end=start+len(quote))
    calls,stats,_=run(monkeypatch,text=raw,quote=quote,queue_change=select)
    assert len(calls)==2
    assert (calls[1]['case_span_start'],calls[1]['case_span_end'])==(start,start+len(quote))


def test_nonenglish_narrow_span_is_not_filtered_by_symptom_vocabulary(monkeypatch):
    raw='Unknown syndrome. 證據：患者於週末住院。\n  敘述保持原樣。'
    calls,stats,_=run(monkeypatch,disease='unknown syndrome',text=raw,quote='患者於週末住院。')
    assert len(calls)==2
    assert calls[1]['case_span_quote']=='患者於週末住院。'


def test_focused_reason_or_round_metadata_does_not_justify_identical_span(monkeypatch):
    def relabel(c):
        return dict(c,recovery_reason='please try a different strategy',recovery_round=7,recovery_action_id='changed')
    calls,stats,_=run(monkeypatch,queue_change=relabel)
    assert len(calls)==1
    assert stats['focused_recovery_call_count']==0
