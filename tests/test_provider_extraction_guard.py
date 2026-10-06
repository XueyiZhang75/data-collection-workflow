from types import SimpleNamespace
from data_collection_workflow import llm_clients, session_runtime as runtime
from data_collection_workflow.nodes import extraction
from data_collection_workflow.models import LLMExtractionOutput
from test_provider_account_guard import context, payload, ProviderError
from test_extraction_scheduler_efficiency import _chunk, _scheduler_policies, _disable_deterministic_and_recovery_paths


def test_batch_undispatched_provider_denials_are_not_model_calls(monkeypatch):
    def denied(chunk, policy):
        raise runtime.ProviderAccountLimit('anthropic',{'reason':'account stopped'},dispatched=False)
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',denied)
    results,stats=extraction._run_bounded_llm_calls([_chunk('a','a'),_chunk('b','b')],policy=SimpleNamespace(),max_concurrency=2)
    assert stats['model_call_count']==0
    assert all(not result['model_call'] for result in results)


def test_primary_after_account_halt_preserves_unattempted_chunks_and_local_fallback(tmp_path,monkeypatch):
    c=context(tmp_path); c.session_dir=tmp_path
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LLM_EXTRACTION_MAX_CONCURRENCY','1')
    monkeypatch.setenv('LLM_EXTRACTION_ROLLING_YIELD_WINDOW','40')
    monkeypatch.setenv('LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS','50')
    monkeypatch.setenv('LLM_EXTRACTION_SAFETY_MAX_CALLS','50')
    monkeypatch.setattr(llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'test'})
    _disable_deterministic_and_recovery_paths(monkeypatch)
    local=[]; network=[]
    def fallback(chunks,*args,**kwargs):
        local.extend(row['chunk_id'] for row in chunks); return [],{}
    monkeypatch.setattr(extraction,'_rule_based_extract_records_from_chunks',fallback)
    def model(chunk,policy):
        def request(): network.append(chunk['chunk_id']); raise ProviderError()
        return c.call('extraction',payload(chunk['chunk_id']),request)
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    lp,dp=_scheduler_policies()
    chunks=[_chunk(str(i),f'Patient {i} was confirmed by PCR.',contains_target_data=True,case_span_id=str(i),case_span_quote=f'Patient {i} was confirmed by PCR.') for i in range(9)]
    with c.activate():
        records,stats=extraction._llm_extract_records_from_chunks(chunks,lp,dp,fallback_to_rule_based=True,context={'collection_mode':'standard'})
    assert len(network)==1
    assert len(set(local))==9
    assert c.ledger.attempted_chunk_ids()==network
    assert stats['scheduler_model_call_count']==1
    assert stats['llm_call_count']==1
    assert stats['llm_error_count']==1
    assert stats['provider_deferred_chunk_count']==8
    assert stats['extraction_budget_ledger']['actual_model_call_count']==1
    assert sum(x['attempted_count'] for x in stats['extraction_budget_by_source'].values())==1


def test_prefetch_halt_race_releases_unadmitted_reservations(tmp_path,monkeypatch):
    from threading import Event
    c=context(tmp_path); c.session_dir=tmp_path
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    for name,value in {'LLM_EXTRACTION_MAX_CONCURRENCY':'4','LLM_EXTRACTION_ROLLING_YIELD_WINDOW':'40','LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS':'50','LLM_EXTRACTION_SAFETY_MAX_CALLS':'50'}.items(): monkeypatch.setenv(name,value)
    monkeypatch.setattr(llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'test'})
    _disable_deterministic_and_recovery_paths(monkeypatch)
    monkeypatch.setattr(extraction,'_ordered_llm_chunks',lambda chunks,*args:chunks)
    stopped=Event(); network=[]
    def model(chunk,policy):
        if chunk['source_id']!='1': stopped.wait(5)
        def request(): network.append(chunk['chunk_id']); raise ProviderError()
        try:return c.call('extraction',payload(chunk['chunk_id']),request)
        finally:
            if chunk['source_id']=='1':stopped.set()
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    lp,dp=_scheduler_policies()
    chunks=[_chunk(str(i),f'Patient {i} was confirmed by PCR.',contains_target_data=True,case_span_id=str(i),case_span_quote=f'Patient {i} was confirmed by PCR.') for i in range(8)]
    with c.activate():
        _,stats=extraction._llm_extract_records_from_chunks(chunks,lp,dp,fallback_to_rule_based=False,context={'collection_mode':'standard'})
    assert network==['chunk_1']
    assert c.ledger.attempted_chunk_ids()==network
    assert stats['scheduler_model_call_count']==stats['llm_call_count']==stats['llm_error_count']==1
    assert stats['extraction_budget_ledger']['actual_model_call_count']==1
    assert stats['provider_deferred_chunk_count']==7
