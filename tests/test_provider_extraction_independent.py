"""Offline independent provider extraction review: no external IO/model construction."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import socket
import subprocess
from types import SimpleNamespace
import pytest
from data_collection_workflow import llm_clients, session_runtime as runtime
from data_collection_workflow.nodes import extraction
from data_collection_workflow.models import LLMExtractionOutput
from test_provider_account_guard import context, payload, ProviderError
from test_extraction_scheduler_efficiency import _chunk, _scheduler_policies, _disable_deterministic_and_recovery_paths

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*a, **k): raise AssertionError('External IO/model construction forbidden')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(llm_clients, 'build_chat_model', forbidden)
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    for key,value in {'LLM_EXTRACTION_MAX_CONCURRENCY':'1','LLM_EXTRACTION_ROLLING_YIELD_WINDOW':'40','LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS':'50','LLM_EXTRACTION_SAFETY_MAX_CALLS':'50','LLM_FOCUSED_RECOVERY_MAX_CALLS':'5','LLM_EXTRACTION_RECOVERY_RESERVED_CALLS':'5'}.items(): monkeypatch.setenv(key,value)
    monkeypatch.setattr(llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'independent-offline'})
    _disable_deterministic_and_recovery_paths(monkeypatch)
    monkeypatch.setattr(extraction,'_ordered_llm_chunks',lambda rows,*a:rows)

def deny_elsewhere(c):
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('model:plan',payload('other-stage'),lambda:(_ for _ in ()).throw(ProviderError()))

def test_admitted_successful_batch_responses_survive_concurrent_halt(tmp_path,monkeypatch):
    c=context(tmp_path); barrier=Barrier(4); failed=Event(); network=[]
    def model(chunk,policy):
        def request():
            network.append(chunk['source_id']); barrier.wait(timeout=5)
            if chunk['source_id']=='0': raise ProviderError()
            failed.wait(timeout=5)
            return LLMExtractionOutput(chunk_is_relevant=True)
        try:return c.call('extraction',payload(chunk['chunk_id']),request)
        finally:
            if chunk['source_id']=='0':failed.set()
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    rows=[_chunk(str(i),f'Patient {i} confirmed.') for i in range(8)]
    out,stats=extraction._run_bounded_llm_calls(rows,policy=SimpleNamespace(),max_concurrency=4)
    assert set(network)=={'0','1','2','3'}
    assert stats['model_call_count']==4
    assert sum(row['output'] is not None for row in out)==3
    assert sum(isinstance(row['error'],runtime.ProviderAccountLimit) for row in out)==5
    assert sum(row['model_call'] for row in out)==4
    assert c.ledger.snapshot()['used']['extraction']==4

def test_successful_cache_under_halt_is_not_new_dispatch(tmp_path,monkeypatch):
    c=context(tmp_path); row=_chunk('saved','Patient confirmed by PCR.'); calls=[]
    def model(chunk,policy):
        return c.call('extraction',payload(chunk['chunk_id']),lambda:(calls.append('called') or LLMExtractionOutput(chunk_is_relevant=True)))
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    first,stats=extraction._run_bounded_llm_calls([row],policy=SimpleNamespace(),max_concurrency=1)
    deny_elsewhere(c)
    second,stats=extraction._run_bounded_llm_calls([row],policy=SimpleNamespace(),max_concurrency=1)
    assert second[0]['output'] is not None
    assert stats['model_call_count']==0
    assert calls==['called']
    assert c.ledger.snapshot()['used']['extraction']==1

@pytest.mark.parametrize('mode',['primary','metric','focused'])
def test_halt_between_gate_and_runtime_dispatch_does_not_claim_attempt(tmp_path,monkeypatch,mode):
    c=context(tmp_path); c.session_dir=tmp_path
    row=_chunk('row','Patient confirmed by PCR.',contains_target_data=True)
    monkeypatch.setattr(extraction,'_direct_llm_chunk_allowed',lambda *a:True)
    if mode=='metric':
        row.update(chunk_kind='metric_row',table_id='cases',row_id='one')
        monkeypatch.setattr(extraction,'_deterministic_metric_row_records',lambda *a,**k:[])
    if mode=='focused':
        monkeypatch.setattr(extraction,'_focused_recovery_queue',lambda *a:[dict(row)])
    chunks=[] if mode=='focused' else [row]
    network=[]
    def model(chunk,policy):
        if c.ledger.provider_status('anthropic') is None:deny_elsewhere(c)
        return c.call('extraction',payload(chunk['chunk_id']),lambda:(network.append(chunk['chunk_id']) or LLMExtractionOutput(chunk_is_relevant=True)))
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    lp,dp=_scheduler_policies()
    with c.activate():
        _,stats=extraction._llm_extract_records_from_chunks(chunks,lp,dp,fallback_to_rule_based=False,context={'collection_mode':'direct_collection' if mode=='metric' else 'standard'})
    assert network==[]
    assert c.ledger.snapshot()['used'].get('extraction',0)==0
    assert stats['scheduler_model_call_count']==0
    assert stats['llm_call_count']==0
    assert stats['llm_error_count']==0
    assert stats['extraction_budget_ledger']['actual_model_call_count']==0
    assert stats['provider_deferred_chunk_ids']==['chunk_row']
    assert sum(r['attempted_count'] for r in stats['extraction_budget_by_source'].values())==0


def test_full_primary_consumes_admitted_results_once_after_halt(tmp_path,monkeypatch):
    c=context(tmp_path); c.session_dir=tmp_path; barrier=Barrier(4); stopped=Event(); network=[]
    monkeypatch.setenv('LLM_EXTRACTION_MAX_CONCURRENCY','4')
    def model(chunk,policy):
        def request():
            network.append(chunk['source_id']); barrier.wait(timeout=5)
            if chunk['source_id']=='0':raise ProviderError()
            stopped.wait(timeout=5)
            return LLMExtractionOutput(chunk_is_relevant=True)
        try:return c.call('extraction',payload(chunk['chunk_id']),request)
        finally:
            if chunk['source_id']=='0':stopped.set()
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    rows=[_chunk(str(i),f'Patient {i} was confirmed by PCR.',contains_target_data=True,case_span_quote=f'Patient {i} was confirmed by PCR.',case_span_id=str(i)) for i in range(8)]
    lp,dp=_scheduler_policies()
    with c.activate():
        _,stats=extraction._llm_extract_records_from_chunks(rows,lp,dp,fallback_to_rule_based=False,context={'collection_mode':'standard'})
    assert set(network)=={'0','1','2','3'}
    assert stats['scheduler_model_call_count']==stats['llm_call_count']==4
    assert stats['llm_success_count']==3
    assert stats['llm_error_count']==1
    assert stats['extraction_budget_ledger']['actual_model_call_count']==4
    assert stats['provider_deferred_chunk_count']==4
    assert set(stats['provider_deferred_chunk_ids'])=={f'chunk_{i}' for i in range(4,8)}

def test_halted_metric_rows_remain_deferred_individually(tmp_path,monkeypatch):
    c=context(tmp_path); c.session_dir=tmp_path; deny_elsewhere(c)
    rows=[_chunk('same',f'Row {i}',chunk_id=f'row_{i}',contains_target_data=True,chunk_kind='metric_row',table_id='cases',row_id=str(i)) for i in range(3)]
    monkeypatch.setattr(extraction,'_direct_llm_chunk_allowed',lambda *a:True)
    monkeypatch.setattr(extraction,'_deterministic_metric_row_records',lambda *a,**k:[])
    def model(chunk,policy):
        return c.call('extraction',payload(chunk['chunk_id']),lambda:pytest.fail('Undispatched model called'))
    monkeypatch.setattr(llm_clients,'extract_chunk_with_llm',model)
    lp,dp=_scheduler_policies()
    with c.activate():
        _,stats=extraction._llm_extract_records_from_chunks(rows,lp,dp,fallback_to_rule_based=False,context={'collection_mode':'direct_collection'})
    assert stats['llm_call_count']==stats['scheduler_model_call_count']==0
    assert stats['llm_error_count']==0
    assert set(stats['provider_deferred_chunk_ids'])=={f'row_{i}' for i in range(3)}
    assert c.ledger.snapshot()['used'].get('extraction',0)==0
    assert sum(r['attempted_count'] for r in stats['extraction_budget_by_source'].values())==0
