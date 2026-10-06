from concurrent.futures import ThreadPoolExecutor

import pytest

from data_collection_workflow.session_runtime import RunContext, BudgetExceeded, ResumeMismatch, derive_budget_limits


def config(limit=3):
    return {'pipeline_mode': 'evidence', 'structured_task': {'disease': 'measles'}, 'universal': {'budget_limits': {'search': limit, 'extraction': 4}, 'extraction_reserve': 2}}


def test_concurrent_dispatch_never_exceeds_budget(tmp_path):
    ctx = RunContext(tmp_path, config())
    def work(i):
        try:
            return ctx.call('search', {'q': i}, lambda: {'result': i})
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=10) as pool:
        values = list(pool.map(work, range(20)))
    assert sum(v is not None for v in values) == 3
    assert ctx.ledger.snapshot()['used']['search'] == 3


def test_empty_response_is_reused_only_in_same_session(tmp_path):
    ctx = RunContext(tmp_path/'one', config())
    assert ctx.call('search', {'q':'x'}, lambda: []) == []
    def forbidden():
        raise AssertionError('cached response called externally again')
    assert ctx.call('search', {'q':'x'}, forbidden) == []
    resumed = RunContext(tmp_path/'one', config(), resume=True)
    assert resumed.call('search', {'q':'x'}, forbidden) == []
    other = RunContext(tmp_path/'two', config())
    assert other.call('search', {'q':'x'}, lambda: [1]) == [1]
    assert other.ledger.snapshot()['used']['search'] == 1


def test_fresh_run_refuses_existing_session_and_resume_fingerprint_mismatch(tmp_path):
    RunContext(tmp_path, config())
    with pytest.raises(ResumeMismatch):
        RunContext(tmp_path, config())
    changed = config(); changed['structured_task']['disease'] = 'dengue'
    with pytest.raises(ResumeMismatch):
        RunContext(tmp_path, changed, resume=True)


def test_failed_attempt_charged_and_recovery_reserve_protected(tmp_path):
    ctx = RunContext(tmp_path, config())
    def bad():
        raise OSError('failed')
    with pytest.raises(OSError):
        ctx.call('extraction', {'id': 1}, bad)
    ctx.call('extraction', {'id': 2}, lambda: {})
    with pytest.raises(BudgetExceeded):
        ctx.call('extraction', {'id': 3}, lambda: {})
    ctx.call('extraction', {'id': 3}, lambda: {}, recovery=True)
    ctx.call('extraction', {'id': 4}, lambda: {}, recovery=True)
    assert ctx.ledger.snapshot()['used']['extraction'] == 4


def test_in_doubt_dispatch_remains_charged_and_never_automatically_repeated(tmp_path):
    ctx = RunContext(tmp_path, config())
    ticket = ctx.ledger.begin('search', {'fingerprint': ctx.fingerprint, 'input': {'q':'uncertain'}})
    resumed = RunContext(tmp_path, config(), resume=True)
    with pytest.raises(ResumeMismatch, match='in_doubt'):
        resumed.call('search', {'q':'uncertain'}, lambda: [])
    assert resumed.ledger.snapshot()['used']['search'] == 1


def test_budget_pool_derivation_and_disabled_pool():
    from data_collection_workflow.runtime_profile import default_workflow_run_config
    c = default_workflow_run_config()
    assert derive_budget_limits(c)['search'] == 76
    assert derive_budget_limits(c)['search_results'] == 144
    c['source_search']['authority_gap_retry']['enabled'] = False
    assert derive_budget_limits(c)['search'] == 20
    c['universal'] = {'budget_limits': {'search': 5}}
    assert derive_budget_limits(c)['search'] == 5


def test_model_dispatch_cache_and_protected_extraction_budget(tmp_path, monkeypatch):
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    ctx = RunContext(tmp_path, config())
    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, messages, config=None):
            self.calls += 1
            return {'records': []}
    model = Model()
    with ctx.activate():
        a = _invoke_with_config(model, [{'content':'source A'}], {'metadata':{'hdc_stage':'structured_extraction'}})
        b = _invoke_with_config(model, [{'content':'source A'}], {'metadata':{'hdc_stage':'structured_extraction'}})
    assert a == b == {'records': []}
    assert model.calls == 1
    assert ctx.ledger.snapshot()['used']['extraction'] == 1


def test_evidence_external_call_without_session_is_blocked(monkeypatch):
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.delenv('UNIVERSAL_SESSION_DIR', raising=False)
    class Model:
        def invoke(self, messages, config=None):
            raise AssertionError('must block before model')
    with pytest.raises(ResumeMismatch):
        _invoke_with_config(Model(), [], {'metadata':{'hdc_stage':'structured_extraction'}})


def test_compact_checkpoint_roundtrip_and_tamper_detection(tmp_path):
    from data_collection_workflow.session_runtime import SessionArtifactSerializer
    serde = SessionArtifactSerializer(tmp_path)
    value = {'documents': [{'clean_text': 'a'*10000}]}
    encoded = serde.dumps_typed(value)
    assert len(encoded[1]) < 1000
    assert serde.loads_typed(encoded) == value
    artifact = next(tmp_path.glob('*.bin'))
    artifact.write_bytes(b'changed')
    with pytest.raises(ResumeMismatch):
        serde.loads_typed(encoded)


def test_evidence_configuration_never_loads_seed_or_reference_evidence():
    from data_collection_workflow.session_runtime import prepare_universal_config
    from data_collection_workflow.runtime_profile import default_workflow_run_config
    original=default_workflow_run_config(); original['pipeline_mode']='evidence'
    original['workflow']['seed_source_overlay_path']='synthetic-seed-overlay.json'
    original['validation']['held_out_records_path']='synthetic-reference.csv'
    original['source_search']['combine_with_seed_catalog']=True
    prepared=prepare_universal_config(original)
    assert prepared['workflow']['seed_source_overlay_path'] is None
    assert not prepared['source_search']['combine_with_seed_catalog']
    assert prepared['validation']['held_out_records_path'] is None
    assert original['workflow']['seed_source_overlay_path']=='synthetic-seed-overlay.json'
    assert original['validation']['held_out_records_path']=='synthetic-reference.csv'
    assert original['source_search']['combine_with_seed_catalog'] is True


def test_preflight_failure_happens_before_session_or_paid_dispatch(tmp_path, monkeypatch):
    from data_collection_workflow.session_runtime import initialize_universal_run
    import data_collection_workflow.document_acquisition as acquisition
    def unavailable(*args, **kwargs): raise RuntimeError('OCR unavailable')
    monkeypatch.setattr(acquisition, 'preflight_acquisition', unavailable)
    with pytest.raises(RuntimeError, match='OCR unavailable'):
        initialize_universal_run(config(), tmp_path/'new')
    assert not (tmp_path/'new/.universal/operations.sqlite').exists()


def test_resume_event_log_keeps_prior_events_and_unique_sequence(tmp_path):
    import json
    from data_collection_workflow.run_events import RunEventWriter
    first=RunEventWriter(session_dir=tmp_path,session_id='session')
    first.append_run_started()
    old=first.events_path.read_bytes()
    second=RunEventWriter(session_dir=tmp_path,session_id='session',resume=True)
    second.append_run_started()
    assert second.events_path.read_bytes().startswith(old)
    rows=[json.loads(line) for line in second.events_path.read_text().splitlines()]
    assert [r['sequence'] for r in rows]==[1,2]


def test_overlapping_sessions_cannot_replace_active_runtime(tmp_path):
    from data_collection_workflow.session_runtime import get_runtime
    a=RunContext(tmp_path/'a',config());b=RunContext(tmp_path/'b',config())
    with a.activate():
        with pytest.raises(ResumeMismatch):
            with b.activate(): pass
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(get_runtime).result() is a



def test_empty_extraction_attempt_ids_persist_and_budget_skip_not_attempted(tmp_path,monkeypatch):
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    ctx=RunContext(tmp_path,config())
    class Model:
        def invoke(self,messages,config=None): return {'records':[]}
    def invoke(chunk,recovery=False):
        return _invoke_with_config(Model(),[{'content':chunk}],{'metadata':{'hdc_stage':'structured_extraction','hdc_chunk_id':chunk,'hdc_recovery':recovery}})
    with ctx.activate():
        invoke('a');invoke('b')
        with pytest.raises(BudgetExceeded): invoke('skipped')
        invoke('focused',True)
    resumed=RunContext(tmp_path,config(),resume=True)
    assert resumed.ledger.attempted_chunk_ids()==['a','b','focused']


def test_search_request_cache_identity_ignores_iteration_metadata():
    from data_collection_workflow.nodes.source_discovery import _search_request_identity
    a={'query':'measles 2024','provider_channel':'web_search','query_id':'a','iteration':1}
    b={**a,'query_id':'b','iteration':2}
    assert _search_request_identity(a)==_search_request_identity(b)


def test_real_chunk_wrapper_tracks_empty_results_and_uses_focused_reserve(tmp_path,monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.nodes.extraction import _load_llm_policy
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        def with_structured_output(self,schema): return self
        def invoke(self,messages,config=None): return {'records':[]}
    monkeypatch.setattr(llm_clients,'build_chat_model',lambda:Model())
    ctx=RunContext(tmp_path,config())
    policy=_load_llm_policy()
    with ctx.activate():
        for key in ['a','b']:
            llm_clients.extract_chunk_with_llm({'chunk_id':key,'text':'4 measles cases'},policy)
        llm_clients.extract_chunk_with_llm({'chunk_id':'c','text':'4 measles cases','focused_recovery_status':'focused_retry_attempted'},policy)
    assert ctx.ledger.attempted_chunk_ids()==['a','b','c']
    assert ctx.ledger.snapshot()['used']['extraction']==3


def test_provider_tokens_persist_once_across_cache_and_resume(tmp_path,monkeypatch):
    from langchain_core.messages import AIMessage
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        def invoke(self,messages,config=None):
            return AIMessage(content='ok',usage_metadata={'input_tokens':11,'output_tokens':3,'total_tokens':14})
    ctx=RunContext(tmp_path,config())
    with ctx.activate():
        _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'planner'}})
        _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'planner'}})
    resumed=RunContext(tmp_path,config(),resume=True)
    def forbidden(*args,**kwargs): raise AssertionError('cache replay dispatched')
    monkeypatch.setattr(Model,'invoke',forbidden)
    with resumed.activate(): _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'planner'}})
    snap=resumed.ledger.snapshot()
    assert snap['token_usage']=={'input_tokens':11,'output_tokens':3,'total_tokens':14,'availability':'complete','known_operations':1,'model_operations':1}
    assert snap['limits']['model:planner']==30
    audit=resumed.ledger.operation_audit()
    assert len(audit)==1 and audit[0]['execution_instance']
    assert audit[0]['budget_changes']=={'model:planner':1}
    assert audit[0]['operation_id']==1 and audit[0]['action_id']


def test_structured_callback_tokens_and_unknown_calls_are_partial(tmp_path,monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration,LLMResult
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        def invoke(self,messages,config=None):
            message=AIMessage(content='{}',usage_metadata={'input_tokens':7,'output_tokens':2,'total_tokens':9})
            for callback in config.get('callbacks') or []:
                callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
            return {'records':[]}
    ctx=RunContext(tmp_path,config())
    with ctx.activate(): _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'structured_extraction'}})
    ctx.call('model:unknown',{},lambda:{})
    snap=ctx.ledger.snapshot()
    assert snap['token_usage']['total_tokens']==9
    assert snap['token_usage']['availability']=='partial'
    assert snap['cost_usd'] is None and snap['cost_availability']=='unavailable'
    assert ctx.ledger.operation_audit()[1]['token_usage'] is None


def test_failed_model_reports_known_tokens_without_recharging_cache(tmp_path,monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration,LLMResult
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        def invoke(self,messages,config=None):
            message=AIMessage(content='invalid',usage_metadata={'input_tokens':5,'output_tokens':1,'total_tokens':6})
            for callback in config['callbacks']: callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
            raise ValueError('structured parser rejected response')
    ctx=RunContext(tmp_path,config())
    with ctx.activate():
        with pytest.raises(ValueError): _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'planner','recovery_round':2}})
    audit=ctx.ledger.operation_audit()[0]
    assert audit['status']=='failed' and audit['token_usage']['total_tokens']==6
    assert audit['recovery_round']==2
    assert ctx.ledger.snapshot()['used']['model:planner']==1
    assert ctx.ledger.snapshot()['used'].get('extraction',0)==0


def test_unknown_token_usage_is_null_and_configured_model_caps_independent(tmp_path):
    conf=config();conf['universal']['model_limits']={'planner':1}
    ctx=RunContext(tmp_path,conf)
    ctx.call('model:planner',{'x':1},lambda:{})
    with pytest.raises(BudgetExceeded): ctx.call('model:planner',{'x':2},lambda:{})
    snap=ctx.ledger.snapshot()
    assert snap['token_usage']['total_tokens'] is None
    assert snap['token_usage']['availability']=='unavailable'
    assert snap['limits']['extraction']==4 and snap['limits']['model:planner']==1
    assert snap['remaining']['extraction']==4


@pytest.mark.parametrize('second_usage,expected,availability', [
    (None,(7,2,9),'partial'),
    ({'total_tokens':6},(7,2,15),'partial'),
    ({'input_tokens':5},(12,2,9),'partial'),
    ({'output_tokens':4},(7,6,9),'partial'),
    ({'input_tokens':0,'output_tokens':0,'total_tokens':0},(7,2,9),'complete'),
])
def test_mixed_generation_usage_preserves_known_sum_as_partial(tmp_path,monkeypatch,second_usage,expected,availability):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration,LLMResult
    from data_collection_workflow.llm_clients import _invoke_with_config
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        def invoke(self,messages,config=None):
            known=AIMessage(content='known',usage_metadata={'input_tokens':7,'output_tokens':2,'total_tokens':9})
            unknown=AIMessage(content='second',response_metadata={'token_usage':second_usage} if second_usage is not None else {})
            response=LLMResult(generations=[[ChatGeneration(message=known),ChatGeneration(message=unknown)]])
            for callback in config['callbacks']: callback.on_llm_end(response)
            return {'records':[]}
    ctx=RunContext(tmp_path,config())
    with ctx.activate(): _invoke_with_config(Model(),[],{'metadata':{'hdc_stage':'structured_extraction'}})
    usage=ctx.ledger.snapshot()['token_usage']
    assert (usage['input_tokens'],usage['output_tokens'],usage['total_tokens'])==expected
    assert usage['availability']==availability
    assert ctx.ledger.operation_audit()[0]['token_usage'].get('availability','complete')==availability



def test_cache_lookup_is_read_only_and_uses_latest_exact_operation(tmp_path):
    ctx = RunContext(tmp_path, config())
    payload = {'url': 'https://example.invalid/report', 'parser_version': 'fixture'}
    assert not ctx.has_cached('model:planner', payload)
    ticket = ctx.ledger.begin('model:planner', {'fingerprint': ctx.fingerprint, 'input': payload})
    before = ctx.ledger.snapshot()
    assert not ctx.has_cached('model:planner', payload)
    assert ctx.ledger.snapshot() == before
    ctx.ledger.finish(ticket, error='failed')
    assert not ctx.has_cached('model:planner', payload)
    ctx.call('model:planner', payload, lambda: {'ok': True})
    before = ctx.ledger.snapshot()
    assert ctx.has_cached('model:planner', payload)
    assert not ctx.has_cached('model:other', payload)
    assert not ctx.has_cached('model:planner', {**payload, 'parser_version': 'changed'})
    assert ctx.ledger.snapshot() == before
