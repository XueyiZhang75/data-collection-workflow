from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
import pytest
from data_collection_workflow import session_runtime as runtime

MESSAGE = 'You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC.'

class ProviderError(Exception):
    def __init__(self, message=MESSAGE, status=400, code='invalid_request_error'):
        super().__init__(message)
        self.status_code=status
        self.body={'type':'error','error':{'type':code,'message':message}}

def context(tmp_path):
    c=runtime.RunContext.__new__(runtime.RunContext)
    c.ledger=runtime.RunBudgetLedger(tmp_path/'ops.sqlite',{'extraction':50,'model:plan':50},extraction_reserve=0,adaptive=True)
    c.fingerprint='same-task'; c.execution_instance='test'; c.recovery=False; c.recovery_round=None
    c.config={'llm':{'provider':'anthropic'}}
    return c

def payload(chunk, provider='anthropic'):
    return {'chunk_id':chunk,'settings':{'provider':provider},'messages':[chunk]}

@pytest.mark.parametrize('error',[
    ProviderError(), ProviderError('Your credit balance is too low to access the Anthropic API.'),
    ProviderError('Account has insufficient credits.',402,'insufficient_quota'),
    ProviderError('You exceeded your current quota, please check your plan and billing details.',429,'insufficient_quota'),
])
def test_explicit_account_denial_classified(error):
    assert runtime.provider_account_limit(error)

@pytest.mark.parametrize('error',[
    ProviderError('Rate limit reached for tokens per minute.',429,'rate_limit_error'),
    ProviderError('Usage limit for this request max_tokens exceeded.',400),
    ProviderError('input is too long',400), ProviderError('overloaded',529,'overloaded_error'),
    ProviderError('monthly limits overview in input invalid schema',400),
    ValueError(MESSAGE), ProviderError('Invalid API key',401,'authentication_error'),
])
def test_transient_or_request_error_does_not_halt(error):
    assert runtime.provider_account_limit(error) is None

def test_halt_is_persisted_same_provider_only_and_no_new_charges(tmp_path):
    c=context(tmp_path); calls=[]
    def deny(): calls.append('deny'); raise ProviderError()
    with pytest.raises(runtime.ProviderAccountLimit) as first:
        c.call('extraction',payload('one'),deny)
    assert first.value.dispatched is True
    with pytest.raises(runtime.ProviderAccountLimit) as blocked:
        c.call('model:plan',payload('two'),lambda:calls.append('forbidden'))
    assert blocked.value.dispatched is False
    assert calls==['deny']
    assert c.ledger.attempted_chunk_ids()==['one']
    assert c.ledger.snapshot()['used']=={'extraction':1}
    assert c.call('model:plan',payload('other','openai'),lambda:'ok')=='ok'
    reloaded=context(tmp_path)
    with pytest.raises(runtime.ProviderAccountLimit):
        reloaded.call('extraction',payload('three'),lambda:'forbidden')
    assert reloaded.ledger.provider_status('anthropic')['status']=='halted'

def test_completed_cache_survives_halt_and_explicit_resume_is_audited(tmp_path):
    c=context(tmp_path)
    assert c.call('extraction',payload('saved'),lambda:{'records':[]})=={'records':[]}
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('bad'),lambda:(_ for _ in ()).throw(ProviderError()))
    assert c.call('extraction',payload('saved'),lambda:pytest.fail('cache made network call'))=={'records':[]}
    old=c.ledger.snapshot()['used']
    event=c.ledger.resume_provider(provider='anthropic',event_id='manual-1',reason='account restored')
    assert c.ledger.resume_provider(provider='anthropic',event_id='manual-1',reason='account restored')==event
    assert c.ledger.snapshot()['used']==old
    assert c.call('extraction',payload('next'),lambda:'ok')=='ok'
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('bad-again'),lambda:(_ for _ in ()).throw(ProviderError()))
    c.ledger.resume_provider(provider='anthropic',event_id='manual-1',reason='account restored')
    assert c.ledger.provider_status('anthropic')['status']=='halted'  # replay cannot unlock a later denial
    with pytest.raises(ValueError):
        c.ledger.resume_provider(provider='anthropic',event_id='manual-1',reason='different')

def test_queued_calls_stop_but_admitted_concurrency_settles(tmp_path):
    c=context(tmp_path); entered=[]; release=Event(); started=Barrier(4)
    def work(i):
        def network():
            entered.append(i)
            if i<4:
                started.wait(timeout=5)
                if i==0: raise ProviderError()
                release.wait(timeout=5)
            return i
        try:return c.call('extraction',payload(str(i)),network)
        except runtime.ProviderAccountLimit:return 'blocked'
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,i) for i in range(16)]
        assert futures[0].result(timeout=10)=='blocked'
        release.set()
        results=[f.result(timeout=10) for f in futures]
    assert set(entered)==set(range(4))
    assert results.count('blocked')==13
    assert c.ledger.snapshot()['used']['extraction']==4
    assert c.ledger.attempted_chunk_ids()==['0','1','2','3']


def test_shared_client_model_stage_halts_extraction_but_not_local_transports(tmp_path,monkeypatch):
    from data_collection_workflow import llm_clients
    c=context(tmp_path); c.session_dir=tmp_path
    c.ledger=runtime.RunBudgetLedger(tmp_path/'transport.sqlite',{'extraction':20,'http_requests':10,'source_targets':10,'ocr':10,'search':10},extraction_reserve=0,adaptive=True)
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LANGSMITH_TRACING','false')
    calls=[]
    class Model:
        model_name='offline'
        def invoke(self,messages,config=None):calls.append(messages);raise ProviderError()
    cfg={'metadata':{'workflow_stage':'planning','llm_settings':{'provider':'anthropic','model':'offline'}}}
    with c.activate():
        with pytest.raises(runtime.ProviderAccountLimit) as first:
            llm_clients._invoke_with_config(Model(),[{'role':'user','content':'plan'}],cfg)
        assert first.value.dispatched
        cfg['metadata']['workflow_stage']='structured_extraction'
        cfg['metadata']['chunk_id']='never-dispatched'
        with pytest.raises(runtime.ProviderAccountLimit) as blocked:
            llm_clients._invoke_with_config(Model(),[{'role':'user','content':'extract'}],cfg)
        assert not blocked.value.dispatched
        for kind in ('http_request','ocr','search'):
            assert c.call(kind,{'resource':kind,'url':'https://offline.invalid/resource'},lambda:kind)==kind
    assert len(calls)==1
    assert c.ledger.attempted_chunk_ids()==[]
    assert c.ledger.snapshot()['used']['model:planning']==1
    assert c.ledger.snapshot()['used'].get('extraction',0)==0


def test_preflight_preserves_typed_account_stop_for_runner_local_completion(tmp_path,monkeypatch):
    from data_collection_workflow import llm_clients
    c=context(tmp_path);c.session_dir=tmp_path
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    class Model:
        model_name='offline'
        def with_structured_output(self,schema):return self
        def invoke(self,messages,config=None):raise ProviderError()
    monkeypatch.setattr(llm_clients,'build_chat_model',lambda *a,**k:Model())
    with c.activate():
        with pytest.raises(runtime.ProviderAccountLimit):
            llm_clients.preflight_llm_model({'provider':'anthropic','model':'offline'})
    assert c.ledger.snapshot()['providers']['anthropic']['status']=='halted'


def test_resume_event_id_cannot_collide_with_automatic_stop_events(tmp_path):
    c=context(tmp_path)
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('denied'),lambda:(_ for _ in ()).throw(ProviderError()))
    with pytest.raises(ValueError,match='reserved'):
        c.ledger.resume_provider(provider='anthropic',event_id='stop:2',reason='manual')
    assert c.ledger.provider_status('anthropic')['status']=='halted'
