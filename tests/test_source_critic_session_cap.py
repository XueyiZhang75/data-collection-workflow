"""Source review caps include recovery and resume, preserving smaller overrides."""
from concurrent.futures import ThreadPoolExecutor
import pytest
from data_collection_workflow.session_runtime import RunContext, BudgetExceeded, derive_budget_limits

KIND = 'model:SourceCriticAgentOutput'

def config(cap, adaptive):
    return {'pipeline_mode':'evidence', 'llm':{'source_critic':{'max_sources':cap}},
        'universal':{'budget_policy':{'version':2,'mode':'adaptive'}} if adaptive else {}}

@pytest.mark.parametrize('adaptive',[False,True])
@pytest.mark.parametrize('cap',[0,1,4])
def test_explicit_source_critic_limit_is_a_session_cap(cap,adaptive):
    assert derive_budget_limits(config(cap,adaptive))[KIND] == cap

@pytest.mark.parametrize('adaptive',[False,True])
def test_smaller_explicit_model_override_wins(adaptive):
    conf=config(4,adaptive)
    conf['universal']['model_limits']={'SourceCriticAgentOutput':2}
    assert derive_budget_limits(conf)[KIND] == 2
    conf['universal']['model_limits']['SourceCriticAgentOutput']=20
    assert derive_budget_limits(conf)[KIND] == 4

@pytest.mark.parametrize('adaptive',[False,True])
def test_recovery_resume_and_failure_share_source_critic_cap(tmp_path,adaptive):
    conf=config(2,adaptive)
    ctx=RunContext(tmp_path,conf)
    def fail(): raise OSError('provider failed')
    with pytest.raises(OSError): ctx.call(KIND,{'source':'a'},fail)
    assert ctx.call(KIND,{'source':'b'},lambda:[],recovery=True)==[]
    resumed=RunContext(tmp_path,conf,resume=True)
    def forbidden(): raise AssertionError('must reuse saved empty response')
    assert resumed.call(KIND,{'source':'b'},forbidden,recovery=True)==[]
    with pytest.raises(BudgetExceeded): resumed.call(KIND,{'source':'c'},forbidden,recovery=True)
    snap=resumed.ledger.snapshot()
    assert snap['used'][KIND]==2 and snap['remaining'][KIND]==0
    assert snap['used'].get('extraction',0)==0

@pytest.mark.parametrize('adaptive',[False,True])
def test_parallel_source_critic_dispatch_cannot_bypass_phase_cap(tmp_path,adaptive):
    ctx=RunContext(tmp_path,config(4,adaptive))
    def work(i):
        try: return ctx.call(KIND,{'source':i},lambda:i,recovery=i%2==0)
        except BudgetExceeded: return None
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(work,range(12)))
    assert sum(v is not None for v in results)==4
    assert ctx.ledger.snapshot()['used'][KIND]==4
