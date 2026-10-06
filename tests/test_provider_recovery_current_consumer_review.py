"""Current consumer-summary and body-only identity recovery controls."""
import importlib
import pytest
from test_provider_account_guard import context,payload,ProviderError
from data_collection_workflow import session_runtime as runtime,source_identity,llm_clients
from data_collection_workflow.workflow_recovery import RecoveryAction,RecoveryPlan,execute_recovery,plan_recovery,assess_collection_gaps

@pytest.mark.parametrize('shape',['summary','historical_ids','local_records'])
def test_current_consumer_deferred_ids_are_not_historical_attempts(tmp_path,monkeypatch,shape):
    from data_collection_workflow.nodes import extraction
    c=context(tmp_path);c.session_dir=tmp_path;c.frontier=None
    monkeypatch.setenv('PIPELINE_MODE','evidence');monkeypatch.setenv('LLM_PROVIDER','anthropic')
    with pytest.raises(ValueError):
        c.call('extraction',payload('c'),lambda:(_ for _ in ()).throw(ValueError('previous parser failure')))
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('other'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=c.ledger.snapshot()['used']
    record={'record_id':'local_r','source_id':'s','supporting_chunk_id':'c','cases_unspecified':4}
    response={'raw_records':[record] if shape=='local_records' else [],'extraction_attempted_chunk_ids':['c'] if shape!='summary' else [],'llm_extraction_summary':{'provider_deferred_chunk_ids':['c'],'provider_deferred_chunk_count':1,'llm_call_count':0}}
    monkeypatch.setattr(extraction,'structured_extraction',lambda state:response)
    state={'evidence_chunks':[{'chunk_id':'c','source_id':'s','text':'4 cases'}]}
    with c.activate():
        delta=execute_recovery(RecoveryPlan([RecoveryAction('extract','c','action-c','facts')]),context=c,artifacts=state,budget=c.ledger)
    assert delta.actions[0]['status']=='provider_deferred'
    assert delta.actions[0]['error']=='provider_account_limit'
    assert delta.attempted_chunk_ids==[]
    assert delta.records==response['raw_records']
    assert c.ledger.snapshot()['used']==before


def test_body_verified_identity_resume_preserves_fit_and_never_researches(tmp_path,monkeypatch):
    screening=importlib.import_module('data_collection_workflow.nodes.source_screening')
    from data_collection_workflow.acquisition_frontier import AcquisitionFrontier
    c=context(tmp_path);c.session_dir=tmp_path;c.frontier=AcquisitionFrontier(c.ledger.path)
    monkeypatch.setenv('PIPELINE_MODE','evidence');monkeypatch.setenv('LLM_PROVIDER','anthropic')
    monkeypatch.setenv('LLM_SOURCE_IDENTITY_REQUIRE_LLM','true');monkeypatch.setenv('LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK','false')
    monkeypatch.setattr(llm_clients,'llm_source_identity_enabled',lambda:True)
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('model:plan',payload('account-denied'),lambda:(_ for _ in ()).throw(ProviderError()))
    c.ledger.resume_provider(provider='anthropic',event_id='body-identity-restored',reason='credits restored')
    fields={'task_fit_evidence_origin':'fetched_content','task_fit_content_hash':'a'*64,'target_verification_status':'verified_target','disease_fit':'match','geography_fit':'match','date_fit':'match','final_screening_decision':'include','blocked_from_fetch':False}
    source={'source_id':'s','url':'https://unknown.test/report','title':'Dengue in Brazil 2025','source_identity_status':'provider_deferred','source_identity_unverified':True,**fields}
    state={'structured_task':{'disease':'dengue','location':'Brazil','start_date':'2025-01-01','end_date':'2025-12-31'},'collection_spec':{'disease':'dengue','geography':'Brazil','time_window':'2025'},'source_registry':[source],'documents':[{'source_id':'s','content_readable':True,'clean_text':'Dengue surveillance in Brazil 2025.'}]}
    calls=[]
    def identity(**kwargs):
        assert kwargs['source_entry']['source_id']=='s'
        return c.call('model:SourceIdentityAgentOutput',{'settings':{'provider':'anthropic'},'source':'s'},lambda:(calls.append('identity') or {}))
    monkeypatch.setattr(source_identity,'assess_source_identity_with_llm',identity)
    monkeypatch.setattr(screening,'source_screening',lambda state:pytest.fail('verified body must not be rescreened as metadata'))
    monkeypatch.setattr(screening,'source_critic_and_uncertainty_routing',lambda state:pytest.fail('body identity recovery must not rerun general critic'))
    with c.activate():
        gaps=[g for g in assess_collection_gaps(state) if g.kind=='source_assessment_pending']
        assert len(gaps)==1
        plan=plan_recovery(gaps,state=state,budget=c.ledger)
        assert [a.kind for a in plan.actions]==['assess_source']
        delta=execute_recovery(plan,context=c,artifacts=state,budget=c.ledger)
    assert calls==['identity']
    assert delta.actions[0]['status']=='completed'
    assert delta.sources[0]['source_identity_status']=='assessed'
    for key,value in fields.items():assert delta.sources[0][key]==value
    assert not delta.records
    assert c.ledger.snapshot()['used']=={'model:plan':1,'model:SourceIdentityAgentOutput':1}
