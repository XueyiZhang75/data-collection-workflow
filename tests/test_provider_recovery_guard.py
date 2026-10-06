import pytest
from data_collection_workflow.workflow_recovery import RecoveryGap, RecoveryAction, RecoveryPlan, plan_recovery, execute_recovery, recovery_control
from data_collection_workflow import session_runtime as runtime
from test_provider_account_guard import context, payload, ProviderError


def test_provider_limit_defers_model_work_but_keeps_local_reparse(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER','anthropic')
    snapshot={'remaining':{'extraction':100,'search':20,'search_results':100},'providers':{'anthropic':{'status':'halted','reason':'account balance'}}}
    gaps=[RecoveryGap('unprocessed_span','c','pending'),RecoveryGap('parse_missing','d','parse')]
    state={'documents':[{'document_id':'d','source_id':'s','clean_text':'measles 4 cases'}]}
    plan=plan_recovery(gaps,state=state,budget=snapshot)
    assert [a.kind for a in plan.actions]==['reparse']
    assert plan.provider_deferred_gaps[0]['target_id']=='c'
    plan=plan_recovery(gaps[:1],state=state,budget=snapshot)
    assert plan.stop_reason=='provider_account_limit'
    assert plan.budget_deferred_gaps==[]


def test_recovery_denial_is_pending_not_skipped_or_completed(tmp_path,monkeypatch):
    from data_collection_workflow.nodes import extraction
    c=context(tmp_path); c.session_dir=tmp_path; c.frontier=None
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('first'),lambda:(_ for _ in ()).throw(ProviderError()))
    monkeypatch.setattr(extraction,'structured_extraction',lambda s:{'raw_records':[], 'extraction_attempted_chunk_ids':[]})
    monkeypatch.setenv('LLM_PROVIDER','anthropic')
    result=execute_recovery(RecoveryPlan([RecoveryAction('extract','c','a','facts')]),context=c,artifacts={'evidence_chunks':[{'chunk_id':'c','source_id':'s','text':'4 cases'}]},budget=c.ledger)
    assert result.actions[0]['status']=='provider_deferred'
    assert result.attempted_chunk_ids==[]


@pytest.mark.parametrize('body_origin',[None,'fetched_content'])
def test_explicit_resume_reopens_initial_deferred_identity_without_research(tmp_path,monkeypatch,body_origin):
    from data_collection_workflow import source_identity
    import importlib
    source_screening=importlib.import_module("data_collection_workflow.nodes.source_screening")
    from data_collection_workflow.acquisition_frontier import AcquisitionFrontier
    from data_collection_workflow.workflow_recovery import assess_collection_gaps
    c=context(tmp_path); c.session_dir=tmp_path; c.frontier=AcquisitionFrontier(c.ledger.path)
    monkeypatch.setenv('PIPELINE_MODE','evidence'); monkeypatch.setenv('LLM_PROVIDER','anthropic')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_IDENTITY','true')
    calls=[]; restored=False
    def identity(**kwargs):
        def request():
            calls.append('identity')
            if not restored:raise ProviderError()
            return {}
        return c.call('model:SourceIdentityAgentOutput',{'settings':{'provider':'anthropic'},'source':kwargs['source_entry']['source_id']},request)
    monkeypatch.setattr(source_identity,'assess_source_identity_with_llm',identity)
    original={'source_id':'source-a','url':'https://unknown.invalid/report','title':'measles surveillance Canada 2025',
        'task_fit_evidence_origin':body_origin,'disease_fit':'match','geography_fit':'match','date_fit':'match',
        'target_verification_status':'verified_target','ready_for_content_fetch':True}
    spec={'disease':'measles','geography':'Canada','time_window':'2025'}
    with c.activate():
        rows,_,_=source_identity.apply_source_identity_to_registry([original],spec,llm_enabled=True,require_llm=True,allow_deterministic_fallback=False)
        state={'structured_task':{'disease':'measles','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},'collection_spec':spec,'source_registry':rows,
            'documents':[{'source_id':'source-a','document_id':'doc','content_readable':True,'clean_text':'Surveillance report'}]}
        gaps=[g for g in assess_collection_gaps(state) if g.kind=='source_assessment_pending']
        assert len(gaps)==1
        assert plan_recovery(gaps,state=state,budget=c.ledger).stop_reason=='provider_account_limit'
        c.ledger.resume_provider(provider='anthropic',event_id='explicit-identity-resume',reason='credits restored')
        restored=True
        plan=plan_recovery(gaps,state=state,budget=c.ledger)
        assert [a.kind for a in plan.actions]==['assess_source']
        monkeypatch.setattr(source_screening,'source_screening',lambda s:pytest.fail('identity-only resume must not rescreen body'))
        monkeypatch.setattr(source_screening,'source_critic_and_uncertainty_routing',lambda s:pytest.fail('identity-only resume must not rewrite routing'))
        delta=execute_recovery(plan,context=c,artifacts=state,budget=c.ledger)
    assert calls==['identity','identity']
    assert delta.actions[0]['status']=='completed'
    assert len(delta.sources)==1
    assert delta.sources[0]['source_identity_status']=='assessed'
    if body_origin:
        for key in ('task_fit_evidence_origin','disease_fit','geography_fit','date_fit','target_verification_status','ready_for_content_fetch'):
            assert delta.sources[0][key]==original[key]
    assert not delta.records
    assert c.ledger.snapshot()['used']=={'model:SourceIdentityAgentOutput':2}
