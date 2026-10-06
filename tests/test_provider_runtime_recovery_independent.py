"""Independent provider budget, no-dispatch and body-identity controls."""
import pytest
from test_provider_account_guard import context,payload,ProviderError
from data_collection_workflow import session_runtime as runtime
from data_collection_workflow.workflow_recovery import RecoveryAction,RecoveryPlan,execute_recovery,_saved_pending_discoveries


def test_repeated_free_denials_do_not_exhaust_same_failed_action_before_explicit_resume(tmp_path):
    c=context(tmp_path)
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('same'),lambda:(_ for _ in ()).throw(ProviderError()))
    for _ in range(4):
        with pytest.raises(runtime.ProviderAccountLimit) as denied:
            c.call('extraction',payload('same'),lambda:pytest.fail('halted call dispatched'))
        assert denied.value.dispatched is False
    assert len(c.ledger.operation_audit())==1
    assert c.ledger.snapshot()['used']=={'extraction':1}
    assert c.ledger.failed_chunk_ids(retryable_only=True)==['same']
    c.ledger.resume_provider(provider='anthropic',event_id='independent-restore',reason='quota restored')
    assert c.call('extraction',payload('same'),lambda:{'records':[]})=={'records':[]}
    assert c.call('extraction',payload('same'),lambda:pytest.fail('successful cached response retried'))=={'records':[]}
    assert c.ledger.snapshot()['used']=={'extraction':2}
    assert c.ledger.failed_chunk_ids()==[]


def test_cached_other_result_preserved_after_halt_and_failed_resume_validation(tmp_path):
    c=context(tmp_path)
    assert c.call('model:plan',payload('cached'),lambda:{'plan':'original'})=={'plan':'original'}
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('bad'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=c.ledger.snapshot()
    with pytest.raises(ValueError):
        c.ledger.resume_provider(provider='anthropic',event_id='stop:99',reason='invalid namespace')
    assert c.ledger.snapshot()['used']==before['used']
    assert c.ledger.provider_status('anthropic')['status']=='halted'
    assert c.call('model:plan',payload('cached'),lambda:pytest.fail('cache dispatch'))=={'plan':'original'}
    assert c.ledger.snapshot()['used']==before['used']


def test_prior_failed_span_cannot_turn_a_free_provider_denial_into_new_failure(tmp_path,monkeypatch):
    from data_collection_workflow.nodes import extraction
    c=context(tmp_path);c.session_dir=tmp_path;c.frontier=None
    with pytest.raises(ValueError):
        c.call('extraction',payload('c'),lambda:(_ for _ in ()).throw(ValueError('previous parser failure')))
    with pytest.raises(runtime.ProviderAccountLimit):
        c.call('extraction',payload('other'),lambda:(_ for _ in ()).throw(ProviderError()))
    before=c.ledger.snapshot()['used']
    monkeypatch.setattr(extraction,'structured_extraction',lambda state:{'raw_records':[],'extraction_attempted_chunk_ids':[]})
    monkeypatch.setenv('LLM_PROVIDER','anthropic')
    state={'evidence_chunks':[{'chunk_id':'c','source_id':'s','text':'4 cases'}]}
    delta=execute_recovery(RecoveryPlan([RecoveryAction('extract','c','action-c','facts')]),context=c,artifacts=state,budget=c.ledger)
    assert delta.actions[0]['status']=='provider_deferred'
    assert delta.actions[0]['error']=='provider_account_limit'
    assert delta.attempted_chunk_ids==[]
    assert c.ledger.snapshot()['used']==before


def test_provider_deferred_identity_with_verified_body_remains_resumable(tmp_path):
    c=context(tmp_path);c.session_dir=tmp_path
    row={'source_id':'s','url':'https://example.test/report','source_identity_status':'provider_deferred','task_fit_evidence_origin':'fetched_content','target_verification_status':'verified_target','disease_fit':'match','geography_fit':'match','date_fit':'match'}
    state={'structured_task':{'disease':'dengue','location':'Brazil'},'source_registry':[row]}
    assert _saved_pending_discoveries(state,c)==[row]
