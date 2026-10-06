import pytest
from data_collection_workflow.session_runtime import RunContext
from data_collection_workflow.workflow_recovery import RecoveryPlan, RecoveryAction, execute_recovery
from data_collection_workflow.nodes import extraction

@pytest.mark.parametrize('text,heading,is_heading',[
 ('Read More »\nSeptember 16, 2026\nNo Comments','Post navigation',False),
 ('CITY COUNCIL TRANSPORT BUDGET','Latest news',True),
 ('af1230'*60,'SHARES',False),
])
def test_forged_or_stale_recovery_plan_cannot_dispatch_non_evidence(tmp_path,monkeypatch,text,heading,is_heading):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def forbidden(*a,**kw):raise AssertionError('noise reached consumer')
    monkeypatch.setattr(extraction,'structured_extraction',forbidden)
    conf={'pipeline_mode':'evidence','universal':{'budget_policy':{'version':2,'mode':'adaptive'}}}
    ctx=RunContext(tmp_path,conf)
    chunk={'source_id':'s','chunk_id':'c','text':text,'source_is_heading':is_heading,
      'contains_target_data':True,'extraction_eligible_for_task_disease':True,'disease_relevance_status':'target_disease_match',
      'fetch_purpose':'data_extraction','chunk_kind':'text','bound_context_spans':[{'role':'heading','quote':heading}]}
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([RecoveryAction('extract','c','a','read existing span')]),context=ctx,artifacts={'evidence_chunks':[chunk]},budget=ctx.ledger)
    assert delta.actions[0]['status']=='skipped'
    assert delta.actions[0]['outcome'] is None
    assert delta.attempted_chunk_ids==[] and delta.records==[]
    assert ctx.ledger.snapshot()['used'].get('extraction',0)==0
