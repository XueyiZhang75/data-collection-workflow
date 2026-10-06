"""Independent pending-source recovery boundaries using actual saved discovery."""
from copy import deepcopy
from dataclasses import replace
import importlib
import pytest
from test_acquisition_identity_budget import offline, setup_search
from data_collection_workflow.workflow_recovery import (
    RecoveryPlan, execute_recovery, merge_recovery_delta,
    assess_collection_gaps, plan_recovery,
)

def pending(monkeypatch,tmp_path):
    ctx,search_plan,initial,searches,fetched=setup_search(monkeypatch,tmp_path,failure="provider")
    with ctx.activate():
        delta=execute_recovery(search_plan,context=ctx,artifacts=initial,budget=ctx.ledger)
    state={**initial,**merge_recovery_delta(initial,delta)}
    state["recovery_action_history"]=list(delta.actions)
    assert state["source_registry"][0]["source_identity_status"]=="pending_assessment"
    assert len(searches)==1
    return ctx,search_plan,state,searches,fetched

def identity_spy(monkeypatch,*,fail=False):
    calls=[]
    def identity(**kwargs):
        calls.append(kwargs["source_entry"]["source_id"])
        if fail:raise TimeoutError("independent identity outage")
        return {"llm_used":False}
    monkeypatch.setattr(importlib.import_module("data_collection_workflow.source_identity"),
                        "assess_source_identity_with_llm",identity)
    return calls

def assessment_plan(ctx,state):
    with ctx.activate():
        gaps=assess_collection_gaps(state)
        plan=plan_recovery(gaps,state=state,budget=ctx.ledger)
    return RecoveryPlan([a for a in plan.actions if a.kind=="assess_source"])

def test_saved_assessment_never_reenters_search_consumer_or_charges_search(monkeypatch,tmp_path):
    ctx,old_plan,state,searches,fetched=pending(monkeypatch,tmp_path)
    calls=identity_spy(monkeypatch)
    def no_search(**kwargs):pytest.fail("Saved pending source must not reenter search consumer")
    monkeypatch.setattr(importlib.import_module("data_collection_workflow.nodes.source_discovery"),
                        "_execute_iterative_query_batch",no_search)
    before=ctx.ledger.snapshot()["used"].copy()
    with ctx.activate():
        delta=execute_recovery(old_plan,context=ctx,artifacts=state,budget=ctx.ledger)
    final={**state,**merge_recovery_delta(state,delta)}
    assert calls==["new"]
    assert final["source_registry"][0]["source_identity_status"]=="assessed"
    assert ctx.ledger.snapshot()["used"]==before
    assert len(searches)==1
    assert not any(a.get("search_executed") or a.get("new_independent_source_ids") for a in delta.actions)

@pytest.mark.parametrize("change",["action","query","task","missing_task"])
def test_saved_pending_relation_cannot_be_reused_by_another_request(monkeypatch,tmp_path,change):
    ctx,old_plan,state,_,_=pending(monkeypatch,tmp_path)
    calls=identity_spy(monkeypatch)
    if change=="action":old_plan=RecoveryPlan([replace(old_plan.actions[0],action_id="different")])
    elif change=="query":
        old_plan=RecoveryPlan([replace(old_plan.actions[0],query={"query":"dengue Brazil 2024","source_type":"official_site_search"})])
    elif change=="missing_task":
        state.pop("structured_task")
    else:
        state["structured_task"]={**state["structured_task"],"disease":"dengue","location":"Brazil"}
    with ctx.activate():
        delta=execute_recovery(old_plan,context=ctx,artifacts=state,budget=ctx.ledger)
    assert not calls
    assert all(s.get("source_identity_status")=="pending_assessment" for s in state["source_registry"])
    assert not any(a.get("new_independent_source_ids") for a in delta.actions)

@pytest.mark.parametrize("boundary",[
    {"blocked_from_fetch":True,"blocked_from_fetch_reason":"user_excluded","final_screening_decision":"exclude"},
    {"task_fit_evidence_origin":"fetched_content","target_fit_status":"wrong_disease","content_hash":"verified-body"},
    {"task_fit_evidence_origin":"fetched_content","target_fit_status":"geography_mismatch","content_hash":"verified-body"},
    {"task_fit_evidence_origin":"fetched_content","target_fit_status":"temporal_mismatch","content_hash":"verified-body"},
    {"source_excluded_by_human_review":True},
    {"source_role_final":"excluded","task_fit_evidence_origin":"fetched_content"},
    {"task_fit_evidence_origin":"fetched_content","target_fit_status":"task_record_collection_candidate","content_hash":"verified-body"},
])
def test_pending_advisory_status_does_not_override_explicit_exclusion(monkeypatch,tmp_path,boundary):
    ctx,old_plan,state,_,fetched=pending(monkeypatch,tmp_path)
    calls=identity_spy(monkeypatch)
    state["source_registry"][0].update(boundary)
    original=deepcopy(state["source_registry"][0])
    assert not assessment_plan(ctx,state).actions
    with ctx.activate():delta=execute_recovery(old_plan,context=ctx,artifacts=state,budget=ctx.ledger)
    assert not calls and not fetched
    assert state["source_registry"][0]==original
    assert not any(a.get("new_independent_source_ids") for a in delta.actions)

def test_stale_assessment_input_cannot_assess_changed_source(monkeypatch,tmp_path):
    ctx,_,state,_,_=pending(monkeypatch,tmp_path)
    plan=assessment_plan(ctx,state)
    assert len(plan.actions)==1
    calls=identity_spy(monkeypatch)
    state["source_registry"][0]["snippet"]="A changed source reports a different observation."
    newer=assessment_plan(ctx,state)
    assert newer.actions and newer.actions[0].action_id!=plan.actions[0].action_id
    with ctx.activate():delta=execute_recovery(plan,context=ctx,artifacts=state,budget=ctx.ledger)
    assert not calls
    assert delta.actions[0]["status"]=="skipped"
    assert not any(a.get("new_independent_source_ids") for a in delta.actions)

def test_failed_assessment_errors_do_not_create_unbounded_action_identities(monkeypatch,tmp_path):
    ctx,_,state,_,_=pending(monkeypatch,tmp_path)
    calls=identity_spy(monkeypatch,fail=True)
    ids=[]
    for attempt in range(2):
        plan=assessment_plan(ctx,state)
        assert len(plan.actions)==1
        ids.append(plan.actions[0].action_id)
        with ctx.activate():delta=execute_recovery(plan,context=ctx,artifacts=state,budget=ctx.ledger)
        state={**state,**merge_recovery_delta(state,delta),
               "recovery_action_history":[*state["recovery_action_history"],*delta.actions]}
    assert ids[0]==ids[1]
    assert len(calls)==2
    assert not assessment_plan(ctx,state).actions


def test_saved_pending_cannot_cross_runtime_session(monkeypatch,tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import _saved_pending_discoveries
    ctx,old_plan,state,_,_=pending(monkeypatch,tmp_path/"first")
    other=RunContext(tmp_path/"other",ctx.config)
    assert not _saved_pending_discoveries(state,other,action=old_plan.actions[0])


def test_source_self_asserted_empty_task_does_not_authorize_missing_task(monkeypatch,tmp_path):
    from data_collection_workflow.workflow_recovery import _saved_pending_discoveries, fingerprint
    ctx,old_plan,state,_,_=pending(monkeypatch,tmp_path)
    state.pop("structured_task")
    state["source_registry"][0]["recovery_discovery"]["task_fingerprint"]=fingerprint({})
    assert not _saved_pending_discoveries(state,ctx,action=old_plan.actions[0])
