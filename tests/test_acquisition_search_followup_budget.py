"""Paid new-source discovery requires a still-affordable downstream path."""
import pytest
from data_collection_workflow.session_runtime import RunContext
import data_collection_workflow.workflow_recovery as recovery

KINDS=("search","search_results","source_targets","http_requests","extraction")
def config(**changes):
    return {"pipeline_mode":"evidence","universal":{"budget_policy":{"version":2,"mode":"adaptive"},"extraction_reserve":0,"budget_limits":{**dict.fromkeys(KINDS,4),**changes}}}
def state():
    return {"structured_task":{"disease":"measles","location":"Canada","start_date":"2025-01-01","end_date":"2025-12-31"}}
def gap():return recovery.RecoveryGap("source_missing","annual","missing coverage")

@pytest.mark.parametrize("empty",KINDS)
def test_search_requires_payable_followup_before_query_is_planned(tmp_path,monkeypatch,empty):
    ctx=RunContext(tmp_path,config(**{empty:0}))
    def forbidden(*args,**kwargs):raise AssertionError("deferred discovery must not construct a query")
    monkeypatch.setattr(recovery,"_recovery_query",forbidden)
    with ctx.activate():plan=recovery.plan_recovery([gap()],state=state(),budget=ctx.ledger)
    assert plan.actions==[] and plan.stop_reason=="budget_exhausted"
    assert plan.budget_deferred_gaps[0]["kind"]=="source_missing"
    assert plan.budget_deferred_gaps[0]["status"]=="budget_deferred"
    assert plan.budget_deferred_gaps[0]["blocked_budget_kinds"]==[empty]
    assert ctx.ledger.snapshot()["used"]=={}

@pytest.mark.parametrize("empty",KINDS)
def test_stale_search_plan_cannot_spend_without_downstream_budget(tmp_path,monkeypatch,empty):
    ctx=RunContext(tmp_path,config(**{empty:0}))
    action=recovery.RecoveryAction("search","annual","a","coverage",query={"query":"measles Canada 2025","search_direction":"national"})
    import importlib
    discovery=importlib.import_module("data_collection_workflow.nodes.source_discovery")
    def forbidden(*args,**kwargs):raise AssertionError("provider must not execute")
    monkeypatch.setattr(discovery,"_execute_iterative_query_batch",forbidden)
    with ctx.activate():delta=recovery.execute_recovery(recovery.RecoveryPlan([action]),context=ctx,artifacts=state(),budget=ctx.ledger)
    assert not delta.sources and not delta.documents and not delta.records
    assert delta.actions[0]["status"]=="budget_deferred"
    assert delta.actions[0]["outcome"]=="not_executed"
    assert delta.actions[0]["search_executed"] is False
    assert delta.actions[0]["new_independent_source_ids"]==[]
    assert delta.actions[0]["blocked_budget_kinds"]==[empty]
    assert ctx.ledger.snapshot()["used"]=={}

def test_affordable_search_still_plans(tmp_path):
    ctx=RunContext(tmp_path,config())
    with ctx.activate():plan=recovery.plan_recovery([gap()],state=state(),budget=ctx.ledger)
    assert [a.kind for a in plan.actions]==["search"]
    assert plan.budget_deferred_gaps==[]

def test_existing_content_extraction_is_not_blocked_by_source_cap(tmp_path):
    ctx=RunContext(tmp_path,config(source_targets=0))
    with ctx.activate():plan=recovery.plan_recovery([gap(),recovery.RecoveryGap("unprocessed_span","saved-span","unprocessed")],state=state(),budget=ctx.ledger)
    assert [a.kind for a in plan.actions]==["extract"]
    assert plan.stop_reason is None
    assert plan.budget_deferred_gaps[0]["blocked_budget_kinds"]==["source_targets"]

def test_budget_stop_reason_and_gap_are_preserved_after_search_stall(tmp_path,monkeypatch):
    ctx=RunContext(tmp_path,config(source_targets=0))
    monkeypatch.setattr(recovery,"_search_stalled",lambda state:True)
    with ctx.activate():result=recovery.recovery_control(state())
    assert result["recovery_stop_reason"]=="budget_exhausted"
    assert result["recovery_plan"]["budget_deferred_gaps"][0]["status"]=="budget_deferred"
    assert result["recovery_gaps"]
