"""A paid discovery survives advisory failure without rediscovering its URLs."""
import importlib
import pytest
from data_collection_workflow.models import SearchResult, SearchProviderResponse
from data_collection_workflow.workflow_recovery import RecoveryAction,RecoveryPlan,execute_recovery,merge_recovery_delta,assess_collection_gaps,plan_recovery
from data_collection_workflow.session_runtime import RunContext

@pytest.fixture
def saved_discovery(monkeypatch,tmp_path,request):
    monkeypatch.setenv("PIPELINE_MODE","evidence")
    for name in ("ENABLE_LLM_SOURCE_CRITIC","ENABLE_LLM_SOURCE_CREDIBILITY","ENABLE_LLM_EXTRACTION","ENABLE_LIVE_FETCH"):
        monkeypatch.setenv(name,"false")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY","true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_REQUIRE_LLM","true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK","false")
    discovery=importlib.import_module("data_collection_workflow.nodes.source_discovery")
    identity=importlib.import_module("data_collection_workflow.source_identity")
    content=importlib.import_module("data_collection_workflow.nodes.content_processing")
    ctx=RunContext(tmp_path,{"pipeline_mode":"evidence","universal":{"budget_policy":{"version":2,"mode":"adaptive"},"budget_limits":{"search":1,"search_results":1,"source_targets":2,"http_requests":3}}})
    calls=[]
    class Provider:
        def search(self,query,**kwargs):
            calls.append(query["query"])
            return SearchProviderResponse(provider="fixture",raw_result_count=1,results=[SearchResult(url="https://public-data.example/report",title="Measles Canada 2025 surveillance",snippet="Canada reported 12 confirmed measles cases during 2025.")])
    settings=discovery.SourceSearchSettings(mode="fixture",provider="fixture",provider_channel_allowlist=["web_search"])
    monkeypatch.setattr(discovery,"_source_search_settings_from_env",lambda:settings)
    monkeypatch.setattr(discovery,"_provider_for_settings",lambda _:Provider())
    def unavailable(**kwargs):raise TimeoutError("identity temporarily unavailable")
    if getattr(request,"param",None)=="real_identity_failure":
        from data_collection_workflow.agents.source_identity_agent import assess_source_identity_with_llm
        from data_collection_workflow import llm_clients
        ctx.identity_payloads=[]
        def runtime_identity(**kwargs):
            def fail():
                ctx.identity_payloads.append(kwargs)
                raise TimeoutError("same identity request temporarily unavailable")
            return ctx.call("model:SourceIdentityAgentOutput",kwargs,fail)
        monkeypatch.setattr(llm_clients,"run_structured_llm_json",runtime_identity)
        monkeypatch.setattr(identity,"assess_source_identity_with_llm",assess_source_identity_with_llm)
    else:
        monkeypatch.setattr(identity,"assess_source_identity_with_llm",unavailable)
    monkeypatch.setattr(content,"content_fetch_and_parse",lambda local:{"documents":[]})
    monkeypatch.setattr(content,"document_quality_check",lambda local:{})
    monkeypatch.setattr(content,"evidence_chunking_and_data_presence_flagging",lambda local:{"evidence_chunks":[]})
    state={"structured_task":{"disease":"measles","location":"Canada","start_date":"2025-01-01","end_date":"2025-12-31"},"collection_spec":{"disease":"measles","geography":"Canada","time_window":"2025","collection_mode":"direct_collection"},"source_registry":[],"human_review_queue":[],"collection_trace":[]}
    action=RecoveryAction("search","annual","paid-action","coverage",query={"query":"measles Canada 2025 report","provider_channel":"web_search","source_type":"official_site_search"})
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([action]),context=ctx,artifacts=state,budget=ctx.ledger)
        state={**state,**merge_recovery_delta(state,delta)}
    assert delta.actions[0]["status"]=="failed",delta.actions
    assert len(state["source_registry"])==1
    assert state["source_registry"][0]["source_identity_status"]=="pending_assessment"
    assert ctx.ledger.snapshot()["remaining"]["search"]==0
    assert ctx.ledger.snapshot()["remaining"]["search_results"]==0
    return ctx,state,action,identity,discovery,calls

def test_real_discovery_pending_source_can_finish_when_search_and_result_budgets_are_zero(saved_discovery,monkeypatch):
    ctx,state,action,identity,discovery,calls=saved_discovery
    monkeypatch.setattr(identity,"assess_source_identity_with_llm",lambda **kwargs:{"llm_used":False})
    def forbidden(*args,**kwargs):raise AssertionError("must use saved source, never rediscover")
    monkeypatch.setattr(discovery,"_execute_iterative_query_batch",forbidden)
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([action]),context=ctx,artifacts=state,budget=ctx.ledger)
        final={**state,**merge_recovery_delta(state,delta)}
    assert final["source_registry"][0]["source_identity_status"]=="assessed"
    assert delta.actions[0]["search_executed"] is False
    assert delta.actions[0]["new_independent_source_ids"]==[]
    assert len(calls)==1
    assert ctx.ledger.snapshot()["used"]["search"]==1
    assert ctx.ledger.snapshot()["used"]["search_results"]==1

def test_pending_source_is_an_executable_assessment_gap_without_search_budget(saved_discovery):
    ctx,state,action,identity,discovery,calls=saved_discovery
    with ctx.activate():
        gaps=assess_collection_gaps(state)
        plan=plan_recovery(gaps,state=state,budget=ctx.ledger)
    assert any(g.kind=="source_assessment_pending" for g in gaps)
    assert [a.kind for a in plan.actions]==["assess_source"]
    assert plan.actions[0].source_version


@pytest.mark.parametrize("saved_discovery",["real_identity_failure"],indirect=True)
def test_real_identity_prompt_initial_and_recovery_share_two_attempt_limit(saved_discovery):
    ctx,state,old_action,identity,discovery,calls=saved_discovery
    for _ in range(3):
        with ctx.activate():
            plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
            delta=execute_recovery(plan,context=ctx,artifacts=state,budget=ctx.ledger)
        state={**state,**merge_recovery_delta(state,delta),
               "recovery_action_history":[*state.get("recovery_action_history",[]),*delta.actions]}
    assert len(ctx.identity_payloads)==2
    assert ctx.identity_payloads[0]==ctx.identity_payloads[1]
    assert ctx.ledger.snapshot()["used"]["model:SourceIdentityAgentOutput"]==2
    assert len(calls)==1
    assert state["source_registry"][0]["source_identity_status"]=="pending_assessment"
    assert not plan.actions


def test_assessment_consumer_leaving_pending_does_not_report_completed(saved_discovery,monkeypatch):
    ctx,state,old_action,identity,discovery,calls=saved_discovery
    screening=importlib.import_module("data_collection_workflow.nodes.source_screening")
    monkeypatch.setattr(screening,"source_screening",lambda local:{})
    monkeypatch.setattr(screening,"source_critic_and_uncertainty_routing",lambda local:{})
    with ctx.activate():
        plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
        delta=execute_recovery(plan,context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]["status"]=="failed"
    assert delta.actions[0]["error"]=="source_assessment_incomplete"
    assert delta.actions[0]["outcome"]!="source_assessment"
    assert not delta.actions[0]["new_independent_source_ids"]


def test_saved_readable_content_allows_assessment_without_more_acquisition(saved_discovery,monkeypatch):
    ctx,state,old_action,identity,discovery,calls=saved_discovery
    sid=state["source_registry"][0]["source_id"]
    state["documents"]=[{"document_id":"saved","source_id":sid,"clean_text":"Canada reported 12 confirmed measles cases in 2025.","content_readable":True,"parse_status":"parsed","acquisition_status":"readable","text_hash":"saved-content"}]
    from data_collection_workflow.workflow_recovery import _pending_assessment_blockers
    assert not _pending_assessment_blockers(state["source_registry"],state,{"remaining":{"extraction":1,"source_targets":0,"http_requests":0}})
    monkeypatch.setattr(identity,"assess_source_identity_with_llm",lambda **kwargs:{"llm_used":False})
    content=importlib.import_module("data_collection_workflow.nodes.content_processing")
    def forbidden(*args,**kwargs):pytest.fail("identity-only work must not fetch or parse content")
    for name in ("content_fetch_and_parse","document_quality_check","evidence_chunking_and_data_presence_flagging"):
        monkeypatch.setattr(content,name,forbidden)
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([old_action]),context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]["status"]=="completed"
    assert not delta.documents and not delta.chunks and not delta.records


@pytest.mark.parametrize("status",[None,"budget_deferred","blocked_llm_required"])
def test_assessment_consumer_without_assessed_source_is_not_completed(saved_discovery,monkeypatch,status):
    ctx,state,old_action,identity,discovery,calls=saved_discovery
    screening=importlib.import_module("data_collection_workflow.nodes.source_screening")
    rows=[] if status is None else [{**state["source_registry"][0],"source_identity_status":status}]
    monkeypatch.setattr(screening,"source_screening",lambda local:{"source_registry":rows})
    monkeypatch.setattr(screening,"source_critic_and_uncertainty_routing",lambda local:{})
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([old_action]),context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]["status"] not in {"completed","completed_empty"}
    assert delta.actions[0]["outcome"]!="source_assessment"
    assert not delta.actions[0]["new_independent_source_ids"]
