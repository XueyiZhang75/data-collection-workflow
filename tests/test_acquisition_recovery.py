"""Recovery must execute useful evidence work and report real outcomes."""
import hashlib

import pytest

from data_collection_workflow.workflow_recovery import (
    RecoveryAction, RecoveryPlan, assess_collection_gaps, execute_recovery,
    plan_recovery, recovery_control,
)
from data_collection_workflow.session_runtime import RunContext


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "false")
    monkeypatch.setenv("ENABLE_LIVE_SEARCH", "false")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")


def runtime(tmp_path):
    return RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {"budget_limits": {"search": 8, "search_results": 40, "fetch": 20, "fetch_ordinary": 10, "extraction": 20}, "extraction_reserve": 0}})


def source_state(text="France reported 12 confirmed measles cases during 2025.", *, readiness="ready"):
    digest = hashlib.sha256(text.encode()).hexdigest()
    source = {"source_id": "s", "url": "https://offline.invalid/report", "final_screening_decision": "include_for_content_fetch", "source_role_final": "collection"}
    document = {"source_id": "s", "document_id": "d", "content_hash": digest, "text_hash": digest, "clean_text": text, "content_readable": True, "acquisition_status": "readable", "parse_status": "parsed"}
    chunk = {"source_id": "s", "chunk_id": "c", "document_hash": digest, "document_id": "d", "text": text, "char_start": 0, "char_end": len(text), "contains_target_data": True, "extraction_eligible_for_task_disease": True, "disease_relevance_status": "target_disease_match", "extraction_readiness": readiness, "chunk_kind": "text", "fetch_purpose": "data_extraction"}
    return {"structured_task": {"disease": "measles", "location": "France", "start_date": "2025-01-01", "end_date": "2025-12-31"}, "source_registry": [source], "documents": [document], "evidence_chunks": [chunk]}


def test_empty_error_response_is_fetch_work_not_saved_content_reparse(tmp_path):
    state = source_state()
    raw = tmp_path / "empty.raw"
    raw.write_bytes(b"")
    state["documents"] = [{"source_id": "s", "document_id": "d", "raw_artifact_path": "empty.raw", "content_hash": hashlib.sha256(b"").hexdigest(), "clean_text": "", "content_readable": False, "request_success": False, "acquisition_status": "request_error", "parse_status": "not_started", "fetch_error": "response exceeds max_bytes"}]
    state["evidence_chunks"] = []
    ctx = runtime(tmp_path)
    with ctx.activate():
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=ctx.ledger)
    assert not any(action.kind == "reparse" for action in plan.actions)
    assert any(action.kind == "fetch" for action in plan.actions)


@pytest.mark.parametrize("quality", ["not_task_relevant", "unsupported_format"])
def test_terminal_document_disposition_does_not_generate_reparse(quality):
    state = source_state()
    state["documents"][0].update(raw_artifact_path="acquisition/saved.raw", quality_status=quality, acquisition_status="unsupported_format" if quality == "unsupported_format" else "readable")
    state["evidence_chunks"] = []
    gaps = assess_collection_gaps(state)
    assert not any(gap.kind in {"parse_missing", "parse_failed"} for gap in gaps)


def test_not_ready_span_is_not_an_executable_extraction_gap():
    state = source_state("For Authors\nFor Referees\nSubmit manuscript", readiness="not_ready")
    gaps = assess_collection_gaps(state)
    assert not any(gap.kind == "unprocessed_span" for gap in gaps)


def test_consumer_skipped_span_is_not_reported_as_completed(monkeypatch, tmp_path):
    from data_collection_workflow.nodes import extraction

    state = source_state("For Authors\nFor Referees\nSubmit manuscript", readiness="not_ready")
    monkeypatch.setattr(extraction.llm_clients, "llm_extraction_enabled", lambda: True)
    monkeypatch.setattr(extraction.llm_clients, "get_llm_settings", lambda: {"provider": "offline", "model": "offline"})
    ctx = runtime(tmp_path)
    with ctx.activate():
        delta = execute_recovery(RecoveryPlan([RecoveryAction("extract", "c", "extract-c", "extract existing content")]), context=ctx, artifacts=state, budget=ctx.ledger)
    assert delta.attempted_chunk_ids == []
    assert delta.actions[0]["status"] != "completed"
    assert delta.actions[0].get("error") or delta.actions[0].get("reason")
    assert not ctx.ledger.snapshot()["used"].get("extraction")


def test_unreadable_reparse_cannot_report_completed(tmp_path):
    raw = tmp_path / "empty.raw"
    raw.write_bytes(b"")
    state = {"structured_task": {"disease": "measles"}, "documents": [{"source_id": "s", "document_id": "d", "url": "https://offline.invalid/report.pdf", "raw_artifact_path": "empty.raw", "content_hash": hashlib.sha256(b"").hexdigest(), "clean_text": "", "content_readable": False, "content_type": "text/plain", "acquisition_status": "request_error"}]}
    ctx = runtime(tmp_path)
    with ctx.activate():
        delta = execute_recovery(RecoveryPlan([RecoveryAction("reparse", "d", "reparse-d", "repair parsing")]), context=ctx, artifacts=state, budget=ctx.ledger)
    assert delta.actions[0]["status"] != "completed"
    assert delta.actions[0].get("error") or delta.actions[0].get("reason")


@pytest.mark.parametrize("disease,country", [("measles", "France"), ("pertussis", "Canada")])
def test_missing_field_gap_has_a_concrete_executable_repair(disease, country):
    state = source_state(f"{country} reported 12 confirmed {disease} cases during 2025.")
    state["structured_task"].update(disease=disease, location=country)
    state["extraction_attempted_chunk_ids"] = ["c"]
    state["raw_records"] = [{"record_id": "r", "source_id": "s", "supporting_chunk_id": "c", "disease": disease, "cases_confirmed": 12}]
    state["candidate_records"] = [{**state["raw_records"][0], "evidence_qualification": {"status": "candidate", "reasons": ["missing_geography", "missing_observation_period"]}}]
    plan = plan_recovery(assess_collection_gaps(state), state=state, budget={"remaining": {"search": 8, "fetch": 20, "fetch_ordinary": 10, "extraction": 20}})
    assert plan.actions, "A field-repair diagnosis needs an executable source-bound action."
    assert any(action.target_id in {"r", "c", "s"} for action in plan.actions)


def test_new_usable_span_is_not_starved_by_one_round_without_fact_gain(tmp_path):
    state = source_state()
    state.update(recovery_round=1, recovery_previous_gain=[], normalized_records=[])
    ctx = runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "extract" and action["target_id"] == "c" for action in result["recovery_plan"]["actions"])


def test_new_independent_source_is_not_starved_by_one_round_without_fact_gain(tmp_path):
    state = source_state()
    state.update(recovery_round=1, recovery_previous_gain=[], normalized_records=[], extraction_attempted_chunk_ids=["c"])
    state["source_registry"].append({"source_id": "independent", "url": "https://another-offline.invalid/report", "final_screening_decision": "include_for_content_fetch", "source_role_final": "collection"})
    ctx = runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "fetch" and action["target_id"] == "independent" for action in result["recovery_plan"]["actions"])


# These controls are adaptive-policy requirements; strict legacy round limits
# are covered separately by the existing compatibility suite.
def adaptive_runtime(tmp_path):
    return RunContext(tmp_path, {
        "pipeline_mode": "evidence",
        "universal": {
            "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 50},
            "budget_limits": {"source_targets": 200, "http_requests": 1000,
                              "search": 88, "search_results": 380, "extraction": 2400},
            "extraction_reserve": 0,
        },
    })


def settled_state(*, covered=False):
    """A real bound observation, with its original span already consumed."""
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    from data_collection_workflow.workflow_recovery import _gain

    state = source_state()
    state["extraction_attempted_chunk_ids"] = ["c"]
    state["source_coverage_requirements"] = [{"requirement_id": "annual",
        "disease": "measles", "country": "France", "reporting_period": "2025"}]
    if covered:
        record = {"record_id": "annual-row", "source_id": "s", "supporting_chunk_id": "c",
                  "disease": "measles", "country": "France", "reporting_period": "2025",
                  "cases_confirmed": 12}
        record["evidence_qualification"] = assess_record_evidence(record,
            contract=state["structured_task"], evidence_index=build_evidence_index(state)).to_dict()
        assert record["evidence_qualification"]["status"] == "qualified"
        state["qualified_records"] = [record]
    state["recovery_previous_gain"] = _gain(state)
    return state


def search_history(*directions, new_ids=(), last_status="completed"):
    rows = [{"kind": "search", "target_id": "annual", "action_id": f"searched-{i}",
             "status": "completed", "search_direction": direction,
             "new_independent_source_ids": [], "query": f"measles France 2025 {direction}"}
            for i, direction in enumerate(directions)]
    if rows:
        rows[-1].update(status=last_status, new_independent_source_ids=list(new_ids))
    return rows


@pytest.mark.parametrize("round_number", [2, 5])
def test_adaptive_recovery_has_no_global_two_round_cutoff_with_usable_work(tmp_path, round_number):
    state = source_state()
    state.update(recovery_round=round_number, recovery_previous_gain=[])
    ctx = adaptive_runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "extract" and action["target_id"] == "c"
               for action in result["recovery_plan"]["actions"])


@pytest.mark.parametrize("contribution", ["new_period", "new_jurisdiction", "new_field", "independent_source"])
def test_main_table_coverage_does_not_stop_new_evidence_work(tmp_path, contribution):
    state = settled_state(covered=True)
    state.update(recovery_round=3, recovery_action_history=search_history("official", "research"))
    if contribution == "independent_source":
        state["source_registry"].append({"source_id": "independent", "url": "https://independent.invalid/report",
            "source_role_final": "collection", "final_screening_decision": "include_for_content_fetch"})
        expected = ("fetch", "independent")
    else:
        text = {"new_period": "France reported 4 confirmed measles cases during June 2025.",
                "new_jurisdiction": "Lyon, France reported 3 confirmed measles cases during 2025.",
                "new_field": "France reported 2 measles deaths during 2025."}[contribution]
        digest = hashlib.sha256(text.encode()).hexdigest()
        state["documents"].append({**state["documents"][0], "document_id": "new-doc", "clean_text": text,
            "content_hash": digest, "text_hash": digest})
        state["evidence_chunks"].append({**state["evidence_chunks"][0], "chunk_id": "new-span",
            "document_id": "new-doc", "document_hash": digest, "text": text, "char_end": len(text)})
        expected = ("extract", "new-span")
    ctx = adaptive_runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["source_coverage_audit"]["coverage_status"] == "complete"
    assert result["recovery_stop_reason"] is None
    assert expected in {(a["kind"], a["target_id"]) for a in result["recovery_plan"]["actions"]}


@pytest.mark.parametrize("history", [
    search_history("official"),
    search_history("official", "official"),
    search_history("official", "research", last_status="failed"),
    search_history("official", "research", new_ids=("new-independent",)),
], ids=["one_direction", "same_direction_twice", "failed_direction", "new_source_resets_streak"])
def test_search_stopping_requires_two_distinct_completed_empty_directions(tmp_path, history):
    state = settled_state()
    state.update(recovery_round=len(history), recovery_action_history=history)
    ctx = adaptive_runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "search" for action in result["recovery_plan"]["actions"])


def test_two_distinct_empty_directions_and_no_executable_work_stop_as_no_progress(tmp_path):
    state = settled_state()
    state.update(recovery_round=2, recovery_action_history=search_history("official", "research"))
    ctx = adaptive_runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] == "no_progress"
    assert result["recovery_plan"]["actions"] == []


def test_completed_main_table_still_requires_independent_search_directions(tmp_path):
    state = settled_state(covered=True)
    state.update(recovery_round=1, recovery_action_history=search_history("official"))
    ctx = adaptive_runtime(tmp_path)
    with ctx.activate():
        result = recovery_control(state)
    assert result["source_coverage_audit"]["coverage_status"] == "complete"
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "search" for action in result["recovery_plan"]["actions"])


def test_persistent_frontier_work_wins_over_two_empty_search_directions(tmp_path):
    from data_collection_workflow.acquisition_frontier import AcquisitionFrontier

    state = settled_state(covered=True)
    state.update(recovery_round=3, recovery_action_history=search_history("official", "research"))
    ctx = adaptive_runtime(tmp_path)
    # Exercise the real shared queue. Task 2 separately verifies automatic
    # RunContext attachment; this test isolates the recovery consumer contract.
    ctx.frontier = AcquisitionFrontier(tmp_path / ".universal" / "operations.sqlite")
    ctx.frontier.enqueue(target_id="https://new.invalid/supplement.csv",
        url="https://new.invalid/supplement.csv", source_id="supplement", priority=50,
        payload={"source_id": "supplement", "url": "https://new.invalid/supplement.csv",
                 "final_screening_decision": "include_for_content_fetch", "source_role_final": "collection"})
    with ctx.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert result["recovery_plan"]["actions"], "Pending frontier acquisition must become executable recovery work."
    assert ctx.frontier.snapshot()["counts"].get("pending") == 1, "Planning must not claim or drop queue work."


def test_recovery_query_rotates_past_a_completed_search_direction():
    from data_collection_workflow.workflow_recovery import _recovery_query

    state = settled_state()
    first = _recovery_query(state, "annual")
    assert first
    state["recovery_action_history"] = [{"kind": "search", "target_id": "annual",
        "action_id": "first-direction", "status": "completed", "query": first["query"],
        "search_direction": first.get("search_direction") or first["query"], "new_independent_source_ids": []}]
    second = _recovery_query(state, "annual")
    assert second and second["query"] != first["query"]


def test_completed_search_action_does_not_hide_a_new_direction_for_same_gap():
    from dataclasses import asdict
    from data_collection_workflow.workflow_recovery import RecoveryGap, _recovery_query

    state = settled_state()
    remaining = {"remaining": {"search": 8, "fetch": 20, "fetch_ordinary": 10,
                               "source_targets": 200, "http_requests": 1000, "extraction": 20}}
    gaps = [RecoveryGap("source_missing", "annual", "qualified coverage missing")]
    initial = plan_recovery(gaps, state=state, budget=remaining)
    assert initial.actions
    first = initial.actions[0]
    query = _recovery_query(state, "annual")
    state["recovery_action_history"] = [{**asdict(first), "status": "completed", "query": query["query"],
        "search_direction": query.get("search_direction") or query["query"], "new_independent_source_ids": []}]
    resumed = plan_recovery(gaps, state=state, budget=remaining)
    assert any(action.kind == "search" and action.action_id != first.action_id for action in resumed.actions)


@pytest.mark.parametrize("independent", [False, True], ids=["same_report_new_format", "independent_corroboration"])
def test_bound_independent_support_is_gain_but_same_report_format_is_not(independent):
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    from data_collection_workflow.workflow_recovery import _gain

    state = settled_state(covered=True)
    state["source_registry"][0].update(original_publisher="First publisher", source_identity_unverified=False)
    before = set(_gain(state))
    text = "France reported 12 confirmed measles cases during 2025. " + (
        "This is an independently published surveillance bulletin." if independent else
        "PDF version of the same surveillance bulletin.")
    digest = hashlib.sha256(text.encode()).hexdigest()
    source = {"source_id": "second", "url": "https://second.invalid/report",
              "original_publisher": "Second publisher" if independent else "First publisher",
              "source_identity_unverified": False}
    if not independent:
        source["upstream_source_mentions"] = [{"source_id": "s"}]
    state["source_registry"].append(source)
    state["documents"].append({**state["documents"][0], "source_id": "second", "document_id": "second-doc",
        "clean_text": text, "content_hash": digest, "text_hash": digest})
    state["evidence_chunks"].append({**state["evidence_chunks"][0], "source_id": "second",
        "chunk_id": "second-span", "document_id": "second-doc", "document_hash": digest,
        "text": text, "char_end": len(text)})
    row = {**state["qualified_records"][0], "record_id": "second-row", "source_id": "second",
           "supporting_chunk_id": "second-span"}
    row.pop("evidence_qualification")
    row["evidence_qualification"] = assess_record_evidence(row, contract=state["structured_task"],
        evidence_index=build_evidence_index(state)).to_dict()
    assert row["evidence_qualification"]["status"] == "qualified"
    state["qualified_records"].append(row)
    after = set(_gain(state))
    assert (after > before) if independent else (after == before)


def test_other_search_scope_cannot_stop_current_recovery(tmp_path):
    state=settled_state()
    history=search_history("official","research")
    for action in history:
        action["search_scope"]="another-disease-or-period"
    state.update(recovery_round=4,recovery_action_history=history)
    ctx=adaptive_runtime(tmp_path)
    with ctx.activate():
        result=recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"]=="search" for action in result["recovery_plan"]["actions"])


def test_frontier_budget_denial_without_document_is_not_completed(tmp_path,monkeypatch):
    from data_collection_workflow.nodes import content_processing as content
    state=source_state()
    state["documents"]=[];state["evidence_chunks"]=[]
    ctx=adaptive_runtime(tmp_path)
    ctx.frontier.enqueue(target_id=state["source_registry"][0]["url"],
        url=state["source_registry"][0]["url"],source_id="s",payload={"entry":state["source_registry"][0]})
    def denied(local):
        job=ctx.frontier.claim_next()
        ctx.frontier.finish(job["target_id"],status="budget_deferred",reason="http_requests exhausted")
        return {"documents":[]}
    monkeypatch.setattr(content,"content_fetch_and_parse",denied)
    monkeypatch.setattr(content,"document_quality_check",lambda state:{})
    monkeypatch.setattr(content,"evidence_chunking_and_data_presence_flagging",lambda state:{"evidence_chunks":[]})
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([RecoveryAction("fetch","s","a","retry")]),context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]["status"]=="budget_exhausted"
    assert delta.actions[0]["error"]
    assert delta.documents==[]


def test_field_repair_uses_original_verified_quote_when_another_observation_has_same_count(tmp_path):
    from data_collection_workflow.workflow_recovery import merge_recovery_delta
    quote="France reported 12 confirmed measles cases during 2025."
    state=source_state(quote+" Canada reported a total of 12 confirmed measles cases during 2025 in its national surveillance bulletin.")
    row={"record_id":"original","source_id":"s","supporting_chunk_id":"c",
         "disease":"measles","cases_confirmed":12,"evidence_quote":quote}
    state.update(raw_records=[row],candidate_records=[{**row,"evidence_qualification":{
        "reasons":["missing_geography","missing_observation_period"]}}],extraction_attempted_chunk_ids=["c"])
    ctx=adaptive_runtime(tmp_path)
    with ctx.activate():
        plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
        actions=[action for action in plan.actions if action.kind=="repair_fields"]
        assert actions
        delta=execute_recovery(RecoveryPlan(actions),context=ctx,artifacts=state,budget=ctx.ledger)
        result=merge_recovery_delta(state,delta)
    assert result["raw_records"][0]["record_id"]=="original"
    assert result["raw_records"][0].get("country")=="France"
    assert result["raw_records"][0].get("reporting_period")=="2025"


@pytest.mark.parametrize("field,value", [
    ("target_fit_status","wrong_disease"),("geography_fit","mismatch"),("period_fit","wrong_period"),
])
def test_explicitly_unrelated_new_source_does_not_reset_search_novelty(field,value):
    from data_collection_workflow.workflow_recovery import _new_independent_sources
    source={"source_id":"new","url":"https://unrelated.invalid/report","source_role_final":"context",field:value}
    assert _new_independent_sources([], [source], [])==[]
