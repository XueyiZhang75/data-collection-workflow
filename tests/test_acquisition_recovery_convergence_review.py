"""Independent real-node checks for stable recovery after repeated consistency."""
from copy import deepcopy
from hashlib import sha256
import socket

import pytest

from data_collection_workflow.nodes.extraction import schema_validation_and_repair
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.nodes.linking_validation import record_linking, cross_source_consistency_check
from data_collection_workflow.run_quality_gates import apply_run_quality_gates
from data_collection_workflow.workflow_recovery import (RecoveryPlan, assess_collection_gaps,
    plan_recovery, execute_recovery, merge_recovery_delta)
from data_collection_workflow.session_runtime import RunContext
from test_validation_refactor import _record


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    for key in ("ENABLE_LLM_EXTRACTION", "ENABLE_LLM_SOURCE_IDENTITY", "ENABLE_LIVE_SEARCH", "ENABLE_LIVE_FETCH", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(key, "false")
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs: pytest.fail("network forbidden"))


def flow_state(disease="measles", country="Canada"):
    rows=[]; docs=[]; chunks=[]; sources=[]
    for index, count in enumerate((100, 150)):
        source_id=f"source-{index}"; chunk_id=f"chunk-{index}"; doc_id=f"doc-{index}"
        # Country is intentionally absent from the original source; field repair
        # cannot certify the existing record's country from a task hint.
        text=f"During 2025, surveillance reported {count} {disease} cases."
        digest=sha256(text.encode()).hexdigest()
        row=_record(f"record-{index}", disease=disease, disease_standard_name=disease,
            virus_or_syndrome=disease, country=country, subnational_location=None,
            reporting_period="2025", date_reported=None, date_anchor=None,
            cases_unspecified=count, deaths=None, hospitalizations=None,
            evidence_quote=text, source_id=source_id, supporting_chunk_id=chunk_id,
            source_url=f"https://source-{index}.example/report", source_type="unknown")
        rows.append(row)
        docs.append({"source_id":source_id,"document_id":doc_id,"content_hash":digest,"text_hash":digest,
            "clean_text":text,"content_readable":True,"acquisition_status":"readable","parse_status":"parsed"})
        chunks.append({"source_id":source_id,"document_id":doc_id,"document_hash":digest,"chunk_id":chunk_id,
            "text":text,"char_start":0,"char_end":len(text),"chunk_kind":"text","contains_target_data":True,
            "disease_relevance_status":"target_disease_match","extraction_eligible_for_task_disease":True,
            "extraction_readiness":"ready","fetch_purpose":"data_extraction"})
        sources.append({"source_id":source_id,"canonical_url":row["source_url"],"source_type":"unknown"})
    return {"raw_records":rows,"documents":docs,"evidence_chunks":chunks,"source_registry":sources,
        "structured_task":{"disease":disease,"location":country,"start_date":"2025-01-01","end_date":"2025-12-31"},
        "collection_spec":{"disease":disease,"geography":country,"time_window":"2025","collection_mode":"direct_collection"},
        "extraction_attempted_chunk_ids":[chunk["chunk_id"] for chunk in chunks],"collection_trace":[],"human_review_queue":[]}


def node_pass(state):
    result=deepcopy(state)
    for node in (schema_validation_and_repair,record_normalization,record_linking,
                 cross_source_consistency_check,apply_run_quality_gates):
        result.update(node(result))
    return result


def runtime(tmp_path):
    return RunContext(tmp_path,{"pipeline_mode":"evidence","universal":{
        "budget_policy":{"version":2,"mode":"adaptive","soft_source_target":50},
        "budget_limits":{"search":0,"search_results":0,"source_targets":0,"http_requests":0,"extraction":0}}})


@pytest.mark.parametrize("disease,country",[("measles","Canada"),("dengue","Brazil")])
def test_real_node_repetition_does_not_reopen_empty_repairs_or_duplicate_conflicts(tmp_path,disease,country):
    state=flow_state(disease,country); ctx=runtime(tmp_path)
    observed=[]; conflict_sets=[]
    with ctx.activate():
        for _ in range(5):
            state=node_pass(state)
            assert state["candidate_records"] and not state["qualified_records"]
            assert state["conflicts"], "The fixture must exercise actual cross-source disagreement."
            conflict_sets.append({row["conflict_id"] for row in state["conflicts"]})
            plan=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
            repairs=[action for action in plan.actions if action.kind=="repair_fields"]
            delta=execute_recovery(RecoveryPlan(repairs),context=ctx,artifacts=state,budget=ctx.ledger)
            observed.append(delta.actions)
            state={**state,**merge_recovery_delta(state,delta),
                   "recovery_action_history":[*state.get("recovery_action_history",[]),*delta.actions]}
    assert observed[0] and all(action["status"]=="completed_empty" for action in observed[0])
    assert all(not actions for actions in observed[1:]), observed
    assert all(ids==conflict_sets[0] for ids in conflict_sets[1:])
    assert not any(ctx.ledger.snapshot()["used"].values())


def test_new_source_span_after_empty_repair_is_executable_again(tmp_path):
    state=flow_state(); ctx=runtime(tmp_path)
    with ctx.activate():
        state=node_pass(state)
        first=plan_recovery(assess_collection_gaps(state),state=state,budget=ctx.ledger)
        repairs=[action for action in first.actions if action.kind=="repair_fields"]
        assert repairs
        delta=execute_recovery(RecoveryPlan(repairs),context=ctx,artifacts=state,budget=ctx.ledger)
        state={**state,**merge_recovery_delta(state,delta),"recovery_action_history":delta.actions}
        # An actual changed source body remains eligible even with the same
        # source/chunk IDs: hashes, bounds and original evidence changed.
        changed=deepcopy(state)
        text="During 2025, Canada reported 100 measles cases."
        digest=sha256(text.encode()).hexdigest()
        changed["documents"][0].update(clean_text=text,content_hash=digest,text_hash=digest)
        changed["evidence_chunks"][0].update(text=text,document_hash=digest,char_end=len(text))
        changed=node_pass(changed)
        again=plan_recovery(assess_collection_gaps(changed),state=changed,budget=ctx.ledger)
    first_ids={action.action_id for action in repairs}
    changed_actions=[action for action in again.actions if action.kind=="repair_fields" and action.target_id=="record-0"]
    assert changed_actions and all(action.action_id not in first_ids for action in changed_actions)


def test_legacy_real_consistency_keeps_append_contract(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE","legacy")
    first=node_pass(flow_state())
    assert first["conflicts"]
    second=cross_source_consistency_check(first)
    assert len(second["conflicts"])==2*len(first["conflicts"])


def test_current_conflict_rebuild_preserves_external_conflict_and_both_observations():
    state=node_pass(flow_state())
    external={**deepcopy(state["conflicts"][0]),"conflict_id":"manual-unresolved",
              "comparison_basis":"human_review_observation","reason":"External unresolved disagreement"}
    state["conflicts"].append(external)
    state["normalized_records"][0]["record_consistency_warnings"].append("manual_context_note")
    before=deepcopy(state)
    result=cross_source_consistency_check(state)
    repeated=cross_source_consistency_check({**state,**result})
    assert state==before
    assert external in repeated["conflicts"]
    assert {row["record_id"]:row["cases_unspecified"] for row in repeated["normalized_records"]}=={
        "record-0":100,"record-1":150}
    assert "manual_context_note" in repeated["normalized_records"][0]["record_consistency_warnings"]
    assert all("manual-unresolved" in row["conflict_ids"] for row in repeated["normalized_records"])
    conflict_ids={row["conflict_id"] for row in repeated["conflicts"]}
    assert all(set(row["conflict_ids"])<=conflict_ids for row in repeated["normalized_records"])
    assert repeated["human_review_queue"]==result["human_review_queue"]


def test_real_new_observation_survives_prior_quality_view_during_consistency():
    first=node_pass(flow_state())
    changed=deepcopy(first)
    raw=next(row for row in changed["raw_records"] if row["record_id"]=="record-1")
    raw["cases_unspecified"]=190
    text="During 2025, surveillance reported 190 measles cases."
    raw["evidence_quote"]=text
    digest=sha256(text.encode()).hexdigest()
    next(doc for doc in changed["documents"] if doc["source_id"]=="source-1").update(
        clean_text=text,content_hash=digest,text_hash=digest)
    next(chunk for chunk in changed["evidence_chunks"] if chunk["source_id"]=="source-1").update(
        text=text,document_hash=digest,char_end=len(text))
    second=node_pass(changed)
    assert next(row for row in second["normalized_records"] if row["record_id"]=="record-1")["cases_unspecified"]==190
    values=[value["value"] for conflict in second["conflicts"] if conflict["field"]=="cases_unspecified"
            for value in conflict["values"]]
    assert 190 in values and 150 not in values


def test_same_comparison_basis_does_not_make_an_external_conflict_owned():
    state=node_pass(flow_state())
    manual={**deepcopy(state["conflicts"][0]),"conflict_id":"external-reviewed-conflict",
            "created_by":"human_review","resolution_status":"unresolved"}
    state["conflicts"].append(manual)
    result=cross_source_consistency_check(state)
    assert manual in result["conflicts"]
