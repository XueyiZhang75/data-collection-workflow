"""Independent recovery contracts across the actual queue and evidence boundary."""
import hashlib
import importlib
import socket

import pytest

from test_acquisition_queue import setup_node
from test_acquisition_recovery import adaptive_runtime, settled_state, search_history, source_state
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
from data_collection_workflow.workflow_recovery import (
    RecoveryAction, RecoveryPlan, assess_collection_gaps, execute_recovery,
    plan_recovery, recovery_control, merge_recovery_delta, _new_independent_sources,
)


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "false")
    def forbidden(*args, **kwargs):
        raise AssertionError("Review tests forbid external network calls")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_frontier_payload_only_source_is_really_fetched_and_restored(tmp_path, monkeypatch):
    runtime, state, visits = setup_node(monkeypatch, tmp_path)
    entry = state["source_registry"][0]
    state["source_registry"] = []
    runtime.frontier.enqueue(target_id=entry["url"], url=entry["url"], source_id=entry["source_id"],
                             priority=50, payload=dict(entry))
    with runtime.activate():
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=runtime.ledger)
        fetches = [action for action in plan.actions if action.kind == "fetch"]
        assert len(fetches) == 1
        delta = execute_recovery(RecoveryPlan(fetches), context=runtime, artifacts=state, budget=runtime.ledger)
        merged = merge_recovery_delta(state, delta)
    assert visits == [entry["url"]]
    assert any(source["source_id"] == entry["source_id"] for source in merged["source_registry"])
    assert any(doc["source_id"] == entry["source_id"] and doc["content_readable"] for doc in merged["documents"])
    assert runtime.frontier.snapshot()["counts"] == {"completed": 1}


def test_two_fetch_actions_do_not_refetch_sources_drained_by_first_batch(tmp_path, monkeypatch):
    runtime, state, visits = setup_node(monkeypatch, tmp_path)
    with runtime.activate():
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=runtime.ledger)
        fetches = [action for action in plan.actions if action.kind == "fetch"]
        assert len(fetches) == 2
        delta = execute_recovery(RecoveryPlan(fetches), context=runtime, artifacts=state, budget=runtime.ledger)
        merged = merge_recovery_delta(state, delta)
    assert set(visits) == {source["url"] for source in state["source_registry"]}
    assert len(visits) == 2
    assert len(merged["documents"]) == 2
    assert runtime.ledger.snapshot()["used"]["http_requests"] == 2
    assert delta.actions[1]["status"] == "skipped"
    assert delta.actions[1]["error"] == "already_handled_by_frontier_batch"


@pytest.mark.parametrize("independent", [False, True])
def test_search_novelty_distinguishes_new_publisher_from_same_report_format(independent):
    before = [{"source_id": "original", "url": "https://first.example/report",
               "original_publisher": "First surveillance agency", "source_identity_unverified": False}]
    after = [{"source_id": "next", "url": "https://second.example/report.pdf",
              "original_publisher": "Second surveillance agency" if independent else "First surveillance agency",
              "source_identity_unverified": False}]
    if not independent:
        after[0]["upstream_source_mentions"] = [{"source_id": "original"}]
    assert _new_independent_sources(before, after, []) == (["next"] if independent else [])


def test_field_repair_cannot_borrow_period_from_different_same_count_observation(tmp_path):
    original_quote = "France reported 12 confirmed measles cases."
    text = original_quote + " Germany reported a total of 12 confirmed measles cases during 2025."
    state = source_state(text)
    row = {"record_id": "original", "source_id": "s", "supporting_chunk_id": "c",
           "disease": "measles", "country": "France", "cases_confirmed": 12,
           "evidence_quote": original_quote,
           "field_provenance_json": {name: {"quote": original_quote} for name in ("disease", "country", "cases_confirmed")}}
    row["evidence_qualification"] = assess_record_evidence(row, contract=state["structured_task"],
                                                         evidence_index=build_evidence_index(state)).to_dict()
    state.update(raw_records=[row], candidate_records=[row], extraction_attempted_chunk_ids=["c"])
    runtime = adaptive_runtime(tmp_path)
    with runtime.activate():
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=runtime.ledger)
        repairs = [action for action in plan.actions if action.kind == "repair_fields"]
        assert repairs
        delta = execute_recovery(RecoveryPlan(repairs), context=runtime, artifacts=state, budget=runtime.ledger)
        merged = merge_recovery_delta(state, delta)
    repaired = next(item for item in merged["raw_records"] if item["record_id"] == "original")
    assert repaired["country"] == "France"
    assert not repaired.get("reporting_period")
    assert not delta.record_updates


def test_new_supported_fact_resets_empty_search_streak_without_erasing_history(tmp_path):
    state = settled_state(covered=True)
    state.update(recovery_round=3, recovery_action_history=search_history("official", "research"))
    text = "France reported 2 deaths from measles during 2025."
    digest = hashlib.sha256(text.encode()).hexdigest()
    state["documents"].append({**state["documents"][0], "document_id": "deaths", "clean_text": text,
                               "content_hash": digest, "text_hash": digest})
    state["evidence_chunks"].append({**state["evidence_chunks"][0], "chunk_id": "deaths",
                                     "document_id": "deaths", "document_hash": digest, "text": text,
                                     "char_end": len(text)})
    state["extraction_attempted_chunk_ids"].append("deaths")
    row = {"record_id": "death-row", "source_id": "s", "supporting_chunk_id": "deaths",
           "disease": "measles", "country": "France", "reporting_period": "2025", "deaths": 2}
    row["evidence_qualification"] = assess_record_evidence(row, contract=state["structured_task"],
                                                         evidence_index=build_evidence_index(state)).to_dict()
    assert row["evidence_qualification"]["status"] == "qualified"
    state["qualified_records"].append(row)
    history = list(state["recovery_action_history"])
    runtime = adaptive_runtime(tmp_path)
    with runtime.activate():
        result = recovery_control(state)
    assert result["recovery_stop_reason"] is None
    assert any(action["kind"] == "search" for action in result["recovery_plan"]["actions"])
    assert state["recovery_action_history"] == history


def test_successful_fetch_is_not_failed_by_its_preserved_prior_error_document(tmp_path, monkeypatch):
    from data_collection_workflow.document_acquisition import parse_response
    runtime, state, visits = setup_node(monkeypatch, tmp_path)
    state["source_registry"] = state["source_registry"][:1]
    entry = state["source_registry"][0]
    old = parse_response(b"server error", url=entry["url"], source_id=entry["source_id"],
                         session_dir=tmp_path, content_type="text/plain", status_code=500)
    old["document_id"] = "old-error-response"
    state["documents"] = [old]
    action = RecoveryAction("fetch", entry["source_id"], "retry-the-source", "get complete source content")
    with runtime.activate():
        delta = execute_recovery(RecoveryPlan([action]), context=runtime, artifacts=state, budget=runtime.ledger)
    assert visits == [entry["url"]]
    assert runtime.frontier.snapshot()["counts"] == {"completed": 1}
    assert any(doc.get("content_hash") == old["content_hash"] for doc in delta.documents)
    assert any(doc.get("content_readable") for doc in delta.documents)
    assert delta.actions[0]["status"] == "completed", delta.actions[0]
