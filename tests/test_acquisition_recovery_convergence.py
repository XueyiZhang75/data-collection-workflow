"""Repeated local recovery must converge without discarding genuinely new input."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from data_collection_workflow.workflow_recovery import RecoveryGap, plan_recovery
from data_collection_workflow.nodes.linking_validation import cross_source_consistency_check
from test_acquisition_recovery import source_state
from test_validation_refactor import _record, _validated_state


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    for key in ("ENABLE_LLM_EXTRACTION", "ENABLE_LLM_SOURCE_IDENTITY", "ENABLE_LIVE_SEARCH", "ENABLE_LIVE_FETCH", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.setenv(key, "false")


def repair_state():
    state = source_state()
    row = {"record_id": "r", "source_id": "s", "supporting_chunk_id": "c",
           "disease": "measles", "cases_confirmed": 12}
    state.update(raw_records=[deepcopy(row)], candidate_records=[deepcopy(row)],
                 extraction_attempted_chunk_ids=["c"])
    return state


def repair_plan(state):
    return plan_recovery([RecoveryGap("candidate_fields_missing", "r", "missing_geography")],
        state=state, budget={"remaining": {"search": 0, "search_results": 0, "extraction": 0}})


@pytest.mark.parametrize("annotation,value", [
    ("conflict_ids", ["conf_006", "conf_012"]),
    ("record_consistency_warnings", ["associated_with_conflict"]),
    ("record_conflict_status", "needs_review"),
    ("linked_event_id", "event_987"),
    ("evidence_qualification", {"status": "candidate", "reasons": ["different_diagnostic"]}),
])
def test_completed_empty_repair_is_not_reopened_by_derived_annotations(annotation, value):
    state = repair_state()
    action = repair_plan(state).actions[0]
    state["recovery_action_history"] = [{**asdict(action), "status": "completed_empty"}]
    for lane in ("raw_records", "candidate_records"):
        state[lane][0][annotation] = value
    assert repair_plan(state).actions == []


def test_failed_attempt_limit_survives_new_conflict_numbers():
    state = repair_state()
    action = repair_plan(state).actions[0]
    state["recovery_action_history"] = [{**asdict(action), "status": "failed"}]
    state["candidate_records"][0]["conflict_ids"] = ["conf_001"]
    retry = repair_plan(state).actions[0]
    assert retry.action_id == action.action_id
    state["recovery_action_history"].append({**asdict(retry), "status": "failed"})
    state["candidate_records"][0]["conflict_ids"].append("conf_002")
    assert repair_plan(state).actions == []


@pytest.mark.parametrize("change", ["count", "geography", "provenance", "document", "span", "task"])
def test_genuinely_changed_repair_input_can_be_planned_again(change):
    state = repair_state()
    action = repair_plan(state).actions[0]
    state["recovery_action_history"] = [{**asdict(action), "status": "completed_empty"}]
    for lane in ("raw_records", "candidate_records"):
        if change == "count": state[lane][0]["cases_confirmed"] = 13
        if change == "geography": state[lane][0]["country"] = "France"
        if change == "provenance": state[lane][0]["field_provenance_json"] = {"cases_confirmed": {"quote": "12 confirmed measles cases", "chunk_id": "c"}}
    if change == "document": state["evidence_chunks"][0]["document_hash"] = "new-source-content-hash"
    if change == "span": state["evidence_chunks"][0]["char_start"] = 1
    if change == "task": state["structured_task"]["target_fields"] = ["country", "reporting_period"]
    again = repair_plan(state).actions
    assert len(again) == 1
    assert again[0].action_id != action.action_id


def conflict_state():
    return _validated_state([
        _record("rec_official", cases_unspecified=100),
        _record("rec_secondary", source_id="src_secondary", source_url="https://secondary.example.org/conflict",
                source_type="news_and_situation_report", source_role_final="collection_support", cases_unspecified=150),
    ])


def test_repeated_consistency_keeps_one_semantic_conflict_and_stable_record_annotations():
    first = conflict_state()
    assert first["conflicts"]
    before = deepcopy(first)
    second = {**first, **cross_source_consistency_check(first)}
    third = {**second, **cross_source_consistency_check(second)}
    assert first == before, "Consistency must not mutate input."
    assert second["conflicts"] == first["conflicts"] == third["conflicts"]
    assert [r["conflict_ids"] for r in first["normalized_records"]] == [r["conflict_ids"] for r in third["normalized_records"]]
    assert third["cross_source_consistency_summary"]["new_conflict_count"] == 0


def test_new_values_update_current_conflicts_instead_of_suppressing_them():
    first = conflict_state()
    changed = deepcopy(first)
    changed["normalized_records"][1]["cases_unspecified"] = 190
    second = {**changed, **cross_source_consistency_check(changed)}
    assert second["conflicts"]
    assert {c["conflict_id"] for c in second["conflicts"]} != {c["conflict_id"] for c in first["conflicts"]}
    assert len(second["conflicts"]) == len(first["conflicts"])
    values = [v["value"] for c in second["conflicts"] if c["field"] == "cases_unspecified" for v in c["values"]]
    assert 190 in values and 150 not in values


def test_resolved_current_values_clear_only_generated_conflict_annotations():
    first = conflict_state()
    resolved = deepcopy(first)
    resolved["normalized_records"][1]["cases_unspecified"] = 100
    resolved["normalized_records"][0]["record_consistency_warnings"].append("manual_context_note")
    result = cross_source_consistency_check(resolved)
    assert not result["conflicts"]
    assert all(not row["conflict_ids"] for row in result["normalized_records"])
    assert "manual_context_note" in result["normalized_records"][0]["record_consistency_warnings"]
    assert all("associated_with_conflict" not in row["record_consistency_warnings"] for row in result["normalized_records"])


def test_legacy_consistency_retains_existing_append_behavior(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "legacy")
    first = conflict_state()
    second = cross_source_consistency_check(first)
    assert len(second["conflicts"]) == 2 * len(first["conflicts"])
