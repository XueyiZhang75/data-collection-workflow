"""Report observed empty output separately from skipped and failed extraction."""
from copy import deepcopy

import pytest

from data_collection_workflow.reporting.source_catalog import build_source_catalog
from data_collection_workflow.source_progress import build_source_progress


@pytest.fixture(autouse=True)
def evidence_mode(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def fixture():
    sources = [dict(source_id=key, url=f"https://example.org/{key}", source_type="unknown")
               for key in ("empty", "failed", "skipped", "pending", "missing", "attempted")]
    docs = [dict(source_id=row["source_id"], clean_text="Readable report body", content_readable=True,
                 parse_status="parsed", fetch_status="success", content_hash=row["source_id"])
            for row in sources]
    chunks = [dict(source_id=key, chunk_id=key, text="The patient developed a rash and was hospitalized.",
                   contains_target_data=True, disease_relevance_status="target_disease_match")
              for key in ("empty", "failed", "skipped", "pending", "attempted")]
    chunks[2]["contains_target_data"] = False
    return {"source_registry": sources, "documents": docs, "evidence_chunks": chunks,
            "extraction_attempted_chunk_ids": ["empty", "failed", "attempted"],
            "structured_extraction_summary": {
                "failed_chunk_ids": ["failed"],
                "llm_empty_output_diagnostics": [{"source_id": "empty", "chunk_id": "empty",
                                                   "reason": "case_span_not_detected"}]}}


def test_actual_empty_failure_skip_and_missing_outcomes_have_separate_counters():
    state = fixture()
    before = deepcopy(state)
    progress = build_source_progress({}, state)
    rows = {row["source_ids"][0]: row for row in progress["sources"]}
    assert rows["failed"]["processing_status"] == "extraction_failed"
    assert rows["failed"]["extraction_outcome"] == "failed"
    assert rows["empty"]["extraction_outcome"] == "empty"
    assert rows["empty"]["processing_reason"] == "case_span_not_detected"
    assert rows["skipped"]["extraction_outcome"] == "skipped"
    assert rows["pending"]["extraction_outcome"] == "pending"
    assert rows["missing"]["extraction_outcome"] == "not_recorded"
    assert rows["attempted"]["extraction_outcome"] == "attempted_unknown"
    assert progress["extraction_empty_sources"] == 1
    assert progress["extraction_failed_sources"] == 1
    assert progress["extraction_skipped_sources"] == 1
    assert progress["missing_extraction_evidence_sources"] == 2
    assert state == before


def test_catalog_labels_do_not_call_operational_failure_successful_empty_output():
    rows = {row["source_ids"][0]: row for row in build_source_catalog({}, fixture())["sources"]}
    assert rows["failed"]["processing"]["label"] == "Extraction failed"
    assert "failed" in rows["failed"]["processing"]["reason"].lower()
    assert "no case-bearing span" in rows["empty"]["processing"]["reason"].lower()
    assert "failed" not in rows["empty"]["processing"]["reason"].lower()
    assert "eligible" in rows["skipped"]["processing"]["reason"].lower()


def test_stale_empty_diagnostic_does_not_override_latest_failed_attempt():
    state = fixture()
    state["structured_extraction_summary"]["llm_empty_output_diagnostics"].append(
        {"source_id": "failed", "chunk_id": "failed", "reason": "case_span_not_detected"})
    rows = {row["source_ids"][0]: row for row in build_source_progress({}, state)["sources"]}
    assert rows["failed"]["processing_status"] == "extraction_failed"


def test_explicit_failed_status_is_not_counted_as_valid_empty_extraction():
    state = fixture()
    state["structured_extraction_summary"] = {}
    state["evidence_chunks"][1].update(extraction_status="failed", extraction_reason="transport_error")
    rows = {row["source_ids"][0]: row for row in build_source_progress({}, state)["sources"]}
    assert rows["failed"]["processing_status"] == "extraction_failed"
    assert rows["failed"]["processing_reason"] == "transport_error"


def test_academic_or_peer_reviewed_category_is_not_displayed_as_verified_literature():
    state = fixture()
    state["source_registry"][0]["source_type"] = "academic_or_peer_reviewed_source"
    row = next(row for row in build_source_catalog({}, state)["sources"] if "empty" in row["source_ids"])
    assert "unverified" in row["source_type_label"].lower()


def test_saved_observation_does_not_become_empty_because_of_earlier_retry_diagnostic():
    state = fixture()
    state["raw_records"] = [{"record_id": "r", "source_id": "empty", "supporting_chunk_id": "empty",
                             "cases_unspecified": 12}]
    rows = {row["source_ids"][0]: row for row in build_source_progress({}, state)["sources"]}
    assert rows["empty"]["extraction_outcome"] == "completed"
    assert rows["empty"]["extraction_outcome_counts"].get("empty", 0) == 0
    assert rows["empty"]["processing_reason"] == "extracted_observations_not_in_output"


def test_mixed_empty_reasons_retain_meaning_in_catalogue():
    state = fixture()
    state["evidence_chunks"].append({**state["evidence_chunks"][0], "chunk_id": "empty_2"})
    state["structured_extraction_summary"]["llm_empty_output_diagnostics"].append(
        {"source_id": "empty", "chunk_id": "empty_2", "reason": "legitimate_no_record_context"})
    row = next(row for row in build_source_catalog({}, state)["sources"] if "empty" in row["source_ids"])
    assert "case-bearing span" in row["processing"]["reason"]
    assert "context-only" in row["processing"]["reason"]


def test_qualified_detail_explains_optional_field_removal_and_partial_task_overlap():
    state = fixture()
    action = {"field": "virus_or_syndrome", "original_value": "HPS",
              "reason": "unsupported_optional_field_set_aside", "quote": "Measles cases"}
    record = {"record_id": "r_supported", "recovered_from_record_id": "r_original", "source_id": "empty",
              "evidence_normalization_actions": [action],
              "evidence_qualification": {"status": "qualified", "product_kind": "aggregate",
                                         "task_temporal_relation": "overlaps_task_boundary"}}
    row = next(row for row in build_source_catalog({"final_dataset": [record]}, state)["sources"]
               if "empty" in row["source_ids"])
    detail = row["evidence_details"][0]
    assert detail["recovered_from_record_id"] == "r_original"
    assert detail["evidence_normalization_actions"] == [action]
    notes = " ".join(detail["reason_labels"])
    assert "HPS" in notes and "unsupported" in notes
    assert "overlaps" in notes and "not split" in notes


def test_unresolved_period_candidate_has_specific_reader_explanation():
    state = fixture()
    record = {"record_id": "r", "source_id": "empty", "evidence_qualification": {
        "status": "candidate", "reasons": ["unresolved_observation_period"]}}
    row = next(row for row in build_source_catalog({"candidate_records": [record]}, state)["sources"]
               if "empty" in row["source_ids"])
    assert "could not be resolved from the source" in row["candidate_reason_labels"][0]


def test_invalid_explicit_span_candidate_explains_failed_source_location():
    state = fixture()
    record = {"record_id": "r", "source_id": "empty", "evidence_qualification": {
        "status": "candidate", "reasons": ["cases_unspecified:invalid_explicit_span"]}}
    row = next(row for row in build_source_catalog({"candidate_records": [record]}, state)["sources"]
               if "empty" in row["source_ids"])
    assert "offsets" in row["candidate_reason_labels"][0]
    assert "source text" in row["candidate_reason_labels"][0]


@pytest.mark.parametrize('current_outcome', ['empty', 'completed'])
def test_current_success_evidence_replaces_stale_chunk_failure_label(current_outcome):
    state = fixture()
    state['evidence_chunks'][1].update(extraction_status='failed', extraction_reason='transport_error')
    state['structured_extraction_summary'] = {'failed_chunk_ids': []}
    if current_outcome == 'empty':
        state['structured_extraction_summary']['llm_empty_output_diagnostics'] = [
            {'chunk_id': 'failed', 'source_id': 'failed', 'reason': 'focused_retry_empty'}]
    else:
        state['raw_records'] = [{'record_id': 'r', 'source_id': 'failed', 'supporting_chunk_id': 'failed',
                                 'cases_unspecified': 12}]
    rows = {row['source_ids'][0]: row for row in build_source_progress({}, state)['sources']}
    assert rows['failed']['extraction_outcome'] == current_outcome
    assert rows['failed']['processing_status'] != 'extraction_failed'


@pytest.mark.parametrize('summary', [{}, {'failed_chunk_ids': []}, {'llm_empty_output_diagnostics': [
    {'chunk_id': 'failed', 'reason': 'focused_retry_empty'}]}])
def test_missing_failure_membership_without_current_success_proof_does_not_invent_success(summary):
    state = fixture()
    state['evidence_chunks'][1].update(extraction_status='failed', extraction_reason='transport_error')
    state['structured_extraction_summary'] = summary
    rows = {row['source_ids'][0]: row for row in build_source_progress({}, state)['sources']}
    assert rows['failed']['extraction_outcome'] == 'failed'
