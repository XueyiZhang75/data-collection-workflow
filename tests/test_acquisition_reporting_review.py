"""Independent projection checks for aliases and retained acquisition history."""
import pytest

from data_collection_workflow.result_manifest import build_result_manifest


@pytest.mark.parametrize("reverse", [False, True])
def test_alias_success_resolves_prior_document_free_budget_refusal(reverse):
    source = {"source_id": "current", "url": "https://example.org/report",
              "official_report_alias_source_ids": ["prior"]}
    denied = {"source_id": "prior", "url": source["url"],
              "acquisition_status": "budget_exhausted",
              "acquisition_incomplete": True, "budget_exhausted_kind": "http_requests",
              "request_success": False, "content_readable": False}
    current = {"source_id": "current", "url": source["url"],
               "acquisition_status": "readable", "fetch_status": "success",
               "parse_status": "parsed", "request_success": True,
               "http_status_code": 200, "content_hash": "actual-content",
               "content_readable": True, "clean_text": "Twelve confirmed cases."}
    docs = [denied, current]
    if reverse:
        docs.reverse()
    state = {"source_registry": [source], "documents": docs,
             "evidence_chunks": [{"source_id": "current", "chunk_id": "c",
                                  "extraction_status": "pending"}]}
    result = build_result_manifest({"source_registry": [source]}, state)
    progress = result["source_progress"]
    assert progress["discovered_unique_sources"] == 1
    assert progress["readable_sources"] == 1
    assert progress["fetch_failed_sources"] == 0
    assert progress["sources"][0]["source_ids"] == ["current", "prior"]
    assert progress["sources"][0]["processing_status"] == "awaiting_extraction"
    assert progress["budget_deferred_sources"] == 0
    assert result["acquisition"]["budget_deferred_document_count"] == 0
    assert result["acquisition"]["budget_deferred_source_count"] == 0
    assert result["acquisition"]["budget_exhausted_causes"] == {}


def test_frontier_alias_without_primary_source_id_maps_by_canonical_url():
    source = {"source_id": "preferred", "url": "https://example.org/report",
              "source_id_aliases": ["old"]}
    state = {"source_registry": [source],
             "acquisition_frontier": {"items": [{
                 "source_id": "queue-id", "url": "https://example.org/report#page",
                 "target_id": "https://example.org/report", "status": "budget_deferred",
                 "reason": "source_targets", "attempts": 0}]}}
    manifest = build_result_manifest({"source_registry": [source]}, state)
    progress = manifest["source_progress"]
    assert progress["discovered_unique_sources"] == 1
    assert progress["budget_deferred_sources"] == 1
    assert progress["fetch_failed_sources"] == 0
    assert progress["sources"][0]["processing_reason"] == "source_targets"
    assert manifest["acquisition"]["budget_deferred_source_count"] == 1
    assert manifest["acquisition"]["budget_exhausted_causes"] == {"source_targets": 1}


@pytest.mark.parametrize(("prior_hash", "budget_kind", "expected_status", "deferred_count"), [
    ("actual-content", "ocr", "awaiting_extraction", 0),
    ("actual-content", None, "awaiting_extraction", 0),
    ("older-content", "ocr", "budget_deferred", 1),
])
def test_alias_partial_supersession_preserves_distinct_document_versions(
        prior_hash, budget_kind, expected_status, deferred_count):
    source = {"source_id": "current", "url": "https://example.org/report",
              "source_id_aliases": ["prior"]}
    prior = {"source_id": "prior", "url": source["url"],
             "content_hash": prior_hash, "acquisition_incomplete": True,
             "acquisition_status": "budget_exhausted" if budget_kind else "readable",
             "content_readable": True, "clean_text": "Partial document content.",
             "unprocessed_pages": [2]}
    if budget_kind:
        prior["budget_exhausted_kind"] = budget_kind
    else:
        prior["quality_issues"] = ["browser_resources_incomplete"]
    current = {"source_id": "current", "url": source["url"],
               "content_hash": "actual-content", "acquisition_status": "readable",
               "request_success": True, "content_readable": True,
               "clean_text": "Complete document content."}
    state = {"source_registry": [source], "documents": [prior, current],
             "evidence_chunks": [{"source_id": "current", "chunk_id": "c",
                                  "extraction_status": "pending"}]}
    manifest = build_result_manifest({"source_registry": [source]}, state)
    assert manifest["source_progress"]["sources"][0]["processing_status"] == expected_status
    acquisition = manifest["acquisition"]
    assert acquisition["budget_deferred_document_count"] == deferred_count
    assert acquisition["budget_deferred_source_count"] == deferred_count
    if prior_hash == "actual-content":
        assert acquisition["unresolved_documents"] == []
        assert acquisition["incomplete_document_count"] == 0
    else:
        assert acquisition["unprocessed_page_count"] == 1


def test_finalization_preserves_excluded_source_in_full_registry_and_audit_view(monkeypatch):
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_HUMAN_REVIEW", "false")
    state = {
        "structured_task": {"disease": "measles", "location": "Canada",
                            "start_date": "2025-01-01", "end_date": "2025-12-31"},
        "source_registry": [
            {"source_id": "pending", "url": "https://example.org/pending",
             "canonical_url": "https://example.org/pending",
             "status": "ready_for_content_fetch"},
            {"source_id": "excluded", "url": "https://example.org/excluded",
             "canonical_url": "https://example.org/excluded", "status": "excluded",
             "source_role_final": "excluded", "final_screening_decision": "exclude",
             "target_fit_status": "unrelated_disease"}],
        "collection_trace": [], "normalized_records": []}
    result = final_data_package_builder(state)
    package = result["final_data_package"]
    for rows in (result["source_registry"], package["source_registry"]):
        indexed = {row["source_id"]: row for row in rows}
        assert set(indexed) == {"pending", "excluded"}
        assert indexed["excluded"]["processing_status"] == "screening_excluded"
    excluded = {row["source_id"]: row for row in package["excluded_sources"]}
    assert excluded["excluded"]["processing_status"] == "screening_excluded"
    manifest = package["result_manifest"]
    assert manifest["counts"]["discovered_sources"] == 2
    assert manifest["source_progress"]["discovered_unique_sources"] == 2
    assert manifest["source_progress"]["screening_excluded_sources"] == 1
    assert package["final_dataset"] == []
