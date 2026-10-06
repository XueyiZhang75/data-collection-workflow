"""Smoke tests for the compiled LangGraph workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.graph import build_graph  # noqa: E402
from data_collection_workflow.nodes.source_discovery import canonicalize_url  # noqa: E402

_RESERVED_SOURCE_IDS = {
    "src_cdc_reported_cases",
    "src_ecdc_surveillance_updates",
    "src_ecdc_annual_report_2023",
    "src_who_hantavirus_fact_sheet",
}


@pytest.fixture(autouse=True)
def _default_standard_collection_mode(monkeypatch):
    monkeypatch.delenv("COLLECTION_MODE", raising=False)


def _sanity_initial_state() -> dict:
    return {
        "user_request": (
            "Collect global human hantavirus case, outbreak, and surveillance data "
            "from 2020 to 2026, including cases, deaths, dates, locations, source URLs, "
            "source types, and evidence quotes."
        ),
        "source_candidates": [],
        "source_discovery_summary": None,
        "source_registry": [],
        "source_registry_summary": None,
        "documents": [],
        "evidence_chunks": [],
        "raw_records": [],
        "validated_records": [],
        "normalized_records": [],
        "linked_events": [],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
        "collection_spec": None,
        "disease_profile": None,
        "collection_schema": None,
        "source_strategy": None,
        "screening_criteria": None,
        "search_queries": None,
        "search_query_inventory": [],
        "final_data_package": None,
        "current_route": None,
    }


def test_build_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_invocation_returns_final_package():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert result.get("final_data_package") is not None


def test_collection_trace_has_at_least_fifteen_events():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    trace = result.get("collection_trace") or []
    assert len(trace) >= 15, f"expected >=15 trace events, got {len(trace)}"


def test_final_data_package_has_expected_keys():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    expected_keys = {
        "final_dataset",
        "source_registry",
        "linked_events",
        "conflicts",
        "human_review_items",
        "excluded_sources",
        "collection_trace",
    }
    missing = expected_keys - set(package.keys())
    assert not missing, f"final_data_package missing keys: {missing}"


def test_task_scope_infers_global_and_time_window():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    spec = result.get("collection_spec") or {}
    assert spec.get("geography") == "global"
    assert spec.get("time_window") == "2020-2026"


def test_query_strategy_inventory_exists():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    inventory = result.get("search_query_inventory") or []
    assert len(inventory) > 0
    required_keys = {
        "query_id",
        "query",
        "source_type",
        "priority",
        "rationale",
        "expected_fields",
    }
    for item in inventory:
        missing = required_keys - set(item.keys())
        assert not missing, f"inventory item missing keys: {missing}"
    query_strings = [item["query"] for item in inventory]
    assert len(query_strings) == len(set(query_strings)), "duplicate query strings detected"


def test_collection_schema_in_state():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert result.get("collection_schema") is not None
    assert result.get("screening_criteria") is not None
    assert result.get("source_strategy") is not None


def test_source_discovery_returns_candidates():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    candidates = result.get("source_candidates") or []
    summary = result.get("source_discovery_summary") or {}
    assert len(candidates) >= 10
    assert summary, "source_discovery_summary should be populated"
    assert summary.get("discovery_method") == "offline_seed_catalog"
    assert summary.get("candidate_count") == len(candidates)


def test_source_registry_deduplicates_and_registers_sources():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    summary = result.get("source_registry_summary") or {}
    assert len(registry) > 0
    # Status is set to "registered" by source_dedup_and_registry but then
    # advanced by source_screening / source_critic_and_uncertainty_routing.
    # We assert it is one of the known lifecycle values.
    valid_statuses = {
        "registered",
        "screened",
        "ready_for_content_fetch",
        "ready_for_context_fetch",
        "deferred_search_expansion",
        "needs_human_review",
        "excluded",
    }
    for entry in registry:
        assert entry.get("source_id")
        assert entry.get("canonical_url")
        assert entry.get("source_type")
        assert entry.get("status") in valid_statuses
    canonical_urls = [e["canonical_url"] for e in registry]
    assert len(canonical_urls) == len(set(canonical_urls)), "canonical_url values must be unique"
    assert summary.get("registry_entry_count") == len(registry)


def test_source_candidates_have_query_metadata_when_possible():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    candidates = result.get("source_candidates") or []
    assert candidates
    with_query = [c for c in candidates if c.get("query_id")]
    assert len(with_query) * 2 >= len(candidates), (
        f"expected at least half of candidates to have a query_id, "
        f"got {len(with_query)} of {len(candidates)}"
    )
    for cand in candidates:
        assert cand.get("discovery_method") == "offline_seed_catalog"


def test_source_type_coverage():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    types = {e.get("source_type") for e in registry}
    required = {
        "official_public_health_agency",
        "international_organization_report",
        "peer_reviewed_literature",
        "structured_database",
        "news_and_situation_report",
    }
    missing = required - types
    assert not missing, f"source_registry missing source_type coverage: {missing}"


def test_canonicalize_url():
    canonical = canonicalize_url(
        " https://www.cdc.gov/hantavirus/about/index.html#section "
    )
    assert canonical == "https://www.cdc.gov/hantavirus/about/index.html"

    seed_uri = "seed://news/hantavirus-outbreak-report-cases-deaths"
    assert canonicalize_url(seed_uri) == seed_uri


def _registry_by_source_id(registry: list[dict]) -> dict[str, dict]:
    return {e.get("source_id"): e for e in registry}


def test_source_screening_populates_decisions():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    assert registry
    for entry in registry:
        assert entry.get("screening_decision")
        assert entry.get("screening_confidence") is not None
        assert entry.get("screening_reason")
        assert entry.get("source_role")
    summary = result.get("source_screening_summary") or {}
    assert summary, "source_screening_summary should be populated"
    assert summary.get("screened_count") == len(registry)


def test_source_critic_populates_final_routing():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    assert registry
    for entry in registry:
        assert entry.get("critic_decision")
        assert entry.get("critic_confidence") is not None
        assert entry.get("critic_reason")
        assert entry.get("final_screening_decision")
        assert entry.get("ready_for_content_fetch") is not None
        assert entry.get("status")
    assert result.get("source_critic_summary")
    assert result.get("source_routing_summary")


def test_case_data_sources_ready_for_content_fetch():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = _registry_by_source_id(result.get("source_registry") or [])
    for source_id in (
        "src_cdc_reported_cases",
        "src_ecdc_surveillance_updates",
        "src_ecdc_annual_report_2023",
    ):
        entry = registry.get(source_id)
        assert entry, f"missing {source_id} in source_registry"
        assert entry.get("final_screening_decision") == "include_for_content_fetch", (
            f"{source_id} final_screening_decision was {entry.get('final_screening_decision')}"
        )
        assert entry.get("ready_for_content_fetch") is True
        assert entry.get("status") == "ready_for_content_fetch"


def test_context_sources_ready_for_context_fetch():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = _registry_by_source_id(result.get("source_registry") or [])
    for source_id in ("src_cdc_about_hantavirus", "src_who_hantavirus_fact_sheet"):
        entry = registry.get(source_id)
        assert entry, f"missing {source_id} in source_registry"
        assert entry.get("final_screening_decision") == "include_for_context_fetch", (
            f"{source_id} final_screening_decision was {entry.get('final_screening_decision')}"
        )
        assert entry.get("ready_for_content_fetch") is True
        assert entry.get("status") == "ready_for_context_fetch"


def test_search_endpoints_deferred():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    search_publishers = {"PubMed", "Europe PMC", "OpenAlex"}
    matched = [e for e in registry if e.get("publisher") in search_publishers]
    assert matched, "expected at least one search-endpoint registry entry"
    for entry in matched:
        assert entry.get("final_screening_decision") == "defer_to_search_expansion"
        assert entry.get("ready_for_content_fetch") is False
        assert entry.get("status") == "deferred_search_expansion"


def test_placeholder_sources_deferred():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = result.get("source_registry") or []
    placeholders = [
        e for e in registry if (e.get("canonical_url") or "").startswith("seed://")
    ]
    assert placeholders, "expected at least one placeholder registry entry"
    for entry in placeholders:
        assert entry.get("final_screening_decision") == "defer_to_search_expansion"
        assert entry.get("ready_for_content_fetch") is False
        assert entry.get("status") == "deferred_search_expansion"


def test_source_type_routing_counts():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    routing_summary = result.get("source_routing_summary") or {}
    assert routing_summary
    assert routing_summary.get("ready_for_content_fetch_count", 0) > 0
    assert routing_summary.get("deferred_search_expansion_count", 0) > 0
    final_counts = routing_summary.get("final_decision_counts") or {}
    for key in (
        "include_for_content_fetch",
        "include_for_context_fetch",
        "defer_to_search_expansion",
    ):
        assert key in final_counts, f"final_decision_counts missing {key}"


def test_default_collection_mode_does_not_mask_reserved_sources(monkeypatch):
    monkeypatch.delenv("COLLECTION_MODE", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = _registry_by_source_id(result.get("source_registry") or [])

    for source_id in _RESERVED_SOURCE_IDS:
        entry = registry.get(source_id)
        assert entry, f"missing {source_id} in source_registry"
        assert entry.get("source_role") != "validation_reserved"
        assert entry.get("final_screening_decision") != "reserved_for_validation"
        assert "validation_reserved" not in (entry.get("routing_flags") or [])

    routing_summary = result.get("source_routing_summary") or {}
    fetch_summary = result.get("content_fetch_summary") or {}
    assert routing_summary.get("collection_mode") == "standard"
    assert routing_summary.get("validation_reserved_source_count") == 0
    assert fetch_summary.get("collection_mode") == "standard"
    assert fetch_summary.get("fetch_request_count") == 10
    assert fetch_summary.get("skipped_validation_reserved_count") == 0


def test_masked_collection_mode_marks_reserved_sources(monkeypatch):
    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    registry = _registry_by_source_id(result.get("source_registry") or [])

    for source_id in _RESERVED_SOURCE_IDS:
        entry = registry.get(source_id)
        assert entry, f"missing {source_id} in source_registry"
        assert entry.get("source_role") == "validation_reserved"
        assert entry.get("ready_for_content_fetch") is False
        assert entry.get("final_screening_decision") == "reserved_for_validation"
        assert entry.get("status") == "reserved_for_validation"
        flags = set(entry.get("routing_flags") or [])
        assert {"validation_reserved", "blocked_from_collection"} <= flags

    routing_summary = result.get("source_routing_summary") or {}
    assert routing_summary.get("collection_mode") == "masked_validation"
    assert routing_summary.get("validation_reserved_source_count") == len(
        _RESERVED_SOURCE_IDS
    )
    assert set(routing_summary.get("validation_reserved_source_ids") or []) == (
        _RESERVED_SOURCE_IDS
    )


def test_human_review_queue_no_duplicates():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    queue = result.get("human_review_queue") or []
    ids = [item.get("review_id") for item in queue if item.get("review_id")]
    assert len(ids) == len(set(ids)), "human_review_queue contains duplicate review_id"


# ---------------------------------------------------------------------------
# Step 5: content fetch / document quality
# ---------------------------------------------------------------------------


def test_content_fetch_requests_created_offline(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    requests = result.get("content_fetch_requests") or []
    assert len(requests) == 10, f"expected 10 fetch requests, got {len(requests)}"
    for r in requests:
        assert r.get("source_id")
        assert r.get("url")
        assert r.get("final_screening_decision")
        assert r.get("fetch_purpose")
    purposes = {r.get("fetch_purpose") for r in requests}
    assert "data_extraction" in purposes
    assert "context_grounding" in purposes


def test_documents_created_as_offline_stubs(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    requests = result.get("content_fetch_requests") or []
    documents = result.get("documents") or []
    assert len(documents) == len(requests)
    for doc in documents:
        assert doc.get("is_offline_stub") is True
        assert doc.get("fetch_status") == "offline_stub"
        assert doc.get("parse_status") == "offline_stub"
        assert doc.get("content_hash")


def test_deferred_sources_not_fetched(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    documents = result.get("documents") or []
    fetched_ids = {d.get("source_id") for d in documents}

    registry = result.get("source_registry") or []
    by_id = {e.get("source_id"): e for e in registry}

    search_publishers = {"PubMed", "Europe PMC", "OpenAlex"}
    for sid, entry in by_id.items():
        if entry.get("publisher") in search_publishers:
            assert sid not in fetched_ids, f"search endpoint {sid} should not be fetched"
        if (entry.get("canonical_url") or "").startswith("seed://"):
            assert sid not in fetched_ids, f"placeholder {sid} should not be fetched"


def test_document_quality_marks_offline_stubs(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    documents = result.get("documents") or []
    summary = result.get("document_quality_summary") or {}
    assert summary, "document_quality_summary should be populated"
    assert summary.get("offline_stub_count") == len(documents)
    for doc in documents:
        assert doc.get("quality_status") == "offline_stub_pending_live_fetch"
        assert "not_real_source_content" in (doc.get("quality_issues") or [])


def test_content_fetch_summary_counts(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("content_fetch_summary") or {}
    assert summary
    assert summary.get("live_fetch_enabled") is False
    assert summary.get("fetch_request_count") == 10
    assert summary.get("document_count") == 10
    # After the Step 6 diagnostic cleanup the 5 non-fetched sources are split
    # across the granular skip buckets rather than all landing in "deferred".
    total_skipped = (
        summary.get("skipped_deferred_count", 0)
        + summary.get("skipped_search_endpoint_count", 0)
        + summary.get("skipped_blocked_scheme_count", 0)
        + summary.get("skipped_human_review_count", 0)
    )
    assert total_skipped >= 5


def test_masked_content_fetch_blocks_reserved_even_if_ready(monkeypatch):
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse

    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    state = {
        "source_registry": [
            {
                "source_id": "src_cdc_reported_cases",
                "canonical_url": "https://www.cdc.gov/hantavirus/surveillance",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_role": "validation_reserved",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
                "requires_human_review": False,
                "routing_flags": ["validation_reserved"],
            }
        ],
        "collection_trace": [],
    }

    result = content_fetch_and_parse(state)

    assert result.get("content_fetch_requests") == []
    assert result.get("documents") == []
    summary = result.get("content_fetch_summary") or {}
    assert summary.get("collection_mode") == "masked_validation"
    assert summary.get("skipped_validation_reserved_count") == 1
    assert summary.get("skipped_validation_reserved_source_ids") == [
        "src_cdc_reported_cases"
    ]


def test_masked_content_fetch_blocks_reserved_source_id_even_without_reserved_flags(
    monkeypatch,
):
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse

    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    state = {
        "source_registry": [
            {
                "source_id": "src_cdc_reported_cases",
                "canonical_url": "https://www.cdc.gov/hantavirus/surveillance",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_role": "data_source",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
                "requires_human_review": False,
                "routing_flags": [],
            }
        ],
        "collection_trace": [],
    }

    result = content_fetch_and_parse(state)

    assert result.get("content_fetch_requests") == []
    assert result.get("documents") == []
    summary = result.get("content_fetch_summary") or {}
    assert summary.get("collection_mode") == "masked_validation"
    assert summary.get("skipped_validation_reserved_count") == 1
    assert summary.get("skipped_validation_reserved_source_ids") == [
        "src_cdc_reported_cases"
    ]


def test_masked_content_fetch_does_not_block_reserved_domain_when_domain_masking_disabled(
    monkeypatch,
):
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse

    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    state = {
        "source_registry": [
            {
                "source_id": "src_non_reserved_cdc_page",
                "canonical_url": "https://www.cdc.gov/hantavirus/about/index.html",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_role": "data_source",
                "final_screening_decision": "include_for_content_fetch",
                "ready_for_content_fetch": True,
                "requires_human_review": False,
                "routing_flags": [],
            }
        ],
        "collection_trace": [],
    }

    result = content_fetch_and_parse(state)

    requests = result.get("content_fetch_requests") or []
    documents = result.get("documents") or []
    assert len(requests) == 1
    assert len(documents) == 1
    assert documents[0].get("is_offline_stub") is True
    summary = result.get("content_fetch_summary") or {}
    assert summary.get("collection_mode") == "masked_validation"
    assert summary.get("skipped_validation_reserved_count") == 0
    assert summary.get("skipped_validation_reserved_source_ids") == []


def test_no_network_by_default(monkeypatch):
    """The default workflow path must never call requests.get."""

    import requests

    def _fail(*args, **kwargs):
        raise AssertionError("Network call attempted in offline mode")

    monkeypatch.setattr(requests, "get", _fail)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)

    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert result.get("final_data_package") is not None
    summary = result.get("content_fetch_summary") or {}
    assert summary.get("live_fetch_enabled") is False


def test_unified_workflow_run_script_exists():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_workflow.py"
    assert script.exists(), f"missing {script}"


# ---------------------------------------------------------------------------
# Step 6: evidence chunking + data presence flagging
# ---------------------------------------------------------------------------


def test_default_offline_mode_skips_stub_documents_for_evidence_chunking(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    documents = result.get("documents") or []
    chunks = result.get("evidence_chunks") or []
    summary = result.get("evidence_chunking_summary") or {}
    assert len(documents) == 10
    assert len(chunks) == 0
    assert summary, "evidence_chunking_summary should be populated"
    assert summary.get("input_document_count") == 10
    assert summary.get("total_chunk_count") == 0
    skip_reason_counts = summary.get("skip_reason_counts") or {}
    assert skip_reason_counts.get("offline_stub") == 10


def test_data_presence_summary_empty_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("data_presence_summary") or {}
    assert summary, "data_presence_summary should be populated"
    assert summary.get("total_chunk_count") == 0
    assert summary.get("target_data_chunk_count") == 0


def test_step5_skip_summary_counts_are_granular(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("content_fetch_summary") or {}
    assert summary
    assert summary.get("skipped_search_endpoint_count", 0) >= 3
    assert summary.get("skipped_blocked_scheme_count", 0) >= 2
    assert summary.get("skipped_deferred_count", 0) >= 0
    assert summary.get("fetch_request_count") == 10
    assert summary.get("document_count") == 10


def _direct_node():
    from data_collection_workflow.nodes.content_processing import (
        evidence_chunking_and_data_presence_flagging,
    )
    return evidence_chunking_and_data_presence_flagging


def test_evidence_chunking_node_chunks_usable_document_directly():
    node = _direct_node()
    state = {
        "documents": [
            {
                "source_id": "src_test_real_doc",
                "document_type": "html",
                "clean_text": (
                    "In 2023, 12 human hantavirus cases and 2 deaths were reported "
                    "in Country X during an outbreak. Surveillance authorities "
                    "reported confirmed cases."
                ),
                "tables": [],
                "metadata": {},
                "parse_status": "parsed_html",
                "quality_status": "usable",
                "quality_issues": [],
                "url": "https://example.org/hantavirus-report",
                "canonical_url": "https://example.org/hantavirus-report",
                "title": "Example Hantavirus Report",
                "publisher": "Example Public Health Agency",
                "source_type": "official_public_health_agency",
                "source_role": "data_source",
                "fetch_purpose": "data_extraction",
                "is_live_fetched": True,
                "is_offline_stub": False,
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    chunks = result.get("evidence_chunks") or []
    chunking_summary = result.get("evidence_chunking_summary") or {}
    presence_summary = result.get("data_presence_summary") or {}

    assert len(chunks) >= 1
    first = chunks[0]
    assert first.get("contains_target_data") is True
    data_types = first.get("data_types") or []
    assert "case_count" in data_types
    assert "death_count" in data_types
    assert (first.get("confidence") or 0) >= 0.70
    assert chunking_summary.get("total_chunk_count", 0) >= 1
    assert presence_summary.get("target_data_chunk_count", 0) >= 1


def test_evidence_chunking_node_context_document_directly():
    node = _direct_node()
    state = {
        "documents": [
            {
                "source_id": "src_test_context_doc",
                "document_type": "html",
                "clean_text": (
                    "Hantavirus pulmonary syndrome is a severe respiratory disease. "
                    "This clinical overview describes symptoms, transmission, "
                    "diagnosis, and prevention."
                ),
                "tables": [],
                "metadata": {},
                "parse_status": "parsed_html",
                "quality_status": "usable",
                "quality_issues": [],
                "url": "https://example.org/hantavirus-overview",
                "canonical_url": "https://example.org/hantavirus-overview",
                "title": "Example Hantavirus Overview",
                "publisher": "Example Public Health Agency",
                "source_type": "official_public_health_agency",
                "source_role": "context_source",
                "fetch_purpose": "context_grounding",
                "is_live_fetched": True,
                "is_offline_stub": False,
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    chunks = result.get("evidence_chunks") or []
    assert len(chunks) >= 1
    first = chunks[0]
    assert first.get("contains_target_data") is False
    context_types = first.get("context_types") or []
    assert any(ct in {"disease_definition", "clinical_or_background"} for ct in context_types)
    assert first.get("confidence") is not None


def test_evidence_chunking_tables_directly():
    node = _direct_node()
    state = {
        "documents": [
            {
                "source_id": "src_test_table_doc",
                "document_type": "html",
                "clean_text": (
                    "Annual hantavirus surveillance summary covering Country X "
                    "for the reporting period 2020-2023."
                ),
                "tables": [
                    {
                        "table_index": 0,
                        "rows": [
                            ["Year", "Country", "Cases", "Deaths"],
                            ["2023", "Country X", "12", "2"],
                        ],
                    }
                ],
                "metadata": {},
                "parse_status": "parsed_html",
                "quality_status": "usable",
                "quality_issues": [],
                "url": "https://example.org/hantavirus-surveillance",
                "canonical_url": "https://example.org/hantavirus-surveillance",
                "title": "Example Hantavirus Surveillance",
                "publisher": "Example Public Health Agency",
                "source_type": "official_public_health_agency",
                "source_role": "data_source",
                "fetch_purpose": "data_extraction",
                "is_live_fetched": True,
                "is_offline_stub": False,
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    chunks = result.get("evidence_chunks") or []
    table_chunks = [c for c in chunks if c.get("chunk_kind") == "table"]
    assert table_chunks, "expected at least one table chunk"
    table_chunk = table_chunks[0]
    assert table_chunk.get("contains_target_data") is True
    data_types = table_chunk.get("data_types") or []
    assert any(dt in {"case_count", "death_count"} for dt in data_types)


# ---------------------------------------------------------------------------
# Step 7: structured extraction + schema validation
# ---------------------------------------------------------------------------


def _extraction_nodes():
    from data_collection_workflow.nodes.extraction import (
        schema_validation_and_repair,
        structured_extraction,
    )
    return structured_extraction, schema_validation_and_repair


def _synthetic_text_chunk() -> dict:
    return {
        "chunk_id": "chunk_src_test_real_doc_001",
        "source_id": "src_test_real_doc",
        "text": (
            "In 2023, 12 human hantavirus cases and 2 deaths were reported in "
            "Country X during an outbreak. Surveillance authorities reported "
            "confirmed cases."
        ),
        "contains_target_data": True,
        "data_types": [
            "case_count",
            "death_count",
            "outbreak",
            "surveillance",
            "date",
            "location",
        ],
        "context_types": ["disease_definition"],
        "confidence": 0.85,
        "document_type": "html",
        "fetch_purpose": "data_extraction",
        "source_url": "https://example.org/hantavirus-report",
        "canonical_url": "https://example.org/hantavirus-report",
        "title": "Example Hantavirus Report",
        "publisher": "Example Public Health Agency",
        "source_type": "official_public_health_agency",
        "source_role": "data_source",
        "quality_status": "usable",
        "chunk_index": 1,
        "chunk_kind": "text",
        "char_start": 0,
        "char_end": 150,
    }


def test_default_offline_mode_has_no_extracted_records(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert (result.get("raw_records") or []) == []
    assert (result.get("validated_records") or []) == []
    assert (result.get("rejected_records") or []) == []
    extraction_summary = result.get("structured_extraction_summary") or {}
    schema_summary = result.get("schema_validation_summary") or {}
    assert extraction_summary, "structured_extraction_summary should be populated"
    assert extraction_summary.get("raw_record_count") == 0
    assert schema_summary, "schema_validation_summary should be populated"
    assert schema_summary.get("validated_record_count") == 0


def test_structured_extraction_from_text_chunk_directly():
    structured_extraction, _ = _extraction_nodes()
    state = {"evidence_chunks": [_synthetic_text_chunk()], "collection_trace": []}
    result = structured_extraction(state)
    raw_records = result.get("raw_records") or []
    assert len(raw_records) == 1
    record = raw_records[0]
    assert record.get("disease") == "Hantavirus disease"
    assert (
        record.get("cases_unspecified") == 12
        or record.get("cases_confirmed") == 12
    )
    assert record.get("deaths") == 2
    assert record.get("date_reported") == "2023"
    assert record.get("country") == "Country X"
    assert record.get("source_url") == "https://example.org/hantavirus-report"
    assert record.get("evidence_quote")
    assert record.get("supporting_chunk_id") == "chunk_src_test_real_doc_001"


def test_schema_validation_accepts_extracted_text_record_directly():
    structured_extraction, schema_validation_and_repair = _extraction_nodes()
    state = {"evidence_chunks": [_synthetic_text_chunk()], "collection_trace": []}
    extraction_result = structured_extraction(state)
    raw_records = extraction_result.get("raw_records") or []
    validation_state = {
        "raw_records": raw_records,
        "human_review_queue": [],
        "collection_trace": [],
    }
    validation_result = schema_validation_and_repair(validation_state)
    validated = validation_result.get("validated_records") or []
    rejected = validation_result.get("rejected_records") or []
    summary = validation_result.get("schema_validation_summary") or {}
    assert len(validated) == 1
    assert validated[0].get("schema_status") == "valid"
    assert validated[0].get("provenance_status") == "verified"
    assert rejected == []
    assert summary.get("validated_record_count") == 1


def test_structured_extraction_from_table_chunk_directly():
    structured_extraction, _ = _extraction_nodes()
    table_chunk = {
        "chunk_id": "chunk_src_table_doc_001",
        "source_id": "src_table_doc",
        "text": "Year | Country | Cases | Deaths\n2023 | Country X | 12 | 2",
        "contains_target_data": True,
        "data_types": ["case_count", "death_count", "date", "location"],
        "context_types": [],
        "confidence": 0.85,
        "document_type": "html",
        "fetch_purpose": "data_extraction",
        "source_url": "https://example.org/table",
        "canonical_url": "https://example.org/table",
        "title": "Example Table",
        "publisher": "Example Public Health Agency",
        "source_type": "official_public_health_agency",
        "source_role": "data_source",
        "quality_status": "usable",
        "chunk_index": 1,
        "chunk_kind": "table",
        "char_start": None,
        "char_end": None,
    }
    state = {"evidence_chunks": [table_chunk], "collection_trace": []}
    result = structured_extraction(state)
    raw_records = result.get("raw_records") or []
    assert len(raw_records) == 1
    record = raw_records[0]
    assert record.get("date_reported") == "2023"
    assert record.get("country") == "Country X"
    assert (
        record.get("cases_unspecified") == 12
        or record.get("cases_confirmed") == 12
    )
    assert record.get("deaths") == 2


def test_schema_validation_flags_missing_country_for_review():
    _, schema_validation_and_repair = _extraction_nodes()
    raw_record = {
        "record_id": "rec_test_missing_country_001",
        "disease": "Hantavirus disease",
        "source_id": "src_test_missing_country",
        "source_url": "https://example.org/missing-country",
        "source_type": "official_public_health_agency",
        "evidence_quote": "Five cases were reported.",
        "supporting_chunk_id": "chunk_src_test_missing_country_001",
        "cases_unspecified": 5,
    }
    state = {
        "raw_records": [raw_record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = schema_validation_and_repair(state)
    validated = result.get("validated_records") or []
    queue = result.get("human_review_queue") or []
    assert len(validated) == 1
    assert validated[0].get("schema_status") == "needs_review"
    assert validated[0].get("requires_human_review") is True
    review_items = [
        item for item in queue
        if item.get("item_type") == "record_schema_validation"
    ]
    assert len(review_items) == 1


def test_schema_validation_rejects_record_without_content():
    _, schema_validation_and_repair = _extraction_nodes()
    raw_record = {
        "record_id": "rec_test_no_content_001",
        "disease": "Hantavirus disease",
        "source_id": "src_test_no_content",
        "source_url": "https://example.org/no-content",
        "source_type": "official_public_health_agency",
        "evidence_quote": "General overview without data.",
        "supporting_chunk_id": "chunk_src_test_no_content_001",
    }
    state = {
        "raw_records": [raw_record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = schema_validation_and_repair(state)
    validated = result.get("validated_records") or []
    rejected = result.get("rejected_records") or []
    assert validated == []
    assert len(rejected) == 1
    assert rejected[0].get("schema_status") == "rejected"


def test_structured_extraction_summary_counts_fields():
    structured_extraction, _ = _extraction_nodes()
    state = {"evidence_chunks": [_synthetic_text_chunk()], "collection_trace": []}
    result = structured_extraction(state)
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("input_chunk_count") == 1
    assert summary.get("raw_record_count") == 1
    field_counts = summary.get("field_detection_counts") or {}
    assert field_counts.get("deaths") == 1
    assert field_counts.get("date_reported") == 1
    assert field_counts.get("country") == 1


# ---------------------------------------------------------------------------
# Step 8: record normalization
# ---------------------------------------------------------------------------


def _normalization_node():
    from data_collection_workflow.nodes.normalization import record_normalization
    return record_normalization


def _baseline_validated_record(**overrides) -> dict:
    base = {
        "record_id": "rec_test_001",
        "disease": "Hantavirus disease",
        "virus_or_syndrome": None,
        "country": None,
        "subnational_location": None,
        "date_reported": None,
        "cases_unspecified": None,
        "deaths": None,
        "case_definition": None,
        "source_id": "src_test",
        "source_url": "https://example.org/report",
        "source_type": "official_public_health_agency",
        "evidence_quote": "Synthetic test record.",
        "supporting_chunk_id": "chunk_test_001",
        "schema_status": "valid",
        "provenance_status": "verified",
        "requires_human_review": False,
    }
    base.update(overrides)
    return base


def test_default_offline_mode_has_no_normalized_records(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert (result.get("normalized_records") or []) == []
    summary = result.get("record_normalization_summary") or {}
    assert summary, "record_normalization_summary should be populated"
    assert summary.get("normalized_record_count") == 0


def test_record_normalization_country_date_syndrome_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        virus_or_syndrome="hantavirus pulmonary syndrome",
        country="USA",
        subnational_location="New Mexico",
        date_reported="March 2023",
        cases_unspecified=12,
        deaths=2,
        case_definition="unspecified",
        evidence_quote=(
            "In March 2023, 12 human hantavirus cases and 2 deaths were "
            "reported in New Mexico, USA."
        ),
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = result.get("normalized_records") or []
    assert len(normalized) == 1
    n = normalized[0]
    assert n.get("country") == "United States of America"
    assert n.get("country_raw") == "USA"
    assert n.get("subnational_location") == "New Mexico"
    assert n.get("date_reported") == "2023-03"
    assert n.get("date_reported_raw") == "March 2023"
    assert n.get("virus_or_syndrome") == "HPS"
    assert n.get("virus_or_syndrome_raw") == "hantavirus pulmonary syndrome"
    assert n.get("normalization_status") in {"normalized", "normalized_with_warnings"}
    actions = n.get("normalization_actions") or []
    assert "normalized_country_alias" in actions
    assert "normalized_date" in actions
    assert "normalized_virus_or_syndrome" in actions


def test_record_normalization_infers_country_from_subnational_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country=None,
        subnational_location="New Mexico",
        date_reported="2023",
        cases_unspecified=5,
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    assert normalized.get("country") == "United States of America"
    assert normalized.get("subnational_location") == "New Mexico"
    actions = normalized.get("normalization_actions") or []
    assert "inferred_country_from_subnational_location" in actions
    assert normalized.get("requires_human_review") is False


def test_record_normalization_non_country_geography_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="Europe",
        subnational_location=None,
        date_reported="2023",
        cases_unspecified=5,
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    queue = result.get("human_review_queue") or []
    assert normalized.get("country") is None
    # Step 16: Europe is now recognized as a region and routed to
    # geographic_scope rather than abused as subnational.
    assert normalized.get("geographic_scope") == "Europe"
    assert normalized.get("geographic_scope_type") in {"region", "multi_country"}
    warnings = normalized.get("normalization_warnings") or []
    assert "regional_geographic_scope_not_country" in warnings
    # Step 16.1: regional aggregate scope is valid geography. The record
    # gets a non-review `regional_or_aggregate_geographic_scope` warning
    # instead of `missing_country_after_normalization`, so the record no
    # longer requires human review.
    assert "regional_or_aggregate_geographic_scope" in warnings
    assert "missing_country_after_normalization" not in warnings
    assert normalized.get("requires_human_review") is False
    review_items = [
        item for item in queue if item.get("item_type") == "record_normalization"
    ]
    assert review_items == []


def test_record_normalization_case_definition_inferred_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="Chile",
        date_reported="2024",
        cases_confirmed=7,
        cases_probable=2,
        case_definition=None,
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    cd = normalized.get("case_definition") or ""
    assert "confirmed" in cd
    assert "probable" in cd
    actions = normalized.get("normalization_actions") or []
    assert "inferred_case_definition_from_case_fields" in actions


def test_record_normalization_numeric_strings_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="Argentina",
        date_reported="2022",
        cases_unspecified="1,234",
        deaths="12",
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    assert normalized.get("cases_unspecified") == 1234.0
    assert normalized.get("deaths") == 12.0
    actions = normalized.get("normalization_actions") or []
    assert "normalized_numeric_field:cases_unspecified" in actions


def test_record_normalization_unrecognized_source_type_review_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="Country X",
        date_reported="2023",
        cases_unspecified=1,
        source_type="unknown_blog",
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    queue = result.get("human_review_queue") or []
    assert normalized.get("source_type") == "unknown_blog"
    warnings = normalized.get("normalization_warnings") or []
    assert "unrecognized_source_type" in warnings
    assert normalized.get("requires_human_review") is True
    review_items = [
        item for item in queue if item.get("item_type") == "record_normalization"
    ]
    assert len(review_items) == 1


def test_record_normalization_preserves_provenance_directly():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="USA",
        date_reported="2023",
        cases_unspecified=10,
        source_id="src_prov_check",
        source_url="https://example.org/prov",
        evidence_quote="Quote text.",
        supporting_chunk_id="chunk_prov_check_001",
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    assert normalized.get("source_id") == "src_prov_check"
    assert normalized.get("source_url") == "https://example.org/prov"
    assert normalized.get("evidence_quote") == "Quote text."
    assert normalized.get("supporting_chunk_id") == "chunk_prov_check_001"


def test_record_normalization_summary_counts_actions():
    node = _normalization_node()
    record_a = _baseline_validated_record(
        record_id="rec_test_a",
        country="USA",
        subnational_location="New Mexico",
        date_reported="March 2023",
        virus_or_syndrome="hantavirus pulmonary syndrome",
        cases_unspecified=10,
        case_definition="unspecified",
    )
    record_b = _baseline_validated_record(
        record_id="rec_test_b",
        country="Country X",
        date_reported="2023",
        cases_unspecified=5,
    )
    state = {
        "validated_records": [record_a, record_b],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    summary = result.get("record_normalization_summary") or {}
    assert summary.get("input_validated_record_count") == 2
    assert summary.get("normalized_record_count") == 2
    action_counts = summary.get("normalization_action_counts") or {}
    assert action_counts, "normalization_action_counts should be non-empty"
    assert summary.get("country_normalized_count", 0) >= 1
    assert summary.get("date_normalized_count", 0) >= 1


# ---------------------------------------------------------------------------
# Step 9: record linking
# ---------------------------------------------------------------------------


def _linking_node():
    from data_collection_workflow.nodes.linking_validation import record_linking
    return record_linking


def _baseline_normalized_record(**overrides) -> dict:
    base = {
        "record_id": "rec_test_001",
        "disease": "Hantavirus disease",
        "virus_or_syndrome": None,
        "country": None,
        "subnational_location": None,
        "date_reported": None,
        "event_start_date": None,
        "event_end_date": None,
        "cases_unspecified": None,
        "deaths": None,
        "case_definition": None,
        "source_id": "src_test",
        "source_url": "https://example.org/report",
        "source_type": "official_public_health_agency",
        "evidence_quote": "Synthetic test record.",
        "supporting_chunk_id": "chunk_test_001",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "requires_human_review": False,
    }
    base.update(overrides)
    return base


def test_default_offline_mode_has_no_linked_events(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert (result.get("linked_events") or []) == []
    summary = result.get("record_linking_summary") or {}
    assert summary, "record_linking_summary should be populated"
    assert summary.get("linked_event_count") == 0


def test_record_linking_groups_same_event_directly():
    node = _linking_node()
    common = dict(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2023-03",
        cases_unspecified=12,
        deaths=2,
    )
    rec_a = _baseline_normalized_record(
        record_id="rec_a",
        source_id="src_1",
        source_url="https://example.org/source-1",
        evidence_quote="source 1 quote",
        supporting_chunk_id="chunk_1",
        **common,
    )
    rec_b = _baseline_normalized_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        evidence_quote="source 2 quote",
        supporting_chunk_id="chunk_2",
        **common,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    normalized_out = result.get("normalized_records") or []
    assert len(linked) == 1
    event = linked[0]
    assert event.get("record_count") == 2
    assert event.get("linking_status") == "linked_multi_source_event"
    assert event.get("requires_human_review") is False
    src_ids = set(event.get("source_ids") or [])
    assert {"src_1", "src_2"}.issubset(src_ids)
    a_event = next(r for r in normalized_out if r.get("record_id") == "rec_a").get("linked_event_id")
    b_event = next(r for r in normalized_out if r.get("record_id") == "rec_b").get("linked_event_id")
    assert a_event and a_event == b_event


def test_record_linking_keeps_different_dates_separate_directly():
    node = _linking_node()
    rec_a = _baseline_normalized_record(
        record_id="rec_a",
        country="Chile",
        date_reported="2023",
        cases_unspecified=5,
    )
    rec_b = _baseline_normalized_record(
        record_id="rec_b",
        country="Chile",
        date_reported="2024",
        cases_unspecified=7,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    normalized_out = result.get("normalized_records") or []
    assert len(linked) == 2
    for event in linked:
        assert event.get("record_count") == 1
    a_event = next(r for r in normalized_out if r.get("record_id") == "rec_a").get("linked_event_id")
    b_event = next(r for r in normalized_out if r.get("record_id") == "rec_b").get("linked_event_id")
    assert a_event and b_event
    assert a_event != b_event


def test_record_linking_single_record_event_directly():
    node = _linking_node()
    rec = _baseline_normalized_record(
        country="Argentina",
        date_reported="2022",
        cases_unspecified=3,
    )
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    assert len(linked) == 1
    event = linked[0]
    assert event.get("linking_status") == "single_record_event"
    assert (event.get("linking_confidence") or 0) >= 0.70
    assert event.get("requires_human_review") is False


def test_record_linking_missing_country_needs_review_directly():
    node = _linking_node()
    rec = _baseline_normalized_record(
        country=None,
        subnational_location=None,
        date_reported="2023",
        cases_unspecified=5,
        normalization_status="normalized_with_warnings",
    )
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    queue = result.get("human_review_queue") or []
    assert len(linked) == 1
    event = linked[0]
    assert event.get("requires_human_review") is True
    warnings = event.get("linking_warnings") or []
    assert "missing_country_for_case_data" in warnings
    review_items = [item for item in queue if item.get("item_type") == "record_linking"]
    assert len(review_items) == 1


def test_record_linking_missing_date_anchor_needs_review_directly():
    node = _linking_node()
    rec = _baseline_normalized_record(
        country="Chile",
        date_reported=None,
        event_start_date=None,
        event_end_date=None,
        cases_unspecified=5,
    )
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    assert len(linked) == 1
    event = linked[0]
    warnings = event.get("linking_warnings") or []
    assert "missing_date_anchor" in warnings
    assert event.get("requires_human_review") is True


def test_record_linking_preserves_provenance_directly():
    node = _linking_node()
    rec = _baseline_normalized_record(
        country="USA",
        date_reported="2023",
        cases_unspecified=4,
        source_id="src_prov",
        source_url="https://example.org/prov",
        evidence_quote="Preserve me.",
        supporting_chunk_id="chunk_prov_001",
    )
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized_out = (result.get("normalized_records") or [])[0]
    assert normalized_out.get("source_id") == "src_prov"
    assert normalized_out.get("source_url") == "https://example.org/prov"
    assert normalized_out.get("evidence_quote") == "Preserve me."
    assert normalized_out.get("supporting_chunk_id") == "chunk_prov_001"
    linked = result.get("linked_events") or []
    assert linked
    assert "https://example.org/prov" in (linked[0].get("source_urls") or [])


def test_record_linking_summary_counts():
    node = _linking_node()
    common = dict(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2023-03",
        cases_unspecified=12,
    )
    rec_a = _baseline_normalized_record(
        record_id="rec_a",
        source_id="src_1",
        source_url="https://example.org/source-1",
        **common,
    )
    rec_b = _baseline_normalized_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        **common,
    )
    rec_c = _baseline_normalized_record(
        record_id="rec_c",
        country="Chile",
        date_reported="2024",
        cases_unspecified=7,
    )
    state = {
        "normalized_records": [rec_a, rec_b, rec_c],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    summary = result.get("record_linking_summary") or {}
    assert summary.get("input_normalized_record_count") == 3
    assert summary.get("linked_event_count") == 2
    assert summary.get("multi_source_event_count") == 1
    assert summary.get("single_record_event_count") == 1
    assert summary.get("records_with_linked_event_id_count") == 3
    action_counts = summary.get("linking_action_counts") or {}
    assert action_counts, "linking_action_counts should be non-empty"


# ---------------------------------------------------------------------------
# Step 10: cross-source consistency check
# ---------------------------------------------------------------------------


def _consistency_node():
    from data_collection_workflow.nodes.linking_validation import cross_source_consistency_check
    return cross_source_consistency_check


def _quality_gate_node():
    from data_collection_workflow.nodes.linking_validation import quality_gate_routing
    return quality_gate_routing


def _consistency_record(**overrides) -> dict:
    base = {
        "record_id": "rec_a",
        "disease": "Hantavirus disease",
        "virus_or_syndrome": "HPS",
        "country": "Country X",
        "subnational_location": None,
        "date_reported": "2023",
        "date_anchor": "2023",
        "cases_unspecified": 12,
        "deaths": 2,
        "case_definition": "unspecified",
        "source_id": "src_1",
        "source_url": "https://example.org/source-1",
        "source_type": "official_public_health_agency",
        "evidence_quote": "Synthetic evidence quote.",
        "supporting_chunk_id": "chunk_1",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "linked_event_id": "event_001",
    }
    base.update(overrides)
    return base


def _consistency_event(record_ids: list[str], **overrides) -> dict:
    base = {
        "linked_event_id": "event_001",
        "record_ids": list(record_ids),
        "linking_basis": ["same_disease", "same_country"],
        "linking_confidence": 0.95,
        "record_count": len(record_ids),
        "source_ids": [],
        "source_urls": [],
        "linking_status": "linked_multi_source_event",
        "linking_method": "deterministic_event_key_linker",
        "linking_warnings": [],
        "requires_human_review": False,
    }
    base.update(overrides)
    return base


def test_default_offline_mode_has_no_conflicts(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    assert (result.get("conflicts") or []) == []
    summary = result.get("cross_source_consistency_summary") or {}
    assert summary, "cross_source_consistency_summary should be populated"
    assert summary.get("conflict_count") == 0
    assert result.get("current_route") == "finalize"


def test_cross_source_consistency_no_conflict_same_values_directly():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1")
    rec_b = _consistency_record(record_id="rec_b", source_id="src_2",
                                 source_url="https://example.org/source-2")
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    events = result.get("linked_events") or []
    records = result.get("normalized_records") or []
    summary = result.get("cross_source_consistency_summary") or {}
    assert conflicts == []
    assert events[0].get("consistency_status") == "consistent"
    for rec in records:
        assert rec.get("record_conflict_status") == "no_conflict"
    assert summary.get("comparable_event_count") == 1
    assert summary.get("conflict_count") == 0


def test_cross_source_consistency_minor_numeric_difference_directly():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", cases_unspecified=12)
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=13,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.get("field") == "cases_unspecified"
    assert c.get("conflict_type") == "minor_numeric_difference"
    assert c.get("severity") == "low"
    assert c.get("requires_human_review") is False
    assert (result.get("linked_events") or [])[0].get(
        "consistency_status"
    ) == "consistent_with_minor_differences"
    assert (result.get("human_review_queue") or []) == []


def test_cross_source_consistency_major_numeric_difference_directly():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", cases_unspecified=12)
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=30,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    queue = result.get("human_review_queue") or []
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.get("conflict_type") == "major_numeric_difference"
    assert c.get("severity") == "high"
    assert c.get("requires_human_review") is True
    assert (result.get("linked_events") or [])[0].get("consistency_status") == "needs_review"
    review_items = [item for item in queue if item.get("item_type") == "cross_source_conflict"]
    assert review_items
    for rec in result.get("normalized_records") or []:
        assert rec.get("record_conflict_status") == "needs_review"


def test_cross_source_consistency_death_mismatch_directly():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", deaths=2)
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        deaths=5,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    deaths_conflicts = [c for c in conflicts if c.get("field") == "deaths"]
    assert deaths_conflicts
    c = deaths_conflicts[0]
    assert c.get("severity") in {"high", "medium"}
    assert c.get("conflict_type") in {"major_numeric_difference", "numeric_mismatch"}


def test_cross_source_consistency_location_mismatch_directly():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", country="Chile")
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        country="Argentina",
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    country_conflicts = [c for c in conflicts if c.get("field") == "country"]
    assert country_conflicts
    c = country_conflicts[0]
    assert c.get("conflict_type") == "location_mismatch"
    assert c.get("severity") == "high"
    assert c.get("requires_human_review") is True


def test_cross_source_consistency_date_mismatch_directly():
    node = _consistency_node()
    rec_a = _consistency_record(
        record_id="rec_a",
        source_id="src_1",
        date_reported="2023",
        date_anchor=None,
    )
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        date_reported="2024",
        date_anchor=None,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    date_conflicts = [
        c for c in conflicts if c.get("field") in {"date_reported", "date_anchor"}
    ]
    assert date_conflicts
    c = date_conflicts[0]
    assert c.get("conflict_type") == "date_mismatch"
    assert c.get("requires_human_review") is True


def test_cross_source_consistency_case_definition_mismatch_directly():
    node = _consistency_node()
    rec_a = _consistency_record(
        record_id="rec_a", source_id="src_1", case_definition="confirmed"
    )
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        case_definition="suspected",
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    cd_conflicts = [c for c in conflicts if c.get("field") == "case_definition"]
    assert cd_conflicts
    c = cd_conflicts[0]
    assert c.get("conflict_type") == "case_definition_mismatch"
    assert c.get("severity") == "medium"
    assert c.get("requires_human_review") is False


def test_cross_source_consistency_preserves_existing_conflicts():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", cases_unspecified=12)
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=30,
    )
    existing_conflict = {
        "conflict_id": "conf_existing",
        "linked_event_id": "event_999",
        "field": "deaths",
        "values": [{"record_id": "rec_z", "value": 1}],
        "conflict_type": "minor_numeric_difference",
        "severity": "low",
        "record_ids": ["rec_z"],
    }
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [existing_conflict],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    assert len(conflicts) == 2
    ids = {c.get("conflict_id") for c in conflicts}
    assert "conf_existing" in ids
    new_ids = ids - {"conf_existing"}
    assert new_ids, "should have created a new conflict id"
    new_id = next(iter(new_ids))
    assert new_id != "conf_existing"
    assert new_id.startswith("conf_")


def test_quality_gate_routes_to_human_review_when_queue_nonempty():
    node = _quality_gate_node()
    state = {
        "human_review_queue": [
            {
                "review_id": "review_conflict_conf_001",
                "item_type": "cross_source_conflict",
                "related_ids": ["conf_001"],
                "reason": "test",
                "status": "pending",
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    assert result.get("current_route") == "human_review"


def test_quality_gate_bypasses_human_review_when_disabled():
    node = _quality_gate_node()
    state = {
        "human_review_enabled": False,
        "human_review_queue": [
            {
                "review_id": "review_conflict_conf_001",
                "item_type": "cross_source_conflict",
                "priority": "high",
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    assert result.get("current_route") == "finalize"
    assert len(result.get("human_review_queue") or []) == 1
    trace = result.get("collection_trace") or []
    assert trace[-1]["metadata"]["human_review_enabled"] is False


def test_quality_gate_routes_to_finalize_when_no_review():
    node = _quality_gate_node()
    state = {"human_review_queue": [], "collection_trace": []}
    result = node(state)
    assert result.get("current_route") == "finalize"


# ---------------------------------------------------------------------------
# Step 11: synthetic fixture document mode
# ---------------------------------------------------------------------------


def _fixture_initial_state() -> dict:
    return _sanity_initial_state()


def test_default_mode_does_not_use_fixture_documents(monkeypatch):
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    documents = result.get("documents") or []
    assert len(documents) == 10
    for doc in documents:
        assert doc.get("is_offline_stub") is True
        assert not doc.get("is_fixture_document")
    assert (result.get("evidence_chunks") or []) == []
    assert (result.get("raw_records") or []) == []
    assert (result.get("validated_records") or []) == []
    assert (result.get("normalized_records") or []) == []
    assert (result.get("linked_events") or []) == []
    assert (result.get("conflicts") or []) == []


def test_fixture_mode_loads_fixture_documents(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    documents = result.get("documents") or []
    assert len(documents) == 10
    fixture_docs = [d for d in documents if d.get("is_fixture_document")]
    assert len(fixture_docs) >= 4
    for d in fixture_docs:
        assert d.get("is_offline_stub") is False
    fixture_summary = result.get("fixture_document_summary") or {}
    assert fixture_summary, "fixture_document_summary should be populated"
    assert fixture_summary.get("loaded_fixture_count", 0) >= 4
    assert "not real public health data" in (
        fixture_summary.get("synthetic_fixture_notice") or ""
    ).lower()


def test_fixture_mode_creates_evidence_chunks_and_records(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    assert len(result.get("evidence_chunks") or []) > 0
    assert len(result.get("raw_records") or []) > 0
    assert len(result.get("validated_records") or []) > 0
    assert len(result.get("normalized_records") or []) > 0
    extraction_summary = result.get("structured_extraction_summary") or {}
    schema_summary = result.get("schema_validation_summary") or {}
    norm_summary = result.get("record_normalization_summary") or {}
    assert extraction_summary.get("raw_record_count", 0) > 0
    assert schema_summary.get("validated_record_count", 0) > 0
    assert norm_summary.get("normalized_record_count", 0) > 0


def test_fixture_mode_creates_linked_events_and_conflicts(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    linked_events = result.get("linked_events") or []
    conflicts = result.get("conflicts") or []
    link_summary = result.get("record_linking_summary") or {}
    cons_summary = result.get("cross_source_consistency_summary") or {}
    assert len(linked_events) > 0
    assert len(conflicts) > 0
    assert link_summary.get("linked_event_count", 0) > 0
    assert cons_summary.get("conflict_count", 0) > 0
    case_conflicts = [
        c for c in conflicts
        if c.get("field") in {"cases_unspecified", "cases_confirmed"}
    ]
    assert case_conflicts, "expected a case-count conflict"
    assert any(c.get("severity") == "high" for c in conflicts)
    assert any(c.get("requires_human_review") for c in conflicts)


def test_fixture_mode_routes_to_human_review(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    queue = result.get("human_review_queue") or []
    trace = result.get("collection_trace") or []
    package = result.get("final_data_package") or {}
    assert len(queue) > 0
    assert result.get("current_route") == "human_review"
    assert any(t.get("node_name") == "human_review" for t in trace)
    assert package, "final_data_package should be populated"
    assert len(package.get("human_review_items") or []) > 0


def test_fixture_mode_final_package_non_empty(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    package = result.get("final_data_package") or {}
    assert package
    assert len(package.get("final_dataset") or []) > 0
    assert len(package.get("linked_events") or []) > 0
    assert len(package.get("conflicts") or []) > 0
    assert len(package.get("human_review_items") or []) > 0


def test_fixture_mode_no_network(monkeypatch):
    import requests

    def _fail(*args, **kwargs):
        raise AssertionError("Network call attempted in fixture mode")

    monkeypatch.setattr(requests, "get", _fail)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())
    assert result.get("final_data_package") is not None
    documents = result.get("documents") or []
    assert any(d.get("is_fixture_document") for d in documents)


def test_fixture_masked_collection_excludes_reserved_records(monkeypatch):
    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())

    documents = result.get("documents") or []
    document_ids = {d.get("source_id") for d in documents}
    assert not (document_ids & _RESERVED_SOURCE_IDS)

    records = result.get("normalized_records") or []
    assert not any(r.get("source_id") in _RESERVED_SOURCE_IDS for r in records)

    package = result.get("final_data_package") or {}
    final_dataset = package.get("final_dataset") or []
    assert not any(r.get("source_id") in _RESERVED_SOURCE_IDS for r in final_dataset)

    fetch_summary = result.get("content_fetch_summary") or {}
    assert fetch_summary.get("collection_mode") == "masked_validation"
    assert fetch_summary.get("skipped_validation_reserved_count") == len(
        _RESERVED_SOURCE_IDS
    )
    assert set(fetch_summary.get("skipped_validation_reserved_source_ids") or []) == (
        _RESERVED_SOURCE_IDS
    )


def test_fixture_masked_collection_has_non_reserved_records(monkeypatch):
    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    graph = build_graph()
    result = graph.invoke(_fixture_initial_state())

    package = result.get("final_data_package") or {}
    final_dataset = package.get("final_dataset") or []
    assert len(final_dataset) >= 1

    source_ids = {record.get("source_id") for record in final_dataset}
    assert "src_paho_hantavirus_americas_guidelines" in source_ids
    assert not (source_ids & _RESERVED_SOURCE_IDS)

    records_with_complete_evidence = [
        record
        for record in final_dataset
        if record.get("source_url")
        and record.get("evidence_quote")
        and record.get("supporting_chunk_id")
        and record.get("linked_event_id")
    ]
    assert records_with_complete_evidence

    fetch_summary = result.get("content_fetch_summary") or {}
    assert fetch_summary.get("collection_mode") == "masked_validation"
    assert fetch_summary.get("live_fetch_enabled") is False
    assert fetch_summary.get("skipped_validation_reserved_count") == len(
        _RESERVED_SOURCE_IDS
    )
    assert set(fetch_summary.get("skipped_validation_reserved_source_ids") or []) == (
        _RESERVED_SOURCE_IDS
    )


def test_workflow_console_builder_script_exists():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "build_console.py"
    assert script.exists(), f"missing {script}"


# ---------------------------------------------------------------------------
# Step 12: human review packets + decision intake
# ---------------------------------------------------------------------------


def _human_review_node():
    from data_collection_workflow.nodes.human_review import human_review
    return human_review


def test_fixture_mode_human_review_packets_created(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    queue = result.get("human_review_queue") or []
    summary = result.get("human_review_summary") or {}
    assert len(queue) > 0
    for item in queue:
        assert item.get("priority") is not None
        assert item.get("review_packet") is not None
    assert any(item.get("item_type") == "cross_source_conflict" for item in queue)
    assert summary
    assert summary.get("pending_count", 0) >= 1
    assert summary.get("fixture_origin_review_count", 0) >= 1


def test_fixture_mode_review_item_contains_conflict_context(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    queue = result.get("human_review_queue") or []
    conflict_items = [i for i in queue if i.get("item_type") == "cross_source_conflict"]
    assert conflict_items
    packet = conflict_items[0].get("review_packet") or {}
    sections = packet.get("packet_sections") or {}
    assert sections.get("conflict") is not None
    related = sections.get("related_records") or []
    assert len(related) >= 2
    field = (sections["conflict"] or {}).get("field")
    assert field in {"cases_unspecified", "cases_confirmed"}


def test_human_review_direct_empty_queue():
    node = _human_review_node()
    state = {"human_review_queue": [], "collection_trace": []}
    result = node(state)
    summary = result.get("human_review_summary") or {}
    assert summary
    assert summary.get("input_review_item_count") == 0
    assert summary.get("pending_count") == 0


def test_human_review_direct_applies_decision():
    node = _human_review_node()
    state = {
        "human_review_queue": [
            {
                "review_id": "review_conflict_conf_001",
                "item_type": "cross_source_conflict",
                "related_ids": ["conf_001", "rec_a", "rec_b"],
                "reason": "test conflict",
                "status": "pending",
            }
        ],
        "conflicts": [
            {
                "conflict_id": "conf_001",
                "linked_event_id": "event_001",
                "field": "cases_unspecified",
                "values": [],
                "conflict_type": "major_numeric_difference",
                "severity": "high",
                "record_ids": ["rec_a", "rec_b"],
                "requires_human_review": True,
            }
        ],
        "normalized_records": [
            {
                "record_id": "rec_a",
                "disease": "Hantavirus disease",
                "source_url": "https://example.org/a",
                "source_type": "official_public_health_agency",
                "evidence_quote": "quote a",
            },
            {
                "record_id": "rec_b",
                "disease": "Hantavirus disease",
                "source_url": "https://example.org/b",
                "source_type": "international_organization_report",
                "evidence_quote": "quote b",
            },
        ],
        "human_review_decisions": [
            {
                "review_id": "review_conflict_conf_001",
                "decision": "needs_more_evidence",
                "reviewer_id": "tester",
                "notes": "Need authoritative source.",
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    queue = result.get("human_review_queue") or []
    summary = result.get("human_review_summary") or {}
    assert len(queue) == 1
    item = queue[0]
    assert item.get("status") == "requires_follow_up"
    assert item.get("human_decision") == "needs_more_evidence"
    assert item.get("decision_applied") is True
    assert item.get("reviewer_id") == "tester"
    assert item.get("review_packet") is not None
    assert summary.get("decision_applied_count") == 1


def test_human_review_direct_invalid_decision():
    node = _human_review_node()
    state = {
        "human_review_queue": [
            {
                "review_id": "review_conflict_conf_001",
                "item_type": "cross_source_conflict",
                "related_ids": ["conf_001"],
                "reason": "test conflict",
                "status": "pending",
            }
        ],
        "conflicts": [],
        "normalized_records": [],
        "human_review_decisions": [
            {
                "review_id": "review_conflict_conf_001",
                "decision": "not_allowed",
                "reviewer_id": "tester",
            }
        ],
        "collection_trace": [],
    }
    result = node(state)
    queue = result.get("human_review_queue") or []
    summary = result.get("human_review_summary") or {}
    assert len(queue) == 1
    item = queue[0]
    assert item.get("status") == "invalid_decision"
    assert item.get("decision_applied") is False
    assert "invalid_decision" in (item.get("decision_warnings") or [])
    assert summary.get("invalid_decision_count", 0) >= 1


def test_studio_sanity_check_script_exists():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "check_studio_app.py"
    assert script.exists(), f"missing {script}"


def test_fixture_mode_final_package_contains_review_packets(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    items = package.get("human_review_items") or []
    assert items
    assert any(item.get("review_packet") for item in items)


# ---------------------------------------------------------------------------
# Step 13: hardened final data package + export utilities
# ---------------------------------------------------------------------------


def test_default_final_package_has_metadata_and_summaries(monkeypatch):
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    assert package
    assert package.get("package_metadata"), "package_metadata missing"
    assert package.get("workflow_summaries"), "workflow_summaries missing"
    assert package.get("data_dictionary"), "data_dictionary missing"
    assert package.get("provenance_manifest"), "provenance_manifest missing"
    finalization_summary = result.get("finalization_summary") or {}
    assert finalization_summary
    assert package.get("contains_synthetic_fixture_data") is False


def test_fixture_final_package_has_synthetic_notice(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    assert package.get("contains_synthetic_fixture_data") is True
    notice = package.get("synthetic_fixture_notice") or ""
    assert notice
    assert "not real public health data" in notice.lower()
    meta = package.get("package_metadata") or {}
    assert meta.get("contains_synthetic_fixture_data") is True


def test_final_package_trace_matches_state_trace_step13(monkeypatch):
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    state_trace = result.get("collection_trace") or []
    package = result.get("final_data_package") or {}
    package_trace = package.get("collection_trace") or []
    assert len(package_trace) == len(state_trace)
    assert state_trace
    assert state_trace[-1]["node_name"] == "final_data_package_builder"
    assert package_trace[-1]["node_name"] == "final_data_package_builder"


def test_final_package_provenance_manifest_counts_fixture(monkeypatch):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    prov = package.get("provenance_manifest") or {}
    assert prov.get("normalized_record_count", 0) > 0
    assert prov.get("linked_event_count", 0) > 0
    assert prov.get("conflict_count", 0) > 0
    assert prov.get("human_review_item_count", 0) > 0
    final_dataset = package.get("final_dataset") or []
    assert prov.get("records_with_source_url_count", 0) > 0
    # All synthesized fixture records should have URLs.
    assert prov.get("records_with_source_url_count", 0) == len(final_dataset)


def test_workflow_summaries_include_major_steps(monkeypatch):
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    summaries = package.get("workflow_summaries") or {}
    for key in (
        "task_intake_summary",
        "disease_intelligence_summary",
        "profile_schema_summary",
        "executable_source_plan_summary",
        "source_discovery_summary",
        "source_screening_summary",
        "content_fetch_summary",
        "evidence_chunking_summary",
        "structured_extraction_summary",
        "schema_validation_summary",
        "record_normalization_summary",
        "record_linking_summary",
        "cross_source_consistency_summary",
    ):
        assert key in summaries, f"workflow_summaries missing {key}"


def test_export_final_data_package_writes_files(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}

    from data_collection_workflow.export import export_final_data_package

    manifest = export_final_data_package(package, tmp_path)
    expected = [
        "final_package.json",
        "final_dataset.csv",
        "source_registry.json",
        "linked_events.json",
        "conflicts.json",
        "human_review_items.json",
        "collection_trace.json",
        "workflow_summaries.json",
        "package_metadata.json",
        "provenance_manifest.json",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), f"missing exported file {name}"
    files = manifest.get("files") or {}
    assert "final_package_json" in files
    assert "final_dataset_csv" in files
    section_counts = manifest.get("section_counts") or {}
    assert section_counts.get("final_dataset", 0) > 0


def test_unified_workflow_runner_exports_final_package():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_workflow.py"
    assert script.exists(), f"missing {script}"
    text = script.read_text(encoding="utf-8")
    assert "export_final_data_package" in text
    assert "write_evaluation_outputs" in text


# ---------------------------------------------------------------------------
# Step 14: optional LLM-based structured extraction
# ---------------------------------------------------------------------------


def _clear_llm_env(monkeypatch):
    for key in (
        "ENABLE_LLM_EXTRACTION",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_FALLBACK_TO_RULE_BASED",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_mode_llm_extraction_disabled(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("llm_enabled") is False
    assert summary.get("llm_call_count") == 0
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    assert metadata.get("llm_used") is False


def test_fixture_mode_llm_extraction_disabled_by_default(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    raw_records = result.get("raw_records") or []
    assert raw_records
    for rec in raw_records:
        assert not rec.get("llm_used")
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("extraction_mode") == "deterministic_rule_based"
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    assert metadata.get("llm_used") is False


def test_llm_settings_support_anthropic_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-claude-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1234")
    from data_collection_workflow import llm_clients

    settings = llm_clients.get_llm_settings()
    assert settings["provider"] == "anthropic"
    assert settings["model"] == "test-claude-model"
    assert settings["max_tokens"] == 1234


def test_llm_settings_support_openai_provider(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "test-openai-model")
    from data_collection_workflow import llm_clients

    settings = llm_clients.get_llm_settings()
    assert settings["provider"] == "openai"
    assert settings["model"] == "test-openai-model"


def test_llm_extraction_path_with_monkeypatched_client(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-claude-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from data_collection_workflow import llm_clients
    from data_collection_workflow.models import LLMExtractedRecord, LLMExtractionOutput

    def mock_extract(chunk, policy):
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    country="Country X",
                    date_reported="2023",
                    cases_unspecified=99,
                    deaths=2,
                    case_definition="unspecified",
                )
            ],
            chunk_is_relevant=True,
            extraction_notes="mocked",
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", mock_extract)

    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())

    raw_records = result.get("raw_records") or []
    assert len(raw_records) > 0
    assert any(r.get("llm_used") for r in raw_records)
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("llm_enabled") is True
    assert summary.get("llm_call_count", 0) > 0
    assert summary.get("llm_success_count", 0) > 0
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    assert metadata.get("llm_used") is True


def test_llm_extraction_fallback_on_error(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-claude-model")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULE_BASED", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from data_collection_workflow import llm_clients

    def mock_fail(chunk, policy):
        raise RuntimeError("mock LLM failure")

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", mock_fail)

    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    raw_records = result.get("raw_records") or []
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("llm_error_count", 0) > 0
    assert summary.get("llm_fallback_count", 0) > 0
    assert raw_records, "expected deterministic fallback records"
    assert any(not r.get("llm_used") for r in raw_records)
    assert result.get("final_data_package") is not None


def test_llm_extraction_no_fallback_on_error(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-claude-model")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULE_BASED", "false")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    from data_collection_workflow import llm_clients

    def mock_fail(chunk, policy):
        raise RuntimeError("mock LLM failure (no fallback)")

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", mock_fail)

    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("structured_extraction_summary") or {}
    assert summary.get("llm_error_count", 0) > 0
    assert summary.get("llm_fallback_count", 0) == 0
    assert result.get("final_data_package") is not None


def test_final_package_workflow_node_count_matches_trace(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    trace = package.get("collection_trace") or []
    assert metadata.get("workflow_node_count") == len(trace)


def test_llm_extraction_summary_in_workflow_summaries(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    package = result.get("final_data_package") or {}
    summaries = package.get("workflow_summaries") or {}
    assert "llm_extraction_summary" in summaries


def test_configured_workflow_script_contains_llm_controls():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_workflow.py"
    assert script.exists(), f"missing {script}"
    text = script.read_text(encoding="utf-8")
    assert "--disable-all-llm" in text
    assert "ENABLE_LLM_SOURCE_PLANNING" in text
    assert "ENABLE_LLM_SOURCE_CRITIC" in text
    assert "ENABLE_LLM_EXTRACTION" in text


# ---------------------------------------------------------------------------
# Controlled live-source workflow env controls
# ---------------------------------------------------------------------------


def test_source_id_allowlist_limits_fetch_requests_offline(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv(
        "SOURCE_ID_ALLOWLIST",
        "src_cdc_reported_cases,src_who_hantavirus_fact_sheet",
    )
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    requests = result.get("content_fetch_requests") or []
    summary = result.get("content_fetch_summary") or {}
    request_ids = {r.get("source_id") for r in requests}
    assert request_ids == {"src_cdc_reported_cases", "src_who_hantavirus_fact_sheet"}
    assert summary.get("source_id_allowlist_enabled") is True
    assert sorted(summary.get("source_id_allowlist") or []) == [
        "src_cdc_reported_cases",
        "src_who_hantavirus_fact_sheet",
    ]
    assert summary.get("skipped_not_in_allowlist_count", 0) > 0


def test_no_allowlist_default_behavior_unchanged(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.delenv("USE_FIXTURE_DOCUMENTS", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_FETCH", raising=False)
    monkeypatch.delenv("SOURCE_ID_ALLOWLIST", raising=False)
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("content_fetch_summary") or {}
    assert summary.get("source_id_allowlist_enabled") is False
    assert summary.get("fetch_request_count") == 10


def test_legacy_llm_max_chunks_maps_to_soft_checkpoint(monkeypatch):
    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("USE_FIXTURE_DOCUMENTS", "true")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "test-claude-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MAX_CHUNKS", "1")

    from data_collection_workflow import llm_clients
    from data_collection_workflow.models import LLMExtractedRecord, LLMExtractionOutput

    def mock_extract(chunk, policy):
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    country="Country X",
                    date_reported="2023",
                    cases_unspecified=11,
                    deaths=2,
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", mock_extract)

    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    summary = result.get("structured_extraction_summary") or {}
    ledger = summary.get("extraction_budget_ledger") or {}
    assert summary.get("deprecated_llm_max_chunks") is True
    assert summary.get("llm_max_chunks") == 1
    assert ledger.get("soft_checkpoint_calls") == 1
    assert 0 < ledger.get("actual_model_call_count", 0) <= ledger.get(
        "safety_max_calls", 0
    )


def test_studio_launcher_script_contains_workflow_controls():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "start_studio.py"
    assert script.exists(), f"missing {script}"
    text = script.read_text(encoding="utf-8")
    assert "--disable-live-fetch" in text
    assert "--enable-all-llm" in text
    assert "--print-config-only" in text


# ---------------------------------------------------------------------------
# Step 16: real-world semantic guardrails
# ---------------------------------------------------------------------------


def test_extraction_guardrail_standardizes_disease():
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails

    out = _apply_extraction_semantic_guardrails(
        {"disease": "hantavirus infection", "country": "United States"},
        {"text": "irrelevant"},
    )
    assert out["disease"] == "Hantavirus disease"
    assert "standardized_disease_name" in (out.get("semantic_warnings") or [])


def test_extraction_guardrail_removes_generic_virus_or_syndrome():
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails

    out = _apply_extraction_semantic_guardrails(
        {
            "disease": "Hantavirus disease",
            "virus_or_syndrome": "hantavirus disease",
            "country": "Chile",
        },
        {"text": "irrelevant"},
    )
    assert out.get("virus_or_syndrome") is None
    assert "removed_generic_virus_or_syndrome" in (out.get("semantic_warnings") or [])


def test_region_geography_normalization_eu_eea():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="EU/EEA (28 countries)",
        subnational_location=None,
        date_reported="2023",
        cases_unspecified=1885,
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    assert normalized.get("country") is None
    assert normalized.get("geographic_scope") == "EU/EEA"
    assert normalized.get("geographic_scope_type") in {"region", "multi_country"}
    warnings = normalized.get("normalization_warnings") or []
    assert "regional_geographic_scope_not_country" in warnings


def test_record_linking_separates_different_statistical_count_types():
    node = _linking_node()
    common = dict(
        disease="Hantavirus disease",
        virus_or_syndrome=None,
        country="United States of America",
        subnational_location=None,
        date_reported="2023",
    )
    rec_a = _baseline_normalized_record(
        record_id="rec_cumulative",
        cases_unspecified=890,
        statistical_count_type="cumulative",
        source_id="src_1",
        **common,
    )
    rec_b = _baseline_normalized_record(
        record_id="rec_annual",
        cases_unspecified=31,
        statistical_count_type="annual",
        source_id="src_2",
        source_url="https://example.org/source-2",
        **common,
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    assert len(linked) == 2
    for event in linked:
        assert event.get("record_count") == 1


def test_cross_source_consistency_skips_numeric_compare_for_different_count_types():
    node = _consistency_node()
    common_kwargs = dict(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="Country X",
        subnational_location=None,
        date_reported="2023",
        date_anchor="2023",
        deaths=2,
        case_definition="unspecified",
    )
    rec_a = _consistency_record(
        record_id="rec_a", source_id="src_1", cases_unspecified=890,
        **{**common_kwargs, "statistical_count_type": "cumulative"},
    )
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=31,
        **{**common_kwargs, "statistical_count_type": "annual"},
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    summary = result.get("cross_source_consistency_summary") or {}
    # No numeric conflict on cases_unspecified because count_type differs.
    numeric_case_conflicts = [
        c for c in conflicts if c.get("field") == "cases_unspecified"
    ]
    assert numeric_case_conflicts == []
    assert summary.get("numeric_comparison_skipped_count", 0) > 0
    skip_reasons = summary.get("numeric_comparison_skip_reason_counts") or {}
    assert "different_statistical_count_type" in skip_reasons


def test_llm_new_fields_model_validation():
    from data_collection_workflow.models import HantavirusRecord, LLMExtractedRecord

    llm_record = LLMExtractedRecord(
        disease="Hantavirus disease",
        statistical_count_type="cumulative",
        reporting_period="through December 2020",
        as_of_date="2020-12-31",
        aggregation_level="national",
        geographic_scope="United States",
        geographic_scope_type="country",
        population_scope="human",
        source_section="Reported Cases",
    )
    assert llm_record.statistical_count_type == "cumulative"
    record = HantavirusRecord(
        record_id="rec_test_001",
        disease="Hantavirus disease",
        statistical_count_type="annual",
        reporting_period="2023",
        as_of_date="2023-12-31",
        aggregation_level="national",
        geographic_scope="United States of America",
        geographic_scope_type="country",
        population_scope="human",
        source_section="Surveillance Update",
        semantic_warnings=["standardized_disease_name"],
    )
    assert record.statistical_count_type == "annual"
    assert "standardized_disease_name" in record.semantic_warnings


def test_cross_source_consistency_summary_counts():
    node = _consistency_node()
    rec_a = _consistency_record(record_id="rec_a", source_id="src_1", cases_unspecified=12)
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=30,
    )
    rec_c = _consistency_record(
        record_id="rec_c",
        source_id="src_3",
        source_url="https://example.org/source-3",
        linked_event_id="event_002",
        cases_unspecified=7,
    )
    events = [
        _consistency_event(["rec_a", "rec_b"], linked_event_id="event_001"),
        _consistency_event(
            ["rec_c"],
            linked_event_id="event_002",
            record_count=1,
            linking_status="single_record_event",
        ),
    ]
    state = {
        "normalized_records": [rec_a, rec_b, rec_c],
        "linked_events": events,
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    summary = result.get("cross_source_consistency_summary") or {}
    assert summary.get("input_linked_event_count") == 2
    assert summary.get("comparable_event_count") == 1
    assert summary.get("single_record_event_count") == 1
    assert summary.get("conflict_count") == 1
    assert summary.get("events_with_conflicts_count") == 1
    assert summary.get("severity_counts"), "severity_counts should be non-empty"
    status_counts = summary.get("event_consistency_status_counts") or {}
    assert "needs_review" in status_counts or "conflict_detected" in status_counts


def test_final_package_trace_matches_state_trace():
    graph = build_graph()
    result = graph.invoke(_sanity_initial_state())
    state_trace = result.get("collection_trace") or []
    package = result.get("final_data_package") or {}
    package_trace = package.get("collection_trace") or []
    assert len(package_trace) == len(state_trace), (
        f"final_data_package trace ({len(package_trace)}) != state trace ({len(state_trace)})"
    )
    assert state_trace, "expected non-empty trace"
    assert state_trace[-1]["node_name"] == "final_data_package_builder"
    assert package_trace[-1]["node_name"] == "final_data_package_builder"


# ---------------------------------------------------------------------------
# Step 16.1: date-anchor fallback + regional-scope geography
# ---------------------------------------------------------------------------


def test_date_anchor_uses_reporting_period_when_date_reported_missing():
    node = _linking_node()
    rec = _baseline_normalized_record(
        record_id="rec_step161_anchor",
        country="United States of America",
        date_reported=None,
        event_start_date=None,
        event_end_date=None,
        reporting_period="December 2020",
        cases_unspecified=5,
    )
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized_out = (result.get("normalized_records") or [])[0]
    linked = result.get("linked_events") or []
    assert normalized_out.get("date_anchor") == "2020-12"
    assert normalized_out.get("date_anchor_field") == "reporting_period"
    warnings = normalized_out.get("record_linking_warnings") or []
    assert "missing_date_anchor" not in warnings
    assert len(linked) == 1
    assert linked[0].get("requires_human_review") is False


def test_reporting_period_normalization_variants():
    from data_collection_workflow.nodes.linking_validation import _normalize_date_anchor_value

    cases = {
        "December 2020": "2020-12",
        "through December 2020": "2020-12",
        "as of December 2020": "2020-12",
        "reported through December 2020": "2020-12",
        "March 2023": "2023-03",
        "through 2023": "2023",
        "2025-03-07": "2025-03-07",
        "2023": "2023",
        "2024-06": "2024-06",
    }
    for raw, expected in cases.items():
        assert _normalize_date_anchor_value(raw) == expected, (
            f"normalize {raw!r} -> {expected!r}"
        )


def test_region_geographic_scope_not_missing_country_after_normalization():
    node = _normalization_node()
    record = _baseline_validated_record(
        record_id="rec_step161_norm",
        country="EU/EEA",
        subnational_location=None,
        date_reported="2023",
        cases_unspecified=1885,
    )
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    queue = result.get("human_review_queue") or []
    warnings = normalized.get("normalization_warnings") or []
    assert normalized.get("country") is None
    assert normalized.get("geographic_scope") == "EU/EEA"
    assert normalized.get("geographic_scope_type") in {"region", "multi_country"}
    assert "regional_or_aggregate_geographic_scope" in warnings
    assert "missing_country_after_normalization" not in warnings
    assert normalized.get("requires_human_review") is False
    review_items = [
        item for item in queue if item.get("item_type") == "record_normalization"
    ]
    assert review_items == []


def test_record_linking_uses_geographic_scope_when_country_missing():
    node = _linking_node()
    rec = _baseline_normalized_record(
        record_id="rec_step161_link",
        country=None,
        subnational_location=None,
        date_reported="2023",
        cases_unspecified=1885,
        normalization_status="normalized_with_warnings",
    )
    # Step 16.1 path: regional scope present, no country.
    rec["geographic_scope"] = "EU/EEA"
    rec["geographic_scope_type"] = "multi_country"
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    queue = result.get("human_review_queue") or []
    assert len(linked) == 1
    event = linked[0]
    warnings = event.get("linking_warnings") or []
    assert "regional_or_aggregate_geographic_scope" in warnings
    assert "missing_country_for_case_data" not in warnings
    assert event.get("requires_human_review") is False
    review_items = [item for item in queue if item.get("item_type") == "record_linking"]
    assert review_items == []


# ---------------------------------------------------------------------------
# Step 16.1.1: enum canonicalization for LLM-emitted variants
# ---------------------------------------------------------------------------


def test_extraction_canonicalizes_geographic_scope_type_multi_country():
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails

    out = _apply_extraction_semantic_guardrails(
        {
            "disease": "Hantavirus disease",
            "country": None,
            "geographic_scope": "EU/EEA",
            "geographic_scope_type": "multi-country",
            "cases_unspecified": 1885,
            "source_id": "src_test",
            "source_url": "https://example.org/eu",
            "source_type": "official_public_health_agency",
            "evidence_quote": "EU/EEA reported 1885 cases.",
        },
        {"text": "EU/EEA reported 1885 cases."},
    )
    assert out["geographic_scope_type"] == "multi_country"
    assert "canonicalized_geographic_scope_type" in (out.get("semantic_warnings") or [])


def test_extraction_canonicalizes_geographic_scope_type_national():
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails

    out = _apply_extraction_semantic_guardrails(
        {
            "disease": "Hantavirus disease",
            "country": "United States",
            "geographic_scope": "United States",
            "geographic_scope_type": "national",
            "cases_unspecified": 12,
            "source_id": "src_test",
            "source_url": "https://example.org/us",
            "source_type": "official_public_health_agency",
            "evidence_quote": "12 cases in the United States.",
        },
        {"text": "12 cases were reported in the United States."},
    )
    assert out["geographic_scope_type"] == "country"
    assert "canonicalized_geographic_scope_type" in (out.get("semantic_warnings") or [])


def test_normalization_multi_country_variant_does_not_trigger_missing_country_review():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="EU/EEA (28 countries)",
        subnational_location=None,
        date_reported="2025-03-07",
        cases_unspecified=1885,
    )
    # Simulate the LLM-emitted hyphenated variant entering normalization
    # directly without going through the extraction guardrails first.
    record["geographic_scope"] = "EU/EEA"
    record["geographic_scope_type"] = "multi-country"
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    queue = result.get("human_review_queue") or []
    warnings = normalized.get("normalization_warnings") or []
    assert normalized.get("country") is None
    assert normalized.get("geographic_scope") == "EU/EEA"
    assert normalized.get("geographic_scope_type") == "multi_country"
    assert "missing_country_after_normalization" not in warnings
    assert "regional_or_aggregate_geographic_scope" in warnings
    assert normalized.get("requires_human_review") is False
    review_items = [
        item for item in queue if item.get("item_type") == "record_normalization"
    ]
    assert review_items == []


def test_linking_multi_country_variant_not_missing_country():
    node = _linking_node()
    rec = _baseline_normalized_record(
        record_id="rec_step1611_link",
        country=None,
        subnational_location=None,
        date_reported="2025-03-07",
        cases_unspecified=1885,
        normalization_status="normalized_with_warnings",
    )
    rec["geographic_scope"] = "EU/EEA"
    rec["geographic_scope_type"] = "multi-country"
    state = {
        "normalized_records": [rec],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    linked = result.get("linked_events") or []
    normalized_out = (result.get("normalized_records") or [])[0]
    assert len(linked) == 1
    event = linked[0]
    warnings = normalized_out.get("record_linking_warnings") or []
    assert "missing_country_for_case_data" not in warnings
    assert "regional_or_aggregate_geographic_scope" in warnings
    assert event.get("requires_human_review") is False
    event_key = event.get("event_key") or ""
    assert "multi_country" in event_key or "region" in event_key


def test_canonicalizes_statistical_count_type_variants():
    node = _normalization_node()
    record = _baseline_validated_record(
        country="United States",
        date_reported="2023",
        cases_unspecified=10,
    )
    record["statistical_count_type"] = "newly-reported"
    state = {
        "validated_records": [record],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    normalized = (result.get("normalized_records") or [])[0]
    assert normalized.get("statistical_count_type") == "newly_reported"
    actions = normalized.get("normalization_actions") or []
    assert "canonicalized_statistical_count_type" in actions


def test_llm_policy_mentions_canonical_scope_type_values():
    from data_collection_workflow.config import load_llm_structured_extraction_policy

    policy = load_llm_structured_extraction_policy()
    rules = policy.get("required_output_rules") or []
    joined = " ".join(rules)
    assert "multi_country" in joined
    assert "newly_reported" in joined


def test_cross_source_consistency_skips_numeric_compare_for_different_reporting_periods():
    node = _consistency_node()
    common_kwargs = dict(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="Country X",
        subnational_location=None,
        date_reported="2023",
        date_anchor="2023",
        deaths=2,
        case_definition="unspecified",
        statistical_count_type="cumulative",
    )
    rec_a = _consistency_record(
        record_id="rec_a",
        source_id="src_1",
        cases_unspecified=890,
        **{**common_kwargs, "reporting_period": "through December 2022"},
    )
    rec_b = _consistency_record(
        record_id="rec_b",
        source_id="src_2",
        source_url="https://example.org/source-2",
        cases_unspecified=950,
        **{**common_kwargs, "reporting_period": "through December 2023"},
    )
    state = {
        "normalized_records": [rec_a, rec_b],
        "linked_events": [_consistency_event(["rec_a", "rec_b"])],
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    result = node(state)
    conflicts = result.get("conflicts") or []
    summary = result.get("cross_source_consistency_summary") or {}
    numeric_case_conflicts = [
        c for c in conflicts if c.get("field") == "cases_unspecified"
    ]
    assert numeric_case_conflicts == []
    assert summary.get("numeric_comparison_skipped_count", 0) > 0
    skip_reasons = summary.get("numeric_comparison_skip_reason_counts") or {}
    assert "different_reporting_period" in skip_reasons
