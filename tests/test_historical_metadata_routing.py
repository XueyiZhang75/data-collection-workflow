"""Index metadata may be listed, but must never become a content fetch request."""
from __future__ import annotations

from copy import deepcopy
import importlib

import pytest

from data_collection_workflow.config import load_content_fetch_policy
from data_collection_workflow.models import ContentFetchPolicy
from data_collection_workflow.nodes.content_processing import (
    _build_fetch_requests,
    _fetch_config_from_env,
)
from data_collection_workflow.nodes.source_screening import (
    source_critic_and_uncertainty_routing,
    source_screening,
)
from data_collection_workflow.source_identity import apply_source_identity_routing_guardrails
from test_evidence_resource_discovery import _env, _node, _source
from test_source_critic_live_integration import _clear_critic_env, _entry, _state


def _historical(entry: dict) -> dict:
    row = deepcopy(entry)
    original = row["canonical_url"]
    replay = f"https://web.archive.org/web/20250501000000/{original}"
    row.update(
        canonical_url=replay,
        url=replay,
        discovery_method="historical_archive_index",
        historical_snapshot={
            "original_url": original,
            "archive_url": replay,
            "capture_timestamp": "20250501000000",
            "parent_source_id": "original",
            "verification_status": "index_only",
        },
        blocked_from_fetch=True,
        blocked_from_fetch_reason="historical_index_only",
        ready_for_content_fetch=False,
    )
    return row


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    _env(monkeypatch)
    _clear_critic_env(monkeypatch)
    for key in (
        "ENABLE_LLM_SOURCE_CRITIC", "ENABLE_LLM_SOURCE_IDENTITY",
        "ENABLE_LLM_SOURCE_CREDIBILITY",
    ):
        monkeypatch.setenv(key, "false")


def _assert_metadata_only(row, metadata):
    assert row["historical_snapshot"] == metadata
    assert row["blocked_from_fetch"] is True
    assert row["blocked_from_fetch_reason"] == "historical_index_only"
    assert row["ready_for_content_fetch"] is False


@pytest.mark.parametrize("pipeline", ["standard", "evidence"])
def test_screening_restores_metadata_only_routing_before_serialization(monkeypatch, pipeline):
    monkeypatch.setenv("PIPELINE_MODE", pipeline)
    entry = _historical(_entry("archive"))
    entry.update(blocked_from_fetch=False, blocked_from_fetch_reason=None,
                 ready_for_content_fetch=True)
    result = source_screening(_state([entry]))
    _assert_metadata_only(result["source_registry"][0], entry["historical_snapshot"])


def test_critic_failure_and_failure_cap_cannot_release_historical_candidates(monkeypatch):
    module = importlib.import_module("data_collection_workflow.nodes.source_screening")
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_SOURCES", "3")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_FAILURES", "1")

    def timeout(*args, **kwargs):
        raise TimeoutError("Offline critic timeout fixture")

    monkeypatch.setattr(module, "assess_source_with_llm", timeout)
    entries = [_historical(_entry(f"archive_{i}", search_rank=i)) for i in (1, 2)]
    state = _state(entries)
    state.update(source_screening(state))
    result = source_critic_and_uncertainty_routing(state)
    assert result["source_critic_summary"]["failed_source_count"] == 1
    assert result["source_critic_summary"]["fail_open_skipped_source_count"] == 1
    for actual, expected in zip(result["source_registry"], entries):
        _assert_metadata_only(actual, expected["historical_snapshot"])
    state.update(result)
    policy = ContentFetchPolicy(**load_content_fetch_policy())
    requests, _, _ = _build_fetch_requests(state, policy, True, _fetch_config_from_env(policy))
    assert requests == []


@pytest.mark.parametrize("override", [
    {"source_identity_llm_used": True, "recommended_source_role": "excluded"},
    {"recommended_fetch_use": "do_not_fetch"},
])
def test_identity_must_fetch_override_is_reblocked_by_final_routing(override):
    entry = _historical(_entry("archive"))
    entry.update(must_fetch=True, coverage_requirement_ids=["fixture_requirement"], **override)
    overridden = apply_source_identity_routing_guardrails(entry)
    assert overridden["blocked_from_fetch"] is False
    assert overridden["ready_for_content_fetch"] is True
    state = _state([overridden])
    state.update(source_screening(state))
    state.update(source_critic_and_uncertainty_routing(state))
    _assert_metadata_only(state["source_registry"][0], entry["historical_snapshot"])
    policy = ContentFetchPolicy(**load_content_fetch_policy())
    requests, _, _ = _build_fetch_requests(state, policy, True, _fetch_config_from_env(policy))
    assert requests == []


@pytest.mark.parametrize("pipeline", ["standard", "evidence"])
def test_fetch_request_builder_enforces_metadata_status_after_block_is_cleared(monkeypatch, pipeline):
    monkeypatch.setenv("PIPELINE_MODE", pipeline)
    historical = _historical(_source("https://example.invalid/archive", 1))
    historical.update(blocked_from_fetch=False, blocked_from_fetch_reason=None,
                      ready_for_content_fetch=True, must_fetch=True,
                      coverage_requirement_ids=["fixture_requirement"])
    ordinary = _source("https://example.invalid/current", 2)
    state = {
        "structured_task": {"disease": "Example fever", "location": "Canada",
                            "start_date": "2025-01-01", "end_date": "2025-12-31"},
        "source_registry": [historical, ordinary],
    }
    policy = ContentFetchPolicy(**load_content_fetch_policy())
    requests, skipped, manifest = _build_fetch_requests(
        state, policy, True, _fetch_config_from_env(policy))
    assert [request.source_id for request in requests] == ["s2"]
    assert skipped["historical_index_only"] == 1
    archived = next(row for row in manifest if row["source_id"] == "s1")
    assert archived["selected_for_fetch"] is False
    assert archived["skip_reason"] == "historical_index_only"


def test_live_acquisition_node_never_opens_index_only_replay(monkeypatch, tmp_path):
    entry = _historical(_source("https://example.invalid/report"))
    entry.update(blocked_from_fetch=False, blocked_from_fetch_reason=None,
                 ready_for_content_fetch=True)
    pages = {entry["canonical_url"]: ("text/html", b"<p>Offline transport fixture.</p>")}
    result, visits, ledger = _node(monkeypatch, tmp_path, pages, sources=[entry])
    assert result["content_fetch_requests"] == []
    assert result["documents"] == []
    assert visits == []
    assert ledger["used"].get("http_requests", 0) == 0
    assert ledger["used"].get("source_targets", 0) == 0


@pytest.mark.parametrize("stale_period", [False, True])
def test_inherited_parent_title_and_capture_do_not_verify_archived_task_fit(stale_period):
    from data_collection_workflow.historical_source_discovery import _candidate
    from data_collection_workflow.nodes.source_discovery import source_dedup_and_registry

    original = "https://www.canada.ca/dengue-surveillance"
    candidate = _candidate(
        {"source_id": "parent", "title": "Dengue cases in Canada in 2020"},
        {"original": original, "timestamp": "20200601000000",
         "captured_at": "2020-06-01T00:00:00Z", "statuscode": "200",
         "mimetype": "text/html", "digest": "INDEX_DIGEST"},
        "https://web.archive.org/cdx/search/cdx?fixture=1",
        "2026-10-06T00:00:00Z",
    )
    state = {"structured_task": {"disease": "dengue", "location": "Canada",
                                 "start_date": "2020-01-01", "end_date": "2020-12-31"},
             "source_candidates": [candidate], "collection_trace": []}
    state.update(source_dedup_and_registry(state))
    if stale_period:
        state["source_registry"][0].update(
            published_date="2020-06-01", reporting_period_start="2020-01-01",
            reporting_period_end="2020-12-31", reporting_period_label="2020",
            period_basis="calendar_year")
    state.update(source_screening(state))
    summary = state["source_screening_summary"]
    assert summary["target_verification_status_counts"].get("verified_target", 0) == 0
    assert summary["triage_role_counts"].get("verified_target_collection", 0) == 0
    triage = state["source_triage_results"][0]
    assert triage["target_verification_status"] == "unverified_candidate"
    assert triage["triage_role"] == "task_record_collection_candidate"

    for result in (state, source_critic_and_uncertainty_routing(state)):
        row = result["source_registry"][0]
        _assert_metadata_only(row, candidate["historical_snapshot"])
        assert row["target_verification_status"] == "unverified_candidate"
        assert row["target_fit_status"] == "task_record_collection_candidate"
        assert row["triage_role"] == "task_record_collection_candidate"
        assert [row[key] for key in ("disease_fit", "geography_fit", "date_fit")] == [
            "unknown", "unknown", "unknown"]
        assert row["published_date"] is None
        assert row["reporting_period_start"] is None
        assert row["reporting_period_end"] is None
        assert row["reporting_period_label"] is None
        assert row["period_basis"] is None
