"""Historical candidates retain URL identity and metadata through discovery."""
import importlib

import pytest

from data_collection_workflow.models import SourceCandidate, SourceRegistryEntry
from data_collection_workflow.session_runtime import RunContext

mod = importlib.import_module("data_collection_workflow.nodes.source_discovery")


def _candidate(stamp="20250501000000"):
    original = "https://health.example/Measles/?view=weekly%2Fcounts&lang=en"
    url = f"https://web.archive.org/web/{stamp}/{original}"
    return {
        "source_id": stamp, "url": url, "canonical_url": url,
        "source_type": "archived_web_page", "title": "Archived version candidate",
        "discovery_method": "historical_archive_index",
        "blocked_from_fetch": True, "blocked_from_fetch_reason": "historical_index_only",
        "historical_snapshot": {
            "original_url": original, "archive_url": url, "capture_timestamp": stamp,
            "verification_status": "index_only", "parent_source_id": "original",
        },
    }


def test_replay_identity_and_provenance_survive_registry_roundtrips():
    candidates = [_candidate(), _candidate("20250601000000")]
    typed = [SourceCandidate(**row).model_dump() for row in candidates]
    registry = mod.source_dedup_and_registry({"source_candidates": typed})["source_registry"]
    assert len(registry) == 2
    for expected, row in zip(candidates, registry):
        restored = SourceRegistryEntry.model_validate_json(
            SourceRegistryEntry(**row).model_dump_json()).model_dump()
        assert restored["canonical_url"] == expected["url"]
        assert restored["historical_snapshot"] == expected["historical_snapshot"]
        assert restored["published_date"] is None
        assert restored["blocked_from_fetch"] is True
        assert restored["blocked_from_fetch_reason"] == "historical_index_only"
        assert restored["ready_for_content_fetch"] is False


def test_replay_preserves_case_insensitive_embedded_scheme():
    url = "https://web.archive.org/web/20250101000000/HTTPS://WWW.CDC.GOV/Measles/"
    assert mod.canonicalize_url(url) == url


@pytest.mark.parametrize("historical_first", [False, True])
def test_search_duplicate_cannot_discard_index_provenance(historical_first):
    archive = _candidate()
    search = {**archive, "source_id": "search", "historical_snapshot": {},
              "blocked_from_fetch": False, "discovery_method": "live_search_result",
              "source_type": "official_public_health_agency", "published_date": "2025-05-01"}
    candidates = [archive, search] if historical_first else [search, archive]
    registry = mod.source_dedup_and_registry({"source_candidates": candidates})["source_registry"]
    assert len(registry) == 1
    row = registry[0]
    assert row["source_id"] == candidates[0]["source_id"]
    assert row["historical_snapshot"]["verification_status"] == "index_only"
    assert row["blocked_from_fetch"] is True
    assert row["source_type"] == "archived_web_page"
    assert row["published_date"] is None
    assert any(item["source_id"] == "search" for item in row["historical_snapshot"]["other_discoveries"])


def _discover(monkeypatch, tmp_path, *, mode="live", pipeline="evidence", enabled=True,
              queries=20, results=100, spent=None):
    monkeypatch.setenv("PIPELINE_MODE", pipeline)
    monkeypatch.setenv("ENABLE_LIVE_SEARCH", "true")
    settings = mod.SourceSearchSettings(
        mode=mode, max_queries=queries, iterative_enabled=True,
        iterative_max_total_queries=queries, max_total_results=results,
        iterative_max_total_results=results)
    monkeypatch.setattr(mod, "_source_search_settings_from_env", lambda: settings)
    monkeypatch.setattr(mod, "_effective_discovery_settings", lambda value: value)
    monkeypatch.setattr(mod, "build_official_coverage_candidates", lambda state: [])
    ordinary = SourceCandidate(source_id="original", url="https://health.example/measles")
    observed = {}
    def search(state, limits):
        observed["search_settings"] = limits
        return [ordinary], [], {
            "selected_query_count": limits.iterative_max_total_queries if spent is None else spent,
            "executed_query_count": limits.iterative_max_total_queries if spent is None else spent,
        }, {}
    monkeypatch.setattr(mod, "_execute_source_search", search)
    def historical(state, candidates, **limits):
        observed["historical_limits"] = limits
        assert candidates[0]["source_id"] == "original"
        return [_candidate()], [{"kind": "index_query"}], {"status": "completed", "index_query_count": 1}
    monkeypatch.setattr(mod, "discover_historical_versions", historical)
    runtime = RunContext(tmp_path / "session", {
        "universal": {"historical_discovery": {"enabled": enabled}}})
    with runtime.activate():
        result = mod.source_discovery({"structured_task": {
            "disease": "measles", "location": "Canada",
            "start_date": "2025-01-01", "end_date": "2025-12-31"}})
    return observed, result


def test_live_hook_reserves_existing_capacity_and_exports_candidate(monkeypatch, tmp_path):
    observed, result = _discover(monkeypatch, tmp_path)
    limits = observed["historical_limits"]
    assert 0 < limits["max_origins"] <= 4
    assert observed["search_settings"].iterative_max_total_queries + limits["max_origins"] <= 20
    assert observed["search_settings"].max_total_results < 100
    assert limits["max_results"] <= 99
    assert len(result["source_candidates"]) == 2
    assert result["source_search_results"] == [{"kind": "index_query"}]
    assert result["source_discovery_summary"]["historical_discovery"]["status"] == "completed"
    assert result["source_search_execution_summary"]["total_candidate_count"] == 2


@pytest.mark.parametrize("kwargs", [
    {"mode": "fixture"}, {"pipeline": "standard"}, {"enabled": False},
    {"queries": 0}, {"results": 0},
])
def test_ineligible_modes_or_zero_capacity_do_not_dispatch(monkeypatch, tmp_path, kwargs):
    observed, result = _discover(monkeypatch, tmp_path, **kwargs)
    assert "historical_limits" not in observed
    assert len(result["source_candidates"]) == 1


def test_existing_search_overspend_never_gets_extra_stage_budget(monkeypatch, tmp_path):
    observed, _ = _discover(monkeypatch, tmp_path, spent=20)
    assert "historical_limits" not in observed
