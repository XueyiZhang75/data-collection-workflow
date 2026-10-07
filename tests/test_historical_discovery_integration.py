"""Historical candidates retain URL identity and metadata through discovery."""
import importlib
import json

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


@pytest.mark.parametrize("authority_retry", [False, True], ids=["without_authority_retry", "with_authority_retry"])
@pytest.mark.parametrize(
    "result_limit,results_per_query,captures_per_origin,ordinary_queries,ordinary_results,archive_queries,archive_results",
    [(100, 1, 1, 16, 16, 4, 4), (20, 4, 2, 4, 15, 3, 5)],
    ids=["query_reservation", "result_reservation"],
)
def test_adaptive_real_iterative_search_preserves_historical_reservation(
    monkeypatch, tmp_path, result_limit, results_per_query, captures_per_origin,
    ordinary_queries, ordinary_results, archive_queries, archive_results, authority_retry,
):
    """Adaptive normalization must not restore caps already reserved by the caller."""
    from data_collection_workflow.agents import iterative_source_discovery_agent as agent
    from data_collection_workflow import historical_source_discovery as historical

    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LIVE_SEARCH", "true")
    settings = mod.SourceSearchSettings(
        mode="live", max_queries=20, max_results_per_query=results_per_query,
        max_total_results=result_limit, iterative_enabled=True,
        iterative_max_iterations=20, iterative_max_queries_per_iteration=4,
        iterative_max_total_queries=20, iterative_max_total_results=result_limit,
        authority_gap_retry_enabled=authority_retry,
    )
    monkeypatch.setattr(mod, "_source_search_settings_from_env", lambda: settings)
    queries = [{"query_id": f"query_{index}",
                "query": f"measles United States surveillance report data {index}",
                "provider_channel": "web_search", "source_type": "official_public_health_agency",
                "role_hint": "collection", "priority": 1}
               for index in range(32)]
    monkeypatch.setattr(agent, "plan_initial_search_iteration", lambda **kwargs: {"query_batch": queries})
    monkeypatch.setattr(agent, "refine_search_iteration", lambda **kwargs: {
        "decision": "continue_search", "decision_reason": "Explore remaining task-grounded reports",
        "next_query_batch": queries,
    })
    provider_queries, index_queries = [], []

    class NovelResultProvider:
        def search(self, query, *, max_results, timeout_seconds):
            provider_queries.append(query["query"])
            number = len(provider_queries)
            return {"provider": "tavily", "raw_result_count": results_per_query,
                    "results": [{"title": "Measles surveillance report United States",
                                 "snippet": "Measles surveillance case counts in the United States",
                                 "url": f"https://www.cdc.gov/measles/surveillance/{number}-{rank}.html",
                                 "rank": rank}
                                for rank in range(1, results_per_query + 1)]}

    monkeypatch.setattr(mod, "_provider_for_settings", lambda limits: NovelResultProvider())

    class IndexResponse:
        status_code = 200
        headers = {}

        def __init__(self, original):
            rows = [["timestamp", "original", "statuscode", "mimetype", "digest"]]
            rows.extend([f"20250{month}01000000", original, "200", "text/html", "CDX-DIGEST"]
                        for month in range(1, captures_per_origin + 1))
            self.body = json.dumps(rows).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            for index in range(0, len(self.body), chunk_size):
                yield self.body[index:index + chunk_size]

    def index_get(url, *, params, **kwargs):
        assert url == "https://web.archive.org/cdx/search/cdx"
        index_queries.append(dict(params))
        return IndexResponse(dict(params)["url"])

    monkeypatch.setattr(historical.requests, "get", index_get)
    ctx = RunContext(tmp_path / "adaptive", {
        "pipeline_mode": "evidence",
        "universal": {"budget_policy": {"version": 2, "mode": "adaptive"},
                      "budget_limits": {"search": 20, "search_results": result_limit},
                      "historical_discovery": {"enabled": True}},
    })
    state = {"structured_task": {"disease": "measles", "location": "United States",
                                 "start_date": "2025-01-01", "end_date": "2025-12-31"},
             "agentic_source_plan": {"planned_queries": queries}, "search_query_inventory": queries}
    with ctx.activate():
        result = mod.source_discovery(state)

    candidates = result["source_candidates"]
    ordinary = [row for row in candidates if row["discovery_method"] == "live_search_result"]
    archived = [row for row in candidates if row["discovery_method"] == "historical_archive_index"]
    summary = result["source_discovery_summary"]["historical_discovery"]
    budget = ctx.ledger.snapshot()
    assert len(provider_queries) == ordinary_queries, summary
    assert len(set(provider_queries)) == ordinary_queries
    assert len(ordinary) == ordinary_results
    assert summary["reserved_query_count"] == 4
    assert summary["reserved_result_count"] == result_limit // 4
    assert len(index_queries) == archive_queries and summary["index_query_count"] == archive_queries
    assert len(archived) == archive_results
    assert budget["used"]["search"] == ordinary_queries + archive_queries <= 20
    assert budget["used"]["search_results"] == ordinary_results + archive_results <= result_limit
    assert budget["limits"]["search"] == 20 and budget["limits"]["search_results"] == result_limit
    assert set(budget["used"]) == {"search", "search_results"}
