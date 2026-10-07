"""Media retrieval must use existing capacity without suppressing other directions."""
from collections import Counter
from copy import deepcopy
import importlib

import pytest

from data_collection_workflow.agents import iterative_source_discovery_agent
from data_collection_workflow.environment import WORKFLOW_ENV_NAMES
from data_collection_workflow.query_policy import assess_query_task_fit

discovery = importlib.import_module("data_collection_workflow.nodes.source_discovery")

HIGH_TRUST = {
    "international_official", "national_or_local_official",
    "structured_database", "peer_reviewed_literature",
}
MEDIA = "news_or_supporting_media"


@pytest.fixture(autouse=True)
def evidence_environment(monkeypatch):
    for name in WORKFLOW_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def _state():
    return {"structured_task": {"disease": "measles", "location": "Canada",
            "start_date": "2025-01-01", "end_date": "2025-12-31", "target_fields": ["cases"]}}


def _official_queries(suffix="cases"):
    return [{"query_id": f"{kind}_{suffix}", "query": f"measles Canada 2025 {terms} {suffix}",
             "source_type": kind, "provider_channel": channel}
            for kind, channel, terms in [
                ("international_organization_report", "official_site_search", "international public health surveillance"),
                ("official_public_health_agency", "official_site_search", "public health surveillance"),
                ("structured_database", "database_search", "public health surveillance"),
                ("peer_reviewed_literature", "literature_api", "epidemiological surveillance")]]


def _media_query(index):
    return {"query_id": f"news_{index}", "query": f"measles Canada 2025 news reported cases {index}",
            "source_type": "news_and_situation_report", "provider_channel": "news_search"}


def _settings(**overrides):
    return discovery.SourceSearchSettings(**{
        "mode": "fixture", "max_queries": 12, "max_total_results": 100,
        "iterative_max_total_results": 100, "iterative_enabled": True,
        "iterative_max_iterations": 3, "iterative_max_queries_per_iteration": 4,
        "iterative_max_total_queries": 12, "authority_gap_retry_enabled": False,
        **overrides})


def _provider(monkeypatch):
    calls = []

    class Provider:
        def search(self, query, **kwargs):
            calls.append(dict(query))
            return {"results": [{"title": "Measles Canada 2025 cases",
                    "url": f"https://reports.example/measles/{len(calls)}"}]}

    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: Provider())
    monkeypatch.setattr(discovery, "external_call", lambda kind, payload, call: call())
    return calls


def _iterative(monkeypatch, **overrides):
    calls = _provider(monkeypatch)
    monkeypatch.setattr(iterative_source_discovery_agent, "plan_initial_search_iteration",
                        lambda **kwargs: {"query_batch": _official_queries()})
    monkeypatch.setattr(iterative_source_discovery_agent, "refine_search_iteration", lambda **kwargs: {
        "decision": "continue_search", "next_query_batch": _official_queries(f"reported cases {len(calls)}")})
    state = _state()
    before = deepcopy(state)
    result = discovery._execute_iterative_source_search(state, _settings(**overrides))
    assert state == before, "Query opportunities must not become source coverage or qualified records."
    return calls, result


@pytest.mark.parametrize("budget", [5, 12])
def test_media_gap_probe_executes_after_four_families_within_existing_budget(monkeypatch, budget):
    calls, (_, _, summary, _) = _iterative(monkeypatch, iterative_max_total_queries=budget)
    assert len(calls) == budget
    assert {discovery._query_source_class(query) for query in calls[:4]} == HIGH_TRUST
    assert discovery._query_source_class(calls[4]) == MEDIA
    assert calls[4].get("temporal_probe"), "The media and temporal opportunities share the fifth slot."
    assert "news" in calls[4]["query"].casefold()
    assert all(assess_query_task_fit(query, _state())["accepted"] for query in calls)
    assert len({discovery._discovery_query_key(query) for query in calls}) == budget
    assert summary["stop_decision"] == "stop_limits_reached"
    for batch in {query["iteration_index"] for query in calls if query["iteration_index"] > 1}:
        rows = [query for query in calls if query["iteration_index"] == batch]
        assert sum(discovery._query_source_class(query) == MEDIA for query in rows) <= 1
        assert sum(bool(query.get("temporal_probe")) for query in rows) <= 1


def test_media_does_not_crowd_out_real_month_archive_and_data_queries(monkeypatch):
    calls, _ = _iterative(monkeypatch)
    text = " ".join(query["query"].casefold() for query in calls)
    assert any(discovery._query_source_class(query) == MEDIA for query in calls)
    assert "january 2025" in text
    assert "historical" in text or "archive" in text
    assert any(term in text for term in ("download", "spreadsheet", "tables"))
    assert len(calls) == 12


@pytest.mark.parametrize("budget,news_enabled", [(4, True), (5, True), (5, False)])
def test_one_shot_dispatches_generic_media_after_family_floor(monkeypatch, budget, news_enabled):
    calls = _provider(monkeypatch)
    state = {**_state(), "search_query_inventory": [*_official_queries(), *_official_queries("case counts")]}
    channels = ["official_site_search", "database_search", "literature_api", "web_search"]
    if news_enabled:
        channels.append("news_search")
    discovery._execute_one_shot_source_search(state, _settings(max_queries=budget, provider_channel_allowlist=channels))
    assert len(calls) == budget
    assert {discovery._query_source_class(query) for query in calls[:4]} == HIGH_TRUST
    media = [query for query in calls if discovery._query_source_class(query) == MEDIA]
    assert len(media) == int(budget > 4 and news_enabled)
    assert all("news" in query["query"].casefold() for query in media)


def test_one_shot_media_floor_does_not_expand_three_query_media_cap():
    queries = [*_official_queries(), *[_media_query(index) for index in range(7)]]
    selected = discovery._select_source_search_queries(queries, _state(), _settings(max_queries=12))
    assert sum(row["selection_bucket"] == MEDIA for row in selected.values()) == 3
    assert sorted(selected.values(), key=lambda row: row["selection_order"])[4]["selection_bucket"] == MEDIA


def test_iterative_media_disabled_channel_is_never_dispatched(monkeypatch):
    calls, _ = _iterative(monkeypatch, provider_channel_allowlist=[
        "official_site_search", "database_search", "literature_api", "web_search"])
    assert len(calls) == 12
    assert all(query["provider_channel"] != "news_search" for query in calls)
    assert {discovery._query_source_class(query) for query in calls} == HIGH_TRUST


def test_disabled_literature_channel_does_not_delay_media_past_last_slot(monkeypatch):
    calls, _ = _iterative(monkeypatch, iterative_max_total_queries=4,
                         provider_channel_allowlist=["official_site_search", "database_search", "news_search"])
    assert len(calls) == 4
    assert {discovery._query_source_class(query) for query in calls[:3]} == {
        "international_official", "national_or_local_official", "structured_database"}
    assert discovery._query_source_class(calls[3]) == MEDIA
    assert calls[3].get("temporal_probe")


def test_standard_one_shot_keeps_existing_planned_query_behavior(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "standard")
    calls = _provider(monkeypatch)
    state = {**_state(), "search_query_inventory": [*_official_queries(), *_official_queries("case counts")]}
    discovery._execute_one_shot_source_search(state, _settings(max_queries=5))
    assert len(calls) == 5
    assert all(query["provider_channel"] != "news_search" for query in calls)


def test_selector_balances_media_without_promoting_context_or_flooding_batch():
    context = {"query": "measles Canada 2025 prevention", "provider_channel": "web_search",
               "source_type": "background_fact_sheet"}
    selected = discovery._remaining_discovery_queries(
        [context, *[_media_query(index) for index in range(5)], *_official_queries()],
        _state(), _settings(), set(), Counter({family: 5 for family in HIGH_TRUST}))
    classes = [discovery._query_source_class(query) for query in selected]
    assert classes[0] == MEDIA
    assert classes.count(MEDIA) == 1
    assert "context_or_other" not in classes


def test_selector_counts_reserved_media_probe_toward_batch_opportunity():
    selected = discovery._remaining_discovery_queries(
        [_media_query(1), *_official_queries()], _state(), _settings(iterative_max_queries_per_iteration=3),
        set(), Counter({**{family: 1 for family in HIGH_TRUST}, MEDIA: 1}),
        reserved_media_count=1)
    assert all(discovery._query_source_class(query) in HIGH_TRUST for query in selected)
