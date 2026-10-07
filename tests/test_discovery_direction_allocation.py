"""Within-family broad rewrites must not crowd out grounded discovery directions."""
from __future__ import annotations

from collections import Counter
import importlib

import pytest

from data_collection_workflow.agents import iterative_source_discovery_agent
from data_collection_workflow.environment import WORKFLOW_ENV_NAMES
from data_collection_workflow.query_policy import assess_query_task_fit

discovery = importlib.import_module("data_collection_workflow.nodes.source_discovery")

FAMILIES = [
    ("international_organization_report", "official_site_search", "international public health surveillance"),
    ("official_public_health_agency", "official_site_search", "public health surveillance"),
    ("structured_database", "database_search", "public health surveillance"),
    ("peer_reviewed_literature", "literature_api", "epidemiological surveillance"),
]
EXPECTED_FAMILIES = {
    "international_official", "national_or_local_official",
    "structured_database", "peer_reviewed_literature",
}


@pytest.fixture(autouse=True)
def isolated_discovery_environment(monkeypatch):
    for name in WORKFLOW_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def _state(start="2025-05-01", end="2025-05-31"):
    return {"structured_task": {
        "disease": "measles", "location": "Canada",
        "start_date": start, "end_date": end, "target_fields": ["cases"],
    }}


def _broad_queries(phrase):
    return [{
        "query": f"measles Canada 2025 {terms} {phrase}",
        "source_type": source_type,
        "provider_channel": channel,
        "query_source": "planner_or_refinement",
    } for source_type, channel, terms in FAMILIES]


def _settings(**overrides):
    values = dict(
        mode="fixture", max_queries=12, max_total_results=100,
        iterative_max_total_results=100, iterative_enabled=True,
        iterative_max_iterations=3, iterative_max_queries_per_iteration=4,
        iterative_max_total_queries=12, authority_gap_retry_enabled=False,
    )
    return discovery.SourceSearchSettings(**(values | overrides))


def _run_repeated_broad_refinement(monkeypatch, state, *, retag_initial_national=False):
    calls = []

    class Provider:
        def search(self, query, **kwargs):
            calls.append(dict(query))
            # Distinct sources prevent stagnation stopping from hiding allocation.
            return {"results": [{
                "title": "Measles Canada 2025 cases",
                "url": f"https://public-health.example/report/{len(calls)}",
            }]}

    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: Provider())
    monkeypatch.setattr(discovery, "external_call", lambda kind, payload, call: call())
    monkeypatch.setattr(
        iterative_source_discovery_agent, "plan_initial_search_iteration",
        lambda **kwargs: {"query_batch": _broad_queries("cases")},
    )
    refinements = []

    def refine(**kwargs):
        refinements.append(kwargs)
        phrase = "case counts" if len(refinements) == 1 else "reported cases"
        retagged_history = []
        if retag_initial_national:
            retagged_history = [
                {**query, "source_type": "international_organization_report"}
                for query in calls[:4]
                if discovery._query_source_class(query) == "national_or_local_official"
            ]
        return {
            "iteration_index": len(refinements),
            "decision": "continue_search",
            "decision_reason": "Try additional task-grounded queries.",
            "next_query_batch": [*retagged_history, *_broad_queries(phrase)],
        }

    monkeypatch.setattr(iterative_source_discovery_agent, "refine_search_iteration", refine)
    _, _, summary, details = discovery._execute_iterative_source_search(state, _settings())
    return calls, summary, details


def _assert_existing_search_contract(calls, state, summary, details):
    assert len(calls) == 12
    keys = {discovery._discovery_query_key(query) for query in calls}
    assert len(keys) == 12
    assert {discovery._query_source_class(query) for query in calls} == EXPECTED_FAMILIES
    assert all(assess_query_task_fit(query, state)["accepted"] for query in calls)
    assert summary["stop_decision"] == "stop_limits_reached"
    assert len(details["search_iteration_observations"]) == 3


def test_month_and_history_directions_survive_repeated_broad_refinement(monkeypatch):
    state = _state()
    calls, summary, details = _run_repeated_broad_refinement(monkeypatch, state)
    _assert_existing_search_contract(calls, state, summary, details)
    assert any("may 2025" in query["query"].lower() for query in calls)
    assert any("historical" in query["query"].lower() or "archive" in query["query"].lower()
               for query in calls)


def test_refinement_cannot_retag_initial_national_history_and_displace_its_month_slot(monkeypatch):
    state = _state()
    calls, summary, details = _run_repeated_broad_refinement(
        monkeypatch, state, retag_initial_national=True)
    _assert_existing_search_contract(calls, state, summary, details)
    # The duplicate changes only the initial national query's family metadata.
    # The first continuation must retain the four month leads of the control.
    month_families = Counter(
        discovery._query_source_class(query) for query in calls[4:8]
        if "may 2025" in query["query"].lower())
    assert month_families == Counter({family: 1 for family in EXPECTED_FAMILIES})


def test_annual_history_and_download_directions_get_existing_budget_slots(monkeypatch):
    state = _state("2025-01-01", "2025-12-31")
    calls, summary, details = _run_repeated_broad_refinement(monkeypatch, state)
    _assert_existing_search_contract(calls, state, summary, details)
    assert any("historical" in query["query"].lower() or "archive" in query["query"].lower()
               for query in calls)
    assert any("download" in query["query"].lower() or "spreadsheet" in query["query"].lower()
               for query in calls)
    assert not any(query.get("time_terms") for query in calls)


def test_selector_keeps_stable_ties_and_existing_eligibility_constraints():
    state = _state()
    base = {"source_type": "official_public_health_agency",
            "provider_channel": "official_site_search"}
    first = {**base, "query": "measles Canada 2025 public health reported cases"}
    second = {**base, "query": "measles Canada 2025 public health case counts"}
    already = {**base, "query": "measles Canada 2025 public health cases"}
    blocked = {**base, "query": "measles Canada may 2025 historical reports",
               "provider_channel": "database_search"}
    unrelated = {**base, "query": "dengue Brazil 2025 historical reports"}
    attempted = {discovery._discovery_query_key(already)}
    family = discovery._query_source_class(first)
    selected = discovery._remaining_discovery_queries(
        [already, blocked, unrelated, first, dict(first), second], state,
        _settings(iterative_max_queries_per_iteration=2,
                  provider_channel_allowlist=["official_site_search"]),
        attempted, Counter({family: 1}),
    )
    assert [query["query"] for query in selected] == [first["query"], second["query"]]


def test_duplicate_history_does_not_count_as_multiple_direction_attempts():
    state = _state()
    base = {"source_type": "official_public_health_agency",
            "provider_channel": "official_site_search"}
    general_old = {**base, "query": "measles Canada 2025 public health cases"}
    history_old = {**base, "query": "measles Canada 2025 historical reports"}
    general_new = {**base, "query": "measles Canada 2025 public health case counts"}
    history_new = {**base, "query": "measles Canada 2025 historical reports archive"}
    attempted = {discovery._discovery_query_key(query) for query in [general_old, history_old]}
    family = discovery._query_source_class(general_old)
    selected = discovery._remaining_discovery_queries(
        [general_new, history_new, general_old, dict(general_old), history_old],
        state, _settings(iterative_max_queries_per_iteration=1),
        attempted, Counter({family: 2}),
        Counter({(family, "general"): 1, (family, "historical"): 1}),
    )
    # One attempted query per direction ties; original pool order wins.
    assert [query["query"] for query in selected] == [general_new["query"]]
