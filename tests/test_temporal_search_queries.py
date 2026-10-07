"""Temporal search must get a bounded opportunity before the shared budget ends."""
from collections import Counter
from copy import deepcopy
import importlib

import pytest

from data_collection_workflow.agents import iterative_source_discovery_agent
from data_collection_workflow.environment import WORKFLOW_ENV_NAMES

discovery = importlib.import_module("data_collection_workflow.nodes.source_discovery")


@pytest.fixture(autouse=True)
def evidence_environment(monkeypatch):
    for name in WORKFLOW_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def _state(start="2025-01-01", end="2025-03-31"):
    return {"structured_task": {"disease": "measles", "location": "Ontario Canada",
            "start_date": start, "end_date": end, "target_fields": ["cases_confirmed"]}}


def _search(monkeypatch, *, total=20, published="2025-03-31"):
    calls = []

    class Provider:
        def search(self, query, **kwargs):
            calls.append(dict(query))
            return {"results": [{"url": "https://health.example/measles-report",
                    "title": "Ontario Canada measles cases 2025", "published_date": published}]}

    def broad(index):
        return {"query": f'"measles" "Ontario Canada" 2025 surveillance cases {index}',
                "provider_channel": "official_site_search", "source_type": "official_public_health_agency"}

    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: Provider())
    monkeypatch.setattr(discovery, "external_call", lambda kind, payload, call: call())
    monkeypatch.setattr(iterative_source_discovery_agent, "plan_initial_search_iteration",
                        lambda **kwargs: {"query_batch": [broad(1), broad(2)]})
    monkeypatch.setattr(iterative_source_discovery_agent, "refine_search_iteration", lambda **kwargs: {
        "decision": "continue_search", "next_query_batch": [broad(len(calls) + 1), broad(len(calls) + 2)]})
    settings = discovery.SourceSearchSettings(mode="fixture", iterative_enabled=True,
        max_total_results=100, iterative_max_total_results=100, iterative_max_iterations=12,
        iterative_max_queries_per_iteration=2, iterative_max_total_queries=total,
        provider_channel_allowlist=["official_site_search"], authority_gap_retry_enabled=False)
    _, _, summary, _ = discovery._execute_iterative_source_search(_state(), settings)
    return calls, summary


def test_quarter_end_candidate_nominates_earlier_probe_before_initial_budget_ends(monkeypatch):
    calls, summary = _search(monkeypatch, total=6)
    probes = [query for query in calls if query.get("temporal_probe")]
    assert probes, "An earlier temporal probe must execute inside initial discovery."
    assert probes[0]["temporal_probe"]["end_date"] < "2025-03-31"
    assert probes[0]["temporal_probe"]["date_basis"] == "candidate_publication_metadata"
    assert len(calls) <= 6
    assert all(count <= 1 for count in Counter(q["iteration_index"] for q in probes).values())


def test_broad_stagnation_allows_only_four_novel_temporal_probes(monkeypatch):
    calls, summary = _search(monkeypatch)
    probes = [query for query in calls if query.get("temporal_probe")]
    assert len(probes) == 4
    assert len({query["query"] for query in probes}) == 4
    assert len(calls) < 20
    assert summary["stop_reason"] == "no_new_sources_in_consecutive_batches"


def test_long_window_fallback_has_bounded_distributed_month_leads():
    queries = discovery._discovery_breadth_queries(_state("2020-02-10", "2030-11-19"))
    temporal = [query for query in queries if query.get("time_terms")]
    assert temporal, "Long windows must offer temporal leads without monthly subruns."
    words = " ".join(query["query"].lower() for query in temporal)
    assert "february 2020" in words and "november 2030" in words
    assert len(temporal) <= 12
    assert any("historical" in query["query"] for query in queries)
    assert any("download" in query["query"] for query in queries)


@pytest.mark.parametrize("start,end", [
    ("2024-02-29", "2024-02-29"), ("2024-02-10", "2024-03-02"),
    ("2025-12-29", "2026-01-03"), ("1900-01-01", "2099-12-31"),
])
def test_probe_windows_are_bounded_and_inside_arbitrary_task_dates(start, end):
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    probes = temporal_search_queries(_state(start, end))
    assert 1 <= len(probes) <= 4
    for query in probes:
        window = query["temporal_probe"]
        assert start <= window["start_date"] <= window["end_date"] <= end


def test_raw_candidate_dates_are_hints_without_relaxing_strict_inventory():
    from data_collection_workflow.report_timeline import build_report_timeline_inventory
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    source = {"source_id": "s", "url": "https://health.example/report", "published_date": "2025-03-31",
              "role_hint": "collection_support"}
    state = {**_state(), "source_registry": [source]}
    original = deepcopy(state)
    assert build_report_timeline_inventory(state)["publication_candidates"]["distinct_date_count"] == 0
    probes = temporal_search_queries(state, candidates=[source])
    assert probes[0]["temporal_probe"]["date_basis"] == "candidate_publication_metadata"
    for negative in ({"blocked_from_fetch": True}, {"source_role_final": "validation"},
                     {"usable_for_task_collection": False}, {"role_hint": "validation_reference"}):
        blocked = temporal_search_queries(state, candidates=[{**source, **negative}])
        assert blocked[0]["temporal_probe"]["date_basis"] == "task_window"
    assert state == original


def test_stale_inventory_cannot_nominate_out_of_window_gap():
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    state = _state()
    state["source_coverage_audit"] = {"reporting_timeline": {
        "publication_candidates": {"distinct_date_count": 1, "date_basis": "candidate_publication_metadata",
            "gaps": [{"start_date": "2030-01-01", "end_date": "2030-12-31"}]}}}
    probes = temporal_search_queries(state)
    assert probes
    assert all("2025-01-01" <= q["temporal_probe"]["start_date"] <= q["temporal_probe"]["end_date"] <= "2025-03-31"
               for q in probes)


def test_session_probe_allowance_combines_initial_and_recovery_history():
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    state = _state()
    first = temporal_search_queries(state)[0]
    initial = {**first, "selected_for_execution": True, "execution_status": "no_results"}
    state["source_search_execution_summary"] = {"query_execution_records": [initial]}
    # A repeated view of one initial execution must not spend another allowance.
    assert temporal_search_queries(state, query_records=[initial])
    state["recovery_action_history"] = []
    for _ in range(3):
        query = temporal_search_queries(state)[0]
        state["recovery_action_history"].append({**query, "kind": "search", "status": "completed",
            "query": query["query"], "search_dispatched": True})
    assert temporal_search_queries(state) == []


def test_recovery_nominates_one_probe_and_respects_disabled_search(monkeypatch):
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery
    state = _state()
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    budget = {"remaining": {"search": 8, "search_results": 30, "fetch": 10, "fetch_ordinary": 10, "extraction": 10}}
    plan = plan_recovery(assess_collection_gaps(state), state=state, budget=budget)
    temporal = [action for action in plan.actions if (action.query or {}).get("temporal_probe")]
    assert len(temporal) == 1
    monkeypatch.setenv("SEARCH_MODE", "disabled")
    plan = plan_recovery(assess_collection_gaps(state), state=state, budget=budget)
    assert not any((action.query or {}).get("temporal_probe") for action in plan.actions)


def test_ordinary_recovery_deduplicates_initial_query_and_preserves_failed_retry():
    from data_collection_workflow.workflow_recovery import _recovery_query
    state = _state()
    query = _recovery_query(state, "task")
    state["source_search_execution_summary"] = {"query_execution_records": [{**query,
        "query": "  " + query["query"].upper() + "  ", "selected_for_execution": True,
        "execution_status": "no_results"}]}
    next_query = _recovery_query(state, "task")
    assert next_query is None or next_query["query"] != query["query"]
    state["source_search_execution_summary"]["query_execution_records"][0]["execution_status"] = "provider_error"
    assert _recovery_query(state, "task")["query"] == query["query"]


def test_initial_llm_cannot_forge_probe_accounting_or_disabled_channel(monkeypatch):
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    forged = {"query": "measles Ontario Canada cases 2025", "temporal_probe": {"probe_id": "forged"}}
    normalized = discovery._normalize_iteration_plan({"query_batch": [forged]}, 1)
    assert "temporal_probe" not in normalized["query_batch"][0]
    settings = discovery.SourceSearchSettings(iterative_max_queries_per_iteration=1)
    selected = discovery._remaining_discovery_queries([forged], _state(), settings, set(), Counter())
    assert "temporal_probe" not in selected[0]
    assert temporal_search_queries(_state(), channels=[]) == []
    assert all(query["provider_channel"] == "database_search"
               for query in temporal_search_queries(_state(), channels=["database_search"]))


def test_overlap_of_persisted_and_current_history_does_not_double_count_probes():
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    state = _state()
    records = []
    for _ in range(3):
        query = temporal_search_queries(state, query_records=records)[0]
        records.append({**query, "selected_for_execution": True, "execution_status": "no_results"})
    state["source_search_execution_summary"] = {"query_execution_records": records}
    assert temporal_search_queries(state, query_records=records)


@pytest.mark.parametrize("start,end", [("2025-02-30", "2025-03-31"), ("2026-01-01", "2025-01-01"), ("", "2025-01-01")])
def test_invalid_windows_do_not_generate_temporal_probes(start, end):
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    assert temporal_search_queries(_state(start, end)) == []


def test_recovery_temporal_action_does_not_bypass_budget_provider_or_round_limit(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import recovery_control, assess_collection_gaps, plan_recovery
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    state = {**_state(), "recovery_round": 1, "recovery_previous_gain": []}
    context = RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {}})
    with context.activate():
        result = recovery_control(state)
        assert any((action.get("query") or {}).get("temporal_probe") for action in result["recovery_plan"]["actions"])
        assert result["recovery_stop_reason"] is None
        assert recovery_control({**state, "recovery_round": 2})["recovery_stop_reason"] == "round_limit"
        gaps = assess_collection_gaps(state)
        assert not plan_recovery(gaps, state=state, budget={"remaining": {"search": 0}}).actions
        monkeypatch.setattr("data_collection_workflow.workflow_recovery._provider_stop", lambda snapshot: {"status": "halted"})
        assert plan_recovery(gaps, state=state, budget=context.ledger).stop_reason == "provider_account_limit"


def test_recovery_empty_probe_keeps_metadata_and_spends_one_real_search(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    from data_collection_workflow.workflow_recovery import execute_recovery, RecoveryAction, RecoveryPlan
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    class EmptyProvider:
        def search(self, query, **kwargs):
            return {"results": []}
    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: EmptyProvider())
    state = _state()
    query = temporal_search_queries(state)[0]
    action = RecoveryAction("search", query["temporal_probe"]["probe_id"], "attempt", "date hint", query=query)
    context = RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {}})
    with context.activate():
        delta = execute_recovery(RecoveryPlan([action]), context=context, artifacts=state, budget=context.ledger)
    row = delta.actions[0]
    assert row["search_dispatched"] is True
    assert row["temporal_probe"] == query["temporal_probe"]
    assert context.ledger.snapshot()["used"]["search"] == 1
    assert all(probe["query"] != query["query"] for probe in temporal_search_queries({**state, "recovery_action_history": delta.actions}))


def test_last_recovery_search_slot_goes_to_novel_probe_after_broad_stagnation(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import recovery_control, execute_recovery
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    calls = []
    class EmptyProvider:
        def search(self, query, **kwargs):
            calls.append(query)
            return {"results": []}
    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: EmptyProvider())
    state = {**_state(), "recovery_round": 4, "recovery_action_history": [
        {"kind": "search", "status": "completed", "search_direction": direction,
         "query": f"measles Ontario Canada {direction}", "new_independent_source_ids": []}
        for direction in ("archive", "dataset")]}
    context = RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive"}, "budget_limits": {"search": 1}}})
    with context.activate():
        result = recovery_control(state)
        execute_recovery(result["recovery_plan"], context=context, artifacts=state, budget=context.ledger)
    assert len(calls) == 1
    assert calls[0].get("temporal_probe"), "Broad corroboration must not consume the final temporal opportunity."
    assert context.ledger.snapshot()["used"]["search"] == 1


def test_probe_scope_falls_back_to_collection_spec_for_none_task_fields():
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    state = {"collection_spec": _state()["structured_task"],
             "structured_task": {"disease": None, "location": None, "start_date": None, "end_date": None}}
    assert temporal_search_queries(state)


@pytest.mark.parametrize("negative", [{"role_hint": "context"}, {"role_hint": "validation"}])
def test_context_or_validation_candidate_date_cannot_guide_timeline_probe(negative):
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    source = {"source_id": "s", "url": "https://health.example/report", "published_date": "2025-03-31", **negative}
    assert temporal_search_queries(_state(), candidates=[source])[0]["temporal_probe"]["date_basis"] == "task_window"


@pytest.mark.parametrize("same_id", [True, False])
def test_saved_human_exclusion_wins_over_raw_candidate_date(same_id):
    from data_collection_workflow.temporal_search_queries import temporal_search_queries
    state = {**_state(), "source_registry": [{"source_id": "s", "url": "https://health.example/report",
             "source_excluded_by_human_review": True}]}
    source = {"source_id": "s" if same_id else "alias", "url": "https://health.example/report", "published_date": "2025-03-31"}
    assert temporal_search_queries(state, candidates=[source])[0]["temporal_probe"]["date_basis"] == "task_window"


def test_denied_temporal_search_reopens_after_explicit_budget_amendment(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery, execute_recovery
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    calls = []
    class Provider:
        def search(self, query, **kwargs):
            calls.append(query)
            return {"results": []}
    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: Provider())
    context = RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive"}, "budget_limits": {"search": 1}}})
    state = _state()
    real_call = discovery.external_call
    def concurrent_consumer(kind, payload, callback, **kwargs):
        if not context.ledger.snapshot()["used"].get("search"):
            context.call("search", {"another": "consumer"}, lambda: {"results": []})
        return real_call(kind, payload, callback, **kwargs)
    monkeypatch.setattr(discovery, "external_call", concurrent_consumer)
    with context.activate():
        gaps = [gap for gap in assess_collection_gaps(state) if gap.kind == "report_timeline_gap"]
        plan = plan_recovery(gaps, state=state, budget=context.ledger)
        delta = execute_recovery(plan, context=context, artifacts=state, budget=context.ledger)
        assert not calls
        assert delta.actions[0]["status"] == "budget_exhausted"
        state["recovery_action_history"] = delta.actions
        context.amend_budget(amendment_id="more-search", increases={"search": 2}, reason="explicit offline continuation")
        retry = plan_recovery(gaps, state=state, budget=context.ledger)
        assert retry.actions
        execute_recovery(retry, context=context, artifacts=state, budget=context.ledger)
    assert len(calls) == 1


def test_interrupted_initial_replay_cannot_reset_probe_cap_after_failed_search_recovers(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    phase, paid_probes = [0], []
    class Provider:
        def search(self, query, **kwargs):
            if query["query"].endswith("initial 1"):
                if phase[0] == 0:
                    raise OSError("temporary initial search failure")
                return {"results": [{"url": "https://health.example/earlier", "title": "Ontario Canada measles 2025",
                                     "published_date": "2025-01-15"}]}
            if query.get("temporal_probe"):
                paid_probes.append(query)
            return {"results": [{"url": "https://health.example/latest", "title": "Ontario Canada measles 2025",
                                 "published_date": "2025-03-31"}]}
    monkeypatch.setattr(discovery, "_provider_for_settings", lambda settings: Provider())
    monkeypatch.setattr(iterative_source_discovery_agent, "plan_initial_search_iteration", lambda **kwargs: {
        "query_batch": [{"query": f"measles Ontario Canada 2025 initial {index}",
                         "source_type": "official_public_health_agency", "provider_channel": "official_site_search"}
                        for index in (1, 2)]})
    def refine(**kwargs):
        if phase[0] == 0 and len(paid_probes) == 4:
            raise KeyboardInterrupt("interrupt before node checkpoint")
        return {"decision": "continue_search", "next_query_batch": []}
    monkeypatch.setattr(iterative_source_discovery_agent, "refine_search_iteration", refine)
    settings = discovery.SourceSearchSettings(mode="fixture", iterative_enabled=True,
        max_total_results=100, iterative_max_total_results=100, iterative_max_iterations=20,
        iterative_max_queries_per_iteration=2, iterative_max_total_queries=40,
        provider_channel_allowlist=["official_site_search"], authority_gap_retry_enabled=False)
    config = {"pipeline_mode": "evidence", "universal": {"budget_limits": {"search": 100}}}
    context = RunContext(tmp_path, config)
    with context.activate(), pytest.raises(KeyboardInterrupt):
        discovery._execute_iterative_source_search(_state(), settings)
    assert len(paid_probes) == 4
    phase[0] = 1
    resumed = RunContext(tmp_path, config, resume=True)
    with resumed.activate():
        discovery._execute_iterative_source_search(_state(), settings)
    assert len(paid_probes) == 4, "Recovered initial search metadata must not reset the persisted four-probe cap."
