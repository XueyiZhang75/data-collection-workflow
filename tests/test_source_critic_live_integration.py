from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


CRITIC_ENV_KEYS = [
    "ENABLE_LLM_SOURCE_CRITIC",
    "LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST",
    "LLM_SOURCE_CRITIC_MAX_SOURCES",
    "LLM_SOURCE_CRITIC_MAX_FAILURES",
    "LLM_SOURCE_CRITIC_MAX_NODE_SECONDS",
    "LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH",
    "ENABLE_LLM_SOURCE_CREDIBILITY",
    "ENABLE_LLM_SOURCE_IDENTITY",
    "LLM_SOURCE_IDENTITY_MAX_SOURCES",
    "LLM_SOURCE_IDENTITY_REQUIRE_LLM",
    "LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK",
    "LLM_PROVIDER",
    "LLM_MODEL",
]


def _clear_critic_env(monkeypatch) -> None:
    for key in CRITIC_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _state(entries: list[dict]) -> dict:
    return {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Shanghai",
            "start_date": "2024-01-01",
            "end_date": "2026-06-09",
        },
        "collection_spec": {
            "disease": "Hantavirus disease",
            "geography": "Shanghai",
            "time_window": "2024-01-01 to 2026-06-09",
            "target_population": "human",
        },
        "disease_intelligence": {
            "disease_standard_name": "Hantavirus disease",
            "aliases": ["hantavirus", "HFRS"],
            "pathogen_terms": ["hantavirus"],
            "syndrome_terms": ["hemorrhagic fever with renal syndrome"],
        },
        "source_registry": entries,
        "human_review_queue": [],
        "collection_trace": [],
    }


def _entry(
    source_id: str,
    *,
    discovery_method: str = "live_search_result",
    title: str = "Shanghai hantavirus cases and deaths surveillance report",
    url: str | None = None,
    source_type: str = "official_public_health_agency",
    role_hint: str = "collection",
    priority: int = 1,
    search_rank: int = 1,
    source_role_final: str | None = None,
    ready_for_content_fetch: bool | None = None,
) -> dict:
    entry = {
        "source_id": source_id,
        "canonical_url": url or f"https://example.org/{source_id}",
        "title": title,
        "publisher": "Example Health Department",
        "source_type": source_type,
        "snippet": "Reported hantavirus cases, deaths, dates, and Shanghai location.",
        "status": "registered",
        "expected_fields": ["cases", "deaths", "date", "location"],
        "matched_terms": ["hantavirus", "Shanghai"],
        "discovery_method": discovery_method,
        "search_provider": "fixture" if discovery_method == "fixture_search_result" else "tavily",
        "query_id": f"q_{source_id}",
        "query_used": '"hantavirus Shanghai" cases deaths',
        "role_hint": role_hint,
        "priority": priority,
        "search_rank": search_rank,
    }
    if source_role_final is not None:
        entry["source_role_final"] = source_role_final
    if ready_for_content_fetch is not None:
        entry["ready_for_content_fetch"] = ready_for_content_fetch
    return entry


def _critic_output(
    source_id: str,
    *,
    decision: str = "suitable_for_collection",
    fetch_recommendation: str = "allow_fetch",
    risk_flags: list[str] | None = None,
    confidence: float = 0.9,
    reason: str = "Source metadata appears suitable for the task.",
    recommended_role: str = "collection",
    review_required: bool = False,
) -> dict:
    return {
        "source_id": source_id,
        "proposed_source_role": recommended_role,
        "proposed_screening_decision": "include",
        "credibility_level": "high",
        "credibility_reason": reason,
        "expected_extractable_fields": ["cases", "deaths", "date", "location"],
        "semantic_leakage_risk": False,
        "semantic_leakage_reason": "",
        "context_only_risk": decision in {"suitable_for_context", "collection_support_only"},
        "validation_candidate_risk": decision == "suitable_for_validation",
        "needs_human_review": review_required,
        "human_review_reason": reason if review_required else "",
        "confidence": confidence,
        "reasoning_summary": reason,
        "critic_decision": decision,
        "risk_flags": risk_flags or [],
        "recommended_role": recommended_role,
        "fetch_recommendation": fetch_recommendation,
        "review_required": review_required,
        "warnings": [],
    }


def _run_critic(monkeypatch, entries: list[dict], outputs: dict[str, dict] | None = None):
    source_screening_module = importlib.import_module(
        "data_collection_workflow.nodes.source_screening"
    )
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    calls: list[str] = []
    outputs = outputs or {}

    def fake_assess(source_entry, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        source_id = source_entry["source_id"]
        calls.append(source_id)
        return outputs.get(source_id) or _critic_output(source_id)

    monkeypatch.setattr(
        source_screening_module,
        "assess_source_with_llm",
        fake_assess,
    )

    state = _state(entries)
    state.update(source_screening(state))
    result = source_critic_and_uncertainty_routing(state)
    state.update(result)
    return state, calls


def test_empty_critic_allowlist_means_no_allowlist(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST", "")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_SOURCES", "2")

    result, calls = _run_critic(
        monkeypatch,
        [
            _entry("src_live_1", search_rank=1),
            _entry("src_live_2", search_rank=2),
            _entry("src_live_3", search_rank=3),
        ],
    )

    summary = result["source_critic_summary"]
    selection = summary["source_critic_selection_summary"]
    assert summary["attempted_source_count"] == 2
    assert summary["assessed_source_count"] == 2
    assert summary["skipped_source_count"] == 1
    assert selection["explicit_allowlist_used"] is False
    assert selection["selected_source_ids"] == calls == ["src_live_1", "src_live_2"]


def test_explicit_non_empty_allowlist_is_respected(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST", "src_live_2")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_SOURCES", "6")

    result, calls = _run_critic(
        monkeypatch,
        [_entry("src_live_1"), _entry("src_live_2"), _entry("src_live_3")],
    )

    summary = result["source_critic_summary"]
    assert calls == ["src_live_2"]
    assert summary["assessed_source_count"] == 1
    assert summary["skipped_reason_counts"]["source_not_in_explicit_critic_allowlist"] == 2


def test_live_search_candidates_prioritized_over_seed_sources(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_SOURCES", "2")

    result, calls = _run_critic(
        monkeypatch,
        [
            _entry("src_seed_1", discovery_method="offline_seed_catalog", priority=1),
            _entry("src_live_1", discovery_method="live_search_result", search_rank=1),
            _entry("src_live_2", discovery_method="live_search_result", search_rank=2),
        ],
    )

    assert calls == ["src_live_1", "src_live_2"]
    selection = result["source_critic_summary"]["source_critic_selection_summary"]
    assert selection["selection_mode"] == "auto_priority"


def test_critic_disease_mismatch_blocks_fetch_and_creates_review(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH", "true")

    result, _calls = _run_critic(
        monkeypatch,
        [
            _entry(
                "src_yahoo_covid",
                title="EXPLAINER-Shanghai death numbers raise questions over its COVID accounting",
                url="https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
                source_type="news_and_situation_report",
            )
        ],
        {
            "src_yahoo_covid": _critic_output(
                "src_yahoo_covid",
                decision="not_task_relevant",
                fetch_recommendation="block_fetch",
                risk_flags=["critical_disease_mismatch"],
                confidence=0.96,
                reason="Source is about COVID-19/SARS-CoV-2, not hantavirus.",
                recommended_role="excluded",
                review_required=True,
            )
        },
    )

    entry = result["source_registry"][0]
    assert entry["ready_for_content_fetch"] is False
    assert entry["blocked_from_fetch"] is True
    assert "llm_source_critic_block_fetch" in entry["blocked_from_fetch_reason"]
    assert entry["source_role_final"] in {"excluded", "needs_human_review"}
    assert result["source_critic_summary"]["blocked_fetch_count"] == 1
    assert any(
        item["item_type"] == "source_critic_blocked_source"
        and "src_yahoo_covid" in item["related_ids"]
        for item in result["human_review_queue"]
    )


def test_critic_no_extractable_data_routes_to_context_or_review(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH", "true")

    result, _calls = _run_critic(
        monkeypatch,
        [_entry("src_context_only", title="Hantavirus background and prevention")],
        {
            "src_context_only": _critic_output(
                "src_context_only",
                decision="no_extractable_data",
                fetch_recommendation="context_fetch_only",
                risk_flags=["only_background_or_context"],
                confidence=0.88,
                reason="Relevant background only; no case or death data expected.",
                recommended_role="context",
                review_required=True,
            )
        },
    )

    entry = result["source_registry"][0]
    assert entry["ready_for_content_fetch"] is False
    assert entry["source_role_final"] in {"context", "needs_human_review"}
    assert entry["llm_source_critic_decision"] == "no_extractable_data"
    assert result["source_critic_summary"]["context_only_count"] >= 1


def test_relevant_official_source_remains_fetchable(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH", "true")

    result, _calls = _run_critic(
        monkeypatch,
        [_entry("src_official_hfrs", title="Shanghai HFRS hantavirus cases 2025")],
        {
            "src_official_hfrs": _critic_output(
                "src_official_hfrs",
                decision="suitable_for_collection",
                fetch_recommendation="allow_fetch",
                confidence=0.92,
                reason="Official source appears task-relevant.",
            )
        },
    )

    entry = result["source_registry"][0]
    assert entry["ready_for_content_fetch"] is True
    assert entry.get("blocked_from_fetch") is False
    assert entry["llm_source_critic_decision"] == "suitable_for_collection"


def test_critic_decision_alias_exclude_is_normalized(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH", "true")

    result, _calls = _run_critic(
        monkeypatch,
        [_entry("src_alias_exclude")],
        {
            "src_alias_exclude": _critic_output(
                "src_alias_exclude",
                decision="exclude",
                fetch_recommendation="block_fetch",
                reason="Model used an alias for not task relevant.",
                recommended_role="excluded",
            )
        },
    )

    entry = result["source_registry"][0]
    assert entry["llm_source_critic_decision"] == "not_task_relevant"
    assert entry["blocked_from_fetch"] is True


def test_critic_failure_does_not_crash_workflow(monkeypatch):
    source_screening_module = importlib.import_module(
        "data_collection_workflow.nodes.source_screening"
    )
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")

    def fail_assess(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("mock critic failure")

    monkeypatch.setattr(source_screening_module, "assess_source_with_llm", fail_assess)
    state = _state([_entry("src_live_fail")])
    state.update(source_screening(state))
    result = source_critic_and_uncertainty_routing(state)

    summary = result["source_critic_summary"]
    assert summary["failed_source_count"] == 1
    assert summary["assessed_source_count"] == 0
    assert result["source_registry"][0]["llm_source_critic_failed"] is True
    assert result["source_registry"][0]["ready_for_content_fetch"] is True


def test_critic_stops_llm_calls_after_failure_cap_and_fails_open(monkeypatch):
    source_screening_module = importlib.import_module(
        "data_collection_workflow.nodes.source_screening"
    )
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_SOURCES", "3")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_MAX_FAILURES", "1")

    calls: list[str] = []

    def fail_assess(source_entry, *_args, **_kwargs):
        calls.append(source_entry["source_id"])
        raise TimeoutError("mock critic timeout")

    monkeypatch.setattr(source_screening_module, "assess_source_with_llm", fail_assess)
    state = _state(
        [
            _entry("src_live_1", search_rank=1),
            _entry("src_live_2", search_rank=2),
            _entry("src_live_3", search_rank=3),
        ]
    )
    state.update(source_screening(state))

    result = source_critic_and_uncertainty_routing(state)

    summary = result["source_critic_summary"]
    by_id = {entry["source_id"]: entry for entry in result["source_registry"]}

    assert calls == ["src_live_1"]
    assert summary["failed_source_count"] == 1
    assert summary["fail_open_skipped_source_count"] == 2
    assert summary["fail_open_reason_counts"] == {
        "source_critic_failure_cap_reached": 2
    }
    assert by_id["src_live_1"]["llm_source_critic_status"] == "timeout"
    assert by_id["src_live_2"]["llm_source_critic_status"] == (
        "source_critic_skipped_timeout_fail_open"
    )
    assert by_id["src_live_3"]["llm_source_critic_status"] == (
        "source_critic_skipped_timeout_fail_open"
    )
    assert by_id["src_live_2"]["ready_for_content_fetch"] is True
    assert by_id["src_live_3"]["ready_for_content_fetch"] is True


def test_required_source_identity_llm_failure_stops_product_workflow(monkeypatch):
    source_identity_module = importlib.import_module("data_collection_workflow.source_identity")
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_IDENTITY", "true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_MAX_SOURCES", "1")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_REQUIRE_LLM", "true")
    monkeypatch.setenv("LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK", "false")

    def fail_identity(**_kwargs):
        raise RuntimeError("mock missing source identity LLM")

    monkeypatch.setattr(
        source_identity_module,
        "assess_source_identity_with_llm",
        fail_identity,
    )

    state = _state([_entry("src_identity_required")])
    state.update(source_screening(state))

    with pytest.raises(RuntimeError, match="source identity LLM required"):
        source_critic_and_uncertainty_routing(state)


def test_disabled_critic_does_not_call_llm(monkeypatch):
    source_screening_module = importlib.import_module(
        "data_collection_workflow.nodes.source_screening"
    )
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "false")

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("LLM critic should not be called when disabled")

    monkeypatch.setattr(
        source_screening_module, "assess_source_with_llm", fail_if_called
    )
    state = _state([_entry("src_live_disabled")])
    state.update(source_screening(state))
    result = source_critic_and_uncertainty_routing(state)

    summary = result["source_critic_summary"]
    assert summary["llm_source_critic_enabled"] is False
    assert summary["attempted_source_count"] == 0
    assert result["source_registry"][0]["llm_source_critic_attempted"] is False


def test_source_critic_results_exported_in_final_package_summaries(monkeypatch):
    from data_collection_workflow.nodes.finalization import final_data_package_builder

    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")

    result, _calls = _run_critic(monkeypatch, [_entry("src_live_export")])
    package_result = final_data_package_builder(
        {
            **result,
            "normalized_records": [],
            "linked_events": [],
            "event_clusters": [],
            "duplicate_clusters": [],
            "validation_cases": [],
            "validation_comparisons": [],
            "validation_results": [],
            "anomaly_results": [],
            "conflicts": [],
        }
    )

    package = package_result["final_data_package"]
    assert "source_critic_summary" in package["workflow_summaries"]
    assert package["source_registry"][0]["llm_source_critic_assessed"] is True


def test_shanghai_like_covid_source_critic_regression(monkeypatch):
    _clear_critic_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CRITIC", "true")
    monkeypatch.setenv("LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH", "true")

    result, calls = _run_critic(
        monkeypatch,
        [
            _entry(
                "src_search_yahoo_covid",
                title="EXPLAINER-Shanghai death numbers raise questions over its COVID accounting",
                url="https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
                source_type="news_and_situation_report",
            )
        ],
        {
            "src_search_yahoo_covid": _critic_output(
                "src_search_yahoo_covid",
                decision="not_task_relevant",
                fetch_recommendation="block_fetch",
                risk_flags=["critical_disease_mismatch", "source_not_about_task"],
                confidence=0.97,
                reason="Shanghai source is about COVID-19/SARS-CoV-2 deaths, not HFRS/hantavirus.",
                recommended_role="excluded",
                review_required=True,
            )
        },
    )

    assert calls == ["src_search_yahoo_covid"]
    summary = result["source_critic_summary"]
    entry = result["source_registry"][0]
    assert summary["attempted_source_count"] == 1
    assert summary["assessed_source_count"] == 1
    assert summary["blocked_fetch_count"] == 1
    assert entry["ready_for_content_fetch"] is False
    assert entry["llm_source_critic_decision"] == "not_task_relevant"
    assert any(
        item["item_type"] == "source_critic_blocked_source"
        for item in result["human_review_queue"]
    )


def test_runtime_env_preserves_empty_source_critic_allowlist():
    from data_collection_workflow.runtime_profile import workflow_run_env

    env = workflow_run_env(
        source_id_allowlist=["src_seed"],
        source_id_allowlist_enabled=True,
        llm_source_critic_source_id_allowlist=[],
    )

    assert env["SOURCE_ID_ALLOWLIST"] == "src_seed"
    assert env["LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST"] == ""
    assert env["LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH"] == "true"


def test_runtime_config_does_not_fallback_critic_allowlist_to_workflow_ids():
    from data_collection_workflow.runtime_profile import (
        default_workflow_run_config,
        workflow_run_env_from_config,
    )

    config = default_workflow_run_config()
    config["source_sets"]["workflow_source_ids"] = ["src_seed"]
    config["source_sets"]["source_id_allowlist_enabled"] = True
    config["source_sets"]["llm_source_critic_source_ids"] = []

    env = workflow_run_env_from_config(config)

    assert env["SOURCE_ID_ALLOWLIST"] == "src_seed"
    assert env["LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST"] == ""
