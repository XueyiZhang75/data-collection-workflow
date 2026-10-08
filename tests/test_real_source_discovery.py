"""Stage 5 tests for controlled real source discovery/search providers."""

from __future__ import annotations

import importlib.util
import importlib
import json
import sys
from pathlib import Path

from synthetic_workflow_inputs import write_workflow_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


SEARCH_ENV_KEYS = [
    "ENABLE_LIVE_SEARCH",
    "ENABLE_ITERATIVE_SOURCE_DISCOVERY",
    "ITERATIVE_SEARCH_ALLOW_DETERMINISTIC_FALLBACK",
    "SEARCH_MODE",
    "SEARCH_PROVIDER",
    "SEARCH_FIXTURE_PATH",
    "SEARCH_MAX_QUERIES",
    "SEARCH_MAX_RESULTS_PER_QUERY",
    "SEARCH_MAX_TOTAL_RESULTS",
    "SEARCH_TIMEOUT_SECONDS",
    "SEARCH_COMBINE_WITH_SEED_CATALOG",
    "SEARCH_PROVIDER_CHANNEL_ALLOWLIST",
    "AUTHORITY_GAP_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_RETRY_RESULT_BUDGET",
    "AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES",
]


def _clear_search_env(monkeypatch) -> None:
    for key in SEARCH_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _state_for(disease: str, location: str, year: str) -> dict:
    return {
        "structured_task": {
            "disease": disease,
            "location": location,
            "start_date": year,
            "end_date": year,
            "target_fields": [
                "cases_confirmed",
                "cases_unspecified",
                "deaths",
                "date_reported",
                "source_url",
                "source_type",
                "evidence_quote",
            ],
            "collection_mode": "standard",
            "user_request": f"Collect {disease} data for {location} in {year}.",
            "run_label": f"stage5_{disease}_{location}_{year}".replace(" ", "_"),
        },
        "collection_trace": [],
    }


def _run_to_source_discovery(state: dict) -> dict:
    from data_collection_workflow.nodes.source_discovery import source_discovery
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        executable_source_planning,
        profile_and_schema_setup,
        query_strategy_builder,
        task_intake_and_scope_planning,
    )

    state.update(task_intake_and_scope_planning(state))
    state.update(disease_intelligence_builder(state))
    state.update(profile_and_schema_setup(state))
    state.update(executable_source_planning(state))
    state.update(query_strategy_builder(state))
    state.update(source_discovery(state))
    return state


def _write_search_fixture(
    tmp_path: Path,
    *,
    disease: str,
    location: str,
    year: str,
    invalid_results: bool = False,
) -> Path:
    result = {
        "title": f"{location} {disease} surveillance {year}",
        "url": "https://health.example.gov/surveillance/update",
        "snippet": f"Official {disease} cases and deaths in {location} during {year}.",
        "published_date": f"{year}-06-01",
        "source": f"{location} Department of Health",
        "rank": 1,
    }
    results = [result]
    if invalid_results:
        results.extend([
            {**result, "url": result["url"] + "#summary"},
            {**result, "url": "ftp://health.example.gov/report"},
            {**result, "url": ""},
            {**result, "url": "https://health.example.gov/empty", "title": "", "snippet": ""},
        ])
    path = tmp_path / "search_results.json"
    path.write_text(json.dumps({"results": results}), encoding="utf-8")
    return path


def test_direct_verified_target_sufficiency_requires_all_target_weeks():
    import importlib

    from data_collection_workflow.models import SourceCandidate

    source_discovery_module = importlib.import_module(
        "data_collection_workflow.nodes.source_discovery"
    )

    task = {
        "disease": "FLU",
        "location": "United States",
        "start_date": "2024-09-29",
        "end_date": "2024-10-12",
        "collection_mode": "direct_collection",
    }
    week_40 = SourceCandidate(
        source_id="src_cdc_week_40",
        url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        title="CDC FluView Week 40, 2024",
        publisher="CDC",
        source_type="official_public_health_agency",
        discovery_method="live_search_result",
    )
    week_41 = SourceCandidate(
        source_id="src_cdc_week_41",
        url="https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
        canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
        title="CDC FluView Week 41, 2024",
        publisher="CDC",
        source_type="official_public_health_agency",
        discovery_method="live_search_result",
    )

    insufficient, one_week = source_discovery_module._verified_target_search_sufficient(
        [week_40],
        task,
    )
    sufficient, two_weeks = source_discovery_module._verified_target_search_sufficient(
        [week_40, week_41],
        task,
    )

    assert insufficient is False
    assert one_week["covered_target_weeks"] == [40]
    assert sufficient is True
    assert two_weeks["covered_target_weeks"] == [40, 41]


def test_subnational_target_verification_requires_geography_signal():
    import importlib

    from data_collection_workflow.models import SourceCandidate

    source_discovery_module = importlib.import_module(
        "data_collection_workflow.nodes.source_discovery"
    )
    task = {
        "disease": "FLU",
        "location": "VIRGINIA",
        "start_date": "2024-10-06",
        "end_date": "2024-10-12",
        "collection_mode": "direct_collection",
    }
    candidates = [
        SourceCandidate(
            source_id="src_cdc_week41_national",
            title=(
                "Weekly US Influenza Surveillance Report: Key Updates for "
                "Week 41, ending October 12, 2024 | FluView | CDC"
            ),
            url="https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
            canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
            publisher="CDC",
            source_type="official_public_health_agency",
            snippet="United States seasonal influenza activity for Week 41.",
            discovery_method="fixture_search_result",
        ),
        SourceCandidate(
            source_id="src_vdh_week41",
            title="Virginia Weekly Respiratory Disease Surveillance Report Week 41 2024",
            url=(
                "https://www.vdh.virginia.gov/content/uploads/sites/3/"
                "2024/10/Weekly-RDS-Report_Week-41.pdf"
            ),
            canonical_url=(
                "https://www.vdh.virginia.gov/content/uploads/sites/3/"
                "2024/10/Weekly-RDS-Report_Week-41.pdf"
            ),
            publisher="Virginia Department of Health",
            source_type="official_public_health_agency",
            snippet="Virginia influenza-like illness activity for Week 41.",
            discovery_method="fixture_search_result",
        ),
    ]

    verification = source_discovery_module._verify_target_sources(candidates, task)

    assert verification["verified_target_source_ids"] == ["src_vdh_week41"]
    assert any(
        "lacks verified target geography evidence" in reason
        for reason in verification["target_source_miss_reasons"]
    )


def test_target_verification_excludes_explicit_validation_role_sources():
    import importlib

    from data_collection_workflow.models import SourceCandidate

    source_discovery_module = importlib.import_module(
        "data_collection_workflow.nodes.source_discovery"
    )
    task = {
        "disease": "FLU",
        "location": "United States",
        "start_date": "2024-09-29",
        "end_date": "2024-10-05",
        "collection_mode": "direct_collection",
    }
    candidates = [
        SourceCandidate(
            source_id="src_cdc_week40_collection",
            title=(
                "Weekly US Influenza Surveillance Report: Key Updates for "
                "Week 40, ending October 5, 2024 | FluView | CDC"
            ),
            url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
            canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
            publisher="CDC",
            source_type="official_public_health_agency",
            snippet="United States seasonal influenza activity for Week 40.",
            discovery_method="fixture_search_result",
            role_hint="collection",
        ),
        SourceCandidate(
            source_id="src_cdc_week40_validation",
            title=(
                "Validation summary for United States influenza Week 40, "
                "ending October 5, 2024"
            ),
            url="https://www.cdc.gov/validation/flu-week-40-2024.html",
            canonical_url="https://www.cdc.gov/validation/flu-week-40-2024.html",
            publisher="CDC",
            source_type="official_public_health_agency",
            snippet=(
                "United States influenza validation context for Week 40, "
                "not a primary collection report."
            ),
            discovery_method="fixture_search_result",
            role_hint="validation",
        ),
    ]

    verification = source_discovery_module._verify_target_sources(candidates, task)

    assert verification["verified_target_source_ids"] == ["src_cdc_week40_collection"]
    assert any(
        "not a target collection source" in reason
        for reason in verification["target_source_miss_reasons"]
    )


def _enable_fixture_search(
    monkeypatch,
    tmp_path: Path,
    *,
    disease: str = "COVID-19",
    location: str = "New York",
    year: str = "2024",
    invalid_results: bool = False,
) -> None:
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    fixture_path = _write_search_fixture(
        tmp_path, disease=disease, location=location, year=year,
        invalid_results=invalid_results,
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "3")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "5")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "15")
    monkeypatch.setenv("SEARCH_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "true")


def _search_candidates(result: dict) -> list[dict]:
    return [
        candidate
        for candidate in result.get("source_candidates") or []
        if candidate.get("discovery_method") in {"fixture_search_result", "live_search_result"}
    ]


def _run_full_graph_from_config(config_path: Path) -> dict:
    from data_collection_workflow.graph import build_graph
    from data_collection_workflow.workflow_run_config import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(config_path)
    config["source_search"]["combine_with_seed_catalog"] = True
    env_updates = workflow_run_env_from_config(config)
    assert env_updates["ENABLE_LIVE_FETCH"] == "false"
    assert env_updates["ENABLE_LLM_SOURCE_PLANNING"] == "false"
    assert env_updates["ENABLE_LLM_SOURCE_CRITIC"] == "false"
    assert env_updates["ENABLE_LLM_EXTRACTION"] == "false"
    assert env_updates["SEARCH_MODE"] == "fixture"
    with temporary_workflow_env(env_updates):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_search_provider_abstraction_and_fixture_provider_exist():
    assert importlib.util.find_spec("data_collection_workflow.search_providers") is not None

    from data_collection_workflow.search_providers import FixtureSearchProvider, SearchProvider

    assert SearchProvider is not None
    assert FixtureSearchProvider is not None


def test_search_disabled_preserves_offline_seed_catalog_behavior(monkeypatch):
    _clear_search_env(monkeypatch)

    result = _run_to_source_discovery(_state_for("hantavirus", "New Mexico", "2024"))

    discovery = result.get("source_discovery_summary") or {}
    search_summary = result.get("source_search_execution_summary") or {}
    assert discovery.get("discovery_method") == "offline_seed_catalog"
    assert search_summary
    assert search_summary["search_enabled"] is False
    assert search_summary["live_search_enabled"] is False
    assert search_summary["fixture_search_enabled"] is False
    assert search_summary["executed_query_count"] == 0
    assert search_summary["candidate_from_search_count"] == 0
    assert {c.get("discovery_method") for c in result.get("source_candidates") or []} == {
        "offline_seed_catalog"
    }


def test_fixture_search_provider_executes_planned_queries(monkeypatch, tmp_path):
    _enable_fixture_search(monkeypatch, tmp_path)

    result = _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))

    summary = result.get("source_search_execution_summary") or {}
    search_candidates = _search_candidates(result)
    assert summary["fixture_search_enabled"] is True
    assert summary["executed_query_count"] > 0
    assert summary["raw_search_result_count"] > 0
    assert summary["candidate_from_search_count"] > 0
    assert search_candidates


def test_iterative_llm_failure_falls_back_to_existing_query_inventory(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "3")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "5")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "15")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("ENABLE_ITERATIVE_SOURCE_DISCOVERY", "true")
    monkeypatch.setenv("ITERATIVE_SEARCH_ALLOW_DETERMINISTIC_FALLBACK", "false")
    fixture_path = tmp_path / "tb_india_results.json"
    fixture_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "title": "India TB Report 2023 annual tuberculosis statistics",
                        "url": "https://tbcindia.gov.in/reports/india-tb-report-2023",
                        "snippet": (
                            "Official India tuberculosis annual surveillance "
                            "statistics for 2023 include incidence, notified "
                            "cases, mortality, and treatment metrics."
                        ),
                        "published_date": "2024-03-01",
                        "source": "National TB Elimination Programme India",
                        "rank": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.agents import iterative_source_discovery_agent

    def _raise_initial_plan(*args, **kwargs):  # noqa: ARG001
        raise ValueError("simulated iterative planner outage")

    monkeypatch.setattr(
        iterative_source_discovery_agent,
        "plan_initial_search_iteration",
        _raise_initial_plan,
    )
    state = _state_for("Tuberculosis", "India", "2023")
    state["structured_task"].update(
        {
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
            "collection_mode": "direct_collection",
            "user_request": (
                "Collect Tuberculosis public health metrics for India in 2023."
            ),
        }
    )

    result = _run_to_source_discovery(state)

    search_summary = result.get("source_search_execution_summary") or {}
    assert search_summary["planned_query_count"] > 0
    assert search_summary["executed_query_count"] > 0
    assert search_summary["candidate_from_search_count"] > 0
    assert search_summary["stop_decision"] == "fallback_to_existing_queries"
    assert any(
        warning.startswith("iterative_llm_initial_plan_failed")
        for warning in search_summary.get("warnings") or []
    )


def test_source_discovery_executes_search_query_inventory_when_plan_is_empty(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "2")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "5")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "10")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    fixture_path = tmp_path / "measles_virginia_results.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "match_terms": ["measles", "virginia", "2023"],
                        "provider_channels": ["web_search", "official_site_search"],
                        "results": [
                            {
                                "title": "Virginia measles annual cases 2023",
                                "url": "https://www.vdh.virginia.gov/measles/2023-cases",
                                "snippet": (
                                    "Virginia Department of Health annual measles "
                                    "case data for 2023."
                                ),
                                "published_date": "2024-01-15",
                                "source": "Virginia Department of Health",
                                "rank": 1,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "Measles",
            "location": "Virginia",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "Measles",
            "geography": "Virginia",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "collection_mode": "direct_collection",
        },
        "agentic_source_plan": {"planned_queries": []},
        "search_query_inventory": [
            {
                "query_id": "q_inventory_001",
                "query": "measles Virginia 2023 annual cases health department",
                "provider_channel": "web_search",
                "query_type": "general_web",
                "source_type": "official_public_health_agency",
                "role_hint": "collection",
                "priority": 1,
                "expected_fields": ["cases", "date", "location", "source_url"],
                "query_source": "query_strategy_inventory",
            }
        ],
        "collection_trace": [],
    }

    result = source_discovery(state)

    search_summary = result.get("source_search_execution_summary") or {}
    assert search_summary["planned_query_count"] == 1
    assert search_summary["executed_query_count"] == 1
    assert search_summary["candidate_from_search_count"] == 1
    assert result["source_candidates"][0]["query_id"] == "q_inventory_001"


def test_source_discovery_reports_query_generation_failure_for_requirements(
    tmp_path,
    monkeypatch,
):
    _enable_fixture_search(monkeypatch, tmp_path)
    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "Measles",
            "location": "Virginia",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "Measles",
            "geography": "Virginia",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "collection_mode": "direct_collection",
        },
        "source_coverage_requirements": [
            {
                "requirement_id": "virginia_measles_annual_2023",
                "disease": "Measles",
                "geography": "Virginia",
                "period_start": "2023-01-01",
                "period_end": "2023-12-31",
            }
        ],
        "agentic_source_plan": {"planned_queries": []},
        "search_query_inventory": [],
        "collection_trace": [],
    }

    result = source_discovery(state)

    search_summary = result.get("source_search_execution_summary") or {}
    assert search_summary["planned_query_count"] == 0
    assert search_summary["executed_query_count"] == 0
    assert search_summary["stop_decision"] == "query_generation_failed_for_requirements"
    assert "query_generation_failed_for_requirements" in search_summary["warnings"]


def test_direct_collection_non_hantavirus_search_does_not_mix_hantavirus_seed_catalog(
    tmp_path,
    monkeypatch,
):
    _enable_fixture_search(monkeypatch, tmp_path)
    state = _state_for("FLU", "California", "2024")
    state["structured_task"]["collection_mode"] = "direct_collection"

    result = _run_to_source_discovery(state)

    discovery = result.get("source_discovery_summary") or {}
    search_candidates = _search_candidates(result)
    assert discovery["search_enabled"] is True
    assert discovery["candidate_from_seed_count"] == 0
    assert discovery["discovery_method"] == "fixture_search_only"
    assert "offline_seed_catalog" not in {
        candidate.get("discovery_method")
        for candidate in result.get("source_candidates") or []
    }
    assert search_candidates == []
    assert "wrong_disease_result" in {
        row.get("rejection_reason")
        for row in result.get("source_search_results") or []
    }


def test_direct_collection_searches_to_validate_generated_official_candidate(
    tmp_path,
    monkeypatch,
):
    _enable_fixture_search(monkeypatch, tmp_path)
    monkeypatch.setenv("DIRECT_FAST_STOP_ON_VERIFIED_TARGET", "true")
    state = _state_for("FLU", "United States", "2024")
    state["structured_task"].update(
        {
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
            "user_request": "Collect FLU surveillance data for United States from 2024-09-29 to 2024-10-05.",
        }
    )

    result = _run_to_source_discovery(state)

    search_summary = result.get("source_search_execution_summary") or {}
    discovery_summary = result.get("source_discovery_summary") or {}
    official_candidates = result.get("official_coverage_candidates") or []
    assert official_candidates
    assert all(candidate.get("must_fetch") is True for candidate in official_candidates)
    assert all(candidate.get("coverage_requirement_ids") for candidate in official_candidates)
    assert {
        candidate.get("triage_role") for candidate in official_candidates
    } == {"predicted_target_candidate"}
    assert search_summary["executed_query_count"] > 0
    assert search_summary["search_stopped_reason"] != "verified_target_source_found"
    assert search_summary["verified_target_source_count"] == 0
    assert search_summary["predicted_target_candidate_count"] >= 1
    assert search_summary["search_verified_target_source_count"] == 0
    assert discovery_summary["search_stopped_reason"] != "verified_target_source_found"
    assert discovery_summary["verified_target_source_count"] == 0
    assert discovery_summary["predicted_target_candidate_count"] >= 1


def test_direct_collection_reports_skipped_search_validation_when_search_disabled(
    monkeypatch,
):
    _clear_search_env(monkeypatch)
    state = _state_for("FLU", "United States", "2024")
    state["structured_task"].update(
        {
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        }
    )

    result = _run_to_source_discovery(state)

    search_summary = result.get("source_search_execution_summary") or {}
    discovery_summary = result.get("source_discovery_summary") or {}
    assert discovery_summary["predicted_target_candidate_count"] >= 1
    assert discovery_summary["verified_target_source_count"] == 0
    assert "search_validation_skipped_search_disabled" in (
        search_summary.get("warnings") or []
    )
    assert "search_validation_skipped_search_disabled" in (
        discovery_summary.get("warnings") or []
    )


def test_direct_collection_partial_week_coverage_does_not_stop_as_sufficient(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("DIRECT_FAST_STOP_ON_VERIFIED_TARGET", "true")
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "4")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "5")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "20")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    fixture_path = tmp_path / "partial_virginia_week42_only.json"
    fixture_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "title": "VDH Weekly Respiratory Disease Surveillance Report Week 42",
                        "url": "https://www.vdh.virginia.gov/content/uploads/sites/3/2024/10/2024-25_Weekly-RDS-Report_Week-42.pdf",
                        "snippet": "Virginia Department of Health Week 42 report for October 13 - October 19, 2024.",
                        "published_date": "2024-10-19",
                        "source": "Virginia Department of Health",
                        "rank": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))
    state = _state_for("FLU", "Virginia", "2024")
    state["structured_task"].update(
        {
            "start_date": "2024-10-11",
            "end_date": "2024-11-01",
            "collection_mode": "direct_collection",
            "user_request": "Collect FLU surveillance data for Virginia from 2024-10-11 to 2024-11-01.",
        }
    )
    result = _run_to_source_discovery(state)

    discovery_summary = result["source_discovery_summary"]
    search_summary = result["source_search_execution_summary"]
    assert discovery_summary["verified_target_source_count"] == 1
    assert discovery_summary["search_stopped_reason"] != "verified_target_source_found"
    assert search_summary["search_stopped_reason"] != "verified_target_source_found"
    assert set(search_summary["target_source_miss_reasons"])


def test_fixture_covid19_search_candidates_are_disease_specific(monkeypatch, tmp_path):
    _enable_fixture_search(monkeypatch, tmp_path)

    result = _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))
    text = "\n".join(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "snippet", "query_used", "publisher", "url")
        )
        for candidate in _search_candidates(result)
    ).lower()

    assert any(term in text for term in ("covid-19", "sars-cov-2", "new york", "2024"))
    assert "hantavirus pulmonary syndrome" not in text
    assert not any(candidate.get("seed_source_id") for candidate in _search_candidates(result))


def test_fixture_dengue_search_candidates_are_disease_specific(monkeypatch, tmp_path):
    _enable_fixture_search(monkeypatch, tmp_path, disease="dengue", location="Florida", year="2025")

    result = _run_to_source_discovery(_state_for("dengue", "Florida", "2025"))
    text = "\n".join(
        " ".join(
            str(candidate.get(key) or "")
            for key in ("title", "snippet", "query_used", "publisher", "url")
        )
        for candidate in _search_candidates(result)
    ).lower()

    assert any(term in text for term in ("dengue", "denv", "florida", "2025"))
    assert "hantavirus pulmonary syndrome" not in text
    assert not any(candidate.get("seed_source_id") for candidate in _search_candidates(result))


def test_search_result_url_validation_and_deduplication(monkeypatch, tmp_path):
    _enable_fixture_search(monkeypatch, tmp_path, invalid_results=True)

    result = _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))
    summary = result.get("source_search_execution_summary") or {}
    rejection_counts = summary.get("rejection_reason_counts") or {}
    candidates = _search_candidates(result)
    canonical_urls = [candidate.get("canonical_url") for candidate in candidates]

    assert candidates
    assert len(canonical_urls) == len(set(canonical_urls))
    assert rejection_counts.get("duplicate_url", 0) >= 1
    assert rejection_counts.get("unsupported_scheme", 0) >= 1
    assert rejection_counts.get("missing_url", 0) >= 1
    assert rejection_counts.get("empty_title_and_snippet", 0) >= 1


def test_query_and_result_limits_are_enforced(monkeypatch, tmp_path):
    from data_collection_workflow.nodes.source_discovery import source_discovery

    _enable_fixture_search(monkeypatch, tmp_path)
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "2")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "3")
    planned_queries = [
        {
            "query_id": f"q_limit_{index}",
            "query": f"COVID-19 New York 2024 official surveillance report {index}",
            "provider_channel": "web_search",
            "query_type": "general_web",
            "source_type": "official_public_health_agency",
            "role_hint": "collection",
            "priority": index,
        }
        for index in range(1, 4)
    ]
    # Each query can return four distinct URLs, so both result limits must bind.
    fixture = {
        "queries": [
            {
                "query_ids": [query["query_id"]],
                "results": [
                    {
                        "title": "New York COVID-19 surveillance 2024",
                        "url": f"https://health.example.gov/{query['query_id']}/{rank}",
                        "snippet": "Official COVID-19 cases in New York during 2024.",
                        "source": "New York Department of Health",
                    }
                    for rank in range(1, 5)
                ],
            }
            for query in planned_queries
        ]
    }
    (tmp_path / "search_results.json").write_text(json.dumps(fixture), encoding="utf-8")
    state = _state_for("COVID-19", "New York", "2024")
    state["agentic_source_plan"] = {"planned_queries": planned_queries}

    result = source_discovery(state)
    summary = result.get("source_search_execution_summary") or {}
    records = summary.get("query_execution_records") or []
    statuses = {record.get("execution_status") for record in records}

    assert summary["executed_query_count"] == 2
    assert summary["candidate_from_search_count"] == 3
    assert summary["raw_search_result_count"] == 8
    assert summary["skipped_query_count"] > 0
    assert "skipped_query_limit" in statuses or "skipped_total_result_limit" in statuses
    assert [record["result_count"] for record in records if record["execution_status"] == "executed"] == [2, 1]
    assert summary["rejection_reason_counts"]["result_limit_reached"] == 1
    assert {candidate["canonical_url"] for candidate in _search_candidates(result)} == {
        "https://health.example.gov/q_limit_1/1",
        "https://health.example.gov/q_limit_1/2",
        "https://health.example.gov/q_limit_2/1",
    }


def test_outbreak_query_budget_keeps_database_and_literature_under_cap(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "5")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "1")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "10")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    fixture_path = tmp_path / "authority_budget_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_news_1"],
                        "results": [
                            {
                                "title": "News outbreak story",
                                "url": "https://example-news.org/hantavirus-story",
                                "snippet": "Media report on a multi-country hantavirus outbreak.",
                                "source": "Example News",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_news_2"],
                        "results": [
                            {
                                "title": "Second news story",
                                "url": "https://example-news.org/hantavirus-followup",
                                "snippet": "Follow-up media report.",
                                "source": "Example News",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_international"],
                        "results": [
                            {
                                "title": "WHO outbreak update",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
                                "snippet": "International official outbreak update.",
                                "source": "WHO",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_national"],
                        "results": [
                            {
                                "title": "National public health update",
                                "url": "https://www.canada.ca/en/public-health/services/example.html",
                                "snippet": "National public health authority update.",
                                "source": "Public Health Agency of Canada",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_database"],
                        "results": [
                            {
                                "title": "Pathogen sequence database",
                                "url": "https://pathoplexus.org/pathogen/hantavirus",
                                "snippet": "Structured genomic database records.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_literature"],
                        "results": [
                            {
                                "title": "Peer reviewed case report",
                                "url": "https://www.nejm.org/doi/full/example",
                                "snippet": "Peer-reviewed outbreak case report.",
                                "source": "NEJM",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_news_1",
                    "query": "hantavirus outbreak news 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_news_2",
                    "query": "hantavirus cruise ship media 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_news_3",
                    "query": "hantavirus travel associated article 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_news_4",
                    "query": "hantavirus latest media report 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_international",
                    "query": "hantavirus WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_national",
                    "query": "hantavirus national public health confirmed cases 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_database",
                    "query": "hantavirus sequence outbreak database 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_literature",
                    "query": "hantavirus case report outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_literature_extra",
                    "query": "hantavirus case series outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_literature_extra_2",
                    "query": "hantavirus journal outbreak report 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    selected = [
        record
        for record in summary["query_execution_records"]
        if record.get("selected_for_execution")
    ]
    selected_ids = {record["query_id"] for record in selected}
    selected_buckets = {record.get("selection_bucket") for record in selected}

    assert summary["selected_query_count"] == 5
    assert summary["candidate_from_search_count"] >= 4
    assert {"q_database", "q_literature"}.issubset(selected_ids)
    assert "structured_database" in selected_buckets
    assert "peer_reviewed_literature" in selected_buckets
    assert sum(1 for record in selected if record["source_type"] == "news_and_situation_report") <= 2
    assert summary["selected_query_count_by_source_class"]["structured_database"] >= 1
    assert summary["selected_query_count_by_source_class"]["peer_reviewed_literature"] >= 1
    assert summary["skipped_high_trust_query_count"] >= 1


def test_outbreak_query_budget_prefers_known_authority_domain_over_generic_database(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "4")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "1")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    fixture_path = tmp_path / "authority_domain_floor_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_international"],
                        "results": [
                            {
                                "title": "WHO outbreak update",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
                                "snippet": "International official outbreak update.",
                                "source": "WHO",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_national"],
                        "results": [
                            {
                                "title": "National public health update",
                                "url": "https://www.canada.ca/en/public-health/services/example.html",
                                "snippet": "National public health authority update.",
                                "source": "Public Health Agency of Canada",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_generic_database"],
                        "results": [
                            {
                                "title": "General genomic discussion",
                                "url": "https://www.congress.gov/crs_external_products/IF/PDF/IF13230/IF13230.1.pdf",
                                "snippet": "Context document mentioning genomic epidemiology.",
                                "source": "Congress",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_pathoplexus"],
                        "results": [
                            {
                                "title": "Pathoplexus Andes virus sequence records",
                                "url": "https://pathoplexus.org/andv/search",
                                "snippet": "Structured pathogen sequence database records.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_nejm"],
                        "results": [
                            {
                                "title": "Peer reviewed outbreak report",
                                "url": "https://www.nejm.org/doi/full/example",
                                "snippet": "Peer-reviewed outbreak case report.",
                                "source": "NEJM",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_international",
                    "query": "hantavirus WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_national",
                    "query": "hantavirus national public health confirmed cases 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_generic_database",
                    "query": "hantavirus genomic epidemiology outbreak data 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_nejm",
                    "query": "hantavirus site:nejm.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    selected_ids = {
        record["query_id"]
        for record in summary["query_execution_records"]
        if record.get("selected_for_execution")
    }

    assert "q_pathoplexus" in selected_ids
    assert "q_nejm" in selected_ids
    assert "q_generic_database" not in selected_ids
    assert summary["selected_known_authority_domain_query_count"] >= 2
    assert summary["skipped_known_authority_domain_query_count"] == 0


def test_authority_gap_retry_generates_jurisdiction_queries_without_benchmark_inputs(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "6")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "2")
    fixture_path = tmp_path / "authority_gap_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_news"],
                        "results": [
                            {
                                "title": "Travel-associated hantavirus cluster in Canada and Spain",
                                "url": "https://example-news.org/hantavirus-canada-spain",
                                "snippet": (
                                    "A cruise ship outbreak involved passengers in Canada "
                                    "and Spain; public health agencies are monitoring cases."
                                ),
                                "source": "Example News",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:canada.ca"],
                        "results": [
                            {
                                "title": "Public health update on hantavirus cases",
                                "url": "https://www.canada.ca/en/public-health/services/hantavirus-update-2026.html",
                                "snippet": "Official Canadian public health update for confirmed cases.",
                                "source": "Public Health Agency of Canada",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sanidad.gob.es"],
                        "results": [
                            {
                                "title": "Spain Ministry of Health hantavirus update",
                                "url": "https://www.sanidad.gob.es/example/hantavirus-2026.pdf",
                                "snippet": "Spanish Ministry of Health official update.",
                                "source": "Ministerio de Sanidad",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_news",
                    "query": "hantavirus cruise ship outbreak news 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                }
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_queries_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    urls = {candidate.get("canonical_url") for candidate in _search_candidates(result)}

    assert "authority_source_recall_incomplete" in summary["warnings"]
    assert "authority_gap_retry_attempted" in summary["warnings"]
    assert "authority_gap_retry_candidates_added" in summary["warnings"]
    assert summary["authority_gap_retry_generated_query_count"] > 0
    assert summary["authority_gap_retry_accepted_candidate_count"] > 0
    assert retry_records
    assert "github" not in retry_queries_text
    assert "gh_id" not in retry_queries_text
    assert any("canada.ca" in str(url) for url in urls)


def test_authority_gap_retry_runs_when_aggregate_authority_misses_event_jurisdictions(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "4")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "16")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "4")
    fixture_path = tmp_path / "authority_jurisdiction_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_who"],
                        "results": [
                            {
                                "title": "WHO multi-country hantavirus outbreak",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
                                "snippet": "WHO reports a cruise-ship related outbreak with affected travellers in Canada and Spain.",
                                "source": "WHO",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_ecdc"],
                        "results": [
                            {
                                "title": "ECDC Andes hantavirus outbreak update",
                                "url": "https://www.ecdc.europa.eu/en/infectious-disease-topics/hantavirus-infection/example",
                                "snippet": "ECDC monitors the same event involving Canada and Spain.",
                                "source": "ECDC",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_pathoplexus"],
                        "results": [
                            {
                                "title": "Pathoplexus Andes virus records",
                                "url": "https://pathoplexus.org/andv/search",
                                "snippet": "Structured genomic database records.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_eurosurveillance"],
                        "results": [
                            {
                                "title": "Eurosurveillance outbreak case report",
                                "url": "https://www.eurosurveillance.org/content/example",
                                "snippet": "Peer reviewed public health surveillance report.",
                                "source": "Eurosurveillance",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:canada.ca"],
                        "results": [
                            {
                                "title": "Public health update on hantavirus cases",
                                "url": "https://www.canada.ca/en/public-health/services/hantavirus-update-2026.html",
                                "snippet": "Official Canadian public health update.",
                                "source": "Public Health Agency of Canada",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sanidad.gob.es"],
                        "results": [
                            {
                                "title": "Spain Ministry of Health hantavirus update",
                                "url": "https://www.sanidad.gob.es/example/hantavirus-2026.pdf",
                                "snippet": "Spanish Ministry of Health official update.",
                                "source": "Ministerio de Sanidad",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_who",
                    "query": "hantavirus WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_ecdc",
                    "query": "hantavirus ECDC outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_eurosurveillance",
                    "query": "hantavirus site:eurosurveillance.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    urls = {candidate.get("canonical_url") for candidate in _search_candidates(result)}

    assert summary["detected_event_jurisdictions"] == ["Canada", "Spain"]
    assert summary["missing_event_jurisdictions"] == ["Canada", "Spain"]
    assert summary["jurisdiction_authority_retry_generated_count"] >= 2
    assert summary["jurisdiction_authority_retry_candidate_count"] >= 2
    assert "site:canada.ca" in retry_text
    assert "site:sanidad.gob.es" in retry_text
    assert "github" not in retry_text
    assert "gh_id" not in retry_text
    assert any("canada.ca" in str(url) for url in urls)
    assert any("sanidad.gob.es" in str(url) for url in urls)


def test_authority_retry_has_independent_budget_for_known_domains_and_jurisdictions(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "5")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "1")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "20")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_ENABLED", "true")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "6")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "4")
    monkeypatch.setenv("AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES", "6")
    fixture_path = tmp_path / "split_authority_retry_budget_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_news"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius cruise ship",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": (
                                    "The outbreak involved Canada, France, Netherlands, "
                                    "Spain and United Kingdom passengers. Public health "
                                    "officials reported cases and deaths in 2026."
                                ),
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:canada.ca"],
                        "results": [
                            {
                                "title": "Canada public health hantavirus outbreak update",
                                "url": "https://www.canada.ca/en/public-health/services/hantavirus-update-2026.html",
                                "snippet": "Official Canadian public health update.",
                                "source": "Government of Canada",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sante.gouv.fr"],
                        "results": [
                            {
                                "title": "France point de situation hantavirus MV Hondius",
                                "url": "https://sante.gouv.fr/soins-et-maladies/maladies/maladies-infectieuses/article/hantavirus-point-de-situation-2026",
                                "snippet": "Point de situation officiel pour le navire MV Hondius.",
                                "source": "French Ministry of Health",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:rivm.nl"],
                        "results": [
                            {
                                "title": "Arrival and cleaning of cruise ship Hondius",
                                "url": "https://www.rivm.nl/en/news/arrival-and-cleaning-of-cruise-ship-hondius",
                                "snippet": "RIVM update about the cruise ship Hondius.",
                                "source": "RIVM",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sanidad.gob.es"],
                        "results": [
                            {
                                "title": "Brote de hantavirus informe de cierre",
                                "url": "https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/hantavirus/informe-cierre-2026.pdf",
                                "snippet": "Informe oficial de casos del Ministerio de Sanidad.",
                                "source": "Ministerio de Sanidad",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:gov.uk"],
                        "results": [
                            {
                                "title": "UKHSA outbreaks under monitoring hantavirus",
                                "url": "https://www.gov.uk/government/publications/outbreaks-under-monitoring/outbreaks-under-monitoring-week-19-2026",
                                "snippet": "UKHSA official monitoring update.",
                                "source": "UK Health Security Agency",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_news",
                    "query": "hantavirus MV Hondius cruise ship outbreak countries 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_nejm",
                    "query": "hantavirus site:nejm.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_euro",
                    "query": "hantavirus site:eurosurveillance.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_science",
                    "query": "hantavirus site:science.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    generated_by_domain = summary.get("jurisdiction_retry_generated_by_domain") or {}
    executed_by_domain = summary.get("jurisdiction_retry_executed_by_domain") or {}

    assert "site:sanidad.gob.es" in retry_text
    assert "site:gov.uk" in retry_text
    assert generated_by_domain["sanidad.gob.es"] >= 1
    assert generated_by_domain["gov.uk"] >= 1
    assert executed_by_domain["sanidad.gob.es"] >= 1
    assert executed_by_domain["gov.uk"] >= 1
    assert summary["known_domain_retry_generated_count"] >= 3
    assert summary["jurisdiction_authority_retry_generated_count"] >= 5
    assert "github" not in retry_text
    assert "gh_id" not in retry_text


def test_authority_retry_protects_jurisdiction_queries_from_broad_domain_result_cap(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "4")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "4")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "5")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "4")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "4")
    monkeypatch.setenv("AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES", "4")
    fixture_path = tmp_path / "jurisdiction_retry_protected_from_result_cap.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_news"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": (
                                    "The cruise ship outbreak involved Canada, France, "
                                    "Netherlands and Spain in 2026."
                                ),
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {"query_ids": ["q_pathoplexus_initial"], "results": []},
                    {"query_ids": ["q_pubmed_initial"], "results": []},
                    {"query_ids": ["q_europepmc_initial"], "results": []},
                    {
                        "match_terms": ["sequence outbreak"],
                        "results": [
                            {
                                "title": f"Pathoplexus Andes virus record {idx}",
                                "url": f"https://pathoplexus.org/seq/PP_TEST_{idx}",
                                "snippet": "Structured database record for Andes virus.",
                                "source": "Pathoplexus",
                            }
                            for idx in range(1, 5)
                        ],
                    },
                    {
                        "match_terms": ["site:canada.ca"],
                        "results": [
                            {
                                "title": "Canada public health hantavirus update",
                                "url": "https://www.canada.ca/en/public-health/services/hantavirus-update-2026.html",
                                "snippet": "Official Canadian public health update.",
                                "source": "Government of Canada",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sante.gouv.fr"],
                        "results": [
                            {
                                "title": "Hantavirus point de situation MV Hondius",
                                "url": "https://sante.gouv.fr/hantavirus-point-de-situation-2026",
                                "snippet": "French official update for the ship event.",
                                "source": "Ministere de la Sante",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:rivm.nl"],
                        "results": [
                            {
                                "title": "Arrival and cleaning of cruise ship Hondius",
                                "url": "https://www.rivm.nl/en/news/arrival-and-cleaning-of-cruise-ship-hondius",
                                "snippet": "RIVM update about the cruise ship Hondius.",
                                "source": "RIVM",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:sanidad.gob.es"],
                        "results": [
                            {
                                "title": "Brote de hantavirus informe de cierre",
                                "url": "https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/alertasActuales/hantavirus/informe-cierre-2026.pdf",
                                "snippet": "Informe oficial del Ministerio de Sanidad.",
                                "source": "Ministerio de Sanidad",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_news",
                    "query": "hantavirus MV Hondius cruise ship outbreak countries 2026",
                    "provider_channel": "web_search",
                    "query_type": "general_web",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus_initial",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pubmed_initial",
                    "query": "hantavirus site:pubmed.ncbi.nlm.nih.gov outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_europepmc_initial",
                    "query": "hantavirus site:europepmc.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    jurisdiction_retry_records = [
        record
        for record in retry_records
        if record.get("retry_category") == "jurisdiction_authority"
    ]
    executed_domains = summary.get("jurisdiction_retry_executed_by_domain") or {}
    skipped_jurisdiction_domains = {
        record.get("official_domain_hint")
        for record in jurisdiction_retry_records
        if record.get("execution_status") == "skipped_total_result_limit"
    }

    assert summary["jurisdiction_authority_retry_generated_count"] >= 4
    assert executed_domains["canada.ca"] >= 1
    assert executed_domains["sante.gouv.fr"] >= 1
    assert executed_domains["rivm.nl"] >= 1
    assert executed_domains["sanidad.gob.es"] >= 1
    assert not skipped_jurisdiction_domains
    assert summary["jurisdiction_retry_skipped_due_to_result_cap_count"] == 0
    assert summary["known_domain_retry_truncated_before_jurisdiction_complete"] is False
    assert summary["retry_category_round_robin_order"][:3] == [
        "jurisdiction_authority",
        "known_domain_no_result",
        "official_page_family",
    ]


def test_authority_retry_caps_broad_jurisdiction_results_so_structured_domain_executes(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "3")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "6")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "6")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "2")
    monkeypatch.setenv("AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES", "3")
    fixture_path = tmp_path / "retry_domain_result_cap_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_event"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": (
                                    "The cruise ship outbreak involved the "
                                    "Netherlands, the United Kingdom and Spain."
                                ),
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {"query_ids": ["q_pathoplexus_initial"], "results": []},
                    {
                        "match_terms": ["site:rivm.nl"],
                        "results": [
                            {
                                "title": f"RIVM Hondius update {idx}",
                                "url": f"https://www.rivm.nl/en/news/hondius-update-{idx}",
                                "snippet": "RIVM official update for the Hondius event.",
                                "source": "RIVM",
                            }
                            for idx in range(1, 7)
                        ],
                    },
                    {
                        "match_terms": ["site:gov.uk"],
                        "results": [
                            {
                                "title": f"UKHSA outbreak update {idx}",
                                "url": f"https://www.gov.uk/government/publications/hantavirus-update-{idx}",
                                "snippet": "UKHSA official update for the cruise ship event.",
                                "source": "UK Health Security Agency",
                            }
                            for idx in range(1, 7)
                        ],
                    },
                    {
                        "match_terms": ['"Andes virus" site:pathoplexus.org'],
                        "results": [
                            {
                                "title": "Pathoplexus Andes virus sequence record",
                                "url": "https://pathoplexus.org/seq/PP_ANDV_001",
                                "snippet": "Structured Andes virus sequence record.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_event",
                    "query": "hantavirus MV Hondius WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_public_health_agency",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus_initial",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    pathoplexus_records = [
        record
        for record in retry_records
        if record.get("official_domain_hint") == "pathoplexus.org"
        or "site:pathoplexus.org" in str(record.get("query") or "")
    ]

    assert pathoplexus_records
    assert pathoplexus_records[0]["execution_status"] in {"executed", "no_results"}
    assert any(
        str(row.get("rejection_reason") or "")
        in {
            "per_domain_retry_result_cap_reached",
            "per_query_result_cap_reached",
        }
        for row in result["source_search_results"]
    )
    assert any(
        "pathoplexus.org" in str(candidate.get("canonical_url") or "")
        for candidate in _search_candidates(result)
    )


def test_authority_retry_generates_event_anchored_science_query_when_domain_only_has_wrong_page(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "5")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "16")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_ENABLED", "true")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "8")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "6")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_RESULT_BUDGET", "12")
    fixture_path = tmp_path / "science_same_domain_wrong_page_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_who"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius cruise ship",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": (
                                    "The MV Hondius cruise ship outbreak involved "
                                    "Andes virus, hantavirus cases, Canada and France."
                                ),
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_science_wrong"],
                        "results": [
                            {
                                "title": "Briefs | Science | AAAS",
                                "url": "https://www.science.org/news/sifter/first-cat-space-gets-her-due",
                                "snippet": "A science news listing about unrelated research briefs.",
                                "source": "Science",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:science.org", "andes virus", "cruise ship"],
                        "results": [
                            {
                                "title": "Cruise ship's hantavirus outbreak puts researchers in uncharted territory",
                                "url": "https://www.science.org/content/article/cruise-ship-s-hantavirus-outbreak-puts-researchers-uncharted-territory",
                                "snippet": (
                                    "Science reports on the Andes virus outbreak linked "
                                    "to the MV Hondius cruise ship."
                                ),
                                "source": "Science",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus MV Hondius cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_who",
                    "query": "hantavirus MV Hondius WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_science_wrong",
                    "query": "hantavirus site:science.org outbreak 2026",
                    "provider_channel": "literature_api",
                    "query_type": "literature",
                    "source_type": "peer_reviewed_literature",
                    "role_hint": "context",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    urls = {candidate.get("canonical_url") for candidate in _search_candidates(result)}

    assert (
        "site:science.org" in retry_text
        or any(
            "cruise-ship-s-hantavirus-outbreak" in str(url)
            for url in urls
        )
    )
    assert any(
        "cruise-ship-s-hantavirus-outbreak" in str(url)
        for url in urls
    )


def test_peer_literature_retry_specs_include_wrong_page_science_domain():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _peer_literature_retry_specs

    specs = _peer_literature_retry_specs(
        state={
            "structured_task": {
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            }
        },
        search_candidates=[
            SourceCandidate(
                source_id="src_who",
                title="WHO reports Andes virus on MV Hondius cruise ship",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="The MV Hondius cruise ship outbreak involved Andes virus.",
            ),
            SourceCandidate(
                source_id="src_science_wrong",
                title="Briefs | Science | AAAS",
                url="https://www.science.org/news/sifter/first-cat-space-gets-her-due",
                canonical_url="https://www.science.org/news/sifter/first-cat-space-gets-her-due",
                domain="science.org",
                snippet="A science news listing about unrelated research briefs.",
            ),
        ],
        year_suffix="2026",
    )

    science_queries = [
        str(spec.get("query") or "")
        for spec in specs
        if spec.get("official_domain_hint") == "science.org"
    ]
    assert science_queries
    assert "site:science.org" in science_queries[0]
    assert "Andes virus" in science_queries[0]


def test_high_confidence_domain_status_distinguishes_event_page_from_same_domain_only():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _high_confidence_domain_query_status

    statuses = _high_confidence_domain_query_status(
        retry_queries=[],
        query_records=[],
        search_candidates=[
            SourceCandidate(
                source_id="src_science_wrong",
                title="Briefs | Science | AAAS",
                url="https://www.science.org/news/sifter/first-cat-space-gets-her-due",
                canonical_url="https://www.science.org/news/sifter/first-cat-space-gets-her-due",
                domain="science.org",
                snippet="A science news listing about unrelated research briefs.",
            ),
            SourceCandidate(
                source_id="src_who_event",
                title="WHO reports Andes virus on MV Hondius cruise ship",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="The MV Hondius cruise ship outbreak involved Andes virus.",
            ),
        ],
        detected_event_jurisdictions=[],
    )

    assert statuses["science.org"] == "domain_found_exact_missing"
    assert statuses["who.int"] == "exact_event_page_found"


def test_official_page_family_retry_targets_national_same_domain_wrong_pages():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _official_page_family_retry_specs

    specs = _official_page_family_retry_specs(
        state={
            "structured_task": {
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            }
        },
        search_candidates=[
            SourceCandidate(
                source_id="src_who",
                title="WHO reports Andes virus on MV Hondius cruise ship",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="The MV Hondius cruise ship outbreak involved Andes virus and hantavirus cases.",
            ),
            SourceCandidate(
                source_id="src_sante_context",
                title="Sante.gouv.fr current public health information",
                url="https://sante.gouv.fr/soins-et-maladies/maladies",
                canonical_url="https://sante.gouv.fr/soins-et-maladies/maladies",
                domain="sante.gouv.fr",
                snippet="French Ministry of Health public health portal.",
            ),
            SourceCandidate(
                source_id="src_rivm_context",
                title="RIVM current information about hantavirus",
                url="https://www.rivm.nl/en/hantavirus/current-information",
                canonical_url="https://www.rivm.nl/en/hantavirus/current-information",
                domain="rivm.nl",
                snippet="General current information about hantavirus.",
            ),
        ],
        candidate_text=(
            "The MV Hondius cruise ship outbreak involved Andes virus cases in "
            "France and the Netherlands."
        ),
        year_suffix="2026",
    )

    query_text = "\n".join(str(spec.get("query") or "") for spec in specs).lower()
    assert "site:sante.gouv.fr" in query_text
    assert "site:rivm.nl" in query_text
    assert "mv hondius" in query_text or "andes virus" in query_text


def test_official_page_family_retry_targets_missing_registry_authority_domain():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _official_page_family_retry_specs

    specs = _official_page_family_retry_specs(
        state={
            "structured_task": {
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            }
        },
        search_candidates=[
            SourceCandidate(
                source_id="src_who",
                title="WHO reports Andes virus on MV Hondius cruise ship",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="The event involved France, but no French authority page was found.",
            ),
        ],
        candidate_text=(
            "The MV Hondius cruise ship outbreak involved Andes virus cases in France."
        ),
        year_suffix="2026",
    )

    france_specs = [
        spec for spec in specs if spec.get("official_domain_hint") == "sante.gouv.fr"
    ]
    assert france_specs
    assert all("site:sante.gouv.fr" in str(spec.get("query")) for spec in france_specs)


def test_structured_database_retry_preserves_domain_diversity_after_one_database_found():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _structured_database_retry_specs

    specs = _structured_database_retry_specs(
        state={
            "structured_task": {
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            "disease_intelligence": {
                "aliases": ["hantavirus", "Andes virus"],
                "abbreviations": ["ANDV"],
            },
        },
        search_candidates=[
            SourceCandidate(
                source_id="src_who",
                title="WHO Andes virus cruise ship outbreak",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="The investigation includes sequence and phylogenetic analysis.",
            ),
            SourceCandidate(
                source_id="src_ncbi",
                title="Andes virus sequence record",
                url="https://www.ncbi.nlm.nih.gov/nuccore/ABC123",
                canonical_url="https://www.ncbi.nlm.nih.gov/nuccore/ABC123",
                domain="ncbi.nlm.nih.gov",
                source_type="structured_database",
                snippet="A genomic sequence record for Andes virus.",
            ),
        ],
        year_suffix="2026",
    )

    domains = {str(spec.get("official_domain_hint")) for spec in specs}
    assert "pathoplexus.org" in domains
    assert "ncbi.nlm.nih.gov" not in domains


def test_source_recall_target_ledger_distinguishes_domain_only_and_event_page():
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.nodes.source_discovery import _build_source_recall_target_ledger

    ledger = _build_source_recall_target_ledger(
        retry_queries=[
            {
                "query_id": "q_sante",
                "official_domain_hint": "sante.gouv.fr",
                "source_type": "national_public_health_agency",
                "jurisdiction_hint": "France",
                "retry_category": "official_page_family",
            }
        ],
        query_records=[
            {
                "query_id": "q_sante",
                "official_domain_hint": "sante.gouv.fr",
                "execution_status": "executed",
                "selected_for_execution": True,
            }
        ],
        search_candidates=[
            SourceCandidate(
                source_id="src_sante_context",
                title="French Ministry of Health",
                url="https://sante.gouv.fr/",
                canonical_url="https://sante.gouv.fr/",
                domain="sante.gouv.fr",
                snippet="General ministry home page.",
            ),
            SourceCandidate(
                source_id="src_who_event",
                title="Disease Outbreak News: Andes virus on MV Hondius",
                url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                canonical_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                domain="who.int",
                snippet="WHO event report for the cruise ship outbreak.",
            ),
        ],
        detected_event_jurisdictions=["France"],
    )

    by_domain = {row["authority_domain"]: row for row in ledger}
    assert by_domain["sante.gouv.fr"]["event_page_status"] == "domain_found_event_page_missing"
    assert by_domain["who.int"]["event_page_status"] == "event_page_found"


def test_search_result_prefilter_rejects_clear_wrong_disease_result(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "4")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    fixture_path = tmp_path / "wrong_disease_search_result_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_mixed"],
                        "results": [
                            {
                                "title": "Pathoplexus Ebola outbreak response",
                                "url": "https://pathoplexus.org/news/2026-ebola-response",
                                "snippet": "Sequence records for an Ebola virus outbreak.",
                                "source": "Pathoplexus",
                            },
                            {
                                "title": "Pathoplexus Andes virus response",
                                "url": "https://pathoplexus.org/news/2026-andes-virus-response",
                                "snippet": "Andes virus sequence records for a hantavirus outbreak.",
                                "source": "Pathoplexus",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    result = source_discovery(
        {
            "structured_task": {
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
                "collection_mode": "standard",
            },
            "agentic_source_plan": {
                "planned_queries": [
                    {
                        "query_id": "q_mixed",
                        "query": "hantavirus outbreak sequence database 2026",
                        "provider_channel": "database_search",
                        "query_type": "database",
                        "source_type": "structured_database",
                        "role_hint": "collection_support",
                        "execution_status": "planned_not_executed",
                    }
                ]
            },
            "collection_trace": [],
        }
    )

    accepted_urls = {
        str(candidate.get("canonical_url") or "")
        for candidate in _search_candidates(result)
    }
    rejected_reasons = {
        row.get("rejection_reason")
        for row in result["source_search_results"]
        if "ebola" in str(row.get("url") or "").lower()
    }

    assert not any("ebola" in url for url in accepted_urls)
    assert "wrong_disease_result" in rejected_reasons


def test_authority_retry_uses_local_result_budget_after_first_pass_result_cap(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "1")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "1")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_RESULT_BUDGET", "4")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "2")
    monkeypatch.setenv("AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES", "2")
    fixture_path = tmp_path / "retry_local_budget_after_first_pass_cap.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_event"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": (
                                    "The cruise ship outbreak involved Canada, "
                                    "France, the Netherlands and Spain."
                                ),
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:pathoplexus.org"],
                        "results": [
                            {
                                "title": "Pathoplexus Andes virus sequence record",
                                "url": "https://pathoplexus.org/seq/PP_ANDV_BUDGET",
                                "snippet": "Structured Andes virus sequence record.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_event",
                    "query": "hantavirus MV Hondius WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_public_health_agency",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                }
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    pathoplexus_records = [
        record
        for record in retry_records
        if record.get("official_domain_hint") == "pathoplexus.org"
        or "site:pathoplexus.org" in str(record.get("query") or "")
    ]

    assert summary["authority_retry_result_budget"] == 4
    assert summary["authority_retry_result_budget_used"] > 0
    assert pathoplexus_records
    assert pathoplexus_records[0]["execution_status"] in {"executed", "no_results"}
    assert any(
        "pathoplexus.org" in str(candidate.get("canonical_url") or "")
        for candidate in _search_candidates(result)
    )


def test_authority_retry_generates_literature_domain_queries_when_peer_sources_missing(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "3")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_RESULT_BUDGET", "8")
    monkeypatch.setenv("AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", "6")
    fixture_path = tmp_path / "literature_domain_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_event"],
                        "results": [
                            {
                                "title": "WHO reports Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": "Andes virus cruise ship outbreak on MV Hondius in 2026.",
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:eurosurveillance.org"],
                        "results": [
                            {
                                "title": "Eurosurveillance rapid communication on Andes virus",
                                "url": "https://www.eurosurveillance.org/content/10.2807/example-andes-virus",
                                "snippet": "Peer-reviewed rapid communication about the cruise ship event.",
                                "source": "Eurosurveillance",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:science.org"],
                        "results": [
                            {
                                "title": "Science reports on cruise ship hantavirus outbreak",
                                "url": "https://www.science.org/content/article/cruise-ship-hantavirus-outbreak-example",
                                "snippet": "Scientific news source describing the Andes virus event.",
                                "source": "Science",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus cruise ship outbreak evidence.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_event",
                    "query": "hantavirus MV Hondius WHO outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_public_health_agency",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                }
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records)
    executed_domains = {
        record.get("official_domain_hint")
        for record in retry_records
        if record.get("execution_status") in {"executed", "no_results"}
    }

    assert "site:eurosurveillance.org" in retry_text
    assert "site:science.org" in retry_text
    assert "eurosurveillance.org" in executed_domains
    assert "science.org" in executed_domains
    assert "github" not in retry_text.lower()
    assert "gh_id" not in retry_text.lower()


def test_authority_retry_caps_same_domain_results_across_retry_queries(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "3")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_RESULT_BUDGET", "12")
    monkeypatch.setenv("AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES", "3")
    fixture_path = tmp_path / "retry_same_domain_cap_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_who_don"],
                        "results": [
                            {
                                "title": "Disease Outbreak News: Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": "WHO reports a cruise ship outbreak linked to Andes virus and MV Hondius.",
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:who.int"],
                        "results": [
                            {
                                "title": f"WHO same event follow-up {idx}",
                                "url": f"https://www.who.int/publications/m/item/hondius-follow-up-{idx}",
                                "snippet": "WHO follow-up source for the same cruise ship event.",
                                "source": "World Health Organization",
                            }
                            for idx in range(1, 4)
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus outbreak evidence for a cruise ship event.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_who_don",
                    "query": "hantavirus WHO DON cruise ship 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_public_health_agency",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                }
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    retry_who_accepted = [
        row
        for row in result["source_search_results"]
        if row.get("query_source") == "authority_gap_retry"
        and row.get("domain") == "who.int"
        and row.get("result_status") == "accepted"
    ]
    cap_rejections = [
        row
        for row in result["source_search_results"]
        if row.get("rejection_reason") == "per_domain_retry_result_cap_reached"
    ]
    summary = result["source_search_execution_summary"]

    assert len(retry_who_accepted) <= 3
    assert len(retry_who_accepted) == 3
    assert summary["retry_domain_result_cap_counts"].get("who.int", 0) == 0


def test_official_page_family_retry_discovers_same_event_who_publication(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "1")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES", "3")
    fixture_path = tmp_path / "who_page_family_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_who_don"],
                        "results": [
                            {
                                "title": "Disease Outbreak News: Andes virus on MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
                                "snippet": "WHO reports a cruise ship outbreak linked to Andes virus and MV Hondius.",
                                "source": "World Health Organization",
                            }
                        ],
                    },
                    {
                        "match_terms": ["site:who.int/publications/m", "risk assessment"],
                        "results": [
                            {
                                "title": "Rapid risk assessment: Andes virus on a cruise ship",
                                "url": "https://www.who.int/publications/m/item/rapid-risk-assessment-andes-virus-mv-hondius-2026",
                                "snippet": "WHO rapid risk assessment for the same cruise ship event.",
                                "source": "World Health Organization",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus outbreak evidence for a cruise ship event.",
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_who_don",
                    "query": "hantavirus WHO DON cruise ship 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_public_health_agency",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                }
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    urls = {candidate.get("canonical_url") for candidate in _search_candidates(result)}

    assert "site:who.int/publications/m" in retry_text
    assert "risk assessment" in retry_text
    assert summary["official_page_family_retry_generated_count"] >= 1
    assert summary["official_page_family_retry_candidate_count"] >= 1
    assert summary["source_recall_target_ledger"]
    assert any(
        row.get("authority_domain") == "who.int"
        and row.get("event_page_status") == "event_page_found"
        for row in summary["source_recall_target_ledger"]
    )
    assert any("who.int/publications/m" in str(url) for url in urls)


def test_high_trust_no_result_domain_retry_uses_event_aliases_without_benchmark_inputs(
    monkeypatch,
    tmp_path,
):
    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "fixture")
    monkeypatch.setenv("SEARCH_PROVIDER", "fixture")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "2")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "2")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "8")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "false")
    monkeypatch.setenv("AUTHORITY_GAP_RETRY_MAX_QUERIES", "4")
    fixture_path = tmp_path / "known_domain_no_result_retry_fixture.json"
    fixture_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_ids": ["q_who_event"],
                        "results": [
                            {
                                "title": "WHO Andes virus outbreak linked to MV Hondius",
                                "url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
                                "snippet": (
                                    "A cruise ship outbreak involved Andes virus, "
                                    "MV Hondius passengers, sequences and case reports."
                                ),
                                "source": "WHO",
                            }
                        ],
                    },
                    {
                        "query_ids": ["q_pathoplexus_initial"],
                        "results": [],
                    },
                    {
                        "match_terms": ["site:pathoplexus.org", "andes virus"],
                        "results": [
                            {
                                "title": "Pathoplexus Andes virus sequence records",
                                "url": "https://pathoplexus.org/andv/search",
                                "snippet": "Structured Andes virus sequence records for the outbreak.",
                                "source": "Pathoplexus",
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SEARCH_FIXTURE_PATH", str(fixture_path))

    from data_collection_workflow.nodes.source_discovery import source_discovery

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "standard",
            "user_request": "Collect hantavirus multi-country cruise ship outbreak evidence.",
        },
        "disease_intelligence": {
            "disease_input": "hantavirus",
            "aliases": ["hantavirus", "Andes virus"],
            "abbreviations": ["HPS", "ANDV"],
            "pathogen_terms": ["hantavirus", "orthohantavirus", "Andes virus"],
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_who_event",
                    "query": "hantavirus WHO Andes virus MV Hondius outbreak 2026",
                    "provider_channel": "official_site_search",
                    "query_type": "official_site",
                    "source_type": "international_organization_report",
                    "role_hint": "validation",
                    "execution_status": "planned_not_executed",
                },
                {
                    "query_id": "q_pathoplexus_initial",
                    "query": "hantavirus site:pathoplexus.org sequence cases 2026-04-01-2026-06-30",
                    "provider_channel": "database_search",
                    "query_type": "database",
                    "source_type": "structured_database",
                    "role_hint": "collection_support",
                    "execution_status": "planned_not_executed",
                },
            ]
        },
        "collection_trace": [],
    }

    result = source_discovery(state)

    summary = result["source_search_execution_summary"]
    retry_records = [
        record
        for record in summary["query_execution_records"]
        if record.get("query_source") == "authority_gap_retry"
    ]
    retry_text = "\n".join(record.get("query") or "" for record in retry_records).lower()
    urls = {candidate.get("canonical_url") for candidate in _search_candidates(result)}

    assert summary["known_authority_domain_no_result_count"] >= 1
    assert summary["known_authority_domain_no_result_retry_generated_count"] >= 1
    assert "site:pathoplexus.org" in retry_text
    assert "andes virus" in retry_text or "andv" in retry_text
    assert "github" not in retry_text
    assert "gh_id" not in retry_text
    assert any("pathoplexus.org" in str(url) for url in urls)


def test_source_candidate_and_registry_preserve_search_provenance(monkeypatch, tmp_path):
    from data_collection_workflow.nodes.source_discovery import source_dedup_and_registry

    _enable_fixture_search(monkeypatch, tmp_path)
    state = _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))
    candidates = _search_candidates(state)
    assert candidates
    candidate = candidates[0]
    for key in (
        "query_id",
        "query_used",
        "search_provider",
        "search_rank",
        "provider_channel",
        "role_hint",
        "discovery_method",
        "planned_query_id",
        "canonical_url",
    ):
        assert candidate.get(key) not in (None, "", [])

    state.update(source_dedup_and_registry(state))
    registry = [
        entry
        for entry in state.get("source_registry") or []
        if entry.get("discovery_method") == "fixture_search_result"
    ]
    assert registry
    entry = registry[0]
    for key in (
        "source_id",
        "canonical_url",
        "source_type",
        "query_id",
        "query_used",
        "search_provider",
        "discovery_method",
    ):
        assert entry.get(key) not in (None, "", [])


def test_full_graph_covid19_fixture_search_smoke(tmp_path):
    result = _run_full_graph_from_config(write_workflow_config(
        tmp_path, disease="COVID-19", location="New York", year="2024", phase="search",
    ))
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    summaries = package.get("workflow_summaries") or {}
    search_summary = summaries.get("source_search_execution_summary") or {}
    discovery = result.get("source_discovery_summary") or {}
    registry = result.get("source_registry") or []

    assert package
    assert metadata.get("disease") == "COVID-19"
    assert metadata.get("geography") == "New York"
    assert metadata.get("time_window") == "2024"
    assert search_summary["executed_query_count"] > 0
    assert search_summary["discovery_method"] == "fixture_search_plus_seed_catalog"
    assert discovery["discovery_method"] == "fixture_search_plus_seed_catalog"
    assert any(entry.get("discovery_method") == "fixture_search_result" for entry in registry)


def test_full_graph_dengue_fixture_search_smoke(tmp_path):
    result = _run_full_graph_from_config(write_workflow_config(
        tmp_path, disease="dengue", location="Florida", year="2025", phase="search",
    ))
    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    summaries = package.get("workflow_summaries") or {}
    search_summary = summaries.get("source_search_execution_summary") or {}
    discovery = result.get("source_discovery_summary") or {}
    registry = result.get("source_registry") or []

    assert package
    assert (metadata.get("disease") or "").lower() == "dengue"
    assert metadata.get("geography") == "Florida"
    assert metadata.get("time_window") == "2025"
    assert search_summary["executed_query_count"] > 0
    assert search_summary["discovery_method"] == "fixture_search_plus_seed_catalog"
    assert discovery["discovery_method"] == "fixture_search_plus_seed_catalog"
    assert any(entry.get("discovery_method") == "fixture_search_result" for entry in registry)


def test_live_search_provider_is_not_called_unless_live_mode_is_explicit(monkeypatch, tmp_path):
    source_discovery_module = importlib.import_module(
        "data_collection_workflow.nodes.source_discovery"
    )

    class ExplodingProvider:
        def search(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("live provider should not be called")

    monkeypatch.setattr(
        source_discovery_module,
        "_build_search_provider",
        lambda settings: ExplodingProvider(),
        raising=False,
    )

    _clear_search_env(monkeypatch)
    _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))

    _enable_fixture_search(monkeypatch, tmp_path)
    _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))


def test_mocked_live_search_provider_works(monkeypatch):
    source_discovery_module = importlib.import_module(
        "data_collection_workflow.nodes.source_discovery"
    )

    class MockLiveProvider:
        provider = "tavily"

        def search(self, planned_query, *, max_results, timeout_seconds):  # noqa: ARG002
            query = planned_query.get("query")
            return {
                "provider": "tavily",
                "query_id": planned_query.get("query_id"),
                "query": query,
                "results": [
                    {
                        "title": "Mocked New York COVID-19 surveillance 2024",
                        "url": "https://health.ny.gov/example/mocked-covid-19-2024",
                        "snippet": "Mocked live provider metadata for COVID-19 cases in New York.",
                        "published_date": "2024-05-01",
                        "source": "Mocked Search",
                        "rank": 1,
                    }
                ],
                "raw_result_count": 1,
                "error": None,
                "warnings": [],
            }

    _clear_search_env(monkeypatch)
    monkeypatch.setenv("SEARCH_MODE", "live")
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("ENABLE_LIVE_SEARCH", "true")
    monkeypatch.setenv("SEARCH_MAX_QUERIES", "1")
    monkeypatch.setenv("SEARCH_MAX_RESULTS_PER_QUERY", "3")
    monkeypatch.setenv("SEARCH_MAX_TOTAL_RESULTS", "3")
    monkeypatch.setenv("SEARCH_COMBINE_WITH_SEED_CATALOG", "true")
    monkeypatch.setattr(
        source_discovery_module,
        "_build_search_provider",
        lambda settings: MockLiveProvider(),
        raising=False,
    )

    result = _run_to_source_discovery(_state_for("COVID-19", "New York", "2024"))
    summary = result.get("source_search_execution_summary") or {}
    candidates = _search_candidates(result)

    assert summary["search_mode"] == "live"
    assert summary["live_search_enabled"] is True
    assert summary["executed_query_count"] > 0
    assert summary["candidate_from_search_count"] > 0
    assert {candidate["discovery_method"] for candidate in candidates} == {
        "live_search_result"
    }
    assert any("mocked-covid-19-2024" in candidate["url"] for candidate in candidates)


def test_tavily_provider_uses_bearer_authorization_header(monkeypatch):
    from data_collection_workflow.search_providers import TavilySearchProvider

    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001
            return False

        def read(self):
            return (
                b'{"results":[{"title":"Official result",'
                b'"url":"https://example.org/report",'
                b'"content":"Official source metadata."}]}'
            )

    def fake_urlopen(request, *, timeout):  # noqa: ANN001
        captured["authorization"] = request.get_header("Authorization")
        captured["content_type"] = request.get_header("Content-type")
        captured["payload"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = TavilySearchProvider().search(
        {"query_id": "q1", "query": "official COVID-19 New York 2024"},
        max_results=1,
        timeout_seconds=7,
    )
    payload = json.loads(captured["payload"].decode("utf-8"))

    assert captured["authorization"] == "Bearer tvly-test-key"
    assert captured["content_type"] == "application/json"
    assert "api_key" not in payload
    assert payload["query"] == "official COVID-19 New York 2024"
    assert captured["timeout"] == 7
    assert response.results[0].url == "https://example.org/report"
