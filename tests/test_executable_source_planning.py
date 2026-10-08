"""Tests for executable, not-yet-executed source planning."""

from __future__ import annotations

from synthetic_workflow_inputs import write_workflow_config

import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _state_for(disease: str, location: str = "New Mexico") -> dict:
    return {
        "structured_task": {
            "disease": disease,
            "location": location,
            "start_date": "2020",
            "end_date": "2026",
            "target_fields": [
                "cases_confirmed",
                "cases_probable",
                "cases_suspected",
                "cases_unspecified",
                "deaths",
                "date_reported",
                "source_url",
                "source_type",
                "evidence_quote",
            ],
            "collection_mode": "standard",
            "user_request": (
                f"Collect {disease} case and death data for {location} "
                "from 2020 to 2026."
            ),
            "run_label": f"{disease}_{location}_stage4".replace(" ", "_"),
        },
        "collection_trace": [],
    }


def _run_to_executable_plan(state: dict) -> dict:
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        executable_source_planning,
        profile_and_schema_setup,
        task_intake_and_scope_planning,
    )

    state.update(task_intake_and_scope_planning(state))
    state.update(disease_intelligence_builder(state))
    state.update(profile_and_schema_setup(state))
    state.update(executable_source_planning(state))
    return state


def _all_planned_query_text(plan: dict) -> str:
    return "\n".join(
        str(item.get("query") or "") for item in plan.get("planned_queries") or []
    ).lower()


def test_deterministic_and_model_plans_describe_pending_execution_without_missing_feature_claim(monkeypatch):
    from data_collection_workflow.models import ExecutableSourcePlan
    from data_collection_workflow.nodes.task_scope import _merge_llm_executable_plan

    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "false")
    monkeypatch.setenv("ENABLE_LLM_DISEASE_INTELLIGENCE", "false")
    state = _run_to_executable_plan(_state_for("measles", "Canada"))
    deterministic = ExecutableSourcePlan(**state["agentic_source_plan"])
    model_plan = _merge_llm_executable_plan({}, deterministic)
    for plan in (deterministic, model_plan):
        assert plan.execution_status == "planned_not_executed"
        assert plan.planned_queries
        assert all(query.execution_status == "planned_not_executed" for query in plan.planned_queries)
        assert not any("not_implemented" in warning or "stage4" in warning for warning in plan.warnings)


def _run_full_graph_from_example_config(config_path: Path) -> dict:
    from data_collection_workflow.graph import build_graph
    from data_collection_workflow.workflow_run_config import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(config_path)
    env_updates = workflow_run_env_from_config(config)
    assert env_updates["USE_FIXTURE_DOCUMENTS"] == "false"
    assert env_updates["ENABLE_LIVE_FETCH"] == "false"
    assert env_updates["ENABLE_LLM_SOURCE_PLANNING"] == "false"
    assert env_updates["ENABLE_LLM_SOURCE_CRITIC"] == "false"
    assert env_updates["ENABLE_LLM_EXTRACTION"] == "false"
    with temporary_workflow_env(env_updates):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def _combined_plan_and_inventory_text(result: dict) -> str:
    plan = result.get("agentic_source_plan") or {}
    planned_queries = plan.get("planned_queries") or []
    inventory = result.get("search_query_inventory") or []
    return "\n".join(
        str(item.get("query") or "") for item in [*planned_queries, *inventory]
    ).lower()


def _assert_no_executable_plan_source_candidates(result: dict) -> None:
    candidates = result.get("source_candidates") or []
    registry = result.get("source_registry") or []
    # These non-hantavirus task examples disable search and supply no seed
    # overlay. Planning must neither invent sources nor inherit another
    # disease's built-in catalog.
    assert candidates == []
    assert registry == []


def test_deterministic_executable_source_plan_is_auditable_and_not_executed(monkeypatch):
    monkeypatch.delenv("ENABLE_LLM_SOURCE_PLANNING", raising=False)
    state = _run_to_executable_plan(_state_for("hantavirus", "New Mexico"))

    plan = state["agentic_source_plan"]
    summary = state["executable_source_plan_summary"]

    assert plan["plan_id"]
    assert plan["disease"] == "Hantavirus disease"
    assert plan["location"] == "New Mexico"
    assert plan["time_window"] == "2020-2026"
    assert plan["generation_method"] == "deterministic_executable_source_plan"
    assert plan["llm_enabled"] is False
    assert plan["execution_status"] == "planned_not_executed"
    assert summary["execution_status"] == "planned_not_executed"
    assert summary["planned_query_count"] == len(plan["planned_queries"])
    assert summary["planned_source_category_count"] == len(
        plan["planned_source_categories"]
    )
    assert plan["source_discovery_objectives"]
    assert plan["planned_source_categories"]
    assert plan["planned_queries"]
    assert plan["source_planning_risks"]

    required_query_keys = {
        "query_id",
        "query",
        "query_type",
        "provider_channel",
        "source_type",
        "role_hint",
        "priority",
        "expected_fields",
        "disease_terms_used",
        "location_terms_used",
        "time_terms_used",
        "rationale",
        "execution_status",
    }
    for query in plan["planned_queries"]:
        assert required_query_keys <= set(query)
        assert query["execution_status"] == "planned_not_executed"
        assert query["provider_channel"] in {
            "web_search",
            "official_site_search",
            "literature_api",
            "news_search",
            "database_search",
            "manual_user_url",
        }
        assert not _URL_RE.search(query["query"])


def test_executable_source_plan_is_disease_aware_for_covid_and_dengue(monkeypatch):
    monkeypatch.delenv("ENABLE_LLM_SOURCE_PLANNING", raising=False)

    covid = _run_to_executable_plan(_state_for("COVID-19", "New York"))
    dengue = _run_to_executable_plan(_state_for("dengue", "Florida"))

    covid_text = _all_planned_query_text(covid["agentic_source_plan"])
    dengue_text = _all_planned_query_text(dengue["agentic_source_plan"])

    assert "covid-19" in covid_text or "sars-cov-2" in covid_text
    assert "hantavirus pulmonary syndrome" not in covid_text
    assert "dengue" in dengue_text or "denv" in dengue_text
    assert "hantavirus pulmonary syndrome" not in dengue_text
    assert covid_text != dengue_text


def test_llm_executable_source_plan_is_called_once_and_consumed_by_query_strategy(
    monkeypatch,
):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.graph import build_graph

    calls = {"count": 0, "user_prompt": ""}

    def fake_structured_llm(*, schema_model, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        calls["user_prompt"] = kwargs.get("user_prompt") or ""
        assert schema_model.__name__ == "ExecutableSourcePlan"
        return {
            "plan_id": "plan_llm_test",
            "disease": "Hantavirus disease",
            "location": "global",
            "time_window": "2020-2026",
            "target_fields": ["cases_confirmed", "deaths", "source_url"],
            "generation_method": "llm_executable_source_plan",
            "llm_enabled": True,
            "execution_status": "planned_not_executed",
            "warnings": [],
            "source_discovery_objectives": [
                {
                    "objective_id": "obj_llm_001",
                    "objective": "Find official case and death updates.",
                    "source_role_hint": "collection",
                    "rationale": "Official sources support extraction.",
                    "priority": 1,
                }
            ],
            "planned_source_categories": [
                {
                    "source_category_id": "cat_llm_001",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "priority": 1,
                    "expected_fields": ["cases_confirmed", "deaths"],
                    "why_relevant": "Likely primary reporting source.",
                    "risk_notes": ["jurisdiction-specific wording"],
                }
            ],
            "planned_queries": [
                {
                    "query_id": "q_llm_001",
                    "query": '"hantavirus" "cases" "deaths" public health',
                    "query_type": "general_web",
                    "provider_channel": "web_search",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "priority": 1,
                    "expected_fields": ["cases_confirmed", "deaths"],
                    "disease_terms_used": ["hantavirus"],
                    "location_terms_used": ["global"],
                    "time_terms_used": ["2020-2026"],
                    "rationale": "LLM planned collection query.",
                    "execution_status": "planned_not_executed",
                }
            ],
            "source_planning_risks": [],
            "_structured_output_mode": "provider_native",
        }

    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "true")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", fake_structured_llm)

    result = build_graph().invoke(
        {
            "user_request": (
                "Collect global human hantavirus case and death data from 2020 to 2026."
            ),
            "collection_trace": [],
        }
    )

    assert calls["count"] == 1
    assert "task_acceptance_contract" in calls["user_prompt"]
    assert "must_match_task_location" in calls["user_prompt"]
    assert result["source_planning_agent_summary"]["status"] == "success"
    assert result["agentic_source_plan"]["generation_method"] == (
        "llm_executable_source_plan"
    )
    assert result["evidence_strategy_plan"] == result["agentic_source_plan"]
    planned_inventory = [
        item for item in result["search_query_inventory"]
        if item.get("query_source") == "executable_source_plan"
    ]
    assert len(planned_inventory) == 1
    assert planned_inventory[0]["query_id"] == "q_llm_001"
    assert planned_inventory[0]["execution_status"] == "planned_not_executed"


def test_llm_planned_urls_are_sanitized_and_not_ingested_as_sources(monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.graph import build_graph

    def fake_structured_llm(*, schema_model, **kwargs):  # noqa: ARG001
        return {
            "plan_id": "plan_url_test",
            "disease": "Hantavirus disease",
            "location": "New Mexico",
            "time_window": "2020-2026",
            "target_fields": ["cases_confirmed", "deaths", "source_url"],
            "generation_method": "llm_executable_source_plan",
            "llm_enabled": True,
            "execution_status": "planned_not_executed",
            "warnings": [],
            "source_discovery_objectives": [],
            "planned_source_categories": [],
            "planned_queries": [
                {
                    "query_id": "q_url_001",
                    "query": "https://evil.example/path hantavirus New Mexico cases",
                    "query_type": "general_web",
                    "provider_channel": "web_search",
                    "source_type": "news_and_situation_report",
                    "role_hint": "collection_support",
                    "priority": 4,
                    "expected_fields": ["cases_confirmed"],
                    "disease_terms_used": ["hantavirus"],
                    "location_terms_used": ["New Mexico"],
                    "time_terms_used": ["2020-2026"],
                    "rationale": "URL must be sanitized into search text only.",
                    "execution_status": "planned_not_executed",
                }
            ],
            "source_planning_risks": [],
        }

    monkeypatch.setenv("ENABLE_LLM_SOURCE_PLANNING", "true")
    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", fake_structured_llm)

    result = build_graph().invoke(
        {
            "user_request": (
                "Collect human hantavirus case and death data for New Mexico "
                "from 2020 to 2026."
            ),
            "collection_trace": [],
        }
    )

    plan = result["agentic_source_plan"]
    queries = plan["planned_queries"]
    assert queries
    assert not any(_URL_RE.search(q["query"]) for q in queries)
    assert "llm_planned_query_url_sanitized:q_url_001" in plan["warnings"]
    assert not any(
        _URL_RE.search(item["query"])
        for item in result["search_query_inventory"]
        if item.get("query_source") == "executable_source_plan"
    )
    assert not [
        candidate for candidate in result["source_candidates"]
        if "evil.example" in (candidate.get("url") or "")
    ]
    assert {
        candidate.get("discovery_method")
        for candidate in result["source_candidates"]
    } == {"offline_seed_catalog"}


def test_final_package_exports_executable_source_plan_summary(monkeypatch):
    from data_collection_workflow.graph import build_graph

    monkeypatch.delenv("ENABLE_LLM_SOURCE_PLANNING", raising=False)
    result = build_graph().invoke(
        {
            "user_request": (
                "Collect global human hantavirus case and death data from 2020 to 2026."
            ),
            "collection_trace": [],
        }
    )
    summaries = result["final_data_package"]["workflow_summaries"]

    assert "executable_source_plan_summary" in summaries
    assert summaries["executable_source_plan_summary"]["execution_status"] == (
        "planned_not_executed"
    )
    assert summaries["executable_source_plan_summary"]["planned_query_count"] > 0


def test_direct_collection_query_strategy_uses_llm_plan_as_primary_inventory():
    from data_collection_workflow.nodes.task_scope import query_strategy_builder

    state = {
        "user_request": "Collect FLU data for California from 2024-10-01 to 2024-10-05.",
        "structured_task": {
            "disease": "FLU",
            "location": "California",
            "start_date": "2024-10-01",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "California",
            "start_date": "2024-10-01",
            "end_date": "2024-10-05",
            "time_window": "2024-10-01 to 2024-10-05",
            "collection_mode": "direct_collection",
            "required_fields": ["cases", "deaths", "hospitalizations"],
        },
        "disease_profile": {
            "disease_standard_name": "Seasonal influenza",
            "include_terms": ["flu", "influenza"],
            "exclude_terms": ["H5N1"],
            "syndrome_terms": ["influenza-like illness"],
            "virus_terms": ["influenza A", "influenza B"],
            "target_population": "humans",
            "primary_data_objects": ["surveillance summaries", "aggregate counts"],
            "required_record_fields": ["date_reported", "location", "evidence_quote"],
        },
        "source_strategy": {
            "source_categories": [
                {
                    "source_type": "official_public_health_agency",
                    "priority": 1,
                    "description": "official sources",
                },
                {
                    "source_type": "news_and_situation_report",
                    "priority": 5,
                    "description": "news sources",
                },
            ],
            "screening_criteria": {
                "include_if_all_apply": [],
                "exclude_if_any_apply": [],
                "uncertain_if_any_apply": [],
            },
        },
        "disease_intelligence": {
            "disease_standard_name": "Seasonal influenza",
            "suggested_query_terms": ["influenza", "flu"],
            "syndrome_terms": ["influenza-like illness"],
            "pathogen_terms": ["influenza A", "influenza B"],
        },
        "agentic_source_plan": {
            "planned_queries": [
                {
                    "query_id": "q_llm_california_001",
                    "query": (
                        'California influenza surveillance "October 5, 2024" '
                        "hospitalizations deaths"
                    ),
                    "query_type": "general_web",
                    "provider_channel": "web_search",
                    "source_type": "official_public_health_agency",
                    "role_hint": "collection",
                    "priority": 1,
                    "expected_fields": ["hospitalizations", "deaths"],
                    "disease_terms_used": ["influenza"],
                    "location_terms_used": ["California"],
                    "time_terms_used": ["2024-10-05"],
                    "rationale": (
                        "LLM evidence strategy selected a California-specific "
                        "collection route."
                    ),
                }
            ],
            "generation_method": "llm_executable_source_plan",
            "llm_enabled": True,
        },
        "collection_trace": [],
    }

    result = query_strategy_builder(state)
    inventory = result["search_query_inventory"]

    assert [item["query_id"] for item in inventory] == ["q_llm_california_001"]
    assert inventory[0]["query_source"] == "executable_source_plan"
    assert result["source_planning_agent_summary"][
        "deterministic_query_template_suppressed"
    ] is True


def test_full_graph_covid19_exports_executable_source_plan_summary(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='COVID-19', location='New York', year='2024', phase='task'))

    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    summaries = package.get("workflow_summaries") or {}
    plan_summary = summaries.get("executable_source_plan_summary") or {}
    source_discovery = result.get("source_discovery_summary") or {}
    executable_queries = [
        item for item in result.get("search_query_inventory") or []
        if item.get("query_source") == "executable_source_plan"
    ]
    all_query_text = _combined_plan_and_inventory_text(result)

    assert package
    assert metadata.get("disease") == "COVID-19"
    assert metadata.get("geography") == "New York"
    assert metadata.get("time_window") == "2024"
    assert plan_summary["execution_status"] == "planned_not_executed"
    assert plan_summary["planned_query_count"] > 0
    assert executable_queries
    assert all(q.get("execution_status") == "planned_not_executed" for q in executable_queries)
    assert "covid-19" in all_query_text or "sars-cov-2" in all_query_text
    assert "new york" in all_query_text
    assert "2024" in all_query_text
    assert source_discovery.get("discovery_method") == "offline_seed_catalog"
    _assert_no_executable_plan_source_candidates(result)


def test_full_graph_dengue_exports_executable_source_plan_summary(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='dengue', location='Florida', year='2025', phase='task'))

    package = result.get("final_data_package") or {}
    metadata = package.get("package_metadata") or {}
    summaries = package.get("workflow_summaries") or {}
    plan_summary = summaries.get("executable_source_plan_summary") or {}
    source_discovery = result.get("source_discovery_summary") or {}
    executable_queries = [
        item for item in result.get("search_query_inventory") or []
        if item.get("query_source") == "executable_source_plan"
    ]
    all_query_text = _combined_plan_and_inventory_text(result)

    assert package
    assert (metadata.get("disease") or "").lower() == "dengue"
    assert metadata.get("geography") == "Florida"
    assert metadata.get("time_window") == "2025"
    assert plan_summary["execution_status"] == "planned_not_executed"
    assert plan_summary["planned_query_count"] > 0
    assert executable_queries
    assert all(q.get("execution_status") == "planned_not_executed" for q in executable_queries)
    assert "dengue" in all_query_text or "denv" in all_query_text
    assert "florida" in all_query_text
    assert "2025" in all_query_text
    assert source_discovery.get("discovery_method") == "offline_seed_catalog"
    _assert_no_executable_plan_source_candidates(result)
