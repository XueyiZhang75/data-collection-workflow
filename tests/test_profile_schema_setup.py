"""Stage 3 tests for generic profile/schema setup."""

from __future__ import annotations

from synthetic_workflow_inputs import write_workflow_config

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _state_for(
    disease: str,
    location: str,
    start: str,
    end: str,
    *,
    target_fields: list[str] | None = None,
) -> dict:
    return {
        "structured_task": {
            "disease": disease,
            "location": location,
            "start_date": start,
            "end_date": end,
            "target_fields": target_fields
            or [
                "cases_confirmed",
                "deaths",
                "date_reported",
                "source_url",
                "evidence_quote",
            ],
            "collection_mode": "standard",
            "user_request": f"Collect {disease} data for {location} from {start} to {end}.",
            "run_label": f"{disease}_{location}_{start}_{end}".replace(" ", "_"),
        },
        "collection_trace": [],
    }


def _run_profile_schema(
    disease: str,
    location: str,
    start: str,
    end: str,
    *,
    target_fields: list[str] | None = None,
) -> dict:
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        profile_and_schema_setup,
        task_intake_and_scope_planning,
    )

    state = _state_for(
        disease,
        location,
        start,
        end,
        target_fields=target_fields,
    )
    state.update(task_intake_and_scope_planning(state))
    state.update(disease_intelligence_builder(state))
    return profile_and_schema_setup(state)


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


def _field_names(schema: dict) -> set[str]:
    return {field["name"] for field in schema.get("core_fields") or []}


def _all_query_text(result: dict) -> str:
    inventory = result.get("search_query_inventory") or []
    return "\n".join(item.get("query", "") for item in inventory).lower()


def _assert_no_stale_profile_schema_warnings(warnings: list[str]) -> None:
    assert "extraction_record_schema_not_yet_disease_generic" not in warnings
    assert "profile_schema_not_yet_generalized" not in warnings
    assert "non_hantavirus_task_with_hantavirus_profile_resources" not in warnings


def test_hantavirus_profile_schema_setup_backward_compatible():
    result = _run_profile_schema("hantavirus", "New Mexico", "2020", "2026")

    profile = result["disease_profile"]
    schema = result["collection_schema"]
    summary = result["profile_schema_summary"]
    assert profile["disease_standard_name"] == "Hantavirus disease"
    assert "HPS" in profile["syndrome_terms"]
    assert schema["schema_name"] == "hantavirus_human_case_outbreak_schema"
    assert {"disease", "cases_confirmed", "deaths", "source_url"} <= _field_names(schema)
    assert summary["profile_generation_method"] == "legacy_hantavirus_profile_schema"
    assert summary["schema_generation_method"] == "legacy_hantavirus_profile_schema"


def test_covid19_generated_profile_schema_is_not_hantavirus_active_resource():
    result = _run_profile_schema("COVID-19", "New York", "2024", "2024")

    profile = result["disease_profile"]
    schema = result["collection_schema"]
    strategy = result["source_strategy"]
    summary = result["profile_schema_summary"]
    warnings = summary["warnings"]
    screening_text = " ".join(
        strategy["screening_criteria"]["include_if_all_apply"]
        + strategy["screening_criteria"]["exclude_if_any_apply"]
        + strategy["screening_criteria"]["uncertain_if_any_apply"]
    ).lower()

    assert "COVID-19" in profile["disease_standard_name"]
    assert "SARS-CoV-2" in profile["virus_terms"]
    assert "hantavirus pulmonary syndrome" not in profile["syndrome_terms"]
    assert schema["schema_name"] != "hantavirus_human_case_outbreak_schema"
    assert {"disease", "cases_confirmed", "deaths", "source_url", "supporting_chunk_id"} <= _field_names(schema)
    assert summary["profile_generation_method"] == "disease_intelligence_generated_profile_schema"
    assert summary["schema_generation_method"] == "disease_intelligence_generated_profile_schema"
    assert "covid-19" in screening_text
    assert "hps" not in screening_text
    assert "hantavirus" not in screening_text
    _assert_no_stale_profile_schema_warnings(warnings)
    assert "source_discovery_not_yet_disease_generic" not in warnings
    assert "extraction_record_model_still_hantavirus_named" not in warnings


def test_profile_schema_setup_exports_task_acceptance_contract():
    result = _run_profile_schema("FLU", "California", "2024-10-01", "2024-10-05")

    contract = result["task_acceptance_contract"]
    assert contract["disease"] == "FLU"
    assert contract["location"] == "California"
    assert contract["start_date"] == "2024-10-01"
    assert contract["end_date"] == "2024-10-05"
    assert "must_match_task_location" in contract["record_acceptance_rules"]
    assert "must_have_interpretable_numeric_or_zero_metric" in contract[
        "record_acceptance_rules"
    ]
    assert "national_or_broader_aggregate_without_target_location_fit" in contract[
        "context_or_quarantine_rules"
    ]
    assert result["profile_schema_summary"]["task_acceptance_contract_version"] == "v1"


def test_dengue_generated_profile_schema_is_not_hantavirus_active_resource():
    result = _run_profile_schema("dengue", "Florida", "2025", "2025")

    profile = result["disease_profile"]
    schema = result["collection_schema"]
    strategy = result["source_strategy"]
    summary = result["profile_schema_summary"]
    warnings = summary["warnings"]
    screening_text = " ".join(
        strategy["screening_criteria"]["include_if_all_apply"]
        + strategy["screening_criteria"]["exclude_if_any_apply"]
        + strategy["screening_criteria"]["uncertain_if_any_apply"]
    ).lower()

    assert "dengue" in profile["disease_standard_name"].lower()
    assert {"DENV", "dengue virus"} & set(profile["virus_terms"])
    assert {"dengue fever", "severe dengue"} <= set(profile["syndrome_terms"])
    assert "hantavirus pulmonary syndrome" not in profile["syndrome_terms"]
    assert schema["schema_name"] != "hantavirus_human_case_outbreak_schema"
    assert summary["profile_generation_method"] == "disease_intelligence_generated_profile_schema"
    assert "dengue" in screening_text
    assert "hps" not in screening_text
    assert "hantavirus" not in screening_text
    _assert_no_stale_profile_schema_warnings(warnings)
    assert "source_discovery_not_yet_disease_generic" not in warnings


def test_full_graph_covid19_exports_profile_schema_summary_and_active_resources(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='COVID-19', location='New York', year='2024', phase='task'))

    package = result["final_data_package"]
    summaries = package["workflow_summaries"]
    metadata = package["package_metadata"]
    assert metadata["disease"] == "COVID-19"
    assert metadata["geography"] == "New York"
    assert metadata["time_window"] == "2024"
    assert "task_intake_summary" in summaries
    assert "disease_intelligence_summary" in summaries
    assert "profile_schema_summary" in summaries
    assert "COVID-19" in result["disease_profile"]["disease_standard_name"]
    assert result["collection_schema"]["schema_name"] != "hantavirus_human_case_outbreak_schema"
    warnings = summaries["profile_schema_summary"]["warnings"]
    _assert_no_stale_profile_schema_warnings(warnings)
    assert "source_discovery_not_yet_disease_generic" not in warnings


def test_full_graph_dengue_exports_profile_schema_summary_and_active_resources(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='dengue', location='Florida', year='2025', phase='task'))

    summaries = result["final_data_package"]["workflow_summaries"]
    assert "profile_schema_summary" in summaries
    assert "dengue" in result["disease_profile"]["disease_standard_name"].lower()
    assert result["collection_schema"]["schema_name"] != "hantavirus_human_case_outbreak_schema"
    warnings = summaries["profile_schema_summary"]["warnings"]
    _assert_no_stale_profile_schema_warnings(warnings)
    assert "source_discovery_not_yet_disease_generic" not in warnings


def test_query_strategy_receives_disease_aware_profile_terms():
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        profile_and_schema_setup,
        query_strategy_builder,
        task_intake_and_scope_planning,
    )

    states = {}
    for disease, location, start, end in (
        ("hantavirus", "New Mexico", "2020", "2026"),
        ("COVID-19", "New York", "2024", "2024"),
        ("dengue", "Florida", "2025", "2025"),
    ):
        state = _state_for(disease, location, start, end)
        state.update(task_intake_and_scope_planning(state))
        state.update(disease_intelligence_builder(state))
        state.update(profile_and_schema_setup(state))
        state.update(query_strategy_builder(state))
        states[disease] = state

    covid_queries = _all_query_text(states["COVID-19"])
    dengue_queries = _all_query_text(states["dengue"])
    hantavirus_queries = _all_query_text(states["hantavirus"])

    assert "covid-19" in covid_queries or "sars-cov-2" in covid_queries
    assert "hantavirus pulmonary syndrome" not in covid_queries
    assert "dengue" in dengue_queries or "denv" in dengue_queries
    assert "hantavirus pulmonary syndrome" not in dengue_queries
    assert "hantavirus" in hantavirus_queries or "hps" in hantavirus_queries


def test_covid19_hospitalizations_target_field_preserved_or_warned():
    result = _run_profile_schema(
        "COVID-19",
        "New York",
        "2024",
        "2024",
        target_fields=[
            "cases_confirmed",
            "deaths",
            "hospitalizations",
            "date_reported",
            "source_url",
            "evidence_quote",
        ],
    )

    schema = result["collection_schema"]
    summary = result["profile_schema_summary"]
    fields = _field_names(schema)
    warnings = summary["warnings"]
    assert "hospitalizations" in fields or (
        "target_field_not_yet_supported_by_record_model:hospitalizations" in warnings
    )
    assert "hospitalizations" in summary["target_fields"]


def test_final_package_policy_includes_profile_schema_summary():
    from data_collection_workflow.config import load_final_package_policy

    policy = load_final_package_policy()
    assert "profile_schema_summary" in policy["workflow_summary_fields"]
    assert "executable_source_plan_summary" in policy["workflow_summary_fields"]
