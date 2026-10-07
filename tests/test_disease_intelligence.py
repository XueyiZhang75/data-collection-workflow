"""Stage 2 tests for disease intelligence."""

from __future__ import annotations

from synthetic_workflow_inputs import write_workflow_config

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_RETIRED_COMPATIBILITY_WARNINGS = {
    "source_discovery_not_yet_disease_generic",
    "extraction_record_model_still_hantavirus_named",
    "extraction_record_schema_not_yet_disease_generic",
}
_STALE_PROFILE_SCHEMA_WARNINGS = {
    "profile_schema_not_yet_generalized",
    "non_hantavirus_task_with_hantavirus_profile_resources",
}


def _state_for(disease: str, location: str, start: str, end: str) -> dict:
    return {
        "structured_task": {
            "disease": disease,
            "location": location,
            "start_date": start,
            "end_date": end,
            "target_fields": [
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


def _build_intelligence(disease: str, location: str, start: str, end: str) -> dict:
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        task_intake_and_scope_planning,
    )

    state = _state_for(disease, location, start, end)
    state.update(task_intake_and_scope_planning(state))
    return disease_intelligence_builder(state)


def _all_terms(intelligence: dict) -> set[str]:
    fields = (
        "aliases",
        "abbreviations",
        "pathogen_terms",
        "syndrome_terms",
        "clinical_terms",
        "transmission_terms",
        "surveillance_terms",
        "official_source_terms",
        "suggested_query_terms",
    )
    terms: set[str] = set()
    for field in fields:
        terms.update(str(item).lower() for item in intelligence.get(field) or [])
    return terms


def test_curated_hantavirus_disease_intelligence_terms():
    result = _build_intelligence("hantavirus", "New Mexico", "2020", "2026")

    intelligence = result["disease_intelligence"]
    assert intelligence["generation_method"] == "curated_profile"
    assert "hantavirus" in _all_terms(intelligence)
    assert "hps" in _all_terms(intelligence)
    assert "sin nombre virus" in _all_terms(intelligence)
    assert any("hantavirus" in term.lower() for term in intelligence["suggested_query_terms"])
    assert result["disease_intelligence_summary"]["query_term_count"] > 0


def test_curated_covid19_disease_intelligence_terms_not_hantavirus_primary():
    result = _build_intelligence("COVID-19", "New York", "2024", "2024")

    intelligence = result["disease_intelligence"]
    terms = _all_terms(intelligence)
    assert intelligence["generation_method"] == "curated_profile"
    assert "COVID-19" in intelligence["disease_standard_name"]
    assert "sars-cov-2" in terms
    assert any("covid" in term.lower() for term in intelligence["suggested_query_terms"])
    assert not {"hps", "sin nombre virus"}.intersection(terms)


def test_curated_dengue_disease_intelligence_terms_not_hantavirus_primary():
    result = _build_intelligence("dengue", "Florida", "2025", "2025")

    intelligence = result["disease_intelligence"]
    terms = _all_terms(intelligence)
    assert intelligence["generation_method"] == "curated_profile"
    assert "denv" in terms
    assert "dengue virus" in terms
    assert "arbovirus surveillance" in terms
    assert "mosquito-borne disease" in terms
    assert any("dengue" in term.lower() for term in intelligence["suggested_query_terms"])
    assert not {"hps", "sin nombre virus"}.intersection(terms)


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
    assert env_updates["ENABLE_LIVE_FETCH"] == "false"
    assert env_updates["ENABLE_LLM_DISEASE_INTELLIGENCE"] == "false"
    with temporary_workflow_env(env_updates):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_full_graph_covid19_exports_disease_intelligence_summary(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='COVID-19', location='New York', year='2024', phase='task'))

    package = result["final_data_package"]
    metadata = package["package_metadata"]
    assert metadata["disease"] == "COVID-19"
    assert metadata["geography"] == "New York"
    assert metadata["time_window"] == "2024"

    summaries = package["workflow_summaries"]
    summary = summaries["disease_intelligence_summary"]
    assert "COVID-19" in summary["disease_standard_name"]
    assert summary["generation_method"] == "curated_profile"
    assert summary["query_term_count"] > 0
    intake_warnings = set(summaries["task_intake_summary"]["warnings"])
    profile_warnings = set(summaries["profile_schema_summary"]["warnings"])
    assert _RETIRED_COMPATIBILITY_WARNINGS.isdisjoint(intake_warnings)
    assert _RETIRED_COMPATIBILITY_WARNINGS.isdisjoint(profile_warnings)
    assert _STALE_PROFILE_SCHEMA_WARNINGS.isdisjoint(intake_warnings)
    assert _STALE_PROFILE_SCHEMA_WARNINGS.isdisjoint(profile_warnings)


def test_full_graph_dengue_exports_disease_intelligence_summary(tmp_path):
    result = _run_full_graph_from_example_config(write_workflow_config(tmp_path, disease='dengue', location='Florida', year='2025', phase='task'))

    package = result["final_data_package"]
    summaries = package["workflow_summaries"]
    summary = summaries["disease_intelligence_summary"]
    assert "dengue" in summary["disease_standard_name"].lower()
    assert summary["generation_method"] == "curated_profile"
    assert summary["query_term_count"] > 0
    intake_warnings = set(summaries["task_intake_summary"]["warnings"])
    profile_warnings = set(summaries["profile_schema_summary"]["warnings"])
    assert _RETIRED_COMPATIBILITY_WARNINGS.isdisjoint(intake_warnings)
    assert _RETIRED_COMPATIBILITY_WARNINGS.isdisjoint(profile_warnings)
    assert _STALE_PROFILE_SCHEMA_WARNINGS.isdisjoint(intake_warnings)
    assert _STALE_PROFILE_SCHEMA_WARNINGS.isdisjoint(profile_warnings)


def test_llm_disease_intelligence_success_and_failure_fallback(monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        task_intake_and_scope_planning,
    )

    monkeypatch.setenv("ENABLE_LLM_DISEASE_INTELLIGENCE", "true")

    def mock_success(**kwargs):
        return {
            "disease_input": "COVID-19",
            "disease_standard_name": "COVID-19",
            "disease_category": "respiratory infectious disease",
            "aliases": ["COVID-19"],
            "abbreviations": [],
            "pathogen_terms": ["SARS-CoV-2"],
            "syndrome_terms": [],
            "clinical_terms": ["hospitalization"],
            "transmission_terms": ["respiratory virus"],
            "case_count_terms": ["cases"],
            "death_terms": ["deaths"],
            "hospitalization_terms": ["hospitalizations"],
            "surveillance_terms": ["respiratory virus surveillance"],
            "outbreak_terms": ["outbreak"],
            "official_source_terms": ["state COVID dashboard"],
            "likely_reporting_agencies": ["state health department"],
            "preferred_source_categories": ["official_public_health_agency"],
            "validation_source_categories": ["structured_database"],
            "suggested_geographic_granularity": "state",
            "suggested_time_granularity": "weekly",
            "extraction_priority_fields": ["cases_confirmed", "deaths"],
            "count_semantics_notes": ["Track cases, deaths, and hospitalizations."],
            "disambiguation_risks": ["Do not mix with all respiratory viruses."],
            "exclusion_terms": ["non-human"],
            "suggested_query_terms": ["COVID-19", "SARS-CoV-2"],
            "suggested_query_templates": ["\"COVID-19\" \"New York\" 2024"],
            "confidence": 0.9,
            "generation_method": "llm_generated",
            "warnings": [],
        }

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_success)
    state = _state_for("COVID-19", "New York", "2024", "2024")
    state.update(task_intake_and_scope_planning(state))
    success = disease_intelligence_builder(state)
    assert success["disease_intelligence"]["generation_method"] == "llm_generated"
    assert success["disease_intelligence"]["pathogen_terms"] == ["SARS-CoV-2"]

    def mock_failure(**kwargs):
        raise RuntimeError("mock disease intelligence LLM failure")

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_failure)
    fallback = disease_intelligence_builder(state)
    assert fallback["disease_intelligence"]["generation_method"] == (
        "llm_failed_curated_fallback"
    )
    assert "llm_disease_intelligence_failed_curated_fallback" in (
        fallback["disease_intelligence"]["warnings"]
    )


def test_required_llm_disease_intelligence_failure_stops_product_workflow(monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.nodes.task_scope import (
        disease_intelligence_builder,
        task_intake_and_scope_planning,
    )

    monkeypatch.setenv("ENABLE_LLM_DISEASE_INTELLIGENCE", "true")
    monkeypatch.setenv("DISEASE_INTELLIGENCE_FORCE_LLM", "true")
    monkeypatch.setenv("DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED", "false")

    def mock_failure(**kwargs):
        raise RuntimeError("mock required disease intelligence LLM failure")

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_failure)
    state = _state_for("Orthoebolavirus zairense", "Democratic Republic of the Congo", "2022", "2026")
    state.update(task_intake_and_scope_planning(state))

    with pytest.raises(RuntimeError, match="disease intelligence LLM required"):
        disease_intelligence_builder(state)


def test_query_terms_differ_by_disease():
    outputs = {
        "hantavirus": _build_intelligence("hantavirus", "New Mexico", "2020", "2026"),
        "covid": _build_intelligence("COVID-19", "New York", "2024", "2024"),
        "dengue": _build_intelligence("dengue", "Florida", "2025", "2025"),
    }
    query_terms = {
        key: set(value["disease_intelligence"]["suggested_query_terms"])
        for key, value in outputs.items()
    }

    assert query_terms["hantavirus"] != query_terms["covid"]
    assert query_terms["hantavirus"] != query_terms["dengue"]
    assert query_terms["covid"] != query_terms["dengue"]
