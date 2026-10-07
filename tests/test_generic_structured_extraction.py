from __future__ import annotations

from synthetic_workflow_inputs import write_workflow_config

import importlib
from pathlib import Path

from data_collection_workflow.graph import build_graph
from data_collection_workflow.models import PublicHealthRecord


def _covid_context_state(chunks: list[dict]) -> dict:
    return {
        "structured_task": {
            "disease": "COVID-19",
            "location": "New York",
            "start_date": "2024",
            "end_date": "2024",
            "target_fields": [
                "cases_confirmed",
                "deaths",
                "hospitalizations",
                "date_reported",
                "source_url",
                "source_type",
                "evidence_quote",
            ],
        },
        "collection_spec": {
            "disease": "COVID-19",
            "geography": "New York",
            "time_window": "2024",
            "target_population": "humans",
        },
        "disease_intelligence": {
            "disease_input": "COVID-19",
            "disease_standard_name": "COVID-19",
            "aliases": ["COVID-19", "coronavirus disease 2019"],
            "abbreviations": ["COVID"],
            "pathogen_terms": ["SARS-CoV-2"],
            "syndrome_terms": ["COVID-19"],
        },
        "evidence_chunks": chunks,
        "human_review_queue": [],
        "collection_trace": [],
    }


def _dengue_context_state(chunks: list[dict]) -> dict:
    return {
        "structured_task": {
            "disease": "dengue",
            "location": "Florida",
            "start_date": "2025",
            "end_date": "2025",
            "target_fields": [
                "cases_unspecified",
                "deaths",
                "date_reported",
                "source_url",
                "source_type",
                "evidence_quote",
            ],
        },
        "collection_spec": {
            "disease": "dengue",
            "geography": "Florida",
            "time_window": "2025",
            "target_population": "humans",
        },
        "disease_intelligence": {
            "disease_input": "dengue",
            "disease_standard_name": "Dengue",
            "aliases": ["dengue", "dengue fever"],
            "abbreviations": ["DENV"],
            "pathogen_terms": ["DENV", "dengue virus"],
            "syndrome_terms": ["dengue"],
        },
        "evidence_chunks": chunks,
        "human_review_queue": [],
        "collection_trace": [],
    }


def _base_chunk(**overrides) -> dict:
    base = {
        "chunk_id": "chunk_src_generic_001",
        "source_id": "src_generic",
        "text": "",
        "contains_target_data": True,
        "data_types": ["case_count", "death_count", "date", "location"],
        "context_types": [],
        "confidence": 0.9,
        "document_type": "html",
        "fetch_purpose": "data_extraction",
        "source_url": "https://example.org/report",
        "canonical_url": "https://example.org/report",
        "title": "Generic public health report",
        "publisher": "Example Health Department",
        "source_type": "official_public_health_agency",
        "source_role": "data_source",
        "source_role_final": "collection",
        "credibility_score": 0.91,
        "credibility_level": "high",
        "discovery_method": "fixture_search_result",
        "search_provider": "fixture",
        "query_id": "q_fixture_001",
        "query_used": "COVID-19 cases deaths New York 2024",
        "quality_status": "usable",
        "chunk_index": 1,
        "chunk_kind": "text",
    }
    base.update(overrides)
    return base


def test_generic_record_model_accepts_covid19_public_health_fields():
    record = PublicHealthRecord(
        record_id="rec_covid_001",
        disease="COVID-19",
        disease_standard_name="COVID-19",
        pathogen_or_syndrome="SARS-CoV-2",
        country="United States of America",
        subnational_location="New York",
        date_reported="2024",
        cases_confirmed=1250,
        deaths=18,
        hospitalizations=74,
        source_id="src_covid",
        source_url="https://health.ny.gov/example/covid",
        source_type="official_public_health_agency",
        evidence_quote="New York reported 1,250 COVID-19 cases, 18 deaths, and 74 hospitalizations in 2024.",
        supporting_chunk_id="chunk_covid_001",
        source_role_final="collection",
        credibility_score=0.95,
        credibility_level="high",
    )

    data = record.model_dump()
    assert data["disease"] == "COVID-19"
    assert data["hospitalizations"] == 74
    assert data["source_url"] == "https://health.ny.gov/example/covid"


def test_generic_record_model_accepts_dengue_public_health_fields():
    record = PublicHealthRecord(
        record_id="rec_dengue_001",
        disease="Dengue",
        disease_standard_name="Dengue",
        pathogen_or_syndrome="DENV",
        country="United States of America",
        subnational_location="Florida",
        date_reported="2025",
        cases_unspecified=42,
        deaths=0,
        source_id="src_dengue",
        source_url="https://floridahealth.gov/example/dengue",
        source_type="official_public_health_agency",
        evidence_quote="Florida reported 42 dengue cases and 0 deaths in 2025.",
        supporting_chunk_id="chunk_dengue_001",
    )

    assert record.disease == "Dengue"
    assert record.cases_unspecified == 42
    assert record.deaths == 0


def test_table_extraction_for_covid19_uses_generic_disease_and_hospitalizations():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(
        source_id="src_covid_table",
        chunk_id="chunk_covid_table_001",
        source_url="https://health.ny.gov/example/covid-19-surveillance-2024",
        title="New York COVID-19 Surveillance Update 2024",
        text=(
            "Week ending | Location | Cases | Deaths | Hospitalizations\n"
            "2024-06-01 | New York | 1250 | 18 | 74"
        ),
        chunk_kind="table",
    )

    result = extraction.structured_extraction(_covid_context_state([chunk]))
    records = result["raw_records"]

    assert len(records) == 1
    record = records[0]
    assert record["disease"] == "COVID-19"
    assert record["disease_standard_name"] == "COVID-19"
    assert record["pathogen_or_syndrome"] in {"COVID-19", "SARS-CoV-2", None}
    assert record["cases_unspecified"] == 1250
    assert record["deaths"] == 18
    assert record["hospitalizations"] == 74
    assert record["subnational_location"] == "New York"
    assert record["date_reported"] == "2024-06-01"


def test_table_extraction_for_dengue_uses_generic_disease():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(
        source_id="src_dengue_table",
        chunk_id="chunk_dengue_table_001",
        source_url="https://www.floridahealth.gov/example/dengue-surveillance-2025",
        title="Florida Dengue Surveillance Update 2025",
        text="Report date | Location | Dengue cases | Deaths\n2025-08-01 | Florida | 42 | 0",
        chunk_kind="table",
    )

    result = extraction.structured_extraction(_dengue_context_state([chunk]))
    record = result["raw_records"][0]

    assert record["disease"] == "Dengue"
    assert record["cases_unspecified"] == 42
    assert record["deaths"] == 0
    assert record["subnational_location"] == "Florida"
    assert record["date_reported"] == "2025-08-01"


def test_narrative_extraction_for_covid19_parses_counts_and_location():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(
        source_id="src_covid_text",
        chunk_id="chunk_covid_text_001",
        text=(
            "New York reported 1,250 COVID-19 cases, 18 deaths, and "
            "74 hospitalizations in 2024. SARS-CoV-2 activity was monitored."
        ),
    )

    result = extraction.structured_extraction(_covid_context_state([chunk]))
    record = result["raw_records"][0]

    assert record["disease"] == "COVID-19"
    assert record["cases_unspecified"] == 1250
    assert record["deaths"] == 18
    assert record["hospitalizations"] == 74
    assert record["subnational_location"] == "New York"
    assert record["date_reported"] == "2024"


def test_narrative_extraction_for_dengue_parses_counts_and_location():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(
        source_id="src_dengue_text",
        chunk_id="chunk_dengue_text_001",
        text="Florida reported 42 dengue cases and 0 deaths in 2025.",
    )

    result = extraction.structured_extraction(_dengue_context_state([chunk]))
    record = result["raw_records"][0]

    assert record["disease"] == "Dengue"
    assert record["cases_unspecified"] == 42
    assert record["deaths"] == 0
    assert record["subnational_location"] == "Florida"
    assert record["date_reported"] == "2025"


def test_non_hantavirus_extraction_does_not_insert_hantavirus_terms():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    covid = _base_chunk(
        source_id="src_covid_text",
        chunk_id="chunk_covid_text_001",
        text="New York reported 100 COVID-19 cases and 2 deaths in 2024.",
    )
    dengue = _base_chunk(
        source_id="src_dengue_text",
        chunk_id="chunk_dengue_text_001",
        text="Florida reported 42 dengue cases and 0 deaths in 2025.",
    )

    covid_records = extraction.structured_extraction(_covid_context_state([covid]))["raw_records"]
    dengue_records = extraction.structured_extraction(_dengue_context_state([dengue]))["raw_records"]

    for record in covid_records + dengue_records:
        assert record["disease"] != "Hantavirus disease"
        combined = " ".join(str(record.get(field) or "") for field in ("virus_or_syndrome", "pathogen_or_syndrome"))
        assert "Sin Nombre" not in combined
        assert "HPS" not in combined


def test_search_and_credibility_provenance_is_preserved_on_generic_records():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(
        text="New York reported 100 COVID-19 cases and 2 deaths in 2024.",
        discovery_method="live_search_result",
        search_provider="tavily",
        query_id="q_exec_001",
        query_used='"COVID-19" cases deaths public health New York 2024',
        source_role_final="collection",
        credibility_score=0.88,
        credibility_level="high",
    )

    record = extraction.structured_extraction(_covid_context_state([chunk]))["raw_records"][0]

    assert record["source_id"] == chunk["source_id"]
    assert record["source_url"] == chunk["source_url"]
    assert record["source_title"] == chunk["title"]
    assert record["publisher"] == chunk["publisher"]
    assert record["discovery_method"] == "live_search_result"
    assert record["search_provider"] == "tavily"
    assert record["query_id"] == "q_exec_001"
    assert record["query_used"] == '"COVID-19" cases deaths public health New York 2024'
    assert record["supporting_chunk_id"] == chunk["chunk_id"]
    assert record["source_role_final"] == "collection"
    assert record["credibility_score"] == 0.88


def test_schema_validation_flags_or_rejects_incomplete_generic_records():
    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    validator = extraction.schema_validation_and_repair
    missing_context_record = PublicHealthRecord(
        record_id="rec_missing_context",
        disease="COVID-19",
        cases_unspecified=5,
        source_id="src_missing",
        source_url="https://example.org/missing",
        source_type="official_public_health_agency",
        evidence_quote="Five COVID-19 cases were reported.",
        supporting_chunk_id="chunk_missing",
    ).model_dump()
    negative_record = {
        **missing_context_record,
        "record_id": "rec_negative",
        "cases_unspecified": -1,
    }
    no_provenance_record = {
        **missing_context_record,
        "record_id": "rec_no_provenance",
        "source_url": None,
        "supporting_chunk_id": None,
    }

    result = validator(
        {
            "raw_records": [missing_context_record, negative_record, no_provenance_record],
            "human_review_queue": [],
            "collection_trace": [],
        }
    )

    validated = result["validated_records"]
    rejected = result["rejected_records"]
    assert any(r["record_id"] == "rec_missing_context" and r["requires_human_review"] for r in validated)
    assert any(r["record_id"] == "rec_negative" for r in rejected)
    assert any(r["record_id"] == "rec_no_provenance" for r in rejected)


def test_optional_llm_generic_extraction_success_is_mocked(monkeypatch):
    from data_collection_workflow import llm_clients
    from data_collection_workflow.models import LLMExtractedRecord, LLMExtractionOutput

    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(text="New York reported 100 COVID-19 cases and 2 deaths in 2024.")

    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock-generic")

    def fake_extract(*args, **kwargs):  # noqa: ANN002, ANN003
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="COVID-19",
                    cases_unspecified=100,
                    deaths=2,
                    subnational_location="New York",
                    date_reported="2024",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    result = extraction.structured_extraction(_covid_context_state([chunk]))
    record = result["raw_records"][0]

    assert record["disease"] == "COVID-19"
    assert record["extraction_method"] == "llm_structured_output_extractor"
    assert record["llm_used"] is True
    assert record["source_url"] == chunk["source_url"]


def test_optional_llm_generic_extraction_failure_falls_back(monkeypatch):
    from data_collection_workflow import llm_clients

    extraction = importlib.import_module("data_collection_workflow.nodes.extraction")
    chunk = _base_chunk(text="New York reported 100 COVID-19 cases and 2 deaths in 2024.")

    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_FALLBACK_TO_RULE_BASED", "true")

    def fake_failure(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("mock llm failure")

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_failure)
    result = extraction.structured_extraction(_covid_context_state([chunk]))

    assert result["llm_extraction_summary"]["llm_error_count"] == 1
    assert result["llm_extraction_summary"]["llm_fallback_count"] == 1
    assert result["raw_records"][0]["disease"] == "COVID-19"


def _run_fixture_config(config_path: Path) -> dict:
    from data_collection_workflow.runtime_profile import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(config_path)
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_full_graph_covid19_fixture_search_fetch_generic_extraction_smoke(tmp_path):
    result = _run_fixture_config(write_workflow_config(tmp_path, disease='COVID-19', location='New York', year='2024', phase='collection'))
    records = result["normalized_records"]
    package = result["final_data_package"]

    assert result["evidence_chunks"]
    assert records
    assert all(r["disease"] == "COVID-19" for r in records)
    assert any(r.get("hospitalizations") == 3 for r in records)
    assert package["final_dataset"][0]["disease"] == "COVID-19"
    assert "generic_record_count" in result["structured_extraction_summary"]


def test_full_graph_dengue_fixture_search_fetch_generic_extraction_smoke(tmp_path):
    result = _run_fixture_config(write_workflow_config(tmp_path, disease='dengue', location='Florida', year='2025', phase='collection'))
    records = result["normalized_records"]
    package = result["final_data_package"]

    assert result["evidence_chunks"]
    assert records
    assert all(r["disease"] == "Dengue" for r in records)
    assert any(r.get("cases_unspecified") == 12 for r in records)
    assert package["final_dataset"][0]["disease"] == "Dengue"


def test_hantavirus_new_mexico_compatibility_still_uses_legacy_label(monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "false")
    result = build_graph().invoke(
        {
            "user_request": "Collect hantavirus data for New Mexico from 2020 to 2026.",
            "source_candidates": [],
            "source_registry": [],
            "documents": [],
            "evidence_chunks": [],
            "raw_records": [],
            "validated_records": [],
            "normalized_records": [],
            "linked_events": [],
            "conflicts": [],
            "human_review_queue": [],
            "collection_trace": [],
        }
    )

    for record in result.get("normalized_records") or []:
        assert record.get("disease") == "Hantavirus disease"
