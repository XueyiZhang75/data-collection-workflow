"""Schema tests for the hantavirus data collection workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.config import (  # noqa: E402
    get_collection_mode,
    load_content_fetch_policy,
    load_cross_source_consistency_policy,
    load_evidence_chunking_policy,
    load_final_package_policy,
    load_hantavirus_collection_schema,
    load_hantavirus_fixture_documents,
    load_hantavirus_profile,
    load_hantavirus_seed_sources,
    load_human_review_policy,
    load_llm_structured_extraction_policy,
    load_record_linking_policy,
    load_record_normalization_policy,
    load_source_role_policy,
    load_source_screening_policy,
    load_source_strategy,
    load_structured_extraction_policy,
)
from data_collection_workflow.models import (  # noqa: E402
    CollectionSchema,
    ContentFetchPolicy,
    CrossSourceConsistencyPolicy,
    DiseaseProfile,
    EvidenceChunkingPolicy,
    FinalPackagePolicy,
    FixtureDocumentCatalog,
    HantavirusRecord,
    HumanReviewDecision,
    HumanReviewPolicy,
    LLMExtractedRecord,
    LLMExtractionOutput,
    LLMStructuredExtractionPolicy,
    PublicHealthRecord,
    RecordLinkingPolicy,
    RecordNormalizationPolicy,
    SeedSourceCatalog,
    SourceScreeningPolicy,
    SourceStrategy,
    StructuredExtractionPolicy,
)


def test_hantavirus_record_accepts_non_negative_values():
    record = HantavirusRecord(
        record_id="r1",
        disease="Hantavirus disease",
        cases_confirmed=3,
        cases_probable=0,
        cases_suspected=2.5,
        cases_unspecified=0,
        deaths=1,
    )
    assert record.cases_confirmed == 3
    assert record.deaths == 1


def test_hantavirus_record_rejects_negative_cases():
    with pytest.raises(ValidationError):
        HantavirusRecord(
            record_id="r2",
            disease="Hantavirus disease",
            cases_confirmed=-1,
        )


def test_hantavirus_record_rejects_negative_deaths():
    with pytest.raises(ValidationError):
        HantavirusRecord(
            record_id="r3",
            disease="Hantavirus disease",
            deaths=-2,
        )


def test_public_health_record_accepts_generic_metric_fields():
    record = PublicHealthRecord(
        record_id="metric_1",
        disease="Seasonal influenza",
        metric_name="nssp_ed_visit_percent",
        metric_value=0.2,
        metric_unit="percent",
        metric_category="ed_visit_percent",
        metric_denominator="emergency_department_visits",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
    )

    assert record.metric_name == "nssp_ed_visit_percent"
    assert record.metric_value == 0.2
    assert record.metric_unit == "percent"
    assert record.metric_category == "ed_visit_percent"
    assert record.metric_denominator == "emergency_department_visits"
    assert record.metric_period_end == "2024-10-05"


def test_public_health_record_rejects_negative_generic_metric_value():
    with pytest.raises(ValidationError):
        PublicHealthRecord(
            record_id="metric_negative",
            disease="Seasonal influenza",
            metric_name="clinical_lab_positive_count",
            metric_value=-1,
            metric_unit="count",
        )


def test_llm_extracted_record_accepts_generic_metric_fields():
    record = LLMExtractedRecord(
        disease="Seasonal influenza",
        country="United States of America",
        metric_name="clinical_lab_positive_count",
        metric_value=56,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
    )

    assert record.metric_name == "clinical_lab_positive_count"
    assert record.metric_value == 56
    assert record.metric_category == "lab_positive_count"


def test_disease_profile_loads_from_config():
    profile_dict = load_hantavirus_profile()
    profile = DiseaseProfile(**profile_dict)
    assert profile.disease_standard_name == "Hantavirus disease"
    assert profile.disease_family == "Hantaviridae"
    assert profile.target_population == "human"
    assert "hantavirus" in profile.include_terms
    assert "HPS" in profile.syndrome_terms
    assert "Sin Nombre virus" in profile.virus_terms
    assert "disease" in profile.required_record_fields


def test_config_loads_collection_schema():
    schema_dict = load_hantavirus_collection_schema()
    schema = CollectionSchema(**schema_dict)
    assert schema.schema_name == "hantavirus_human_case_outbreak_schema"
    assert len(schema.core_fields) >= 16
    assert schema.extraction_rules, "extraction_rules should not be empty"


def test_config_loads_source_strategy():
    strategy_dict = load_source_strategy()
    strategy = SourceStrategy(**strategy_dict)
    assert len(strategy.source_categories) >= 5
    assert strategy.screening_criteria.include_if_all_apply
    assert strategy.screening_criteria.exclude_if_any_apply
    assert strategy.screening_criteria.uncertain_if_any_apply


def test_config_loads_seed_sources():
    catalog_dict = load_hantavirus_seed_sources()
    catalog = SeedSourceCatalog(**catalog_dict)
    assert catalog.catalog_name == "hantavirus_seed_source_catalog"
    assert len(catalog.seed_sources) >= 10
    for seed in catalog.seed_sources:
        assert seed.seed_source_id
        assert seed.title
        assert seed.url
        assert seed.publisher
        assert seed.source_type
        assert isinstance(seed.priority, int)
        assert seed.expected_fields, "expected_fields should not be empty"
        assert seed.match_terms, "match_terms should not be empty"


def test_config_loads_source_screening_policy():
    policy_dict = load_source_screening_policy()
    assert policy_dict["policy_name"] == "data_collection_source_screening_policy"
    assert policy_dict["target_data_fields"], "target_data_fields should be non-empty"
    assert policy_dict["context_fields"], "context_fields should be non-empty"
    thresholds = policy_dict.get("thresholds") or {}
    for key in ("high_confidence", "medium_confidence", "low_confidence"):
        assert key in thresholds, f"thresholds missing {key}"


def test_source_screening_policy_model_validates():
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    assert policy.policy_name == "data_collection_source_screening_policy"
    assert "screening_decisions" in policy.decision_labels
    assert "source_roles" in policy.decision_labels
    assert "final_decisions" in policy.decision_labels


def test_config_loads_source_role_policy():
    policy = load_source_role_policy()
    assert policy["policy_name"] == "data_collection_source_role_policy"
    assert policy["default_collection_mode"] == "standard"
    assert "standard" in policy.get("supported_collection_modes", [])
    assert "masked_validation" in policy.get("supported_collection_modes", [])
    assert policy.get("domain_masking_enabled") is False
    reserved_ids = set(policy.get("validation_reserved_source_ids") or [])
    assert {
        "src_cdc_reported_cases",
        "src_ecdc_surveillance_updates",
        "src_ecdc_annual_report_2023",
        "src_who_hantavirus_fact_sheet",
    } <= reserved_ids


def test_get_collection_mode_defaults_standard(monkeypatch):
    monkeypatch.delenv("COLLECTION_MODE", raising=False)
    assert get_collection_mode(load_source_role_policy()) == "standard"


def test_get_collection_mode_reads_masked_validation(monkeypatch):
    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")
    assert get_collection_mode(load_source_role_policy()) == "masked_validation"


def test_get_collection_mode_falls_back_for_unknown_value(monkeypatch):
    monkeypatch.setenv("COLLECTION_MODE", "unexpected")
    assert get_collection_mode(load_source_role_policy()) == "standard"


def test_config_loads_content_fetch_policy():
    policy_dict = load_content_fetch_policy()
    assert policy_dict["policy_name"] == "data_collection_content_fetch_policy"
    fetchable = policy_dict.get("fetchable_final_decisions") or []
    assert "include_for_content_fetch" in fetchable
    assert "include_for_context_fetch" in fetchable
    quality = policy_dict.get("document_quality") or {}
    assert "min_clean_text_chars_for_usable" in quality


def test_content_fetch_policy_model_validates():
    policy = ContentFetchPolicy(**load_content_fetch_policy())
    assert policy.policy_name == "data_collection_content_fetch_policy"
    assert "include_for_content_fetch" in policy.fetchable_final_decisions
    assert "seed" in policy.blocked_url_schemes


def test_config_loads_evidence_chunking_policy():
    policy_dict = load_evidence_chunking_policy()
    assert policy_dict["policy_name"] == "data_collection_evidence_chunking_policy"
    assert "usable" in policy_dict.get("chunkable_quality_statuses", [])
    assert "offline_stub_pending_live_fetch" in policy_dict.get(
        "excluded_quality_statuses", []
    )
    assert "case_count" in policy_dict.get("target_data_signals", {})
    assert "disease_definition" in policy_dict.get("context_signals", {})


def test_evidence_chunking_policy_model_validates():
    policy = EvidenceChunkingPolicy(**load_evidence_chunking_policy())
    assert policy.policy_name == "data_collection_evidence_chunking_policy"
    assert policy.max_chunk_chars > 0
    assert "case_count" in policy.target_data_signals
    assert "disease_definition" in policy.context_signals


def test_config_loads_structured_extraction_policy():
    policy_dict = load_structured_extraction_policy()
    assert policy_dict["policy_name"] == "generic_public_health_structured_extraction_policy"
    assert policy_dict["default_disease"] == ""
    assert policy_dict["extraction_method"] == "deterministic_rule_based_extractor"
    required_prov = policy_dict.get("required_provenance_fields") or []
    assert "source_url" in required_prov
    assert "evidence_quote" in required_prov


def test_structured_extraction_policy_model_validates():
    policy = StructuredExtractionPolicy(**load_structured_extraction_policy())
    assert policy.policy_name == "generic_public_health_structured_extraction_policy"
    assert policy.default_disease == ""
    assert "HPS" in policy.virus_or_syndrome_terms
    assert "confirmed" in policy.case_keywords
    assert policy.death_keywords


def test_config_loads_record_normalization_policy():
    policy_dict = load_record_normalization_policy()
    assert policy_dict["policy_name"] == "data_collection_record_normalization_policy"
    assert "USA" in policy_dict.get("country_aliases", {})
    assert "hantavirus pulmonary syndrome" in policy_dict.get(
        "virus_or_syndrome_aliases", {}
    )
    assert "official_public_health_agency" in policy_dict.get(
        "allowed_source_types", []
    )


def test_record_normalization_policy_model_validates():
    policy = RecordNormalizationPolicy(**load_record_normalization_policy())
    assert policy.policy_name == "data_collection_record_normalization_policy"
    assert "USA" in policy.country_aliases
    assert "New Mexico" in policy.subnational_country_map
    assert "Europe" in policy.non_country_geographic_terms


def test_config_loads_record_linking_policy():
    policy_dict = load_record_linking_policy()
    assert policy_dict["policy_name"] == "data_collection_record_linking_policy"
    assert policy_dict["linking_method"] == "deterministic_event_key_linker"
    event_key_fields = policy_dict.get("event_key_fields") or []
    assert "country" in event_key_fields
    assert "date_anchor" in event_key_fields
    assert "missing_country_for_case_data" in (policy_dict.get("review_triggers") or [])


def test_record_linking_policy_model_validates():
    policy = RecordLinkingPolicy(**load_record_linking_policy())
    assert policy.policy_name == "data_collection_record_linking_policy"
    assert policy.linking_method == "deterministic_event_key_linker"
    assert "country" in policy.event_key_fields
    assert "missing_country_for_case_data" in policy.review_triggers


def test_config_loads_cross_source_consistency_policy():
    policy_dict = load_cross_source_consistency_policy()
    assert policy_dict["policy_name"] == "data_collection_cross_source_consistency_policy"
    numeric_fields = policy_dict.get("comparable_numeric_fields") or []
    assert "cases_unspecified" in numeric_fields
    assert "deaths" in numeric_fields
    assert "country" in (policy_dict.get("comparable_text_fields") or [])
    assert "date_anchor" in (policy_dict.get("comparable_date_fields") or [])
    assert "major_numeric_difference" in (policy_dict.get("human_review_triggers") or [])


def test_cross_source_consistency_policy_model_validates():
    policy = CrossSourceConsistencyPolicy(**load_cross_source_consistency_policy())
    assert policy.policy_name == "data_collection_cross_source_consistency_policy"
    assert policy.consistency_method == "deterministic_linked_event_consistency_checker"
    assert "cases_unspecified" in policy.comparable_numeric_fields
    assert "country" in policy.comparable_text_fields
    assert "date_anchor" in policy.comparable_date_fields


def test_config_loads_fixture_documents():
    catalog_dict = load_hantavirus_fixture_documents()
    assert catalog_dict["catalog_name"] == "hantavirus_fixture_document_catalog"
    fixtures = catalog_dict.get("fixture_documents") or []
    assert len(fixtures) >= 4
    for fixture in fixtures:
        assert fixture.get("fixture_id")
        assert fixture.get("source_id")
        assert fixture.get("clean_text")


def test_fixture_document_catalog_model_validates():
    catalog = FixtureDocumentCatalog(**load_hantavirus_fixture_documents())
    assert catalog.catalog_name == "hantavirus_fixture_document_catalog"
    assert len(catalog.fixture_documents) >= 4
    source_ids = {f.source_id for f in catalog.fixture_documents}
    assert "src_cdc_reported_cases" in source_ids
    assert "src_who_hantavirus_fact_sheet" in source_ids


def test_config_loads_human_review_policy():
    policy_dict = load_human_review_policy()
    assert policy_dict["policy_name"] == "data_collection_human_review_policy"
    assert "cross_source_conflict" in (policy_dict.get("review_item_types") or [])
    allowed = policy_dict.get("allowed_decisions") or []
    assert "accept" in allowed
    assert "needs_more_evidence" in allowed
    priority = policy_dict.get("priority_by_item_type") or {}
    assert priority.get("cross_source_conflict") == 1


def test_human_review_policy_model_validates():
    policy = HumanReviewPolicy(**load_human_review_policy())
    assert policy.policy_name == "data_collection_human_review_policy"
    assert "needs_more_evidence" in policy.allowed_decisions
    assert policy.priority_by_item_type["cross_source_conflict"] == 1


def test_human_review_decision_model_validates():
    decision = HumanReviewDecision(
        review_id="review_test_001",
        decision="needs_more_evidence",
        reviewer_id="tester",
        notes="Need more evidence.",
    )
    assert decision.review_id == "review_test_001"
    assert decision.decision == "needs_more_evidence"
    assert decision.reviewer_id == "tester"
    assert decision.modified_values == {}


def test_config_loads_final_package_policy():
    policy_dict = load_final_package_policy()
    assert policy_dict["policy_name"] == "data_collection_final_package_policy"
    field_order = policy_dict.get("final_dataset_field_order") or []
    assert "record_id" in field_order
    assert "source_url" in field_order
    for field in [
        "metric_name",
        "metric_value",
        "metric_unit",
        "metric_category",
        "metric_denominator",
        "metric_period_start",
        "metric_period_end",
    ]:
        assert field in field_order
    summary_fields = policy_dict.get("workflow_summary_fields") or []
    assert "human_review_summary" in summary_fields
    exportable_sections = policy_dict.get("exportable_sections") or []
    assert "final_dataset" in exportable_sections
    for field in [
        "task_acceptance_contract",
        "evidence_strategy_plan",
        "source_triage_results",
        "chunk_relevance_assessments",
        "record_task_fit_assessments",
        "collection_decision_summary",
    ]:
        assert field in summary_fields
        assert field in exportable_sections


def test_final_package_policy_model_validates():
    policy = FinalPackagePolicy(**load_final_package_policy())
    assert policy.policy_name == "data_collection_final_package_policy"
    assert policy.package_builder == "deterministic_final_data_package_builder"
    assert "record_id" in policy.final_dataset_field_order
    assert "human_review_summary" in policy.workflow_summary_fields


def test_config_loads_llm_structured_extraction_policy():
    policy_dict = load_llm_structured_extraction_policy()
    assert policy_dict["policy_name"] == "generic_public_health_llm_structured_extraction_policy"
    assert policy_dict.get("required_output_rules"), "required_output_rules empty"
    assert "data_extraction" in (policy_dict.get("allowed_fetch_purposes") or [])
    assert "text" in (policy_dict.get("allowed_chunk_kinds") or [])


def test_llm_structured_extraction_policy_model_validates():
    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    assert policy.policy_name == "generic_public_health_llm_structured_extraction_policy"
    assert policy.llm_extraction_method == "llm_structured_output_extractor"
    assert policy.max_records_per_chunk >= 1


def test_llm_extraction_output_model_validates():
    output = LLMExtractionOutput(
        records=[
            LLMExtractedRecord(
                disease="Hantavirus disease",
                country="Country X",
                cases_unspecified=1,
            )
        ],
        chunk_is_relevant=True,
    )
    assert output.chunk_is_relevant is True
    assert len(output.records) == 1
    assert output.records[0].country == "Country X"
