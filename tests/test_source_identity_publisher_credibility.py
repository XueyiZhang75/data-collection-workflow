from __future__ import annotations

import sys
from pathlib import Path
from typing import get_args

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.claim_corroboration import (  # noqa: E402
    build_claims_from_state,
    build_corroborated_events,
    compare_claims,
)
from data_collection_workflow.export import export_final_data_package  # noqa: E402
from data_collection_workflow.models import (  # noqa: E402
    ClaimSupportRole,
    RecommendedExtractionUse,
    RecommendedFetchUse,
    SourceTypeFinal,
)
from data_collection_workflow.source_identity import (  # noqa: E402
    apply_source_identity_routing_guardrails,
    apply_source_identity_to_registry,
    assess_source_identity,
    build_source_identity_summary,
    enrich_source_identity_post_fetch,
    enrich_source_identity_registry_post_fetch,
    lookup_source_identity_registry,
)


def _entry(source_id: str = "src_test", **overrides) -> dict:
    row = {
        "source_id": source_id,
        "canonical_url": "https://www.vdh.virginia.gov/hantavirus",
        "title": "Hantavirus - Virginia Department of Health",
        "publisher": "Tavily",
        "source_type": "official_public_health_agency",
        "snippet": "Hantavirus information from Virginia.",
        "status": "registered",
        "search_provider": "tavily",
        "search_rank": 1,
        "result_source": "Tavily",
        "query_used": "hantavirus Virginia 2025 cases",
        "query_id": "q1",
        "discovery_method": "live_search_result",
        "ready_for_content_fetch": True,
        "source_role_final": "collection",
        "routing_flags": [],
        "warnings": [],
    }
    row.update(overrides)
    return row


def _spec() -> dict:
    return {
        "disease": "hantavirus",
        "geography": "Virginia",
        "start_date": "2025-01-01",
        "end_date": "2026-06-01",
        "time_window": "2025-01-01 to 2026-06-01",
    }


def _record(record_id: str, source_id: str, **overrides) -> dict:
    row = {
        "record_id": record_id,
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "country": "United States of America",
        "subnational_location": "Virginia",
        "geographic_scope": "Virginia",
        "date_reported": "2025-06-01",
        "reporting_period": "2025",
        "cases_confirmed": 1.0,
        "statistical_count_type": "incident",
        "count_semantics": "confirmed case count",
        "count_unit": "persons",
        "source_id": source_id,
        "source_url": f"https://example.org/{source_id}",
        "source_title": "Virginia hantavirus report",
        "publisher": "Search metadata publisher",
        "source_type": "official_public_health_agency",
        "source_role_final": "collection",
        "credibility_score": 0.9,
        "credibility_level": "high",
        "supporting_chunk_id": f"chunk_{record_id}",
        "evidence_quote": "Virginia reported one confirmed hantavirus case.",
        "extraction_method": "fixture_extractor",
        "extraction_confidence": 0.9,
        "normalization_status": "normalized",
        "schema_status": "valid",
        "provenance_status": "verified",
        "event_cluster_id": "event_virginia_2025",
        "linked_event_id": "linked_virginia_2025",
        "countable": True,
    }
    row.update(overrides)
    return row


def test_source_identity_allowed_values_match_optimization_contract():
    assert set(get_args(SourceTypeFinal)) == {
        "official_public_health_agency",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "international_public_health_agency",
        "academic_or_peer_reviewed_source",
        "structured_database",
        "hospital_or_health_system",
        "news_media",
        "secondary_aggregator",
        "social_media",
        "personal_blog_or_forum",
        "commercial_site",
        "search_endpoint",
        "background_fact_sheet",
        "public_health_context_page",
        "unknown",
    }
    assert set(get_args(ClaimSupportRole)) == {
        "primary_case_claim_support",
        "corroboration_support",
        "zero_case_statement_support",
        "exposure_monitoring_support",
        "context_only",
        "search_discovery_only",
        "not_task_relevant",
        "insufficient_information",
    }
    assert set(get_args(RecommendedFetchUse)) == {
        "fetch_for_extraction",
        "fetch_for_context",
        "fetch_only_after_review",
        "do_not_fetch",
        "already_fetched_review_only",
        "insufficient_information",
    }
    assert set(get_args(RecommendedExtractionUse)) == {
        "extract_primary_case_claims",
        "extract_public_health_observations",
        "extract_context_only",
        "do_not_extract",
        "needs_human_review",
        "insufficient_information",
    }


def test_registry_recognizes_state_specialty_public_health_surveillance_domain():
    entry = lookup_source_identity_registry("https://westnile.ca.gov/resources.php")

    assert entry is not None
    assert entry["publisher_name"] == "California Department of Public Health"
    assert entry["source_type"] == "state_or_local_public_health_agency"
    assert entry["jurisdiction_name"] == "California"


def test_registry_recognizes_global_outbreak_authority_domains():
    cases = [
        (
            "https://pathoplexus.org/pathogen/hantavirus",
            "Pathoplexus",
            "structured_database",
            "structured_database",
        ),
        (
            "https://www.nejm.org/doi/full/example",
            "New England Journal of Medicine",
            "academic_or_peer_reviewed_source",
            "academic_source",
        ),
        (
            "https://www.eurosurveillance.org/content/example",
            "Eurosurveillance",
            "academic_or_peer_reviewed_source",
            "academic_source",
        ),
        (
            "https://www.science.org/doi/example",
            "Science",
            "academic_or_peer_reviewed_source",
            "academic_source",
        ),
        (
            "https://www.canada.ca/en/public-health/services/diseases.html",
            "Government of Canada",
            "national_public_health_agency",
            "official_authority",
        ),
        (
            "https://www.rivm.nl/en/hantavirus",
            "National Institute for Public Health and the Environment",
            "national_public_health_agency",
            "official_authority",
        ),
        (
            "https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/",
            "Ministry of Health of Spain",
            "national_public_health_agency",
            "official_authority",
        ),
        (
            "https://sante.gouv.fr/soins-et-maladies/maladies/",
            "French Ministry of Health",
            "national_public_health_agency",
            "official_authority",
        ),
        (
            "https://www.gov.uk/government/publications/outbreaks-under-monitoring",
            "UK Health Security Agency",
            "national_public_health_agency",
            "official_authority",
        ),
        (
            "https://ukhsa-newsroom.prgloo.com/news/hantavirus-update",
            "UK Health Security Agency Newsroom",
            "national_public_health_agency",
            "official_authority",
        ),
    ]

    for url, publisher, source_type, authority_bucket in cases:
        registry_entry = lookup_source_identity_registry(url)
        assessment = assess_source_identity(
            _entry(
                canonical_url=url,
                title=f"{publisher} hantavirus update",
                publisher="Tavily",
                result_source="Tavily",
                source_type="news_and_situation_report",
            ),
            collection_spec={"disease": "hantavirus", "geography": "global"},
        )

        assert registry_entry is not None
        assert registry_entry["publisher_name"] == publisher
        assert registry_entry["source_type"] == source_type
        assert assessment["actual_publisher"] == publisher
        assert assessment["source_type_final"] == source_type
        assert assessment["authority_bucket"] == authority_bucket


def test_search_provider_is_not_treated_as_publisher():
    assessment = assess_source_identity(_entry(), collection_spec=_spec())

    assert assessment["search_provider"] == "tavily"
    assert assessment["search_result_source_raw"] == "Tavily"
    assert assessment["actual_publisher"] != "Tavily"
    assert assessment["actual_publisher"] == "Virginia Department of Health"
    assert "search_provider_not_publisher" in assessment["warnings"]


def test_secondary_and_social_sources_are_not_official_authority():
    cases = [
        (
            "https://www.ajmc.com/view/hantavirus-outbreak",
            "American Journal of Managed Care",
            "news_media",
            "secondary_media",
        ),
        (
            "https://outbreaknewstoday.substack.com/p/hantavirus-in-the-americas",
            "Outbreak News Today",
            "secondary_aggregator",
            "secondary_aggregator",
        ),
        (
            "https://en.wikipedia.org/wiki/MV_Hondius_hantavirus_outbreak",
            "Wikipedia",
            "secondary_aggregator",
            "secondary_aggregator",
        ),
        (
            "https://www.instagram.com/reel/DYE-JUXjp_K",
            "Instagram",
            "social_media",
            "social_media",
        ),
    ]

    for url, publisher, source_type, expected_bucket in cases:
        assessment = assess_source_identity(
            _entry(
                canonical_url=url,
                title="Hantavirus source",
                publisher=publisher,
                result_source="Tavily",
                source_type=source_type,
            ),
            collection_spec={"disease": "hantavirus", "geography": "global"},
        )

        assert assessment["authority_bucket"] == expected_bucket
        assert assessment["source_type_final"] != "official_public_health_agency"
        assert assessment["source_type_final"] not in {
            "national_public_health_agency",
            "international_public_health_agency",
            "state_or_local_public_health_agency",
        }


def test_news_domain_quoting_official_agency_is_not_promoted_to_official():
    assessment = assess_source_identity(
        _entry(
            "src_nvdaily_vdh_quote",
            canonical_url="https://www.nvdaily.com/news/measles-cases-virginia-2024/article_123.html",
            title="Virginia measles cases rise, VDH says",
            publisher="Northern Virginia Daily",
            result_source="Tavily",
            source_type="news_media",
            snippet="The Virginia Department of Health said measles cases were reported in the past month.",
        ),
        collection_spec={"disease": "measles", "geography": "Virginia"},
    )

    assert assessment["actual_publisher"] != "Virginia Department of Health"
    assert assessment["source_type_final"] in {"news_media", "secondary_aggregator"}
    assert assessment["source_type_final"] not in {
        "official_public_health_agency",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "international_public_health_agency",
    }
    assert assessment["recommended_source_role"] in {
        "context",
        "collection_support",
        "needs_human_review",
    }


def test_academic_secondary_domain_quoting_cdc_is_not_promoted_to_cdc_publisher():
    assessment = assess_source_identity(
        _entry(
            "src_cidrap_cdc_quote",
            canonical_url="https://www.cidrap.umn.edu/influenza/cdc-reports-flu-activity",
            title="CDC reports flu activity in weekly update",
            publisher="CIDRAP",
            result_source="Tavily",
            source_type="secondary_aggregator",
            snippet="CIDRAP summarizes CDC FluView surveillance data.",
        ),
        collection_spec={"disease": "flu", "geography": "United States"},
    )

    assert assessment["actual_publisher"] != "Centers for Disease Control and Prevention"
    assert assessment["source_type_final"] in {
        "academic_or_peer_reviewed_source",
        "secondary_aggregator",
        "news_media",
    }
    assert assessment["source_type_final"] not in {
        "official_public_health_agency",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "international_public_health_agency",
    }


def test_global_task_routes_local_health_department_as_local_context():
    assessment = assess_source_identity(
        _entry(
            canonical_url="https://www.nmhealth.org/about/erd/ideb/zdp/hps",
            title="Hantavirus Pulmonary Syndrome - New Mexico Department of Health",
            publisher="Tavily",
            result_source="Tavily",
            source_type="official_public_health_agency",
        ),
        collection_spec={"disease": "hantavirus", "geography": "global"},
    )

    assert assessment["actual_publisher"] == "New Mexico Department of Health"
    assert assessment["source_type_final"] == "state_or_local_public_health_agency"
    assert assessment["jurisdiction_scope"] == "subnational"
    assert assessment["recommended_source_role"] in {"context", "collection_support"}
    assert assessment["authority_bucket"] == "local_official_context"


def test_llm_identity_assessment_can_set_actual_publisher(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "Virginia Department of Health",
            "actual_publisher_normalized": "virginia_department_of_health",
            "actual_publisher_confidence": "high",
            "source_type_final": "state_or_local_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "exposure_monitoring_support",
            "recommended_source_role": "context",
            "recommended_fetch_use": "fetch_for_context",
            "recommended_extraction_use": "extract_context_only",
            "credibility_level_llm": "high",
            "credibility_rationale": "Metadata points to VDH and monitoring context.",
            "trust_basis": "Official state health department metadata.",
            "source_independence_group": "publisher:virginia_department_of_health",
            "warnings": [],
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, assessments, summary = apply_source_identity_to_registry(
        [_entry("src_vdh_llm", publisher="Tavily", result_source="Tavily")],
        collection_spec=_spec(),
        llm_enabled=True,
        max_sources=1,
    )

    assert updated[0]["actual_publisher"] == "Virginia Department of Health"
    assert updated[0]["source_type_final"] == "state_or_local_public_health_agency"
    assert updated[0]["claim_support_role"] == "exposure_monitoring_support"
    assert assessments[0]["llm_used"] is True
    assert summary["llm_identity_assessed_count"] == 1


def test_must_fetch_target_official_source_cannot_be_overwritten_to_excluded():
    entry = _entry(
        "src_vdh_week_40",
        canonical_url="https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-40.pdf",
        title="Weekly-RDS-Report_Week-40.pdf",
        publisher="Virginia Department of Health",
        must_fetch=True,
        must_fetch_reason="Target Virginia official weekly surveillance report.",
        coverage_requirement_ids=["virginia_influenza_official_week_40_2024"],
        source_identity_llm_used=True,
        recommended_source_role="excluded",
        recommended_fetch_use="do_not_fetch",
        recommended_extraction_use="do_not_extract",
        final_screening_decision="include_for_content_fetch",
        source_role_final="collection",
        ready_for_content_fetch=True,
    )

    updated = apply_source_identity_routing_guardrails(entry)

    assert updated["source_role_final"] == "collection"
    assert updated["final_screening_decision"] == "include_for_content_fetch"
    assert updated["ready_for_content_fetch"] is True
    assert updated.get("blocked_from_fetch") is not True
    assert "source_identity_recommended_excluded_for_must_fetch" in updated["routing_flags"]
    assert "source_identity_do_not_fetch_for_must_fetch" in updated["routing_flags"]


def test_llm_source_triage_controls_role_and_fetch_use(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "Centers for Disease Control and Prevention",
            "actual_publisher_normalized": "centers_for_disease_control_and_prevention",
            "actual_publisher_confidence": "high",
            "source_type_final": "national_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "corroboration_support",
            "recommended_source_role": "validation",
            "recommended_fetch_use": "fetch_for_extraction",
            "recommended_extraction_use": "extract_public_health_observations",
            "credibility_level_llm": "high",
            "credibility_rationale": "Official national public health source.",
            "trust_basis": "Official CDC domain and task-compatible surveillance wording.",
            "source_independence_group": "publisher:centers_for_disease_control_and_prevention",
            "warnings": [],
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, assessments, summary = apply_source_identity_to_registry(
        [
            _entry(
                "src_cdc_validation",
                canonical_url="https://www.cdc.gov/hantavirus/surveillance",
                title="Hantavirus reported cases surveillance",
                publisher="Tavily",
                result_source="Tavily",
                source_role_final="collection",
                source_type="news_and_situation_report",
            )
        ],
        collection_spec=_spec(),
        llm_enabled=True,
        require_llm=True,
        allow_deterministic_fallback=False,
    )

    assert assessments[0]["llm_used"] is True
    assert updated[0]["source_role_final"] == "validation"
    assert updated[0]["ready_for_content_fetch"] is True
    assert updated[0]["recommended_fetch_use"] == "fetch_for_extraction"
    assert updated[0]["recommended_extraction_use"] == "extract_public_health_observations"
    assert summary["claim_support_role_counts"]["corroboration_support"] == 1


def test_direct_collection_source_identity_fast_path_skips_non_target_sources(
    monkeypatch,
):
    calls: list[str] = []

    def fake_llm_identity(source_entry, **_kwargs):
        calls.append(source_entry.get("source_id"))
        return {
            "actual_publisher": "Centers for Disease Control and Prevention",
            "actual_publisher_normalized": "centers_for_disease_control_and_prevention",
            "actual_publisher_confidence": "high",
            "source_type_final": "national_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "primary_case_claim_support",
            "recommended_source_role": "collection",
            "recommended_fetch_use": "fetch_for_extraction",
            "recommended_extraction_use": "extract_public_health_observations",
            "credibility_level_llm": "high",
            "trust_basis": "Official target report.",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, assessments, summary = apply_source_identity_to_registry(
        [
            _entry(
                "src_cdc_week_40",
                canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                title="CDC FluView Week 40",
                publisher="CDC",
                source_type="official_public_health_agency",
                must_fetch=True,
            ),
            _entry(
                "src_cdc_context",
                canonical_url="https://www.cdc.gov/fluview/surveillance/2025-week-04.html",
                title="CDC FluView Week 04 2025",
                publisher="CDC",
                source_type="official_public_health_agency",
            ),
            _entry(
                "src_forum",
                canonical_url="https://flutrackers.com/forum/example",
                title="Forum thread",
                publisher="FluTrackers",
                source_type="personal_blog_or_forum",
            ),
        ],
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        llm_enabled=True,
        require_llm=True,
        allow_deterministic_fallback=False,
    )

    assert calls == []
    assert updated[0]["source_identity_llm_used"] is False
    assert updated[1]["source_identity_llm_used"] is False
    assert updated[2]["source_identity_llm_used"] is False
    assert summary["llm_identity_assessed_count"] == 0
    assert summary["direct_identity_fast_path"] is True
    assert summary["direct_identity_fast_path_skipped_count"] == 3
    assert {
        item["source_id"]
        for item in assessments
        if item.get("source_identity_llm_skipped_reason")
        == "direct_target_official_fast_path_skips_source_identity"
    } == {"src_cdc_week_40", "src_cdc_context", "src_forum"}


def test_direct_collection_post_fetch_identity_fast_path_skips_verified_targets(
    monkeypatch,
):
    calls: list[str] = []

    def fake_llm_identity(source_entry, **_kwargs):
        calls.append(source_entry.get("source_id"))
        return {
            "actual_publisher": "Centers for Disease Control and Prevention",
            "source_type_final": "national_public_health_agency",
            "claim_support_role": "primary_case_claim_support",
            "recommended_fetch_use": "fetch_for_extraction",
            "recommended_extraction_use": "extract_public_health_observations",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    registry = [
        _entry(
            "src_cdc_week_40",
            canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
            title="CDC FluView Week 40",
            publisher="CDC",
            source_type="official_public_health_agency",
            source_type_final="national_public_health_agency",
            must_fetch=True,
            coverage_requirement_ids=["req_week_40"],
        )
    ]
    assessments = [
        assess_source_identity(
            registry[0],
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
        )
    ]
    documents = [
        {
            "source_id": "src_cdc_week_40",
            "title": "CDC FluView Week 40",
            "clean_text": "Centers for Disease Control and Prevention FluView report.",
            "parse_status": "parsed_text",
            "http_status_code": 200,
        }
    ]

    updated, post_fetch_assessments, summary = enrich_source_identity_registry_post_fetch(
        registry,
        assessments,
        documents,
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        llm_enabled=True,
        require_llm=True,
        allow_deterministic_fallback=False,
    )

    assert calls == []
    assert updated[0]["source_identity_llm_used"] is False
    assert post_fetch_assessments[0]["llm_used"] is False
    assert post_fetch_assessments[0]["post_fetch_identity_assessed"] is True
    assert (
        post_fetch_assessments[0]["source_identity_llm_skipped_reason"]
        == "direct_target_official_fast_path_skips_source_identity"
    )
    assert summary["llm_identity_assessed_count"] == 0
    assert summary["post_fetch_identity_assessed_count"] == 1


def test_secondary_aggregator_is_not_classified_as_official(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "source_type_final": "secondary_aggregator",
            "claim_support_role": "corroboration_support",
            "recommended_source_role": "collection_support",
            "recommended_extraction_use": "needs_human_review",
            "actual_publisher": "Outbreak News Today",
            "actual_publisher_normalized": "outbreak_news_today",
            "actual_publisher_confidence": "medium",
            "source_independence_group": "publisher:outbreak_news_today",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_aggregator",
                canonical_url="https://outbreaknewstoday.substack.com/p/hantavirus-virginia",
                title="Hantavirus Virginia case report",
                publisher="Tavily",
                result_source="Tavily",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec=_spec(),
        llm_enabled=True,
    )

    assert updated[0]["source_type_final"] == "secondary_aggregator"
    assert updated[0]["source_type_final"] != "official_public_health_agency"
    assert updated[0]["claim_support_role"] != "primary_case_claim_support"


def test_news_source_can_support_but_not_automatically_official(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "WSLS",
            "actual_publisher_normalized": "wsls",
            "actual_publisher_confidence": "high",
            "source_type_final": "news_media",
            "source_type_confidence": "high",
            "claim_support_role": "corroboration_support",
            "recommended_source_role": "collection_support",
            "recommended_extraction_use": "needs_human_review",
            "source_independence_group": "publisher:wsls",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_news",
                canonical_url="https://www.wsls.com/news/virginia/hantavirus-case",
                title="Virginia hantavirus case",
                publisher="Tavily",
                result_source="Tavily",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec=_spec(),
        llm_enabled=True,
    )

    assert updated[0]["source_type_final"] == "news_media"
    assert updated[0]["claim_support_role"] == "corroboration_support"
    assert updated[0]["source_type_final"] != "official_public_health_agency"


def test_news_domain_cannot_be_upgraded_to_official_by_upstream_mentions(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "Virginia Department of Health",
            "actual_publisher_normalized": "virginia_department_of_health",
            "actual_publisher_confidence": "high",
            "source_type_final": "official_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "primary_case_claim_support",
            "recommended_source_role": "collection",
            "recommended_extraction_use": "extract_primary_case_claims",
            "source_independence_group": "publisher:virginia_department_of_health",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_nvdaily",
                canonical_url="https://www.nvdaily.com/nvdaily/measles-on-the-rise-in-virginia/article_df1aefc2-8513-51b2-9516-37885787e942.html",
                title="Measles on the rise in Virginia",
                snippet="The article quotes the Virginia Department of Health about measles outbreaks.",
                publisher="Tavily",
                result_source="Tavily",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec={
            "disease": "Measles",
            "geography": "Virginia",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
        llm_enabled=True,
    )

    row = updated[0]
    assert row["source_type_final"] == "news_media"
    assert row["source_type_final"] != "official_public_health_agency"
    assert row.get("actual_publisher") != "Virginia Department of Health"
    assert row["recommended_source_role"] in {"collection_support", "needs_human_review"}
    assert "llm_official_identity_conflicts_with_domain" in row["source_identity_warnings"]


def test_unknown_commercial_health_domain_with_official_metadata_is_unverified_not_official():
    assessment = assess_source_identity(
        _entry(
            "src_woodsidemd",
            canonical_url="https://www.woodsidemd.com/post/hantavirus-what-you-need-to-know-about-the-2026-global-surge",
            title="Hantavirus: what you need to know about the 2026 global surge",
            publisher="Centers for Disease Control and Prevention",
            result_source="Centers for Disease Control and Prevention",
            source_type="official_public_health_agency",
            snippet=(
                "CDC and WHO updates describe seven reported cases among "
                "passengers and three fatalities."
            ),
        ),
        collection_spec={"disease": "hantavirus", "geography": "Global"},
    )

    assert assessment["domain"] == "www.woodsidemd.com"
    assert assessment["source_type_final"] not in {
        "official_public_health_agency",
        "national_public_health_agency",
        "international_public_health_agency",
        "state_or_local_public_health_agency",
    }
    assert assessment["authority_bucket"] != "official_authority"
    assert assessment["actual_publisher"] != "Centers for Disease Control and Prevention"
    assert assessment.get("source_identity_unverified") is True
    assert assessment.get("metadata_only_identity") is True
    assert "official_source_type_claim_without_domain_support" in assessment["warnings"]


def test_unknown_news_domain_with_llm_official_claim_is_not_promoted_to_official(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "World Health Organization",
            "actual_publisher_confidence": "high",
            "source_type_final": "official_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "primary_case_claim_support",
            "recommended_source_role": "collection",
            "recommended_extraction_use": "extract_primary_case_claims",
            "source_independence_group": "publisher:world_health_organization",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_ndtv",
                canonical_url="https://www.ndtv.com/health/hantavirus-confirmed-cases-by-nationality-11478772",
                title="Hantavirus confirmed cases by nationality",
                snippet="NDTV reports confirmed cases by nationality, citing WHO updates.",
                publisher="NDTV",
                result_source="NDTV",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec={"disease": "hantavirus", "geography": "Global"},
        llm_enabled=True,
    )

    row = updated[0]
    assert row["source_type_final"] in {"news_media", "unknown", "commercial_site"}
    assert row["source_type_final"] != "official_public_health_agency"
    assert row["actual_publisher"] != "World Health Organization"
    assert row.get("source_identity_unverified") is True
    assert {
        "llm_official_identity_conflicts_with_unverified_domain",
        "llm_official_identity_conflicts_with_domain",
    } & set(row["source_identity_warnings"])


def test_academic_secondary_domain_keeps_own_publisher_when_quoting_cdc(monkeypatch):
    def fake_llm_identity(**_kwargs):
        return {
            "actual_publisher": "Centers for Disease Control and Prevention",
            "actual_publisher_normalized": "centers_for_disease_control_and_prevention",
            "actual_publisher_confidence": "high",
            "source_type_final": "official_public_health_agency",
            "source_type_confidence": "high",
            "claim_support_role": "primary_case_claim_support",
            "recommended_source_role": "collection",
            "recommended_extraction_use": "extract_primary_case_claims",
            "source_independence_group": "publisher:centers_for_disease_control_and_prevention",
        }

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        fake_llm_identity,
    )

    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_cidrap",
                canonical_url="https://www.cidrap.umn.edu/measles/us-measles-cases-continue-climb-especially-virginia",
                title="US measles cases continue to climb, especially in Virginia - CIDRAP",
                snippet="CIDRAP summarizes CDC and VDH measles case reports.",
                publisher="Tavily",
                result_source="Tavily",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec={
            "disease": "Measles",
            "geography": "Virginia",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
        },
        llm_enabled=True,
    )

    row = updated[0]
    assert row["actual_publisher"] == "CIDRAP"
    assert row["actual_publisher"] != "Centers for Disease Control and Prevention"
    assert row["source_type_final"] == "academic_or_peer_reviewed_source"
    assert row["source_type_final"] != "official_public_health_agency"
    assert "llm_official_identity_conflicts_with_domain" in row["source_identity_warnings"]


def test_social_media_source_is_not_treated_as_collection_official():
    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_social",
                canonical_url="https://www.facebook.com/vdh/posts/hantavirus",
                title="VDH Facebook post about hantavirus",
                publisher="Tavily",
                result_source="Tavily",
                source_type="official_public_health_agency",
            )
        ],
        collection_spec=_spec(),
        llm_enabled=False,
    )

    assert updated[0]["source_type_final"] == "social_media"
    assert updated[0]["recommended_source_role"] in {"needs_human_review", "context"}
    assert updated[0]["source_type_final"] != "official_public_health_agency"


def test_search_endpoint_not_fetchable_for_extraction():
    updated, _, _ = apply_source_identity_to_registry(
        [
            _entry(
                "src_pubmed_search",
                canonical_url="https://pubmed.ncbi.nlm.nih.gov/?term=hantavirus+Virginia",
                title="PubMed search results",
                publisher="PubMed",
                result_source="PubMed",
                source_type="literature_api",
            )
        ],
        collection_spec=_spec(),
        llm_enabled=False,
    )

    assert updated[0]["source_type_final"] == "search_endpoint"
    assert updated[0]["recommended_fetch_use"] == "do_not_fetch"
    assert updated[0]["recommended_extraction_use"] == "do_not_extract"
    assert updated[0]["ready_for_content_fetch"] is False


def test_post_fetch_page_metadata_corrects_publisher():
    entry = _entry(
        "src_unknown",
        canonical_url="https://www.vdh.virginia.gov/hantavirus",
        publisher=None,
        result_source=None,
    )
    assessment = assess_source_identity(entry, collection_spec=_spec())
    enriched = enrich_source_identity_post_fetch(
        entry,
        assessment,
        {
            "source_id": "src_unknown",
            "canonical_url": "https://www.vdh.virginia.gov/hantavirus",
            "title": "Hantavirus - Virginia Department of Health",
            "clean_text": "Virginia Department of Health Hantavirus information and monitoring guidance.",
            "parse_status": "parsed_html",
            "http_status_code": 200,
        },
    )

    assert enriched["post_fetch_identity_assessed"] is True
    assert enriched["actual_publisher"] == "Virginia Department of Health"
    assert enriched["actual_publisher_confidence"] in {"medium", "high"}
    assert "page_title" in enriched["publisher_evidence_fields"]


def test_post_fetch_short_publisher_alias_requires_token_match():
    source = _entry(
        "src_private_clinic",
        canonical_url="https://priveclinics.com/hantavirus-outbreak-2026",
        title="Hantavirus Outbreak 2026 | Private clinic",
        publisher="Tavily",
        result_source="Tavily",
        source_type="international_organization_report",
    )
    assessment = assess_source_identity(source, collection_spec=_spec())

    enriched = enrich_source_identity_post_fetch(
        source,
        assessment,
        {
            "source_id": "src_private_clinic",
            "canonical_url": "https://priveclinics.com/hantavirus-outbreak-2026",
            "title": "Hantavirus Outbreak 2026 | Private clinic",
            "metadata": {},
            "clean_text": (
                "Private clinic appointment booking page describing a hantavirus "
                "outbreak. The page includes appointment and map links but no "
                "publisher byline."
            ),
        },
    )

    assert enriched["actual_publisher"] != "Associated Press"
    assert enriched["publisher_source"] != "page_metadata_or_text"


def test_conflicting_search_and_page_metadata_lowers_confidence():
    entry = _entry(
        "src_conflict",
        canonical_url="https://www.vdh.virginia.gov/hantavirus",
        publisher="Reuters",
        result_source="Reuters",
        search_provider="tavily",
    )
    assessment = assess_source_identity(entry, collection_spec=_spec())
    enriched = enrich_source_identity_post_fetch(
        entry,
        assessment,
        {
            "source_id": "src_conflict",
            "title": "Hantavirus - Virginia Department of Health",
            "clean_text": "Virginia Department of Health page on hantavirus.",
            "parse_status": "parsed_html",
            "http_status_code": 200,
        },
    )

    assert "publisher_conflict_between_search_and_page_metadata" in enriched["warnings"]
    assert enriched["actual_publisher_confidence"] != "high"


def test_claim_provenance_uses_actual_publisher_and_independence_group():
    state = {
        "normalized_records": [_record("rec_a", "src_a")],
        "final_dataset_pre_quality_gate": [_record("rec_a", "src_a")],
        "source_registry": [
            _entry(
                "src_a",
                canonical_url="https://www.vdh.virginia.gov/hantavirus",
                actual_publisher="Virginia Department of Health",
                actual_publisher_normalized="virginia_department_of_health",
                source_independence_group="publisher:virginia_department_of_health",
                source_type_final="state_or_local_public_health_agency",
                claim_support_role="primary_case_claim_support",
            )
        ],
    }

    claims = build_claims_from_state(state)

    assert claims[0]["actual_publisher"] == "Virginia Department of Health"
    assert (
        claims[0]["source_independence_group"]
        == "publisher:virginia_department_of_health"
    )
    assert claims[0]["claim_support_role"] == "primary_case_claim_support"


def test_same_upstream_aggregated_sources_do_not_count_as_independent():
    state = {
        "normalized_records": [
            _record("rec_a", "src_news_a", source_url="https://news-a.example/case"),
            _record("rec_b", "src_news_b", source_url="https://news-b.example/case"),
        ],
        "source_registry": [
            _entry(
                "src_news_a",
                canonical_url="https://news-a.example/case",
                actual_publisher="News A",
                source_independence_group="upstream:virginia_department_of_health",
                likely_syndicated_or_aggregated=True,
                upstream_source_mentions=["Virginia Department of Health"],
                source_type_final="news_media",
            ),
            _entry(
                "src_news_b",
                canonical_url="https://news-b.example/case",
                actual_publisher="News B",
                source_independence_group="upstream:virginia_department_of_health",
                likely_syndicated_or_aggregated=True,
                upstream_source_mentions=["Virginia Department of Health"],
                source_type_final="news_media",
            ),
        ],
    }

    claims = build_claims_from_state(state)
    comparisons = compare_claims(claims)
    events = build_corroborated_events(claims, comparisons)

    assert comparisons[0]["source_independence_status"] == "same_source"
    assert comparisons[0]["corroboration_match_status"] == "duplicate_same_source"
    assert events[0]["independent_source_count"] == 1
    assert events[0]["corroboration_status"] != "corroborated"


def test_source_identity_summary_and_exports(tmp_path):
    registry, assessments, summary = apply_source_identity_to_registry(
        [_entry("src_export")],
        collection_spec=_spec(),
        llm_enabled=False,
    )
    rebuilt_summary = build_source_identity_summary(assessments)

    package = {
        "final_dataset": [],
        "source_registry": registry,
        "source_identity_assessments": assessments,
        "source_identity_summary": summary,
        "workflow_summaries": {"source_identity_summary": rebuilt_summary},
    }
    manifest = export_final_data_package(package, tmp_path)

    assert summary["identity_assessed_count"] == 1
    assert rebuilt_summary["source_type_counts"]
    assert (tmp_path / "source_identity_assessments.json").exists()
    assert (tmp_path / "source_identity_assessments.csv").exists()
    assert "source_identity_assessments_json" in manifest["files"]


def test_required_llm_identity_records_blocked_when_unavailable(monkeypatch):
    def failing_llm_identity(**_kwargs):
        raise RuntimeError("missing api key")

    monkeypatch.setattr(
        "data_collection_workflow.source_identity.assess_source_identity_with_llm",
        failing_llm_identity,
    )

    updated, assessments, summary = apply_source_identity_to_registry(
        [_entry("src_require_llm")],
        collection_spec=_spec(),
        llm_enabled=True,
        max_sources=1,
        require_llm=True,
        allow_deterministic_fallback=False,
    )

    assert assessments[0]["source_identity_status"] == "blocked_llm_required"
    assert assessments[0]["llm_used"] is False
    assert "llm_source_identity_required_but_unavailable" in assessments[0]["warnings"]
    assert updated[0]["source_identity_status"] == "blocked_llm_required"
    assert summary["blocked_llm_required_count"] == 1
