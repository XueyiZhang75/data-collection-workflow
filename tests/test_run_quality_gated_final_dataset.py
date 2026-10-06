from __future__ import annotations

import csv
import json

from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.human_review_application import apply_human_review_decisions
from data_collection_workflow.nodes.finalization import final_data_package_builder


def _record(record_id: str, **overrides) -> dict:
    source_id = overrides.get("source_id", f"src_{record_id}")
    chunk_id = overrides.get("supporting_chunk_id", f"chunk_{record_id}")
    record = {
        "record_id": record_id,
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "virus_or_syndrome": "Hantavirus pulmonary syndrome",
        "pathogen_or_syndrome": "hantavirus",
        "country": "United States of America",
        "subnational_location": "New Mexico",
        "date_reported": "2025-03-07",
        "date_anchor": "2025-03-07",
        "reporting_period": "2025",
        "cases_unspecified": 1.0,
        "deaths": 0.0,
        "source_id": source_id,
        "source_url": f"https://example.org/{source_id}",
        "source_title": "Example public health source",
        "source_type": "official_public_health_agency",
        "publisher": "Example Department of Health",
        "source_role_final": "collection",
        "credibility_score": 0.95,
        "credibility_level": "high",
        "discovery_method": "fixture_search_result",
        "search_provider": "fixture",
        "query_id": "q_fixture_001",
        "query_used": "hantavirus New Mexico cases deaths 2025",
        "evidence_quote": (
            "New Mexico reported a hantavirus pulmonary syndrome case in 2025."
        ),
        "supporting_chunk_id": chunk_id,
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "record_schema": "generic_public_health_record",
        "event_cluster_id": "event_001",
        "countable": True,
        "requires_human_review": False,
    }
    record.update(overrides)
    return record


def _source(source_id: str, **overrides) -> dict:
    row = {
        "source_id": source_id,
        "canonical_url": f"https://example.org/{source_id}",
        "title": "Example source",
        "publisher": "Example Department of Health",
        "source_type": "official_public_health_agency",
        "status": "ready_for_content_fetch",
        "source_role_final": "collection",
        "credibility_score": 0.95,
        "credibility_level": "high",
        "final_screening_decision": "include_for_content_fetch",
        "ready_for_content_fetch": True,
        "source_disease_relevance_status": "target_disease_match",
    }
    row.update(overrides)
    return row


def _document(source_id: str, **overrides) -> dict:
    row = {
        "document_id": f"doc_{source_id}",
        "source_id": source_id,
        "url": f"https://example.org/{source_id}",
        "canonical_url": f"https://example.org/{source_id}",
        "title": "Example source",
        "publisher": "Example Department of Health",
        "source_type": "official_public_health_agency",
        "source_role_final": "collection",
        "fetch_purpose": "data_extraction",
        "fetch_status": "fetched",
        "parse_status": "parsed_html",
        "quality_status": "usable",
        "document_disease_relevance_status": "target_disease_match",
        "not_extractable_for_task_disease": False,
        "clean_text": "New Mexico reported a hantavirus pulmonary syndrome case.",
    }
    row.update(overrides)
    return row


def _chunk(chunk_id: str, source_id: str, **overrides) -> dict:
    row = {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "text": "New Mexico reported a hantavirus pulmonary syndrome case.",
        "contains_target_data": True,
        "disease_relevance_status": "target_disease_match",
        "extraction_eligible_for_task_disease": True,
        "source_url": f"https://example.org/{source_id}",
        "source_role_final": "collection",
        "quality_status": "usable",
        "chunk_index": 1,
        "chunk_kind": "text",
    }
    row.update(overrides)
    return row


def _validation_result(record_id: str, **overrides) -> dict:
    row = {
        "validation_result_id": f"val_result_{record_id}",
        "validation_case_id": "val_case_001",
        "validation_type": "scope_check",
        "validation_unit": "record",
        "comparison_id": f"val_cmp_{record_id}",
        "left_record_ids": [record_id],
        "right_record_ids": [],
        "left_event_cluster_ids": ["event_001"],
        "right_event_cluster_ids": [],
        "left_source_ids": [f"src_{record_id}"],
        "right_source_ids": [],
        "left_source_urls": [f"https://example.org/src_{record_id}"],
        "right_source_urls": [],
        "compared_field": "record_scope",
        "disease": "Hantavirus disease",
        "location": "New Mexico",
        "date_or_period": "2025",
        "left_value": "New Mexico / 2025",
        "right_value": "New Mexico / 2025",
        "comparability_status": "comparable",
        "match_status": "matched",
        "validation_status": "validated",
        "confidence": 0.9,
        "reason": "record is within requested task scope",
        "evidence_summary": "scope check",
        "needs_human_review": False,
        "warnings": [],
    }
    row.update(overrides)
    return row


def _anomaly(record_id: str, **overrides) -> dict:
    row = {
        "anomaly_id": f"anom_{record_id}",
        "anomaly_type": "deaths_greater_than_cases",
        "anomaly_unit": "record",
        "severity": "high",
        "record_id": record_id,
        "source_id": f"src_{record_id}",
        "source_ids": [f"src_{record_id}"],
        "source_urls": [f"https://example.org/src_{record_id}"],
        "reason": "deaths exceed available case count for the same record",
        "recommended_action": "review_case_and_death_counts",
        "needs_human_review": True,
        "human_review_reason": "review high-severity anomaly",
    }
    row.update(overrides)
    return row


def _state(records: list[dict] | None = None, **overrides) -> dict:
    records = records if records is not None else [_record("rec_001")]
    source_ids = sorted(
        {
            str(record["source_id"])
            for record in records
            if isinstance(record, dict) and record.get("source_id")
        }
    )
    chunks = [
        _chunk(
            str(record.get("supporting_chunk_id") or f"chunk_{record['record_id']}"),
            str(record["source_id"]),
        )
        for record in records
        if isinstance(record, dict) and record.get("source_id")
    ]
    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "New Mexico",
            "start_date": "2024",
            "end_date": "2026",
            "target_fields": ["cases", "deaths", "dates", "locations", "source URLs"],
        },
        "collection_spec": {
            "disease": "hantavirus",
            "geography": "New Mexico",
            "start_date": "2024",
            "end_date": "2026",
            "time_window": "2024 to 2026",
        },
        "disease_intelligence": {
            "disease_input": "hantavirus",
            "disease_standard_name": "Hantavirus disease",
            "aliases": ["hantavirus", "hantavirus disease"],
            "abbreviations": ["HPS", "HFRS"],
            "pathogen_terms": ["hantavirus", "orthohantavirus"],
            "syndrome_terms": [
                "hantavirus pulmonary syndrome",
                "hemorrhagic fever with renal syndrome",
            ],
        },
        "normalized_records": records,
        "source_registry": [_source(source_id) for source_id in source_ids],
        "documents": [_document(source_id) for source_id in source_ids],
        "evidence_chunks": chunks,
        "linked_events": [],
        "event_clusters": [],
        "duplicate_clusters": [],
        "validation_cases": [],
        "validation_comparisons": [],
        "validation_results": [],
        "conflicts": [],
        "human_review_queue": [],
        "human_review_decisions": [],
        "collection_trace": [],
    }
    state.update(overrides)
    return state


def _finalize(state: dict) -> tuple[dict, dict]:
    result = final_data_package_builder(state)
    return result["final_data_package"], result


def _record_ids(rows: list[dict]) -> set[str]:
    return {str(row.get("record_id")) for row in rows}


def test_clean_record_is_accepted_into_quality_gated_final_dataset():
    package, result = _finalize(_state())

    assert _record_ids(package["final_dataset"]) == {"rec_001"}
    record = package["final_dataset"][0]
    assert record["final_dataset_included"] is True
    assert record["record_final_inclusion_status"] in {"accepted", "accepted_with_warnings"}
    assert package["quarantined_records"] == []
    assert result["run_quality_summary"]["run_quality_status"] in {"passed", "passed_with_review"}
    assert result["run_quality_summary"]["primary_case_dataset_eligible_count"] == 0
    assert result["final_dataset_quality_summary"]["primary_case_dataset_eligible_count"] == 0


def test_final_package_exports_evidence_product_dataset_and_bundle_metadata():
    record = _record(
        "rec_event_summary",
        source_id="src_event_report",
        source_url="https://health.example.gov/news/hantavirus-outbreak-update-2026",
        source_title="Hantavirus outbreak update",
        cases_confirmed=12.0,
        cases_probable=1.0,
        cases_unspecified=None,
        deaths=3.0,
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary", "death_record"],
        record_final_inclusion_status="pending_human_review",
        quality_gate_reasons=["official single-source outbreak candidate requires review"],
    )
    package, _ = _finalize(
        _state(
            [record],
            source_registry=[
                _source(
                    "src_event_report",
                    canonical_url="https://health.example.gov/news/hantavirus-outbreak-update-2026",
                    title="Hantavirus outbreak update",
                    data_product_type="event_outbreak_report",
                    expected_evidence_role="aggregate_event_evidence",
                    task_specificity="event_specific",
                    machine_readability="narrative_extractable",
                    authority_bucket="official_authority",
                    source_type_final="national_public_health_agency",
                )
            ],
        )
    )

    evidence_rows = package["evidence_product_dataset"]
    assert evidence_rows
    evidence = evidence_rows[0]
    assert evidence["source_product_type"] == "event_outbreak_report"
    assert evidence["evidence_role"] == "aggregate_event_evidence"
    assert evidence["candidate_type"] == "aggregate_event_candidate"
    assert evidence["case_count"] == 13.0
    assert evidence["death_count"] == 3.0

    bundle = package["case_evidence_bundles"][0]
    assert bundle["bundle_type"] == "aggregate_event"
    assert bundle["verified_authority_source_count"] == 1
    assert bundle["main_blocking_reason"]
    assert bundle["recommended_review_action"] == "review_aggregate_event_evidence"


def test_unverified_non_official_single_source_case_is_reviewable_not_final_case_dataset():
    record = _record(
        "rec_woodsidemd",
        source_id="src_woodsidemd",
        source_url="https://www.woodsidemd.com/post/hantavirus-what-you-need-to-know-about-the-2026-global-surge",
        source_title="Hantavirus: what you need to know about the 2026 global surge",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        credibility_level="high",
        credibility_score=0.95,
        cases_unspecified=7.0,
        deaths=3.0,
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote=(
            "A private medical site reported seven cases among passengers "
            "and three confirmed fatalities."
        ),
    )
    source = _source(
        "src_woodsidemd",
        canonical_url="https://www.woodsidemd.com/post/hantavirus-what-you-need-to-know-about-the-2026-global-surge",
        title="Hantavirus: what you need to know about the 2026 global surge",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        credibility_level="high",
        credibility_score=0.95,
        source_identity_unverified=True,
        metadata_only_identity=True,
        source_identity_warnings=[
            "official_source_type_claim_without_domain_support",
            "source_identity_unverified",
        ],
    )
    state = _state(
        [record],
        source_registry=[source],
        structured_task={
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["final_case_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_woodsidemd"}
    assert _record_ids(package["reviewable_case_dataset"]) == {"rec_woodsidemd"}
    assert package["case_candidate_dataset"][0]["source_id"] == "src_woodsidemd"
    assert package["case_candidate_dataset"][0]["case_candidate_status"] == "reviewable"
    decision = result["record_inclusion_decisions"][0]
    assert decision["record_final_inclusion_status"] == "pending_human_review"
    assert "source_identity_unverified" in decision["quality_gate_warnings"]


def test_private_single_source_case_with_review_warning_is_not_final_case_dataset():
    record = _record(
        "rec_private_review",
        source_id="src_private_review",
        source_url="https://example-clinic.invalid/hantavirus-outbreak-2026",
        source_title="Hantavirus outbreak 2026",
        publisher="Example Clinic",
        actual_publisher="Example Clinic",
        source_type="international_organization_report",
        credibility_level="high",
        credibility_score=0.86,
        cases_unspecified=3.0,
        deaths=1.0,
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        requires_human_review=True,
        evidence_quote=(
            "A private clinic website described three passenger cases and one death, "
            "but did not link an official case report."
        ),
    )
    source = _source(
        "src_private_review",
        canonical_url="https://example-clinic.invalid/hantavirus-outbreak-2026",
        title="Hantavirus outbreak 2026",
        publisher="Example Clinic",
        actual_publisher="Example Clinic",
        source_type="international_organization_report",
        credibility_level="high",
        credibility_score=0.86,
        credibility_flags=[
            "primary_or_authoritative_source",
            "missing_publisher",
        ],
        source_identity_warnings=[
            "search_provider_not_publisher",
            "publisher_from_search_metadata_unverified",
            "actual_publisher_unknown",
        ],
    )
    state = _state(
        [record],
        source_registry=[source],
        structured_task={
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
        human_review_enabled=False,
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["final_case_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_private_review"}
    assert _record_ids(package["reviewable_case_dataset"]) == {"rec_private_review"}
    decision = result["record_inclusion_decisions"][0]
    assert decision["record_final_inclusion_status"] == "pending_human_review"
    assert "source_trust_requires_human_review" in decision["quality_gate_warnings"]


def test_source_identity_warning_without_review_warning_is_not_final_case_dataset():
    record = _record(
        "rec_identity_warning_only",
        source_id="src_identity_warning_only",
        source_url="https://example-news.invalid/hantavirus-report",
        source_title="Hantavirus report",
        publisher="Example News",
        actual_publisher="Example News",
        source_type="official_public_health_agency",
        source_type_final="commercial_site",
        credibility_level="high",
        credibility_score=0.87,
        cases_unspecified=None,
        deaths=3.0,
        metric_name="deaths",
        metric_value=3.0,
        observation_type="death_record",
        observation_types=["death_record"],
        primary_case_dataset_eligible=True,
        requires_human_review=False,
        source_identity_warnings=[
            "official_source_type_claim_without_domain_support",
            "source_identity_unverified",
            "metadata_only_identity",
        ],
        source_identity_unverified=True,
        metadata_only_identity=True,
    )
    source = _source(
        "src_identity_warning_only",
        canonical_url="https://example-news.invalid/hantavirus-report",
        title="Hantavirus report",
        publisher="Example News",
        actual_publisher="Example News",
        source_type="official_public_health_agency",
        source_type_final="commercial_site",
        credibility_level="high",
        credibility_score=0.87,
        source_identity_warnings=[
            "official_source_type_claim_without_domain_support",
            "source_identity_unverified",
            "metadata_only_identity",
        ],
        source_identity_unverified=True,
        metadata_only_identity=True,
    )
    state = _state(
        [record],
        source_registry=[source],
        human_review_enabled=False,
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["final_case_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {
        "rec_identity_warning_only"
    }
    decision = result["record_inclusion_decisions"][0]
    assert decision["record_final_inclusion_status"] == "pending_human_review"
    assert "source_identity_requires_human_review" in decision[
        "quality_gate_warnings"
    ]


def test_high_trust_source_without_extracted_records_remains_in_source_inventory():
    source = _source(
        "src_sanidad",
        canonical_url="https://www.sanidad.gob.es/areas/alertasEmergenciasSanitarias/hantavirus-2026.pdf",
        title="Spain Ministry of Health hantavirus update",
        publisher="Ministry of Health of Spain",
        actual_publisher="Ministry of Health of Spain",
        source_type="national_public_health_agency",
        source_type_final="national_public_health_agency",
        authority_bucket="official_authority",
        credibility_level="high",
        credibility_score=0.98,
    )
    document = _document(
        "src_sanidad",
        canonical_url=source["canonical_url"],
        title=source["title"],
        publisher=source["publisher"],
        source_type="national_public_health_agency",
        fetch_status="fetched",
        parse_status="parsed_html",
        quality_status="usable",
        clean_text=(
            "Spanish Ministry of Health context on a hantavirus outbreak, "
            "but no extractable line-list metric in this fixture."
        ),
    )
    state = _state(
        records=[],
        source_registry=[source],
        documents=[document],
        evidence_chunks=[],
        structured_task={
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
    )

    package, _ = _finalize(state)

    inventory = package["source_inventory"]
    authority_inventory = package["authority_source_inventory"]
    assert len(inventory) == 1
    assert inventory[0]["source_id"] == "src_sanidad"
    assert inventory[0]["fetch_status"] == "fetched"
    assert inventory[0]["parse_status"] == "parsed_html"
    assert inventory[0]["extractable_status"] == "parsed_no_extracted_records"
    assert inventory[0]["evidence_role"] == "national_or_subnational_authority"
    assert [row["source_id"] for row in authority_inventory] == ["src_sanidad"]


def test_case_candidate_evidence_bundles_group_same_event_sources():
    official = _record(
        "rec_who",
        source_id="src_who",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/example",
        source_title="WHO MV Hondius hantavirus outbreak update",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        geographic_scope="Multi-locations",
        event_cluster_id="event_mv_hondius_hantavirus_2026",
        cases_unspecified=13.0,
        deaths=3.0,
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="WHO reported 13 cases and three deaths linked to the cruise ship outbreak.",
    )
    media = _record(
        "rec_media",
        source_id="src_media",
        source_url="https://www.example-news.org/mv-hondius-hantavirus",
        source_title="MV Hondius hantavirus cases by nationality",
        publisher="Example News",
        actual_publisher="Example News",
        source_type_final="news_media",
        geographic_scope="Multi-locations",
        event_cluster_id="event_mv_hondius_hantavirus_2026",
        cases_confirmed=1.0,
        cases_unspecified=None,
        deaths=1.0,
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        requires_human_review=True,
        evidence_quote="The outlet described one confirmed fatal case linked to the same event.",
    )
    state = _state(
        [official, media],
        source_registry=[
            _source(
                "src_who",
                canonical_url=official["source_url"],
                title=official["source_title"],
                publisher="World Health Organization",
                actual_publisher="World Health Organization",
                source_type_final="international_public_health_agency",
            ),
            _source(
                "src_media",
                canonical_url=media["source_url"],
                title=media["source_title"],
                publisher="Example News",
                actual_publisher="Example News",
                source_type_final="news_media",
            ),
        ],
    )

    package, _ = _finalize(state)

    bundles = package["case_evidence_bundles"]
    assert len(bundles) == 2
    assert {bundle["bundle_scope"] for bundle in bundles} == {
        "event_snapshot",
        "individual_case",
    }
    assert {
        record_id
        for bundle in bundles
        for record_id in bundle["member_record_ids"]
    } == {"rec_who", "rec_media"}
    assert all(bundle["supporting_source_count"] == 1 for bundle in bundles)


def test_core_metric_extraction_gap_creates_human_review_item_without_records():
    state = _state(
        records=[],
        structured_task={
            "disease": "Tuberculosis",
            "location": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_vdh_tb_annual",
                canonical_url="https://www.vdh.virginia.gov/tuberculosis/data-reports",
                title="Virginia Tuberculosis Annual Surveillance Report 2025",
                publisher="Virginia Department of Health",
                source_type_final="state_or_local_public_health_agency",
            )
        ],
        documents=[
            _document(
                "src_vdh_tb_annual",
                title="Virginia Tuberculosis Annual Surveillance Report 2025",
                canonical_url="https://www.vdh.virginia.gov/tuberculosis/data-reports",
                usable_for_task_collection=True,
            )
        ],
        structured_extraction_summary={
            "raw_record_count": 0,
            "core_metric_extraction_gap_count": 1,
            "core_metric_extraction_gaps": [
                {
                    "source_id": "src_vdh_tb_annual",
                    "reason": "core_metric_text_attempted_but_no_records_extracted",
                    "core_metric_chunk_count": 2,
                    "sample_chunks": [
                        {
                            "chunk_id": "chunk_tb_gap",
                            "text_preview": "Virginia TB cases and incidence were discussed.",
                        }
                    ],
                }
            ],
        },
        metric_extraction_plan={
            "core_metric_extraction_gap_count": 1,
        },
        human_review_queue=[],
    )

    package, result = _finalize(state)

    review_items = package["human_review_items"]
    assert len(review_items) == 1
    assert review_items[0]["item_type"] == "core_metric_extraction_gap"
    assert review_items[0]["source_ids"] == ["src_vdh_tb_annual"]
    assert review_items[0]["reason"] == "core_metric_text_attempted_but_no_records_extracted"
    assert result["direct_collection_summary"]["core_metric_extraction_gap_count"] == 1


def test_direct_collection_news_domain_claiming_official_source_routes_to_pending_review():
    record = _record(
        "rec_news_measles",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=259.0,
        source_id="src_abc_news",
        source_url="https://abcnews.go.com/Health/measles-cases-linked-texas-outbreak/story?id=119799576",
        source_title="Measles cases linked to Texas outbreak grows to 259 - ABC News",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        publisher="ABC News",
        actual_publisher="Centers for Disease Control and Prevention",
        credibility_score=0.91,
        credibility_level="high",
        evidence_quote="ABC News reported CDC measles counts for an outbreak.",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        metric_name="measles outbreak cases",
        metric_category="case_count",
        metric_value=259.0,
        metric_unit="count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual",
        cases_confirmed=None,
    )
    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_abc_news",
                    canonical_url="https://abcnews.go.com/Health/measles-cases-linked-texas-outbreak/story?id=119799576",
                    title="Measles cases linked to Texas outbreak grows to 259 - ABC News",
                    publisher="ABC News",
                    actual_publisher="Centers for Disease Control and Prevention",
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    credibility_score=0.91,
                    credibility_level="high",
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_news_measles"}
    pending = package["pending_review_records"][0]
    assert pending["record_final_inclusion_status"] == "pending_human_review"
    assert pending["requires_human_review"] is True
    assert pending["human_review_reason"]
    assert "source_trust_requires_human_review" in pending["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["pending_review_record_count"] == 1


def test_best_available_context_records_get_period_reason_for_near_miss():
    record = _record(
        "rec_tb_annual_best",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        subnational_location=None,
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=None,
        source_id="src_ntep",
        source_url="https://tbcindia.mohfw.gov.in/india-tb-report-2024",
        source_title="India TB Report 2024",
        source_type="official_public_health_agency",
        source_type_final="national_public_health_agency",
        publisher="National Tuberculosis Elimination Programme",
        actual_publisher="National Tuberculosis Elimination Programme",
        metric_name="TB patient notifications",
        metric_category="case_count",
        metric_value=2600000.0,
        metric_unit="count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual",
        evidence_quote="India notified 26 lakh TB patients in 2024.",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
    )
    package, _ = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_ntep",
                    canonical_url="https://tbcindia.mohfw.gov.in/india-tb-report-2024",
                    title="India TB Report 2024",
                    publisher="National Tuberculosis Elimination Programme",
                    actual_publisher="National Tuberculosis Elimination Programme",
                    source_type="official_public_health_agency",
                    source_type_final="national_public_health_agency",
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {"rec_tb_annual_best"}
    best = package["best_available_context_records"][0]
    assert best["best_available_reason"] == "period_mismatch_best_available_context"
    assert best["record_period_fit_status"] == "broader_than_task"


def test_final_package_uses_source_critic_fast_path_summary_fallback():
    fast_path = {
        "collection_mode": "direct_collection",
        "target_source_count": 2,
        "critic_attempted_source_count": 0,
        "critic_skipped_source_count": 4,
        "critic_skipped_reason_counts": {
            "direct_target_official_fast_path_skips_source_critic": 4
        },
    }

    package, result = _finalize(
        _state(
            source_critic_summary={
                "direct_fast_path_summary": fast_path,
            }
        )
    )

    assert package["direct_fast_path_summary"] == fast_path
    assert result["direct_fast_path_summary"] == fast_path


def test_coverage_refresh_does_not_let_one_accepted_record_complete_other_requirements_from_same_source():
    req_2024 = {
        "requirement_id": "india_tuberculosis_annual_2024",
        "disease": "tuberculosis",
        "location": "India",
        "geography": "India",
        "period_basis": "annual",
        "reporting_period_start": "2024-01-01",
        "reporting_period_end": "2024-12-31",
    }
    req_2025 = {
        "requirement_id": "india_tuberculosis_annual_2025",
        "disease": "tuberculosis",
        "location": "India",
        "geography": "India",
        "period_basis": "annual",
        "reporting_period_start": "2025-01-01",
        "reporting_period_end": "2025-12-31",
    }
    record = _record(
        "rec_india_tb_2024_cases",
        source_id="src_india_tb_report",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="Mycobacterium tuberculosis",
        country="India",
        subnational_location="",
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=1000.0,
        metric_name="notified TB cases",
        metric_value=1000.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        count_semantics="annual case aggregate",
        statistical_count_type="annual",
        coverage_requirement_ids=["india_tuberculosis_annual_2024"],
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="India reported 1,000 notified tuberculosis cases in 2024.",
        source_url="https://tbcindia.mohfw.gov.in/annual-report-2024",
        source_title="India TB Annual Report 2024",
        publisher="National Tuberculosis Elimination Programme",
        actual_publisher="National Tuberculosis Elimination Programme",
        source_type_final="national_public_health_agency",
    )

    package, _ = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-01-01",
                "end_date": "2025-12-31",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "Tuberculosis",
                "disease_standard_name": "Tuberculosis",
                "aliases": ["tuberculosis", "TB"],
                "pathogen_terms": ["Mycobacterium tuberculosis"],
                "syndrome_terms": ["tuberculosis"],
            },
            task_evidence_contract={
                "time_granularity": "annual",
                "requirements": [req_2024, req_2025],
            },
            source_coverage_requirements=[req_2024, req_2025],
            source_registry=[
                _source(
                    "src_india_tb_report",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type_final="national_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.98,
                    coverage_requirement_ids=[
                        "india_tuberculosis_annual_2024",
                        "india_tuberculosis_annual_2025",
                    ],
                )
            ],
            documents=[
                _document(
                    "src_india_tb_report",
                    url=record["source_url"],
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type_final="national_public_health_agency",
                    clean_text=record["evidence_quote"],
                    usable_for_task_collection=True,
                )
            ],
        )
    )

    audit = package["source_coverage_audit"]
    by_id = {row["requirement_id"]: row for row in audit["requirements"]}
    assert by_id["india_tuberculosis_annual_2024"]["accepted"] is True
    assert by_id["india_tuberculosis_annual_2024"]["accepted_record_ids"] == [
        "rec_india_tb_2024_cases"
    ]
    assert by_id["india_tuberculosis_annual_2025"]["accepted"] is False
    assert by_id["india_tuberculosis_annual_2025"]["accepted_record_ids"] == []
    assert audit["coverage_completeness_status"] == "partial_target_coverage"
    assert audit["missing_requirement_ids"] == ["india_tuberculosis_annual_2025"]


def test_dynamic_outbreak_snapshots_do_not_quarantine_as_validation_conflict():
    record = _record(
        "rec_cdc_may8_snapshot",
        source_id="src_cdc_han",
        source_url="https://www.cdc.gov/han/php/notices/han00528.html",
        source_title="2026 Multi-country Hantavirus Cluster Linked to Cruise Ship",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type_final="national_public_health_agency",
        country="",
        subnational_location="",
        geographic_scope="travel-associated outbreak",
        geographic_scope_type="multi_country",
        date_reported="2026-05-08",
        date_anchor="2026-05-08",
        reporting_period="as of May 8, 2026",
        cases_unspecified=8.0,
        deaths=3.0,
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "As of May 8, 2026, WHO has reported eight cases, including "
            "three deaths, in a travel-associated hantavirus outbreak."
        ),
    )
    validation = _validation_result(
        "rec_cdc_may8_snapshot",
        validation_type="held_out_source_comparison",
        compared_field="cases_unspecified",
        validation_status="conflict",
        match_status="conflict",
        left_value="8",
        right_value="13",
        date_or_period="as of May 8, 2026 compared with later update",
        reason=(
            "collection and held-out validation disagree for cases_unspecified "
            "because the held-out source reports a later outbreak snapshot"
        ),
    )

    _package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            source_registry=[
                _source(
                    "src_cdc_han",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type_final="national_public_health_agency",
                    data_product_type="event_outbreak_report",
                    source_role_final="collection",
                )
            ],
            validation_results=[validation],
        )
    )

    decision = result["record_inclusion_decisions"][0]
    assert "validation_conflict" not in decision["quality_gate_blocking_flags"]
    assert "compatible_dynamic_outbreak_snapshot" in decision["quality_gate_warnings"]


def test_direct_collection_event_snapshot_not_quarantined_by_period_semantics():
    record = _record(
        "rec_who_asof_snapshot",
        source_id="src_who_don600",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600",
        source_title="Hantavirus disease - multi-country outbreak update",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        country="",
        subnational_location="",
        geographic_scope="MV Hondius cruise ship outbreak",
        geographic_scope_type="vessel",
        date_reported="2026-05-08",
        date_anchor="2026-05-08",
        as_of_date="2026-05-08",
        reporting_period="as of May 8, 2026",
        cases_unspecified=8.0,
        deaths=3.0,
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        statistical_count_type="event_total",
        count_semantics="event_outbreak_snapshot",
        resolved_column_period_type="cumulative_period",
        primary_case_dataset_eligible=True,
        requires_human_review=True,
        evidence_quote=(
            "As of May 8, 2026, eight hantavirus cases, including three "
            "deaths, were linked to the MV Hondius cruise ship outbreak."
        ),
    )

    _package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            source_registry=[
                _source(
                    "src_who_don600",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type_final="international_public_health_agency",
                    data_product_type="event_outbreak_report",
                    source_role_final="collection",
                )
            ],
        )
    )

    decision = result["record_inclusion_decisions"][0]
    assert "record_period_semantics_not_exact_for_task_window" not in decision[
        "quality_gate_blocking_flags"
    ]
    assert decision["record_final_inclusion_status"] == "pending_human_review"


def test_case_evidence_bundles_group_compatible_outbreak_snapshots_across_sources():
    records = [
        _record(
            "rec_cdc_snapshot",
            source_id="src_cdc",
            source_url="https://www.cdc.gov/han/php/notices/han00528.html",
            source_title="CDC HAN hantavirus cruise ship outbreak",
            publisher="Centers for Disease Control and Prevention",
            source_type_final="national_public_health_agency",
            country="",
            subnational_location="",
            geographic_scope="travel-associated outbreak",
            geographic_scope_type="multi_country",
            date_reported="2026-05-08",
            reporting_period="as of May 8, 2026",
            cases_unspecified=8.0,
            deaths=3.0,
            observation_type="outbreak_summary",
            observation_types=["outbreak_summary"],
            evidence_quote="As of May 8, 2026, eight cases were linked to the MV Hondius cruise ship outbreak.",
        ),
        _record(
            "rec_nejm_snapshot",
            source_id="src_nejm",
            source_url="https://www.nejm.org/doi/full/10.1056/NEJMc2606496",
            source_title="Andes Hantavirus Outbreak on a Cruise Ship",
            publisher="New England Journal of Medicine",
            source_type_final="academic_or_peer_reviewed_source",
            country="",
            subnational_location="",
            geographic_scope="MV Hondius cruise ship",
            geographic_scope_type="vessel",
            date_reported="2026-05-13",
            reporting_period="as of May 13, 2026",
            cases_unspecified=10.0,
            deaths=3.0,
            observation_type="outbreak_summary",
            observation_types=["outbreak_summary"],
            evidence_quote="A total of 10 cases had been identified in the MV Hondius cruise ship outbreak.",
        ),
        _record(
            "rec_clinicallab_snapshot",
            source_id="src_clinicallab",
            source_url="https://www.clinicallab.com/hantavirus-diagnosis-surveillance-and-2026-outbreak-updates-28664",
            source_title="Hantavirus Diagnosis, Surveillance, and 2026 Outbreak Updates",
            publisher="Clinical Lab",
            source_type_final="news_media",
            country="",
            subnational_location="",
            geographic_scope="MV Hondius cruise ship",
            geographic_scope_type="vessel",
            date_reported="2026-05-27",
            reporting_period="as of May 27, 2026",
            cases_unspecified=13.0,
            deaths=3.0,
            observation_type="outbreak_summary",
            observation_types=["outbreak_summary"],
            evidence_quote="The MV Hondius cruise ship outbreak has been linked to 13 cases and three deaths.",
        ),
    ]
    package, _ = _finalize(
        _state(
            records,
            structured_task={
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            source_registry=[
                _source(
                    "src_cdc",
                    canonical_url=records[0]["source_url"],
                    title=records[0]["source_title"],
                    source_type_final="national_public_health_agency",
                    data_product_type="event_outbreak_report",
                    authority_bucket="official_authority",
                ),
                _source(
                    "src_nejm",
                    canonical_url=records[1]["source_url"],
                    title=records[1]["source_title"],
                    source_type_final="academic_or_peer_reviewed_source",
                    data_product_type="case_report_article",
                ),
                _source(
                    "src_clinicallab",
                    canonical_url=records[2]["source_url"],
                    title=records[2]["source_title"],
                    source_type_final="news_media",
                    data_product_type="news_case_report",
                ),
            ],
        )
    )

    outbreak_bundles = [
        row
        for row in package["case_evidence_bundles"]
        if row.get("bundle_type") == "aggregate_event"
    ]
    assert len(outbreak_bundles) == 3
    event_family_ids = {bundle["event_family_id"] for bundle in outbreak_bundles}
    assert len(event_family_ids) == 1
    assert next(iter(event_family_ids))
    assert all(bundle["supporting_source_count"] == 1 for bundle in outbreak_bundles)
    assert all(
        bundle["snapshot_compatibility_status"] == "single_snapshot_or_not_applicable"
        for bundle in outbreak_bundles
    )
    assert {
        source_id
        for bundle in outbreak_bundles
        for source_id in bundle["source_ids"]
    } == {"src_cdc", "src_nejm", "src_clinicallab"}


def test_review_warning_does_not_block_when_human_review_disabled():
    record = _record(
        "rec_who_review_warning",
        source_id="src_who_review_warning",
        cases_unspecified=None,
        cases_confirmed=13.0,
        deaths=3.0,
        country="",
        subnational_location="",
        geographic_scope="Multi-locations",
        geographic_scope_type="multi_country",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
        source_title="Hantavirus outbreak linked to cruise ship travel, Multi-locations",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        source_independence_group="WHO",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        requires_human_review=True,
        evidence_quote=(
            "As of 27 May, a total of 13 hantavirus cases, including three deaths, have "
            "been reported. International contact tracing and follow up of "
            "contacts is ongoing."
        ),
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "global",
                "start_date": "2024",
                "end_date": "2026",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "global",
                "start_date": "2024",
                "end_date": "2026",
                "time_window": "2024-2026",
            },
            human_review_enabled=False,
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_who_review_warning"}
    assert package["final_dataset"][0]["record_final_inclusion_status"] in {
        "accepted_with_review_warning",
        "accepted_with_warnings",
    }
    assert "accepted_with_review_warning" in package["final_dataset"][0][
        "quality_gate_warnings"
    ]
    assert result["run_quality_summary"]["accepted_record_count"] == 1


def test_global_task_multi_country_record_not_blocked_as_broader_geography():
    record = _record(
        "rec_who_multi_country",
        source_id="src_who_multi_country",
        cases_unspecified=None,
        cases_confirmed=1.0,
        deaths=1.0,
        country="",
        subnational_location="",
        geographic_scope="Multi-country",
        geographic_scope_type="multi_country",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        source_title="Hantavirus outbreak linked to cruise ship travel",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        date_reported="2026-05-04",
        date_anchor="2026-05-04",
        reporting_period="2026-04-01 to 2026-06-30",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote=(
            "Case 4: An adult female with fever and pneumonia died during a "
            "multi-country cruise ship outbreak."
        ),
    )

    _package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
                "time_window": "2026-04-01 to 2026-06-30",
            },
        ),
    )

    decision = result["record_inclusion_decisions"][0]
    assert "record_geography_broader_than_task" not in decision[
        "quality_gate_blocking_flags"
    ]


def test_global_task_vessel_record_not_blocked_as_broader_geography():
    record = _record(
        "rec_vessel_case",
        source_id="src_ecdc_vessel",
        cases_unspecified=1.0,
        country="",
        subnational_location="",
        geographic_scope="MV Hondius cruise ship",
        geographic_scope_type="vessel",
        source_url="https://www.ecdc.europa.eu/sites/default/files/documents/TAB-hantavirus-06052026.pdf",
        source_title="Threat Assessment Brief: Andes hantavirus outbreak",
        publisher="European Centre for Disease Prevention and Control",
        actual_publisher="European Centre for Disease Prevention and Control",
        source_type_final="international_public_health_agency",
        date_reported="2026-05-06",
        date_anchor="2026-05-06",
        reporting_period="2026-04-01 to 2026-06-30",
        observation_type="suspected_case_record",
        observation_types=["suspected_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote="The fourth individual on the cruise ship died on board.",
    )

    _package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
                "time_window": "2026-04-01 to 2026-06-30",
            },
        ),
    )

    decision = result["record_inclusion_decisions"][0]
    assert "record_geography_broader_than_task" not in decision[
        "quality_gate_blocking_flags"
    ]


def test_global_task_global_scope_record_not_blocked_as_broader_geography():
    from data_collection_workflow.run_quality_gates import _record_geography_scope_block

    record = _record(
        "rec_global_summary",
        source_id="src_who_global_summary",
        cases_unspecified=13.0,
        country="",
        subnational_location="",
        geographic_scope="Global",
        geographic_scope_type="global",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
        source_title="Hantavirus outbreak linked to cruise ship travel",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        date_reported="2026-05-27",
        date_anchor="2026-05-27",
        reporting_period="2026-04-01 to 2026-06-30",
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="WHO reported 13 cases in a global outbreak summary.",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "time_window": "2026-04-01 to 2026-06-30",
        },
    )

    assert _record_geography_scope_block(record, state) is None

    _package, result = _finalize(state)

    decision = result["record_inclusion_decisions"][0]
    assert "record_geography_broader_than_task" not in decision[
        "quality_gate_blocking_flags"
    ]


def test_verified_official_domain_search_metadata_warnings_do_not_block_final_inclusion():
    record = _record(
        "rec_who_verified_warning",
        source_id="src_who_verified_warning",
        cases_unspecified=None,
        cases_confirmed=13.0,
        deaths=3.0,
        country="",
        subnational_location="",
        geographic_scope="Multi-country",
        geographic_scope_type="multi_country",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
        source_title="Hantavirus outbreak linked to cruise ship travel",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        source_identity_warnings=[
            "publisher_from_search_metadata_unverified",
            "search_provider_not_publisher",
        ],
        date_reported="2026-05-27",
        date_anchor="2026-05-27",
        reporting_period="2026-04-01 to 2026-06-30",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote=(
            "As of 27 May 2026, a total of 13 hantavirus cases, including three deaths, "
            "have been reported."
        ),
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "hantavirus",
                "location": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
                "time_window": "2026-04-01 to 2026-06-30",
            },
            human_review_enabled=False,
        )
    )

    decision = result["record_inclusion_decisions"][0]
    assert "source_identity_requires_human_review" not in decision[
        "quality_gate_blocking_flags"
    ]
    assert _record_ids(package["final_dataset"]) == {"rec_who_verified_warning"}


def test_non_primary_claim_observation_is_quarantined_and_counted():
    record = _record(
        "rec_zero_case",
        cases_unspecified=0.0,
        observation_type="zero_case_statement",
        primary_case_dataset_eligible=False,
        claim_corroboration_warnings=["not_primary_case_record"],
        evidence_quote="No confirmed hantavirus cases were reported.",
    )

    package, result = _finalize(_state([record]))

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {"rec_zero_case"}
    assert _record_ids(package["quarantined_records"]) == {"rec_zero_case"}
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] in {
        "quarantined_zero_case_statement",
        "quarantined_non_primary_observation",
    }
    assert "primary_case_dataset_eligible_false" in quarantined["quality_gate_blocking_flags"]
    assert "not_primary_case_record" in quarantined["quality_gate_blocking_flags"]
    assert "claim_observation_type_zero_case_statement" in quarantined["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["primary_case_dataset_eligible_count"] == 0
    assert result["run_quality_summary"]["non_primary_observation_count"] == 1
    assert result["final_dataset_quality_summary"]["primary_case_dataset_eligible_count"] == 0
    assert result["final_dataset_quality_summary"]["non_primary_observation_count"] == 1


def test_explicit_primary_case_dataset_false_is_blocked_from_final_dataset():
    record = _record(
        "rec_ambiguous",
        cases_unspecified=None,
        deaths=None,
        observation_type="ambiguous_public_health_observation",
        primary_case_dataset_eligible=False,
        evidence_quote="Public health officials issued general prevention guidance.",
    )

    package, result = _finalize(_state([record]))

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {"rec_ambiguous"}
    assert _record_ids(package["quarantined_records"]) == {"rec_ambiguous"}
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] in {
        "quarantined_ambiguous_non_primary_observation",
        "quarantined_non_primary_observation",
    }
    assert "primary_case_dataset_eligible_false" in quarantined["quality_gate_blocking_flags"]
    assert "not_primary_case_record" in quarantined["quality_gate_blocking_flags"]
    assert (
        "claim_observation_type_ambiguous_public_health_observation"
        in quarantined["quality_gate_blocking_flags"]
    )
    assert result["run_quality_summary"]["run_quality_status"] in {
        "no_primary_case_dataset_records",
        "failed_quality_gate",
    }
    assert result["run_quality_summary"]["no_primary_case_dataset_records"] is True
    assert result["run_quality_summary"]["primary_case_dataset_status"] in {
        "no_primary_case_dataset_records",
        "non_primary_observations_only",
    }


def test_official_task_relevant_surveillance_summary_is_accepted_into_task_aware_final_dataset():
    record = _record(
        "rec_vdh_week_40_surveillance",
        source_id="src_vdh_week_40",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        cases_confirmed=None,
        tests_positive=42.0,
        hospitalizations=7.0,
        deaths=0.0,
        count_semantics="weekly aggregate surveillance count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "VDH reported 42 positive influenza laboratory tests and 7 "
            "influenza-associated hospitalizations during week 40."
        ),
        source_url="https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-40.pdf",
        source_title="Weekly-RDS-Report_Week-40.pdf",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type_final="state_or_local_public_health_agency",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "time_window": "2024-10-01 to 2024-10-10",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_vdh_week_40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                actual_publisher="Virginia Department of Health",
                source_type_final="state_or_local_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        documents=[
            _document(
                "src_vdh_week_40",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                actual_publisher="Virginia Department of Health",
                source_type_final="state_or_local_public_health_agency",
                clean_text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_vdh_week_40_surveillance"}
    assert package["final_case_dataset"] == []
    accepted = package["final_dataset"][0]
    assert accepted["dataset_view"] == "task_aware_surveillance_summary"
    assert accepted["record_final_inclusion_status"] in {
        "accepted",
        "accepted_with_warnings",
    }
    assert "primary_case_dataset_eligible_false" not in accepted["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["final_dataset_mode"] == "task_aware_quality_gated_records"
    assert result["run_quality_summary"]["surveillance_summary_record_count"] == 1


def test_direct_collection_accepts_official_aggregate_record_without_observation_type():
    record = _record(
        "rec_vdh_lab_positive_without_type",
        source_id="src_vdh_week_40",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza A/H3N2",
        pathogen_or_syndrome="influenza A/H3N2",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        cases_confirmed=None,
        tests_positive=42.0,
        hospitalizations=7.0,
        deaths=0.0,
        count_semantics="weekly aggregate surveillance count",
        observation_type=None,
        observation_types=[],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "VDH reported 42 positive influenza laboratory tests and 7 "
            "influenza-associated hospitalizations during week 40."
        ),
        source_url="https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-40.pdf",
        source_title="Weekly-RDS-Report_Week-40.pdf",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type_final="state_or_local_public_health_agency",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "time_window": "2024-10-01 to 2024-10-10",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza", "H1N1", "H3N2", "influenza A", "influenza B"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_vdh_week_40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                actual_publisher="Virginia Department of Health",
                source_type_final="state_or_local_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        documents=[
            _document(
                "src_vdh_week_40",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                actual_publisher="Virginia Department of Health",
                source_type_final="state_or_local_public_health_agency",
                clean_text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_vdh_lab_positive_without_type"}
    accepted = package["final_dataset"][0]
    assert accepted["dataset_view"] == "task_aware_surveillance_summary"
    assert accepted["observation_types"] == ["surveillance_summary"]
    assert "accepted_direct_collection_official_aggregate" in accepted["quality_gate_warnings"]
    assert result["run_quality_summary"]["collection_mode"] == "direct_collection"
    assert result["run_quality_summary"]["official_source_record_count"] == 1


def test_direct_collection_accepts_supported_official_gov_source_despite_low_machine_score():
    record = _record(
        "rec_fresno_wnv_official_pdf",
        source_id="src_fresno_wnv_pdf",
        disease="West Nile virus disease",
        disease_standard_name="West Nile virus disease",
        virus_or_syndrome="West Nile virus",
        pathogen_or_syndrome="West Nile virus",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        geographic_scope_type="state",
        date_reported="2024-08-31",
        date_anchor="2024-08-31",
        reporting_period="2024-08-01 to 2024-08-31",
        cases_unspecified=38.0,
        deaths=4.0,
        metric_name="Human West Nile virus cases",
        metric_value=38.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-08-01",
        metric_period_end="2024-08-31",
        count_semantics="monthly human case surveillance count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        source_url=(
            "https://www.fresnocountyca.gov/files/sharedassets/county/v/1/"
            "public-health/dph-news-releases/2024/09-13-2024-news-release-"
            "first-human-death-caused-by-west-nile-virus-in-fresno-county.pdf"
        ),
        source_title="First Human Death Caused by West Nile Virus in Fresno County",
        publisher=None,
        actual_publisher=None,
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.5352,
        credibility_flags=[
            "official_public_health_authority",
            "missing_publisher",
            "low_machine_readability",
        ],
        evidence_quote=(
            "California reported 38 human West Nile virus cases and 4 deaths "
            "during August 2024."
        ),
    )
    state = _state(
        [record],
        structured_task={
            "disease": "West Nile virus",
            "location": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "West Nile virus",
            "geography": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "time_window": "2024-08-01 to 2024-08-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "West Nile virus",
            "disease_standard_name": "West Nile virus disease",
            "aliases": ["West Nile virus", "WNV", "West Nile virus disease"],
            "pathogen_terms": ["West Nile virus", "WNV"],
            "syndrome_terms": ["West Nile fever", "West Nile neuroinvasive disease"],
        },
        source_registry=[
            _source(
                "src_fresno_wnv_pdf",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher=None,
                actual_publisher=None,
                source_type="official_public_health_agency",
                source_type_final="official_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.5352,
                credibility_flags=[
                    "official_public_health_authority",
                    "missing_publisher",
                    "low_machine_readability",
                ],
            )
        ],
        documents=[
            _document(
                "src_fresno_wnv_pdf",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher=None,
                actual_publisher=None,
                source_type="official_public_health_agency",
                source_type_final="official_public_health_agency",
                clean_text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_fresno_wnv_official_pdf"}
    accepted = package["final_dataset"][0]
    assert "source_trust_requires_human_review" not in accepted["quality_gate_blocking_flags"]
    assert package["pending_review_records"] == []
    assert result["run_quality_summary"]["pending_review_record_count"] == 0


def test_direct_collection_accepts_official_vector_control_district_org_source():
    record = _record(
        "rec_placer_vector_control_wnv",
        source_id="src_placer_vector_control",
        disease="West Nile virus disease",
        disease_standard_name="West Nile virus disease",
        virus_or_syndrome="West Nile virus",
        pathogen_or_syndrome="West Nile virus",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        geographic_scope_type="state",
        date_reported="2024-08-31",
        date_anchor="2024-08-31",
        reporting_period="2024-08-01 to 2024-08-31",
        cases_unspecified=2.0,
        metric_name="Human West Nile virus cases",
        metric_value=2.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-08-01",
        metric_period_end="2024-08-31",
        count_semantics="monthly human case surveillance count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        source_url="https://placermosquito.org/west-nile-information",
        source_title="West Nile Virus Information - Placer Mosquito Vector Control District",
        publisher=None,
        actual_publisher=None,
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="collection",
        credibility_level="medium",
        credibility_score=0.5768,
        credibility_flags=[
            "official_public_health_authority",
            "missing_publisher",
            "ambiguous_disease",
        ],
        evidence_quote=(
            "The Placer Mosquito and Vector Control District reported "
            "2 human West Nile virus cases for California in August 2024."
        ),
    )
    state = _state(
        [record],
        structured_task={
            "disease": "West Nile virus",
            "location": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "West Nile virus",
            "geography": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "time_window": "2024-08-01 to 2024-08-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "West Nile virus",
            "disease_standard_name": "West Nile virus disease",
            "aliases": ["West Nile virus", "WNV", "West Nile virus disease"],
            "pathogen_terms": ["West Nile virus", "WNV"],
            "syndrome_terms": ["West Nile fever", "West Nile neuroinvasive disease"],
        },
        source_registry=[
            _source(
                "src_placer_vector_control",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher=None,
                actual_publisher=None,
                source_type="official_public_health_agency",
                source_type_final="official_public_health_agency",
                source_role_final="collection",
                credibility_level="medium",
                credibility_score=0.5768,
                credibility_flags=[
                    "official_public_health_authority",
                    "missing_publisher",
                    "ambiguous_disease",
                ],
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_placer_vector_control_wnv"}
    accepted = package["final_dataset"][0]
    assert "source_trust_requires_human_review" not in accepted["quality_gate_blocking_flags"]
    assert package["pending_review_records"] == []
    assert result["run_quality_summary"]["pending_review_record_count"] == 0


def test_direct_collection_rescues_official_metric_from_machine_excluded_source_role():
    record = _record(
        "rec_westnile_ca_weekly_metric",
        source_id="src_westnile_ca_week32_pdf",
        disease="West Nile virus disease",
        disease_standard_name="West Nile virus disease",
        virus_or_syndrome="West Nile virus",
        pathogen_or_syndrome="West Nile virus",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        geographic_scope_type="state",
        date_reported="2024-08-31",
        date_anchor="2024-08-31",
        reporting_period="2024-08-01 to 2024-08-31",
        cases_confirmed=2.0,
        metric_name="Newly reported human WNV disease cases",
        metric_value=2.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-08-01",
        metric_period_end="2024-08-31",
        count_semantics="monthly human case surveillance count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=True,
        source_url="https://westnile.ca.gov/download?download_id=5072",
        source_title="Arbobulletin_2024_32.pdf",
        publisher=None,
        actual_publisher=None,
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="excluded",
        credibility_level="excluded",
        credibility_score=0.5192,
        credibility_flags=[
            "official_public_health_authority",
            "missing_publisher",
            "low_machine_readability",
            "ambiguous_disease",
        ],
        evidence_quote="California reported 2 newly reported human WNV disease cases.",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "West Nile virus",
            "location": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "West Nile virus",
            "geography": "California",
            "start_date": "2024-08-01",
            "end_date": "2024-08-31",
            "time_window": "2024-08-01 to 2024-08-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "West Nile virus",
            "disease_standard_name": "West Nile virus disease",
            "aliases": ["West Nile virus", "WNV", "West Nile virus disease"],
            "pathogen_terms": ["West Nile virus", "WNV"],
            "syndrome_terms": ["West Nile fever", "West Nile neuroinvasive disease"],
        },
        source_registry=[
            _source(
                "src_westnile_ca_week32_pdf",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher=None,
                actual_publisher=None,
                source_type="official_public_health_agency",
                source_type_final="official_public_health_agency",
                source_role_final="excluded",
                credibility_level="excluded",
                credibility_score=0.5192,
                credibility_flags=[
                    "official_public_health_authority",
                    "missing_publisher",
                    "low_machine_readability",
                    "ambiguous_disease",
                ],
                status="ready_for_content_fetch",
                final_screening_decision="include_for_content_fetch",
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_westnile_ca_weekly_metric"}
    accepted = package["final_dataset"][0]
    assert "source_role_final_excluded" not in accepted["quality_gate_blocking_flags"]
    assert "source_trust_requires_human_review" not in accepted["quality_gate_blocking_flags"]
    assert package["pending_review_records"] == []
    assert result["run_quality_summary"]["final_dataset_count"] == 1


def test_direct_collection_quarantines_prior_year_comparator_metric_row():
    record = _record(
        "rec_westnile_prior_year_comparator",
        source_id="src_westnile_ca_bulletin",
        disease="West Nile virus disease",
        disease_standard_name="West Nile virus disease",
        virus_or_syndrome="West Nile virus",
        pathogen_or_syndrome="West Nile virus",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        geographic_scope_type="state",
        date_reported="2024-08-31",
        date_anchor="2024-08-31",
        reporting_period="2023 to same point as Bulletin #29",
        cases_unspecified=98.0,
        metric_name="No. Human Cases (prior year comparator)",
        metric_value=98.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-08-01",
        metric_period_end="2024-08-31",
        count_semantics="prior year comparator case count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        source_url="https://westnile.ca.gov/download?download_id=5061",
        source_title="Arbobulletin_2024_29.pdf",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="excluded",
        credibility_level="excluded",
        credibility_score=0.5192,
        evidence_quote=(
            "California No. Human Cases 98 for 2023 to same point as Bulletin #29."
        ),
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "West Nile virus",
                "location": "California",
                "start_date": "2024-08-01",
                "end_date": "2024-08-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "West Nile virus",
                "geography": "California",
                "start_date": "2024-08-01",
                "end_date": "2024-08-31",
                "time_window": "2024-08-01 to 2024-08-31",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "West Nile virus",
                "disease_standard_name": "West Nile virus disease",
                "aliases": ["West Nile virus", "WNV", "West Nile virus disease"],
                "pathogen_terms": ["West Nile virus", "WNV"],
                "syndrome_terms": ["West Nile fever", "West Nile neuroinvasive disease"],
            },
            source_registry=[
                _source(
                    "src_westnile_ca_bulletin",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    source_role_final="excluded",
                    credibility_level="excluded",
                    credibility_score=0.5192,
                    status="ready_for_content_fetch",
                    final_screening_decision="include_for_content_fetch",
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_westnile_prior_year_comparator"
    }
    quarantined = package["quarantined_records"][0]
    assert "record_period_semantics_not_exact_for_task_window" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_direct_collection_quarantines_cumulative_metric_name_for_short_task_window():
    record = _record(
        "rec_westnile_cumulative_annual_metric",
        source_id="src_westnile_ca_bulletin",
        disease="West Nile virus disease",
        disease_standard_name="West Nile virus disease",
        virus_or_syndrome="West Nile virus",
        pathogen_or_syndrome="West Nile virus",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        geographic_scope_type="state",
        date_reported="2024-08-31",
        date_anchor="2024-08-31",
        reporting_period="2024",
        cases_unspecified=112.0,
        metric_name="cumulative 2024 human WNV disease cases",
        metric_value=112.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-08-01",
        metric_period_end="2024-08-31",
        count_semantics="case_count",
        statistical_count_type="current_period",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        source_url="https://westnile.ca.gov/download?download_id=5072",
        source_title="Arbobulletin_2024_32.pdf",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="excluded",
        credibility_level="excluded",
        credibility_score=0.5192,
        evidence_quote="California cumulative 2024 human WNV disease cases: 112.",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "West Nile virus",
                "location": "California",
                "start_date": "2024-08-01",
                "end_date": "2024-08-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "West Nile virus",
                "geography": "California",
                "start_date": "2024-08-01",
                "end_date": "2024-08-31",
                "time_window": "2024-08-01 to 2024-08-31",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "West Nile virus",
                "disease_standard_name": "West Nile virus disease",
                "aliases": ["West Nile virus", "WNV", "West Nile virus disease"],
                "pathogen_terms": ["West Nile virus", "WNV"],
                "syndrome_terms": ["West Nile fever", "West Nile neuroinvasive disease"],
            },
            source_registry=[
                _source(
                    "src_westnile_ca_bulletin",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    source_role_final="excluded",
                    credibility_level="excluded",
                    credibility_score=0.5192,
                    status="ready_for_content_fetch",
                    final_screening_decision="include_for_content_fetch",
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_westnile_cumulative_annual_metric"
    }
    quarantined = package["quarantined_records"][0]
    assert "record_period_semantics_not_exact_for_task_window" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_best_available_records_receive_reason_and_period_fit_when_missing():
    from data_collection_workflow.nodes.finalization import _build_best_available_context_records

    record = _record(
        "rec_govuk_2024_tb_context",
        source_id="src_govuk_tb_2024",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        country="United Kingdom",
        geographic_scope="England",
        geographic_scope_type="national_or_subnational",
        metric_name="TB incidence rate",
        metric_category="incidence_rate",
        metric_value=8.5,
        metric_unit="per 100,000 population",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        date_reported="2024-12-31",
        evidence_quote="England tuberculosis incidence was 8.5 per 100,000 in 2024.",
        source_url="https://www.gov.uk/government/statistics/tuberculosis-in-england-2024",
        source_title="Tuberculosis in England, 2024",
        publisher="UK Health Security Agency",
        actual_publisher="UK Health Security Agency",
        source_type_final="official_public_health_agency",
        credibility_level="high",
        quality_gate_blocking_flags=["record_period_outside_task_window"],
        best_available_reason=None,
        record_period_fit_status=None,
    )
    state = _state(
        [record],
        structured_task={
            "disease": "Tuberculosis",
            "location": "United Kingdom",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "United Kingdom",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_govuk_tb_2024",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="UK Health Security Agency",
                actual_publisher="UK Health Security Agency",
                source_type_final="official_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
            )
        ],
    )

    best = _build_best_available_context_records([record], state)

    assert len(best) == 1
    assert best[0]["record_period_fit_status"] == "outside_task_window"
    assert best[0]["best_available_reason"] == "period_mismatch_best_available_context"
    assert best[0]["source_url"] == record["source_url"]


def test_direct_collection_treats_validation_outside_scope_as_audit_for_official_aggregate():
    record = _record(
        "rec_ny_week_40_surveillance",
        source_id="src_ny_week_40",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza A",
        pathogen_or_syndrome="influenza A",
        country="United States of America",
        subnational_location="New York",
        geographic_scope="New York",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="week ending October 5, 2024",
        cases_unspecified=None,
        cases_confirmed=None,
        tests_positive=42.0,
        hospitalizations=7.0,
        count_semantics="weekly aggregate surveillance count",
        observation_type=None,
        observation_types=[],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "New York State reported 42 positive influenza laboratory tests "
            "and 7 influenza-associated hospitalizations for the week ending "
            "October 5, 2024."
        ),
        source_url=(
            "https://www.health.ny.gov/diseases/communicable/influenza/"
            "surveillance/2024-2025/archive/2024-10-05_flu_report.pdf"
        ),
        source_title="New York State Influenza Surveillance Report",
        publisher="New York State Department of Health",
        actual_publisher="New York State Department of Health",
        source_type_final="state_or_local_public_health_agency",
    )
    validation = _validation_result(
        "rec_ny_week_40_surveillance",
        validation_status="outside_scope",
        match_status="outside_requested_scope",
        reason="held-out audit source mistakenly marked outside_time_window",
        warnings=["outside_time_window"],
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "New York",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "time_window": "2024-10-01 to 2024-10-10",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_ny_week_40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="New York State Department of Health",
                actual_publisher="New York State Department of Health",
                source_type_final="state_or_local_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        documents=[
            _document(
                "src_ny_week_40",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="New York State Department of Health",
                actual_publisher="New York State Department of Health",
                source_type_final="state_or_local_public_health_agency",
                clean_text=record["evidence_quote"],
            )
        ],
        validation_results=[validation],
        source_coverage_audit={"coverage_status": "target_official_source_parsed"},
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_ny_week_40_surveillance"}
    accepted = package["final_dataset"][0]
    assert "validation_outside_scope" not in accepted["quality_gate_blocking_flags"]
    assert "validation_outside_scope_audit_only" in accepted["quality_gate_warnings"]
    assert result["direct_collection_summary"]["coverage_status"] == "accepted"


def test_direct_collection_quarantines_current_week_record_without_date_or_metric():
    record = _record(
        "rec_current_week_empty",
        source_id="src_ny_current_week",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="New York",
        geographic_scope="New York",
        date_reported="",
        date_anchor="",
        reporting_period="",
        cases_unspecified=None,
        cases_confirmed=None,
        cases_probable=None,
        cases_suspected=None,
        tests_positive=None,
        tests_total=None,
        positivity_rate=None,
        hospitalizations=None,
        deaths=None,
        count_semantics="newly_reported",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="The current week influenza report was updated.",
        source_url=(
            "https://www.health.ny.gov/diseases/communicable/influenza/"
            "surveillance/flu_report_current_week.pdf"
        ),
        source_title="New York State Influenza Surveillance Report - Current Week",
        publisher="New York State Department of Health",
        actual_publisher="New York State Department of Health",
        source_type_final="state_or_local_public_health_agency",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_ny_current_week",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="New York State Department of Health",
                actual_publisher="New York State Department of Health",
                source_type_final="state_or_local_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_id"] == "rec_current_week_empty"
    assert "missing_direct_collection_metric" in quarantined["quality_gate_blocking_flags"]
    assert "missing_direct_collection_date_anchor" in quarantined["quality_gate_blocking_flags"]
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_quarantines_metric_record_without_date_anchor():
    record = _record(
        "rec_ed_visit_no_date",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="",
        cases_unspecified=None,
        tests_positive=None,
        tests_total=None,
        positivity_rate=None,
        metric_name="nssp_ed_visit_percent",
        metric_value=0.2,
        metric_unit="percent",
        metric_category="ed_visit_percent",
        metric_denominator="emergency_department_visits",
        count_semantics="weekly ED visit percent",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="NSSP emergency department visits for influenza were 0.2%.",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert "missing_direct_collection_metric" not in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert "missing_direct_collection_date_anchor" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_accepts_metric_record_with_metric_period_anchor():
    record = _record(
        "rec_ed_visit_with_period",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="",
        metric_name="nssp_ed_visit_percent",
        metric_value=0.2,
        metric_unit="percent",
        metric_category="ed_visit_percent",
        metric_denominator="emergency_department_visits",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        count_semantics="weekly ED visit percent",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="NSSP emergency department visits for influenza were 0.2%.",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_ed_visit_with_period"}
    accepted = package["final_dataset"][0]
    assert accepted["metric_value"] == 0.2
    assert accepted["metric_period_end"] == "2024-10-05"
    assert result["direct_collection_summary"]["final_dataset_count"] == 1


def test_direct_collection_quarantines_unresolved_metric_row_binding():
    record = _record(
        "rec_positive_wrong_quote",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="",
        metric_name="Number of positive specimens",
        metric_value=359,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory positive count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_tested",
        metric_row_binding_status="unresolved",
        evidence_quote="| No. of specimens tested | 46,025 | 107,292 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert "metric_row_binding_unresolved" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_routes_ambiguous_metric_column_semantics_to_review():
    record = _record(
        "rec_ambiguous_column",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        deaths=None,
        metric_name="Number of specimens tested",
        metric_value=1113,
        metric_unit="count",
        metric_category="lab_test_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory test count",
        statistical_count_type="newly_reported",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_specimens_tested",
        source_column_label="column_2",
        metric_row_binding_status="resolved",
        metric_column_semantics_status="ambiguous",
        resolved_column_period_type="ambiguous_column",
        evidence_quote="| No. of specimens tested | 638 | 1,113 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["quarantined_records"] == []
    pending = package["pending_review_records"][0]
    assert "ambiguous_metric_column_semantics_requires_human_review" in pending[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_quarantines_previous_week_column_with_current_period_source():
    record = _record(
        "rec_previous_week_current_period",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        deaths=None,
        metric_name="Number of specimens tested",
        metric_value=53424,
        metric_unit="count",
        metric_category="lab_test_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory test count",
        statistical_count_type="newly_reported",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_specimens_tested",
        source_column_label="column 2 (previous week)",
        metric_row_binding_status="resolved",
        metric_column_semantics_status="resolved",
        resolved_column_period_type="previous_period",
        evidence_quote="| No. of specimens tested | 53,699 | 53,424 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert "previous_week_metric_uses_current_source_period" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_accepts_resolved_positivity_metric_despite_case_only_flags():
    record = _record(
        "rec_lab_positivity_percent",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        deaths=None,
        metric_name="Percent positive specimens",
        metric_value=0.8,
        metric_unit="percent",
        metric_category="lab_positivity_percent",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory positivity percent",
        statistical_count_type="weekly",
        observation_type="background_context",
        observation_types=["background_context"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_percent_positive",
        metric_row_binding_status="resolved",
        evidence_quote="| Percent positive specimens | 0.8% | Week 40 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        corroboration_summary={
            "claim_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_lab_positivity_percent"}
    accepted = package["final_dataset"][0]
    assert accepted["metric_category"] == "lab_positivity_percent"
    assert accepted["record_final_inclusion_status"] in {
        "accepted",
        "accepted_with_warnings",
    }
    assert "primary_case_dataset_eligible_false" not in accepted[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["accepted_positivity_rate_count"] == 1


def test_direct_collection_accepts_metric_even_when_observation_has_ambiguous_tag():
    record = _record(
        "rec_ed_percent",
        source_id="src_cdc_weekly",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="MMWR week 40, 2024",
        metric_name="ED visits with discharge diagnosis of influenza",
        metric_value=0.2,
        metric_unit="percent",
        metric_category="ed_visit_percent",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly emergency department visit percent",
        statistical_count_type="weekly",
        observation_type="surveillance_summary",
        observation_types=[
            "surveillance_summary",
            "ambiguous_public_health_observation",
        ],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_ed_visits",
        metric_row_binding_status="resolved",
        evidence_quote=(
            "The percentage of emergency department visits with a discharge "
            "diagnosis of influenza was 0.2% overall during Week 40."
        ),
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_weekly",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        corroboration_summary={
            "claim_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_ed_percent"}
    accepted = package["final_dataset"][0]
    assert accepted["metric_category"] == "ed_visit_percent"
    assert "primary_case_dataset_eligible_false" not in accepted[
        "quality_gate_blocking_flags"
    ]
    assert "claim_observation_type_ambiguous_public_health_observation" not in accepted[
        "quality_gate_blocking_flags"
    ]
    assert result["direct_collection_summary"]["final_dataset_count"] == 1


def test_direct_collection_quarantines_descriptive_hospital_delay_metric():
    record = _record(
        "rec_hospital_delay_days",
        source_id="src_cdc_mmwr",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles",
        country="United States of America",
        subnational_location="West Texas",
        geographic_scope="West Texas",
        geographic_scope_type="subnational",
        date_reported="2025-03-31",
        date_anchor="2025-01-01",
        reporting_period="January-March 2025",
        cases_unspecified=None,
        deaths=None,
        hospitalizations=None,
        metric_name="Days from rash onset to hospital admission (median)",
        metric_value=2.0,
        metric_unit="days",
        metric_category="other",
        metric_period_start="2025-01-01",
        metric_period_end="2025-03-31",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="other",
        statistical_count_type="subset",
        observation_type="ambiguous_public_health_observation",
        observation_types=["ambiguous_public_health_observation"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_hospital_delay",
        metric_row_binding_status="resolved",
        source_column_labels=["No. (%)"],
        metric_column_label="median (range)",
        evidence_quote=(
            "| Days from rash onset to hospital admission, median (range) | "
            "2 (-2 to 10) |"
        ),
        source_url="https://www.cdc.gov/mmwr/volumes/75/wr/mm7520a1.htm",
        source_title=(
            "Characteristics of Patients Hospitalized with Measles During an "
            "Outbreak - West Texas, January-March 2025"
        ),
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "Measles",
            "location": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Measles",
            "geography": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_mmwr",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Centers for Disease Control and Prevention",
                actual_publisher="Centers for Disease Control and Prevention",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        corroboration_summary={
            "claim_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {
        "rec_hospital_delay_days"
    }
    non_primary = package["non_primary_observations"][0]
    assert non_primary["record_final_inclusion_status"] in {
        "quarantined_ambiguous_non_primary_observation",
        "quarantined_non_primary_observation",
    }
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_quarantines_count_metric_with_percent_unit():
    record = _record(
        "rec_hospitalized_age_percent",
        source_id="src_cdc_measles_cases",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2025-03-31",
        date_anchor="2025-01-01",
        reporting_period="January-March 2025",
        cases_unspecified=None,
        deaths=None,
        hospitalizations=None,
        metric_name="Percent of age group hospitalized - under 5 years",
        metric_value=52.0,
        metric_unit="percent",
        metric_category="hospitalization_count",
        metric_period_start="2025-01-01",
        metric_period_end="2025-03-31",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="hospitalization count",
        statistical_count_type="subset",
        observation_type="hospitalization_record",
        observation_types=["hospitalization_record"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_hospitalized_age_percent",
        metric_row_binding_status="resolved",
        source_column_labels=["% hospitalized"],
        metric_column_label="percent hospitalized",
        evidence_quote="| Under 5 years | 52% hospitalized |",
        source_url="https://www.cdc.gov/measles/data-research/index.html",
        source_title="Measles Cases and Outbreaks",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "Measles",
            "location": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Measles",
            "geography": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-03-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_measles_cases",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Centers for Disease Control and Prevention",
                actual_publisher="Centers for Disease Control and Prevention",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        corroboration_summary={
            "claim_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {
        "rec_hospitalized_age_percent"
    }
    non_primary = package["non_primary_observations"][0]
    assert non_primary["record_final_inclusion_status"] == (
        "quarantined_non_primary_observation"
    )
    assert result["direct_collection_summary"]["final_dataset_count"] == 0


def test_direct_collection_accepts_annual_aggregate_metric_without_weekly_column_label():
    record = _record(
        "rec_india_tb_incidence_2024",
        source_id="src_india_tb_report",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        metric_name="TB incidence rate",
        metric_value=187,
        metric_unit="per 100,000 population",
        metric_category="incidence_rate",
        incidence_rate=187,
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual national incidence rate",
        statistical_count_type="annual",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        row_context_type="markdown_metric_line",
        source_row_id="row_tb_incidence_2024",
        metric_row_binding_status="resolved",
        metric_column_semantics_status="ambiguous",
        resolved_column_period_type="ambiguous_column",
        evidence_quote=(
            "TB incidence in India dropped to 187 per lakh population in 2024."
        ),
        source_url="https://www.pib.gov.in/PressReleasePage.aspx?PRID=2189415",
        source_title="India TB incidence in 2024",
        publisher="Press Information Bureau, Government of India",
        actual_publisher="Press Information Bureau, Government of India",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "Tuberculosis",
            "location": "India",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "India",
            "start_date": "2023-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_india_tb_report",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Press Information Bureau, Government of India",
                actual_publisher="Press Information Bureau, Government of India",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
        corroboration_summary={
            "claim_count": 1,
            "corroborated_primary_case_event_count": 0,
        },
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_india_tb_incidence_2024"}
    accepted = package["final_dataset"][0]
    assert accepted["metric_category"] == "incidence_rate"
    assert accepted["resolved_column_period_type"] == "annual_period"
    assert accepted["metric_column_semantics_status"] == "resolved"
    assert result["direct_collection_summary"]["final_dataset_count"] == 1


def test_finalization_refreshes_source_coverage_audit_after_records_are_accepted():
    record = _record(
        "rec_lab_positive",
        source_id="src_cdc_week_40",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="",
        date_anchor="",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        deaths=None,
        metric_name="Clinical lab positive specimens",
        metric_value=359,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory positive count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_positive",
        metric_row_binding_status="resolved",
        evidence_quote="| No. of positive specimens | 359 | Week 40 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    requirement = {
        "requirement_id": "req_week_40",
        "disease": "FLU",
        "location": "United States",
        "start_date": "2024-09-29",
        "end_date": "2024-10-05",
        "reporting_period_start": "2024-09-29",
        "reporting_period_end": "2024-10-05",
        "official_candidate_urls": [record["source_url"]],
        "reason": "target weekly surveillance report",
    }
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_cdc_week_40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                must_fetch=True,
                coverage_requirement_ids=["req_week_40"],
            )
        ],
        documents=[
            _document(
                "src_cdc_week_40",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_html",
            )
        ],
        source_coverage_requirements=[requirement],
        source_coverage_audit={
            "coverage_status": "target_official_source_discovered_not_fetched",
            "requirements": [
                {
                    **requirement,
                    "discovered": True,
                    "fetched": False,
                    "parsed": False,
                }
            ],
        },
    )

    package, result = _finalize(state)

    audit = package["source_coverage_audit"]
    assert audit["coverage_status"] == "target_official_source_accepted"
    assert audit["accepted_record_count"] == 1
    row = audit["requirements"][0]
    assert row["fetched"] is True
    assert row["parsed"] is True
    assert row["extracted"] is True
    assert row["accepted_record_count"] == 1
    assert row["accepted_record_ids"] == ["rec_lab_positive"]
    assert row["accepted_source_ids"] == ["src_cdc_week_40"]
    assert result["source_coverage_audit"]["coverage_status"] == (
        "target_official_source_accepted"
    )


def test_direct_collection_rejects_task_geography_and_period_inherited_without_source_evidence():
    record = _record(
        "rec_cdc_weekly_boston_inherited",
        source_id="src_cdc_weekly_national",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="",
        geographic_scope="Boston",
        geographic_scope_type="city",
        reporting_period="CDC FluView Week 47, 2025",
        metric_name="Clinical laboratory specimens tested",
        metric_value=52021,
        metric_unit="count",
        metric_category="lab_test_count",
        metric_period_start="2025-01-01",
        metric_period_end="2025-12-01",
        metric_period_source="filled_from_task_window",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_lab_tests",
        metric_row_binding_status="resolved",
        evidence_quote="| No. of specimens tested | 52,021 | Week 47 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2025-week-47.html",
        source_title="CDC FluView Week 47",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    requirement = {
        "requirement_id": "boston_flu_task_window_2025_01_01_2025_12_01",
        "disease": "FLU",
        "geography": "Boston",
        "location": "Boston",
        "period_start": "2025-01-01",
        "period_end": "2025-12-01",
        "period_basis": "task_window",
        "time_granularity": "task_window",
        "accepted_metric_families": ["lab_test_count", "lab_positive_count"],
    }
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "Boston",
            "start_date": "2025-01-01",
            "end_date": "2025-12-01",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "Boston",
            "start_date": "2025-01-01",
            "end_date": "2025-12-01",
            "collection_mode": "direct_collection",
        },
        task_evidence_contract={
            "time_granularity": "task_window",
            "requirements": [requirement],
        },
        source_coverage_requirements=[requirement],
        source_registry=[
            _source(
                "src_cdc_weekly_national",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                coverage_requirement_ids=[requirement["requirement_id"]],
            )
        ],
        documents=[
            _document(
                "src_cdc_weekly_national",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_html",
                quality_status="usable",
                clean_text=record["evidence_quote"],
            )
        ],
        evidence_chunks=[
            _chunk(
                "chunk_rec_cdc_weekly_boston_inherited",
                "src_cdc_weekly_national",
                text=record["evidence_quote"],
                source_role_final="collection",
                coverage_requirement_ids=[requirement["requirement_id"]],
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert "record_geography_inherited_without_source_evidence" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert "record_period_inherited_from_task_window_without_source_evidence" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["source_coverage_audit"]["coverage_completeness_status"] != (
        "complete_target_coverage"
    )


def test_coverage_not_complete_when_only_edge_metric_records_are_accepted():
    record = _record(
        "rec_tb_missing_age",
        source_id="src_cdc_tb_annual",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="United States",
        geographic_scope="United States",
        geographic_scope_type="country",
        reporting_period="2025 annual provisional TB data",
        cases_unspecified=None,
        metric_name="TB cases with age missing or unknown",
        metric_value=91,
        metric_unit="count",
        metric_category="missing_demographic_count",
        metric_period_start="2025-01-01",
        metric_period_end="2025-12-31",
        metric_period_source="llm_extracted",
        resolved_column_period_type="annual_period",
        metric_column_semantics_status="resolved",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_missing_age",
        metric_row_binding_status="resolved",
        evidence_quote="Age was missing or unknown for 91 TB cases in 2025.",
        source_url="https://www.cdc.gov/tb-data/provisional-2025.html",
        source_title="Provisional 2025 Tuberculosis Data, United States",
        publisher="CDC",
        actual_publisher="CDC",
        source_type_final="national_public_health_agency",
        credibility_level="high",
    )
    requirement = {
        "requirement_id": "united_states_tuberculosis_annual_2025",
        "disease": "Tuberculosis",
        "geography": "United States",
        "location": "United States",
        "period_start": "2025-01-01",
        "period_end": "2025-12-31",
        "period_basis": "annual",
        "time_granularity": "annual",
        "accepted_metric_families": [
            "case_count",
            "incidence_rate",
            "death_count",
        ],
    }
    state = _state(
        [record],
        structured_task={
            "disease": "Tuberculosis",
            "location": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "United States",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        task_evidence_contract={
            "time_granularity": "annual",
            "requirements": [requirement],
        },
        source_coverage_requirements=[requirement],
        source_registry=[
            _source(
                "src_cdc_tb_annual",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CDC",
                actual_publisher="CDC",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                coverage_requirement_ids=[requirement["requirement_id"]],
            )
        ],
        documents=[
            _document(
                "src_cdc_tb_annual",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_html",
                quality_status="usable",
                clean_text=record["evidence_quote"],
            )
        ],
        structured_extraction_summary={
            "core_metric_extraction_gap_count": 1,
            "core_metric_extraction_gaps": [
                {
                    "source_id": "src_cdc_tb_annual",
                    "reason": "core_metric_text_attempted_but_no_records_extracted",
                    "core_metric_chunk_count": 2,
                }
            ],
        },
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_tb_missing_age"}
    audit = result["source_coverage_audit"]
    assert audit["coverage_completeness_status"] != "complete_target_coverage"
    row = audit["requirements"][0]
    assert row["accepted_record_count"] == 1
    assert row["strict_status"] == "edge_metric_only"
    assert row["task_value_status"] == "edge_metric_only"
    assert result["direct_collection_summary"]["core_metric_extraction_gap_count"] == 1


def test_best_available_records_inherit_requirement_linkage_and_audit_reason():
    record = _record(
        "rec_germany_measles_europe_context",
        source_id="src_ecdc_measles_europe",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="Romania",
        geographic_scope="EU/EEA",
        geographic_scope_type="region",
        reporting_period="2024 annual surveillance",
        metric_name="Measles deaths",
        metric_value=8,
        metric_unit="count",
        metric_category="death_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        resolved_column_period_type="annual_period",
        metric_column_semantics_status="resolved",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="Romania reported eight measles deaths in 2024 in the EU/EEA report.",
        source_url="https://www.ecdc.europa.eu/en/measles-annual-report-2024",
        source_title="ECDC measles annual report 2024",
        publisher="ECDC",
        actual_publisher="ECDC",
        source_type_final="supranational_public_health_agency",
        credibility_level="high",
    )
    requirement = {
        "requirement_id": "germany_measles_annual_2024",
        "disease": "Measles",
        "geography": "Germany",
        "location": "Germany",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "period_basis": "annual",
        "time_granularity": "annual",
        "accepted_metric_families": ["case_count", "death_count"],
    }
    state = _state(
        [record],
        structured_task={
            "disease": "Measles",
            "location": "Germany",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Measles",
            "geography": "Germany",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        source_coverage_requirements=[requirement],
        source_registry=[
            _source(
                "src_ecdc_measles_europe",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="ECDC",
                actual_publisher="ECDC",
                source_type_final="supranational_public_health_agency",
                source_role_final="best_available_context_candidate",
                credibility_level="high",
                credibility_score=0.9,
                coverage_requirement_ids=[requirement["requirement_id"]],
            )
        ],
        documents=[
            _document(
                "src_ecdc_measles_europe",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_html",
                quality_status="usable",
                clean_text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_germany_measles_europe_context"
    }
    best = package["best_available_context_records"][0]
    assert best["coverage_requirement_ids"] == ["germany_measles_annual_2024"]
    assert best["best_available_reason"] == "geography_mismatch_best_available_context"
    assert best["record_geography_fit_status"] == "broader_than_task"
    assert best["record_period_fit_status"] == "exact"
    audit_row = result["source_coverage_audit"]["requirements"][0]
    assert audit_row["strict_status"] == "best_available_only"
    assert audit_row["best_available_record_ids"] == [
        "rec_germany_measles_europe_context"
    ]


def test_final_records_inherit_requirement_linkage_from_source_metadata():
    req_id = "united_states_influenza_official_week_40_2024"
    record = _record(
        "rec_cdc_week40_current",
        source_id="src_cdc_week40",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="",
        geographic_scope="United States",
        geographic_scope_type="country",
        reporting_period="Week 40, 2024",
        metric_name="Influenza positivity",
        metric_value=0.7,
        metric_unit="percent",
        metric_category="lab_positivity_percent",
        cases_unspecified=None,
        deaths=None,
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="source_reporting_period",
        resolved_column_period_type="current_period",
        metric_column_semantics_status="resolved",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="During Week 40, 0.7% of specimens tested positive for influenza.",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="Weekly US Influenza Surveillance Report: Week 40",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type_final="national_public_health_agency",
        coverage_requirement_ids=[],
    )
    requirement = {
        "requirement_id": req_id,
        "disease": "Influenza",
        "geography": "United States",
        "location": "United States",
        "period_start": "2024-09-29",
        "period_end": "2024-10-05",
        "period_basis": "mmwr_week",
        "time_granularity": "weekly",
        "accepted_metric_families": ["lab_positivity_percent"],
    }
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_coverage_requirements=[requirement],
        source_registry=[
            _source(
                "src_cdc_week40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher=record["publisher"],
                actual_publisher=record["actual_publisher"],
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                coverage_requirement_ids=[req_id],
            )
        ],
        documents=[
            _document(
                "src_cdc_week40",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                coverage_requirement_ids=[req_id],
                clean_text=record["evidence_quote"],
            )
        ],
        evidence_chunks=[
            _chunk(
                "chunk_rec_cdc_week40_current",
                "src_cdc_week40",
                coverage_requirement_ids=[req_id],
                text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_cdc_week40_current"}
    assert package["final_dataset"][0]["coverage_requirement_ids"] == [req_id]
    audit_row = result["source_coverage_audit"]["requirements"][0]
    assert audit_row["accepted_record_ids"] == ["rec_cdc_week40_current"]
    assert audit_row["strict_status"] in {
        "accepted_exact_record",
        "core_metric_accepted",
    }


def test_pending_review_records_inherit_requirement_linkage_and_summary_counts():
    req_id = "london_measles_task_window_2024_09_29_2024_10_05"
    record = _record(
        "rec_cidrap_london_measles",
        source_id="src_cidrap_london_measles",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United Kingdom",
        subnational_location="London",
        geographic_scope="London",
        geographic_scope_type="city",
        reporting_period="September 29 - October 5, 2024",
        metric_name="Measles cases",
        metric_value=12,
        metric_unit="count",
        metric_category="case_count",
        cases_unspecified=12,
        deaths=None,
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="source_reporting_period",
        resolved_column_period_type="current_period",
        metric_column_semantics_status="resolved",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        evidence_quote="CIDRAP reported 12 measles cases in London during September 29 through October 5, 2024.",
        source_url="https://www.cidrap.umn.edu/measles/london-update",
        source_title="London measles update",
        publisher="CIDRAP",
        actual_publisher="CIDRAP",
        source_type_final="secondary_media",
        source_role_final="needs_human_review",
        credibility_level="medium",
        coverage_requirement_ids=[],
    )
    requirement = {
        "requirement_id": req_id,
        "disease": "Measles",
        "geography": "London",
        "location": "London",
        "period_start": "2024-09-29",
        "period_end": "2024-10-05",
        "period_basis": "task_window",
        "time_granularity": "task_window",
        "accepted_metric_families": ["case_count"],
    }
    state = _state(
        [record],
        structured_task={
            "disease": "Measles",
            "location": "London",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Measles",
            "geography": "London",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_coverage_requirements=[requirement],
        source_registry=[
            _source(
                "src_cidrap_london_measles",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="CIDRAP",
                actual_publisher="CIDRAP",
                source_type_final="secondary_media",
                source_role_final="needs_human_review",
                requires_human_review=True,
                coverage_requirement_ids=[req_id],
            )
        ],
        documents=[
            _document(
                "src_cidrap_london_measles",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                coverage_requirement_ids=[req_id],
                clean_text=record["evidence_quote"],
            )
        ],
        evidence_chunks=[
            _chunk(
                "chunk_rec_cidrap_london_measles",
                "src_cidrap_london_measles",
                coverage_requirement_ids=[req_id],
                text=record["evidence_quote"],
            )
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {
        "rec_cidrap_london_measles"
    }
    pending = package["pending_review_records"][0]
    assert pending["coverage_requirement_ids"] == [req_id]
    assert pending["matched_requirement_id"] == req_id
    assert pending["matched_requirement_ids"] == [req_id]
    assert pending["requirement_match_status"] == "linked_to_task_requirement"
    assert pending["requirement_geography"] == "London"
    assert pending["requirement_period_start"] == "2024-09-29"
    assert pending["requirement_period_end"] == "2024-10-05"
    assert pending["requirement_time_granularity"] == "task_window"
    assert pending["requires_human_review"] is True
    summary = result["direct_collection_summary"]
    assert summary["pending_review_record_count"] == 1
    assert summary["quarantined_record_count"] == 0
    assert summary["best_available_context_record_count"] == 0


def test_finalization_does_not_complete_coverage_for_parsed_source_without_accepted_records():
    record = _record(
        "rec_wrong_period_context",
        source_id="src_vdh_news_context",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        date_reported="",
        date_anchor="",
        reporting_period="2024-25 season",
        metric_name="Virginia influenza outbreaks last season",
        metric_value=366,
        metric_unit="count",
        metric_category="outbreak_count",
        metric_period_start="2024-09-01",
        metric_period_end="2025-05-31",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="Last season, Virginia reported six influenza-associated pediatric deaths and 366 influenza outbreaks.",
        source_url="https://www.vdh.virginia.gov/news/tag/influenza",
        source_title="VDH influenza news tag",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type_final="state_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
    )
    req_id = "virginia_influenza_official_week_40_2024"
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "Virginia",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_vdh_news_context",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                source_type_final="state_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                coverage_requirement_ids=[req_id],
            )
        ],
        documents=[
            _document(
                "src_vdh_news_context",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_html",
                quality_status="usable",
                clean_text=record["evidence_quote"],
            )
        ],
        source_coverage_requirements=[
            {
                "requirement_id": req_id,
                "disease": "FLU",
                "location": "Virginia",
                "reporting_period_start": "2024-09-29",
                "reporting_period_end": "2024-10-05",
                "reason": "target weekly surveillance report",
            }
        ],
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    audit = result["source_coverage_audit"]
    assert audit["accepted_record_count"] == 0
    assert audit["coverage_completeness_status"] != "complete_target_coverage"
    assert audit["coverage_status"] in {
        "records_quarantined",
        "best_available_only",
        "partial_target_coverage",
        "no_target_coverage",
    }
    assert audit["missing_requirement_ids"] == [req_id]
    assert result["direct_collection_summary"]["coverage_completeness_status"] != (
        "complete_target_coverage"
    )


def test_finalization_reports_partial_coverage_when_only_one_requirement_accepted():
    record = _record(
        "rec_week_42_metric",
        source_id="src_vdh_week_42",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        reporting_period="MMWR week 42, 2024",
        metric_name="Influenza ED visits",
        metric_value=386,
        metric_unit="count",
        metric_category="ed_visit_count",
        metric_period_start="2024-10-13",
        metric_period_end="2024-10-19",
        metric_period_source="filled_from_source_reporting_period",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        source_row_id="row_ed_visits",
        metric_row_binding_status="resolved",
        evidence_quote="| Diagnosed influenza ED visits | 386 |",
        source_url="https://www.vdh.virginia.gov/content/uploads/sites/3/2024/10/2024-25_Weekly-RDS-Report_Week-42.pdf",
        source_title="VDH Weekly RDS Report Week 42",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type_final="state_public_health_agency",
        credibility_level="high",
    )
    req_41 = {
        "requirement_id": "virginia_influenza_official_week_41_2024",
        "disease": "FLU",
        "location": "Virginia",
        "reporting_period_start": "2024-10-06",
        "reporting_period_end": "2024-10-12",
        "official_candidate_urls": [
            "https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-41.pdf"
        ],
        "reason": "target weekly surveillance report",
    }
    req_42 = {
        "requirement_id": "virginia_influenza_official_week_42_2024",
        "disease": "FLU",
        "location": "Virginia",
        "reporting_period_start": "2024-10-13",
        "reporting_period_end": "2024-10-19",
        "official_candidate_urls": [record["source_url"]],
        "reason": "target weekly surveillance report",
    }
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-19",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-19",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_vdh_week_41_bad",
                canonical_url=req_41["official_candidate_urls"][0],
                title="VDH Weekly RDS Report Week 41",
                publisher="Virginia Department of Health",
                source_type_final="state_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                must_fetch=True,
                coverage_requirement_ids=[req_41["requirement_id"]],
            ),
            _source(
                "src_vdh_week_42",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Virginia Department of Health",
                source_type_final="state_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                must_fetch=True,
                coverage_requirement_ids=[req_42["requirement_id"]],
            ),
        ],
        documents=[
            _document(
                "src_vdh_week_41_bad",
                url=req_41["official_candidate_urls"][0],
                canonical_url=req_41["official_candidate_urls"][0],
                title="Page not found",
                parse_status="parsed_html",
                quality_status="unusable",
                clean_text="Page not found",
            ),
            _document(
                "src_vdh_week_42",
                url=record["source_url"],
                canonical_url=record["source_url"],
                title=record["source_title"],
                parse_status="parsed_pdf",
            ),
        ],
        source_coverage_requirements=[req_41, req_42],
    )

    package, result = _finalize(state)

    audit = package["source_coverage_audit"]
    assert audit["coverage_status"] == "partial_target_coverage"
    assert audit["coverage_completeness_status"] == "partial_target_coverage"
    assert audit["complete_requirement_count"] == 1
    assert audit["partial_requirement_count"] == 1
    assert audit["missing_requirement_ids"] == [req_41["requirement_id"]]
    assert audit["accepted_requirement_count"] == 1
    assert audit["requirements"][0]["missing_reason"] == "target_alias_error_page"
    assert result["source_coverage_audit"]["coverage_status"] == "partial_target_coverage"
    assert result["direct_collection_summary"]["coverage_status"] == (
        "partial_target_coverage"
    )


def test_finalization_reports_target_alias_error_page_needs_fallback():
    req_id = "virginia_influenza_official_week_40_2024"
    bad_url = (
        "https://www.vdh.virginia.gov/content/uploads/sites/13/"
        "2024/10/Weekly-RDS-Report_Week-40.pdf"
    )
    state = _state(
        [],
        structured_task={
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "Virginia",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_vdh_week_40_bad",
                canonical_url=bad_url,
                title="VDH Weekly RDS Report Week 40",
                publisher="Virginia Department of Health",
                source_type_final="state_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
                must_fetch=True,
                coverage_requirement_ids=[req_id],
            )
        ],
        documents=[
            _document(
                "src_vdh_week_40_bad",
                url=bad_url,
                canonical_url=bad_url,
                title="Page not found - Virginia Department of Health",
                parse_status="parsed_html",
                quality_status="unusable",
                clean_text="Page not found. The page you requested was not found.",
            )
        ],
        source_coverage_requirements=[
            {
                "requirement_id": req_id,
                "disease": "FLU",
                "location": "Virginia",
                "reporting_period_start": "2024-09-29",
                "reporting_period_end": "2024-10-05",
                "official_candidate_urls": [bad_url],
                "reason": "target weekly surveillance report",
            }
        ],
        content_fetch_summary={
            "target_unusable_needs_fallback": True,
            "fallback_fetch_attempted": False,
            "usable_task_collection_document_count": 0,
            "error_alias_urls": [bad_url],
        },
        structured_extraction_summary={
            "no_task_collection_document": True,
            "extraction_blocking_reason": "no_task_collection_document",
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert result["source_coverage_audit"]["coverage_status"] == (
        "target_alias_error_page_needs_fallback"
    )
    assert result["direct_collection_summary"]["coverage_status"] == (
        "target_alias_error_page_needs_fallback"
    )
    assert result["direct_collection_summary"]["missing_requirement_ids"] == [req_id]


def test_direct_collection_summary_final_case_count_matches_exported_view():
    record = _record(
        "rec_outbreak_summary",
        source_id="src_ny_week_44",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="New York",
        geographic_scope="New York",
        date_reported="2024-11-02",
        date_anchor="2024-11-02",
        reporting_period="week ending November 2, 2024",
        cases_unspecified=None,
        cumulative_count=13.0,
        count_semantics="season-to-date outbreak count",
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "Thirteen influenza outbreaks have been reported from hospitals "
            "and nursing homes season to date."
        ),
        source_url=(
            "https://www.health.ny.gov/diseases/communicable/influenza/"
            "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf"
        ),
        source_title="New York State Influenza Surveillance Report",
        publisher="New York State Department of Health",
        actual_publisher="New York State Department of Health",
        source_type_final="state_or_local_public_health_agency",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_ny_week_44",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="New York State Department of Health",
                actual_publisher="New York State Department of Health",
                source_type_final="state_or_local_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_outbreak_summary"}
    assert package["final_case_dataset"] == []
    assert result["direct_collection_summary"]["final_case_dataset_count"] == 0
    assert result["run_quality_summary"]["final_case_dataset_count"] == 0


def test_h5n1_records_are_excluded_from_default_seasonal_flu_task():
    record = _record(
        "rec_h5n1_context",
        disease="Influenza A(H5N1)",
        disease_standard_name="Avian influenza A(H5N1)",
        virus_or_syndrome="H5N1 avian influenza",
        pathogen_or_syndrome="H5N1",
        country="United States of America",
        subnational_location="California",
        geographic_scope="California",
        date_reported="2024-10-04",
        date_anchor="2024-10-04",
        cases_confirmed=1.0,
        cases_unspecified=None,
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote="One human H5N1 case was reported in a dairy worker.",
        source_title="CDC H5N1 dairy worker update",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "user_request": "Collect FLU surveillance data for Virginia.",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_h5n1_context"}
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] == "quarantined_disease_mismatch"
    assert "non_seasonal_influenza_subtype" in quarantined["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_default_seasonal_flu_accepts_h1n1_record_from_mixed_h5_table():
    record = _record(
        "rec_h1n1_lab_positive",
        disease="Seasonal influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza A",
        pathogen_or_syndrome="A(H1N1)pdm09",
        country="United States of America",
        subnational_location="United States",
        geographic_scope="United States",
        geographic_scope_type="national",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        tests_positive=17.0,
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        count_semantics="weekly public health laboratory influenza subtype count",
        evidence_quote=(
            "Public health laboratories reported A(H1N1)pdm09, A(H3N2), "
            "Influenza B, and a separate H5 row in the same table."
        ),
        source_title="CDC FluView public health laboratory subtype table",
        source_id="src_cdc_week40",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        publisher="Centers for Disease Control and Prevention",
        source_type="official_public_health_agency",
        credibility_level="high",
    )
    state = _state(
        [record],
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "user_request": (
                "Collect FLU surveillance data for United States from "
                "2024-09-29 to 2024-10-05."
            ),
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza", "A(H1N1)pdm09", "A(H3N2)"],
            "syndrome_terms": ["influenza-like illness"],
        },
        source_registry=[
            _source(
                "src_cdc_week40",
                canonical_url=record["source_url"],
                title=record["source_title"],
                publisher="Centers for Disease Control and Prevention",
                source_type_final="national_public_health_agency",
                source_role_final="collection",
                credibility_level="high",
                credibility_score=0.98,
            )
        ],
    )

    package, result = _finalize(state)

    assert _record_ids(package["final_dataset"]) == {"rec_h1n1_lab_positive"}
    assert package["quarantined_records"] == []
    assert result["run_quality_summary"]["final_dataset_count"] == 1


def test_record_date_outside_task_window_is_quarantined_even_from_official_source():
    record = _record(
        "rec_vdh_outside_window",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        date_reported="2025-10-23",
        date_anchor="2025-10-23",
        reporting_period="2025-26 influenza season",
        cases_unspecified=366.0,
        deaths=None,
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        source_url="https://www.vdh.virginia.gov/news/tag/flu",
        source_title="flu Archives - Newsroom - Virginia Department of Health",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        evidence_quote=(
            "October 23, 2025. Last season, Virginia reported six "
            "influenza-associated pediatric deaths and 366 influenza outbreaks."
        ),
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "VIRGINIA",
                "start_date": "2024-11-01",
                "end_date": "2024-11-20",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "VIRGINIA",
                "start_date": "2024-11-01",
                "end_date": "2024-11-20",
                "time_window": "2024-11-01 to 2024-11-20",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_rec_vdh_outside_window",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Virginia Department of Health",
                    actual_publisher="Virginia Department of Health",
                    source_type="official_public_health_agency",
                    source_type_final="state_or_local_public_health_agency",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_vdh_outside_window"}
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] == "quarantined_outside_scope"
    assert "record_date_outside_task_window" in quarantined["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["outside_scope_record_count"] == 1


def test_direct_collection_quarantines_explicit_event_start_after_task_window():
    record = _record(
        "rec_fda_november_salmonella_outbreak_deaths",
        source_id="src_fda_november_salmonella",
        disease="Salmonellosis (non-typhoidal Salmonella)",
        disease_standard_name="Salmonellosis (non-typhoidal Salmonella)",
        virus_or_syndrome="Salmonella",
        pathogen_or_syndrome="Salmonella",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-07-31",
        date_anchor="2024-07-31",
        event_start_date="2024-11-01",
        event_end_date=None,
        reporting_period="November 2024 outbreak investigation",
        cases_unspecified=None,
        deaths=0.0,
        metric_name="Deaths",
        metric_value=0.0,
        metric_unit="count",
        metric_category="death_count",
        metric_period_start="2024-05-01",
        metric_period_end="2024-07-31",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="death_count",
        statistical_count_type="cumulative",
        observation_type="death_record",
        observation_types=["death_record"],
        primary_case_dataset_eligible=False,
        source_url=(
            "https://www.fda.gov/food/outbreaks-foodborne-illness/"
            "outbreak-investigation-salmonella-cucumbers-november-2024"
        ),
        source_title="Outbreak Investigation of Salmonella: Cucumbers (November 2024)",
        publisher=None,
        actual_publisher=None,
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.8112,
        evidence_quote="Deaths: 0",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Salmonella",
                "location": "United States",
                "start_date": "2024-05-01",
                "end_date": "2024-07-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Salmonella",
                "geography": "United States",
                "start_date": "2024-05-01",
                "end_date": "2024-07-31",
                "time_window": "2024-05-01 to 2024-07-31",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "Salmonella",
                "disease_standard_name": "Salmonellosis (non-typhoidal Salmonella)",
                "aliases": ["Salmonella", "Salmonellosis", "NTS"],
                "pathogen_terms": ["Salmonella", "Salmonella enterica"],
                "syndrome_terms": ["gastroenteritis", "foodborne illness"],
            },
            source_registry=[
                _source(
                    "src_fda_november_salmonella",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=None,
                    actual_publisher=None,
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.8112,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_fda_november_salmonella_outbreak_deaths"
    }
    quarantined = package["quarantined_records"][0]
    assert "record_event_period_outside_task_window" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_as_of_date_outside_task_window_overrides_inherited_metric_period():
    record = _record(
        "rec_official_as_of_outside_window",
        disease="Legionnaires' Disease",
        disease_standard_name="Legionnaires' Disease",
        virus_or_syndrome="Legionella pneumophila",
        pathogen_or_syndrome="Legionella pneumophila",
        country="United States of America",
        subnational_location="New York City",
        geographic_scope="New York City",
        geographic_scope_type="subnational",
        date_reported="2025-08-26",
        date_anchor="2025-08-26",
        as_of_date="2025-08-26",
        reporting_period="2024-06-01 to 2024-08-31",
        cases_confirmed=113.0,
        deaths=6.0,
        metric_name="confirmed cases",
        metric_value=113.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-06-01",
        metric_period_end="2024-08-31",
        metric_period_source="filled_from_source_reporting_period",
        statistical_count_type="current_period",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        source_id="src_nyc_health_cluster",
        source_url="https://www.nyc.gov/site/doh/health/health-topics/legionnaires.page",
        source_title="Legionnaires disease community cluster update",
        publisher="New York City Department of Health and Mental Hygiene",
        actual_publisher="New York City Department of Health and Mental Hygiene",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.98,
        evidence_quote=(
            "On August 26, health officials reported 113 confirmed Legionnaires' disease cases "
            "and six deaths in New York City."
        ),
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Legionnaires' disease",
                "location": "New York City",
                "start_date": "2024-06-01",
                "end_date": "2024-08-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Legionnaires' disease",
                "geography": "New York City",
                "start_date": "2024-06-01",
                "end_date": "2024-08-31",
                "time_window": "2024-06-01 to 2024-08-31",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_nyc_health_cluster",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="official_public_health_agency",
                    source_type_final="state_or_local_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_official_as_of_outside_window"
    }
    quarantined = package["quarantined_records"][0]
    assert "record_as_of_date_outside_task_window" in quarantined[
        "quality_gate_blocking_flags"
    ]
    assert result["run_quality_summary"]["outside_scope_record_count"] == 1


def test_direct_collection_accepts_metric_record_with_partially_overlapping_period():
    record = _record(
        "rec_weekly_metric_partial_overlap",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="United States",
        geographic_scope="United States",
        geographic_scope_type="national",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="MMWR week 40, 2024",
        metric_name="Number of positive specimens",
        metric_value=380.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        tests_positive=380.0,
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory positive specimens",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        source_id="src_cdc_week40",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type="official_public_health_agency",
        source_type_final="national_public_health_agency",
        credibility_level="high",
        credibility_score=0.98,
        evidence_quote="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
        source_row_id="row_positive_week40",
        source_column_label="Current Week",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-01",
                "end_date": "2024-10-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-01",
                "end_date": "2024-10-01",
                "time_window": "2024-09-01 to 2024-10-01",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_cdc_week40",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Centers for Disease Control and Prevention",
                    actual_publisher="Centers for Disease Control and Prevention",
                    source_type="official_public_health_agency",
                    source_type_final="national_public_health_agency",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_weekly_metric_partial_overlap"}
    assert package["quarantined_records"] == []
    accepted = package["final_dataset"][0]
    assert accepted["period_overlap_status"] == "partial_overlap"
    assert "metric_period_partially_overlaps_task_window" in accepted[
        "quality_gate_warning_flags"
    ]
    assert result["run_quality_summary"]["final_dataset_count"] == 1


def test_direct_collection_quarantines_national_season_record_for_state_short_window():
    record = _record(
        "rec_cidrap_national_season",
        source_id="src_cidrap_fluview",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza",
        country="United States of America",
        subnational_location="",
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-10-01",
        date_anchor="2024-10-01",
        event_start_date="2024-10-01",
        event_end_date="2025-05-31",
        reporting_period="2024-25 influenza season",
        cases_unspecified=None,
        tests_positive=3200.0,
        hospitalizations=110.0,
        deaths=12.0,
        count_semantics="national season-to-date aggregate surveillance estimate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "National 2024-25 influenza season estimates include laboratory "
            "positives, hospitalizations, and deaths across the United States."
        ),
        source_url="https://www.cidrap.umn.edu/influenza-general/fluview-national-2024-25",
        source_title="CIDRAP summary of national CDC FluView estimates",
        publisher="CIDRAP",
        actual_publisher="CIDRAP",
        source_type_final="academic_or_peer_reviewed_source",
        credibility_level="high",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "California",
                "start_date": "2024-10-01",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "California",
                "start_date": "2024-10-01",
                "end_date": "2024-10-05",
                "time_window": "2024-10-01 to 2024-10-05",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_cidrap_fluview",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="CIDRAP",
                    actual_publisher="CIDRAP",
                    source_type_final="academic_or_peer_reviewed_source",
                    source_role_final="context",
                    credibility_level="high",
                    credibility_score=0.9,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_cidrap_national_season"}
    pending = package["pending_review_records"][0]
    assert pending["record_final_inclusion_status"] == "pending_human_review"
    assert "source_trust_requires_human_review" in pending["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["pending_review_record_count"] == 1


def test_direct_collection_quarantines_record_when_source_year_conflicts_with_metric_period():
    record = _record(
        "rec_wrong_year_vdh_metric",
        source_id="src_vdh_week_41_2023",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza",
        country="United States",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        date_reported="2023-10",
        date_anchor="2023-10-14",
        reporting_period="Season-to-date through Flu season to date (October 2023, Week 41)",
        cases_confirmed=None,
        tests_positive=119.0,
        metric_name="Influenza positive lab reports",
        metric_value=119.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-10-06",
        metric_period_end="2024-10-12",
        metric_period_source="filled_from_column_label",
        count_semantics="weekly laboratory positive specimens",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="Flu season to date, positive lab reports: 119.",
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
        source_row_id="row_positive_week41",
        source_url=(
            "https://www.vdh.virginia.gov/content/uploads/sites/3/2023/11/"
            "Weekly-RDS-Report_Week-41.pdf"
        ),
        source_title="Weekly Respiratory Disease Surveillance Report Week 41",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        credibility_level="high",
        credibility_score=0.98,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "Virginia",
                "start_date": "2024-10-06",
                "end_date": "2024-10-12",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "Virginia",
                "start_date": "2024-10-06",
                "end_date": "2024-10-12",
                "time_window": "2024-10-06 to 2024-10-12",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_vdh_week_41_2023",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Virginia Department of Health",
                    actual_publisher="Virginia Department of Health",
                    source_type_final="state_or_local_public_health_agency",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert package["final_case_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_wrong_year_vdh_metric"}
    quarantined = package["quarantined_records"][0]
    assert "source_period_mismatch" in quarantined["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_direct_collection_exports_best_available_context_without_relaxing_strict_final():
    record = _record(
        "rec_vdh_best_available_season_context",
        source_id="src_vdh_end_of_season",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza",
        country="United States",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        date_reported="2024-06-30",
        date_anchor="2024-06-30",
        reporting_period="2023-2024 influenza season",
        cases_confirmed=None,
        tests_positive=1200.0,
        metric_name="Influenza positive lab reports",
        metric_value=1200.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2023-10-01",
        metric_period_end="2024-06-30",
        metric_period_source="llm_extracted",
        count_semantics="season-to-date aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="VDH reported 1,200 positive influenza lab reports during the 2023-2024 season.",
        chunk_kind="text",
        metric_row_binding_status="not_applicable",
        source_url="https://www.vdh.virginia.gov/epidemiology/influenza-flu-in-virginia",
        source_title="Influenza (Flu) in Virginia",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        credibility_level="high",
        credibility_score=0.98,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "Virginia",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "Virginia",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "time_window": "2024-09-29 to 2024-10-05",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_vdh_end_of_season",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Virginia Department of Health",
                    actual_publisher="Virginia Department of Health",
                    source_type_final="state_or_local_public_health_agency",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_vdh_best_available_season_context"
    }
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_vdh_best_available_season_context"
    }
    summary = result["direct_collection_summary"]
    assert summary["best_available_context_record_count"] == 1
    assert result["run_quality_summary"]["final_dataset_count"] == 0


def test_direct_collection_lab_positive_metric_is_not_case_dataset_record():
    record = _record(
        "rec_vdh_lab_positive_metric",
        source_id="src_vdh_week_41_2024",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza",
        country="United States",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        date_reported="2024-10-12",
        date_anchor="2024-10-12",
        reporting_period="MMWR week 41, 2024",
        cases_confirmed=119.0,
        tests_positive=None,
        metric_name="Influenza positive lab reports",
        metric_value=119.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-10-06",
        metric_period_end="2024-10-12",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="weekly laboratory positive specimens",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        evidence_quote="Influenza positive lab reports: 119.",
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
        source_row_id="row_positive_week41",
        source_url=(
            "https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/"
            "Weekly-RDS-Report_Week-41.pdf"
        ),
        source_title="Weekly Respiratory Disease Surveillance Report Week 41",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        credibility_level="high",
        credibility_score=0.98,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "Virginia",
                "start_date": "2024-10-06",
                "end_date": "2024-10-12",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "Virginia",
                "start_date": "2024-10-06",
                "end_date": "2024-10-12",
                "time_window": "2024-10-06 to 2024-10-12",
                "collection_mode": "direct_collection",
            },
            disease_intelligence={
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            source_registry=[
                _source(
                    "src_vdh_week_41_2024",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Virginia Department of Health",
                    actual_publisher="Virginia Department of Health",
                    source_type_final="state_or_local_public_health_agency",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_vdh_lab_positive_metric"}
    accepted = package["final_dataset"][0]
    assert accepted["tests_positive"] == 119.0
    assert accepted.get("cases_confirmed") in (None, "")
    assert accepted["primary_case_dataset_eligible"] is False
    assert accepted["observation_type"] == "surveillance_summary"
    assert "surveillance_summary" in accepted["observation_types"]
    assert package["final_case_dataset"] == []
    assert result["run_quality_summary"]["final_dataset_count"] == 1
    assert result["run_quality_summary"]["final_case_dataset_count"] == 0


def test_direct_collection_social_source_metric_goes_to_human_review_not_strict_final():
    record = _record(
        "rec_social_tb_metric",
        source_id="src_facebook_tb_report",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        subnational_location=None,
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2023-12-31",
        date_anchor="2023-12-31",
        reporting_period="2023",
        cases_unspecified=None,
        deaths=None,
        metric_name="Missing TB cases",
        metric_value=230000.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2023-01-01",
        metric_period_end="2023-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual case aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="The India TB Report 2024 noted 230,000 missing TB cases in 2023.",
        chunk_kind="text",
        metric_row_binding_status="not_applicable",
        source_url="https://www.facebook.com/statecraftdaily/posts/india-tb-report-2024",
        source_title="Statecraft - India TB Report 2024",
        publisher="Statecraft",
        actual_publisher="Statecraft",
        source_type="social_media",
        source_type_final="social_media",
        source_role_final="collection",
        credibility_level="low",
        credibility_score=0.25,
        requires_human_review=True,
        human_review_reason="social source cannot establish strict public-health provenance",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_facebook_tb_report",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Statecraft",
                    actual_publisher="Statecraft",
                    source_type="social_media",
                    source_type_final="social_media",
                    source_role_final="collection",
                    credibility_level="low",
                    credibility_score=0.25,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_social_tb_metric"}
    decision = result["record_inclusion_decisions"][0]
    assert "source_trust_requires_human_review" in decision["quality_gate_blocking_flags"]
    assert result["direct_collection_summary"]["human_review_record_count"] == 1


def test_direct_collection_social_source_metric_goes_to_review_even_if_misclassified_high_trust():
    record = _record(
        "rec_facebook_measles_metric",
        source_id="src_facebook_measles",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-08-01",
        date_anchor="2024-08-01",
        reporting_period="2024 through August 1",
        cases_unspecified=None,
        metric_name="Total measles cases",
        metric_value=203.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-08-01",
        metric_period_source="llm_extracted",
        count_semantics="year-to-date case aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="As of August 1, 2024, a total of 203 measles cases were reported.",
        source_url="https://www.facebook.com/example/posts/measles-cases-2024",
        source_title="Measles cases in 2024 As of August 1, 2024 a total of ... - Facebook",
        publisher="Facebook",
        actual_publisher="Facebook",
        source_type="social_media",
        source_type_final="social_media",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.92,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_facebook_measles",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Facebook",
                    actual_publisher="Facebook",
                    source_type="social_media",
                    source_type_final="social_media",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.92,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_facebook_measles_metric"}
    decision = result["record_inclusion_decisions"][0]
    assert "source_trust_requires_human_review" in decision["quality_gate_blocking_flags"]


def test_direct_collection_secondary_source_metric_goes_to_review_even_if_high_credibility():
    record = _record(
        "rec_secondary_measles_metric",
        source_id="src_secondary_measles",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=None,
        metric_name="Total measles cases",
        metric_value=285.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual case aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="The article summarizes 285 measles cases reported in the United States in 2024.",
        source_url="https://example-news.invalid/measles-2024-summary",
        source_title="US measles cases in 2024",
        publisher="Example Health News",
        actual_publisher="Example Health News",
        source_type="secondary_media",
        source_type_final="secondary_media",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.86,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2024-01-01",
                "end_date": "2025-01-01",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_secondary_measles",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="Example Health News",
                    actual_publisher="Example Health News",
                    source_type="secondary_media",
                    source_type_final="secondary_media",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.86,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_secondary_measles_metric"}
    decision = result["record_inclusion_decisions"][0]
    assert "source_trust_requires_human_review" in decision["quality_gate_blocking_flags"]


def test_direct_collection_unknown_source_metric_goes_to_review_even_with_collection_role():
    record = _record(
        "rec_unknown_tb_metric",
        source_id="src_unknown_tb",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        subnational_location=None,
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=None,
        metric_name="TB notified cases",
        metric_value=2600000.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual case aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="India notified 2.6 million TB cases in 2024.",
        source_url="https://scienceopen.example/article/india-tb-burden",
        source_title="India TB burden 2024",
        publisher="",
        actual_publisher="",
        source_type="unknown",
        source_type_final="unknown",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.9,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_unknown_tb",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher="",
                    actual_publisher="",
                    source_type="unknown",
                    source_type_final="unknown",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.9,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {"rec_unknown_tb_metric"}
    decision = result["record_inclusion_decisions"][0]
    assert "source_trust_requires_human_review" in decision["quality_gate_blocking_flags"]


def test_direct_collection_annual_task_sends_campaign_partial_overlap_to_best_available():
    record = _record(
        "rec_tb_100_day_campaign",
        source_id="src_ctd_campaign",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        subnational_location=None,
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2025-02-22",
        date_anchor="2025-02-22",
        reporting_period="100-day TB campaign, 2024-12-07 to 2025-02-22",
        cases_unspecified=None,
        metric_name="TB patient notifications during 100-day campaign",
        metric_value=510000.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-12-07",
        metric_period_end="2025-02-22",
        metric_period_source="llm_extracted",
        count_semantics="campaign partial-overlap aggregate",
        statistical_count_type="partial_overlap_campaign",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="The 100-day TB campaign notified 5.10 lakh TB patients from Dec 7, 2024 to Feb 22, 2025.",
        source_url="https://tbcindia.mohfw.gov.in/100-day-tb-campaign",
        source_title="Press release link on PIB 6 - Central Tuberculosis Division",
        publisher="Central Tuberculosis Division",
        actual_publisher="Central Tuberculosis Division",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.95,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "annual",
                "requirements": [
                    {
                        "requirement_id": "india_tuberculosis_annual_2024",
                        "period_start": "2024-01-01",
                        "period_end": "2024-12-31",
                    }
                ],
            },
            source_registry=[
                _source(
                    "src_ctd_campaign",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.95,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {"rec_tb_100_day_campaign"}
    quarantined = package["quarantined_records"][0]
    assert "record_period_partial_overlap_for_annual_requirement" in quarantined["quality_gate_blocking_flags"]
    assert result["direct_collection_summary"]["best_available_context_record_count"] == 1


def test_direct_collection_annual_task_sends_as_of_partial_year_record_to_best_available():
    record = _record(
        "rec_virginia_measles_as_of_july",
        source_id="src_vdh_measles_update",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="state",
        date_reported="2024-07-11",
        date_anchor="2024-07-11",
        reporting_period="as of July 11, 2024",
        cases_unspecified=None,
        metric_name="measles outbreak count",
        metric_value=12.0,
        metric_unit="count",
        metric_category="outbreak_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-07-11",
        metric_period_source="llm_extracted",
        count_semantics="year-to-date as-of aggregate",
        statistical_count_type="year_to_date",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="In Virginia, 12 outbreaks of measles had been detected as of July 11.",
        source_url="https://www.vdh.virginia.gov/news/measles-update",
        source_title="Virginia measles update",
        publisher="Virginia Department of Health",
        actual_publisher="Virginia Department of Health",
        source_type="official_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.95,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "Virginia",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "Virginia",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "annual",
                "requirements": [
                    {
                        "requirement_id": "virginia_measles_annual_2024",
                        "period_start": "2024-01-01",
                        "period_end": "2024-12-31",
                    }
                ],
            },
            source_registry=[
                _source(
                    "src_vdh_measles_update",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="official_public_health_agency",
                    source_type_final="state_or_local_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.95,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_virginia_measles_as_of_july"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "record_period_partial_overlap_for_annual_requirement" in decision[
        "quality_gate_blocking_flags"
    ]


def test_direct_collection_trusted_large_annual_metric_can_enter_final_despite_simple_threshold():
    record = _record(
        "rec_ntep_large_tb_burden",
        source_id="src_ntep_tb_report",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        virus_or_syndrome="tuberculosis",
        pathogen_or_syndrome="mycobacterium tuberculosis",
        country="India",
        subnational_location=None,
        geographic_scope="India",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        date_anchor="2024-12-31",
        reporting_period="2024",
        cases_unspecified=None,
        deaths=None,
        metric_name="TB cases notified",
        metric_value=2600000.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual case aggregate",
        statistical_count_type="annual",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="India notified 2.6 million TB cases in 2024.",
        chunk_kind="narrative_metric",
        metric_row_binding_status="not_applicable",
        source_url="https://tbcindia.mohfw.gov.in/india-tb-report-2024",
        source_title="India TB Report 2024",
        publisher="Central TB Division, Ministry of Health and Family Welfare",
        actual_publisher="National Tuberculosis Elimination Programme",
        source_type="official_public_health_agency",
        source_type_final="official_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.95,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_ntep_tb_report",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="official_public_health_agency",
                    source_type_final="official_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.95,
                )
            ],
            anomaly_results=[
                _anomaly(
                    "rec_ntep_large_tb_burden",
                    severity="critical",
                    anomaly_type="abrupt_spike_simple_threshold",
                    reason="value exceeded generic simple anomaly threshold",
                )
            ],
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_ntep_large_tb_burden"}
    assert package["final_case_dataset"] == []
    assert package["quarantined_records"] == []
    assert package["pending_review_records"] == []
    decision = result["record_inclusion_decisions"][0]
    assert "source_aware_anomaly_requires_human_review" not in decision["quality_gate_blocking_flags"]
    assert "source_aware_simple_anomaly_accepted_for_high_trust_source" in decision[
        "quality_gate_warnings"
    ]
    assert result["direct_collection_summary"]["human_review_record_count"] == 0


def test_direct_collection_national_task_rejects_broader_regional_record_from_strict_final():
    record = _record(
        "rec_paho_americas_measles",
        source_id="src_paho_measles",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="Region of the Americas",
        subnational_location=None,
        geographic_scope="Region of the Americas",
        geographic_scope_type="region",
        date_reported="2025-12-31",
        date_anchor="2025-12-31",
        reporting_period="2025",
        cases_unspecified=None,
        metric_name="confirmed measles cases",
        metric_value=12600.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2025-01-01",
        metric_period_end="2025-12-31",
        metric_period_source="llm_extracted",
        count_semantics="annual regional aggregate",
        statistical_count_type="annual",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="The Region of the Americas reported 12,600 confirmed measles cases in 2025.",
        source_url="https://www.paho.org/en/documents/situation-report-2-measles-americas-region",
        source_title="Measles situation report, Region of the Americas",
        publisher="Pan American Health Organization",
        actual_publisher="Pan American Health Organization",
        source_type="international_public_health_agency",
        source_type_final="international_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.96,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "annual",
                "requirements": [
                    {
                        "requirement_id": "united_states_measles_annual_2025",
                        "period_start": "2025-01-01",
                        "period_end": "2025-12-31",
                        "geography": "United States",
                        "disease": "measles",
                    }
                ],
            },
            source_registry=[
                _source(
                    "src_paho_measles",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="international_public_health_agency",
                    source_type_final="international_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.96,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_paho_americas_measles"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "record_geography_broader_than_task" in decision["quality_gate_blocking_flags"]


def test_direct_collection_national_task_rejects_regional_record_even_when_quote_mentions_task_country():
    record = _record(
        "rec_paho_americas_with_us_mention",
        source_id="src_paho_measles_2026",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States of America",
        subnational_location=None,
        geographic_scope="Americas",
        geographic_scope_type="region",
        date_reported="2026-06-18",
        date_anchor="2026-06-18",
        reporting_period="EW21-EW22 2026",
        cases_unspecified=None,
        metric_name="new confirmed measles cases",
        metric_value=827.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2026-01-01",
        metric_period_end="2026-02-01",
        metric_period_source="EW21-EW22 2026",
        count_semantics="regional two-week aggregate",
        statistical_count_type="period_aggregate",
        resolved_column_period_type="period_aggregate",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "During EW21 and EW22 of 2026, the Region of the Americas "
            "reported 827 new confirmed measles cases. The United States "
            "showed plateauing trends."
        ),
        source_url="https://www.paho.org/en/documents/situation-report-5-measles-americas-region-18-june-2026",
        source_title="Situation Report #5: Measles in the Americas Region",
        publisher="Pan American Health Organization",
        actual_publisher="Pan American Health Organization",
        source_type="international_public_health_agency",
        source_type_final="international_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.96,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "task_window",
                "requirements": [
                    {
                        "requirement_id": "united_states_measles_task_window_2026_01_01_2026_02_01",
                        "period_start": "2026-01-01",
                        "period_end": "2026-02-01",
                        "geography": "United States",
                        "disease": "measles",
                    }
                ],
            },
            source_registry=[
                _source(
                    "src_paho_measles_2026",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="international_public_health_agency",
                    source_type_final="international_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.96,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_paho_americas_with_us_mention"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "record_geography_broader_than_task" in decision["quality_gate_blocking_flags"]


def test_direct_collection_task_window_rejects_ytd_record_even_when_dates_are_filled_to_window():
    record = _record(
        "rec_measles_ytd_filled_to_window",
        source_id="src_paho_measles_ytd",
        disease="Measles",
        disease_standard_name="Measles",
        virus_or_syndrome="measles",
        pathogen_or_syndrome="measles virus",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2026-05-07",
        date_anchor="2026-05-07",
        reporting_period="EW1-EW16 2026",
        cases_unspecified=None,
        metric_name="confirmed measles cases",
        metric_value=1792.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2026-01-01",
        metric_period_end="2026-02-01",
        metric_period_source="EW1-EW16 2026",
        count_semantics="season-to-date confirmed cases",
        statistical_count_type="season_to_date",
        resolved_column_period_type="season_to_date",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "Between epidemiological weeks EW1 and EW16 of 2026, "
            "the United States reported 1,792 confirmed measles cases."
        ),
        source_url="https://www.paho.org/en/documents/situation-report-2-measles-americas-region",
        source_title="Situation Report #2: Measles in the Americas Region",
        publisher="Pan American Health Organization",
        actual_publisher="Pan American Health Organization",
        source_type="international_public_health_agency",
        source_type_final="international_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.96,
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "Measles",
                "location": "United States",
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "Measles",
                "geography": "United States",
                "start_date": "2026-01-01",
                "end_date": "2026-02-01",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_paho_measles_ytd",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="international_public_health_agency",
                    source_type_final="international_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.96,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_measles_ytd_filled_to_window"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "record_period_semantics_not_exact_for_task_window" in decision[
        "quality_gate_blocking_flags"
    ]


def test_direct_collection_annual_task_rejects_weekly_current_row_filled_to_full_year():
    record = _record(
        "rec_florida_week51_current_row_filled_annual",
        source_id="src_cdc_week51",
        disease="Influenza",
        disease_standard_name="Influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza virus",
        country="United States of America",
        subnational_location="Florida",
        geographic_scope="Florida",
        geographic_scope_type="subnational",
        date_reported="2024-12-21",
        date_anchor="2024-12-21",
        reporting_period="MMWR Week 51 ending December 21, 2024",
        cases_unspecified=None,
        metric_name="Number of specimens tested",
        metric_value=74587.0,
        metric_unit="count",
        metric_category="lab_test_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="laboratory test count",
        statistical_count_type="current_period",
        resolved_column_period_type="current_period",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="| **No. of specimens tested** | 74,587 | 941,281 |",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-51.html",
        source_title="Weekly US Influenza Surveillance Report: Key Updates for Week 51",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type="national_public_health_agency",
        source_type_final="national_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.98,
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "Florida",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "Florida",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "annual",
                "requirements": [
                    {
                        "requirement_id": "florida_flu_annual_2024",
                        "period_start": "2024-01-01",
                        "period_end": "2024-12-31",
                        "period_basis": "annual",
                        "geography": "Florida",
                        "disease": "flu",
                    }
                ],
            },
            source_registry=[
                _source(
                    "src_cdc_week51",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="national_public_health_agency",
                    source_type_final="national_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.98,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["best_available_context_records"]) == {
        "rec_florida_week51_current_row_filled_annual"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "record_period_semantics_not_exact_for_annual_requirement" in decision[
        "quality_gate_blocking_flags"
    ]


def test_direct_collection_official_ambiguous_metric_column_routes_to_pending_review():
    record = _record(
        "rec_ny_ambiguous_outbreak_metric",
        source_id="src_ny_flu_press",
        disease="Influenza",
        disease_standard_name="Influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza virus",
        country="United States of America",
        subnational_location="New York",
        geographic_scope="New York",
        geographic_scope_type="state",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="week ending October 5, 2024",
        cases_unspecified=None,
        metric_name="Lab-confirmed influenza outbreaks in hospitals and nursing homes",
        metric_value=3.0,
        metric_unit="count",
        metric_category="outbreak_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="llm_extracted",
        source_column_label="Column 1",
        metric_column_label="Column 1",
        metric_column_semantics_status="ambiguous",
        resolved_column_period_type="ambiguous_column",
        count_semantics="weekly public-health metric",
        statistical_count_type="current_period",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote="Lab-confirmed influenza outbreaks in hospitals and nursing homes: 3.",
        source_url="https://www.health.ny.gov/press/releases/2024/2024-11-01_flu.htm",
        source_title="State Health Department Encourages New Yorkers to Get Flu Vaccine",
        publisher="New York State Department of Health",
        actual_publisher="New York State Department of Health",
        source_type="state_or_local_public_health_agency",
        source_type_final="state_or_local_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.95,
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "New York",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "New York",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            source_registry=[
                _source(
                    "src_ny_flu_press",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="state_or_local_public_health_agency",
                    source_type_final="state_or_local_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.95,
                )
            ],
        )
    )

    assert package["final_dataset"] == []
    assert package["quarantined_records"] == []
    assert _record_ids(package["pending_review_records"]) == {
        "rec_ny_ambiguous_outbreak_metric"
    }
    decision = result["record_inclusion_decisions"][0]
    assert "ambiguous_metric_column_semantics_requires_human_review" in decision[
        "quality_gate_blocking_flags"
    ]


def test_weekly_contract_full_calendar_year_does_not_require_exact_annual_period():
    record = _record(
        "rec_flu_weekly_current",
        source_id="src_cdc_flu_week_40",
        disease="Influenza",
        disease_standard_name="Influenza",
        virus_or_syndrome="seasonal influenza",
        pathogen_or_syndrome="influenza virus",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-10-05",
        date_anchor="2024-10-05",
        reporting_period="MMWR week 40, 2024",
        cases_unspecified=None,
        metric_name="Clinical lab positive specimens",
        metric_value=380.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        metric_period_start="2024-09-29",
        metric_period_end="2024-10-05",
        metric_period_source="filled_from_source_reporting_period",
        count_semantics="current weekly laboratory positive specimens",
        statistical_count_type="current_period",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        evidence_quote=(
            "During influenza Week 40, clinical laboratories reported 380 "
            "positive specimens."
        ),
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_title="CDC FluView Week 40",
        publisher="Centers for Disease Control and Prevention",
        actual_publisher="Centers for Disease Control and Prevention",
        source_type="national_public_health_agency",
        source_type_final="national_public_health_agency",
        source_role_final="collection",
        credibility_level="high",
        credibility_score=0.98,
        coverage_requirement_ids=["united_states_influenza_official_week_40_2024"],
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
    )

    package, result = _finalize(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "collection_mode": "direct_collection",
            },
            task_evidence_contract={
                "time_granularity": "weekly",
                "requirements": [
                    {
                        "requirement_id": "united_states_influenza_official_week_40_2024",
                        "period_basis": "week_ending_saturday",
                        "period_start": "2024-09-29",
                        "period_end": "2024-10-05",
                    }
                ],
            },
            source_coverage_requirements=[
                {
                    "requirement_id": "united_states_influenza_official_week_40_2024",
                    "period_basis": "week_ending_saturday",
                    "period_start": "2024-09-29",
                    "period_end": "2024-10-05",
                    "reporting_period_start": "2024-09-29",
                    "reporting_period_end": "2024-10-05",
                }
            ],
            source_registry=[
                _source(
                    "src_cdc_flu_week_40",
                    canonical_url=record["source_url"],
                    title=record["source_title"],
                    publisher=record["publisher"],
                    actual_publisher=record["actual_publisher"],
                    source_type="national_public_health_agency",
                    source_type_final="national_public_health_agency",
                    source_role_final="collection",
                    credibility_level="high",
                    credibility_score=0.98,
                    coverage_requirement_ids=[
                        "united_states_influenza_official_week_40_2024"
                    ],
                )
            ],
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_flu_weekly_current"}
    decision = result["record_inclusion_decisions"][0]
    assert "record_period_partial_overlap_for_annual_requirement" not in decision[
        "quality_gate_blocking_flags"
    ]


def test_disease_mismatch_record_is_quarantined():
    bad = _record(
        "rec_covid_as_hanta",
        disease="COVID-19",
        disease_standard_name="COVID-19",
        virus_or_syndrome="SARS-CoV-2",
        pathogen_or_syndrome="SARS-CoV-2",
        evidence_quote="Shanghai reported COVID-19 cases and SARS-CoV-2 deaths.",
        record_disease_compatibility_status="incompatible_disease",
        record_disease_compatibility_reject=True,
    )

    package, _ = _finalize(_state([bad]))

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_covid_as_hanta"}
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] == "quarantined_disease_mismatch"
    assert "disease_pathogen_incompatible_with_task" in quarantined["quality_gate_blocking_flags"]


def test_quality_gate_never_overrides_local_numeric_disease_mismatch_with_bound_source():
    record = _record(
        "rec_bound_generic_count",
        source_id="src_bound_generic_count",
        source_url="https://www.nmhealth.org/publication/view/report/weekly",
        source_title="Weekly public health report",
        evidence_quote="A total of 12 confirmed cases and 3 deaths were reported.",
        cases_unspecified=12.0,
        deaths=3.0,
        metric_name="Reported cases",
        metric_value=12.0,
        metric_unit="count",
        metric_category="case_count",
        count_semantics="current reporting-period case count",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=False,
        chunk_kind="metric_row",
        metric_row_binding_status="resolved",
        coverage_requirement_ids=["req_hantavirus_weekly"],
    )
    state = _state(
        [record],
        structured_task={
            "disease": "hantavirus",
            "location": "New Mexico",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "New Mexico",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        source_registry=[
            _source(
                "src_bound_generic_count",
                title=record["source_title"],
                canonical_url=record["source_url"],
                source_type_final="state_or_local_public_health_agency",
                must_fetch=True,
                coverage_requirement_ids=["req_hantavirus_weekly"],
            )
        ],
    )

    package, _ = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {
        "rec_bound_generic_count"
    }
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_local_numeric_disease_status"] == (
        "incompatible_disease"
    )
    assert "local_evidence_disease_mismatch" in quarantined[
        "quality_gate_blocking_flags"
    ]


def test_source_critic_blocked_record_is_quarantined():
    record = _record("rec_blocked")
    state = _state(
        [record],
        source_registry=[
            _source(
                "src_rec_blocked",
                blocked_from_fetch=True,
                blocked_from_fetch_reason="source critic marked not task relevant",
                llm_source_critic_block_fetch=True,
            )
        ],
    )

    package, _ = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] == "quarantined_source_not_task_relevant"
    assert any("source" in reason for reason in quarantined["quality_gate_reasons"])


def test_document_or_chunk_not_task_relevant_record_is_quarantined():
    record = _record("rec_bad_chunk")
    state = _state(
        [record],
        documents=[
            _document(
                "src_rec_bad_chunk",
                quality_status="not_task_relevant",
                document_disease_relevance_status="unrelated_disease",
                not_extractable_for_task_disease=True,
            )
        ],
        evidence_chunks=[
            _chunk(
                "chunk_rec_bad_chunk",
                "src_rec_bad_chunk",
                contains_target_data=False,
                disease_relevance_status="unrelated_disease",
                extraction_eligible_for_task_disease=False,
            )
        ],
    )

    package, _ = _finalize(state)

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] in {
        "quarantined_document_not_task_relevant",
        "quarantined_chunk_not_task_relevant",
    }
    assert any("task relevant" in reason for reason in quarantined["quality_gate_reasons"])


def test_outside_scope_validation_result_blocks_accepted_final_dataset():
    record = _record("rec_outside")
    validation = _validation_result(
        "rec_outside",
        match_status="outside_requested_scope",
        validation_status="outside_scope",
        reason="record is outside_time_window for requested task",
        warnings=["outside_time_window"],
    )

    package, _ = _finalize(_state([record], validation_results=[validation]))

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_outside"}
    assert package["quarantined_records"][0]["record_final_inclusion_status"] == "quarantined_outside_scope"


def test_trusted_source_validation_conflict_quarantines_record():
    record = _record("rec_conflict")
    validation = _validation_result(
        "rec_conflict",
        validation_type="trusted_source_comparison",
        match_status="conflict",
        validation_status="conflict",
        reason="collection and held-out validation disagree for cases_unspecified",
        warnings=["trusted_source_conflict"],
    )

    package, _ = _finalize(_state([record], validation_results=[validation]))

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_conflict"}
    assert package["quarantined_records"][0]["record_final_inclusion_status"] == "quarantined_validation_conflict"


def test_no_compatible_validation_source_warns_but_does_not_quarantine_clean_record():
    package, result = _finalize(
        _state(
            validation_source_compatibility_summary={
                "compatibility_status": "no_task_compatible_validation_source",
                "warnings": ["no_task_compatible_validation_source"],
            }
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_001"}
    assert package["quarantined_records"] == []
    assert result["run_quality_summary"]["validation_limited"] is True
    assert result["run_quality_summary"]["no_compatible_validation_source"] is True
    assert "no_task_compatible_validation_source" in result["run_quality_summary"]["warnings"]
    assert result["run_quality_summary"]["run_quality_status"] != "failed_quality_gate"


def test_live_validation_pending_is_reported_as_validation_limited():
    package, result = _finalize(
        _state(
            validation_source_compatibility_summary={
                "validation_mode": "live_cross_source",
                "compatibility_status": "live_validation_pending",
                "active_validation_record_count": 0,
                "compatibility_reason": (
                    "Live validation limited: no compatible validation source was found."
                ),
                "warnings": [],
            }
        )
    )

    assert _record_ids(package["final_dataset"]) == {"rec_001"}
    assert result["run_quality_summary"]["validation_limited"] is True
    assert result["run_quality_summary"]["no_compatible_validation_source"] is True
    assert "validation_limited_no_compatible_source" in result["run_quality_summary"]["warnings"]


def test_exposure_monitoring_is_excluded_and_exported_as_non_primary():
    record = _record(
        "rec_monitoring",
        cases_unspecified=1.0,
        observation_type="exposure_monitoring_record",
        primary_case_dataset_eligible=False,
        evidence_quote="A traveler is in good health and under public health monitoring.",
    )

    package, result = _finalize(_state([record]))

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {"rec_monitoring"}
    assert package["non_primary_observations"][0]["record_final_inclusion_status"] in {
        "quarantined_exposure_monitoring",
        "quarantined_non_primary_observation",
    }
    assert result["run_quality_summary"]["non_primary_observation_count"] == 1
    assert "confirmed" not in str(
        package["non_primary_observations"][0].get("record_final_inclusion_status")
    )


def test_exposure_monitoring_language_overrides_mislabeled_suspected_case():
    record = _record(
        "rec_vdh_monitoring",
        cases_unspecified=None,
        observation_type="suspected_case_record",
        primary_case_dataset_eligible=True,
        publisher="Virginia Department of Health",
        source_title="Hantavirus - Statement from the Virginia Department of Health",
        evidence_quote=(
            "To date, one Virginia traveler disembarked the ship and has returned "
            "home. This person is currently in good health and is under public "
            "health monitoring. A small number of other potentially exposed "
            "Virginians might be identified in the days ahead."
        ),
    )

    package, result = _finalize(_state([record]))

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {"rec_vdh_monitoring"}
    non_primary = package["non_primary_observations"][0]
    assert non_primary["record_final_inclusion_status"] == "quarantined_exposure_monitoring"
    assert "exposure_monitoring_language" in non_primary["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["non_primary_observation_count"] == 1
    assert result["run_quality_summary"]["accepted_primary_case_record_count"] == 0


def test_quarantined_or_monitored_people_are_not_accepted_as_cases():
    record = _record(
        "rec_quarantined_people",
        cases_unspecified=3.0,
        observation_type="unspecified_case_record",
        primary_case_dataset_eligible=True,
        case_definition="3 New Yorkers among 18 quarantined",
        count_semantics="people quarantined after possible hantavirus exposure",
        source_title="3 New Yorkers among 18 quarantined after deadly hantavirus outbreak",
        evidence_quote=(
            "Three New Yorkers were among 18 passengers quarantined and monitored "
            "after possible exposure; no confirmed New York cases were reported."
        ),
    )

    package, result = _finalize(_state([record]))

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {
        "rec_quarantined_people"
    }
    non_primary = package["non_primary_observations"][0]
    assert non_primary["record_final_inclusion_status"] == "quarantined_exposure_monitoring"
    assert "exposure_monitoring_language" in non_primary["quality_gate_blocking_flags"]
    assert result["run_quality_summary"]["accepted_primary_case_record_count"] == 0


def test_positive_primary_case_claim_remains_accepted_when_gates_pass():
    record = _record(
        "rec_primary",
        cases_confirmed=1.0,
        cases_unspecified=None,
        observation_type="confirmed_case_record",
        primary_case_dataset_eligible=True,
    )

    package, result = _finalize(_state([record]))

    assert _record_ids(package["final_dataset"]) == {"rec_primary"}
    assert package["non_primary_observations"] == []
    assert result["run_quality_summary"]["accepted_primary_case_record_count"] == 1
    assert result["run_quality_summary"]["primary_case_dataset_status"] == "primary_case_records_present"
    assert result["run_quality_summary"]["run_quality_status"] in {
        "passed",
        "passed_with_review",
    }


def test_official_single_source_review_candidate_is_not_final_accepted():
    record = _record(
        "rec_official_single_source",
        cases_confirmed=None,
        cases_unspecified=8.0,
        deaths=3.0,
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        record_schema="official_outbreak_page_extraction_schema",
        extraction_method="deterministic_official_outbreak_page_extractor",
        requires_human_review=True,
        human_review_reason="official_single_source_not_cross_validated",
        quality_status="official_single_source_not_cross_validated",
        evidence_quote="A total of eight cases, including three deaths, have been reported.",
    )

    package, result = _finalize(_state([record], human_review_enabled=False))

    assert package["final_dataset"] == []
    assert _record_ids(package["pending_review_records"]) == {
        "rec_official_single_source"
    }
    pending = package["pending_review_records"][0]
    assert pending["final_dataset_included"] is False
    assert pending["record_final_inclusion_status"] == "pending_human_review"
    assert "official_single_source_not_cross_validated" in pending[
        "quality_gate_blocking_flags"
    ]
    assert result["run_quality_summary"]["pending_review_record_count"] == 1


def test_missing_primary_case_dataset_eligibility_preserves_legacy_compatibility():
    record = _record("rec_legacy")
    record.pop("primary_case_dataset_eligible", None)
    record.pop("observation_type", None)

    package, result = _finalize(_state([record]))

    assert _record_ids(package["final_dataset"]) == {"rec_legacy"}
    assert package["non_primary_observations"] == []
    assert result["run_quality_summary"]["run_quality_status"] in {
        "passed",
        "passed_with_review",
    }


def test_all_non_primary_records_yield_empty_final_dataset_and_post_review():
    records = [
        _record(
            "rec_ambiguous",
            cases_unspecified=None,
            deaths=None,
            observation_type="ambiguous_public_health_observation",
            primary_case_dataset_eligible=False,
        ),
        _record(
            "rec_surveillance",
            source_id="src_untrusted_surveillance",
            source_url="https://example-blog.test/flu-surveillance-summary",
            publisher="Example Blog",
            source_type="news_media",
            credibility_level="low",
            cases_unspecified=12.0,
            observation_type="surveillance_summary",
            primary_case_dataset_eligible=False,
        ),
    ]

    state = _state(
        records,
        source_registry=[
            _source("src_rec_ambiguous"),
            _source(
                "src_untrusted_surveillance",
                canonical_url="https://example-blog.test/flu-surveillance-summary",
                publisher="Example Blog",
                source_type="news_media",
                source_type_final="news_media",
                credibility_level="low",
                credibility_score=0.2,
            ),
        ],
    )
    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["final_dataset_post_review"] == []
    assert _record_ids(package["non_primary_observations"]) == {
        "rec_ambiguous",
        "rec_surveillance",
    }
    assert result["run_quality_summary"]["non_primary_observation_count"] == 2
    assert result["run_quality_summary"]["run_quality_status"] in {
        "no_primary_case_dataset_records",
        "failed_quality_gate",
    }


def test_finalization_enriches_stale_records_from_claim_outputs_before_quality_gate():
    record = _record(
        "rec_stale",
        cases_unspecified=1.0,
        evidence_quote="A traveler remained healthy while under public health monitoring.",
    )
    claim = {
        "claim_id": "claim_rec_stale_cases_unspecified",
        "source_record_id": "rec_stale",
        "observation_type": "exposure_monitoring_record",
        "primary_case_dataset_eligible": False,
        "source_id": record["source_id"],
        "source_url": record["source_url"],
        "supporting_chunk_id": record["supporting_chunk_id"],
    }
    event = {
        "corroborated_event_id": "corr_event_stale",
        "supporting_claim_ids": ["claim_rec_stale_cases_unspecified"],
        "unverified_claim_ids": ["claim_rec_stale_cases_unspecified"],
        "corroboration_status": "exposure_monitoring_only",
        "independent_source_count": 1,
        "official_source_support_count": 1,
        "secondary_source_support_count": 0,
        "primary_case_dataset_eligible": False,
        "corroboration_reason": "Exposure monitoring only; not a primary case record.",
        "warnings": ["not_primary_case_record"],
    }

    package, result = _finalize(
        _state(
            [record],
            claims=[claim],
            corroborated_events=[event],
            corroboration_summary={
                "claim_count": 1,
                "corroborated_event_count": 1,
                "corroborated_primary_case_event_count": 0,
            },
        )
    )

    assert package["final_dataset"] == []
    assert _record_ids(package["non_primary_observations"]) == {"rec_stale"}
    decision = result["record_inclusion_decisions"][0]
    assert "primary_case_dataset_eligible_false" in decision["quality_gate_blocking_flags"]
    assert "claim_observation_type_exposure_monitoring_record" in decision["quality_gate_blocking_flags"]


def test_high_or_critical_anomaly_quarantines_record():
    record = _record("rec_anom", cases_unspecified=1, deaths=5)

    package, _ = _finalize(_state([record], anomaly_results=[_anomaly("rec_anom")]))

    assert package["final_dataset"] == []
    quarantined = package["quarantined_records"][0]
    assert quarantined["record_final_inclusion_status"] == "quarantined_critical_anomaly"
    assert "deaths_greater_than_cases" in quarantined["quality_gate_blocking_flags"]


def test_human_review_reject_excludes_post_review_dataset_and_preserves_audit():
    state = _state(
        human_review_decisions=[
            {
                "decision_id": "decision_reject_001",
                "review_id": "review_001",
                "decision_type": "reject_record",
                "reviewer_id": "reviewer_test_public_health_001",
                "decided_at": "2026-06-01T00:00:00Z",
                "target_type": "record",
                "target_ids": ["rec_001"],
                "reason": "erroneous record",
                "apply_decision": True,
            }
        ]
    )
    state.update(apply_human_review_decisions(state))

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert package["final_dataset_post_review"] == []
    assert _record_ids(package["records_excluded_by_human_review"]) == {"rec_001"}
    assert result["human_review_audit_trail"]


def test_post_review_dataset_matches_quality_gated_final_dataset_without_decisions():
    clean = _record("rec_clean")
    bad = _record(
        "rec_bad",
        disease="COVID-19",
        virus_or_syndrome="SARS-CoV-2",
        evidence_quote="COVID-19 evidence",
        record_disease_compatibility_reject=True,
    )

    package, _ = _finalize(_state([clean, bad]))

    assert _record_ids(package["final_dataset"]) == {"rec_clean"}
    assert _record_ids(package["final_dataset_post_review"]) == {"rec_clean"}
    assert _record_ids(package["final_dataset_pre_quality_gate"]) == {"rec_clean", "rec_bad"}


def test_shanghai_failure_style_wrong_disease_records_do_not_look_successful():
    bad = _record(
        "rec_shanghai_covid",
        disease="COVID-19",
        disease_standard_name="COVID-19",
        virus_or_syndrome="SARS-CoV-2",
        pathogen_or_syndrome="SARS-CoV-2",
        country="China",
        subnational_location="Shanghai",
        date_reported="2024-04-01",
        reporting_period="2024",
        query_used="hantavirus Shanghai 2024",
        evidence_quote="Shanghai reported COVID-19 and SARS-CoV-2 case counts.",
    )
    state = _state(
        [bad],
        structured_task={
            "disease": "hantavirus",
            "location": "Shanghai",
            "start_date": "2024",
            "end_date": "2026",
        },
        collection_spec={
            "disease": "hantavirus",
            "geography": "Shanghai",
            "time_window": "2024 to 2026",
        },
        validation_source_compatibility_summary={
            "compatibility_status": "incompatible_validation_source_disabled",
            "warnings": ["no_task_compatible_validation_source"],
        },
    )

    package, result = _finalize(state)

    assert package["final_dataset"] == []
    assert _record_ids(package["quarantined_records"]) == {"rec_shanghai_covid"}
    assert result["run_quality_summary"]["run_quality_status"] in {
        "failed_quality_gate",
        "no_task_relevant_records",
    }
    assert all(
        "SARS-CoV-2" not in json.dumps(record, ensure_ascii=False)
        for record in package["final_dataset"]
    )


def test_no_records_extracted_gets_explicit_run_quality_status():
    package, result = _finalize(
        _state(
            [],
            disease_relevance_summary={
                "target_data_chunk_count": 0,
                "chunk_status_counts": {"unrelated_disease": 2},
            },
        )
    )

    assert package["final_dataset"] == []
    assert result["run_quality_summary"]["run_quality_status"] in {
        "no_records_extracted",
        "no_task_relevant_records",
    }
    assert result["run_quality_summary"]["recommended_user_message"]


def test_final_package_and_export_include_all_quality_dataset_views(tmp_path):
    clean = _record("rec_clean")
    bad = _record(
        "rec_bad",
        disease="COVID-19",
        virus_or_syndrome="SARS-CoV-2",
        evidence_quote="COVID-19 evidence",
        record_disease_compatibility_reject=True,
    )
    package, _ = _finalize(_state([clean, bad]))

    for key in (
        "final_dataset",
        "final_dataset_pre_quality_gate",
        "quarantined_records",
        "pending_review_records",
        "non_primary_observations",
        "record_inclusion_decisions",
        "run_quality_summary",
        "final_dataset_quality_summary",
    ):
        assert key in package
    assert "run_quality_summary" in package["workflow_summaries"]
    assert "final_dataset_quality_summary" in package["workflow_summaries"]
    assert {chunk["chunk_id"] for chunk in package["evidence_chunks"]} == {
        "chunk_rec_bad",
        "chunk_rec_clean",
    }

    manifest = export_final_data_package(package, tmp_path)
    expected_files = {
        "final_dataset_json": "final_dataset.json",
        "final_dataset_pre_quality_gate_json": "final_dataset_pre_quality_gate.json",
        "final_dataset_pre_quality_gate_csv": "final_dataset_pre_quality_gate.csv",
        "quarantined_records_json": "quarantined_records.json",
        "quarantined_records_csv": "quarantined_records.csv",
        "pending_review_records_json": "pending_review_records.json",
        "pending_review_records_csv": "pending_review_records.csv",
        "non_primary_observations_json": "non_primary_observations.json",
        "non_primary_observations_csv": "non_primary_observations.csv",
        "record_inclusion_decisions_json": "record_inclusion_decisions.json",
        "evidence_chunks_json": "evidence_chunks.json",
    }
    for manifest_key, filename in expected_files.items():
        assert manifest_key in manifest["files"]
        assert (tmp_path / filename).exists()

    with (tmp_path / "evidence_chunks.json").open(encoding="utf-8") as f:
        exported_chunks = json.load(f)
    assert {chunk["chunk_id"] for chunk in exported_chunks} == {
        "chunk_rec_bad",
        "chunk_rec_clean",
    }

    with (tmp_path / "final_dataset.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["record_id"] for row in rows] == ["rec_clean"]


def test_clean_covid_dengue_and_hantavirus_records_remain_accepted():
    cases = [
        (
            "COVID-19",
            "New York",
            _record(
                "rec_covid",
                disease="COVID-19",
                disease_standard_name="COVID-19",
                virus_or_syndrome="SARS-CoV-2",
                pathogen_or_syndrome="SARS-CoV-2",
                subnational_location="New York",
                evidence_quote="New York reported COVID-19 cases in 2024.",
            ),
        ),
        (
            "dengue",
            "Florida",
            _record(
                "rec_dengue",
                disease="Dengue",
                disease_standard_name="Dengue",
                virus_or_syndrome="DENV",
                pathogen_or_syndrome="dengue virus",
                subnational_location="Florida",
                evidence_quote="Florida reported dengue cases in 2025.",
            ),
        ),
        ("hantavirus", "New Mexico", _record("rec_hanta")),
    ]

    for disease, location, record in cases:
        package, _ = _finalize(
            _state(
                [record],
                structured_task={
                    "disease": disease,
                    "location": location,
                    "start_date": "2024",
                    "end_date": "2026",
                },
                collection_spec={
                    "disease": disease,
                    "geography": location,
                    "time_window": "2024 to 2026",
                },
            )
        )
        assert _record_ids(package["final_dataset"]) == {record["record_id"]}
        assert package["quarantined_records"] == []
