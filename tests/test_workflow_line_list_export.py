from __future__ import annotations

import csv
import json

from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.nodes.finalization import (
    _build_case_evidence_bundles,
    _case_event_key,
    final_data_package_builder,
)
from data_collection_workflow.workflow_line_list_export import (
    build_workflow_case_bundle_line_list,
    build_workflow_case_candidate_line_list,
)


def _source(source_id: str, **overrides) -> dict:
    row = {
        "source_id": source_id,
        "canonical_url": f"https://example.org/{source_id}",
        "title": f"Source {source_id}",
        "publisher": "Example Publisher",
        "source_type_final": "news_media",
        "source_product_type": "news_case_report",
        "data_product_type": "news_case_report",
        "authority_bucket": "",
        "status": "ready_for_content_fetch",
    }
    row.update(overrides)
    return row


def _candidate(record_id: str, source_id: str, **overrides) -> dict:
    row = {
        "record_id": record_id,
        "case_candidate_id": f"case_candidate_{record_id}",
        "candidate_type": "individual_case_candidate",
        "case_candidate_status": "reviewable",
        "event_cluster_key": "hantavirus|global|mv_hondius",
        "disease": "Hantavirus disease",
        "country": "France",
        "subnational_location": "",
        "geographic_scope": "France",
        "case_status": "confirmed_case_record",
        "cases_confirmed": 1,
        "cases_probable": None,
        "cases_suspected": None,
        "cases_unspecified": None,
        "deaths": 0,
        "date_reported": "2026-05-01",
        "reporting_period": "2026-04-01 to 2026-06-30",
        "source_id": source_id,
        "source_url": f"https://example.org/{source_id}",
        "source_type_final": "news_media",
        "source_product_type": "news_case_report",
        "expected_evidence_role": "individual_case_evidence",
        "evidence_quote": "A passenger was confirmed with Andes virus disease.",
        "quality_gate_status": "pending_human_review",
        "review_reason": "single source requires review",
    }
    row.update(overrides)
    return row


def test_bundle_line_list_orders_sources_and_preserves_overflow_json():
    sources = [
        _source(
            "src_media",
            canonical_url="https://news.example/case",
            source_type_final="news_media",
            source_product_type="news_case_report",
        ),
        _source(
            "src_database",
            canonical_url="https://pathoplexus.org/pathogen/hantavirus",
            source_type_final="structured_database",
            source_product_type="sequence_database_record",
        ),
        _source(
            "src_literature",
            canonical_url="https://www.nejm.org/doi/full/10.1056/example",
            source_type_final="academic_or_peer_reviewed_source",
            source_product_type="case_report_article",
        ),
        _source(
            "src_official",
            canonical_url="https://www.who.int/example",
            source_type_final="international_public_health_agency",
            source_product_type="event_outbreak_report",
            authority_bucket="official_authority",
        ),
        _source("src_extra_1"),
        _source("src_extra_2"),
        _source("src_extra_3"),
        _source("src_extra_4"),
    ]
    candidates = [
        _candidate("rec_official", "src_official", source_type_final="international_public_health_agency"),
        _candidate("rec_literature", "src_literature", source_type_final="academic_or_peer_reviewed_source"),
        _candidate("rec_database", "src_database", source_type_final="structured_database"),
        _candidate("rec_media", "src_media"),
        _candidate("rec_extra_1", "src_extra_1"),
        _candidate("rec_extra_2", "src_extra_2"),
        _candidate("rec_extra_3", "src_extra_3"),
        _candidate("rec_extra_4", "src_extra_4"),
    ]
    bundles = [
        {
            "case_evidence_bundle_id": "case_bundle_0001",
            "event_cluster_key": "hantavirus|global|mv_hondius",
            "bundle_type": "individual_case_like",
            "case_candidate_ids": [row["case_candidate_id"] for row in candidates],
            "source_ids": [row["source_id"] for row in candidates],
            "source_urls": [row["source_url"] for row in candidates],
            "supporting_source_count": 8,
            "verified_authority_source_count": 1,
            "high_trust_source_count": 3,
            "official_source_count": 1,
            "peer_reviewed_source_count": 1,
            "structured_database_source_count": 1,
            "source_stage_summary": "3 fetched / 3 parsed / 3 extracted",
            "source_gap_status": "jurisdiction_authority_partial",
            "snapshot_compatibility_status": "compatible_dynamic_snapshots",
            "best_evidence_quote": (
                "![](https://example.org/logo.png) The authors wrote, "
                "â€œA passenger was confirmed with Andes virus disease.â€"
            ),
            "main_blocking_reason": "single source requires review",
            "recommended_review_action": "review_individual_case_evidence",
            "candidate_statuses": ["reviewable"],
            "source_product_types": ["event_outbreak_report", "case_report_article"],
        }
    ]

    rows = build_workflow_case_bundle_line_list(
        case_evidence_bundles=bundles,
        case_candidate_dataset=candidates,
        source_inventory=sources,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["workflow_row_id"] == "WFL-0001"
    assert row["bundle_id"] == "case_bundle_0001"
    assert row["source_I_url"] == "https://www.who.int/example"
    assert row["source_II_url"] == "https://www.nejm.org/doi/full/10.1056/example"
    assert row["source_III_url"] == "https://pathoplexus.org/pathogen/hantavirus"
    assert row["source_VII_url"]
    assert "src_extra_4" in json.loads(row["all_source_ids_json"])
    assert "https://example.org/src_extra_4" in json.loads(row["all_source_urls_json"])
    assert row["high_trust_source_count"] == 3
    assert row["official_source_count"] == 1
    assert row["peer_reviewed_source_count"] == 1
    assert row["structured_database_source_count"] == 1
    assert row["source_stage_summary"] == "3 fetched / 3 parsed / 3 extracted"
    assert row["source_gap_status"] == "jurisdiction_authority_partial"
    assert row["snapshot_compatibility_status"] == "compatible_dynamic_snapshots"
    assert row["best_evidence_quote"] == (
        'The authors wrote, "A passenger was confirmed with Andes virus disease."'
    )
    assert "Gh_ID" not in row
    assert row["claim_boundary"] == "reviewable_evidence_not_final_truth"


def test_candidate_line_list_preserves_source_specific_review_fields():
    source = _source(
        "src_official",
        canonical_url="https://www.who.int/example",
        title="WHO outbreak report",
        source_type_final="international_public_health_agency",
        data_product_type="event_outbreak_report",
        authority_bucket="official_authority",
        source_exact_page_status="event_page_found",
    )
    candidate = _candidate(
        "rec_case_1",
        "src_official",
        case_candidate_status="quarantined",
        quality_gate_status="quarantined_non_primary_observation",
        review_reason="tested negative contact is not a case",
        quarantine_reason="non_case_or_monitoring_evidence",
        candidate_type="non_case_or_monitoring_candidate",
        evidence_quote="The contact tested negative and was monitored.",
        symptoms="fever; pneumonia",
        date_onset="2026-04-28",
        date_confirmation="2026-05-01",
        date_death="2026-05-02",
        occupation_or_role="passenger",
        hospitalized=True,
        intensive_care=False,
        isolated=True,
        cruise_crew=False,
        cruise_passenger_guest=True,
        contact_with_case=True,
        contact_setting="cruise ship",
        ship_board_date="2026-04-01",
        ship_disembark_date="2026-05-06",
        travel_from="South America; Argentina",
        travel_to="United Kingdom",
        confirmation_method="PCR",
        accession_id="PP_123",
        source_gap_status="structured_database_missing",
        case_identity_fingerprint="hantavirus|mv_hondius|case_4|female",
        field_provenance_json='{"gender":{"case_span_id":"case_span_001_case_4"}}',
        unsupported_case_fields=["age"],
        focused_recovery_status="focused_retry_succeeded",
        record_local_disease_relevance_status="target_disease_match",
        record_local_numeric_disease_status="target_disease_match",
        record_local_numeric_target_terms_found=["Andes virus"],
        record_local_numeric_incompatible_terms_found=[],
        case_span_id="case_span_001_case_4",
        case_span_quote="Case 4: An adult female died on 2 May 2026.",
        case_span_start=10,
        case_span_end=62,
        case_span_extraction_method="deterministic_case_label_span",
    )

    rows = build_workflow_case_candidate_line_list(
        case_candidate_dataset=[candidate],
        case_evidence_bundles=[],
        source_inventory=[source],
    )

    assert rows == [
            {
                "workflow_candidate_id": "WFC-0001",
                "candidate_id": "case_candidate_rec_case_1",
                "bundle_id": "",
                "event_family_id": "",
                "bundle_scope": "",
                "record_id": "rec_case_1",
            "source_id": "src_official",
            "source_url": "https://www.who.int/example",
            "source_name": "WHO outbreak report",
            "source_type": "international_public_health_agency",
            "source_product_type": "event_outbreak_report",
            "evidence_role": "individual_case_evidence",
            "candidate_type": "non_case_or_monitoring_candidate",
            "observation_type": "confirmed_case_record",
                "case_status": "confirmed_case_record",
                "workflow_case_label": "",
                "case_label_normalized": "",
                "location": "France",
                "date_or_period": "2026-05-01",
                "snapshot_as_of_date": "",
                "case_count": 1,
            "death_count": 0,
            "age": "",
            "gender": "",
            "nationality": "",
            "outcome": "",
            "symptoms": "fever; pneumonia",
            "date_onset": "2026-04-28",
            "date_confirmation": "2026-05-01",
            "date_death": "2026-05-02",
            "occupation_or_role": "passenger",
            "hospitalized": True,
            "intensive_care": False,
            "isolated": True,
            "cruise_crew": False,
            "cruise_passenger_guest": True,
            "contact_with_case": True,
            "contact_setting": "cruise ship",
            "ship_board_date": "2026-04-01",
            "ship_disembark_date": "2026-05-06",
            "travel_from": "South America; Argentina",
            "travel_to": "United Kingdom",
            "travel_or_vessel_context": "",
            "confirmation_method": "PCR",
            "accession_id": "PP_123",
            "source_gap_status": "structured_database_missing",
            "high_confidence_domain_status_json": "",
            "source_exact_page_status": "event_page_found",
            "source_to_evidence_status": "",
            "source_only_reason": "",
            "extraction_failure_substage": "",
            "case_identity_fingerprint": "hantavirus|mv_hondius|case_4|female",
            "case_entity_id": "",
            "link_status": "",
            "link_score": "",
            "link_reasons": "",
            "field_conflicts_json": "",
            "field_provenance_json": '{"gender":{"case_span_id":"case_span_001_case_4"}}',
            "unsupported_case_fields": '["age"]',
            "focused_recovery_status": "focused_retry_succeeded",
                "local_disease_relevance_status": "target_disease_match",
                "record_local_numeric_disease_status": "target_disease_match",
                "record_local_numeric_sentence": "",
                "record_local_numeric_disease_decision": "",
                "record_local_numeric_target_terms_found": '["Andes virus"]',
                "record_local_numeric_incompatible_terms_found": "[]",
            "case_span_id": "case_span_001_case_4",
            "case_span_quote": "Case 4: An adult female died on 2 May 2026.",
            "case_span_start": 10,
            "case_span_end": 62,
            "case_span_extraction_method": "deterministic_case_label_span",
            "field_completeness_score": "0.54",
            "missing_key_fields": (
                "workflow_case_label; age; gender; nationality; outcome; "
                "travel_or_vessel_context"
            ),
            "confidence": "quarantined_non_primary_observation",
            "quality_status": "quarantined_non_primary_observation",
            "review_reason": "tested negative contact is not a case",
            "quarantine_reason": "non_case_or_monitoring_evidence",
            "evidence_quote": "The contact tested negative and was monitored.",
            "is_verified_authority_source": True,
            "human_review_required": True,
        }
    ]
    assert "Gh_ID" not in rows[0]


def test_candidate_line_list_reports_field_completeness_for_sparse_candidates():
    source = _source(
        "src_case_report",
        canonical_url="https://www.nejm.org/doi/full/10.1056/example",
        source_type_final="academic_or_peer_reviewed_source",
        data_product_type="case_report_article",
    )
    candidate = _candidate(
        "rec_sparse",
        "src_case_report",
        workflow_case_label="Case 4",
        age="adult",
        gender="female",
        outcome="death",
        date_onset="2026-04-28",
        record_local_disease_relevance_status="target_disease_match",
    )

    rows = build_workflow_case_candidate_line_list(
        case_candidate_dataset=[candidate],
        case_evidence_bundles=[],
        source_inventory=[source],
    )

    assert rows[0]["field_completeness_score"] != ""
    assert float(rows[0]["field_completeness_score"]) > 0.3
    assert "nationality" in rows[0]["missing_key_fields"]
    assert "workflow_case_label" not in rows[0]["missing_key_fields"]


def test_bundle_line_list_reports_field_completeness_and_missing_key_fields():
    source = _source(
        "src_official",
        canonical_url="https://www.who.int/example",
        source_type_final="international_public_health_agency",
        data_product_type="event_outbreak_report",
        authority_bucket="official_authority",
    )
    candidate = _candidate(
        "rec_case_4",
        "src_official",
        case_candidate_status="reviewable",
        workflow_case_label="Case 4",
        age="adult",
        gender="female",
        outcome="death",
        symptoms="pneumonia; fever",
        date_onset="2026-04-28",
        date_death="2026-05-02",
        source_gap_status="jurisdiction_authority_partial",
        record_local_disease_relevance_status="target_disease_match",
    )
    bundle = {
        "case_evidence_bundle_id": "case_bundle_0001",
        "event_cluster_key": candidate["event_cluster_key"],
        "bundle_type": "individual_case_like",
        "case_candidate_ids": [candidate["case_candidate_id"]],
        "source_ids": ["src_official"],
        "candidate_statuses": ["reviewable"],
        "source_gap_status": "jurisdiction_authority_partial",
        "high_confidence_domain_status_json": '{"canada.ca":"executed_no_result"}',
        "source_to_evidence_status_json": '{"src_official":"evidence_extracted"}',
        "non_target_disease_block_count": 0,
        "main_blocking_reason": candidate["review_reason"],
        "best_evidence_quote": candidate["evidence_quote"],
        "recommended_review_action": "review_individual_case_evidence",
    }

    rows = build_workflow_case_bundle_line_list(
        case_evidence_bundles=[bundle],
        case_candidate_dataset=[candidate],
        source_inventory=[source],
    )

    row = rows[0]
    assert row["workflow_case_label"] == "Case 4"
    assert row["age"] == "adult"
    assert row["gender"] == "female"
    assert row["outcome"] == "death"
    assert row["date_onset"] == "2026-04-28"
    assert row["source_gap_status"] == "jurisdiction_authority_partial"
    assert row["high_confidence_domain_status_json"] == (
        '{"canada.ca":"executed_no_result"}'
    )
    assert row["source_to_evidence_status_json"] == (
        '{"src_official":"evidence_extracted"}'
    )
    assert row["non_target_disease_block_count"] == 0
    assert row["local_disease_relevance_status"] == "target_disease_match"
    assert row["line_list_detail_level"] == "individual_detail"
    assert float(row["field_completeness_score"]) > 0.5
    assert "nationality" in row["missing_key_fields"]


def test_bundle_line_list_merges_complementary_case_fields_with_provenance():
    sources = [
        _source(
            "src_official_a",
            canonical_url="https://www.who.int/case-5",
            source_type_final="international_public_health_agency",
            data_product_type="event_outbreak_report",
            authority_bucket="official_authority",
        ),
        _source(
            "src_official_b",
            canonical_url="https://www.rivm.nl/case-5",
            source_type_final="national_public_health_agency",
            data_product_type="event_outbreak_report",
            authority_bucket="official_authority",
        ),
    ]
    candidates = [
        _candidate(
            "rec_case_5_a",
            "src_official_a",
            workflow_case_label="Case 5",
            case_label_normalized="case_5",
            age="41",
            gender="male",
            nationality="Dutch",
            symptoms="fever; fatigue; myalgia",
            evidence_quote="Case 5 was a 41-year-old Dutch male with fever and fatigue.",
            field_provenance_json=json.dumps(
                {
                    "age": {"source_id": "src_official_a", "case_span_id": "span_a"},
                    "nationality": {
                        "source_id": "src_official_a",
                        "case_span_id": "span_a",
                    },
                }
            ),
        ),
        _candidate(
            "rec_case_5_b",
            "src_official_b",
            workflow_case_label="Case 5",
            case_label_normalized="case_5",
            date_onset="2026-04-30",
            date_confirmation="2026-05-07",
            hospitalized=True,
            occupation_or_role="ship doctor",
            confirmation_method="PCR",
            evidence_quote="The ship doctor developed symptoms on 30 April and was confirmed by PCR on 7 May 2026.",
            field_provenance_json=json.dumps(
                {
                    "date_onset": {
                        "source_id": "src_official_b",
                        "case_span_id": "span_b",
                    },
                    "date_confirmation": {
                        "source_id": "src_official_b",
                        "case_span_id": "span_b",
                    },
                }
            ),
        ),
    ]
    bundle = {
        "case_evidence_bundle_id": "case_bundle_0005",
        "event_cluster_key": candidates[0]["event_cluster_key"],
        "bundle_type": "individual_case_like",
        "case_label_normalized": "case_5",
        "case_candidate_ids": [row["case_candidate_id"] for row in candidates],
        "source_ids": [row["source_id"] for row in candidates],
        "candidate_statuses": ["reviewable"],
        "best_evidence_quote": candidates[0]["evidence_quote"],
    }

    row = build_workflow_case_bundle_line_list(
        case_evidence_bundles=[bundle],
        case_candidate_dataset=candidates,
        source_inventory=sources,
    )[0]

    assert row["age"] == "41"
    assert row["nationality"] == "Dutch"
    assert row["symptoms"] == "fever; fatigue; myalgia"
    assert row["date_onset"] == "2026-04-30"
    assert row["date_confirmation"] == "2026-05-07"
    assert row["hospitalized"] is True
    assert row["occupation_or_role"] == "ship doctor"
    assert row["confirmation_method"] == "PCR"
    provenance = json.loads(row["field_source_ids_json"])
    assert provenance["age"] == ["src_official_a"]
    assert provenance["date_confirmation"] == ["src_official_b"]
    field_provenance = json.loads(row["field_provenance_json"])
    assert field_provenance["age"]["source_id"] == "src_official_a"
    assert field_provenance["date_confirmation"]["source_id"] == "src_official_b"
    assert json.loads(row["field_conflicts_json"]) == {}


def test_bundle_line_list_leaves_conflicting_case_fields_unresolved():
    source = _source("src_official")
    candidates = [
        _candidate(
            "rec_case_a",
            "src_official",
            workflow_case_label="Case 4",
            case_label_normalized="case_4",
            gender="female",
        ),
        _candidate(
            "rec_case_b",
            "src_official",
            workflow_case_label="Case 4",
            case_label_normalized="case_4",
            gender="male",
        ),
    ]
    bundle = {
        "case_evidence_bundle_id": "case_bundle_conflict",
        "event_cluster_key": candidates[0]["event_cluster_key"],
        "bundle_type": "individual_case_like",
        "case_candidate_ids": [row["case_candidate_id"] for row in candidates],
        "source_ids": ["src_official"],
        "candidate_statuses": ["reviewable"],
    }

    row = build_workflow_case_bundle_line_list(
        case_evidence_bundles=[bundle],
        case_candidate_dataset=candidates,
        source_inventory=[source],
    )[0]

    assert row["gender"] == ""
    assert json.loads(row["field_conflicts_json"])["gender"] == ["female", "male"]


def test_case_evidence_bundle_preserves_merged_field_audit_metadata():
    candidates = [
        _candidate(
            "rec_case_5_a",
            "src_official_a",
            bundle_scope="individual_case",
            workflow_case_label="Case 5",
            case_label_normalized="case_5",
            age="41",
            nationality="Dutch",
            evidence_quote="Case 5 was a 41-year-old Dutch national.",
            field_provenance_json=json.dumps(
                {
                    "age": {"source_id": "src_official_a", "case_span_id": "span_a"},
                    "nationality": {
                        "source_id": "src_official_a",
                        "case_span_id": "span_a",
                    },
                }
            ),
        ),
        _candidate(
            "rec_case_5_b",
            "src_official_b",
            bundle_scope="individual_case",
            workflow_case_label="Case 5",
            case_label_normalized="case_5",
            date_onset="2026-04-30",
            confirmation_method="PCR",
            evidence_quote="Case 5 developed symptoms on 30 April and was confirmed by PCR.",
            field_provenance_json=json.dumps(
                {
                    "date_onset": {
                        "source_id": "src_official_b",
                        "case_span_id": "span_b",
                    }
                }
            ),
        ),
    ]

    bundle = _build_case_evidence_bundles(candidates)[0]

    assert bundle["merged_case_fields"]["age"] == "41"
    assert bundle["merged_case_fields"]["nationality"] == "Dutch"
    assert bundle["merged_case_fields"]["date_onset"] == "2026-04-30"
    assert bundle["merged_case_fields"]["confirmation_method"] == "PCR"
    assert bundle["field_source_ids"]["age"] == ["src_official_a"]
    assert bundle["field_source_ids"]["date_onset"] == ["src_official_b"]
    field_provenance = json.loads(bundle["field_provenance_json"])
    assert field_provenance["age"]["source_id"] == "src_official_a"
    assert field_provenance["date_onset"]["source_id"] == "src_official_b"
    assert bundle["field_conflicts"] == {}
    assert bundle["field_recovery_method"] == "explicit_candidate_field_merge_v1"


def test_case_bundle_links_compatible_unlabelled_cases_but_keeps_labels_separate():
    compatible = [
        _candidate(
            "rec_unlabelled_a",
            "src_official_a",
            bundle_scope="individual_case",
            workflow_case_label="",
            case_label_normalized="",
            age="64",
            gender="male",
            nationality="Swiss",
            occupation_or_role="passenger",
            date_confirmation="2026-05-05",
        ),
        _candidate(
            "rec_unlabelled_b",
            "src_literature_b",
            bundle_scope="individual_case",
            workflow_case_label="",
            case_label_normalized="",
            age="64",
            gender="male",
            nationality="Swiss",
            occupation_or_role="passenger",
            date_confirmation="2026-05-05",
        ),
    ]
    labelled = [
        _candidate(
            "rec_case_3",
            "src_case_3",
            bundle_scope="individual_case",
            workflow_case_label="Case 3",
            case_label_normalized="case_3",
            age="64",
            gender="male",
            nationality="Swiss",
        ),
        _candidate(
            "rec_case_4",
            "src_case_4",
            bundle_scope="individual_case",
            workflow_case_label="Case 4",
            case_label_normalized="case_4",
            age="64",
            gender="male",
            nationality="Swiss",
        ),
    ]

    bundles = _build_case_evidence_bundles([*compatible, *labelled])

    assert len(bundles) == 3
    unlabelled_bundle = next(
        bundle for bundle in bundles if set(bundle["source_ids"]) == {
            "src_official_a",
            "src_literature_b",
        }
    )
    assert unlabelled_bundle["case_identity_fingerprint"]
    labelled_fingerprints = {
        bundle["case_identity_fingerprint"]
        for bundle in bundles
        if bundle is not unlabelled_bundle
    }
    assert len(labelled_fingerprints) == 2


def test_case_bundle_different_explicit_labels_have_distinct_auditable_entities():
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_case_3",
                "src_case_3",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Case 3",
                case_label_normalized="case_3",
            ),
            _candidate(
                "rec_case_4",
                "src_case_4",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Case 4",
                case_label_normalized="case_4",
            ),
        ]
    )

    assert len(bundles) == 2
    assert len({bundle["case_entity_id"] for bundle in bundles}) == 2
    assert {bundle["link_status"] for bundle in bundles} == {
        "explicit_label_singleton"
    }
    assert all(bundle["link_score"] >= 0 for bundle in bundles)
    assert all("explicit_case_label" in bundle["link_reasons"] for bundle in bundles)


def test_grouped_case_remains_one_grouped_entity_without_synthetic_members():
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_cases_1_2",
                "src_grouped",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Cases 1 and 2",
                case_label_normalized="cases_1_and_2",
                cases_confirmed=2,
            )
        ]
    )

    assert len(bundles) == 1
    assert bundles[0]["case_label_normalized"] == "cases_1_and_2"
    assert bundles[0]["member_record_ids"] == ["rec_cases_1_2"]
    assert bundles[0]["link_status"] == "grouped_case_preserved"
    assert "grouped_case_not_split" in bundles[0]["link_reasons"]


def test_aggregate_snapshots_form_compatible_family_across_as_of_dates():
    common = {
        "candidate_type": "aggregate_event_candidate",
        "event_family_id": "event_hondius",
        "event_cluster_key": "event_hondius",
        "bundle_scope": "event_snapshot",
        "cases_confirmed": 8,
    }
    candidates = [
        _candidate(
            "rec_may_8",
            "src_may_8",
            **common,
            snapshot_as_of_date="2026-05-08",
            count_semantics="event_outbreak_snapshot",
            geographic_scope="France",
            country="France",
            reporting_period="as of 2026-05-08",
        ),
        _candidate(
            "rec_may_12",
            "src_may_12",
            **common,
            snapshot_as_of_date="2026-05-12",
            count_semantics="event_outbreak_snapshot",
            geographic_scope="France",
            country="France",
            reporting_period="as of 2026-05-12",
        ),
        _candidate(
            "rec_incident",
            "src_incident",
            **common,
            snapshot_as_of_date="2026-05-08",
            count_semantics="incident_cases",
            geographic_scope="France",
            country="France",
            reporting_period="as of 2026-05-08",
        ),
        _candidate(
            "rec_canada",
            "src_canada",
            **common,
            snapshot_as_of_date="2026-05-08",
            count_semantics="event_outbreak_snapshot",
            geographic_scope="Canada",
            country="Canada",
            reporting_period="as of 2026-05-08",
        ),
    ]

    bundles = _build_case_evidence_bundles(candidates)

    assert len(bundles) == 3
    dynamic = next(
        bundle
        for bundle in bundles
        if set(bundle["member_record_ids"]) == {"rec_may_8", "rec_may_12"}
    )
    assert dynamic["snapshot_compatibility_status"] == "compatible_dynamic_snapshots"
    assert dynamic["link_status"] == "aggregate_snapshot_family"
    assert all(
        bundle["snapshot_compatibility_status"] != "conflicting_snapshots"
        for bundle in bundles
    )
    assert {bundle["event_family_id"] for bundle in bundles} == {"event_hondius"}


def test_event_family_uses_generic_vessel_location_and_time_window_anchors():
    def event_record(*, vessel: str, report_date: str, reporting_period: str) -> dict:
        return {
            "record_id": f"rec_{vessel}_{report_date}",
            "disease": "Novel arenavirus disease",
            "country": "Norway",
            "geographic_scope": "Norway",
            "date_reported": report_date,
            "reporting_period": reporting_period,
            "travel_or_vessel_context": vessel,
            "source_title": "Cruise outbreak situation update",
            "evidence_quote": "An outbreak investigation was updated.",
            "cases_confirmed": 1,
        }

    first = event_record(
        vessel="MV Aurora",
        report_date="2026-05-08",
        reporting_period="2026-04-01 to 2026-06-30",
    )
    update = event_record(
        vessel="MV Aurora",
        report_date="2026-05-12",
        reporting_period="2026-04-01 to 2026-06-30",
    )
    other_vessel = event_record(
        vessel="MV Borealis",
        report_date="2026-05-12",
        reporting_period="2026-04-01 to 2026-06-30",
    )
    later_period = event_record(
        vessel="MV Aurora",
        report_date="2027-02-02",
        reporting_period="2027-01-01 to 2027-03-31",
    )

    first_key = _case_event_key(first)
    update_key = _case_event_key(update)
    assert first_key == update_key
    assert first_key != _case_event_key(other_vessel)
    assert first_key != _case_event_key(later_period)

    candidates = [
        _candidate(
            record["record_id"],
            source_id,
            disease=record["disease"],
            country=record["country"],
            geographic_scope=record["geographic_scope"],
            date_reported=record["date_reported"],
            reporting_period=record["reporting_period"],
            travel_or_vessel_context=record["travel_or_vessel_context"],
            event_family_id=_case_event_key(record),
            event_cluster_key=_case_event_key(record),
            workflow_case_label="Case 1",
            case_label_normalized="case_1",
        )
        for record, source_id in ((first, "src_first"), (update, "src_update"))
    ]
    bundles = _build_case_evidence_bundles(candidates)

    assert len(bundles) == 1
    assert bundles[0]["link_status"] == "explicit_label_linked"


def test_case_label_aliases_link_only_without_hard_descriptor_conflicts():
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_case_1_a",
                "src_a",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Case 1",
                age="70",
                gender="female",
            ),
            _candidate(
                "rec_case_1_b",
                "src_b",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Case 1 (first death)",
                age="70",
                gender="female",
            ),
            _candidate(
                "rec_case_1_conflict",
                "src_conflict",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="Case 1 - 55-year-old female",
                age="55",
                gender="female",
            ),
        ]
    )

    assert len(bundles) == 2
    linked = next(bundle for bundle in bundles if set(bundle["source_ids"]) == {"src_a", "src_b"})
    assert linked["case_label_normalized"] == "case_1"
    assert linked["link_status"] == "explicit_label_linked"
    assert "canonical_explicit_case_label" in linked["link_reasons"]
    assert all(
        json.loads(bundle["field_conflicts_json"]) == {} for bundle in bundles
    )


def test_same_explicit_case_label_never_crosses_event_family():
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_event_a",
                "src_a",
                event_family_id="event_alpha",
                bundle_scope="individual_case",
                workflow_case_label="Case 1",
            ),
            _candidate(
                "rec_event_b",
                "src_b",
                event_family_id="event_beta",
                bundle_scope="individual_case",
                workflow_case_label="Case 1 (first death)",
            ),
        ]
    )

    assert len(bundles) == 2
    assert {bundle["event_family_id"] for bundle in bundles} == {
        "event_alpha",
        "event_beta",
    }


def test_unlabelled_records_require_three_compatible_anchors_to_link():
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_unlabelled_complete",
                "src_complete",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="",
                age="64",
                gender="male",
                nationality="Swiss",
                occupation_or_role="passenger",
            ),
            _candidate(
                "rec_unlabelled_compatible",
                "src_compatible",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="",
                age="64",
                gender="male",
                nationality="Swiss",
            ),
            _candidate(
                "rec_unlabelled_possible",
                "src_possible",
                event_family_id="event_hondius",
                bundle_scope="individual_case",
                workflow_case_label="",
                age="64",
                gender="male",
            ),
        ]
    )

    assert len(bundles) == 2
    linked = next(bundle for bundle in bundles if set(bundle["source_ids"]) == {"src_complete", "src_compatible"})
    possible = next(bundle for bundle in bundles if bundle["source_ids"] == ["src_possible"])
    assert linked["link_status"] == "compatible_unlabelled_records_linked"
    assert linked["link_score"] >= 3
    assert possible["link_status"] == "possible_same_case"


def test_case_bundle_scopes_keep_grouped_monitoring_and_historical_snapshots_separate():
    common = {
        "event_family_id": "event_hondius",
        "event_cluster_key": "event_hondius",
    }
    bundles = _build_case_evidence_bundles(
        [
            _candidate(
                "rec_grouped",
                "src_grouped",
                **common,
                bundle_scope="individual_case",
                workflow_case_label="Cases 1 and 2",
                cases_confirmed=2,
            ),
            _candidate(
                "rec_monitoring",
                "src_monitoring",
                **common,
                candidate_type="non_case_or_monitoring_candidate",
                bundle_scope="non_case_monitoring",
                workflow_case_label="",
            ),
            _candidate(
                "rec_historical",
                "src_historical",
                **common,
                candidate_type="aggregate_event_candidate",
                bundle_scope="event_snapshot",
                cases_confirmed=13,
                snapshot_as_of_date="2020-12-31",
                count_semantics="historical_total",
                geographic_scope="France",
                reporting_period="2020 annual total",
            ),
            _candidate(
                "rec_current",
                "src_current",
                **common,
                candidate_type="aggregate_event_candidate",
                bundle_scope="event_snapshot",
                cases_confirmed=13,
                snapshot_as_of_date="2026-05-08",
                count_semantics="event_outbreak_snapshot",
                geographic_scope="France",
                reporting_period="as of 2026-05-08",
            ),
        ]
    )

    assert len(bundles) == 4
    assert {bundle["bundle_scope"] for bundle in bundles} == {
        "grouped_case",
        "non_case_monitoring",
        "event_snapshot",
    }
    snapshot_bundles = [bundle for bundle in bundles if bundle["bundle_scope"] == "event_snapshot"]
    assert len(snapshot_bundles) == 2
    assert {bundle["event_family_id"] for bundle in snapshot_bundles} == {"event_hondius"}


def test_case_entity_ids_are_deterministic_across_candidate_order():
    candidates = [
        _candidate(
            "rec_case_1_a",
            "src_a",
            event_family_id="event_hondius",
            bundle_scope="individual_case",
            workflow_case_label="Case 1",
            age="70",
        ),
        _candidate(
            "rec_case_1_b",
            "src_b",
            event_family_id="event_hondius",
            bundle_scope="individual_case",
            workflow_case_label="Case 1 (first death)",
            age="70",
        ),
    ]

    forward = _build_case_evidence_bundles(candidates)
    reverse = _build_case_evidence_bundles(list(reversed(candidates)))

    assert [bundle["case_entity_id"] for bundle in forward] == [
        bundle["case_entity_id"] for bundle in reverse
    ]


def test_bundle_export_does_not_absorb_candidates_when_member_ids_are_missing():
    candidates = [
        _candidate("rec_case_1", "src_a", age="41"),
        _candidate("rec_case_2", "src_b", age="55"),
    ]
    rows = build_workflow_case_bundle_line_list(
        case_evidence_bundles=[
            {
                "case_evidence_bundle_id": "case_bundle_legacy",
                "event_cluster_key": "hantavirus|global|mv_hondius",
                "bundle_type": "individual_case_like",
                "source_ids": ["src_a", "src_b"],
            }
        ],
        case_candidate_dataset=candidates,
        source_inventory=[_source("src_a"), _source("src_b")],
    )

    assert rows[0]["link_status"] == "bundle_members_unresolved"
    assert rows[0]["age"] == ""
    assert json.loads(rows[0]["field_conflicts_json"]) == {}


def _record(record_id: str, **overrides) -> dict:
    row = {
        "record_id": record_id,
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "pathogen_or_syndrome": "hantavirus",
        "country": "France",
        "date_reported": "2026-05-01",
        "reporting_period": "2026-04-01 to 2026-06-30",
        "cases_confirmed": 1,
        "deaths": 0,
        "source_id": "src_official",
        "source_url": "https://www.who.int/example",
        "source_title": "WHO outbreak report",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "publisher": "World Health Organization",
        "credibility_score": 0.95,
        "credibility_level": "high",
        "evidence_quote": "A passenger was confirmed with Andes virus disease.",
        "supporting_chunk_id": "chunk_1",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "record_schema": "generic_public_health_record",
        "observation_type": "confirmed_case_record",
        "primary_case_dataset_eligible": True,
        "requires_human_review": True,
        "human_review_reason": "official_single_source_not_cross_validated",
    }
    row.update(overrides)
    return row


def test_final_package_and_export_include_workflow_line_list_outputs(tmp_path):
    case_quote = (
        "Case 5: A 41-year-old Dutch male national, serving as the ship's "
        "doctor, developed fever on 30 April 2026. PCR testing confirmed "
        "Andes virus infection on 6 May 2026."
    )
    record = _record(
        "rec_case_1",
        evidence_quote=case_quote,
        case_span_quote=case_quote,
        case_span_id="case_span_case_5",
        workflow_case_label="Case 5",
    )
    source = _source(
        "src_official",
        canonical_url="https://www.who.int/example",
        title="WHO outbreak report",
        source_type_final="international_public_health_agency",
        data_product_type="event_outbreak_report",
        source_product_type="event_outbreak_report",
        authority_bucket="official_authority",
    )
    source_only = _source(
        "src_sante_no_record",
        canonical_url="https://sante.gouv.fr/example-hondius",
        title="Cas d'Hantavirus à bord du navire MV Hondius",
        source_type_final="national_public_health_agency",
        data_product_type="event_outbreak_report",
        task_specificity="event_specific",
        authority_bucket="official_authority",
        domain="sante.gouv.fr",
    )
    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
        "collection_spec": {
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
        "normalized_records": [record],
        "source_registry": [source, source_only],
        "documents": [
            {
                "document_id": "doc_sante_no_record",
                "source_id": "src_sante_no_record",
                "url": "https://sante.gouv.fr/example-hondius",
                "canonical_url": "https://sante.gouv.fr/example-hondius",
                "parse_status": "parsed_html",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "clean_text": "MV Hondius hantavirus situation page without extractable counts.",
            }
        ],
        "evidence_chunks": [
            {
                "chunk_id": "chunk_1",
                "source_id": "src_official",
                "text": "A passenger was confirmed with Andes virus disease.",
            },
            {
                "chunk_id": "chunk_sante_no_record",
                "source_id": "src_sante_no_record",
                "text": "MV Hondius hantavirus situation page without extractable counts.",
                "disease_relevance_status": "target_disease_match",
            }
        ],
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
        "human_review_enabled": False,
        "structured_extraction_summary": {
            "focused_recovery_status_by_source": {
                "src_sante_no_record": "focused_retry_empty",
            },
            "llm_empty_output_diagnostics": [
                {
                    "source_id": "src_sante_no_record",
                    "chunk_id": "chunk_sante_no_record",
                    "reason": "focused_retry_empty",
                    "stage": "focused_llm_recovery",
                }
            ],
            "extraction_budget_by_source": {
                "src_sante_no_record": {
                    "source_id": "src_sante_no_record",
                    "queued_count": 1,
                    "eligible_count": 1,
                    "attempted_count": 0,
                    "record_count": 0,
                    "skipped_due_to_cap_count": 0,
                    "budget_bucket": "official_or_high_trust",
                }
            },
            "official_extraction_failures": [
                {
                    "source_id": "src_sante_no_record",
                    "reason": "must_fetch_source_not_attempted_for_extraction",
                }
            ],
        },
        "source_search_execution_summary": {
            "high_confidence_source_gap_status": "partial",
            "source_recall_target_ledger": [
                {
                    "authority_domain": "sante.gouv.fr",
                    "event_page_status": "event_page_found",
                }
            ],
            "high_confidence_domain_query_status": {
                "canada.ca": "executed_no_result",
                "who.int": "exact_event_page_found",
            },
        },
    }

    result = final_data_package_builder(state)
    package = result["final_data_package"]

    assert package["workflow_case_bundle_line_list"]
    assert package["workflow_case_candidate_line_list"]
    recovered_candidate = package["case_candidate_dataset"][0]
    assert recovered_candidate["age"] == "41"
    assert recovered_candidate["nationality"] == "Dutch"
    assert recovered_candidate["date_onset"] == "2026-04-30"
    assert recovered_candidate["date_confirmation"] == "2026-05-06"
    assert recovered_candidate["confirmation_method"] == "PCR"
    assert json.loads(recovered_candidate["field_provenance_json"])["age"][
        "case_span_id"
    ] == "case_span_case_5"
    source_only_inventory = {
        row["source_id"]: row for row in package["source_inventory"]
    }["src_sante_no_record"]
    assert source_only_inventory["source_to_evidence_status"] == (
        "parsed_target_source_no_record_extracted"
    )
    assert source_only_inventory["target_chunk_count"] == 1
    assert source_only_inventory["target_record_count"] == 0
    assert source_only_inventory["source_only_reason"] == (
        "event_source_focused_retry_empty"
    )
    assert source_only_inventory["extraction_attempted"] is True
    assert source_only_inventory["extraction_queued"] is True
    assert source_only_inventory["extraction_failure_substage"] == (
        "focused_llm_recovery"
    )
    assert source_only_inventory["extraction_failure_reason"] == (
        "focused_retry_empty"
    )
    assert source_only_inventory["focused_recovery_status"] == "focused_retry_empty"
    assert source_only_inventory["llm_empty_output_reason"] == "focused_retry_empty"
    assert source_only_inventory["source_exact_page_status"] == "event_page_found"
    assert package["export_manifest"]["section_counts"]["workflow_case_bundle_line_list"] == 1
    assert package["export_manifest"]["section_counts"]["workflow_case_candidate_line_list"] == 1

    manifest = export_final_data_package(package, tmp_path)
    for key, filename in {
        "workflow_case_bundle_line_list_csv": "workflow_case_bundle_line_list.csv",
        "workflow_case_bundle_line_list_json": "workflow_case_bundle_line_list.json",
        "workflow_case_candidate_line_list_csv": "workflow_case_candidate_line_list.csv",
        "workflow_case_candidate_line_list_json": "workflow_case_candidate_line_list.json",
    }.items():
        assert key in manifest["files"]
        assert (tmp_path / filename).exists()

    with (tmp_path / "workflow_case_candidate_line_list.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["workflow_candidate_id"] == "WFC-0001"
    assert rows[0]["source_to_evidence_status"] == "evidence_extracted"
    assert rows[0]["source_gap_status"] == "partial"
    assert json.loads(rows[0]["high_confidence_domain_status_json"]) == {
        "canada.ca": "executed_no_result",
        "who.int": "exact_event_page_found",
    }
    assert "Gh_ID" not in rows[0]

    bundle_header = (
        tmp_path / "workflow_case_bundle_line_list.csv"
    ).read_text(encoding="utf-8").splitlines()[0].split(",")
    with (tmp_path / "workflow_case_bundle_line_list.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        bundle_rows = list(csv.DictReader(handle))
    assert json.loads(bundle_rows[0]["high_confidence_domain_status_json"]) == {
        "canada.ca": "executed_no_result",
        "who.int": "exact_event_page_found",
    }
    assert bundle_header[:6] == [
        "workflow_row_id",
        "bundle_id",
        "event_family_id",
        "bundle_scope",
        "bundle_type",
        "workflow_inclusion_status",
    ]
    assert bundle_header.index("source_I_url") < bundle_header.index("source_II_url")
    assert bundle_header.index("source_II_url") < bundle_header.index("source_III_url")
    for column in (
        "event_family_id",
        "bundle_scope",
        "case_label_normalized",
        "snapshot_as_of_date",
        "high_trust_source_count",
        "official_source_count",
        "peer_reviewed_source_count",
        "structured_database_source_count",
        "source_stage_summary",
        "snapshot_compatibility_status",
    ):
        assert column in bundle_header
    candidate_header = (
        tmp_path / "workflow_case_candidate_line_list.csv"
    ).read_text(encoding="utf-8").splitlines()[0].split(",")
    for column in (
        "event_family_id",
        "bundle_scope",
        "case_label_normalized",
        "snapshot_as_of_date",
        "record_local_numeric_sentence",
        "record_local_numeric_disease_decision",
    ):
        assert column in candidate_header


def test_export_derives_workflow_line_lists_for_existing_packages(tmp_path):
    candidate = _candidate("rec_case_1", "src_official")
    bundle = {
        "case_evidence_bundle_id": "case_bundle_0001",
        "event_cluster_key": candidate["event_cluster_key"],
        "bundle_type": "individual_case_like",
        "case_candidate_ids": [candidate["case_candidate_id"]],
        "source_ids": ["src_official"],
        "candidate_statuses": ["reviewable"],
        "supporting_source_count": 1,
        "verified_authority_source_count": 1,
        "best_evidence_quote": candidate["evidence_quote"],
        "main_blocking_reason": candidate["review_reason"],
        "recommended_review_action": "review_individual_case_evidence",
    }
    source = _source(
        "src_official",
        canonical_url="https://www.who.int/example",
        source_type_final="international_public_health_agency",
        data_product_type="event_outbreak_report",
        authority_bucket="official_authority",
    )
    package = {
        "final_dataset": [],
        "case_candidate_dataset": [candidate],
        "case_evidence_bundles": [bundle],
        "source_inventory": [source],
    }

    manifest = export_final_data_package(package, tmp_path)

    assert manifest["section_counts"]["workflow_case_bundle_line_list"] == 1
    assert manifest["section_counts"]["workflow_case_candidate_line_list"] == 1
    with (tmp_path / "workflow_case_bundle_line_list.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["bundle_id"] == "case_bundle_0001"
    assert rows[0]["source_I_url"] == "https://www.who.int/example"
