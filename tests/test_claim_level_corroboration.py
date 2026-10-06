from __future__ import annotations

import json
from pathlib import Path

from data_collection_workflow.claim_corroboration import (
    _date_key,
    annotate_records_with_claim_corroboration,
    build_claims_from_state,
    build_corroborated_events,
    compare_claims,
    run_claim_corroboration,
)
from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.nodes.linking_validation import cross_source_consistency_check


def _record(record_id: str, **overrides) -> dict:
    source_id = overrides.get("source_id", f"src_{record_id}")
    row = {
        "record_id": record_id,
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "virus_or_syndrome": "Hantavirus pulmonary syndrome",
        "pathogen_or_syndrome": "hantavirus",
        "country": "United States of America",
        "subnational_location": "Virginia",
        "locality": None,
        "geographic_scope": "Virginia",
        "date_reported": "2025-06-15",
        "event_start_date": None,
        "event_end_date": None,
        "reporting_period": "2025-06",
        "as_of_date": None,
        "cases_confirmed": 1.0,
        "cases_probable": None,
        "cases_suspected": None,
        "cases_unspecified": None,
        "deaths": None,
        "hospitalizations": None,
        "statistical_count_type": "incident",
        "count_semantics": "confirmed case count",
        "count_unit": "persons",
        "source_id": source_id,
        "source_url": f"https://example.org/{source_id}",
        "source_title": "Virginia hantavirus report",
        "publisher": "Example Department of Health",
        "source_type": "official_public_health_agency",
        "source_role_final": "collection",
        "credibility_score": 0.9,
        "credibility_level": "high",
        "discovery_method": "fixture_search_result",
        "search_provider": "fixture",
        "query_id": "q_fixture",
        "query_used": "hantavirus Virginia 2025 case",
        "document_id": f"doc_{source_id}",
        "supporting_chunk_id": f"chunk_{record_id}",
        "evidence_quote": (
            "Virginia reported one confirmed hantavirus case in June 2025."
        ),
        "evidence_context": "case report paragraph",
        "extraction_method": "fixture_extractor",
        "extraction_confidence": 0.9,
        "normalization_status": "normalized",
        "schema_status": "valid",
        "provenance_status": "verified",
        "event_cluster_id": "event_virginia_2025",
        "linked_event_id": "linked_virginia_2025",
        "countable": True,
        "requires_human_review": False,
    }
    row.update(overrides)
    return row


def _state(records: list[dict], **overrides) -> dict:
    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2026-06-01",
        },
        "collection_spec": {
            "disease": "hantavirus",
            "geography": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2026-06-01",
            "time_window": "2025-01-01 to 2026-06-01",
        },
        "normalized_records": records,
        "validated_records": records,
        "raw_records": records,
        "source_registry": [
            {
                "source_id": record.get("source_id"),
                "canonical_url": record.get("source_url"),
                "title": record.get("source_title"),
                "publisher": record.get("publisher"),
                "source_type": record.get("source_type"),
                "source_role_final": record.get("source_role_final"),
                "credibility_score": record.get("credibility_score"),
                "credibility_level": record.get("credibility_level"),
                "status": "ready_for_content_fetch",
                "final_screening_decision": "include_for_content_fetch",
            }
            for record in records
        ],
        "documents": [],
        "evidence_chunks": [],
        "linked_events": [],
        "event_clusters": [],
        "duplicate_clusters": [],
        "validation_cases": [],
        "validation_comparisons": [],
        "validation_results": [],
        "validation_records": [],
        "active_validation_records": [],
        "inactive_validation_records": [],
        "validation_source_compatibility_summary": {
            "compatibility_status": "no_task_compatible_validation_source",
            "warnings": ["no_task_compatible_validation_source"],
        },
        "conflicts": [],
        "human_review_queue": [],
        "collection_trace": [],
    }
    state.update(overrides)
    return state


def _claims_by_record(claims: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for claim in claims:
        out.setdefault(claim["source_record_id"], []).append(claim)
    return out


def test_claims_are_built_from_normalized_records_with_provenance():
    records = [
        _record("rec_a", source_id="src_a"),
        _record("rec_b", source_id="src_b", source_url="https://b.example.org/case"),
    ]

    claims = build_claims_from_state(_state(records))

    assert len(claims) == 2
    first = claims[0]
    assert first["claim_id"] == "claim_rec_a_cases_confirmed"
    assert first["source_record_id"] == "rec_a"
    assert first["source_id"] == "src_a"
    assert first["source_url"] == "https://example.org/src_a"
    assert first["supporting_chunk_id"] == "chunk_rec_a"
    assert first["evidence_quote"]
    assert first["observation_type"] == "confirmed_case_record"
    assert first["count_field"] == "cases_confirmed"
    assert first["count_value"] == 1.0


def test_independent_matching_claims_create_corroborated_event():
    claims = build_claims_from_state(
        _state(
            [
                _record("rec_a", source_id="src_a"),
                _record(
                    "rec_b",
                    source_id="src_b",
                    publisher="Local Hospital Epidemiology Unit",
                    source_type="news_and_situation_report",
                ),
            ]
        )
    )

    comparisons = compare_claims(claims)
    events = build_corroborated_events(claims, comparisons)

    assert any(row["corroboration_match_status"] == "corroborates" for row in comparisons)
    event = events[0]
    assert event["corroboration_status"] in {"corroborated", "cross_source_supported"}
    assert event["independent_source_count"] == 2
    assert set(event["supporting_claim_ids"]) == {claim["claim_id"] for claim in claims}


def test_date_key_preserves_day_precision():
    assert _date_key("2025-06-15") == "2025-06-15"
    assert _date_key("reported 2025/6/5") == "2025-06-05"


def test_incompatible_count_semantics_do_not_corroborate():
    incident = _record(
        "incident",
        source_id="src_incident",
        statistical_count_type="incident",
        count_semantics="new cases reported during the week",
        cases_confirmed=8,
    )
    cumulative = _record(
        "cumulative",
        source_id="src_cumulative",
        statistical_count_type="cumulative",
        count_semantics="cumulative cases as of the report date",
        cases_confirmed=8,
    )
    claims = build_claims_from_state(_state([incident, cumulative]))

    comparisons = compare_claims(claims, storage_mode="full")
    events = build_corroborated_events(claims, comparisons)

    assert len(comparisons) == 1
    assert comparisons[0]["count_semantics_match_status"] == "not_matched"
    assert comparisons[0]["corroboration_match_status"] == "not_comparable"
    assert len(events) == 2
    assert all(event["corroboration_status"] != "corroborated" for event in events)


def test_same_source_duplicate_does_not_count_as_independent_corroboration():
    claims = build_claims_from_state(
        _state(
            [
                _record("rec_a", source_id="src_same", source_url="https://same.example/case"),
                _record("rec_b", source_id="src_same", source_url="https://same.example/case"),
            ]
        )
    )

    comparisons = compare_claims(claims)
    events = build_corroborated_events(claims, comparisons)

    assert comparisons[0]["corroboration_match_status"] == "duplicate_same_source"
    assert events[0]["independent_source_count"] == 1
    assert events[0]["corroboration_status"] != "corroborated"


def test_conflicting_counts_route_to_human_review():
    state = _state(
        [
            _record("rec_one_case", source_id="src_a", cases_confirmed=1.0),
            _record("rec_three_cases", source_id="src_b", cases_confirmed=3.0),
        ]
    )

    result = run_claim_corroboration(state)

    assert any(
        row["corroboration_match_status"] == "conflicts"
        for row in result["claim_comparisons"]
    )
    assert any(
        event["corroboration_status"] == "conflicting_claims"
        for event in result["corroborated_events"]
    )
    assert any(item["item_type"] == "claim_corroboration_conflict" for item in result["human_review_items"])


def test_minor_numeric_difference_is_recorded_without_human_review():
    state = _state(
        [
            _record("rec_twelve", source_id="src_a", cases_confirmed=None, cases_unspecified=12.0),
            _record("rec_thirteen", source_id="src_b", cases_confirmed=None, cases_unspecified=13.0),
        ]
    )

    result = run_claim_corroboration(state)

    comparison = result["claim_comparisons"][0]
    assert comparison["count_value_match_status"] == "minor_numeric_difference"
    assert comparison["corroboration_match_status"] == "partially_supports"
    assert comparison["needs_human_review"] is False
    assert result["human_review_items"] == []


def test_zero_case_statement_is_not_confirmed_case():
    record = _record(
        "rec_zero",
        cases_confirmed=0.0,
        count_semantics="No confirmed cases reported in Virginia yet.",
        evidence_quote="No confirmed cases reported in Virginia yet.",
    )

    claims = build_claims_from_state(_state([record]))
    claim = claims[0]
    events = build_corroborated_events(claims, [])

    assert claim["observation_type"] == "zero_case_statement"
    assert claim["is_zero_case_statement"] is True
    assert claim["primary_case_dataset_eligible"] is False
    assert events[0]["corroboration_status"] == "zero_case_statement_unverified"

    result = run_claim_corroboration(_state([record]))
    annotated = result["normalized_records"][0]
    assert annotated["primary_case_dataset_eligible"] is False
    assert annotated["independent_source_count"] == 1
    assert "not_primary_case_record" in annotated["claim_corroboration_warnings"]


def test_exposure_monitoring_is_not_case_record():
    record = _record(
        "rec_monitoring",
        cases_confirmed=None,
        count_semantics=(
            "Three people completed 42-day public health monitoring period and remained healthy; no cases reported"
        ),
        evidence_quote=(
            "To date, three people in Virginia have completed their 42-day public health monitoring period. "
            "All three remained healthy."
        ),
    )

    claims = build_claims_from_state(_state([record]))

    assert claims[0]["observation_type"] == "exposure_monitoring_record"
    assert claims[0]["is_exposure_monitoring_claim"] is True
    assert claims[0]["primary_case_dataset_eligible"] is False


def test_case_counts_win_over_contact_tracing_context():
    record = {
        "record_id": "rec_who_don604",
        "source_id": "src_who_don604",
        "disease": "Hantavirus disease",
        "virus_or_syndrome": "Andes virus",
        "date_reported": "2026-05-27",
        "cases_confirmed": 11.0,
        "cases_probable": 2.0,
        "deaths": 3.0,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
        "source_title": "Hantavirus outbreak linked to cruise ship travel, Multi-locations",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type_final": "international_public_health_agency",
        "source_independence_group": "WHO",
        "evidence_quote": (
            "As of 27 May, a total of 13 cases, including three deaths, "
            "have been reported. Eleven cases have been laboratory-confirmed "
            "for Andes virus infection, and two are probable cases. "
            "International contact tracing and follow up of contacts is ongoing."
        ),
        "requires_human_review": False,
    }
    source = {
        "source_id": "src_who_don604",
        "actual_publisher": "World Health Organization",
        "source_type_final": "international_public_health_agency",
        "source_independence_group": "WHO",
    }

    claims = run_claim_corroboration({"normalized_records": [record], "source_registry": [source]})["claims"]
    types = {claim["count_field"]: claim["observation_type"] for claim in claims}

    assert types["cases_confirmed"] == "confirmed_case_record"
    assert types["cases_probable"] == "probable_case_record"
    assert types["deaths"] == "death_record"
    assert all(not claim["is_exposure_monitoring_claim"] for claim in claims)
    assert all(claim["primary_case_dataset_eligible"] for claim in claims)


def test_record_annotation_preserves_multiple_observation_types_and_reason():
    record = _record("rec_multi", cases_confirmed=1.0, deaths=1.0)
    claims = [
        {
            "claim_id": "claim_rec_multi_cases_confirmed",
            "source_record_id": "rec_multi",
            "observation_type": "confirmed_case_record",
            "primary_case_dataset_eligible": True,
        },
        {
            "claim_id": "claim_rec_multi_deaths",
            "source_record_id": "rec_multi",
            "observation_type": "death_record",
            "primary_case_dataset_eligible": True,
        },
    ]
    events = [
        {
            "corroborated_event_id": "corr_multi",
            "supporting_claim_ids": [
                "claim_rec_multi_cases_confirmed",
                "claim_rec_multi_deaths",
            ],
            "unverified_claim_ids": [
                "claim_rec_multi_cases_confirmed",
                "claim_rec_multi_deaths",
            ],
            "corroboration_status": "single_source_unverified",
            "independent_source_count": 1,
            "official_source_support_count": 1,
            "secondary_source_support_count": 0,
            "primary_case_dataset_eligible": True,
            "corroboration_reason": "Primary case claim has no independent corroborating source.",
            "warnings": ["single_source_unverified"],
        }
    ]

    annotated = annotate_records_with_claim_corroboration([record], claims, events)
    row = annotated[0]

    assert set(row["observation_types"]) == {
        "confirmed_case_record",
        "death_record",
    }
    assert row["primary_case_dataset_eligible"] is True
    assert row["corroboration_reason"] == (
        "Primary case claim has no independent corroborating source."
    )
    assert set(row["claim_ids"]) == {
        "claim_rec_multi_cases_confirmed",
        "claim_rec_multi_deaths",
    }


def test_virginia_like_exposure_and_zero_case_do_not_enter_final_dataset():
    exposure = _record(
        "rec_vdh_monitoring",
        cases_confirmed=None,
        cases_unspecified=1.0,
        count_semantics="traveler under monitoring and remained healthy",
        evidence_quote=(
            "A Virginia traveler is currently in good health and is under public health monitoring."
        ),
    )
    zero_case = _record(
        "rec_zero_case",
        cases_confirmed=0.0,
        cases_unspecified=None,
        count_semantics="No confirmed hantavirus cases were reported in Virginia.",
        evidence_quote="No confirmed hantavirus cases were reported in Virginia.",
        source_id="src_zero",
        source_type="news_and_situation_report",
        publisher="Secondary source",
    )
    state = _state([exposure, zero_case])

    validation_result = cross_source_consistency_check(state)
    final_state = {**state, **validation_result}
    package_result = final_data_package_builder(final_state)
    package = package_result["final_data_package"]

    assert package["final_dataset"] == []
    assert {row["record_id"] for row in package["non_primary_observations"]} == {
        "rec_vdh_monitoring",
        "rec_zero_case",
    }
    assert package["corroboration_summary"]["corroborated_primary_case_event_count"] == 0
    assert package["run_quality_summary"]["corroborated_primary_case_event_count"] == 0
    assert package["run_quality_summary"]["run_quality_status"] in {
        "no_primary_case_dataset_records",
        "failed_quality_gate",
    }


def test_official_first_case_text_without_numeric_field_is_primary_case_claim():
    record = _record(
        "rec_nmdoh_first_case",
        cases_confirmed=None,
        cases_unspecified=None,
        source_title="New Mexico reports first hantavirus pulmonary syndrome case of 2024",
        publisher="New Mexico Department of Health",
        source_type="official_public_health_agency",
        evidence_quote=(
            "The Scientific Laboratory Division of the New Mexico Department of "
            "Health has confirmed the first case of hantavirus pulmonary syndrome "
            "this year. A man living in San Juan County was hospitalized, released, "
            "and is at home recovering."
        ),
    )

    claims = build_claims_from_state(_state([record]))

    assert len(claims) == 1
    assert claims[0]["observation_type"] == "confirmed_case_record"
    assert claims[0]["count_field"] == "cases_confirmed"
    assert claims[0]["count_value"] == 1.0
    assert claims[0]["primary_case_dataset_eligible"] is True


def test_official_death_confirmed_text_without_numeric_field_is_primary_claim():
    record = _record(
        "rec_nmdoh_death",
        deaths=None,
        cases_confirmed=None,
        source_title="Hantavirus death confirmed in Santa Fe County woman",
        publisher="New Mexico Department of Health",
        source_type="official_public_health_agency",
        evidence_quote=(
            "The New Mexico Department of Health confirmed today that a 65-year-old "
            "woman from Santa Fe County has died of Hantavirus Pulmonary Syndrome, "
            "marking the first reported case in New Mexico this year."
        ),
    )

    claims = build_claims_from_state(_state([record]))

    assert len(claims) == 1
    assert claims[0]["observation_type"] == "death_record"
    assert claims[0]["count_field"] == "deaths"
    assert claims[0]["count_value"] == 1.0
    assert claims[0]["primary_case_dataset_eligible"] is True


def test_national_aggregate_supports_local_claim_only_when_location_is_explicit():
    local = _record("rec_local", source_id="src_local", cases_confirmed=1.0)
    explicit_national = _record(
        "rec_us_explicit",
        source_id="src_us",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States of America",
        cases_confirmed=1.0,
        evidence_quote="The United States reported 6 hantavirus cases, including 1 in Virginia.",
        count_semantics="national aggregate explicitly includes Virginia",
    )
    generic_national = _record(
        "rec_us_generic",
        source_id="src_us_generic",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States of America",
        cases_confirmed=6.0,
        evidence_quote="The United States reported 6 hantavirus cases.",
        count_semantics="national aggregate without subnational breakdown",
    )

    explicit_comparison = compare_claims(
        build_claims_from_state(_state([local, explicit_national])),
        storage_mode="full",
    )
    generic_comparison = compare_claims(
        build_claims_from_state(_state([local, generic_national])),
        storage_mode="full",
    )

    assert explicit_comparison[0]["corroboration_match_status"] in {
        "partially_supports",
        "corroborates",
    }
    assert explicit_comparison[0]["geography_match_status"] in {
        "matched",
        "partially_matched",
    }
    assert generic_comparison[0]["corroboration_match_status"] in {
        "not_comparable",
        "insufficient_information",
    }


def test_background_context_does_not_corroborate_case_count():
    background = _record(
        "rec_background",
        cases_confirmed=None,
        count_semantics=None,
        source_title="Hantavirus prevention fact sheet",
        evidence_quote=(
            "Hantaviruses are spread by rodents. Prevention includes cleaning rodent droppings safely."
        ),
    )

    claims = build_claims_from_state(_state([background]))
    events = build_corroborated_events(claims, [])

    assert claims[0]["observation_type"] == "background_context"
    assert claims[0]["primary_case_dataset_eligible"] is False
    assert events[0]["corroboration_status"] == "context_only"


def test_corroboration_outputs_are_in_state_final_package_and_exports(tmp_path):
    records = [
        _record("rec_a", source_id="src_a"),
        _record("rec_b", source_id="src_b"),
    ]
    state = _state(records)
    validation_result = cross_source_consistency_check(state)
    final_state = {**state, **validation_result}
    package_result = final_data_package_builder(final_state)
    package = package_result["final_data_package"]
    manifest = export_final_data_package(package, tmp_path)

    assert validation_result["claims"]
    assert validation_result["claim_comparisons"]
    assert validation_result["corroborated_events"]
    assert validation_result["corroboration_summary"]["claim_count"] == 2
    assert package["claims"]
    assert package["claim_comparisons"]
    assert package["corroborated_events"]
    assert package["workflow_summaries"]["corroboration_summary"]["claim_count"] == 2
    assert "claims_json" in manifest["files"]
    assert "corroborated_events_json" in manifest["files"]
    assert "corroborated_case_events_csv" in manifest["files"]


def test_meaningful_storage_blocks_implausible_pairs_and_keeps_meaningful_edges(monkeypatch):
    records = [
        _record("rec_a", source_id="src_a"),
        _record(
            "rec_a_duplicate",
            source_id="src_a",
            source_url="https://example.org/src_a",
        ),
        _record("rec_b", source_id="src_b"),
        _record("rec_conflict", source_id="src_c", cases_confirmed=3.0),
        _record("rec_other_location", source_id="src_d", subnational_location="Oregon"),
        _record("rec_other_time", source_id="src_e", date_reported="2024-06-15"),
    ]
    monkeypatch.setenv("CLAIM_COMPARISON_STORAGE", "meaningful_edges")

    result = run_claim_corroboration(_state(records))

    statuses = {
        comparison["corroboration_match_status"]
        for comparison in result["claim_comparisons"]
    }
    summary = result["corroboration_summary"]
    assert statuses == {"corroborates", "conflicts", "duplicate_same_source"}
    assert summary["claim_comparison_storage"] == "meaningful_edges"
    assert summary["total_candidate_pair_count"] == 15
    assert summary["compared_pair_count"] == 4
    assert summary["skipped_pair_count"] == 11
    assert summary["claim_comparison_count"] == 4


def test_full_comparison_storage_retains_nonmeaningful_plausible_edges():
    death_claim = _record(
        "rec_death_same_block",
        source_id="src_death",
        cases_confirmed=None,
        deaths=1.0,
        count_semantics="death count",
        evidence_quote="Virginia reported one hantavirus death in June 2025.",
    )
    claims = build_claims_from_state(
        _state([_record("rec_a", source_id="src_a"), _record("rec_b", source_id="src_b"), death_claim])
    )

    meaningful = compare_claims(claims, storage_mode="meaningful_edges")
    full = compare_claims(claims, storage_mode="full")

    assert len(meaningful) == 1
    assert meaningful[0]["corroboration_match_status"] == "corroborates"
    assert len(full) == 3
    assert {row["corroboration_match_status"] for row in full} == {
        "corroborates",
        "not_comparable",
    }


def test_compact_package_externalizes_large_arrays_and_references_single_claim_artifact(tmp_path):
    package = {
        "final_dataset": [{"record_id": "rec_a"}],
        "claims": [{"claim_id": "claim_a"}],
        "claim_comparisons": [{"comparison_id": "claim_cmp_001"}],
        "evidence_chunks": [{"chunk_id": "chunk_a", "text": "large chunk"}],
        "raw_chunks": [{"chunk_id": "raw_chunk_a", "text": "raw chunk"}],
        "model_traces": [{"model": "fixture", "trace": "large trace"}],
        "corroboration_summary": {"claim_count": 1, "claim_comparison_count": 1},
        "workflow_summaries": {"corroboration_summary": {"claim_count": 1}},
    }

    manifest = export_final_data_package(package, tmp_path, package_mode="compact")
    final_package = json.loads(
        Path(manifest["files"]["final_package_json"]).read_text(encoding="utf-8")
    )

    assert final_package["package_mode"] == "compact"
    assert final_package["final_dataset"] == [{"record_id": "rec_a"}]
    assert "claim_comparisons" not in final_package
    assert "evidence_chunks" not in final_package
    assert "raw_chunks" not in final_package
    assert "model_traces" not in final_package
    assert final_package["artifact_manifest"]["files"]["claim_comparisons_json"] == manifest[
        "files"
    ]["claim_comparisons_json"]
    assert len(list(tmp_path.rglob("claim_comparisons.json"))) == 1
    assert Path(manifest["files"]["evidence_chunks_json"]).exists()
    assert Path(manifest["files"]["raw_chunks_json"]).exists()
    assert Path(manifest["files"]["model_traces_json"]).exists()


def test_full_audit_package_retains_embedded_large_arrays(tmp_path):
    package = {
        "final_dataset": [{"record_id": "rec_a"}],
        "claim_comparisons": [{"comparison_id": "claim_cmp_001"}],
        "evidence_chunks": [{"chunk_id": "chunk_a"}],
        "model_traces": [{"trace": "fixture"}],
    }

    manifest = export_final_data_package(package, tmp_path, package_mode="full_audit")
    final_package = json.loads(
        Path(manifest["files"]["final_package_json"]).read_text(encoding="utf-8")
    )

    assert final_package["package_mode"] == "full_audit"
    assert final_package["claim_comparisons"] == package["claim_comparisons"]
    assert final_package["evidence_chunks"] == package["evidence_chunks"]
    assert final_package["model_traces"] == package["model_traces"]
