from __future__ import annotations

import json

from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.observation_type_datasets import (
    DATASET_VIEW_KEYS,
    build_observation_type_dataset_split,
)


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
        "geographic_scope": "Virginia",
        "date_reported": "2025-06-15",
        "reporting_period": "2025",
        "cases_confirmed": None,
        "cases_probable": None,
        "cases_suspected": None,
        "cases_unspecified": None,
        "deaths": None,
        "hospitalizations": None,
        "source_id": source_id,
        "source_url": f"https://example.org/{source_id}",
        "source_title": "Virginia hantavirus source",
        "publisher": "Virginia Department of Health",
        "actual_publisher": "Virginia Department of Health",
        "source_type": "official_public_health_agency",
        "source_type_final": "state_or_local_public_health_agency",
        "source_independence_group": "virginia_state_government",
        "claim_support_role": "primary_case_claim_support",
        "source_role_final": "collection",
        "credibility_score": 0.9,
        "credibility_level": "high",
        "document_id": f"doc_{source_id}",
        "supporting_chunk_id": f"chunk_{record_id}",
        "evidence_quote": "Virginia reported one hantavirus public-health observation.",
        "evidence_context": "public health paragraph",
        "extraction_method": "fixture_extractor",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "event_cluster_id": f"event_{record_id}",
        "linked_event_id": f"linked_{record_id}",
        "countable": True,
        "requires_human_review": False,
        "review_status": "unreviewed",
        "record_final_inclusion_status": "accepted",
        "final_dataset_included": True,
        "primary_case_dataset_eligible": True,
        "observation_type": "confirmed_case_record",
        "observation_types": ["confirmed_case_record"],
        "claim_ids": [f"claim_{record_id}"],
        "corroborated_event_ids": [f"corr_{record_id}"],
        "corroboration_status": "corroborated",
        "independent_source_count": 2,
    }
    row.update(overrides)
    return row


def _claim(claim_id: str, record_id: str, observation_type: str, **overrides) -> dict:
    row = {
        "claim_id": claim_id,
        "source_record_id": record_id,
        "claim_type": "public_health_observation",
        "observation_type": observation_type,
        "disease": "Hantavirus disease",
        "country": "United States of America",
        "subnational_location": "Virginia",
        "date_or_period": "2025",
        "count_field": None,
        "count_value": None,
        "is_case_claim": observation_type.endswith("_case_record"),
        "is_death_claim": observation_type == "death_record",
        "is_zero_case_statement": observation_type == "zero_case_statement",
        "is_exposure_monitoring_claim": observation_type
        == "exposure_monitoring_record",
        "is_background_context_claim": observation_type == "background_context",
        "primary_case_dataset_eligible": observation_type
        in {
            "confirmed_case_record",
            "probable_case_record",
            "suspected_case_record",
            "unspecified_case_record",
        },
        "source_id": f"src_{record_id}",
        "source_url": f"https://example.org/src_{record_id}",
        "actual_publisher": "Virginia Department of Health",
        "source_type_final": "state_or_local_public_health_agency",
        "source_independence_group": "virginia_state_government",
        "claim_support_role": "primary_case_claim_support",
        "supporting_chunk_id": f"chunk_{record_id}",
        "evidence_quote": "Evidence quote for claim.",
        "claim_status": "active",
    }
    row.update(overrides)
    return row


def _event(event_id: str, claim_ids: list[str], observation_type: str, **overrides) -> dict:
    row = {
        "corroborated_event_id": event_id,
        "observation_type": observation_type,
        "supporting_claim_ids": claim_ids,
        "source_ids": ["src_a", "src_b"],
        "source_urls": ["https://example.org/src_a", "https://example.org/src_b"],
        "actual_publishers": ["Virginia Department of Health", "CDC"],
        "source_independence_groups": [
            "virginia_state_government",
            "us_federal_public_health",
        ],
        "independent_source_count": 2,
        "corroboration_status": "corroborated",
        "primary_case_dataset_eligible": observation_type
        in {
            "confirmed_case_record",
            "probable_case_record",
            "suspected_case_record",
            "unspecified_case_record",
        },
    }
    row.update(overrides)
    return row


def _state(records: list[dict], **overrides) -> dict:
    claims = []
    events = []
    for record in records:
        observation_type = record.get("observation_type") or "ambiguous_public_health_observation"
        claim_id = f"claim_{record['record_id']}"
        claims.append(_claim(claim_id, record["record_id"], observation_type))
        events.append(_event(f"corr_{record['record_id']}", [claim_id], observation_type))
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
        },
        "final_dataset_pre_quality_gate": records,
        "normalized_records": records,
        "validated_records": records,
        "raw_records": records,
        "final_dataset": [
            row
            for row in records
            if row.get("final_dataset_included") is True
            and str(row.get("record_final_inclusion_status")) in {"accepted", "accepted_with_warnings"}
        ],
        "final_dataset_post_review": [],
        "non_primary_observations": [
            row for row in records if row.get("primary_case_dataset_eligible") is False
        ],
        "quarantined_records": [
            row
            for row in records
            if str(row.get("record_final_inclusion_status", "")).startswith("quarantined")
        ],
        "pending_review_records": [
            row
            for row in records
            if str(row.get("record_final_inclusion_status", "")) == "pending_review"
        ],
        "claims": claims,
        "claim_comparisons": [],
        "corroborated_events": events,
        "source_registry": [],
        "record_inclusion_decisions": [],
        "run_quality_summary": {},
    }
    state.update(overrides)
    return state


def _ids(rows: list[dict]) -> set[str]:
    return {str(row.get("record_id") or row.get("source_record_id")) for row in rows}


def test_confirmed_case_record_enters_final_case_dataset():
    record = _record("rec_confirmed", cases_confirmed=1.0)

    split = build_observation_type_dataset_split(_state([record]))

    assert _ids(split["final_case_dataset"]) == {"rec_confirmed"}
    assert split["observation_type_dataset_summary"]["dataset_view_counts"][
        "final_case_dataset"
    ] == 1
    assert split["zero_case_statements"] == []
    assert split["exposure_monitoring_records"] == []
    assert split["context_records"] == []


def test_probable_and_suspected_records_enter_specific_case_views():
    probable = _record(
        "rec_probable",
        observation_type="probable_case_record",
        observation_types=["probable_case_record"],
        cases_probable=1.0,
        cases_confirmed=None,
    )
    suspected = _record(
        "rec_suspected",
        observation_type="suspected_case_record",
        observation_types=["suspected_case_record"],
        cases_suspected=2.0,
        cases_confirmed=None,
    )

    split = build_observation_type_dataset_split(_state([probable, suspected]))

    assert _ids(split["probable_case_dataset"]) == {"rec_probable"}
    assert _ids(split["suspected_case_dataset"]) == {"rec_suspected"}
    assert _ids(split["final_case_dataset"]) == {"rec_probable", "rec_suspected"}


def test_zero_case_statement_goes_to_zero_case_view_not_final_case_dataset():
    zero = _record(
        "rec_zero",
        observation_type="zero_case_statement",
        observation_types=["zero_case_statement"],
        cases_confirmed=0.0,
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_zero_case_statement",
        evidence_quote="No confirmed hantavirus cases were reported in Virginia.",
    )

    split = build_observation_type_dataset_split(_state([zero]))

    assert _ids(split["zero_case_statements"]) == {"rec_zero"}
    assert split["final_case_dataset"] == []
    assert _ids(split["non_primary_observations"]) == {"rec_zero"}


def test_exposure_monitoring_record_goes_to_exposure_view():
    monitoring = _record(
        "rec_monitoring",
        observation_type="exposure_monitoring_record",
        observation_types=["exposure_monitoring_record"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_exposure_monitoring",
        evidence_quote="The traveler remained healthy and completed public health monitoring.",
    )

    split = build_observation_type_dataset_split(_state([monitoring]))

    assert _ids(split["exposure_monitoring_records"]) == {"rec_monitoring"}
    assert split["final_case_dataset"] == []


def test_background_context_goes_to_context_records():
    context = _record(
        "rec_context",
        observation_type="background_context",
        observation_types=["background_context"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_background_context",
        claim_support_role="context_only",
        source_type_final="background_fact_sheet",
        evidence_quote="Hantavirus symptoms and prevention information.",
    )

    split = build_observation_type_dataset_split(_state([context]))

    assert _ids(split["context_records"]) == {"rec_context"}
    assert split["final_case_dataset"] == []


def test_surveillance_summary_is_separate_unless_primary_eligible():
    non_primary = _record(
        "rec_surveillance_context",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        cases_unspecified=10.0,
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_non_primary_observation",
    )
    primary = _record(
        "rec_surveillance_primary",
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        cases_unspecified=1.0,
        primary_case_dataset_eligible=True,
    )

    split = build_observation_type_dataset_split(_state([non_primary, primary]))

    assert _ids(split["surveillance_summary_records"]) == {
        "rec_surveillance_context",
        "rec_surveillance_primary",
    }
    assert _ids(split["final_case_dataset"]) == {"rec_surveillance_primary"}


def test_public_health_metric_mislabeled_as_case_stays_out_of_final_case_dataset():
    metric = _record(
        "rec_lab_positive_metric",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        cases_confirmed=119.0,
        tests_positive=119.0,
        metric_name="Influenza positive lab reports",
        metric_value=119.0,
        metric_unit="count",
        metric_category="lab_positive_count",
        count_semantics="weekly laboratory positive specimens",
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted",
        source_type_final="state_or_local_public_health_agency",
        evidence_quote="Influenza positive lab reports: 119.",
    )

    split = build_observation_type_dataset_split(_state([metric]))

    assert split["final_case_dataset"] == []
    assert _ids(split["surveillance_summary_records"]) == {"rec_lab_positive_metric"}
    assert _ids(split["non_primary_observations"]) == {"rec_lab_positive_metric"}


def test_percentage_public_health_metric_stays_out_of_final_case_dataset():
    metric = _record(
        "rec_hospitalization_percent",
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        cases_unspecified=114.0,
        metric_name="Percent of cases hospitalized",
        metric_value=40.0,
        metric_unit="percent",
        metric_category="public_health_metric",
        count_semantics="percentage of reported cases hospitalized",
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted",
        source_type_final="official_public_health_agency",
        evidence_quote="40% of cases hospitalized (114 of 285).",
    )

    split = build_observation_type_dataset_split(_state([metric]))

    assert split["final_case_dataset"] == []
    assert _ids(split["surveillance_summary_records"]) == {"rec_hospitalization_percent"}
    assert _ids(split["non_primary_observations"]) == {"rec_hospitalization_percent"}


def test_variant_influenza_case_record_stays_out_of_default_flu_case_dataset():
    record = _record(
        "rec_variant_h3n2v",
        disease="Influenza",
        disease_standard_name="Seasonal influenza",
        virus_or_syndrome="variant influenza A(H3N2)v",
        pathogen_or_syndrome="influenza A(H3N2)v",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-09-07",
        reporting_period="MMWR week 36, 2024",
        cases_confirmed=2.0,
        metric_name="variant influenza A(H3N2)v cases",
        metric_value=2.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-09-01",
        metric_period_end="2024-09-07",
        count_semantics="variant influenza case count",
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted_with_warnings",
        evidence_quote="Two variant influenza A(H3N2)v cases were reported.",
    )

    split = build_observation_type_dataset_split(
        _state(
            [record],
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-10-01",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-01-01",
                "end_date": "2024-10-01",
                "collection_mode": "direct_collection",
            },
        )
    )

    assert split["final_case_dataset"] == []
    assert _ids(split["non_primary_observations"]) == {"rec_variant_h3n2v"}


def test_missing_unknown_demographic_case_table_row_stays_out_of_case_dataset():
    record = _record(
        "rec_tb_missing_age",
        disease="Tuberculosis",
        disease_standard_name="Tuberculosis",
        country="United States of America",
        subnational_location=None,
        geographic_scope="United States",
        geographic_scope_type="country",
        date_reported="2024-12-31",
        reporting_period="2024",
        cases_unspecified=33.0,
        metric_name="TB cases with missing or unknown age",
        metric_value=33.0,
        metric_unit="count",
        metric_category="case_count",
        metric_period_start="2024-01-01",
        metric_period_end="2024-12-31",
        count_semantics="data quality submetric for cases with missing or unknown age",
        observation_type="unspecified_case_record",
        observation_types=["unspecified_case_record"],
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted",
        source_type_final="national_public_health_agency",
        evidence_quote="Age was missing or unknown for 33 TB cases.",
    )

    split = build_observation_type_dataset_split(_state([record]))

    assert split["final_case_dataset"] == []
    assert _ids(split["surveillance_summary_records"]) == {"rec_tb_missing_age"}
    assert _ids(split["non_primary_observations"]) == {"rec_tb_missing_age"}


def test_outbreak_summary_outside_scope_stays_out_of_final_case_dataset():
    outbreak = _record(
        "rec_outbreak",
        observation_type="outbreak_summary",
        observation_types=["outbreak_summary"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_outside_scope",
        evidence_quote="A cruise outbreak summary outside Virginia.",
    )

    split = build_observation_type_dataset_split(_state([outbreak]))

    assert _ids(split["outbreak_summary_records"]) == {"rec_outbreak"}
    assert split["final_case_dataset"] == []


def test_global_who_outbreak_record_enters_global_outbreak_view():
    who = _record(
        "rec_who_don604",
        source_id="src_who_don604",
        source_url="https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
        source_title="Hantavirus outbreak linked to cruise ship travel, Multi-locations",
        publisher="World Health Organization",
        actual_publisher="World Health Organization",
        source_type_final="international_public_health_agency",
        geographic_scope="Multi-locations",
        geographic_scope_type="multi_country",
        country="",
        subnational_location="",
        date_reported="2026-05-27",
        cases_confirmed=11.0,
        cases_probable=2.0,
        deaths=3.0,
        observation_type="confirmed_case_record",
        observation_types=["confirmed_case_record"],
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted_with_review_warning",
        evidence_quote=(
            "As of 27 May, a total of 13 cases, including three deaths, "
            "have been reported. International contact tracing is ongoing."
        ),
    )

    split = build_observation_type_dataset_split(
        _state(
            [who],
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
            },
        )
    )

    assert _ids(split["global_outbreak_event_dataset"]) == {"rec_who_don604"}
    assert split["exposure_monitoring_records"] == []


def test_global_paho_regional_aggregate_enters_regional_surveillance_view():
    paho = _record(
        "rec_paho_alert",
        source_id="src_paho_alert",
        source_url="https://www.paho.org/sites/default/files/2025-12/hantavirus-alert.pdf",
        source_title="Epidemiological Alert Hantavirus Pulmonary Syndrome in the Americas",
        publisher="Pan American Health Organization",
        actual_publisher="Pan American Health Organization",
        source_type_final="international_public_health_agency",
        geographic_scope="Americas",
        geographic_scope_type="region",
        country="",
        subnational_location="",
        date_reported="2025-12-19",
        cases_unspecified=229.0,
        deaths=59.0,
        observation_type="surveillance_summary",
        observation_types=["surveillance_summary"],
        primary_case_dataset_eligible=True,
        final_dataset_included=True,
        record_final_inclusion_status="accepted_with_review_warning",
        evidence_quote=(
            "PAHO reported 229 hantavirus pulmonary syndrome cases and "
            "59 deaths in the Americas in 2025."
        ),
    )

    split = build_observation_type_dataset_split(
        _state(
            [paho],
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
            },
        )
    )

    assert _ids(split["regional_surveillance_dataset"]) == {"rec_paho_alert"}
    assert _ids(split["official_alert_dataset"]) == {"rec_paho_alert"}


def test_ambiguous_record_goes_to_unclassified_observation_records():
    ambiguous = _record(
        "rec_ambiguous",
        observation_type="ambiguous_public_health_observation",
        observation_types=["ambiguous_public_health_observation"],
        primary_case_dataset_eligible=None,
        final_dataset_included=False,
        record_final_inclusion_status="pending_review",
    )

    split = build_observation_type_dataset_split(_state([ambiguous]))

    assert _ids(split["unclassified_observation_records"]) == {"rec_ambiguous"}
    assert split["final_case_dataset"] == []


def test_dataset_split_uses_claims_when_record_lacks_observation_type():
    record = _record(
        "rec_claim_only",
        observation_type=None,
        observation_types=[],
        primary_case_dataset_eligible=None,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_exposure_monitoring",
    )
    claim = _claim(
        "claim_rec_claim_only",
        "rec_claim_only",
        "exposure_monitoring_record",
        primary_case_dataset_eligible=False,
        is_exposure_monitoring_claim=True,
    )

    split = build_observation_type_dataset_split(
        _state([record], claims=[claim], corroborated_events=[])
    )

    assert _ids(split["exposure_monitoring_records"]) == {"rec_claim_only"}
    assert split["final_case_dataset"] == []


def test_export_includes_all_new_dataset_views(tmp_path):
    records = [
        _record("rec_case", cases_confirmed=1.0),
        _record(
            "rec_zero",
            observation_type="zero_case_statement",
            observation_types=["zero_case_statement"],
            primary_case_dataset_eligible=False,
            final_dataset_included=False,
            record_final_inclusion_status="quarantined_zero_case_statement",
        ),
    ]
    package = final_data_package_builder(_state(records))["final_data_package"]

    manifest = export_final_data_package(package, tmp_path)

    for key in DATASET_VIEW_KEYS:
        assert key in package
    assert "observation_type_dataset_summary" in package
    assert (tmp_path / "final_case_dataset.csv").exists()
    assert (tmp_path / "final_case_dataset.json").exists()
    assert (tmp_path / "zero_case_statements.csv").exists()
    assert (tmp_path / "zero_case_statements.json").exists()
    assert (tmp_path / "exposure_monitoring_records.csv").exists()
    assert (tmp_path / "context_records.json").exists()
    assert (tmp_path / "observation_type_dataset_summary.json").exists()
    assert manifest["section_counts"]["final_case_dataset"] == 1


def test_export_includes_professor_facing_dataset_product_aliases(tmp_path):
    records = [
        _record("rec_case", cases_confirmed=1.0),
        _record(
            "rec_zero",
            observation_type="zero_case_statement",
            observation_types=["zero_case_statement"],
            primary_case_dataset_eligible=False,
            final_dataset_included=False,
            record_final_inclusion_status="quarantined_zero_case_statement",
            quality_gate_blocking_flags=["not_primary_case_record"],
        ),
    ]
    package = final_data_package_builder(_state(records))["final_data_package"]

    manifest = export_final_data_package(package, tmp_path)

    assert (tmp_path / "primary_case_dataset.csv").exists()
    assert (tmp_path / "primary_case_dataset.json").exists()
    assert (tmp_path / "task_aware_observation_dataset.csv").exists()
    assert (tmp_path / "task_aware_observation_dataset.json").exists()
    assert (tmp_path / "reviewable_dataset.csv").exists()
    assert (tmp_path / "reviewable_dataset.json").exists()
    assert manifest["section_counts"]["primary_case_dataset"] == 1
    assert manifest["section_counts"]["task_aware_observation_dataset"] == 1
    assert manifest["section_counts"]["reviewable_dataset"] == 1

    saved = json.loads((tmp_path / "final_package.json").read_text(encoding="utf-8"))
    assert saved["primary_case_dataset"][0]["record_id"] == "rec_case"
    assert saved["task_aware_observation_dataset"][0]["record_id"] == "rec_zero"
    assert saved["reviewable_dataset"][0]["record_id"] == "rec_zero"


def test_run_quality_summary_includes_dataset_view_counts():
    zero = _record(
        "rec_zero",
        observation_type="zero_case_statement",
        observation_types=["zero_case_statement"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_zero_case_statement",
    )
    package = final_data_package_builder(_state([zero]))["final_data_package"]
    summary = package["run_quality_summary"]
    quality = package["final_dataset_quality_summary"]

    assert summary["final_case_dataset_count"] == 0
    assert summary["zero_case_statement_count"] == 1
    assert summary["dataset_view_counts"]["zero_case_statements"] == 1
    assert quality["final_case_dataset_count"] == 0
    assert quality["dataset_view_counts"]["zero_case_statements"] == 1


def test_virginia_like_offline_regression_separates_non_primary_views():
    monitoring = _record(
        "rec_monitoring",
        observation_type="exposure_monitoring_record",
        observation_types=["exposure_monitoring_record"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_exposure_monitoring",
    )
    zero = _record(
        "rec_zero",
        observation_type="zero_case_statement",
        observation_types=["zero_case_statement"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_zero_case_statement",
    )
    context = _record(
        "rec_context",
        observation_type="background_context",
        observation_types=["background_context"],
        primary_case_dataset_eligible=False,
        final_dataset_included=False,
        record_final_inclusion_status="quarantined_background_context",
    )

    package = final_data_package_builder(
        _state([monitoring, zero, context])
    )["final_data_package"]

    assert package["final_case_dataset"] == []
    assert len(package["exposure_monitoring_records"]) == 1
    assert len(package["zero_case_statements"]) == 1
    assert len(package["context_records"]) == 1
    assert package["run_quality_summary"]["no_primary_case_dataset_records"] is True
    assert package["run_quality_summary"]["recommended_primary_dataset_message"]


def test_final_package_json_preserves_observation_type_sections(tmp_path):
    record = _record("rec_case", cases_confirmed=1.0)
    package = final_data_package_builder(_state([record]))["final_data_package"]
    export_final_data_package(package, tmp_path)

    saved = json.loads((tmp_path / "final_package.json").read_text(encoding="utf-8"))

    assert saved["final_case_dataset"][0]["record_id"] == "rec_case"
    assert saved["observation_type_dataset_summary"]["dataset_view_counts"][
        "final_case_dataset"
    ] == 1
