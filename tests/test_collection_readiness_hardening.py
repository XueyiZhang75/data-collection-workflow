import csv
import json
from pathlib import Path

from data_collection_workflow.collection_readiness import (
    build_collection_readiness_summary,
    write_collection_readiness_outputs,
)
from data_collection_workflow.evaluation_report_builder import (
    build_evaluation_report,
    write_evaluation_outputs,
)
from data_collection_workflow.geography import resolve_record_geography
from data_collection_workflow.numeric_semantics import sanitize_case_death_numeric_fields
from data_collection_workflow.reporting.report_facts import build_report_facts
from data_collection_workflow.source_identity import lookup_source_identity_registry


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _session(tmp_path: Path, *, executed_queries=1, fetched_documents=1, chunks=1):
    session = tmp_path / "session"
    _write_json(
        session / "workflow_run_summary.json",
        {
            "live_search_enabled": True,
            "live_fetch_enabled": True,
            "fixture_documents_enabled": False,
            "source_search_execution_summary": {
                "executed_query_count": executed_queries,
                "candidate_from_search_count": 1,
            },
            "document_count": fetched_documents,
            "accepted_record_count": 1,
        },
    )
    _write_json(
        session / "diagnostics" / "source_search_execution_summary.json",
        {
            "executed_query_count": executed_queries,
            "candidate_from_search_count": 1,
        },
    )
    _write_json(
        session / "diagnostics" / "content_fetch_summary.json",
        {
            "fixture_documents_enabled": False,
            "offline_metadata_stub_only": False,
        },
    )
    _write_json(
        session / "diagnostics" / "source_registry.json",
        [
            {
                "source_id": "src_1",
                "source_role_final": "collection",
                "actual_publisher": "Example Health Department",
            }
        ],
    )
    _write_json(
        session / "diagnostics" / "documents.json",
        [
            {
                "document_id": f"doc_{i}",
                "source_id": "src_1",
                "fetch_status": "success",
                "parse_status": "usable",
                "quality_status": "usable",
            }
            for i in range(fetched_documents)
        ],
    )
    _write_json(
        session / "diagnostics" / "evidence_chunks.json",
        [
            {
                "chunk_id": f"chunk_{i}",
                "text": "One confirmed human case was reported in New Mexico in 2024.",
            }
            for i in range(chunks)
        ],
    )
    _write_json(session / "diagnostics" / "raw_records.json", [{"record_id": "raw_1"}])
    _write_json(session / "diagnostics" / "validated_records.json", [{"record_id": "val_1"}])
    _write_json(session / "diagnostics" / "normalized_records.json", [{"record_id": "rec_1"}])
    final = [
        {
            "record_id": "rec_1",
            "disease": "Hantavirus disease",
            "country": "United States",
            "subnational_location": "New Mexico",
            "cases_confirmed": 1,
            "source_url": "https://example.gov/report",
            "evidence_quote": "One confirmed human case was reported.",
            "source_id": "src_1",
        }
    ]
    _write_json(session / "collection" / "final_dataset.json", final)
    _write_csv(session / "collection" / "final_dataset.csv", final)
    _write_json(session / "collection" / "final_dataset_pre_quality_gate.json", final)
    _write_json(session / "collection" / "pending_review_records.json", [])
    _write_json(session / "collection" / "quarantined_records.json", [])
    _write_csv(session / "human_review" / "top_review_items.csv", [])
    _write_csv(
        session / "evaluation" / "evaluation_report.csv",
        [{"evaluation_row_id": "eval_001", "overall_match_status": "match"}],
    )
    _write_json(session / "diagnostics" / "run_quality_summary.json", {})
    _write_json(session / "diagnostics" / "source_identity_summary.json", {})
    _write_json(session / "diagnostics" / "corroboration_summary.json", {})
    _write_json(session / "diagnostics" / "validation_source_compatibility_summary.json", {})
    return session


def test_case_study_real_mode_disallows_offline_stub_as_real(tmp_path):
    session = _session(tmp_path)
    _write_json(
        session / "diagnostics" / "content_fetch_summary.json",
        {"fixture_documents_enabled": True, "offline_metadata_stub_only": True},
    )

    summary = build_collection_readiness_summary(
        session,
        case_study_real_mode=True,
    )

    assert summary["failure_stage"] == "completed_without_real_collection"
    assert summary["root_failure_class"] == "fixture_or_offline_stub"
    assert summary["collection_readiness_status"] == "not_ready"


def test_case_study_real_mode_executed_search_queries_zero_failure(tmp_path):
    session = _session(tmp_path, executed_queries=0)

    summary = build_collection_readiness_summary(
        session,
        case_study_real_mode=True,
    )

    assert summary["failure_stage"] == "diagnostic_failure_at_live_search"
    assert summary["root_failure_class"] == "live_search_not_executed"


def test_case_study_real_mode_fetched_documents_zero_failure(tmp_path):
    session = _session(tmp_path, fetched_documents=0)

    summary = build_collection_readiness_summary(
        session,
        case_study_real_mode=True,
    )

    assert summary["failure_stage"] == "diagnostic_failure_at_fetch"
    assert summary["root_failure_class"] == "fetch_not_successful"


def test_case_study_real_mode_evidence_chunks_zero_failure(tmp_path):
    session = _session(tmp_path, chunks=0)

    summary = build_collection_readiness_summary(
        session,
        case_study_real_mode=True,
    )

    assert summary["failure_stage"] == "diagnostic_failure_at_evidence_chunking"
    assert summary["root_failure_class"] == "no_evidence_chunks"


def test_readiness_outputs_json_csv_and_markdown(tmp_path):
    session = _session(tmp_path)

    outputs = write_collection_readiness_outputs(session, case_study_real_mode=True)

    assert Path(outputs["collection_readiness_summary_json"]).exists()
    assert Path(outputs["collection_readiness_summary_csv"]).exists()
    markdown = Path(outputs["collection_readiness_summary_md"]).read_text(encoding="utf-8")
    assert "collection_readiness_status" in markdown
    payload = json.loads(
        Path(outputs["collection_readiness_summary_json"]).read_text(encoding="utf-8")
    )
    assert payload["collection_readiness_status"] == "ready"


def test_readiness_outputs_quality_gate_failure_breakdown(tmp_path):
    session = _session(tmp_path)
    pre_quality = [
        {
            "record_id": "rec_candidate",
            "record_final_inclusion_status": "quarantined_unsupported_numeric_claim",
            "quality_gate_blocking_flags": [
                "unsupported_numeric_claim",
                "missing_provenance",
            ],
            "semantic_warnings": ["numeric_semantics_rejected:cases_confirmed"],
            "geography_status": "mismatch",
        }
    ]
    _write_json(session / "collection" / "final_dataset.json", [])
    _write_json(session / "collection" / "final_case_dataset.json", [])
    _write_json(session / "collection" / "non_primary_observations.json", [])
    _write_json(session / "collection" / "final_dataset_pre_quality_gate.json", pre_quality)
    _write_json(session / "collection" / "quarantined_records.json", pre_quality)

    outputs = write_collection_readiness_outputs(session, case_study_real_mode=True)

    breakdown_path = Path(outputs["quality_gate_failure_breakdown_json"])
    assert breakdown_path.exists()
    breakdown = json.loads(breakdown_path.read_text(encoding="utf-8"))
    assert breakdown["pre_quality_records_count"] == 1
    assert breakdown["accepted_primary_case_records_count"] == 0
    assert breakdown["quarantined_count"] == 1
    assert breakdown["unsupported_numeric_claim_count"] == 1
    assert breakdown["missing_provenance_count"] == 1
    assert breakdown["geography_mismatch_count"] == 1


def test_readiness_is_partial_when_validation_rows_are_zero(tmp_path):
    session = _session(tmp_path)
    _write_csv(session / "evaluation" / "evaluation_report.csv", [])

    summary = build_collection_readiness_summary(session, case_study_real_mode=True)

    assert summary["failure_stage"] == "none"
    assert summary["root_failure_class"] == "benchmark_validation_not_available"
    assert summary["collection_readiness_status"] == "partial"


def test_readiness_distinguishes_non_primary_observations_only(tmp_path):
    session = _session(tmp_path)
    observation = {
        "record_id": "rec_observation",
        "observation_type": "surveillance_summary",
        "primary_case_dataset_eligible": False,
        "record_final_inclusion_status": "quarantined_non_primary_observation",
        "evidence_quote": "Official surveillance indicator was reported.",
        "source_url": "https://example.gov/surveillance",
    }
    _write_json(session / "collection" / "final_dataset.json", [])
    _write_json(session / "collection" / "final_case_dataset.json", [])
    _write_json(session / "collection" / "primary_case_dataset.json", [])
    _write_json(session / "collection" / "non_primary_observations.json", [observation])
    _write_json(
        session / "collection" / "task_aware_observation_dataset.json",
        [observation],
    )
    _write_json(session / "collection" / "final_dataset_pre_quality_gate.json", [observation])
    _write_json(session / "collection" / "quarantined_records.json", [observation])

    summary = build_collection_readiness_summary(session, case_study_real_mode=True)

    assert summary["failure_stage"] == "diagnostic_failure_at_quality_gate"
    assert summary["root_failure_class"] == "non_primary_observations_only"
    assert summary["accepted_primary_case_records"] == 0
    assert summary["accepted_task_aware_observations"] == 1
    assert summary["collection_readiness_status"] == "partial"


def test_source_identity_registry_recognizes_core_public_health_domains():
    cases = {
        "https://www.cdc.gov/hantavirus/": "Centers for Disease Control and Prevention",
        "https://www.who.int/news-room": "World Health Organization",
        "https://www.ecdc.europa.eu/en": "European Centre for Disease Prevention and Control",
        "https://www.nmhealth.org/news/": "New Mexico Department of Health",
    }

    for url, publisher in cases.items():
        entry = lookup_source_identity_registry(url)
        assert entry is not None
        assert entry["publisher_name"] == publisher
        assert entry["source_type"]
        assert entry["credibility_tier"]


def test_state_local_geography_does_not_drift_to_us_only():
    record = {"record_id": "rec_nm", "evidence_quote": "A case was reported today."}
    source = lookup_source_identity_registry("https://www.nmhealth.org/news/")

    resolved = resolve_record_geography(
        record,
        source_identity=source,
        task={"location": "New Mexico"},
    )

    assert resolved["geographic_scope"] == "New Mexico"
    assert resolved["subnational_location"] == "New Mexico"
    assert resolved["country"] == "United States"
    assert resolved["geography_inference_method"] == "source_jurisdiction"
    assert resolved["geography_inference_warning"] is True


def test_california_and_county_geography_are_preserved():
    california = resolve_record_geography(
        {"record_id": "rec_ca", "evidence_quote": "West Nile virus activity increased."},
        task={"location": "California"},
    )
    county = resolve_record_geography(
        {
            "record_id": "rec_county",
            "locality": "Los Angeles County",
            "subnational_location": "California",
            "evidence_quote": "Los Angeles County reported one case.",
        },
        task={"location": "California"},
    )

    assert california["geographic_scope"] == "California"
    assert california["subnational_location"] == "California"
    assert county["locality"] == "Los Angeles County"
    assert county["subnational_location"] == "California"
    assert county["geographic_scope_type"] == "county"


def test_national_cdc_record_can_be_national_when_scope_is_national():
    source = lookup_source_identity_registry("https://www.cdc.gov/hantavirus/")

    resolved = resolve_record_geography(
        {
            "record_id": "rec_cdc",
            "country": "United States",
            "geographic_scope": "United States",
            "geographic_scope_type": "country",
            "evidence_quote": "National surveillance reported 10 cases.",
        },
        source_identity=source,
        task={"location": "United States"},
    )

    assert resolved["geographic_scope"] == "United States"
    assert resolved["geographic_scope_type"] == "country"
    assert resolved["geography_inference_warning"] is False


def test_counties_weeks_percentages_are_not_extracted_as_case_counts():
    record = {
        "record_id": "rec_bad_numeric",
        "cases_confirmed": 3,
        "evidence_quote": "Data were reported from 3 counties in week 40; percent positive was 5%.",
        "metric_name": "county_week_percent_summary",
    }

    sanitized = sanitize_case_death_numeric_fields(record)

    assert sanitized["cases_confirmed"] is None
    assert "numeric_semantics_rejected:cases_confirmed" in sanitized["semantic_warnings"]


def test_missing_numeric_label_warns_but_preserves_values_without_non_case_marker():
    record = {
        "record_id": "rec_sparse_numeric",
        "cases_confirmed": 2,
        "deaths": 1,
        "evidence_quote": "Synthetic test record.",
    }

    sanitized = sanitize_case_death_numeric_fields(record)

    assert sanitized["cases_confirmed"] == 2
    assert sanitized["deaths"] == 1
    assert "case_death_count_requires_explicit_label" in sanitized["semantic_warnings"]
    assert "numeric_semantics_rejected:cases_confirmed" not in sanitized["semantic_warnings"]
    assert "numeric_semantics_rejected:deaths" not in sanitized["semantic_warnings"]


def test_pending_and_quarantined_records_are_not_in_final_dataset(tmp_path):
    session = _session(tmp_path)
    _write_json(session / "collection" / "final_dataset.json", [])
    _write_json(
        session / "collection" / "pending_review_records.json",
        [{"record_id": "pending_1", "review_reason": "publisher_unknown"}],
    )
    _write_json(
        session / "collection" / "quarantined_records.json",
        [{"record_id": "quarantined_1", "blocking_reason": "unsupported_numeric_claim"}],
    )

    summary = build_collection_readiness_summary(session)

    assert summary["accepted_records"] == 0
    assert summary["pending_review_records"] == 1
    assert summary["quarantined_records"] == 1


def test_evaluation_scaffold_toy_benchmark_emits_benchmark_aliases(tmp_path):
    collection = {
        "record_id": "rec_collection",
        "linked_event_id": "event_1",
        "disease": "Hantavirus disease",
        "country": "United States",
        "subnational_location": "New Mexico",
        "reporting_period": "2024",
        "statistical_count_type": "annual",
        "cases_confirmed": 1,
        "deaths": 0,
        "source_id": "src_collection",
        "source_url": "https://www.nmhealth.org/report",
        "evidence_quote": "One confirmed case was reported in 2024.",
        "supporting_chunk_id": "chunk_1",
    }
    benchmark = {
        **collection,
        "record_id": "rec_benchmark",
        "source_id": "src_benchmark",
        "source_url": "https://benchmark.example/record",
        "evidence_quote": "Benchmark confirms one case in 2024.",
    }

    rows, summary = build_evaluation_report(
        collection_records=[collection],
        validation_records=[benchmark],
        collection_source_registry=[],
        reserved_source_ids={"src_benchmark"},
    )
    outputs = write_evaluation_outputs(rows, summary, tmp_path / "evaluation")

    assert Path(outputs["evaluation_report_csv"]).exists()
    assert rows[0]["benchmark_case_count"] == "1"
    assert rows[0]["benchmark_source_urls"] == "https://benchmark.example/record"
    assert rows[0]["overall_match_status"] == "match"


def test_final_report_displays_collection_readiness_summary(tmp_path):
    session = _session(tmp_path)
    write_collection_readiness_outputs(session, case_study_real_mode=True)

    facts = build_report_facts(session)
    assert facts["collection_readiness"]["collection_readiness_status"]
    assert facts["collection_readiness"]["case_study_real_mode"] is True
