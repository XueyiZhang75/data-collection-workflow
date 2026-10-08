from __future__ import annotations

import csv
import json
from pathlib import Path

from data_collection_workflow.reporting.report_facts import build_report_facts


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _record(**overrides) -> dict:
    row = {
        "record_id": "rec_accepted_1",
        "source_id": "src_known",
        "disease": "hantavirus",
        "location": "Virginia",
        "date_reported": "2025-03-01",
        "reporting_period": "2025 week 09",
        "cases_confirmed": 2,
        "deaths": 0,
        "statistical_count_type": "weekly",
        "source_name": "Virginia Department of Health weekly report",
        "source_title": "Virginia Department of Health weekly report",
        "source_url": "https://example.test/vdh-weekly",
        "actual_publisher": "Virginia Department of Health",
        "quality_status": "accepted_with_warnings",
        "record_final_inclusion_status": "accepted_with_warnings",
        "evidence_quote": "Two confirmed hantavirus cases were reported in week 09.",
        "corroboration_status": "official_single_source",
    }
    row.update(overrides)
    return row


def _source(**overrides) -> dict:
    row = {
        "source_id": "src_known",
        "title": "Virginia Department of Health weekly report",
        "publisher": "Virginia Department of Health",
        "actual_publisher": "Virginia Department of Health",
        "source_type_final": "state_or_local_public_health_agency",
        "source_role": "collection",
        "canonical_url": "https://example.test/vdh-weekly",
        "fetch_status": "success",
        "final_screening_decision": "include",
        "ready_for_content_fetch": True,
    }
    row.update(overrides)
    return row


def _make_session(
    tmp_path: Path,
    *,
    final_records: list[dict] | None = None,
    pre_quality_records: list[dict] | None = None,
    pending_records: list[dict] | None = None,
    quarantined_records: list[dict] | None = None,
    raw_records: list[dict] | None = None,
    source_registry: list[dict] | None = None,
    source_identity_summary: dict | None = None,
    run_quality_status: str = "passed",
    validation_limited: bool = False,
    write_evaluation: bool = True,
) -> Path:
    session = tmp_path / "session"
    collection = session / "collection"
    diagnostics = session / "diagnostics"
    evaluation = session / "evaluation"
    final_records = list(final_records if final_records is not None else [_record()])
    pre_quality_records = list(
        pre_quality_records if pre_quality_records is not None else final_records
    )
    pending_records = list(pending_records or [])
    quarantined_records = list(quarantined_records or [])
    raw_records = list(raw_records if raw_records is not None else pre_quality_records)
    source_registry = list(source_registry if source_registry is not None else [_source()])
    run_summary = {
        "session_id": "session",
        "user_request": "Collect hantavirus cases, deaths, dates, locations, source URLs, source types, and evidence quotes for Virginia from 2025-03-01 to 2025-03-07.",
        "live_search_enabled": True,
        "live_fetch_enabled": True,
        "run_quality_status": run_quality_status,
        "task_disease": "hantavirus",
        "task_location": "Virginia",
        "task_start_date": "2025-03-01",
        "task_end_date": "2025-03-07",
        "source_registry_count": len(source_registry),
        "document_count": 1,
        "normalized_record_count": len(pre_quality_records),
        "evaluation_row_count": 1 if write_evaluation else 0,
    }
    run_quality = {
        "run_quality_status": run_quality_status,
        "final_dataset_mode": "task_aware_quality_gated_records",
        "task_disease": "hantavirus",
        "task_location": "Virginia",
        "task_start_date": "2025-03-01",
        "task_end_date": "2025-03-07",
        "accepted_record_count": len(final_records),
        "final_dataset_count": len(final_records),
        "final_dataset_pre_quality_gate_count": len(pre_quality_records),
        "pending_review_record_count": len(pending_records),
        "quarantined_record_count": len(quarantined_records),
        "validation_limited": validation_limited,
        "collection_decision_summary": {
            "quarantine_reason_counts": {
                "numeric claim lacks evidence": len(quarantined_records)
            }
            if quarantined_records
            else {},
        },
    }
    source_identity_summary = source_identity_summary or {
        "identity_assessed_count": len(source_registry),
        "unknown_publisher_count": 0,
        "source_type_counts": {"state_or_local_public_health_agency": len(source_registry)},
    }
    package = {
        "final_dataset": final_records,
        "final_dataset_pre_quality_gate": pre_quality_records,
        "pending_review_records": pending_records,
        "quarantined_records": quarantined_records,
        "source_registry": source_registry,
        "source_identity_summary": source_identity_summary,
        "run_quality_summary": run_quality,
        "human_review_items": [],
    }
    _write_json(session / "workflow_run_summary.json", run_summary)
    _write_json(collection / "final_package.json", package)
    for name, rows in {
        "final_dataset": final_records,
        "final_dataset_pre_quality_gate": pre_quality_records,
        "pending_review_records": pending_records,
        "quarantined_records": quarantined_records,
        "raw_records": raw_records,
        "normalized_records": pre_quality_records,
        "validated_records": pre_quality_records,
        "source_registry": source_registry,
        "source_identity_assessments": source_registry,
        "evidence_chunks": [
            {
                "chunk_id": "chunk_1",
                "source_id": "src_known",
                "case_signal": True,
                "death_signal": False,
                "date_signal": True,
                "location_signal": True,
            }
        ],
        "human_review_items": [],
    }.items():
        _write_json(collection / f"{name}.json", rows)
        _write_json(diagnostics / f"{name}.json", rows)
    for name, value in {
        "run_quality_summary": run_quality,
        "source_identity_summary": source_identity_summary,
        "content_fetch_summary": {
            "fetch_status_counts": {"success": 1, "failed": 0},
            "fetch_failures_blocking_count": 0,
        },
        "corroboration_summary": {
            "multi_source_corroborated_count": 0,
            "official_single_source_count": len(final_records),
            "single_source_unverified_count": 0,
            "conflicting_claim_count": 0,
        },
    }.items():
        _write_json(diagnostics / f"{name}.json", value)
    if write_evaluation:
        _write_csv(
            evaluation / "evaluation_report.csv",
            [{"record_id": "rec_accepted_1", "match_status": "not_evaluated"}],
        )
    return session


def test_final_report_counts_accepted_records_when_present(tmp_path):
    session = _make_session(tmp_path, final_records=[_record()])
    facts = build_report_facts(session)
    assert facts["executive_summary"]["accepted_records_count"] == 1


def test_final_report_preserves_original_multilingual_evidence(tmp_path):
    quote = 'Two confirmed hantavirus cases were reported. 原文记录保留。'
    session = _make_session(tmp_path, final_records=[_record(evidence_quote=quote)])
    facts = build_report_facts(session)
    assert facts['final_statistics']['accepted_primary_dataset'][0]['evidence_quote'] == quote


def test_final_report_uses_untruncated_export_reviewable_total_and_shows_page_size(
    tmp_path,
):
    pending = [
        _record(
            record_id=f"rec_pending_{index:02d}",
            record_final_inclusion_status="pending_human_review",
            quality_status="pending_review",
        )
        for index in range(34)
    ]
    quarantined = _record(
        record_id="rec_quarantined_34",
        record_final_inclusion_status="quarantined_unsupported_numeric_claim",
        quality_status="quarantined",
    )
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[*pending, quarantined],
        pending_records=pending,
        quarantined_records=[quarantined],
        run_quality_status="reviewable_evidence_only",
    )
    _write_json(
        session / "collection" / "evidence_product_dataset.json",
        [
            {
                "evidence_id": f"evidence_{row['record_id']}",
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "quality_status": row["quality_status"],
                "evidence_quote": row["evidence_quote"],
            }
            for row in [*pending, quarantined]
        ],
    )
    facts = build_report_facts(session)
    assert facts["executive_summary"]["reviewable_records_count"] == 35
    assert facts["readable_outputs"]["reviewable_evidence_total_count"] == 35
    assert len(facts["readable_outputs"]["reviewable_evidence_matrix"]) == 30


def test_final_report_surfaces_source_to_evidence_and_disease_local_summaries(tmp_path):
    quarantined = _record(
        record_id="rec_h5n6_wrong_disease",
        cases_confirmed=1,
        disease="hantavirus",
        source_id="src_govuk",
        source_name="GOV.UK outbreaks under monitoring",
        source_url="https://example.test/govuk-monitoring",
        record_final_inclusion_status="quarantined_source_not_task_relevant",
        quarantine_reason="local_evidence_disease_mismatch",
        quality_gate_reasons=["local_evidence_disease_mismatch"],
        record_local_disease_relevance_status="incompatible_disease",
        record_local_numeric_disease_status="incompatible_disease",
        record_local_numeric_target_terms_found=[],
        record_local_numeric_incompatible_terms_found=["H5N6"],
        evidence_quote="One confirmed human case of avian influenza A(H5N6) was reported.",
    )
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[quarantined],
        quarantined_records=[quarantined],
        run_quality_status="no_primary_case_dataset_records",
    )
    _write_json(
        session / "collection" / "source_inventory.json",
        [
            {
                "source_id": "src_sante",
                "source_name": "Sante live outbreak page",
                "source_type_final": "national_public_health_authority",
                "source_product_type": "event_outbreak_report",
                "source_url": "https://example.test/sante-live",
                "task_specificity": "event_specific",
                "authority_score": 0.92,
                "fetch_status": "success",
                "parse_status": "success",
                "source_to_evidence_status": "parsed_target_source_no_record_extracted",
                "target_chunk_count": 2,
                "target_record_count": 0,
                "source_only_reason": "no_extracted_records",
                "extraction_failure_substage": "record_extraction",
            }
        ],
    )
    _write_json(
        session / "collection" / "case_candidate_dataset.json",
        [
            {
                "case_candidate_id": "cand_reviewable_1",
                "record_id": "rec_reviewable_1",
                "source_id": "src_known",
                "case_candidate_status": "reviewable",
                "field_completeness_score": 0.31,
                "missing_key_fields": "age;gender;date_onset;outcome",
            }
        ],
    )
    facts = build_report_facts(session)
    outputs = facts['readable_outputs']
    assert outputs['source_to_evidence_funnel']['source_to_evidence_status_counts']['parsed_target_source_no_record_extracted'] == 1
    assert outputs['high_confidence_sources_no_record'][0]['source_id'] == 'src_sante'
    assert outputs['disease_local_rejection_counts']['local_evidence_disease_mismatch'] == 1
    assert outputs['case_field_completeness_summary']['missing_key_fields_top']['age'] == 1


def test_final_report_surfaces_exact_page_empty_recovery_and_case_span_coverage(
    tmp_path,
):
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[],
        quarantined_records=[],
        run_quality_status="reviewable_evidence_only",
    )
    _write_json(
        session / "diagnostics" / "source_search_execution_summary.json",
        {
            "source_recall_target_ledger": [
                {
                    "authority_domain": "who.int",
                    "source_class": "official_public_health",
                    "event_page_status": "event_page_found",
                },
                {
                    "authority_domain": "sante.gouv.fr",
                    "source_class": "official_public_health",
                    "event_page_status": "domain_found_event_page_missing",
                },
                {
                    "authority_domain": "pathoplexus.org",
                    "source_class": "structured_database",
                    "event_page_status": "query_executed_no_result",
                },
            ]
        },
    )
    _write_json(
        session / "diagnostics" / "structured_extraction_summary.json",
        {
            "llm_call_count": 12,
            "primary_llm_call_count": 9,
            "llm_success_count": 12,
            "llm_transport_success_count": 12,
            "llm_empty_output_count": 5,
            "llm_non_empty_output_count": 7,
            "valid_record_count": 9,
            "unique_valid_record_count": 8,
            "llm_error_count": 0,
            "llm_empty_output_reasons": {
                "legitimate_no_record_context": 2,
                "llm_empty_strong_signal": 3,
            },
            "deterministic_recovery_succeeded_count": 1,
            "focused_recovery_call_count": 3,
            "focused_retry_succeeded_count": 2,
            "focused_retry_empty_count": 1,
            "focused_recovery_valid_record_count": 3,
            "focused_recovery_field_gain_count": 7,
            "metric_row_batch_deferred_for_case_coverage_count": 4,
            "extraction_budget_ledger": {
                "soft_primary_calls": 8,
                "hard_primary_calls": 16,
                "soft_cap_extension_call_count": 1,
                "total_call_count": 12,
            },
            "focused_recovery_status_by_source": {
                "src_who": "focused_retry_succeeded",
                "src_sante": "focused_retry_empty",
            },
        },
    )
    _write_json(
        session / "collection" / "source_inventory.json",
        [
            {
                "source_id": "src_who",
                "source_type_final": "international_public_health_authority",
                "source_url": "https://who.int/event-report",
                "source_exact_page_status": "event_page_found",
                "fetch_status": "success",
                "parse_status": "success",
                "source_to_evidence_status": "evidence_extracted",
                "target_chunk_count": 2,
                "target_record_count": 1,
            },
            {
                "source_id": "src_sante",
                "source_type_final": "national_public_health_authority",
                "source_url": "https://sante.gouv.fr/background",
                "source_exact_page_status": "domain_found_event_page_missing",
                "fetch_status": "success",
                "parse_status": "success",
                "source_to_evidence_status": "parsed_target_source_no_record_extracted",
                "target_chunk_count": 1,
                "target_record_count": 0,
            },
        ],
    )
    _write_json(
        session / "collection" / "case_candidate_dataset.json",
        [
            {
                "case_candidate_id": "cand_case_4",
                "source_id": "src_who",
                "case_span_id": "span_case_4",
                "case_span_quote": "Case 4 was confirmed with Andes virus.",
                "field_provenance_json": json.dumps(
                    {
                        "workflow_case_label": {
                            "value": "Case 4",
                            "case_span_id": "span_case_4",
                        }
                    }
                ),
                "unsupported_case_fields": "",
                "field_completeness_score": 0.5,
                "missing_key_fields": "age;nationality",
            },
            {
                "case_candidate_id": "cand_aggregate",
                "source_id": "src_who",
                "field_completeness_score": 0.2,
                "missing_key_fields": "age;gender;nationality",
            },
        ],
    )
    facts = build_report_facts(session)
    outputs = facts["readable_outputs"]
    exact = outputs["high_confidence_exact_page_recall"]
    recovery = outputs["llm_empty_output_recovery"]
    spans = outputs["case_span_extraction_coverage"]
    assert exact["recall_target_count"] == 3
    assert exact["event_page_found_count"] == 1
    assert exact["domain_found_event_page_missing_count"] == 1
    assert exact["query_executed_no_result_count"] == 1
    assert recovery["llm_empty_output_rate"] == 0.417
    assert recovery["recovered_output_count"] == 3
    assert recovery["focused_retry_empty_count"] == 1
    assert recovery["actual_model_call_count"] == 12
    assert recovery["non_empty_call_count"] == 7
    assert recovery["valid_record_count"] == 9
    assert recovery["unique_valid_record_count"] == 8
    assert recovery["focused_recovery_field_gain_count"] == 7
    assert recovery["soft_cap_extension_call_count"] == 1
    assert spans["candidate_rows"] == 2
    assert spans["rows_with_case_span_evidence"] == 1
    assert spans["rows_with_field_provenance"] == 1


def test_final_report_distinguishes_official_single_source_not_cross_validated(tmp_path):
    session = _make_session(
        tmp_path,
        final_records=[
            _record(
                corroboration_status="single_source_unverified",
                source_type_final="state_or_local_public_health_agency",
            )
        ],
        source_identity_summary={
            "identity_assessed_count": 1,
            "unknown_publisher_count": 0,
            "source_type_counts": {"state_or_local_public_health_agency": 1},
        },
    )
    _write_json(
        session / "diagnostics" / "corroboration_summary.json",
        {
            "multi_source_corroborated_count": 0,
            "official_single_source_count": 0,
            "single_source_unverified_count": 1,
            "conflicting_claim_count": 0,
        },
    )
    facts = build_report_facts(session)
    assert (
        facts["quality_and_trustworthiness"]["source_corroboration_status"][
            "official_single_source_not_cross_validated"
        ]
        == 1
    )


def test_final_report_exposes_primary_and_task_aware_dataset_counts(tmp_path):
    primary = _record(record_id="rec_primary", cases_confirmed=1)
    observation = _record(
        record_id="rec_observation",
        cases_confirmed=None,
        metric_name="Influenza percent positive",
        metric_value=7.2,
        metric_unit="percent",
        observation_type="surveillance_summary",
        primary_case_dataset_eligible=False,
        record_final_inclusion_status="quarantined_non_primary_observation",
    )
    session = _make_session(
        tmp_path,
        final_records=[primary],
        pre_quality_records=[primary, observation],
        quarantined_records=[observation],
    )
    _write_json(session / "collection" / "primary_case_dataset.json", [primary])
    _write_json(session / "collection" / "final_case_dataset.json", [primary])
    _write_json(
        session / "collection" / "task_aware_observation_dataset.json",
        [observation],
    )
    _write_json(session / "collection" / "non_primary_observations.json", [observation])
    _write_json(session / "collection" / "reviewable_dataset.json", [observation])
    facts = build_report_facts(session)
    assert facts["executive_summary"]["accepted_primary_case_records_count"] == 1
    assert facts["executive_summary"]["task_aware_observation_records_count"] == 1
    assert facts["collection_funnel"]["accepted_primary_case_records"] == 1
    assert facts["collection_funnel"]["accepted_task_aware_observations"] == 1


def test_final_report_empty_final_dataset_with_pre_quality_records_is_quality_gate_not_program_failure(tmp_path):
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[_record(record_id="rec_candidate")],
        raw_records=[_record(record_id="rec_candidate")],
        run_quality_status="failed_quality_gate",
    )
    facts = build_report_facts(session)
    assert facts['executive_summary']['real_collection_status'] == 'failed_quality_gate'
    assert facts['executive_summary']['suitable_as_final_epidemiological_dataset'] == 'no'


def test_final_report_distinguishes_non_primary_observations_only(tmp_path):
    observation = _record(
        record_id="rec_observation",
        cases_confirmed=None,
        observation_type="surveillance_summary",
        primary_case_dataset_eligible=False,
        record_final_inclusion_status="quarantined_non_primary_observation",
        evidence_quote="Official surveillance indicator was reported.",
    )
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[observation],
        raw_records=[observation],
        quarantined_records=[observation],
        run_quality_status="no_primary_case_dataset_records",
    )
    _write_json(session / "collection" / "primary_case_dataset.json", [])
    _write_json(session / "collection" / "final_case_dataset.json", [])
    _write_json(
        session / "collection" / "task_aware_observation_dataset.json",
        [observation],
    )
    _write_json(session / "collection" / "non_primary_observations.json", [observation])
    facts = build_report_facts(session)
    assert facts["executive_summary"]["real_collection_status"] == (
        "non_primary_observations_only"
    )


def test_final_report_no_raw_records_reports_no_records_extracted(tmp_path):
    session = _make_session(
        tmp_path,
        final_records=[],
        pre_quality_records=[],
        raw_records=[],
        run_quality_status="no_records_extracted",
    )
    facts = build_report_facts(session)
    assert facts['executive_summary']['real_collection_status'] == 'no_records_extracted'


def test_final_report_validation_limited_does_not_claim_validated(tmp_path):
    session = _make_session(
        tmp_path,
        validation_limited=True,
        run_quality_status="validation_limited_no_compatible_source",
        write_evaluation=False,
    )
    facts = build_report_facts(session)
    assert facts["validation_readiness"]["evaluation_rows_count"] == "not_available"
    assert facts['validation_readiness']['ready_for_benchmark_comparison'] == 'no'


def test_final_report_displays_publisher_unknown_count(tmp_path):
    session = _make_session(
        tmp_path,
        source_registry=[
            _source(
                source_id="src_unknown",
                publisher="unknown",
                actual_publisher="unknown",
                source_type_final="unknown",
            )
        ],
        source_identity_summary={
            "identity_assessed_count": 1,
            "unknown_publisher_count": 1,
            "source_type_counts": {"unknown": 1},
        },
    )
    facts = build_report_facts(session)
    assert facts['data_source_summary']['publisher_unknown_count'] == 1


def test_final_report_pending_review_records_are_not_counted_as_accepted(tmp_path):
    pending = _record(
        record_id="rec_pending_99",
        cases_confirmed=99,
        record_final_inclusion_status="pending_human_review",
        quality_status="pending_review",
    )
    session = _make_session(
        tmp_path,
        final_records=[_record(record_id="rec_accepted_1", cases_confirmed=2)],
        pre_quality_records=[_record(record_id="rec_accepted_1", cases_confirmed=2), pending],
        pending_records=[pending],
    )
    facts = build_report_facts(session)
    assert facts["executive_summary"]["accepted_records_count"] == 1
    assert facts["executive_summary"]["pending_review_records_count"] == 1
    assert all(row["record_id"] != "rec_pending_99" for row in facts["final_statistics"]["accepted_primary_dataset"])


def test_final_report_shows_quarantine_reasons(tmp_path):
    session = _make_session(
        tmp_path,
        quarantined_records=[
            _record(
                record_id="rec_quarantined",
                quality_status="quarantined",
                record_final_inclusion_status="quarantined_unsupported_numeric_claim",
                exclusion_reason="numeric claim lacks evidence",
            )
        ],
        run_quality_status="partial_with_quarantined_records",
    )
    facts = build_report_facts(session)
    assert facts['quality_and_trustworthiness']['main_quarantine_reasons']['numeric claim lacks evidence'] == 1


def test_final_report_does_not_aggregate_mixed_statistical_count_types(tmp_path):
    session = _make_session(
        tmp_path,
        final_records=[
            _record(record_id="rec_weekly", cases_confirmed=2, statistical_count_type="weekly"),
            _record(record_id="rec_cumulative", cases_confirmed=10, statistical_count_type="cumulative"),
        ],
    )
    facts = build_report_facts(session)
    aggregation = facts['final_statistics']['aggregation']
    assert aggregation['status'] == 'not aggregated because count types are not comparable'
    assert aggregation['aggregate_cases'] == 'not_available'


def test_final_report_counts_match_source_artifacts(tmp_path):
    pending = _record(record_id="rec_pending", quality_status="pending_review")
    quarantined = _record(record_id="rec_quarantined", quality_status="quarantined")
    session = _make_session(
        tmp_path,
        final_records=[_record(record_id="rec_accepted")],
        pre_quality_records=[_record(record_id="rec_accepted"), pending, quarantined],
        pending_records=[pending],
        quarantined_records=[quarantined],
        raw_records=[_record(record_id="raw_1"), _record(record_id="raw_2")],
    )
    facts = build_report_facts(session)
    assert facts["collection_funnel"]["raw_records"] == 2
    assert facts["collection_funnel"]["pre_quality_records"] == 3
    assert facts["collection_funnel"]["accepted_records"] == 1
    assert facts["collection_funnel"]["pending_review_records"] == 1
    assert facts["collection_funnel"]["quarantined_records"] == 1


def test_final_report_recomputes_final_reviewable_and_quarantine_from_package_state(
    tmp_path,
):
    accepted = _record(record_id="rec_accepted")
    pending = _record(record_id="rec_pending", quality_status="pending_review")
    quarantined = _record(record_id="rec_quarantined", quality_status="quarantined")
    session = _make_session(
        tmp_path,
        final_records=[accepted],
        pre_quality_records=[accepted, pending, quarantined],
        pending_records=[pending],
        quarantined_records=[quarantined],
    )
    stale_final_rows = [accepted, _record(record_id="stale_duplicate")]
    _write_json(session / "collection" / "final_dataset.json", stale_final_rows)
    _write_json(session / "diagnostics" / "final_dataset.json", stale_final_rows)
    facts = build_report_facts(session)
    assert facts["executive_summary"]["accepted_records_count"] == 1
    assert facts["executive_summary"]["final_case_records_count"] == 1
    assert facts["executive_summary"]["reviewable_records_count"] == 2
    assert facts["executive_summary"]["quarantined_records_count"] == 1
    assert facts["collection_funnel"]["accepted_records"] == 1
    assert facts["collection_funnel"]["pending_review_records"] == 1
    assert facts["collection_funnel"]["quarantined_records"] == 1


def test_final_report_separates_all_accepted_records_from_final_case_records(tmp_path):
    primary = _record(record_id="rec_primary")
    accepted_context = _record(
        record_id="rec_context",
        observation_type="aggregate_outbreak_snapshot",
    )
    session = _make_session(
        tmp_path,
        final_records=[primary, accepted_context],
    )
    package_path = session / "collection" / "final_package.json"
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["primary_case_dataset"] = [primary]
    package["final_case_dataset"] = [primary]
    _write_json(package_path, package)
    facts = build_report_facts(session)
    assert facts["executive_summary"]["accepted_records_count"] == 2
    assert facts["executive_summary"]["accepted_primary_case_records_count"] == 1
    assert facts["executive_summary"]["final_case_records_count"] == 1
    assert facts["collection_funnel"]["accepted_records"] == 2
    assert facts["collection_funnel"]["accepted_primary_case_records"] == 1


def test_final_report_does_not_invent_source_url_or_evidence_quote(tmp_path):
    session = _make_session(
        tmp_path,
        final_records=[
            _record(
                record_id="rec_missing_provenance",
                source_url="",
                evidence_quote="",
            )
        ],
        source_registry=[_source(canonical_url="")],
        write_evaluation=False,
    )
    facts = build_report_facts(session)
    row = facts['final_statistics']['accepted_primary_dataset'][0]
    assert row['record_id'] == 'rec_missing_provenance'
    assert row['source_url'] == 'not_available'
    assert row['evidence_quote'] == 'not_available'
    assert 'missing_source_url' in json.dumps(facts['final_statistics']['provenance_warnings'])
    assert 'missing_evidence_quote' in json.dumps(facts['final_statistics']['provenance_warnings'])


def test_final_report_counts_data_source_role_as_collection_allowed_and_uses_document_count_fallback(tmp_path):
    session = _make_session(
        tmp_path,
        source_registry=[
            _source(source_id="src_data", source_role="data_source", ready_for_content_fetch=True)
        ],
    )
    facts = build_report_facts(session)
    assert facts["collection_funnel"]["collection_allowed_sources"] == 1
    assert facts["data_source_summary"]["fetched_source_count"] == 1


def test_final_report_zero_row_evaluation_file_reports_zero_rows_not_missing(tmp_path):
    session = _make_session(tmp_path, write_evaluation=False)
    evaluation_path = session / "evaluation" / "evaluation_report.csv"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text("evaluation_row_id,record_id\n", encoding="utf-8")
    facts = build_report_facts(session)
    assert facts["validation_readiness"]["evaluation_rows_count"] == 0
    assert facts["validation_readiness"]["reason_if_not_ready"] == (
        "evaluation_report.csv has zero rows"
    )


def test_final_report_masking_checker_rows_do_not_claim_open_run_was_masked(tmp_path):
    session = _make_session(tmp_path, write_evaluation=False)
    _write_csv(
        session / "evaluation" / "evaluation_report.csv",
        [
            {
                "evaluation_row_id": "eval_001",
                "record_id": "rec_accepted_1",
                "masking_compliance_status": "passed",
            }
        ],
    )
    facts = build_report_facts(session)
    assert (
        facts["validation_readiness"]["github_benchmark_visible_or_masked"]
        == "masking_checked_no_leakage"
    )
