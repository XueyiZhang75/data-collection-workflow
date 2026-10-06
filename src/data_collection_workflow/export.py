"""Deterministic export utilities for the final data package.

No LangGraph, no LLM, no network. Writes JSON and CSV artifacts under a
user-supplied output directory.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import csv
import json
from pathlib import Path

from .config import load_final_package_policy
from .models import FinalPackagePolicy
from .workflow_line_list_export import (
    WORKFLOW_CASE_BUNDLE_LINE_LIST_FIELDS,
    WORKFLOW_CASE_CANDIDATE_LINE_LIST_FIELDS,
    build_workflow_case_bundle_line_list,
    build_workflow_case_candidate_line_list,
)


DEFAULT_FINAL_PACKAGE_MODE = "compact"
FINAL_PACKAGE_MODES = {"compact", "full_audit"}
COMPACT_EXTERNALIZED_SECTIONS = {
    "claim_comparisons",
    "content_chunks",
    "evidence_chunks",
    "model_traces",
    "raw_chunks",
    "raw_records",
}


def ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def write_csv_rows(
    rows: list[dict],
    path: str | Path,
    field_order: list[str] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(field_order or [])

    if not rows:
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered)
            writer.writeheader()
        return p

    extras: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            if key not in ordered:
                extras.add(key)
    fieldnames = ordered + sorted(extras)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            cleaned: dict = {}
            for key in fieldnames:
                value = row.get(key)
                if value is None:
                    cleaned[key] = ""
                elif isinstance(value, (list, dict)):
                    cleaned[key] = json.dumps(value, ensure_ascii=False)
                else:
                    cleaned[key] = value
            writer.writerow(cleaned)
    return p


def _final_package_mode(package_mode: str | None = None) -> str:
    value = str(
        package_mode
        or get_env("FINAL_PACKAGE_MODE")
        or get_env("PACKAGE_MODE")
        or ""
    ).strip().lower()
    if value in FINAL_PACKAGE_MODES:
        return value
    return DEFAULT_FINAL_PACKAGE_MODE


def _package_for_final_json(package: dict, package_mode: str, manifest: dict) -> dict:
    final_package = dict(package)
    final_package["package_mode"] = package_mode
    final_package["artifact_manifest"] = {
        "files": dict(manifest["files"]),
        "section_counts": dict(manifest["section_counts"]),
    }
    if package_mode == "compact":
        for section in COMPACT_EXTERNALIZED_SECTIONS:
            final_package.pop(section, None)
    return final_package


def export_final_data_package(
    package: dict,
    output_dir: str | Path,
    package_mode: str | None = None,
) -> dict:
    """Write the final data package to JSON + CSV files. Returns a manifest."""

    policy = FinalPackagePolicy(**load_final_package_policy())
    out_dir = ensure_output_dir(output_dir)
    package = dict(package)
    package_mode = _final_package_mode(package_mode)

    primary_case_dataset = list(
        package.get("primary_case_dataset")
        if package.get("primary_case_dataset") is not None
        else package.get("final_case_dataset") or []
    )
    task_aware_observation_dataset = list(
        package.get("task_aware_observation_dataset")
        if package.get("task_aware_observation_dataset") is not None
        else package.get("non_primary_observations") or []
    )
    reviewable_dataset = list(
        package.get("reviewable_dataset")
        if package.get("reviewable_dataset") is not None
        else (package.get("pending_review_records") or [])
        + (package.get("quarantined_records") or [])
    )
    package.setdefault("primary_case_dataset", primary_case_dataset)
    package.setdefault(
        "task_aware_observation_dataset", task_aware_observation_dataset
    )
    package.setdefault("reviewable_dataset", reviewable_dataset)
    if not package.get("workflow_case_bundle_line_list"):
        package["workflow_case_bundle_line_list"] = build_workflow_case_bundle_line_list(
            case_evidence_bundles=package.get("case_evidence_bundles") or [],
            case_candidate_dataset=package.get("case_candidate_dataset") or [],
            source_inventory=package.get("source_inventory")
            or package.get("source_registry")
            or [],
        )
    if not package.get("workflow_case_candidate_line_list"):
        package["workflow_case_candidate_line_list"] = (
            build_workflow_case_candidate_line_list(
                case_candidate_dataset=package.get("case_candidate_dataset") or [],
                case_evidence_bundles=package.get("case_evidence_bundles") or [],
                source_inventory=package.get("source_inventory")
                or package.get("source_registry")
                or [],
            )
        )

    files: dict[str, str] = {}

    files["final_dataset_json"] = str(
        write_json(package.get("final_dataset") or [], out_dir / "final_dataset.json")
    )
    files["final_dataset_csv"] = str(
        write_csv_rows(
            package.get("final_dataset") or [],
            out_dir / "final_dataset.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    files["final_dataset_pre_quality_gate_json"] = str(
        write_json(
            package.get("final_dataset_pre_quality_gate") or [],
            out_dir / "final_dataset_pre_quality_gate.json",
        )
    )
    files["final_dataset_pre_quality_gate_csv"] = str(
        write_csv_rows(
            package.get("final_dataset_pre_quality_gate") or [],
            out_dir / "final_dataset_pre_quality_gate.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    files["final_dataset_post_review_json"] = str(
        write_json(
            package.get("final_dataset_post_review") or [],
            out_dir / "final_dataset_post_review.json",
        )
    )
    files["final_dataset_post_review_csv"] = str(
        write_csv_rows(
            package.get("final_dataset_post_review") or [],
            out_dir / "final_dataset_post_review.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    files["quarantined_records_json"] = str(
        write_json(
            package.get("quarantined_records") or [],
            out_dir / "quarantined_records.json",
        )
    )
    files["quarantined_records_csv"] = str(
        write_csv_rows(
            package.get("quarantined_records") or [],
            out_dir / "quarantined_records.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    files["pending_review_records_json"] = str(
        write_json(
            package.get("pending_review_records") or [],
            out_dir / "pending_review_records.json",
        )
    )
    files["pending_review_records_csv"] = str(
        write_csv_rows(
            package.get("pending_review_records") or [],
            out_dir / "pending_review_records.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    files["non_primary_observations_json"] = str(
        write_json(
            package.get("non_primary_observations") or [],
            out_dir / "non_primary_observations.json",
        )
    )
    files["non_primary_observations_csv"] = str(
        write_csv_rows(
            package.get("non_primary_observations") or [],
            out_dir / "non_primary_observations.csv",
            field_order=policy.final_dataset_field_order,
        )
    )
    professor_facing_sections = {
        "primary_case_dataset": primary_case_dataset,
        "task_aware_observation_dataset": task_aware_observation_dataset,
        "reviewable_dataset": reviewable_dataset,
    }
    for section, rows in professor_facing_sections.items():
        files[f"{section}_json"] = str(write_json(rows, out_dir / f"{section}.json"))
        files[f"{section}_csv"] = str(
            write_csv_rows(
                rows,
                out_dir / f"{section}.csv",
                field_order=policy.final_dataset_field_order,
            )
        )
    observation_dataset_sections = {
        "final_case_dataset": package.get("final_case_dataset") or [],
        "global_outbreak_event_dataset": package.get(
            "global_outbreak_event_dataset"
        )
        or [],
        "regional_surveillance_dataset": package.get(
            "regional_surveillance_dataset"
        )
        or [],
        "country_year_aggregate_dataset": package.get(
            "country_year_aggregate_dataset"
        )
        or [],
        "official_alert_dataset": package.get("official_alert_dataset") or [],
        "probable_case_dataset": package.get("probable_case_dataset") or [],
        "suspected_case_dataset": package.get("suspected_case_dataset") or [],
        "unspecified_case_dataset": package.get("unspecified_case_dataset") or [],
        "death_dataset": package.get("death_dataset") or [],
        "hospitalization_dataset": package.get("hospitalization_dataset") or [],
        "zero_case_statements": package.get("zero_case_statements") or [],
        "exposure_monitoring_records": package.get("exposure_monitoring_records") or [],
        "surveillance_summary_records": package.get("surveillance_summary_records") or [],
        "outbreak_summary_records": package.get("outbreak_summary_records") or [],
        "context_records": package.get("context_records") or [],
        "best_available_context_records": package.get("best_available_context_records")
        or [],
        "unclassified_observation_records": package.get(
            "unclassified_observation_records"
        )
        or [],
    }
    for section, rows in observation_dataset_sections.items():
        files[f"{section}_json"] = str(write_json(rows, out_dir / f"{section}.json"))
        files[f"{section}_csv"] = str(
            write_csv_rows(
                rows,
                out_dir / f"{section}.csv",
                field_order=policy.final_dataset_field_order,
            )
        )
    files["observation_type_dataset_summary_json"] = str(
        write_json(
            package.get("observation_type_dataset_summary") or {},
            out_dir / "observation_type_dataset_summary.json",
        )
    )
    files["record_inclusion_decisions_json"] = str(
        write_json(
            package.get("record_inclusion_decisions") or [],
            out_dir / "record_inclusion_decisions.json",
        )
    )
    files["run_quality_summary_json"] = str(
        write_json(
            package.get("run_quality_summary") or {},
            out_dir / "run_quality_summary.json",
        )
    )
    files["collection_decision_summary_json"] = str(
        write_json(
            package.get("collection_decision_summary") or {},
            out_dir / "collection_decision_summary.json",
        )
    )
    files["final_dataset_quality_summary_json"] = str(
        write_json(
            package.get("final_dataset_quality_summary") or {},
            out_dir / "final_dataset_quality_summary.json",
        )
    )
    files["task_acceptance_contract_json"] = str(
        write_json(
            package.get("task_acceptance_contract") or {},
            out_dir / "task_acceptance_contract.json",
        )
    )
    files["task_evidence_contract_json"] = str(
        write_json(
            package.get("task_evidence_contract") or {},
            out_dir / "task_evidence_contract.json",
        )
    )
    files["evidence_strategy_plan_json"] = str(
        write_json(
            package.get("evidence_strategy_plan") or {},
            out_dir / "evidence_strategy_plan.json",
        )
    )
    files["source_triage_results_json"] = str(
        write_json(
            package.get("source_triage_results") or [],
            out_dir / "source_triage_results.json",
        )
    )
    files["evidence_chunks_json"] = str(
        write_json(
            package.get("evidence_chunks") or [],
            out_dir / "evidence_chunks.json",
        )
    )
    files["chunk_relevance_assessments_json"] = str(
        write_json(
            package.get("chunk_relevance_assessments") or [],
            out_dir / "chunk_relevance_assessments.json",
        )
    )
    files["record_task_fit_assessments_json"] = str(
        write_json(
            package.get("record_task_fit_assessments") or [],
            out_dir / "record_task_fit_assessments.json",
        )
    )
    files["direct_fast_path_summary_json"] = str(
        write_json(
            package.get("direct_fast_path_summary") or {},
            out_dir / "direct_fast_path_summary.json",
        )
    )
    files["metric_extraction_plan_json"] = str(
        write_json(
            package.get("metric_extraction_plan") or {},
            out_dir / "metric_extraction_plan.json",
        )
    )
    files["metric_row_extraction_audit_json"] = str(
        write_json(
            package.get("metric_row_extraction_audit") or [],
            out_dir / "metric_row_extraction_audit.json",
        )
    )
    files["records_excluded_by_human_review_json"] = str(
        write_json(
            package.get("records_excluded_by_human_review") or [],
            out_dir / "records_excluded_by_human_review.json",
        )
    )
    for section in (
        "source_inventory",
        "authority_source_inventory",
        "case_candidate_dataset",
        "reviewable_case_dataset",
        "case_evidence_bundles",
        "evidence_product_dataset",
        "workflow_case_bundle_line_list",
        "workflow_case_candidate_line_list",
    ):
        rows = package.get(section) or []
        files[f"{section}_json"] = str(write_json(rows, out_dir / f"{section}.json"))
        field_order = None
        if section == "workflow_case_bundle_line_list":
            field_order = WORKFLOW_CASE_BUNDLE_LINE_LIST_FIELDS
        elif section == "workflow_case_candidate_line_list":
            field_order = WORKFLOW_CASE_CANDIDATE_LINE_LIST_FIELDS
        files[f"{section}_csv"] = str(
            write_csv_rows(rows, out_dir / f"{section}.csv", field_order=field_order)
        )
    files["source_registry_json"] = str(
        write_json(package.get("source_registry") or [], out_dir / "source_registry.json")
    )
    files["source_identity_assessments_json"] = str(
        write_json(
            package.get("source_identity_assessments") or [],
            out_dir / "source_identity_assessments.json",
        )
    )
    files["source_identity_assessments_csv"] = str(
        write_csv_rows(
            package.get("source_identity_assessments") or [],
            out_dir / "source_identity_assessments.csv",
        )
    )
    files["source_identity_summary_json"] = str(
        write_json(
            package.get("source_identity_summary") or {},
            out_dir / "source_identity_summary.json",
        )
    )
    files["official_coverage_candidates_json"] = str(
        write_json(
            package.get("official_coverage_candidates") or [],
            out_dir / "official_coverage_candidates.json",
        )
    )
    files["source_coverage_requirements_json"] = str(
        write_json(
            package.get("source_coverage_requirements") or [],
            out_dir / "source_coverage_requirements.json",
        )
    )
    files["source_coverage_audit_json"] = str(
        write_json(
            package.get("source_coverage_audit") or {},
            out_dir / "source_coverage_audit.json",
        )
    )
    files["target_official_fetch_plan_json"] = str(
        write_json(
            package.get("target_official_fetch_plan") or [],
            out_dir / "target_official_fetch_plan.json",
        )
    )
    files["must_fetch_sources_json"] = str(
        write_json(
            package.get("must_fetch_sources") or [],
            out_dir / "must_fetch_sources.json",
        )
    )
    files["fetch_failures_blocking_json"] = str(
        write_json(
            package.get("fetch_failures_blocking") or [],
            out_dir / "fetch_failures_blocking.json",
        )
    )
    files["linked_events_json"] = str(
        write_json(package.get("linked_events") or [], out_dir / "linked_events.json")
    )
    files["event_clusters_json"] = str(
        write_json(package.get("event_clusters") or [], out_dir / "event_clusters.json")
    )
    files["duplicate_clusters_json"] = str(
        write_json(
            package.get("duplicate_clusters") or [],
            out_dir / "duplicate_clusters.json",
        )
    )
    files["validation_cases_json"] = str(
        write_json(
            package.get("validation_cases") or [],
            out_dir / "validation_cases.json",
        )
    )
    files["validation_comparisons_json"] = str(
        write_json(
            package.get("validation_comparisons") or [],
            out_dir / "validation_comparisons.json",
        )
    )
    files["validation_results_json"] = str(
        write_json(
            package.get("validation_results") or [],
            out_dir / "validation_results.json",
        )
    )
    files["validation_results_csv"] = str(
        write_csv_rows(
            package.get("validation_results") or [],
            out_dir / "validation_results.csv",
        )
    )
    claims = package.get("claims") or []
    claim_comparisons = package.get("claim_comparisons") or []
    corroborated_events = package.get("corroborated_events") or []
    corroborated_case_events = [
        row
        for row in corroborated_events
        if row.get("primary_case_dataset_eligible")
        and row.get("corroboration_status")
        in {"corroborated", "cross_source_supported"}
    ]
    conflicting_claim_ids = {
        str(claim_id)
        for event in corroborated_events
        for claim_id in (event.get("conflicting_claim_ids") or [])
    }
    supported_claim_ids = {
        str(claim_id)
        for event in corroborated_events
        if event.get("corroboration_status")
        in {"corroborated", "cross_source_supported"}
        for claim_id in (event.get("supporting_claim_ids") or [])
    }
    conflicting_claims = [
        claim for claim in claims if str(claim.get("claim_id")) in conflicting_claim_ids
    ]
    uncorroborated_claims = [
        claim
        for claim in claims
        if str(claim.get("claim_id")) not in supported_claim_ids
    ]
    files["claims_json"] = str(write_json(claims, out_dir / "claims.json"))
    files["claims_csv"] = str(write_csv_rows(claims, out_dir / "claims.csv"))
    files["claim_comparisons_json"] = str(
        write_json(claim_comparisons, out_dir / "claim_comparisons.json")
    )
    for section in sorted(COMPACT_EXTERNALIZED_SECTIONS - {"claim_comparisons"}):
        if section in package:
            files[f"{section}_json"] = str(
                write_json(package.get(section), out_dir / f"{section}.json")
            )
    files["corroborated_events_json"] = str(
        write_json(corroborated_events, out_dir / "corroborated_events.json")
    )
    files["corroborated_events_csv"] = str(
        write_csv_rows(corroborated_events, out_dir / "corroborated_events.csv")
    )
    files["corroborated_case_events_csv"] = str(
        write_csv_rows(
            corroborated_case_events,
            out_dir / "corroborated_case_events.csv",
        )
    )
    files["uncorroborated_claims_csv"] = str(
        write_csv_rows(uncorroborated_claims, out_dir / "uncorroborated_claims.csv")
    )
    files["conflicting_claims_csv"] = str(
        write_csv_rows(conflicting_claims, out_dir / "conflicting_claims.csv")
    )
    files["corroboration_summary_json"] = str(
        write_json(
            package.get("corroboration_summary") or {},
            out_dir / "corroboration_summary.json",
        )
    )
    files["anomaly_results_json"] = str(
        write_json(
            package.get("anomaly_results") or [],
            out_dir / "anomaly_results.json",
        )
    )
    files["anomaly_results_csv"] = str(
        write_csv_rows(
            package.get("anomaly_results") or [],
            out_dir / "anomaly_results.csv",
        )
    )
    files["applied_human_review_decisions_json"] = str(
        write_json(
            package.get("applied_human_review_decisions") or [],
            out_dir / "applied_human_review_decisions.json",
        )
    )
    files["rejected_human_review_decisions_json"] = str(
        write_json(
            package.get("rejected_human_review_decisions") or [],
            out_dir / "rejected_human_review_decisions.json",
        )
    )
    files["human_review_audit_trail_json"] = str(
        write_json(
            package.get("human_review_audit_trail") or [],
            out_dir / "human_review_audit_trail.json",
        )
    )
    files["human_review_application_summary_json"] = str(
        write_json(
            package.get("human_review_application_summary") or {},
            out_dir / "human_review_application_summary.json",
        )
    )
    files["conflicts_json"] = str(
        write_json(package.get("conflicts") or [], out_dir / "conflicts.json")
    )
    files["human_review_items_json"] = str(
        write_json(
            package.get("human_review_items") or [],
            out_dir / "human_review_items.json",
        )
    )
    files["collection_trace_json"] = str(
        write_json(package.get("collection_trace") or [], out_dir / "collection_trace.json")
    )
    files["workflow_summaries_json"] = str(
        write_json(
            package.get("workflow_summaries") or {},
            out_dir / "workflow_summaries.json",
        )
    )
    files["package_metadata_json"] = str(
        write_json(
            package.get("package_metadata") or {},
            out_dir / "package_metadata.json",
        )
    )
    files["provenance_manifest_json"] = str(
        write_json(
            package.get("provenance_manifest") or {},
            out_dir / "provenance_manifest.json",
        )
    )

    files["final_package_json"] = str(out_dir / "final_package.json")
    manifest = {
        "output_dir": str(out_dir),
        "package_mode": package_mode,
        "files": files,
        "section_counts": {
            "final_dataset": len(package.get("final_dataset") or []),
            "final_dataset_pre_quality_gate": len(
                package.get("final_dataset_pre_quality_gate") or []
            ),
            "final_dataset_post_review": len(
                package.get("final_dataset_post_review") or []
            ),
            "quarantined_records": len(package.get("quarantined_records") or []),
            "pending_review_records": len(package.get("pending_review_records") or []),
            "non_primary_observations": len(
                package.get("non_primary_observations") or []
            ),
            **{
                section: len(rows)
                for section, rows in professor_facing_sections.items()
            },
            **{
                section: len(rows)
                for section, rows in observation_dataset_sections.items()
            },
            "observation_type_dataset_summary": (
                1 if package.get("observation_type_dataset_summary") else 0
            ),
            "record_inclusion_decisions": len(
                package.get("record_inclusion_decisions") or []
            ),
            "run_quality_summary": 1 if package.get("run_quality_summary") else 0,
            "final_dataset_quality_summary": (
                1 if package.get("final_dataset_quality_summary") else 0
            ),
            "records_excluded_by_human_review": len(
                package.get("records_excluded_by_human_review") or []
            ),
            "source_registry": len(package.get("source_registry") or []),
            "source_identity_assessments": len(
                package.get("source_identity_assessments") or []
            ),
            "source_identity_summary": (
                1 if package.get("source_identity_summary") else 0
            ),
            "linked_events": len(package.get("linked_events") or []),
            "event_clusters": len(package.get("event_clusters") or []),
            "duplicate_clusters": len(package.get("duplicate_clusters") or []),
            "validation_cases": len(package.get("validation_cases") or []),
            "validation_comparisons": len(package.get("validation_comparisons") or []),
            "validation_results": len(package.get("validation_results") or []),
            "claims": len(claims),
            "claim_comparisons": len(claim_comparisons),
            "corroborated_events": len(corroborated_events),
            "corroborated_case_events": len(corroborated_case_events),
            "uncorroborated_claims": len(uncorroborated_claims),
            "conflicting_claims": len(conflicting_claims),
            "anomaly_results": len(package.get("anomaly_results") or []),
            "applied_human_review_decisions": len(
                package.get("applied_human_review_decisions") or []
            ),
            "rejected_human_review_decisions": len(
                package.get("rejected_human_review_decisions") or []
            ),
            "human_review_audit_trail": len(
                package.get("human_review_audit_trail") or []
            ),
            "conflicts": len(package.get("conflicts") or []),
            "human_review_items": len(package.get("human_review_items") or []),
            "excluded_sources": len(package.get("excluded_sources") or []),
            "collection_trace": len(package.get("collection_trace") or []),
            "data_dictionary": len(package.get("data_dictionary") or []),
            "workflow_case_bundle_line_list": len(
                package.get("workflow_case_bundle_line_list") or []
            ),
            "workflow_case_candidate_line_list": len(
                package.get("workflow_case_candidate_line_list") or []
            ),
        },
    }
    write_json(
        _package_for_final_json(package, package_mode, manifest),
        files["final_package_json"],
    )
    if package.get('pipeline_mode') == 'evidence':
        from .result_manifest import write_universal_outputs, OUTPUT_DATASETS
        files.update(write_universal_outputs(package, out_dir))
        manifest['result_manifest'] = package['result_manifest']
        manifest['section_counts'].update({key:len(package.get(key) or []) for key in OUTPUT_DATASETS})
        write_json(_package_for_final_json(package, package_mode, manifest), files['final_package_json'])
    return manifest
