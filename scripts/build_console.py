"""Build a workflow-run console for the Data Collection LangGraph workflow.

The console prefers the most recent configured workflow-run artifacts and
renders a self-contained HTML page that highlights node boundaries, state
growth, conditional routing, traceability, LLM extraction, validation
comparison, and human-review packaging.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOW_RUN_ROOT = _PROJECT_ROOT / "outputs"
_DEFAULT_CONSOLE_OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "workflow_console"
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.graph import build_graph  # noqa: E402


NODE_STORY = [
    (
        "task_intake_and_scope_planning",
        "Task structure",
        "Create a collection specification covering disease, population, geography, time window, and requested fields.",
    ),
    (
        "executable_source_planning",
        "Executable source plan",
        "Define source objectives, categories, and queries, with their initial status marked planned_not_executed.",
    ),
    (
        "query_strategy_builder",
        "Recorded search strategy",
        "Record the query inventory so the search strategy can be inspected and reproduced.",
    ),
    (
        "source_discovery",
        "Source discovery",
        "Create source candidates from the configured catalog and explicitly enabled fixture or live searches, retaining discovery provenance.",
    ),
    (
        "source_screening",
        "Source screening",
        "Assign source roles such as data_source, context_source, search_endpoint, and placeholder_source.",
    ),
    (
        "content_fetch_and_parse",
        "Document acquisition",
        "Fetch permitted source documents and record acquisition, parsing, and quality status in the workflow state.",
    ),
    (
        "evidence_chunking_and_data_presence_flagging",
        "Evidence chunking",
        "Each chunk carries data signals, context signals, confidence, and provenance.",
    ),
    (
        "structured_extraction",
        "Evidence becomes records",
        "Extracts PublicHealthRecord-style generic public-health records from target-data chunks; rule-based and LLM modes remain switchable.",
    ),
    (
        "record_normalization",
        "Record normalization",
        "Normalize countries, dates, case definitions, source types, and numeric fields while preserving raw values.",
    ),
    (
        "record_linking",
        "Event linking",
        "Group records describing the same event using a shared event key.",
    ),
    (
        "cross_source_consistency_check",
        "Cross-source consistency",
        "Compare case counts, deaths, dates, and locations across sources for the same event.",
    ),
    (
        "quality_gate_routing",
        "Conditional routing",
        "Route queued review items to human_review; otherwise proceed to finalization.",
    ),
    (
        "human_review",
        "Human review package",
        "Package conflicts, related records, linked events, and supporting text for review.",
    ),
    (
        "final_data_package_builder",
        "Auditable outputs",
        "Write the final dataset, source registry, linked events, conflicts, review items, and provenance manifest.",
    ),
]


def _initial_state() -> dict:
    return {
        "user_request": (
            "Collect global human hantavirus case, outbreak, and surveillance data "
            "from 2020 to 2026, including cases, deaths, dates, locations, source URLs, "
            "source types, and evidence quotes."
        ),
        "source_candidates": [],
        "source_registry": [],
        "source_identity_assessments": [],
        "source_identity_summary": None,
        "documents": [],
        "evidence_chunks": [],
        "raw_records": [],
        "validated_records": [],
        "normalized_records": [],
        "linked_events": [],
        "event_clusters": [],
        "duplicate_clusters": [],
        "active_validation_records": [],
        "inactive_validation_records": [],
        "validation_source_compatibility_summary": None,
        "validation_cases": [],
        "validation_comparisons": [],
        "validation_results": [],
        "validation_summary": None,
        "trusted_source_validation_summary": None,
        "cross_source_validation_summary": None,
        "conflicts": [],
        "human_review_queue": [],
        "human_review_decisions": [],
        "collection_trace": [],
        "collection_spec": None,
        "disease_profile": None,
        "collection_schema": None,
        "source_strategy": None,
        "screening_criteria": None,
        "search_queries": None,
        "search_query_inventory": [],
        "source_search_results": [],
        "source_search_execution_summary": None,
        "content_fetch_requests": [],
        "content_fetch_summary": None,
        "document_parse_summary": None,
        "fetch_manifest": [],
        "fixture_document_summary": None,
        "document_quality_summary": None,
        "final_data_package": None,
        "current_route": None,
    }


def _read_json_file(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _summary_value(summaries: dict, key: str):
    return (summaries or {}).get(key)


_UNAVAILABLE = "collection_spec unavailable in artifacts"


def _dict_or_empty(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value) -> list:
    return value if isinstance(value, list) else []


def _diagnostic_summary(
    diagnostics_dir: Path,
    workflow_summaries: dict,
    key: str,
) -> dict:
    value = _read_json_file(diagnostics_dir / f"{key}.json")
    if not isinstance(value, dict):
        value = _summary_value(workflow_summaries, key)
    return _sanitize_workflow_artifact(value or {})


def _collection_spec_from_artifacts(
    workflow_summaries: dict,
    run_summary: dict,
    package: dict,
) -> dict:
    task_intake = _dict_or_empty(_summary_value(workflow_summaries, "task_intake_summary"))
    spec = _dict_or_empty(task_intake.get("collection_spec"))
    if spec:
        return {
            "task_type": spec.get("task_type")
            or "public_health_case_and_outbreak_collection",
            "disease": spec.get("disease") or _UNAVAILABLE,
            "target_population": spec.get("target_population") or "human",
            "geography": spec.get("geography")
            or spec.get("location")
            or _UNAVAILABLE,
            "start_date": spec.get("start_date") or _UNAVAILABLE,
            "end_date": spec.get("end_date") or _UNAVAILABLE,
            "time_window": spec.get("time_window")
            or _time_window(spec.get("start_date"), spec.get("end_date")),
            "user_request": spec.get("user_request")
            or run_summary.get("user_request")
            or _UNAVAILABLE,
            "run_label": spec.get("run_label")
            or run_summary.get("session_id")
            or _UNAVAILABLE,
            "collection_spec_status": "available",
        }

    if any(task_intake.get(key) for key in ("disease", "location", "start_date", "end_date")):
        return {
            "task_type": task_intake.get("task_type")
            or "public_health_case_and_outbreak_collection",
            "disease": task_intake.get("disease") or _UNAVAILABLE,
            "target_population": task_intake.get("target_population") or "human",
            "geography": task_intake.get("location")
            or task_intake.get("geography")
            or _UNAVAILABLE,
            "start_date": task_intake.get("start_date") or _UNAVAILABLE,
            "end_date": task_intake.get("end_date") or _UNAVAILABLE,
            "time_window": task_intake.get("time_window")
            or _time_window(task_intake.get("start_date"), task_intake.get("end_date")),
            "user_request": run_summary.get("user_request")
            or task_intake.get("user_request")
            or _UNAVAILABLE,
            "run_label": task_intake.get("run_label")
            or run_summary.get("session_id")
            or _UNAVAILABLE,
            "collection_spec_status": "derived_from_task_intake_summary",
        }

    metadata = _dict_or_empty(package.get("package_metadata"))
    if metadata.get("disease") or metadata.get("geography"):
        return {
            "task_type": "public_health_case_and_outbreak_collection",
            "disease": _UNAVAILABLE,
            "target_population": "human",
            "geography": _UNAVAILABLE,
            "start_date": _UNAVAILABLE,
            "end_date": _UNAVAILABLE,
            "time_window": _UNAVAILABLE,
            "user_request": run_summary.get("user_request") or _UNAVAILABLE,
            "run_label": run_summary.get("session_id") or _UNAVAILABLE,
            "collection_spec_status": "unavailable",
        }

    return {
        "task_type": "public_health_case_and_outbreak_collection",
        "disease": _UNAVAILABLE,
        "target_population": "human",
        "geography": _UNAVAILABLE,
        "start_date": _UNAVAILABLE,
        "end_date": _UNAVAILABLE,
        "time_window": _UNAVAILABLE,
        "user_request": run_summary.get("user_request") or _UNAVAILABLE,
        "run_label": run_summary.get("session_id") or _UNAVAILABLE,
        "collection_spec_status": "unavailable",
    }


def _time_window(start_date, end_date) -> str:
    if start_date and end_date:
        return f"{start_date}-{end_date}"
    return _UNAVAILABLE


def user_facing_run_status(run_quality_summary: dict | None) -> str:
    summary = run_quality_summary or {}
    status = summary.get("run_quality_status")
    mapping = {
        "passed": "PASSED: accepted quality-gated records produced.",
        "passed_with_review": (
            "PASSED WITH REVIEW: accepted records produced, but review or "
            "warnings exist."
        ),
        "partial_with_quarantined_records": (
            "PARTIAL: accepted records produced, but some records were quarantined."
        ),
        "no_records_extracted": (
            "COMPLETED WITH NO RECORDS: workflow completed but no records were extracted."
        ),
        "no_task_relevant_records": (
            "COMPLETED WITH NO TASK-RELEVANT RECORDS: no accepted records passed "
            "the disease/task relevance gates."
        ),
        "failed_quality_gate": (
            "FAILED QUALITY GATE: records were produced but none passed final "
            "quality gates."
        ),
        "no_primary_case_dataset_records": (
            "COMPLETED WITH NO PRIMARY CASE DATASET RECORDS: workflow completed, "
            "but no primary case records were accepted into final_dataset."
        ),
        "failed_source_relevance": (
            "FAILED SOURCE RELEVANCE: workflow completed but source relevance "
            "gates blocked accepted records."
        ),
        "failed_extraction_relevance": (
            "FAILED EXTRACTION RELEVANCE: workflow completed but extracted "
            "records did not pass task relevance gates."
        ),
        "validation_limited_no_compatible_source": (
            "VALIDATION LIMITED: no task-compatible validation source "
            "was available."
        ),
    }
    text = mapping.get(str(status), f"STATUS: {status or 'unknown'}")
    if summary.get("validation_limited"):
        if summary.get("validation_mode") == "live_cross_source":
            text += (
                " Live cross-source validation was limited because this run "
                "did not find a task-compatible validation source."
            )
        else:
            text += (
                " Held-out validation was limited because no task-compatible "
                "validation source was available."
            )
    return text


def _sanitize_workflow_artifact(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key)
            if "fixture" in key_text.lower():
                continue
            cleaned[key_text] = _sanitize_workflow_artifact(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_workflow_artifact(item) for item in value]
    if isinstance(value, str):
        return (
            value.replace("fixture_documents_enabled", "local_test_documents_enabled")
            .replace("fixture documents", "local test documents")
            .replace("fixture", "local test")
            .replace("Fixture", "Local test")
        )
    return value


def _evidence_chunks_from_records(records: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    seen: set[str] = set()
    for record in records:
        chunk_id = record.get("supporting_chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_id": record.get("source_id"),
                "source_url": record.get("source_url"),
                "source_type": record.get("source_type"),
                "publisher": record.get("publisher"),
                "title": record.get("source_title"),
                "document_type": record.get("document_type"),
                "fetch_purpose": record.get("fetch_purpose"),
                "chunk_kind": record.get("chunk_kind"),
                "contains_target_data": True,
                "data_types": record.get("data_types"),
                "context_types": record.get("context_types"),
                "confidence": record.get("extraction_confidence"),
                "presence_reason": "Reconstructed from live collection final_dataset evidence_quote.",
                "text": record.get("evidence_quote"),
            }
        )
    return chunks


def _latest_session_dir(root: Path) -> Path:
    sessions_dir = root / "sessions"
    if not sessions_dir.exists():
        return root
    sessions = [
        item
        for item in sessions_dir.iterdir()
        if item.is_dir() and (item / "collection" / "final_package.json").exists()
    ]
    if not sessions:
        return root
    return sorted(sessions, key=lambda item: item.name)[-1]


def _load_current_workflow_result(run_output_dir: Path) -> dict | None:
    collection_dir = run_output_dir / "collection"
    diagnostics_dir = run_output_dir / "diagnostics"
    summary_path = run_output_dir / "workflow_run_summary.json"

    package = _read_json_file(collection_dir / "final_package.json")
    if not isinstance(package, dict):
        return None
    package = _sanitize_workflow_artifact(package)

    workflow_summaries = _read_json_file(diagnostics_dir / "workflow_summaries.json")
    if not isinstance(workflow_summaries, dict):
        workflow_summaries = package.get("workflow_summaries") or {}
    workflow_summaries = _sanitize_workflow_artifact(workflow_summaries)

    live_fetch_summary = _read_json_file(diagnostics_dir / "live_fetch_summary.json")
    if not isinstance(live_fetch_summary, dict):
        live_fetch_summary = {}
    live_fetch_summary = _sanitize_workflow_artifact(live_fetch_summary)

    document_parse_summary = _read_json_file(
        diagnostics_dir / "document_parse_summary.json"
    )
    if not isinstance(document_parse_summary, dict):
        document_parse_summary = (
            workflow_summaries.get("document_parse_summary")
            if isinstance(workflow_summaries, dict)
            else {}
        )
    document_parse_summary = _sanitize_workflow_artifact(
        document_parse_summary or {}
    )

    fetch_manifest = _read_json_file(diagnostics_dir / "fetch_manifest.json")
    if not isinstance(fetch_manifest, list):
        fetch_manifest = []
    fetch_manifest = _sanitize_workflow_artifact(fetch_manifest)

    llm_stage_summary = _read_json_file(diagnostics_dir / "llm_stage_summary.json")
    if not isinstance(llm_stage_summary, dict):
        llm_stage_summary = {}
    llm_stage_summary = _sanitize_workflow_artifact(llm_stage_summary)

    source_credibility_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "source_credibility_summary",
    )

    source_credibility_assessments = _read_json_file(
        diagnostics_dir / "source_credibility_assessments.json"
    )
    if not isinstance(source_credibility_assessments, list):
        source_credibility_assessments = []
    source_credibility_assessments = _sanitize_workflow_artifact(
        source_credibility_assessments
    )

    disease_relevance_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "disease_relevance_summary",
    )

    run_summary = _read_json_file(summary_path)
    if not isinstance(run_summary, dict):
        run_summary = {}
    run_summary = _sanitize_workflow_artifact(run_summary)
    interpretive_report_summary = _read_json_file(
        run_output_dir / "workflow_interpretive_report_summary.json"
    )
    if not isinstance(interpretive_report_summary, dict):
        interpretive_report_summary = {}
    interpretive_report_summary = _sanitize_workflow_artifact(
        interpretive_report_summary
    )
    workflow_visualization_summary = _read_json_file(
        run_output_dir
        / "workflow_visualization"
        / "workflow_visualization_summary.json"
    )
    if not isinstance(workflow_visualization_summary, dict):
        workflow_visualization_summary = _read_json_file(
            diagnostics_dir / "workflow_visualization_summary.json"
        )
    if not isinstance(workflow_visualization_summary, dict):
        workflow_visualization_summary = {}
    workflow_visualization_summary = _sanitize_workflow_artifact(
        workflow_visualization_summary
    )

    records = _list_or_empty(package.get("final_dataset"))
    final_dataset_pre_quality_gate = list(
        package.get("final_dataset_pre_quality_gate") or []
    )
    final_dataset_post_review = list(package.get("final_dataset_post_review") or [])
    quarantined_records = list(package.get("quarantined_records") or [])
    pending_review_records = list(package.get("pending_review_records") or [])
    non_primary_observations = list(package.get("non_primary_observations") or [])
    final_case_dataset = list(package.get("final_case_dataset") or [])
    zero_case_statements = list(package.get("zero_case_statements") or [])
    exposure_monitoring_records = list(
        package.get("exposure_monitoring_records") or []
    )
    surveillance_summary_records = list(
        package.get("surveillance_summary_records") or []
    )
    outbreak_summary_records = list(package.get("outbreak_summary_records") or [])
    context_records = list(package.get("context_records") or [])
    unclassified_observation_records = list(
        package.get("unclassified_observation_records") or []
    )
    observation_type_dataset_summary = (
        package.get("observation_type_dataset_summary")
        or _summary_value(workflow_summaries, "observation_type_dataset_summary")
        or {}
    )
    record_inclusion_decisions = list(package.get("record_inclusion_decisions") or [])
    run_quality_summary = (
        package.get("run_quality_summary")
        or _summary_value(workflow_summaries, "run_quality_summary")
        or {}
    )
    final_dataset_quality_summary = (
        package.get("final_dataset_quality_summary")
        or _summary_value(workflow_summaries, "final_dataset_quality_summary")
        or {}
    )
    anomaly_results = list(package.get("anomaly_results") or [])
    human_review_application_summary = (
        package.get("human_review_application_summary")
        or _summary_value(workflow_summaries, "human_review_application_summary")
        or {}
    )
    applied_human_review_decisions = list(
        package.get("applied_human_review_decisions") or []
    )
    rejected_human_review_decisions = list(
        package.get("rejected_human_review_decisions") or []
    )
    human_review_audit_trail = list(package.get("human_review_audit_trail") or [])
    records_excluded_by_human_review = list(
        package.get("records_excluded_by_human_review") or []
    )
    review_items = list(package.get("human_review_items") or [])
    collection_spec = _collection_spec_from_artifacts(
        workflow_summaries,
        run_summary,
        package,
    )
    normalized_records = _read_json_file(diagnostics_dir / "normalized_records.json")
    if not isinstance(normalized_records, list):
        normalized_records = final_dataset_pre_quality_gate or records
    raw_records = _read_json_file(diagnostics_dir / "raw_records.json")
    if not isinstance(raw_records, list):
        raw_records = normalized_records
    validated_records = _read_json_file(diagnostics_dir / "validated_records.json")
    if not isinstance(validated_records, list):
        validated_records = normalized_records
    source_search_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "source_search_execution_summary",
    )
    iterative_source_discovery_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "iterative_source_discovery_summary",
    )
    localized_source_planning_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "localized_source_planning_summary",
    )
    source_critic_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "source_critic_summary",
    )
    validation_source_compatibility_summary = _diagnostic_summary(
        diagnostics_dir,
        workflow_summaries,
        "validation_source_compatibility_summary",
    )

    return {
        "user_request": collection_spec["user_request"],
        "collection_trace": package.get("collection_trace") or [],
        "collection_spec": collection_spec,
        "search_queries": None,
        "search_query_inventory": [],
        "source_registry": package.get("source_registry") or [],
        "documents": live_fetch_summary.get("documents") or [],
        "evidence_chunks": _evidence_chunks_from_records(records),
        "raw_records": raw_records,
        "validated_records": validated_records,
        "rejected_records": [],
        "normalized_records": normalized_records,
        "linked_events": package.get("linked_events") or [],
        "event_clusters": package.get("event_clusters") or [],
        "duplicate_clusters": package.get("duplicate_clusters") or [],
        "validation_cases": package.get("validation_cases") or [],
        "validation_comparisons": package.get("validation_comparisons") or [],
        "validation_results": package.get("validation_results") or [],
        "active_validation_records": [],
        "inactive_validation_records": [],
        "anomaly_results": anomaly_results,
        "anomaly_summary": package.get("anomaly_summary")
        or _summary_value(workflow_summaries, "anomaly_summary"),
        "validation_summary": package.get("validation_summary")
        or _summary_value(workflow_summaries, "validation_summary"),
        "validation_source_compatibility_summary": validation_source_compatibility_summary,
        "trusted_source_validation_summary": package.get(
            "trusted_source_validation_summary"
        )
        or _summary_value(workflow_summaries, "trusted_source_validation_summary"),
        "cross_source_validation_summary": package.get(
            "cross_source_validation_summary"
        )
        or _summary_value(workflow_summaries, "cross_source_validation_summary"),
        "conflicts": package.get("conflicts") or [],
        "human_review_queue": review_items,
        "applied_human_review_decisions": applied_human_review_decisions,
        "rejected_human_review_decisions": rejected_human_review_decisions,
        "human_review_audit_trail": human_review_audit_trail,
        "human_review_application_summary": human_review_application_summary,
        "final_dataset_post_review": final_dataset_post_review,
        "final_dataset_pre_quality_gate": final_dataset_pre_quality_gate,
        "quarantined_records": quarantined_records,
        "pending_review_records": pending_review_records,
        "non_primary_observations": non_primary_observations,
        "final_case_dataset": final_case_dataset,
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
        "zero_case_statements": zero_case_statements,
        "exposure_monitoring_records": exposure_monitoring_records,
        "surveillance_summary_records": surveillance_summary_records,
        "outbreak_summary_records": outbreak_summary_records,
        "context_records": context_records,
        "unclassified_observation_records": unclassified_observation_records,
        "observation_type_dataset_summary": observation_type_dataset_summary,
        "record_inclusion_decisions": record_inclusion_decisions,
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": final_dataset_quality_summary,
        "records_excluded_by_human_review": records_excluded_by_human_review,
        "final_data_package": package,
        "current_route": run_summary.get("current_route")
        or ("human_review" if review_items else "final_data_package_builder"),
        "executable_source_plan_summary": _summary_value(
            workflow_summaries, "executable_source_plan_summary"
        ),
        "localized_source_planning_summary": localized_source_planning_summary,
        "source_planning_agent_summary": _summary_value(
            workflow_summaries, "source_planning_agent_summary"
        ),
        "source_search_execution_summary": source_search_summary,
        "iterative_source_discovery_summary": iterative_source_discovery_summary,
        "source_discovery_summary": _summary_value(
            workflow_summaries, "source_discovery_summary"
        ),
        "source_screening_summary": _summary_value(
            workflow_summaries, "source_screening_summary"
        ),
        "source_critic_summary": source_critic_summary,
        "source_routing_summary": _summary_value(
            workflow_summaries, "source_routing_summary"
        ),
        "source_credibility_summary": _summary_value(
            workflow_summaries, "source_credibility_summary"
        )
        or source_credibility_summary,
        "source_credibility_assessments": source_credibility_assessments,
        "disease_relevance_summary": _summary_value(
            workflow_summaries, "disease_relevance_summary"
        )
        or disease_relevance_summary,
        "content_fetch_summary": _summary_value(
            workflow_summaries, "content_fetch_summary"
        ),
        "document_parse_summary": _summary_value(
            workflow_summaries, "document_parse_summary"
        )
        or document_parse_summary,
        "fetch_manifest": fetch_manifest,
        "fixture_document_summary": None,
        "document_quality_summary": _summary_value(
            workflow_summaries, "document_quality_summary"
        ),
        "evidence_chunking_summary": _summary_value(
            workflow_summaries, "evidence_chunking_summary"
        ),
        "data_presence_summary": _summary_value(
            workflow_summaries, "data_presence_summary"
        ),
        "structured_extraction_summary": _summary_value(
            workflow_summaries, "structured_extraction_summary"
        ),
        "llm_extraction_summary": _summary_value(
            workflow_summaries, "llm_extraction_summary"
        ),
        "schema_validation_summary": _summary_value(
            workflow_summaries, "schema_validation_summary"
        ),
        "record_normalization_summary": _summary_value(
            workflow_summaries, "record_normalization_summary"
        ),
        "record_linking_summary": _summary_value(
            workflow_summaries, "record_linking_summary"
        ),
        "event_clustering_summary": _summary_value(
            workflow_summaries, "event_clustering_summary"
        ),
        "duplicate_detection_summary": _summary_value(
            workflow_summaries, "duplicate_detection_summary"
        ),
        "cross_source_consistency_summary": _summary_value(
            workflow_summaries, "cross_source_consistency_summary"
        ),
        "human_review_summary": _summary_value(
            workflow_summaries, "human_review_summary"
        ),
        "run_quality_summary": _summary_value(
            workflow_summaries, "run_quality_summary"
        )
        or run_quality_summary,
        "final_dataset_quality_summary": _summary_value(
            workflow_summaries, "final_dataset_quality_summary"
        )
        or final_dataset_quality_summary,
        "interpretive_report_summary": interpretive_report_summary,
        "workflow_visualization_summary": workflow_visualization_summary,
        "finalization_summary": _summary_value(
            workflow_summaries, "finalization_summary"
        ),
        "workflow_run_result": {
            "live_fetch_summary": live_fetch_summary,
            "document_parse_summary": document_parse_summary,
            "fetch_manifest": fetch_manifest,
            "llm_stage_summary": llm_stage_summary,
            "disease_relevance_summary": disease_relevance_summary,
            "run_quality_summary": run_quality_summary,
            "final_dataset_quality_summary": final_dataset_quality_summary,
            "run_summary": run_summary,
            "interpretive_report_summary": interpretive_report_summary,
            "workflow_visualization_summary": workflow_visualization_summary,
        },
    }


def _load_workflow_result(run_output_dir: Path | None = None) -> dict | None:
    resolved_dir = run_output_dir or _latest_session_dir(_DEFAULT_WORKFLOW_RUN_ROOT)
    return _load_current_workflow_result(resolved_dir)


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _json(value) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2), quote=False)


def _count(result: dict, key: str) -> int:
    return len(result.get(key) or [])


def _summary(result: dict, key: str) -> dict:
    value = result.get(key) or {}
    return value if isinstance(value, dict) else {}


def _metric_card(label: str, value, note: str = "") -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{_esc(label)}</div>'
        f'<div class="metric-value">{_esc(value)}</div>'
        f'<div class="metric-note">{_esc(note)}</div>'
        "</div>"
    )


def _node_card(node_name: str, title: str, description: str, executed: bool) -> str:
    status = "executed" if executed else "not-run"
    status_text = "executed" if executed else "not in this run"
    return (
        f'<section class="node-card {status}">'
        f'<div class="node-kicker">{_esc(status_text)}</div>'
        f"<h3>{_esc(title)}</h3>"
        f'<code>{_esc(node_name)}</code>'
        f"<p>{_esc(description)}</p>"
        "</section>"
    )


def _row(cells: list[str], tag: str = "td") -> str:
    return "<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>"


def _stage_rows(result: dict) -> str:
    summaries = {
        "source_discovery": _summary(result, "source_discovery_summary"),
        "source_screening": _summary(result, "source_screening_summary"),
        "content_fetch_and_parse": _summary(result, "content_fetch_summary"),
        "evidence_chunking_and_data_presence_flagging": _summary(
            result, "evidence_chunking_summary"
        ),
        "structured_extraction": _summary(result, "structured_extraction_summary"),
        "schema_validation_and_repair": _summary(result, "schema_validation_summary"),
        "record_normalization": _summary(result, "record_normalization_summary"),
        "record_linking": _summary(result, "record_linking_summary"),
        "cross_source_consistency_check": _summary(
            result, "cross_source_consistency_summary"
        ),
        "human_review": _summary(result, "human_review_summary"),
    }
    rows = [
        (
            "source_discovery",
            "source_candidates",
            summaries["source_discovery"].get("candidate_count"),
            "Pass discovered sources to the next step through the workflow state.",
        ),
        (
            "source_screening",
            "source_registry",
            summaries["source_screening"].get("source_role_counts"),
            "Store screening decisions as structured state fields.",
        ),
        (
            "content_fetch_and_parse",
            "documents",
            summaries["content_fetch_and_parse"].get("fetch_status_counts"),
            "The acquisition node supports configured offline, fixture, and live-fetch modes.",
        ),
        (
            "evidence_chunking_and_data_presence_flagging",
            "evidence_chunks",
            summaries["evidence_chunking_and_data_presence_flagging"].get(
                "total_chunk_count"
            ),
            "Chunks include signals, confidence, and source provenance.",
        ),
        (
            "structured_extraction",
            "raw_records",
            summaries["structured_extraction"].get("raw_record_count"),
            "The extraction node supports deterministic rules and configured model extraction.",
        ),
        (
            "schema_validation_and_repair",
            "validated_records / rejected_records",
            summaries["schema_validation_and_repair"].get("schema_status_counts"),
            "Validation results determine which records enter the review queue.",
        ),
        (
            "record_normalization",
            "normalized_records",
            summaries["record_normalization"].get("normalization_status_counts"),
            "Retain raw fields so normalization changes can be audited.",
        ),
        (
            "record_linking",
            "linked_events",
            summaries["record_linking"].get("linking_status_counts"),
            "Group records from multiple sources into candidate events.",
        ),
        (
            "cross_source_consistency_check",
            "conflicts",
            summaries["cross_source_consistency_check"].get("conflict_type_counts"),
            "Represent cross-source conflicts as traceable records.",
        ),
        (
            "human_review",
            "human_review_queue",
            summaries["human_review"].get("item_type_counts"),
            "Package conflicts and context for human review.",
        ),
    ]
    return "\n".join(
        _row(
            [
                f"<code>{_esc(node)}</code>",
                f"<code>{_esc(field)}</code>",
                f"<pre>{_json(value)}</pre>",
                _esc(why),
            ]
        )
        for node, field, value, why in rows
    )


def _records_table(records: list[dict]) -> str:
    header = _row(
        [
            "record_id",
            "source",
            "country",
            "date",
            "cases_unspecified",
            "deaths",
            "linked_event",
            "conflicts",
        ],
        tag="th",
    )
    body = "\n".join(
        _row(
            [
                f"<code>{_esc(r.get('record_id'))}</code>",
                _esc(r.get("publisher") or r.get("source_id")),
                _esc(r.get("country")),
                _esc(r.get("date_reported") or r.get("date_anchor")),
                _esc(r.get("cases_unspecified")),
                _esc(r.get("deaths")),
                f"<code>{_esc(r.get('linked_event_id'))}</code>",
                _esc(", ".join(r.get("conflict_ids") or [])),
            ]
        )
        for r in records
    )
    return f"<table>{header}{body}</table>"


def _conflict_panels(conflicts: list[dict]) -> str:
    if not conflicts:
        return '<p class="muted">No conflicts in this run.</p>'
    panels: list[str] = []
    for conflict in conflicts:
        values = conflict.get("values") or []
        value_items = "".join(
            "<li>"
            f"<code>{_esc(v.get('record_id'))}</code> "
            f"{_esc(v.get('source_id'))}: "
            f"<strong>{_esc(v.get('value'))}</strong>"
            "</li>"
            for v in values
        )
        panels.append(
            '<article class="conflict">'
            f"<h3>{_esc(conflict.get('field'))}: {_esc(conflict.get('conflict_type'))}</h3>"
            f'<p><strong>Severity:</strong> {_esc(conflict.get("severity"))} '
            f'| <strong>Human review:</strong> {_esc(conflict.get("requires_human_review"))}</p>'
            f"<ul>{value_items}</ul>"
            f'<p class="muted">{_esc(conflict.get("recommended_action"))}</p>'
            "</article>"
        )
    return "\n".join(panels)


def _review_panels(items: list[dict]) -> str:
    if not items:
        return '<p class="muted">No review items in this run.</p>'
    panels: list[str] = []
    for item in items:
        packet = item.get("review_packet") or {}
        sections = packet.get("packet_sections") or {}
        section_names = ", ".join(sections.keys())
        panels.append(
            '<article class="review">'
            f"<h3>{_esc(item.get('review_id'))}</h3>"
            f'<p><strong>Type:</strong> {_esc(item.get("item_type"))} '
            f'| <strong>Priority:</strong> {_esc(item.get("priority"))} '
            f'| <strong>Status:</strong> {_esc(item.get("status"))}</p>'
            f'<p>{_esc(item.get("reason"))}</p>'
            f'<p><strong>Packet sections:</strong> {_esc(section_names)}</p>'
            "</article>"
        )
    return "\n".join(panels)


def _short_text(value, limit: int = 520) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sample(items, limit: int = 5) -> list:
    return list(items or [])[:limit]


def _trace_by_node(result: dict) -> dict[str, dict]:
    return {
        event.get("node_name"): event
        for event in result.get("collection_trace", [])
        if event.get("node_name")
    }


def _source_lookup(result: dict) -> dict[str, dict]:
    return {
        item.get("source_id"): item
        for item in result.get("source_registry", [])
        if item.get("source_id")
    }


def _contains_synthetic_data(package: dict) -> bool:
    return bool(
        package.get("contains_synthetic_data")
        or package.get("contains_synthetic_fixture_data")
    )


def _stage_payload(result: dict) -> list[dict]:
    trace = _trace_by_node(result)
    return [
        {
            "node": "task_intake_and_scope_planning",
            "title": "Plan: Define the collection task",
            "agentic_role": "Planning / state initialization",
            "state_writes": ["collection_spec", "collection_trace"],
            "trace": trace.get("task_intake_and_scope_planning", {}),
            "show": {
                "collection_spec": result.get("collection_spec"),
            },
            "takeaway": "Store the task disease, location, time range, requested fields, and validation rules in the collection specification.",
        },
        {
            "node": "executable_source_planning",
            "title": "Plan: Prepare source discovery",
            "agentic_role": "Executable source planning",
            "state_writes": [
                "agentic_source_plan",
                "executable_source_plan_summary",
                "localized_source_planning_summary",
                "source_planning_agent_summary",
            ],
            "trace": trace.get("executable_source_planning", {}),
            "show": {
                "executable_source_plan_summary": result.get(
                    "executable_source_plan_summary"
                ),
                "localized_source_planning_summary": result.get(
                    "localized_source_planning_summary"
                ),
                "planned_query_sample": _sample(
                    (result.get("agentic_source_plan") or {}).get("planned_queries"),
                    6,
                ),
            },
            "takeaway": "Prepare an auditable source plan. Queries remain planned_not_executed until the source-discovery step executes them.",
        },
        {
            "node": "query_strategy_builder",
            "title": "Plan: Build the search strategy and query inventory",
            "agentic_role": "Search strategy",
            "state_writes": ["search_queries", "search_query_inventory"],
            "trace": trace.get("query_strategy_builder", {}),
            "show": {
                "search_queries": result.get("search_queries"),
                "search_query_inventory_sample": _sample(
                    result.get("search_query_inventory"), 6
                ),
            },
            "takeaway": "Store the search strategy in state so it can be inspected, adjusted, and rerun.",
        },
        {
            "node": "source_screening",
            "title": "Act and filter: Discover sources and assign roles",
            "agentic_role": "Tool-like source discovery + rule gate",
            "state_writes": ["source_candidates", "source_registry"],
            "trace": trace.get("source_screening", {}),
            "show": {
                "source_search_execution_summary": result.get(
                    "source_search_execution_summary"
                ),
                "source_discovery_summary": result.get("source_discovery_summary"),
                "source_screening_summary": result.get("source_screening_summary"),
                "source_registry_sample": _sample(result.get("source_registry"), 6),
            },
            "takeaway": "Build the source registry and identify sources permitted for extraction, validation, or contextual use.",
        },
        {
            "node": "source_screening",
            "title": "Verify: source credibility scoring and final role assignment",
            "agentic_role": "Credibility rubric / role gate",
            "state_writes": [
                "source_credibility_assessments",
                "source_credibility_summary",
                "source_identity_assessments",
                "source_identity_summary",
                "source_registry.source_role_final",
            ],
            "trace": trace.get("source_screening", {}),
            "show": {
                "source_credibility_summary": result.get(
                    "source_credibility_summary"
                ),
                "source_credibility_assessments": _sample(
                    result.get("source_credibility_assessments"), 6
                ),
                "source_identity_summary": result.get("source_identity_summary"),
                "source_identity_assessments": _sample(
                    result.get("source_identity_assessments"), 8
                ),
                "source_registry_sample": _sample(result.get("source_registry"), 6),
            },
            "takeaway": "Each registry source receives task-aware credibility scoring and an auditable final role: collection, validation, context, collection_support, search_endpoint, excluded, or needs_human_review.",
        },
        {
            "node": "content_fetch_and_parse",
            "title": "Act: Retrieve source documents",
            "agentic_role": "Document acquisition tool",
            "state_writes": [
                "content_fetch_requests",
                "documents",
                "document_parse_summary",
                "fetch_manifest",
            ],
            "trace": trace.get("content_fetch_and_parse", {}),
            "show": {
                "content_fetch_summary": result.get("content_fetch_summary"),
                "document_parse_summary": result.get("document_parse_summary"),
                "fetch_manifest": _sample(result.get("fetch_manifest"), 8),
                "documents_sample": _sample(result.get("documents"), 5),
            },
            "takeaway": "Retrieve permitted source URLs and record fetch status, HTTP status, content type, and parsing status.",
        },
        {
            "node": "evidence_chunking_and_data_presence_flagging",
            "title": "Observe: Identify text containing relevant data",
            "agentic_role": "Evidence triage",
            "state_writes": ["evidence_chunks", "evidence_chunking_summary"],
            "trace": trace.get("evidence_chunking_and_data_presence_flagging", {}),
            "show": {
                "evidence_chunking_summary": result.get("evidence_chunking_summary"),
                "data_presence_summary": result.get("data_presence_summary"),
                "evidence_chunks": result.get("evidence_chunks"),
            },
            "takeaway": "Split document text into evidence chunks and identify case-count, death-count, date, and location signals.",
        },
        {
            "node": "structured_extraction",
            "title": "Extract: Create structured records from evidence",
            "agentic_role": "Structured extraction agent",
            "state_writes": ["raw_records", "llm_extraction_summary"],
            "trace": trace.get("structured_extraction", {}),
            "show": {
                "structured_extraction_summary": result.get(
                    "structured_extraction_summary"
                ),
                "llm_extraction_summary": result.get("llm_extraction_summary"),
                "raw_records": result.get("raw_records"),
            },
            "takeaway": "Apply deterministic extraction and, when enabled, model extraction to eligible collection evidence chunks.",
        },
        {
            "node": "schema_validation_and_repair",
            "title": "Verify: schema validation / repair",
            "agentic_role": "Guardrail",
            "state_writes": ["validated_records", "rejected_records"],
            "trace": trace.get("schema_validation_and_repair", {}),
            "show": {
                "schema_validation_summary": result.get("schema_validation_summary"),
                "validated_records": result.get("validated_records"),
                "rejected_records": result.get("rejected_records"),
            },
            "takeaway": "Record validation decisions in state so each decision can be audited.",
        },
        {
            "node": "record_linking",
            "title": "Reason: Link records describing the same event",
            "agentic_role": "Entity / event linking",
            "state_writes": ["normalized_records", "linked_events", "event_clusters"],
            "trace": trace.get("record_linking", {}),
            "show": {
                "record_normalization_summary": result.get(
                    "record_normalization_summary"
                ),
                "record_linking_summary": result.get("record_linking_summary"),
                "event_clustering_summary": result.get("event_clustering_summary"),
                "duplicate_detection_summary": result.get("duplicate_detection_summary"),
                "normalized_records": result.get("normalized_records"),
                "linked_events": result.get("linked_events"),
                "event_clusters": result.get("event_clusters"),
            },
            "takeaway": "Link records by disease, location, date, and count semantics while preserving their source evidence.",
        },
        {
            "node": "cross_source_consistency_check",
            "title": "Critique: Check cross-source consistency",
            "agentic_role": "Critic / verifier",
            "state_writes": [
                "validation_results",
                "validation_summary",
                "trusted_source_validation_summary",
                "cross_source_validation_summary",
                "conflicts",
                "cross_source_consistency_summary",
            ],
            "trace": trace.get("cross_source_consistency_check", {}),
            "show": {
                "validation_summary": result.get("validation_summary"),
                "trusted_source_validation_summary": result.get(
                    "trusted_source_validation_summary"
                ),
                "cross_source_validation_summary": result.get(
                    "cross_source_validation_summary"
                ),
                "validation_results": result.get("validation_results"),
                "cross_source_consistency_summary": result.get(
                    "cross_source_consistency_summary"
                ),
                "conflicts": result.get("conflicts"),
            },
            "takeaway": "Check whether counts, dates, and locations are comparable. Retain conflicts and insufficient validation evidence for review.",
        },
        {
            "node": "quality_gate_routing",
            "title": "Route: Apply quality gates",
            "agentic_role": "Conditional routing",
            "state_writes": ["current_route"],
            "trace": trace.get("quality_gate_routing", {}),
            "show": {
                "current_route": result.get("current_route"),
                "human_review_queue_count": len(result.get("human_review_queue") or []),
                "conflict_count": len(result.get("conflicts") or []),
                "anomaly_summary": result.get("anomaly_summary"),
                "anomaly_results": result.get("anomaly_results"),
            },
            "takeaway": "Graph edges make the quality-gate decision and subsequent route explicit.",
        },
        {
            "node": "human_review",
            "title": "Human review: Prepare the review package",
            "agentic_role": "Human oversight handoff",
            "state_writes": [
                "human_review_queue",
                "human_review_summary",
                "human_review_application_summary",
                "final_dataset_post_review",
            ],
            "trace": trace.get("human_review", {}),
            "show": {
                "human_review_summary": result.get("human_review_summary"),
                "human_review_application_summary": result.get(
                    "human_review_application_summary"
                ),
                "human_review_queue": result.get("human_review_queue"),
                "applied_human_review_decisions": result.get(
                    "applied_human_review_decisions"
                ),
                "rejected_human_review_decisions": result.get(
                    "rejected_human_review_decisions"
                ),
                "human_review_audit_trail": result.get("human_review_audit_trail"),
            },
            "takeaway": "Provide reviewers with conflicts, supporting evidence, related records, and linked events.",
        },
        {
            "node": "final_data_package_builder",
            "title": "Finalize: Export auditable data",
            "agentic_role": "Audit package",
            "state_writes": ["final_data_package", "finalization_summary"],
            "trace": trace.get("final_data_package_builder", {}),
            "show": {
                "finalization_summary": result.get("finalization_summary"),
                "final_dataset_post_review": result.get("final_dataset_post_review"),
                "records_excluded_by_human_review": result.get(
                    "records_excluded_by_human_review"
                ),
                "package_metadata": (result.get("final_data_package") or {}).get(
                    "package_metadata"
                ),
                "provenance_manifest": (result.get("final_data_package") or {}).get(
                    "provenance_manifest"
                ),
            },
            "takeaway": "Export records together with the trace, source registry, provenance, diagnostics, evaluation report, and review items.",
        },
    ]


def _report_payload(result: dict) -> dict:
    sources = _source_lookup(result)
    records = result.get("normalized_records") or []
    chunks = result.get("evidence_chunks") or []
    conflicts = result.get("conflicts") or []
    review_items = result.get("human_review_queue") or []
    package = result.get("final_data_package") or {}
    collection_spec = result.get("collection_spec") or {}
    workflow_result = result.get("workflow_run_result") or {}
    interpretive_report_summary = (
        result.get("interpretive_report_summary")
        or workflow_result.get("interpretive_report_summary")
        or {}
    )
    workflow_visualization_summary = (
        result.get("workflow_visualization_summary")
        or workflow_result.get("workflow_visualization_summary")
        or {}
    )
    live_fetch_summary = workflow_result.get("live_fetch_summary") or {}
    document_parse_summary = (
        result.get("document_parse_summary")
        or workflow_result.get("document_parse_summary")
        or {}
    )
    fetch_manifest = (
        result.get("fetch_manifest") or workflow_result.get("fetch_manifest") or []
    )
    llm_replay_summary = workflow_result.get("llm_replay_summary") or {}
    source_search_summary = result.get("source_search_execution_summary") or {}
    iterative_source_discovery_summary = (
        result.get("iterative_source_discovery_summary") or {}
    )
    localized_source_planning_summary = (
        result.get("localized_source_planning_summary") or {}
    )
    source_critic_summary = result.get("source_critic_summary") or {}
    source_credibility_summary = result.get("source_credibility_summary") or {}
    source_identity_summary = result.get("source_identity_summary") or {}
    source_identity_assessments = result.get("source_identity_assessments") or []
    disease_relevance_summary = result.get("disease_relevance_summary") or {}
    validation_source_compatibility_summary = (
        result.get("validation_source_compatibility_summary") or {}
    )
    anomaly_results = result.get("anomaly_results") or []
    applied_decisions = result.get("applied_human_review_decisions") or []
    rejected_decisions = result.get("rejected_human_review_decisions") or []
    audit_trail = result.get("human_review_audit_trail") or []
    final_dataset_post_review = result.get("final_dataset_post_review") or []
    final_dataset_pre_quality_gate = result.get("final_dataset_pre_quality_gate") or []
    quarantined_records = result.get("quarantined_records") or []
    pending_review_records = result.get("pending_review_records") or []
    non_primary_observations = result.get("non_primary_observations") or []
    final_case_dataset = result.get("final_case_dataset") or package.get(
        "final_case_dataset"
    ) or []
    global_outbreak_event_dataset = result.get(
        "global_outbreak_event_dataset"
    ) or package.get("global_outbreak_event_dataset") or []
    regional_surveillance_dataset = result.get(
        "regional_surveillance_dataset"
    ) or package.get("regional_surveillance_dataset") or []
    country_year_aggregate_dataset = result.get(
        "country_year_aggregate_dataset"
    ) or package.get("country_year_aggregate_dataset") or []
    official_alert_dataset = result.get("official_alert_dataset") or package.get(
        "official_alert_dataset"
    ) or []
    zero_case_statements = result.get("zero_case_statements") or package.get(
        "zero_case_statements"
    ) or []
    exposure_monitoring_records = result.get(
        "exposure_monitoring_records"
    ) or package.get("exposure_monitoring_records") or []
    surveillance_summary_records = result.get(
        "surveillance_summary_records"
    ) or package.get("surveillance_summary_records") or []
    outbreak_summary_records = result.get("outbreak_summary_records") or package.get(
        "outbreak_summary_records"
    ) or []
    context_records = result.get("context_records") or package.get("context_records") or []
    unclassified_observation_records = result.get(
        "unclassified_observation_records"
    ) or package.get("unclassified_observation_records") or []
    observation_type_dataset_summary = result.get(
        "observation_type_dataset_summary"
    ) or package.get("observation_type_dataset_summary") or {}
    record_inclusion_decisions = result.get("record_inclusion_decisions") or []
    run_quality_summary = result.get("run_quality_summary") or {}
    final_dataset_quality_summary = result.get("final_dataset_quality_summary") or {}
    accepted_records = package.get("final_dataset") or []
    user_status = user_facing_run_status(run_quality_summary)
    recommended_user_message = run_quality_summary.get("recommended_user_message")

    evidence_links = []
    chunk_by_id = {chunk.get("chunk_id"): chunk for chunk in chunks}
    for record in records:
        chunk = chunk_by_id.get(record.get("supporting_chunk_id")) or {}
        evidence_links.append(
            {
                "record_id": record.get("record_id"),
                "source_id": record.get("source_id"),
                "publisher": record.get("publisher"),
                "cases_unspecified": record.get("cases_unspecified"),
                "deaths": record.get("deaths"),
                "date_reported": record.get("date_reported"),
                "country": record.get("country"),
                "chunk_id": record.get("supporting_chunk_id"),
                "extraction_method": record.get("extraction_method"),
                "llm_used": record.get("llm_used"),
                "source_priority": (sources.get(record.get("source_id")) or {}).get(
                    "source_priority"
                ),
                "evidence_quote": _short_text(record.get("evidence_quote"), 900),
                "chunk_signals": {
                    "data_types": chunk.get("data_types"),
                    "confidence": chunk.get("confidence"),
                    "presence_reason": chunk.get("presence_reason"),
                },
            }
        )

    return {
        "mode": {
            "local_test_documents": bool(
                live_fetch_summary.get("fixture_documents_enabled")
            ),
            "live_fetch": bool(live_fetch_summary.get("live_fetch_enabled")),
            "llm_extraction": bool(llm_replay_summary.get("llm_call_succeeded")),
            "source_search_mode": source_search_summary.get("search_mode"),
            "source_search_provider": source_search_summary.get("search_provider"),
            "source_search_executed_queries": source_search_summary.get(
                "executed_query_count"
            ),
            "search_derived_candidates": source_search_summary.get(
                "candidate_from_search_count"
            ),
            "iterative_source_discovery_enabled": (
                iterative_source_discovery_summary.get(
                    "iterative_source_discovery_enabled"
                )
            ),
            "iterative_search_iterations": iterative_source_discovery_summary.get(
                "search_iteration_count"
            ),
            "iterative_stop_decision": iterative_source_discovery_summary.get(
                "stop_decision"
            ),
            "truth_label": (
                f"Current session task: {collection_spec.get('disease')} / "
                f"{collection_spec.get('geography')} / "
                f"{collection_spec.get('time_window')}. {user_status}"
            ),
        },
        "task": {
            "collection_spec": collection_spec,
            "disease": collection_spec.get("disease"),
            "location": collection_spec.get("geography"),
            "start_date": collection_spec.get("start_date"),
            "end_date": collection_spec.get("end_date"),
            "time_window": collection_spec.get("time_window"),
            "user_request": collection_spec.get("user_request"),
            "run_label": collection_spec.get("run_label"),
            "collection_spec_status": collection_spec.get("collection_spec_status"),
        },
        "user_facing_run_status": user_status,
        "recommended_user_message": recommended_user_message,
        "counts": {
            "trace_nodes": len(result.get("collection_trace") or []),
            "sources": len(result.get("source_registry") or []),
            "source_identity_assessments": source_identity_summary.get(
                "identity_assessed_count", 0
            ),
            "documents": len(result.get("documents") or []),
            "evidence_chunks": len(chunks),
            "records": len(records),
            "accepted_records": len(accepted_records),
            "pre_quality_records": len(final_dataset_pre_quality_gate),
            "quarantined_records": len(quarantined_records),
            "pending_review_records": len(pending_review_records),
            "non_primary_observations": len(non_primary_observations),
            "final_case_dataset": len(final_case_dataset),
            "global_outbreak_event_dataset": len(global_outbreak_event_dataset),
            "regional_surveillance_dataset": len(regional_surveillance_dataset),
            "country_year_aggregate_dataset": len(country_year_aggregate_dataset),
            "official_alert_dataset": len(official_alert_dataset),
            "zero_case_statements": len(zero_case_statements),
            "exposure_monitoring_records": len(exposure_monitoring_records),
            "surveillance_summary_records": len(surveillance_summary_records),
            "outbreak_summary_records": len(outbreak_summary_records),
            "context_records": len(context_records),
            "unclassified_observation_records": len(unclassified_observation_records),
            "primary_case_dataset_eligible": run_quality_summary.get(
                "primary_case_dataset_eligible_count",
                final_dataset_quality_summary.get(
                    "primary_case_dataset_eligible_count", 0
                ),
            ),
            "corroborated_primary_case_events": run_quality_summary.get(
                "corroborated_primary_case_event_count",
                final_dataset_quality_summary.get(
                    "corroborated_primary_case_event_count", 0
                ),
            ),
            "linked_events": len(result.get("linked_events") or []),
            "event_clusters": len(result.get("event_clusters") or []),
            "duplicate_clusters": len(result.get("duplicate_clusters") or []),
            "validation_results": len(result.get("validation_results") or []),
            "anomaly_results": len(anomaly_results),
            "conflicts": len(conflicts),
            "human_review_items": len(review_items),
            "applied_review_decisions": len(applied_decisions),
            "rejected_review_decisions": len(rejected_decisions),
            "review_audit_entries": len(audit_trail),
            "final_dataset_post_review": len(final_dataset_post_review),
            "rejected_incompatible_records": disease_relevance_summary.get(
                "rejected_incompatible_record_count", 0
            ),
            "iterative_search_iterations": iterative_source_discovery_summary.get(
                "search_iteration_count", 0
            ),
            "iterative_llm_refinement_calls": iterative_source_discovery_summary.get(
                "llm_refinement_call_count", 0
            ),
        },
        "route": result.get("current_route"),
        "source_search": source_search_summary,
        "iterative_source_discovery": iterative_source_discovery_summary,
        "localized_source_planning": localized_source_planning_summary,
        "source_critic": source_critic_summary,
        "source_credibility": {
            "summary": source_credibility_summary,
            "assessment_sample": _sample(
                result.get("source_credibility_assessments"), 6
            ),
        },
        "source_identity": {
            "summary": source_identity_summary,
            "assessment_sample": _sample(source_identity_assessments, 8),
        },
        "disease_relevance": disease_relevance_summary,
        "run_quality": {
            "summary": run_quality_summary,
            "final_dataset_quality_summary": final_dataset_quality_summary,
            "observation_type_dataset_summary": observation_type_dataset_summary,
            "final_case_dataset_sample": _sample(final_case_dataset, 8),
            "global_outbreak_event_dataset_sample": _sample(
                global_outbreak_event_dataset, 8
            ),
            "regional_surveillance_dataset_sample": _sample(
                regional_surveillance_dataset, 8
            ),
            "country_year_aggregate_dataset_sample": _sample(
                country_year_aggregate_dataset, 8
            ),
            "official_alert_dataset_sample": _sample(official_alert_dataset, 8),
            "zero_case_statements_sample": _sample(zero_case_statements, 8),
            "exposure_monitoring_records_sample": _sample(
                exposure_monitoring_records, 8
            ),
            "context_records_sample": _sample(context_records, 8),
            "record_inclusion_decisions_sample": _sample(
                record_inclusion_decisions, 10
            ),
            "quarantined_records_sample": _sample(quarantined_records, 8),
            "pending_review_records_sample": _sample(pending_review_records, 8),
            "non_primary_observations_sample": _sample(
                non_primary_observations, 8
            ),
            "pre_quality_records_sample": _sample(final_dataset_pre_quality_gate, 8),
        },
        "interpretive_report": interpretive_report_summary,
        "workflow_visualization": workflow_visualization_summary,
        "fetch_parse": {
            "live_fetch_summary": live_fetch_summary,
            "document_parse_summary": document_parse_summary,
            "fetch_manifest_sample": _sample(fetch_manifest, 8),
        },
        "stages": _stage_payload(result),
        "validation": {
            "validation_source_compatibility_summary": validation_source_compatibility_summary,
            "validation_summary": result.get("validation_summary") or {},
            "trusted_source_validation_summary": result.get(
                "trusted_source_validation_summary"
            )
            or {},
            "cross_source_validation_summary": result.get(
                "cross_source_validation_summary"
            )
            or {},
            "validation_results_sample": _sample(
                result.get("validation_results"), 8
            ),
        },
        "stage11": {
            "anomaly_summary": result.get("anomaly_summary") or {},
            "anomaly_results_sample": _sample(anomaly_results, 8),
            "human_review_application_summary": result.get(
                "human_review_application_summary"
            )
            or {},
            "applied_human_review_decisions_sample": _sample(applied_decisions, 8),
            "rejected_human_review_decisions_sample": _sample(rejected_decisions, 8),
            "human_review_audit_trail_sample": _sample(audit_trail, 8),
            "final_dataset_post_review_sample": _sample(
                final_dataset_post_review, 8
            ),
        },
        "evidence_links": evidence_links,
        "conflicts": conflicts,
        "review_items": review_items,
        "trace_order": [
            event.get("node_name") for event in result.get("collection_trace", [])
        ],
        "final_package_flags": {
            "contains_synthetic_data": _contains_synthetic_data(package),
            "synthetic_notice": package.get("synthetic_fixture_notice"),
            "live_fetch_enabled": live_fetch_summary.get("live_fetch_enabled"),
            "llm_call_succeeded": llm_replay_summary.get("llm_call_succeeded"),
        },
        "key_artifact_paths": {
            "workflow_interpretive_report.md": "workflow_interpretive_report.md",
            "workflow_interpretive_report_summary.json": "workflow_interpretive_report_summary.json",
            "workflow_visualization/index.html": "workflow_visualization/index.html",
            "workflow_visualization/workflow_timeline.html": "workflow_visualization/workflow_timeline.html",
            "workflow_visualization/evidence_flow_graph.html": "workflow_visualization/evidence_flow_graph.html",
            "workflow_visualization/claim_comparison_cards.html": "workflow_visualization/claim_comparison_cards.html",
            "workflow_visualization/dataset_decision_flow.html": "workflow_visualization/dataset_decision_flow.html",
            "workflow_visualization/human_review_workflow.html": "workflow_visualization/human_review_workflow.html",
            "workflow_visualization/workflow_visualization_summary.json": "workflow_visualization/workflow_visualization_summary.json",
            "human_review/human_review_priority_summary.json": "human_review/human_review_priority_summary.json",
            "human_review/human_review_priority_summary.md": "human_review/human_review_priority_summary.md",
            "human_review/top_review_items.csv": "human_review/top_review_items.csv",
            "human_review/top_review_items.json": "human_review/top_review_items.json",
            "human_review/review_decision_template.json": "human_review/review_decision_template.json",
            "human_review/review_decision_prefill.json": "human_review/review_decision_prefill.json",
            "human_review/review_decision_prefill.csv": "human_review/review_decision_prefill.csv",
            "human_review/review_packet_index.json": "human_review/review_packet_index.json",
            "human_review/review_action_guide.md": "human_review/review_action_guide.md",
            "collection/final_dataset.csv": "collection/final_dataset.csv",
            "collection/final_dataset_pre_quality_gate.csv": "collection/final_dataset_pre_quality_gate.csv",
            "collection/quarantined_records.csv": "collection/quarantined_records.csv",
            "collection/pending_review_records.csv": "collection/pending_review_records.csv",
            "collection/non_primary_observations.csv": "collection/non_primary_observations.csv",
            "collection/non_primary_observations.json": "collection/non_primary_observations.json",
            "collection/final_case_dataset.csv": "collection/final_case_dataset.csv",
            "collection/global_outbreak_event_dataset.csv": "collection/global_outbreak_event_dataset.csv",
            "collection/regional_surveillance_dataset.csv": "collection/regional_surveillance_dataset.csv",
            "collection/country_year_aggregate_dataset.csv": "collection/country_year_aggregate_dataset.csv",
            "collection/official_alert_dataset.csv": "collection/official_alert_dataset.csv",
            "collection/zero_case_statements.csv": "collection/zero_case_statements.csv",
            "collection/exposure_monitoring_records.csv": "collection/exposure_monitoring_records.csv",
            "collection/surveillance_summary_records.csv": "collection/surveillance_summary_records.csv",
            "collection/outbreak_summary_records.csv": "collection/outbreak_summary_records.csv",
            "collection/context_records.csv": "collection/context_records.csv",
            "collection/unclassified_observation_records.csv": "collection/unclassified_observation_records.csv",
            "collection/observation_type_dataset_summary.json": "collection/observation_type_dataset_summary.json",
            "collection/final_dataset_post_review.csv": "collection/final_dataset_post_review.csv",
            "collection/record_inclusion_decisions.json": "collection/record_inclusion_decisions.json",
            "diagnostics/run_quality_summary.json": "diagnostics/run_quality_summary.json",
            "diagnostics/final_dataset_quality_summary.json": "diagnostics/final_dataset_quality_summary.json",
            "diagnostics/non_primary_observations.json": "diagnostics/non_primary_observations.json",
            "diagnostics/final_case_dataset.json": "diagnostics/final_case_dataset.json",
            "diagnostics/global_outbreak_event_dataset.json": "diagnostics/global_outbreak_event_dataset.json",
            "diagnostics/regional_surveillance_dataset.json": "diagnostics/regional_surveillance_dataset.json",
            "diagnostics/country_year_aggregate_dataset.json": "diagnostics/country_year_aggregate_dataset.json",
            "diagnostics/official_alert_dataset.json": "diagnostics/official_alert_dataset.json",
            "diagnostics/zero_case_statements.json": "diagnostics/zero_case_statements.json",
            "diagnostics/exposure_monitoring_records.json": "diagnostics/exposure_monitoring_records.json",
            "diagnostics/surveillance_summary_records.json": "diagnostics/surveillance_summary_records.json",
            "diagnostics/outbreak_summary_records.json": "diagnostics/outbreak_summary_records.json",
            "diagnostics/context_records.json": "diagnostics/context_records.json",
            "diagnostics/unclassified_observation_records.json": "diagnostics/unclassified_observation_records.json",
            "diagnostics/observation_type_dataset_summary.json": "diagnostics/observation_type_dataset_summary.json",
            "diagnostics/source_critic_summary.json": "diagnostics/source_critic_summary.json",
            "diagnostics/source_identity_summary.json": "diagnostics/source_identity_summary.json",
            "diagnostics/source_identity_assessments.json": "diagnostics/source_identity_assessments.json",
            "diagnostics/iterative_source_discovery_summary.json": "diagnostics/iterative_source_discovery_summary.json",
            "diagnostics/search_iteration_plans.json": "diagnostics/search_iteration_plans.json",
            "diagnostics/search_iteration_observations.json": "diagnostics/search_iteration_observations.json",
            "diagnostics/search_refinement_decisions.json": "diagnostics/search_refinement_decisions.json",
            "diagnostics/iterative_search_queries.json": "diagnostics/iterative_search_queries.json",
            "diagnostics/disease_relevance_summary.json": "diagnostics/disease_relevance_summary.json",
            "diagnostics/validation_source_compatibility_summary.json": "diagnostics/validation_source_compatibility_summary.json",
            "diagnostics/workflow_summaries.json": "diagnostics/workflow_summaries.json",
        },
    }


def _json_for_script(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


# This second definition intentionally replaces the earlier static report renderer.
# The first version was useful as a summary; this version is an interactive run
# inspector that is closer to how official LangGraph examples communicate value.
def _build_html(result: dict) -> str:
    report = _report_payload(result)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>data collection workflow Run Inspector</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #5b6678;
      --line: #d9e1ec;
      --blue: #155eaa;
      --cyan: #007c89;
      --green: #13795b;
      --amber: #9a6700;
      --red: #b42318;
      --violet: #6d4aff;
      --code: #edf3f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Arial, "Microsoft YaHei", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 34px 44px 26px;
      background: #0f2747;
      color: #fff;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: 0;
    }}
    header p {{
      max-width: 1180px;
      margin: 0;
      color: #dbeafe;
      font-size: 16px;
    }}
    main {{ padding: 24px 32px 52px; }}
    section.band {{
      margin: 0 0 18px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 0 0 10px; }}
    code {{
      padding: 2px 5px;
      border-radius: 4px;
      background: var(--code);
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
    }}
    pre {{
      margin: 0;
      max-height: 360px;
      overflow: auto;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f7f9fc;
      color: #172033;
      white-space: pre-wrap;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 12px;
    }}
    .top-grid {{
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 16px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      min-height: 98px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 6px; font-size: 27px; }}
    .mode {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .mode-item {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
    }}
    .mode-item strong {{ display: block; margin-bottom: 4px; }}
    .mode-item.false strong {{ color: var(--red); }}
    .mode-item.true strong {{ color: var(--green); }}
    .official {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .official article, .note, .evidence-card, .conflict-card, .review-card {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .note {{ border-left: 5px solid var(--amber); background: #fff9eb; }}
    .run-inspector {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 14px;
      min-height: 620px;
    }}
    .stage-list {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdff;
      overflow: hidden;
    }}
    .stage-button {{
      width: 100%;
      padding: 13px 14px;
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
    }}
    .stage-button:hover {{ background: #eef5fd; }}
    .stage-button.active {{
      background: #e8f1fb;
      border-left: 5px solid var(--blue);
      padding-left: 9px;
    }}
    .stage-title {{ display: block; font-weight: 700; }}
    .stage-node {{ display: block; margin-top: 3px; color: var(--muted); font-size: 12px; }}
    .detail {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .detail-head {{
      padding: 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfdff;
    }}
    .detail-body {{ padding: 18px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: #25324a;
      font-size: 12px;
    }}
    .badge.route {{ border-color: #b7a4ff; color: var(--violet); }}
    .tabs {{ display: flex; gap: 8px; margin: 14px 0; }}
    .tab {{
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font-weight: 600;
    }}
    .tab.active {{ background: var(--blue); border-color: var(--blue); color: #fff; }}
    .artifact-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .artifact {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}
    .artifact h4 {{
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #f2f6fb;
      font-size: 13px;
    }}
    .artifact pre {{
      max-height: 290px;
      border: 0;
      border-radius: 0;
    }}
    .pipeline {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
    }}
    .step {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 130px;
      border-top: 5px solid var(--cyan);
    }}
    .step small {{ color: var(--muted); }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .evidence-card {{
      border-left: 5px solid var(--green);
    }}
    .quote {{
      margin-top: 10px;
      padding: 12px;
      border-radius: 6px;
      background: #f4f8f6;
      color: #23362f;
      font-size: 13px;
    }}
    mark {{
      padding: 0 3px;
      border-radius: 3px;
      background: #ffe8a3;
    }}
    .conflict-card {{ border-left: 5px solid var(--red); }}
    .value-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .value {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff7f7;
    }}
    .value strong {{ display: block; font-size: 26px; color: var(--red); }}
    .review-card {{ border-left: 5px solid var(--violet); }}
    .studio-steps {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .studio-steps ol {{ margin-top: 0; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1180px) {{
      .top-grid, .run-inspector, .official, .pipeline, .evidence-grid, .studio-steps {{
        grid-template-columns: 1fr;
      }}
      .metrics, .mode, .artifact-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      main {{ padding-left: 18px; padding-right: 18px; }}
      header {{ padding-left: 24px; padding-right: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>data collection workflow Run Console</h1>
    <p>This console shows the node order, state artifacts, quality gates, validation outputs, anomaly detection, and human-review routing for a completed data collection workflow run.</p>
  </header>
  <main>
    <section class="band top-grid">
      <div>
        <h2>1. Workflow run overview</h2>
        <div class="metrics" id="metrics"></div>
      </div>
      <div>
        <h2>Workflow run controls</h2>
        <div class="mode">
          <div class="mode-item true"><strong>Configured run</strong><span>Runtime mode, live search/fetch, and LLM stages are read from current session artifacts.</span></div>
          <div class="mode-item true"><strong>PublicHealthRecord output</strong><span>Structured extraction produces generic public-health records; accepted output is quality-gated.</span></div>
          <div class="mode-item true"><strong>Workflow inspection</strong><span>Node trace, state artifacts, and conditional routing are inspectable for this run.</span></div>
        </div>
        <p class="note" style="margin-top:12px;">This console is rebuilt from the selected run folder and should describe the current task, counts, validation compatibility, and run-quality result.</p>
      </div>
    </section>

    <section class="band">
      <h2>2. Run quality gate and current task</h2>
      <div id="statusPanel"></div>
    </section>

    <section class="band">
      <h2>3. How the workflow run is inspected</h2>
      <div class="official">
        <article>
          <h3>Runnable workflow</h3>
          <p>Start with an explicit task and inspect graph execution and the state produced by each node.</p>
        </article>
        <article>
          <h3>Studio trace</h3>
          <p>Use Studio to inspect intermediate state, executed nodes, thread history, and output artifacts.</p>
        </article>
        <article>
          <h3>Debug / fork / human-in-the-loop</h3>
          <p>When checks identify conflicts or insufficient evidence, the workflow prepares a review package for a human decision.</p>
        </article>
      </div>
    </section>

    <section class="band">
      <h2>3. Workflow decisions: plan, acquire, verify, and route</h2>
      <div class="pipeline">
        <div class="step"><small>PLAN</small><h3>Scope and strategy</h3><p>Turn the task into a schema, queries, and source strategy.</p></div>
        <div class="step"><small>ACT</small><h3>Acquire sources</h3><p>Discover sources, screen the registry, and retrieve documents.</p></div>
        <div class="step"><small>OBSERVE</small><h3>Evidence triage</h3><p>Identify text containing case, death, date, and location signals.</p></div>
        <div class="step"><small>EXTRACT + VERIFY</small><h3>Records and linked event</h3><p>Extract and normalize records, then assess whether they describe the same event.</p></div>
        <div class="step"><small>CRITIQUE + ROUTE</small><h3>Quality gate to review</h3><p>Route missing fields, incompatible validation, or insufficient evidence to human review.</p></div>
      </div>
    </section>

    <section class="band">
      <h2>4. Interactive run inspector: inspect node state</h2>
      <div class="run-inspector">
        <div class="stage-list" id="stageList"></div>
        <div class="detail">
          <div class="detail-head" id="detailHead"></div>
          <div class="detail-body">
            <div id="detailTakeaway"></div>
            <div class="tabs">
              <button class="tab active" data-tab="artifacts">State artifacts</button>
              <button class="tab" data-tab="trace">Trace metadata</button>
            </div>
            <div id="tabArtifacts"></div>
            <div id="tabTrace" style="display:none;"></div>
          </div>
        </div>
      </div>
    </section>

    <section class="band">
      <h2>5. Final dataset views</h2>
      <div id="datasetViews"></div>
    </section>

    <section class="band">
      <h2>6. Source critic and fetch blocking</h2>
      <div id="sourceCriticPanel"></div>
    </section>

    <section class="band">
      <h2>7. Disease relevance gates</h2>
      <div id="diseaseRelevancePanel"></div>
    </section>

    <section class="band">
      <h2>8. Validation source compatibility</h2>
      <div id="validationCompatibilityPanel"></div>
    </section>

    <section class="band">
      <h2>9. Localized source planning</h2>
      <div id="localizedPlanningPanel"></div>
    </section>

    <section class="band">
      <h2>10. Artifact paths</h2>
      <div id="artifactPanel"></div>
    </section>

    <section class="band">
      <h2>11. Evidence lineage: from text to structured records</h2>
      <div class="evidence-grid" id="evidenceGrid"></div>
    </section>

    <section class="band">
      <h2>6. Quality checks and reasons for human review</h2>
      <div id="conflictPanel"></div>
    </section>

    <section class="band">
      <h2>7. Human review package: records and supporting evidence</h2>
      <div id="reviewPanel"></div>
    </section>

    <section class="band">
      <h2>8. Workflow run sequence</h2>
      <div class="studio-steps">
        <div>
          <h3>Instructions</h3>
          <ol>
            <li>Open or generate a runtime profile for the intended disease, location, and time window.</li>
            <li>Submit the task request shown in this session's task panel.</li>
            <li>Start Studio: <code>python scripts/start_studio.py</code></li>
            <li>Inspect the node order and the conditional branch at <code>quality_gate_routing</code>.</li>
            <li>Export a report and this console using the same configuration: <code>python scripts/run_workflow.py</code></li>
            <li>Open the dataset views and diagnostics listed above.</li>
          </ol>
        </div>
        <div>
          <h3>Run notes</h3>
          <p>This view separates technical completion from accepted data success. A run can complete while producing zero accepted quality-gated records, and that status is shown explicitly above.</p>
          <p class="muted">Regenerate this console after each run so status, counts, validation compatibility, and artifact links match the selected session.</p>
        </div>
      </div>
    </section>
  </main>

  <script id="run-data" type="application/json">{_json_for_script(report)}</script>
  <script>
    const RUN = JSON.parse(document.getElementById("run-data").textContent);
    const state = {{ activeStage: 0, tab: "artifacts" }};

    function esc(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[ch]));
    }}
    function pretty(value) {{
      return esc(JSON.stringify(value, null, 2));
    }}
    function highlight(text) {{
      const escaped = esc(text);
      function escapeRegExpTerm(value) {{
        const slash = String.fromCharCode(92);
        const specials = [slash, ".", "*", "+", "?", "^", "$", "{{", "}}", "(", ")", "|", "[", "]"];
        let out = String(value);
        specials.forEach(ch => {{
          out = out.split(ch).join(slash + ch);
        }});
        return out;
      }}
      const terms = [RUN.task?.location, RUN.task?.disease, RUN.task?.start_date, RUN.task?.end_date]
        .filter(Boolean)
        .map(escapeRegExpTerm);
      if (!terms.length) return escaped;
      return escaped.replace(new RegExp(`(${{terms.join("|")}})`, "gi"), "<mark>$1</mark>");
    }}
    function renderStatusPanel() {{
      const task = RUN.task || {{}};
      const quality = RUN.run_quality?.summary || {{}};
      document.getElementById("statusPanel").innerHTML = `
        <div class="grid-2">
          <article class="note">
            <h3>${{esc(RUN.user_facing_run_status)}}</h3>
            <p>${{esc(RUN.recommended_user_message || quality.recommended_user_message || "")}}</p>
            <p><strong>run_quality_status:</strong> <code>${{esc(quality.run_quality_status)}}</code></p>
            <p><strong>final_dataset_mode:</strong> <code>${{esc(quality.final_dataset_mode)}}</code></p>
            <p><strong>primary_case_dataset_status:</strong> <code>${{esc(quality.primary_case_dataset_status)}}</code></p>
            ${{quality.no_primary_case_dataset_records ? `<p><strong>Warning:</strong> no primary case dataset records were accepted; non-primary observations are preserved separately.</p>` : ""}}
          </article>
          <article class="artifact">
            <h4>Current task</h4>
            <pre>${{pretty(task)}}</pre>
          </article>
        </div>
      `;
    }}
    function renderDatasetViews() {{
      const counts = RUN.counts || {{}};
      const quality = RUN.run_quality?.summary || {{}};
      const datasetCounts = [
        ["Accepted final_dataset", counts.accepted_records],
        ["Pre-quality-gate records", counts.pre_quality_records],
        ["Quarantined records", counts.quarantined_records],
        ["Pending-review records", counts.pending_review_records],
        ["Non-primary observations", counts.non_primary_observations],
        ["Primary-case eligible accepted", counts.primary_case_dataset_eligible],
        ["Corroborated primary case events", counts.corroborated_primary_case_events],
        ["Post-review records", counts.final_dataset_post_review],
        ["Normalized records", counts.records],
      ];
      document.getElementById("datasetViews").innerHTML = `
        <div class="metrics">
          ${{datasetCounts.map(([label, value]) => `<div class="metric"><span>${{esc(label)}}</span><strong>${{esc(value)}}</strong></div>`).join("")}}
        </div>
        <div class="artifact-grid" style="margin-top:12px;">
          <div class="artifact"><h4>Run quality summary</h4><pre>${{pretty(quality)}}</pre></div>
          <div class="artifact"><h4>Final dataset quality summary</h4><pre>${{pretty(RUN.run_quality?.final_dataset_quality_summary || {{}})}}</pre></div>
          <div class="artifact"><h4>Interpretive report summary</h4><pre>${{pretty(RUN.interpretive_report || {{}})}}</pre></div>
          <div class="artifact"><h4>Quarantined records sample</h4><pre>${{pretty(RUN.run_quality?.quarantined_records_sample || [])}}</pre></div>
          <div class="artifact"><h4>Non-primary observations sample</h4><pre>${{pretty(RUN.run_quality?.non_primary_observations_sample || [])}}</pre></div>
          <div class="artifact"><h4>Record inclusion decisions sample</h4><pre>${{pretty(RUN.run_quality?.record_inclusion_decisions_sample || [])}}</pre></div>
        </div>
      `;
    }}
    function renderSummaryPanel(id, value) {{
      document.getElementById(id).innerHTML = `<pre>${{pretty(value || {{}})}}</pre>`;
    }}
    function renderArtifactPanel() {{
      const paths = RUN.key_artifact_paths || {{}};
      document.getElementById("artifactPanel").innerHTML = `
        <table>
          <thead><tr><th>Artifact</th><th>Relative path</th></tr></thead>
          <tbody>
            ${{Object.entries(paths).map(([label, value]) => `<tr><td>${{esc(label)}}</td><td><code>${{esc(value)}}</code></td></tr>`).join("")}}
          </tbody>
        </table>
      `;
    }}
    function renderMetrics() {{
      const labels = [
        ["trace_nodes", "Trace nodes"],
        ["sources", "Sources"],
        ["source_identity_assessments", "Source identity assessments"],
        ["documents", "Documents"],
        ["evidence_chunks", "Evidence chunks"],
        ["accepted_records", "Accepted records"],
        ["records", "Normalized records"],
        ["pre_quality_records", "Pre-quality records"],
        ["quarantined_records", "Quarantined"],
        ["pending_review_records", "Pending review"],
        ["non_primary_observations", "Non-primary observations"],
        ["primary_case_dataset_eligible", "Primary-case eligible accepted"],
        ["corroborated_primary_case_events", "Corroborated primary case events"],
        ["linked_events", "Linked events"],
        ["event_clusters", "Event clusters"],
        ["duplicate_clusters", "Duplicate clusters"],
        ["validation_results", "Validation results"],
        ["anomaly_results", "Anomaly results"],
        ["conflicts", "Conflicts"],
        ["human_review_items", "Review items"],
        ["applied_review_decisions", "Applied review decisions"],
        ["final_dataset_post_review", "Post-review records"],
        ["rejected_incompatible_records", "Disease-mismatch rejected records"],
        ["iterative_search_iterations", "Iterative search iterations"],
        ["iterative_llm_refinement_calls", "Iterative LLM refinement calls"],
      ];
      document.getElementById("metrics").innerHTML = labels.map(([key, label]) => `
        <div class="metric"><span>${{label}}</span><strong>${{esc(RUN.counts[key])}}</strong></div>
      `).join("")
        + `<div class="metric"><span>Run quality</span><strong>${{esc(RUN.run_quality?.summary?.run_quality_status)}}</strong></div>`
        + `<div class="metric"><span>Primary-case dataset status</span><strong>${{esc(RUN.run_quality?.summary?.primary_case_dataset_status)}}</strong></div>`
        + `<div class="metric"><span>Iterative stop decision</span><strong>${{esc(RUN.iterative_source_discovery?.stop_decision)}}</strong></div>`
        + `<div class="metric"><span>Route</span><strong>${{esc(RUN.route)}}</strong></div>`;
    }}
    function renderStageList() {{
      document.getElementById("stageList").innerHTML = RUN.stages.map((stage, idx) => `
        <button class="stage-button ${{idx === state.activeStage ? "active" : ""}}" data-stage="${{idx}}">
          <span class="stage-title">${{esc(stage.title)}}</span>
          <span class="stage-node">${{esc(stage.node)}} · ${{esc(stage.agentic_role)}}</span>
        </button>
      `).join("");
      document.querySelectorAll(".stage-button").forEach(btn => {{
        btn.addEventListener("click", () => {{
          state.activeStage = Number(btn.dataset.stage);
          renderStageList();
          renderStageDetail();
        }});
      }});
    }}
    function renderStageDetail() {{
      const stage = RUN.stages[state.activeStage];
      document.getElementById("detailHead").innerHTML = `
        <h3>${{esc(stage.title)}}</h3>
        <code>${{esc(stage.node)}}</code>
        <div class="badge-row">
          <span class="badge">${{esc(stage.agentic_role)}}</span>
          ${{stage.state_writes.map(k => `<span class="badge route">writes: ${{esc(k)}}</span>`).join("")}}
        </div>
      `;
      document.getElementById("detailTakeaway").innerHTML = `<p class="note">${{esc(stage.takeaway)}}</p>`;
      const artifactEntries = Object.entries(stage.show || {{}});
      document.getElementById("tabArtifacts").innerHTML = `
        <div class="artifact-grid">
          ${{artifactEntries.map(([label, value]) => `
            <div class="artifact"><h4>${{esc(label)}}</h4><pre>${{pretty(value)}}</pre></div>
          `).join("")}}
        </div>
      `;
      document.getElementById("tabTrace").innerHTML = `<pre>${{pretty(stage.trace || {{}})}}</pre>`;
      renderTabs();
    }}
    function renderTabs() {{
      document.querySelectorAll(".tab").forEach(tab => {{
        tab.classList.toggle("active", tab.dataset.tab === state.tab);
        tab.onclick = () => {{
          state.tab = tab.dataset.tab;
          renderTabs();
        }};
      }});
      document.getElementById("tabArtifacts").style.display = state.tab === "artifacts" ? "" : "none";
      document.getElementById("tabTrace").style.display = state.tab === "trace" ? "" : "none";
    }}
    function renderEvidence() {{
      if (!RUN.evidence_links.length) {{
        document.getElementById("evidenceGrid").innerHTML = `
          <article class="note">
            <h3>No accepted evidence-linked records in this run.</h3>
            <p>The workflow may still have completed technically. Check the run quality gate, quarantined records, source critic, and disease relevance panels above.</p>
          </article>
        `;
        return;
      }}
      document.getElementById("evidenceGrid").innerHTML = RUN.evidence_links.map(item => `
        <article class="evidence-card">
          <h3>${{esc(item.publisher)}} · ${{esc(item.source_id)}}</h3>
          <p><strong>${{esc(item.cases_unspecified)}} cases</strong>, ${{esc(item.deaths)}} deaths, ${{esc(item.country)}} / ${{esc(item.date_reported)}}</p>
          <p><code>${{esc(item.record_id)}}</code> from <code>${{esc(item.chunk_id)}}</code></p>
          <p class="muted">method: ${{esc(item.extraction_method)}} · llm_used: ${{esc(item.llm_used)}}</p>
          <div class="quote">${{highlight(item.evidence_quote)}}</div>
        </article>
      `).join("");
    }}
    function renderConflict() {{
      const conflict = RUN.conflicts[0];
      if (!conflict) {{
        document.getElementById("conflictPanel").innerHTML = `<p class="muted">No conflict in this run.</p>`;
        return;
      }}
      document.getElementById("conflictPanel").innerHTML = `
        <article class="conflict-card">
          <h3>${{esc(conflict.conflict_id)}} · ${{esc(conflict.field)}} · ${{esc(conflict.conflict_type)}}</h3>
          <p><strong>Severity:</strong> ${{esc(conflict.severity)}} · <strong>Requires review:</strong> ${{esc(conflict.requires_human_review)}}</p>
          <div class="value-strip">
            ${{(conflict.values || []).map(v => `
              <div class="value">
                <strong>${{esc(v.value)}}</strong>
                <span>${{esc(v.source_id)}}</span><br />
                <code>${{esc(v.record_id)}}</code>
              </div>
            `).join("")}}
          </div>
          <pre style="margin-top:12px;">${{pretty(conflict)}}</pre>
        </article>
      `;
    }}
    function renderReview() {{
      const item = RUN.review_items[0];
      if (!item) {{
        document.getElementById("reviewPanel").innerHTML = `<p class="muted">No review packet in this run.</p>`;
        return;
      }}
      const sections = item.review_packet?.packet_sections || {{}};
      document.getElementById("reviewPanel").innerHTML = `
        <article class="review-card">
          <h3>${{esc(item.review_id)}} · ${{esc(item.item_type)}} · priority ${{esc(item.priority)}}</h3>
          <p>${{esc(item.reason)}}</p>
          <div class="badge-row">
            ${{Object.keys(sections).map(name => `<span class="badge route">${{esc(name)}}</span>`).join("")}}
          </div>
          <pre style="margin-top:12px;">${{pretty(item.review_packet)}}</pre>
        </article>
      `;
    }}
    renderStatusPanel();
    renderDatasetViews();
    renderSummaryPanel("sourceCriticPanel", RUN.source_critic);
    renderSummaryPanel("diseaseRelevancePanel", RUN.disease_relevance);
    renderSummaryPanel("validationCompatibilityPanel", RUN.validation?.validation_source_compatibility_summary || {{}});
    renderSummaryPanel("localizedPlanningPanel", RUN.localized_source_planning);
    renderArtifactPanel();
    renderMetrics();
    renderStageList();
    renderStageDetail();
    renderEvidence();
    renderConflict();
    renderReview();
  </script>
</body>
</html>
"""


def _build_static_html(result: dict) -> str:
    trace = result.get("collection_trace") or []
    executed_nodes = {event.get("node_name") for event in trace}
    package = result.get("final_data_package") or {}
    records = result.get("normalized_records") or []
    conflicts = result.get("conflicts") or []
    review_items = result.get("human_review_queue") or []
    route = result.get("current_route")

    metrics = "\n".join(
        [
            _metric_card("Trace nodes", len(trace), "Number of executed nodes recorded by LangGraph"),
            _metric_card("Sources", _count(result, "source_registry"), "seed catalog + screening"),
            _metric_card("Documents", _count(result, "documents"), "fixture + offline stubs"),
            _metric_card("Evidence chunks", _count(result, "evidence_chunks"), "Traceable evidence chunks"),
            _metric_card("Records", _count(result, "normalized_records"), "Structured observation records"),
            _metric_card("Final case dataset", _count(result, "final_case_dataset"), "primary case records"),
            _metric_card("Zero-case", _count(result, "zero_case_statements"), "no-case statements"),
            _metric_card("Exposure monitoring", _count(result, "exposure_monitoring_records"), "monitoring observations"),
            _metric_card("Context", _count(result, "context_records"), "background/context records"),
            _metric_card("Linked events", _count(result, "linked_events"), "Records grouped by event"),
            _metric_card("Event clusters", _count(result, "event_clusters"), "duplicate-aware event clusters"),
            _metric_card("Duplicate clusters", _count(result, "duplicate_clusters"), "non-countable duplicate groups"),
            _metric_card("Validation results", _count(result, "validation_results"), "trusted-source + cross-source checks"),
            _metric_card("Conflicts", _count(result, "conflicts"), "Cross-source consistency issues"),
            _metric_card("Route", route, "Route selected by the quality gate"),
        ]
    )

    node_cards = "\n".join(
        _node_card(node, title, description, node in executed_nodes)
        for node, title, description in NODE_STORY
    )

    spec = result.get("collection_spec") or {}
    package_meta = package.get("package_metadata") or {}
    provenance = package.get("provenance_manifest") or {}

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangGraph data collection workflow Console</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5e6a7d;
      --line: #d7e0ee;
      --blue: #0b5cab;
      --green: #178254;
      --red: #b42318;
      --gold: #9a6700;
      --bg: #f6f8fb;
      --panel: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Segoe UI", Arial, "Microsoft YaHei", sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 44px 56px 34px;
      color: #fff;
      background: linear-gradient(135deg, #073b78, #0b5cab 54%, #178254);
    }}
    header h1 {{
      margin: 0 0 8px;
      font-size: 42px;
      letter-spacing: 0;
    }}
    header p {{ max-width: 980px; margin: 0; font-size: 18px; opacity: .94; }}
    main {{ padding: 28px 56px 64px; }}
    section.band {{
      margin: 0 0 24px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    h2 {{ margin: 0 0 16px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 17px; }}
    code {{
      padding: 2px 5px;
      border-radius: 4px;
      background: #eef4fb;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: 13px;
    }}
    pre {{
      max-height: 150px;
      overflow: auto;
      margin: 0;
      padding: 10px;
      border-radius: 6px;
      background: #f2f5fa;
      white-space: pre-wrap;
      font-size: 12px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    .metric-label {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ margin-top: 4px; font-size: 28px; font-weight: 700; }}
    .metric-note {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .node-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .node-card {{
      min-height: 160px;
      padding: 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--blue);
      border-radius: 8px;
      background: #fff;
    }}
    .node-card.not-run {{ opacity: .55; border-left-color: #8a95a8; }}
    .node-kicker {{
      margin-bottom: 8px;
      color: var(--green);
      text-transform: uppercase;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: .04em;
    }}
    .comparison {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .comparison article {{
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    ul {{ margin: 8px 0 0 20px; padding: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    th, td {{
      padding: 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{ background: #eef4fb; color: #21395f; }}
    .conflict, .review {{
      margin: 0 0 12px;
      padding: 16px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--red);
      border-radius: 8px;
      background: #fff;
    }}
    .review {{ border-left-color: #6f42c1; }}
    .muted {{ color: var(--muted); }}
    .callout {{
      padding: 16px;
      border-left: 5px solid var(--gold);
      border-radius: 8px;
      background: #fff8e6;
    }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .pill {{
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      font-size: 13px;
      font-weight: 600;
    }}
    .arrow {{ color: var(--muted); }}
    @media (max-width: 1100px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .node-grid, .grid-2, .comparison {{ grid-template-columns: 1fr; }}
      header, main {{ padding-left: 24px; padding-right: 24px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>LangGraph data collection workflow</h1>
    <p>A product workflow console: after a user submits a collection task, the workflow exposes source routing, content fetch, structured extraction, validation comparison, anomaly checks, and human review routing as inspectable nodes.</p>
  </header>
  <main>
    <section class="band">
      <h2>1. Run results</h2>
      <div class="metrics">{metrics}</div>
    </section>

    <section class="band">
      <h2>2. Execution and inspection</h2>
      <div class="comparison">
        <article>
          <h3>Terminal inspection</h3>
          <ul>
            <li>The terminal displays a run summary.</li>
            <li>Use logs to locate failed operations.</li>
            <li>Saved intermediate state supports reproducibility.</li>
            <li>Review, reruns, and branching follow the configured workflow.</li>
          </ul>
        </article>
        <article>
          <h3>Graph inspection</h3>
          <ul>
            <li>Node inputs and outputs form part of the workflow state.</li>
            <li>The trace shows how each node updates state.</li>
            <li>Conditional edges show the route selected by quality gates.</li>
            <li>Studio supports node inspection, reruns, debugging, and human review integration.</li>
          </ul>
        </article>
      </div>
    </section>

    <section class="band">
      <h2>3. Data flow through the graph</h2>
      <div class="flow">
        <span class="pill">Source catalog</span><span class="arrow">→</span>
        <span class="pill">Screening rules</span><span class="arrow">→</span>
        <span class="pill">Live webpage documents</span><span class="arrow">→</span>
        <span class="pill">Evidence chunks</span><span class="arrow">→</span>
        <span class="pill">Structured records</span><span class="arrow">→</span>
        <span class="pill">Linked event</span><span class="arrow">→</span>
        <span class="pill">Conflict</span><span class="arrow">→</span>
        <span class="pill">Human review</span><span class="arrow">→</span>
        <span class="pill">Final package</span>
      </div>
      <p class="muted">The node view follows configured document collection and optional model extraction, showing state updates and review routing.</p>
    </section>

    <section class="band">
      <h2>4. Node details</h2>
      <div class="node-grid">{node_cards}</div>
    </section>

    <section class="band">
      <h2>5. State updates by node</h2>
      <table>
        {_row(["Node", "Main output fields", "Run result", "Inspection notes"], tag="th")}
        {_stage_rows(result)}
      </table>
    </section>

    <section class="band">
      <h2>6. From records to conflicts</h2>
      {_records_table(records)}
    </section>

    <section class="band">
      <h2>7. Cross-source conflicts</h2>
      {_conflict_panels(conflicts)}
    </section>

    <section class="band">
      <h2>8. Human Review Packet</h2>
      {_review_panels(review_items)}
    </section>

    <section class="band">
      <h2>9. Inspect the run in Studio</h2>
      <div class="grid-2">
        <div>
          <h3>Steps</h3>
          <ol>
            <li>Set the provider and model, and supply API credentials through the environment.</li>
            <li>Provide the task disease, location, and time range.</li>
            <li>Start Studio: <code>python scripts/start_studio.py</code></li>
            <li>Export a report and this console using the same configuration: <code>python scripts/run_workflow.py</code></li>
            <li>Open <code>task_intake_and_scope_planning</code> in the trace to inspect the collection specification.</li>
            <li>Open <code>cross_source_consistency_check</code> to inspect conflicts.</li>
            <li>Open <code>quality_gate_routing</code> to inspect the reason for review routing.</li>
            <li>Open <code>human_review</code> to inspect the review package.</li>
          </ol>
        </div>
        <div>
          <h3>Workflow summary</h3>
          <div class="callout">
            The workflow records each collection step in state, preserves structured evidence, routes quality issues to review, and produces auditable outputs.
          </div>
        </div>
      </div>
    </section>

    <section class="band">
      <h2>10. Run metadata</h2>
      <div class="grid-2">
        <div>
          <h3>Collection Spec</h3>
          <pre>{_json(spec)}</pre>
        </div>
        <div>
          <h3>Package Metadata</h3>
          <pre>{_json(package_meta)}</pre>
        </div>
        <div>
          <h3>Provenance Manifest</h3>
          <pre>{_json(provenance)}</pre>
        </div>
        <div>
          <h3>Trace Node Order</h3>
          <pre>{_json([event.get("node_name") for event in trace])}</pre>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def build_report(output_dir: Path, run_output_dir: Path | None = None) -> dict:
    data_source = "workflow_run_artifacts"
    result = _load_workflow_result(run_output_dir)
    if result is None:
        data_source = "fallback_local_graph"
        os.environ["USE_FIXTURE_DOCUMENTS"] = "true"
        os.environ["ENABLE_LIVE_FETCH"] = "false"
        os.environ.setdefault("ENABLE_LLM_EXTRACTION", "false")
        graph = build_graph()
        result = graph.invoke(_initial_state())

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "data_collection_workflow_console.html"
    summary_path = output_dir / "data_collection_workflow_console_summary.json"

    html_path.write_text(_build_html(result), encoding="utf-8")
    report_payload = _report_payload(result)
    workflow_run_result = result.get("workflow_run_result") or {}
    llm_stage_summary = workflow_run_result.get("llm_stage_summary") or {}
    structured_llm = llm_stage_summary.get("structured_extraction") or {}
    source_planning = llm_stage_summary.get("source_planning") or {}
    source_critic = llm_stage_summary.get("source_critic") or {}
    legacy_llm_replay = workflow_run_result.get("llm_replay_summary") or {}
    live_fetch_summary = (
        report_payload.get("fetch_parse", {}).get("live_fetch_summary")
        or workflow_run_result.get("live_fetch_summary", {})
        or {}
    )
    document_parse_summary = (
        report_payload.get("fetch_parse", {}).get("document_parse_summary")
        or result.get("document_parse_summary")
        or workflow_run_result.get("document_parse_summary")
        or {}
    )
    fetch_manifest = (
        report_payload.get("fetch_parse", {}).get("fetch_manifest_sample")
        or result.get("fetch_manifest")
        or workflow_run_result.get("fetch_manifest")
        or []
    )
    live_fetch_enabled = live_fetch_summary.get("live_fetch_enabled")
    if live_fetch_enabled is None and data_source == "workflow_run_artifacts":
        live_fetch_enabled = bool(result.get("documents"))
    counts = report_payload.get("counts") or {}
    task = report_payload.get("task") or {}
    collection_spec = task.get("collection_spec") or {}
    source_search_summary = report_payload.get("source_search") or {}
    iterative_source_discovery_summary = (
        report_payload.get("iterative_source_discovery") or {}
    )
    localized_source_planning_summary = (
        report_payload.get("localized_source_planning") or {}
    )
    source_critic_summary = report_payload.get("source_critic") or {}
    source_credibility_summary = (
        report_payload.get("source_credibility", {}).get("summary") or {}
    )
    source_identity_summary = (
        report_payload.get("source_identity", {}).get("summary") or {}
    )
    disease_relevance_summary = report_payload.get("disease_relevance") or {}
    validation_source_compatibility_summary = (
        report_payload.get("validation", {}).get(
            "validation_source_compatibility_summary"
        )
        or {}
    )
    run_quality_summary = report_payload.get("run_quality", {}).get("summary") or {}
    final_dataset_quality_summary = (
        report_payload.get("run_quality", {}).get("final_dataset_quality_summary")
        or {}
    )
    interpretive_report_summary = report_payload.get("interpretive_report") or {}
    key_artifact_paths = report_payload.get("key_artifact_paths") or {}
    summary = {
        "html_path": str(html_path),
        "summary_path": str(summary_path),
        "run_output_dir": str(run_output_dir or _latest_session_dir(_DEFAULT_WORKFLOW_RUN_ROOT)),
        "data_source": data_source,
        "session_id": workflow_run_result.get("session_id")
        or Path(run_output_dir).name
        if run_output_dir
        else None,
        "task_disease": task.get("disease"),
        "task_location": task.get("location"),
        "task_start_date": task.get("start_date"),
        "task_end_date": task.get("end_date"),
        "task_time_window": task.get("time_window"),
        "task_user_request": task.get("user_request"),
        "collection_spec": collection_spec,
        "collection_spec_status": task.get("collection_spec_status"),
        "user_facing_run_status": report_payload.get("user_facing_run_status"),
        "recommended_user_message": report_payload.get("recommended_user_message"),
        "counts": {
            "trace_nodes": counts.get("trace_nodes", len(result.get("collection_trace") or [])),
            "source_registry": _count(result, "source_registry"),
            "documents": counts.get("documents", _count(result, "documents")),
            "evidence_chunks": counts.get(
                "evidence_chunks", _count(result, "evidence_chunks")
            ),
            "normalized_records": counts.get(
                "records", _count(result, "normalized_records")
            ),
            "accepted_records": counts.get("accepted_records", 0),
            "pre_quality_records": counts.get("pre_quality_records", 0),
            "quarantined_records": counts.get("quarantined_records", 0),
            "pending_review_records": counts.get("pending_review_records", 0),
            "non_primary_observations": counts.get("non_primary_observations", 0),
            "final_case_dataset": counts.get("final_case_dataset", 0),
            "zero_case_statements": counts.get("zero_case_statements", 0),
            "exposure_monitoring_records": counts.get(
                "exposure_monitoring_records", 0
            ),
            "surveillance_summary_records": counts.get(
                "surveillance_summary_records", 0
            ),
            "outbreak_summary_records": counts.get("outbreak_summary_records", 0),
            "context_records": counts.get("context_records", 0),
            "unclassified_observation_records": counts.get(
                "unclassified_observation_records", 0
            ),
            "primary_case_dataset_eligible": counts.get(
                "primary_case_dataset_eligible", 0
            ),
            "corroborated_primary_case_events": counts.get(
                "corroborated_primary_case_events", 0
            ),
            "final_dataset_post_review": counts.get("final_dataset_post_review", 0),
            "linked_events": counts.get("linked_events", _count(result, "linked_events")),
            "event_clusters": counts.get("event_clusters", _count(result, "event_clusters")),
            "duplicate_clusters": counts.get(
                "duplicate_clusters", _count(result, "duplicate_clusters")
            ),
            "validation_results": counts.get(
                "validation_results", _count(result, "validation_results")
            ),
            "conflicts": counts.get("conflicts", _count(result, "conflicts")),
            "human_review_items": counts.get(
                "human_review_items", _count(result, "human_review_queue")
            ),
            "rejected_incompatible_records": disease_relevance_summary.get(
                "rejected_incompatible_record_count", 0
            ),
            "iterative_search_iterations": counts.get(
                "iterative_search_iterations", 0
            ),
            "iterative_llm_refinement_calls": counts.get(
                "iterative_llm_refinement_calls", 0
            ),
        },
        "accepted_record_count": counts.get("accepted_records", 0),
        "pre_quality_record_count": counts.get("pre_quality_records", 0),
        "quarantined_record_count": counts.get("quarantined_records", 0),
        "pending_review_record_count": counts.get("pending_review_records", 0),
        "non_primary_observation_count": counts.get("non_primary_observations", 0),
        "final_case_dataset_count": counts.get("final_case_dataset", 0),
        "zero_case_statement_count": counts.get("zero_case_statements", 0),
        "exposure_monitoring_record_count": counts.get(
            "exposure_monitoring_records", 0
        ),
        "surveillance_summary_record_count": counts.get(
            "surveillance_summary_records", 0
        ),
        "outbreak_summary_record_count": counts.get("outbreak_summary_records", 0),
        "context_record_count": counts.get("context_records", 0),
        "unclassified_observation_count": counts.get(
            "unclassified_observation_records", 0
        ),
        "observation_type_dataset_summary": (
            report_payload.get("observation_type_dataset_summary") or {}
        ),
        "primary_case_dataset_eligible_count": counts.get(
            "primary_case_dataset_eligible", 0
        ),
        "corroborated_primary_case_event_count": counts.get(
            "corroborated_primary_case_events", 0
        ),
        "primary_case_dataset_status": run_quality_summary.get(
            "primary_case_dataset_status"
        )
        or final_dataset_quality_summary.get("primary_case_dataset_status"),
        "post_review_record_count": counts.get("final_dataset_post_review", 0),
        "current_route": result.get("current_route"),
        "contains_synthetic_data": _contains_synthetic_data(
            result.get("final_data_package") or {}
        ),
        "live_fetch_enabled": live_fetch_enabled,
        "live_search_enabled": bool(
            source_search_summary.get("live_search_enabled")
            if "live_search_enabled" in source_search_summary
            else source_search_summary.get("search_enabled")
        ),
        "source_search_execution_summary": source_search_summary,
        "iterative_source_discovery_summary": iterative_source_discovery_summary,
        "iterative_search_iteration_count": iterative_source_discovery_summary.get(
            "search_iteration_count", 0
        ),
        "iterative_llm_refinement_call_count": iterative_source_discovery_summary.get(
            "llm_refinement_call_count", 0
        ),
        "iterative_search_stop_decision": iterative_source_discovery_summary.get(
            "stop_decision"
        ),
        "localized_source_planning_summary": localized_source_planning_summary,
        "source_critic_summary": source_critic_summary,
        "selected_search_derived_fetch_count": live_fetch_summary.get(
            "selected_search_derived_fetch_count"
        ),
        "parser_status_counts": document_parse_summary.get("parser_status_counts")
        or live_fetch_summary.get("parser_status_counts")
        or {},
        "fetch_manifest_count": len(fetch_manifest),
        "all_three_llm_stages_enabled": bool(
            source_planning.get("enabled")
            and source_critic.get("enabled")
            and structured_llm.get("enabled")
        ),
        "llm_source_planning_status": source_planning.get("status"),
        "llm_source_critic_assessed_source_count": source_critic.get(
            "assessed_source_count"
        ),
        "llm_source_critic_blocked_fetch_count": source_critic.get(
            "blocked_fetch_count", 0
        ),
        "llm_call_succeeded": bool(
            structured_llm.get("success_count")
            or legacy_llm_replay.get("llm_call_succeeded")
        ),
        "llm_structured_extraction_call_count": structured_llm.get("call_count"),
        "source_credibility_assessed_source_count": source_credibility_summary.get(
            "assessed_source_count"
        ),
        "source_credibility_role_counts": source_credibility_summary.get(
            "role_counts"
        )
        or {},
        "source_credibility_needs_review_count": source_credibility_summary.get(
            "needs_review_count"
        ),
        "source_identity_assessed_source_count": source_identity_summary.get(
            "identity_assessed_count"
        ),
        "llm_source_identity_assessed_source_count": source_identity_summary.get(
            "llm_identity_assessed_count"
        ),
        "source_identity_warning_counts": source_identity_summary.get(
            "warning_counts"
        )
        or {},
        "disease_relevance_summary": disease_relevance_summary,
        "validation_source_compatibility_summary": (
            validation_source_compatibility_summary
        ),
        "run_quality_status": run_quality_summary.get("run_quality_status"),
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": final_dataset_quality_summary,
        "interpretive_report_summary": interpretive_report_summary,
        "interpretive_one_sentence_conclusion_en": interpretive_report_summary.get(
            "one_sentence_conclusion_en"
        ),
        "interpretive_suitable_as_final_epidemiological_dataset": (
            interpretive_report_summary.get(
                "suitable_as_final_epidemiological_dataset"
            )
        ),
        "key_artifact_paths": key_artifact_paths,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an HTML console from exported Data Collection Workflow artifacts."
    )
    parser.add_argument(
        "--run-output-dir",
        default=None,
        help="Workflow run session directory to visualize. Defaults to latest session.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_CONSOLE_OUTPUT_DIR),
        help="Directory where the HTML console should be written.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _PROJECT_ROOT / output_dir
    run_output_dir = Path(args.run_output_dir) if args.run_output_dir else None
    if run_output_dir is not None and not run_output_dir.is_absolute():
        run_output_dir = _PROJECT_ROOT / run_output_dir
    summary = build_report(output_dir, run_output_dir)
    print("Data Collection Workflow console generated.")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
