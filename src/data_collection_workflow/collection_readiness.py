"""Case-study readiness diagnostics built from completed run artifacts.

This module does not add graph nodes, fetch content, search the web, call an
LLM, or mutate quality gates. It turns existing artifacts into a stage-level
diagnostic funnel so a case-study run cannot silently look "complete" when live
collection never happened.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "not_available"

READINESS_METRICS = [
    "planned_queries",
    "executed_search_queries",
    "source_candidates",
    "collection_allowed_sources",
    "validation_reserved_sources",
    "context_only_sources",
    "rejected_sources",
    "fetched_documents",
    "offline_stub_documents",
    "usable_documents",
    "partial_documents",
    "unusable_documents",
    "parse_deferred_documents",
    "dashboard_or_manual_access_required_sources",
    "evidence_chunks",
    "evidence_chunks_with_case_signal",
    "evidence_chunks_with_death_signal",
    "evidence_chunks_with_date_signal",
    "evidence_chunks_with_location_signal",
    "raw_records",
    "validated_records",
    "normalized_records",
    "pre_quality_records",
    "accepted_primary_case_records",
    "accepted_task_aware_observations",
    "accepted_records",
    "pending_review_records",
    "quarantined_records",
    "validation_rows",
    "human_review_tasks",
]


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _rows(session: Path, stem: str, *, fallback_stems: tuple[str, ...] = ()) -> list[dict]:
    for name in (stem, *fallback_stems):
        for base in (session / "collection", session / "diagnostics"):
            value = _read_json(base / f"{name}.json")
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
            rows = _read_csv(base / f"{name}.csv")
            if rows:
                return rows
    package = _as_dict(_read_json(session / "collection" / "final_package.json"))
    value = package.get(stem)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _dict_artifact(session: Path, stem: str) -> dict:
    for base in (session / "diagnostics", session / "collection"):
        value = _read_json(base / f"{stem}.json")
        if isinstance(value, dict):
            return value
    package = _as_dict(_read_json(session / "collection" / "final_package.json"))
    return _as_dict(package.get(stem))


def _safe_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def _first_int(*values: Any, default: int = 0) -> int:
    for value in values:
        if value not in (None, "", [], {}):
            return _safe_int(value, default)
    return default


def _bool_value(*values: Any, default: bool = False) -> bool:
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)
    return default


def _role(row: dict) -> str:
    return str(
        row.get("source_role_final")
        or row.get("source_role")
        or row.get("recommended_source_role")
        or row.get("default_source_role")
        or ""
    ).strip().lower()


def _status_text(row: dict) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "status",
            "fetch_status",
            "parse_status",
            "quality_status",
            "document_status",
            "document_quality_status",
            "content_status",
            "fetch_mode",
            "notes",
        )
    ).lower()


def _count_source_roles(registry: list[dict]) -> dict[str, int]:
    counts = {
        "collection_allowed_sources": 0,
        "validation_reserved_sources": 0,
        "context_only_sources": 0,
        "rejected_sources": 0,
        "dashboard_or_manual_access_required_sources": 0,
    }
    for row in registry:
        role = _role(row)
        if role in {"collection", "collection_source", "collection_allowed", "data_source"}:
            counts["collection_allowed_sources"] += 1
        elif role in {"validation", "validation_reserved", "reserved_validation"}:
            counts["validation_reserved_sources"] += 1
        elif role in {"context", "context_only", "context_source"}:
            counts["context_only_sources"] += 1
        elif role in {"excluded", "blocked", "search_endpoint", "not_task_relevant"}:
            counts["rejected_sources"] += 1
        text = _status_text(row)
        if "dashboard" in text or "manual" in text or "manual_access_required" in text:
            counts["dashboard_or_manual_access_required_sources"] += 1
    return counts


def _count_document_statuses(documents: list[dict], content_fetch: dict) -> dict[str, int]:
    counts = Counter()
    for row in documents:
        text = _status_text(row)
        if "stub" in text or row.get("offline_metadata_stub"):
            counts["offline_stub_documents"] += 1
        if "partial" in text:
            counts["partial_documents"] += 1
        elif any(token in text for token in ("unusable", "failed", "not_task_relevant")):
            counts["unusable_documents"] += 1
        elif "deferred" in text:
            counts["parse_deferred_documents"] += 1
        else:
            counts["usable_documents"] += 1
    counts["offline_stub_documents"] += _safe_int(
        content_fetch.get("offline_stub_document_count")
        or content_fetch.get("offline_metadata_stub_document_count")
    )
    return dict(counts)


def _field_signal(row: dict, *tokens: str) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "case_signal",
            "death_signal",
            "date_signal",
            "location_signal",
            "signal_types",
            "text",
            "chunk_text",
            "evidence_quote",
        )
    ).lower()
    return any(token in text for token in tokens)


def _validation_rows(session: Path) -> int:
    rows = _read_csv(session / "evaluation" / "evaluation_report.csv")
    return len(rows)


def _human_review_rows(session: Path) -> int:
    rows = _rows(session, "human_review_items")
    if rows:
        return len(rows)
    return len(_read_csv(session / "human_review" / "top_review_items.csv"))


def _text_blob(row: dict) -> str:
    pieces: list[str] = []
    for key in (
        "record_final_inclusion_status",
        "quality_status",
        "quarantine_reason",
        "blocking_reason",
        "review_reason",
        "human_review_reason",
        "quality_gate_blocking_flags",
        "quality_gate_reasons",
        "quality_gate_warnings",
        "semantic_warnings",
        "normalization_warnings",
        "geography_status",
        "geography_warnings",
        "source_trust_status",
    ):
        value = row.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        else:
            pieces.append(str(value or ""))
    return " ".join(pieces).lower()


def _reason_counter(rows: list[dict], keys: tuple[str, ...]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                for item in value:
                    if item not in (None, ""):
                        counts[str(item)] += 1
            elif value not in (None, ""):
                counts[str(value)] += 1
    return dict(counts)


def _dedupe_records(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        rid = str(row.get("record_id") or row.get("source_record_id") or f"row_{index}")
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(row)
    return deduped


def build_quality_gate_failure_breakdown(session_dir: Path | str) -> dict:
    """Explain why candidate records did not become primary case records."""

    session = Path(session_dir)
    pre_quality = _rows(session, "final_dataset_pre_quality_gate")
    primary = _rows(session, "primary_case_dataset", fallback_stems=("final_case_dataset",))
    task_aware = _rows(
        session,
        "task_aware_observation_dataset",
        fallback_stems=("non_primary_observations",),
    )
    pending = _rows(session, "pending_review_records")
    quarantined = _rows(session, "quarantined_records")
    reviewable = pending + quarantined
    all_candidates = _dedupe_records(pre_quality + reviewable)

    def has_any(row: dict, *tokens: str) -> bool:
        text = _text_blob(row)
        return any(token in text for token in tokens)

    return {
        "pre_quality_records_count": len(pre_quality),
        "accepted_primary_case_records_count": len(primary),
        "accepted_task_aware_observations_count": len(task_aware),
        "pending_review_count": len(pending),
        "quarantined_count": len(quarantined),
        "blocking_reason_counts": _reason_counter(
            reviewable or pre_quality,
            (
                "quality_gate_blocking_flags",
                "quarantine_reason",
                "blocking_reason",
                "record_final_inclusion_status",
            ),
        ),
        "review_reason_counts": _reason_counter(
            pending,
            ("review_reason", "human_review_reason", "quality_gate_warnings"),
        ),
        "non_primary_observation_count": len(task_aware)
        or sum(
            1
            for row in all_candidates
            if row.get("primary_case_dataset_eligible") is False
            or has_any(row, "not_primary", "non_primary", "zero_case_statement")
        ),
        "target_metric_not_available_count": sum(
            1
            for row in all_candidates
            if has_any(row, "target_metric_not_available")
        ),
        "unsupported_numeric_claim_count": sum(
            1
            for row in all_candidates
            if has_any(row, "unsupported_numeric", "numeric_semantics_rejected")
        ),
        "time_window_mismatch_count": sum(
            1 for row in all_candidates if has_any(row, "time_window_mismatch")
        ),
        "geography_mismatch_count": sum(
            1 for row in all_candidates if has_any(row, "geography_mismatch", "mismatch")
        ),
        "source_trust_pending_count": sum(
            1
            for row in all_candidates
            if has_any(row, "publisher_unknown", "source_trust_pending")
        ),
        "missing_provenance_count": sum(
            1
            for row in all_candidates
            if has_any(row, "missing_provenance")
            or not row.get("source_url")
            or not row.get("evidence_quote")
        ),
    }


def _failure_stage(metrics: dict, flags: dict, *, real_mode: bool) -> tuple[str, str]:
    if real_mode and (
        not flags["live_search_enabled"]
        or not flags["live_fetch_enabled"]
        or flags["fixture_documents_enabled"]
        or flags["offline_metadata_stub_only"]
        or metrics["offline_stub_documents"] > 0
    ):
        return "completed_without_real_collection", "fixture_or_offline_stub"
    if metrics["executed_search_queries"] <= 0:
        return "diagnostic_failure_at_live_search", "live_search_not_executed"
    if metrics["source_candidates"] <= 0 and real_mode:
        return "diagnostic_failure_at_source_planning", "source_planning_not_available"
    if metrics["fetched_documents"] <= 0:
        return "diagnostic_failure_at_fetch", "fetch_not_successful"
    if metrics["usable_documents"] + metrics["partial_documents"] <= 0:
        return "diagnostic_failure_at_parsing", "parse_unusable"
    if metrics["evidence_chunks"] <= 0:
        return "diagnostic_failure_at_evidence_chunking", "no_evidence_chunks"
    if metrics["raw_records"] <= 0:
        return (
            "diagnostic_failure_at_structured_extraction",
            "no_structured_records",
        )
    if (
        metrics.get("accepted_primary_case_records", 0) <= 0
        and metrics.get("accepted_task_aware_observations", 0) > 0
    ):
        return "diagnostic_failure_at_quality_gate", "non_primary_observations_only"
    if metrics["accepted_records"] <= 0 and (
        metrics["pre_quality_records"] > 0
        or metrics["pending_review_records"] > 0
        or metrics["quarantined_records"] > 0
    ):
        return "diagnostic_failure_at_quality_gate", "quality_gate_no_accepted_records"
    return "none", "ready"


def _readiness_status(metrics: dict, failure_stage: str, *, real_mode: bool) -> str:
    real_requirements_met = (
        metrics["executed_search_queries"] > 0
        and metrics["fetched_documents"] > 0
        and metrics["usable_documents"] + metrics["partial_documents"] > 0
        and metrics["evidence_chunks"] > 0
    )
    if (
        failure_stage == "none"
        and metrics["accepted_records"] > 0
        and metrics["validation_rows"] > 0
        and real_requirements_met
    ):
        return "ready"
    if failure_stage == "none" and metrics["accepted_records"] > 0 and real_requirements_met:
        return "partial"
    if failure_stage == "completed_without_real_collection":
        return "not_ready"
    if failure_stage != "diagnostic_failure_at_quality_gate" and real_mode:
        return "not_ready"
    if real_requirements_met and (
        metrics["raw_records"] > 0
        or metrics["pending_review_records"] > 0
        or metrics["quarantined_records"] > 0
    ):
        return "partial"
    if not real_mode and metrics["accepted_records"] > 0:
        return "partial"
    return "not_ready"


def build_collection_readiness_summary(
    session_dir: Path | str,
    *,
    case_study_real_mode: bool | None = None,
) -> dict:
    """Return a deterministic case-study readiness funnel summary."""

    session = Path(session_dir)
    run_summary = _as_dict(_read_json(session / "workflow_run_summary.json"))
    source_search = _dict_artifact(session, "source_search_execution_summary")
    content_fetch = _dict_artifact(session, "content_fetch_summary")
    registry = _rows(session, "source_registry")
    if not registry:
        registry = _rows(session, "source_candidates", fallback_stems=("search_results_manifest",))
    documents = _rows(session, "documents", fallback_stems=("fetch_manifest",))
    chunks = _rows(session, "evidence_chunks")
    raw_records = _rows(session, "raw_records")
    validated = _rows(session, "validated_records")
    normalized = _rows(session, "normalized_records")
    pre_quality = _rows(session, "final_dataset_pre_quality_gate")
    primary_case = _rows(
        session,
        "primary_case_dataset",
        fallback_stems=("final_case_dataset",),
    )
    task_aware = _rows(
        session,
        "task_aware_observation_dataset",
        fallback_stems=("non_primary_observations",),
    )
    final = _rows(session, "final_dataset")
    pending = _rows(session, "pending_review_records")
    quarantined = _rows(session, "quarantined_records")

    role_counts = _count_source_roles(registry)
    doc_counts = _count_document_statuses(documents, content_fetch)
    real_mode = (
        case_study_real_mode
        if case_study_real_mode is not None
        else get_env("RUN_MODE") == "case_study_real"
    )
    metrics = {
        "planned_queries": _first_int(
            source_search.get("planned_query_count"),
            source_search.get("total_queries_planned"),
            run_summary.get("planned_query_count"),
            len(_rows(session, "search_query_inventory")),
        ),
        "executed_search_queries": _first_int(
            source_search.get("executed_query_count"),
            source_search.get("total_queries_executed"),
            run_summary.get("source_search_executed_query_count"),
        ),
        "source_candidates": _first_int(
            source_search.get("candidate_from_search_count"),
            source_search.get("total_candidates_created"),
            run_summary.get("source_registry_count"),
            len(registry),
        ),
        **role_counts,
        "fetched_documents": _first_int(
            run_summary.get("document_count"),
            content_fetch.get("document_count"),
            len(documents),
        ),
        "offline_stub_documents": doc_counts.get("offline_stub_documents", 0),
        "usable_documents": doc_counts.get("usable_documents", 0),
        "partial_documents": doc_counts.get("partial_documents", 0),
        "unusable_documents": doc_counts.get("unusable_documents", 0),
        "parse_deferred_documents": doc_counts.get("parse_deferred_documents", 0),
        "evidence_chunks": len(chunks),
        "evidence_chunks_with_case_signal": sum(
            1 for row in chunks if _field_signal(row, "case", "cases")
        ),
        "evidence_chunks_with_death_signal": sum(
            1 for row in chunks if _field_signal(row, "death", "deaths", "fatal")
        ),
        "evidence_chunks_with_date_signal": sum(
            1 for row in chunks if _field_signal(row, "date", "week", "month", "202")
        ),
        "evidence_chunks_with_location_signal": sum(
            1 for row in chunks if _field_signal(row, "location", "county", "state", "city", "new mexico", "california")
        ),
        "raw_records": len(raw_records),
        "validated_records": len(validated),
        "normalized_records": len(normalized),
        "pre_quality_records": len(pre_quality),
        "accepted_primary_case_records": len(primary_case),
        "accepted_task_aware_observations": len(task_aware),
        "accepted_records": len(final),
        "pending_review_records": len(pending),
        "quarantined_records": len(quarantined),
        "validation_rows": _validation_rows(session),
        "human_review_tasks": _human_review_rows(session),
    }
    flags = {
        "case_study_real_mode": bool(real_mode),
        "live_search_enabled": _bool_value(
            run_summary.get("live_search_enabled"),
            get_env("ENABLE_LIVE_SEARCH"),
        ),
        "live_fetch_enabled": _bool_value(
            run_summary.get("live_fetch_enabled"),
            get_env("ENABLE_LIVE_FETCH"),
        ),
        "fixture_documents_enabled": _bool_value(
            run_summary.get("fixture_documents_enabled"),
            content_fetch.get("fixture_documents_enabled"),
            content_fetch.get("fixture_documents_present"),
        ),
        "offline_metadata_stub_only": _bool_value(
            run_summary.get("offline_metadata_stub_only"),
            content_fetch.get("offline_metadata_stub_only"),
        ),
    }
    failure, root_class = _failure_stage(metrics, flags, real_mode=bool(real_mode))
    status = _readiness_status(metrics, failure, real_mode=bool(real_mode))
    if failure == "none" and status == "partial" and metrics["validation_rows"] <= 0:
        root_class = "benchmark_validation_not_available"
    not_available_metrics = [
        name
        for name in READINESS_METRICS
        if name not in metrics or metrics[name] == NOT_AVAILABLE
    ]
    return {
        **flags,
        **{name: metrics.get(name, 0) for name in READINESS_METRICS},
        "failure_stage": failure,
        "root_failure_class": root_class,
        "collection_readiness_status": status,
        "not_available_metrics": not_available_metrics,
        "readiness_method": "deterministic_collection_readiness_funnel_v1",
    }


def _write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def _summary_markdown(summary: dict) -> str:
    lines = [
        "# Case Study Readiness Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def write_collection_readiness_outputs(
    session_dir: Path | str,
    *,
    case_study_real_mode: bool | None = None,
) -> dict:
    """Write JSON, CSV, and Markdown readiness diagnostics."""

    session = Path(session_dir)
    diagnostics_dir = session / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    summary = build_collection_readiness_summary(
        session,
        case_study_real_mode=case_study_real_mode,
    )
    json_path = diagnostics_dir / "collection_readiness_summary.json"
    csv_path = diagnostics_dir / "collection_readiness_summary.csv"
    md_path = diagnostics_dir / "collection_readiness_summary.md"
    breakdown_path = diagnostics_dir / "quality_gate_failure_breakdown.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(csv_path, summary)
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")
    breakdown = build_quality_gate_failure_breakdown(session)
    breakdown_path.write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "collection_readiness_summary_json": str(json_path),
        "collection_readiness_summary_csv": str(csv_path),
        "collection_readiness_summary_md": str(md_path),
        "quality_gate_failure_breakdown_json": str(breakdown_path),
    }
