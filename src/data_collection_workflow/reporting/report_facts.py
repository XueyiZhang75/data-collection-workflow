"""Deterministic fact builder for professor-facing final reports.

This module reads completed session artifacts only. It does not call an LLM,
search provider, fetcher, graph node, or quality gate mutator.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


NOT_AVAILABLE = "not_available"
FINAL_REPORT_FACTS_JSON = "final_report_facts.json"
FINAL_REPORT_DIAGNOSTICS_JSON = "final_report_diagnostics.json"

FUNNEL_METRICS = (
    "planned_queries",
    "executed_search_queries",
    "source_candidates",
    "collection_allowed_sources",
    "validation_reserved_sources",
    "context_only_sources",
    "rejected_sources",
    "fetched_documents",
    "offline_stub_documents",
    "usable_or_partial_documents",
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
)

UNKNOWN_PUBLISHER_VALUES = {
    "",
    "unknown",
    "unknown publisher",
    "publisher unknown",
    "publisher_unknown",
    "not_available",
    "not available",
    "none",
    "null",
    "nan",
}

FINAL_STATE_ROW_ARTIFACTS = {
    "final_dataset",
    "primary_case_dataset",
    "final_case_dataset",
    "task_aware_observation_dataset",
    "non_primary_observations",
    "reviewable_dataset",
    "pending_review_records",
    "quarantined_records",
}


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


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _first_present(*values: Any, default: Any = NOT_AVAILABLE) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _count_rows(rows: list[dict], fallback: Any = None) -> int | str:
    if rows:
        return len(rows)
    fallback_int = _safe_int(fallback)
    if fallback_int is not None:
        return fallback_int
    return NOT_AVAILABLE


def _load_artifacts(session_dir: Path | str) -> dict:
    session = Path(session_dir)
    package = _as_dict(_read_json(session / "collection" / "final_package.json"))
    run_summary = _as_dict(_read_json(session / "workflow_run_summary.json"))
    return {
        "session_dir": session,
        "package": package,
        "run_summary": run_summary,
    }


def _artifact_dict(artifacts: dict, stem: str) -> dict:
    session = artifacts["session_dir"]
    package = artifacts["package"]
    for path in (
        session / "diagnostics" / f"{stem}.json",
        session / "collection" / f"{stem}.json",
    ):
        value = _read_json(path)
        if isinstance(value, dict):
            return value
    return _as_dict(package.get(stem))


def _artifact_rows(artifacts: dict, stem: str) -> list[dict]:
    session = artifacts["session_dir"]
    package = artifacts["package"]
    if stem in FINAL_STATE_ROW_ARTIFACTS and stem in package:
        package_value = package.get(stem)
        if isinstance(package_value, list):
            return [row for row in package_value if isinstance(row, dict)]
    for base in (session / "collection", session / "diagnostics"):
        value = _read_json(base / f"{stem}.json")
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        rows = _read_csv(base / f"{stem}.csv")
        if rows:
            return rows
    package_value = package.get(stem)
    if isinstance(package_value, list):
        return [row for row in package_value if isinstance(row, dict)]
    return []


def _has_explicit_row_artifact(artifacts: dict, stem: str) -> bool:
    package = artifacts["package"]
    if isinstance(package.get(stem), list):
        return True
    session = artifacts["session_dir"]
    for base in (session / "collection", session / "diagnostics"):
        json_path = base / f"{stem}.json"
        if json_path.exists() and isinstance(_read_json(json_path), list):
            return True
        if (base / f"{stem}.csv").exists():
            return True
    return False


def _reviewable_records(artifacts: dict, rows: dict[str, list[dict]]) -> list[dict]:
    if _has_explicit_row_artifact(artifacts, "reviewable_dataset"):
        return list(rows["reviewable_dataset"])
    return [
        *rows["pending_review_records"],
        *rows["quarantined_records"],
    ]


def _reviewable_evidence_rows(
    rows: dict[str, list[dict]],
    reviewable_records: list[dict],
) -> list[dict]:
    evidence_rows = (
        rows["evidence_product_dataset"]
        or rows["reviewable_case_dataset"]
        or rows["case_candidate_dataset"]
    )
    evidence_by_record: dict[str, dict] = {}
    for row in evidence_rows:
        record_id = str(row.get("record_id") or "").strip()
        if record_id and record_id not in evidence_by_record:
            evidence_by_record[record_id] = row

    output: list[dict] = []
    for record in reviewable_records:
        record_id = str(record.get("record_id") or "").strip()
        evidence = evidence_by_record.get(record_id)
        output.append({**record, **evidence} if evidence else dict(record))
    return output


def _evaluation_rows(session: Path) -> list[dict] | str:
    evaluation_report = session / "evaluation" / "evaluation_report.csv"
    if not evaluation_report.exists():
        return NOT_AVAILABLE
    return _read_csv(evaluation_report)


def _is_unknown_publisher(value: Any) -> bool:
    return str(value or "").strip().lower() in UNKNOWN_PUBLISHER_VALUES


def _source_id(row: dict) -> str:
    return str(row.get("source_id") or row.get("id") or "")


def _source_role(row: dict) -> str:
    flags = {str(flag) for flag in _as_list(row.get("routing_flags"))}
    role = str(row.get("source_role") or row.get("role") or "").strip()
    if role in {"data_source", "collection_source", "collection_allowed"}:
        return "collection"
    if role:
        return role
    if "validation_reserved" in flags:
        return "validation_reserved"
    if "context_only" in flags:
        return "context_only"
    if row.get("ready_for_content_fetch") is True:
        return "collection"
    return ""


def _source_type(row: dict) -> str:
    return str(
        row.get("source_type_final")
        or row.get("source_type")
        or row.get("source_category")
        or row.get("type")
        or "unknown"
    )


def _source_publisher(row: dict) -> str:
    return str(
        row.get("actual_publisher")
        or row.get("publisher")
        or row.get("source_publisher")
        or ""
    )


def _registry_by_id(rows: list[dict]) -> dict[str, dict]:
    return {_source_id(row): row for row in rows if _source_id(row)}


def _record_publisher(record: dict, source_by_id: dict[str, dict]) -> str:
    source = source_by_id.get(str(record.get("source_id") or "")) or {}
    return str(
        record.get("actual_publisher")
        or record.get("publisher")
        or record.get("source_publisher")
        or source.get("actual_publisher")
        or source.get("publisher")
        or ""
    )


def _is_excluded_source(row: dict) -> bool:
    values = {
        str(row.get("status") or "").lower(),
        str(row.get("screening_decision") or "").lower(),
        str(row.get("critic_decision") or "").lower(),
        str(row.get("final_screening_decision") or "").lower(),
    }
    return bool(values & {"exclude", "excluded", "blocked", "reject", "rejected"})


def _field_signal(row: dict, *keys: str, tokens: tuple[str, ...]) -> bool:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            if value:
                return True
            continue
        text = str(value or "").lower()
        if any(token in text for token in tokens):
            return True
    return False


def _build_collection_funnel(
    artifacts: dict,
    rows: dict[str, list[dict]],
    diagnostics: dict[str, dict],
    evaluation_rows: list[dict] | str,
) -> tuple[dict, list[str]]:
    run_summary = artifacts["run_summary"]
    registry = rows["source_registry"]
    chunks = rows["evidence_chunks"]
    documents = rows["documents"]
    source_search = diagnostics["source_search_execution_summary"]
    content_fetch = diagnostics["content_fetch_summary"]
    missing: list[str] = []

    def metric(name: str, value: Any) -> Any:
        if value == NOT_AVAILABLE:
            missing.append(name)
        return value

    collection_allowed = (
        sum(1 for row in registry if _source_role(row) in {"collection", "collection_source"})
        if registry
        else NOT_AVAILABLE
    )
    validation_reserved = (
        sum(1 for row in registry if _source_role(row) == "validation_reserved")
        if registry
        else NOT_AVAILABLE
    )
    context_only = (
        sum(1 for row in registry if _source_role(row) in {"context_only", "context_source"})
        if registry
        else NOT_AVAILABLE
    )
    rejected = sum(1 for row in registry if _is_excluded_source(row)) if registry else NOT_AVAILABLE
    usable_documents = (
        sum(
            1
            for row in documents
            if str(row.get("quality_status") or row.get("parse_status") or "").lower()
            not in {"failed", "fetch_failed", "unusable", "not_task_relevant"}
        )
        if documents
        else NOT_AVAILABLE
    )
    validation_row_count = (
        len(evaluation_rows) if isinstance(evaluation_rows, list) else NOT_AVAILABLE
    )
    values = {
        "planned_queries": metric(
            "planned_queries",
            _count_rows(rows["search_query_inventory"], run_summary.get("planned_query_count")),
        ),
        "executed_search_queries": metric(
            "executed_search_queries",
            _first_present(
                source_search.get("executed_query_count"),
                run_summary.get("source_search_executed_query_count"),
            ),
        ),
        "source_candidates": metric(
            "source_candidates",
            _count_rows(rows["source_candidates"], run_summary.get("source_registry_count")),
        ),
        "collection_allowed_sources": metric("collection_allowed_sources", collection_allowed),
        "validation_reserved_sources": metric(
            "validation_reserved_sources", validation_reserved
        ),
        "context_only_sources": metric("context_only_sources", context_only),
        "rejected_sources": metric("rejected_sources", rejected),
        "fetched_documents": metric(
            "fetched_documents",
            _count_rows(documents, run_summary.get("document_count")),
        ),
        "offline_stub_documents": metric(
            "offline_stub_documents",
            _first_present(
                content_fetch.get("offline_stub_document_count"),
                content_fetch.get("offline_metadata_stub_document_count"),
                0,
            ),
        ),
        "usable_or_partial_documents": metric(
            "usable_or_partial_documents", usable_documents
        ),
        "evidence_chunks": metric("evidence_chunks", _count_rows(chunks)),
        "evidence_chunks_with_case_signal": metric(
            "evidence_chunks_with_case_signal",
            sum(
                1
                for row in chunks
                if _field_signal(
                    row,
                    "case_signal",
                    "signal_types",
                    "text",
                    "chunk_text",
                    tokens=("case", "cases"),
                )
            )
            if chunks
            else NOT_AVAILABLE,
        ),
        "evidence_chunks_with_death_signal": metric(
            "evidence_chunks_with_death_signal",
            sum(
                1
                for row in chunks
                if _field_signal(
                    row,
                    "death_signal",
                    "signal_types",
                    "text",
                    "chunk_text",
                    tokens=("death", "deaths", "fatal"),
                )
            )
            if chunks
            else NOT_AVAILABLE,
        ),
        "evidence_chunks_with_date_signal": metric(
            "evidence_chunks_with_date_signal",
            sum(
                1
                for row in chunks
                if _field_signal(
                    row,
                    "date_signal",
                    "signal_types",
                    "text",
                    "chunk_text",
                    tokens=("date", "week", "month", "202"),
                )
            )
            if chunks
            else NOT_AVAILABLE,
        ),
        "evidence_chunks_with_location_signal": metric(
            "evidence_chunks_with_location_signal",
            sum(
                1
                for row in chunks
                if _field_signal(
                    row,
                    "location_signal",
                    "signal_types",
                    "text",
                    "chunk_text",
                    tokens=("location", "county", "state", "city"),
                )
            )
            if chunks
            else NOT_AVAILABLE,
        ),
        "raw_records": metric(
            "raw_records",
            _count_rows(rows["raw_records"], run_summary.get("raw_record_count")),
        ),
        "validated_records": metric("validated_records", _count_rows(rows["validated_records"])),
        "normalized_records": metric(
            "normalized_records",
            _count_rows(rows["normalized_records"], run_summary.get("normalized_record_count")),
        ),
        "pre_quality_records": metric(
            "pre_quality_records", len(rows["final_dataset_pre_quality_gate"])
        ),
        "accepted_primary_case_records": metric(
            "accepted_primary_case_records",
            len(rows.get("primary_case_dataset") or rows.get("final_case_dataset") or []),
        ),
        "accepted_task_aware_observations": metric(
            "accepted_task_aware_observations",
            len(
                rows.get("task_aware_observation_dataset")
                or rows.get("non_primary_observations")
                or []
            ),
        ),
        "accepted_records": metric("accepted_records", len(rows["final_dataset"])),
        "pending_review_records": metric(
            "pending_review_records", len(rows["pending_review_records"])
        ),
        "quarantined_records": metric(
            "quarantined_records", len(rows["quarantined_records"])
        ),
        "validation_rows": metric("validation_rows", validation_row_count),
        "human_review_tasks": metric(
            "human_review_tasks", _count_rows(rows["human_review_items"])
        ),
    }
    if not documents and content_fetch.get("fetch_status_counts"):
        fetched = sum(
            int(count or 0)
            for status, count in content_fetch.get("fetch_status_counts", {}).items()
            if "success" in str(status).lower() or "fetched" in str(status).lower()
        )
        values["fetched_documents"] = fetched
    return {name: values.get(name, NOT_AVAILABLE) for name in FUNNEL_METRICS}, missing


def _count_type(record: dict) -> str:
    return str(
        record.get("statistical_count_type")
        or record.get("count_semantics")
        or record.get("resolved_column_period_type")
        or "unspecified"
    )


def _record_period(record: dict) -> str:
    return str(
        record.get("reporting_period")
        or record.get("metric_period_label")
        or record.get("metric_period_start")
        or record.get("period_start_date")
        or record.get("date_reported")
        or record.get("date_anchor")
        or ""
    )


def _case_value(record: dict) -> Any:
    for key in (
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "case_count",
    ):
        value = record.get(key)
        if value not in (None, ""):
            return value
    if str(record.get("metric_category") or "").lower() == "case_count":
        return record.get("metric_value")
    return ""


def _death_value(record: dict) -> Any:
    value = record.get("deaths")
    if value not in (None, ""):
        return value
    if str(record.get("metric_category") or "").lower() == "death_count":
        return record.get("metric_value")
    return ""


def _source_name(record: dict, source_by_id: dict[str, dict]) -> str:
    source = source_by_id.get(str(record.get("source_id") or "")) or {}
    return str(
        record.get("source_name")
        or record.get("source_title")
        or record.get("title")
        or source.get("title")
        or source.get("name")
        or NOT_AVAILABLE
    )


def _source_url(record: dict, source_by_id: dict[str, dict]) -> str:
    source = source_by_id.get(str(record.get("source_id") or "")) or {}
    return str(
        record.get("source_url")
        or record.get("url")
        or record.get("canonical_url")
        or source.get("canonical_url")
        or source.get("url")
        or ""
    )


def _evidence_quote(record: dict) -> str:
    return str(record.get("evidence_quote") or record.get("evidence") or "")


def _final_statistics(
    accepted: list[dict],
    source_by_id: dict[str, dict],
) -> dict:
    table: list[dict] = []
    provenance_warnings: list[dict] = []
    for record in accepted:
        source_url = _source_url(record, source_by_id)
        evidence_quote = _evidence_quote(record)
        warnings = []
        if not source_url:
            warnings.append("missing_source_url")
        if not evidence_quote:
            warnings.append("missing_evidence_quote")
        if warnings:
            provenance_warnings.append(
                {
                    "record_id": record.get("record_id") or NOT_AVAILABLE,
                    "warnings": warnings,
                }
            )
        table.append(
            {
                "record_id": record.get("record_id") or NOT_AVAILABLE,
                "date_or_reporting_period": _record_period(record) or NOT_AVAILABLE,
                "location": record.get("location")
                or record.get("geographic_scope")
                or record.get("subnational_location")
                or NOT_AVAILABLE,
                "cases": _case_value(record),
                "deaths": _death_value(record),
                "statistical_count_type": _count_type(record),
                "source_name": _source_name(record, source_by_id),
                "source_url": source_url or NOT_AVAILABLE,
                "evidence_quote": evidence_quote or NOT_AVAILABLE,
                "quality_status": record.get("quality_status")
                or record.get("record_final_inclusion_status")
                or NOT_AVAILABLE,
            }
        )

    count_types = {row["statistical_count_type"] for row in table}
    periods = {row["date_or_reporting_period"] for row in table}
    aggregation = {
        "status": "no accepted records to aggregate",
        "aggregate_cases": NOT_AVAILABLE,
        "aggregate_deaths": NOT_AVAILABLE,
    }
    if table:
        if len(count_types) == 1 and len(periods) == 1:
            case_values = [_safe_int(row["cases"]) for row in table]
            death_values = [_safe_int(row["deaths"]) for row in table]
            aggregation = {
                "status": "aggregated because count type and reporting period are compatible",
                "statistical_count_type": next(iter(count_types)),
                "reporting_period": next(iter(periods)),
                "aggregate_cases": sum(value for value in case_values if value is not None),
                "aggregate_deaths": sum(value for value in death_values if value is not None),
            }
        else:
            aggregation = {
                "status": "not aggregated because count types are not comparable",
                "statistical_count_types": sorted(count_types),
                "reporting_periods": sorted(periods),
                "aggregate_cases": NOT_AVAILABLE,
                "aggregate_deaths": NOT_AVAILABLE,
            }
    return {
        "accepted_primary_dataset": table,
        "aggregation": aggregation,
        "provenance_warnings": provenance_warnings,
    }


def _source_summary(
    rows: dict[str, list[dict]],
    diagnostics: dict[str, dict],
    accepted: list[dict],
    run_summary: dict,
) -> dict:
    registry = rows["source_registry"]
    identity_summary = diagnostics["source_identity_summary"]
    source_by_id = _registry_by_id(registry)
    source_type_counts = identity_summary.get("source_type_counts")
    if not isinstance(source_type_counts, dict):
        source_type_counts = dict(Counter(_source_type(row) for row in registry))
    unknown_publisher_count = _safe_int(
        identity_summary.get("unknown_publisher_count")
        or identity_summary.get("publisher_unknown_count")
    )
    if unknown_publisher_count is None:
        unknown_publisher_count = sum(
            1 for row in registry if _is_unknown_publisher(_source_publisher(row))
        )
    assessed_count = _safe_int(
        identity_summary.get("identity_assessed_count")
        or identity_summary.get("source_identity_assessed_count")
    )
    if assessed_count is None:
        assessed_count = len(registry)
    known_publisher_count = max(assessed_count - unknown_publisher_count, 0)
    type_text = [(key, str(key).lower()) for key in source_type_counts.keys()]
    official_public_health_sources = sum(
        count
        for key, lower in type_text
        for count in [int(source_type_counts.get(key) or 0)]
        if "official" in lower or "public_health" in lower or "health_agency" in lower
    )

    def count_type_tokens(*tokens: str) -> int:
        return sum(
            int(source_type_counts.get(key) or 0)
            for key, lower in type_text
            if any(token in lower for token in tokens)
        )

    fetch_failures = rows["fetch_failures_blocking"]
    documents = rows["documents"]
    content_fetch = diagnostics["content_fetch_summary"]
    failed_source_count = len(fetch_failures)
    if failed_source_count == 0:
        status_counts = content_fetch.get("fetch_status_counts") or {}
        failed_source_count = sum(
            int(count or 0)
            for status, count in status_counts.items()
            if "fail" in str(status).lower() or "blocked" in str(status).lower()
        )
    fetched_source_count = len(documents)
    if fetched_source_count == 0:
        fetched_source_count = _safe_int(run_summary.get("document_count")) or 0
    if fetched_source_count == 0:
        status_counts = content_fetch.get("fetch_status_counts") or {}
        fetched_source_count = sum(
            int(count or 0)
            for status, count in status_counts.items()
            if "success" in str(status).lower() or "fetched" in str(status).lower()
        )
    accepted_source_counts = Counter(
        _source_name(record, source_by_id) for record in accepted
    )
    excluded_sources = [
        {
            "source_id": row.get("source_id") or NOT_AVAILABLE,
            "source_name": row.get("title") or row.get("name") or NOT_AVAILABLE,
            "reason": row.get("exclusion_reason")
            or row.get("screening_reason")
            or row.get("final_screening_reason")
            or row.get("status")
            or NOT_AVAILABLE,
        }
        for row in registry
        if _is_excluded_source(row)
    ]
    dashboard_manual_pdf_blocked = sum(
        1
        for row in registry
        if any(
            token in str(row.get(key) or "").lower()
            for key in ("source_type_final", "source_type", "content_type", "fetch_status", "status", "notes")
            for token in ("dashboard", "manual", "pdf")
        )
    )
    accepted_unknown_count = sum(
        1
        for record in accepted
        if _is_unknown_publisher(_record_publisher(record, source_by_id))
    )
    return {
        "total_source_candidates": len(registry),
        "source_type_counts": source_type_counts,
        "known_publisher_count": known_publisher_count,
        "publisher_unknown_count": unknown_publisher_count,
        "official_public_health_sources_count": official_public_health_sources,
        "national_source_count": count_type_tokens("national"),
        "state_source_count": count_type_tokens("state"),
        "local_source_count": count_type_tokens("local"),
        "international_source_count": count_type_tokens("international", "global"),
        "news_source_count": count_type_tokens("news"),
        "academic_source_count": count_type_tokens("academic", "journal", "research"),
        "social_source_count": count_type_tokens("social"),
        "unknown_source_count": count_type_tokens("unknown"),
        "fetched_source_count": fetched_source_count,
        "failed_source_count": failed_source_count,
        "dashboard_manual_pdf_blocked_sources_count": dashboard_manual_pdf_blocked,
        "top_accepted_sources": [
            {"source_name": name, "accepted_record_count": count}
            for name, count in accepted_source_counts.most_common(5)
        ],
        "top_excluded_sources": excluded_sources[:5],
        "accepted_publisher_unknown_record_count": accepted_unknown_count,
        "warnings": (
            ["accepted_records_include_publisher_unknown_source"]
            if accepted_unknown_count
            else []
        ),
    }


def _reason_counts(records: list[dict], summary_counts: dict | None = None) -> dict:
    if isinstance(summary_counts, dict) and summary_counts:
        return summary_counts
    counter = Counter()
    for row in records:
        reason = _first_present(
            row.get("exclusion_reason"),
            row.get("review_reason"),
            row.get("quality_gate_reason"),
            row.get("record_final_inclusion_status"),
            default="unspecified",
        )
        counter[str(reason)] += 1
    return dict(counter)


def _provenance_completeness(accepted: list[dict], source_by_id: dict[str, dict]) -> dict:
    total = len(accepted)

    def present_count(check) -> str:
        return f"{sum(1 for record in accepted if check(record))}/{total}"

    if total == 0:
        return {
            "source_url_present": "0/0",
            "evidence_quote_present": "0/0",
            "publisher_known": "0/0",
            "date_anchor_present": "0/0",
            "geography_anchor_present": "0/0",
        }
    return {
        "source_url_present": present_count(lambda record: bool(_source_url(record, source_by_id))),
        "evidence_quote_present": present_count(lambda record: bool(_evidence_quote(record))),
        "publisher_known": present_count(
            lambda record: not _is_unknown_publisher(_record_publisher(record, source_by_id))
        ),
        "date_anchor_present": present_count(
            lambda record: bool(
                record.get("date_anchor")
                or record.get("date_reported")
                or record.get("reporting_period")
                or record.get("metric_period_start")
            )
        ),
        "geography_anchor_present": present_count(
            lambda record: bool(
                record.get("location")
                or record.get("geographic_scope")
                or record.get("subnational_location")
                or record.get("country")
            )
        ),
    }


def _corroboration_status(accepted: list[dict], diagnostics: dict[str, dict]) -> dict:
    summary = diagnostics["corroboration_summary"]
    taxonomy_counter: Counter = Counter()
    if accepted:
        from data_collection_workflow.collection_diagnostics import classify_single_source_record

        taxonomy_counter = Counter(
            classify_single_source_record(record) for record in accepted
        )
    official_not_cross_validated = taxonomy_counter.get(
        "official_single_source_not_cross_validated", 0
    )
    non_official_unverified = taxonomy_counter.get(
        "non_official_single_source_unverified", 0
    )
    multi_source_corroborated = taxonomy_counter.get("multi_source_corroborated", 0)
    conflicting_sources = taxonomy_counter.get("conflicting_sources", 0)
    needs_human_review = taxonomy_counter.get("needs_human_review", 0)
    if summary:
        if not accepted:
            official_not_cross_validated = _first_present(
                summary.get("official_single_source_not_cross_validated_count"),
                summary.get("official_single_source_count"),
                default=0,
            )
            non_official_unverified = _first_present(
                summary.get("non_official_single_source_unverified_count"),
                summary.get("single_source_unverified_count"),
                default=0,
            )
            multi_source_corroborated = _first_present(
                summary.get("multi_source_corroborated_count"),
                summary.get("corroborated_primary_case_event_count"),
                default=0,
            )
            conflicting_sources = _first_present(
                summary.get("conflicting_source_count"),
                summary.get("conflicting_claim_count"),
                summary.get("conflicting_claims_count"),
                default=0,
            )
        return {
            "multi_source_corroborated": _first_present(
                summary.get("multi_source_corroborated_count"),
                summary.get("corroborated_primary_case_event_count"),
                default=multi_source_corroborated,
            ),
            "official_single_source": _first_present(
                summary.get("official_single_source_count"),
                default=official_not_cross_validated,
            ),
            "official_single_source_not_cross_validated": (
                official_not_cross_validated
            ),
            "non_official_single_source_unverified": non_official_unverified,
            "single_source_unverified": _first_present(
                summary.get("single_source_unverified_count"),
                default=non_official_unverified,
            ),
            "conflicting_claims": _first_present(
                summary.get("conflicting_claim_count"),
                summary.get("conflicting_claims_count"),
                default=conflicting_sources,
            ),
            "conflicting_sources": conflicting_sources,
            "needs_human_review": needs_human_review,
        }
    counter = Counter(
        str(record.get("corroboration_status") or "single_source_unverified")
        for record in accepted
    )
    return {
        "multi_source_corroborated": counter.get("multi_source_corroborated", 0)
        + counter.get("corroborated", 0)
        + counter.get("cross_source_supported", 0),
        "official_single_source": counter.get("official_single_source", 0),
        "official_single_source_not_cross_validated": (
            official_not_cross_validated
        ),
        "non_official_single_source_unverified": non_official_unverified,
        "single_source_unverified": counter.get("single_source_unverified", 0),
        "conflicting_claims": counter.get("conflicting_claims", 0),
        "conflicting_sources": conflicting_sources,
        "needs_human_review": needs_human_review,
    }


def _review_action_for(row: dict, item_type: str) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "reason",
            "review_reason",
            "exclusion_reason",
            "record_final_inclusion_status",
            "quality_status",
            "item_type",
        )
    ).lower()
    checks = []
    if "case" in text or row.get("cases_confirmed") not in (None, ""):
        checks.append("case count")
    if "death" in text or row.get("deaths") not in (None, ""):
        checks.append("death count")
    if "date" in text or "period" in text:
        checks.append("date window")
    if "location" in text or "geograph" in text:
        checks.append("location/geography")
    if "publisher" in text or "source" in text or "identity" in item_type:
        checks.append("source identity")
    if "count type" in text or row.get("statistical_count_type"):
        checks.append("count type")
    if "conflict" in text:
        checks.append("source conflict")
    if not checks:
        checks = [
            "case count",
            "death count",
            "date window",
            "location/geography",
            "source identity",
            "count type",
            "source conflict",
        ]
    return "Verify " + ", ".join(dict.fromkeys(checks)) + "."


def _human_review_tasks(rows: dict[str, list[dict]]) -> list[dict]:
    tasks: list[dict] = []
    for idx, item in enumerate(rows["human_review_items"], start=1):
        item_type = str(item.get("review_task_type") or item.get("item_type") or "general_review")
        tasks.append(
            {
                "priority": item.get("priority") or item.get("priority_level") or f"P{min(idx, 3)}",
                "review_task_type": item_type,
                "record_id_source_id_packet_id": _first_present(
                    item.get("record_id"),
                    item.get("source_id"),
                    item.get("packet_id"),
                    item.get("review_id"),
                ),
                "reason": item.get("reason") or item.get("review_reason") or NOT_AVAILABLE,
                "expected_human_action": item.get("expected_human_action")
                or item.get("suggested_action")
                or _review_action_for(item, item_type),
            }
        )
    for item_type, records in (
        ("pending_record_review", rows["pending_review_records"]),
        ("quarantined_record_audit", rows["quarantined_records"]),
    ):
        for record in records:
            tasks.append(
                {
                    "priority": "P1" if item_type == "pending_record_review" else "P2",
                    "review_task_type": item_type,
                    "record_id_source_id_packet_id": record.get("record_id")
                    or record.get("source_id")
                    or NOT_AVAILABLE,
                    "reason": _first_present(
                        record.get("review_reason"),
                        record.get("exclusion_reason"),
                        record.get("quality_status"),
                        record.get("record_final_inclusion_status"),
                    ),
                    "expected_human_action": _review_action_for(record, item_type),
                }
            )
    return tasks


def _validation_readiness(
    session: Path,
    evaluation_rows: list[dict] | str,
    diagnostics: dict[str, dict],
    run_summary: dict,
) -> dict:
    validation_summary = diagnostics["validation_source_compatibility_summary"]
    active_validation = _safe_int(
        validation_summary.get("active_validation_record_count")
        or run_summary.get("active_validation_record_count")
    )
    held_out_available = "yes" if active_validation and active_validation > 0 else "no"
    evaluation_count: int | str = (
        len(evaluation_rows) if isinstance(evaluation_rows, list) else NOT_AVAILABLE
    )
    masking_leakage_count: int | str = NOT_AVAILABLE
    github_status: str = NOT_AVAILABLE
    if isinstance(evaluation_rows, list):
        masking_leakage_count = sum(
            1
            for row in evaluation_rows
            if str(row.get("masking_leakage") or row.get("leakage") or "").lower()
            in {"1", "true", "yes", "leakage"}
        )
        masking_fields_present = any(
            any(
                field in row
                for field in (
                    "masking_compliance_status",
                    "masking_leakage",
                    "leakage",
                )
            )
            for row in evaluation_rows
        )
        if masking_fields_present:
            github_status = (
                "masking_leakage_detected"
                if masking_leakage_count
                else "masking_checked_no_leakage"
            )
        elif evaluation_rows:
            github_status = "not_masked_open_diagnostic"
    validation_limited = bool(
        validation_summary.get("validation_limited")
        or validation_summary.get("no_compatible_validation_source")
    )
    if isinstance(evaluation_count, int) and evaluation_count > 0 and not validation_limited:
        ready = "yes"
        reason = "evaluation_report.csv is available and validation is not marked limited"
    elif isinstance(evaluation_count, int) and evaluation_count > 0:
        ready = "partial"
        reason = "evaluation rows exist, but validation is marked limited"
    elif isinstance(evaluation_count, int):
        ready = "no"
        reason = "evaluation_report.csv has zero rows"
    else:
        ready = "no"
        reason = "evaluation_report.csv is not available"
    return {
        "held_out_validation_sources_available": held_out_available,
        "github_benchmark_visible_or_masked": github_status,
        "masking_leakage_count": masking_leakage_count,
        "evaluation_rows_count": evaluation_count,
        "ready_for_benchmark_comparison": ready,
        "reason_if_not_ready": reason if ready != "yes" else "",
        "evaluation_report_path": (
            str(session / "evaluation" / "evaluation_report.csv")
            if (session / "evaluation" / "evaluation_report.csv").exists()
            else NOT_AVAILABLE
        ),
    }


def _exported_artifacts(session: Path) -> list[dict]:
    artifact_paths = [
        "collection/final_dataset.csv",
        "collection/primary_case_dataset.csv",
        "collection/task_aware_observation_dataset.csv",
        "collection/reviewable_dataset.csv",
        "collection/evidence_product_dataset.csv",
        "collection/case_candidate_dataset.csv",
        "collection/reviewable_case_dataset.csv",
        "collection/case_evidence_bundles.csv",
        "collection/workflow_case_bundle_line_list.csv",
        "collection/workflow_case_candidate_line_list.csv",
        "collection/source_inventory.csv",
        "collection/authority_source_inventory.csv",
        "collection/final_dataset_pre_quality_gate.csv",
        "collection/pending_review_records.csv",
        "collection/quarantined_records.csv",
        "collection/source_registry.json",
        "collection/source_registry.csv",
        "diagnostics/run_quality_summary.json",
        "diagnostics/collection_readiness_summary.json",
        "diagnostics/quality_gate_failure_breakdown.json",
        "diagnostics/source_coverage_audit.json",
        "diagnostics/human_review_audit_trail.json",
        "evaluation/evaluation_report.csv",
    ]
    return [
        {
            "artifact": artifact,
            "available": (session / artifact).exists(),
            "path": str(session / artifact) if (session / artifact).exists() else NOT_AVAILABLE,
        }
        for artifact in artifact_paths
    ]


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _run_status(
    *,
    run_quality_status: str,
    accepted_count: int,
    task_aware_count: int,
    pre_quality_count: int,
    raw_count: int | str,
    pending_count: int,
    validation_limited: bool,
    live_search: bool,
    live_fetch: bool,
) -> str:
    if not live_search and not live_fetch:
        return "completed_without_real_collection"
    if raw_count == 0 or run_quality_status == "no_records_extracted":
        return "no_records_extracted"
    if run_quality_status == "no_task_relevant_records":
        return "no_task_relevant_records"
    if accepted_count == 0 and task_aware_count > 0:
        return "non_primary_observations_only"
    if accepted_count == 0 and pending_count > 0:
        return "pending_review"
    if accepted_count == 0 and pre_quality_count > 0:
        return "failed_quality_gate"
    if validation_limited and run_quality_status in {
        "validation_limited_no_compatible_source",
        "no_task_compatible_validation_source",
    }:
        return "validation_limited_no_compatible_source"
    return run_quality_status or "completed"


def _main_limitation(
    *,
    real_status: str,
    pending_count: int,
    quarantined_count: int,
    validation_rows_count: int | str,
    final_epi: str,
) -> str:
    if real_status == "no_records_extracted":
        return "No raw task-relevant public health records were extracted."
    if real_status == "failed_quality_gate":
        return "Candidate records existed, but none passed deterministic quality gates."
    if real_status == "pending_review":
        return "Records remain pending human review and cannot be treated as accepted."
    if real_status == "validation_limited_no_compatible_source":
        return "No task-compatible validation source was available for benchmark-grade validation."
    if validation_rows_count == NOT_AVAILABLE:
        return "No evaluation_report.csv artifact is available, so validation readiness is not established."
    if pending_count:
        return "Some records remain pending review and are excluded from primary statistics."
    if quarantined_count:
        return "Some candidate records were quarantined and are excluded from primary statistics."
    if final_epi == "no":
        return "The accepted evidence is not sufficient for final epidemiological use."
    return "No major deterministic reporting limitation was identified."


def _source_registry_profile(rows: dict[str, list[dict]]) -> list[dict]:
    source_rows = rows.get("source_inventory") or rows.get("source_registry") or []
    profile: list[dict] = []
    for row in source_rows[:20]:
        profile.append(
            {
                "source_id": row.get("source_id") or row.get("id") or NOT_AVAILABLE,
                "source_name": row.get("title")
                or row.get("name")
                or row.get("source_name")
                or NOT_AVAILABLE,
                "source_product_type": row.get("data_product_type")
                or row.get("source_product_type")
                or NOT_AVAILABLE,
                "source_type": row.get("source_type_final")
                or row.get("source_type")
                or NOT_AVAILABLE,
                "authority_score": row.get("authority_score")
                or row.get("credibility_score")
                or row.get("authority_bucket")
                or NOT_AVAILABLE,
                "task_specificity": row.get("task_specificity") or NOT_AVAILABLE,
                "time_window_fit": row.get("time_window_fit") or NOT_AVAILABLE,
                "machine_readability": row.get("machine_readability") or NOT_AVAILABLE,
                "screening_decision": row.get("screening_decision")
                or row.get("final_screening_decision")
                or row.get("fetch_status")
                or NOT_AVAILABLE,
            }
        )
    return profile


def _fetch_manifest_rows(
    rows: dict[str, list[dict]],
    diagnostics: dict[str, dict],
) -> list[dict]:
    manifest = diagnostics["content_fetch_summary"].get("selection_manifest")
    if isinstance(manifest, list) and manifest:
        return [
            {
                "source_id": row.get("source_id") or NOT_AVAILABLE,
                "fetch_selected": row.get("fetch_selected")
                if row.get("fetch_selected") is not None
                else row.get("selected", NOT_AVAILABLE),
                "data_product_type": row.get("data_product_type") or NOT_AVAILABLE,
                "task_specificity": row.get("task_specificity") or NOT_AVAILABLE,
                "skip_reason": row.get("source_product_skip_reason")
                or row.get("skip_reason")
                or row.get("selection_reason")
                or NOT_AVAILABLE,
                "source_url": row.get("url")
                or row.get("canonical_url")
                or row.get("source_url")
                or NOT_AVAILABLE,
            }
            for row in manifest[:30]
            if isinstance(row, dict)
        ]
    document_rows = rows.get("documents") or []
    fetch_failures = rows.get("fetch_failures_blocking") or []
    return [
        {
            "source_id": row.get("source_id") or NOT_AVAILABLE,
            "fetch_selected": "yes",
            "data_product_type": row.get("data_product_type") or NOT_AVAILABLE,
            "task_specificity": row.get("task_specificity") or NOT_AVAILABLE,
            "skip_reason": row.get("fetch_status") or row.get("parse_status") or NOT_AVAILABLE,
            "source_url": row.get("source_url")
            or row.get("canonical_url")
            or row.get("url")
            or NOT_AVAILABLE,
        }
        for row in document_rows[:20]
    ] + [
        {
            "source_id": row.get("source_id") or NOT_AVAILABLE,
            "fetch_selected": "no",
            "data_product_type": row.get("data_product_type") or NOT_AVAILABLE,
            "task_specificity": row.get("task_specificity") or NOT_AVAILABLE,
            "skip_reason": row.get("reason") or row.get("message") or NOT_AVAILABLE,
            "source_url": row.get("source_url") or row.get("url") or NOT_AVAILABLE,
        }
        for row in fetch_failures[:10]
    ]


def _record_provenance_rows(
    rows: dict[str, list[dict]],
    source_by_id: dict[str, dict],
) -> list[dict]:
    evidence_rows = rows.get("evidence_product_dataset") or []
    if evidence_rows:
        return [
            {
                "record_or_evidence_id": row.get("record_id")
                or row.get("evidence_id")
                or NOT_AVAILABLE,
                "source_id": row.get("source_id") or NOT_AVAILABLE,
                "source_url": row.get("source_url") or NOT_AVAILABLE,
                "source_product_type": row.get("source_product_type") or NOT_AVAILABLE,
                "quality_status": row.get("quality_status") or NOT_AVAILABLE,
                "review_reason": row.get("review_reason") or NOT_AVAILABLE,
                "evidence_quote": row.get("evidence_quote") or NOT_AVAILABLE,
            }
            for row in evidence_rows[:30]
        ]
    candidates = (
        rows.get("case_candidate_dataset")
        or rows.get("pending_review_records")
        or rows.get("quarantined_records")
        or rows.get("final_dataset")
        or []
    )
    return [
        {
            "record_or_evidence_id": row.get("record_id") or NOT_AVAILABLE,
            "source_id": row.get("source_id") or NOT_AVAILABLE,
            "source_url": _source_url(row, source_by_id) or NOT_AVAILABLE,
            "source_product_type": row.get("data_product_type") or NOT_AVAILABLE,
            "quality_status": row.get("record_final_inclusion_status")
            or row.get("quality_status")
            or NOT_AVAILABLE,
            "review_reason": row.get("review_reason")
            or row.get("exclusion_reason")
            or NOT_AVAILABLE,
            "evidence_quote": _evidence_quote(row) or NOT_AVAILABLE,
        }
        for row in candidates[:30]
    ]


def _failure_funnel(
    rows: dict[str, list[dict]],
    diagnostics: dict[str, dict],
    collection_funnel: dict,
) -> dict:
    content_fetch = diagnostics["content_fetch_summary"]
    selection_manifest = content_fetch.get("selection_manifest")
    skipped_fetch = 0
    skip_reasons = Counter()
    if isinstance(selection_manifest, list):
        for row in selection_manifest:
            if not isinstance(row, dict):
                continue
            selected = row.get("fetch_selected")
            if selected is False or str(selected).lower() in {"false", "no", "0"}:
                skipped_fetch += 1
                reason = (
                    row.get("source_product_skip_reason")
                    or row.get("skip_reason")
                    or row.get("selection_reason")
                    or "unspecified"
                )
                skip_reasons[str(reason)] += 1
    return {
        "source_candidates": collection_funnel.get("source_candidates"),
        "rejected_sources": collection_funnel.get("rejected_sources"),
        "fetch_skipped_sources": skipped_fetch,
        "fetch_skip_top_reasons": dict(skip_reasons.most_common(5)),
        "fetched_documents": collection_funnel.get("fetched_documents"),
        "evidence_chunks": collection_funnel.get("evidence_chunks"),
        "raw_records": collection_funnel.get("raw_records"),
        "pre_quality_records": collection_funnel.get("pre_quality_records"),
        "accepted_records": collection_funnel.get("accepted_records"),
        "pending_review_records": collection_funnel.get("pending_review_records"),
        "quarantined_records": collection_funnel.get("quarantined_records"),
    }


def _source_to_evidence_funnel(rows: dict[str, list[dict]]) -> dict:
    source_rows = rows.get("source_inventory") or rows.get("source_registry") or []

    def row_int(row: dict, key: str) -> int:
        value = _safe_int(row.get(key))
        return value if value is not None else 0

    fetched = 0
    parsed = 0
    target_chunks = 0
    target_records = 0
    status_counts: Counter[str] = Counter()
    for row in source_rows:
        status = str(row.get("source_to_evidence_status") or "").strip()
        if status:
            status_counts[status] += 1
        fetch_status = str(row.get("fetch_status") or "").lower()
        parse_status = str(row.get("parse_status") or "").lower()
        if fetch_status in {"success", "fetched", "ok"} or status.startswith(
            ("fetched", "parsed", "evidence")
        ):
            fetched += 1
        if parse_status in {"success", "parsed", "ok"} or status.startswith(
            ("parsed", "evidence")
        ):
            parsed += 1
        target_chunks += row_int(row, "target_chunk_count")
        target_records += row_int(row, "target_record_count")
    if target_records == 0:
        target_records = len(rows.get("case_candidate_dataset") or [])
    return {
        "sources": len(source_rows),
        "fetched_sources": fetched,
        "parsed_sources": parsed,
        "target_chunks": target_chunks,
        "target_records": target_records,
        "case_candidates": len(rows.get("case_candidate_dataset") or []),
        "reviewable_case_candidates": len(rows.get("reviewable_case_dataset") or []),
        "final_records": len(
            rows.get("primary_case_dataset")
            or rows.get("final_case_dataset")
            or rows.get("final_dataset")
            or []
        ),
        "quarantined_records": len(rows.get("quarantined_records") or []),
        "source_to_evidence_status_counts": dict(status_counts.most_common()),
    }


def _is_high_confidence_source(row: dict) -> bool:
    source_type = str(
        row.get("source_type_final")
        or row.get("source_type")
        or row.get("source_category")
        or ""
    ).lower()
    product = str(
        row.get("source_product_type")
        or row.get("data_product_type")
        or ""
    ).lower()
    publisher = str(
        row.get("actual_publisher")
        or row.get("publisher")
        or row.get("source_name")
        or ""
    ).lower()
    score = row.get("authority_score") or row.get("credibility_score")
    score_float = None
    try:
        score_float = float(score)
    except (TypeError, ValueError):
        pass
    trusted_tokens = (
        "official",
        "public_health",
        "government",
        "ministry",
        "authority",
        "structured",
        "database",
        "peer",
        "scientific",
        "literature",
        "journal",
    )
    if score_float is not None and score_float >= 0.8:
        return True
    return any(token in source_type for token in trusted_tokens) or any(
        token in product for token in trusted_tokens
    ) or any(token in publisher for token in ("who", "cdc", "ministry", "health"))


def _high_confidence_sources_with_no_records(rows: dict[str, list[dict]]) -> list[dict]:
    output: list[dict] = []
    for row in rows.get("source_inventory") or []:
        status = str(row.get("source_to_evidence_status") or "")
        target_records = _safe_int(row.get("target_record_count")) or 0
        target_chunks = _safe_int(row.get("target_chunk_count")) or 0
        if status != "parsed_target_source_no_record_extracted" and not (
            target_chunks > 0 and target_records == 0 and str(row.get("parse_status") or "").lower()
            in {"success", "parsed", "ok"}
        ):
            continue
        if not _is_high_confidence_source(row):
            continue
        output.append(
            {
                "source_id": row.get("source_id") or NOT_AVAILABLE,
                "source_name": row.get("source_name")
                or row.get("title")
                or row.get("name")
                or NOT_AVAILABLE,
                "source_type": row.get("source_type_final")
                or row.get("source_type")
                or NOT_AVAILABLE,
                "source_product_type": row.get("source_product_type")
                or row.get("data_product_type")
                or NOT_AVAILABLE,
                "source_to_evidence_status": status or NOT_AVAILABLE,
                "target_chunk_count": target_chunks,
                "target_record_count": target_records,
                "source_only_reason": row.get("source_only_reason") or NOT_AVAILABLE,
                "extraction_failure_substage": row.get("extraction_failure_substage")
                or NOT_AVAILABLE,
                "source_url": row.get("source_url")
                or row.get("canonical_url")
                or row.get("url")
                or NOT_AVAILABLE,
            }
        )
    return output[:20]


def _disease_local_rejection_counts(rows: dict[str, list[dict]]) -> dict:
    quarantined = rows.get("quarantined_records") or []
    reason_counts: Counter[str] = Counter()
    incompatible_terms: Counter[str] = Counter()
    local_status_counts: Counter[str] = Counter()
    for row in quarantined:
        reason = str(
            row.get("quarantine_reason")
            or row.get("exclusion_reason")
            or row.get("record_final_inclusion_status")
            or "unspecified"
        )
        if "local_evidence_disease_mismatch" in reason or str(
            row.get("record_local_disease_relevance_status") or ""
        ) == "incompatible_disease":
            reason_counts["local_evidence_disease_mismatch"] += 1
        else:
            reason_counts[reason] += 1
        status = str(
            row.get("record_local_numeric_disease_status")
            or row.get("record_local_disease_relevance_status")
            or ""
        )
        if status:
            local_status_counts[status] += 1
        terms = row.get("record_local_numeric_incompatible_terms_found") or row.get(
            "record_local_incompatible_terms_found"
        )
        if isinstance(terms, str):
            terms = [term.strip() for term in terms.split(";") if term.strip()]
        for term in terms or []:
            incompatible_terms[str(term)] += 1
    return {
        "quarantined_records": len(quarantined),
        "local_evidence_disease_mismatch": reason_counts.get(
            "local_evidence_disease_mismatch", 0
        ),
        "rejection_reason_counts": dict(reason_counts.most_common()),
        "local_numeric_status_counts": dict(local_status_counts.most_common()),
        "incompatible_terms_found": dict(incompatible_terms.most_common(10)),
    }


def _case_field_completeness_summary(rows: dict[str, list[dict]]) -> dict:
    candidate_rows = (
        rows.get("workflow_case_candidate_line_list")
        or rows.get("case_candidate_dataset")
        or rows.get("reviewable_case_dataset")
        or []
    )
    scores: list[float] = []
    missing_fields: Counter[str] = Counter()
    with_case_span = 0
    for row in candidate_rows:
        score = row.get("field_completeness_score")
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            pass
        missing = row.get("missing_key_fields")
        if isinstance(missing, str):
            parts = [item.strip() for item in missing.replace(",", ";").split(";")]
        elif isinstance(missing, list):
            parts = [str(item).strip() for item in missing]
        else:
            parts = []
        for field in parts:
            if field:
                missing_fields[field] += 1
        if row.get("case_span_quote") or row.get("case_span_id"):
            with_case_span += 1
    average = round(sum(scores) / len(scores), 3) if scores else NOT_AVAILABLE
    return {
        "candidate_rows": len(candidate_rows),
        "average_field_completeness_score": average,
        "rows_with_case_span_evidence": with_case_span,
        "missing_key_fields_top": dict(missing_fields.most_common(10)),
    }


def _high_confidence_exact_page_recall(
    rows: dict[str, list[dict]], diagnostics: dict[str, dict]
) -> dict:
    search_summary = diagnostics.get("source_search_execution_summary") or {}
    ledger = [
        row
        for row in search_summary.get("source_recall_target_ledger") or []
        if isinstance(row, dict)
    ]
    status_counts: Counter[str] = Counter(
        str(row.get("event_page_status") or "status_unknown") for row in ledger
    )

    exact_page_sources = []
    for row in rows.get("source_inventory") or []:
        if not _is_high_confidence_source(row):
            continue
        if str(row.get("source_exact_page_status") or "") == "event_page_found":
            exact_page_sources.append(row)

    def stage_count(stage: str) -> int:
        count = 0
        for row in exact_page_sources:
            source_status = str(row.get("source_to_evidence_status") or "").lower()
            fetch_status = str(row.get("fetch_status") or "").lower()
            parse_status = str(row.get("parse_status") or "").lower()
            target_records = _safe_int(row.get("target_record_count")) or 0
            if stage == "fetched" and (
                fetch_status in {"success", "fetched", "ok"}
                or source_status.startswith(("fetched", "parsed", "evidence"))
            ):
                count += 1
            elif stage == "parsed" and (
                parse_status in {"success", "parsed", "ok", "parsed_html", "parsed_pdf"}
                or source_status.startswith(("parsed", "evidence"))
            ):
                count += 1
            elif stage == "evidence" and (
                target_records > 0 or source_status == "evidence_extracted"
            ):
                count += 1
        return count

    return {
        "recall_target_count": len(ledger),
        "event_page_found_count": status_counts.get("event_page_found", 0),
        "domain_found_event_page_missing_count": status_counts.get(
            "domain_found_event_page_missing", 0
        ),
        "query_not_generated_count": status_counts.get("query_not_generated", 0),
        "query_generated_not_executed_count": status_counts.get(
            "query_generated_not_executed", 0
        ),
        "query_executed_no_result_count": status_counts.get(
            "query_executed_no_result", 0
        ),
        "query_provider_error_count": status_counts.get("query_provider_error", 0),
        "exact_page_source_records": len(exact_page_sources),
        "exact_page_sources_fetched": stage_count("fetched"),
        "exact_page_sources_parsed": stage_count("parsed"),
        "exact_page_sources_with_evidence": stage_count("evidence"),
        "status_counts": dict(status_counts.most_common()),
    }


def _llm_empty_output_recovery(diagnostics: dict[str, dict]) -> dict:
    extraction = diagnostics.get("structured_extraction_summary") or {}
    budget = extraction.get("extraction_budget_ledger") or {}
    calls = _safe_int(extraction.get("llm_call_count")) or 0
    actual_calls = _safe_int(budget.get("total_call_count")) or calls
    primary_calls = _safe_int(extraction.get("primary_llm_call_count")) or 0
    successes = _safe_int(extraction.get("llm_success_count")) or 0
    transport_successes = (
        _safe_int(extraction.get("llm_transport_success_count")) or successes
    )
    empty = _safe_int(extraction.get("llm_empty_output_count")) or 0
    non_empty = _safe_int(extraction.get("llm_non_empty_output_count"))
    if non_empty is None:
        non_empty = max(0, transport_successes - empty)
    valid_records = _safe_int(extraction.get("valid_record_count")) or 0
    unique_valid_records = (
        _safe_int(extraction.get("unique_valid_record_count")) or valid_records
    )
    errors = _safe_int(extraction.get("llm_error_count")) or 0
    deterministic = (
        _safe_int(extraction.get("deterministic_recovery_succeeded_count")) or 0
    )
    focused_calls = _safe_int(extraction.get("focused_recovery_call_count")) or 0
    focused_succeeded = (
        _safe_int(extraction.get("focused_retry_succeeded_count")) or 0
    )
    focused_empty = _safe_int(extraction.get("focused_retry_empty_count")) or 0
    focused_valid = (
        _safe_int(extraction.get("focused_recovery_valid_record_count")) or 0
    )
    focused_field_gain = (
        _safe_int(extraction.get("focused_recovery_field_gain_count")) or 0
    )
    statuses = Counter(
        str(status)
        for status in (extraction.get("focused_recovery_status_by_source") or {}).values()
        if status
    )
    return {
        "actual_model_call_count": actual_calls,
        "primary_model_call_count": primary_calls,
        "llm_call_count": calls,
        "llm_success_count": successes,
        "transport_success_call_count": transport_successes,
        "non_empty_call_count": non_empty,
        "llm_empty_output_count": empty,
        "llm_empty_output_rate": round(empty / calls, 3) if calls else NOT_AVAILABLE,
        "llm_error_count": errors,
        "valid_record_count": valid_records,
        "unique_valid_record_count": unique_valid_records,
        "empty_output_reason_counts": extraction.get("llm_empty_output_reasons") or {},
        "deterministic_recovery_succeeded_count": deterministic,
        "focused_recovery_call_count": focused_calls,
        "focused_retry_succeeded_count": focused_succeeded,
        "focused_retry_empty_count": focused_empty,
        "focused_recovery_valid_record_count": focused_valid,
        "focused_recovery_field_gain_count": focused_field_gain,
        "focused_recovery_valid_record_yield_per_call": (
            round(focused_valid / focused_calls, 3)
            if focused_calls
            else NOT_AVAILABLE
        ),
        "recovered_output_count": (
            focused_valid if focused_valid else deterministic + focused_succeeded
        ),
        "soft_primary_call_limit": _safe_int(budget.get("soft_primary_calls")),
        "hard_primary_call_limit": _safe_int(budget.get("hard_primary_calls")),
        "soft_cap_extension_call_count": (
            _safe_int(budget.get("soft_cap_extension_call_count")) or 0
        ),
        "metric_rows_deferred_for_case_coverage": (
            _safe_int(
                extraction.get(
                    "metric_row_batch_deferred_for_case_coverage_count"
                )
            )
            or 0
        ),
        "focused_recovery_status_counts": dict(statuses.most_common()),
    }


def _case_span_extraction_coverage(rows: dict[str, list[dict]]) -> dict:
    candidate_rows = (
        rows.get("workflow_case_candidate_line_list")
        or rows.get("case_candidate_dataset")
        or rows.get("reviewable_case_dataset")
        or []
    )
    rows_with_span = 0
    rows_with_provenance = 0
    unique_span_ids: set[str] = set()
    unsupported_fields: Counter[str] = Counter()
    quote_labels: dict[str, set[str]] = {}

    for row in candidate_rows:
        span_id = str(row.get("case_span_id") or "").strip()
        span_quote = str(row.get("case_span_quote") or "").strip()
        if span_id or span_quote:
            rows_with_span += 1
        if span_id:
            unique_span_ids.add(span_id)

        provenance = row.get("field_provenance_json")
        if isinstance(provenance, str) and provenance.strip():
            try:
                provenance = json.loads(provenance)
            except json.JSONDecodeError:
                provenance = None
        if isinstance(provenance, dict) and provenance:
            rows_with_provenance += 1

        unsupported = row.get("unsupported_case_fields") or []
        if isinstance(unsupported, str):
            unsupported = [
                item.strip()
                for item in unsupported.replace(",", ";").split(";")
                if item.strip()
            ]
        for field in unsupported if isinstance(unsupported, list) else []:
            unsupported_fields[str(field)] += 1

        label = str(
            row.get("case_label_normalized")
            or row.get("workflow_case_label")
            or ""
        ).strip()
        normalized_quote = " ".join(span_quote.lower().split())
        if label and normalized_quote:
            quote_labels.setdefault(normalized_quote, set()).add(label)

    cross_label_quote_count = sum(
        1 for labels in quote_labels.values() if len(labels) > 1
    )
    return {
        "candidate_rows": len(candidate_rows),
        "rows_with_case_span_evidence": rows_with_span,
        "unique_case_span_count": len(unique_span_ids),
        "rows_with_field_provenance": rows_with_provenance,
        "unsupported_case_field_count": sum(unsupported_fields.values()),
        "unsupported_case_fields_top": dict(unsupported_fields.most_common(10)),
        "cross_label_duplicated_case_quote_count": cross_label_quote_count,
    }


def build_report_facts(session_dir: Path | str) -> dict:
    """Build deterministic facts for final_report.md."""

    artifacts = _load_artifacts(session_dir)
    session = artifacts["session_dir"]
    run_summary = artifacts["run_summary"]
    rows = {
        name: _artifact_rows(artifacts, name)
        for name in (
            "search_query_inventory",
            "source_candidates",
            "source_registry",
            "documents",
            "evidence_chunks",
            "raw_records",
            "validated_records",
            "normalized_records",
            "final_dataset_pre_quality_gate",
            "primary_case_dataset",
            "final_case_dataset",
            "task_aware_observation_dataset",
            "non_primary_observations",
            "reviewable_dataset",
            "final_dataset",
            "reviewable_dataset",
            "pending_review_records",
            "quarantined_records",
            "source_inventory",
            "authority_source_inventory",
            "case_candidate_dataset",
            "reviewable_case_dataset",
            "case_evidence_bundles",
            "workflow_case_bundle_line_list",
            "workflow_case_candidate_line_list",
            "evidence_product_dataset",
            "record_inclusion_decisions",
            "human_review_items",
            "fetch_failures_blocking",
        )
    }
    if not rows["human_review_items"]:
        rows["human_review_items"] = _read_csv(session / "human_review" / "top_review_items.csv")
    diagnostics = {
        name: _artifact_dict(artifacts, name)
        for name in (
            "run_quality_summary",
            "source_identity_summary",
            "source_search_execution_summary",
            "content_fetch_summary",
            "structured_extraction_summary",
            "llm_stage_summary",
            "corroboration_summary",
            "validation_source_compatibility_summary",
        )
    }
    run_quality = diagnostics["run_quality_summary"]
    evaluation_rows = _evaluation_rows(session)
    collection_funnel, missing_metrics = _build_collection_funnel(
        artifacts,
        rows,
        diagnostics,
        evaluation_rows,
    )
    source_by_id = _registry_by_id(rows["source_registry"])
    accepted_primary = (
        rows["primary_case_dataset"]
        or rows["final_case_dataset"]
        or rows["final_dataset"]
    )
    accepted = rows["final_dataset"]
    task_aware_observations = (
        rows["task_aware_observation_dataset"] or rows["non_primary_observations"]
    )
    pending = rows["pending_review_records"]
    quarantined = rows["quarantined_records"]
    reviewable = _reviewable_records(artifacts, rows)
    pre_quality = rows["final_dataset_pre_quality_gate"]
    final_statistics = _final_statistics(accepted_primary, source_by_id)
    validation_readiness = _validation_readiness(
        session,
        evaluation_rows,
        diagnostics,
        run_summary,
    )
    collection_readiness = _artifact_dict(artifacts, "collection_readiness_summary")
    validation_limited = bool(
        run_quality.get("validation_limited")
        or diagnostics["validation_source_compatibility_summary"].get("validation_limited")
        or diagnostics["validation_source_compatibility_summary"].get(
            "no_compatible_validation_source"
        )
    )
    run_quality_status = str(
        _first_present(
            run_quality.get("run_quality_status"),
            run_summary.get("run_quality_status"),
            default="",
        )
    )
    raw_count = collection_funnel["raw_records"]
    pre_quality_count = len(pre_quality)
    accepted_count = len(accepted)
    accepted_primary_count = len(accepted_primary)
    task_aware_count = len(task_aware_observations)
    pending_count = len(pending)
    quarantined_count = len(quarantined)
    reviewable_count = len(reviewable)
    real_collection_status = _run_status(
        run_quality_status=run_quality_status,
        accepted_count=accepted_count,
        task_aware_count=task_aware_count,
        pre_quality_count=pre_quality_count,
        raw_count=raw_count,
        pending_count=pending_count,
        validation_limited=validation_limited,
        live_search=bool(run_summary.get("live_search_enabled")),
        live_fetch=bool(run_summary.get("live_fetch_enabled")),
    )
    source_summary = _source_summary(rows, diagnostics, accepted, run_summary)
    provenance = _provenance_completeness(accepted_primary, source_by_id)
    reviewable_evidence = _reviewable_evidence_rows(rows, reviewable)
    readable_outputs = {
        "reviewable_evidence_matrix": reviewable_evidence[:30],
        "reviewable_evidence_shown_count": min(len(reviewable_evidence), 30),
        "reviewable_evidence_total_count": reviewable_count,
        "case_evidence_bundles": rows["case_evidence_bundles"][:30],
        "source_registry_profile": _source_registry_profile(rows),
        "fetch_manifest": _fetch_manifest_rows(rows, diagnostics),
        "record_provenance": _record_provenance_rows(rows, source_by_id),
        "failure_funnel": _failure_funnel(rows, diagnostics, collection_funnel),
        "source_to_evidence_funnel": _source_to_evidence_funnel(rows),
        "high_confidence_sources_no_record": _high_confidence_sources_with_no_records(
            rows
        ),
        "disease_local_rejection_counts": _disease_local_rejection_counts(rows),
        "case_field_completeness_summary": _case_field_completeness_summary(rows),
        "high_confidence_exact_page_recall": _high_confidence_exact_page_recall(
            rows, diagnostics
        ),
        "llm_empty_output_recovery": _llm_empty_output_recovery(diagnostics),
        "case_span_extraction_coverage": _case_span_extraction_coverage(rows),
    }
    final_epi_yes = (
        accepted_primary_count > 0
        and pending_count == 0
        and not validation_limited
        and not final_statistics["provenance_warnings"]
        and source_summary["accepted_publisher_unknown_record_count"] == 0
    )
    final_epi = _yes_no(final_epi_yes)
    evaluation_count = validation_readiness["evaluation_rows_count"]
    if accepted_count > 0 and isinstance(evaluation_count, int) and evaluation_count > 0:
        case_study = "partial" if validation_limited or pending_count else "yes"
    elif accepted_count > 0:
        case_study = "partial"
    else:
        case_study = "no"
    disease = _first_present(
        run_quality.get("task_disease"),
        run_summary.get("task_disease"),
        default="unknown",
    )
    location = _first_present(
        run_quality.get("task_location"),
        run_summary.get("task_location"),
        default="unknown",
    )
    start_date = _first_present(
        run_quality.get("task_start_date"),
        run_summary.get("task_start_date"),
        default="unknown",
    )
    end_date = _first_present(
        run_quality.get("task_end_date"),
        run_summary.get("task_end_date"),
        default="unknown",
    )
    executive_summary = {
        "disease": disease,
        "location": location,
        "time_range": f"{start_date} to {end_date}",
        "run_status": run_quality_status or "unknown",
        "real_collection_status": real_collection_status,
        "accepted_records_count": accepted_count,
        "accepted_primary_case_records_count": accepted_primary_count,
        "final_case_records_count": accepted_primary_count,
        "task_aware_observation_records_count": task_aware_count,
        "candidate_or_pre_quality_records_count": pre_quality_count,
        "pending_review_records_count": pending_count,
        "reviewable_records_count": reviewable_count,
        "quarantined_records_count": quarantined_count,
        "validation_rows_count": evaluation_count,
        "suitable_for_case_study_comparison": case_study,
        "suitable_as_final_epidemiological_dataset": final_epi,
        "one_sentence_main_limitation": _main_limitation(
            real_status=real_collection_status,
            pending_count=pending_count,
            quarantined_count=quarantined_count,
            validation_rows_count=evaluation_count,
            final_epi=final_epi,
        ),
    }
    task_scope = {
        "original_user_request": run_summary.get("user_request") or NOT_AVAILABLE,
        "interpreted_disease_pathogen": disease,
        "interpreted_geography": location,
        "interpreted_time_range": f"{start_date} to {end_date}",
        "target_fields": _first_present(run_summary.get("target_fields")),
        "scope_expansion_restriction": _first_present(
            run_quality.get("scope_expansion_restriction"),
            run_summary.get("scope_expansion_restriction"),
        ),
        "hps_hfrs_hantavirus_aliases_used": _first_present(
            run_summary.get("hantavirus_aliases_used"),
            run_quality.get("hantavirus_aliases_used"),
        ),
        "county_state_national_aggregation_allowed": _first_present(
            run_summary.get("aggregation_allowed"),
            run_quality.get("aggregation_allowed"),
        ),
    }
    quality_and_trustworthiness = {
        "accepted_count": accepted_count,
        "accepted_primary_case_records_count": accepted_primary_count,
        "final_case_records_count": accepted_primary_count,
        "accepted_task_aware_observations_count": task_aware_count,
        "pending_review_count": pending_count,
        "quarantined_count": quarantined_count,
        "main_quarantine_reasons": _reason_counts(
            quarantined,
            (run_quality.get("collection_decision_summary") or {}).get(
                "quarantine_reason_counts"
            ),
        ),
        "main_pending_review_reasons": _reason_counts(pending),
        "source_corroboration_status": _corroboration_status(accepted, diagnostics),
        "provenance_completeness": provenance,
    }
    facts = {
        "report_metadata": {
            "session_id": run_summary.get("session_id") or session.name,
            "generated_from_artifacts_only": True,
            "llm_called_for_report": False,
            "search_called_for_report": False,
            "fetch_called_for_report": False,
        },
        "executive_summary": executive_summary,
        "task_scope": task_scope,
        "collection_funnel": collection_funnel,
        "data_source_summary": source_summary,
        "final_statistics": final_statistics,
        "quality_and_trustworthiness": quality_and_trustworthiness,
        "validation_readiness": validation_readiness,
        "collection_readiness": collection_readiness,
        "readable_outputs": readable_outputs,
        "human_review_tasks": _human_review_tasks(rows),
        "exported_artifacts": _exported_artifacts(session),
        "diagnostics": {
            "missing_report_metrics": sorted(set(missing_metrics)),
            "reporting_warnings": (
                source_summary["warnings"]
                + [
                    warning
                    for record in final_statistics["provenance_warnings"]
                    for warning in record["warnings"]
                ]
            ),
            "legacy_reports_treated_as_diagnostics": [
                name
                for name in (
                    "workflow_run_report.md",
                    "workflow_interpretive_report.md",
                    "workflow_interpretive_report_summary.json",
                )
                if (session / name).exists()
            ],
        },
    }
    return facts
