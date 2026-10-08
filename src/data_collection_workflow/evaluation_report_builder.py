"""Deterministic masked-validation evaluation report utilities.

This module is intentionally offline-only: it reads local records, compares
collection output with held-out validation records, and writes CSV/JSON/Markdown
artifacts. It does not call network clients, LLMs, or dotenv loaders.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import HumanReviewItem
from .validation_source_compatibility import (
    DISABLED_STATUS,
    EMPTY_STATUS,
    INCOMPATIBLE_DISABLED_STATUS,
    MISSING_STATUS,
    NO_COMPATIBLE_STATUS,
)


EVALUATION_FIELDNAMES = [
    "evaluation_row_id",
    "linked_event_id",
    "disease",
    "virus_or_syndrome",
    "location",
    "country",
    "subnational_location",
    "date_start",
    "date_end",
    "date_anchor",
    "date_anchor_field",
    "reporting_period",
    "as_of_date",
    "statistical_count_type",
    "collection_case_count",
    "collection_death_count",
    "collection_source_ids",
    "collection_source_urls",
    "collection_evidence_quotes",
    "validation_source_ids",
    "validation_source_urls",
    "validation_case_count",
    "validation_death_count",
    "validation_evidence_quotes",
    "benchmark_case_count",
    "benchmark_death_count",
    "benchmark_source_urls",
    "benchmark_evidence_quotes",
    "benchmark_overall_match_status",
    "case_count_difference",
    "death_count_difference",
    "field_level_match_status",
    "overall_match_status",
    "masking_compliance_status",
    "provenance_completeness_status",
    "human_review_flag",
    "review_reason",
]

_CASE_FIELDS = [
    "cases_confirmed",
    "cases_unspecified",
    "cases_probable",
    "cases_suspected",
]
_PROVENANCE_FIELDS = [
    "source_url",
    "evidence_quote",
    "supporting_chunk_id",
    "linked_event_id",
]
_MISSING_TOKENS = {"", "none", "null", "nan", "n/a", "na", "unspecified"}


def read_csv_records(path: Path | str) -> list[dict]:
    """Read a CSV file into a list of dictionaries."""

    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv_records(
    path: Path | str,
    records: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    """Write dictionaries to CSV with stable field ordering."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ordered = list(fieldnames or [])
    extras: set[str] = set()
    for record in records:
        for key in record.keys():
            if key not in ordered:
                extras.add(key)
    if not ordered:
        ordered = sorted(extras)
        extras = set()
    all_fieldnames = ordered + sorted(extras)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in all_fieldnames})


def read_json_file(path: Path | str):
    """Read a local JSON file."""

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_collection_artifacts(export_dir: Path | str) -> dict:
    """Read collection artifacts from an exported final package directory."""

    p = Path(export_dir)
    final_package_path = p / "final_package.json"
    if final_package_path.exists():
        package = read_json_file(final_package_path)
        return {
            "final_package": package,
            "final_dataset": list(package.get("final_dataset") or []),
            "source_registry": list(package.get("source_registry") or []),
            "conflicts": list(package.get("conflicts") or []),
            "human_review_items": list(package.get("human_review_items") or []),
        }
    return {
        "final_package": None,
        "final_dataset": read_csv_records(p / "final_dataset.csv"),
        "source_registry": read_json_file(p / "source_registry.json"),
        "conflicts": read_json_file(p / "conflicts.json"),
        "human_review_items": read_json_file(p / "human_review_items.json"),
    }


def build_evaluation_report(
    collection_records: list[dict],
    validation_records: list[dict],
    collection_source_registry: list[dict] | None = None,
    reserved_source_ids: set[str] | None = None,
    conflicts: list[dict] | None = None,
    human_review_items: list[dict] | None = None,
    validation_source_compatibility_summary: dict | None = None,
) -> tuple[list[dict], dict]:
    """Build row-level masked-validation evaluation rows and a summary."""

    compatibility_status = None
    if validation_source_compatibility_summary:
        compatibility_status = validation_source_compatibility_summary.get(
            "compatibility_status"
        )
    no_task_compatible_validation = (
        not validation_records
        and compatibility_status
        in {
            NO_COMPATIBLE_STATUS,
            INCOMPATIBLE_DISABLED_STATUS,
            MISSING_STATUS,
            EMPTY_STATUS,
            DISABLED_STATUS,
        }
    )

    if no_task_compatible_validation:
        rows: list[dict] = []
    else:
        collection_groups = _group_records(collection_records)
        validation_groups = _group_records(validation_records)
        all_keys = sorted(set(collection_groups) | set(validation_groups))

        rows = []
        for index, key in enumerate(all_keys, start=1):
            collection_group = collection_groups.get(key, [])
            validation_group = validation_groups.get(key, [])
            row = _build_evaluation_row(
                index=index,
                key=key,
                collection_group=collection_group,
                validation_group=validation_group,
                reserved_source_ids=reserved_source_ids,
            )
            rows.append(row)

    reserved_ids = sorted(reserved_source_ids) if reserved_source_ids else []
    leaked_ids = sorted(
        {
            str(record.get("source_id"))
            for record in collection_records
            if reserved_source_ids
            and record.get("source_id")
            and record.get("source_id") in reserved_source_ids
        }
    )
    rows_with_collection_evidence_count = sum(
        1 for row in rows if _has_value(row.get("collection_evidence_quotes"))
    )
    rows_with_validation_evidence_count = sum(
        1 for row in rows if _has_value(row.get("validation_evidence_quotes"))
    )
    rows_with_both_evidence_count = sum(
        1
        for row in rows
        if _has_value(row.get("collection_evidence_quotes"))
        and _has_value(row.get("validation_evidence_quotes"))
    )
    summary = {
        "evaluation_method": "deterministic_masked_validation_v0",
        "collection_record_count": len(collection_records),
        "validation_record_count": len(validation_records),
        "evaluation_row_count": len(rows),
        "collection_source_registry_count": len(collection_source_registry or []),
        "conflict_count": len(conflicts or []),
        "human_review_item_count": len(human_review_items or []),
        "human_review_flagged_row_count": sum(
            1 for row in rows if _truthy(row.get("human_review_flag"))
        ),
        "rows_with_collection_evidence_count": rows_with_collection_evidence_count,
        "rows_with_validation_evidence_count": rows_with_validation_evidence_count,
        "rows_with_both_collection_and_validation_evidence_count": (
            rows_with_both_evidence_count
        ),
        "reserved_source_ids": reserved_ids,
        "reserved_source_id_count": len(reserved_ids),
        "reserved_source_leakage_count": len(leaked_ids),
        "reserved_source_leakage_source_ids": leaked_ids,
        "overall_match_status_counts": dict(
            Counter(row.get("overall_match_status") for row in rows)
        ),
        "masking_compliance_status_counts": dict(
            Counter(row.get("masking_compliance_status") for row in rows)
        ),
        "provenance_completeness_status_counts": dict(
            Counter(row.get("provenance_completeness_status") for row in rows)
        ),
        "workflow_limitations": [
            "This comparison uses only the supplied collection and held-out reference records; it does not retrieve additional evidence.",
            "Agreement with a reference record does not establish complete coverage of the requested scope.",
            "Missing collection records can occur when held-out validation sources contain the only extractable data.",
        ],
    }
    if validation_source_compatibility_summary:
        summary.update(
            {
                "validation_source_compatibility_status": compatibility_status,
                "active_validation_record_count": (
                    validation_source_compatibility_summary.get(
                        "active_validation_record_count", 0
                    )
                ),
                "inactive_validation_record_count": (
                    validation_source_compatibility_summary.get(
                        "inactive_validation_record_count", 0
                    )
                ),
                "raw_validation_record_count": (
                    validation_source_compatibility_summary.get(
                        "validation_record_count", 0
                    )
                ),
                "validation_source_compatibility_warnings": (
                    validation_source_compatibility_summary.get("warnings") or []
                ),
            }
        )
    if no_task_compatible_validation:
        summary["workflow_limitations"].append(
            "No task-compatible held-out validation source was configured/found."
        )
    return rows, summary


def write_evaluation_outputs(
    evaluation_rows: list[dict],
    evaluation_summary: dict,
    output_dir: Path | str,
) -> dict:
    """Write evaluation CSV, summary JSON, and human-readable Markdown."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = out_dir / "evaluation_report.csv"
    summary_json = out_dir / "evaluation_summary.json"
    markdown_path = out_dir / "readable_evaluation_report.md"

    write_csv_records(report_csv, evaluation_rows, EVALUATION_FIELDNAMES)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, ensure_ascii=False, indent=2)
    markdown_path.write_text(
        _build_readable_markdown(evaluation_rows, evaluation_summary),
        encoding="utf-8",
    )

    return {
        "evaluation_report_csv": str(report_csv),
        "evaluation_summary_json": str(summary_json),
        "readable_evaluation_report_md": str(markdown_path),
    }


def build_evaluation_review_items(
    evaluation_rows: list[dict],
    existing_review_ids: set[str] | None = None,
) -> list[dict]:
    """Convert flagged validation rows into standard human review items."""

    seen = set(existing_review_ids or set())
    items: list[dict] = []
    for row in evaluation_rows:
        if not _truthy(row.get("human_review_flag")):
            continue
        row_id = str(row.get("evaluation_row_id") or "").strip()
        if not row_id:
            continue
        review_id = f"review_validation_{row_id}"
        if review_id in seen:
            continue
        related_ids = [row_id]
        linked_event_id = str(row.get("linked_event_id") or "").strip()
        if linked_event_id:
            related_ids.append(linked_event_id)
        item = HumanReviewItem(
            review_id=review_id,
            item_type="masked_validation",
            related_ids=related_ids,
            reason=row.get("review_reason")
            or "Masked validation row requires human review.",
            status="pending",
            priority=2,
            review_packet={
                "review_id": review_id,
                "item_type": "masked_validation",
                "priority": 2,
                "status": "pending",
                "reason": row.get("review_reason")
                or "Masked validation row requires human review.",
                "related_ids": related_ids,
                "packet_sections": {
                    "evaluation_row": row,
                    "collection_evidence_quotes": row.get(
                        "collection_evidence_quotes"
                    ),
                    "validation_evidence_quotes": row.get(
                        "validation_evidence_quotes"
                    ),
                },
            },
        )
        items.append(item.model_dump())
        seen.add(review_id)
    return items


def _build_evaluation_row(
    index: int,
    key: tuple[str, ...],
    collection_group: list[dict],
    validation_group: list[dict],
    reserved_source_ids: set[str] | None,
) -> dict:
    source_records = collection_group or validation_group
    representative = _first_record(source_records)
    collection_case_values = _numeric_values(collection_group, _case_count)
    validation_case_values = _numeric_values(validation_group, _case_count)
    collection_death_values = _numeric_values(collection_group, lambda r: r.get("deaths"))
    validation_death_values = _numeric_values(validation_group, lambda r: r.get("deaths"))

    case_status, case_diff = _compare_numeric_values(
        collection_case_values, validation_case_values, "case_count"
    )
    death_status, death_diff = _compare_numeric_values(
        collection_death_values, validation_death_values, "death_count"
    )
    field_statuses = [case_status, death_status]
    masking_status = _masking_status(collection_group, reserved_source_ids)
    provenance_status = _provenance_status(collection_group)
    overall_status = _overall_status(
        collection_group,
        validation_group,
        field_statuses,
    )
    human_review = (
        overall_status != "match"
        or masking_status == "failed_validation_source_in_collection"
    )
    review_reason = _review_reason(
        overall_status,
        masking_status,
        field_statuses,
    )

    return {
        "evaluation_row_id": f"eval_{index:03d}",
        "linked_event_id": _first_non_empty(source_records, "linked_event_id"),
        "disease": _display_value(representative, "disease", key[0]),
        "virus_or_syndrome": _display_value(representative, "virus_or_syndrome", key[1]),
        "location": _display_location(representative, key),
        "country": _display_geo(representative, key[2]),
        "subnational_location": _display_value(
            representative, "subnational_location", key[3]
        ),
        "date_start": _first_non_empty(source_records, "event_start_date"),
        "date_end": _first_non_empty(source_records, "event_end_date"),
        "date_anchor": _first_non_empty(
            source_records, "date_anchor", "date_reported", "reporting_period"
        ),
        "date_anchor_field": _first_non_empty(source_records, "date_anchor_field"),
        "reporting_period": _first_non_empty(source_records, "reporting_period"),
        "as_of_date": _first_non_empty(source_records, "as_of_date"),
        "statistical_count_type": _display_value(
            representative, "statistical_count_type", key[6]
        ),
        "collection_case_count": _format_values(collection_case_values),
        "collection_death_count": _format_values(collection_death_values),
        "collection_source_ids": _join_unique(collection_group, "source_id"),
        "collection_source_urls": _join_unique(collection_group, "source_url"),
        "collection_evidence_quotes": _join_unique(collection_group, "evidence_quote"),
        "validation_source_ids": _join_unique(validation_group, "source_id"),
        "validation_source_urls": _join_unique(validation_group, "source_url"),
        "validation_case_count": _format_values(validation_case_values),
        "validation_death_count": _format_values(validation_death_values),
        "validation_evidence_quotes": _join_unique(validation_group, "evidence_quote"),
        "benchmark_case_count": _format_values(validation_case_values),
        "benchmark_death_count": _format_values(validation_death_values),
        "benchmark_source_urls": _join_unique(validation_group, "source_url"),
        "benchmark_evidence_quotes": _join_unique(validation_group, "evidence_quote"),
        "benchmark_overall_match_status": _standard_match_status(overall_status),
        "case_count_difference": _format_number(case_diff),
        "death_count_difference": _format_number(death_diff),
        "field_level_match_status": ";".join(field_statuses),
        "overall_match_status": overall_status,
        "masking_compliance_status": masking_status,
        "provenance_completeness_status": provenance_status,
        "human_review_flag": human_review,
        "review_reason": review_reason,
    }


def _group_records(records: list[dict]) -> dict[tuple[str, ...], list[dict]]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for record in records:
        grouped[_comparison_key(record)].append(record)
    return grouped


def _comparison_key(record: dict) -> tuple[str, ...]:
    geo = record.get("country") or record.get("geographic_scope")
    date_anchor = _comparison_date_anchor(record)
    return (
        _normalize(record.get("disease")),
        _normalize(record.get("virus_or_syndrome")),
        _normalize(geo),
        _normalize(record.get("subnational_location")),
        _normalize(date_anchor),
        _normalize(record.get("reporting_period")),
        _normalize(record.get("statistical_count_type")),
    )


def _comparison_date_anchor(record: dict):
    if (
        _normalize(record.get("statistical_count_type")) == "annual"
        and _has_value(record.get("reporting_period"))
    ):
        return record.get("reporting_period")
    return (
        record.get("date_anchor")
        or record.get("date_reported")
        or record.get("reporting_period")
    )


def _normalize(value) -> str:
    text = str(value or "").strip()
    if text.lower() in _MISSING_TOKENS:
        return "unspecified"
    return " ".join(text.lower().split())


def _display_value(record: dict, field: str, fallback: str) -> str:
    value = record.get(field)
    if _has_value(value):
        return str(value)
    return "" if fallback == "unspecified" else fallback


def _display_geo(record: dict, fallback: str) -> str:
    value = record.get("country") or record.get("geographic_scope")
    if _has_value(value):
        return str(value)
    return "" if fallback == "unspecified" else fallback


def _display_location(record: dict, key: tuple[str, ...]) -> str:
    value = (
        record.get("locality")
        or record.get("subnational_location")
        or record.get("geographic_scope")
        or record.get("country")
    )
    if _has_value(value):
        return str(value)
    for index in (3, 2):
        if len(key) > index and key[index] != "unspecified":
            return key[index]
    return ""


def _standard_match_status(status: str) -> str:
    mapping = {
        "partial_match_not_comparable": "partial_match",
        "insufficient_comparable_values": "not_comparable",
        "missing_collection_record": "insufficient_evidence",
        "missing_validation_record": "insufficient_evidence",
    }
    return mapping.get(status, status)


def _first_record(records: list[dict]) -> dict:
    return records[0] if records else {}


def _first_non_empty(records: list[dict], *fields: str) -> str:
    for record in records:
        for field in fields:
            value = record.get(field)
            if _has_value(value):
                return str(value)
    return ""


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in _MISSING_TOKENS
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _case_count(record: dict):
    for field in _CASE_FIELDS:
        value = record.get(field)
        if _has_value(value):
            return value
    return None


def _numeric_values(records: list[dict], getter) -> list[float]:
    values: set[float] = set()
    for record in records:
        numeric = _to_float(getter(record))
        if numeric is not None:
            values.add(numeric)
    return sorted(values)


def _to_float(value) -> float | None:
    if not _has_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compare_numeric_values(
    collection_values: list[float],
    validation_values: list[float],
    label: str,
) -> tuple[str, float | None]:
    if len(collection_values) == 1 and len(validation_values) == 1:
        diff = collection_values[0] - validation_values[0]
        status = "match" if diff == 0 else "mismatch"
        return f"{label}_{status}", diff
    return f"{label}_not_comparable", None


def _overall_status(
    collection_group: list[dict],
    validation_group: list[dict],
    field_statuses: Iterable[str],
) -> str:
    if validation_group and not collection_group:
        return "missing_collection_record"
    if collection_group and not validation_group:
        return "missing_validation_record"
    statuses = list(field_statuses)
    if any(status.endswith("_mismatch") for status in statuses):
        return "mismatch"
    if all(status.endswith("_match") for status in statuses):
        return "match"
    if any(status.endswith("_match") for status in statuses) and any(
        status.endswith("_not_comparable") for status in statuses
    ):
        return "partial_match_not_comparable"
    if all(status.endswith("_not_comparable") for status in statuses):
        return "insufficient_comparable_values"
    return "partial_match"


def _masking_status(
    collection_group: list[dict],
    reserved_source_ids: set[str] | None,
) -> str:
    if reserved_source_ids is None:
        return "not_checked"
    for record in collection_group:
        if record.get("source_id") in reserved_source_ids:
            return "failed_validation_source_in_collection"
    return "passed"


def _provenance_status(collection_group: list[dict]) -> str:
    if not collection_group:
        return "not_applicable_no_collection_record"
    adequate = [
        all(_has_value(record.get(field)) for field in _PROVENANCE_FIELDS)
        for record in collection_group
    ]
    if all(adequate):
        return "complete"
    if any(adequate):
        return "partial"
    return "missing"


def _review_reason(
    overall_status: str,
    masking_status: str,
    field_statuses: list[str],
) -> str:
    reasons: list[str] = []
    if masking_status == "failed_validation_source_in_collection":
        reasons.append("Collection output contains a validation-reserved source.")
    if overall_status == "missing_collection_record":
        reasons.append("Held-out validation record had no collection counterpart.")
    elif overall_status == "missing_validation_record":
        reasons.append("Collection record could not be validated against held-out records.")
    elif overall_status == "insufficient_comparable_values":
        reasons.append("Neither side has comparable numeric values.")
    elif overall_status == "partial_match_not_comparable":
        reasons.append(
            "Some comparable numeric fields matched, but at least one core "
            "numeric field was not comparable."
        )
    elif overall_status == "mismatch":
        mismatches = [
            status.replace("_mismatch", "")
            for status in field_statuses
            if status.endswith("_mismatch")
        ]
        reasons.append(f"Comparable numeric fields differ: {', '.join(mismatches)}.")
    return " ".join(reasons)


def _join_unique(records: list[dict], field: str) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for record in records:
        value = record.get(field)
        if not _has_value(value):
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            values.append(text)
    return " | ".join(values)


def _format_values(values: list[float]) -> str:
    return " | ".join(_format_number(value) for value in values)


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _build_readable_markdown(rows: list[dict], summary: dict) -> str:
    overall_counts = summary.get("overall_match_status_counts") or {}
    masking_counts = summary.get("masking_compliance_status_counts") or {}
    provenance_counts = summary.get("provenance_completeness_status_counts") or {}
    reserved_ids = summary.get("reserved_source_ids") or []
    review_rows = [row for row in rows if _truthy(row.get("human_review_flag"))]
    compatibility_status = summary.get("validation_source_compatibility_status")
    compatibility_warnings = summary.get("validation_source_compatibility_warnings") or []

    lines = [
        "# Masked Validation Evaluation Report",
        "",
        (
            "This report summarizes the workflow's collection output against "
            "held-out validation evidence. It is an audit-oriented workflow "
            "evaluation report, not a full epidemiological benchmark."
        ),
        "",
        "## Collection Summary",
        "",
        f"- Collection record count: {summary.get('collection_record_count', 0)}",
        f"- Validation record count: {summary.get('validation_record_count', 0)}",
        (
            "- Validation source compatibility status: "
            f"{compatibility_status or 'not_checked'}"
        ),
        (
            "- Active / inactive / raw validation records: "
            f"{summary.get('active_validation_record_count', summary.get('validation_record_count', 0))} / "
            f"{summary.get('inactive_validation_record_count', 0)} / "
            f"{summary.get('raw_validation_record_count', summary.get('validation_record_count', 0))}"
        ),
        f"- Evaluation row count: {summary.get('evaluation_row_count', 0)}",
        (
            "- Rows with collection evidence: "
            f"{summary.get('rows_with_collection_evidence_count', 0)}"
        ),
        (
            "- Rows with validation evidence: "
            f"{summary.get('rows_with_validation_evidence_count', 0)}"
        ),
        (
            "- Rows with both collection and validation evidence: "
            f"{summary.get('rows_with_both_collection_and_validation_evidence_count', 0)}"
        ),
        f"- Masking compliance summary: {_format_counts(masking_counts)}",
        (
            "- Human review flagged row count: "
            f"{summary.get('human_review_flagged_row_count', 0)}"
        ),
        "",
        "## Held-Out Source Policy",
        "",
        f"- Reserved source IDs: {', '.join(reserved_ids) if reserved_ids else 'none'}",
        (
            "- The held-out policy requires reserved sources to stay out of "
            "collection. The recorded masking compliance is shown below."
        ),
        (
            "- Validation compatibility warnings: "
            + ("; ".join(str(item) for item in compatibility_warnings) if compatibility_warnings else "none")
        ),
        "",
        "## Evaluation Status Counts",
        "",
        f"- Overall match status counts: {_format_counts(overall_counts)}",
        f"- Masking compliance status counts: {_format_counts(masking_counts)}",
        f"- Provenance completeness status counts: {_format_counts(provenance_counts)}",
        "",
        "## Evaluation Row Preview",
        "",
        *_row_preview_lines(rows),
        "",
        "## Human Review Rows",
        "",
    ]
    if not review_rows:
        lines.append("- None")
    else:
        for row in review_rows:
            location = row.get("country") or row.get("subnational_location") or "unknown"
            date = row.get("date_anchor") or row.get("reporting_period") or "unknown"
            lines.append(
                "- "
                f"{row.get('evaluation_row_id')}: {location} / {date}; "
                f"status={row.get('overall_match_status')}; "
                f"reason={row.get('review_reason')}"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            *(f"- {item}" for item in summary.get("workflow_limitations") or []),
            "",
        ]
    )
    return "\n".join(lines)


def _row_preview_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["- None"]

    lines: list[str] = []
    for row in rows[:10]:
        location = _display_cell(
            row.get("country") or row.get("subnational_location") or "unknown"
        )
        date = _display_cell(
            row.get("date_anchor") or row.get("reporting_period") or "unknown"
        )
        parts = [
            f"{row.get('evaluation_row_id')}: location/date={location} / {date}",
            f"collection_case_count={_display_cell(row.get('collection_case_count'))}",
            f"collection_death_count={_display_cell(row.get('collection_death_count'))}",
            f"collection_source_ids={_display_cell(row.get('collection_source_ids'))}",
            (
                "collection_evidence_quote_preview="
                f"\"{_preview_text(row.get('collection_evidence_quotes'))}\""
            ),
            f"validation_case_count={_display_cell(row.get('validation_case_count'))}",
            f"validation_death_count={_display_cell(row.get('validation_death_count'))}",
            f"validation_source_ids={_display_cell(row.get('validation_source_ids'))}",
            (
                "validation_evidence_quote_preview="
                f"\"{_preview_text(row.get('validation_evidence_quotes'))}\""
            ),
            f"overall_match_status={_display_cell(row.get('overall_match_status'))}",
            f"human_review_flag={_format_bool(row.get('human_review_flag'))}",
        ]
        if _has_value(row.get("review_reason")):
            parts.append(f"review_reason={_preview_text(row.get('review_reason'), 180)}")
        lines.append("- " + "; ".join(parts))

    if len(rows) > 10:
        lines.append(f"- ... {len(rows) - 10} additional rows omitted")
    return lines


def _display_cell(value) -> str:
    return str(value) if _has_value(value) else "none"


def _preview_text(value, max_chars: int = 160) -> str:
    if not _has_value(value):
        return "none"
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_bool(value) -> str:
    return "true" if _truthy(value) else "false"


def _format_counts(counts: dict) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
