"""Task-compatible validation source resolution.

This module is deterministic and offline-only. It decides whether held-out
validation records are compatible with the active task before those records are
used as trusted validation evidence.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

from collections import Counter
from pathlib import Path
import re

from .disease_relevance import (
    COMPATIBLE,
    build_disease_relevance_context,
    assess_record_disease_compatibility,
)

COMPATIBLE_STATUS = "compatible"
PARTIALLY_COMPATIBLE_STATUS = "partially_compatible"
NO_COMPATIBLE_STATUS = "no_task_compatible_validation_source"
INCOMPATIBLE_DISABLED_STATUS = "incompatible_validation_source_disabled"
MISSING_STATUS = "validation_source_missing"
EMPTY_STATUS = "validation_source_empty"
INSUFFICIENT_METADATA_STATUS = "insufficient_validation_metadata"
EXPLICIT_OVERRIDE_STATUS = "explicit_validation_source_loaded_with_warning"
DISABLED_STATUS = "validation_disabled_by_config"
LIVE_VALIDATION_PENDING_STATUS = "live_validation_pending"

_DATE_FIELDS = (
    "date_reported",
    "event_start_date",
    "event_end_date",
    "reporting_period",
    "as_of_date",
    "date_anchor",
)
_LOCATION_FIELDS = (
    "locality",
    "subnational_location",
    "geographic_scope",
    "country",
)
_DISEASE_FIELDS = (
    "disease",
    "disease_standard_name",
    "virus_or_syndrome",
    "pathogen_or_syndrome",
)


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _norm(value) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = _norm(clean)
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def _extract_years(value) -> list[int]:
    return [
        int(match.group(0))
        for match in re.finditer(r"\b(?:19|20)\d{2}\b", str(value or ""))
    ]


def _task_context(state_or_task_context: dict | None) -> dict:
    state_or_task_context = state_or_task_context or {}
    structured = state_or_task_context.get("structured_task") or state_or_task_context
    spec = state_or_task_context.get("collection_spec") or {}
    disease_intelligence = state_or_task_context.get("disease_intelligence") or {}
    disease = structured.get("disease") or spec.get("disease")
    location = structured.get("location") or spec.get("geography") or spec.get("location")
    start_date = structured.get("start_date") or spec.get("start_date")
    end_date = structured.get("end_date") or spec.get("end_date")
    years: list[int] = []
    for value in (start_date, end_date, spec.get("time_window")):
        years.extend(_extract_years(value))
    return {
        "structured_task": structured if isinstance(structured, dict) else {},
        "collection_spec": spec if isinstance(spec, dict) else {},
        "disease_intelligence": disease_intelligence
        if isinstance(disease_intelligence, dict)
        else {},
        "task_disease": str(disease or ""),
        "task_location": str(location or ""),
        "task_start_year": min(years) if years else None,
        "task_end_year": max(years) if years else None,
    }


def _record_years(record: dict) -> list[int]:
    years: list[int] = []
    for field in _DATE_FIELDS:
        years.extend(_extract_years(record.get(field)))
    return sorted(set(years))


def _record_locations(record: dict) -> list[str]:
    return _unique([str(record.get(field) or "") for field in _LOCATION_FIELDS])


def _record_diseases(record: dict) -> list[str]:
    return _unique([str(record.get(field) or "") for field in _DISEASE_FIELDS])


def _location_compatible(record: dict, task_location: str) -> tuple[bool | None, str]:
    task = _norm(task_location)
    if not task or task in {"global", "worldwide", "all"}:
        return True, "task location is broad or unspecified"
    values = [_norm(value) for value in _record_locations(record)]
    values = [value for value in values if value]
    if not values:
        return None, "insufficient geography metadata"
    if any(task == value or task in value or value in task for value in values):
        return True, "geography overlaps active task"
    return False, "geography mismatch"


def _time_compatible(record: dict, task_start: int | None, task_end: int | None) -> tuple[bool | None, str]:
    years = _record_years(record)
    if task_start is None and task_end is None:
        return True, "task time window is unspecified"
    if not years:
        return None, "insufficient time metadata"
    start = min(years)
    end = max(years)
    if task_start is not None and end < task_start:
        return False, "time window does not overlap"
    if task_end is not None and start > task_end:
        return False, "time window does not overlap"
    return True, "time window overlaps active task"


def _disease_compatible(record: dict, context: dict) -> tuple[bool | None, str, dict]:
    assessment = assess_record_disease_compatibility(
        record,
        build_disease_relevance_context(context),
    )
    status = assessment.get("status")
    if status == COMPATIBLE:
        return True, "disease is compatible with active task", assessment
    if status in {"insufficient_text", "ambiguous_disease"}:
        return None, assessment.get("reason") or "insufficient disease metadata", assessment
    return False, assessment.get("reason") or "disease mismatch", assessment


def assess_validation_record_compatibility(record: dict, context: dict) -> dict:
    task = _task_context(context)
    disease_ok, disease_reason, disease_assessment = _disease_compatible(
        record, context
    )
    location_ok, location_reason = _location_compatible(
        record, task["task_location"]
    )
    time_ok, time_reason = _time_compatible(
        record,
        task["task_start_year"],
        task["task_end_year"],
    )
    reasons: list[str] = []
    warnings: list[str] = []
    for label, ok, reason in (
        ("disease", disease_ok, disease_reason),
        ("geography", location_ok, location_reason),
        ("time", time_ok, time_reason),
    ):
        if ok is False:
            reasons.append(f"{label}: {reason}")
            warnings.append(f"{label}_mismatch: {reason}")
        elif ok is None:
            reasons.append(f"{label}: {reason}")
            warnings.append(f"insufficient_{label}_metadata: {reason}")

    is_compatible = disease_ok is True and location_ok is True and time_ok is True
    insufficient = disease_ok is None or location_ok is None or time_ok is None
    status = (
        "compatible"
        if is_compatible
        else ("insufficient_validation_metadata" if insufficient and not reasons else "incompatible")
    )
    return {
        "record_id": record.get("record_id"),
        "source_id": record.get("source_id"),
        "is_compatible": is_compatible,
        "compatibility_status": status,
        "compatibility_reason": "; ".join(reasons)
        if reasons
        else "validation record is task-compatible",
        "warnings": warnings,
        "disease_compatible": disease_ok,
        "geography_compatible": location_ok,
        "time_compatible": time_ok,
        "disease_assessment": disease_assessment,
    }


def _summary_status(
    *,
    active_count: int,
    inactive_count: int,
    total_count: int,
    explicit: bool,
    allow_incompatible: bool,
    incompatible_count: int = 0,
    missing_path: bool,
) -> str:
    if missing_path:
        return MISSING_STATUS
    if total_count == 0:
        return EMPTY_STATUS
    if allow_incompatible and incompatible_count:
        return EXPLICIT_OVERRIDE_STATUS
    if active_count == total_count:
        return COMPATIBLE_STATUS
    if active_count > 0:
        return PARTIALLY_COMPATIBLE_STATUS
    if explicit:
        return INCOMPATIBLE_DISABLED_STATUS
    return NO_COMPATIBLE_STATUS


def resolve_task_compatible_validation_records(
    *,
    validation_records: list[dict] | None = None,
    state_or_task_context: dict | None = None,
    validation_records_path: str | Path | None = None,
    validation_records_path_requested: str | Path | None = None,
    validation_records_explicit: bool = False,
    validation_records_defaulted: bool | None = None,
    allow_incompatible_validation_records: bool | None = None,
    validation_mode: str | None = None,
) -> dict:
    """Split validation records into active and inactive task-compatible sets."""

    records = list(validation_records or [])
    context = state_or_task_context or {}
    task = _task_context(context)
    allow_incompatible = (
        _truthy(get_env("ALLOW_INCOMPATIBLE_VALIDATION_RECORDS"))
        if allow_incompatible_validation_records is None
        else bool(allow_incompatible_validation_records)
    )
    path = Path(validation_records_path) if validation_records_path else None
    requested_path = (
        Path(validation_records_path_requested)
        if validation_records_path_requested
        else path
    )
    mode = str(validation_mode or get_env("VALIDATION_MODE") or "").strip()
    if not mode:
        mode = "held_out_file"
    if mode == "live_cross_source" and not path and not records:
        summary = {
            "validation_mode": "live_cross_source",
            "validation_records_path": None,
            "validation_records_path_requested": str(requested_path)
            if requested_path
            else None,
            "validation_records_path_used": None,
            "validation_records_source": "none",
            "validation_records_explicit": bool(validation_records_explicit),
            "validation_records_defaulted": False,
            "validation_records_loaded": False,
            "validation_records_enabled": False,
            "validation_record_count": 0,
            "validation_compatible_record_count": 0,
            "validation_incompatible_record_count": 0,
            "active_validation_record_count": 0,
            "inactive_validation_record_count": 0,
            "compatibility_status": LIVE_VALIDATION_PENDING_STATUS,
            "compatibility_reason": (
                "Live cross-source validation will be derived from fetched "
                "task-compatible validation sources."
            ),
            "task_disease": task["task_disease"],
            "task_location": task["task_location"],
            "task_start_date": (context.get("structured_task") or {}).get(
                "start_date"
            ),
            "task_end_date": (context.get("structured_task") or {}).get("end_date"),
            "validation_diseases_seen": [],
            "validation_locations_seen": [],
            "validation_time_windows_seen": [],
            "warnings": [],
            "record_status_counts": {},
            "allow_incompatible_validation_records": False,
        }
        return {
            "validation_records": [],
            "active_validation_records": [],
            "inactive_validation_records": [],
            "validation_record_compatibility": [],
            "validation_source_compatibility_summary": summary,
        }
    path_missing = bool(path and not path.exists())

    assessments = [
        assess_validation_record_compatibility(record, context) for record in records
    ]
    active: list[dict] = []
    inactive: list[dict] = []
    warnings: list[str] = []
    for record, assessment in zip(records, assessments):
        annotated = {
            **record,
            "validation_compatibility_status": assessment[
                "compatibility_status"
            ],
            "validation_compatibility_reason": assessment[
                "compatibility_reason"
            ],
            "validation_compatibility_warnings": assessment["warnings"],
        }
        if assessment["is_compatible"] or (allow_incompatible and validation_records_explicit):
            active.append(annotated)
        else:
            inactive.append(annotated)
        warnings.extend(assessment["warnings"])

    status = _summary_status(
        active_count=len(active),
        inactive_count=len(inactive),
        total_count=len(records),
        explicit=validation_records_explicit,
        allow_incompatible=allow_incompatible and validation_records_explicit,
        incompatible_count=sum(1 for item in assessments if not item["is_compatible"]),
        missing_path=path_missing,
    )
    if status == EXPLICIT_OVERRIDE_STATUS:
        warnings.append("override enabled: incompatible validation records loaded as active")
    elif status in {NO_COMPATIBLE_STATUS, INCOMPATIBLE_DISABLED_STATUS}:
        warnings.append("no task-compatible held-out validation source is active")

    disease_values: list[str] = []
    location_values: list[str] = []
    time_values: list[str] = []
    for record in records:
        disease_values.extend(_record_diseases(record))
        location_values.extend(_record_locations(record))
        for field in _DATE_FIELDS:
            if record.get(field):
                time_values.append(str(record.get(field)))

    summary = {
        "validation_mode": mode,
        "validation_records_path": str(path) if path else None,
        "validation_records_path_requested": str(requested_path)
        if requested_path
        else None,
        "validation_records_path_used": str(path)
        if path and len(active) > 0
        else None,
        "validation_records_source": "csv_path" if path else "state",
        "validation_records_explicit": bool(validation_records_explicit),
        "validation_records_defaulted": (
            not validation_records_explicit
            if validation_records_defaulted is None
            else bool(validation_records_defaulted)
        ),
        "validation_records_loaded": bool(records),
        "validation_records_enabled": bool(active),
        "validation_record_count": len(records),
        "validation_compatible_record_count": len(active),
        "validation_incompatible_record_count": len(inactive),
        "active_validation_record_count": len(active),
        "inactive_validation_record_count": len(inactive),
        "compatibility_status": status,
        "compatibility_reason": (
            "Task-compatible validation records are active."
            if active
            else "No task-compatible held-out validation source was configured/found."
        ),
        "task_disease": task["task_disease"],
        "task_location": task["task_location"],
        "task_start_date": (context.get("structured_task") or {}).get("start_date"),
        "task_end_date": (context.get("structured_task") or {}).get("end_date"),
        "validation_diseases_seen": _unique(disease_values),
        "validation_locations_seen": _unique(location_values),
        "validation_time_windows_seen": _unique(time_values),
        "warnings": _unique(warnings),
        "record_status_counts": dict(
            Counter(item["compatibility_status"] for item in assessments)
        ),
        "allow_incompatible_validation_records": bool(
            allow_incompatible and validation_records_explicit
        ),
    }
    return {
        "validation_records": list(records),
        "active_validation_records": active,
        "inactive_validation_records": inactive,
        "validation_record_compatibility": assessments,
        "validation_source_compatibility_summary": summary,
    }
