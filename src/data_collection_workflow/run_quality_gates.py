"""Deterministic final dataset quality gates.

This module evaluates records that have already passed through extraction,
schema validation, normalization, linking, validation, anomaly detection, and
human review application. It does not search, fetch, call an LLM, or alter graph
topology.
"""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit

from .disease_relevance import (
    INCOMPATIBLE_DISEASE,
    UNRELATED_DISEASE,
    assess_record_disease_compatibility,
    build_disease_relevance_context,
)

QUALITY_GATE_METHOD = "deterministic_run_quality_gate_v1"

_TEXT_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_TEXT_MONTH_DATE_RE = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december"
    r")\s+(\d{1,2}),\s*(20\d{2})\b",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

ACCEPTED_STATUSES = {
    "accepted",
    "accepted_with_warnings",
    "accepted_with_review_warning",
    "accepted_after_human_review",
    "corrected_after_human_review",
}

VALIDATION_LIMITED_STATUSES = {
    "no_task_compatible_validation_source",
    "incompatible_validation_source_disabled",
    "validation_limited_no_compatible_source",
    "no_compatible_validation_source",
    "live_validation_pending",
}

PRIMARY_COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
)
EXPLICIT_PRIMARY_COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "deaths",
    "hospitalizations",
)

_SOURCE_BLOCK_STATUSES = {"excluded", "blocked", "not_task_relevant"}
_SOURCE_EXCLUDED_ROLES = {"excluded", "search_endpoint"}
_SOURCE_UNRELATED_STATUSES = {UNRELATED_DISEASE, INCOMPATIBLE_DISEASE}
_DOCUMENT_BLOCK_STATUSES = {UNRELATED_DISEASE, INCOMPATIBLE_DISEASE}
_DOCUMENT_BLOCK_QUALITIES = {"not_task_relevant"}
_CHUNK_BLOCK_STATUSES = {UNRELATED_DISEASE, INCOMPATIBLE_DISEASE}
_SCHEMA_BLOCK_STATUSES = {"invalid", "rejected"}
_ANOMALY_BLOCK_SEVERITIES = {"high", "critical"}
_OUTSIDE_SCOPE_TOKENS = {
    "outside_scope",
    "outside_requested_scope",
    "outside_geography",
    "outside_time_window",
    "disease_mismatch",
}
_NON_PRIMARY_OBSERVATION_TYPES = {
    "zero_case_statement",
    "exposure_monitoring_record",
    "background_context",
    "context_only",
    "ambiguous_public_health_observation",
    "exposure_monitoring_only",
    "zero_case_statement_unverified",
    "surveillance_summary",
    "outbreak_summary",
}
_TASK_AWARE_ACCEPTABLE_OBSERVATION_TYPES = {
    "surveillance_summary",
    "regional_surveillance_dataset",
    "country_year_aggregate_dataset",
    "hospitalization_dataset",
    "death_dataset",
    "official_alert_dataset",
    "outbreak_summary",
}
_TASK_AWARE_BLOCKED_OBSERVATION_TYPES = {
    "zero_case_statement",
    "zero_case_statement_unverified",
    "exposure_monitoring_record",
    "exposure_monitoring_only",
    "ambiguous_public_health_observation",
}
_DIRECT_METRIC_SOFT_OBSERVATION_TYPES = {
    "ambiguous_public_health_observation",
}
_TASK_AWARE_CONTEXT_OBSERVATION_TYPES = {
    "background_context",
    "context_only",
}
_TASK_AWARE_COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "icu_admissions",
    "tests_positive",
    "tests_total",
    "positivity_rate",
    "incidence_rate",
    "cumulative_count",
    "new_count",
    "metric_value",
)
_EXPOSURE_MONITORING_MARKERS = {
    "under public health monitoring",
    "public health monitoring",
    "under monitoring",
    "being monitored",
    "monitored",
    "monitoring",
    "quarantined",
    "quarantine",
    "contacts observed",
    "under observation",
    "travelers monitored",
    "monitoring this situation",
}
_EXPOSURE_CONTEXT_MARKERS = {
    "currently in good health",
    "in good health",
    "potentially exposed",
    "potential exposure",
    "possible exposure",
    "exposed",
    "aboard",
    "passenger",
    "passengers",
    "contacts",
    "contact",
    "returned home",
    "disembarked",
    "traveler",
    "travelers",
}


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    return []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _record_id(record: dict) -> str:
    return str(record.get("record_id") or "")


def _source_id(record: dict) -> str:
    return str(record.get("source_id") or "")


def _supporting_chunk_id(record: dict) -> str:
    return str(record.get("supporting_chunk_id") or "")


def _source_url(record: dict) -> str | None:
    value = record.get("source_url")
    return str(value) if value else None


def _reason_text(*values: Any) -> str:
    return " ".join(str(value or "") for value in values).lower()


def _index_by_id(rows: list[dict], field: str) -> dict[str, dict]:
    return {
        str(row.get(field)): row
        for row in rows
        if isinstance(row, dict) and row.get(field)
    }


def _documents_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("source_id")
        if sid:
            out.setdefault(str(sid), []).append(row)
    return out


def _chunks_by_source(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = row.get("source_id")
        if sid:
            out.setdefault(str(sid), []).append(row)
    return out


def _validation_results_by_record(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ids = set()
        ids.update(str(v) for v in _as_list(row.get("left_record_ids")) if v)
        ids.update(str(v) for v in _as_list(row.get("right_record_ids")) if v)
        if row.get("record_id"):
            ids.add(str(row.get("record_id")))
        for rid in ids:
            out.setdefault(rid, []).append(row)
    return out


def _anomalies_by_record(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = row.get("record_id")
        if rid:
            out.setdefault(str(rid), []).append(row)
    return out


def _review_items_by_record(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ids = set(str(v) for v in _as_list(row.get("related_ids")) if v)
        if row.get("record_id"):
            ids.add(str(row.get("record_id")))
        for rid in ids:
            out.setdefault(rid, []).append(row)
    return out


def _human_review_rejected_record_ids(state: dict) -> set[str]:
    rejected: set[str] = set()
    for row in _as_list(state.get("records_excluded_by_human_review")):
        if isinstance(row, dict) and row.get("record_id"):
            rejected.add(str(row["record_id"]))
    for decision in _as_list(state.get("applied_human_review_decisions")):
        if not isinstance(decision, dict):
            continue
        if decision.get("decision_type") != "reject_record":
            continue
        rejected.update(str(v) for v in _as_list(decision.get("target_ids")) if v)
    for decision in _as_list(state.get("human_review_decisions")):
        if not isinstance(decision, dict):
            continue
        if decision.get("decision_type") != "reject_record":
            continue
        if decision.get("apply_decision") is False:
            continue
        rejected.update(str(v) for v in _as_list(decision.get("target_ids")) if v)
    return rejected


def _has_explicit_human_review_decisions(state: dict) -> bool:
    return bool(
        _as_list(state.get("applied_human_review_decisions"))
        or _as_list(state.get("rejected_human_review_decisions"))
        or _as_list(state.get("human_review_decisions"))
        or _as_list(state.get("records_excluded_by_human_review"))
    )


def _add_block(
    blocks: list[tuple[str, str, str]],
    status: str,
    flag: str,
    reason: str,
) -> None:
    blocks.append((status, flag, reason))


def _add_warning(warnings: list[str], warning: str) -> None:
    if warning and warning not in warnings:
        warnings.append(warning)


def _positive_number(value: Any) -> bool:
    if value in (None, ""):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _has_positive_primary_count(record: dict) -> bool:
    return any(_positive_number(record.get(field)) for field in PRIMARY_COUNT_FIELDS)


def _has_positive_explicit_primary_count(record: dict) -> bool:
    return any(
        _positive_number(record.get(field)) for field in EXPLICIT_PRIMARY_COUNT_FIELDS
    )


def _human_review_blocks_quality_gate(state: dict) -> bool:
    if state.get("human_review_enabled") is False:
        return False
    return True


def _first_block(blocks: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    if not blocks:
        return None
    priority = {
        "excluded_by_human_review": 0,
        "quarantined_disease_mismatch": 1,
        "quarantined_source_not_task_relevant": 2,
        "quarantined_document_not_task_relevant": 3,
        "quarantined_chunk_not_task_relevant": 4,
        "quarantined_schema_invalid": 5,
        "quarantined_normalization_rejected": 6,
        "quarantined_outside_scope": 7,
        "quarantined_validation_conflict": 8,
        "quarantined_critical_anomaly": 9,
        "quarantined_zero_case_statement": 10,
        "quarantined_exposure_monitoring": 10,
        "quarantined_background_context": 10,
        "quarantined_ambiguous_non_primary_observation": 10,
        "quarantined_non_primary_observation": 10,
    }
    return sorted(blocks, key=lambda item: priority.get(item[0], 100))[0]


def _record_observation_types(record: dict) -> list[str]:
    values = _as_list(record.get("observation_types"))
    if not values and record.get("observation_type"):
        values = [record.get("observation_type")]
    observation_types = [str(value) for value in values if value not in (None, "")]
    if (
        _has_exposure_monitoring_language(record)
        and not _has_positive_explicit_primary_count(record)
        and "exposure_monitoring_record" not in observation_types
    ):
        observation_types.append("exposure_monitoring_record")
    return observation_types


def _has_exposure_monitoring_language(record: dict) -> bool:
    text = _reason_text(
        record.get("evidence_quote"),
        record.get("evidence_context"),
        record.get("notes"),
        record.get("source_title"),
        record.get("publisher"),
        record.get("case_definition"),
        record.get("count_semantics"),
    )
    return any(marker in text for marker in _EXPOSURE_MONITORING_MARKERS) and any(
        marker in text for marker in _EXPOSURE_CONTEXT_MARKERS
    )


def _specific_non_primary_status(observation_types: list[str]) -> str:
    types = set(observation_types)
    if "zero_case_statement" in types or "zero_case_statement_unverified" in types:
        return "quarantined_zero_case_statement"
    if "exposure_monitoring_record" in types or "exposure_monitoring_only" in types:
        return "quarantined_exposure_monitoring"
    if "background_context" in types or "context_only" in types:
        return "quarantined_background_context"
    if "ambiguous_public_health_observation" in types:
        return "quarantined_ambiguous_non_primary_observation"
    return "quarantined_non_primary_observation"


def _explicit_review_accepts_record(record: dict) -> bool:
    return bool(record.get("human_review_applied")) and _lower(
        record.get("review_status")
    ) in {
        "accepted",
        "corrected",
        "accepted_after_human_review",
        "corrected_after_human_review",
    }


def _is_non_primary_observation(record: dict) -> bool:
    observation_types = _record_observation_types(record)
    if record.get("primary_case_dataset_eligible") is False:
        return True
    if _has_positive_explicit_primary_count(record):
        return False
    if _has_exposure_monitoring_language(record):
        return True
    if record.get("primary_case_dataset_eligible") is True:
        return False
    if _has_positive_primary_count(record):
        return False
    return any(item in _NON_PRIMARY_OBSERVATION_TYPES for item in observation_types)


def _task_metadata(state: dict) -> dict:
    structured_task = _as_dict(state.get("structured_task"))
    collection_spec = _as_dict(state.get("collection_spec"))
    return {
        "task_disease": structured_task.get("disease") or collection_spec.get("disease"),
        "task_location": structured_task.get("location")
        or collection_spec.get("geography")
        or collection_spec.get("location"),
        "task_start_date": structured_task.get("start_date")
        or collection_spec.get("start_date"),
        "task_end_date": structured_task.get("end_date") or collection_spec.get("end_date"),
    }


def _task_requires_exact_annual_period(state: dict) -> bool:
    contract = _as_dict(state.get("task_evidence_contract"))
    if _lower(contract.get("time_granularity")) == "annual":
        return True
    requirements = contract.get("requirements")
    if isinstance(requirements, list):
        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            period_basis = _lower(requirement.get("period_basis"))
            req_id = _lower(requirement.get("requirement_id") or requirement.get("id"))
            if period_basis == "annual" or "_annual_" in req_id or req_id.endswith("_annual"):
                return True
    return False


def _task_time_granularity(state: dict) -> str:
    contract = _as_dict(state.get("task_evidence_contract"))
    granularity = _lower(contract.get("time_granularity"))
    if granularity:
        return granularity
    requirements = contract.get("requirements")
    if isinstance(requirements, list) and requirements:
        values = {
            _lower(requirement.get("time_granularity") or requirement.get("period_basis"))
            for requirement in requirements
            if isinstance(requirement, dict)
        }
        values.discard("")
        if len(values) == 1:
            return next(iter(values))
    task = _task_metadata(state)
    task_start = _parse_scope_date(task.get("task_start_date"))
    task_end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or task_start
    if task_start and task_end:
        if task_end < task_start:
            task_start, task_end = task_end, task_start
        is_full_calendar_year = (
            task_start.month == 1
            and task_start.day == 1
            and task_end.month == 12
            and task_end.day == 31
            and task_start.year == task_end.year
        )
        if not is_full_calendar_year:
            return "task_window"
    return ""


def _collection_mode(state: dict) -> str:
    structured_task = _as_dict(state.get("structured_task"))
    collection_spec = _as_dict(state.get("collection_spec"))
    workflow = _as_dict(state.get("workflow"))
    return str(
        structured_task.get("collection_mode")
        or collection_spec.get("collection_mode")
        or workflow.get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip() or "standard"


def _direct_collection_enabled(state: dict) -> bool:
    return _collection_mode(state) == "direct_collection"


def _parse_scope_date(value: Any, *, end_of_period: bool = False) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) >= 10:
        text = text[:10]
    if text.isdigit() and len(text) == 4:
        year = int(text)
        return date(year, 12, 31) if end_of_period else date(year, 1, 1)
    parts = text.split("-")
    if len(parts) == 3:
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (TypeError, ValueError):
            return None
    try:
        return datetime.fromisoformat(text).date()
    except (TypeError, ValueError):
        return None


def _parse_date_from_text(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    iso_match = _TEXT_ISO_DATE_RE.search(text)
    if iso_match:
        try:
            return date(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
            )
        except (TypeError, ValueError):
            pass
    month_match = _TEXT_MONTH_DATE_RE.search(text)
    if month_match:
        try:
            return date(
                int(month_match.group(3)),
                _MONTHS[month_match.group(1).lower()],
                int(month_match.group(2)),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _record_scope_date(record: dict) -> date | None:
    for key in ("date_reported", "date_anchor", "date_onset", "date"):
        parsed = _parse_scope_date(record.get(key))
        if parsed:
            return parsed
    return None


def _record_period_dates(record: dict) -> tuple[date | None, date | None]:
    start = None
    end = None
    for key in (
        "event_start_date",
        "period_start_date",
        "metric_period_start",
        "start_date",
    ):
        start = _parse_scope_date(record.get(key))
        if start:
            break
    for key in (
        "event_end_date",
        "period_end_date",
        "metric_period_end",
        "end_date",
    ):
        end = _parse_scope_date(record.get(key), end_of_period=True)
        if end:
            break
    if not start:
        start = _parse_date_from_text(record.get("reporting_period"))
    if not end:
        end = _parse_date_from_text(record.get("reporting_period"))
    return start, end


def _record_event_period_scope_block(record: dict, state: dict) -> tuple[str, str] | None:
    task = _task_metadata(state)
    task_start = _parse_scope_date(task.get("task_start_date"))
    task_end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or task_start
    event_start = _parse_scope_date(record.get("event_start_date"))
    event_end = (
        _parse_scope_date(record.get("event_end_date"), end_of_period=True)
        or event_start
    )
    if not task_start or not task_end or not event_start or not event_end:
        return None
    if task_end < task_start:
        task_start, task_end = task_end, task_start
    if event_end < event_start:
        event_start, event_end = event_end, event_start
    if not (event_end < task_start or event_start > task_end):
        return None
    record["record_period_fit_status"] = "outside_task_window"
    record["record_period_source"] = "event_period"
    return (
        "record_event_period_outside_task_window",
        (
            f"record event period {event_start.isoformat()} to "
            f"{event_end.isoformat()} does not overlap requested task window "
            f"{task_start.isoformat()} to {task_end.isoformat()}"
        ),
    )


def _direct_record_scope_date(record: dict) -> date | None:
    direct_date = _record_scope_date(record)
    if direct_date:
        return direct_date
    for key in (
        "as_of_date",
        "metric_period_end",
        "metric_period_start",
        "reporting_period",
        "source_url",
        "source_title",
        "evidence_quote",
        "count_semantics",
    ):
        parsed = _parse_date_from_text(record.get(key))
        if parsed:
            return parsed
    return None


def _record_date_scope_block(record: dict, state: dict) -> tuple[str, str] | None:
    task = _task_metadata(state)
    start = _parse_scope_date(task.get("task_start_date"))
    end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or start
    record_date = _record_scope_date(record)
    if not start or not end or not record_date:
        return None
    if end < start:
        start, end = end, start
    if start <= record_date <= end:
        return None
    return (
        "record_date_outside_task_window",
        (
            f"record date {record_date.isoformat()} is outside requested "
            f"task window {start.isoformat()} to {end.isoformat()}"
        ),
    )


def _parse_explicit_as_of_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return _parse_scope_date(text)
    return _parse_date_from_text(text)


def _record_as_of_date_scope_block(record: dict, state: dict) -> tuple[str, str] | None:
    task = _task_metadata(state)
    start = _parse_scope_date(task.get("task_start_date"))
    end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or start
    as_of_date = _parse_explicit_as_of_date(record.get("as_of_date"))
    if not start or not end or not as_of_date:
        return None
    if end < start:
        start, end = end, start
    if start <= as_of_date <= end:
        return None
    record["record_period_fit_status"] = "outside_task_window"
    record["record_period_source"] = "as_of_date"
    return (
        "record_as_of_date_outside_task_window",
        (
            f"record as-of date {as_of_date.isoformat()} is outside requested "
            f"task window {start.isoformat()} to {end.isoformat()}"
        ),
    )


def _normalize_location_text(value: Any) -> str:
    text = _lower(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _task_location_requires_subnational_fit(location: str | None) -> bool:
    normalized = _normalize_location_text(location)
    if not normalized:
        return False
    country_or_global = {
        "global",
        "worldwide",
        "world",
        "united states",
        "united states of america",
        "usa",
        "us",
        "u s",
        "u s a",
    }
    return normalized not in country_or_global


def _record_location_text(record: dict) -> str:
    return _normalize_location_text(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "subnational_location",
                "locality",
                "admin_area",
                "geographic_scope",
                "location",
                "place",
                "evidence_quote",
            )
        )
    )


def _record_geography_scope_block(record: dict, state: dict) -> tuple[str, str] | None:
    task = _task_metadata(state)
    task_location = str(task.get("task_location") or "").strip()
    task_location_norm = _normalize_location_text(task_location)
    if not task_location_norm:
        return None
    if task_location_norm in {"global", "world", "worldwide"}:
        return None

    scope_type = _normalize_location_text(record.get("geographic_scope_type"))
    scope = _normalize_location_text(record.get("geographic_scope"))
    country = _normalize_location_text(record.get("country"))
    subnational = _normalize_location_text(record.get("subnational_location"))
    broader_scope_types = {
        "global",
        "multi country",
        "region",
        "international",
        "multi-country",
        "supranational",
    }
    if not _task_location_requires_subnational_fit(task_location):
        if scope_type in broader_scope_types or scope in {
            "global",
            "world",
            "worldwide",
            "region of the americas",
            "americas",
            "europe",
            "africa",
            "asia",
        }:
            return (
                "record_geography_broader_than_task",
                (
                    f"record geography is broader than requested location "
                    f"{task_location}; record has geographic_scope_type="
                    f"{record.get('geographic_scope_type')!r}, geographic_scope="
                    f"{record.get('geographic_scope')!r}, country="
                    f"{record.get('country')!r}"
                ),
            )
        return None

    record_location = _record_location_text(record)
    if task_location_norm in record_location:
        return None

    national_scope_types = {
        "country",
        "national",
        "global",
        "multi country",
        "region",
        "international",
    }
    if (
        scope_type in national_scope_types
        or scope in {"united states", "united states of america", "usa", "us"}
        or (country and not subnational)
    ):
        return (
            "record_geography_broader_than_task",
            (
                f"record geography is broader than requested location "
                f"{task_location}; record has geographic_scope_type="
                f"{record.get('geographic_scope_type')!r}, geographic_scope="
                f"{record.get('geographic_scope')!r}, subnational_location="
                f"{record.get('subnational_location')!r}"
            ),
        )
    return None


def _task_fit_evidence_text(
    record: dict,
    source: dict | None,
    documents: list[dict],
    chunks: list[dict],
) -> str:
    values: list[str] = []
    for key in (
        "evidence_quote",
        "evidence_context",
        "source_title",
        "source_url",
        "publisher",
        "actual_publisher",
    ):
        values.append(str(record.get(key) or ""))
    if source:
        for key in (
            "title",
            "canonical_url",
            "url",
            "publisher",
            "actual_publisher",
            "snippet",
            "description",
        ):
            values.append(str(source.get(key) or ""))
    for doc in documents:
        for key in ("title", "url", "canonical_url", "clean_text", "text"):
            values.append(str(doc.get(key) or ""))
    for chunk in chunks:
        for key in ("text", "heading_context", "table_header"):
            values.append(str(chunk.get(key) or ""))
    return _normalize_location_text(" ".join(values))


def _direct_record_task_fit_blocks(
    record: dict,
    source: dict | None,
    documents: list[dict],
    chunks: list[dict],
    state: dict,
) -> list[tuple[str, str, str]]:
    if not _direct_collection_enabled(state):
        return []
    blocks: list[tuple[str, str, str]] = []
    reasons: list[str] = []
    task = _task_metadata(state)
    task_location = str(task.get("task_location") or "").strip()
    task_location_norm = _normalize_location_text(task_location)
    evidence_text = _task_fit_evidence_text(record, source, documents, chunks)

    if task_location_norm and _task_location_requires_subnational_fit(task_location):
        record_location = _record_location_text(record)
        if task_location_norm in record_location and task_location_norm not in evidence_text:
            reason = (
                "record claims the requested subnational geography, but the "
                "source/chunk/evidence text does not explicitly support that "
                "geography; task geography cannot be inherited into strict final"
            )
            blocks.append(
                (
                    "quarantined_outside_scope",
                    "record_geography_inherited_without_source_evidence",
                    reason,
                )
            )
            reasons.append("record_geography_inherited_without_source_evidence")
            record["record_geography_fit_status"] = "inherited_without_evidence"
            record["record_geography_source"] = "task_context_unverified"
        elif task_location_norm in evidence_text:
            record["record_geography_fit_status"] = (
                record.get("record_geography_fit_status") or "exact"
            )
            record["record_geography_source"] = (
                record.get("record_geography_source") or "source_or_evidence_text"
            )

    metric_period_source = _lower(record.get("metric_period_source"))
    if "task_window" in metric_period_source or "user_window" in metric_period_source:
        reason = (
            "record metric period was filled from the user/task window rather "
            "than a source, row, column, or narrative period; native source "
            "period semantics must be preserved before strict final inclusion"
        )
        blocks.append(
            (
                "quarantined_outside_scope",
                "record_period_inherited_from_task_window_without_source_evidence",
                reason,
            )
        )
        reasons.append("record_period_inherited_from_task_window_without_source_evidence")
        record["record_period_fit_status"] = "inherited_without_evidence"
        record["record_period_source"] = "task_context_unverified"
    elif record.get("metric_period_start") or record.get("metric_period_end"):
        record["record_period_source"] = (
            record.get("record_period_source")
            or str(record.get("metric_period_source") or "record_metric_period")
        )

    if blocks:
        record["record_task_fit_status"] = "failed"
        existing = [str(value) for value in _as_list(record.get("record_task_fit_reasons"))]
        for reason in reasons:
            if reason not in existing:
                existing.append(reason)
        record["record_task_fit_reasons"] = existing
    else:
        record["record_task_fit_status"] = (
            record.get("record_task_fit_status") or "passed"
        )
    return blocks


def _record_period_scope_blocks(record: dict, state: dict) -> list[tuple[str, str]]:
    task = _task_metadata(state)
    task_start = _parse_scope_date(task.get("task_start_date"))
    task_end = (
        _parse_scope_date(task.get("task_end_date"), end_of_period=True)
        or task_start
    )
    event_scope_block = _record_event_period_scope_block(record, state)
    if event_scope_block:
        return [event_scope_block]
    period_start, period_end = _record_period_dates(record)
    if not task_start or not task_end or not period_start or not period_end:
        return []
    if task_end < task_start:
        task_start, task_end = task_end, task_start
    if period_end < period_start:
        period_start, period_end = period_end, period_start
    blocks: list[tuple[str, str]] = []
    if period_end < task_start or period_start > task_end:
        blocks.append(
            (
                "record_period_outside_task_window",
                (
                    f"record period {period_start.isoformat()} to "
                    f"{period_end.isoformat()} does not overlap requested "
                    f"task window {task_start.isoformat()} to {task_end.isoformat()}"
                ),
            )
        )
        return blocks

    task_span_days = (task_end - task_start).days + 1
    period_span_days = (period_end - period_start).days + 1
    allowed_slack = timedelta(days=14)
    if (
        period_span_days > max(task_span_days + 14, 31)
        and (
            period_start < task_start - allowed_slack
            or period_end > task_end + allowed_slack
        )
    ):
        blocks.append(
            (
                "record_period_too_broad_for_task_window",
                (
                    f"record period {period_start.isoformat()} to "
                    f"{period_end.isoformat()} is too broad for requested "
                    f"task window {task_start.isoformat()} to {task_end.isoformat()}"
                ),
            )
        )
    return blocks


def _explicit_years_from_text(value: Any) -> set[int]:
    years: set[int] = set()
    text = str(value or "")
    for match in re.finditer(r"\b20\d{2}\b", text):
        try:
            years.add(int(match.group(0)))
        except (TypeError, ValueError):
            continue
    return years


def _task_years(state: dict) -> set[int]:
    task = _task_metadata(state)
    start = _parse_scope_date(task.get("task_start_date"))
    end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or start
    if not start:
        return set()
    if end and end < start:
        start, end = end, start
    if not end:
        return {start.year}
    return set(range(start.year, end.year + 1))


def _source_explicit_years(record: dict, source: dict | None) -> set[int]:
    values: list[Any] = []
    if source:
        values.extend(
            [
                source.get("canonical_url"),
                source.get("url"),
                source.get("title"),
                source.get("source_title"),
                source.get("reporting_period_label"),
            ]
        )
    values.extend(
        [
            record.get("source_url"),
            record.get("source_title"),
            record.get("source_reporting_period_label"),
        ]
    )
    years: set[int] = set()
    for value in values:
        years.update(_explicit_years_from_text(value))
    return years


def _source_period_scope_block(
    record: dict,
    source: dict | None,
    state: dict,
) -> tuple[str, str] | None:
    source_years = _source_explicit_years(record, source)
    task_years = _task_years(state)
    if not source_years or not task_years:
        return None
    if source_years & task_years:
        return None
    record["source_period_validation_status"] = "source_period_mismatch"
    record["source_period_mismatch_reasons"] = [
        (
            f"source explicit years {sorted(source_years)} do not overlap "
            f"task years {sorted(task_years)}"
        )
    ]
    return (
        "source_period_mismatch",
        (
            f"source explicit years {sorted(source_years)} do not overlap "
            f"requested task years {sorted(task_years)}"
        ),
    )


def _task_text(state: dict) -> str:
    structured_task = _as_dict(state.get("structured_task"))
    collection_spec = _as_dict(state.get("collection_spec"))
    disease_intelligence = _as_dict(state.get("disease_intelligence"))
    return _lower(
        " ".join(
            str(part or "")
            for part in (
                structured_task.get("disease"),
                structured_task.get("user_request"),
                collection_spec.get("disease"),
                collection_spec.get("user_request"),
                disease_intelligence.get("disease_input"),
                disease_intelligence.get("disease_standard_name"),
            )
        )
    )


def _default_seasonal_flu_task(state: dict) -> bool:
    text = _task_text(state)
    if not any(token in text for token in ("flu", "influenza")):
        return False
    return not any(token in text for token in ("h5n1", "h5", "avian", "bird flu"))


def _has_nonseasonal_flu_token(text: str) -> bool:
    return any(
        token in text
        for token in ("h5n1", "h5", "avian influenza", "bird flu")
    )


def _has_seasonal_flu_subtype_token(text: str) -> bool:
    return any(
        token in text
        for token in (
            "h1n1",
            "h3n2",
            "a(h1n1)",
            "a(h3n2)",
            "a(h1n1)pdm09",
            "influenza b",
            "influenza-like illness",
            "ili",
        )
    )


def _record_has_nonseasonal_influenza_subtype(record: dict) -> bool:
    identity_text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "disease",
                "disease_standard_name",
                "virus_or_syndrome",
                "pathogen_or_syndrome",
                "count_semantics",
                "statistical_count_type",
                "observation_type",
                "case_definition",
            )
        )
    )
    observation_types = " ".join(
        str(value or "") for value in _as_list(record.get("observation_types"))
    )
    identity_text = f"{identity_text} {_lower(observation_types)}"
    if _has_nonseasonal_flu_token(identity_text):
        return True
    if _has_seasonal_flu_subtype_token(identity_text):
        return False
    context_text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "case_definition",
                "source_title",
                "evidence_quote",
            )
        )
    )
    return _has_nonseasonal_flu_token(context_text)


_SECONDARY_OR_REVIEW_HOST_TOKENS = (
    "abcnews.",
    "nvdaily.",
    "cidrap.umn.edu",
    "usafacts.org",
    "aha.org",
    "facebook.",
    "instagram.",
    "twitter.",
    "x.com",
    "tiktok.",
    "reddit.",
    "flutrackers.",
    "substack.",
    "wikipedia.",
)

_OFFICIAL_PUBLIC_HEALTH_HOST_TOKENS = (
    ".gov",
    ".gov.",
    ".gov/",
    "cdc.gov",
    "who.int",
    "paho.org",
    "ecdc.europa.eu",
    "canada.ca",
    "gov.uk",
    "europa.eu",
)

_LOCAL_VECTOR_CONTROL_IDENTITY_TOKENS = (
    "mosquito and vector control district",
    "mosquito vector control district",
    "vector control district",
    "mosquito abatement district",
    "mosquito control district",
    "vector-borne disease control district",
)


def _source_host(source: dict | None, record: dict) -> str:
    for value in (
        record.get("source_url"),
        (source or {}).get("canonical_url"),
        (source or {}).get("url"),
        record.get("canonical_url"),
        record.get("url"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        try:
            host = urlsplit(text).netloc.lower()
        except ValueError:
            host = ""
        if host:
            return host[4:] if host.startswith("www.") else host
    return ""


def _host_requires_source_trust_review(host: str) -> bool:
    value = str(host or "").lower()
    return bool(value) and any(token in value for token in _SECONDARY_OR_REVIEW_HOST_TOKENS)


def _host_supports_official_public_health_identity(host: str) -> bool:
    value = str(host or "").lower()
    return bool(value) and any(token in value for token in _OFFICIAL_PUBLIC_HEALTH_HOST_TOKENS)


_UNVERIFIED_SOURCE_IDENTITY_WARNINGS = {
    "source_identity_unverified",
    "metadata_only_identity",
    "publisher_domain_mismatch",
    "official_source_type_claim_without_domain_support",
    "llm_official_identity_conflicts_with_unverified_domain",
    "llm_official_identity_conflicts_with_domain",
    "publisher_from_upstream_quote_not_page_publisher",
    "publisher_from_search_metadata_unverified",
    "search_provider_not_publisher",
    "actual_publisher_unknown",
    "missing_publisher",
}


def _source_metadata_identity_requires_review(
    source: dict | None,
    record: dict,
) -> bool:
    host = _source_host(source, record)
    if not host or _host_supports_official_public_health_identity(host):
        return False
    if _source_has_local_vector_control_identity(source, record):
        return False
    warning_values: list[str] = []
    for key in (
        "source_identity_warnings",
        "publisher_warning_flags",
        "source_type_warning_flags",
        "credibility_flags",
    ):
        warning_values.extend(_as_list((source or {}).get(key)))
        warning_values.extend(_as_list(record.get(key)))
    warning_text = {_lower(value) for value in warning_values if str(value or "").strip()}
    if not (warning_text & _UNVERIFIED_SOURCE_IDENTITY_WARNINGS):
        return False
    identity_values = [
        (source or {}).get("source_type"),
        (source or {}).get("source_type_final"),
        (source or {}).get("publisher_type"),
        record.get("source_type"),
        record.get("source_type_final"),
        record.get("publisher_type"),
        (source or {}).get("credibility_level"),
        record.get("credibility_level"),
    ]
    identity_text = _lower(" ".join(str(value or "") for value in identity_values))
    return any(
        token in identity_text
        for token in (
            "official",
            "public_health",
            "international_organization_report",
            "national_public_health_agency",
            "state_or_local_public_health_agency",
            "high",
        )
    )


def _source_identity_text(source: dict | None, record: dict) -> str:
    values = []
    if source:
        values.extend(
            [
                source.get("title"),
                source.get("source_title"),
                source.get("publisher"),
                source.get("actual_publisher"),
                source.get("canonical_url"),
                source.get("url"),
                " ".join(str(flag) for flag in _as_list(source.get("credibility_flags"))),
            ]
        )
    values.extend(
        [
            record.get("source_title"),
            record.get("publisher"),
            record.get("actual_publisher"),
            record.get("source_url"),
            record.get("evidence_quote"),
            " ".join(str(flag) for flag in _as_list(record.get("credibility_flags"))),
        ]
    )
    return _lower(" ".join(str(value or "") for value in values))


def _source_has_local_vector_control_identity(
    source: dict | None,
    record: dict,
) -> bool:
    host = _source_host(source, record)
    if not host or _host_requires_source_trust_review(host):
        return False
    text = _source_identity_text(source, record)
    return any(token in text for token in _LOCAL_VECTOR_CONTROL_IDENTITY_TOKENS)


def _source_has_supported_official_public_health_identity(
    source: dict | None,
    record: dict,
) -> bool:
    if not _source_claims_public_health_agency(source, record):
        return False
    host = _source_host(source, record)
    return _host_supports_official_public_health_identity(
        host
    ) or _source_has_local_vector_control_identity(source, record)


def _source_claims_public_health_agency(source: dict | None, record: dict) -> bool:
    text = _lower(
        " ".join(
            str(value or "")
            for value in (
                (source or {}).get("source_type"),
                (source or {}).get("source_type_final"),
                record.get("source_type"),
                record.get("source_type_final"),
            )
        )
    )
    return any(
        token in text
        for token in (
            "official_public_health_agency",
            "national_public_health_agency",
            "state_or_local_public_health_agency",
            "international_public_health_agency",
            "public_health_agency",
        )
    )


def _source_domain_identity_requires_review(source: dict | None, record: dict) -> bool:
    host = _source_host(source, record)
    if not host:
        return False
    if (source or {}).get("source_identity_unverified") or record.get(
        "source_identity_unverified"
    ):
        return True
    if _source_metadata_identity_requires_review(source, record):
        return True
    if _host_requires_source_trust_review(host):
        return True
    if _source_claims_public_health_agency(source, record) and not _source_has_supported_official_public_health_identity(source, record):
        return True
    return False


def _source_identity_unverified_flags(source: dict | None, record: dict) -> list[str]:
    flags: list[str] = []
    host = _source_host(source, record)
    verified_official_host = bool(
        host and _host_supports_official_public_health_identity(host)
    )
    for key in (
        "source_identity_unverified",
        "metadata_only_identity",
        "publisher_domain_mismatch",
    ):
        if verified_official_host and key == "metadata_only_identity":
            continue
        if (source or {}).get(key) or record.get(key):
            flags.append(key)
    for warning in _as_list((source or {}).get("source_identity_warnings")) + _as_list(
        record.get("source_identity_warnings")
    ):
        text = str(warning or "").strip()
        if verified_official_host and text in {
            "publisher_from_search_metadata_unverified",
            "search_provider_not_publisher",
            "actual_publisher_unknown",
            "missing_publisher",
            "metadata_only_identity",
        }:
            continue
        if text in {
            *list(_UNVERIFIED_SOURCE_IDENTITY_WARNINGS),
        }:
            flags.append(text)
    return sorted(dict.fromkeys(flags))


def _source_is_official_or_high_trust(source: dict | None, record: dict) -> bool:
    if _source_domain_identity_requires_review(source, record):
        return False
    if _lower((source or {}).get("credibility_level")) == "high":
        return True
    if _lower(record.get("credibility_level")) == "high":
        return True
    values = []
    if source:
        values.extend(
            [
                source.get("source_type"),
                source.get("source_type_final"),
                source.get("publisher"),
                source.get("actual_publisher"),
                source.get("canonical_url"),
                source.get("url"),
                source.get("credibility_level"),
            ]
        )
    values.extend(
        [
            record.get("source_type"),
            record.get("source_type_final"),
            record.get("publisher"),
            record.get("actual_publisher"),
            record.get("source_url"),
            record.get("credibility_level"),
        ]
    )
    text = _lower(" ".join(str(value or "") for value in values))
    return (
        "official_public_health_agency" in text
        or "public_health_agency" in text
        or "department of health" in text
        or ".gov" in text
    )


def _source_requires_trust_review(source: dict | None, record: dict) -> bool:
    if _source_domain_identity_requires_review(source, record):
        return True
    source_type_values = [
        (source or {}).get("source_type"),
        (source or {}).get("source_type_final"),
        record.get("source_type"),
        record.get("source_type_final"),
    ]
    if any(
        _lower(value) in {"unknown", "unknown_source", "unclassified_source"}
        for value in source_type_values
    ):
        return True

    publisher_values = [
        (source or {}).get("publisher"),
        (source or {}).get("actual_publisher"),
        record.get("publisher"),
        record.get("actual_publisher"),
    ]
    provenance_text = _lower(
        " ".join(
            str(value or "")
            for value in (
                *publisher_values,
                (source or {}).get("canonical_url"),
                (source or {}).get("url"),
                record.get("source_url"),
                *source_type_values,
            )
        )
    )
    if (
        not any(str(value or "").strip() for value in publisher_values)
        and ".gov" not in provenance_text
        and "department of health" not in provenance_text
        and "official_public_health_agency" not in provenance_text
        and "public_health_agency" not in provenance_text
    ):
        return True

    values = []
    if source:
        values.extend(
            [
                source.get("source_type"),
                source.get("source_type_final"),
                source.get("source_role"),
                source.get("source_role_final"),
                source.get("publisher"),
                source.get("actual_publisher"),
                source.get("canonical_url"),
                source.get("url"),
                source.get("credibility_level"),
                " ".join(str(flag) for flag in _as_list(source.get("credibility_flags"))),
            ]
        )
    values.extend(
        [
            record.get("source_type"),
            record.get("source_type_final"),
            record.get("publisher"),
            record.get("actual_publisher"),
            record.get("source_url"),
            record.get("credibility_level"),
            " ".join(str(flag) for flag in _as_list(record.get("credibility_flags"))),
        ]
    )
    text = _lower(" ".join(str(value or "") for value in values))
    if any(
        token in text
        for token in (
            "social_media",
            "social post",
            "facebook.com",
            "instagram.com",
            "x.com",
            "twitter.com",
            "forum",
            "community_forum",
            "flutrackers",
            "secondary_media",
            "secondary source",
            "commercial_media",
            "news_media",
            "media_report",
            "news article",
            "unknown_source",
            "unknown source",
            "unclassified_source",
            "needs_review",
            "low credibility",
        )
    ):
        return True
    if _source_has_supported_official_public_health_identity(source, record):
        return False
    if _lower((source or {}).get("credibility_level")) in {"low", "excluded", "needs_review"}:
        return True
    if _lower(record.get("credibility_level")) in {"low", "excluded", "needs_review"}:
        return True
    for value in (
        (source or {}).get("credibility_score"),
        record.get("credibility_score"),
    ):
        try:
            if value not in (None, "") and float(value) < 0.55:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _accepted_review_warning_pending_block(
    record: dict,
    source: dict | None,
    warnings: list[str],
) -> tuple[str, str] | None:
    if "accepted_with_review_warning" not in warnings:
        return None
    if _source_identity_unverified_flags(source, record):
        return (
            "source_identity_requires_human_review",
            (
                "record has review-required source identity warnings and cannot "
                "enter the final dataset without human adjudication"
            ),
        )
    if record.get("primary_case_dataset_eligible") and not _source_is_official_or_high_trust(
        source, record
    ):
        return (
            "source_trust_requires_human_review",
            (
                "record is a primary case candidate from a source that is not "
                "verified high-trust enough for final inclusion"
            ),
        )
    return None


def _source_identity_pending_block(
    record: dict,
    source: dict | None,
) -> tuple[str, str] | None:
    if not _source_identity_unverified_flags(source, record):
        return None
    if not (
        record.get("primary_case_dataset_eligible")
        or _has_task_aware_count(record)
        or _has_direct_collection_count_semantics(record)
    ):
        return None
    return (
        "source_identity_requires_human_review",
        (
            "record has review-required source identity warnings and cannot "
            "enter the final dataset without human adjudication"
        ),
    )


def _has_task_aware_count(record: dict) -> bool:
    present = [
        record.get(field)
        for field in _TASK_AWARE_COUNT_FIELDS
        if record.get(field) not in (None, "")
    ]
    return bool(present)


def _task_aware_dataset_view(observation_types: list[str]) -> str:
    types = set(observation_types)
    if "hospitalization_dataset" in types:
        return "task_aware_hospitalization_dataset"
    if "death_dataset" in types:
        return "task_aware_death_dataset"
    if "regional_surveillance_dataset" in types:
        return "task_aware_regional_surveillance_dataset"
    if "country_year_aggregate_dataset" in types:
        return "task_aware_country_year_aggregate_dataset"
    if "official_alert_dataset" in types:
        return "task_aware_official_alert_dataset"
    if "outbreak_summary" in types:
        return "task_aware_outbreak_summary"
    return "task_aware_surveillance_summary"


def _is_task_aware_accepted_observation(
    record: dict,
    source: dict | None,
    observation_types: list[str],
) -> bool:
    types = set(observation_types)
    if types & _TASK_AWARE_BLOCKED_OBSERVATION_TYPES:
        return False
    if not (types & _TASK_AWARE_ACCEPTABLE_OBSERVATION_TYPES):
        return False
    if _has_inconsistent_count_metric_unit(record):
        return False
    if not _has_task_aware_count(record):
        return False
    return _source_is_official_or_high_trust(source, record)


def _has_direct_collection_count_semantics(record: dict) -> bool:
    if not _has_task_aware_count(record):
        return False
    if _is_descriptive_duration_metric(record):
        return False
    if _has_inconsistent_count_metric_unit(record):
        return False
    text = _reason_text(
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("reporting_period"),
        record.get("source_title"),
        record.get("evidence_quote"),
    )
    if any(
        token in text
        for token in (
            "surveillance",
            "aggregate",
            "weekly",
            "monthly",
            "annual",
            "laboratory",
            "positive",
            "hospitalization",
            "death",
            "ili",
            "ed visit",
            "emergency department",
            "metric",
            "percent",
            "rate",
            "case",
            "confirmed",
            "probable",
            "suspected",
            "cumulative",
            "newly reported",
        )
    ):
        return True
    return any(record.get(field) not in (None, "") for field in _TASK_AWARE_COUNT_FIELDS)


def _has_public_health_metric(record: dict) -> bool:
    if record.get("metric_value") in (None, ""):
        return False
    if _is_descriptive_duration_metric(record):
        return False
    if _has_inconsistent_count_metric_unit(record):
        return False
    text = _reason_text(
        record.get("metric_name"),
        record.get("metric_category"),
        record.get("metric_unit"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
    )
    if not text:
        return False
    return any(
        token in text
        for token in (
            "positivity",
            "positive",
            "percent",
            "percentage",
            "rate",
            "ili",
            "ed visit",
            "emergency department",
            "hospital",
            "death",
            "outbreak",
            "test",
            "specimen",
            "case",
            "surveillance",
            "metric",
        )
    )


def _structured_metric_has_bound_disease_context(
    record: dict,
    source: dict | None,
    state: dict,
    disease_assessment: dict,
) -> bool:
    """Accept a metric row whose disease identity is supplied by its bound table context."""

    if not _direct_collection_enabled(state):
        return False
    if "local_evidence_disease_mismatch" not in str(
        disease_assessment.get("reason") or ""
    ):
        return False
    if (
        disease_assessment.get("incompatible_disease_terms_found")
        or disease_assessment.get("record_local_incompatible_terms_found")
        or disease_assessment.get("record_local_numeric_incompatible_terms_found")
    ):
        return False
    if not disease_assessment.get("target_disease_terms_found"):
        return False
    if _lower(record.get("metric_row_binding_status")) not in {
        "resolved",
        "bound",
        "verified",
    }:
        return False
    if not (
        _as_list(record.get("coverage_requirement_ids"))
        or record.get("matched_requirement_id")
        or _as_list(record.get("matched_requirement_ids"))
    ):
        return False
    if not _source_is_official_or_high_trust(source, record):
        return False
    return _has_public_health_metric(record)


_NON_CASE_PUBLIC_HEALTH_METRIC_CATEGORIES = {
    "lab_positive_count",
    "positive_test_count",
    "lab_positivity_percent",
    "positivity_percent",
    "lab_test_count",
    "test_count",
    "ili_percent",
    "ed_visit_percent",
    "emergency_department_visit_percent",
    "hospitalization_rate",
    "hospitalization_percent",
    "death_rate",
    "outbreak_count",
}
_NON_CASE_PUBLIC_HEALTH_METRIC_TOKENS = (
    "lab positive",
    "positive lab",
    "positive specimen",
    "positive test",
    "positivity",
    "specimen tested",
    "specimens tested",
    "test count",
    "tests total",
    "ili",
    "ed visit",
    "emergency department",
    "hospitalization rate",
    "hospitalization percent",
    "death rate",
    "outbreak count",
)
_DESCRIPTIVE_DURATION_METRIC_UNITS = {
    "day",
    "days",
    "hour",
    "hours",
    "week",
    "weeks",
}
_DESCRIPTIVE_DURATION_METRIC_TOKENS = (
    "median",
    "range",
    "duration",
    "length of stay",
    "time to",
    "days from",
    "hours from",
    "weeks from",
    "onset to",
    "admission",
)


def _is_descriptive_duration_metric(record: dict) -> bool:
    unit = _lower(record.get("metric_unit"))
    if unit not in _DESCRIPTIVE_DURATION_METRIC_UNITS:
        return False
    category = _lower(record.get("metric_category"))
    if category and category not in {"other", "duration", "time_interval"}:
        return False
    text = _reason_text(
        record.get("metric_name"),
        record.get("metric_column_label"),
        " ".join(str(value) for value in _as_list(record.get("source_column_labels"))),
        record.get("table_header"),
        record.get("evidence_quote"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
    )
    return any(token in text for token in _DESCRIPTIVE_DURATION_METRIC_TOKENS)


def _has_inconsistent_count_metric_unit(record: dict) -> bool:
    category = _lower(record.get("metric_category"))
    if not category.endswith("_count"):
        return False
    unit = _lower(record.get("metric_unit"))
    if not unit:
        return False
    if unit in {"percent", "percentage", "%", "rate"}:
        return True
    if "%" in unit:
        return True
    return unit.startswith("per ") or " per " in unit


def _is_non_case_public_health_metric(record: dict) -> bool:
    if not _has_public_health_metric(record):
        return False
    category = _lower(record.get("metric_category"))
    if category in _NON_CASE_PUBLIC_HEALTH_METRIC_CATEGORIES:
        return True
    text = _reason_text(
        record.get("metric_name"),
        record.get("metric_category"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("evidence_quote"),
    )
    if any(token in text for token in _NON_CASE_PUBLIC_HEALTH_METRIC_TOKENS):
        return True
    if "case" in text and not any(
        token in text
        for token in (
            "lab",
            "test",
            "specimen",
            "positivity",
            "ili",
            "ed visit",
            "emergency department",
            "outbreak",
        )
    ):
        return False
    return False


def _coerce_non_case_public_health_metric(record: dict) -> dict:
    if not _is_non_case_public_health_metric(record):
        return record

    coerced = dict(record)
    metric_value = coerced.get("metric_value")
    category = _lower(coerced.get("metric_category"))
    text = _reason_text(
        coerced.get("metric_name"),
        coerced.get("metric_category"),
        coerced.get("count_semantics"),
        coerced.get("statistical_count_type"),
    )
    if category in {"lab_positive_count", "positive_test_count"} or (
        "positive" in text and "percent" not in text and "positivity" not in text
    ):
        coerced.setdefault("tests_positive", metric_value)
        if coerced.get("tests_positive") in (None, ""):
            coerced["tests_positive"] = metric_value
    if category in {"lab_test_count", "test_count"} or "specimen tested" in text:
        coerced.setdefault("tests_total", metric_value)
        if coerced.get("tests_total") in (None, ""):
            coerced["tests_total"] = metric_value
    if category in {"lab_positivity_percent", "positivity_percent"} or "positivity" in text:
        coerced.setdefault("positivity_rate", metric_value)
        if coerced.get("positivity_rate") in (None, ""):
            coerced["positivity_rate"] = metric_value

    original_case_values: dict[str, Any] = {}
    for field in (
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
    ):
        if coerced.get(field) not in (None, ""):
            original_case_values[field] = coerced.get(field)
            coerced[field] = None
    if original_case_values:
        coerced["non_case_metric_original_case_field_values"] = original_case_values

    observation_types = [
        value
        for value in _record_observation_types(coerced)
        if value not in {
            "confirmed_case_record",
            "probable_case_record",
            "suspected_case_record",
            "unspecified_case_record",
        }
    ]
    if "surveillance_summary" not in observation_types:
        observation_types.insert(0, "surveillance_summary")
    coerced["observation_types"] = observation_types
    coerced["observation_type"] = observation_types[0]
    coerced["primary_case_dataset_eligible"] = False
    semantic_warnings = _as_list(coerced.get("semantic_warnings"))
    if "public_health_metric_not_primary_case_record" not in semantic_warnings:
        semantic_warnings.append("public_health_metric_not_primary_case_record")
    coerced["semantic_warnings"] = semantic_warnings
    return coerced


def _has_resolved_metric_row_binding(record: dict) -> bool:
    if str(record.get("chunk_kind") or "").lower() != "metric_row":
        return True
    status = str(record.get("metric_row_binding_status") or "").lower()
    return status not in {"", "unresolved", "missing", "failed"}


def _has_strong_direct_public_health_metric(record: dict) -> bool:
    return _has_public_health_metric(record) and _has_resolved_metric_row_binding(record)


_PREVIOUS_COLUMN_RE = re.compile(
    r"\b(previous|prior|last)\s+(week|period)\b",
    re.IGNORECASE,
)
_WEAK_COLUMN_LABEL_RE = re.compile(r"^column[_\s-]?\d+$", re.IGNORECASE)


def _metric_period_dates(record: dict) -> tuple[date | None, date | None]:
    start = _parse_scope_date(record.get("metric_period_start"))
    end = _parse_scope_date(record.get("metric_period_end"), end_of_period=True)
    if start and end and end < start:
        start, end = end, start
    return start, end


def _metric_period_day_count(record: dict) -> int | None:
    start, end = _metric_period_dates(record)
    if not start or not end:
        return None
    return (end - start).days + 1


def _metric_period_semantics_text(record: dict) -> str:
    return _reason_text(
        record.get("metric_period_label"),
        record.get("reporting_period"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("metric_name"),
        record.get("metric_category"),
        record.get("source_title"),
        record.get("evidence_quote"),
    )


def _normalize_non_weekly_metric_column_semantics(record: dict) -> None:
    if not _has_public_health_metric(record):
        return
    if str(record.get("metric_column_semantics_status") or "").lower() not in {
        "ambiguous",
        "",
    }:
        return
    day_count = _metric_period_day_count(record)
    if not day_count:
        return
    text = _metric_period_semantics_text(record)
    period_type: str | None = None
    reason: str | None = None
    if re.search(r"\bannual\b|\byear(?:ly)?\b|\b20\d{2}\b", text) and day_count >= 300:
        period_type = "annual_period"
        reason = "resolved_from_explicit_annual_metric_period"
    elif any(marker in text for marker in ("multi-year", "multi year", "trend")) and day_count >= 365:
        period_type = "multi_year_trend"
        reason = "resolved_from_explicit_multi_year_metric_period"
    elif any(marker in text for marker in ("season-to-date", "season to date", "cumulative")):
        period_type = "season_to_date"
        reason = "resolved_from_explicit_cumulative_metric_period"
        record["count_semantics"] = record.get("count_semantics") or "cumulative"
        record["statistical_count_type"] = (
            record.get("statistical_count_type") or "cumulative"
        )
    elif day_count >= 28 and str(record.get("metric_period_source") or "").strip():
        period_type = "period_aggregate"
        reason = "resolved_from_explicit_metric_period"
    if not period_type:
        return
    record["metric_column_semantics_status"] = "resolved"
    record["resolved_column_period_type"] = period_type
    record["column_period_resolution_reason"] = reason
    record["column_semantics_resolution_method"] = "metric_period"
    record["column_semantics_confidence"] = max(
        float(record.get("column_semantics_confidence") or 0.0),
        0.85,
    )
    warnings = _as_list(record.get("semantic_warnings"))
    if "non_weekly_metric_period_semantics_resolved" not in warnings:
        warnings.append("non_weekly_metric_period_semantics_resolved")
    record["semantic_warnings"] = warnings


def _direct_metric_column_semantics_blocks(record: dict) -> list[tuple[str, str, str]]:
    if not _has_public_health_metric(record):
        return []
    _normalize_non_weekly_metric_column_semantics(record)
    source_label = str(record.get("source_column_label") or "").strip()
    metric_label = str(record.get("metric_column_label") or "").strip()
    semantics_status = str(
        record.get("metric_column_semantics_status") or ""
    ).strip().lower()
    period_type = str(record.get("resolved_column_period_type") or "").strip().lower()
    metric_period_source = str(record.get("metric_period_source") or "").strip().lower()
    labels_text = " ".join(label for label in (source_label, metric_label) if label)
    blocks: list[tuple[str, str, str]] = []
    if semantics_status == "ambiguous" or period_type in {
        "ambiguous_column",
        "ambiguous_previous_period",
    }:
        blocks.append(
            (
                "quarantined_schema_invalid",
                "ambiguous_metric_column_semantics",
                (
                    "direct_collection metric records must resolve whether the "
                    "table column is current-period, previous-period, or cumulative"
                ),
            )
        )
    if (
        (source_label and _WEAK_COLUMN_LABEL_RE.match(source_label))
        or (metric_label and _WEAK_COLUMN_LABEL_RE.match(metric_label))
    ) and semantics_status != "resolved":
        blocks.append(
            (
                "quarantined_schema_invalid",
                "ambiguous_metric_column_semantics",
                "weak table column labels such as column_1/column_2 need resolved semantics",
            )
        )
    if _PREVIOUS_COLUMN_RE.search(labels_text) and metric_period_source not in {
        "filled_from_previous_column_label",
        "llm_extracted",
    }:
        blocks.append(
            (
                "quarantined_outside_scope",
                "previous_week_metric_uses_current_source_period",
                (
                    "previous/prior week metric columns cannot reuse the current "
                    "source reporting period"
                ),
            )
        )
    return blocks


def _blocked_observation_types_are_soft_for_direct_metric(
    record: dict,
    blocked_types: set[str],
) -> bool:
    return bool(blocked_types) and blocked_types <= _DIRECT_METRIC_SOFT_OBSERVATION_TYPES and _has_strong_direct_public_health_metric(record)


def _is_direct_collection_official_aggregate(
    record: dict,
    source: dict | None,
    observation_types: list[str],
    state: dict,
) -> bool:
    if not _direct_collection_enabled(state):
        return False
    observation_type_set = set(observation_types)
    blocked_types = observation_type_set & _TASK_AWARE_BLOCKED_OBSERVATION_TYPES
    if blocked_types and not _blocked_observation_types_are_soft_for_direct_metric(
        record,
        blocked_types,
    ):
        return False
    if not _source_is_official_or_high_trust(source, record):
        return False
    if _has_strong_direct_public_health_metric(record):
        return True
    return _has_direct_collection_count_semantics(record)


def _direct_source_trust_pending_review(
    record: dict,
    source: dict | None,
    state: dict,
) -> tuple[str, str] | None:
    if not _direct_collection_enabled(state):
        return None
    if not (
        _has_strong_direct_public_health_metric(record)
        or _has_direct_collection_count_semantics(record)
        or _has_task_aware_count(record)
    ):
        return None
    if not _source_requires_trust_review(source, record):
        if _source_is_official_or_high_trust(source, record):
            return None
        return None
    return (
        "source_trust_requires_human_review",
        (
            "record contains task-compatible public-health data, but the source "
            "is not trusted enough for strict final inclusion without human review"
        ),
    )


def _direct_collection_hard_blocks(
    record: dict,
    source: dict | None,
    observation_types: list[str],
    state: dict,
) -> list[tuple[str, str, str]]:
    if not _direct_collection_enabled(state):
        return []
    if not _source_is_official_or_high_trust(source, record):
        return []
    observation_type_set = set(observation_types)
    likely_direct_record = bool(
        observation_type_set & _TASK_AWARE_ACCEPTABLE_OBSERVATION_TYPES
    ) or _has_task_aware_count(record) or any(
        token in _reason_text(
            record.get("count_semantics"),
            record.get("statistical_count_type"),
            record.get("reporting_period"),
            record.get("source_title"),
            record.get("evidence_quote"),
        )
        for token in (
            "surveillance",
            "weekly",
            "laboratory",
            "positive",
            "hospitalization",
            "death",
            "ili",
            "outbreak",
        )
    )
    if not likely_direct_record:
        return []

    blocks: list[tuple[str, str, str]] = []
    if not _has_task_aware_count(record):
        blocks.append(
            (
                "quarantined_schema_invalid",
                "missing_direct_collection_metric",
                (
                    "direct_collection records must contain at least one "
                    "task-aware numeric metric or explicit zero count"
                ),
            )
        )
    if (
        str(record.get("chunk_kind") or "").lower() == "metric_row"
        and record.get("metric_value") not in (None, "")
        and str(record.get("metric_row_binding_status") or "").lower()
        in {"unresolved", "missing", "failed"}
    ):
        blocks.append(
            (
                "quarantined_schema_invalid",
                "metric_row_binding_unresolved",
                (
                    "direct_collection metric-row records must resolve to a "
                    "specific source row before entering final_dataset"
                ),
            )
        )
    blocks.extend(_direct_metric_column_semantics_blocks(record))

    task = _task_metadata(state)
    start = _parse_scope_date(task.get("task_start_date"))
    end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or start
    record_date = _direct_record_scope_date(record)
    if not record_date:
        blocks.append(
            (
                "quarantined_outside_scope",
                "missing_direct_collection_date_anchor",
                (
                    "direct_collection records must include a usable date "
                    "anchor, reporting period, or report date"
                ),
            )
        )
        return blocks

    if start and end:
        if end < start:
            start, end = end, start
        if not (start <= record_date <= end):
            period_overlap_status = _record_period_overlap_status(record, state)
            if period_overlap_status == "partial_overlap":
                record["period_overlap_status"] = "partial_overlap"
                warning_flags = _as_list(record.get("quality_gate_warning_flags"))
                if "metric_period_partially_overlaps_task_window" not in warning_flags:
                    warning_flags.append("metric_period_partially_overlaps_task_window")
                record["quality_gate_warning_flags"] = warning_flags
            elif period_overlap_status == "within_window":
                record["period_overlap_status"] = "within_window"
            else:
                blocks.append(
                    (
                        "quarantined_outside_scope",
                        "record_date_outside_task_window",
                        (
                            f"record date {record_date.isoformat()} is outside requested "
                            f"task window {start.isoformat()} to {end.isoformat()}"
                        ),
                    )
                )
    geography_block = _record_geography_scope_block(record, state)
    if geography_block:
        flag, reason = geography_block
        blocks.append(("quarantined_outside_scope", flag, reason))
    source_period_block = _source_period_scope_block(record, source, state)
    if source_period_block:
        flag, reason = source_period_block
        blocks.append(("quarantined_outside_scope", flag, reason))
    for flag, reason in _record_period_scope_blocks(record, state):
        blocks.append(("quarantined_outside_scope", flag, reason))
    return blocks


def _record_period_overlap_status(record: dict, state: dict) -> str | None:
    task = _task_metadata(state)
    task_start = _parse_scope_date(task.get("task_start_date"))
    task_end = _parse_scope_date(task.get("task_end_date"), end_of_period=True) or task_start
    period_start, period_end = _record_period_dates(record)
    if not task_start or not task_end or not period_start or not period_end:
        return None
    if task_end < task_start:
        task_start, task_end = task_end, task_start
    if period_end < period_start:
        period_start, period_end = period_end, period_start
    if period_end < task_start or period_start > task_end:
        return "outside_window"
    if period_start < task_start or period_end > task_end:
        return "partial_overlap"
    return "within_window"


def _record_period_semantics_block(record: dict, state: dict) -> tuple[str, str] | None:
    if not _direct_collection_enabled(state):
        return None
    if _looks_like_event_outbreak_snapshot_record(record):
        record["count_semantics"] = record.get("count_semantics") or "event_outbreak_snapshot"
        record["compatible_dynamic_outbreak_snapshot"] = True
        return None
    granularity = _task_time_granularity(state)
    resolved_type = _lower(record.get("resolved_column_period_type"))
    statistical_type = _lower(record.get("statistical_count_type"))
    semantics_text = _reason_text(
        record.get("resolved_column_period_type"),
        record.get("metric_column_semantics_status"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("metric_name"),
        record.get("metric_period_label"),
        record.get("metric_period_source"),
        record.get("metric_column_label"),
        record.get("source_column_label"),
        record.get("table_header"),
        record.get("reporting_period"),
        record.get("source_title"),
        record.get("source_url"),
        record.get("evidence_quote"),
    )
    non_exact_task_window_types = {
        "annual_period",
        "season_to_date",
        "cumulative_period",
        "previous_period",
        "period_aggregate",
        "partial_overlap",
    }
    non_exact_annual_types = {
        "current_period",
        "previous_period",
        "season_to_date",
        "cumulative_period",
        "period_aggregate",
        "partial_overlap",
    }
    if granularity in {"task_window", "weekly"}:
        if (
            resolved_type in non_exact_task_window_types
            or statistical_type in non_exact_task_window_types
            or any(
                token in semantics_text
                for token in (
                    "year-to-date",
                    "year to date",
                    "ytd",
                    "cumulative",
                    "ew1",
                    "ew 1",
                    "prior year",
                    "prior-year",
                    "previous year",
                    "last year",
                    "same point",
                    "same time last year",
                    "historical comparator",
                    "comparator",
                )
            )
        ):
            return (
                "record_period_semantics_not_exact_for_task_window",
                (
                    "task-window direct_collection final records require exact "
                    "current-period semantics; season-to-date, cumulative, "
                    "annual, previous-period, or broader EW-range records are "
                    "kept as best-available context or human review"
                ),
            )
    if granularity == "annual" or _task_requires_exact_annual_period(state):
        if (
            resolved_type in non_exact_annual_types
            or statistical_type in non_exact_annual_types
            or any(
                token in semantics_text
                for token in (
                    "week ",
                    "weekly",
                    "mmwr week",
                    "current week",
                    "previous week",
                    "season-to-date",
                    "season to date",
                    "year-to-date",
                    "year to date",
                    "ytd",
                )
            )
        ):
            return (
                "record_period_semantics_not_exact_for_annual_requirement",
                (
                    "annual direct_collection final records require exact "
                    "full-year annual semantics; weekly, current-period, "
                    "previous-period, season-to-date, YTD, or cumulative rows "
                    "are kept as best-available context"
                ),
            )
    return None


def _annual_exact_period_block(record: dict, state: dict) -> tuple[str, str] | None:
    if not (_direct_collection_enabled(state) and _task_requires_exact_annual_period(state)):
        return None
    if _looks_like_event_outbreak_snapshot_record(record):
        record["count_semantics"] = record.get("count_semantics") or "event_outbreak_snapshot"
        record["compatible_dynamic_outbreak_snapshot"] = True
        return None
    period_start, period_end = _record_period_dates(record)
    if not period_start or not period_end:
        return None
    if period_end < period_start:
        period_start, period_end = period_end, period_start
    if period_start.month == 1 and period_start.day == 1 and period_end.month == 12 and period_end.day == 31 and period_start.year == period_end.year:
        return None
    text = _reason_text(
        record.get("reporting_period"),
        record.get("metric_period_label"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("evidence_quote"),
    )
    partial_markers = (
        "as of",
        "year-to-date",
        "year to date",
        "ytd",
        "past month",
        "last month",
        "campaign",
        "season-to-date",
        "season to date",
        "partial",
        "through ",
    )
    if any(marker in text for marker in partial_markers) or period_start.year != period_end.year or period_end.month != 12 or period_end.day != 31:
        record["period_overlap_status"] = "partial_overlap"
        return (
            "record_period_partial_overlap_for_annual_requirement",
            (
                "annual direct_collection requirements require exact full-year "
                "records; as-of, year-to-date, campaign, season-to-date, or "
                "other partial-period records are kept as best-available context"
            ),
        )
    return None


def _validation_limited(state: dict) -> tuple[bool, bool, list[str]]:
    summary = _as_dict(state.get("validation_source_compatibility_summary"))
    status = _lower(summary.get("compatibility_status") or summary.get("status"))
    warnings = [str(w) for w in _as_list(summary.get("warnings"))]
    limited = status in VALIDATION_LIMITED_STATUSES or any(
        _lower(w) in VALIDATION_LIMITED_STATUSES for w in warnings
    )
    if limited:
        if "no_task_compatible_validation_source" not in warnings:
            warnings.append("no_task_compatible_validation_source")
    return limited, limited, warnings


def _document_candidates(
    record: dict,
    docs_by_id: dict[str, dict],
    docs_by_source: dict[str, list[dict]],
) -> list[dict]:
    doc_id = record.get("document_id")
    if doc_id and str(doc_id) in docs_by_id:
        return [docs_by_id[str(doc_id)]]
    return list(docs_by_source.get(_source_id(record), []))


def _chunk_candidates(
    record: dict,
    chunks_by_id: dict[str, dict],
    chunks_by_source: dict[str, list[dict]],
) -> list[dict]:
    chunk_id = _supporting_chunk_id(record)
    if chunk_id and chunk_id in chunks_by_id:
        return [chunks_by_id[chunk_id]]
    return list(chunks_by_source.get(_source_id(record), []))


def _source_machine_exclusion_can_be_rescued(
    source: dict | None,
    record: dict,
    state: dict,
) -> bool:
    if not source or not _direct_collection_enabled(state):
        return False
    if source.get("source_excluded_by_human_review"):
        return False
    if source.get("blocked_from_fetch") or source.get("llm_source_critic_block_fetch"):
        return False
    if _lower(source.get("status")) in _SOURCE_BLOCK_STATUSES:
        return False
    if _lower(source.get("source_disease_relevance_status")) in _SOURCE_UNRELATED_STATUSES:
        return False
    for field in (
        "screening_decision",
        "critic_decision",
        "final_screening_decision",
        "llm_source_critic_decision",
        "llm_source_critic_fetch_recommendation",
    ):
        if _lower(source.get(field)) in {"exclude", "excluded", "not_task_relevant", "block_fetch"}:
            return False
    role = _lower(source.get("source_role_final"))
    if role not in _SOURCE_EXCLUDED_ROLES:
        return False
    if not _source_has_supported_official_public_health_identity(source, record):
        return False
    return (
        _has_strong_direct_public_health_metric(record)
        or _has_direct_collection_count_semantics(record)
        or _has_task_aware_count(record)
    )


def _source_is_blocking(
    source: dict | None,
    record: dict | None = None,
    state: dict | None = None,
) -> tuple[str, str] | None:
    if not source:
        return None
    if source.get("blocked_from_fetch") or source.get("llm_source_critic_block_fetch"):
        return (
            "source_critic_blocked_from_fetch",
            source.get("blocked_from_fetch_reason")
            or source.get("llm_source_critic_reason")
            or "source critic blocked fetch",
        )
    role = _lower(source.get("source_role_final"))
    if role in _SOURCE_EXCLUDED_ROLES:
        if record is not None and state is not None and _source_machine_exclusion_can_be_rescued(source, record, state):
            return None
        return ("source_role_final_excluded", f"source_role_final={role}")
    status = _lower(source.get("status"))
    if status in _SOURCE_BLOCK_STATUSES:
        return ("source_status_excluded", f"source status={status}")
    for field in (
        "screening_decision",
        "critic_decision",
        "final_screening_decision",
        "llm_source_critic_decision",
        "llm_source_critic_fetch_recommendation",
    ):
        value = _lower(source.get(field))
        if value in {"exclude", "excluded", "not_task_relevant", "block_fetch"}:
            return (f"{field}_blocked", f"{field}={value}")
    if source.get("source_excluded_by_human_review"):
        return ("source_excluded_by_human_review", "source excluded by human review")
    if _lower(source.get("source_disease_relevance_status")) in _SOURCE_UNRELATED_STATUSES:
        return (
            "source_disease_relevance_status_unrelated",
            "source disease relevance is not task relevant",
        )
    return None


def _document_block(documents: list[dict]) -> tuple[str, str] | None:
    for doc in documents:
        status = _lower(doc.get("document_disease_relevance_status"))
        quality = _lower(doc.get("quality_status"))
        if status in _DOCUMENT_BLOCK_STATUSES:
            return (
                "document_disease_relevance_status_unrelated",
                "document is not task relevant",
            )
        if doc.get("not_extractable_for_task_disease") is True:
            return (
                "not_extractable_for_task_disease",
                "document is not extractable for the task disease",
            )
        if quality in _DOCUMENT_BLOCK_QUALITIES:
            return ("document_quality_not_task_relevant", "document is not task relevant")
    return None


def _chunk_block(chunks: list[dict]) -> tuple[str, str] | None:
    for chunk in chunks:
        status = _lower(chunk.get("disease_relevance_status"))
        if status in _CHUNK_BLOCK_STATUSES:
            return (
                "chunk_disease_relevance_status_unrelated",
                "evidence chunk is not task relevant",
            )
        if chunk.get("extraction_eligible_for_task_disease") is False:
            return (
                "chunk_not_extractable_for_task_disease",
                "evidence chunk is not extractable for the task disease",
            )
        if chunk.get("contains_target_data") is False:
            return ("chunk_contains_target_data_false", "evidence chunk is not task relevant")
    return None


def _looks_like_dynamic_outbreak_snapshot_conflict(record: dict, row: dict) -> bool:
    compared_field = _lower(row.get("compared_field"))
    if compared_field and not any(
        token in compared_field for token in ("case", "death", "hospital")
    ):
        return False
    record_text = _reason_text(
        record.get("observation_type"),
        " ".join(_record_observation_types(record)),
        record.get("geographic_scope"),
        record.get("geographic_scope_type"),
        record.get("reporting_period"),
        record.get("date_reported"),
        record.get("date_anchor"),
        record.get("evidence_quote"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
    )
    validation_text = _reason_text(
        row.get("reason"),
        row.get("date_or_period"),
        row.get("evidence_summary"),
        row.get("left_value"),
        row.get("right_value"),
    )
    outbreak_markers = (
        "outbreak",
        "cluster",
        "travel associated",
        "travel-associated",
        "cruise",
        "ship",
        "vessel",
        "multi country",
        "multi-country",
    )
    snapshot_markers = (
        "as of",
        "later",
        "update",
        "snapshot",
        "reported",
        "ongoing",
    )
    if not any(marker in record_text for marker in outbreak_markers):
        return False
    return any(marker in f"{record_text} {validation_text}" for marker in snapshot_markers)


def _looks_like_event_outbreak_snapshot_record(record: dict) -> bool:
    text = _reason_text(
        record.get("observation_type"),
        " ".join(_record_observation_types(record)),
        record.get("geographic_scope"),
        record.get("geographic_scope_type"),
        record.get("reporting_period"),
        record.get("date_reported"),
        record.get("date_anchor"),
        record.get("as_of_date"),
        record.get("count_semantics"),
        record.get("statistical_count_type"),
        record.get("source_title"),
        record.get("source_url"),
        record.get("evidence_quote"),
    )
    outbreak_markers = (
        "outbreak",
        "cluster",
        "travel associated",
        "travel-associated",
        "cruise",
        "ship",
        "vessel",
        "multi country",
        "multi-country",
    )
    snapshot_markers = (
        "as of",
        "update",
        "snapshot",
        "ongoing",
        "reported",
        "event_total",
        "event outbreak snapshot",
        "event_outbreak_snapshot",
    )
    if not any(marker in text for marker in outbreak_markers):
        return False
    if not any(marker in text for marker in snapshot_markers):
        return False
    return any(token in text for token in ("case", "death", "hospital", "confirmed"))


def _validation_block(record: dict, rows: list[dict]) -> tuple[str, str, str] | None:
    for row in rows:
        text = _reason_text(
            row.get("validation_status"),
            row.get("match_status"),
            row.get("comparability_status"),
            row.get("reason"),
            " ".join(str(w) for w in _as_list(row.get("warnings"))),
        )
        if any(token in text for token in _OUTSIDE_SCOPE_TOKENS):
            return (
                "quarantined_outside_scope",
                "validation_outside_scope",
                row.get("reason") or "validation marked record outside requested scope",
            )
        validation_type = _lower(row.get("validation_type"))
        trusted_conflict = validation_type in {
            "trusted_source_comparison",
            "held_out_source_comparison",
            "aggregate_comparison",
        }
        if trusted_conflict and (
            _lower(row.get("validation_status")) == "conflict"
            or _lower(row.get("match_status")) == "conflict"
        ):
            if _looks_like_dynamic_outbreak_snapshot_conflict(record, row):
                record["compatible_dynamic_outbreak_snapshot"] = True
                return None
            return (
                "quarantined_validation_conflict",
                "validation_conflict",
                row.get("reason") or "validation result is a conflict",
            )
    return None


def _anomaly_block(rows: list[dict]) -> tuple[str, str] | None:
    for row in rows:
        if _lower(row.get("anomaly_status")) in {"resolved", "dismissed"}:
            continue
        severity = _lower(row.get("severity"))
        anomaly_type = str(row.get("anomaly_type") or "high_or_critical_anomaly")
        if severity in _ANOMALY_BLOCK_SEVERITIES:
            return (
                anomaly_type,
                row.get("reason") or f"{severity} anomaly requires blocking review",
            )
    return None


def _direct_source_aware_anomaly_pending_review(
    record: dict,
    source: dict | None,
    rows: list[dict],
    state: dict,
) -> tuple[str, str] | None:
    if not _direct_collection_enabled(state):
        return None
    if not _source_is_official_or_high_trust(source, record):
        return None
    if not (
        _has_strong_direct_public_health_metric(record)
        or _has_direct_collection_count_semantics(record)
        or _has_task_aware_count(record)
    ):
        return None
    for row in rows:
        if _lower(row.get("anomaly_status")) in {"resolved", "dismissed"}:
            continue
        severity = _lower(row.get("severity"))
        if severity not in _ANOMALY_BLOCK_SEVERITIES:
            continue
        anomaly_type = _lower(row.get("anomaly_type"))
        reason_text = _reason_text(row.get("reason"), anomaly_type)
        if "simple_threshold" in anomaly_type or "simple anomaly threshold" in reason_text:
            return (
                "source_aware_anomaly_requires_human_review",
                (
                    "high-trust aggregate public-health metric exceeded a generic "
                    "anomaly threshold; route to human review instead of automatic "
                    "strict-final rejection"
                ),
            )
    return None


def _direct_source_aware_simple_anomaly_acceptance_warning(
    record: dict,
    source: dict | None,
    rows: list[dict],
    state: dict,
) -> tuple[str, str] | None:
    if not _direct_collection_enabled(state):
        return None
    if _source_requires_trust_review(source, record):
        return None
    if not _source_is_official_or_high_trust(source, record):
        return None
    if not (
        _has_strong_direct_public_health_metric(record)
        or _has_direct_collection_count_semantics(record)
        or _has_task_aware_count(record)
    ):
        return None
    if _record_geography_scope_block(record, state):
        return None
    if _source_period_scope_block(record, source, state):
        return None
    if _record_period_scope_blocks(record, state):
        return None
    for row in rows:
        if _lower(row.get("anomaly_status")) in {"resolved", "dismissed"}:
            continue
        severity = _lower(row.get("severity"))
        if severity not in _ANOMALY_BLOCK_SEVERITIES:
            continue
        anomaly_type = _lower(row.get("anomaly_type"))
        reason_text = _reason_text(row.get("reason"), anomaly_type)
        if "simple_threshold" in anomaly_type or "simple anomaly threshold" in reason_text:
            return (
                "source_aware_simple_anomaly_accepted_for_high_trust_source",
                (
                    "high-trust exact public-health aggregate exceeded a generic "
                    "simple anomaly threshold; retain as strict final with audit "
                    "warning instead of treating the threshold as semantic evidence"
                ),
            )
    return None


def _direct_ambiguous_metric_semantics_pending_review(
    record: dict,
    source: dict | None,
    state: dict,
    blocks: list[tuple[str, str, str]],
) -> tuple[str, str] | None:
    if not _direct_collection_enabled(state):
        return None
    if not any(flag == "ambiguous_metric_column_semantics" for _, flag, _ in blocks):
        return None
    if _source_requires_trust_review(source, record):
        return None
    if not _source_is_official_or_high_trust(source, record):
        return None
    if not (
        _has_strong_direct_public_health_metric(record)
        or _has_direct_collection_count_semantics(record)
        or _has_task_aware_count(record)
    ):
        return None
    return (
        "ambiguous_metric_column_semantics_requires_human_review",
        (
            "official or high-trust task-compatible metric has unresolved column "
            "semantics; route to human review instead of strict final inclusion"
        ),
    )


def _pending_review_block(
    record: dict,
    review_items: list[dict],
    *,
    human_review_blocks: bool = True,
) -> tuple[str, str] | None:
    if record.get("requires_human_review"):
        review_reason = record.get("human_review_reason") or "record requires human review"
        official_single_source = (
            _lower(record.get("quality_status"))
            == "official_single_source_not_cross_validated"
            or _lower(record.get("human_review_reason"))
            == "official_single_source_not_cross_validated"
            or (
                _lower(record.get("record_schema"))
                == "official_outbreak_page_extraction_schema"
                and _lower(record.get("corroboration_reason"))
                == "official_single_source_not_cross_validated"
            )
        )
        if official_single_source:
            return (
                "official_single_source_not_cross_validated",
                (
                    "official single-source outbreak candidate requires human "
                    "review before final dataset inclusion"
                ),
            )
        if not human_review_blocks:
            return None
        return ("requires_human_review", review_reason)
    if record.get("countable") is False and record.get("duplicate_of_record_id"):
        return ("non_countable_duplicate", "record is marked as a non-countable duplicate")
    for item in review_items:
        if _lower(item.get("status") or "pending") == "pending":
            severity = _lower(item.get("severity"))
            if severity in {"high", "critical"}:
                return (
                    str(item.get("item_type") or "pending_human_review"),
                    item.get("reason") or "unresolved human review item targets record",
                )
    return None


def _review_acceptance_status(record: dict, default_status: str) -> str:
    if record.get("human_review_applied"):
        review_status = _lower(record.get("review_status"))
        if review_status == "corrected":
            return "corrected_after_human_review"
        if review_status in {"accepted", "accepted_as_is", "resolved"}:
            return "accepted_after_human_review"
    return default_status


def _evaluate_record(record: dict, state: dict, indexes: dict) -> tuple[dict, dict]:
    record = _coerce_non_case_public_health_metric(record)
    enriched = deepcopy(record)
    rid = _record_id(record)
    blocks: list[tuple[str, str, str]] = []
    warnings: list[str] = []
    source = indexes["sources_by_id"].get(_source_id(record))

    context = indexes["disease_context"]
    disease_assessment = dict(assess_record_disease_compatibility(record, context))
    local_disease_mismatch = any(
        _lower(value) in {"incompatible_disease", "unrelated_disease", "disease_mismatch"}
        for value in (
            disease_assessment.get("record_local_disease_relevance_status"),
            disease_assessment.get("record_local_numeric_disease_status"),
            record.get("record_local_disease_relevance_status"),
            record.get("record_local_numeric_disease_status"),
        )
    ) or "local_evidence_disease_mismatch" in _reason_text(
        disease_assessment.get("reason"),
        record.get("record_disease_compatibility_reason"),
    )
    if not local_disease_mismatch and _structured_metric_has_bound_disease_context(
        record,
        source,
        state,
        disease_assessment,
    ):
        disease_assessment.update(
            {
                "reject_record": False,
                "status": "compatible_structured_metric_context",
                "reason": (
                    "target disease identity is supplied by a verified official "
                    "metric-row and task-requirement binding"
                ),
                "record_local_disease_relevance_status": (
                    "compatible_structured_metric_context"
                ),
                "record_local_numeric_disease_status": (
                    "compatible_structured_metric_context"
                ),
                "record_local_numeric_disease_decision": (
                    "accepted_bound_structured_metric_context"
                ),
            }
        )
    if disease_assessment.get("reject_record") or record.get(
        "record_disease_compatibility_reject"
    ):
        local_mismatch = (
            "local_evidence_disease_mismatch"
            in str(disease_assessment.get("reason") or "")
        )
        _add_block(
            blocks,
            "quarantined_source_not_task_relevant"
            if local_mismatch
            else "quarantined_disease_mismatch",
            "local_evidence_disease_mismatch"
            if local_mismatch
            else "disease_pathogen_incompatible_with_task",
            disease_assessment.get("reason")
            or record.get("record_disease_compatibility_reason")
            or "record disease/pathogen is incompatible with task disease",
        )
    elif _lower(record.get("record_disease_compatibility_status")) in {
        "incompatible_disease",
        "unrelated_disease",
        "disease_mismatch",
    }:
        _add_block(
            blocks,
            "quarantined_disease_mismatch",
            "disease_mismatch",
            record.get("record_disease_compatibility_reason")
            or "record disease compatibility status is blocking",
        )

    if _default_seasonal_flu_task(state) and _record_has_nonseasonal_influenza_subtype(record):
        _add_block(
            blocks,
            "quarantined_disease_mismatch",
            "non_seasonal_influenza_subtype",
            (
                "Default FLU tasks target seasonal influenza surveillance; "
                "H5/H5N1/avian influenza records are excluded unless the user explicitly requests that subtype."
            ),
        )

    annual_exact_block = _annual_exact_period_block(record, state)
    if annual_exact_block:
        flag, reason = annual_exact_block
        _add_block(
            blocks,
            "quarantined_outside_scope",
            flag,
            reason,
        )

    period_semantics_block = _record_period_semantics_block(record, state)
    if period_semantics_block:
        flag, reason = period_semantics_block
        _add_block(
            blocks,
            "quarantined_outside_scope",
            flag,
            reason,
        )

    as_of_scope_block = _record_as_of_date_scope_block(record, state)
    if as_of_scope_block:
        flag, reason = as_of_scope_block
        _add_block(
            blocks,
            "quarantined_outside_scope",
            flag,
            reason,
        )

    date_scope_block = _record_date_scope_block(record, state)
    if date_scope_block and not as_of_scope_block:
        period_overlap_status = _record_period_overlap_status(record, state)
        if (
            _direct_collection_enabled(state)
            and period_overlap_status == "partial_overlap"
            and _task_requires_exact_annual_period(state)
        ):
            record["period_overlap_status"] = "partial_overlap"
            _add_block(
                blocks,
                "quarantined_outside_scope",
                "record_period_partial_overlap_for_annual_requirement",
                (
                    "annual direct_collection requirements require exact annual "
                    "period records; partial-overlap campaign or intervention "
                    "periods are kept as best-available context"
                ),
            )
            date_scope_block = None
        if date_scope_block is None:
            pass
        elif (
            _direct_collection_enabled(state)
            and _has_public_health_metric(record)
            and period_overlap_status in {"partial_overlap", "within_window"}
        ):
            record["period_overlap_status"] = period_overlap_status
            if period_overlap_status == "partial_overlap":
                warning_flags = _as_list(record.get("quality_gate_warning_flags"))
                if "metric_period_partially_overlaps_task_window" not in warning_flags:
                    warning_flags.append("metric_period_partially_overlaps_task_window")
                record["quality_gate_warning_flags"] = warning_flags
        else:
            flag, reason = date_scope_block
            _add_block(
                blocks,
                "quarantined_outside_scope",
                flag,
                reason,
            )

    if rid in indexes["human_review_rejected_ids"] or record.get(
        "record_excluded_by_human_review"
    ):
        _add_block(
            blocks,
            "excluded_by_human_review",
            "explicit_human_review_reject",
            "explicit human review rejected this record",
        )

    schema_status = _lower(record.get("schema_status"))
    if schema_status in _SCHEMA_BLOCK_STATUSES:
        _add_block(
            blocks,
            "quarantined_schema_invalid",
            f"schema_status_{schema_status}",
            f"schema_status={schema_status}",
        )
    validation_errors = " ".join(str(v) for v in _as_list(record.get("validation_errors")))
    if "disease_mismatch" in validation_errors:
        _add_block(
            blocks,
            "quarantined_disease_mismatch",
            "disease_mismatch",
            "schema validation reported disease_mismatch",
        )

    normalization_status = _lower(record.get("normalization_status"))
    if any(token in normalization_status for token in ("rejected", "quarantined")):
        _add_block(
            blocks,
            "quarantined_normalization_rejected",
            "normalization_rejected",
            f"normalization_status={normalization_status}",
        )
    if "disease_mismatch" in normalization_status:
        _add_block(
            blocks,
            "quarantined_disease_mismatch",
            "disease_mismatch",
            f"normalization_status={normalization_status}",
        )
    if _lower(record.get("provenance_status")) == "failed":
        _add_block(
            blocks,
            "quarantined_schema_invalid",
            "provenance_failed",
            "provenance_status=failed",
        )

    source_block = _source_is_blocking(source, record, state)
    if source_block:
        flag, reason = source_block
        _add_block(
            blocks,
            "quarantined_source_not_task_relevant",
            flag,
            reason,
        )

    record_documents = _document_candidates(
        record, indexes["documents_by_id"], indexes["documents_by_source"]
    )
    record_chunks = _chunk_candidates(
        record, indexes["chunks_by_id"], indexes["chunks_by_source"]
    )
    document_block = _document_block(record_documents)
    if document_block:
        flag, reason = document_block
        _add_block(
            blocks,
            "quarantined_document_not_task_relevant",
            flag,
            reason,
        )

    chunk_block = _chunk_block(record_chunks)
    if chunk_block:
        flag, reason = chunk_block
        _add_block(
            blocks,
            "quarantined_chunk_not_task_relevant",
            flag,
            reason,
        )

    observation_types = _record_observation_types(record)
    source_trust_pending_block = _direct_source_trust_pending_review(
        record,
        source,
        state,
    )
    for warning in _source_identity_unverified_flags(source, record):
        _add_warning(warnings, warning)
    for status, flag, reason in _direct_record_task_fit_blocks(
        record,
        source,
        record_documents,
        record_chunks,
        state,
    ):
        _add_block(blocks, status, flag, reason)
    for status, flag, reason in _direct_collection_hard_blocks(
        record,
        source,
        observation_types,
        state,
    ):
        _add_block(blocks, status, flag, reason)
    ambiguous_semantics_pending_block = _direct_ambiguous_metric_semantics_pending_review(
        record,
        source,
        state,
        blocks,
    )
    if ambiguous_semantics_pending_block is not None:
        blocks = [
            block
            for block in blocks
            if block[1] != "ambiguous_metric_column_semantics"
        ]
    for key in (
        "metric_column_semantics_status",
        "resolved_column_period_type",
        "column_period_resolution_reason",
        "column_semantics_resolution_method",
        "column_semantics_confidence",
        "column_period_warning_flags",
        "semantic_warnings",
        "count_semantics",
        "statistical_count_type",
        "metric_period_label",
        "metric_period_source",
        "record_task_fit_status",
        "record_geography_source",
        "record_period_source",
        "record_task_fit_reasons",
        "record_geography_fit_status",
        "record_period_fit_status",
    ):
        if key in record:
            enriched[key] = record.get(key)
    if record.get("period_overlap_status"):
        enriched["period_overlap_status"] = record.get("period_overlap_status")
    for warning in _as_list(record.get("quality_gate_warning_flags")):
        _add_warning(warnings, warning)
    direct_collection_observation = _is_direct_collection_official_aggregate(
        record,
        source,
        observation_types,
        state,
    )
    if direct_collection_observation and not observation_types:
        observation_types = ["surveillance_summary"]
    if observation_types:
        enriched["observation_types"] = observation_types
    task_aware_observation = _is_task_aware_accepted_observation(
        record,
        source,
        observation_types,
    ) or direct_collection_observation
    validation_block = _validation_block(
        record,
        indexes["validation_by_record"].get(rid, []),
    )
    if record.get("compatible_dynamic_outbreak_snapshot"):
        enriched["compatible_dynamic_outbreak_snapshot"] = True
        _add_warning(warnings, "compatible_dynamic_outbreak_snapshot")
    if validation_block:
        status, flag, reason = validation_block
        if direct_collection_observation:
            _add_warning(warnings, f"{flag}_audit_only")
        else:
            _add_block(blocks, status, flag, reason)
    if task_aware_observation:
        enriched["dataset_view"] = _task_aware_dataset_view(observation_types)
        _add_warning(warnings, "accepted_task_aware_non_primary_observation")
        if direct_collection_observation:
            _add_warning(warnings, "accepted_direct_collection_official_aggregate")
    if (
        _is_non_primary_observation(record)
        and not _explicit_review_accepts_record(record)
        and not task_aware_observation
        and source_trust_pending_block is None
    ):
        non_primary_status = _specific_non_primary_status(observation_types)
        first_observation_type = (
            observation_types[0] if observation_types else "non_primary_observation"
        )
        if record.get("primary_case_dataset_eligible") is False:
            _add_block(
                blocks,
                non_primary_status,
                "primary_case_dataset_eligible_false",
                "claim-level corroboration marked this record as not eligible for the primary case dataset",
            )
        if _has_exposure_monitoring_language(
            record
        ) and not _has_positive_explicit_primary_count(record):
            _add_block(
                blocks,
                "quarantined_exposure_monitoring",
                "exposure_monitoring_language",
                (
                    "evidence describes exposure monitoring or a healthy monitored "
                    "traveler, not an accepted case dataset record"
                ),
            )
        _add_block(
            blocks,
            non_primary_status,
            "not_primary_case_record",
            "record is a non-primary public-health observation, not an accepted case dataset record",
        )
        _add_block(
            blocks,
            non_primary_status,
            f"claim_observation_type_{first_observation_type}",
            (
                f"claim-level corroboration classified record as {first_observation_type}, "
                "not a primary confirmed case dataset record"
            ),
        )
        corroboration_summary = _as_dict(state.get("corroboration_summary"))
        if (
            corroboration_summary.get("claim_count", 0)
            and not corroboration_summary.get("corroborated_primary_case_event_count", 0)
        ):
            _add_block(
                blocks,
                non_primary_status,
                "no_corroborated_primary_case_event",
                "no corroborated primary case event supports this record",
            )

    anomaly_rows = indexes["anomalies_by_record"].get(rid, [])
    anomaly_acceptance_warning = _direct_source_aware_simple_anomaly_acceptance_warning(
        record,
        source,
        anomaly_rows,
        state,
    )
    if anomaly_acceptance_warning is not None:
        _add_warning(warnings, anomaly_acceptance_warning[0])
    anomaly_pending_block = (
        None
        if anomaly_acceptance_warning is not None
        else _direct_source_aware_anomaly_pending_review(
            record,
            source,
            anomaly_rows,
            state,
        )
    )
    anomaly_block = (
        None
        if (anomaly_pending_block is not None or anomaly_acceptance_warning is not None)
        else _anomaly_block(anomaly_rows)
    )
    if anomaly_block:
        flag, reason = anomaly_block
        _add_block(
            blocks,
            "quarantined_critical_anomaly",
            flag,
            reason,
        )

    pending_block = _pending_review_block(
        record,
        indexes["review_items_by_record"].get(rid, []),
        human_review_blocks=_human_review_blocks_quality_gate(state),
    )
    if source_trust_pending_block is not None:
        pending_block = source_trust_pending_block
    elif anomaly_pending_block is not None:
        pending_block = anomaly_pending_block
    elif ambiguous_semantics_pending_block is not None:
        pending_block = ambiguous_semantics_pending_block
    if pending_block is None:
        source_identity_pending_block = _source_identity_pending_block(record, source)
        if source_identity_pending_block is not None:
            pending_block = source_identity_pending_block
    if (
        record.get("requires_human_review")
        and not _human_review_blocks_quality_gate(state)
        and pending_block is None
    ):
        _add_warning(warnings, "accepted_with_review_warning")
    if pending_block is None:
        review_warning_pending_block = _accepted_review_warning_pending_block(
            record, source, warnings
        )
        if review_warning_pending_block is not None:
            pending_block = review_warning_pending_block

    validation_limited, _, validation_warnings = _validation_limited(state)
    if validation_limited:
        _add_warning(warnings, "no_task_compatible_validation_source")
    for warning in validation_warnings:
        _add_warning(warnings, warning)

    block = _first_block(blocks)
    if source_trust_pending_block is not None and block:
        hard_statuses = {
            "quarantined_disease_mismatch",
            "quarantined_source_not_task_relevant",
            "quarantined_schema_invalid",
            "quarantined_normalization_rejected",
            "excluded_by_human_review",
        }
        if block[0] not in hard_statuses:
            block = None
    if block:
        status, _, reason = block
        final_dataset_included = False
        quarantine_reason = reason
    elif pending_block:
        status = "pending_human_review"
        final_dataset_included = False
        quarantine_reason = pending_block[1]
        _add_warning(warnings, pending_block[0])
        enriched["requires_human_review"] = True
        enriched["human_review_reason"] = pending_block[1]
        enriched["review_status"] = enriched.get("review_status") or "pending"
    else:
        default_status = "accepted_with_warnings" if warnings else "accepted"
        if "accepted_with_review_warning" in warnings:
            default_status = "accepted_with_review_warning"
        status = _review_acceptance_status(record, default_status)
        final_dataset_included = True
        quarantine_reason = None

    blocking_flags = [flag for _, flag, _ in blocks]
    reasons = [reason for _, _, reason in blocks]
    if pending_block and not block:
        blocking_flags.append(pending_block[0])
        reasons.append(pending_block[1])
    if not reasons and final_dataset_included:
        reasons.append("record passed deterministic run quality gates")

    enriched.update(
        {
            "final_dataset_included": final_dataset_included,
            "record_final_inclusion_status": status,
            "quality_gate_reasons": reasons,
            "quality_gate_blocking_flags": blocking_flags,
            "quarantine_reason": quarantine_reason,
            "quality_gate_method": QUALITY_GATE_METHOD,
            "quality_gate_warnings": warnings,
            "quality_gate_warning_flags": warnings,
            "period_overlap_status": record.get("period_overlap_status")
            or enriched.get("period_overlap_status"),
            "record_disease_compatibility_status": disease_assessment.get("status")
            or record.get("record_disease_compatibility_status"),
            "record_disease_compatibility_reason": disease_assessment.get("reason")
            or record.get("record_disease_compatibility_reason"),
            "record_target_disease_terms_found": list(
                disease_assessment.get("target_disease_terms_found") or []
            ),
            "record_incompatible_disease_terms_found": list(
                disease_assessment.get("incompatible_disease_terms_found") or []
            ),
            "record_disease_compatibility_reject": bool(
                disease_assessment.get("reject_record")
                or record.get("record_disease_compatibility_reject")
            ),
            "record_local_disease_relevance_status": (
                disease_assessment.get("record_local_disease_relevance_status")
                or record.get("record_local_disease_relevance_status")
            ),
            "record_local_target_terms_found": list(
                disease_assessment.get("record_local_target_terms_found")
                or record.get("record_local_target_terms_found")
                or []
            ),
            "record_local_incompatible_terms_found": list(
                disease_assessment.get("record_local_incompatible_terms_found")
                or record.get("record_local_incompatible_terms_found")
                or []
            ),
            "record_local_numeric_disease_status": (
                disease_assessment.get("record_local_numeric_disease_status")
                or record.get("record_local_numeric_disease_status")
            ),
            "record_local_numeric_sentence": (
                disease_assessment.get("record_local_numeric_sentence")
                or record.get("record_local_numeric_sentence")
            ),
            "record_local_numeric_disease_decision": (
                disease_assessment.get("record_local_numeric_disease_decision")
                or record.get("record_local_numeric_disease_decision")
            ),
            "record_local_numeric_target_terms_found": list(
                disease_assessment.get("record_local_numeric_target_terms_found")
                or record.get("record_local_numeric_target_terms_found")
                or []
            ),
            "record_local_numeric_incompatible_terms_found": list(
                disease_assessment.get(
                    "record_local_numeric_incompatible_terms_found"
                )
                or record.get("record_local_numeric_incompatible_terms_found")
                or []
            ),
        }
    )

    decision = {
        "record_id": rid,
        "source_id": _source_id(record) or None,
        "source_url": _source_url(record),
        "supporting_chunk_id": _supporting_chunk_id(record) or None,
        "evidence_quote": record.get("evidence_quote"),
        "final_dataset_included": final_dataset_included,
        "record_final_inclusion_status": status,
        "quality_gate_reasons": reasons,
        "quality_gate_blocking_flags": blocking_flags,
        "quarantine_reason": quarantine_reason,
        "quality_gate_method": QUALITY_GATE_METHOD,
        "quality_gate_warnings": warnings,
    }
    return enriched, decision


def _build_indexes(state: dict) -> dict:
    documents = [row for row in _as_list(state.get("documents")) if isinstance(row, dict)]
    chunks = [row for row in _as_list(state.get("evidence_chunks")) if isinstance(row, dict)]
    return {
        "disease_context": build_disease_relevance_context(state),
        "sources_by_id": _index_by_id(_as_list(state.get("source_registry")), "source_id"),
        "documents_by_id": _index_by_id(documents, "document_id"),
        "documents_by_source": _documents_by_source(documents),
        "chunks_by_id": _index_by_id(chunks, "chunk_id"),
        "chunks_by_source": _chunks_by_source(chunks),
        "validation_by_record": _validation_results_by_record(
            _as_list(state.get("validation_results"))
        ),
        "anomalies_by_record": _anomalies_by_record(
            _as_list(state.get("anomaly_results"))
        ),
        "review_items_by_record": _review_items_by_record(
            _as_list(state.get("human_review_queue"))
        ),
        "human_review_rejected_ids": _human_review_rejected_record_ids(state),
    }


def _split_evaluated(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    accepted: list[dict] = []
    quarantined: list[dict] = []
    pending: list[dict] = []
    for record in records:
        status = record.get("record_final_inclusion_status")
        if record.get("final_dataset_included") and status in ACCEPTED_STATUSES:
            accepted.append(record)
        elif status == "pending_human_review":
            pending.append(record)
        else:
            quarantined.append(record)
    return accepted, quarantined, pending


def _collection_decision_summary(
    *,
    state: dict,
    final_dataset: list[dict],
    quarantined_records: list[dict],
    pending_review_records: list[dict],
    decisions: list[dict],
) -> dict:
    """Summarize final collection decisions without re-running quality gates."""

    status_counter = Counter(
        str(decision.get("record_final_inclusion_status") or "unknown")
        for decision in decisions
    )
    quarantine_counter = Counter(
        str(
            decision.get("quarantine_reason")
            or "; ".join(decision.get("quality_gate_reasons") or [])
            or "unspecified"
        )
        for decision in decisions
        if not decision.get("final_dataset_included")
    )
    warning_counter = Counter(
        str(warning)
        for decision in decisions
        for warning in (decision.get("quality_gate_warnings") or [])
    )
    return {
        "collection_mode": _collection_mode(state),
        "direct_collection_enabled": _direct_collection_enabled(state),
        "final_dataset_count": len(final_dataset),
        "quarantined_record_count": len(quarantined_records),
        "pending_review_record_count": len(pending_review_records),
        "accepted_record_ids": [
            str(row.get("record_id"))
            for row in final_dataset
            if row.get("record_id")
        ],
        "quarantined_record_ids": [
            str(row.get("record_id"))
            for row in quarantined_records
            if row.get("record_id")
        ],
        "pending_review_record_ids": [
            str(row.get("record_id"))
            for row in pending_review_records
            if row.get("record_id")
        ],
        "record_final_inclusion_status_counts": dict(status_counter),
        "quarantine_reason_counts": dict(quarantine_counter),
        "warning_counts": dict(warning_counter),
        "decision_count": len(decisions),
        "summary_method": QUALITY_GATE_METHOD,
    }


def _evaluate_records(records: list[dict], state: dict, indexes: dict) -> tuple[list[dict], list[dict]]:
    evaluated: list[dict] = []
    decisions: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        enriched, decision = _evaluate_record(record, state, indexes)
        evaluated.append(enriched)
        decisions.append(decision)
    return evaluated, decisions


def _non_primary_observations(records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if not record.get("final_dataset_included")
        and _is_non_primary_observation(record)
    ]


def _post_review_records(
    state: dict,
    accepted_records: list[dict],
    indexes: dict,
) -> tuple[list[dict], list[dict]]:
    if not _has_explicit_human_review_decisions(state):
        return deepcopy(accepted_records), []
    source_rows = [
        row
        for row in _as_list(state.get("final_dataset_post_review"))
        if isinstance(row, dict)
    ]
    if not source_rows:
        return [], []
    evaluated, decisions = _evaluate_records(source_rows, state, indexes)
    accepted, _, _ = _split_evaluated(evaluated)
    return accepted, decisions


def _no_task_relevant_signal(state: dict) -> bool:
    summary = _as_dict(state.get("disease_relevance_summary"))
    if summary.get("target_data_chunk_count") == 0:
        return True
    for key in (
        "chunk_status_counts",
        "document_status_counts",
        "source_status_counts",
    ):
        counts = _as_dict(summary.get(key))
        if counts and counts.get("target_disease_match", 0) == 0:
            return True
    data_presence = _as_dict(state.get("data_presence_summary"))
    if data_presence and data_presence.get("target_data_chunk_count", 0) == 0:
        return True
    return False


def _build_quality_summary(
    *,
    state: dict,
    pre_quality_records: list[dict],
    final_dataset: list[dict],
    non_primary_observations: list[dict],
    quarantined_records: list[dict],
    pending_review_records: list[dict],
    final_dataset_post_review: list[dict],
    records_excluded_by_review: list[dict],
    decisions: list[dict],
) -> tuple[dict, dict]:
    blocking_counter = Counter()
    warning_counter = Counter()
    status_counter = Counter()
    for decision in decisions:
        status_counter[str(decision.get("record_final_inclusion_status") or "unknown")] += 1
        for flag in _as_list(decision.get("quality_gate_blocking_flags")):
            blocking_counter[str(flag)] += 1
        for warning in _as_list(decision.get("quality_gate_warnings")):
            warning_counter[str(warning)] += 1

    validation_limited, no_compatible_validation, validation_warnings = _validation_limited(state)
    accepted_count = len(final_dataset)
    primary_case_dataset_eligible_count = sum(
        1 for record in final_dataset if record.get("primary_case_dataset_eligible") is True
    )
    accepted_primary_case_record_count = primary_case_dataset_eligible_count
    accepted_non_primary_count = sum(
        1 for record in final_dataset if record.get("primary_case_dataset_eligible") is False
    )
    surveillance_summary_record_count = sum(
        1
        for record in final_dataset
        if "surveillance_summary" in _record_observation_types(record)
    )
    task_aware_non_primary_record_count = sum(
        1
        for record in final_dataset
        if record.get("primary_case_dataset_eligible") is False
        and str(record.get("dataset_view") or "").startswith("task_aware_")
    )
    accepted_records_are_not_primary_case_records = bool(
        accepted_count > 0
        and accepted_non_primary_count == accepted_count
        and task_aware_non_primary_record_count == 0
    )
    non_primary_observation_count = len(non_primary_observations)
    corroboration_summary = _as_dict(state.get("corroboration_summary"))
    claim_count = int(corroboration_summary.get("claim_count") or 0)
    corroborated_primary_case_event_count = int(
        corroboration_summary.get("corroborated_primary_case_event_count") or 0
    )
    no_corroborated_primary_case_events = bool(
        claim_count and corroborated_primary_case_event_count == 0
    )
    no_primary_case_dataset_records = bool(
        normalized_count := len(pre_quality_records)
    ) and accepted_primary_case_record_count == 0 and non_primary_observation_count > 0
    if accepted_primary_case_record_count > 0:
        primary_case_dataset_status = "primary_case_records_present"
    elif no_primary_case_dataset_records:
        primary_case_dataset_status = "no_primary_case_dataset_records"
    elif no_corroborated_primary_case_events:
        primary_case_dataset_status = "no_corroborated_primary_case_events"
    elif claim_count == 0:
        primary_case_dataset_status = "unknown_no_claim_outputs"
    else:
        primary_case_dataset_status = "unknown_no_claim_outputs"
    quarantined_count = len(quarantined_records)
    pending_count = len(pending_review_records)
    human_review_required = pending_count > 0 or any(
        row.get("requires_human_review") for row in pre_quality_records
    )

    if normalized_count == 0:
        run_quality_status = (
            "no_task_relevant_records"
            if _no_task_relevant_signal(state)
            else "no_records_extracted"
        )
    elif accepted_records_are_not_primary_case_records:
        run_quality_status = "failed_quality_gate"
    elif accepted_count == 0 and non_primary_observation_count > 0:
        run_quality_status = "no_primary_case_dataset_records"
    elif accepted_count > 0 and quarantined_count > 0:
        run_quality_status = "partial_with_quarantined_records"
    elif accepted_count > 0 and human_review_required:
        run_quality_status = "passed_with_review"
    elif accepted_count > 0 and validation_limited:
        run_quality_status = "passed_with_review"
    elif accepted_count > 0:
        run_quality_status = "passed"
    elif pending_count > 0:
        run_quality_status = "human_review_required"
    elif quarantined_count > 0:
        run_quality_status = "failed_quality_gate"
    elif validation_limited:
        run_quality_status = "validation_limited_no_compatible_source"
    else:
        run_quality_status = "failed_quality_gate"

    if accepted_count > 0:
        acceptance_reason = "quality-gated accepted records are available"
        recommended = "Review final_dataset and warnings before use."
        if accepted_records_are_not_primary_case_records:
            acceptance_reason = "accepted records are not primary case dataset records"
            recommended = (
                "Do not use final_dataset as an epidemiological case dataset; "
                "inspect non_primary_observations and record_inclusion_decisions."
            )
    elif normalized_count == 0:
        acceptance_reason = "no normalized records were extracted"
        recommended = "No reliable task-relevant records were accepted; inspect search, fetch, extraction, and quarantine diagnostics."
    elif non_primary_observation_count > 0:
        acceptance_reason = "no primary case dataset records were accepted"
        recommended = (
            "Workflow completed, but no primary case records were accepted; "
            "non-primary observations were preserved separately."
        )
    elif quarantined_count:
        acceptance_reason = "all candidate records failed deterministic quality gates"
        recommended = "Use quarantined_records and record_inclusion_decisions to inspect why no records were accepted."
    else:
        acceptance_reason = "records remain pending human review"
        recommended = "Resolve pending human review items before using post-review data."

    warnings = list(validation_warnings)
    if validation_limited and "validation_limited_no_compatible_source" not in warnings:
        warnings.append("validation_limited_no_compatible_source")
    if accepted_records_are_not_primary_case_records:
        _add_warning(warnings, "accepted_records_are_not_primary_case_records")
    if no_primary_case_dataset_records:
        _add_warning(warnings, "no_primary_case_dataset_records")
    if no_corroborated_primary_case_events:
        _add_warning(warnings, "no_corroborated_primary_case_events")

    task = _task_metadata(state)
    disease_mismatch_count = status_counter.get("quarantined_disease_mismatch", 0)
    outside_scope_count = status_counter.get("quarantined_outside_scope", 0)
    validation_conflict_count = status_counter.get("quarantined_validation_conflict", 0)
    critical_anomaly_count = sum(
        1
        for row in quarantined_records
        if row.get("record_final_inclusion_status") == "quarantined_critical_anomaly"
        and any(
            _lower((anom or {}).get("severity")) == "critical"
            for anom in _as_list(state.get("anomaly_results"))
            if isinstance(anom, dict)
            and anom.get("record_id") == row.get("record_id")
        )
    )
    high_anomaly_count = sum(
        1
        for row in quarantined_records
        if row.get("record_final_inclusion_status") == "quarantined_critical_anomaly"
        and any(
            _lower((anom or {}).get("severity")) == "high"
            for anom in _as_list(state.get("anomaly_results"))
            if isinstance(anom, dict)
            and anom.get("record_id") == row.get("record_id")
        )
    )
    source_blocked_count = status_counter.get(
        "quarantined_source_not_task_relevant", 0
    )
    sources_by_id = _index_by_id(_as_list(state.get("source_registry")), "source_id")
    official_source_record_count = sum(
        1
        for record in final_dataset
        if _source_is_official_or_high_trust(
            sources_by_id.get(_source_id(record)),
            record,
        )
    )
    metric_record_count = sum(
        1 for record in final_dataset if record.get("metric_value") not in (None, "")
    )
    coverage_audit = _as_dict(state.get("source_coverage_audit"))
    coverage_status = (
        coverage_audit.get("coverage_status")
        or coverage_audit.get("status")
        or coverage_audit.get("summary_status")
    )
    requirement_count = coverage_audit.get("requirement_count")
    coverage_complete = not requirement_count or coverage_audit.get(
        "coverage_completeness_status"
    ) in {
        "complete_target_coverage",
        "not_required",
    } or (
        requirement_count
        and coverage_audit.get("accepted_requirement_count") == requirement_count
    )
    if (
        _direct_collection_enabled(state)
        and official_source_record_count > 0
        and len(final_dataset) > 0
        and coverage_complete
    ):
        coverage_status = "accepted"
    official_extraction_failure_count = len(
        _as_list(state.get("official_extraction_failures"))
    )
    accepted_metric_category_counter = Counter(
        str(record.get("metric_category") or "unknown")
        for record in final_dataset
        if record.get("metric_value") not in (None, "")
    )
    quarantined_metric_category_counter = Counter(
        str(record.get("metric_category") or "unknown")
        for record in quarantined_records
        if record.get("metric_value") not in (None, "")
    )
    accepted_positivity_rate_count = sum(
        count
        for category, count in accepted_metric_category_counter.items()
        if "positivity" in category.lower()
        or "positive_percent" in category.lower()
        or "percent_positive" in category.lower()
    )
    accepted_column_semantics_counter = Counter(
        str(record.get("resolved_column_period_type") or "unspecified")
        for record in final_dataset
        if record.get("metric_value") not in (None, "")
    )
    quarantined_column_semantics_counter = Counter(
        str(record.get("resolved_column_period_type") or "unspecified")
        for record in quarantined_records
        if record.get("metric_value") not in (None, "")
    )
    accepted_narrative_metric_count = sum(
        1
        for record in final_dataset
        if record.get("metric_value") not in (None, "")
        and str(record.get("row_context_type") or "").lower()
        == "markdown_metric_line"
    )
    quarantined_narrative_metric_count = sum(
        1
        for record in quarantined_records
        if record.get("metric_value") not in (None, "")
        and str(record.get("row_context_type") or "").lower()
        == "markdown_metric_line"
    )
    accepted_ed_visit_percent_count = sum(
        count
        for category, count in accepted_metric_category_counter.items()
        if "ed_visit_percent" in category.lower()
    )
    direct_collection_summary = {
        "collection_mode": _collection_mode(state),
        "direct_collection_enabled": _direct_collection_enabled(state),
        "metric_record_count": metric_record_count,
        "official_source_record_count": official_source_record_count,
        "coverage_status": coverage_status,
        "official_extraction_failure_count": official_extraction_failure_count,
        "final_dataset_count": len(final_dataset),
        "final_case_dataset_count": accepted_primary_case_record_count,
        "accepted_metric_category_counts": dict(accepted_metric_category_counter),
        "quarantined_metric_category_counts": dict(
            quarantined_metric_category_counter
        ),
        "accepted_column_semantic_counts": dict(accepted_column_semantics_counter),
        "quarantined_column_semantic_counts": dict(
            quarantined_column_semantics_counter
        ),
        "accepted_positivity_rate_count": accepted_positivity_rate_count,
        "accepted_ed_visit_percent_count": accepted_ed_visit_percent_count,
        "accepted_narrative_metric_count": accepted_narrative_metric_count,
        "quarantined_narrative_metric_count": quarantined_narrative_metric_count,
        "human_review_record_count": pending_count,
    }

    run_quality_summary = {
        "run_quality_status": run_quality_status,
        "final_dataset_mode": "task_aware_quality_gated_records",
        "collection_mode": _collection_mode(state),
        "direct_collection_enabled": _direct_collection_enabled(state),
        **task,
        "normalized_record_count": normalized_count,
        "accepted_record_count": accepted_count,
        "primary_case_dataset_eligible_count": primary_case_dataset_eligible_count,
        "accepted_primary_case_record_count": accepted_primary_case_record_count,
        "accepted_non_primary_observation_count": accepted_non_primary_count,
        "task_aware_non_primary_record_count": task_aware_non_primary_record_count,
        "surveillance_summary_record_count": surveillance_summary_record_count,
        "non_primary_observation_count": non_primary_observation_count,
        "corroborated_primary_case_event_count": corroborated_primary_case_event_count,
        "primary_case_dataset_status": primary_case_dataset_status,
        "no_primary_case_dataset_records": no_primary_case_dataset_records,
        "no_corroborated_primary_case_events": no_corroborated_primary_case_events,
        "accepted_records_are_not_primary_case_records": accepted_records_are_not_primary_case_records,
        "quarantined_record_count": quarantined_count,
        "pending_review_record_count": pending_count,
        "final_dataset_count": len(final_dataset),
        "final_dataset_post_review_count": len(final_dataset_post_review),
        "rejected_record_count": len(records_excluded_by_review),
        "disease_mismatch_record_count": disease_mismatch_count,
        "outside_scope_record_count": outside_scope_count,
        "validation_conflict_record_count": validation_conflict_count,
        "critical_anomaly_record_count": critical_anomaly_count,
        "high_anomaly_record_count": high_anomaly_count,
        "source_blocked_record_count": source_blocked_count,
        "official_source_record_count": official_source_record_count,
        "coverage_status": coverage_status,
        "official_extraction_failure_count": official_extraction_failure_count,
        "direct_collection_summary": direct_collection_summary,
        "no_compatible_validation_source": no_compatible_validation,
        "validation_limited": validation_limited,
        "human_review_required": human_review_required,
        "blocking_reason_counts": dict(blocking_counter),
        "warning_counts": dict(warning_counter),
        "record_final_inclusion_status_counts": dict(status_counter),
        "acceptance_reason": acceptance_reason,
        "recommended_user_message": recommended,
        "warnings": warnings,
    }
    final_dataset_quality_summary = {
        "quality_gate_method": QUALITY_GATE_METHOD,
        "final_dataset_mode": "task_aware_quality_gated_records",
        "collection_mode": _collection_mode(state),
        "direct_collection_enabled": _direct_collection_enabled(state),
        "normalized_record_count": normalized_count,
        "accepted_record_count": accepted_count,
        "primary_case_dataset_eligible_count": primary_case_dataset_eligible_count,
        "accepted_primary_case_record_count": accepted_primary_case_record_count,
        "accepted_non_primary_observation_count": accepted_non_primary_count,
        "task_aware_non_primary_record_count": task_aware_non_primary_record_count,
        "surveillance_summary_record_count": surveillance_summary_record_count,
        "non_primary_observation_count": non_primary_observation_count,
        "corroborated_primary_case_event_count": corroborated_primary_case_event_count,
        "primary_case_dataset_status": primary_case_dataset_status,
        "no_primary_case_dataset_records": no_primary_case_dataset_records,
        "no_corroborated_primary_case_events": no_corroborated_primary_case_events,
        "accepted_records_are_not_primary_case_records": accepted_records_are_not_primary_case_records,
        "quarantined_record_count": quarantined_count,
        "pending_review_record_count": pending_count,
        "post_review_record_count": len(final_dataset_post_review),
        "record_final_inclusion_status_counts": dict(status_counter),
        "blocking_reason_counts": dict(blocking_counter),
        "warning_counts": dict(warning_counter),
        "accepted_statuses": sorted(ACCEPTED_STATUSES),
        "official_source_record_count": official_source_record_count,
        "coverage_status": coverage_status,
        "official_extraction_failure_count": official_extraction_failure_count,
    }
    return run_quality_summary, final_dataset_quality_summary


def apply_run_quality_gates(state: dict) -> dict:
    """Return quality-gated final dataset views and run quality summaries."""

    from .evidence_qualification import evidence_qualification_enabled, build_evidence_index, qualify_records
    if evidence_qualification_enabled():
        contract = dict(state.get("structured_task") or {})
        contract.update(state.get("task_evidence_contract") or {})
        records = list(state.get("normalized_records") or [])
        # Review may change a value, so corrected rows must be qualified afresh.
        reviewed = {str(r.get("record_id")): r for r in state.get("final_dataset_post_review") or []}
        records = [reviewed.get(str(r.get("record_id")), r) for r in records]
        input_ids = {str(r.get("record_id")) for r in records}
        # A recovered observation has its own ID. Preserve a user's correction
        # to that observation instead of recreating its old value from the parent.
        records.extend(row for key, row in reviewed.items() if key not in input_ids
                       and str(row.get("recovered_from_record_id")) in input_ids)
        groups = qualify_records(records, contract=contract, evidence_index=build_evidence_index(state))
        excluded = {str(r.get("record_id")) for r in state.get("records_excluded_by_human_review") or []}
        def excluded_by_review(row):
            return any(str(value) in excluded for value in
                       (row.get("record_id"), row.get("recovered_from_record_id"))
                       if value is not None)
        for key in ("qualified_records", "qualified_case_records", "qualified_aggregate_records", "qualified_context_records"):
            groups[key] = [r for r in groups[key] if not excluded_by_review(r)]
        final = [r for r in groups["qualified_records"] if r["product_kind"] in {"case_level", "aggregate"} and not excluded_by_review(r)]
        summary = {"quality_gate_method":"evidence", "qualified_record_count":len(final), "candidate_record_count":len(groups["candidate_records"]), "run_quality_status":"qualified_evidence_available" if final else "no_qualified_evidence"}
        return {**groups, "final_dataset_pre_quality_gate":records, "final_dataset":final,
                "final_dataset_post_review":final, "quarantined_records":groups["candidate_records"],
                "pending_review_records":groups["candidate_records"],
                "non_primary_observations":groups["qualified_context_records"],
                "record_inclusion_decisions":groups["record_qualifications"],
                "run_quality_summary":summary, "final_dataset_quality_summary":dict(summary),
                "direct_collection_summary":{}, "collection_decision_summary":dict(summary)}

    normalized_records = [
        row for row in _as_list(state.get("normalized_records")) if isinstance(row, dict)
    ]
    indexes = _build_indexes(state)
    pre_quality_records, decisions = _evaluate_records(normalized_records, state, indexes)
    final_dataset, quarantined_records, pending_review_records = _split_evaluated(
        pre_quality_records
    )
    non_primary_observations = _non_primary_observations(pre_quality_records)
    final_dataset_post_review, post_review_decisions = _post_review_records(
        state, final_dataset, indexes
    )
    all_decisions = decisions + post_review_decisions
    records_excluded_by_review = [
        row
        for row in _as_list(state.get("records_excluded_by_human_review"))
        if isinstance(row, dict)
    ]
    run_quality_summary, final_dataset_quality_summary = _build_quality_summary(
        state=state,
        pre_quality_records=pre_quality_records,
        final_dataset=final_dataset,
        non_primary_observations=non_primary_observations,
        quarantined_records=quarantined_records,
        pending_review_records=pending_review_records,
        final_dataset_post_review=final_dataset_post_review,
        records_excluded_by_review=records_excluded_by_review,
        decisions=all_decisions,
    )
    collection_decision_summary = _collection_decision_summary(
        state=state,
        final_dataset=final_dataset,
        quarantined_records=quarantined_records,
        pending_review_records=pending_review_records,
        decisions=all_decisions,
    )
    run_quality_summary["collection_decision_summary"] = collection_decision_summary
    final_dataset_quality_summary["collection_decision_summary"] = (
        collection_decision_summary
    )
    return {
        "final_dataset_pre_quality_gate": pre_quality_records,
        "final_dataset": final_dataset,
        "final_dataset_post_review": final_dataset_post_review,
        "quarantined_records": quarantined_records,
        "pending_review_records": pending_review_records,
        "non_primary_observations": non_primary_observations,
        "record_inclusion_decisions": all_decisions,
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": final_dataset_quality_summary,
        "direct_collection_summary": run_quality_summary.get(
            "direct_collection_summary"
        )
        or {},
        "collection_decision_summary": collection_decision_summary,
    }
