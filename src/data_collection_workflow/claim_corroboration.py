"""Claim-level cross-source corroboration for workflow records.

This module is deterministic. It does not search, fetch, call an LLM, or change
graph topology. It converts existing records into public-health claims, compares
claims across sources, and builds auditable corroborated-event summaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import os
import re
from typing import Any

from .models import (
    ClaimComparison,
    CorroboratedEvent,
    CorroborationSummary,
    HumanReviewItem,
    PublicHealthClaim,
)

CLAIM_METHOD = "deterministic_claim_level_corroboration_v1"
CLAIM_MINOR_ABSOLUTE_DIFFERENCE = 1.0
CLAIM_MINOR_RELATIVE_DIFFERENCE = 0.05
DEFAULT_CLAIM_COMPARISON_STORAGE = "meaningful_edges"
CLAIM_COMPARISON_STORAGE_MODES = {"meaningful_edges", "full"}
MEANINGFUL_COMPARISON_STATUSES = {
    "corroborates",
    "partially_supports",
    "conflicts",
    "duplicate_same_source",
}

CASE_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
)
COUNT_FIELDS = CASE_FIELDS + ("deaths", "hospitalizations")

EXPOSURE_MONITORING_TERMS = (
    "completed their 42-day public health monitoring period",
    "completed 42-day public health monitoring",
    "public health monitoring period",
    "remained healthy",
    "remained well",
    "no symptoms",
    "without symptoms",
    "exposed people",
    "exposed persons",
    "exposed residents",
    "monitor their health",
    "under monitoring",
)

ZERO_CASE_TERMS = (
    "no confirmed cases",
    "no cases reported",
    "no reported cases",
    "zero confirmed cases",
    "0 confirmed cases",
    "no human cases",
    "no hantavirus cases",
)

BACKGROUND_TERMS = (
    "fact sheet",
    "general information",
    "prevention",
    "symptoms",
    "signs and symptoms",
    "transmission",
    "rodents",
    "cleaning rodent droppings",
    "about hantavirus",
)

INFERRED_DEATH_CLAIM_TERMS = (
    "death confirmed",
    "has died of",
    "died of hantavirus",
    "died of hantavirus pulmonary syndrome",
)

INFERRED_CONFIRMED_CASE_TERMS = (
    "confirmed the first case",
    "reports first hantavirus pulmonary syndrome case",
    "reported the first case",
    "first reported case",
    "hantavirus confirmed in",
    "resident was hospitalized with hantavirus pulmonary syndrome",
    "has confirmed the first case",
)

OFFICIAL_SOURCE_MARKERS = (
    "official_public_health_agency",
    "state_or_local_public_health_agency",
    "national_public_health_agency",
    "international_public_health_agency",
    "international_organization_report",
    "structured_database",
)


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _first_present(row: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return None


def _date_or_period(record: dict) -> str | None:
    return _first_present(
        record,
        ("date_reported", "reporting_period", "event_start_date", "as_of_date"),
    )


def _date_key(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    match = re.search(r"(20\d{2})(?:[-/](\d{1,2})(?:[-/](\d{1,2}))?)?", text)
    if not match:
        return text.lower()
    if match.group(3):
        return (
            f"{match.group(1)}-{int(match.group(2)):02d}"
            f"-{int(match.group(3)):02d}"
        )
    if match.group(2):
        return f"{match.group(1)}-{int(match.group(2)):02d}"
    return match.group(1)


def _task_location(state: dict) -> str | None:
    structured = _as_dict(state.get("structured_task"))
    spec = _as_dict(state.get("collection_spec"))
    return (
        structured.get("location")
        or spec.get("geography")
        or spec.get("location")
        or None
    )


def _task_disease(state: dict) -> str | None:
    structured = _as_dict(state.get("structured_task"))
    spec = _as_dict(state.get("collection_spec"))
    return structured.get("disease") or spec.get("disease") or None


def _source_map(state: dict) -> dict[str, dict]:
    return {
        str(row.get("source_id")): row
        for row in _as_list(state.get("source_registry"))
        if isinstance(row, dict) and row.get("source_id")
    }


def _records_for_claims(state: dict) -> list[dict]:
    for key in (
        "final_dataset_pre_quality_gate",
        "normalized_records",
        "validated_records",
        "raw_records",
    ):
        rows = [row for row in _as_list(state.get(key)) if isinstance(row, dict)]
        if rows:
            return rows
    return []


def _record_evidence(record: dict) -> str:
    return " ".join(
        _clean(record.get(field))
        for field in (
            "evidence_quote",
            "evidence_context",
            "count_semantics",
            "source_title",
            "notes",
        )
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def _claim_status(record: dict, observation_type: str) -> str:
    status = _lower(record.get("record_final_inclusion_status"))
    if "quarantined" in status:
        return "quarantined"
    if "rejected" in status or _lower(record.get("schema_status")) == "rejected":
        return "rejected"
    if observation_type == "background_context":
        return "context_only"
    if record.get("requires_human_review"):
        return "pending_review"
    return "active"


def _observation_type(record: dict, count_field: str | None) -> tuple[str, str]:
    evidence = _record_evidence(record)
    text = evidence.lower()
    count_value = _number(record.get(count_field)) if count_field else None

    if count_field == "cases_confirmed" and count_value and count_value > 0:
        return "confirmed_case_record", "record reports confirmed case count"
    if count_field == "cases_probable" and count_value and count_value > 0:
        return "probable_case_record", "record reports probable case count"
    if count_field == "cases_suspected" and count_value and count_value > 0:
        return "suspected_case_record", "record reports suspected case count"
    if count_field == "deaths" and count_value and count_value > 0:
        return "death_record", "record reports death count"
    if count_field == "hospitalizations" and count_value and count_value > 0:
        return "hospitalization_record", "record reports hospitalization count"
    if _contains_any(text, EXPOSURE_MONITORING_TERMS):
        return (
            "exposure_monitoring_record",
            "evidence describes exposed people under monitoring or remaining healthy",
        )
    if (
        (count_field in CASE_FIELDS and count_value == 0)
        or _contains_any(text, ZERO_CASE_TERMS)
    ):
        return (
            "zero_case_statement",
            "evidence states no confirmed or reported cases",
        )
    if count_field == "cases_unspecified" and count_value and count_value > 0:
        return "unspecified_case_record", "record reports unspecified case count"
    if count_field == "cases_confirmed":
        return "confirmed_case_record", "record reports confirmed case count"
    if count_field == "cases_probable":
        return "probable_case_record", "record reports probable case count"
    if count_field == "cases_suspected":
        return "suspected_case_record", "record reports suspected case count"
    if count_field == "cases_unspecified":
        return "unspecified_case_record", "record reports unspecified case count"
    if count_field == "deaths":
        return "death_record", "record reports death count"
    if count_field == "hospitalizations":
        return "hospitalization_record", "record reports hospitalization count"
    if "surveillance" in text and any(term in text for term in ("case", "count")):
        return "surveillance_summary", "evidence is a surveillance summary"
    if "outbreak" in text and any(term in text for term in ("case", "death")):
        return "outbreak_summary", "evidence is an outbreak summary"
    if _contains_any(text, BACKGROUND_TERMS):
        return "background_context", "evidence is background or prevention context"
    return (
        "ambiguous_public_health_observation",
        "record lacks a deterministic case/death/hospitalization claim",
    )


def _claim_flags(observation_type: str, count_value: float | None) -> dict:
    is_case = observation_type in {
        "confirmed_case_record",
        "probable_case_record",
        "suspected_case_record",
        "unspecified_case_record",
    }
    is_death = observation_type == "death_record"
    is_hospital = observation_type == "hospitalization_record"
    is_zero = observation_type == "zero_case_statement"
    is_exposure = observation_type == "exposure_monitoring_record"
    is_background = observation_type == "background_context"
    primary_eligible = bool(
        (is_case or is_death or is_hospital)
        and not is_zero
        and not is_exposure
        and not is_background
        and count_value is not None
        and count_value > 0
    )
    return {
        "is_case_claim": is_case,
        "is_death_claim": is_death,
        "is_zero_case_statement": is_zero,
        "is_exposure_monitoring_claim": is_exposure,
        "is_background_context_claim": is_background,
        "primary_case_dataset_eligible": primary_eligible,
    }


def _claim_count_fields(record: dict) -> list[str | None]:
    fields = [field for field in COUNT_FIELDS if _number(record.get(field)) is not None]
    if fields:
        return fields
    inferred = _inferred_claim_count_fields(record)
    if inferred:
        return inferred
    return [None]


def _inferred_claim_count_fields(record: dict) -> list[str]:
    text = _record_evidence(record).lower()
    if _contains_any(text, INFERRED_DEATH_CLAIM_TERMS):
        return ["deaths"]
    if _contains_any(text, INFERRED_CONFIRMED_CASE_TERMS):
        return ["cases_confirmed"]
    return []


def _inferred_count_value(record: dict, count_field: str | None) -> float | None:
    if not count_field:
        return None
    explicit = _number(record.get(count_field))
    if explicit is not None:
        return explicit
    if count_field in _inferred_claim_count_fields(record):
        return 1.0
    return None


def _build_claim(record: dict, source: dict, count_field: str | None) -> dict:
    count_value = _inferred_count_value(record, count_field)
    observation_type, reason = _observation_type(record, count_field)
    flags = _claim_flags(observation_type, count_value)
    record_id = _clean(record.get("record_id"))
    claim_suffix = count_field or "observation"
    actual_publisher = (
        record.get("actual_publisher")
        or source.get("actual_publisher")
    )
    publisher_value = actual_publisher or record.get("publisher") or source.get("publisher")
    actual_publisher_normalized = (
        record.get("actual_publisher_normalized")
        or source.get("actual_publisher_normalized")
    )
    source_type_final = (
        record.get("source_type_final")
        or source.get("source_type_final")
        or record.get("source_type")
        or source.get("source_type")
    )
    claim = {
        "claim_id": f"claim_{record_id}_{claim_suffix}",
        "source_record_id": record_id or None,
        "event_cluster_id": record.get("event_cluster_id"),
        "linked_event_id": record.get("linked_event_id"),
        "claim_type": "public_health_observation",
        "observation_type": observation_type,
        "disease": record.get("disease") or record.get("disease_standard_name"),
        "disease_standard_name": record.get("disease_standard_name"),
        "pathogen_or_syndrome": record.get("pathogen_or_syndrome")
        or record.get("virus_or_syndrome"),
        "country": record.get("country"),
        "subnational_location": record.get("subnational_location"),
        "locality": record.get("locality"),
        "geographic_scope": record.get("geographic_scope")
        or record.get("subnational_location")
        or record.get("country"),
        "date_or_period": _date_or_period(record),
        "date_reported": record.get("date_reported"),
        "event_start_date": record.get("event_start_date"),
        "event_end_date": record.get("event_end_date"),
        "reporting_period": record.get("reporting_period"),
        "as_of_date": record.get("as_of_date"),
        "count_field": count_field,
        "count_value": count_value,
        "cases_confirmed": _number(record.get("cases_confirmed")),
        "cases_probable": _number(record.get("cases_probable")),
        "cases_suspected": _number(record.get("cases_suspected")),
        "cases_unspecified": _number(record.get("cases_unspecified")),
        "deaths": _number(record.get("deaths")),
        "hospitalizations": _number(record.get("hospitalizations")),
        "statistical_count_type": record.get("statistical_count_type"),
        "count_semantics": record.get("count_semantics"),
        "count_unit": record.get("count_unit"),
        "observation_semantics_reason": reason,
        "source_id": record.get("source_id"),
        "source_url": record.get("source_url") or source.get("canonical_url"),
        "source_title": record.get("source_title") or source.get("title"),
        "publisher": publisher_value,
        "source_type": record.get("source_type") or source.get("source_type"),
        "source_role_final": record.get("source_role_final")
        or source.get("source_role_final"),
        "actual_publisher": actual_publisher,
        "actual_publisher_normalized": actual_publisher_normalized,
        "source_type_final": source_type_final,
        "source_independence_group": record.get("source_independence_group")
        or source.get("source_independence_group"),
        "claim_support_role": record.get("claim_support_role")
        or source.get("claim_support_role"),
        "recommended_source_role": record.get("recommended_source_role")
        or source.get("recommended_source_role"),
        "recommended_fetch_use": record.get("recommended_fetch_use")
        or source.get("recommended_fetch_use"),
        "recommended_extraction_use": record.get("recommended_extraction_use")
        or source.get("recommended_extraction_use"),
        "likely_syndicated_or_aggregated": bool(
            record.get("likely_syndicated_or_aggregated")
            or source.get("likely_syndicated_or_aggregated")
        ),
        "upstream_source_mentions": list(
            record.get("upstream_source_mentions")
            or source.get("upstream_source_mentions")
            or []
        ),
        "credibility_score": record.get("credibility_score")
        or source.get("credibility_score"),
        "credibility_level": record.get("credibility_level")
        or source.get("credibility_level"),
        "discovery_method": record.get("discovery_method")
        or source.get("discovery_method"),
        "search_provider": record.get("search_provider") or source.get("search_provider"),
        "query_used": record.get("query_used") or source.get("query_used"),
        "document_id": record.get("document_id"),
        "supporting_chunk_id": record.get("supporting_chunk_id"),
        "evidence_quote": record.get("evidence_quote"),
        "evidence_context": record.get("evidence_context"),
        "claim_status": _claim_status(record, observation_type),
        "extraction_method": record.get("extraction_method"),
        "confidence": record.get("extraction_confidence")
        or record.get("count_confidence")
        or 0.75,
        "warnings": list(record.get("semantic_warnings") or []),
        "requires_human_review": bool(record.get("requires_human_review")),
        "human_review_reason": record.get("human_review_reason"),
    }
    claim.update(flags)
    return PublicHealthClaim(**claim).model_dump()


def build_claims_from_state(state: dict) -> list[dict]:
    """Build public-health claims from current workflow records."""

    sources = _source_map(state)
    claims: list[dict] = []
    for record in _records_for_claims(state):
        if not record.get("record_id"):
            continue
        source = sources.get(str(record.get("source_id"))) or {}
        for count_field in _claim_count_fields(record):
            claims.append(_build_claim(record, source, count_field))
    return claims


def _disease_status(left: dict, right: dict) -> str:
    a = _lower(left.get("disease_standard_name") or left.get("disease"))
    b = _lower(right.get("disease_standard_name") or right.get("disease"))
    if not a or not b:
        return "insufficient_information"
    if a == b or a in b or b in a:
        return "matched"
    return "not_matched"


def _claim_location(claim: dict) -> str:
    return _clean(
        claim.get("locality")
        or claim.get("subnational_location")
        or claim.get("geographic_scope")
        or claim.get("country")
    )


def _evidence_mentions_location(claim: dict, location: str) -> bool:
    text = " ".join(
        _clean(claim.get(field))
        for field in ("evidence_quote", "evidence_context", "count_semantics")
    ).lower()
    return bool(location and location.lower() in text)


def _geography_status(left: dict, right: dict) -> str:
    left_sub = _clean(left.get("subnational_location") or left.get("locality"))
    right_sub = _clean(right.get("subnational_location") or right.get("locality"))
    left_country = _lower(left.get("country"))
    right_country = _lower(right.get("country"))
    if left_sub and right_sub:
        return "matched" if left_sub.lower() == right_sub.lower() else "not_matched"
    if left_sub and not right_sub:
        if left_country and right_country and left_country != right_country:
            return "not_matched"
        return "partially_matched" if _evidence_mentions_location(right, left_sub) else "insufficient_information"
    if right_sub and not left_sub:
        if left_country and right_country and left_country != right_country:
            return "not_matched"
        return "partially_matched" if _evidence_mentions_location(left, right_sub) else "insufficient_information"
    left_loc = _claim_location(left).lower()
    right_loc = _claim_location(right).lower()
    if not left_loc or not right_loc:
        return "insufficient_information"
    return "matched" if left_loc == right_loc else "not_matched"


def _time_status(left: dict, right: dict) -> str:
    a = _date_key(left.get("date_or_period"))
    b = _date_key(right.get("date_or_period"))
    if not a or not b:
        return "insufficient_information"
    if a == b:
        return "matched"
    if len(a) == 4 and b.startswith(a):
        return "partially_matched"
    if len(b) == 4 and a.startswith(b):
        return "partially_matched"
    return "not_matched"


def _observation_status(left: dict, right: dict) -> str:
    a = left.get("observation_type")
    b = right.get("observation_type")
    if not a or not b:
        return "insufficient_information"
    if a == b:
        return "matched"
    case_types = {
        "confirmed_case_record",
        "probable_case_record",
        "suspected_case_record",
        "unspecified_case_record",
    }
    if a in case_types and b in case_types:
        return "partially_matched"
    if "zero_case_statement" in {a, b} and (a in case_types or b in case_types):
        return "conflict"
    if "background_context" in {a, b}:
        return "not_comparable"
    if "exposure_monitoring_record" in {a, b}:
        return "not_comparable"
    return "not_comparable"


def _count_field_status(left: dict, right: dict) -> str:
    a = left.get("count_field")
    b = right.get("count_field")
    if not a or not b:
        return "insufficient_information"
    if a == b:
        return "matched"
    if str(a).startswith("cases_") and str(b).startswith("cases_"):
        return "partially_matched"
    return "not_matched"


def _count_semantics_status(left: dict, right: dict) -> str:
    a = _lower(left.get("statistical_count_type") or left.get("count_semantics"))
    b = _lower(right.get("statistical_count_type") or right.get("count_semantics"))
    if not a or not b:
        return "insufficient_information"
    if a == b:
        return "matched"
    a_family = _count_semantics_family(a)
    b_family = _count_semantics_family(b)
    if a_family and b_family:
        return "partially_matched" if a_family == b_family else "not_matched"
    if "confirmed" in a and "confirmed" in b:
        return "partially_matched"
    if "case" in a and "case" in b:
        return "partially_matched"
    return "not_matched"


def _count_semantics_family(value: str) -> str | None:
    text = _lower(value)
    if any(
        marker in text
        for marker in ("cumulative", "to date", "as of", "running total", "total since")
    ):
        return "cumulative_snapshot"
    if any(
        marker in text
        for marker in (
            "incident",
            "incidence",
            "new case",
            "during the",
            "daily",
            "weekly",
            "monthly",
        )
    ):
        return "incident_period"
    if "prevalence" in text or "prevalent" in text:
        return "prevalence"
    return None


def _count_value_status(left: dict, right: dict) -> str:
    a = _number(left.get("count_value"))
    b = _number(right.get("count_value"))
    if a is None or b is None:
        return "insufficient_information"
    if a == b:
        return "matched"
    absolute_difference = abs(a - b)
    denominator = max(abs(a), abs(b), 1.0)
    relative_difference = absolute_difference / denominator
    if (
        absolute_difference <= CLAIM_MINOR_ABSOLUTE_DIFFERENCE
        or relative_difference <= CLAIM_MINOR_RELATIVE_DIFFERENCE
    ):
        return "minor_numeric_difference"
    return "conflict"


def _independence_key(claim: dict) -> str:
    return _clean(
        claim.get("source_independence_group")
        or claim.get("actual_publisher_normalized")
        or claim.get("actual_publisher")
        or claim.get("source_url")
        or claim.get("source_id")
        or claim.get("publisher")
    )


def _source_independence_status(left: dict, right: dict) -> str:
    left_key = _independence_key(left)
    right_key = _independence_key(right)
    if not left_key or not right_key:
        return "insufficient_information"
    return "same_source" if left_key == right_key else "independent"


def _compare_pair(left: dict, right: dict, index: int) -> dict:
    disease = _disease_status(left, right)
    geography = _geography_status(left, right)
    time = _time_status(left, right)
    observation = _observation_status(left, right)
    count_field = _count_field_status(left, right)
    semantics = _count_semantics_status(left, right)
    value = _count_value_status(left, right)
    independence = _source_independence_status(left, right)

    warnings: list[str] = []
    needs_review = False
    human_review_reason = None
    confidence = 0.75

    if independence == "same_source" and disease == "matched" and geography in {
        "matched",
        "partially_matched",
    }:
        comparability = "comparable"
        match = "duplicate_same_source"
        reason = "claims come from the same source and do not count as independent corroboration"
    elif disease != "matched":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "disease does not match"
    elif geography == "not_matched":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "geography does not match"
    elif geography == "insufficient_information":
        comparability = "insufficient_information"
        match = "insufficient_information"
        reason = "geography is insufficient for local claim corroboration"
    elif time == "not_matched":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "time window does not match"
    elif observation == "not_comparable":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "observation types are not comparable as the same public-health claim"
    elif count_field == "not_matched":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "count fields are not comparable"
    elif semantics == "not_matched":
        comparability = "not_comparable"
        match = "not_comparable"
        reason = "count semantics are not comparable"
    elif "conflict" in {observation, value}:
        comparability = "comparable"
        match = "conflicts"
        reason = "claims are comparable but conflict in observation type or count value"
        needs_review = True
        human_review_reason = reason
        warnings.append("conflicting_claims")
        confidence = 0.85
    elif value == "matched" and independence == "independent" and geography == "matched":
        comparability = "comparable"
        match = "corroborates"
        reason = "independent claims match on disease, geography, time, count field, and value"
        confidence = 0.9
    elif value == "matched" and independence == "independent" and geography == "partially_matched":
        comparability = "partially_comparable"
        match = "partially_supports"
        reason = "broader source explicitly mentions the local geography and matching count value"
        confidence = 0.8
    elif value == "minor_numeric_difference" and independence == "independent":
        comparability = "partially_comparable"
        match = "partially_supports"
        reason = "independent claims have only a minor numeric difference within the configured tolerance"
        warnings.append("minor_numeric_difference")
        confidence = 0.75
    elif value == "insufficient_information":
        comparability = "insufficient_information"
        match = "insufficient_information"
        reason = "one or both claims lack a count value"
    else:
        comparability = "partially_comparable"
        match = "partially_supports"
        reason = "claims partially align but do not fully satisfy corroboration criteria"
        warnings.append("partial_claim_match")

    comparison = {
        "comparison_id": f"claim_cmp_{index:03d}",
        "left_claim_id": left["claim_id"],
        "right_claim_id": right["claim_id"],
        "left_source_id": left.get("source_id"),
        "right_source_id": right.get("source_id"),
        "left_record_id": left.get("source_record_id"),
        "right_record_id": right.get("source_record_id"),
        "compared_field": left.get("count_field") or right.get("count_field"),
        "disease_match_status": disease,
        "geography_match_status": geography,
        "time_match_status": time,
        "observation_type_match_status": observation,
        "count_semantics_match_status": semantics,
        "count_value_match_status": value,
        "source_independence_status": independence,
        "comparability_status": comparability,
        "corroboration_match_status": match,
        "confidence": confidence,
        "reason": reason,
        "warnings": warnings,
        "needs_human_review": needs_review,
        "human_review_reason": human_review_reason,
    }
    return ClaimComparison(**comparison).model_dump()


def _comparison_storage_mode(storage_mode: str | None = None) -> str:
    value = _lower(storage_mode or os.getenv("CLAIM_COMPARISON_STORAGE"))
    if value in CLAIM_COMPARISON_STORAGE_MODES:
        return value
    return DEFAULT_CLAIM_COMPARISON_STORAGE


def _canonical_claim_key(claim: dict) -> tuple[str, ...]:
    """Collapse repeated source claims before cross-source comparison."""

    return (
        _lower(claim.get("disease_standard_name") or claim.get("disease")),
        _lower(claim.get("event_cluster_id") or claim.get("linked_event_id")),
        _lower(claim.get("country")),
        _lower(claim.get("geographic_scope")),
        _lower(_claim_location(claim)),
        _date_key(claim.get("date_or_period")) or "",
        _lower(claim.get("observation_type")),
        _lower(claim.get("count_field")),
        _clean(claim.get("count_value")),
        _lower(claim.get("statistical_count_type") or claim.get("count_semantics")),
        _lower(_independence_key(claim)),
    )


def _plausibly_same_claim_block(left: dict, right: dict) -> bool:
    """Use disease, event, scope, location, and time to reject impossible pairs."""

    if _disease_status(left, right) != "matched":
        return False
    left_event = _clean(left.get("event_cluster_id") or left.get("linked_event_id"))
    right_event = _clean(right.get("event_cluster_id") or right.get("linked_event_id"))
    if left_event and right_event and left_event != right_event:
        return False
    left_country = _lower(left.get("country"))
    right_country = _lower(right.get("country"))
    if left_country and right_country and left_country != right_country:
        return False
    left_location = _lower(left.get("locality") or left.get("subnational_location"))
    right_location = _lower(right.get("locality") or right.get("subnational_location"))
    if left_location and right_location and left_location != right_location:
        return False
    if _time_status(left, right) == "not_matched":
        return False
    return True


def _planned_claim_pairs(claims: list[dict]) -> tuple[list[tuple[dict, dict]], dict]:
    comparable_claims = sorted(
        (
            claim
            for claim in claims
            if claim.get("claim_status") in {"active", "pending_review", "quarantined"}
        ),
        key=lambda claim: str(claim.get("claim_id") or ""),
    )
    canonical_groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for claim in comparable_claims:
        canonical_groups[_canonical_claim_key(claim)].append(claim)

    representatives: list[dict] = []
    planned_pairs: list[tuple[dict, dict]] = []
    for members in canonical_groups.values():
        ordered_members = sorted(members, key=lambda claim: str(claim.get("claim_id") or ""))
        representative = ordered_members[0]
        representatives.append(representative)
        planned_pairs.extend((representative, duplicate) for duplicate in ordered_members[1:])

    representatives.sort(key=lambda claim: str(claim.get("claim_id") or ""))
    for left, right in combinations(representatives, 2):
        if _plausibly_same_claim_block(left, right):
            planned_pairs.append((left, right))

    total_pair_count = len(comparable_claims) * (len(comparable_claims) - 1) // 2
    comparison_metadata = {
        "total_candidate_pair_count": total_pair_count,
        "canonical_claim_count": len(representatives),
        "compared_pair_count": len(planned_pairs),
        "skipped_pair_count": total_pair_count - len(planned_pairs),
    }
    return planned_pairs, comparison_metadata


def compare_claims_with_metadata(
    claims: list[dict],
    storage_mode: str | None = None,
) -> tuple[list[dict], dict]:
    """Compare only plausible canonical blocks and retain the requested edge set."""

    mode = _comparison_storage_mode(storage_mode)
    pairs, metadata = _planned_claim_pairs(claims)
    comparisons: list[dict] = []
    for index, (left, right) in enumerate(pairs, 1):
        comparison = _compare_pair(left, right, index)
        if mode == "full" or comparison.get("corroboration_match_status") in MEANINGFUL_COMPARISON_STATUSES:
            comparisons.append(comparison)
    metadata.update(
        {
            "claim_comparison_storage": mode,
            "persisted_comparison_count": len(comparisons),
        }
    )
    return comparisons, metadata


def compare_claims(
    claims: list[dict],
    storage_mode: str | None = None,
) -> list[dict]:
    comparisons, _ = compare_claims_with_metadata(claims, storage_mode=storage_mode)
    return comparisons


class _UnionFind:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent.setdefault(item, item)
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        lroot = self.find(left)
        rroot = self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def _supportive_status(status: str) -> bool:
    return status in {"corroborates", "partially_supports", "duplicate_same_source", "conflicts"}


def _is_official_claim(claim: dict) -> bool:
    source_type = _lower(claim.get("source_type_final") or claim.get("source_type"))
    role = _lower(claim.get("source_role_final"))
    return source_type in OFFICIAL_SOURCE_MARKERS or role == "validation"


def _event_status(claims: list[dict], comparisons: list[dict]) -> tuple[str, bool, str | None, list[str]]:
    warnings: list[str] = []
    if not claims:
        return "no_claims", False, None, warnings
    if any(c.get("corroboration_match_status") == "conflicts" for c in comparisons):
        return (
            "conflicting_claims",
            True,
            "Comparable public-health claims conflict and require review.",
            ["conflicting_claims"],
        )
    observation_types = {c.get("observation_type") for c in claims}
    if observation_types <= {"background_context"}:
        return "context_only", False, None, warnings
    if observation_types <= {"exposure_monitoring_record"}:
        return "exposure_monitoring_only", False, None, warnings
    if observation_types <= {"zero_case_statement"}:
        return "zero_case_statement_unverified", False, None, warnings
    support_count = sum(
        1
        for cmp in comparisons
        if cmp.get("corroboration_match_status") in {"corroborates", "partially_supports"}
    )
    independent_sources = {_independence_key(claim) for claim in claims if _independence_key(claim)}
    if len(independent_sources) >= 2 and support_count:
        if any(cmp.get("corroboration_match_status") == "corroborates" for cmp in comparisons):
            return "corroborated", False, None, warnings
        return "cross_source_supported", False, None, warnings
    if any(claim.get("primary_case_dataset_eligible") for claim in claims):
        return (
            "single_source_unverified",
            True,
            "Primary case claim has no independent corroborating source.",
            ["single_source_unverified"],
        )
    return "insufficient_information", False, None, warnings


def _group_claims(claims: list[dict], comparisons: list[dict]) -> list[tuple[list[dict], list[dict]]]:
    by_id = {claim["claim_id"]: claim for claim in claims}
    uf = _UnionFind(list(by_id))
    for cmp in comparisons:
        if _supportive_status(cmp.get("corroboration_match_status") or ""):
            uf.union(cmp["left_claim_id"], cmp["right_claim_id"])
    grouped_ids: dict[str, list[str]] = defaultdict(list)
    for claim_id in by_id:
        grouped_ids[uf.find(claim_id)].append(claim_id)
    groups: list[tuple[list[dict], list[dict]]] = []
    for ids in grouped_ids.values():
        id_set = set(ids)
        group_claims = [by_id[item] for item in ids]
        group_comparisons = [
            cmp
            for cmp in comparisons
            if cmp["left_claim_id"] in id_set and cmp["right_claim_id"] in id_set
        ]
        groups.append((group_claims, group_comparisons))
    return groups


def build_corroborated_events(claims: list[dict], comparisons: list[dict]) -> list[dict]:
    events: list[dict] = []
    for index, (group_claims, group_comparisons) in enumerate(
        _group_claims(claims, comparisons),
        1,
    ):
        primary = group_claims[0]
        status, needs_review, review_reason, warnings = _event_status(
            group_claims, group_comparisons
        )
        source_groups = _unique([_independence_key(claim) for claim in group_claims])
        conflicting_claim_ids = _unique(
            [
                cmp.get("left_claim_id")
                for cmp in group_comparisons
                if cmp.get("corroboration_match_status") == "conflicts"
            ]
            + [
                cmp.get("right_claim_id")
                for cmp in group_comparisons
                if cmp.get("corroboration_match_status") == "conflicts"
            ]
        )
        supporting_ids = _unique(
            [
                claim.get("claim_id")
                for claim in group_claims
                if claim.get("claim_id") not in conflicting_claim_ids
            ]
        )
        unverified_ids = (
            _unique([claim.get("claim_id") for claim in group_claims])
            if status in {"single_source_unverified", "zero_case_statement_unverified"}
            else []
        )
        event = {
            "corroborated_event_id": f"corr_event_{index:03d}",
            "event_cluster_id": primary.get("event_cluster_id"),
            "disease": primary.get("disease"),
            "country": primary.get("country"),
            "subnational_location": primary.get("subnational_location"),
            "locality": primary.get("locality"),
            "date_or_period": primary.get("date_or_period"),
            "observation_type": primary.get("observation_type"),
            "count_field": primary.get("count_field"),
            "canonical_count_value": primary.get("count_value"),
            "statistical_count_type": primary.get("statistical_count_type"),
            "count_semantics": primary.get("count_semantics"),
            "primary_claim_id": primary.get("claim_id"),
            "supporting_claim_ids": supporting_ids,
            "conflicting_claim_ids": conflicting_claim_ids,
            "unverified_claim_ids": unverified_ids,
            "source_ids": _unique([claim.get("source_id") for claim in group_claims]),
            "source_urls": _unique([claim.get("source_url") for claim in group_claims]),
            "publishers": _unique([claim.get("publisher") for claim in group_claims]),
            "actual_publishers": _unique(
                [claim.get("actual_publisher") for claim in group_claims]
            ),
            "independent_source_count": len(source_groups),
            "official_source_support_count": sum(
                1 for claim in group_claims if _is_official_claim(claim)
            ),
            "secondary_source_support_count": sum(
                1 for claim in group_claims if not _is_official_claim(claim)
            ),
            "source_independence_groups": source_groups,
            "corroboration_status": status,
            "corroboration_confidence": 0.9
            if status in {"corroborated", "cross_source_supported"}
            else 0.7,
            "corroboration_reason": review_reason
            or f"Claim group classified as {status}.",
            "primary_case_dataset_eligible": bool(
                status in {"corroborated", "cross_source_supported"}
                and any(claim.get("primary_case_dataset_eligible") for claim in group_claims)
            ),
            "needs_human_review": needs_review,
            "human_review_reason": review_reason,
            "warnings": warnings,
        }
        events.append(CorroboratedEvent(**event).model_dump())
    return events


def _make_review_items(events: list[dict]) -> list[dict]:
    items: list[dict] = []
    for event in events:
        if not event.get("needs_human_review"):
            continue
        event_id = event.get("corroborated_event_id")
        item_type = (
            "claim_corroboration_conflict"
            if event.get("corroboration_status") == "conflicting_claims"
            else "claim_corroboration_review"
        )
        item = HumanReviewItem(
            review_id=f"review_{item_type}_{event_id}",
            item_type=item_type,
            related_ids=_unique(
                [event_id]
                + _as_list(event.get("supporting_claim_ids"))
                + _as_list(event.get("conflicting_claim_ids"))
                + _as_list(event.get("unverified_claim_ids"))
            ),
            reason=event.get("human_review_reason")
            or event.get("corroboration_reason")
            or "Claim-level corroboration requires review.",
            priority=1 if item_type == "claim_corroboration_conflict" else 3,
            source_ids=_as_list(event.get("source_ids")),
            source_urls=_as_list(event.get("source_urls")),
            severity="high" if item_type == "claim_corroboration_conflict" else "medium",
            evidence_summary=event.get("corroboration_reason"),
            suggested_action="review_claim_corroboration",
            decision_options=[
                "accept_corroborated_event",
                "reject_claim",
                "mark_unverified",
                "defer",
            ],
        )
        items.append(item.model_dump())
    return items


def _summarize(
    claims: list[dict],
    comparisons: list[dict],
    events: list[dict],
    review_items: list[dict],
    comparison_metadata: dict | None = None,
) -> dict:
    status_counts = Counter(str(event.get("corroboration_status")) for event in events)
    observation_counts = Counter(str(claim.get("observation_type")) for claim in claims)
    summary = CorroborationSummary(
        claim_count=len(claims),
        claim_comparison_count=len(comparisons),
        corroborated_event_count=len(events),
        corroborated_primary_case_event_count=sum(
            1
            for event in events
            if event.get("primary_case_dataset_eligible")
            and event.get("corroboration_status")
            in {"corroborated", "cross_source_supported"}
        ),
        zero_case_statement_count=observation_counts.get("zero_case_statement", 0),
        exposure_monitoring_claim_count=observation_counts.get(
            "exposure_monitoring_record", 0
        ),
        conflicting_claim_count=sum(
            len(event.get("conflicting_claim_ids") or [])
            for event in events
            if event.get("corroboration_status") == "conflicting_claims"
        ),
        single_source_unverified_count=status_counts.get("single_source_unverified", 0),
        status_counts=dict(status_counts),
        observation_type_counts=dict(observation_counts),
        human_review_item_count=len(review_items),
        **(comparison_metadata or {}),
    )
    return summary.model_dump()


def annotate_records_with_claim_corroboration(
    records: list[dict],
    claims: list[dict],
    events: list[dict],
) -> list[dict]:
    claims_by_record: dict[str, list[dict]] = defaultdict(list)
    event_by_claim: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        if claim.get("source_record_id"):
            claims_by_record[str(claim["source_record_id"])].append(claim)
    for event in events:
        for claim_id in _as_list(event.get("supporting_claim_ids")) + _as_list(
            event.get("conflicting_claim_ids")
        ) + _as_list(event.get("unverified_claim_ids")):
            event_by_claim[str(claim_id)].append(event)

    annotated: list[dict] = []
    for record in records:
        row = dict(record)
        record_claims = claims_by_record.get(str(row.get("record_id")), [])
        if not record_claims:
            annotated.append(row)
            continue
        claim_ids = _unique([claim.get("claim_id") for claim in record_claims])
        related_events: list[dict] = []
        for claim_id in claim_ids:
            related_events.extend(event_by_claim.get(claim_id, []))
        observation_counter = Counter(
            str(claim.get("observation_type")) for claim in record_claims
        )
        observation_type = observation_counter.most_common(1)[0][0]
        observation_types = _unique(
            [claim.get("observation_type") for claim in record_claims]
        )
        statuses = _unique([event.get("corroboration_status") for event in related_events])
        claim_primary_values = [
            claim.get("primary_case_dataset_eligible")
            for claim in record_claims
            if claim.get("primary_case_dataset_eligible") in {True, False}
        ]
        if any(value is True for value in claim_primary_values):
            record_primary_eligible = True
        elif claim_primary_values and all(value is False for value in claim_primary_values):
            record_primary_eligible = False
        else:
            record_primary_eligible = any(
                event.get("primary_case_dataset_eligible") for event in related_events
            )
        if observation_types and set(observation_types) <= {
            "zero_case_statement",
            "exposure_monitoring_record",
            "background_context",
            "context_only",
            "ambiguous_public_health_observation",
            "exposure_monitoring_only",
            "zero_case_statement_unverified",
        }:
            record_primary_eligible = False
        related_event_warnings = _unique(
            warning
            for event in related_events
            for warning in _as_list(event.get("warnings"))
        )
        related_event_reasons = _unique(
            event.get("corroboration_reason") for event in related_events
        )
        if not record_primary_eligible:
            related_event_warnings = _unique(
                related_event_warnings + ["not_primary_case_record"]
            )
        row.update(
            {
                "observation_type": observation_type,
                "observation_types": observation_types,
                "claim_ids": claim_ids,
                "corroborated_event_ids": _unique(
                    [event.get("corroborated_event_id") for event in related_events]
                ),
                "corroboration_status": statuses[0] if len(statuses) == 1 else ";".join(statuses),
                "corroboration_reason": (
                    related_event_reasons[0]
                    if len(related_event_reasons) == 1
                    else "; ".join(related_event_reasons)
                ),
                "independent_source_count": max(
                    [
                        int(event.get("independent_source_count") or 0)
                        for event in related_events
                    ]
                    or [0]
                ),
                "official_source_support_count": max(
                    [
                        int(event.get("official_source_support_count") or 0)
                        for event in related_events
                    ]
                    or [0]
                ),
                "secondary_source_support_count": max(
                    [
                        int(event.get("secondary_source_support_count") or 0)
                        for event in related_events
                    ]
                    or [0]
                ),
                "primary_case_dataset_eligible": record_primary_eligible,
                "claim_corroboration_warnings": related_event_warnings,
            }
        )
        annotated.append(row)
    return annotated


def run_claim_corroboration(state: dict) -> dict:
    claims = build_claims_from_state(state)
    comparisons, comparison_metadata = compare_claims_with_metadata(claims)
    events = build_corroborated_events(claims, comparisons)
    review_items = _make_review_items(events)
    summary = _summarize(
        claims,
        comparisons,
        events,
        review_items,
        comparison_metadata,
    )
    annotated_records = annotate_records_with_claim_corroboration(
        _records_for_claims(state),
        claims,
        events,
    )
    return {
        "claims": claims,
        "claim_comparisons": comparisons,
        "corroborated_events": events,
        "corroboration_summary": summary,
        "human_review_items": review_items,
        "normalized_records": annotated_records,
    }
