"""Observation-type dataset views for final workflow outputs.

This module is deterministic. It does not search, fetch, call an LLM, or make
truth determinations. It separates already collected records and claims into
auditable dataset views so primary case records are not confused with
zero-case statements, exposure monitoring, context, or other non-primary
public-health observations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any

OBSERVATION_SPLIT_METHOD = "deterministic_observation_type_dataset_split_v1"

CASE_OBSERVATION_TYPES = {
    "confirmed_case_record",
    "probable_case_record",
    "suspected_case_record",
    "unspecified_case_record",
}

ZERO_CASE_STATUSES = {"zero_case_statement_unverified"}
EXPOSURE_STATUSES = {"exposure_monitoring_only"}
CONTEXT_STATUSES = {"context_only"}

NON_PRIMARY_OBSERVATION_TYPES = {
    "zero_case_statement",
    "exposure_monitoring_record",
    "background_context",
    "ambiguous_public_health_observation",
    "non_task_record",
    "surveillance_summary",
    "outbreak_summary",
}

DATASET_VIEW_KEYS = [
    "final_case_dataset",
    "global_outbreak_event_dataset",
    "regional_surveillance_dataset",
    "country_year_aggregate_dataset",
    "official_alert_dataset",
    "probable_case_dataset",
    "suspected_case_dataset",
    "unspecified_case_dataset",
    "death_dataset",
    "hospitalization_dataset",
    "zero_case_statements",
    "exposure_monitoring_records",
    "surveillance_summary_records",
    "outbreak_summary_records",
    "context_records",
    "non_primary_observations",
    "unclassified_observation_records",
]

_ALL_RECORD_INPUT_KEYS = (
    "normalized_records",
    "final_dataset_pre_quality_gate",
    "final_dataset",
    "final_dataset_post_review",
    "non_primary_observations",
    "quarantined_records",
    "pending_review_records",
)

_COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
)
_EXPLICIT_PRIMARY_COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "deaths",
    "hospitalizations",
)

_EXPOSURE_TERMS = (
    "public health monitoring",
    "under monitoring",
    "being monitored",
    "monitored",
    "monitoring",
    "quarantined",
    "quarantine",
    "possible exposure",
    "potential exposure",
    "exposed",
    "aboard",
    "passenger",
    "passengers",
    "contacts observed",
    "contact tracing",
    "contacts",
    "remained healthy",
    "remained well",
    "no symptoms",
    "without symptoms",
    "monitored for symptoms",
    "monitor their health",
)

_ZERO_CASE_TERMS = (
    "no confirmed cases",
    "no cases reported",
    "no reported cases",
    "zero confirmed cases",
    "0 confirmed cases",
    "no human cases",
    "no hantavirus cases",
)

_CONTEXT_TERMS = (
    "fact sheet",
    "symptoms",
    "prevention",
    "about hantavirus",
    "general information",
    "transmission",
)

_NON_CASE_METRIC_CATEGORIES = {
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
_NON_CASE_METRIC_TOKENS = (
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
_DATA_QUALITY_CASE_SUBMETRIC_TOKENS = (
    "missing or unknown",
    "unknown or missing",
    "missing age",
    "unknown age",
    "missing race",
    "unknown race",
    "missing ethnicity",
    "unknown ethnicity",
    "missing birth origin",
    "unknown birth origin",
    "missing country of birth",
    "unknown country of birth",
    "missing demographic",
    "unknown demographic",
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


def _has_positive_number(record: dict, field: str) -> bool:
    value = _number(record.get(field))
    return value is not None and value > 0


def _has_positive_primary_count(record: dict) -> bool:
    return any(_has_positive_number(record, field) for field in _COUNT_FIELDS)


def _has_positive_explicit_primary_count(record: dict) -> bool:
    return any(
        _has_positive_number(record, field) for field in _EXPLICIT_PRIMARY_COUNT_FIELDS
    )


def _has_public_health_metric(record: dict) -> bool:
    return record.get("metric_value") not in (None, "") and bool(
        _lower(
            " ".join(
                str(record.get(key) or "")
                for key in (
                    "metric_name",
                    "metric_category",
                    "metric_unit",
                    "count_semantics",
                    "statistical_count_type",
                )
            )
        )
    )


def _is_non_case_public_health_metric(record: dict, claims: list[dict] | None = None) -> bool:
    if not _has_public_health_metric(record):
        return False
    category = _lower(record.get("metric_category"))
    if category in _NON_CASE_METRIC_CATEGORIES:
        return True
    unit = _lower(record.get("metric_unit"))
    if unit in {"percent", "percentage", "%", "rate"} and category not in {
        "case_count",
        "cases",
        "confirmed_case_count",
        "probable_case_count",
        "suspected_case_count",
    }:
        return True
    text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "metric_name",
                "metric_category",
                "count_semantics",
                "statistical_count_type",
                "evidence_quote",
            )
        )
    )
    if any(token in text for token in _DATA_QUALITY_CASE_SUBMETRIC_TOKENS):
        return True
    if any(token in text for token in _NON_CASE_METRIC_TOKENS):
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


def _task_is_global(state: dict) -> bool:
    structured = _as_dict(state.get("structured_task"))
    spec = _as_dict(state.get("collection_spec"))
    location = _lower(
        structured.get("location")
        or spec.get("geography")
        or spec.get("location")
    )
    return location in {"global", "worldwide", "world wide", "all"}


def _default_seasonal_flu_task(state: dict) -> bool:
    structured = _as_dict(state.get("structured_task"))
    spec = _as_dict(state.get("collection_spec"))
    disease = _lower(
        structured.get("disease")
        or structured.get("virus")
        or spec.get("disease")
        or spec.get("virus")
    )
    if disease not in {"flu", "influenza", "seasonal influenza"}:
        return False
    explicit_subtype_tokens = (
        "h5",
        "h5n1",
        "avian",
        "bird flu",
        "variant",
        "novel influenza",
    )
    request_text = _lower(
        " ".join(
            str(value or "")
            for value in (
                structured.get("user_request"),
                structured.get("raw_request"),
                spec.get("user_request"),
                spec.get("query"),
            )
        )
    )
    return not any(token in request_text for token in explicit_subtype_tokens)


def _record_has_nonseasonal_influenza_subtype(record: dict) -> bool:
    text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "disease",
                "disease_standard_name",
                "virus_or_syndrome",
                "pathogen_or_syndrome",
                "metric_name",
                "count_semantics",
                "statistical_count_type",
                "case_definition",
                "evidence_quote",
            )
        )
    )
    subtype_tokens = (
        "h5n1",
        " h5 ",
        "avian influenza",
        "bird flu",
        "variant influenza",
        "novel influenza",
        "a(h3n2)v",
        "a(h1n2)v",
        "a(h1n1)v",
    )
    return any(token in f" {text} " for token in subtype_tokens)


def _scope_type(record: dict) -> str:
    return _lower(record.get("geographic_scope_type") or record.get("aggregation_level"))


def _is_official_alert_source(record: dict) -> bool:
    publisher = _lower(record.get("actual_publisher") or record.get("publisher"))
    source_type = _lower(record.get("source_type_final") or record.get("source_type"))
    title = _lower(record.get("source_title"))
    url = _lower(record.get("source_url"))
    official_type = source_type in {
        "official_public_health_agency",
        "national_public_health_agency",
        "international_public_health_agency",
        "state_or_local_public_health_agency",
    }
    official_publisher = any(
        token in publisher
        for token in (
            "world health organization",
            "pan american health organization",
            "centers for disease control",
            "european centre for disease prevention",
            "department of health",
        )
    )
    alert_page = any(
        token in f"{title} {url}"
        for token in (
            "disease-outbreak-news",
            "epidemiological alert",
            "han",
            "rapid scientific advice",
            "annual epidemiological report",
        )
    )
    return (official_type or official_publisher) and alert_page


def _record_id(record: dict) -> str:
    return _clean(record.get("record_id") or record.get("source_record_id"))


def _merge_record(existing: dict | None, update: dict) -> dict:
    if existing is None:
        return deepcopy(update)
    out = deepcopy(existing)
    for key, value in update.items():
        if value not in (None, "", [], {}):
            out[key] = deepcopy(value)
        elif key not in out:
            out[key] = deepcopy(value)
    return out


def _records_by_id(state: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for key in _ALL_RECORD_INPUT_KEYS:
        for row in _as_list(state.get(key)):
            if not isinstance(row, dict):
                continue
            rid = _record_id(row)
            if not rid:
                continue
            records[rid] = _merge_record(records.get(rid), row)
    return records


def _id_set(rows: list) -> set[str]:
    return {
        _record_id(row)
        for row in rows
        if isinstance(row, dict) and _record_id(row)
    }


def _claims_by_record(claims: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        rid = _clean(claim.get("source_record_id"))
        if rid:
            grouped[rid].append(claim)
    return grouped


def _events_by_claim(events: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        claim_ids = (
            _as_list(event.get("supporting_claim_ids"))
            + _as_list(event.get("conflicting_claim_ids"))
            + _as_list(event.get("unverified_claim_ids"))
        )
        for claim_id in claim_ids:
            if claim_id:
                grouped[str(claim_id)].append(event)
    return grouped


def _source_by_id(sources: list[dict]) -> dict[str, dict]:
    return {
        str(row.get("source_id")): row
        for row in sources
        if isinstance(row, dict) and row.get("source_id")
    }


def _source_enriched_record(record: dict, source: dict | None) -> dict:
    out = deepcopy(record)
    if not source:
        return out
    for key in (
        "actual_publisher",
        "actual_publisher_normalized",
        "source_type_final",
        "source_independence_group",
        "claim_support_role",
        "recommended_source_role",
        "recommended_fetch_use",
        "recommended_extraction_use",
    ):
        if out.get(key) in (None, "", [], {}) and source.get(key) not in (
            None,
            "",
            [],
            {},
        ):
            out[key] = deepcopy(source.get(key))
    return out


def _record_text(record: dict, claims: list[dict]) -> str:
    pieces = [
        record.get("evidence_quote"),
        record.get("evidence_context"),
        record.get("count_semantics"),
        record.get("source_title"),
        record.get("notes"),
        record.get("claim_support_role"),
        record.get("source_type_final"),
    ]
    for claim in claims:
        pieces.extend(
            [
                claim.get("evidence_quote"),
                claim.get("evidence_context"),
                claim.get("claim_support_role"),
                claim.get("source_type_final"),
            ]
        )
    return " ".join(_clean(value) for value in pieces).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _observation_types(record: dict, claims: list[dict], events: list[dict]) -> list[str]:
    values: list[str] = []
    for claim in claims:
        if claim.get("observation_type"):
            values.append(str(claim["observation_type"]))
    for value in _as_list(record.get("observation_types")):
        if value:
            values.append(str(value))
    if record.get("observation_type"):
        values.append(str(record["observation_type"]))
    for event in events:
        if event.get("observation_type"):
            values.append(str(event["observation_type"]))
    text = _record_text(record, claims)
    statuses = {_lower(event.get("corroboration_status")) for event in events}
    statuses.add(_lower(record.get("corroboration_status")))
    if (
        (_contains_any(text, _EXPOSURE_TERMS) or statuses & EXPOSURE_STATUSES)
        and not _has_positive_explicit_primary_count(record)
    ):
        values.append("exposure_monitoring_record")
    if _contains_any(text, _ZERO_CASE_TERMS) or statuses & ZERO_CASE_STATUSES:
        values.append("zero_case_statement")
    if (
        _contains_any(text, _CONTEXT_TERMS)
        or statuses & CONTEXT_STATUSES
        or _lower(record.get("claim_support_role")) == "context_only"
        or _lower(record.get("source_type_final"))
        in {"background_fact_sheet", "public_health_context_page"}
    ):
        values.append("background_context")
    if not values:
        for field, observation_type in (
            ("cases_confirmed", "confirmed_case_record"),
            ("cases_probable", "probable_case_record"),
            ("cases_suspected", "suspected_case_record"),
            ("cases_unspecified", "unspecified_case_record"),
            ("deaths", "death_record"),
            ("hospitalizations", "hospitalization_record"),
        ):
            if _has_positive_number(record, field):
                values.append(observation_type)
                break
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _claim_ids(claims: list[dict]) -> list[str]:
    return sorted({str(claim["claim_id"]) for claim in claims if claim.get("claim_id")})


def _event_ids(events: list[dict]) -> list[str]:
    return sorted(
        {
            str(event["corroborated_event_id"])
            for event in events
            if event.get("corroborated_event_id")
        }
    )


def _primary_eligible(record: dict, claims: list[dict], events: list[dict]) -> bool:
    observation_types = _as_list(record.get("observation_types"))
    if record.get("observation_type"):
        observation_types = observation_types + [record.get("observation_type")]
    text = _record_text(record, claims)
    if _is_non_case_public_health_metric(record, claims):
        return False
    if (
        "exposure_monitoring_record" in observation_types
        or "exposure_monitoring_only" in observation_types
        or (
            any(term in text for term in _EXPOSURE_TERMS)
            and not _has_positive_explicit_primary_count(record)
        )
    ):
        return False
    if record.get("primary_case_dataset_eligible") is True:
        return True
    if any(claim.get("primary_case_dataset_eligible") is True for claim in claims):
        return True
    return any(event.get("primary_case_dataset_eligible") is True for event in events)


def _blocked_from_primary(record: dict, quarantined_ids: set[str], pending_ids: set[str]) -> bool:
    rid = _record_id(record)
    status = _lower(record.get("record_final_inclusion_status"))
    if rid in quarantined_ids or status.startswith("quarantined"):
        return True
    if rid in pending_ids or status in {"pending_review", "needs_human_review"}:
        return True
    if "rejected" in status or record.get("record_excluded_by_human_review"):
        return True
    if record.get("final_dataset_included") is False:
        return True
    return False


def _case_bearing(record: dict, observation_types: list[str]) -> bool:
    text = _record_text(record, [])
    if record.get("case_only_exclusion_reason"):
        return False
    if _is_non_case_public_health_metric(record):
        return False
    if (
        "exposure_monitoring_record" in observation_types
        or "exposure_monitoring_only" in observation_types
        or (
            any(term in text for term in _EXPOSURE_TERMS)
            and not _has_positive_explicit_primary_count(record)
        )
    ):
        return False
    if any(item in CASE_OBSERVATION_TYPES for item in observation_types):
        return True
    if "surveillance_summary" in observation_types and any(
        _has_positive_number(record, field)
        for field in (
            "cases_confirmed",
            "cases_probable",
            "cases_suspected",
            "cases_unspecified",
        )
    ):
        return True
    return False


def _enrich_for_view(record: dict, claims: list[dict], events: list[dict]) -> dict:
    row = deepcopy(record)
    claim_ids = _claim_ids(claims)
    event_ids = _event_ids(events)
    if claim_ids:
        row["claim_ids"] = sorted(set(_as_list(row.get("claim_ids")) + claim_ids))
    if event_ids:
        row["corroborated_event_ids"] = sorted(
            set(_as_list(row.get("corroborated_event_ids")) + event_ids)
        )
    if not row.get("corroboration_status") and events:
        row["corroboration_status"] = events[0].get("corroboration_status")
    if not row.get("independent_source_count") and events:
        row["independent_source_count"] = events[0].get("independent_source_count")
    if not row.get("source_independence_group"):
        groups = [
            str(group)
            for event in events
            for group in _as_list(event.get("source_independence_groups"))
            if group
        ]
        if groups:
            row["source_independence_group"] = groups[0]
    return row


def _append_unique(view: list[dict], record: dict) -> None:
    rid = _record_id(record)
    if not rid:
        return
    if any(_record_id(row) == rid for row in view):
        return
    view.append(deepcopy(record))


def _view_count_fields(views: dict[str, list[dict]]) -> dict[str, int]:
    return {key: len(views.get(key) or []) for key in DATASET_VIEW_KEYS}


def _quality_summary_fields(views: dict[str, list[dict]]) -> dict:
    return {
        "final_case_dataset_count": len(views["final_case_dataset"]),
        "global_outbreak_event_dataset_count": len(
            views["global_outbreak_event_dataset"]
        ),
        "regional_surveillance_dataset_count": len(
            views["regional_surveillance_dataset"]
        ),
        "country_year_aggregate_dataset_count": len(
            views["country_year_aggregate_dataset"]
        ),
        "official_alert_dataset_count": len(views["official_alert_dataset"]),
        "zero_case_statement_count": len(views["zero_case_statements"]),
        "exposure_monitoring_record_count": len(
            views["exposure_monitoring_records"]
        ),
        "surveillance_summary_record_count": len(
            views["surveillance_summary_records"]
        ),
        "outbreak_summary_record_count": len(views["outbreak_summary_records"]),
        "context_record_count": len(views["context_records"]),
        "death_dataset_count": len(views["death_dataset"]),
        "hospitalization_dataset_count": len(views["hospitalization_dataset"]),
        "unclassified_observation_count": len(
            views["unclassified_observation_records"]
        ),
        "primary_case_dataset_record_count": len(views["final_case_dataset"]),
        "non_primary_observation_count": len(views["non_primary_observations"]),
        "dataset_view_counts": _view_count_fields(views),
    }


def build_observation_type_dataset_split(state: dict) -> dict:
    """Build observation-type dataset views from existing workflow state."""

    records_by_id = _records_by_id(state)
    task_is_global = _task_is_global(state)
    claims = [row for row in _as_list(state.get("claims")) if isinstance(row, dict)]
    events = [
        row
        for row in _as_list(state.get("corroborated_events"))
        if isinstance(row, dict)
    ]
    claims_by_record = _claims_by_record(claims)
    events_by_claim = _events_by_claim(events)
    sources_by_id = _source_by_id(_as_list(state.get("source_registry")))
    quarantined_ids = _id_set(_as_list(state.get("quarantined_records")))
    pending_ids = _id_set(_as_list(state.get("pending_review_records")))
    accepted_ids = _id_set(_as_list(state.get("final_dataset")))

    views: dict[str, list[dict]] = {key: [] for key in DATASET_VIEW_KEYS}
    observation_type_counts: Counter[str] = Counter()
    assigned_ids: set[str] = set()
    warnings: list[str] = []

    for rid, record in sorted(records_by_id.items()):
        record_claims = claims_by_record.get(rid, [])
        related_events: list[dict] = []
        for claim in record_claims:
            related_events.extend(events_by_claim.get(str(claim.get("claim_id")), []))
        record = _source_enriched_record(record, sources_by_id.get(str(record.get("source_id"))))
        observation_types = _observation_types(record, record_claims, related_events)
        if _is_non_case_public_health_metric(record, record_claims):
            observation_types = [
                value
                for value in observation_types
                if value not in CASE_OBSERVATION_TYPES
            ]
            if "surveillance_summary" not in observation_types:
                observation_types.insert(0, "surveillance_summary")
            record["primary_case_dataset_eligible"] = False
        if _default_seasonal_flu_task(state) and _record_has_nonseasonal_influenza_subtype(record):
            observation_types = [
                value
                for value in observation_types
                if value not in CASE_OBSERVATION_TYPES
            ]
            if "surveillance_summary" not in observation_types:
                observation_types.insert(0, "surveillance_summary")
            record["primary_case_dataset_eligible"] = False
            warnings_list = _as_list(record.get("semantic_warnings"))
            if "nonseasonal_influenza_subtype_not_case_only_for_default_flu_task" not in warnings_list:
                warnings_list.append(
                    "nonseasonal_influenza_subtype_not_case_only_for_default_flu_task"
                )
            record["semantic_warnings"] = warnings_list
            record["case_only_exclusion_reason"] = (
                "nonseasonal_influenza_subtype_not_case_only_for_default_flu_task"
            )
        if observation_types:
            record["observation_types"] = observation_types
            if record.get("observation_type") in CASE_OBSERVATION_TYPES:
                record["observation_type"] = observation_types[0]
            else:
                record.setdefault("observation_type", observation_types[0])
        else:
            observation_types = ["ambiguous_public_health_observation"]
            record["observation_types"] = observation_types
            record.setdefault("observation_type", observation_types[0])
        observation_type_counts.update(observation_types)
        row = _enrich_for_view(record, record_claims, related_events)

        is_blocked = _blocked_from_primary(row, quarantined_ids, pending_ids)
        primary_eligible = _primary_eligible(row, record_claims, related_events)
        accepted_or_explicit = rid in accepted_ids or row.get("final_dataset_included") is True
        final_case_ok = (
            accepted_or_explicit
            and primary_eligible
            and not is_blocked
            and _case_bearing(row, observation_types)
        )

        if final_case_ok:
            _append_unique(views["final_case_dataset"], row)
            assigned_ids.add(rid)
            if "probable_case_record" in observation_types:
                _append_unique(views["probable_case_dataset"], row)
            if "suspected_case_record" in observation_types:
                _append_unique(views["suspected_case_dataset"], row)
            if "unspecified_case_record" in observation_types or (
                "surveillance_summary" in observation_types
                and _has_positive_number(row, "cases_unspecified")
            ):
                _append_unique(views["unspecified_case_dataset"], row)

        if task_is_global and not is_blocked and _has_positive_primary_count(row):
            scope_type = _scope_type(row)
            is_multi_scope = scope_type in {"multi_country", "multicountry", "region", "global"}
            is_outbreak = (
                "outbreak_summary" in observation_types
                or "outbreak" in _record_text(row, record_claims)
                or "disease-outbreak-news" in _lower(row.get("source_url"))
            )
            if is_multi_scope and is_outbreak:
                _append_unique(views["global_outbreak_event_dataset"], row)
                assigned_ids.add(rid)
            if scope_type in {"region", "regional"} or any(
                token in _lower(row.get("geographic_scope"))
                for token in ("americas", "europe", "european", "global")
            ):
                _append_unique(views["regional_surveillance_dataset"], row)
                assigned_ids.add(rid)
            if scope_type in {"country", "national"} and row.get("country"):
                _append_unique(views["country_year_aggregate_dataset"], row)
                assigned_ids.add(rid)
            if _is_official_alert_source(row):
                _append_unique(views["official_alert_dataset"], row)
                assigned_ids.add(rid)

        if _has_positive_number(row, "deaths") or "death_record" in observation_types:
            if not is_blocked:
                _append_unique(views["death_dataset"], row)
                assigned_ids.add(rid)
        if _has_positive_number(row, "hospitalizations") or "hospitalization_record" in observation_types:
            if not is_blocked:
                _append_unique(views["hospitalization_dataset"], row)
                assigned_ids.add(rid)
        if "zero_case_statement" in observation_types or any(
            claim.get("is_zero_case_statement") for claim in record_claims
        ):
            _append_unique(views["zero_case_statements"], row)
            assigned_ids.add(rid)
        if "exposure_monitoring_record" in observation_types or any(
            claim.get("is_exposure_monitoring_claim") for claim in record_claims
        ):
            _append_unique(views["exposure_monitoring_records"], row)
            assigned_ids.add(rid)
        if "surveillance_summary" in observation_types:
            _append_unique(views["surveillance_summary_records"], row)
            assigned_ids.add(rid)
        if "outbreak_summary" in observation_types:
            _append_unique(views["outbreak_summary_records"], row)
            assigned_ids.add(rid)
        if "background_context" in observation_types or any(
            claim.get("is_background_context_claim") for claim in record_claims
        ):
            _append_unique(views["context_records"], row)
            assigned_ids.add(rid)
        if (
            "ambiguous_public_health_observation" in observation_types
            or "non_task_record" in observation_types
        ) and not final_case_ok:
            _append_unique(views["unclassified_observation_records"], row)
            assigned_ids.add(rid)

        non_primary = (
            not final_case_ok
            and (
                row.get("primary_case_dataset_eligible") is False
                or any(item in NON_PRIMARY_OBSERVATION_TYPES for item in observation_types)
                or rid in _id_set(_as_list(state.get("non_primary_observations")))
            )
        )
        if non_primary:
            _append_unique(views["non_primary_observations"], row)
            assigned_ids.add(rid)

    claim_observation_type_counts = Counter(
        str(claim.get("observation_type") or "unknown") for claim in claims
    )
    unassigned_ids = set(records_by_id) - assigned_ids
    if unassigned_ids:
        warnings.append(
            f"{len(unassigned_ids)} records were not assigned to any observation dataset view"
        )

    dataset_view_counts = _view_count_fields(views)
    summary = {
        "method": OBSERVATION_SPLIT_METHOD,
        "dataset_view_counts": dataset_view_counts,
        "observation_type_counts": dict(observation_type_counts),
        "claim_observation_type_counts": dict(claim_observation_type_counts),
        "primary_case_dataset_eligible_count": sum(
            1
            for row in records_by_id.values()
            if row.get("primary_case_dataset_eligible") is True
        ),
        "records_assigned_count": len(assigned_ids),
        "records_unassigned_count": len(unassigned_ids),
        "warnings": warnings,
    }

    result = {key: views[key] for key in DATASET_VIEW_KEYS}
    result["observation_type_dataset_summary"] = summary
    result["quality_summary_fields"] = _quality_summary_fields(views)
    return result


def apply_observation_type_counts_to_summaries(
    run_quality_summary: dict,
    final_dataset_quality_summary: dict,
    split_result: dict,
) -> tuple[dict, dict]:
    """Return summaries enriched with observation dataset view counts."""

    fields = dict(split_result.get("quality_summary_fields") or {})
    views = fields.get("dataset_view_counts") or {}
    final_case_count = int(fields.get("final_case_dataset_count") or 0)

    run_summary = dict(run_quality_summary or {})
    final_summary = dict(final_dataset_quality_summary or {})
    accepted_count = int(
        run_summary.get("accepted_record_count")
        or run_summary.get("final_dataset_count")
        or final_summary.get("final_dataset_count")
        or 0
    )
    accepted_not_primary = accepted_count > final_case_count
    message = (
        "No primary case dataset records were accepted; inspect zero-case, "
        "exposure-monitoring, context, and other observation views."
        if final_case_count == 0
        else "Primary case dataset records are available in final_case_dataset."
    )
    enrichment = {
        **fields,
        "accepted_records_not_primary_case_dataset": accepted_not_primary,
        "no_primary_case_dataset_records": final_case_count == 0,
        "recommended_primary_dataset_message": message,
    }
    run_summary.update(enrichment)
    final_summary.update(enrichment)
    final_summary.setdefault(
        "observation_type_counts",
        split_result.get("observation_type_dataset_summary", {}).get(
            "observation_type_counts", {}
        ),
    )
    final_summary.setdefault("dataset_view_counts", views)
    run_summary.setdefault("dataset_view_counts", views)
    return run_summary, final_summary
