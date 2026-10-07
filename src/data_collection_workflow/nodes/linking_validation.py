"""Record linking, cross-source consistency, and quality-gate routing.

Links related observations, compares source claims, evaluates compatible
validation evidence, and routes unresolved findings for human review.
"""

from __future__ import annotations

import re
from collections import Counter

from ..claim_corroboration import run_claim_corroboration
from ..config import load_cross_source_consistency_policy, load_record_linking_policy
from ..anomaly_detection import detect_anomalies
from ..human_review_application import has_human_review_decision_input
from ..validation_source_compatibility import (
    INCOMPATIBLE_DISABLED_STATUS,
    LIVE_VALIDATION_PENDING_STATUS,
    NO_COMPATIBLE_STATUS,
    resolve_task_compatible_validation_records,
)
from ..models import (
    Conflict,
    CrossSourceConsistencyPolicy,
    EventCluster,
    EventClusterMember,
    FieldComparisonResult,
    HumanReviewItem,
    LinkedEvent,
    RecordLinkingPolicy,
    RecordLinkingResult,
    ValidationCase,
    ValidationComparison,
    ValidationResult,
)
from ..state import DataCollectionState, append_trace

_NUMERIC_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
)

_DUPLICATE_METHOD = "deterministic_event_clusterer"
_INCOMPATIBLE_COUNT_SEMANTICS = {
    ("annual", "weekly"),
    ("annual", "newly_reported"),
    ("annual", "cumulative"),
    ("annual", "historical_total"),
    ("cumulative", "newly_reported"),
    ("cumulative", "weekly"),
    ("historical_total", "newly_reported"),
    ("historical_total", "weekly"),
    ("newly_reported", "weekly"),
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _blank_to_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s if s else None
    return value


def _has_case_or_death_data(record: dict) -> bool:
    return any(record.get(f) is not None for f in _NUMERIC_FIELDS)


_MONTH_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "sept": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

_RE_ANCHOR_ISO_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_ANCHOR_ISO_YM = re.compile(r"^\d{4}-\d{2}$")
_RE_ANCHOR_YEAR = re.compile(r"^\d{4}$")
_RE_ANCHOR_MONTH_YEAR = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")

# Cutoff-style prefixes commonly emitted by LLMs inside `reporting_period`
# (e.g. "as of December 2020") — strip them so we can extract a clean anchor.
_DATE_ANCHOR_PREFIXES = (
    "reported through",
    "as of",
    "through",
    "as at",
    "up to",
)

# Step 16.1 backward-compat: previously-deployed policy files may not list
# as_of_date / reporting_period in date_anchor_preference. Treat them as
# implicit fallbacks so linking works even with older policy snapshots.
_FALLBACK_DATE_ANCHOR_FIELDS = ("as_of_date", "reporting_period")

_AGGREGATE_GEOGRAPHIC_SCOPE_TYPES = {"region", "multi_country", "global"}

# Step 16.1.1: lightweight alias map sufficient for linking-side decisions.
# Kept in sync with the canonical maps in extraction.py / normalization.py.
_GEOGRAPHIC_SCOPE_TYPE_LINKING_ALIASES: dict[str, str] = {
    "country": "country",
    "national": "country",
    "nation": "country",
    "single country": "country",
    "single_country": "country",
    "subnational": "subnational",
    "sub-national": "subnational",
    "sub national": "subnational",
    "sub_national": "subnational",
    "state": "subnational",
    "province": "subnational",
    "region": "region",
    "regional": "region",
    "multi_country": "multi_country",
    "multi-country": "multi_country",
    "multi country": "multi_country",
    "multicountry": "multi_country",
    "multinational": "multi_country",
    "multi-national": "multi_country",
    "multi national": "multi_country",
    "global": "global",
    "worldwide": "global",
    "world wide": "global",
    "world-wide": "global",
    "unknown": "unknown",
}

_STATISTICAL_COUNT_TYPE_LINKING_ALIASES: dict[str, str] = {
    "cumulative": "cumulative",
    "total": "cumulative",
    "cumulative total": "cumulative",
    "cumulative_total": "cumulative",
    "annual": "annual",
    "yearly": "annual",
    "newly_reported": "newly_reported",
    "newly reported": "newly_reported",
    "newly-reported": "newly_reported",
    "new cases": "newly_reported",
    "additional cases": "newly_reported",
    "historical_total": "historical_total",
    "historical total": "historical_total",
    "historic total": "historical_total",
    "reported through": "historical_total",
    "subset": "subset",
    "subgroup": "subset",
    "of which": "subset",
    "unknown": "unknown",
}


def _canonicalize_geographic_scope_type_for_linking(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    key = re.sub(r"\s+", " ", stripped.lower())
    return _GEOGRAPHIC_SCOPE_TYPE_LINKING_ALIASES.get(key, stripped)


def _canonicalize_statistical_count_type_for_linking(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    key = re.sub(r"\s+", " ", stripped.lower())
    return _STATISTICAL_COUNT_TYPE_LINKING_ALIASES.get(key, stripped)


def _normalize_date_anchor_value(value):
    """Best-effort canonicalization of a value selected as the date anchor.

    Handles ISO formats and common natural-language forms emitted by LLMs in
    `reporting_period` / `as_of_date` such as ``"December 2020"``,
    ``"through December 2020"``, and ``"as of December 2020"``.
    """

    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    lowered = s.lower()
    for prefix in _DATE_ANCHOR_PREFIXES:
        if lowered.startswith(prefix):
            s = s[len(prefix) :].strip()
            lowered = s.lower()
            break
    if not s:
        return None

    if _RE_ANCHOR_ISO_YMD.fullmatch(s):
        return s
    if _RE_ANCHOR_ISO_YM.fullmatch(s):
        return s
    if _RE_ANCHOR_YEAR.fullmatch(s):
        return s

    m = _RE_ANCHOR_MONTH_YEAR.fullmatch(s)
    if m:
        month = _MONTH_TO_NUM.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}"

    return s


def _date_anchor(
    record: dict,
    policy: RecordLinkingPolicy,
) -> tuple[str | None, str | None, list[str]]:
    fields = list(policy.date_anchor_preference)
    for fallback in _FALLBACK_DATE_ANCHOR_FIELDS:
        if fallback not in fields:
            fields.append(fallback)
    for field in fields:
        value = _blank_to_none(record.get(field))
        if value:
            normalized = _normalize_date_anchor_value(value)
            if normalized:
                return str(normalized), field, [f"selected_date_anchor:{field}"]
    return None, None, []


def _normalized_key_part(value, missing_token: str) -> str:
    if value is None:
        return missing_token
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return missing_token
        return re.sub(r"\s+", " ", s).lower()
    return str(value).lower()


def _unique_non_empty(values: list) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        result.append(s)
    return result


# ---------------------------------------------------------------------------
# Event-key construction
# ---------------------------------------------------------------------------


def _build_event_key(
    record: dict,
    policy: RecordLinkingPolicy,
) -> tuple[str, dict, list[str], list[str], str | None, str | None]:
    actions: list[str] = []
    warnings: list[str] = []

    disease = _blank_to_none(record.get("disease"))
    virus = _blank_to_none(record.get("virus_or_syndrome"))
    country = _blank_to_none(record.get("country"))
    subnational = _blank_to_none(record.get("subnational_location"))
    date_anchor, date_anchor_field, date_actions = _date_anchor(record, policy)
    actions.extend(date_actions)

    # Step 16 semantic fields. Step 16.1.1 canonicalizes the enum-like ones
    # before they enter the event key so hyphen / underscore / free-text
    # variants from the LLM do not split otherwise-identical events.
    statistical_count_type = _canonicalize_statistical_count_type_for_linking(
        _blank_to_none(record.get("statistical_count_type"))
    )
    reporting_period = _blank_to_none(record.get("reporting_period"))
    geographic_scope = _blank_to_none(record.get("geographic_scope"))
    geographic_scope_type = _canonicalize_geographic_scope_type_for_linking(
        _blank_to_none(record.get("geographic_scope_type"))
    )

    if virus is None:
        actions.append("used_unspecified_virus_or_syndrome_token")
    if subnational is None:
        actions.append("used_unspecified_subnational_location_token")
    if country is None or date_anchor is None:
        actions.append("used_missing_value_token")

    components = {
        "disease": _normalized_key_part(disease, policy.missing_value_token),
        "virus_or_syndrome": _normalized_key_part(virus, policy.unspecified_virus_token),
        "country": _normalized_key_part(country, policy.missing_value_token),
        "subnational_location": _normalized_key_part(
            subnational, policy.unspecified_subnational_token
        ),
        "date_anchor": _normalized_key_part(date_anchor, policy.missing_value_token),
        "statistical_count_type": _normalized_key_part(
            statistical_count_type, policy.missing_value_token
        ),
        "reporting_period": _normalized_key_part(
            reporting_period, policy.missing_value_token
        ),
        "geographic_scope": _normalized_key_part(
            geographic_scope, policy.missing_value_token
        ),
        "geographic_scope_type": _normalized_key_part(
            geographic_scope_type, policy.missing_value_token
        ),
    }
    event_key = (
        f"disease={components['disease']}|"
        f"virus_or_syndrome={components['virus_or_syndrome']}|"
        f"country={components['country']}|"
        f"subnational_location={components['subnational_location']}|"
        f"date_anchor={components['date_anchor']}|"
        f"statistical_count_type={components['statistical_count_type']}|"
        f"reporting_period={components['reporting_period']}|"
        f"geographic_scope={components['geographic_scope']}|"
        f"geographic_scope_type={components['geographic_scope_type']}"
    )

    if country is None and _has_case_or_death_data(record):
        if (
            geographic_scope
            and geographic_scope_type in _AGGREGATE_GEOGRAPHIC_SCOPE_TYPES
        ):
            warnings.append("regional_or_aggregate_geographic_scope")
        else:
            warnings.append("missing_country_for_case_data")
    if date_anchor is None:
        warnings.append("missing_date_anchor")
    if "non_country_geography_used_as_country" in (
        record.get("normalization_warnings") or []
    ):
        warnings.append("non_country_geography")
    if record.get("requires_human_review"):
        warnings.append("existing_record_requires_human_review")

    return event_key, components, actions, warnings, date_anchor, date_anchor_field


def _is_linkable_record(
    record: dict,
    key_components: dict,  # noqa: ARG001 — reserved for future LLM linker
    policy: RecordLinkingPolicy,
) -> tuple[bool, list[str]]:
    if not _blank_to_none(record.get("disease")):
        return False, ["missing_disease"]
    country = _blank_to_none(record.get("country"))
    subnational = _blank_to_none(record.get("subnational_location"))
    date_anchor, _, _ = _date_anchor(record, policy)
    # Step 16.1.1: any non-empty geographic_scope (including canonical
    # aggregate scopes such as EU/EEA or Europe) counts as valid geography
    # for linking, so regional records still produce a linkable event.
    geographic_scope = _blank_to_none(record.get("geographic_scope"))
    if (
        country is None
        and subnational is None
        and date_anchor is None
        and geographic_scope is None
    ):
        return False, ["insufficient_event_key"]
    return True, []


# ---------------------------------------------------------------------------
# LinkedEvent assembly
# ---------------------------------------------------------------------------


def _status_for_event(
    records: list[dict],
    warnings: list[str],
    policy: RecordLinkingPolicy,
) -> tuple[str, bool]:
    review_triggers = set(policy.review_triggers or [])
    if "insufficient_event_key" in warnings:
        return "not_linkable", True
    if any(w in review_triggers for w in warnings):
        return "needs_review", True
    if len(records) == 1:
        return "single_record_event", False
    source_ids = {r.get("source_id") for r in records if _blank_to_none(r.get("source_id"))}
    source_urls = {r.get("source_url") for r in records if _blank_to_none(r.get("source_url"))}
    if len(source_ids) > 1 or len(source_urls) > 1:
        return "linked_multi_source_event", False
    return "linked_multi_record_event", False


def _event_id_for_index(index: int) -> str:
    return f"event_{index:03d}"


def _make_linked_event(
    event_id: str,
    event_key: str,
    records: list[dict],
    key_components: dict,
    policy: RecordLinkingPolicy,
    warnings: list[str],
) -> LinkedEvent:
    sorted_records = sorted(records, key=lambda r: r.get("record_id") or "")
    record_ids = [r.get("record_id") for r in sorted_records if r.get("record_id")]

    source_ids = _unique_non_empty([r.get("source_id") for r in sorted_records])
    source_urls = _unique_non_empty([r.get("source_url") for r in sorted_records])
    source_types = _unique_non_empty([r.get("source_type") for r in sorted_records])
    publishers = _unique_non_empty([r.get("publisher") for r in sorted_records])

    status, requires_review = _status_for_event(sorted_records, warnings, policy)

    basis: list[str] = ["same_disease"]
    if key_components["virus_or_syndrome"] != policy.unspecified_virus_token:
        basis.append("same_virus_or_syndrome")
    if key_components["country"] != policy.missing_value_token:
        basis.append("same_country")
    if key_components["subnational_location"] != policy.unspecified_subnational_token:
        basis.append("same_subnational_location")
    if key_components["date_anchor"] != policy.missing_value_token:
        basis.append("same_date_anchor")
    if key_components.get("statistical_count_type", policy.missing_value_token) != policy.missing_value_token:
        basis.append("same_statistical_count_type")
    if key_components.get("reporting_period", policy.missing_value_token) != policy.missing_value_token:
        basis.append("same_reporting_period")
    if key_components.get("geographic_scope", policy.missing_value_token) != policy.missing_value_token:
        basis.append("same_geographic_scope")

    country_present = key_components["country"] != policy.missing_value_token
    date_present = key_components["date_anchor"] != policy.missing_value_token
    if status in {"needs_review", "not_linkable"}:
        confidence = 0.60
    elif len(sorted_records) > 1 and country_present and date_present:
        confidence = 0.95
    elif len(sorted_records) > 1:
        confidence = 0.85
    else:
        confidence = 0.75

    representative = record_ids[0] if record_ids else None
    first = sorted_records[0] if sorted_records else {}

    return LinkedEvent(
        linked_event_id=event_id,
        record_ids=record_ids,
        linking_basis=basis,
        linking_confidence=confidence,
        event_key=event_key,
        disease=first.get("disease"),
        virus_or_syndrome=first.get("virus_or_syndrome"),
        country=first.get("country"),
        subnational_location=first.get("subnational_location"),
        date_anchor=first.get("date_anchor"),
        date_anchor_field=first.get("date_anchor_field"),
        record_count=len(sorted_records),
        source_ids=source_ids,
        source_urls=source_urls,
        source_types=source_types,
        publishers=publishers,
        representative_record_id=representative,
        linking_status=status,
        linking_method=policy.linking_method,
        linking_warnings=list(warnings),
        requires_human_review=requires_review,
    )


# ---------------------------------------------------------------------------
# Duplicate detection and event-cluster assembly
# ---------------------------------------------------------------------------


def _count_values(record: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in _NUMERIC_FIELDS:
        value = record.get(field)
        if value in (None, ""):
            continue
        try:
            values[field] = float(value)
        except (TypeError, ValueError):
            continue
    return values


def _count_signature(record: dict) -> tuple[tuple[str, float], ...]:
    values = _count_values(record)
    return tuple(sorted(values.items()))


def _source_type_rank(value: str | None) -> int:
    ranks = {
        "official_public_health_agency": 0,
        "international_organization_report": 1,
        "peer_reviewed_literature": 2,
        "structured_database": 3,
        "news_and_situation_report": 4,
    }
    return ranks.get(value or "", 20)


def _source_role_rank(value: str | None) -> int:
    ranks = {
        "collection": 0,
        "validation": 1,
        "collection_support": 2,
        "context": 3,
        "context_only": 3,
        "validation_reserved": 4,
    }
    return ranks.get(value or "", 20)


def _record_completeness_score(record: dict) -> int:
    fields = (
        "disease",
        "country",
        "subnational_location",
        "date_anchor",
        "date_reported",
        "source_url",
        "evidence_quote",
        "supporting_chunk_id",
        "statistical_count_type",
        "count_semantics",
    )
    score = sum(1 for field in fields if _blank_to_none(record.get(field)) is not None)
    score += len(_count_values(record))
    return score


def _select_representative(records: list[dict]) -> tuple[dict | None, str]:
    if not records:
        return None, "no_records"

    ranked = sorted(
        records,
        key=lambda r: (
            _source_role_rank(r.get("source_role_final")),
            _source_type_rank(r.get("source_type")),
            -float(r.get("credibility_score") or 0.0),
            -_record_completeness_score(r),
            str(r.get("record_id") or ""),
        ),
    )
    representative = ranked[0]
    reason = (
        "higher_source_priority; source_role_final="
        f"{representative.get('source_role_final') or 'unknown'}; "
        f"source_type={representative.get('source_type') or 'unknown'}; "
        f"credibility_score={representative.get('credibility_score')}"
    )
    return representative, reason


def _normalized_semantics(record: dict) -> str | None:
    for field in ("statistical_count_type", "count_semantics"):
        value = _canonicalize_statistical_count_type_for_linking(
            _blank_to_none(record.get(field))
        )
        if value:
            return str(value)
    return None


def _semantics_pair_incompatible(a: str | None, b: str | None) -> bool:
    if not a or not b or a == b:
        return False
    return (a, b) in _INCOMPATIBLE_COUNT_SEMANTICS or (
        b,
        a,
    ) in _INCOMPATIBLE_COUNT_SEMANTICS


def _cluster_status_for_records(records: list[dict]) -> tuple[str, str, float, bool, list[str]]:
    if not records:
        return (
            "invalid_or_unclustered",
            "no records available for clustering",
            0.0,
            True,
            ["empty_cluster"],
        )
    if len(records) == 1:
        record = records[0]
        if not record.get("linked_event_id"):
            return (
                "invalid_or_unclustered",
                "record was not linkable",
                0.30,
                True,
                ["missing_linked_event_id"],
            )
        return (
            "singleton",
            "single normalized record in event cluster",
            0.75,
            False,
            [],
        )

    signatures = [_count_signature(record) for record in records]
    non_empty_signatures = [signature for signature in signatures if signature]
    unique_signatures = set(signatures)
    semantics = {_normalized_semantics(record) for record in records if _normalized_semantics(record)}

    if len(non_empty_signatures) < len(records):
        return (
            "related_records",
            "same event key but one or more records lack count fields",
            0.65,
            True,
            ["count_fields_incomplete"],
        )
    if len(unique_signatures) > 1:
        return (
            "conflict_needs_review",
            "same event key but count values conflict",
            0.70,
            True,
            ["count_conflict"],
        )
    if len(semantics) == 0:
        return (
            "related_records",
            "same event key and counts but count_semantics are unclear",
            0.70,
            True,
            ["count_semantics_missing"],
        )
    return (
        "duplicate_cluster",
        "same disease/location/date/count signature; duplicate-safe representative selected",
        0.95,
        False,
        [],
    )


def _location_key(record: dict) -> str:
    parts = [
        _normalized_key_part(record.get("country"), "UNKNOWN"),
        _normalized_key_part(record.get("subnational_location"), "UNKNOWN"),
        _normalized_key_part(record.get("locality"), "UNKNOWN"),
    ]
    return "|".join(parts)


def _date_key(record: dict) -> str:
    for field in ("date_anchor", "date_reported", "reporting_period", "as_of_date"):
        value = _blank_to_none(record.get(field))
        if value:
            return str(value)
    return "UNKNOWN"


def _date_keys_related(a: str, b: str) -> bool:
    if a == b:
        return True
    if _RE_ANCHOR_YEAR.fullmatch(a) and b.startswith(f"{a}-"):
        return True
    if _RE_ANCHOR_YEAR.fullmatch(b) and a.startswith(f"{b}-"):
        return True
    return False


def _records_are_related_not_duplicate(a: dict, b: dict) -> tuple[bool, str | None]:
    if _normalized_key_part(a.get("disease"), "UNKNOWN") != _normalized_key_part(
        b.get("disease"), "UNKNOWN"
    ):
        return False, None
    if _location_key(a) != _location_key(b):
        return False, None
    if not _date_keys_related(_date_key(a), _date_key(b)):
        return False, None
    sem_a = _normalized_semantics(a)
    sem_b = _normalized_semantics(b)
    if _semantics_pair_incompatible(sem_a, sem_b):
        return True, "count_semantics_incompatible"
    if sem_a != sem_b and (sem_a or sem_b):
        return True, "count_semantics_differ"
    return False, None


def _credibility_range(records: list[dict]) -> dict:
    values = [
        float(r.get("credibility_score"))
        for r in records
        if r.get("credibility_score") not in (None, "")
    ]
    if not values:
        return {}
    return {"min": min(values), "max": max(values)}


def _members_for_cluster(
    records: list[dict],
    representative_id: str | None,
    cluster_status: str,
    duplicate_reason: str,
    same_event_score: float,
) -> list[EventClusterMember]:
    members: list[EventClusterMember] = []
    for record in records:
        rid = record.get("record_id")
        if cluster_status == "duplicate_cluster":
            member_status = "representative" if rid == representative_id else "non_countable_duplicate"
            countable = rid == representative_id
            duplicate_of = None if rid == representative_id else representative_id
        elif cluster_status == "conflict_needs_review":
            member_status = "representative" if rid == representative_id else "conflicting_member"
            countable = True
            duplicate_of = None
        elif cluster_status == "related_records":
            member_status = "representative" if rid == representative_id else "related_not_merged"
            countable = True
            duplicate_of = None
        elif cluster_status == "singleton":
            member_status = "singleton"
            countable = True
            duplicate_of = None
        elif cluster_status == "not_comparable":
            member_status = "not_comparable"
            countable = True
            duplicate_of = None
        else:
            member_status = "invalid"
            countable = False
            duplicate_of = None
        members.append(
            EventClusterMember(
                record_id=rid or "",
                event_member_status=member_status,
                countable=countable,
                duplicate_of_record_id=duplicate_of,
                duplicate_detection_confidence=same_event_score,
                duplicate_detection_reason=duplicate_reason,
                source_id=record.get("source_id"),
                source_url=record.get("source_url"),
                source_type=record.get("source_type"),
                publisher=record.get("publisher"),
                evidence_quote=record.get("evidence_quote"),
            )
        )
    return members


def _make_event_cluster(
    linked_event: dict,
    records: list[dict],
) -> EventCluster:
    representative, selection_reason = _select_representative(records)
    representative_id = representative.get("record_id") if representative else None
    cluster_status, reason, score, needs_review, warnings = _cluster_status_for_records(records)
    members = _members_for_cluster(
        records, representative_id, cluster_status, reason, score
    )
    member_record_ids = [m.record_id for m in members if m.record_id]
    countable_record_ids = [m.record_id for m in members if m.countable and m.record_id]
    duplicate_record_ids = [
        m.record_id
        for m in members
        if m.event_member_status == "non_countable_duplicate" and m.record_id
    ]
    conflict_record_ids = [
        m.record_id
        for m in members
        if m.event_member_status == "conflicting_member" and m.record_id
    ]

    first = representative or (records[0] if records else {})
    values = _count_values(first)
    source_ids = _unique_non_empty([r.get("source_id") for r in records])
    source_urls = _unique_non_empty([r.get("source_url") for r in records])
    source_types = _unique_non_empty([r.get("source_type") for r in records])
    publishers = _unique_non_empty([r.get("publisher") for r in records])
    discovery_methods = _unique_non_empty([r.get("discovery_method") for r in records])
    search_providers = _unique_non_empty([r.get("search_provider") for r in records])
    source_roles = _unique_non_empty([r.get("source_role_final") for r in records])

    return EventCluster(
        event_cluster_id=linked_event.get("linked_event_id") or "",
        cluster_status=cluster_status,
        disease=first.get("disease"),
        disease_standard_name=first.get("disease_standard_name"),
        location_key=_location_key(first),
        country=first.get("country"),
        subnational_location=first.get("subnational_location"),
        locality=first.get("locality"),
        admin_level=first.get("admin_level"),
        date_key=_date_key(first),
        date_reported=first.get("date_reported"),
        event_start_date=first.get("event_start_date"),
        event_end_date=first.get("event_end_date"),
        reporting_period=first.get("reporting_period"),
        as_of_date=first.get("as_of_date"),
        statistical_count_type=first.get("statistical_count_type"),
        count_semantics=first.get("count_semantics"),
        representative_record_id=representative_id,
        representative_selection_reason=selection_reason,
        member_record_ids=member_record_ids,
        members=members,
        countable_record_ids=countable_record_ids,
        non_countable_duplicate_record_ids=duplicate_record_ids,
        related_record_ids=[],
        conflict_record_ids=conflict_record_ids,
        source_ids=source_ids,
        source_urls=source_urls,
        source_types=source_types,
        publishers=publishers,
        discovery_methods=discovery_methods,
        search_providers=search_providers,
        source_role_final_values=source_roles,
        credibility_score_range=_credibility_range(records),
        canonical_cases_confirmed=values.get("cases_confirmed"),
        canonical_cases_probable=values.get("cases_probable"),
        canonical_cases_suspected=values.get("cases_suspected"),
        canonical_cases_unspecified=values.get("cases_unspecified"),
        canonical_deaths=values.get("deaths"),
        canonical_hospitalizations=values.get("hospitalizations"),
        canonical_count_notes=reason,
        source_count=len(source_ids),
        independent_source_count=len(set(source_urls or source_ids)),
        same_event_score=score,
        cluster_reason=reason,
        duplicate_reason=reason if cluster_status == "duplicate_cluster" else None,
        needs_human_review=needs_review,
        human_review_reason=reason if needs_review else None,
        warnings=warnings,
    )


def _make_duplicate_review_item(cluster: dict, reason: str) -> HumanReviewItem:
    member_ids = list(cluster.get("member_record_ids") or [])
    cluster_id = cluster.get("event_cluster_id") or "unknown_cluster"
    return HumanReviewItem(
        review_id=f"review_duplicate_{cluster_id}",
        item_type="duplicate_event_clustering",
        related_ids=[cluster_id] + member_ids,
        reason=f"Duplicate/event clustering requires review: {reason}",
        status="pending",
        review_packet={
            "event_cluster_id": cluster_id,
            "member_record_ids": member_ids,
            "representative_record_id": cluster.get("representative_record_id"),
            "reason": reason,
            "suggested_action": "Review duplicate/related-event classification; do not apply changes automatically.",
            "source_ids": cluster.get("source_ids") or [],
            "source_urls": cluster.get("source_urls") or [],
            "count_comparison_summary": {
                "canonical_cases_confirmed": cluster.get("canonical_cases_confirmed"),
                "canonical_cases_unspecified": cluster.get("canonical_cases_unspecified"),
                "canonical_deaths": cluster.get("canonical_deaths"),
                "canonical_hospitalizations": cluster.get("canonical_hospitalizations"),
            },
        },
    )


def _annotate_records_with_clusters(
    records: list[dict],
    clusters: list[dict],
) -> list[dict]:
    member_lookup: dict[str, tuple[dict, dict]] = {}
    for cluster in clusters:
        for member in cluster.get("members") or []:
            rid = member.get("record_id")
            if rid:
                member_lookup[rid] = (cluster, member)

    updated: list[dict] = []
    for record in records:
        new_record = dict(record)
        rid = new_record.get("record_id")
        cluster, member = member_lookup.get(rid, ({}, {}))
        cluster_id = cluster.get("event_cluster_id") or new_record.get("linked_event_id")
        new_record["event_cluster_id"] = cluster_id
        if cluster_id and not new_record.get("linked_event_id"):
            new_record["linked_event_id"] = cluster_id
        new_record["event_cluster_status"] = cluster.get("cluster_status")
        new_record["event_member_status"] = member.get("event_member_status") or "invalid"
        new_record["countable"] = bool(member.get("countable", False))
        new_record["duplicate_of_record_id"] = member.get("duplicate_of_record_id")
        new_record["representative_record_id"] = cluster.get("representative_record_id")
        new_record["duplicate_detection_method"] = _DUPLICATE_METHOD
        new_record["duplicate_detection_confidence"] = member.get(
            "duplicate_detection_confidence"
        )
        new_record["duplicate_detection_reason"] = member.get(
            "duplicate_detection_reason"
        ) or cluster.get("cluster_reason")
        needs_review = bool(cluster.get("needs_human_review"))
        new_record["duplicate_review_required"] = needs_review
        new_record["duplicate_review_reason"] = (
            cluster.get("human_review_reason") if needs_review else None
        )
        new_record["event_cluster_warnings"] = list(cluster.get("warnings") or [])
        updated.append(new_record)
    return updated


def _mark_related_clusters(
    clusters: list[dict],
    record_map: dict[str, dict],
) -> tuple[list[dict], list[tuple[dict, str]]]:
    updated = [dict(cluster) for cluster in clusters]
    review_requests: list[tuple[dict, str]] = []
    for index, left in enumerate(updated):
        left_rep = record_map.get(left.get("representative_record_id"))
        if not left_rep:
            continue
        for right in updated[index + 1:]:
            right_rep = record_map.get(right.get("representative_record_id"))
            if not right_rep:
                continue
            related, reason = _records_are_related_not_duplicate(left_rep, right_rep)
            if not related:
                continue
            for cluster, other in ((left, right), (right, left)):
                cluster["cluster_status"] = (
                    "related_records"
                    if cluster.get("cluster_status") == "singleton"
                    else cluster.get("cluster_status")
                )
                cluster["needs_human_review"] = True
                cluster["human_review_reason"] = reason
                warnings = list(cluster.get("warnings") or [])
                if reason not in warnings:
                    warnings.append(reason)
                cluster["warnings"] = warnings
                related_ids = list(cluster.get("related_record_ids") or [])
                for rid in other.get("member_record_ids") or []:
                    if rid not in related_ids:
                        related_ids.append(rid)
                cluster["related_record_ids"] = related_ids
                cluster["cluster_reason"] = reason
                for member in cluster.get("members") or []:
                    if member.get("event_member_status") in {"singleton", "countable"}:
                        member["event_member_status"] = "related_not_merged"
                        member["countable"] = True
                        member["duplicate_detection_reason"] = reason
            review_requests.append((left, reason or "related_event_ambiguity"))
            review_requests.append((right, reason or "related_event_ambiguity"))
    return updated, review_requests


def _summaries_for_clusters(
    records: list[dict],
    clusters: list[dict],
    new_review_count: int,
) -> tuple[dict, dict]:
    cluster_status_counter = Counter(
        cluster.get("cluster_status") or "unknown" for cluster in clusters
    )
    member_status_counter = Counter(
        record.get("event_member_status") or "unknown" for record in records
    )
    countable_count = sum(1 for record in records if record.get("countable") is True)
    non_countable_count = sum(
        1
        for record in records
        if record.get("event_member_status") == "non_countable_duplicate"
    )
    records_with_cluster = sum(1 for record in records if record.get("event_cluster_id"))
    event_summary = {
        "input_normalized_record_count": len(records),
        "event_cluster_count": len(clusters),
        "singleton_cluster_count": cluster_status_counter.get("singleton", 0),
        "duplicate_cluster_count": cluster_status_counter.get("duplicate_cluster", 0),
        "related_cluster_count": cluster_status_counter.get("related_records", 0),
        "conflict_cluster_count": cluster_status_counter.get("conflict_needs_review", 0),
        "not_comparable_cluster_count": cluster_status_counter.get("not_comparable", 0),
        "invalid_cluster_count": cluster_status_counter.get("invalid_or_unclustered", 0),
        "countable_record_count": countable_count,
        "non_countable_duplicate_count": non_countable_count,
        "records_with_event_cluster_id_count": records_with_cluster,
        "human_review_duplicate_item_count": new_review_count,
        "cluster_status_counts": dict(cluster_status_counter),
        "event_member_status_counts": dict(member_status_counter),
        "clustering_method": _DUPLICATE_METHOD,
    }
    duplicate_summary = {
        "input_normalized_record_count": len(records),
        "duplicate_cluster_count": cluster_status_counter.get("duplicate_cluster", 0),
        "duplicate_record_count": non_countable_count,
        "non_countable_duplicate_count": non_countable_count,
        "countable_record_count": countable_count,
        "conflict_needs_review_cluster_count": cluster_status_counter.get(
            "conflict_needs_review", 0
        ),
        "related_record_cluster_count": cluster_status_counter.get("related_records", 0),
        "human_review_duplicate_item_count": new_review_count,
        "duplicate_detection_method": _DUPLICATE_METHOD,
    }
    return event_summary, duplicate_summary


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def record_linking(state: DataCollectionState) -> dict:
    """Group normalized records into candidate LinkedEvent objects."""

    policy = RecordLinkingPolicy(**load_record_linking_policy())
    normalized_records = list(state.get("normalized_records") or [])
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {item.get("review_id") for item in existing_queue}

    action_counter: Counter = Counter()
    warning_counter: Counter = Counter()
    status_counter: Counter = Counter()

    # Per-record processing first.
    groups: dict[str, list[dict]] = {}
    group_components: dict[str, dict] = {}
    group_warnings: dict[str, list[str]] = {}
    not_linkable: list[tuple[str, dict, list[str], dict]] = []
    updated_records: list[dict] = []

    for original in normalized_records:
        record = dict(original)
        norm_status = record.get("normalization_status")
        if (
            norm_status is not None
            and norm_status not in policy.allowed_input_normalization_statuses
        ):
            # Pass through without linking metadata; no event assignment.
            updated_records.append(record)
            continue

        record_warnings: list[str] = []
        if norm_status is None:
            record_warnings.append("missing_normalization_status")

        (
            event_key,
            components,
            actions,
            key_warnings,
            date_anchor,
            date_anchor_field,
        ) = _build_event_key(record, policy)

        warnings = record_warnings + key_warnings
        linkable, lnk_warnings = _is_linkable_record(record, components, policy)
        warnings.extend(lnk_warnings)

        record["event_key"] = event_key
        record["date_anchor"] = date_anchor
        record["date_anchor_field"] = date_anchor_field
        record["record_linking_actions"] = list(actions)
        record["record_linking_warnings"] = list(warnings)

        for a in actions:
            action_counter[a] += 1
        for w in warnings:
            warning_counter[w] += 1

        if linkable:
            groups.setdefault(event_key, []).append(record)
            group_components[event_key] = components
            existing_group_warnings = group_warnings.setdefault(event_key, [])
            for w in warnings:
                if w not in existing_group_warnings:
                    existing_group_warnings.append(w)
        else:
            not_linkable.append((event_key, components, warnings, record))

        updated_records.append(record)

    # Assemble linked events.
    linked_events: list[LinkedEvent] = []
    new_review_items: list[HumanReviewItem] = []
    event_index = 0

    for event_key in sorted(groups.keys()):
        event_index += 1
        event_id = _event_id_for_index(event_index)
        records_in_group = groups[event_key]
        components = group_components[event_key]
        warnings = group_warnings[event_key]

        linked_event = _make_linked_event(
            event_id, event_key, records_in_group, components, policy, warnings
        )
        linked_events.append(linked_event)
        status_counter[linked_event.linking_status] += 1

        for r in records_in_group:
            r["linked_event_id"] = event_id
            r["record_linking_status"] = linked_event.linking_status

        if linked_event.requires_human_review:
            review_id = f"review_linking_{event_id}"
            if review_id not in existing_review_ids:
                reason = (
                    "Linked event requires review: " + ", ".join(warnings)
                ) if warnings else "Linked event requires review."
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="record_linking",
                        related_ids=list(linked_event.record_ids),
                        reason=reason,
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

    # Singleton not-linkable groups follow, deterministic ordering by record_id.
    for event_key, components, warnings, record in sorted(
        not_linkable, key=lambda t: t[3].get("record_id") or ""
    ):
        event_index += 1
        event_id = _event_id_for_index(event_index)
        linked_event = _make_linked_event(
            event_id, event_key, [record], components, policy, warnings
        )
        linked_events.append(linked_event)
        status_counter[linked_event.linking_status] += 1
        record["linked_event_id"] = event_id
        record["record_linking_status"] = linked_event.linking_status

        if linked_event.requires_human_review:
            review_id = f"review_linking_{event_id}"
            if review_id not in existing_review_ids:
                reason = (
                    "Linked event requires review: " + ", ".join(warnings)
                ) if warnings else "Linked event requires review."
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="record_linking",
                        related_ids=[record.get("record_id")] if record.get("record_id") else [],
                        reason=reason,
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

    linked_event_dicts = [event.model_dump() for event in linked_events]
    record_map = _record_by_id(updated_records)
    event_cluster_dicts = [
        _make_event_cluster(
            linked_event_dict,
            _records_for_event(linked_event_dict, record_map),
        ).model_dump()
        for linked_event_dict in linked_event_dicts
    ]
    event_cluster_dicts, _ = _mark_related_clusters(
        event_cluster_dicts,
        record_map,
    )

    for cluster in event_cluster_dicts:
        if not cluster.get("needs_human_review"):
            continue
        review_item = _make_duplicate_review_item(
            cluster,
            cluster.get("human_review_reason")
            or cluster.get("cluster_reason")
            or "duplicate_or_related_event_uncertainty",
        )
        if review_item.review_id not in existing_review_ids:
            new_review_items.append(review_item)
            existing_review_ids.add(review_item.review_id)

    updated_records = _annotate_records_with_clusters(
        updated_records,
        event_cluster_dicts,
    )
    record_map = _record_by_id(updated_records)

    cluster_by_id = {
        cluster.get("event_cluster_id"): cluster
        for cluster in event_cluster_dicts
        if cluster.get("event_cluster_id")
    }
    updated_linked_events: list[dict] = []
    for linked_event_dict in linked_event_dicts:
        event_id = linked_event_dict.get("linked_event_id")
        cluster = cluster_by_id.get(event_id, {})
        if cluster:
            linked_event_dict["event_cluster_id"] = event_id
            linked_event_dict["cluster_status"] = cluster.get("cluster_status")
            linked_event_dict["representative_record_id"] = cluster.get(
                "representative_record_id"
            )
        updated_linked_events.append(linked_event_dict)

    duplicate_clusters = [
        cluster
        for cluster in event_cluster_dicts
        if cluster.get("cluster_status") == "duplicate_cluster"
    ]

    human_review_queue = list(existing_queue) + [
        item.model_dump() for item in new_review_items
    ]

    records_with_event_id = sum(1 for r in updated_records if r.get("linked_event_id"))
    single_record_event_count = sum(
        1 for e in linked_events if e.linking_status == "single_record_event"
    )
    multi_record_event_count = sum(
        1 for e in linked_events if e.linking_status == "linked_multi_record_event"
    )
    multi_source_event_count = sum(
        1 for e in linked_events if e.linking_status == "linked_multi_source_event"
    )
    needs_review_event_count = sum(
        1 for e in linked_events if e.linking_status == "needs_review"
    )
    not_linkable_event_count = sum(
        1 for e in linked_events if e.linking_status == "not_linkable"
    )

    event_clustering_summary, duplicate_detection_summary = _summaries_for_clusters(
        updated_records,
        event_cluster_dicts,
        len(
            [
                item
                for item in new_review_items
                if item.item_type == "duplicate_event_clustering"
            ]
        ),
    )

    summary = {
        "input_normalized_record_count": len(normalized_records),
        "linked_event_count": len(updated_linked_events),
        "single_record_event_count": single_record_event_count,
        "multi_record_event_count": multi_record_event_count,
        "multi_source_event_count": multi_source_event_count,
        "needs_review_event_count": needs_review_event_count,
        "not_linkable_event_count": not_linkable_event_count,
        "records_with_linked_event_id_count": records_with_event_id,
        "linking_status_counts": dict(status_counter),
        "linking_warning_counts": dict(warning_counter),
        "linking_action_counts": dict(action_counter),
        "human_review_item_count": len(new_review_items),
        "event_cluster_count": event_clustering_summary["event_cluster_count"],
        "duplicate_cluster_count": event_clustering_summary["duplicate_cluster_count"],
        "countable_record_count": event_clustering_summary["countable_record_count"],
        "non_countable_duplicate_count": event_clustering_summary[
            "non_countable_duplicate_count"
        ],
    }

    trace = append_trace(
        state,
        node_name="record_linking",
        message=(
            f"Linked {records_with_event_id}/{len(normalized_records)} normalized "
            f"records into {len(linked_events)} candidate events."
        ),
        metadata=summary,
    )
    return {
        "normalized_records": updated_records,
        "linked_events": updated_linked_events,
        "event_clusters": event_cluster_dicts,
        "duplicate_clusters": duplicate_clusters,
        "human_review_queue": human_review_queue,
        "record_linking_summary": summary,
        "event_clustering_summary": event_clustering_summary,
        "duplicate_detection_summary": duplicate_detection_summary,
        "collection_trace": trace,
    }


# ---------------------------------------------------------------------------
# Cross-source consistency helpers
# ---------------------------------------------------------------------------


def _record_by_id(records: list[dict]) -> dict[str, dict]:
    return {r.get("record_id"): r for r in records if r.get("record_id")}


def _records_for_event(linked_event: dict, record_map: dict[str, dict]) -> list[dict]:
    out: list[dict] = []
    for rid in linked_event.get("record_ids") or []:
        rec = record_map.get(rid)
        if rec is not None:
            out.append(rec)
    return out


def _canonical_compare_value(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        return re.sub(r"\s+", " ", s).lower()
    return value


def _source_priority(
    source_type: str | None,
    policy: CrossSourceConsistencyPolicy,
) -> int:
    if not source_type:
        return 99
    return int(policy.source_authority_priority.get(source_type, 99))


def _value_packet(
    record: dict,
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> dict:
    raw = record.get(field)
    return {
        "record_id": record.get("record_id"),
        "value": raw,
        "canonical_value": _canonical_compare_value(raw),
        "source_id": record.get("source_id"),
        "source_url": record.get("source_url"),
        "source_type": record.get("source_type"),
        "source_priority": _source_priority(record.get("source_type"), policy),
        "publisher": record.get("publisher"),
        "evidence_quote": record.get("evidence_quote"),
    }


def _non_null_packets(
    records: list[dict],
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> list[dict]:
    packets = []
    for r in records:
        p = _value_packet(r, field, policy)
        if p["canonical_value"] is not None:
            packets.append(p)
    return packets


def _numeric_difference_stats(values: list[float]) -> dict:
    if len(values) < 2:
        return {
            "min": values[0] if values else None,
            "max": values[0] if values else None,
            "absolute_difference": 0.0,
            "relative_difference": 0.0,
        }
    mn = min(values)
    mx = max(values)
    abs_diff = abs(mx - mn)
    denom = max(abs(mn), 1.0)
    return {
        "min": mn,
        "max": mx,
        "absolute_difference": abs_diff,
        "relative_difference": abs_diff / denom,
    }


def _ordered_unique_values(packets: list[dict]) -> list:
    seen: set = set()
    out: list = []
    for p in packets:
        cv = p["canonical_value"]
        if cv in seen:
            continue
        seen.add(cv)
        out.append(p["value"])
    return out


def _compare_numeric_field(
    linked_event_id: str,
    records: list[dict],
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> FieldComparisonResult:
    packets = _non_null_packets(records, field, policy)
    compared_ids = [p["record_id"] for p in packets if p.get("record_id")]
    if len(packets) < 2:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=_ordered_unique_values(packets),
            conflict_detected=False,
        )

    canonical_values = [p["canonical_value"] for p in packets]
    unique_canon = {round(v, 6) for v in canonical_values}
    unique_values = _ordered_unique_values(packets)
    if len(unique_canon) <= 1:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=False,
        )

    thresholds = policy.numeric_conflict_thresholds or {}
    minor_abs = float(thresholds.get("minor_absolute_difference", 1))
    minor_rel = float(thresholds.get("minor_relative_difference", 0.05))
    major_rel = float(thresholds.get("major_relative_difference", 0.20))

    stats = _numeric_difference_stats(canonical_values)
    abs_diff = stats["absolute_difference"]
    rel_diff = stats["relative_difference"]

    if abs_diff <= minor_abs or rel_diff <= minor_rel:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=True,
            conflict_type="minor_numeric_difference",
            severity="low",
            possible_reason=(
                "Different sources may use slightly different reporting cutoffs or rounding."
            ),
            recommended_action=(
                "Keep all values with conflict flag; no immediate human review required."
            ),
            requires_human_review=False,
        )
    if rel_diff >= major_rel:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=True,
            conflict_type="major_numeric_difference",
            severity="high",
            possible_reason=(
                "Sources report substantially different numeric values for the same linked event."
            ),
            recommended_action=(
                "Send to human review and verify against the highest-authority source."
            ),
            requires_human_review=True,
        )
    return FieldComparisonResult(
        linked_event_id=linked_event_id,
        field=field,
        compared_record_ids=compared_ids,
        unique_values=unique_values,
        conflict_detected=True,
        conflict_type="numeric_mismatch",
        severity="medium",
        possible_reason="Sources report different numeric values for the same linked event.",
        recommended_action="Review source definitions, reporting dates, and case definitions.",
        requires_human_review=True,
    )


_TEXT_FIELD_PROFILE: dict[str, tuple[str, str, bool, str, str]] = {
    "country": (
        "location_mismatch",
        "high",
        True,
        "Records linked into the same event have inconsistent geographic fields.",
        "Review whether records are truly the same event or should be separated.",
    ),
    "subnational_location": (
        "location_mismatch",
        "high",
        True,
        "Records linked into the same event have inconsistent geographic fields.",
        "Review whether records are truly the same event or should be separated.",
    ),
    "virus_or_syndrome": (
        "virus_or_syndrome_mismatch",
        "high",
        True,
        "Records linked into the same event refer to different virus or syndrome labels.",
        "Review disease terminology and whether records should be split.",
    ),
    "case_definition": (
        "case_definition_mismatch",
        "medium",
        False,
        "Sources may use different case definitions or reporting categories.",
        "Keep all values but flag case definition differences.",
    ),
}


def _compare_text_field(
    linked_event_id: str,
    records: list[dict],
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> FieldComparisonResult:
    packets = _non_null_packets(records, field, policy)
    compared_ids = [p["record_id"] for p in packets if p.get("record_id")]
    unique_values = _ordered_unique_values(packets)
    if len(packets) < 2:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=False,
        )
    unique_canon = {p["canonical_value"] for p in packets}
    if len(unique_canon) <= 1:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=False,
        )

    profile = _TEXT_FIELD_PROFILE.get(field)
    if profile is None:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=True,
            conflict_type="numeric_mismatch",
            severity="medium",
            possible_reason=f"Records disagree on textual field {field}.",
            recommended_action="Review source values.",
            requires_human_review=True,
        )
    conflict_type, severity, requires_review, reason, action = profile
    return FieldComparisonResult(
        linked_event_id=linked_event_id,
        field=field,
        compared_record_ids=compared_ids,
        unique_values=unique_values,
        conflict_detected=True,
        conflict_type=conflict_type,
        severity=severity,
        possible_reason=reason,
        recommended_action=action,
        requires_human_review=requires_review,
    )


def _compare_date_field(
    linked_event_id: str,
    records: list[dict],
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> FieldComparisonResult:
    packets = _non_null_packets(records, field, policy)
    compared_ids = [p["record_id"] for p in packets if p.get("record_id")]
    unique_values = _ordered_unique_values(packets)
    if len(packets) < 2:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=False,
        )
    unique_canon = {p["canonical_value"] for p in packets}
    if len(unique_canon) <= 1:
        return FieldComparisonResult(
            linked_event_id=linked_event_id,
            field=field,
            compared_record_ids=compared_ids,
            unique_values=unique_values,
            conflict_detected=False,
        )
    return FieldComparisonResult(
        linked_event_id=linked_event_id,
        field=field,
        compared_record_ids=compared_ids,
        unique_values=unique_values,
        conflict_detected=True,
        conflict_type="date_mismatch",
        severity="medium",
        possible_reason=(
            "Sources use different reporting dates, event dates, or update cutoffs."
        ),
        recommended_action=(
            "Review whether dates represent reporting date, event start, event end, or publication update."
        ),
        requires_human_review=True,
    )


def _make_conflict(
    conflict_id: str,
    comparison: FieldComparisonResult,
    records: list[dict],
    field: str,
    policy: CrossSourceConsistencyPolicy,
) -> Conflict:
    packets = _non_null_packets(records, field, policy)
    values = [
        {
            "record_id": p.get("record_id"),
            "value": p.get("value"),
            "source_id": p.get("source_id"),
            "source_url": p.get("source_url"),
            "source_type": p.get("source_type"),
            "source_priority": p.get("source_priority"),
        }
        for p in packets
    ]
    record_ids = _unique_non_empty([p.get("record_id") for p in packets])
    source_ids = _unique_non_empty([p.get("source_id") for p in packets])
    source_urls = _unique_non_empty([p.get("source_url") for p in packets])
    source_types = _unique_non_empty([p.get("source_type") for p in packets])
    evidence_quotes = _unique_non_empty([p.get("evidence_quote") for p in packets])

    warnings = ["human_review_required"] if comparison.requires_human_review else []

    return Conflict(
        conflict_id=conflict_id,
        linked_event_id=comparison.linked_event_id,
        field=field,
        values=values,
        conflict_type=comparison.conflict_type or "missing_comparable_values",
        severity=comparison.severity or "low",
        possible_reason=comparison.possible_reason,
        recommended_action=comparison.recommended_action,
        record_ids=record_ids,
        source_ids=source_ids,
        source_urls=source_urls,
        source_types=source_types,
        evidence_quotes=evidence_quotes,
        requires_human_review=comparison.requires_human_review,
        resolution_status="unresolved",
        comparison_basis="linked_event_field_comparison",
        conflict_warnings=warnings,
        created_by=policy.consistency_method,
    )


def _event_status(
    event_conflicts: list[Conflict],
    record_count: int,
) -> tuple[str, list[str], bool]:
    if record_count <= 1:
        return "single_record_no_comparison", [], False
    if not event_conflicts:
        return "consistent", [], False
    has_high = any(c.severity == "high" for c in event_conflicts)
    has_review = any(c.requires_human_review for c in event_conflicts)
    if has_high or has_review:
        return "needs_review", ["high_or_reviewable_conflicts_detected"], True
    only_low = all(c.severity == "low" for c in event_conflicts)
    if only_low:
        return "consistent_with_minor_differences", ["minor_differences_detected"], False
    return "conflict_detected", ["conflicts_detected"], False


def _conflict_record_ids(conflict_dict: dict) -> list[str]:
    explicit = conflict_dict.get("record_ids") or []
    if explicit:
        return [str(x) for x in explicit if x]
    return [
        v.get("record_id") for v in (conflict_dict.get("values") or [])
        if v.get("record_id")
    ]


def _annotate_records_with_conflicts(
    records: list[dict],
    conflicts: list[dict],
    *, replace_generated: bool = False,
) -> list[dict]:
    # Build per-record conflict lookups.
    by_record: dict[str, list[dict]] = {}
    for c in conflicts:
        for rid in _conflict_record_ids(c):
            by_record.setdefault(rid, []).append(c)

    updated: list[dict] = []
    for record in records:
        new_record = dict(record)
        existing_ids = [] if replace_generated else list(new_record.get("conflict_ids") or [])
        warnings = list(new_record.get("record_consistency_warnings") or [])
        if replace_generated:
            warnings = [w for w in warnings if w not in {"associated_with_conflict", "associated_with_reviewable_conflict"}]
        rid = new_record.get("record_id")
        associated = by_record.get(rid, [])

        added_ids: list[str] = []
        any_review = False
        for c in associated:
            cid = c.get("conflict_id")
            if cid and cid not in existing_ids and cid not in added_ids:
                added_ids.append(cid)
            if c.get("requires_human_review"):
                any_review = True

        new_conflict_ids = existing_ids + added_ids

        has_link = bool(new_record.get("linked_event_id"))
        if any_review:
            status = "needs_review"
            if "associated_with_reviewable_conflict" not in warnings:
                warnings.append("associated_with_reviewable_conflict")
            if "associated_with_conflict" not in warnings:
                warnings.append("associated_with_conflict")
        elif new_conflict_ids:
            status = "has_conflict"
            if "associated_with_conflict" not in warnings:
                warnings.append("associated_with_conflict")
        elif has_link:
            status = "no_conflict"
        else:
            status = "not_checked"

        new_record["conflict_ids"] = new_conflict_ids
        new_record["record_conflict_status"] = status
        new_record["record_consistency_warnings"] = warnings
        updated.append(new_record)
    return updated


def _make_conflict_review_item(conflict: Conflict) -> HumanReviewItem:
    related = [conflict.conflict_id] + list(conflict.record_ids or [])
    return HumanReviewItem(
        review_id=f"review_conflict_{conflict.conflict_id}",
        item_type="cross_source_conflict",
        related_ids=related,
        reason=(
            "Cross-source conflict requires review: "
            f"{conflict.field} / {conflict.conflict_type}"
        ),
        status="pending",
    )


# ---------------------------------------------------------------------------
# Stage 10 validation helpers
# ---------------------------------------------------------------------------


_VALIDATION_METHOD = "deterministic_validation_refactor_v1"
_VALIDATION_NUMERIC_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
)


def _first_non_empty(record: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = _blank_to_none(record.get(field))
        if value is not None:
            return str(value)
    return None


def _record_period(record: dict) -> str | None:
    return _first_non_empty(
        record,
        ("reporting_period", "date_anchor", "date_reported", "as_of_date"),
    )


def _record_year(record: dict) -> str | None:
    value = _record_period(record)
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(value))
    return match.group(0) if match else None


def _record_location(record: dict) -> str | None:
    return _first_non_empty(
        record,
        ("locality", "subnational_location", "geographic_scope", "country"),
    )


def _record_location_key_for_validation(record: dict) -> str:
    return _normalized_key_part(_record_location(record), "unknown")


def _record_disease_key(record: dict) -> str:
    return _normalized_key_part(
        record.get("disease_standard_name") or record.get("disease"),
        "unknown",
    )


def _task_disease(state: DataCollectionState) -> str | None:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    if isinstance(structured, dict) and structured.get("disease"):
        return str(structured.get("disease"))
    if isinstance(spec, dict) and spec.get("disease"):
        return str(spec.get("disease"))
    return None


def _task_location(state: DataCollectionState) -> str | None:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    if isinstance(structured, dict) and structured.get("location"):
        return str(structured.get("location"))
    if isinstance(spec, dict) and spec.get("geography"):
        return str(spec.get("geography"))
    return None


def _task_start_end_years(state: DataCollectionState) -> tuple[int | None, int | None]:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    start = None
    end = None
    for source in (structured, spec):
        if not isinstance(source, dict):
            continue
        if start is None and source.get("start_date"):
            match = re.search(r"\b(19|20)\d{2}\b", str(source.get("start_date")))
            if match:
                start = int(match.group(0))
        if end is None and source.get("end_date"):
            match = re.search(r"\b(19|20)\d{2}\b", str(source.get("end_date")))
            if match:
                end = int(match.group(0))
        if (start is None or end is None) and source.get("time_window"):
            years = [
                int(match.group(0))
                for match in re.finditer(r"\b(19|20)\d{2}\b", str(source.get("time_window")))
            ]
            if years:
                start = start if start is not None else min(years)
                end = end if end is not None else max(years)
    return start, end


def _disease_matches_task(record: dict, task_disease: str | None) -> bool:
    if not task_disease:
        return True
    task = _normalized_key_part(task_disease, "")
    disease = _record_disease_key(record)
    if not task or not disease:
        return True
    if task in disease or disease in task:
        return True
    aliases = {
        "hantavirus": {"hantavirus disease", "hps"},
        "covid-19": {"covid", "covid-19", "sars-cov-2"},
        "dengue": {"dengue", "denv", "dengue virus"},
    }
    for canonical, terms in aliases.items():
        if task in terms or task == canonical:
            return disease in terms or disease == canonical
    return False


def _location_matches_task(record: dict, task_location: str | None) -> bool:
    if not task_location:
        return True
    task = _normalized_key_part(task_location, "")
    if task in {"global", "worldwide", "all"}:
        return True
    candidates = [
        record.get("locality"),
        record.get("subnational_location"),
        record.get("geographic_scope"),
        record.get("country"),
    ]
    normalized = [_normalized_key_part(value, "") for value in candidates if value]
    return any(task in value or value in task for value in normalized if value)


def _count_bearing(record: dict) -> bool:
    return any(record.get(field) not in (None, "") for field in _VALIDATION_NUMERIC_FIELDS)


def _scope_issues(record: dict, state: DataCollectionState) -> list[str]:
    issues: list[str] = []
    if not _disease_matches_task(record, _task_disease(state)):
        issues.append("disease_mismatch")
    if not _location_matches_task(record, _task_location(state)):
        issues.append("outside_geography")

    start, end = _task_start_end_years(state)
    year = _record_year(record)
    if year is None and _count_bearing(record):
        issues.append("insufficient_scope_information")
    elif year is not None:
        y = int(year)
        if start is not None and y < start:
            issues.append("outside_time_window")
        if end is not None and y > end:
            issues.append("outside_time_window")

    if _count_bearing(record) and not _record_location(record):
        if "insufficient_scope_information" not in issues:
            issues.append("insufficient_scope_information")
    return issues


def _source_ids(records: list[dict]) -> list[str]:
    return _unique_non_empty([record.get("source_id") for record in records])


def _source_urls(records: list[dict]) -> list[str]:
    return _unique_non_empty([record.get("source_url") for record in records])


def _source_roles(records: list[dict]) -> list[str]:
    return _unique_non_empty([record.get("source_role_final") for record in records])


def _discovery_methods(records: list[dict]) -> list[str]:
    return _unique_non_empty([record.get("discovery_method") for record in records])


def _evidence_summary(records: list[dict]) -> str:
    quotes = _unique_non_empty([record.get("evidence_quote") for record in records])
    if quotes:
        return quotes[0]
    chunks = _unique_non_empty([record.get("supporting_chunk_id") for record in records])
    if chunks:
        return "supporting_chunk_id=" + chunks[0]
    return "No evidence quote available."


def _comparison_location(left_records: list[dict], right_records: list[dict]) -> str | None:
    return _record_location(left_records[0]) if left_records else (
        _record_location(right_records[0]) if right_records else None
    )


def _comparison_disease(left_records: list[dict], right_records: list[dict]) -> str | None:
    return (left_records[0].get("disease") if left_records else None) or (
        right_records[0].get("disease") if right_records else None
    )


def _comparison_period(left_records: list[dict], right_records: list[dict]) -> str | None:
    return _record_period(left_records[0]) if left_records else (
        _record_period(right_records[0]) if right_records else None
    )


def _comparison_stat_type(left_records: list[dict], right_records: list[dict]) -> str | None:
    return (left_records[0].get("statistical_count_type") if left_records else None) or (
        right_records[0].get("statistical_count_type") if right_records else None
    )


def _comparison_count_semantics(left_records: list[dict], right_records: list[dict]) -> str | None:
    return (left_records[0].get("count_semantics") if left_records else None) or (
        right_records[0].get("count_semantics") if right_records else None
    )


def _validation_builder() -> dict:
    return {
        "cases": [],
        "comparisons": [],
        "results": [],
        "case_index": 1,
        "comparison_index": 1,
        "result_index": 1,
    }


def _add_validation_result(
    builder: dict,
    *,
    validation_type: str,
    validation_unit: str,
    compared_field: str,
    left_records: list[dict] | None = None,
    right_records: list[dict] | None = None,
    left_event_cluster_ids: list[str] | None = None,
    right_event_cluster_ids: list[str] | None = None,
    left_value=None,
    right_value=None,
    comparability_status: str,
    match_status: str,
    validation_status: str,
    reason: str,
    confidence: float = 0.80,
    needs_human_review: bool = False,
    human_review_reason: str | None = None,
    warnings: list[str] | None = None,
    tolerance: float | None = 0.0,
) -> dict:
    left_records = list(left_records or [])
    right_records = list(right_records or [])
    left_event_cluster_ids = list(left_event_cluster_ids or [])
    right_event_cluster_ids = list(right_event_cluster_ids or [])
    warnings = list(warnings or [])

    case_id = f"val_case_{builder['case_index']:03d}"
    builder["case_index"] += 1
    comparison_id = f"val_cmp_{builder['comparison_index']:03d}"
    builder["comparison_index"] += 1
    result_id = f"val_result_{builder['result_index']:03d}"
    builder["result_index"] += 1

    case = ValidationCase(
        validation_case_id=case_id,
        validation_type=validation_type,
        validation_unit=validation_unit,
        record_ids=_unique_non_empty(
            [record.get("record_id") for record in left_records + right_records]
        ),
        event_cluster_ids=_unique_non_empty(
            left_event_cluster_ids + right_event_cluster_ids
        ),
        source_ids=_source_ids(left_records + right_records),
        source_urls=_source_urls(left_records + right_records),
        disease=_comparison_disease(left_records, right_records),
        location=_comparison_location(left_records, right_records),
        date_or_period=_comparison_period(left_records, right_records),
        reason=reason,
    ).model_dump()
    comparison = ValidationComparison(
        comparison_id=comparison_id,
        validation_case_id=case_id,
        validation_type=validation_type,
        validation_unit=validation_unit,
        left_record_ids=_unique_non_empty(
            [record.get("record_id") for record in left_records]
        ),
        right_record_ids=_unique_non_empty(
            [record.get("record_id") for record in right_records]
        ),
        left_event_cluster_ids=left_event_cluster_ids,
        right_event_cluster_ids=right_event_cluster_ids,
        left_source_ids=_source_ids(left_records),
        right_source_ids=_source_ids(right_records),
        compared_field=compared_field,
        left_value=left_value,
        right_value=right_value,
        comparability_status=comparability_status,
        reason=reason,
    ).model_dump()
    result = ValidationResult(
        validation_result_id=result_id,
        validation_case_id=case_id,
        validation_type=validation_type,
        validation_unit=validation_unit,
        comparison_id=comparison_id,
        left_record_ids=comparison["left_record_ids"],
        right_record_ids=comparison["right_record_ids"],
        left_event_cluster_ids=left_event_cluster_ids,
        right_event_cluster_ids=right_event_cluster_ids,
        left_source_ids=_source_ids(left_records),
        right_source_ids=_source_ids(right_records),
        left_source_urls=_source_urls(left_records),
        right_source_urls=_source_urls(right_records),
        left_source_roles=_source_roles(left_records),
        right_source_roles=_source_roles(right_records),
        left_discovery_methods=_discovery_methods(left_records),
        right_discovery_methods=_discovery_methods(right_records),
        compared_field=compared_field,
        disease=_comparison_disease(left_records, right_records),
        location=_comparison_location(left_records, right_records),
        geographic_scope=(
            left_records[0].get("geographic_scope") if left_records else None
        )
        or (right_records[0].get("geographic_scope") if right_records else None),
        date_or_period=_comparison_period(left_records, right_records),
        reporting_period=(
            left_records[0].get("reporting_period") if left_records else None
        )
        or (right_records[0].get("reporting_period") if right_records else None),
        as_of_date=(left_records[0].get("as_of_date") if left_records else None)
        or (right_records[0].get("as_of_date") if right_records else None),
        statistical_count_type=_comparison_stat_type(left_records, right_records),
        count_semantics=_comparison_count_semantics(left_records, right_records),
        left_value=left_value,
        right_value=right_value,
        tolerance=tolerance,
        comparability_status=comparability_status,
        match_status=match_status,
        validation_status=validation_status,
        confidence=confidence,
        reason=reason,
        evidence_summary=_evidence_summary(left_records + right_records),
        needs_human_review=needs_human_review,
        human_review_reason=human_review_reason,
        warnings=warnings,
    ).model_dump()

    builder["cases"].append(case)
    builder["comparisons"].append(comparison)
    builder["results"].append(result)
    return result


def _records_comparable_for_validation(left: dict, right: dict) -> tuple[str, str]:
    if _record_disease_key(left) != _record_disease_key(right):
        return "not_comparable", "different_disease"
    if _record_location_key_for_validation(left) != _record_location_key_for_validation(right):
        return "not_comparable", "different_location"
    left_year = _record_year(left)
    right_year = _record_year(right)
    if left_year and right_year and left_year != right_year:
        return "not_comparable", "different_time_period"
    left_sem = _normalized_semantics(left)
    right_sem = _normalized_semantics(right)
    if _semantics_pair_incompatible(left_sem, right_sem):
        return "not_comparable", "incompatible_count_semantics"
    if (left_sem or right_sem) and left_sem != right_sem:
        return "partially_comparable", "count_semantics_partially_comparable"
    return "comparable", "comparable"


def _numeric_fields_for_pair(left: dict, right: dict) -> list[str]:
    fields: list[str] = []
    for field in _VALIDATION_NUMERIC_FIELDS:
        if left.get(field) not in (None, "") or right.get(field) not in (None, ""):
            fields.append(field)
    return fields


def _number_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _canonical_independent_source_keys(records: list[dict]) -> set[str]:
    keys: set[str] = set()
    for record in records:
        source_url = _blank_to_none(record.get("source_url"))
        source_id = _blank_to_none(record.get("source_id"))
        if source_url:
            keys.add(str(source_url).strip().lower())
        elif source_id:
            keys.add(str(source_id).strip().lower())
    return keys


def _make_validation_review_item(result: dict, item_type: str) -> HumanReviewItem:
    related_ids = [result.get("validation_result_id")]
    related_ids.extend(result.get("left_record_ids") or [])
    related_ids.extend(result.get("right_record_ids") or [])
    related_ids = [str(value) for value in related_ids if value]
    return HumanReviewItem(
        review_id=f"review_validation_{result.get('validation_result_id')}",
        item_type=item_type,
        related_ids=related_ids,
        reason=result.get("human_review_reason")
        or f"Validation requires review: {result.get('reason')}",
        status="pending",
        review_packet={
            "validation_result_id": result.get("validation_result_id"),
            "validation_case_id": result.get("validation_case_id"),
            "event_cluster_id": (result.get("left_event_cluster_ids") or [None])[0],
            "record_ids": _unique_non_empty(
                (result.get("left_record_ids") or [])
                + (result.get("right_record_ids") or [])
            ),
            "source_ids": _unique_non_empty(
                (result.get("left_source_ids") or [])
                + (result.get("right_source_ids") or [])
            ),
            "source_urls": _unique_non_empty(
                (result.get("left_source_urls") or [])
                + (result.get("right_source_urls") or [])
            ),
            "compared_field": result.get("compared_field"),
            "left_value": result.get("left_value"),
            "right_value": result.get("right_value"),
            "reason": result.get("reason"),
            "suggested_action": "Review validation comparison; do not apply changes automatically.",
            "evidence_summary": result.get("evidence_summary"),
        },
    )


def _summarize_validation_results(results: list[dict]) -> tuple[dict, dict, dict]:
    type_counter = Counter(row.get("validation_type") or "unknown" for row in results)
    status_counter = Counter(row.get("validation_status") or "unknown" for row in results)
    match_counter = Counter(row.get("match_status") or "unknown" for row in results)
    comparability_counter = Counter(
        row.get("comparability_status") or "unknown" for row in results
    )
    unit_counter = Counter(row.get("validation_unit") or "unknown" for row in results)

    validation_summary = {
        "validation_method": _VALIDATION_METHOD,
        "validation_result_count": len(results),
        "validation_case_count": len({row.get("validation_case_id") for row in results}),
        "validation_comparison_count": len({row.get("comparison_id") for row in results}),
        "needs_human_review_count": sum(1 for row in results if row.get("needs_human_review")),
        "validation_type_counts": dict(type_counter),
        "validation_unit_counts": dict(unit_counter),
        "validation_status_counts": dict(status_counter),
        "match_status_counts": dict(match_counter),
        "comparability_status_counts": dict(comparability_counter),
    }
    trusted_results = [
        row
        for row in results
        if row.get("validation_type")
        in {
            "trusted_source_comparison",
            "held_out_source_comparison",
            "aggregate_comparison",
            "count_semantics_check",
        }
    ]
    trusted_summary = {
        "validation_method": _VALIDATION_METHOD,
        "trusted_source_validation_result_count": len(trusted_results),
        "trusted_source_comparison_count": sum(
            1 for row in trusted_results if row.get("validation_type") == "trusted_source_comparison"
        ),
        "aggregate_comparison_count": sum(
            1 for row in trusted_results if row.get("validation_type") == "aggregate_comparison"
        ),
        "missing_validation_count": sum(
            1 for row in trusted_results if row.get("match_status") == "missing_validation"
        ),
        "missing_collection_count": sum(
            1 for row in trusted_results if row.get("match_status") == "missing_collection"
        ),
        "not_comparable_count": sum(
            1 for row in trusted_results if row.get("match_status") == "not_comparable"
        ),
        "conflict_count": sum(1 for row in trusted_results if row.get("match_status") == "conflict"),
        "validated_count": sum(
            1 for row in trusted_results if row.get("validation_status") == "validated"
        ),
    }
    cross_results = [
        row
        for row in results
        if row.get("validation_type")
        in {"cross_source_support", "cross_source_conflict", "event_cluster_support"}
    ]
    cross_summary = {
        "validation_method": _VALIDATION_METHOD,
        "cross_source_validation_result_count": len(cross_results),
        "cross_source_supported_cluster_count": sum(
            1
            for row in cross_results
            if row.get("validation_type") == "cross_source_support"
            and row.get("validation_status") == "validated"
        ),
        "single_source_only_cluster_count": sum(
            1
            for row in cross_results
            if row.get("validation_type") == "cross_source_support"
            and row.get("match_status") == "missing_validation"
        ),
        "cross_source_conflict_count": sum(
            1 for row in cross_results if row.get("validation_type") == "cross_source_conflict"
        ),
        "needs_human_review_count": sum(1 for row in cross_results if row.get("needs_human_review")),
    }
    return validation_summary, trusted_summary, cross_summary


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def cross_source_consistency_check(state: DataCollectionState) -> dict:
    """Detect cross-source inconsistencies within each linked event."""

    policy = CrossSourceConsistencyPolicy(**load_cross_source_consistency_policy())
    normalized_records = list(state.get("normalized_records") or [])
    linked_events = list(state.get("linked_events") or [])
    event_clusters = list(state.get("event_clusters") or [])
    raw_validation_records = list(state.get("validation_records") or [])
    source_by_id = {
        str(source.get("source_id")): source
        for source in (state.get("source_registry") or [])
        if isinstance(source, dict) and source.get("source_id")
    }

    def _with_source_metadata(record: dict) -> dict:
        if not isinstance(record, dict):
            return record
        source = source_by_id.get(str(record.get("source_id"))) or {}
        if not source:
            return record
        out = dict(record)
        for key in (
            "source_role_final",
            "source_type_final",
            "actual_publisher",
            "authority_bucket",
            "publisher_type",
            "jurisdiction_scope",
            "primary_vs_secondary",
            "page_function",
        ):
            if out.get(key) in (None, "") and source.get(key) not in (None, ""):
                out[key] = source.get(key)
        return out

    normalized_records = [_with_source_metadata(record) for record in normalized_records]
    raw_validation_records = [
        _with_source_metadata(record) for record in raw_validation_records
    ]
    if state.get("validation_source_compatibility_summary") is not None:
        active_validation_records = list(state.get("active_validation_records") or [])
        inactive_validation_records = list(
            state.get("inactive_validation_records") or []
        )
        validation_source_compatibility_summary = dict(
            state.get("validation_source_compatibility_summary") or {}
        )
    else:
        resolved_validation = resolve_task_compatible_validation_records(
            validation_records=raw_validation_records,
            state_or_task_context=state,
            validation_mode=state.get("validation_mode"),
        )
        active_validation_records = list(
            resolved_validation.get("active_validation_records") or []
        )
        inactive_validation_records = list(
            resolved_validation.get("inactive_validation_records") or []
        )
        validation_source_compatibility_summary = dict(
            resolved_validation.get("validation_source_compatibility_summary") or {}
        )
    validation_mode = str(
        state.get("validation_mode")
        or validation_source_compatibility_summary.get("validation_mode")
        or "held_out_file"
    )
    if validation_mode == "live_cross_source":
        def _is_live_validation_candidate(record: dict) -> bool:
            role = (record.get("source_role_final") or "").strip().lower()
            bucket = (record.get("authority_bucket") or "").strip().lower()
            source_type = (record.get("source_type_final") or "").strip().lower()
            if role == "validation":
                return True
            if bucket in {
                "official_authority",
                "structured_database",
                "academic_source",
            }:
                return _has_case_or_death_data(record)
            return source_type in {
                "national_public_health_agency",
                "international_public_health_agency",
                "structured_database",
                "academic_or_peer_reviewed_source",
            } and _has_case_or_death_data(record)

        live_validation_records = [
            {
                **dict(record),
                "source_role_final": "validation",
                "live_validation_source_role_derivation_reason": (
                    "explicit_validation_role"
                    if (record.get("source_role_final") or "").strip().lower()
                    == "validation"
                    else "authoritative_live_source"
                ),
            }
            for record in normalized_records
            if _is_live_validation_candidate(record)
        ]
        if live_validation_records:
            raw_validation_records = list(raw_validation_records) + [
                record
                for record in live_validation_records
                if record.get("record_id")
                not in {
                    existing.get("record_id")
                    for existing in raw_validation_records
                    if isinstance(existing, dict)
                }
            ]
            active_validation_records = list(live_validation_records)
            inactive_validation_records = list(inactive_validation_records)
            validation_source_compatibility_summary = {
                **validation_source_compatibility_summary,
                "validation_mode": "live_cross_source",
                "validation_records_source": "live_fetched_records",
                "validation_records_path": None,
                "validation_records_path_used": None,
                "validation_records_loaded": True,
                "validation_records_enabled": True,
                "compatibility_status": "compatible",
                "compatibility_reason": (
                    "Task-compatible validation records were derived from "
                    "live-fetched validation sources."
                ),
                "validation_record_count": len(raw_validation_records),
                "active_validation_record_count": len(active_validation_records),
                "inactive_validation_record_count": len(inactive_validation_records),
                "validation_compatible_record_count": len(active_validation_records),
                "validation_incompatible_record_count": len(
                    inactive_validation_records
                ),
            }
        else:
            validation_source_compatibility_summary = {
                **validation_source_compatibility_summary,
                "validation_mode": "live_cross_source",
                "validation_records_source": "none",
                "validation_records_path": None,
                "validation_records_path_used": None,
                "compatibility_status": LIVE_VALIDATION_PENDING_STATUS,
                "compatibility_reason": (
                    "Live validation limited: no compatible validation source "
                    "was found in this run."
                ),
            }
    validation_records = list(active_validation_records)
    validation_compatibility_status = (
        validation_source_compatibility_summary.get("compatibility_status")
    )
    no_task_compatible_validation = (
        not validation_records
        and validation_compatibility_status
        in {
            NO_COMPATIBLE_STATUS,
            INCOMPATIBLE_DISABLED_STATUS,
            LIVE_VALIDATION_PENDING_STATUS,
        }
    )
    from ..evidence_qualification import evidence_qualification_enabled
    from ..session_runtime import fingerprint
    current_conflict_view = evidence_qualification_enabled()
    prior_conflicts = list(state.get("conflicts") or [])
    prior_conflict_ids = {c.get("conflict_id") for c in prior_conflicts}
    existing_conflicts = [c for c in prior_conflicts if not current_conflict_view
        or c.get("comparison_basis") != "linked_event_field_comparison"
        or c.get("created_by") != policy.consistency_method]

    def _current_conflict(conflict_id, comparison, records, field):
        conflict = _make_conflict(conflict_id, comparison, records, field, policy)
        if current_conflict_view:
            # A conflict is defined by source facts, not loop count or display IDs.
            keys = ("record_id", "source_id", "disease", "country", "subnational_location",
                    "locality", "geographic_scope", "reporting_period", "date_reported",
                    "metric_period_start", "metric_period_end", "count_semantics", "statistical_count_type", field)
            facts = [{k: record.get(k) for k in keys} for record in records]
            conflict.conflict_id = "conf_" + fingerprint({"field": field,
                "kind": comparison.conflict_type, "facts": sorted(facts, key=fingerprint)})[:24]
        return conflict
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {item.get("review_id") for item in existing_queue}

    record_map = _record_by_id(normalized_records)
    cluster_by_id = {
        cluster.get("event_cluster_id"): cluster
        for cluster in event_clusters
        if cluster.get("event_cluster_id")
    }
    validation_builder = _validation_builder()

    new_conflicts: list[Conflict] = []
    new_review_items: list[HumanReviewItem] = []
    updated_events: list[dict] = []

    next_conflict_index = len(existing_conflicts) + 1
    comparable_event_count = 0
    single_record_event_count = 0
    events_with_conflicts_count = 0
    events_requiring_human_review_count = 0
    event_status_counter: Counter = Counter()
    conflict_field_counter: Counter = Counter()
    conflict_type_counter: Counter = Counter()
    severity_counter: Counter = Counter()
    numeric_comparison_skipped_count = 0
    numeric_comparison_skip_reason_counter: Counter = Counter()

    # Scope validation is explicit and record-level. It never removes records.
    for record in normalized_records:
        issues = _scope_issues(record, state)
        if issues:
            status = (
                "needs_human_review"
                if "insufficient_scope_information" in issues
                else "outside_scope"
            )
            match_status = (
                "insufficient_information"
                if "insufficient_scope_information" in issues
                else "outside_requested_scope"
            )
            scope_result = _add_validation_result(
                validation_builder,
                validation_type="scope_check",
                validation_unit="scope",
                compared_field="task_scope",
                left_records=[record],
                left_event_cluster_ids=[record.get("event_cluster_id")]
                if record.get("event_cluster_id")
                else [],
                left_value={
                    "disease": record.get("disease"),
                    "location": _record_location(record),
                    "date_or_period": _record_period(record),
                },
                right_value={
                    "disease": _task_disease(state),
                    "location": _task_location(state),
                    "time_window": list(_task_start_end_years(state)),
                },
                comparability_status=(
                    "insufficient_information"
                    if "insufficient_scope_information" in issues
                    else "not_comparable"
                ),
                match_status=match_status,
                validation_status=status,
                reason=";".join(sorted(set(issues))),
                confidence=0.90,
                needs_human_review=True,
                human_review_reason="Scope validation requires review: "
                + ";".join(sorted(set(issues))),
                warnings=issues,
            )
            review_item = _make_validation_review_item(
                scope_result, "validation_scope_issue"
            )
            if review_item.review_id not in existing_review_ids:
                new_review_items.append(review_item)
                existing_review_ids.add(review_item.review_id)
        else:
            _add_validation_result(
                validation_builder,
                validation_type="scope_check",
                validation_unit="scope",
                compared_field="task_scope",
                left_records=[record],
                left_event_cluster_ids=[record.get("event_cluster_id")]
                if record.get("event_cluster_id")
                else [],
                left_value={
                    "disease": record.get("disease"),
                    "location": _record_location(record),
                    "date_or_period": _record_period(record),
                },
                right_value={
                    "disease": _task_disease(state),
                    "location": _task_location(state),
                    "time_window": list(_task_start_end_years(state)),
                },
                comparability_status="comparable",
                match_status="matched",
                validation_status="validated",
                reason="record is within requested task scope",
                confidence=0.90,
            )

    def _numeric_comparison_skip_reason(recs: list[dict]) -> str | None:
        types = {
            r.get("statistical_count_type")
            for r in recs
            if r.get("statistical_count_type")
        }
        if len(types) > 1:
            return "different_statistical_count_type"
        periods = {
            r.get("reporting_period") for r in recs if r.get("reporting_period")
        }
        if len(periods) > 1:
            return "different_reporting_period"
        return None

    for event in linked_events:
        new_event = dict(event)
        event_id = new_event.get("linked_event_id") or ""
        records_in_event = _records_for_event(new_event, record_map)
        record_count = len(records_in_event)

        event_conflicts: list[Conflict] = []
        extra_event_warnings: list[str] = []
        if record_count <= 1:
            single_record_event_count += 1
        else:
            comparable_event_count += 1
            # Step 16: skip numeric comparison when records inside this event
            # carry different statistical meanings — comparing cumulative vs
            # annual vs subset counts produces spurious "conflicts".
            skip_reason = _numeric_comparison_skip_reason(records_in_event)
            if skip_reason is not None:
                numeric_comparison_skipped_count += 1
                numeric_comparison_skip_reason_counter[skip_reason] += 1
                if skip_reason == "different_statistical_count_type":
                    extra_event_warnings.append(
                        "numeric_comparison_skipped_due_to_different_statistical_count_type"
                    )
                else:
                    extra_event_warnings.append(
                        "numeric_comparison_skipped_due_to_different_reporting_period"
                    )
            else:
                for field in policy.comparable_numeric_fields:
                    cmp = _compare_numeric_field(event_id, records_in_event, field, policy)
                    if cmp.conflict_detected:
                        conflict_id = f"conf_{next_conflict_index:03d}"
                        next_conflict_index += 1
                        conflict = _current_conflict(conflict_id, cmp, records_in_event, field)
                        new_conflicts.append(conflict)
                        event_conflicts.append(conflict)
            for field in policy.comparable_text_fields:
                cmp = _compare_text_field(event_id, records_in_event, field, policy)
                if cmp.conflict_detected:
                    conflict_id = f"conf_{next_conflict_index:03d}"
                    next_conflict_index += 1
                    conflict = _current_conflict(conflict_id, cmp, records_in_event, field)
                    new_conflicts.append(conflict)
                    event_conflicts.append(conflict)
            for field in policy.comparable_date_fields:
                cmp = _compare_date_field(event_id, records_in_event, field, policy)
                if cmp.conflict_detected:
                    conflict_id = f"conf_{next_conflict_index:03d}"
                    next_conflict_index += 1
                    conflict = _current_conflict(conflict_id, cmp, records_in_event, field)
                    new_conflicts.append(conflict)
                    event_conflicts.append(conflict)

        status, warnings, requires_review = _event_status(event_conflicts, record_count)

        # Merge in any numeric-comparison-skip warnings from Step 16.
        merged_warnings = list(warnings)
        for w in extra_event_warnings:
            if w not in merged_warnings:
                merged_warnings.append(w)

        new_event["consistency_status"] = status
        new_event["conflict_ids"] = [c.conflict_id for c in event_conflicts]
        new_event["consistency_warnings"] = merged_warnings
        new_event["checked_by"] = policy.consistency_method
        if requires_review:
            new_event["requires_human_review"] = True
        # else: preserve any earlier requires_human_review value
        event_status_counter[status] += 1
        if event_conflicts:
            events_with_conflicts_count += 1
        if requires_review:
            events_requiring_human_review_count += 1

        # Per-event review items.
        for conflict in event_conflicts:
            conflict_field_counter[conflict.field] += 1
            conflict_type_counter[conflict.conflict_type] += 1
            severity_counter[conflict.severity] += 1
            validation_conflict = _add_validation_result(
                validation_builder,
                validation_type="cross_source_conflict",
                validation_unit="field",
                compared_field=conflict.field,
                left_records=records_in_event,
                left_event_cluster_ids=[event_id] if event_id else [],
                left_value=[
                    {"record_id": value.get("record_id"), "value": value.get("value")}
                    for value in conflict.values
                ],
                comparability_status="comparable",
                match_status="conflict",
                validation_status="conflict",
                reason=conflict.possible_reason
                or f"cross-source conflict detected for {conflict.field}",
                confidence=0.85,
                needs_human_review=conflict.requires_human_review,
                human_review_reason=(
                    f"Cross-source validation conflict: {conflict.field} / "
                    f"{conflict.conflict_type}"
                )
                if conflict.requires_human_review
                else None,
                warnings=conflict.conflict_warnings,
            )
            if conflict.requires_human_review:
                review_item = _make_conflict_review_item(conflict)
                if review_item.review_id not in existing_review_ids:
                    new_review_items.append(review_item)
                    existing_review_ids.add(review_item.review_id)
                validation_review = _make_validation_review_item(
                    validation_conflict, "validation_conflict"
                )
                if validation_review.review_id not in existing_review_ids:
                    new_review_items.append(validation_review)
                    existing_review_ids.add(validation_review.review_id)

        cluster = cluster_by_id.get(event_id, {})
        support_member_ids = list(cluster.get("member_record_ids") or [])
        support_records = [
            record_map[rid] for rid in support_member_ids if rid in record_map
        ]
        if not support_records:
            support_records = list(records_in_event)
        independent_sources = _canonical_independent_source_keys(support_records)
        support_result_status = "validated" if len(independent_sources) >= 2 and not event_conflicts else "unvalidated"
        support_match_status = "matched" if support_result_status == "validated" else "missing_validation"
        support_reason = (
            "event cluster has independent cross-source support"
            if support_result_status == "validated"
            else "event cluster is single-source-only or duplicate-suppressed; no independent cross-source support"
        )
        _add_validation_result(
            validation_builder,
            validation_type="cross_source_support",
            validation_unit="event_cluster",
            compared_field="independent_source_support",
            left_records=support_records,
            left_event_cluster_ids=[event_id] if event_id else [],
            left_value=len(independent_sources),
            right_value=2,
            comparability_status="comparable",
            match_status=support_match_status,
            validation_status=support_result_status,
            reason=support_reason,
            confidence=0.80,
            warnings=[] if support_result_status == "validated" else ["single_source_only"],
        )

        updated_events.append(new_event)

    countable_collection_records = [
        record
        for record in normalized_records
        if record.get("countable") is not False
        and (record.get("source_role_final") or "collection") != "validation"
    ]
    validation_records = [
        {**record, "source_role_final": record.get("source_role_final") or "validation"}
        for record in validation_records
    ]

    matched_collection_ids: set[str] = set()
    matched_validation_ids: set[str] = set()
    if countable_collection_records and validation_records:
        for left in countable_collection_records:
            comparable_candidates: list[tuple[dict, str, str]] = []
            not_comparable_candidates: list[tuple[dict, str]] = []
            for right in validation_records:
                if (
                    left.get("record_id")
                    and left.get("record_id") == right.get("record_id")
                ) or (
                    left.get("source_id")
                    and left.get("source_id") == right.get("source_id")
                ):
                    continue
                comparability, reason = _records_comparable_for_validation(left, right)
                if comparability == "comparable":
                    comparable_candidates.append((right, comparability, reason))
                else:
                    not_comparable_candidates.append((right, reason))

            if not comparable_candidates and not_comparable_candidates:
                reason = not_comparable_candidates[0][1]
                validation_type = (
                    "count_semantics_check"
                    if "count_semantics" in reason
                    else "held_out_source_comparison"
                )
                result = _add_validation_result(
                    validation_builder,
                    validation_type=validation_type,
                    validation_unit="record",
                    compared_field="record_comparability",
                    left_records=[left],
                    right_records=[not_comparable_candidates[0][0]],
                    left_event_cluster_ids=[left.get("event_cluster_id")]
                    if left.get("event_cluster_id")
                    else [],
                    left_value=_record_period(left),
                    right_value=_record_period(not_comparable_candidates[0][0]),
                    comparability_status="not_comparable",
                    match_status="not_comparable",
                    validation_status="not_comparable",
                    reason=reason,
                    confidence=0.85,
                    needs_human_review=True,
                    human_review_reason=f"Held-out validation not comparable: {reason}",
                    warnings=[reason],
                )
                review_item = _make_validation_review_item(result, "validation_conflict")
                if review_item.review_id not in existing_review_ids:
                    new_review_items.append(review_item)
                    existing_review_ids.add(review_item.review_id)
                continue

            if not comparable_candidates:
                _add_validation_result(
                    validation_builder,
                    validation_type="held_out_source_comparison",
                    validation_unit="record",
                    compared_field="validation_counterpart",
                    left_records=[left],
                    left_event_cluster_ids=[left.get("event_cluster_id")]
                    if left.get("event_cluster_id")
                    else [],
                    comparability_status="insufficient_information",
                    match_status="missing_validation",
                    validation_status="missing_counterpart",
                    reason="no comparable held-out validation record found",
                    confidence=0.70,
                    warnings=["missing_validation"],
                )
                continue

            for right, _, _ in comparable_candidates:
                matched_collection_ids.add(str(left.get("record_id")))
                matched_validation_ids.add(str(right.get("record_id")))
                for field in _numeric_fields_for_pair(left, right):
                    left_value = _number_or_none(left.get(field))
                    right_value = _number_or_none(right.get(field))
                    if left_value is None or right_value is None:
                        match_status = "insufficient_information"
                        validation_status = "unvalidated"
                        reason = f"field {field} missing on one side"
                        needs_review = True
                    elif left_value == right_value:
                        match_status = "matched"
                        validation_status = "validated"
                        reason = f"collection and held-out validation match for {field}"
                        needs_review = False
                    else:
                        match_status = "conflict"
                        validation_status = "conflict"
                        reason = f"collection and held-out validation disagree for {field}"
                        needs_review = True
                    result = _add_validation_result(
                        validation_builder,
                        validation_type="trusted_source_comparison",
                        validation_unit="field",
                        compared_field=field,
                        left_records=[left],
                        right_records=[right],
                        left_event_cluster_ids=[left.get("event_cluster_id")]
                        if left.get("event_cluster_id")
                        else [],
                        left_value=left_value,
                        right_value=right_value,
                        comparability_status="comparable",
                        match_status=match_status,
                        validation_status=validation_status,
                        reason=reason,
                        confidence=0.90,
                        needs_human_review=needs_review,
                        human_review_reason=reason if needs_review else None,
                        warnings=["trusted_source_conflict"] if needs_review else [],
                    )
                    if needs_review:
                        review_item = _make_validation_review_item(
                            result, "validation_conflict"
                        )
                        if review_item.review_id not in existing_review_ids:
                            new_review_items.append(review_item)
                            existing_review_ids.add(review_item.review_id)

    elif countable_collection_records and not validation_records:
        if no_task_compatible_validation:
            pass
        else:
            for left in countable_collection_records:
                _add_validation_result(
                    validation_builder,
                    validation_type="held_out_source_comparison",
                    validation_unit="record",
                    compared_field="validation_counterpart",
                    left_records=[left],
                    left_event_cluster_ids=[left.get("event_cluster_id")]
                    if left.get("event_cluster_id")
                    else [],
                    comparability_status="insufficient_information",
                    match_status="missing_validation",
                    validation_status="missing_counterpart",
                    reason="no held-out validation records were available",
                    confidence=0.70,
                    warnings=["missing_validation"],
                )
    elif validation_records and not countable_collection_records:
        for right in validation_records:
            result = _add_validation_result(
                validation_builder,
                validation_type="held_out_source_comparison",
                validation_unit="record",
                compared_field="collection_counterpart",
                right_records=[right],
                comparability_status="insufficient_information",
                match_status="missing_collection",
                validation_status="missing_counterpart",
                reason="held-out validation record has no comparable collection record",
                confidence=0.70,
                needs_human_review=True,
                human_review_reason="Validation source indicates missing collection counterpart.",
                warnings=["missing_collection"],
            )
            review_item = _make_validation_review_item(result, "validation_conflict")
            if review_item.review_id not in existing_review_ids:
                new_review_items.append(review_item)
                existing_review_ids.add(review_item.review_id)

    # Aggregate comparison uses Stage 9 countable records only.
    if countable_collection_records and validation_records:
        for right in validation_records:
            comparable_left = [
                left
                for left in countable_collection_records
                if _records_comparable_for_validation(left, right)[0] == "comparable"
            ]
            if not comparable_left:
                continue
            for field in _VALIDATION_NUMERIC_FIELDS:
                if right.get(field) in (None, ""):
                    continue
                left_values = [
                    _number_or_none(left.get(field))
                    for left in comparable_left
                    if _number_or_none(left.get(field)) is not None
                ]
                if not left_values:
                    continue
                left_total = float(sum(float(v) for v in left_values))
                right_value = _number_or_none(right.get(field))
                match_status = "matched" if left_total == right_value else "conflict"
                validation_status = "validated" if match_status == "matched" else "conflict"
                reason = (
                    f"countable aggregate matches held-out validation for {field}"
                    if match_status == "matched"
                    else f"countable aggregate conflicts with held-out validation for {field}"
                )
                result = _add_validation_result(
                    validation_builder,
                    validation_type="aggregate_comparison",
                    validation_unit="aggregate",
                    compared_field=field,
                    left_records=comparable_left,
                    right_records=[right],
                    left_event_cluster_ids=_unique_non_empty(
                        [left.get("event_cluster_id") for left in comparable_left]
                    ),
                    left_value=left_total,
                    right_value=right_value,
                    comparability_status="comparable",
                    match_status=match_status,
                    validation_status=validation_status,
                    reason=reason,
                    confidence=0.88,
                    needs_human_review=match_status == "conflict",
                    human_review_reason=reason if match_status == "conflict" else None,
                    warnings=["aggregate_conflict"] if match_status == "conflict" else [],
                )
                if result.get("needs_human_review"):
                    review_item = _make_validation_review_item(
                        result, "validation_conflict"
                    )
                    if review_item.review_id not in existing_review_ids:
                        new_review_items.append(review_item)
                        existing_review_ids.add(review_item.review_id)

    for left in countable_collection_records:
        if str(left.get("record_id")) not in matched_collection_ids and validation_records:
            # Ensure missing counterpart is explicit even when other records matched.
            if not any(
                row.get("match_status") == "missing_validation"
                and left.get("record_id") in (row.get("left_record_ids") or [])
                for row in validation_builder["results"]
            ):
                _add_validation_result(
                    validation_builder,
                    validation_type="held_out_source_comparison",
                    validation_unit="record",
                    compared_field="validation_counterpart",
                    left_records=[left],
                    left_event_cluster_ids=[left.get("event_cluster_id")]
                    if left.get("event_cluster_id")
                    else [],
                    comparability_status="insufficient_information",
                    match_status="missing_validation",
                    validation_status="missing_counterpart",
                    reason="no comparable held-out validation counterpart matched this collection record",
                    confidence=0.70,
                    warnings=["missing_validation"],
                )
    for right in validation_records:
        if str(right.get("record_id")) not in matched_validation_ids and countable_collection_records:
            if not any(
                row.get("match_status") == "missing_collection"
                and right.get("record_id") in (row.get("right_record_ids") or [])
                for row in validation_builder["results"]
            ):
                _add_validation_result(
                    validation_builder,
                    validation_type="held_out_source_comparison",
                    validation_unit="record",
                    compared_field="collection_counterpart",
                    right_records=[right],
                    comparability_status="insufficient_information",
                    match_status="missing_collection",
                    validation_status="missing_counterpart",
                    reason="no comparable collection counterpart matched this validation record",
                    confidence=0.70,
                    warnings=["missing_collection"],
                )

    combined_conflict_dicts = list(existing_conflicts) + [c.model_dump() for c in new_conflicts]
    annotated_records = _annotate_records_with_conflicts(
        normalized_records, combined_conflict_dicts, replace_generated=current_conflict_view
    )

    record_status_counter: Counter = Counter()
    records_with_conflicts_count = 0
    for r in annotated_records:
        status = r.get("record_conflict_status") or "not_checked"
        record_status_counter[status] += 1
        if status in {"has_conflict", "needs_review"}:
            records_with_conflicts_count += 1

    human_review_queue = list(existing_queue) + [
        item.model_dump() for item in new_review_items
    ]

    claim_state = dict(state)
    claim_state.update(
        {
            "normalized_records": annotated_records,
            "linked_events": updated_events,
            "event_clusters": event_clusters,
            "validation_cases": validation_builder["cases"],
            "validation_comparisons": validation_builder["comparisons"],
            "validation_results": validation_builder["results"],
            "validation_records": raw_validation_records,
            "active_validation_records": active_validation_records,
            "inactive_validation_records": inactive_validation_records,
            "conflicts": combined_conflict_dicts,
            "human_review_queue": human_review_queue,
        }
    )
    if current_conflict_view:
        # A prior quality-gate projection must not overwrite this pass's facts
        # or freshly rebuilt conflict annotations inside claim corroboration.
        claim_state["final_dataset_pre_quality_gate"] = annotated_records
    claim_result = run_claim_corroboration(claim_state)
    annotated_records = claim_result.get("normalized_records") or annotated_records
    claim_review_items = []
    for item in claim_result.get("human_review_items") or []:
        review_id = item.get("review_id")
        if review_id in existing_review_ids:
            continue
        existing_review_ids.add(review_id)
        claim_review_items.append(item)
    human_review_queue = human_review_queue + claim_review_items

    summary = {
        "input_linked_event_count": len(linked_events),
        "input_normalized_record_count": len(normalized_records),
        "comparable_event_count": comparable_event_count,
        "single_record_event_count": single_record_event_count,
        "conflict_count": len(combined_conflict_dicts),
        "new_conflict_count": (sum(c.conflict_id not in prior_conflict_ids for c in new_conflicts)
                               if current_conflict_view else len(new_conflicts)),
        "events_with_conflicts_count": events_with_conflicts_count,
        "events_requiring_human_review_count": events_requiring_human_review_count,
        "records_with_conflicts_count": records_with_conflicts_count,
        "human_review_conflict_count": len(new_review_items),
        "claim_human_review_item_count": len(claim_review_items),
        "claim_count": len(claim_result.get("claims") or []),
        "claim_comparison_count": len(claim_result.get("claim_comparisons") or []),
        "corroborated_event_count": len(
            claim_result.get("corroborated_events") or []
        ),
        "conflict_field_counts": dict(conflict_field_counter),
        "conflict_type_counts": dict(conflict_type_counter),
        "severity_counts": dict(severity_counter),
        "event_consistency_status_counts": dict(event_status_counter),
        "record_conflict_status_counts": dict(record_status_counter),
        "numeric_comparison_skipped_count": numeric_comparison_skipped_count,
        "numeric_comparison_skip_reason_counts": dict(
            numeric_comparison_skip_reason_counter
        ),
    }
    validation_summary, trusted_source_validation_summary, cross_source_validation_summary = (
        _summarize_validation_results(validation_builder["results"])
    )
    validation_summary.update(
        {
            "validation_source_compatibility_status": validation_compatibility_status,
            "active_validation_record_count": len(active_validation_records),
            "inactive_validation_record_count": len(inactive_validation_records),
            "validation_record_count": len(active_validation_records),
            "raw_validation_record_count": len(raw_validation_records),
            "validation_source_compatibility_warnings": (
                validation_source_compatibility_summary.get("warnings") or []
            ),
        }
    )
    trusted_status = (
        validation_compatibility_status
        if no_task_compatible_validation
        else "completed"
    )
    trusted_source_validation_summary.update(
        {
            "status": trusted_status,
            "validation_source_compatibility_status": validation_compatibility_status,
            "active_validation_record_count": len(active_validation_records),
            "inactive_validation_record_count": len(inactive_validation_records),
            "validation_record_count": len(active_validation_records),
            "raw_validation_record_count": len(raw_validation_records),
        }
    )
    cross_source_validation_summary.update(
        {
            "validation_source_compatibility_status": validation_compatibility_status,
        }
    )
    summary.update(
        {
            "validation_result_count": validation_summary["validation_result_count"],
            "trusted_source_validation_result_count": trusted_source_validation_summary[
                "trusted_source_validation_result_count"
            ],
            "cross_source_validation_result_count": cross_source_validation_summary[
                "cross_source_validation_result_count"
            ],
            "validation_source_compatibility_status": validation_compatibility_status,
            "active_validation_record_count": len(active_validation_records),
            "inactive_validation_record_count": len(inactive_validation_records),
            "raw_validation_record_count": len(raw_validation_records),
        }
    )

    trace = append_trace(
        state,
        node_name="cross_source_consistency_check",
        message=(
            f"Checked {comparable_event_count} multi-record events; "
            f"found {summary['new_conflict_count']} new conflicts "
            f"and {validation_summary['validation_result_count']} validation results "
            f"({events_requiring_human_review_count} events need review)."
        ),
        metadata=summary,
    )
    return {
        "normalized_records": annotated_records,
        "linked_events": updated_events,
        "validation_cases": validation_builder["cases"],
        "validation_comparisons": validation_builder["comparisons"],
        "validation_results": validation_builder["results"],
        "validation_records": raw_validation_records,
        "active_validation_records": active_validation_records,
        "inactive_validation_records": inactive_validation_records,
        "validation_source_compatibility_summary": validation_source_compatibility_summary,
        "validation_summary": validation_summary,
        "trusted_source_validation_summary": trusted_source_validation_summary,
        "cross_source_validation_summary": cross_source_validation_summary,
        "claims": claim_result.get("claims") or [],
        "claim_comparisons": claim_result.get("claim_comparisons") or [],
        "corroborated_events": claim_result.get("corroborated_events") or [],
        "corroboration_summary": claim_result.get("corroboration_summary") or {},
        "conflicts": combined_conflict_dicts,
        "human_review_queue": human_review_queue,
        "cross_source_consistency_summary": summary,
        "collection_trace": trace,
    }


def quality_gate_routing(state: DataCollectionState) -> dict:
    """Route to human_review if any review items are pending, else finalize."""

    anomaly_output = detect_anomalies(state)
    human_review_enabled = bool(state.get("human_review_enabled", True))
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {
        item.get("review_id")
        for item in existing_queue
        if isinstance(item, dict) and item.get("review_id")
    }
    anomaly_review_items = []
    for item in anomaly_output.get("anomaly_review_items") or []:
        if item.get("review_id") in existing_review_ids:
            continue
        existing_review_ids.add(item.get("review_id"))
        anomaly_review_items.append(item)
    queue = existing_queue + anomaly_review_items
    if not human_review_enabled:
        route = "finalize"
        message = (
            "Human review disabled; preserving review flags as audit metadata "
            "and routing to final_data_package_builder."
        )
    elif queue:
        route = "human_review"
        message = (
            f"Human review required: {len(queue)} item(s) in human_review_queue."
        )
    elif has_human_review_decision_input(state):
        route = "human_review"
        message = "Human review decision input supplied; routing to human_review for application."
    else:
        route = "finalize"
        message = "No human review required; routing to final_data_package_builder."
    trace = append_trace(
        state,
        node_name="quality_gate_routing",
        message=message,
        metadata={
            "current_route": route,
            "human_review_enabled": human_review_enabled,
            "human_review_queue_size": len(queue),
            "anomaly_result_count": (
                anomaly_output.get("anomaly_summary") or {}
            ).get("anomaly_result_count", 0),
            "anomaly_review_item_count": len(anomaly_review_items),
        },
    )
    return {
        "current_route": route,
        "human_review_queue": queue,
        "anomaly_results": anomaly_output.get("anomaly_results") or [],
        "anomaly_summary": anomaly_output.get("anomaly_summary") or {},
        "anomaly_review_items": anomaly_review_items,
        "collection_trace": trace,
    }
