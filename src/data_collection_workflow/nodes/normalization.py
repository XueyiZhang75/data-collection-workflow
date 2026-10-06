"""Deterministic record normalization (Step 8).

Normalizes validated records' country names, dates, virus/syndrome labels,
case definitions, source types, and numeric fields. Raw input values are
preserved on `*_raw` fields so downstream record_linking can compare both
forms. No LLM and no network.
"""

from __future__ import annotations

import re
from collections import Counter

from ..config import load_record_normalization_policy
from ..disease_relevance import (
    AMBIGUOUS_DISEASE,
    INSUFFICIENT_TEXT,
    assess_record_disease_compatibility,
    build_disease_relevance_context,
    record_compatibility_fields,
    update_disease_relevance_summary,
)
from ..models import (
    ObservationRecord,
    HumanReviewItem,
    PublicHealthRecord,
    RecordNormalizationPolicy,
    RecordNormalizationResult,
)
from ..geography import resolve_record_geography
from ..numeric_semantics import sanitize_case_death_numeric_fields
from ..state import DataCollectionState, append_trace

_NORMALIZED_BY = "deterministic_record_normalization"

_NUMERIC_FIELDS = (
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

# Step 16.1.1: kept in sync with `_GEOGRAPHIC_SCOPE_TYPE_ALIASES` /
# `_STATISTICAL_COUNT_TYPE_ALIASES` in `extraction.py`. A second deterministic
# canonicalization pass at normalization time ensures records that arrive
# without going through the extraction guardrails (e.g. fixture documents,
# legacy state) still end up with the canonical enum values that downstream
# nodes rely on.
_GEOGRAPHIC_SCOPE_TYPE_ALIASES: dict[str, str] = {
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

_STATISTICAL_COUNT_TYPE_ALIASES: dict[str, str] = {
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

_AGGREGATE_GEOGRAPHIC_SCOPE_TYPES = {"region", "multi_country", "global"}


def _canonicalize_geographic_scope_type(
    value: str | None,
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return value, []
    stripped = value.strip()
    if not stripped:
        return None, []
    key = re.sub(r"\s+", " ", stripped.lower())
    canonical = _GEOGRAPHIC_SCOPE_TYPE_ALIASES.get(key)
    if canonical is None:
        return stripped, ["unrecognized_geographic_scope_type"]
    if canonical != stripped:
        return canonical, ["canonicalized_geographic_scope_type"]
    return canonical, []


def _canonicalize_statistical_count_type(
    value: str | None,
) -> tuple[str | None, list[str]]:
    if value is None:
        return None, []
    if not isinstance(value, str):
        return value, []
    stripped = value.strip()
    if not stripped:
        return None, []
    key = re.sub(r"\s+", " ", stripped.lower())
    canonical = _STATISTICAL_COUNT_TYPE_ALIASES.get(key)
    if canonical is None:
        return stripped, ["unrecognized_statistical_count_type"]
    if canonical != stripped:
        return canonical, ["canonicalized_statistical_count_type"]
    return canonical, []


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _blank_to_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _normalize_numeric(value):
    """Return float or None. Negative or unparseable inputs return None."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return None
        try:
            num = float(s)
        except (TypeError, ValueError):
            return None
        if num < 0:
            return None
        return num
    return None


def _normalize_disease(value, disease_standard_name=None) -> tuple[str | None, list[str], list[str]]:
    raw = _blank_to_none(value)
    standard = _blank_to_none(disease_standard_name)
    candidate = standard or raw
    if candidate is None:
        return raw, [], ["missing_disease"]
    lowered = str(candidate).strip().lower()
    mapping = {
        "hantavirus": "Hantavirus disease",
        "hantavirus disease": "Hantavirus disease",
        "hps": "Hantavirus disease",
        "hantavirus pulmonary syndrome": "Hantavirus disease",
        "covid": "COVID-19",
        "covid-19": "COVID-19",
        "covid 19": "COVID-19",
        "sars-cov-2": "COVID-19",
        "dengue": "Dengue",
        "dengue fever": "Dengue",
        "denv": "Dengue",
        "dengue virus": "Dengue",
    }
    canonical = mapping.get(lowered, str(candidate).strip())
    actions = []
    if raw and canonical != raw:
        actions.append("normalized_disease_name")
    return canonical, actions, []


# ---------------------------------------------------------------------------
# Country + subnational
# ---------------------------------------------------------------------------


def _normalize_country_and_subnational(
    record: dict,
    policy: RecordNormalizationPolicy,
) -> tuple[str | None, str | None, list[str], list[str], str | None, str | None]:
    actions: list[str] = []
    warnings: list[str] = []

    country = _blank_to_none(record.get("country"))
    subnational = _blank_to_none(record.get("subnational_location"))
    # Step 16 preserves any geographic_scope already set by extraction
    # guardrails (e.g. EU/EEA already moved out of country slot).
    geographic_scope = _blank_to_none(record.get("geographic_scope"))
    geographic_scope_type = _blank_to_none(record.get("geographic_scope_type"))

    country_aliases_lower = {k.lower(): v for k, v in policy.country_aliases.items()}
    canonical_values_lower = {v.lower() for v in policy.country_aliases.values()}
    non_country_lower = {t.lower(): t for t in policy.non_country_geographic_terms}
    region_terms_lower = {t.lower() for t in policy.region_geographic_terms}
    subnational_map_lower = {
        k.lower(): v for k, v in policy.subnational_country_map.items()
    }

    # 0. Region detection — runs BEFORE country alias normalization so that
    #    EU/EEA does not get unrecognized_country_name. If country slot
    #    holds a region term, move it out and set geographic_scope fields.
    if country and country.lower() in region_terms_lower:
        canonical_region = country_aliases_lower.get(country.lower(), country)
        if not geographic_scope:
            geographic_scope = canonical_region
        if not geographic_scope_type:
            geographic_scope_type = (
                "multi_country" if canonical_region == "EU/EEA" else "region"
            )
        warnings.append("regional_geographic_scope_not_country")
        country = None

    # 1. Non-country geographic term placed in the country slot.
    if country and country.lower() in non_country_lower:
        if not subnational:
            subnational = non_country_lower[country.lower()]
        warnings.append("non_country_geography_used_as_country")
        country = None

    # 2. Country alias normalization.
    if country:
        lowered = country.lower()
        if lowered in country_aliases_lower:
            canonical = country_aliases_lower[lowered]
            country = canonical
            actions.append("normalized_country_alias")
        elif lowered in canonical_values_lower:
            # Already a canonical value — no action, no warning.
            pass
        else:
            warnings.append("unrecognized_country_name")

    # 3. Infer country from subnational when country is missing.
    if not country and subnational:
        if subnational.lower() in subnational_map_lower:
            country = subnational_map_lower[subnational.lower()]
            actions.append("inferred_country_from_subnational_location")

    # 4. Single-country canonical mapped into geographic_scope when scope
    #    is not already set (preserves extraction-set scope).
    if country and not geographic_scope:
        geographic_scope = country
        if not geographic_scope_type:
            geographic_scope_type = "country"

    # 5. Both geography fields missing.
    if not country and not subnational and not geographic_scope:
        warnings.append("missing_geography")

    return country, subnational, actions, warnings, geographic_scope, geographic_scope_type


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------


_RE_ISO_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_ISO_YM = re.compile(r"^\d{4}-\d{2}$")
_RE_YEAR = re.compile(r"^\d{4}$")
_RE_MONTH_YEAR = re.compile(r"^([A-Za-z]+)\s+(\d{4})$")
_RE_YEAR_MONTH = re.compile(r"^(\d{4})\s+([A-Za-z]+)$")
_RE_MONTH_DAY_YEAR = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})$")
_RE_DAY_MONTH_YEAR = re.compile(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$")


def _normalize_date_value(
    value,
    policy: RecordNormalizationPolicy,
) -> tuple[str | None, list[str], list[str]]:
    if value is None:
        return None, [], []
    s = str(value).strip()
    if not s:
        return None, [], []

    month_aliases = (policy.date_normalization or {}).get("month_aliases", {}) or {}
    month_map = {k.lower(): v for k, v in month_aliases.items()}

    if _RE_ISO_YMD.fullmatch(s):
        return s, [], []
    if _RE_ISO_YM.fullmatch(s):
        return s, [], []
    if _RE_YEAR.fullmatch(s):
        return s, [], []

    m = _RE_MONTH_YEAR.fullmatch(s)
    if m:
        month = month_map.get(m.group(1).lower())
        if month:
            return f"{m.group(2)}-{month}", ["normalized_date"], []

    m = _RE_YEAR_MONTH.fullmatch(s)
    if m:
        month = month_map.get(m.group(2).lower())
        if month:
            return f"{m.group(1)}-{month}", ["normalized_date"], []

    m = _RE_MONTH_DAY_YEAR.fullmatch(s)
    if m:
        month = month_map.get(m.group(1).lower())
        if month:
            day = m.group(2).zfill(2)
            return f"{m.group(3)}-{month}-{day}", ["normalized_date"], []

    m = _RE_DAY_MONTH_YEAR.fullmatch(s)
    if m:
        month = month_map.get(m.group(2).lower())
        if month:
            day = m.group(1).zfill(2)
            return f"{m.group(3)}-{month}-{day}", ["normalized_date"], []

    return s, [], ["unrecognized_date_format"]


# ---------------------------------------------------------------------------
# Virus / syndrome
# ---------------------------------------------------------------------------


def _normalize_virus_or_syndrome(
    value,
    policy: RecordNormalizationPolicy,
) -> tuple[str | None, list[str], list[str]]:
    raw = _blank_to_none(value)
    if raw is None:
        return None, [], []
    aliases_lower = {k.lower(): v for k, v in policy.virus_or_syndrome_aliases.items()}
    canonical = aliases_lower.get(raw.lower())
    if canonical is not None:
        return canonical, ["normalized_virus_or_syndrome"], []
    return raw, [], ["unrecognized_virus_or_syndrome"]


# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------


def _normalize_case_definition(
    record: dict,
    policy: RecordNormalizationPolicy,
) -> tuple[str | None, list[str], list[str]]:
    actions: list[str] = []
    warnings: list[str] = []
    aliases_lower = {k.lower(): v for k, v in policy.case_definition_aliases.items()}

    raw_cd = _blank_to_none(record.get("case_definition"))
    if raw_cd:
        tokens = re.split(r"[,;/|]+", raw_cd)
        normalized: list[str] = []
        seen: set[str] = set()
        any_changed = False
        for tok in tokens:
            t = tok.strip()
            if not t:
                continue
            canonical = aliases_lower.get(t.lower())
            if canonical is not None:
                label = canonical
                if canonical != t:
                    any_changed = True
            else:
                label = t
                if "unrecognized_case_definition" not in warnings:
                    warnings.append("unrecognized_case_definition")
            if label not in seen:
                seen.add(label)
                normalized.append(label)
        if normalized:
            joined = ",".join(normalized)
            if joined != raw_cd or any_changed:
                actions.append("normalized_case_definition")
            return joined, actions, warnings
        return None, actions, warnings

    # Infer from populated case fields.
    inferred: list[str] = []
    if record.get("cases_confirmed") is not None:
        inferred.append("confirmed")
    if record.get("cases_probable") is not None:
        inferred.append("probable")
    if record.get("cases_suspected") is not None:
        inferred.append("suspected")
    if record.get("cases_unspecified") is not None:
        inferred.append("unspecified")
    if inferred:
        actions.append("inferred_case_definition_from_case_fields")
        return ",".join(inferred), actions, warnings
    return None, actions, warnings


# ---------------------------------------------------------------------------
# Source type
# ---------------------------------------------------------------------------


def _normalize_source_type(
    value,
    policy: RecordNormalizationPolicy,
) -> tuple[str | None, list[str], list[str]]:
    raw = _blank_to_none(value)
    if raw is None:
        return None, [], ["missing_source_type"]
    if raw in policy.allowed_source_types:
        return raw, [], []
    lowered = raw.lower()
    for allowed in policy.allowed_source_types:
        if allowed.lower() == lowered:
            return allowed, ["normalized_source_type_case"], []
    return raw, [], ["unrecognized_source_type"]


# ---------------------------------------------------------------------------
# Whole-record normalization
# ---------------------------------------------------------------------------


def _normalize_record(
    record: dict,
    policy: RecordNormalizationPolicy,
) -> tuple[dict, RecordNormalizationResult]:
    normalized = dict(record)
    actions: list[str] = list(record.get("normalization_actions") or [])
    warnings: list[str] = list(record.get("normalization_warnings") or [])

    # 1. Preserve raw inputs.
    for raw_field, src_field in (
        ("disease_raw", "disease"),
        ("country_raw", "country"),
        ("subnational_location_raw", "subnational_location"),
        ("locality_raw", "locality"),
        ("date_reported_raw", "date_reported"),
        ("event_start_date_raw", "event_start_date"),
        ("event_end_date_raw", "event_end_date"),
        ("reporting_period_raw", "reporting_period"),
        ("as_of_date_raw", "as_of_date"),
        ("virus_or_syndrome_raw", "virus_or_syndrome"),
        ("case_definition_raw", "case_definition"),
        ("source_type_raw", "source_type"),
    ):
        if normalized.get(raw_field) is None:
            normalized[raw_field] = record.get(src_field)

    disease, disease_actions, disease_warnings = _normalize_disease(
        normalized.get("disease"),
        normalized.get("disease_standard_name"),
    )
    normalized["disease"] = disease
    if normalized.get("disease_standard_name") is None:
        normalized["disease_standard_name"] = disease
    actions.extend(disease_actions)
    warnings.extend(disease_warnings)

    # 2. Numeric cleanup. Done before case_definition so that inference sees the
    #    cleaned values.
    for field in _NUMERIC_FIELDS:
        original = record.get(field)
        if original is None:
            continue
        cleaned = _normalize_numeric(original)
        if cleaned is None:
            normalized[field] = None
            warnings.append(f"invalid_numeric_value:{field}")
            if isinstance(original, (int, float)) and original < 0:
                if "invalid_negative_numeric_value" not in warnings:
                    warnings.append("invalid_negative_numeric_value")
            elif isinstance(original, str):
                s = original.strip().replace(",", "")
                try:
                    if float(s) < 0:
                        if "invalid_negative_numeric_value" not in warnings:
                            warnings.append("invalid_negative_numeric_value")
                except (TypeError, ValueError):
                    pass
        else:
            normalized[field] = cleaned
            if isinstance(original, str):
                actions.append(f"normalized_numeric_field:{field}")
    before_numeric_semantics = {
        field: normalized.get(field)
        for field in (
            "cases_confirmed",
            "cases_probable",
            "cases_suspected",
            "cases_unspecified",
            "deaths",
        )
    }
    from ..evidence_qualification import evidence_qualification_enabled
    if not evidence_qualification_enabled():
        # EVIDENCE preserves claims for the shared source-bound evidence qualifier.
        # Keep the destructive English-label heuristic only for older revisions.
        normalized = sanitize_case_death_numeric_fields(normalized)
    after_numeric_semantics = {
        field: normalized.get(field)
        for field in before_numeric_semantics
    }
    if after_numeric_semantics != before_numeric_semantics:
        actions.append("sanitized_case_death_numeric_semantics")
        for warning in normalized.get("semantic_warnings") or []:
            if warning not in warnings:
                warnings.append(warning)

    # 3. Country + subnational + geographic_scope.
    (
        country,
        subnational,
        c_actions,
        c_warnings,
        geographic_scope,
        geographic_scope_type,
    ) = _normalize_country_and_subnational(normalized, policy)
    normalized["country"] = country
    normalized["subnational_location"] = subnational
    if geographic_scope is not None or normalized.get("geographic_scope") is None:
        normalized["geographic_scope"] = geographic_scope
    if geographic_scope_type is not None or normalized.get("geographic_scope_type") is None:
        normalized["geographic_scope_type"] = geographic_scope_type
    actions.extend(c_actions)
    warnings.extend(c_warnings)

    # 3b. Step 16.1.1 — canonicalize semantic enum fields (idempotent if the
    # extraction guardrails already canonicalized them).
    raw_scope_type = normalized.get("geographic_scope_type")
    canonical_scope_type, scope_type_warnings = _canonicalize_geographic_scope_type(
        raw_scope_type
    )
    if canonical_scope_type != raw_scope_type:
        normalized["geographic_scope_type"] = canonical_scope_type
        if "canonicalized_geographic_scope_type" in scope_type_warnings:
            actions.append("canonicalized_geographic_scope_type")
    for w in scope_type_warnings:
        if w == "canonicalized_geographic_scope_type":
            continue
        if w not in warnings:
            warnings.append(w)

    raw_count_type = normalized.get("statistical_count_type")
    canonical_count_type, count_type_warnings = _canonicalize_statistical_count_type(
        raw_count_type
    )
    if canonical_count_type != raw_count_type:
        normalized["statistical_count_type"] = canonical_count_type
        if "canonicalized_statistical_count_type" in count_type_warnings:
            actions.append("canonicalized_statistical_count_type")
    for w in count_type_warnings:
        if w == "canonicalized_statistical_count_type":
            continue
        if w not in warnings:
            warnings.append(w)

    # 4. Dates.
    for date_field in ("date_reported", "event_start_date", "event_end_date", "as_of_date"):
        value = normalized.get(date_field)
        new_value, d_actions, d_warnings = _normalize_date_value(value, policy)
        normalized[date_field] = new_value
        actions.extend(d_actions)
        warnings.extend(d_warnings)

    # 5. Virus / syndrome.
    vs, vs_actions, vs_warnings = _normalize_virus_or_syndrome(
        normalized.get("virus_or_syndrome"), policy
    )
    normalized["virus_or_syndrome"] = vs
    actions.extend(vs_actions)
    warnings.extend(vs_warnings)

    # 6. Case definition (after numeric so inference works).
    cd, cd_actions, cd_warnings = _normalize_case_definition(normalized, policy)
    normalized["case_definition"] = cd
    actions.extend(cd_actions)
    warnings.extend(cd_warnings)

    # 7. Source type.
    st, st_actions, st_warnings = _normalize_source_type(
        normalized.get("source_type"), policy
    )
    normalized["source_type"] = st
    actions.extend(st_actions)
    warnings.extend(st_warnings)

    # 8. Missing-country-after-normalization check. Step 16.1: regional or
    #    multi-country aggregate scopes (e.g. EU/EEA, global) are valid
    #    geography and should not trigger a missing-country review.
    #    Step 16.1.1: rely on the canonicalized scope_type so "multi-country"
    #    and "national" variants from the LLM are accepted here too.
    has_case_or_death = any(
        normalized.get(f) is not None for f in _NUMERIC_FIELDS
    )
    if not normalized.get("country") and has_case_or_death:
        scope = normalized.get("geographic_scope")
        scope_type = normalized.get("geographic_scope_type")
        scope_type_canonical = (
            scope_type if isinstance(scope_type, str) else None
        )
        if scope and scope_type_canonical in _AGGREGATE_GEOGRAPHIC_SCOPE_TYPES:
            if "regional_or_aggregate_geographic_scope" not in warnings:
                warnings.append("regional_or_aggregate_geographic_scope")
        elif "missing_country_after_normalization" not in warnings:
            warnings.append("missing_country_after_normalization")

    # 9. Determine requires_human_review.
    review_triggers = set(policy.review_triggers or [])
    triggered = [w for w in warnings if w in review_triggers]
    existing_review = bool(record.get("requires_human_review"))
    requires_human_review = existing_review or bool(triggered)

    # 10. Determine normalization_status.
    if triggered:
        status = "needs_review"
    elif warnings:
        status = "normalized_with_warnings"
    else:
        status = "normalized"

    normalized["normalization_status"] = status
    normalized["normalization_actions"] = actions
    normalized["normalization_warnings"] = warnings
    normalized["normalized_by"] = _NORMALIZED_BY
    normalized["requires_human_review"] = requires_human_review

    if evidence_qualification_enabled():
        # Preserve observation facts and precision before legacy inference.
        # Qualification, not normalization, decides whether a fact is supported.
        for key in ("case_definition", "country", "subnational_location", "locality", "geographic_scope", "reporting_period", "as_of_date", "event_start_date", "event_end_date", "metric_period_start", "metric_period_end"):
            normalized[key] = record.get(key)

    # 11. Pydantic validation produces a clean canonical dict.
    validated_dict = PublicHealthRecord(**normalized).model_dump()

    result = RecordNormalizationResult(
        record_id=validated_dict.get("record_id", ""),
        normalization_status=status,
        normalization_actions=actions,
        normalization_warnings=warnings,
        requires_human_review=requires_human_review,
    )
    return validated_dict, result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def record_normalization(state: DataCollectionState) -> dict:
    """Normalize validated records and queue review items for problem records."""

    policy = RecordNormalizationPolicy(**load_record_normalization_policy())
    validated_records = list(state.get("validated_records") or [])
    source_by_id = {
        str(row.get("source_id")): row
        for row in list(state.get("source_registry") or [])
        if isinstance(row, dict) and row.get("source_id")
    }
    task_scope = (
        state.get("structured_task")
        or state.get("task")
        or state.get("collection_spec")
        or {}
    )
    disease_context = build_disease_relevance_context(state)
    existing_queue = list(state.get("human_review_queue") or [])
    existing_review_ids = {item.get("review_id") for item in existing_queue}

    normalized_records: list[dict] = []
    disease_mismatch_records: list[dict] = []
    new_review_items: list[HumanReviewItem] = []

    status_counter: Counter = Counter()
    action_counter: Counter = Counter()
    warning_counter: Counter = Counter()
    country_normalized_count = 0
    date_normalized_count = 0
    virus_normalized_count = 0
    case_def_normalized_count = 0
    source_type_warning_count = 0
    needs_review_count = 0
    disease_counter: Counter = Counter()
    source_type_counter: Counter = Counter()
    extraction_method_counter: Counter = Counter()
    generic_record_count = 0
    legacy_hantavirus_record_count = 0
    disease_mismatch_quarantined_record_count = 0
    disease_uncertain_review_record_count = 0

    country_action_set = {
        "normalized_country_alias",
        "inferred_country_from_subnational_location",
    }
    case_def_action_set = {
        "normalized_case_definition",
        "inferred_case_definition_from_case_fields",
    }
    source_type_warning_set = {"missing_source_type", "unrecognized_source_type"}

    for record in validated_records:
        compatibility = assess_record_disease_compatibility(
            record,
            disease_context,
        )
        compatibility_fields = record_compatibility_fields(compatibility)
        if compatibility.get("reject_record"):
            quarantined = dict(record)
            quarantined.update(compatibility_fields)
            quarantined["normalization_status"] = "quarantined_disease_mismatch"
            warnings = list(quarantined.get("normalization_warnings") or [])
            if "disease_mismatch_quarantined_before_normalization" not in warnings:
                warnings.append("disease_mismatch_quarantined_before_normalization")
            quarantined["normalization_warnings"] = warnings
            disease_mismatch_records.append(quarantined)
            disease_mismatch_quarantined_record_count += 1
            review_id = f"review_disease_mismatch_{record.get('record_id', '')}"
            if review_id and review_id not in existing_review_ids:
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="record_disease_relevance",
                        related_ids=[record.get("record_id", "")],
                        reason=compatibility.get("reason")
                        or "Record disease compatibility failed before normalization.",
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)
            continue
        record_for_normalization = dict(record)
        record_for_normalization.update(compatibility_fields)
        if compatibility.get("status") in {AMBIGUOUS_DISEASE, INSUFFICIENT_TEXT}:
            warnings = list(record_for_normalization.get("normalization_warnings") or [])
            if "disease_relevance_uncertain_before_normalization" not in warnings:
                warnings.append("disease_relevance_uncertain_before_normalization")
            record_for_normalization["normalization_warnings"] = warnings
            record_for_normalization["requires_human_review"] = True
            disease_uncertain_review_record_count += 1
        normalized, result = _normalize_record(record_for_normalization, policy)
        before_geography = {
            key: normalized.get(key)
            for key in (
                "country",
                "subnational_location",
                "locality",
                "geographic_scope",
                "geographic_scope_type",
            )
        }
        normalized = resolve_record_geography(
            normalized,
            source_identity=source_by_id.get(str(normalized.get("source_id") or "")),
            task=task_scope if isinstance(task_scope, dict) else {},
        )
        after_geography = {key: normalized.get(key) for key in before_geography}
        if after_geography != before_geography:
            normalized["normalization_actions"] = list(
                normalized.get("normalization_actions") or []
            ) + ["anchored_record_geography"]
            result.normalization_actions.append("anchored_record_geography")
            if normalized.get("geography_inference_warning"):
                warnings_for_record = list(normalized.get("normalization_warnings") or [])
                if "geography_inferred_from_source_or_task" not in warnings_for_record:
                    warnings_for_record.append("geography_inferred_from_source_or_task")
                normalized["normalization_warnings"] = warnings_for_record
                result.normalization_warnings.append(
                    "geography_inferred_from_source_or_task"
                )
        normalized_records.append(normalized)
        disease_counter[normalized.get("disease") or "unknown"] += 1
        source_type_counter[normalized.get("source_type") or "unknown"] += 1
        extraction_method_counter[
            normalized.get("extraction_method") or "unknown"
        ] += 1
        if normalized.get("record_schema") == "generic_public_health_record":
            generic_record_count += 1
        if normalized.get("disease") == "Hantavirus disease":
            legacy_hantavirus_record_count += 1

        status_counter[result.normalization_status] += 1
        for action in result.normalization_actions:
            action_counter[action] += 1
        for warning in result.normalization_warnings:
            warning_counter[warning] += 1

        if any(a in country_action_set for a in result.normalization_actions):
            country_normalized_count += 1
        if "normalized_date" in result.normalization_actions:
            date_normalized_count += 1
        if "normalized_virus_or_syndrome" in result.normalization_actions:
            virus_normalized_count += 1
        if any(a in case_def_action_set for a in result.normalization_actions):
            case_def_normalized_count += 1
        if any(w in source_type_warning_set for w in result.normalization_warnings):
            source_type_warning_count += 1

        if result.requires_human_review:
            needs_review_count += 1
            review_id = f"review_normalization_{result.record_id}"
            if review_id and review_id not in existing_review_ids:
                reason = (
                    "Record requires review after normalization: "
                    + ", ".join(result.normalization_warnings)
                    if result.normalization_warnings
                    else "Record requires review after normalization."
                )
                new_review_items.append(
                    HumanReviewItem(
                        review_id=review_id,
                        item_type="record_normalization",
                        related_ids=[result.record_id],
                        reason=reason,
                        status="pending",
                    )
                )
                existing_review_ids.add(review_id)

    human_review_queue = list(existing_queue) + [
        item.model_dump() for item in new_review_items
    ]

    summary = {
        "input_validated_record_count": len(validated_records),
        "normalized_record_count": len(normalized_records),
        "needs_review_count": needs_review_count,
        "normalization_status_counts": dict(status_counter),
        "normalization_action_counts": dict(action_counter),
        "normalization_warning_counts": dict(warning_counter),
        "country_normalized_count": country_normalized_count,
        "date_normalized_count": date_normalized_count,
        "virus_or_syndrome_normalized_count": virus_normalized_count,
        "case_definition_normalized_count": case_def_normalized_count,
        "source_type_warning_count": source_type_warning_count,
        "generic_record_count": generic_record_count,
        "legacy_hantavirus_record_count": legacy_hantavirus_record_count,
        "disease_counts": dict(disease_counter),
        "source_type_counts": dict(source_type_counter),
        "extraction_method_counts": dict(extraction_method_counter),
        "review_required_record_count": needs_review_count,
        "unsupported_target_field_count": 0,
        "warnings": dict(warning_counter),
        "disease_mismatch_quarantined_record_count": (
            disease_mismatch_quarantined_record_count
        ),
        "disease_uncertain_review_record_count": disease_uncertain_review_record_count,
    }

    trace = append_trace(
        state,
        node_name="record_normalization",
        message=(
            f"Normalized {len(normalized_records)}/{len(validated_records)} records "
            f"({needs_review_count} need review)."
        ),
        metadata=summary,
    )
    return {
        "normalized_records": normalized_records,
        "disease_mismatch_records": disease_mismatch_records,
        "human_review_queue": human_review_queue,
        "record_normalization_summary": summary,
        "disease_relevance_summary": update_disease_relevance_summary(
            {
                **state,
                "normalized_records": normalized_records,
                "rejected_records": list(state.get("rejected_records") or [])
                + disease_mismatch_records,
            },
            disease_mismatch_quarantined_record_count=(
                disease_mismatch_quarantined_record_count
            ),
        ),
        "collection_trace": trace,
    }
