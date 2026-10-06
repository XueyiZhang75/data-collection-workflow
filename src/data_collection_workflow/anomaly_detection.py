"""Deterministic anomaly detection for workflow trust checks.

The detector consumes already-produced workflow state. It does not fetch,
search, call LLMs, correct records, or delete records.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import re
from collections import Counter

from .models import AnomalyResult, AnomalySummary, HumanReviewItem
from .state import DataCollectionState

DETECTION_METHOD = "deterministic_anomaly_detection"

COUNT_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "icu_admissions",
    "tests_positive",
    "tests_total",
    "cumulative_count",
    "new_count",
    "incidence_rate",
    "positivity_rate",
)
CASE_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "cumulative_count",
    "new_count",
)
DATE_FIELDS = (
    "date_reported",
    "event_start_date",
    "event_end_date",
    "reporting_period",
    "as_of_date",
    "date_anchor",
)
LOCATION_FIELDS = ("country", "subnational_location", "locality", "geographic_scope")
INCOMPATIBLE_SEMANTICS = {
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


def _env_float(name: str, default: float) -> float:
    raw = get_env(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = get_env(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def anomaly_thresholds() -> dict:
    return {
        "max_cases_threshold": _env_float("ANOMALY_MAX_CASES_THRESHOLD", 1_000_000.0),
        "max_deaths_threshold": _env_float("ANOMALY_MAX_DEATHS_THRESHOLD", 100_000.0),
        "spike_multiplier": _env_float("ANOMALY_SPIKE_MULTIPLIER", 10.0),
        "min_prior_records": _env_int("ANOMALY_MIN_PRIOR_RECORDS", 1),
    }


def _number(value):
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_count(record: dict) -> bool:
    return any(_number(record.get(field)) is not None for field in COUNT_FIELDS)


def _record_case_reference(record: dict) -> float | None:
    values = [_number(record.get(field)) for field in CASE_FIELDS]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return max(values)


def _first_non_empty(record: dict, fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = record.get(field)
        if value in (None, "", [], {}):
            continue
        return str(value)
    return None


def _record_location(record: dict) -> str | None:
    return _first_non_empty(record, LOCATION_FIELDS)


def _record_period(record: dict) -> str | None:
    return _first_non_empty(record, DATE_FIELDS)


def _task_disease(state: DataCollectionState) -> str | None:
    for source in (state.get("structured_task") or {}, state.get("collection_spec") or {}):
        value = source.get("disease")
        if value:
            return str(value)
    return None


def _normal_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _disease_matches(record: dict, task_disease: str | None, state: DataCollectionState) -> bool:
    record_disease = record.get("disease") or record.get("disease_standard_name")
    if not task_disease or not record_disease:
        return False
    task = _normal_text(task_disease)
    observed = _normal_text(str(record_disease))
    if task == observed or task in observed or observed in task:
        return True
    profile = state.get("disease_intelligence") or {}
    aliases = []
    for key in ("aliases", "abbreviations", "pathogen_terms", "syndrome_terms"):
        aliases.extend(profile.get(key) or [])
    aliases.append(profile.get("disease_standard_name"))
    aliases.append(profile.get("disease_input"))
    return any(_normal_text(alias) == observed for alias in aliases if alias)


def _source_ids_from_record(record: dict) -> list[str]:
    sid = record.get("source_id")
    return [str(sid)] if sid else []


def _source_urls_from_record(record: dict) -> list[str]:
    url = record.get("source_url")
    return [str(url)] if url else []


def _evidence(record: dict | None = None, validation: dict | None = None) -> str | None:
    parts: list[str] = []
    if record:
        for field in ("evidence_quote", "evidence_context", "count_notes"):
            if record.get(field):
                parts.append(str(record[field]))
                break
    if validation and validation.get("evidence_summary"):
        parts.append(str(validation["evidence_summary"]))
    return " | ".join(parts) or None


class _AnomalyBuilder:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add(
        self,
        *,
        anomaly_type: str,
        anomaly_unit: str,
        severity: str,
        reason: str,
        record: dict | None = None,
        event_cluster: dict | None = None,
        validation_result: dict | None = None,
        source_id: str | None = None,
        compared_field: str | None = None,
        observed_value=None,
        expected_or_reference_value=None,
        threshold=None,
        recommended_action: str | None = None,
        needs_human_review: bool = True,
        human_review_reason: str | None = None,
        warnings: list[str] | None = None,
        source_ids: list[str] | None = None,
        source_urls: list[str] | None = None,
    ) -> dict:
        anomaly_id = f"anom_{len(self.results) + 1:03d}"
        row = AnomalyResult(
            anomaly_id=anomaly_id,
            anomaly_type=anomaly_type,
            anomaly_unit=anomaly_unit,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            record_id=(record or {}).get("record_id"),
            event_cluster_id=(event_cluster or {}).get("event_cluster_id")
            or (
                (validation_result or {}).get("left_event_cluster_ids") or [None]
            )[0],
            validation_result_id=(validation_result or {}).get("validation_result_id"),
            source_id=source_id or (record or {}).get("source_id"),
            compared_field=compared_field,
            disease=(record or validation_result or event_cluster or {}).get("disease"),
            location=(record or validation_result or event_cluster or {}).get("location")
            or (record or event_cluster or {}).get("subnational_location")
            or (record or event_cluster or {}).get("geographic_scope")
            or (record or event_cluster or {}).get("country"),
            date_or_period=(validation_result or {}).get("date_or_period")
            or _record_period(record or event_cluster or {}),
            source_ids=source_ids
            if source_ids is not None
            else _source_ids_from_record(record or {})
            or list((validation_result or {}).get("left_source_ids") or []),
            source_urls=source_urls
            if source_urls is not None
            else _source_urls_from_record(record or {})
            or list((validation_result or {}).get("left_source_urls") or []),
            evidence_summary=_evidence(record, validation_result),
            observed_value=observed_value,
            expected_or_reference_value=expected_or_reference_value,
            threshold=threshold,
            reason=reason,
            recommended_action=recommended_action or "human_review",
            needs_human_review=needs_human_review,
            human_review_reason=human_review_reason or reason if needs_human_review else None,
            detection_method=DETECTION_METHOD,
            warnings=warnings or [],
        ).model_dump()
        self.results.append(row)
        return row


def _review_item_for_anomaly(anomaly: dict) -> dict:
    related_ids = [
        value
        for value in (
            anomaly.get("record_id"),
            anomaly.get("event_cluster_id"),
            anomaly.get("validation_result_id"),
            anomaly.get("source_id"),
        )
        if value
    ]
    item = HumanReviewItem(
        review_id=f"review_anomaly_{anomaly['anomaly_id']}",
        item_type="anomaly",
        related_ids=related_ids or [anomaly["anomaly_id"]],
        reason=anomaly.get("human_review_reason") or anomaly.get("reason") or "",
        status="pending",
        anomaly_id=anomaly.get("anomaly_id"),
        record_id=anomaly.get("record_id"),
        event_cluster_id=anomaly.get("event_cluster_id"),
        validation_result_id=anomaly.get("validation_result_id"),
        source_ids=list(anomaly.get("source_ids") or []),
        source_urls=list(anomaly.get("source_urls") or []),
        severity=anomaly.get("severity"),
        evidence_summary=anomaly.get("evidence_summary"),
        suggested_action=anomaly.get("recommended_action") or "human_review",
        decision_options=[
            "accept_anomaly",
            "dismiss_anomaly",
            "confirm_anomaly",
            "mark_anomaly_resolved",
            "mark_anomaly_needs_more_evidence",
        ],
    )
    payload = item.model_dump()
    payload["issue_type"] = anomaly.get("anomaly_type")
    payload["recommended_action"] = anomaly.get("recommended_action")
    return payload


def _add_record_anomalies(builder: _AnomalyBuilder, state: DataCollectionState) -> None:
    task_disease = _task_disease(state)
    thresholds = anomaly_thresholds()
    prior_by_key: dict[tuple[str, str], list[float]] = {}

    for record in state.get("normalized_records") or []:
        if not isinstance(record, dict):
            continue
        case_reference = _record_case_reference(record)
        deaths = _number(record.get("deaths"))
        if deaths is not None and case_reference is None:
            builder.add(
                anomaly_type="deaths_without_case_reference",
                anomaly_unit="record",
                severity="low",
                record=record,
                compared_field="deaths",
                observed_value=deaths,
                reason="deaths present but no comparable case count is available",
                needs_human_review=False,
                warnings=["insufficient_case_reference"],
            )
        elif deaths is not None and case_reference is not None and deaths > case_reference:
            builder.add(
                anomaly_type="deaths_greater_than_cases",
                anomaly_unit="record",
                severity="critical" if deaths > case_reference * 2 else "high",
                record=record,
                compared_field="deaths",
                observed_value=deaths,
                expected_or_reference_value=case_reference,
                reason="deaths exceed available case count for the same record",
                recommended_action="review_case_and_death_counts",
            )

        for field in COUNT_FIELDS:
            value = _number(record.get(field))
            if value is None:
                continue
            if value < 0:
                builder.add(
                    anomaly_type="negative_count_value",
                    anomaly_unit="record",
                    severity="critical",
                    record=record,
                    compared_field=field,
                    observed_value=value,
                    expected_or_reference_value="non_negative_count",
                    reason=f"{field} is negative",
                    recommended_action="correct_or_reject_record",
                )

        if not _has_count(record):
            continue

        if _record_period(record) is None:
            builder.add(
                anomaly_type="missing_date_for_count_bearing_record",
                anomaly_unit="record",
                severity="medium",
                record=record,
                compared_field="date_or_period",
                reason="count-bearing record has no usable date, reporting period, as-of date, or date anchor",
                recommended_action="supply_date_or_mark_needs_more_evidence",
            )
        if _record_location(record) is None:
            builder.add(
                anomaly_type="missing_location_for_count_bearing_record",
                anomaly_unit="record",
                severity="medium",
                record=record,
                compared_field="location",
                reason="count-bearing record has no usable country, subnational location, locality, or geographic scope",
                recommended_action="supply_location_or_mark_needs_more_evidence",
            )
        if not _disease_matches(record, task_disease, state):
            builder.add(
                anomaly_type="disease_mismatch_or_unknown_for_count_bearing_record",
                anomaly_unit="record",
                severity="high",
                record=record,
                compared_field="disease",
                observed_value=record.get("disease") or record.get("disease_standard_name"),
                expected_or_reference_value=task_disease,
                reason="count-bearing record disease is missing or does not match task disease",
                recommended_action="confirm_record_scope_or_reject",
            )

        cases = _record_case_reference(record)
        if cases is not None and cases > thresholds["max_cases_threshold"]:
            builder.add(
                anomaly_type="abrupt_spike_simple_threshold",
                anomaly_unit="record",
                severity="high" if cases > thresholds["max_cases_threshold"] * 2 else "medium",
                record=record,
                compared_field="case_count",
                observed_value=cases,
                threshold=thresholds["max_cases_threshold"],
                reason="case count exceeds configured simple anomaly threshold",
                recommended_action="review_possible_spike_or_extraction_error",
            )
        if deaths is not None and deaths > thresholds["max_deaths_threshold"]:
            builder.add(
                anomaly_type="abrupt_spike_simple_threshold",
                anomaly_unit="record",
                severity="high",
                record=record,
                compared_field="deaths",
                observed_value=deaths,
                threshold=thresholds["max_deaths_threshold"],
                reason="death count exceeds configured simple anomaly threshold",
                recommended_action="review_possible_spike_or_extraction_error",
            )

        positivity = _number(record.get("positivity_rate"))
        if positivity is not None and (positivity < 0 or positivity > 1):
            builder.add(
                anomaly_type="test_positivity_or_rate_invalid",
                anomaly_unit="record",
                severity="high" if positivity > 100 else "medium",
                record=record,
                compared_field="positivity_rate",
                observed_value=positivity,
                expected_or_reference_value="0_to_1_proportion_or_valid_percent",
                reason="positivity_rate is outside expected proportion bounds",
                recommended_action="review_rate_scale_or_extraction",
            )
        incidence = _number(record.get("incidence_rate"))
        if incidence is not None and incidence < 0:
            builder.add(
                anomaly_type="test_positivity_or_rate_invalid",
                anomaly_unit="record",
                severity="high",
                record=record,
                compared_field="incidence_rate",
                observed_value=incidence,
                expected_or_reference_value="non_negative_rate",
                reason="incidence_rate is negative",
                recommended_action="review_rate_or_extraction",
            )

        key = (
            _normal_text(record.get("disease") or ""),
            _normal_text(_record_location(record) or ""),
        )
        prior = prior_by_key.setdefault(key, [])
        if cases is not None and len(prior) >= thresholds["min_prior_records"]:
            baseline = max(prior) if prior else 0
            if baseline > 0 and cases >= baseline * thresholds["spike_multiplier"]:
                builder.add(
                    anomaly_type="abrupt_spike_simple_threshold",
                    anomaly_unit="record",
                    severity="medium",
                    record=record,
                    compared_field="case_count",
                    observed_value=cases,
                    expected_or_reference_value=baseline,
                    threshold=thresholds["spike_multiplier"],
                    reason="case count is a simple-threshold spike over prior comparable records",
                    recommended_action="review_possible_spike_or_count_semantics",
                )
        if cases is not None:
            prior.append(cases)


def _add_validation_anomalies(builder: _AnomalyBuilder, state: DataCollectionState) -> None:
    for row in state.get("validation_results") or []:
        if not isinstance(row, dict):
            continue
        validation_status = row.get("validation_status")
        match_status = row.get("match_status")
        validation_type = row.get("validation_type")
        reason = str(row.get("reason") or "")
        warnings = [str(w) for w in (row.get("warnings") or [])]
        if validation_type == "scope_check" and (
            match_status == "outside_requested_scope"
            or validation_status == "outside_scope"
            or "outside" in reason
        ):
            builder.add(
                anomaly_type="out_of_scope_count_bearing_record",
                anomaly_unit="validation_result",
                severity="high",
                validation_result=row,
                compared_field=row.get("compared_field"),
                observed_value=row.get("left_value"),
                expected_or_reference_value=row.get("right_value"),
                reason=f"Stage 10 validation marked record outside requested scope: {reason}",
                recommended_action="confirm_scope_or_reject_record",
                warnings=warnings,
            )
        if (
            validation_type == "count_semantics_check"
            or "count_semantics" in reason
            or any("count_semantics" in w for w in warnings)
        ) and (
            row.get("comparability_status") in {"not_comparable", "needs_human_review"}
            or match_status in {"not_comparable", "needs_human_review"}
        ):
            builder.add(
                anomaly_type="count_semantics_conflict",
                anomaly_unit="validation_result",
                severity="medium",
                validation_result=row,
                compared_field=row.get("compared_field"),
                observed_value=row.get("left_value"),
                expected_or_reference_value=row.get("right_value"),
                reason="Stage 10 marked count semantics as not safely comparable",
                recommended_action="review_count_semantics",
                warnings=warnings,
            )
        if match_status == "conflict" or validation_status == "conflict":
            builder.add(
                anomaly_type="validation_conflict_anomaly",
                anomaly_unit="validation_result",
                severity="high",
                validation_result=row,
                compared_field=row.get("compared_field"),
                observed_value=row.get("left_value"),
                expected_or_reference_value=row.get("right_value"),
                reason=f"Validation result is a conflict: {reason}",
                recommended_action="resolve_validation_conflict",
                warnings=warnings,
            )


def _add_conflict_anomalies(builder: _AnomalyBuilder, state: DataCollectionState) -> None:
    by_source = {
        source.get("source_id"): source
        for source in state.get("source_registry") or []
        if isinstance(source, dict) and source.get("source_id")
    }
    for conflict in state.get("conflicts") or []:
        if not isinstance(conflict, dict):
            continue
        source_ids = list(conflict.get("source_ids") or [])
        high_sources = [
            sid
            for sid in source_ids
            if (by_source.get(sid) or {}).get("credibility_level") == "high"
            or float((by_source.get(sid) or {}).get("credibility_score") or 0) >= 0.85
        ]
        if len(set(high_sources)) >= 2:
            builder.add(
                anomaly_type="high_credibility_source_conflict",
                anomaly_unit="source",
                severity="high",
                compared_field=conflict.get("field"),
                observed_value=conflict.get("values"),
                reason="high-credibility sources disagree on comparable fields",
                source_ids=source_ids,
                source_urls=list(conflict.get("source_urls") or []),
                recommended_action="review_high_credibility_conflict",
            )


def _add_cluster_anomalies(builder: _AnomalyBuilder, state: DataCollectionState) -> None:
    records_by_id = {
        record.get("record_id"): record
        for record in state.get("normalized_records") or []
        if isinstance(record, dict) and record.get("record_id")
    }
    canonical_map = {
        "canonical_cases_confirmed": "cases_confirmed",
        "canonical_cases_probable": "cases_probable",
        "canonical_cases_suspected": "cases_suspected",
        "canonical_cases_unspecified": "cases_unspecified",
        "canonical_deaths": "deaths",
        "canonical_hospitalizations": "hospitalizations",
    }
    for cluster in state.get("event_clusters") or []:
        if not isinstance(cluster, dict):
            continue
        countable_ids = list(cluster.get("countable_record_ids") or [])
        if not countable_ids:
            continue
        member_records = [records_by_id[rid] for rid in countable_ids if rid in records_by_id]
        semantics = {
            str(record.get("count_semantics") or record.get("statistical_count_type") or "")
            for record in member_records
            if record.get("count_semantics") or record.get("statistical_count_type")
        }
        for a in semantics:
            for b in semantics:
                if (a, b) in INCOMPATIBLE_SEMANTICS:
                    builder.add(
                        anomaly_type="count_semantics_conflict",
                        anomaly_unit="event_cluster",
                        severity="medium",
                        event_cluster=cluster,
                        compared_field="count_semantics",
                        observed_value=sorted(semantics),
                        reason="event cluster countable members mix incompatible count semantics",
                        recommended_action="review_cluster_countability",
                    )
                    break
        for canonical_field, member_field in canonical_map.items():
            canonical = _number(cluster.get(canonical_field))
            if canonical is None:
                continue
            values = [
                _number(record.get(member_field))
                for record in member_records
                if _number(record.get(member_field)) is not None
            ]
            if not values:
                continue
            total = sum(values)
            if total != canonical:
                builder.add(
                    anomaly_type="aggregate_member_mismatch",
                    anomaly_unit="event_cluster",
                    severity="medium",
                    event_cluster=cluster,
                    compared_field=member_field,
                    observed_value=canonical,
                    expected_or_reference_value=total,
                    reason="event cluster canonical count differs from sum of comparable countable members",
                    recommended_action="review_cluster_canonical_count",
                )


def _summary(results: list[dict], thresholds: dict) -> dict:
    severity_counter = Counter(str(row.get("severity") or "unknown") for row in results)
    type_counter = Counter(str(row.get("anomaly_type") or "unknown") for row in results)
    model = AnomalySummary(
        anomaly_result_count=len(results),
        needs_human_review_count=sum(1 for row in results if row.get("needs_human_review")),
        severity_counts=dict(severity_counter),
        anomaly_type_counts=dict(type_counter),
        thresholds=thresholds,
    )
    return model.model_dump()


def detect_anomalies(state: DataCollectionState) -> dict:
    """Return anomaly results, summary, and review items for the current state."""

    builder = _AnomalyBuilder()
    _add_record_anomalies(builder, state)
    _add_validation_anomalies(builder, state)
    _add_conflict_anomalies(builder, state)
    _add_cluster_anomalies(builder, state)
    existing_ids = {
        item.get("review_id")
        for item in (state.get("human_review_queue") or [])
        if isinstance(item, dict)
    }
    review_items: list[dict] = []
    for anomaly in builder.results:
        if not anomaly.get("needs_human_review"):
            continue
        item = _review_item_for_anomaly(anomaly)
        if item["review_id"] in existing_ids:
            continue
        existing_ids.add(item["review_id"])
        review_items.append(item)
    return {
        "anomaly_results": builder.results,
        "anomaly_summary": _summary(builder.results, anomaly_thresholds()),
        "anomaly_review_items": review_items,
    }
