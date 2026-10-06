"""Recover explicit case fields from completed workflow evidence.

The recovery layer is deliberately evidence-bound: it reads only workflow
candidate quotes, fills fields stated in those quotes, and records conflicts
instead of resolving them by guesswork.
"""

from __future__ import annotations

import json
from copy import deepcopy

from .nodes.extraction import _official_line_list_details
from .workflow_line_list_export import merge_case_candidate_fields


RECOVERABLE_CASE_FIELDS = (
    "workflow_case_label",
    "age",
    "gender",
    "nationality",
    "occupation_or_role",
    "symptoms",
    "date_onset",
    "date_confirmation",
    "date_death",
    "outcome",
    "hospitalized",
    "intensive_care",
    "isolated",
    "cruise_crew",
    "cruise_passenger_guest",
    "contact_with_case",
    "contact_setting",
    "ship_board_date",
    "ship_disembark_date",
    "travel_from",
    "travel_to",
    "travel_or_vessel_context",
    "confirmation_method",
    "accession_id",
)

INDIVIDUAL_COMPLETENESS_FIELDS = (
    "workflow_case_label",
    "case_status",
    "case_count",
    "age",
    "gender",
    "nationality",
    "occupation_or_role",
    "symptoms",
    "date_onset",
    "date_confirmation",
    "outcome",
    "location_admin_0",
    "travel_or_vessel_context",
    "confirmation_method",
)

AGGREGATE_COMPLETENESS_FIELDS = (
    "case_status",
    "case_count",
    "death_count",
    "date_reported",
    "reporting_period",
    "location_admin_0",
)

MONITORING_COMPLETENESS_FIELDS = (
    "workflow_case_label",
    "case_status",
    "date_reported",
    "location_admin_0",
    "contact_with_case",
    "contact_setting",
    "travel_or_vessel_context",
)


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _normalized(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold()


def _symptom_terms(*values) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not _has_value(value):
            continue
        for term in str(value).replace(",", ";").split(";"):
            cleaned = term.strip()
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                terms.append(cleaned)
    return terms


def _completeness_fields(row: dict, *, candidate: bool = False) -> tuple[str, ...]:
    if candidate:
        candidate_type = str(row.get("candidate_type") or "")
        if candidate_type == "aggregate_event_candidate":
            return AGGREGATE_COMPLETENESS_FIELDS
        if candidate_type == "non_case_or_monitoring_candidate":
            return MONITORING_COMPLETENESS_FIELDS
        return INDIVIDUAL_COMPLETENESS_FIELDS

    bundle_type = str(row.get("bundle_type") or "")
    if bundle_type == "aggregate_event":
        return AGGREGATE_COMPLETENESS_FIELDS
    if bundle_type == "non_case_monitoring":
        return MONITORING_COMPLETENESS_FIELDS
    if bundle_type == "individual_case_like":
        return INDIVIDUAL_COMPLETENESS_FIELDS
    return (
        "case_status",
        "case_count",
        "date_reported",
        "location_admin_0",
    )


def _apply_completeness(row: dict, *, candidate: bool = False) -> None:
    fields = _completeness_fields(row, candidate=candidate)
    missing = [field for field in fields if not _has_value(row.get(field))]
    row["field_completeness_score"] = f"{(len(fields) - len(missing)) / len(fields):.2f}"
    row["missing_key_fields"] = "; ".join(missing)


def recover_candidate_fields(candidate: dict) -> tuple[dict, int]:
    """Fill explicit fields from one candidate's own case-level quote."""

    recovered = deepcopy(candidate)
    if str(recovered.get("candidate_type") or "") != "individual_case_candidate":
        recovered["recovered_fields"] = ""
        recovered["field_recovery_conflicts_json"] = "{}"
        recovered["field_recovery_method"] = "not_applicable_for_non_individual_candidate"
        _apply_completeness(recovered, candidate=True)
        return recovered, 0

    quote = str(
        recovered.get("case_span_quote")
        or recovered.get("evidence_quote")
        or ""
    ).strip()
    details = _official_line_list_details(quote) if quote else {}
    changed: list[str] = []
    conflicts: dict[str, list[object]] = {}

    for field in RECOVERABLE_CASE_FIELDS:
        new_value = details.get(field)
        if not _has_value(new_value):
            continue
        old_value = recovered.get(field)
        if field == "symptoms":
            combined = "; ".join(_symptom_terms(old_value, new_value))
            if combined and combined != str(old_value or ""):
                recovered[field] = combined
                changed.append(field)
            continue
        if field == "age" and str(old_value or "").strip().casefold() == "adult":
            if str(new_value).isdigit():
                recovered[field] = new_value
                changed.append(field)
                continue
        if not _has_value(old_value):
            recovered[field] = new_value
            changed.append(field)
        elif _normalized(old_value) != _normalized(new_value):
            conflicts[field] = [old_value, new_value]

    recovered["recovered_fields"] = "; ".join(changed)
    recovered["field_recovery_conflicts_json"] = json.dumps(
        conflicts,
        ensure_ascii=False,
        sort_keys=True,
    )
    recovered["field_recovery_quote"] = quote
    recovered["field_recovery_method"] = "deterministic_case_span_recovery_v1"
    existing_provenance = recovered.get("field_provenance_json") or {}
    if isinstance(existing_provenance, str):
        try:
            existing_provenance = json.loads(existing_provenance)
        except json.JSONDecodeError:
            existing_provenance = {}
    if not isinstance(existing_provenance, dict):
        existing_provenance = {}
    for field in changed:
        existing_provenance[field] = {
            "field_name": field,
            "extracted_value": recovered.get(field),
            "source_id": recovered.get("source_id"),
            "chunk_id": recovered.get("supporting_chunk_id")
            or recovered.get("chunk_id"),
            "case_span_id": recovered.get("case_span_id"),
            "supporting_quote": quote,
            "extraction_method": "deterministic_case_span_recovery_v1",
        }
    recovered["field_provenance_json"] = json.dumps(
        existing_provenance,
        ensure_ascii=False,
        sort_keys=True,
    )
    recovered["unsupported_case_fields"] = sorted(
        {
            *(str(item) for item in (recovered.get("unsupported_case_fields") or [])),
            *conflicts.keys(),
        }
    )
    _apply_completeness(recovered, candidate=True)
    return recovered, len(changed)


def recover_line_list_rows(
    bundle_rows: list[dict],
    candidate_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    """Recover candidate fields, then merge complementary bundle evidence."""

    recovered_candidates: list[dict] = []
    recovered_candidate_field_count = 0
    for candidate in candidate_rows:
        recovered, changed_count = recover_candidate_fields(candidate)
        recovered_candidates.append(recovered)
        recovered_candidate_field_count += changed_count

    candidates_by_bundle: dict[str, list[dict]] = {}
    for candidate in recovered_candidates:
        bundle_id = str(candidate.get("bundle_id") or "")
        if bundle_id:
            candidates_by_bundle.setdefault(bundle_id, []).append(candidate)

    recovered_bundles: list[dict] = []
    recovered_bundle_field_count = 0
    for original in bundle_rows:
        row = deepcopy(original)
        bundle_id = str(row.get("bundle_id") or "")
        bundle_type = str(row.get("bundle_type") or "")
        candidates = candidates_by_bundle.get(bundle_id, [])
        merged = merge_case_candidate_fields(candidates, bundle_type=bundle_type)
        changed: list[str] = []
        for field in RECOVERABLE_CASE_FIELDS:
            if field in merged["conflicts"]:
                if _has_value(row.get(field)):
                    row[field] = ""
                    changed.append(field)
                continue
            value = merged["values"].get(field)
            if not _has_value(value):
                continue
            if field == "symptoms":
                combined = "; ".join(_symptom_terms(row.get(field), value))
                if combined and combined != str(row.get(field) or ""):
                    row[field] = combined
                    changed.append(field)
            elif not _has_value(row.get(field)):
                row[field] = value
                changed.append(field)

        row["field_source_ids_json"] = json.dumps(
            merged["source_ids"], ensure_ascii=False, sort_keys=True
        )
        row["field_evidence_quotes_json"] = json.dumps(
            merged["evidence_quotes"], ensure_ascii=False, sort_keys=True
        )
        row["field_conflicts_json"] = json.dumps(
            merged["conflicts"], ensure_ascii=False, sort_keys=True
        )
        row["field_recovery_method"] = merged["method"]
        row["recovered_fields"] = "; ".join(changed)
        _apply_completeness(row)
        recovered_bundle_field_count += len(changed)
        recovered_bundles.append(row)

    summary = {
        "bundle_row_count": len(recovered_bundles),
        "candidate_row_count": len(recovered_candidates),
        "recovered_bundle_field_count": recovered_bundle_field_count,
        "recovered_candidate_field_count": recovered_candidate_field_count,
        "benchmark_data_used": False,
        "method": "workflow_evidence_only_case_span_recovery_v1",
    }
    return recovered_bundles, recovered_candidates, summary


def field_completeness_audit(
    before_rows: list[dict],
    after_rows: list[dict],
    *,
    layer: str,
) -> list[dict]:
    """Return per-field before/after fill rates for an exported layer."""

    fields = [
        field
        for field in RECOVERABLE_CASE_FIELDS
        if any(field in row for row in before_rows) or any(field in row for row in after_rows)
    ]
    total = max(len(after_rows), len(before_rows), 1)
    audit: list[dict] = []
    for field in fields:
        before_count = sum(1 for row in before_rows if _has_value(row.get(field)))
        after_count = sum(1 for row in after_rows if _has_value(row.get(field)))
        audit.append(
            {
                "layer": layer,
                "field": field,
                "row_count": max(len(after_rows), len(before_rows)),
                "filled_before": before_count,
                "filled_after": after_count,
                "fields_recovered": after_count - before_count,
                "fill_rate_before": f"{before_count / total:.3f}",
                "fill_rate_after": f"{after_count / total:.3f}",
            }
        )
    return audit
