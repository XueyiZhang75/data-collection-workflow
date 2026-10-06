"""Workflow-native line-list style exports.

These helpers reshape existing workflow evidence outputs into professor-facing
CSV/JSON tables. They do not perform benchmark matching and do not alter any
quality-gate decision.
"""

from __future__ import annotations

import json
import re


SOURCE_SLOT_ROMANS = ("I", "II", "III", "IV", "V", "VI", "VII")

WORKFLOW_CASE_BUNDLE_LINE_LIST_FIELDS = [
    "workflow_row_id",
    "bundle_id",
    "event_family_id",
    "bundle_scope",
    "bundle_type",
    "workflow_inclusion_status",
    "recommended_review_action",
    "main_blocking_reason",
    "confidence",
    "disease",
    "pathogen_or_syndrome",
    "event_cluster_key",
    "event_name_or_context",
    "workflow_case_label",
    "case_label_normalized",
    "case_identity_fingerprint",
    "case_entity_id",
    "link_status",
    "link_score",
    "link_reasons",
    "observation_type",
    "case_status",
    "case_count",
    "death_count",
    "outcome",
    "symptoms",
    "date_onset",
    "date_confirmation",
    "date_death",
    "date_reported",
    "snapshot_as_of_date",
    "count_semantics",
    "location_scope",
    "reporting_period",
    "location_admin_0",
    "location_admin_1",
    "location_admin_2",
    "nationality",
    "age",
    "gender",
    "occupation_or_role",
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
    "field_source_ids_json",
    "field_evidence_quotes_json",
    "field_conflicts_json",
    "field_provenance_json",
    "unsupported_case_fields",
    "field_recovery_method",
    "focused_recovery_status",
    "supporting_source_count",
    "verified_authority_source_count",
    "high_trust_source_count",
    "official_source_count",
    "peer_reviewed_source_count",
    "structured_database_source_count",
    "source_stage_summary",
    "source_gap_status",
    "high_confidence_domain_status_json",
    "source_exact_page_status",
    "source_to_evidence_status_json",
    "non_target_disease_block_count",
    "local_disease_relevance_status",
    "field_completeness_score",
    "missing_key_fields",
    "line_list_detail_level",
    "snapshot_compatibility_status",
    "source_product_types",
    "source_I_url",
    "source_I_type",
    "source_II_url",
    "source_II_type",
    "source_III_url",
    "source_III_type",
    "source_IV_url",
    "source_IV_type",
    "source_V_url",
    "source_V_type",
    "source_VI_url",
    "source_VI_type",
    "source_VII_url",
    "source_VII_type",
    "best_evidence_quote",
    "all_source_ids_json",
    "all_source_urls_json",
    "human_review_required",
    "claim_boundary",
]

WORKFLOW_CASE_CANDIDATE_LINE_LIST_FIELDS = [
    "workflow_candidate_id",
    "candidate_id",
    "bundle_id",
    "event_family_id",
    "bundle_scope",
    "record_id",
    "source_id",
    "source_url",
    "source_name",
    "source_type",
    "source_product_type",
    "evidence_role",
    "candidate_type",
    "observation_type",
    "case_status",
    "workflow_case_label",
    "case_label_normalized",
    "case_identity_fingerprint",
    "case_entity_id",
    "link_status",
    "link_score",
    "link_reasons",
    "field_conflicts_json",
    "location",
    "date_or_period",
    "snapshot_as_of_date",
    "case_count",
    "death_count",
    "age",
    "gender",
    "nationality",
    "outcome",
    "symptoms",
    "date_onset",
    "date_confirmation",
    "date_death",
    "occupation_or_role",
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
    "source_gap_status",
    "high_confidence_domain_status_json",
    "source_exact_page_status",
    "source_to_evidence_status",
    "source_only_reason",
    "extraction_failure_substage",
    "field_provenance_json",
    "unsupported_case_fields",
    "focused_recovery_status",
    "local_disease_relevance_status",
    "record_local_numeric_disease_status",
    "record_local_numeric_sentence",
    "record_local_numeric_disease_decision",
    "record_local_numeric_target_terms_found",
    "record_local_numeric_incompatible_terms_found",
    "case_span_id",
    "case_span_quote",
    "case_span_start",
    "case_span_end",
    "case_span_extraction_method",
    "field_completeness_score",
    "missing_key_fields",
    "confidence",
    "quality_status",
    "review_reason",
    "quarantine_reason",
    "evidence_quote",
    "is_verified_authority_source",
    "human_review_required",
]


def _as_str(value) -> str:
    if value in (None, [], {}):
        return ""
    return str(value)


def _clean_evidence_quote(value) -> str:
    text = _as_str(value)
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]*\]\(([^)]+)\)", r"\1", text)
    replacements = {
        "â€œ": '"',
        "â€": '"',
        "â€\u009d": '"',
        "â€™": "'",
        "â€˜": "'",
        "â€“": "-",
        "â€”": "-",
        "Â": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_present(*values):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return ""


def _source_lookup(rows: list[dict]) -> dict[str, dict]:
    return {
        str(row.get("source_id") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("source_id")
    }


def _candidate_lookup(rows: list[dict]) -> dict[str, dict]:
    return {
        str(row.get("case_candidate_id") or row.get("candidate_id") or ""): row
        for row in rows
        if isinstance(row, dict)
        and (row.get("case_candidate_id") or row.get("candidate_id"))
    }


def _candidate_source_map(rows: list[dict]) -> dict[str, dict]:
    by_source: dict[str, dict] = {}
    for row in rows:
        source_id = str(row.get("source_id") or "")
        if source_id and source_id not in by_source:
            by_source[source_id] = row
    return by_source


def _source_url(source: dict, candidate: dict | None = None) -> str:
    candidate = candidate or {}
    return _as_str(
        _first_present(
            source.get("canonical_url"),
            source.get("source_url"),
            source.get("url"),
            candidate.get("source_url"),
        )
    )


def _source_name(source: dict, candidate: dict | None = None) -> str:
    candidate = candidate or {}
    return _as_str(
        _first_present(
            source.get("title"),
            source.get("source_name"),
            source.get("publisher"),
            candidate.get("source_title"),
            candidate.get("source_name"),
        )
    )


def _source_type(source: dict, candidate: dict | None = None) -> str:
    candidate = candidate or {}
    return _as_str(
        _first_present(
            source.get("source_type_final"),
            source.get("source_type"),
            candidate.get("source_type_final"),
            candidate.get("source_type"),
        )
    )


def _source_product_type(source: dict, candidate: dict | None = None) -> str:
    candidate = candidate or {}
    return _as_str(
        _first_present(
            source.get("data_product_type"),
            source.get("source_product_type"),
            candidate.get("data_product_type"),
            candidate.get("source_product_type"),
        )
    )


def _is_verified_authority_source(source: dict, candidate: dict | None = None) -> bool:
    candidate = candidate or {}
    text = " ".join(
        _as_str(value).lower()
        for value in (
            _source_type(source, candidate),
            source.get("authority_bucket"),
            candidate.get("authority_bucket"),
            source.get("evidence_role"),
            candidate.get("expected_evidence_role"),
        )
    )
    return any(
        token in text
        for token in (
            "official",
            "public_health",
            "health_agency",
            "government",
            "ministry",
            "international",
            "national",
            "authority",
        )
    )


def _source_priority(source: dict, candidate: dict | None = None) -> tuple[int, str]:
    source_type = _source_type(source, candidate).lower()
    product_type = _source_product_type(source, candidate).lower()
    if _is_verified_authority_source(source, candidate):
        return (0, _source_url(source, candidate))
    if any(token in source_type for token in ("peer", "academic", "scientific")) or any(
        token in product_type
        for token in ("case_report_article", "literature", "peer_reviewed")
    ):
        return (1, _source_url(source, candidate))
    if "structured_database" in source_type or "database" in product_type:
        return (2, _source_url(source, candidate))
    if any(token in source_type for token in ("news", "media")):
        return (3, _source_url(source, candidate))
    return (4, _source_url(source, candidate))


def _source_family_counts(
    ordered_sources: list[dict],
    *,
    source_by_id: dict[str, dict],
    candidate_by_source: dict[str, dict],
) -> dict[str, int]:
    official = 0
    peer = 0
    structured = 0
    high_trust_ids: set[str] = set()
    for source_row in ordered_sources:
        source_id = source_row["source_id"]
        source = source_by_id.get(source_id) or {}
        candidate = candidate_by_source.get(source_id) or {}
        source_type = _source_type(source, candidate).lower()
        product_type = _source_product_type(source, candidate).lower()
        if _is_verified_authority_source(source, candidate):
            official += 1
            high_trust_ids.add(source_id)
        if any(token in source_type for token in ("peer", "academic", "journal", "literature", "scientific")) or any(
            token in product_type for token in ("case_report_article", "peer_reviewed")
        ):
            peer += 1
            high_trust_ids.add(source_id)
        if "structured" in source_type or "database" in source_type or "database" in product_type:
            structured += 1
            high_trust_ids.add(source_id)
    return {
        "official_source_count": official,
        "peer_reviewed_source_count": peer,
        "structured_database_source_count": structured,
        "high_trust_source_count": len(high_trust_ids),
    }


def _ordered_bundle_sources(
    source_ids: list[str],
    *,
    source_by_id: dict[str, dict],
    candidate_by_source: dict[str, dict],
) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for source_id in source_ids:
        source_id = str(source_id or "")
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        source = source_by_id.get(source_id) or {"source_id": source_id}
        candidate = candidate_by_source.get(source_id) or {}
        rows.append(
            {
                "source_id": source_id,
                "url": _source_url(source, candidate),
                "type": _source_type(source, candidate),
                "product_type": _source_product_type(source, candidate),
                "priority": _source_priority(source, candidate),
            }
        )
    return sorted(rows, key=lambda row: row["priority"])


def _case_count(row: dict) -> int | float | str:
    for key in ("case_count", "cases_confirmed", "cases_probable", "cases_suspected", "cases_unspecified"):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _death_count(row: dict) -> int | float | str:
    for key in ("death_count", "deaths"):
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


LINE_LIST_KEY_FIELDS = (
    "workflow_case_label",
    "case_status",
    "case_count",
    "death_count",
    "date_onset",
    "date_reported",
    "location_admin_0",
    "age",
    "gender",
    "nationality",
    "outcome",
    "symptoms",
    "travel_or_vessel_context",
)


def _field_completeness(row: dict) -> tuple[str, str]:
    missing: list[str] = []
    filled = 0
    for field in LINE_LIST_KEY_FIELDS:
        value = row.get(field)
        if value not in (None, "", [], {}):
            filled += 1
        else:
            missing.append(field)
    score = filled / len(LINE_LIST_KEY_FIELDS)
    return f"{score:.2f}", "; ".join(missing)


def _line_list_detail_level(row: dict, bundle_type: str) -> str:
    if bundle_type == "aggregate_event":
        return "aggregate_event"
    individual_fields = ("workflow_case_label", "age", "gender", "date_onset", "outcome")
    if any(row.get(field) not in (None, "") for field in individual_fields):
        return "individual_detail"
    if bundle_type == "individual_case_like":
        return "individual_sparse"
    if bundle_type == "non_case_monitoring":
        return "non_case_monitoring"
    return "mixed_or_context"


def _claim_boundary(status: str, bundle_type: str) -> str:
    status = status.lower()
    if "final" in status:
        return "final_accepted_record"
    if "quarantine" in status:
        return "quarantined"
    if bundle_type in {"context_only", "context"}:
        return "context_only"
    return "reviewable_evidence_not_final_truth"


def _line_list_location(candidate: dict) -> str:
    return _as_str(
        _first_present(
            candidate.get("geographic_scope"),
            candidate.get("subnational_location"),
            candidate.get("country"),
            candidate.get("location"),
        )
    )


def _json_list(values: list[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


MERGEABLE_CASE_FIELDS = (
    "workflow_case_label",
    "age",
    "gender",
    "nationality",
    "outcome",
    "symptoms",
    "date_onset",
    "date_confirmation",
    "date_death",
    "hospitalized",
    "intensive_care",
    "isolated",
    "occupation_or_role",
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


def _meaningful_field_value(value) -> bool:
    return value not in (None, "", [], {})


def _normalized_field_value(value):
    if isinstance(value, bool):
        return value
    return re.sub(r"\s+", " ", str(value)).strip()


def _canonical_case_label(value: object) -> str:
    text = _normalized_field_value(value).casefold()
    match = re.search(r"\bcase\s*(?:no\.?\s*)?#?\s*(\d+)\b", text)
    if match:
        return f"case_{match.group(1)}"
    return text


def _symptom_terms(values: list[object]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        for term in re.split(r"\s*[;,]\s*", str(value or "")):
            term = re.sub(r"\s+", " ", term).strip()
            key = term.casefold()
            if term and key not in seen:
                seen.add(key)
                terms.append(term)
    return terms


def _candidate_field_provenance(candidate: dict, field: str) -> list[dict]:
    provenance = candidate.get("field_provenance_json") or {}
    if isinstance(provenance, str):
        try:
            provenance = json.loads(provenance)
        except json.JSONDecodeError:
            provenance = {}
    if not isinstance(provenance, dict):
        return []
    value = provenance.get(field)
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def merge_case_candidate_fields(
    candidates: list[dict],
    *,
    bundle_type: str,
) -> dict:
    """Merge explicit, non-conflicting case fields with field-level provenance."""

    if bundle_type != "individual_case_like":
        return {
            "values": {},
            "source_ids": {},
            "evidence_quotes": {},
            "field_provenance": {},
            "conflicts": {},
            "method": "not_applicable_for_non_individual_bundle",
        }

    values: dict[str, object] = {}
    source_ids: dict[str, list[str]] = {}
    evidence_quotes: dict[str, list[str]] = {}
    field_provenance: dict[str, dict] = {}
    conflicts: dict[str, list[object]] = {}
    for field in MERGEABLE_CASE_FIELDS:
        entries = [
            {
                "value": candidate.get(field),
                "source_id": str(candidate.get("source_id") or ""),
                "quote": _clean_evidence_quote(
                    candidate.get("case_span_quote") or candidate.get("evidence_quote")
                ),
                "provenance": _candidate_field_provenance(candidate, field),
            }
            for candidate in candidates
            if _meaningful_field_value(candidate.get(field))
        ]
        if not entries:
            continue

        if field == "symptoms":
            terms = _symptom_terms([entry["value"] for entry in entries])
            if terms:
                values[field] = "; ".join(terms)
        else:
            unique: list[object] = []
            unique_keys: set[object] = set()
            for entry in entries:
                normalized = _normalized_field_value(entry["value"])
                key = normalized if isinstance(normalized, bool) else normalized.casefold()
                if key not in unique_keys:
                    unique_keys.add(key)
                    unique.append(normalized)

            if field == "workflow_case_label" and len(unique) > 1:
                canonical_labels = {_canonical_case_label(value) for value in unique}
                if len(canonical_labels) == 1:
                    unique = [min(unique, key=lambda value: (len(str(value)), str(value)))]

            if field == "age" and len(unique) > 1:
                numeric = [value for value in unique if str(value).isdigit()]
                if len(numeric) == 1:
                    unique = numeric
            if field == "occupation_or_role" and len(unique) > 1:
                specific = [
                    value
                    for value in unique
                    if str(value).casefold() not in {"crew", "passenger", "contact"}
                ]
                if len(specific) == 1:
                    unique = specific

            if len(unique) == 1:
                values[field] = unique[0]
            else:
                conflicts[field] = sorted(unique, key=lambda value: str(value))
                continue

        field_sources = sorted(
            {entry["source_id"] for entry in entries if entry["source_id"]}
        )
        field_quotes = []
        for entry in entries:
            quote = entry["quote"]
            if quote and quote not in field_quotes:
                field_quotes.append(quote)
        source_ids[field] = field_sources
        evidence_quotes[field] = field_quotes[:5]
        provenance_records: list[dict] = []
        seen_provenance: set[str] = set()
        for entry in entries:
            for provenance in entry["provenance"]:
                key = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
                if key not in seen_provenance:
                    seen_provenance.add(key)
                    provenance_records.append(provenance)
        if len(provenance_records) == 1:
            field_provenance[field] = provenance_records[0]
        elif provenance_records:
            field_provenance[field] = {
                "field_name": field,
                "extracted_value": values.get(field),
                "source_ids": field_sources,
                "provenance_records": provenance_records,
            }

    return {
        "values": values,
        "source_ids": source_ids,
        "evidence_quotes": evidence_quotes,
        "field_provenance": field_provenance,
        "conflicts": conflicts,
        "method": "explicit_candidate_field_merge_v1",
    }


def _merged_case_value(field: str, merged: dict, *fallbacks):
    if field in (merged.get("conflicts") or {}):
        return ""
    if field in (merged.get("values") or {}):
        return merged["values"][field]
    return _first_present(*fallbacks)


def build_workflow_case_bundle_line_list(
    *,
    case_evidence_bundles: list[dict],
    case_candidate_dataset: list[dict],
    source_inventory: list[dict],
) -> list[dict]:
    """Build one professor-facing line-list row per evidence bundle."""

    source_by_id = _source_lookup(source_inventory)
    candidate_by_id = _candidate_lookup(case_candidate_dataset)
    candidate_by_source = _candidate_source_map(case_candidate_dataset)
    rows: list[dict] = []
    for index, bundle in enumerate(case_evidence_bundles or [], start=1):
        candidate_ids = [
            str(value)
            for value in (bundle.get("case_candidate_ids") or [])
            if value not in (None, "")
        ]
        candidates = [candidate_by_id[cid] for cid in candidate_ids if cid in candidate_by_id]
        unresolved_membership = not candidate_ids or len(candidates) != len(candidate_ids)
        representative = candidates[0] if candidates else {}
        source_ids = list(bundle.get("source_ids") or [])
        if not source_ids:
            source_ids = [candidate.get("source_id") for candidate in candidates]
        ordered_sources = _ordered_bundle_sources(
            source_ids,
            source_by_id=source_by_id,
            candidate_by_source=candidate_by_source,
        )
        family_counts = _source_family_counts(
            ordered_sources,
            source_by_id=source_by_id,
            candidate_by_source=candidate_by_source,
        )
        statuses = list(bundle.get("candidate_statuses") or [])
        status = str(statuses[0] if statuses else representative.get("case_candidate_status") or "")
        bundle_type = _as_str(bundle.get("bundle_type"))
        merged_case_fields = merge_case_candidate_fields(
            candidates,
            bundle_type=bundle_type,
        )
        row = {
            "workflow_row_id": f"WFL-{index:04d}",
            "bundle_id": _as_str(bundle.get("case_evidence_bundle_id") or bundle.get("bundle_id")),
            "event_family_id": _as_str(
                bundle.get("event_family_id")
                or representative.get("event_family_id")
            ),
            "bundle_scope": _as_str(
                bundle.get("bundle_scope") or representative.get("bundle_scope")
            ),
            "bundle_type": bundle_type,
            "workflow_inclusion_status": status or _as_str(bundle.get("bundle_status")),
            "recommended_review_action": _as_str(bundle.get("recommended_review_action")),
            "main_blocking_reason": _as_str(bundle.get("main_blocking_reason")),
            "confidence": _as_str(
                _first_present(
                    representative.get("confidence"),
                    representative.get("quality_gate_status"),
                    status,
                )
            ),
            "disease": _as_str(representative.get("disease")),
            "pathogen_or_syndrome": _as_str(representative.get("pathogen_or_syndrome")),
            "event_cluster_key": _as_str(bundle.get("event_cluster_key")),
            "event_name_or_context": _as_str(
                _first_present(
                    representative.get("event_name_or_context"),
                    representative.get("event_context"),
                    bundle.get("event_cluster_key"),
                )
            ),
            "workflow_case_label": _as_str(
                _merged_case_value(
                    "workflow_case_label",
                    merged_case_fields,
                    representative.get("workflow_case_label"),
                    representative.get("source_row_id"),
                    representative.get("case_label"),
                )
            ),
            "case_label_normalized": _as_str(
                bundle.get("case_label_normalized")
                or representative.get("case_label_normalized")
            ),
            "case_identity_fingerprint": _as_str(
                bundle.get("case_identity_fingerprint")
                or representative.get("case_identity_fingerprint")
            ),
            "case_entity_id": _as_str(
                bundle.get("case_entity_id") or representative.get("case_entity_id")
            ),
            "link_status": (
                "bundle_members_unresolved"
                if unresolved_membership
                else _as_str(bundle.get("link_status") or representative.get("link_status"))
            ),
            "link_score": _first_present(
                bundle.get("link_score"), representative.get("link_score")
            ),
            "link_reasons": "; ".join(
                [
                    *(
                        str(value)
                        for value in (
                            bundle.get("link_reasons")
                            or representative.get("link_reasons")
                            or []
                        )
                    ),
                    *(
                        ["case_candidate_ids_missing_or_unresolved"]
                        if unresolved_membership
                        else []
                    ),
                ]
            ),
            "observation_type": _as_str(representative.get("case_status")),
            "case_status": _as_str(representative.get("case_status")),
            "case_count": _case_count(representative),
            "death_count": _death_count(representative),
            "outcome": _as_str(
                _merged_case_value("outcome", merged_case_fields, representative.get("outcome"))
            ),
            "symptoms": _as_str(
                _merged_case_value("symptoms", merged_case_fields, representative.get("symptoms"))
            ),
            "date_onset": _as_str(
                _merged_case_value("date_onset", merged_case_fields, representative.get("date_onset"))
            ),
            "date_confirmation": _as_str(
                _merged_case_value(
                    "date_confirmation",
                    merged_case_fields,
                    representative.get("date_confirmation"),
                )
            ),
            "date_death": _as_str(
                _merged_case_value("date_death", merged_case_fields, representative.get("date_death"))
            ),
            "date_reported": _as_str(representative.get("date_reported")),
            "snapshot_as_of_date": _as_str(
                bundle.get("snapshot_as_of_date")
                or representative.get("snapshot_as_of_date")
            ),
            "count_semantics": _as_str(
                bundle.get("count_semantics")
                or representative.get("count_semantics")
                or representative.get("statistical_count_type")
            ),
            "location_scope": _as_str(
                bundle.get("location_scope")
                or representative.get("geographic_scope")
            ),
            "reporting_period": _as_str(
                bundle.get("reporting_period")
                or representative.get("reporting_period")
            ),
            "location_admin_0": _as_str(representative.get("country")),
            "location_admin_1": _as_str(representative.get("subnational_location")),
            "location_admin_2": _as_str(representative.get("locality")),
            "nationality": _as_str(
                _merged_case_value(
                    "nationality", merged_case_fields, representative.get("nationality")
                )
            ),
            "age": _as_str(
                _merged_case_value("age", merged_case_fields, representative.get("age"))
            ),
            "gender": _as_str(
                _merged_case_value("gender", merged_case_fields, representative.get("gender"))
            ),
            "occupation_or_role": _as_str(
                _merged_case_value(
                    "occupation_or_role",
                    merged_case_fields,
                    representative.get("occupation_or_role"),
                    representative.get("occupation"),
                    representative.get("role"),
                )
            ),
            "hospitalized": _merged_case_value(
                "hospitalized", merged_case_fields, representative.get("hospitalized")
            ),
            "intensive_care": _merged_case_value(
                "intensive_care", merged_case_fields, representative.get("intensive_care")
            ),
            "isolated": _merged_case_value(
                "isolated", merged_case_fields, representative.get("isolated")
            ),
            "cruise_crew": _merged_case_value(
                "cruise_crew", merged_case_fields, representative.get("cruise_crew")
            ),
            "cruise_passenger_guest": _merged_case_value(
                "cruise_passenger_guest",
                merged_case_fields,
                representative.get("cruise_passenger_guest"),
            ),
            "contact_with_case": _as_str(
                _merged_case_value(
                    "contact_with_case", merged_case_fields, representative.get("contact_with_case")
                )
            ),
            "contact_setting": _as_str(
                _merged_case_value(
                    "contact_setting", merged_case_fields, representative.get("contact_setting")
                )
            ),
            "ship_board_date": _as_str(
                _merged_case_value(
                    "ship_board_date", merged_case_fields, representative.get("ship_board_date")
                )
            ),
            "ship_disembark_date": _as_str(
                _merged_case_value(
                    "ship_disembark_date",
                    merged_case_fields,
                    representative.get("ship_disembark_date"),
                )
            ),
            "travel_from": _as_str(
                _merged_case_value("travel_from", merged_case_fields, representative.get("travel_from"))
            ),
            "travel_to": _as_str(
                _merged_case_value("travel_to", merged_case_fields, representative.get("travel_to"))
            ),
            "travel_or_vessel_context": _as_str(
                _merged_case_value(
                    "travel_or_vessel_context",
                    merged_case_fields,
                    representative.get("travel_history"),
                    representative.get("travel_or_vessel_context"),
                    representative.get("exposure_context"),
                )
            ),
            "confirmation_method": _as_str(
                _merged_case_value(
                    "confirmation_method",
                    merged_case_fields,
                    representative.get("confirmation_method"),
                )
            ),
            "accession_id": _as_str(
                _merged_case_value(
                    "accession_id", merged_case_fields, representative.get("accession_id")
                )
            ),
            "field_source_ids_json": json.dumps(
                merged_case_fields.get("source_ids") or {}, ensure_ascii=False
            ),
            "field_evidence_quotes_json": json.dumps(
                merged_case_fields.get("evidence_quotes") or {}, ensure_ascii=False
            ),
            "field_conflicts_json": json.dumps(
                merged_case_fields.get("conflicts") or {}, ensure_ascii=False
            ),
            "field_provenance_json": (
                json.dumps(
                    merged_case_fields.get("field_provenance") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if merged_case_fields.get("field_provenance")
                else _as_str(
                    bundle.get("field_provenance_json")
                    or representative.get("field_provenance_json")
                )
            ),
            "unsupported_case_fields": json.dumps(
                sorted(
                    {
                        str(field)
                        for candidate in candidates
                        for field in (candidate.get("unsupported_case_fields") or [])
                        if field not in (None, "")
                    }
                ),
                ensure_ascii=False,
            ),
            "field_recovery_method": _as_str(merged_case_fields.get("method")),
            "focused_recovery_status": _as_str(
                bundle.get("focused_recovery_status")
                or representative.get("focused_recovery_status")
            ),
            "supporting_source_count": bundle.get("supporting_source_count") or len(ordered_sources),
            "verified_authority_source_count": bundle.get(
                "verified_authority_source_count"
            )
            or sum(
                1
                for source in ordered_sources
                if _is_verified_authority_source(
                    source_by_id.get(source["source_id"]) or {},
                    candidate_by_source.get(source["source_id"]) or {},
                )
            ),
            "high_trust_source_count": bundle.get("high_trust_source_count")
            or family_counts["high_trust_source_count"],
            "official_source_count": bundle.get("official_source_count")
            or family_counts["official_source_count"],
            "peer_reviewed_source_count": bundle.get("peer_reviewed_source_count")
            or family_counts["peer_reviewed_source_count"],
            "structured_database_source_count": bundle.get("structured_database_source_count")
            or family_counts["structured_database_source_count"],
            "source_stage_summary": _as_str(bundle.get("source_stage_summary")),
            "source_gap_status": _as_str(bundle.get("source_gap_status")),
            "high_confidence_domain_status_json": _as_str(
                bundle.get("high_confidence_domain_status_json")
            ),
            "source_exact_page_status": _as_str(
                bundle.get("source_exact_page_status")
                or representative.get("source_exact_page_status")
            ),
            "source_to_evidence_status_json": _as_str(
                bundle.get("source_to_evidence_status_json")
            ),
            "non_target_disease_block_count": bundle.get(
                "non_target_disease_block_count"
            )
            or 0,
            "local_disease_relevance_status": _as_str(
                _first_present(
                    bundle.get("local_disease_relevance_status"),
                    representative.get("record_local_disease_relevance_status"),
                )
            ),
            "snapshot_compatibility_status": _as_str(
                bundle.get("snapshot_compatibility_status")
            ),
            "source_product_types": ", ".join(
                str(value)
                for value in (bundle.get("source_product_types") or [])
                if value not in (None, "")
            ),
            "best_evidence_quote": _clean_evidence_quote(
                bundle.get("best_evidence_quote")
            ),
            "all_source_ids_json": _json_list([source["source_id"] for source in ordered_sources]),
            "all_source_urls_json": _json_list([source["url"] for source in ordered_sources]),
            "human_review_required": status != "final",
            "claim_boundary": _claim_boundary(status, bundle_type),
        }
        completeness_score, missing_key_fields = _field_completeness(row)
        row["field_completeness_score"] = completeness_score
        row["missing_key_fields"] = missing_key_fields
        row["line_list_detail_level"] = _line_list_detail_level(row, bundle_type)
        for slot, source in zip(SOURCE_SLOT_ROMANS, ordered_sources[: len(SOURCE_SLOT_ROMANS)]):
            row[f"source_{slot}_url"] = source["url"]
            row[f"source_{slot}_type"] = source["type"]
        for slot in SOURCE_SLOT_ROMANS:
            row.setdefault(f"source_{slot}_url", "")
            row.setdefault(f"source_{slot}_type", "")
        rows.append(row)
    return rows


def _bundle_id_by_candidate_id(case_evidence_bundles: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for bundle in case_evidence_bundles or []:
        bundle_id = _as_str(bundle.get("case_evidence_bundle_id") or bundle.get("bundle_id"))
        for candidate_id in bundle.get("case_candidate_ids") or []:
            mapping[str(candidate_id)] = bundle_id
    return mapping


def build_workflow_case_candidate_line_list(
    *,
    case_candidate_dataset: list[dict],
    case_evidence_bundles: list[dict],
    source_inventory: list[dict],
) -> list[dict]:
    """Build one line-list row per source-specific candidate."""

    source_by_id = _source_lookup(source_inventory)
    bundle_by_candidate = _bundle_id_by_candidate_id(case_evidence_bundles)
    rows: list[dict] = []
    for index, candidate in enumerate(case_candidate_dataset or [], start=1):
        candidate_id = _as_str(candidate.get("case_candidate_id") or candidate.get("candidate_id"))
        source_id = _as_str(candidate.get("source_id"))
        source = source_by_id.get(source_id) or {}
        quality_status = _as_str(
            _first_present(
                candidate.get("quality_gate_status"),
                candidate.get("quality_status"),
                candidate.get("case_candidate_status"),
            )
        )
        status = _as_str(candidate.get("case_candidate_status"))
        row = {
                "workflow_candidate_id": f"WFC-{index:04d}",
                "candidate_id": candidate_id,
                "bundle_id": bundle_by_candidate.get(candidate_id, ""),
                "event_family_id": _as_str(candidate.get("event_family_id")),
                "bundle_scope": _as_str(candidate.get("bundle_scope")),
                "record_id": _as_str(candidate.get("record_id")),
                "source_id": source_id,
                "source_url": _source_url(source, candidate),
                "source_name": _source_name(source, candidate),
                "source_type": _source_type(source, candidate),
                "source_product_type": _source_product_type(source, candidate),
                "evidence_role": _as_str(candidate.get("expected_evidence_role")),
                "candidate_type": _as_str(candidate.get("candidate_type")),
                "observation_type": _as_str(candidate.get("case_status")),
                "case_status": _as_str(candidate.get("case_status")),
                "workflow_case_label": _as_str(candidate.get("workflow_case_label")),
                "case_label_normalized": _as_str(
                    candidate.get("case_label_normalized")
                ),
                "case_identity_fingerprint": _as_str(
                    candidate.get("case_identity_fingerprint")
                ),
                "case_entity_id": _as_str(candidate.get("case_entity_id")),
                "link_status": _as_str(candidate.get("link_status")),
                "link_score": _first_present(candidate.get("link_score")),
                "link_reasons": "; ".join(
                    str(value) for value in candidate.get("link_reasons") or []
                ),
                "field_conflicts_json": _as_str(
                    candidate.get("field_conflicts_json")
                ),
                "location": _line_list_location(candidate),
                "date_or_period": _as_str(
                    _first_present(
                        candidate.get("date_reported"),
                        candidate.get("reporting_period"),
                    )
                ),
                "snapshot_as_of_date": _as_str(candidate.get("snapshot_as_of_date")),
                "case_count": _case_count(candidate),
                "death_count": _death_count(candidate),
                "age": _as_str(candidate.get("age")),
                "gender": _as_str(candidate.get("gender")),
                "nationality": _as_str(candidate.get("nationality")),
                "outcome": _as_str(candidate.get("outcome")),
                "symptoms": _as_str(candidate.get("symptoms")),
                "date_onset": _as_str(candidate.get("date_onset")),
                "date_confirmation": _as_str(candidate.get("date_confirmation")),
                "date_death": _as_str(candidate.get("date_death")),
                "occupation_or_role": _as_str(
                    _first_present(
                        candidate.get("occupation_or_role"),
                        candidate.get("occupation"),
                        candidate.get("role"),
                    )
                ),
                "hospitalized": candidate.get("hospitalized")
                if candidate.get("hospitalized") is not None
                else "",
                "intensive_care": candidate.get("intensive_care")
                if candidate.get("intensive_care") is not None
                else "",
                "isolated": candidate.get("isolated")
                if candidate.get("isolated") is not None
                else "",
                "cruise_crew": candidate.get("cruise_crew")
                if candidate.get("cruise_crew") is not None
                else "",
                "cruise_passenger_guest": candidate.get("cruise_passenger_guest")
                if candidate.get("cruise_passenger_guest") is not None
                else "",
                "contact_with_case": candidate.get("contact_with_case")
                if candidate.get("contact_with_case") is not None
                else "",
                "contact_setting": _as_str(candidate.get("contact_setting")),
                "ship_board_date": _as_str(candidate.get("ship_board_date")),
                "ship_disembark_date": _as_str(
                    candidate.get("ship_disembark_date")
                ),
                "travel_from": _as_str(candidate.get("travel_from")),
                "travel_to": _as_str(candidate.get("travel_to")),
                "confirmation_method": _as_str(candidate.get("confirmation_method")),
                "accession_id": _as_str(candidate.get("accession_id")),
                "travel_or_vessel_context": _as_str(
                    _first_present(
                        candidate.get("travel_or_vessel_context"),
                        candidate.get("travel_history"),
                        candidate.get("exposure_context"),
                    )
                ),
                "source_gap_status": _as_str(candidate.get("source_gap_status")),
                "high_confidence_domain_status_json": _as_str(
                    candidate.get("high_confidence_domain_status_json")
                ),
                "source_exact_page_status": _as_str(
                    candidate.get("source_exact_page_status")
                    or source.get("source_exact_page_status")
                ),
                "source_to_evidence_status": _as_str(
                    candidate.get("source_to_evidence_status")
                    or source.get("source_to_evidence_status")
                ),
                "source_only_reason": _as_str(
                    candidate.get("source_only_reason")
                    or source.get("source_only_reason")
                ),
                "extraction_failure_substage": _as_str(
                    candidate.get("extraction_failure_substage")
                    or source.get("extraction_failure_substage")
                ),
                "field_provenance_json": (
                    json.dumps(
                        candidate.get("field_provenance_json"),
                        ensure_ascii=False,
                    )
                    if isinstance(candidate.get("field_provenance_json"), (dict, list))
                    else _as_str(candidate.get("field_provenance_json"))
                ),
                "unsupported_case_fields": json.dumps(
                    list(candidate.get("unsupported_case_fields") or []),
                    ensure_ascii=False,
                ),
                "focused_recovery_status": _as_str(
                    candidate.get("focused_recovery_status")
                    or source.get("focused_recovery_status")
                ),
                "local_disease_relevance_status": _as_str(
                    candidate.get("record_local_disease_relevance_status")
                ),
                "record_local_numeric_disease_status": _as_str(
                    candidate.get("record_local_numeric_disease_status")
                ),
                "record_local_numeric_sentence": _clean_evidence_quote(
                    candidate.get("record_local_numeric_sentence")
                ),
                "record_local_numeric_disease_decision": _as_str(
                    candidate.get("record_local_numeric_disease_decision")
                ),
                "record_local_numeric_target_terms_found": json.dumps(
                    list(candidate.get("record_local_numeric_target_terms_found") or []),
                    ensure_ascii=False,
                ),
                "record_local_numeric_incompatible_terms_found": json.dumps(
                    list(
                        candidate.get(
                            "record_local_numeric_incompatible_terms_found"
                        )
                        or []
                    ),
                    ensure_ascii=False,
                ),
                "case_span_id": _as_str(candidate.get("case_span_id")),
                "case_span_quote": _clean_evidence_quote(
                    candidate.get("case_span_quote")
                ),
                "case_span_start": candidate.get("case_span_start")
                if candidate.get("case_span_start") is not None
                else "",
                "case_span_end": candidate.get("case_span_end")
                if candidate.get("case_span_end") is not None
                else "",
                "case_span_extraction_method": _as_str(
                    candidate.get("case_span_extraction_method")
                ),
                "confidence": quality_status,
                "quality_status": quality_status,
                "review_reason": _as_str(candidate.get("review_reason")),
                "quarantine_reason": _as_str(candidate.get("quarantine_reason")),
                "evidence_quote": _clean_evidence_quote(candidate.get("evidence_quote")),
                "is_verified_authority_source": _is_verified_authority_source(
                    source, candidate
                ),
                "human_review_required": status != "final",
            }
        completeness_row = dict(row)
        completeness_row["date_reported"] = row.get("date_or_period")
        completeness_row["location_admin_0"] = row.get("location")
        completeness_score, missing_key_fields = _field_completeness(completeness_row)
        row["field_completeness_score"] = completeness_score
        row["missing_key_fields"] = missing_key_fields
        rows.append(row)
    return rows
