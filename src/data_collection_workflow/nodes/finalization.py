"""Final data package builder (Step 13).

Assembles a hardened, auditable `FinalDataPackage` from the workflow state.
Adds package metadata, workflow-summary aggregation, data dictionary,
provenance manifest, export manifest, and synthetic-fixture detection.

Does NOT call LLMs, does NOT touch the network, does NOT resolve conflicts,
and does NOT apply human review decisions to modify records.
"""

from __future__ import annotations

import json
import re
import hashlib
from datetime import date, datetime, timezone

from ..config import load_final_package_policy
from ..claim_corroboration import annotate_records_with_claim_corroboration
from ..models import (
    AnomalyResult,
    AppliedHumanReviewDecision,
    ClaimComparison,
    Conflict,
    CorroboratedEvent,
    EventCluster,
    FinalDataPackage,
    FinalPackagePolicy,
    HumanReviewAuditEntry,
    HumanReviewItem,
    LinkedEvent,
    PublicHealthRecord,
    PublicHealthClaim,
    RejectedHumanReviewDecision,
    SourceIdentityAssessment,
    SourceRegistryEntry,
    ValidationCase,
    ValidationComparison,
    ValidationResult,
)
from ..observation_type_datasets import (
    DATASET_VIEW_KEYS,
    apply_observation_type_counts_to_summaries,
    build_observation_type_dataset_split,
)
from ..run_quality_gates import apply_run_quality_gates
from ..source_coverage import (
    build_source_coverage_audit,
    build_task_evidence_contract,
)
from ..source_product_profile import apply_source_product_profiles
from ..state import DataCollectionState, append_trace
from ..workflow_line_list_export import (
    build_workflow_case_bundle_line_list,
    build_workflow_case_candidate_line_list,
    merge_case_candidate_fields,
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _package_generated_at(
    state: DataCollectionState,
    policy: FinalPackagePolicy,
    contains_fixture: bool,
    llm_used: bool,
) -> str:
    """Keep fixed dates only for synthetic offline packages."""
    search = state.get("source_search_execution_summary") or {}
    fetch = state.get("content_fetch_summary") or {}
    live_run = (
        state.get("pipeline_mode") == "evidence"
        or llm_used
        or bool(fetch.get("live_fetch_enabled"))
        or str(search.get("provider") or "").lower() not in {"", "fixture"}
    )
    if contains_fixture and not live_run:
        return policy.fixed_generated_at
    return datetime.now(timezone.utc).isoformat()


def _safe_list(state: DataCollectionState, key: str) -> list:
    return list(state.get(key) or [])


def _safe_dict(value) -> dict:
    return dict(value or {}) if isinstance(value, dict) else {}


def _source_lookup(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("source_id") or ""): row for row in rows if row.get("source_id")}


def _human_review_items_from_core_metric_gaps(
    gaps: list[dict],
    source_registry: list[dict],
    documents: list[dict],
) -> list[dict]:
    source_by_id = _source_lookup(source_registry)
    docs_by_source: dict[str, list[dict]] = {}
    for doc in documents:
        source_id = str(doc.get("source_id") or "")
        if source_id:
            docs_by_source.setdefault(source_id, []).append(doc)

    items: list[dict] = []
    for idx, gap in enumerate(gaps, start=1):
        source_id = str(gap.get("source_id") or "")
        source = source_by_id.get(source_id) or {}
        docs = docs_by_source.get(source_id) or []
        source_urls = []
        for value in (
            source.get("canonical_url"),
            source.get("url"),
            *(doc.get("canonical_url") or doc.get("url") for doc in docs),
        ):
            if value and value not in source_urls:
                source_urls.append(value)
        reason = str(
            gap.get("reason") or "core_metric_text_attempted_but_no_records_extracted"
        )
        items.append(
            {
                "review_id": f"review_core_metric_gap_{source_id or idx}",
                "item_type": "core_metric_extraction_gap",
                "related_ids": [source_id] if source_id else [],
                "reason": reason,
                "status": "pending",
                "priority": 1,
                "source_ids": [source_id] if source_id else [],
                "source_urls": source_urls,
                "severity": "medium",
                "evidence_summary": (
                    "Task-relevant source text appeared to contain core "
                    "epidemiology metric signals, but extraction produced no "
                    "structured records."
                ),
                "suggested_action": "review_source_for_core_metric_extraction",
                "decision_options": [
                    "extract_metric_record",
                    "mark_no_extractable_task_metric",
                    "send_to_best_available_context",
                ],
                "review_packet": {
                    "core_metric_extraction_gap": gap,
                    "source": source,
                    "documents": docs[:3],
                },
            }
        )
    return items


def _append_missing_human_review_items(
    existing: list[dict],
    additions: list[dict],
) -> list[dict]:
    output = list(existing or [])
    seen = {str(item.get("review_id") or "") for item in output}
    for item in additions:
        review_id = str(item.get("review_id") or "")
        if review_id and review_id in seen:
            continue
        output.append(item)
        if review_id:
            seen.add(review_id)
    return output


def _lower(value) -> str:
    return str(value or "").strip().lower()


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _lower(value)).strip("_")
    return text or "unknown"


def _iso_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None


def _record_requirement_ids(record: dict) -> set[str]:
    return {
        str(value)
        for value in (record.get("coverage_requirement_ids") or [])
        if value not in (None, "")
    }


def _record_period_dates(record: dict) -> tuple[date | None, date | None]:
    start = None
    end = None
    for key in ("metric_period_start", "period_start_date", "event_start_date", "start_date"):
        start = _iso_date(record.get(key))
        if start:
            break
    for key in ("metric_period_end", "period_end_date", "event_end_date", "end_date"):
        end = _iso_date(record.get(key))
        if end:
            break
    return start, end


def _requirement_period_dates(requirement: dict) -> tuple[date | None, date | None]:
    start = _iso_date(
        requirement.get("period_start") or requirement.get("reporting_period_start")
    )
    end = _iso_date(
        requirement.get("period_end") or requirement.get("reporting_period_end")
    )
    return start, end


def _record_period_conflicts_requirement(record: dict, requirement: dict) -> bool:
    req_start, req_end = _requirement_period_dates(requirement)
    if not req_start or not req_end:
        return False
    rec_start, rec_end = _record_period_dates(record)
    if rec_start and rec_end:
        if rec_end < rec_start:
            rec_start, rec_end = rec_end, rec_start
        if req_end < req_start:
            req_start, req_end = req_end, req_start
        return rec_start != req_start or rec_end != req_end
    requirement_year = requirement.get("year")
    if not requirement_year and req_start.year == req_end.year:
        requirement_year = req_start.year
    if _lower(requirement.get("period_basis")) == "annual" and requirement_year:
        text = _lower(
            " ".join(
                str(record.get(key) or "")
                for key in (
                    "reporting_period",
                    "metric_period_label",
                    "count_semantics",
                    "statistical_count_type",
                    "evidence_quote",
                )
            )
        )
        if str(requirement_year) in text and "annual" in text:
            return False
        return True
    anchor = _iso_date(record.get("date_anchor") or record.get("date_reported"))
    if anchor and (req_start <= anchor <= req_end):
        return False
    return False


def _record_matches_requirement_exact(
    record: dict,
    requirement: dict,
    source_ids: set[str],
) -> bool:
    requirement_id = str(requirement.get("requirement_id") or "")
    explicit_ids = _record_requirement_ids(record)
    if explicit_ids:
        if requirement_id not in explicit_ids:
            return False
    elif str(record.get("source_id") or "") not in source_ids:
        return False
    if _record_period_conflicts_requirement(record, requirement):
        return False
    req_location = _lower(requirement.get("geography") or requirement.get("location"))
    if req_location:
        record_geo = _lower(
            " ".join(
                str(record.get(key) or "")
                for key in (
                    "geographic_scope",
                    "subnational_location",
                    "country",
                    "location",
                )
            )
        )
        if req_location not in record_geo:
            return False
    req_disease = _lower(requirement.get("disease"))
    if req_disease:
        record_disease = _lower(
            " ".join(
                str(record.get(key) or "")
                for key in (
                    "disease",
                    "disease_standard_name",
                    "virus_or_syndrome",
                    "pathogen_or_syndrome",
                )
            )
        )
        if req_disease not in record_disease:
            return False
    return True


def _direct_collection_mode(state: dict) -> bool:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    return (
        str(
            structured.get("collection_mode")
            or spec.get("collection_mode")
            or state.get("collection_mode")
            or ""
        ).strip()
        == "direct_collection"
    )


def _record_has_public_health_metric(record: dict) -> bool:
    if record.get("metric_name") or record.get("metric_category"):
        return True
    if record.get("metric_value") not in (None, ""):
        return True
    metric_fields = (
        "tests_positive",
        "tests_total",
        "positivity_rate",
        "ili_percentage",
        "ed_visit_percentage",
        "hospitalizations",
        "deaths",
        "outbreak_count",
    )
    return any(record.get(field) not in (None, "") for field in metric_fields)


_EDGE_METRIC_RE = re.compile(
    r"\b("
    r"missing|unknown|not\s+reported|not\s+available|"
    r"race|ethnicity|birth\s+origin|age\s+missing|demographic"
    r")\b",
    re.IGNORECASE,
)


def _requirement_core_metric_families(requirement: dict) -> set[str]:
    raw = (
        requirement.get("core_metric_families")
        or requirement.get("accepted_metric_families")
        or []
    )
    return {
        re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        for value in raw
        if str(value or "").strip()
    }


def _record_is_edge_metric(record: dict) -> bool:
    text = " ".join(
        str(record.get(key) or "")
        for key in (
            "metric_name",
            "metric_category",
            "evidence_quote",
            "source_row_label",
            "source_column_label",
            "metric_column_label",
        )
    )
    return bool(_EDGE_METRIC_RE.search(text))


def _record_matches_core_metric_family(record: dict, families: set[str]) -> bool:
    if not families:
        return True
    if _record_is_edge_metric(record):
        return False
    text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "metric_name",
                "metric_category",
                "observation_type",
                " ".join(str(value) for value in record.get("observation_types") or []),
                "evidence_quote",
            )
        )
    )
    normalized_text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if families & set(filter(None, normalized_text.split("_"))):
        return True
    if "case_count" in families and (
        record.get("cases_confirmed") not in (None, "")
        or record.get("cases_probable") not in (None, "")
        or record.get("cases_suspected") not in (None, "")
        or record.get("cases_unspecified") not in (None, "")
        or "case_count" in normalized_text
        or "total_cases" in normalized_text
        or "confirmed_cases" in normalized_text
        or "reported_cases" in normalized_text
        or "notified_cases" in normalized_text
    ):
        return True
    if "incidence_rate" in families and (
        record.get("incidence_rate") not in (None, "")
        or "incidence_rate" in normalized_text
        or ("incidence" in text and "rate" in text)
    ):
        return True
    if families & {"death_count", "mortality", "mortality_rate", "death_rate"} and (
        record.get("deaths") not in (None, "")
        or "death" in text
        or "mortality" in text
    ):
        return True
    if families & {"hospitalization", "hospitalization_count", "hospitalization_rate"} and (
        record.get("hospitalizations") not in (None, "")
        or "hospital" in text
    ):
        return True
    if families & {"treatment_coverage", "vaccination_coverage", "coverage"} and (
        "coverage" in text or "treatment" in text or "vaccination" in text
    ):
        return True
    for family in families:
        if family and family in normalized_text:
            return True
    return False


def _record_source_high_trust(record: dict, registry_by_id: dict[str, dict]) -> bool:
    source_id = str(record.get("source_id") or "")
    source = registry_by_id.get(source_id) or {}
    level = str(
        record.get("credibility_level") or source.get("credibility_level") or ""
    ).strip().lower()
    if level in {"high", "medium"}:
        return True
    publisher = str(
        record.get("publisher")
        or record.get("actual_publisher")
        or source.get("publisher")
        or source.get("actual_publisher")
        or ""
    ).lower()
    url = str(record.get("source_url") or source.get("canonical_url") or "").lower()
    source_type = str(
        record.get("source_type_final")
        or record.get("source_type")
        or source.get("source_type_final")
        or source.get("source_type")
        or ""
    ).lower()
    return (
        ".gov" in url
        or "department of health" in publisher
        or "public health" in publisher
        or "health agency" in source_type
        or "official" in source_type
    )


def _task_period_from_state(state: dict) -> tuple[date | None, date | None]:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    start = _iso_date(
        structured.get("start_date")
        or structured.get("date_start")
        or spec.get("start_date")
        or spec.get("date_start")
    )
    end = _iso_date(
        structured.get("end_date")
        or structured.get("date_end")
        or spec.get("end_date")
        or spec.get("date_end")
    ) or start
    if start and end and end < start:
        start, end = end, start
    return start, end


def _record_period_from_record(record: dict) -> tuple[date | None, date | None]:
    start = _iso_date(
        record.get("metric_period_start")
        or record.get("date_start")
        or record.get("event_start_date")
        or record.get("date_reported")
        or record.get("date_anchor")
    )
    end = _iso_date(
        record.get("metric_period_end")
        or record.get("date_end")
        or record.get("event_end_date")
        or record.get("date_reported")
        or record.get("date_anchor")
    ) or start
    if start and end and end < start:
        start, end = end, start
    return start, end


def _best_available_period_fit(record: dict, state: dict) -> str:
    task_start, task_end = _task_period_from_state(state)
    record_start, record_end = _record_period_from_record(record)
    if not task_start or not task_end or not record_start or not record_end:
        return "period_uncertain"
    if record_start == task_start and record_end == task_end:
        return "exact"
    if record_start <= task_start and record_end >= task_end:
        return "broader_than_task"
    if record_end < task_start or record_start > task_end:
        return "outside_task_window"
    return "partial_overlap"


def _best_available_geography_fit(record: dict, state: dict) -> str:
    task = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    task_location = _lower(task.get("location") or spec.get("geography"))
    if not task_location:
        return "unknown_geography_fit"
    record_geo = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "geographic_scope",
                "subnational_location",
                "country",
                "locality",
                "admin_area",
                "location",
                "place",
                "evidence_quote",
            )
        )
    )
    scope_type = _lower(record.get("geographic_scope_type") or record.get("location_type"))
    scope = _lower(record.get("geographic_scope"))
    if task_location in {"global", "world", "worldwide"}:
        vessel_terms = {
            "vessel",
            "ship",
            "cruise",
            "cruise ship",
            "travel",
            "travel associated",
            "travel-associated",
            "traveler",
            "traveller",
        }
        if scope_type in {"vessel", "travel", "travel_associated", "vessel_or_travel_associated"} or any(
            term in record_geo for term in vessel_terms
        ):
            return "vessel_or_travel_associated_within_global_scope"
        if scope_type in {"region", "regional", "international", "multi country", "multi-country", "multi_country"}:
            return "regional_within_global_scope"
        if scope in {"americas", "region of the americas", "europe", "eu/eea"}:
            return "regional_within_global_scope"
        if scope_type in {"subnational", "state", "province", "county", "city", "local"} or record.get("subnational_location") or record.get("locality"):
            return "subnational_within_global_scope"
        if scope_type in {"country", "national"} or record.get("country"):
            return "country_within_global_scope"
        if scope in {"global", "world", "worldwide"}:
            return "within_global_scope"
        if record_geo:
            return "within_global_scope"
        return "unknown_geography_fit"
    if task_location and task_location in record_geo:
        if scope_type in {"county", "city", "local"} or record.get("locality"):
            return "subnational_within_task_geography"
        return "within_task_geography"
    if scope_type in {"global", "region", "international", "multi country", "multi-country"}:
        return "broader_than_task"
    if scope in {"global", "world", "worldwide", "americas", "europe", "eu/eea", "region of the americas"}:
        return "broader_than_task"
    if record_geo:
        return "outside_task_geography"
    return "unknown_geography_fit"


def _apply_best_available_geography_fit(row: dict, state: dict) -> None:
    current = str(row.get("record_geography_fit_status") or "").strip()
    task = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    task_location = _lower(task.get("location") or spec.get("geography"))
    should_refresh_global = task_location in {"global", "world", "worldwide"} and current in {
        "",
        "outside_task_geography",
        "broader_than_task",
        "exact",
        "unknown",
    }
    if not current or should_refresh_global:
        row["record_geography_fit_status"] = _best_available_geography_fit(row, state)
    if (
        row.get("record_geography_fit_status")
        == "vessel_or_travel_associated_within_global_scope"
        and not row.get("location_type")
    ):
        row["location_type"] = "vessel_or_travel_associated"


def _requirement_ids_for_best_available_record(
    record: dict,
    state: dict,
    registry_by_id: dict[str, dict],
) -> list[str]:
    ids = [
        str(value)
        for value in (record.get("coverage_requirement_ids") or [])
        if str(value or "").strip()
    ]
    if ids:
        return sorted(dict.fromkeys(ids))
    source = registry_by_id.get(str(record.get("source_id") or "")) or {}
    ids = [
        str(value)
        for value in (source.get("coverage_requirement_ids") or [])
        if str(value or "").strip()
    ]
    if ids:
        return sorted(dict.fromkeys(ids))
    requirements = _safe_list(state, "source_coverage_requirements")
    if len(requirements) == 1 and requirements[0].get("requirement_id"):
        return [str(requirements[0]["requirement_id"])]
    return []


def _build_record_linkage_indexes(state: dict) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict]]:
    registry_by_id = {
        str(row.get("source_id")): row
        for row in _safe_list(state, "source_registry")
        if isinstance(row, dict) and row.get("source_id")
    }
    documents_by_source: dict[str, list[dict]] = {}
    for document in _safe_list(state, "documents"):
        if not isinstance(document, dict):
            continue
        source_id = str(document.get("source_id") or "")
        if source_id:
            documents_by_source.setdefault(source_id, []).append(document)
    chunks_by_id = {
        str(row.get("chunk_id")): row
        for row in _safe_list(state, "evidence_chunks")
        if isinstance(row, dict) and row.get("chunk_id")
    }
    return registry_by_id, documents_by_source, chunks_by_id


def _record_requirement_ids_from_lineage(
    record: dict,
    state: dict,
    registry_by_id: dict[str, dict],
    documents_by_source: dict[str, list[dict]],
    chunks_by_id: dict[str, dict],
) -> list[str]:
    ids: list[str] = [
        str(value)
        for value in (record.get("coverage_requirement_ids") or [])
        if str(value or "").strip()
    ]
    if ids:
        return sorted(dict.fromkeys(ids))
    chunk = chunks_by_id.get(str(record.get("supporting_chunk_id") or ""))
    if chunk:
        ids.extend(
            str(value)
            for value in (chunk.get("coverage_requirement_ids") or [])
            if str(value or "").strip()
        )
    source_id = str(record.get("source_id") or "")
    source = registry_by_id.get(source_id) or {}
    ids.extend(
        str(value)
        for value in (source.get("coverage_requirement_ids") or [])
        if str(value or "").strip()
    )
    for document in documents_by_source.get(source_id, []):
        ids.extend(
            str(value)
            for value in (document.get("coverage_requirement_ids") or [])
            if str(value or "").strip()
        )
    if ids:
        return sorted(dict.fromkeys(ids))
    requirements = _safe_list(state, "source_coverage_requirements")
    if len(requirements) == 1 and requirements[0].get("requirement_id"):
        return [str(requirements[0]["requirement_id"])]
    return []


def _requirement_lookup(state: dict) -> dict[str, dict]:
    return {
        str(row.get("requirement_id")): row
        for row in _safe_list(state, "source_coverage_requirements")
        if isinstance(row, dict) and row.get("requirement_id")
    }


def _unique_nonempty(values) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        output.append(text)
        seen.add(text)
    return output


_AUTHORITY_INVENTORY_SOURCE_TYPES = {
    "official_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "international_public_health_agency",
    "structured_database",
    "academic_or_peer_reviewed_source",
}


def _source_inventory_role(source: dict) -> str:
    source_type = _lower(source.get("source_type_final") or source.get("source_type"))
    scope = _lower(source.get("jurisdiction_scope"))
    if source_type == "structured_database":
        return "structured_database"
    if source_type == "academic_or_peer_reviewed_source":
        return "peer_reviewed_or_scientific"
    if source_type == "international_public_health_agency" or scope in {
        "international",
        "regional",
    }:
        return "international_or_regional_authority"
    if source_type in {
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "official_public_health_agency",
    }:
        return "national_or_subnational_authority"
    if source_type in {"news_media", "secondary_aggregator"}:
        return "supporting_media_or_secondary"
    if source_type in {"background_fact_sheet", "public_health_context_page"}:
        return "context_source"
    return "unknown_or_other"


def _doc_status_for_source(documents: list[dict]) -> tuple[str, str, str]:
    if not documents:
        return "not_fetched", "not_parsed", "not_fetched"
    doc = documents[0]
    fetch_status = str(doc.get("fetch_status") or "fetched").strip() or "fetched"
    parse_status = str(doc.get("parse_status") or "unknown").strip() or "unknown"
    quality = _lower(doc.get("quality_status"))
    if _lower(fetch_status) == "fetch_failed":
        return fetch_status, parse_status, "fetch_failed"
    if parse_status in {"parse_failed", "fetch_failed"} or quality == "unusable":
        return fetch_status, parse_status, "parsed_unusable_or_failed"
    if parse_status.startswith("parsed"):
        return fetch_status, parse_status, "parsed_no_extracted_records"
    return fetch_status, parse_status, "fetched_not_parsed"


def _build_source_inventory(
    source_registry: list[dict],
    documents: list[dict],
    evidence_chunks: list[dict],
    records: list[dict],
    structured_extraction_summary: dict | None = None,
    source_recall_target_ledger: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    structured_extraction_summary = structured_extraction_summary or {}
    source_recall_target_ledger = source_recall_target_ledger or []
    extraction_budget_by_source = (
        structured_extraction_summary.get("extraction_budget_by_source") or {}
    )
    focused_status_by_source = (
        structured_extraction_summary.get("focused_recovery_status_by_source")
        or {}
    )
    empty_reason_by_source: dict[str, str] = {}
    for diagnostic in (
        structured_extraction_summary.get("llm_empty_output_diagnostics") or []
    ):
        if not isinstance(diagnostic, dict):
            continue
        diagnostic_source_id = str(diagnostic.get("source_id") or "")
        if diagnostic_source_id and diagnostic.get("reason"):
            empty_reason_by_source[diagnostic_source_id] = str(
                diagnostic.get("reason")
            )
    exact_status_by_domain = {
        str(row.get("authority_domain") or "")
        .strip()
        .lower()
        .removeprefix("www."): str(row.get("event_page_status") or "")
        for row in source_recall_target_ledger
        if isinstance(row, dict)
        and row.get("authority_domain")
        and row.get("event_page_status")
    }
    extraction_failures_by_source: dict[str, dict] = {}
    for failure in structured_extraction_summary.get("official_extraction_failures") or []:
        if not isinstance(failure, dict):
            continue
        source_id = str(failure.get("source_id") or "")
        if source_id and source_id not in extraction_failures_by_source:
            extraction_failures_by_source[source_id] = failure
    official_diagnostics_by_source: dict[str, list[dict]] = {}
    for diagnostic in (
        structured_extraction_summary.get("official_outbreak_deterministic_diagnostics")
        or []
    ):
        if not isinstance(diagnostic, dict):
            continue
        source_id = str(diagnostic.get("source_id") or "")
        if source_id:
            official_diagnostics_by_source.setdefault(source_id, []).append(diagnostic)
    docs_by_source: dict[str, list[dict]] = {}
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        source_id = str(doc.get("source_id") or "")
        if source_id:
            docs_by_source.setdefault(source_id, []).append(doc)
    records_by_source: dict[str, list[dict]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = str(record.get("source_id") or "")
        if source_id:
            records_by_source.setdefault(source_id, []).append(record)
    chunks_by_source: dict[str, list[dict]] = {}
    for chunk in evidence_chunks:
        if not isinstance(chunk, dict):
            continue
        source_id = str(chunk.get("source_id") or "")
        if source_id:
            chunks_by_source.setdefault(source_id, []).append(chunk)

    inventory: list[dict] = []
    for source in source_registry:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("source_id") or "")
        source_docs = docs_by_source.get(source_id, [])
        fetch_status, parse_status, extractable_status = _doc_status_for_source(source_docs)
        extracted_count = len(records_by_source.get(source_id, []))
        budget = (
            extraction_budget_by_source.get(source_id)
            if isinstance(extraction_budget_by_source, dict)
            else None
        ) or {}
        diagnostic_rows = official_diagnostics_by_source.get(source_id, [])
        failure = extraction_failures_by_source.get(source_id) or {}
        attempted_count = int(budget.get("attempted_count") or 0)
        queued_count = int(budget.get("queued_count") or 0)
        eligible_count = int(budget.get("eligible_count") or 0)
        skipped_due_to_cap_count = int(budget.get("skipped_due_to_cap_count") or 0)
        diagnostic_attempted = any(
            bool(row.get("extraction_attempted")) for row in diagnostic_rows
        )
        focused_recovery_status = str(
            focused_status_by_source.get(source_id)
            or budget.get("focused_recovery_status")
            or source.get("focused_recovery_status")
            or ""
        )
        focused_recovery_attempted = focused_recovery_status in {
            "deterministic_recovery_succeeded",
            "focused_retry_attempted",
            "focused_retry_succeeded",
            "focused_retry_empty",
            "focused_retry_error",
        }
        extraction_attempted = bool(
            extracted_count
            or attempted_count > 0
            or diagnostic_attempted
            or focused_recovery_attempted
            or source.get("extraction_attempted") is True
        )
        extraction_queued = bool(
            queued_count > 0
            or eligible_count > 0
            or diagnostic_rows
            or failure
            or source.get("extraction_queued") is True
        )
        source_chunks = chunks_by_source.get(source_id, [])
        target_chunk_count = sum(
            1
            for chunk in source_chunks
            if chunk.get("contains_target_data") is True
            or str(chunk.get("disease_relevance_status") or "")
            == "target_disease_match"
        )
        if extracted_count:
            extractable_status = "extracted_records"
        if extracted_count:
            source_to_evidence_status = "evidence_extracted"
            source_only_reason = ""
        elif fetch_status == "fetch_failed":
            source_to_evidence_status = "fetch_failed"
            source_only_reason = "fetch_failed"
        elif not source_docs:
            source_to_evidence_status = "not_fetched"
            source_only_reason = "not_fetched"
        elif parse_status.startswith("parsed") and target_chunk_count:
            source_to_evidence_status = "parsed_target_source_no_record_extracted"
            product = _lower(source.get("data_product_type"))
            expected_role = _lower(source.get("expected_evidence_role"))
            task_specificity = _lower(source.get("task_specificity"))
            if (
                product
                in {
                    "background_fact_sheet",
                    "search_or_index_page",
                    "policy_or_protocol",
                    "public_health_context_page",
                }
                or expected_role == "source_context"
                or task_specificity == "disease_specific_but_context"
            ):
                source_only_reason = "context_or_background_no_record_expected"
            elif (
                product
                in {
                    "event_outbreak_report",
                    "case_report_article",
                    "line_list_or_case_series",
                    "official_surveillance_report",
                    "sequence_database_record",
                }
                or task_specificity == "event_specific"
            ):
                if focused_recovery_status == "focused_retry_empty":
                    source_only_reason = "event_source_focused_retry_empty"
                elif focused_recovery_status == "focused_retry_error":
                    source_only_reason = "event_source_focused_retry_error"
                elif focused_recovery_status == "focused_retry_budget_exhausted":
                    source_only_reason = "event_source_focused_retry_budget_exhausted"
                else:
                    source_only_reason = (
                        "event_source_no_record_extracted"
                        if extraction_attempted
                        else "event_source_not_attempted_for_extraction"
                    )
            else:
                source_only_reason = "no_extracted_records"
        elif parse_status.startswith("parsed"):
            source_to_evidence_status = "parsed_no_target_chunks"
            source_only_reason = "no_target_chunks"
        elif fetch_status == "fetched":
            source_to_evidence_status = "fetched_not_parsed"
            source_only_reason = "not_parsed"
        else:
            source_to_evidence_status = extractable_status
            source_only_reason = extractable_status if not extracted_count else ""
        extraction_failure_substage = source.get("extraction_failure_substage")
        if not extraction_failure_substage:
            diagnostic_failure = next(
                (
                    str(row.get("failure_substage") or "")
                    for row in diagnostic_rows
                    if row.get("failure_substage")
                ),
                "",
            )
            if diagnostic_failure:
                extraction_failure_substage = diagnostic_failure
            elif source_only_reason == "event_source_not_attempted_for_extraction":
                extraction_failure_substage = "extraction_not_attempted"
            elif skipped_due_to_cap_count > 0:
                extraction_failure_substage = "chunk_cap"
            elif source_only_reason == "event_source_no_record_extracted":
                extraction_failure_substage = "record_extraction"
            elif source_only_reason in {
                "event_source_focused_retry_empty",
                "event_source_focused_retry_error",
                "event_source_focused_retry_budget_exhausted",
            }:
                extraction_failure_substage = "focused_llm_recovery"
            elif source_only_reason == "no_target_chunks":
                extraction_failure_substage = "chunk_relevance"
            else:
                extraction_failure_substage = ""
        llm_empty_output_reason = empty_reason_by_source.get(source_id, "")
        extraction_failure_reason = (
            llm_empty_output_reason
            or failure.get("reason")
            or source.get("extraction_failure_reason")
            or extraction_failure_substage
        )
        source_domain = (
            str(source.get("domain") or "")
            .strip()
            .lower()
            .removeprefix("www.")
        )
        source_exact_page_status = str(
            source.get("source_exact_page_status")
            or exact_status_by_domain.get(source_domain)
            or ""
        )
        row = {
            "source_id": source_id,
            "canonical_url": source.get("canonical_url") or source.get("url"),
            "title": source.get("title"),
            "publisher": source.get("actual_publisher") or source.get("publisher"),
            "domain": source.get("domain"),
            "source_type_final": source.get("source_type_final") or source.get("source_type"),
            "authority_bucket": source.get("authority_bucket"),
            "data_product_type": source.get("data_product_type"),
            "task_specificity": source.get("task_specificity"),
            "time_window_fit": source.get("time_window_fit"),
            "machine_readability": source.get("machine_readability"),
            "expected_evidence_role": source.get("expected_evidence_role"),
            "evidence_role": _source_inventory_role(source),
            "fetch_status": fetch_status,
            "parse_status": parse_status,
            "extractable_status": extractable_status,
            "extracted_record_count": extracted_count,
            "source_to_evidence_status": source_to_evidence_status,
            "target_chunk_count": target_chunk_count,
            "target_record_count": extracted_count,
            "source_only_reason": source_only_reason,
            "extraction_attempted": extraction_attempted,
            "extraction_queued": extraction_queued,
            "extraction_attempted_count": attempted_count,
            "extraction_eligible_count": eligible_count,
            "extraction_queued_count": queued_count,
            "extraction_record_count": int(budget.get("record_count") or extracted_count),
            "extraction_skipped_due_to_cap_count": skipped_due_to_cap_count,
            "extraction_failure_substage": extraction_failure_substage,
            "extraction_failure_reason": extraction_failure_reason,
            "focused_recovery_status": focused_recovery_status,
            "llm_empty_output_reason": llm_empty_output_reason,
            "source_exact_page_status": source_exact_page_status,
            "source_identity_unverified": bool(source.get("source_identity_unverified")),
            "source_identity_warnings": list(source.get("source_identity_warnings") or []),
        }
        inventory.append(row)
    authority_inventory = [
        row
        for row in inventory
        if _lower(row.get("source_type_final")) in _AUTHORITY_INVENTORY_SOURCE_TYPES
        or row.get("evidence_role")
        in {
            "international_or_regional_authority",
            "national_or_subnational_authority",
            "structured_database",
            "peer_reviewed_or_scientific",
        }
    ]
    return inventory, authority_inventory


def _record_key(record: dict) -> str:
    return str(record.get("record_id") or record.get("source_record_id") or "").strip()


def _case_count_present(record: dict) -> bool:
    return any(
        record.get(field) not in (None, "")
        for field in (
            "cases_confirmed",
            "cases_probable",
            "cases_suspected",
            "cases_unspecified",
            "deaths",
        )
    )


def _case_event_key(record: dict) -> str:
    explicit_event_id = next(
        (
            str(record.get(field)).strip()
            for field in (
                "event_family_id",
                "event_cluster_id",
                "linked_event_id",
                "outbreak_id",
            )
            if record.get(field) not in (None, "", [], {})
        ),
        None,
    )
    if explicit_event_id:
        return explicit_event_id
    normalized = _normalized_outbreak_event_key(record)
    if normalized:
        return normalized
    disease = _slug(
        str(record.get("disease_standard_name") or record.get("disease") or "unknown")
    )
    anchors = _event_family_anchors(record)
    pieces = [f"event_family|disease:{disease}"]
    pieces.extend(
        f"{field}:{value}"
        for field, value in anchors.items()
        if value
    )
    return "|".join(pieces) if len(pieces) > 1 else _record_key(record)


def _case_label_normalized(value: str | None) -> str | None:
    text = _lower(value)
    if not text:
        return None
    singular_match = re.search(r"\bcase\s*(?:no\.?\s*)?#?\s*(\d+)\b", text)
    if singular_match:
        return f"case_{singular_match.group(1)}"
    return _slug(text)


_HARD_CASE_ANCHOR_FIELDS = (
    "age",
    "gender",
    "nationality",
    "date_onset",
    "date_confirmation",
    "date_death",
)


def _case_descriptor_anchors(record: dict) -> dict[str, str]:
    """Retain identifying details embedded in an explicit case label."""

    text = _lower(record.get("workflow_case_label"))
    anchors: dict[str, str] = {}
    age_match = re.search(r"\b(\d{1,3})\s*(?:-| )?year(?:-| )?old\b", text)
    if age_match:
        anchors["age"] = age_match.group(1)
    gender_match = re.search(r"\b(male|female|man|woman)\b", text)
    if gender_match:
        anchors["gender"] = {
            "man": "male",
            "woman": "female",
        }.get(gender_match.group(1), gender_match.group(1))
    return anchors


def _case_anchor_values(record: dict) -> dict[str, str]:
    descriptor_anchors = _case_descriptor_anchors(record)
    values: dict[str, str] = {}
    for field, value in (
        ("age", record.get("age")),
        ("gender", record.get("gender")),
        ("nationality", record.get("nationality")),
        ("role", record.get("occupation_or_role") or record.get("occupation") or record.get("role")),
        ("onset", record.get("date_onset")),
        ("confirmation", record.get("date_confirmation")),
        ("death", record.get("date_death")),
        ("outcome", record.get("outcome")),
        (
            "travel",
            record.get("travel_or_vessel_context")
            or record.get("travel_history")
            or record.get("contact_setting"),
        ),
    ):
        text = str(value or "").strip()
        if text:
            values[field] = _slug(text)
    for field, value in descriptor_anchors.items():
        values.setdefault(field, _slug(value))
    return values


def _compatible_case_anchor_count(left: dict, right: dict) -> int:
    left_anchors = _case_anchor_values(left)
    right_anchors = _case_anchor_values(right)
    return sum(
        1
        for field, value in left_anchors.items()
        if value and right_anchors.get(field) == value
    )


def _has_hard_case_conflict(left: dict, right: dict) -> bool:
    left_anchors = _case_anchor_values(left)
    right_anchors = _case_anchor_values(right)
    for field in _HARD_CASE_ANCHOR_FIELDS:
        left_value = left_anchors.get(field)
        right_value = right_anchors.get(field)
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _effective_bundle_scope(candidate: dict) -> str:
    scope = str(candidate.get("bundle_scope") or "").strip()
    if not scope:
        scope = _bundle_scope_for_candidate_type(
            str(candidate.get("candidate_type") or "")
        )
    if scope == "individual_case" and _is_grouped_case_label(
        candidate.get("workflow_case_label")
    ):
        return "grouped_case"
    return scope


def _case_identity_fingerprint(
    record: dict,
    *,
    event_family: str | None = None,
    bundle_scope: str | None = None,
    case_label_normalized: str | None = None,
) -> str | None:
    existing = str(record.get("case_identity_fingerprint") or "").strip()
    if existing:
        return existing
    event_key = str(
        event_family
        or record.get("event_family_id")
        or record.get("event_cluster_key")
        or _case_event_key(record)
        or ""
    ).strip()
    scope = str(bundle_scope or record.get("bundle_scope") or "").strip()
    label = str(
        case_label_normalized
        or record.get("case_label_normalized")
        or _case_label_normalized(record.get("workflow_case_label"))
        or ""
    ).strip()
    if label:
        return f"{event_key}|{scope or 'individual_case'}|label:{label}"
    if scope and scope != "individual_case":
        return f"{event_key}|{scope}"

    anchors = [
        f"{key}:{value}" for key, value in sorted(_case_anchor_values(record).items())
    ]
    if len(anchors) < 3:
        return None
    return "|".join([event_key, scope or "individual_case", *anchors])


def _snapshot_as_of_date(record: dict) -> str | None:
    for key in ("as_of_date", "date_anchor", "date_reported", "report_date"):
        value = record.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    reporting_period = str(record.get("reporting_period") or "")
    if "as of" in reporting_period.lower():
        return reporting_period
    return None


def _snapshot_count_signature(record: dict) -> str:
    """Keep materially different outbreak snapshots in separate bundles."""
    values: list[str] = []
    for field in (
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
        "outbreak_count",
    ):
        value = record.get(field)
        if value not in (None, "", [], {}):
            values.append(f"{field}:{value}")
    metric_value = record.get("metric_value")
    if metric_value not in (None, "", [], {}):
        metric_name = _slug(
            str(record.get("metric_category") or record.get("metric_name") or "metric")
        )
        values.append(f"{metric_name}:{metric_value}")
    return _slug("|".join(values) or "unspecified")


def _bundle_scope_for_candidate_type(candidate_type: str) -> str:
    if candidate_type == "individual_case_candidate":
        return "individual_case"
    if candidate_type == "aggregate_event_candidate":
        return "event_snapshot"
    if candidate_type == "non_case_or_monitoring_candidate":
        return "non_case_monitoring"
    return "source_context"


def _case_bundle_group_key(candidate: dict) -> str:
    event_family = str(candidate.get("event_family_id") or candidate.get("event_cluster_key") or "")
    disease = _slug(
        str(candidate.get("disease_standard_name") or candidate.get("disease") or "")
    )
    scope = _effective_bundle_scope(candidate)
    label = str(
        candidate.get("case_label_normalized")
        or _case_label_normalized(candidate.get("workflow_case_label"))
        or ""
    )
    if scope == "individual_case" and label:
        return f"{event_family}|disease:{disease}|{scope}|{label}"
    if scope == "individual_case":
        return f"{event_family}|disease:{disease}|{scope}|unlabelled"
    if scope == "event_snapshot":
        count_semantics = _slug(
            str(
                candidate.get("count_semantics")
                or candidate.get("statistical_count_type")
                or candidate.get("observation_type")
                or candidate.get("case_status")
                or "unspecified"
            )
        )
        location_scope = _slug(
            "|".join(
                _unique_nonempty(
                    (
                        candidate.get("geographic_scope"),
                        candidate.get("country"),
                        candidate.get("subnational_location"),
                    )
                )
            )
            or "unspecified"
        )
        return "|".join(
            (
                event_family,
                f"disease:{disease}",
                scope,
                f"semantics:{count_semantics or 'unspecified'}",
                f"location:{location_scope or 'unspecified'}",
                f"period_family:{_snapshot_period_family(candidate)}",
                f"counts:{_snapshot_count_signature(candidate)}",
            )
        )
    if scope == "non_case_monitoring":
        snapshot_as_of_date = _slug(
            str(candidate.get("snapshot_as_of_date") or _snapshot_as_of_date(candidate) or "")
        )
        count_semantics = _slug(
            str(
                candidate.get("count_semantics")
                or candidate.get("statistical_count_type")
                or candidate.get("observation_type")
                or candidate.get("case_status")
                or "unspecified"
            )
        )
        location_scope = _slug(
            "|".join(
                _unique_nonempty(
                    (
                        candidate.get("geographic_scope"),
                        candidate.get("country"),
                        candidate.get("subnational_location"),
                    )
                )
            )
            or "unspecified"
        )
        reporting_period = _slug(str(candidate.get("reporting_period") or "unspecified"))
        return "|".join(
            (
                event_family,
                f"disease:{disease}",
                scope,
                f"as_of:{snapshot_as_of_date or 'unspecified'}",
                f"semantics:{count_semantics or 'unspecified'}",
                f"location:{location_scope or 'unspecified'}",
                f"period:{reporting_period or 'unspecified'}",
            )
        )
    return f"{event_family}|disease:{disease}|{scope}|{candidate.get('case_candidate_id') or candidate.get('record_id')}"


def _case_entity_id(bundle_key: str) -> str:
    digest = hashlib.sha256(bundle_key.encode("utf-8")).hexdigest()[:16]
    return f"case_entity_{digest}"


def _is_grouped_case_label(value: str | None) -> bool:
    text = _lower(value)
    if not text.startswith("cases "):
        return False
    return bool(re.search(r"\b(?:and|to|through)\b|[,/&-]", text))


def _bundle_link_decision(rows: list[dict], bundle_type: str) -> tuple[str, int, list[str]]:
    labels = _unique_nonempty(
        row.get("case_label_normalized")
        or _case_label_normalized(row.get("workflow_case_label"))
        for row in rows
    )
    raw_labels = _unique_nonempty(row.get("workflow_case_label") for row in rows)
    if any(_is_grouped_case_label(label) for label in raw_labels):
        return "grouped_case_preserved", 10, [
            "explicit_case_label",
            "grouped_case_not_split",
        ]
    if labels:
        return (
            "explicit_label_linked" if len(rows) > 1 else "explicit_label_singleton",
            10,
            ["explicit_case_label", "canonical_explicit_case_label"],
        )
    if bundle_type == "aggregate_event":
        if len(rows) > 1:
            return "aggregate_snapshot_family", 0, [
                "aggregate_scope",
                "compatible_snapshot_family_dimensions",
                "as_of_date_treated_as_snapshot_version",
            ]
        return "aggregate_snapshot_partitioned", 0, [
            "aggregate_scope",
            "snapshot_dimensions_in_bundle_key",
        ]
    compatible_anchor_count = min(
        (
            _compatible_case_anchor_count(left, right)
            for offset, left in enumerate(rows)
            for right in rows[offset + 1 :]
        ),
        default=0,
    )
    if len(rows) > 1 and compatible_anchor_count >= 3:
        return "compatible_unlabelled_records_linked", compatible_anchor_count, [
            "compatible_case_identity_fingerprint",
            "minimum_three_identity_anchors",
        ]
    if any(row.get("_possible_same_case") for row in rows):
        return "possible_same_case", compatible_anchor_count, [
            "fewer_than_three_compatible_identity_anchors"
        ]
    return "independent", 0, ["insufficient_linking_anchors"]


def _event_name_anchor(record: dict) -> str | None:
    for field in (
        "event_name",
        "outbreak_name",
        "event_name_or_context",
        "event_context",
    ):
        value = str(record.get(field) or "").strip()
        if value:
            return _slug(value)
    return None


def _named_vessel_anchor(record: dict) -> str | None:
    for field in ("vessel_name", "ship_name", "vessel"):
        value = str(record.get(field) or "").strip()
        if value:
            return _slug(value)

    text = " ".join(
        str(record.get(field) or "")
        for field in (
            "travel_or_vessel_context",
            "travel_history",
            "contact_setting",
            "source_title",
            "evidence_quote",
        )
    )
    match = re.search(
        r"\b(?:m[/.]?v|m[/.]?s|ss)\s+[a-z0-9][a-z0-9'-]*",
        text,
        flags=re.IGNORECASE,
    )
    return _slug(match.group(0)) if match else None


def _event_location_anchor(record: dict) -> str | None:
    values = _unique_nonempty(
        (
            record.get("geographic_scope"),
            record.get("country"),
            record.get("subnational_location"),
            record.get("locality"),
            record.get("location"),
        )
    )
    return _slug("|".join(values)) if values else None


def _event_time_window_anchor(record: dict) -> str | None:
    start, end = _record_period_dates(record)
    if start or end:
        return f"{start.isoformat() if start else 'open'}_{end.isoformat() if end else 'open'}"

    reporting_period = str(record.get("reporting_period") or "").strip()
    period_dates = re.findall(r"20\d{2}-\d{2}-\d{2}", reporting_period)
    if len(period_dates) >= 2:
        return f"{period_dates[0]}_{period_dates[-1]}"
    if reporting_period and "as of" not in reporting_period.lower():
        return _slug(reporting_period)

    for field in ("as_of_date", "date_anchor", "date_reported", "report_date"):
        match = re.search(r"20\d{2}", str(record.get(field) or ""))
        if match:
            return match.group(0)
    return None


def _event_family_anchors(record: dict) -> dict[str, str]:
    anchors = {
        "event": _event_name_anchor(record),
        "vessel": _named_vessel_anchor(record),
        "location": _event_location_anchor(record),
        "window": _event_time_window_anchor(record),
    }
    return {field: value for field, value in anchors.items() if value}


def _snapshot_period_family(record: dict) -> str:
    reporting_period = str(record.get("reporting_period") or "").strip()
    if reporting_period and "as of" in reporting_period.lower():
        return "dynamic_as_of_snapshot"
    period_dates = re.findall(r"20\d{2}-\d{2}-\d{2}", reporting_period)
    if len(period_dates) >= 2:
        return f"{period_dates[0]}_{period_dates[-1]}"
    if reporting_period:
        return _slug(reporting_period)
    return _event_time_window_anchor(record) or "unspecified"


def _normalized_outbreak_event_key(record: dict) -> str | None:
    text = _lower(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "event_name",
                "outbreak_name",
                "event_name_or_context",
                "event_context",
                "travel_or_vessel_context",
                "geographic_scope",
                "location",
                "source_title",
                "observation_type",
                "count_semantics",
                "evidence_quote",
            )
        )
    )
    case_total = _record_case_total(record)
    death_total = _record_death_total(record)
    aggregate_like = (
        (case_total is not None and case_total > 1)
        or (death_total is not None and death_total > 1)
        or any(token in text for token in ("outbreak", "cluster", "summary"))
    )
    if not aggregate_like:
        return None

    disease = _slug(
        str(record.get("disease_standard_name") or record.get("disease") or "unknown")
    )
    pieces = [f"event_family|disease:{disease}"]
    pieces.extend(
        f"{field}:{value}"
        for field, value in _event_family_anchors(record).items()
    )
    return "|".join(pieces)


def _numeric_value(value) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _record_case_total(record: dict) -> float | None:
    values = [
        _numeric_value(record.get(field))
        for field in (
            "cases_confirmed",
            "cases_probable",
            "cases_suspected",
            "cases_unspecified",
            "case_count",
        )
    ]
    present = [value for value in values if value is not None]
    if present:
        return sum(present)
    metric_value = _numeric_value(record.get("metric_value"))
    metric_category = _lower(record.get("metric_category"))
    if metric_value is not None and "case" in metric_category:
        return metric_value
    return None


def _record_death_total(record: dict) -> float | None:
    value = _numeric_value(record.get("deaths"))
    if value is not None:
        return value
    metric_value = _numeric_value(record.get("metric_value"))
    metric_category = _lower(record.get("metric_category"))
    if metric_value is not None and "death" in metric_category:
        return metric_value
    return None


def _observation_tokens(record: dict) -> set[str]:
    values = [
        record.get("observation_type"),
        record.get("case_definition"),
        record.get("statistical_count_type"),
        record.get("count_semantics"),
        record.get("metric_category"),
    ]
    values.extend(record.get("observation_types") or [])
    return {
        token
        for value in values
        for token in re.split(r"[^a-z0-9_]+", _lower(value))
        if token
    }


def _case_candidate_type(record: dict) -> str:
    tokens = _observation_tokens(record)
    if tokens & {
        "contact",
        "exposure",
        "monitoring",
        "monitored",
        "negative",
        "zero",
        "non_case",
        "noncase",
    }:
        return "non_case_or_monitoring_candidate"
    case_total = _record_case_total(record)
    death_total = _record_death_total(record)
    if (
        tokens
        & {
            "confirmed_case_record",
            "probable_case_record",
            "suspected_case_record",
            "individual_case",
            "patient_record",
        }
        and (case_total is None or case_total <= 1)
        and (death_total is None or death_total <= 1)
    ):
        return "individual_case_candidate"
    if tokens & {
        "outbreak",
        "summary",
        "surveillance",
        "aggregate",
        "regional",
        "country",
        "alert",
    }:
        return "aggregate_event_candidate"
    if case_total is not None and case_total > 1:
        return "aggregate_event_candidate"
    if death_total is not None and death_total > 1:
        return "aggregate_event_candidate"
    if case_total is not None or death_total is not None:
        return "individual_case_candidate"
    return "context_only_candidate"


def _evidence_role_for_candidate(candidate_type: str, source: dict) -> str:
    role = str(source.get("expected_evidence_role") or "").strip()
    if role:
        return role
    if candidate_type == "individual_case_candidate":
        return "individual_case_evidence"
    if candidate_type == "aggregate_event_candidate":
        return "aggregate_event_evidence"
    if candidate_type == "non_case_or_monitoring_candidate":
        return "non_case_or_monitoring_evidence"
    return "source_context"


def _first_quality_reason(row: dict) -> str | None:
    for key in (
        "quality_gate_reasons",
        "quality_gate_warnings",
        "review_reasons",
        "quarantine_reasons",
    ):
        values = row.get(key)
        if isinstance(values, list) and values:
            return str(values[0])
        if values:
            return str(values)
    for key in (
        "review_reason",
        "quality_gate_status",
        "record_final_inclusion_status",
        "case_candidate_status",
    ):
        value = row.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return None


def _is_verified_authority_source(row: dict) -> bool:
    source_type = _lower(row.get("source_type_final") or row.get("source_type"))
    bucket = _lower(row.get("authority_bucket"))
    role = _lower(row.get("evidence_role") or row.get("expected_evidence_role"))
    return any(
        token in source_type
        for token in (
            "official",
            "public_health",
            "health_agency",
            "government",
            "ministry",
            "international",
            "national",
        )
    ) or bucket in {"official_authority", "official", "authority"} or "authority" in role


def _build_case_candidate_datasets(
    *,
    final_case_records: list[dict],
    accepted_records: list[dict],
    pending_review_records: list[dict],
    quarantined_records: list[dict],
    non_primary_observation_records: list[dict],
    source_registry: list[dict] | None = None,
    source_gap_status: str | None = None,
    high_confidence_domain_status_json: str | None = None,
) -> tuple[list[dict], list[dict]]:
    # Imported here to avoid a package-initialization cycle: the recovery
    # helper reuses the extraction span parser, while nodes.__init__ exports
    # this finalization node.
    from ..case_field_recovery import recover_candidate_fields

    source_by_id = _source_lookup(source_registry or [])
    final_ids = {_record_key(record) for record in final_case_records}
    accepted_ids = {_record_key(record) for record in accepted_records}
    pending_ids = {_record_key(record) for record in pending_review_records}
    candidates: list[dict] = []
    seen: set[str] = set()
    for status, rows in (
        ("final", final_case_records),
        ("accepted_context_or_event", accepted_records),
        ("reviewable", pending_review_records),
        ("quarantined", quarantined_records),
        ("context_or_event", non_primary_observation_records),
    ):
        for record in rows:
            rid = _record_key(record)
            if not rid or rid in seen or not _case_count_present(record):
                continue
            seen.add(rid)
            source_id = str(record.get("source_id") or "")
            source = source_by_id.get(source_id) or {}
            candidate_type = _case_candidate_type(record)
            data_product_type = (
                record.get("data_product_type")
                or source.get("data_product_type")
                or source.get("source_product_type")
            )
            expected_evidence_role = _evidence_role_for_candidate(
                candidate_type,
                source,
            )
            event_family_id = _case_event_key(record)
            bundle_scope = _bundle_scope_for_candidate_type(candidate_type)
            case_label_normalized = _case_label_normalized(
                record.get("workflow_case_label")
            )
            case_identity_fingerprint = _case_identity_fingerprint(
                record,
                event_family=event_family_id,
                bundle_scope=bundle_scope,
                case_label_normalized=case_label_normalized,
            )
            candidates.append(
                {
                    "record_id": rid,
                    "case_candidate_id": f"case_candidate_{rid}",
                    "candidate_type": candidate_type,
                    "event_family_id": event_family_id,
                    "bundle_scope": bundle_scope,
                    "case_label_normalized": case_label_normalized,
                    "case_identity_fingerprint": case_identity_fingerprint,
                    "snapshot_as_of_date": _snapshot_as_of_date(record),
                    "count_semantics": record.get("count_semantics"),
                    "statistical_count_type": record.get("statistical_count_type"),
                    "case_candidate_status": (
                        "final"
                        if rid in final_ids
                        else "accepted_context_or_event"
                        if rid in accepted_ids
                        else "reviewable"
                        if rid in pending_ids
                        else status
                    ),
                    "event_cluster_key": event_family_id,
                    "disease": record.get("disease_standard_name") or record.get("disease"),
                    "country": record.get("country"),
                    "subnational_location": record.get("subnational_location"),
                    "geographic_scope": record.get("geographic_scope"),
                    "case_status": record.get("case_definition")
                    or record.get("observation_type"),
                    "cases_confirmed": record.get("cases_confirmed"),
                    "cases_probable": record.get("cases_probable"),
                    "cases_suspected": record.get("cases_suspected"),
                    "cases_unspecified": record.get("cases_unspecified"),
                    "deaths": record.get("deaths"),
                    "date_reported": record.get("date_reported"),
                    "reporting_period": record.get("reporting_period"),
                    "date_onset": record.get("date_onset"),
                    "date_confirmation": record.get("date_confirmation"),
                    "date_death": record.get("date_death"),
                    "age": record.get("age"),
                    "gender": record.get("gender"),
                    "nationality": record.get("nationality"),
                    "outcome": record.get("outcome"),
                    "symptoms": record.get("symptoms"),
                    "hospitalized": record.get("hospitalized"),
                    "intensive_care": record.get("intensive_care"),
                    "isolated": record.get("isolated"),
                    "occupation_or_role": record.get("occupation_or_role"),
                    "cruise_crew": record.get("cruise_crew"),
                    "cruise_passenger_guest": record.get("cruise_passenger_guest"),
                    "contact_with_case": record.get("contact_with_case"),
                    "contact_setting": record.get("contact_setting"),
                    "ship_board_date": record.get("ship_board_date"),
                    "ship_disembark_date": record.get("ship_disembark_date"),
                    "travel_from": record.get("travel_from"),
                    "travel_to": record.get("travel_to"),
                    "confirmation_method": record.get("confirmation_method"),
                    "accession_id": record.get("accession_id"),
                    "workflow_case_label": record.get("workflow_case_label"),
                    "travel_or_vessel_context": record.get(
                        "travel_or_vessel_context"
                    ),
                    "record_local_disease_relevance_status": record.get(
                        "record_local_disease_relevance_status"
                    ),
                    "record_local_target_terms_found": list(
                        record.get("record_local_target_terms_found") or []
                    ),
                    "record_local_incompatible_terms_found": list(
                        record.get("record_local_incompatible_terms_found") or []
                    ),
                    "record_local_numeric_disease_status": record.get(
                        "record_local_numeric_disease_status"
                    ),
                    "record_local_numeric_sentence": record.get(
                        "record_local_numeric_sentence"
                    ),
                    "record_local_numeric_disease_decision": record.get(
                        "record_local_numeric_disease_decision"
                    ),
                    "record_local_numeric_target_terms_found": list(
                        record.get("record_local_numeric_target_terms_found") or []
                    ),
                    "record_local_numeric_incompatible_terms_found": list(
                        record.get(
                            "record_local_numeric_incompatible_terms_found"
                        )
                        or []
                    ),
                    "case_span_id": record.get("case_span_id"),
                    "case_span_quote": record.get("case_span_quote"),
                    "case_span_start": record.get("case_span_start"),
                    "case_span_end": record.get("case_span_end"),
                    "case_span_extraction_method": record.get(
                        "case_span_extraction_method"
                    ),
                    "supporting_chunk_id": record.get("supporting_chunk_id"),
                    "field_provenance_json": record.get("field_provenance_json"),
                    "unsupported_case_fields": list(
                        record.get("unsupported_case_fields") or []
                    ),
                    "focused_recovery_status": record.get(
                        "focused_recovery_status"
                    )
                    or source.get("focused_recovery_status"),
                    "source_gap_status": record.get("source_gap_status")
                    or source_gap_status,
                    "high_confidence_domain_status_json": record.get(
                        "high_confidence_domain_status_json"
                    )
                    or high_confidence_domain_status_json,
                    "source_to_evidence_status": source.get(
                        "source_to_evidence_status"
                    ),
                    "source_only_reason": source.get("source_only_reason"),
                    "extraction_failure_substage": source.get(
                        "extraction_failure_substage"
                    ),
                    "source_exact_page_status": source.get(
                        "source_exact_page_status"
                    ),
                    "source_id": source_id,
                    "source_url": record.get("source_url")
                    or source.get("canonical_url")
                    or source.get("url"),
                    "source_type_final": record.get("source_type_final")
                    or source.get("source_type_final")
                    or source.get("source_type"),
                    "authority_bucket": source.get("authority_bucket"),
                    "data_product_type": data_product_type,
                    "source_product_type": data_product_type,
                    "task_specificity": source.get("task_specificity"),
                    "time_window_fit": source.get("time_window_fit"),
                    "machine_readability": source.get("machine_readability"),
                    "expected_evidence_role": expected_evidence_role,
                    "evidence_quote": record.get("evidence_quote"),
                    "quality_gate_status": record.get("record_final_inclusion_status"),
                    "quality_gate_reasons": list(record.get("quality_gate_reasons") or []),
                    "quality_gate_warnings": list(record.get("quality_gate_warnings") or []),
                    "review_reason": _first_quality_reason(record),
                }
            )
    recovered_candidates: list[dict] = []
    for candidate in candidates:
        recovered, _changed_count = recover_candidate_fields(candidate)
        recovered["case_label_normalized"] = _case_label_normalized(
            recovered.get("workflow_case_label")
        )
        recovered["case_identity_fingerprint"] = _case_identity_fingerprint(
            recovered,
            event_family=recovered.get("event_family_id"),
            bundle_scope=recovered.get("bundle_scope"),
            case_label_normalized=recovered.get("case_label_normalized"),
        )
        recovered_candidates.append(recovered)
    candidates = recovered_candidates
    reviewable = [
        row
        for row in candidates
        if row.get("case_candidate_status") in {"final", "accepted_context_or_event", "reviewable"}
    ]
    return candidates, reviewable


def _bundle_type(rows: list[dict]) -> str:
    types = {str(row.get("candidate_type") or "") for row in rows}
    if types and types <= {"individual_case_candidate"}:
        return "individual_case_like"
    if types and types <= {"aggregate_event_candidate"}:
        return "aggregate_event"
    if types and types <= {"non_case_or_monitoring_candidate"}:
        return "non_case_monitoring"
    return "mixed_or_unclear"


def _recommended_review_action(bundle_type: str) -> str:
    if bundle_type == "individual_case_like":
        return "review_individual_case_evidence"
    if bundle_type == "aggregate_event":
        return "review_aggregate_event_evidence"
    if bundle_type == "non_case_monitoring":
        return "review_non_case_monitoring_evidence"
    return "review_mixed_evidence_bundle"


def _stable_candidate_sort_key(candidate: dict) -> tuple[str, str, str]:
    return (
        str(candidate.get("case_candidate_id") or ""),
        str(candidate.get("record_id") or ""),
        str(candidate.get("source_id") or ""),
    )


def _partition_explicit_label_candidates(rows: list[dict]) -> list[list[dict]]:
    """Keep a canonical label together unless its identifying anchors conflict."""

    partitions: list[list[dict]] = []
    for row in sorted(rows, key=_stable_candidate_sort_key):
        for partition in partitions:
            if all(not _has_hard_case_conflict(row, member) for member in partition):
                partition.append(row)
                break
        else:
            partitions.append([row])
    return partitions


def _partition_unlabelled_case_candidates(rows: list[dict]) -> list[list[dict]]:
    """Link only records sharing three compatible identity anchors."""

    ordered_rows = sorted(rows, key=_stable_candidate_sort_key)
    for row in ordered_rows:
        row["_possible_same_case"] = any(
            not _has_hard_case_conflict(row, other)
            and 0 < _compatible_case_anchor_count(row, other) < 3
            for other in ordered_rows
            if other is not row
        )

    partitions: list[list[dict]] = []
    for row in ordered_rows:
        for partition in partitions:
            if all(
                not _has_hard_case_conflict(row, member)
                and _compatible_case_anchor_count(row, member) >= 3
                for member in partition
            ):
                partition.append(row)
                break
        else:
            partitions.append([row])
    return partitions


def _case_bundle_groups(case_candidates: list[dict]) -> dict[str, list[dict]]:
    primary_groups: dict[str, list[dict]] = {}
    for candidate in case_candidates:
        scope = _effective_bundle_scope(candidate)
        candidate["bundle_scope"] = scope
        candidate["case_label_normalized"] = (
            _case_label_normalized(candidate.get("workflow_case_label"))
            or candidate.get("case_label_normalized")
        )
        primary_groups.setdefault(_case_bundle_group_key(candidate).strip(), []).append(
            candidate
        )

    grouped: dict[str, list[dict]] = {}
    for key, rows in primary_groups.items():
        scope = _effective_bundle_scope(rows[0])
        label = rows[0].get("case_label_normalized")
        if scope == "individual_case" and label:
            partitions = _partition_explicit_label_candidates(rows)
        elif scope == "individual_case":
            partitions = _partition_unlabelled_case_candidates(rows)
        else:
            partitions = [sorted(rows, key=_stable_candidate_sort_key)]
        for partition in partitions:
            if len(partitions) == 1:
                partition_key = key
            else:
                anchors = _case_anchor_values(partition[0])
                partition_suffix = "|".join(
                    f"{field}:{value}"
                    for field, value in sorted(anchors.items())
                    if field in _HARD_CASE_ANCHOR_FIELDS
                )
                partition_key = f"{key}|partition:{partition_suffix or partition[0].get('record_id')}"
            grouped[partition_key] = partition
    return grouped


def _build_case_evidence_bundles(case_candidates: list[dict]) -> list[dict]:
    grouped = _case_bundle_groups(case_candidates)

    bundles: list[dict] = []
    for index, (key, rows) in enumerate(sorted(grouped.items()), start=1):
        source_ids = _unique_nonempty(row.get("source_id") for row in rows)
        source_urls = _unique_nonempty(row.get("source_url") for row in rows)
        source_types = _unique_nonempty(row.get("source_type_final") for row in rows)
        source_product_types = _unique_nonempty(
            row.get("source_product_type") or row.get("data_product_type")
            for row in rows
        )
        statuses = _unique_nonempty(row.get("case_candidate_status") for row in rows)
        evidence_quotes = _unique_nonempty(row.get("evidence_quote") for row in rows)
        reasons = _unique_nonempty(_first_quality_reason(row) for row in rows)
        source_gap_statuses = _unique_nonempty(
            row.get("source_gap_status") for row in rows
        )
        local_disease_statuses = _unique_nonempty(
            row.get("record_local_disease_relevance_status") for row in rows
        )
        high_confidence_domain_statuses = _unique_nonempty(
            row.get("high_confidence_domain_status_json") for row in rows
        )
        source_to_evidence_status = {
            str(row.get("source_id")): row.get("source_to_evidence_status")
            for row in rows
            if row.get("source_id") and row.get("source_to_evidence_status")
        }
        non_target_disease_block_count = sum(
            1
            for row in rows
            if row.get("case_candidate_status") == "quarantined"
            and (
                row.get("record_local_disease_relevance_status")
                == "incompatible_disease"
                or row.get("record_local_numeric_disease_status")
                == "incompatible_disease"
                or "local_evidence_disease_mismatch"
                in " ".join(str(v) for v in row.get("quality_gate_reasons") or [])
            )
        )
        current_bundle_type = _bundle_type(rows)
        merged_case_fields = merge_case_candidate_fields(
            rows,
            bundle_type=current_bundle_type,
        )
        official_source_count = len(
            {
                str(row.get("source_id") or "")
                for row in rows
                if row.get("source_id") and _is_verified_authority_source(row)
            }
        )
        peer_reviewed_source_count = len(
            {
                str(row.get("source_id") or "")
                for row in rows
                if row.get("source_id")
                and any(
                    token in _lower(row.get("source_type_final"))
                    for token in ("academic", "peer", "journal", "literature")
                )
            }
        )
        structured_database_source_count = len(
            {
                str(row.get("source_id") or "")
                for row in rows
                if row.get("source_id")
                and (
                    "structured" in _lower(row.get("source_type_final"))
                    or "database" in _lower(row.get("source_product_type"))
                    or "sequence_database" in _lower(row.get("source_product_type"))
                )
            }
        )
        high_trust_source_count = len(
            {
                source_id
                for source_id in [
                    *(str(row.get("source_id") or "") for row in rows if _is_verified_authority_source(row)),
                    *(
                        str(row.get("source_id") or "")
                        for row in rows
                        if any(
                            token in _lower(row.get("source_type_final"))
                            for token in ("academic", "peer", "journal", "literature", "structured")
                        )
                    ),
                ]
                if source_id
            }
        )
        date_values = _unique_nonempty(
            row.get("snapshot_as_of_date")
            or row.get("date_reported")
            or row.get("reporting_period")
            for row in rows
        )
        count_values = _unique_nonempty(
            str(_record_case_total(row) or "") for row in rows if _record_case_total(row) is not None
        )
        snapshot_status = (
            "compatible_dynamic_snapshots"
            if current_bundle_type == "aggregate_event"
            and len(source_ids) > 1
            and (len(date_values) > 1 or len(count_values) > 1)
            else "single_snapshot_or_not_applicable"
        )
        link_status, link_score, link_reasons = _bundle_link_decision(
            rows,
            current_bundle_type,
        )
        count_semantics = rows[0].get("count_semantics") or rows[0].get(
            "statistical_count_type"
        )
        location_scope = "|".join(
            _unique_nonempty(
                (
                    rows[0].get("geographic_scope"),
                    rows[0].get("country"),
                    rows[0].get("subnational_location"),
                )
            )
        )
        bundles.append(
            {
                "case_evidence_bundle_id": f"case_bundle_{index:04d}",
                "case_entity_id": _case_entity_id(key),
                "link_status": link_status,
                "link_score": link_score,
                "link_reasons": link_reasons,
                "field_conflicts_json": json.dumps(
                    merged_case_fields["conflicts"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "event_family_id": rows[0].get("event_family_id") or key,
                "bundle_scope": _effective_bundle_scope(rows[0]),
                "case_label_normalized": rows[0].get("case_label_normalized"),
                "case_identity_fingerprint": (
                    rows[0].get("case_identity_fingerprint")
                    or _case_identity_fingerprint(rows[0])
                ),
                "snapshot_as_of_date": rows[0].get("snapshot_as_of_date"),
                "count_semantics": count_semantics,
                "location_scope": location_scope,
                "reporting_period": rows[0].get("reporting_period"),
                "event_cluster_key": key,
                "bundle_type": current_bundle_type,
                "member_record_ids": _unique_nonempty(row.get("record_id") for row in rows),
                "case_candidate_ids": _unique_nonempty(
                    row.get("case_candidate_id") for row in rows
                ),
                "source_ids": source_ids,
                "source_urls": source_urls,
                "source_type_finals": source_types,
                "source_product_types": source_product_types,
                "candidate_statuses": statuses,
                "supporting_source_count": len(source_ids),
                "verified_authority_source_count": official_source_count,
                "high_trust_source_count": high_trust_source_count,
                "official_source_count": official_source_count,
                "peer_reviewed_source_count": peer_reviewed_source_count,
                "structured_database_source_count": structured_database_source_count,
                "source_stage_summary": (
                    f"{len(source_ids)} sources / {len(rows)} extracted candidates"
                ),
                "source_gap_status": (
                    source_gap_statuses[0] if source_gap_statuses else None
                ),
                "local_disease_relevance_status": (
                    local_disease_statuses[0] if local_disease_statuses else None
                ),
                "high_confidence_domain_status_json": (
                    high_confidence_domain_statuses[0]
                    if high_confidence_domain_statuses
                    else None
                ),
                "source_exact_page_status": json.dumps(
                    {
                        str(row.get("source_id")): row.get(
                            "source_exact_page_status"
                        )
                        for row in rows
                        if row.get("source_id")
                        and row.get("source_exact_page_status")
                    },
                    ensure_ascii=False,
                ),
                "focused_recovery_status": json.dumps(
                    {
                        str(row.get("source_id")): row.get(
                            "focused_recovery_status"
                        )
                        for row in rows
                        if row.get("source_id")
                        and row.get("focused_recovery_status")
                    },
                    ensure_ascii=False,
                ),
                "source_to_evidence_status_json": (
                    json.dumps(source_to_evidence_status, ensure_ascii=False)
                    if source_to_evidence_status
                    else None
                ),
                "merged_case_fields": merged_case_fields["values"],
                "field_source_ids": merged_case_fields["source_ids"],
                "field_evidence_quotes": merged_case_fields["evidence_quotes"],
                "field_provenance_json": json.dumps(
                    merged_case_fields["field_provenance"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "field_conflicts": merged_case_fields["conflicts"],
                "field_recovery_method": merged_case_fields["method"],
                "non_target_disease_block_count": non_target_disease_block_count,
                "snapshot_compatibility_status": snapshot_status,
                "member_record_count": len(rows),
                "evidence_quotes": evidence_quotes[:5],
                "best_evidence_quote": evidence_quotes[0] if evidence_quotes else None,
                "main_blocking_reason": reasons[0] if reasons else None,
                "recommended_review_action": _recommended_review_action(
                    current_bundle_type
                ),
                "bundle_status": (
                    "contains_final_case_candidate"
                    if "final" in statuses
                    else "reviewable_or_quarantined"
                ),
                "method": "deterministic_case_candidate_evidence_bundle_v1",
            }
        )
    return bundles


def _annotate_case_candidates_with_link_decisions(
    case_candidates: list[dict],
    case_evidence_bundles: list[dict],
) -> None:
    candidate_by_id = {
        str(candidate.get("case_candidate_id") or ""): candidate
        for candidate in case_candidates
        if candidate.get("case_candidate_id")
    }
    for bundle in case_evidence_bundles:
        link_fields = {
            "case_entity_id": bundle.get("case_entity_id"),
            "link_status": bundle.get("link_status"),
            "link_score": bundle.get("link_score"),
            "link_reasons": list(bundle.get("link_reasons") or []),
            "field_conflicts_json": bundle.get("field_conflicts_json"),
        }
        for candidate_id in bundle.get("case_candidate_ids") or []:
            candidate = candidate_by_id.get(str(candidate_id))
            if candidate is not None:
                candidate.update(link_fields)


def _build_evidence_product_dataset(
    case_candidates: list[dict],
    source_registry: list[dict],
) -> list[dict]:
    source_by_id = _source_lookup(source_registry)
    evidence_rows: list[dict] = []
    used_sources: set[str] = set()
    for index, candidate in enumerate(case_candidates, start=1):
        source_id = str(candidate.get("source_id") or "")
        source = source_by_id.get(source_id) or {}
        used_sources.add(source_id)
        candidate_type = str(candidate.get("candidate_type") or "context_only_candidate")
        evidence_rows.append(
            {
                "evidence_id": f"evidence_{index:04d}",
                "record_id": candidate.get("record_id"),
                "case_candidate_id": candidate.get("case_candidate_id"),
                "candidate_type": candidate_type,
                "source_id": source_id,
                "source_url": candidate.get("source_url")
                or source.get("canonical_url")
                or source.get("url"),
                "source_product_type": candidate.get("source_product_type")
                or source.get("data_product_type"),
                "evidence_role": candidate.get("expected_evidence_role")
                or _evidence_role_for_candidate(candidate_type, source),
                "observation_type": candidate.get("case_status"),
                "location": candidate.get("geographic_scope")
                or candidate.get("subnational_location")
                or candidate.get("country"),
                "date_or_period": candidate.get("date_reported")
                or candidate.get("reporting_period"),
                "case_status": candidate.get("case_status"),
                "case_count": _record_case_total(candidate),
                "death_count": _record_death_total(candidate),
                "confidence": candidate.get("confidence")
                or candidate.get("quality_gate_status")
                or candidate.get("case_candidate_status"),
                "quality_status": candidate.get("quality_gate_status")
                or candidate.get("case_candidate_status"),
                "review_reason": candidate.get("review_reason")
                or _first_quality_reason(candidate),
                "evidence_quote": candidate.get("evidence_quote"),
            }
        )

    next_index = len(evidence_rows) + 1
    high_value_roles = {
        "individual_case_evidence",
        "aggregate_event_evidence",
        "non_case_or_monitoring_evidence",
    }
    low_value_products = {
        "background_fact_sheet",
        "search_or_index_page",
        "policy_or_protocol",
        "unrelated_or_other",
    }
    for source in source_registry:
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in used_sources:
            continue
        role = str(source.get("expected_evidence_role") or "").strip()
        product_type = str(source.get("data_product_type") or "").strip()
        if role not in high_value_roles or product_type in low_value_products:
            continue
        evidence_rows.append(
            {
                "evidence_id": f"evidence_{next_index:04d}",
                "record_id": None,
                "case_candidate_id": None,
                "candidate_type": "context_only_candidate",
                "source_id": source_id,
                "source_url": source.get("canonical_url") or source.get("url"),
                "source_product_type": product_type,
                "evidence_role": role,
                "observation_type": None,
                "location": source.get("jurisdiction") or source.get("country"),
                "date_or_period": source.get("publication_date")
                or source.get("date")
                or source.get("published_date"),
                "case_status": None,
                "case_count": None,
                "death_count": None,
                "confidence": source.get("confidence") or "source_only",
                "quality_status": source.get("extractable_status") or "source_only",
                "review_reason": "high_value_source_retained_without_extracted_record",
                "evidence_quote": None,
            }
        )
        next_index += 1
    return evidence_rows


def _joined_requirement_value(requirements: list[dict], *keys: str) -> str | None:
    values = _unique_nonempty(
        requirement.get(key)
        for requirement in requirements
        for key in keys
    )
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return " | ".join(values)


def _requirement_period_bound(
    requirements: list[dict],
    *,
    start: bool,
) -> str | None:
    keys = (
        ("period_start", "reporting_period_start")
        if start
        else ("period_end", "reporting_period_end")
    )
    dated: list[tuple[date, str]] = []
    fallback: list[str] = []
    for requirement in requirements:
        for key in keys:
            text = str(requirement.get(key) or "").strip()
            if not text:
                continue
            parsed = _iso_date(text)
            if parsed:
                dated.append((parsed, text))
            elif text not in fallback:
                fallback.append(text)
    if dated:
        selected = (
            min(dated, key=lambda item: item[0])
            if start
            else max(dated, key=lambda item: item[0])
        )
        return selected[1]
    if not fallback:
        return None
    return fallback[0] if len(fallback) == 1 else " | ".join(fallback)


def _apply_requirement_linkage_metadata(
    record: dict,
    requirement_ids: list[str],
    requirements_by_id: dict[str, dict],
) -> None:
    if not requirement_ids:
        return
    record["matched_requirement_id"] = requirement_ids[0]
    record["matched_requirement_ids"] = requirement_ids
    matched = [requirements_by_id[req_id] for req_id in requirement_ids if req_id in requirements_by_id]
    if len(requirement_ids) == 1:
        record["requirement_match_status"] = "linked_to_task_requirement"
    else:
        record["requirement_match_status"] = "linked_to_multiple_task_requirements"
    if not matched:
        return
    geography = _joined_requirement_value(matched, "geography", "location")
    if geography:
        record["requirement_geography"] = geography
    period_start = _requirement_period_bound(matched, start=True)
    if period_start:
        record["requirement_period_start"] = period_start
    period_end = _requirement_period_bound(matched, start=False)
    if period_end:
        record["requirement_period_end"] = period_end
    period_label = _joined_requirement_value(
        matched,
        "reporting_period_label",
        "period_label",
        "year",
    )
    if period_label:
        record["requirement_period_label"] = period_label
    period_basis = _joined_requirement_value(matched, "period_basis")
    if period_basis:
        record["requirement_period_basis"] = period_basis
    granularity = _joined_requirement_value(matched, "time_granularity")
    if granularity:
        record["requirement_time_granularity"] = granularity


def _enrich_records_with_requirement_linkage(records: list[dict], state: dict) -> list[dict]:
    if not records:
        return []
    registry_by_id, documents_by_source, chunks_by_id = _build_record_linkage_indexes(state)
    requirements_by_id = _requirement_lookup(state)
    has_requirements = bool(_safe_list(state, "source_coverage_requirements"))
    enriched_records: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row = dict(record)
        requirement_ids = _record_requirement_ids_from_lineage(
            row,
            state,
            registry_by_id,
            documents_by_source,
            chunks_by_id,
        )
        if requirement_ids:
            row["coverage_requirement_ids"] = requirement_ids
            _apply_requirement_linkage_metadata(row, requirement_ids, requirements_by_id)
        elif has_requirements:
            reasons = [
                str(value)
                for value in (row.get("record_task_fit_reasons") or [])
                if str(value or "").strip()
            ]
            if "requirement_linkage_missing" not in reasons:
                reasons.append("requirement_linkage_missing")
            row["record_task_fit_reasons"] = reasons
        source = registry_by_id.get(str(row.get("source_id") or "")) or {}
        if source:
            row.setdefault(
                "source_url",
                source.get("canonical_url") or source.get("url"),
            )
            row.setdefault("source_role_final", source.get("source_role_final"))
            row.setdefault("target_fit_status", source.get("target_fit_status"))
        _apply_best_available_geography_fit(row, state)
        if not row.get("record_period_fit_status"):
            row["record_period_fit_status"] = _best_available_period_fit(row, state)
        enriched_records.append(row)
    return enriched_records


def _best_available_reason(record: dict, state: dict) -> str:
    flags = {
        str(flag)
        for flag in (record.get("quality_gate_blocking_flags") or [])
        if flag
    }
    reason_text = _lower(
        " ".join(
            str(value or "")
            for value in (
                record.get("quarantine_reason"),
                " ".join(str(value) for value in record.get("quality_gate_reasons") or []),
                " ".join(flags),
            )
        )
    )
    period_fit = _best_available_period_fit(record, state)
    if period_fit != "exact" or "period" in reason_text or "date" in reason_text:
        return "period_mismatch_best_available_context"
    if "geography" in reason_text or "broader_than_task" in reason_text:
        return "geography_mismatch_best_available_context"
    return "near_match_best_available_context"


def _build_best_available_context_records(
    quarantined_records: list[dict],
    state: dict,
) -> list[dict]:
    """Keep useful near-miss records visible without relaxing strict final gates."""

    if not _direct_collection_mode(state):
        return []
    registry_by_id = {
        str(row.get("source_id")): row
        for row in _safe_list(state, "source_registry")
        if isinstance(row, dict) and row.get("source_id")
    }
    disallowed_flags = {
        "disease_pathogen_incompatible_with_task",
        "non_seasonal_influenza_subtype",
        "record_geography_outside_task",
        "missing_direct_collection_geography",
        "source_not_task_relevant",
        "source_period_mismatch",
        "metric_row_binding_unresolved",
        "missing_direct_collection_metric",
        "missing_direct_collection_source_provenance",
    }
    best_available: list[dict] = []
    for record in quarantined_records:
        if not isinstance(record, dict):
            continue
        flags = {
            str(flag)
            for flag in (record.get("quality_gate_blocking_flags") or [])
            if flag
        }
        if flags & disallowed_flags:
            continue
        if not _record_has_public_health_metric(record):
            continue
        if not _record_source_high_trust(record, registry_by_id):
            continue
        if not (record.get("source_url") or record.get("source_id")):
            continue
        if not (record.get("evidence_quote") or record.get("source_title")):
            continue
        row = dict(record)
        requirement_ids = _requirement_ids_for_best_available_record(
            row, state, registry_by_id
        )
        if requirement_ids:
            row["coverage_requirement_ids"] = requirement_ids
            row.setdefault("source_url", (registry_by_id.get(str(row.get("source_id") or "")) or {}).get("canonical_url"))
        _apply_best_available_geography_fit(row, state)
        if not row.get("record_period_fit_status"):
            row["record_period_fit_status"] = _best_available_period_fit(row, state)
        if not row.get("best_available_reason"):
            row["best_available_reason"] = _best_available_reason(row, state)
        best_available.append(row)
    return best_available


def _refresh_source_coverage_audit(
    requirements: list[dict],
    registry: list[dict],
    documents: list[dict],
    extracted_records: list[dict],
    accepted_records: list[dict],
    existing_audit: dict | None = None,
) -> dict:
    if not requirements:
        return existing_audit or {}
    audit = build_source_coverage_audit(
        requirements,
        registry,
        documents,
        evidence_rows=extracted_records,
    )
    extracted_by_source: dict[str, int] = {}
    accepted_by_source: dict[str, int] = {}
    extracted_record_ids_by_requirement: dict[str, set[str]] = {}
    accepted_record_ids_by_requirement: dict[str, set[str]] = {}
    for record in extracted_records:
        source_id = str(record.get("source_id") or "")
        if source_id:
            extracted_by_source[source_id] = extracted_by_source.get(source_id, 0) + 1
        record_id = str(record.get("record_id") or record.get("source_record_id") or "")
        for requirement_id in record.get("coverage_requirement_ids") or []:
            if record_id and requirement_id:
                extracted_record_ids_by_requirement.setdefault(str(requirement_id), set()).add(record_id)
    for record in accepted_records:
        source_id = str(record.get("source_id") or "")
        if source_id:
            accepted_by_source[source_id] = accepted_by_source.get(source_id, 0) + 1
        record_id = str(record.get("record_id") or record.get("source_record_id") or "")
        for requirement_id in record.get("coverage_requirement_ids") or []:
            if record_id and requirement_id:
                accepted_record_ids_by_requirement.setdefault(str(requirement_id), set()).add(record_id)

    extracted_requirement_count = 0
    accepted_requirement_count = 0
    best_available_requirement_count = 0
    total_extracted = 0
    total_accepted = 0
    total_best_available = 0
    parsed_requirement_count = 0
    rows: list[dict] = []
    for row in audit.get("requirements") or []:
        source_ids = {
            str(source_id)
            for source_id in (
                row.get("matched_source_ids")
                or row.get("fetched_source_ids")
                or row.get("parsed_source_ids")
                or []
            )
            if source_id
        }
        extracted_count = sum(extracted_by_source.get(source_id, 0) for source_id in source_ids)
        accepted_count = sum(accepted_by_source.get(source_id, 0) for source_id in source_ids)
        requirement_id = str(row.get("requirement_id") or "")
        extracted_record_ids = {
            str(record.get("record_id") or record.get("source_record_id") or "")
            for record in extracted_records
            if _record_matches_requirement_exact(record, row, source_ids)
            and (record.get("record_id") or record.get("source_record_id"))
        }
        accepted_record_ids = {
            str(record.get("record_id") or record.get("source_record_id") or "")
            for record in accepted_records
            if _record_matches_requirement_exact(record, row, source_ids)
            and (record.get("record_id") or record.get("source_record_id"))
        }
        core_families = _requirement_core_metric_families(row)
        accepted_core_record_ids = {
            str(record.get("record_id") or record.get("source_record_id") or "")
            for record in accepted_records
            if str(record.get("record_id") or record.get("source_record_id") or "")
            in accepted_record_ids
            and _record_matches_core_metric_family(record, core_families)
        }
        source_record_ids = {
            str(record.get("record_id") or record.get("source_record_id") or "")
            for record in extracted_records
            if str(record.get("source_id") or "") in source_ids
            and (record.get("record_id") or record.get("source_record_id"))
        }
        best_available_record_ids = sorted(source_record_ids - extracted_record_ids)
        accepted_source_ids = {
            str(record.get("source_id") or "")
            for record in accepted_records
            if str(record.get("record_id") or record.get("source_record_id") or "") in accepted_record_ids
            and record.get("source_id")
        }
        extracted_count = len(extracted_record_ids)
        accepted_count = len(accepted_record_ids)
        has_core_requirement = bool(core_families)
        core_accepted = bool(accepted_core_record_ids)
        requirement_complete = bool(accepted_count) and (
            core_accepted or not has_core_requirement
        )
        if requirement_complete:
            strict_status = "core_metric_accepted" if core_accepted else "accepted_exact_record"
        elif accepted_count:
            strict_status = "edge_metric_only"
        elif best_available_record_ids:
            strict_status = "best_available_only"
        elif extracted_count:
            strict_status = "records_quarantined"
        elif row.get("parsed"):
            strict_status = "parsed_no_records"
        elif row.get("unusable"):
            strict_status = "target_source_unusable"
        else:
            strict_status = "target_source_missing"
        refreshed = dict(row)
        refreshed.update(
            {
                "extracted": extracted_count > 0,
                "extracted_record_count": extracted_count,
                "extracted_record_ids": sorted(extracted_record_ids),
                "accepted": accepted_count > 0,
                "coverage_complete": requirement_complete,
                "strict_status": strict_status,
                "task_value_status": strict_status,
                "accepted_core_record_count": len(accepted_core_record_ids),
                "accepted_core_record_ids": sorted(accepted_core_record_ids),
                "accepted_record_count": accepted_count,
                "accepted_record_ids": sorted(accepted_record_ids),
                "accepted_source_ids": sorted(accepted_source_ids),
                "best_available_record_count": len(best_available_record_ids),
                "best_available_record_ids": best_available_record_ids,
            }
        )
        if refreshed.get("parsed"):
            parsed_requirement_count += 1
        if extracted_count:
            extracted_requirement_count += 1
            total_extracted += extracted_count
        if accepted_count:
            accepted_requirement_count += 1
            total_accepted += accepted_count
        if best_available_record_ids and not accepted_count:
            best_available_requirement_count += 1
            total_best_available += len(best_available_record_ids)
        if requirement_complete:
            refreshed["missing_reason"] = None
        elif accepted_count:
            refreshed["missing_reason"] = "edge_metric_only"
        elif refreshed.get("extracted"):
            refreshed["missing_reason"] = "records_quarantined"
        elif best_available_record_ids:
            refreshed["missing_reason"] = "best_available_only"
        elif refreshed.get("parsed"):
            refreshed["missing_reason"] = "parsed_no_records"
        elif refreshed.get("unusable"):
            refreshed["missing_reason"] = "target_alias_error_page"
        elif refreshed.get("fetch_failed") or (
            refreshed.get("fetch_attempted") and not refreshed.get("fetched")
        ):
            refreshed["missing_reason"] = "target_fetch_failed"
        elif refreshed.get("discovered"):
            refreshed["missing_reason"] = "target_source_discovered_not_fetched"
        else:
            refreshed["missing_reason"] = "target_source_missing"
        rows.append(refreshed)

    requirement_count = len(rows)
    complete_requirement_count = sum(1 for row in rows if row.get("coverage_complete"))
    partial_requirement_count = (
        requirement_count - complete_requirement_count if requirement_count else 0
    )
    missing_requirement_ids = [
        row.get("requirement_id")
        for row in rows
        if row.get("requirement_id")
        and not row.get("coverage_complete")
    ]
    if not requirement_count:
        coverage_completeness_status = "not_required"
    elif complete_requirement_count == requirement_count:
        coverage_completeness_status = "complete_target_coverage"
    elif complete_requirement_count:
        coverage_completeness_status = "partial_target_coverage"
    else:
        coverage_completeness_status = "no_target_coverage"

    audit.update(
        {
            "requirements": rows,
            "extracted_requirement_count": extracted_requirement_count,
            "accepted_requirement_count": accepted_requirement_count,
            "extracted_record_count": total_extracted,
            "accepted_record_count": total_accepted,
            "best_available_requirement_count": best_available_requirement_count,
            "best_available_record_count": total_best_available,
            "complete_requirement_count": complete_requirement_count,
            "partial_requirement_count": partial_requirement_count,
            "missing_requirement_ids": missing_requirement_ids,
            "coverage_completeness_status": coverage_completeness_status,
        }
    )
    if requirement_count and complete_requirement_count == requirement_count:
        audit["coverage_status"] = "target_official_source_accepted"
    elif complete_requirement_count:
        audit["coverage_status"] = "partial_target_coverage"
    elif accepted_requirement_count:
        audit["coverage_status"] = "edge_metric_only"
    elif extracted_requirement_count:
        audit["coverage_status"] = "records_quarantined"
    elif best_available_requirement_count:
        audit["coverage_status"] = "best_available_only"
    elif parsed_requirement_count:
        audit["coverage_status"] = "parsed_no_records"
    return audit


def _refine_direct_coverage_status(
    audit: dict,
    *,
    content_fetch_summary: dict | None = None,
    structured_extraction_summary: dict | None = None,
) -> dict:
    if not audit:
        return audit
    coverage_completeness_status = str(
        audit.get("coverage_completeness_status") or ""
    )
    if coverage_completeness_status in {
        "complete_target_coverage",
        "partial_target_coverage",
    }:
        return audit

    content_fetch_summary = content_fetch_summary or {}
    structured_extraction_summary = structured_extraction_summary or {}
    usable_task_docs = int(
        content_fetch_summary.get("usable_task_collection_document_count") or 0
    )
    if usable_task_docs > 0:
        return audit

    refined_status: str | None = None
    if content_fetch_summary.get("target_unusable_needs_fallback"):
        if content_fetch_summary.get("fallback_fetch_attempted"):
            refined_status = "fallback_target_fetch_failed"
        else:
            refined_status = "target_alias_error_page_needs_fallback"
    elif structured_extraction_summary.get("no_task_collection_document"):
        refined_status = "no_task_collection_document"

    if not refined_status:
        return audit

    out = dict(audit)
    out["coverage_status"] = refined_status
    out["coverage_failure_stage"] = refined_status
    rows: list[dict] = []
    for row in out.get("requirements") or []:
        refreshed = dict(row)
        if not (
            refreshed.get("accepted")
            or refreshed.get("extracted")
            or refreshed.get("parsed")
        ):
            refreshed["missing_reason"] = refined_status
        rows.append(refreshed)
    out["requirements"] = rows
    return out


def _records_need_claim_annotation(records: list[dict]) -> bool:
    return any(
        isinstance(record, dict)
        and record.get("record_id")
        and not record.get("claim_ids")
        for record in records
    )


def _claim_annotated_records_for_finalization(
    state: DataCollectionState,
    records: list[dict],
) -> list[dict]:
    claims = _safe_list(state, "claims")
    if not claims or not _records_need_claim_annotation(records):
        return records
    return annotate_records_with_claim_corroboration(
        [dict(record) for record in records if isinstance(record, dict)],
        claims,
        _safe_list(state, "corroborated_events"),
    )


def _collect_workflow_summaries(
    state: DataCollectionState,
    policy: FinalPackagePolicy,
) -> dict:
    return {field: state.get(field) for field in policy.workflow_summary_fields}


def _build_data_dictionary(
    state: DataCollectionState,
    policy: FinalPackagePolicy,
) -> list[dict]:
    schema = state.get("collection_schema") or {}
    core_fields = schema.get("core_fields") if isinstance(schema, dict) else None
    if core_fields:
        result: list[dict] = []
        for field in core_fields:
            result.append(
                {
                    "name": field.get("name"),
                    "type": field.get("type"),
                    "required": field.get("required", False),
                    "description": field.get("description") or "",
                }
            )
        return result
    return [
        {"name": name, "type": "unknown", "required": False, "description": ""}
        for name in policy.final_dataset_field_order
    ]


# ---------------------------------------------------------------------------
# Synthetic fixture detection
# ---------------------------------------------------------------------------


def _section_has_fixture(obj, markers: list[str]) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        if obj.get("is_fixture_document") is True:
            return True
        if obj.get("fixture_id"):
            return True
        meta = obj.get("metadata")
        if isinstance(meta, dict):
            if meta.get("synthetic_fixture") is True:
                return True
            if meta.get("not_real_public_health_data") is True:
                return True
            if meta.get("fixture_id"):
                return True
        for value in obj.values():
            if _section_has_fixture(value, markers):
                return True
        return False
    if isinstance(obj, list):
        return any(_section_has_fixture(v, markers) for v in obj)
    if isinstance(obj, str):
        return any(marker in obj for marker in markers)
    return False


def _detect_synthetic_fixture_data(
    state: DataCollectionState,
    policy: FinalPackagePolicy,
) -> tuple[bool, str | None]:
    markers = list(policy.synthetic_fixture_markers or [])
    for key in (
        "documents",
        "evidence_chunks",
        "raw_records",
        "validated_records",
        "normalized_records",
        "conflicts",
        "human_review_queue",
    ):
        items = state.get(key) or []
        if _section_has_fixture(items, markers):
            return (
                True,
                (
                    "This final data package contains synthetic fixture data used "
                    "only for deterministic workflow testing; it is not real "
                    "public health data."
                ),
            )
    return False, None


# ---------------------------------------------------------------------------
# Provenance + metadata
# ---------------------------------------------------------------------------


def _build_provenance_manifest(state: DataCollectionState) -> dict:
    normalized = _safe_list(state, "normalized_records")
    conflicts = _safe_list(state, "conflicts")

    def _count(records, field):
        return sum(
            1
            for r in records
            if isinstance(r, dict)
            and r.get(field) not in (None, "", [], {})
        )

    return {
        "source_count": len(_safe_list(state, "source_registry")),
        "source_identity_assessment_count": len(
            _safe_list(state, "source_identity_assessments")
        ),
        "document_count": len(_safe_list(state, "documents")),
        "evidence_chunk_count": len(_safe_list(state, "evidence_chunks")),
        "raw_record_count": len(_safe_list(state, "raw_records")),
        "validated_record_count": len(_safe_list(state, "validated_records")),
        "normalized_record_count": len(normalized),
        "linked_event_count": len(_safe_list(state, "linked_events")),
        "event_cluster_count": len(_safe_list(state, "event_clusters")),
        "duplicate_cluster_count": len(_safe_list(state, "duplicate_clusters")),
        "validation_case_count": len(_safe_list(state, "validation_cases")),
        "validation_comparison_count": len(_safe_list(state, "validation_comparisons")),
        "validation_result_count": len(_safe_list(state, "validation_results")),
        "claim_count": len(_safe_list(state, "claims")),
        "claim_comparison_count": len(_safe_list(state, "claim_comparisons")),
        "corroborated_event_count": len(_safe_list(state, "corroborated_events")),
        "anomaly_result_count": len(_safe_list(state, "anomaly_results")),
        "applied_human_review_decision_count": len(
            _safe_list(state, "applied_human_review_decisions")
        ),
        "rejected_human_review_decision_count": len(
            _safe_list(state, "rejected_human_review_decisions")
        ),
        "human_review_audit_entry_count": len(
            _safe_list(state, "human_review_audit_trail")
        ),
        "final_dataset_post_review_count": len(
            _safe_list(state, "final_dataset_post_review")
        ),
        "conflict_count": len(conflicts),
        "human_review_item_count": len(_safe_list(state, "human_review_queue")),
        "records_with_source_url_count": _count(normalized, "source_url"),
        "records_with_evidence_quote_count": _count(normalized, "evidence_quote"),
        "records_with_supporting_chunk_id_count": _count(normalized, "supporting_chunk_id"),
        "records_with_linked_event_id_count": _count(normalized, "linked_event_id"),
        "records_with_event_cluster_id_count": _count(normalized, "event_cluster_id"),
        "countable_record_count": sum(
            1 for r in normalized if isinstance(r, dict) and r.get("countable") is True
        ),
        "non_countable_duplicate_count": sum(
            1
            for r in normalized
            if isinstance(r, dict)
            and r.get("event_member_status") == "non_countable_duplicate"
        ),
        "generic_record_count": sum(
            1
            for r in normalized
            if isinstance(r, dict)
            and r.get("record_schema") == "generic_public_health_record"
        ),
        "legacy_hantavirus_record_count": sum(
            1
            for r in normalized
            if isinstance(r, dict) and r.get("disease") == "Hantavirus disease"
        ),
        "conflicts_with_record_ids_count": sum(
            1
            for c in conflicts
            if isinstance(c, dict) and (c.get("record_ids") or [])
        ),
        "conflicts_requiring_human_review_count": sum(
            1
            for c in conflicts
            if isinstance(c, dict) and c.get("requires_human_review")
        ),
    }


def _detect_llm_used(state: DataCollectionState) -> bool:
    for key in ("normalized_records", "validated_records", "raw_records"):
        for record in state.get(key) or []:
            if isinstance(record, dict) and record.get("llm_used"):
                return True
    summary = state.get("llm_extraction_summary") or {}
    if isinstance(summary, dict):
        if summary.get("llm_enabled") and (summary.get("llm_success_count") or 0) > 0:
            return True
    return False


def _build_package_metadata(
    state: DataCollectionState,
    policy: FinalPackagePolicy,
    contains_fixture: bool,
    fixture_notice: str | None,
    trace: list[dict],
    llm_used: bool,
) -> dict:
    spec = state.get("collection_spec") or {}
    llm_summary = state.get("llm_extraction_summary") or {}
    search_summary = state.get("source_search_execution_summary") or {}
    return {
        "package_name": "data_collection_workflow_final_package",
        "package_version": policy.package_version,
        "package_builder": policy.package_builder,
        "generated_at": _package_generated_at(state, policy, contains_fixture, llm_used),
        "disease": spec.get("disease") if isinstance(spec, dict) else None,
        "geography": spec.get("geography") if isinstance(spec, dict) else None,
        "time_window": spec.get("time_window") if isinstance(spec, dict) else None,
        "workflow_node_count": len(trace),
        "contains_synthetic_fixture_data": contains_fixture,
        "synthetic_fixture_notice": fixture_notice,
        "llm_used": llm_used,
        "llm_provider": llm_summary.get("llm_provider") if isinstance(llm_summary, dict) else None,
        "llm_model": llm_summary.get("llm_model") if isinstance(llm_summary, dict) else None,
        "web_search_used": bool(
            isinstance(search_summary, dict)
            and (search_summary.get("executed_query_count") or 0) > 0
        ),
        "baseline_comparison_included": False,
        "evaluation_metrics_included": False,
    }


def _build_export_manifest(
    package: FinalDataPackage,
    policy: FinalPackagePolicy,
) -> dict:
    return {
        "exportable_sections": list(policy.exportable_sections),
        "section_counts": {
            "final_dataset": len(package.final_dataset),
            "final_dataset_pre_quality_gate": len(
                package.final_dataset_pre_quality_gate
            ),
            "final_dataset_post_review": len(package.final_dataset_post_review),
            "quarantined_records": len(package.quarantined_records),
            "pending_review_records": len(package.pending_review_records),
            "non_primary_observations": len(package.non_primary_observations),
            "final_case_dataset": len(package.final_case_dataset),
            "global_outbreak_event_dataset": len(package.global_outbreak_event_dataset),
            "regional_surveillance_dataset": len(package.regional_surveillance_dataset),
            "country_year_aggregate_dataset": len(package.country_year_aggregate_dataset),
            "official_alert_dataset": len(package.official_alert_dataset),
            "zero_case_statements": len(package.zero_case_statements),
            "exposure_monitoring_records": len(
                package.exposure_monitoring_records
            ),
            "surveillance_summary_records": len(
                package.surveillance_summary_records
            ),
            "outbreak_summary_records": len(package.outbreak_summary_records),
            "context_records": len(package.context_records),
            "unclassified_observation_records": len(
                package.unclassified_observation_records
            ),
            "source_inventory": len(package.source_inventory),
            "authority_source_inventory": len(package.authority_source_inventory),
            "case_candidate_dataset": len(package.case_candidate_dataset),
            "reviewable_case_dataset": len(package.reviewable_case_dataset),
            "case_evidence_bundles": len(package.case_evidence_bundles),
            "workflow_case_bundle_line_list": len(
                package.workflow_case_bundle_line_list
            ),
            "workflow_case_candidate_line_list": len(
                package.workflow_case_candidate_line_list
            ),
            "observation_type_dataset_summary": (
                1 if package.observation_type_dataset_summary else 0
            ),
            "record_inclusion_decisions": len(package.record_inclusion_decisions),
            "run_quality_summary": 1 if package.run_quality_summary else 0,
            "final_dataset_quality_summary": (
                1 if package.final_dataset_quality_summary else 0
            ),
            "records_excluded_by_human_review": len(
                package.records_excluded_by_human_review
            ),
            "source_registry": len(package.source_registry),
            "source_identity_assessments": len(package.source_identity_assessments),
            "source_identity_summary": 1 if package.source_identity_summary else 0,
            "linked_events": len(package.linked_events),
            "event_clusters": len(package.event_clusters),
            "duplicate_clusters": len(package.duplicate_clusters),
            "validation_cases": len(package.validation_cases),
            "validation_comparisons": len(package.validation_comparisons),
            "validation_results": len(package.validation_results),
            "claims": len(package.claims),
            "claim_comparisons": len(package.claim_comparisons),
            "corroborated_events": len(package.corroborated_events),
            "corroboration_summary": 1 if package.corroboration_summary else 0,
            "anomaly_results": len(package.anomaly_results),
            "applied_human_review_decisions": len(
                package.applied_human_review_decisions
            ),
            "rejected_human_review_decisions": len(
                package.rejected_human_review_decisions
            ),
            "human_review_audit_trail": len(package.human_review_audit_trail),
            "conflicts": len(package.conflicts),
            "human_review_items": len(package.human_review_items),
            "excluded_sources": len(package.excluded_sources),
            "collection_trace": len(package.collection_trace),
            "data_dictionary": len(package.data_dictionary),
            "evidence_chunks": len(package.evidence_chunks),
        },
    }


def _build_finalization_summary(
    package: FinalDataPackage,
    workflow_summaries: dict,
    contains_fixture: bool,
) -> dict:
    package_dict = package.model_dump()
    disease_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    extraction_method_counts: dict[str, int] = {}
    for record in package_dict.get("final_dataset") or []:
        disease = str(record.get("disease") or "unknown")
        source_type = str(record.get("source_type") or "unknown")
        method = str(record.get("extraction_method") or "unknown")
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        extraction_method_counts[method] = extraction_method_counts.get(method, 0) + 1
    return {
        "final_dataset_count": len(package.final_dataset),
        "final_dataset_pre_quality_gate_count": len(
            package.final_dataset_pre_quality_gate
        ),
        "final_dataset_post_review_count": len(package.final_dataset_post_review),
        "quarantined_record_count": len(package.quarantined_records),
        "pending_review_record_count": len(package.pending_review_records),
        "non_primary_observation_count": len(package.non_primary_observations),
        "final_case_dataset_count": len(package.final_case_dataset),
        "global_outbreak_event_dataset_count": len(
            package.global_outbreak_event_dataset
        ),
        "regional_surveillance_dataset_count": len(
            package.regional_surveillance_dataset
        ),
        "country_year_aggregate_dataset_count": len(
            package.country_year_aggregate_dataset
        ),
        "official_alert_dataset_count": len(package.official_alert_dataset),
        "zero_case_statement_count": len(package.zero_case_statements),
        "exposure_monitoring_record_count": len(
            package.exposure_monitoring_records
        ),
        "surveillance_summary_record_count": len(
            package.surveillance_summary_records
        ),
        "outbreak_summary_record_count": len(package.outbreak_summary_records),
        "context_record_count": len(package.context_records),
        "unclassified_observation_count": len(
            package.unclassified_observation_records
        ),
        "source_inventory_count": len(package.source_inventory),
        "authority_source_inventory_count": len(package.authority_source_inventory),
        "case_candidate_dataset_count": len(package.case_candidate_dataset),
        "reviewable_case_dataset_count": len(package.reviewable_case_dataset),
        "case_evidence_bundle_count": len(package.case_evidence_bundles),
        "run_quality_status": package.run_quality_summary.get("run_quality_status"),
        "final_dataset_mode": package.run_quality_summary.get("final_dataset_mode"),
        "records_excluded_by_human_review_count": len(
            package.records_excluded_by_human_review
        ),
        "generic_record_count": sum(
            1
            for record in package_dict.get("final_dataset") or []
            if record.get("record_schema") == "generic_public_health_record"
        ),
        "legacy_hantavirus_record_count": sum(
            1
            for record in package_dict.get("final_dataset") or []
            if record.get("disease") == "Hantavirus disease"
        ),
        "disease_counts": disease_counts,
        "source_type_counts": source_type_counts,
        "extraction_method_counts": extraction_method_counts,
        "source_registry_count": len(package.source_registry),
        "source_identity_assessment_count": len(package.source_identity_assessments),
        "excluded_source_count": len(package.excluded_sources),
        "linked_event_count": len(package.linked_events),
        "event_cluster_count": len(package.event_clusters),
        "duplicate_cluster_count": len(package.duplicate_clusters),
        "validation_case_count": len(package.validation_cases),
        "validation_comparison_count": len(package.validation_comparisons),
        "validation_result_count": len(package.validation_results),
        "claim_count": len(package.claims),
        "claim_comparison_count": len(package.claim_comparisons),
        "corroborated_event_count": len(package.corroborated_events),
        "corroborated_primary_case_event_count": package.corroboration_summary.get(
            "corroborated_primary_case_event_count", 0
        ),
        "anomaly_result_count": len(package.anomaly_results),
        "applied_human_review_decision_count": len(
            package.applied_human_review_decisions
        ),
        "rejected_human_review_decision_count": len(
            package.rejected_human_review_decisions
        ),
        "human_review_audit_entry_count": len(package.human_review_audit_trail),
        "countable_record_count": sum(
            1
            for record in package_dict.get("final_dataset") or []
            if record.get("countable") is True
        ),
        "non_countable_duplicate_count": sum(
            1
            for record in package_dict.get("final_dataset") or []
            if record.get("event_member_status") == "non_countable_duplicate"
        ),
        "conflict_count": len(package.conflicts),
        "human_review_item_count": len(package.human_review_items),
        "collection_trace_count": len(package.collection_trace),
        "workflow_summary_count": len(workflow_summaries),
        "contains_synthetic_fixture_data": contains_fixture,
        "final_package_keys": sorted(package_dict.keys()),
}


def _default_human_review_application_summary(records: list[dict]) -> dict:
    return {
        "records_before_review": len(records),
        "records_after_review": len(records),
        "records_excluded_by_review": 0,
        "records_corrected_by_review": 0,
        "clusters_modified_by_review": 0,
        "validation_results_modified_by_review": 0,
        "sources_modified_by_review": 0,
        "anomalies_resolved_by_review": 0,
        "decisions_provided_count": 0,
        "decisions_applied_count": 0,
        "decisions_rejected_count": 0,
        "audit_entry_count": 0,
    }


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def _build_final_data_package(state: DataCollectionState) -> dict:
    """Assemble the hardened, auditable FinalDataPackage."""

    policy = FinalPackagePolicy(**load_final_package_policy())

    normalized_records = _claim_annotated_records_for_finalization(
        state,
        _safe_list(state, "normalized_records"),
    )
    state = dict(state)
    state["normalized_records"] = normalized_records
    post_review_records = _safe_list(state, "final_dataset_post_review")
    records_excluded_by_review = _safe_list(state, "records_excluded_by_human_review")
    quality_state = dict(state)
    quality_state.update(
        {
            "normalized_records": normalized_records,
            "final_dataset_post_review": post_review_records,
            "records_excluded_by_human_review": records_excluded_by_review,
        }
    )
    quality_result = apply_run_quality_gates(quality_state)
    pre_quality_records = _safe_list(
        quality_result, "final_dataset_pre_quality_gate"
    )
    accepted_records = _safe_list(quality_result, "final_dataset")
    post_review_records = _safe_list(quality_result, "final_dataset_post_review")
    quarantined_records = _safe_list(quality_result, "quarantined_records")
    pending_review_records = _safe_list(quality_result, "pending_review_records")
    non_primary_observations = _safe_list(
        quality_result, "non_primary_observations"
    )
    record_inclusion_decisions = _safe_list(
        quality_result, "record_inclusion_decisions"
    )
    run_quality_summary = quality_result.get("run_quality_summary") or {}
    final_dataset_quality_summary = (
        quality_result.get("final_dataset_quality_summary") or {}
    )
    direct_collection_summary = quality_result.get("direct_collection_summary") or {}
    collection_decision_summary = (
        quality_result.get("collection_decision_summary") or {}
    )
    task_acceptance_contract = state.get("task_acceptance_contract") or {}
    task_evidence_contract = state.get("task_evidence_contract") or build_task_evidence_contract(
        state
    )
    evidence_strategy_plan = state.get("evidence_strategy_plan") or {}
    source_triage_results = _safe_list(state, "source_triage_results")
    evidence_chunks = _safe_list(state, "evidence_chunks")
    chunk_relevance_assessments = _safe_list(
        state, "chunk_relevance_assessments"
    )
    record_task_fit_assessments = _safe_list(
        state, "record_task_fit_assessments"
    )
    source_critic_summary = state.get("source_critic_summary") or {}
    direct_fast_path_summary = (
        state.get("direct_fast_path_summary")
        or source_critic_summary.get("direct_fast_path_summary")
        or {}
    )
    metric_extraction_plan = state.get("metric_extraction_plan") or {}
    metric_row_extraction_audit = _safe_list(state, "metric_row_extraction_audit")
    content_fetch_summary = state.get("content_fetch_summary") or {}
    structured_extraction_summary = state.get("structured_extraction_summary") or {}
    source_registry = apply_source_product_profiles(
        _safe_list(state, "source_registry"),
        state,
    )
    documents = _safe_list(state, "documents")
    source_identity_assessments = _safe_list(state, "source_identity_assessments")
    official_coverage_candidates = _safe_list(state, "official_coverage_candidates")
    source_coverage_requirements = _safe_list(state, "source_coverage_requirements")

    lineage_state = dict(state)
    lineage_state.update(
        {
            "source_registry": source_registry,
            "documents": documents,
            "evidence_chunks": evidence_chunks,
            "source_coverage_requirements": source_coverage_requirements,
        }
    )
    normalized_records = _enrich_records_with_requirement_linkage(
        normalized_records,
        lineage_state,
    )
    pre_quality_records = _enrich_records_with_requirement_linkage(
        pre_quality_records,
        lineage_state,
    )
    accepted_records = _enrich_records_with_requirement_linkage(
        accepted_records,
        lineage_state,
    )
    post_review_records = _enrich_records_with_requirement_linkage(
        post_review_records,
        lineage_state,
    )
    quarantined_records = _enrich_records_with_requirement_linkage(
        quarantined_records,
        lineage_state,
    )
    pending_review_records = _enrich_records_with_requirement_linkage(
        pending_review_records,
        lineage_state,
    )
    non_primary_observations = _enrich_records_with_requirement_linkage(
        non_primary_observations,
        lineage_state,
    )

    source_coverage_audit = _refresh_source_coverage_audit(
        source_coverage_requirements,
        source_registry,
        documents,
        normalized_records,
        accepted_records,
        state.get("source_coverage_audit") or {},
    )
    source_coverage_audit = _refine_direct_coverage_status(
        source_coverage_audit,
        content_fetch_summary=content_fetch_summary,
        structured_extraction_summary=structured_extraction_summary,
    )
    if source_coverage_audit:
        refreshed_coverage_status = source_coverage_audit.get("coverage_status")
        should_override_summary_coverage = bool(
            source_coverage_audit.get("requirement_count")
            or refreshed_coverage_status
            in {
                "partial_target_coverage",
                "target_alias_error_page",
                "all_target_aliases_unusable",
                "no_task_collection_document",
                "target_alias_error_page_needs_fallback",
                "fallback_target_fetch_failed",
                "target_official_source_missing",
                "target_official_source_fetch_failed",
                "target_official_source_unusable",
            }
        )
        direct_collection_summary = dict(direct_collection_summary)
        run_quality_summary = dict(run_quality_summary)
        collection_decision_summary = dict(collection_decision_summary)
        if refreshed_coverage_status and should_override_summary_coverage:
            direct_collection_summary["coverage_status"] = refreshed_coverage_status
            run_quality_summary["coverage_status"] = refreshed_coverage_status
            collection_decision_summary["coverage_status"] = refreshed_coverage_status
        for key in (
            "coverage_completeness_status",
            "complete_requirement_count",
            "partial_requirement_count",
            "missing_requirement_ids",
        ):
            if key in source_coverage_audit:
                direct_collection_summary[key] = source_coverage_audit.get(key)
                collection_decision_summary[key] = source_coverage_audit.get(key)
    target_official_fetch_plan = _safe_list(state, "target_official_fetch_plan")
    must_fetch_sources = _safe_list(state, "must_fetch_sources")
    fetch_failures_blocking = _safe_list(state, "fetch_failures_blocking")
    linked_events = _safe_list(state, "linked_events")
    event_clusters = _safe_list(state, "event_clusters")
    duplicate_clusters = _safe_list(state, "duplicate_clusters")
    validation_cases = _safe_list(state, "validation_cases")
    validation_comparisons = _safe_list(state, "validation_comparisons")
    validation_results = _safe_list(state, "validation_results")
    claims = _safe_list(state, "claims")
    claim_comparisons = _safe_list(state, "claim_comparisons")
    corroborated_events = _safe_list(state, "corroborated_events")
    corroboration_summary = state.get("corroboration_summary") or {}
    anomaly_results = _safe_list(state, "anomaly_results")
    applied_decisions = _safe_list(state, "applied_human_review_decisions")
    rejected_decisions = _safe_list(state, "rejected_human_review_decisions")
    audit_trail = _safe_list(state, "human_review_audit_trail")
    conflicts = _safe_list(state, "conflicts")
    human_review_queue = _safe_list(state, "human_review_queue")
    core_metric_extraction_gaps = (
        structured_extraction_summary.get("core_metric_extraction_gaps")
        or metric_extraction_plan.get("core_metric_extraction_gaps")
        or []
    )
    human_review_queue = _append_missing_human_review_items(
        human_review_queue,
        _human_review_items_from_core_metric_gaps(
            core_metric_extraction_gaps,
            source_registry,
            documents,
        ),
    )

    split_state = dict(state)
    split_state.update(
        {
            "final_dataset_pre_quality_gate": pre_quality_records,
            "final_dataset": accepted_records,
            "final_dataset_post_review": post_review_records,
            "quarantined_records": quarantined_records,
            "pending_review_records": pending_review_records,
            "non_primary_observations": non_primary_observations,
            "record_inclusion_decisions": record_inclusion_decisions,
            "run_quality_summary": run_quality_summary,
            "final_dataset_quality_summary": final_dataset_quality_summary,
            "direct_collection_summary": direct_collection_summary,
            "collection_decision_summary": collection_decision_summary,
        }
    )
    observation_type_dataset_split = build_observation_type_dataset_split(split_state)
    run_quality_summary, final_dataset_quality_summary = (
        apply_observation_type_counts_to_summaries(
            run_quality_summary,
            final_dataset_quality_summary,
            observation_type_dataset_split,
        )
    )
    observation_type_dataset_summary = observation_type_dataset_split[
        "observation_type_dataset_summary"
    ]
    observation_dataset_views = {
        key: _safe_list(observation_type_dataset_split, key)
        for key in DATASET_VIEW_KEYS
    }
    source_inventory, authority_source_inventory = _build_source_inventory(
        source_registry,
        documents,
        evidence_chunks,
        normalized_records,
        structured_extraction_summary=structured_extraction_summary,
        source_recall_target_ledger=(
            _safe_dict(state.get("source_search_execution_summary")).get(
                "source_recall_target_ledger"
            )
            or []
        ),
    )
    source_search_summary_for_candidates = _safe_dict(
        state.get("source_search_execution_summary")
    )
    source_gap_status_for_candidates = (
        source_search_summary_for_candidates.get("high_confidence_source_gap_status")
        or source_search_summary_for_candidates.get("authority_source_recall_status")
    )
    high_confidence_domain_status_for_candidates = (
        json.dumps(
            source_search_summary_for_candidates.get(
                "high_confidence_domain_query_status"
            )
            or {},
            ensure_ascii=False,
            sort_keys=True,
        )
        if source_search_summary_for_candidates.get(
            "high_confidence_domain_query_status"
        )
        else None
    )
    case_candidate_dataset, reviewable_case_dataset = _build_case_candidate_datasets(
        final_case_records=observation_dataset_views["final_case_dataset"],
        accepted_records=accepted_records,
        pending_review_records=pending_review_records,
        quarantined_records=quarantined_records,
        non_primary_observation_records=non_primary_observations,
        source_registry=source_inventory,
        source_gap_status=source_gap_status_for_candidates,
        high_confidence_domain_status_json=(
            high_confidence_domain_status_for_candidates
        ),
    )
    case_evidence_bundles = _build_case_evidence_bundles(case_candidate_dataset)
    _annotate_case_candidates_with_link_decisions(
        case_candidate_dataset,
        case_evidence_bundles,
    )
    evidence_product_dataset = _build_evidence_product_dataset(
        case_candidate_dataset,
        source_inventory,
    )
    workflow_case_bundle_line_list = build_workflow_case_bundle_line_list(
        case_evidence_bundles=case_evidence_bundles,
        case_candidate_dataset=case_candidate_dataset,
        source_inventory=source_inventory,
    )
    workflow_case_candidate_line_list = build_workflow_case_candidate_line_list(
        case_candidate_dataset=case_candidate_dataset,
        case_evidence_bundles=case_evidence_bundles,
        source_inventory=source_inventory,
    )
    best_available_context_records = _build_best_available_context_records(
        quarantined_records,
        state,
    )

    final_dataset_pre_quality_gate = [
        PublicHealthRecord(**r) for r in pre_quality_records
    ]
    final_dataset = [PublicHealthRecord(**r) for r in accepted_records]
    final_dataset_post_review = [
        PublicHealthRecord(**r) for r in post_review_records
    ]
    quarantined_record_models = [
        PublicHealthRecord(**r) for r in quarantined_records
    ]
    pending_review_record_models = [
        PublicHealthRecord(**r) for r in pending_review_records
    ]
    non_primary_observation_models = [
        PublicHealthRecord(**r) for r in non_primary_observations
    ]
    final_case_dataset_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["final_case_dataset"]
    ]
    global_outbreak_event_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["global_outbreak_event_dataset"]
    ]
    regional_surveillance_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["regional_surveillance_dataset"]
    ]
    country_year_aggregate_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["country_year_aggregate_dataset"]
    ]
    official_alert_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["official_alert_dataset"]
    ]
    probable_case_dataset_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["probable_case_dataset"]
    ]
    suspected_case_dataset_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["suspected_case_dataset"]
    ]
    unspecified_case_dataset_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["unspecified_case_dataset"]
    ]
    death_dataset_models = [
        PublicHealthRecord(**r) for r in observation_dataset_views["death_dataset"]
    ]
    hospitalization_dataset_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["hospitalization_dataset"]
    ]
    zero_case_statement_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["zero_case_statements"]
    ]
    exposure_monitoring_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["exposure_monitoring_records"]
    ]
    surveillance_summary_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["surveillance_summary_records"]
    ]
    outbreak_summary_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["outbreak_summary_records"]
    ]
    context_record_models = [
        PublicHealthRecord(**r) for r in observation_dataset_views["context_records"]
    ]
    best_available_context_record_models = [
        PublicHealthRecord(**r) for r in best_available_context_records
    ]
    unclassified_observation_models = [
        PublicHealthRecord(**r)
        for r in observation_dataset_views["unclassified_observation_records"]
    ]
    records_excluded_models = [
        PublicHealthRecord(**r) for r in records_excluded_by_review
    ]
    registry_models = [SourceRegistryEntry(**e) for e in source_registry]
    source_identity_models = [
        SourceIdentityAssessment(**item) for item in source_identity_assessments
    ]
    excluded_sources = [
        e
        for e in registry_models
        if e.screening_decision == "exclude"
        or e.critic_decision == "exclude"
        or e.final_screening_decision == "exclude"
        or e.status == "excluded"
    ]
    included_registry = [e for e in registry_models if e not in excluded_sources]

    direct_collection_summary = dict(direct_collection_summary or {})
    direct_collection_summary["final_dataset_count"] = len(final_dataset)
    direct_collection_summary["quarantined_record_count"] = len(
        quarantined_record_models
    )
    direct_collection_summary["pending_review_record_count"] = len(
        pending_review_record_models
    )
    direct_collection_summary["human_review_record_count"] = len(
        pending_review_record_models
    )
    direct_collection_summary["final_case_dataset_count"] = len(
        final_case_dataset_models
    )
    direct_collection_summary["dataset_view_counts"] = (
        observation_type_dataset_summary.get("dataset_view_counts") or {}
    )
    direct_collection_summary["best_available_context_record_count"] = len(
        best_available_context_record_models
    )
    direct_collection_summary["best_available_context_record_ids"] = [
        record.record_id for record in best_available_context_record_models
    ]
    direct_collection_summary["core_metric_extraction_gap_count"] = int(
        structured_extraction_summary.get("core_metric_extraction_gap_count")
        or metric_extraction_plan.get("core_metric_extraction_gap_count")
        or len(core_metric_extraction_gaps)
        or 0
    )
    direct_collection_summary["core_metric_extraction_gap_source_ids"] = sorted(
        {
            str(gap.get("source_id") or "")
            for gap in core_metric_extraction_gaps
            if gap.get("source_id")
        }
    )
    run_quality_summary = dict(run_quality_summary or {})
    final_dataset_quality_summary = dict(final_dataset_quality_summary or {})
    run_quality_summary["direct_collection_summary"] = direct_collection_summary
    final_dataset_quality_summary["direct_collection_summary"] = (
        direct_collection_summary
    )

    linked_event_models = [LinkedEvent(**le) for le in linked_events]
    event_cluster_models = [EventCluster(**cluster) for cluster in event_clusters]
    duplicate_cluster_models = [
        EventCluster(**cluster) for cluster in duplicate_clusters
    ]
    validation_case_models = [ValidationCase(**item) for item in validation_cases]
    validation_comparison_models = [
        ValidationComparison(**item) for item in validation_comparisons
    ]
    validation_result_models = [
        ValidationResult(**item) for item in validation_results
    ]
    claim_models = [PublicHealthClaim(**item) for item in claims]
    claim_comparison_models = [
        ClaimComparison(**item) for item in claim_comparisons
    ]
    corroborated_event_models = [
        CorroboratedEvent(**item) for item in corroborated_events
    ]
    anomaly_result_models = [AnomalyResult(**item) for item in anomaly_results]
    applied_decision_models = [
        AppliedHumanReviewDecision(**item) for item in applied_decisions
    ]
    rejected_decision_models = [
        RejectedHumanReviewDecision(**item) for item in rejected_decisions
    ]
    audit_models = [HumanReviewAuditEntry(**item) for item in audit_trail]
    conflict_models = [Conflict(**c) for c in conflicts]
    human_review_items = [HumanReviewItem(**item) for item in human_review_queue]

    contains_fixture, fixture_notice = _detect_synthetic_fixture_data(state, policy)
    summary_state = dict(state)
    summary_state.update(
        {
            "run_quality_summary": run_quality_summary,
            "final_dataset_quality_summary": final_dataset_quality_summary,
            "direct_collection_summary": direct_collection_summary,
            "collection_decision_summary": collection_decision_summary,
        }
    )
    workflow_summaries = _collect_workflow_summaries(summary_state, policy)
    data_dictionary = _build_data_dictionary(state, policy)
    provenance_manifest = _build_provenance_manifest(state)
    llm_used = _detect_llm_used(state)
    human_review_application_summary = state.get(
        "human_review_application_summary"
    ) or _default_human_review_application_summary(normalized_records)

    # Append trace before building the package so package.collection_trace
    # includes this very event and stays length-aligned with state trace.
    trace = append_trace(
        state,
        node_name="final_data_package_builder",
        message="Assembled hardened FinalDataPackage from current state.",
        metadata={
            "final_dataset_size": len(final_dataset),
            "final_dataset_pre_quality_gate_size": len(
                final_dataset_pre_quality_gate
            ),
            "quarantined_record_count": len(quarantined_record_models),
            "pending_review_record_count": len(pending_review_record_models),
            "non_primary_observation_count": len(non_primary_observation_models),
            "final_case_dataset_count": len(final_case_dataset_models),
            "global_outbreak_event_dataset_count": len(global_outbreak_event_models),
            "regional_surveillance_dataset_count": len(regional_surveillance_models),
            "country_year_aggregate_dataset_count": len(country_year_aggregate_models),
            "official_alert_dataset_count": len(official_alert_models),
            "zero_case_statement_count": len(zero_case_statement_models),
            "exposure_monitoring_record_count": len(exposure_monitoring_models),
            "context_record_count": len(context_record_models),
            "best_available_context_record_count": len(
                best_available_context_record_models
            ),
            "observation_type_dataset_summary": observation_type_dataset_summary,
            "run_quality_status": run_quality_summary.get("run_quality_status"),
            "source_registry_size": len(included_registry),
            "source_identity_assessment_count": len(source_identity_models),
            "excluded_source_count": len(excluded_sources),
            "linked_event_count": len(linked_event_models),
            "event_cluster_count": len(event_cluster_models),
            "duplicate_cluster_count": len(duplicate_cluster_models),
            "validation_result_count": len(validation_result_models),
            "claim_count": len(claim_models),
            "claim_comparison_count": len(claim_comparison_models),
            "corroborated_event_count": len(corroborated_event_models),
            "anomaly_result_count": len(anomaly_result_models),
            "applied_human_review_decision_count": len(applied_decision_models),
            "rejected_human_review_decision_count": len(rejected_decision_models),
            "human_review_audit_entry_count": len(audit_models),
            "conflict_count": len(conflict_models),
            "human_review_item_count": len(human_review_items),
            "contains_synthetic_fixture_data": contains_fixture,
            "llm_used": llm_used,
            "package_version": policy.package_version,
        },
    )
    package_metadata = _build_package_metadata(
        state, policy, contains_fixture, fixture_notice, trace, llm_used
    )

    package = FinalDataPackage(
        final_dataset=final_dataset,
        final_dataset_pre_quality_gate=final_dataset_pre_quality_gate,
        final_dataset_post_review=final_dataset_post_review,
        quarantined_records=quarantined_record_models,
        pending_review_records=pending_review_record_models,
        non_primary_observations=non_primary_observation_models,
        final_case_dataset=final_case_dataset_models,
        global_outbreak_event_dataset=global_outbreak_event_models,
        regional_surveillance_dataset=regional_surveillance_models,
        country_year_aggregate_dataset=country_year_aggregate_models,
        official_alert_dataset=official_alert_models,
        probable_case_dataset=probable_case_dataset_models,
        suspected_case_dataset=suspected_case_dataset_models,
        unspecified_case_dataset=unspecified_case_dataset_models,
        death_dataset=death_dataset_models,
        hospitalization_dataset=hospitalization_dataset_models,
        zero_case_statements=zero_case_statement_models,
        exposure_monitoring_records=exposure_monitoring_models,
        surveillance_summary_records=surveillance_summary_models,
        outbreak_summary_records=outbreak_summary_models,
        context_records=context_record_models,
        best_available_context_records=best_available_context_record_models,
        unclassified_observation_records=unclassified_observation_models,
        source_inventory=source_inventory,
        authority_source_inventory=authority_source_inventory,
        case_candidate_dataset=case_candidate_dataset,
        reviewable_case_dataset=reviewable_case_dataset,
        case_evidence_bundles=case_evidence_bundles,
        evidence_product_dataset=evidence_product_dataset,
        workflow_case_bundle_line_list=workflow_case_bundle_line_list,
        workflow_case_candidate_line_list=workflow_case_candidate_line_list,
        observation_type_dataset_summary=observation_type_dataset_summary,
        record_inclusion_decisions=record_inclusion_decisions,
        run_quality_summary=run_quality_summary,
        final_dataset_quality_summary=final_dataset_quality_summary,
        task_acceptance_contract=task_acceptance_contract,
        task_evidence_contract=task_evidence_contract,
        evidence_strategy_plan=evidence_strategy_plan,
        source_triage_results=source_triage_results,
        evidence_chunks=evidence_chunks,
        chunk_relevance_assessments=chunk_relevance_assessments,
        record_task_fit_assessments=record_task_fit_assessments,
        direct_fast_path_summary=direct_fast_path_summary,
        metric_extraction_plan=metric_extraction_plan,
        metric_row_extraction_audit=metric_row_extraction_audit,
        collection_decision_summary=collection_decision_summary,
        records_excluded_by_human_review=records_excluded_models,
        source_registry=included_registry,
        source_identity_assessments=source_identity_models,
        source_identity_summary=state.get("source_identity_summary") or {},
        official_coverage_candidates=official_coverage_candidates,
        source_coverage_requirements=source_coverage_requirements,
        source_coverage_audit=source_coverage_audit,
        target_official_fetch_plan=target_official_fetch_plan,
        must_fetch_sources=must_fetch_sources,
        fetch_failures_blocking=fetch_failures_blocking,
        linked_events=linked_event_models,
        event_clusters=event_cluster_models,
        duplicate_clusters=duplicate_cluster_models,
        validation_cases=validation_case_models,
        validation_comparisons=validation_comparison_models,
        validation_results=validation_result_models,
        validation_summary=state.get("validation_summary") or {},
        trusted_source_validation_summary=state.get(
            "trusted_source_validation_summary"
        )
        or {},
        cross_source_validation_summary=state.get(
            "cross_source_validation_summary"
        )
        or {},
        claims=claim_models,
        claim_comparisons=claim_comparison_models,
        corroborated_events=corroborated_event_models,
        corroboration_summary=corroboration_summary,
        anomaly_results=anomaly_result_models,
        anomaly_summary=state.get("anomaly_summary") or {},
        applied_human_review_decisions=applied_decision_models,
        rejected_human_review_decisions=rejected_decision_models,
        human_review_audit_trail=audit_models,
        human_review_application_summary=state.get(
            "human_review_application_summary"
        )
        or human_review_application_summary,
        conflicts=conflict_models,
        human_review_items=human_review_items,
        excluded_sources=excluded_sources,
        collection_trace=trace,
        package_metadata=package_metadata,
        workflow_summaries=workflow_summaries,
        data_dictionary=data_dictionary,
        provenance_manifest=provenance_manifest,
        export_manifest={},
        export_warnings=(
            ["contains_synthetic_fixture_data"] if contains_fixture else []
        ),
        contains_synthetic_fixture_data=contains_fixture,
        synthetic_fixture_notice=fixture_notice,
    )
    # Export manifest needs the final package; fill it now.
    package.export_manifest = _build_export_manifest(package, policy)

    finalization_summary = _build_finalization_summary(
        package, workflow_summaries, contains_fixture
    )

    return {
        "final_data_package": package.model_dump(),
        "finalization_summary": finalization_summary,
        "final_dataset_pre_quality_gate": [
            record.model_dump() for record in final_dataset_pre_quality_gate
        ],
        "quarantined_records": [
            record.model_dump() for record in quarantined_record_models
        ],
        "pending_review_records": [
            record.model_dump() for record in pending_review_record_models
        ],
        "non_primary_observations": [
            record.model_dump() for record in non_primary_observation_models
        ],
        "final_case_dataset": [
            record.model_dump() for record in final_case_dataset_models
        ],
        "global_outbreak_event_dataset": [
            record.model_dump() for record in global_outbreak_event_models
        ],
        "regional_surveillance_dataset": [
            record.model_dump() for record in regional_surveillance_models
        ],
        "country_year_aggregate_dataset": [
            record.model_dump() for record in country_year_aggregate_models
        ],
        "official_alert_dataset": [
            record.model_dump() for record in official_alert_models
        ],
        "probable_case_dataset": [
            record.model_dump() for record in probable_case_dataset_models
        ],
        "suspected_case_dataset": [
            record.model_dump() for record in suspected_case_dataset_models
        ],
        "unspecified_case_dataset": [
            record.model_dump() for record in unspecified_case_dataset_models
        ],
        "death_dataset": [record.model_dump() for record in death_dataset_models],
        "hospitalization_dataset": [
            record.model_dump() for record in hospitalization_dataset_models
        ],
        "zero_case_statements": [
            record.model_dump() for record in zero_case_statement_models
        ],
        "exposure_monitoring_records": [
            record.model_dump() for record in exposure_monitoring_models
        ],
        "surveillance_summary_records": [
            record.model_dump() for record in surveillance_summary_models
        ],
        "outbreak_summary_records": [
            record.model_dump() for record in outbreak_summary_models
        ],
        "context_records": [record.model_dump() for record in context_record_models],
        "best_available_context_records": [
            record.model_dump() for record in best_available_context_record_models
        ],
        "unclassified_observation_records": [
            record.model_dump() for record in unclassified_observation_models
        ],
        "source_inventory": list(source_inventory),
        "authority_source_inventory": list(authority_source_inventory),
        "case_candidate_dataset": list(case_candidate_dataset),
        "reviewable_case_dataset": list(reviewable_case_dataset),
        "case_evidence_bundles": list(case_evidence_bundles),
        "evidence_product_dataset": list(evidence_product_dataset),
        "workflow_case_bundle_line_list": list(workflow_case_bundle_line_list),
        "workflow_case_candidate_line_list": list(workflow_case_candidate_line_list),
        "observation_type_dataset_summary": observation_type_dataset_summary,
        "record_inclusion_decisions": list(record_inclusion_decisions),
        "run_quality_summary": run_quality_summary,
        "final_dataset_quality_summary": final_dataset_quality_summary,
        "direct_collection_summary": direct_collection_summary,
        "collection_decision_summary": collection_decision_summary,
        "task_acceptance_contract": task_acceptance_contract,
        "task_evidence_contract": task_evidence_contract,
        "evidence_strategy_plan": evidence_strategy_plan,
        "source_triage_results": source_triage_results,
        "evidence_chunks": evidence_chunks,
        "chunk_relevance_assessments": chunk_relevance_assessments,
        "record_task_fit_assessments": record_task_fit_assessments,
        "direct_fast_path_summary": direct_fast_path_summary,
        "metric_extraction_plan": metric_extraction_plan,
        "metric_row_extraction_audit": metric_row_extraction_audit,
        "source_coverage_requirements": source_coverage_requirements,
        "source_coverage_audit": source_coverage_audit,
        "final_dataset_post_review": [
            record.model_dump() for record in final_dataset_post_review
        ],
        "records_excluded_by_human_review": [
            record.model_dump() for record in records_excluded_models
        ],
        "applied_human_review_decisions": [
            decision.model_dump() for decision in applied_decision_models
        ],
        "rejected_human_review_decisions": [
            decision.model_dump() for decision in rejected_decision_models
        ],
        "human_review_audit_trail": [entry.model_dump() for entry in audit_models],
        "human_review_application_summary": human_review_application_summary,
        "collection_trace": trace,
    }


def final_data_package_builder(state: DataCollectionState) -> dict:
    """Preserve the established package contract, enforcing evidence products last."""
    from ..evidence_qualification import evidence_qualification_enabled, qualified_coverage
    if not evidence_qualification_enabled():
        return _build_final_data_package(state)
    result = apply_run_quality_gates(state)
    qualified_state = dict(state)
    qualified_state.update(result)
    qualified_state["pipeline_mode"] = "evidence"
    qualified_state["normalized_records"] = result["qualified_records"]
    package_result = _build_final_data_package(qualified_state)
    coverage = qualified_coverage(state.get("source_coverage_requirements") or [], result["qualified_records"])
    from ..report_timeline import build_report_timeline_inventory
    coverage["reporting_timeline"] = build_report_timeline_inventory(qualified_state)
    additions = {**result, "pipeline_mode":"evidence",
                 "source_coverage_audit":coverage,
                 "final_case_dataset":result["qualified_case_records"],
                 "aggregate_dataset":result["qualified_aggregate_records"],
                 "context_records":result["qualified_context_records"],
                 "case_candidate_dataset":[], "reviewable_case_dataset":[],
                 "case_evidence_bundles":[], "workflow_case_bundle_line_list":[],
                 "workflow_case_candidate_line_list":[],
                 "evidence_product_dataset":result["qualified_records"] + result["candidate_records"],
                 "release_status":"not_evaluated"}
    package_result.update(additions)
    package_result["final_data_package"].update(additions)
    for target in (package_result["finalization_summary"], package_result["final_data_package"].get("package_metadata", {})):
        target.update(pipeline_mode="evidence", final_record_count=len(result["final_dataset"]), candidate_record_count=len(result["candidate_records"]), final_case_dataset_count=len(result["qualified_case_records"]), release_status="not_evaluated")
    manifest = package_result["final_data_package"].get("export_manifest") or {}
    if isinstance(manifest.get("record_counts"), dict):
        for key,value in additions.items():
            if isinstance(value,list):
                manifest["record_counts"][key] = len(value)
    from ..evidence_products import build_evidence_products
    from ..evidence_qualification import build_evidence_index
    from ..result_manifest import build_result_manifest, normalize_result_views, synchronize_package_summary
    from ..session_runtime import get_runtime
    runtime = get_runtime()
    manifest_state = {**state, **package_result}
    if runtime:
        manifest_state['run_budget_ledger'] = runtime.ledger.snapshot()
        if runtime.frontier is not None:
            manifest_state['acquisition_frontier'] = runtime.frontier.snapshot()
            package_result['acquisition_frontier'] = manifest_state['acquisition_frontier']
    package = package_result['final_data_package']
    # The evidence registry is the complete discovery audit, including excluded sources.
    package['source_registry'] = [dict(row) for row in state.get('source_registry') or []]
    package['evidence_products'] = build_evidence_products(
        result['qualified_records'], evidence_index=build_evidence_index(qualified_state),
        sources=state.get('source_registry') or [])
    normalize_result_views(package)
    package['result_manifest'] = build_result_manifest(package, manifest_state)
    summary = synchronize_package_summary(package)
    package_result['finalization_summary'] = dict(summary)
    for key in ('run_quality_summary', 'final_dataset_quality_summary', 'collection_decision_summary', 'direct_collection_summary', 'observation_type_dataset_summary', 'primary_case_dataset', 'task_aware_observation_dataset', 'reviewable_dataset', 'non_primary_observations'):
        package_result[key] = package[key]
    package_result['source_registry'] = package.get('source_registry') or []
    package_result['result_manifest'] = package['result_manifest']
    package_result['evidence_products'] = package['evidence_products']
    package_result['run_budget_ledger'] = package['result_manifest']['budget']
    return package_result
