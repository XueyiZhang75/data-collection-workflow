"""Artifact-only human review workflow productization helpers.

This module reads completed workflow session artifacts and writes prioritized,
actionable review files. It does not call LLMs, search providers, or fetchers,
and it never applies human review decisions.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .export import write_csv_rows, write_json


PRIORITY_LEVELS = ("P0_critical", "P1_high", "P2_medium", "P3_low")
TOP_N_RECOMMENDED = 10

SUPPORTED_DECISION_TYPES_BY_TARGET = {
    "record": [
        "accept_as_is",
        "correct_fields",
        "reject_record",
        "mark_requires_review",
        "mark_review_resolved",
        "mark_non_countable",
        "mark_countable",
        "mark_duplicate",
        "mark_not_duplicate",
        "needs_more_evidence",
        "defer_decision",
    ],
    "validation_result": [
        "accept_validation_result",
        "mark_validation_result_not_applicable",
        "confirm_conflict",
        "resolve_conflict_as_left",
        "resolve_conflict_as_right",
        "mark_needs_more_evidence",
        "override_validation_status",
    ],
    "anomaly": [
        "accept_anomaly",
        "dismiss_anomaly",
        "confirm_anomaly",
        "mark_anomaly_resolved",
        "mark_anomaly_needs_more_evidence",
    ],
    "source": [
        "approve_source_role",
        "override_source_role",
        "exclude_source",
        "mark_source_needs_review",
    ],
    "event_cluster": [
        "merge_records_or_clusters",
        "split_cluster",
        "defer_decision",
    ],
    "claim": ["needs_more_evidence", "defer_decision"],
    "run": ["needs_more_evidence", "defer_decision"],
    "general": ["needs_more_evidence", "defer_decision"],
}

ALL_SUPPORTED_DECISION_TYPES = sorted(
    {
        decision
        for decisions in SUPPORTED_DECISION_TYPES_BY_TARGET.values()
        for decision in decisions
    }
)

CSV_COLUMNS = [
    "priority_rank",
    "priority_level",
    "issue_category",
    "issue_type",
    "short_title",
    "target_type",
    "target_ids",
    "disease",
    "location",
    "date_or_period",
    "why_it_matters",
    "suggested_reviewer_question",
    "suggested_action",
    "allowed_decision_types",
    "source_title",
    "source_url",
    "actual_publisher",
    "evidence_quote",
    "recommended_artifacts_to_open",
]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _clean_text(value: Any, limit: int | None = None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if limit and len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _unique(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            nested = _unique(value)
            for item in nested:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _by_id(rows: list[dict], keys: tuple[str, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if value:
                out[str(value)] = row
                break
    return out


def _read_artifact(
    session_dir: Path,
    package: dict,
    key: str,
    default: Any,
) -> Any:
    collection_path = session_dir / "collection" / f"{key}.json"
    diagnostics_path = session_dir / "diagnostics" / f"{key}.json"
    value = _read_json(collection_path, None)
    if value is not None:
        return value
    value = _read_json(diagnostics_path, None)
    if value is not None:
        return value
    return package.get(key, default)


def load_human_review_artifacts(session_dir: Path) -> dict:
    """Load existing human-review-related artifacts from a completed session."""

    session_dir = Path(session_dir)
    package = _as_dict(_read_json(session_dir / "collection" / "final_package.json", {}))
    run_summary = _as_dict(_read_json(session_dir / "workflow_run_summary.json", {}))
    interpretive_summary = _as_dict(
        _read_json(session_dir / "workflow_interpretive_report_summary.json", {})
    )
    workflow_summaries = _as_dict(
        _read_json(session_dir / "diagnostics" / "workflow_summaries.json", {})
    )

    keys_with_defaults = {
        "human_review_items": [],
        "anomaly_results": [],
        "validation_results": [],
        "validation_comparisons": [],
        "source_identity_assessments": [],
        "source_registry": [],
        "claims": [],
        "claim_comparisons": [],
        "corroborated_events": [],
        "conflicts": [],
        "final_dataset": [],
        "final_case_dataset": [],
        "final_dataset_pre_quality_gate": [],
        "context_records": [],
        "quarantined_records": [],
        "pending_review_records": [],
        "non_primary_observations": [],
        "record_inclusion_decisions": [],
        "run_quality_summary": {},
        "final_dataset_quality_summary": {},
        "observation_type_dataset_summary": {},
        "validation_source_compatibility_summary": {},
        "source_critic_summary": {},
        "human_review_application_summary": {},
        "applied_human_review_decisions": [],
        "rejected_human_review_decisions": [],
        "human_review_audit_trail": [],
    }
    artifacts: dict[str, Any] = {
        "session_dir": session_dir,
        "session_id": session_dir.name,
        "package": package,
        "workflow_run_summary": run_summary,
        "interpretive_report_summary": interpretive_summary,
        "workflow_summaries": workflow_summaries,
    }
    for key, default in keys_with_defaults.items():
        artifacts[key] = _read_artifact(session_dir, package, key, default)

    artifacts["task"] = _task_metadata(artifacts)
    return artifacts


def _task_metadata(artifacts: dict) -> dict:
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    interpretive = _as_dict(artifacts.get("interpretive_report_summary"))
    run_summary = _as_dict(artifacts.get("workflow_run_summary"))
    package = _as_dict(artifacts.get("package"))
    metadata = _as_dict(package.get("package_metadata"))
    return {
        "session_id": artifacts.get("session_id") or run_summary.get("session_id"),
        "disease": (
            run_quality.get("task_disease")
            or interpretive.get("task_disease")
            or metadata.get("disease")
            or ""
        ),
        "location": (
            run_quality.get("task_location")
            or interpretive.get("task_location")
            or metadata.get("geography")
            or metadata.get("location")
            or ""
        ),
        "start_date": (
            run_quality.get("task_start_date")
            or interpretive.get("task_start_date")
            or metadata.get("start_date")
            or ""
        ),
        "end_date": (
            run_quality.get("task_end_date")
            or interpretive.get("task_end_date")
            or metadata.get("end_date")
            or ""
        ),
    }


def _artifact_maps(artifacts: dict) -> dict[str, dict[str, dict]]:
    records = []
    for key in (
        "final_dataset",
        "final_case_dataset",
        "final_dataset_pre_quality_gate",
        "context_records",
        "quarantined_records",
        "pending_review_records",
        "non_primary_observations",
    ):
        records.extend(_as_list(artifacts.get(key)))
    return {
        "records": _by_id(records, ("record_id",)),
        "sources": _by_id(_as_list(artifacts.get("source_registry")), ("source_id",)),
        "source_identity": _by_id(
            _as_list(artifacts.get("source_identity_assessments")),
            ("source_id",),
        ),
        "claims": _by_id(_as_list(artifacts.get("claims")), ("claim_id",)),
        "events": _by_id(_as_list(artifacts.get("corroborated_events")), ("event_id",)),
        "anomalies": _by_id(_as_list(artifacts.get("anomaly_results")), ("anomaly_id",)),
        "validation_results": _by_id(
            _as_list(artifacts.get("validation_results")),
            ("validation_result_id", "result_id"),
        ),
    }


def _validation_limited_item(artifacts: dict) -> dict | None:
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    compat = _as_dict(artifacts.get("validation_source_compatibility_summary"))
    status = str(compat.get("compatibility_status") or "")
    limited = bool(run_quality.get("validation_limited")) or status.startswith(
        "incompatible"
    )
    if not limited:
        return None
    return {
        "review_id": "review_validation_limited_current_task",
        "item_type": "validation_limited",
        "related_ids": ["validation_source_compatibility_summary"],
        "reason": compat.get("compatibility_reason")
        or "No compatible held-out validation source was available.",
        "status": "pending",
        "suggested_action": "inspect_validation_source_coverage",
        "review_packet": {
            "packet_sections": {
                "validation_source_compatibility_summary": compat,
                "run_quality_summary": run_quality,
            }
        },
    }


def _source_from_packet(item: dict) -> dict:
    packet = _as_dict(item.get("review_packet"))
    sections = _as_dict(packet.get("packet_sections"))
    source = _as_dict(sections.get("source_registry_entry"))
    return source


def _records_from_packet(item: dict) -> list[dict]:
    packet = _as_dict(item.get("review_packet"))
    sections = _as_dict(packet.get("packet_sections"))
    rows = sections.get("related_records")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    record = sections.get("record")
    return [record] if isinstance(record, dict) else []


def _ids_for_item(item: dict, maps: dict[str, dict[str, dict]]) -> dict[str, list[str]]:
    related = _as_list(item.get("related_ids"))
    packet_records = _records_from_packet(item)
    source_ids = _unique(
        [
            item.get("source_ids"),
            item.get("source_id"),
            _source_from_packet(item).get("source_id"),
            *[rid for rid in related if str(rid).startswith("src_")],
            *[record.get("source_id") for record in packet_records],
        ]
    )
    record_ids = _unique(
        [
            item.get("record_id"),
            *[rid for rid in related if str(rid).startswith("rec_")],
            *[record.get("record_id") for record in packet_records],
        ]
    )
    claim_ids = _unique([rid for rid in related if str(rid).startswith("claim")])
    event_ids = _unique(
        [
            item.get("event_cluster_id"),
            *[
                rid
                for rid in related
                if str(rid).startswith("event") or str(rid).startswith("cluster")
            ],
        ]
    )
    anomaly_ids = _unique(
        [item.get("anomaly_id"), *[rid for rid in related if str(rid).startswith("anom")]]
    )
    validation_ids = _unique(
        [
            item.get("validation_result_id"),
            *[rid for rid in related if str(rid).startswith("val")],
        ]
    )

    for event_id in list(event_ids):
        event = maps["events"].get(event_id) or {}
        record_ids = _unique([record_ids, event.get("record_ids")])
        claim_ids = _unique([claim_ids, event.get("claim_ids")])
        source_ids = _unique([source_ids, event.get("source_ids")])
    for claim_id in list(claim_ids):
        claim = maps["claims"].get(claim_id) or {}
        record_ids = _unique([record_ids, claim.get("record_id")])
        source_ids = _unique([source_ids, claim.get("source_id")])
    for anomaly_id in list(anomaly_ids):
        anomaly = maps["anomalies"].get(anomaly_id) or {}
        record_ids = _unique([record_ids, anomaly.get("record_id")])
        source_ids = _unique([source_ids, anomaly.get("source_ids")])

    target_ids = _unique(
        [record_ids, claim_ids, source_ids, event_ids, anomaly_ids, validation_ids, related]
    )
    return {
        "target_ids": target_ids,
        "record_ids": record_ids,
        "claim_ids": claim_ids,
        "source_ids": source_ids,
        "event_ids": event_ids,
        "anomaly_ids": anomaly_ids,
        "validation_ids": validation_ids,
    }


def _target_type(ids: dict, item_type: str) -> str:
    item_lower = item_type.lower()
    if "validation_limited" in item_lower:
        return "run"
    if ids["anomaly_ids"] or "anomaly" in item_lower:
        return "anomaly"
    if ids["validation_ids"] or "validation" in item_lower:
        return "validation_result"
    if ids["record_ids"]:
        return "record"
    if ids["event_ids"] or "cluster" in item_lower:
        return "event_cluster"
    if ids["claim_ids"] or "claim" in item_lower:
        return "claim"
    if ids["source_ids"] or "source" in item_lower:
        return "source"
    if "validation_limited" in item_lower:
        return "run"
    return "general"


def _category_for_item(
    item: dict,
    ids: dict,
    artifacts: dict,
    maps: dict[str, dict[str, dict]],
) -> str:
    item_type = str(item.get("item_type") or "").lower()
    reason = str(item.get("reason") or item.get("human_review_reason") or "").lower()
    joined = f"{item_type} {reason}"
    run_quality = _as_dict(artifacts.get("run_quality_summary"))

    if "validation_limited" in joined or run_quality.get("validation_limited") and item.get("review_id") == "review_validation_limited_current_task":
        return "validation_limited_review"
    if "source_critic" in joined or "blocked_source" in joined or "block_fetch" in joined:
        return "source_critic_block_review"
    if "source_identity" in joined or "publisher" in joined or "actual_publisher_unknown" in joined:
        return "source_identity_review"
    if "source_credibility" in joined or item_type == "source_credibility":
        return "source_credibility_review"
    if "conflict" in joined:
        if "claim" in joined:
            return "conflicting_claims_review"
        return "validation_conflict_review"
    if "claim" in joined or "single_source" in joined or ids["claim_ids"] or ids["event_ids"]:
        for rid in ids["record_ids"]:
            record = maps["records"].get(rid) or {}
            if record.get("primary_case_dataset_eligible") or "primary" in joined:
                if int(run_quality.get("final_case_dataset_count") or 0) == 0:
                    return "possible_primary_case_evidence"
                return "claim_corroboration_review"
        return "claim_corroboration_review"
    if "zero" in joined:
        return "zero_case_statement_review"
    if "exposure" in joined or "monitoring" in joined:
        return "exposure_monitoring_review"
    if "outside" in joined or "scope" in joined or "geography" in joined:
        return "outside_scope_review"
    if "missing" in joined:
        return "missing_required_fields_review"
    if "provenance" in joined or "evidence" in joined:
        return "provenance_review"
    if "anomaly" in joined or ids["anomaly_ids"]:
        return "anomaly_review"
    if "duplicate" in joined or "cluster" in joined:
        return "duplicate_or_event_cluster_review"
    if "non_primary" in joined:
        return "non_primary_observation_review"
    if "context" in joined:
        return "context_only_review"
    if ids["record_ids"]:
        for rid in ids["record_ids"]:
            record = maps["records"].get(rid) or {}
            if record.get("primary_case_dataset_eligible"):
                return "primary_case_dataset_blocker"
            if record.get("observation_type") == "background_context":
                return "context_only_review"
    return "general_review"


def _primary_record(ids: dict, maps: dict[str, dict[str, dict]], item: dict) -> dict:
    for rid in ids["record_ids"]:
        record = maps["records"].get(rid)
        if record:
            return record
    packet_records = _records_from_packet(item)
    return packet_records[0] if packet_records else {}


def _primary_source(ids: dict, maps: dict[str, dict[str, dict]], item: dict) -> dict:
    for sid in ids["source_ids"]:
        source = maps["sources"].get(sid)
        if source:
            return source
    packet_source = _source_from_packet(item)
    if packet_source:
        return packet_source
    record = _primary_record(ids, maps, item)
    sid = record.get("source_id")
    return maps["sources"].get(sid) or {}


def _date_or_period(record: dict, event: dict, task: dict) -> str:
    return (
        _clean_text(record.get("date_reported"))
        or _clean_text(record.get("event_start_date"))
        or _clean_text(record.get("reporting_period"))
        or _clean_text(event.get("date_or_period"))
        or "-".join([x for x in [task.get("start_date"), task.get("end_date")] if x])
    )


def _dataset_view(record_id: str, artifacts: dict) -> str:
    if not record_id:
        return ""
    view_keys = (
        "final_case_dataset",
        "final_dataset",
        "quarantined_records",
        "pending_review_records",
        "context_records",
        "non_primary_observations",
        "final_dataset_pre_quality_gate",
    )
    for key in view_keys:
        for row in _as_list(artifacts.get(key)):
            if isinstance(row, dict) and row.get("record_id") == record_id:
                return key
    return ""


def _category_message(category: str) -> tuple[str, str, str]:
    messages = {
        "primary_case_dataset_blocker": (
            "This item can change whether a record is eligible for the primary case dataset.",
            "Does this evidence support a task-compatible primary case record?",
            "Inspect the record, evidence quote, source identity, and quality-gate reasons before editing a decision file.",
        ),
        "possible_primary_case_evidence": (
            "No accepted primary case dataset records exist, and this item may contain possible primary case evidence.",
            "Is this a task-compatible primary case observation or only context/secondary evidence?",
            "Inspect corroboration and source identity before deciding whether a structured human decision is warranted.",
        ),
        "validation_limited_review": (
            "Held-out validation was limited. This is not automatic proof that no case exists.",
            "Is there a task-compatible validation source that should be configured or inspected?",
            "Inspect validation source compatibility before interpreting absence of accepted records.",
        ),
        "source_identity_review": (
            "Publisher or source identity uncertainty can affect source independence and credibility.",
            "Who is the actual publisher, and is it independent of other evidence?",
            "Open source identity assessments and source registry metadata.",
        ),
        "source_credibility_review": (
            "Source credibility affects whether evidence should support collection or only context.",
            "Should this source be collection, support, context, or excluded?",
            "Review source registry and source credibility/source critic outputs.",
        ),
        "source_critic_block_review": (
            "A source critic or routing issue may have affected whether evidence was fetched or trusted.",
            "Was a potentially relevant source blocked or downgraded appropriately?",
            "Inspect source critic results and source registry routing fields.",
        ),
        "claim_corroboration_review": (
            "Claim-level support is unresolved or depends on a single source.",
            "Is the claim independently corroborated, conflicting, or only single-source evidence?",
            "Open claims, claim comparisons, and corroborated events.",
        ),
        "conflicting_claims_review": (
            "Comparable claims conflict and may affect cases, date, location, or count semantics.",
            "Which claim should be treated as comparable evidence after review?",
            "Compare claims and source identity before editing decisions.",
        ),
        "outside_scope_review": (
            "The item may be outside the requested disease, location, or time window.",
            "Does the evidence actually match the requested scope?",
            "Inspect the evidence quote and record inclusion decision.",
        ),
        "anomaly_review": (
            "An anomaly rule flagged the item for expert inspection.",
            "Is this a real public-health signal or an extraction/count-semantics problem?",
            "Open anomaly results and the linked record/evidence.",
        ),
        "context_only_review": (
            "The item appears to be context rather than primary case data.",
            "Should this remain context-only?",
            "Inspect context records and source role fields.",
        ),
        "non_primary_observation_review": (
            "The item may be useful public-health context but not a primary case record.",
            "Is the observation correctly separated from the primary dataset?",
            "Inspect observation-type dataset outputs.",
        ),
    }
    return messages.get(
        category,
        (
            "This review item requires human inspection before it should affect downstream interpretation.",
            "What decision, if any, should a human reviewer record?",
            "Inspect the linked artifacts and only then edit a decision file if needed.",
        ),
    )


def _recommended_artifacts(category: str, target_type: str) -> list[str]:
    artifacts = ["human_review_items", "record_inclusion_decisions"]
    if target_type == "record":
        artifacts.extend(["quarantined_records", "context_records", "final_case_dataset"])
    if target_type == "source" or "source" in category:
        artifacts.extend(["source_registry", "source_identity_assessments"])
    if (
        "claim" in category
        or category
        in {"possible_primary_case_evidence", "primary_case_dataset_blocker"}
        or target_type in {"claim", "event_cluster"}
    ):
        artifacts.extend(["claims", "claim_comparisons", "corroborated_events"])
    if "validation" in category or target_type == "validation_result":
        artifacts.extend(["validation_results", "validation_source_compatibility_summary"])
    if target_type == "anomaly" or category == "anomaly_review":
        artifacts.append("anomaly_results")
    if "observation" in category or category == "context_only_review":
        artifacts.append("observation_type_dataset_summary")
    return _unique(artifacts)


def _score_and_level(
    category: str,
    item: dict,
    ids: dict,
    artifacts: dict,
    record: dict,
) -> tuple[int, str]:
    base = {
        "primary_case_dataset_blocker": 105,
        "possible_primary_case_evidence": 100,
        "conflicting_claims_review": 98,
        "validation_conflict_review": 92,
        "claim_corroboration_review": 82,
        "source_identity_review": 76,
        "source_critic_block_review": 76,
        "validation_limited_review": 74,
        "outside_scope_review": 70,
        "anomaly_review": 68,
        "source_credibility_review": 64,
        "missing_required_fields_review": 62,
        "non_primary_observation_review": 52,
        "context_only_review": 35,
        "general_review": 30,
    }.get(category, 40)
    severity = str(item.get("severity") or record.get("severity") or "").lower()
    reason = str(item.get("reason") or "").lower()
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    if severity == "critical":
        base += 20
    elif severity == "high":
        base += 12
    if "single_source" in reason or "primary" in reason:
        base += 8
    if int(run_quality.get("final_case_dataset_count") or 0) == 0 and (
        record.get("primary_case_dataset_eligible") or category == "possible_primary_case_evidence"
    ):
        base += 8
    if ids["target_ids"]:
        base += 2

    if base >= 95:
        level = "P0_critical"
    elif base >= 70:
        level = "P1_high"
    elif base >= 45:
        level = "P2_medium"
    else:
        level = "P3_low"
    return base, level


def _item_title(category: str, item: dict, record: dict, source: dict) -> str:
    source_title = source.get("title") or record.get("source_title")
    bits = [
        category.replace("_", " "),
        item.get("review_id"),
        source_title or record.get("record_id") or ",".join(_as_list(item.get("related_ids"))[:2]),
    ]
    return _clean_text(" - ".join(str(bit) for bit in bits if bit), 140)


def _build_one_item(
    item: dict,
    artifacts: dict,
    maps: dict[str, dict[str, dict]],
) -> dict:
    task = _as_dict(artifacts.get("task"))
    ids = _ids_for_item(item, maps)
    target_type = _target_type(ids, str(item.get("item_type") or ""))
    category = _category_for_item(item, ids, artifacts, maps)
    record = _primary_record(ids, maps, item)
    source = _primary_source(ids, maps, item)
    event = maps["events"].get(ids["event_ids"][0], {}) if ids["event_ids"] else {}
    source_identity = (
        maps["source_identity"].get(source.get("source_id"))
        or maps["source_identity"].get(record.get("source_id"))
        or {}
    )
    score, level = _score_and_level(category, item, ids, artifacts, record)
    why, question, default_action = _category_message(category)
    allowed = SUPPORTED_DECISION_TYPES_BY_TARGET.get(
        target_type,
        SUPPORTED_DECISION_TYPES_BY_TARGET["general"],
    )
    source_url = (
        record.get("source_url")
        or source.get("canonical_url")
        or source.get("url")
        or _as_list(item.get("source_urls"))[:1]
    )
    if isinstance(source_url, list):
        source_url = source_url[0] if source_url else ""
    warnings = _unique(
        [
            source.get("source_identity_warnings"),
            source_identity.get("source_identity_warnings"),
            source.get("publisher_warning_flags"),
            record.get("quality_gate_reasons"),
            item.get("decision_warnings"),
        ]
    )
    current_dataset_view = _dataset_view(
        ids["record_ids"][0] if ids["record_ids"] else "",
        artifacts,
    )
    affects_primary = category in {
        "primary_case_dataset_blocker",
        "possible_primary_case_evidence",
        "claim_corroboration_review",
        "conflicting_claims_review",
    } or bool(record.get("primary_case_dataset_eligible"))
    return {
        "review_item_id": item.get("review_id") or "review_item_unidentified",
        "priority_rank": 0,
        "priority_level": level,
        "priority_score": score,
        "issue_type": item.get("item_type") or "general_review",
        "issue_category": category,
        "target_type": target_type,
        "target_ids": ids["target_ids"],
        "target_record_ids": ids["record_ids"],
        "target_claim_ids": ids["claim_ids"],
        "target_source_ids": ids["source_ids"],
        "target_event_ids": ids["event_ids"],
        "disease": record.get("disease") or task.get("disease") or "",
        "location": (
            record.get("subnational_location")
            or record.get("locality")
            or task.get("location")
            or ""
        ),
        "date_or_period": _date_or_period(record, event, task),
        "short_title": _item_title(category, item, record, source),
        "why_it_matters": why,
        "evidence_quote": _clean_text(
            record.get("evidence_quote") or item.get("evidence_summary"), 800
        ),
        "source_url": source_url or "",
        "source_title": source.get("title") or record.get("source_title") or "",
        "actual_publisher": (
            source.get("actual_publisher")
            or source_identity.get("actual_publisher")
            or source.get("publisher")
            or record.get("publisher")
            or ""
        ),
        "source_type_final": (
            source.get("source_type_final")
            or source_identity.get("source_type_final")
            or source.get("source_type")
            or record.get("source_type")
            or ""
        ),
        "source_identity_confidence": (
            source.get("actual_publisher_confidence")
            or source_identity.get("actual_publisher_confidence")
            or source.get("source_type_confidence")
            or ""
        ),
        "current_status": item.get("status") or "pending",
        "current_dataset_view": current_dataset_view,
        "suggested_reviewer_question": question,
        "suggested_action": item.get("suggested_action") or default_action,
        "allowed_decision_types": allowed,
        "suggested_decision_template_id": f"decision_template_{item.get('review_id') or 'unidentified'}",
        "blocking_final_case_dataset": category in {
            "primary_case_dataset_blocker",
            "possible_primary_case_evidence",
        }
        or (
            bool(record.get("primary_case_dataset_eligible"))
            and current_dataset_view in {"quarantined_records", "pending_review_records"}
        ),
        "affects_primary_case_dataset": affects_primary,
        "affects_validation": "validation" in category or target_type == "validation_result",
        "affects_source_identity": "source_identity" in category,
        "affects_corroboration": "claim" in category or target_type in {"claim", "event_cluster"},
        "affects_quarantine": current_dataset_view == "quarantined_records",
        "recommended_artifacts_to_open": _recommended_artifacts(category, target_type),
        "warnings": warnings,
        "source_provenance": {
            "source_id": (ids["source_ids"][:1] or [record.get("source_id") or ""])[0],
            "source_url": source_url or "",
            "evidence_quote": _clean_text(record.get("evidence_quote"), 800),
            "evidence_chunk_id": record.get("supporting_chunk_id") or "",
            "reason": item.get("reason") or "",
        },
    }


def build_top_review_items(artifacts: dict) -> list[dict]:
    """Return priority-sorted review items from existing artifacts."""

    maps = _artifact_maps(artifacts)
    raw_items = [row for row in _as_list(artifacts.get("human_review_items")) if isinstance(row, dict)]
    validation_item = _validation_limited_item(artifacts)
    if validation_item:
        raw_items.append(validation_item)

    enriched = [_build_one_item(item, artifacts, maps) for item in raw_items]
    level_order = {level: index for index, level in enumerate(PRIORITY_LEVELS)}
    enriched.sort(
        key=lambda row: (
            level_order.get(row["priority_level"], 99),
            -int(row["priority_score"]),
            row["review_item_id"],
        )
    )
    for index, row in enumerate(enriched, start=1):
        row["priority_rank"] = index
    return enriched


def build_human_review_priority_summary(artifacts: dict) -> dict:
    """Build a compact product-facing summary of review priority outputs."""

    items = build_top_review_items(artifacts)
    task = _as_dict(artifacts.get("task"))
    run_quality = _as_dict(artifacts.get("run_quality_summary"))
    final_quality = _as_dict(artifacts.get("final_dataset_quality_summary"))
    compat = _as_dict(artifacts.get("validation_source_compatibility_summary"))
    categories = Counter(row["issue_category"] for row in items)
    levels = Counter(row["priority_level"] for row in items)
    issue_types = Counter(row["issue_type"] for row in items)
    target_types = Counter(row["target_type"] for row in items)
    top = items[:TOP_N_RECOMMENDED]
    validation_limited = bool(run_quality.get("validation_limited")) or str(
        compat.get("compatibility_status") or ""
    ).startswith("incompatible")

    return {
        "session_id": task.get("session_id") or artifacts.get("session_id"),
        "task_disease": task.get("disease"),
        "task_location": task.get("location"),
        "task_start_date": task.get("start_date"),
        "task_end_date": task.get("end_date"),
        "review_item_count": len(_as_list(artifacts.get("human_review_items"))),
        "prioritized_review_item_count": len(items),
        "top_n_recommended": TOP_N_RECOMMENDED,
        "priority_level_counts": {level: levels.get(level, 0) for level in PRIORITY_LEVELS},
        "issue_category_counts": dict(categories),
        "issue_type_counts": dict(issue_types),
        "target_type_counts": dict(target_types),
        "primary_case_dataset_blocker_count": categories.get("primary_case_dataset_blocker", 0),
        "possible_primary_case_evidence_count": categories.get("possible_primary_case_evidence", 0),
        "non_primary_observation_review_count": categories.get("non_primary_observation_review", 0),
        "source_identity_review_count": categories.get("source_identity_review", 0),
        "claim_corroboration_review_count": categories.get("claim_corroboration_review", 0)
        + categories.get("conflicting_claims_review", 0),
        "validation_limited_review_count": categories.get("validation_limited_review", 0),
        "anomaly_review_count": categories.get("anomaly_review", 0),
        "quarantined_record_review_count": sum(
            1 for row in items if row.get("current_dataset_view") == "quarantined_records"
        ),
        "accepted_record_review_count": sum(
            1
            for row in items
            if row.get("current_dataset_view") in {"final_dataset", "final_case_dataset"}
        ),
        "final_case_dataset_count": int(
            run_quality.get("final_case_dataset_count")
            or final_quality.get("final_case_dataset_count")
            or len(_as_list(artifacts.get("final_case_dataset")))
        ),
        "final_dataset_count": int(
            run_quality.get("final_dataset_count")
            or final_quality.get("accepted_record_count")
            or len(_as_list(artifacts.get("final_dataset")))
        ),
        "quarantined_record_count": int(
            run_quality.get("quarantined_record_count")
            or final_quality.get("quarantined_record_count")
            or len(_as_list(artifacts.get("quarantined_records")))
        ),
        "context_record_count": int(
            run_quality.get("context_record_count")
            or final_quality.get("context_record_count")
            or len(_as_list(artifacts.get("context_records")))
        ),
        "validation_limited": validation_limited,
        "run_quality_status": run_quality.get("run_quality_status"),
        "primary_case_dataset_status": run_quality.get("primary_case_dataset_status")
        or final_quality.get("primary_case_dataset_status"),
        "recommended_review_order": [row["review_item_id"] for row in top],
        "top_review_item_ids": [row["review_item_id"] for row in top],
        "top_review_item_titles": [row["short_title"] for row in top],
        "recommended_next_steps": [
            "Open human_review/top_review_items.csv and start with P0/P1 rows.",
            "Inspect linked source, record, claim, anomaly, and validation artifacts before editing any decision file.",
            "Copy review_decision_prefill.json to a working decision file, then set apply_decision=true only for decisions a human has actually made.",
            "Re-run the workflow with the explicit decision file and inspect final_dataset_post_review plus audit trail outputs.",
        ],
        "warnings": _unique(
            [
                run_quality.get("warnings"),
                compat.get("warnings"),
                "validation_limited_is_not_automatic_no_case_proof"
                if validation_limited
                else None,
            ]
        ),
        "generated_from_artifacts_only": True,
        "llm_called_for_review_productization": False,
        "search_called_for_review_productization": False,
        "fetch_called_for_review_productization": False,
    }


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "No prioritized review items were generated.\n"
    lines = [
        "| Rank | Priority | Category | Title | Action |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {level} | {cat} | {title} | {action} |".format(
                rank=row["priority_rank"],
                level=row["priority_level"],
                cat=row["issue_category"],
                title=_clean_text(row["short_title"], 90).replace("|", "/"),
                action=_clean_text(row["suggested_action"], 90).replace("|", "/"),
            )
        )
    return "\n".join(lines) + "\n"


def build_human_review_priority_summary_markdown(artifacts: dict) -> str:
    summary = build_human_review_priority_summary(artifacts)
    items = build_top_review_items(artifacts)
    top = items[:TOP_N_RECOMMENDED]
    lines = [
        "# Human Review Priority Summary",
        "",
        "## 1. Review status",
        "",
        f"- Session: `{summary.get('session_id')}`",
        f"- Task: {summary.get('task_disease')} / {summary.get('task_location')} / {summary.get('task_start_date')} to {summary.get('task_end_date')}",
        f"- Run quality status: `{summary.get('run_quality_status')}`",
        f"- Primary case dataset status: `{summary.get('primary_case_dataset_status')}`",
        f"- Review items in queue: `{summary.get('review_item_count')}`",
        f"- Prioritized items generated: `{summary.get('prioritized_review_item_count')}`",
        "",
        "## 2. Priority counts",
        "",
        "```json",
        _json_block(summary.get("priority_level_counts") or {}),
        "```",
        "",
        "## 3. Top review items",
        "",
        _markdown_table(top),
        "## 4. Why these items matter",
        "",
        "P0/P1 items are listed first because they may affect primary case dataset inclusion, source identity, validation limits, or claim corroboration. Validation-limited items do not prove a negative finding; they tell the reviewer that held-out validation coverage needs inspection.",
        "",
        "## 5. Suggested review workflow",
        "",
        "1. Open `human_review/top_review_items.csv` and review P0/P1 rows first.",
        "2. Use `recommended_artifacts_to_open` to inspect source, record, claim, anomaly, and validation evidence.",
        "3. Copy `human_review/review_decision_prefill.json` to a working decision file.",
        "4. Edit reviewer metadata, target IDs, decision type, reason, notes, and patches only after human review.",
        "5. Set `apply_decision=true` only for explicit decisions the reviewer wants to apply.",
        "6. Re-run the workflow with the existing decision-application mechanism and inspect post-review outputs.",
        "",
        "## 6. Decision file instructions",
        "",
        "`review_decision_template.json` and `review_decision_prefill.json` are templates only. Every generated decision keeps `apply_decision=false` by default. Optimization 6 does not approve, reject, correct, or determine truth automatically.",
        "",
        "## 7. Key artifacts",
        "",
        "- `human_review/human_review_priority_summary.json`",
        "- `human_review/top_review_items.csv`",
        "- `human_review/top_review_items.json`",
        "- `human_review/review_decision_template.json`",
        "- `human_review/review_decision_prefill.json`",
        "- `human_review/review_packet_index.json`",
        "- `human_review/review_action_guide.md`",
        "",
        "## 8. Boundaries",
        "",
        "This artifact is generated from existing session outputs only. It does not call LLMs, search the web, fetch pages, apply human review decisions, provide medical advice, or make official surveillance conclusions.",
        "",
    ]
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


def build_review_decision_template(artifacts: dict) -> dict:
    items = build_top_review_items(artifacts)
    decisions = []
    for row in items[:TOP_N_RECOMMENDED]:
        allowed = row.get("allowed_decision_types") or SUPPORTED_DECISION_TYPES_BY_TARGET["general"]
        decisions.append(
            {
                "decision_id": row["suggested_decision_template_id"],
                "review_id": row["review_item_id"],
                "decision_type": allowed[0],
                "target_type": row["target_type"],
                "target_ids": list(row.get("target_ids") or []),
                "reviewer_id": "REPLACE_WITH_REVIEWER_ID",
                "decided_at": "REPLACE_WITH_ISO_TIMESTAMP",
                "reason": "REPLACE_WITH_HUMAN_REVIEW_REASON",
                "notes": "REPLACE_WITH_OPTIONAL_NOTES",
                "patch": {},
                "corrected_fields": {},
                "confidence": None,
                "apply_decision": False,
                "issue_context": {
                    "priority_level": row["priority_level"],
                    "issue_category": row["issue_category"],
                    "short_title": row["short_title"],
                    "suggested_reviewer_question": row["suggested_reviewer_question"],
                },
            }
        )
    return {
        "template_type": "human_review_decision_template",
        "session_id": _as_dict(artifacts.get("task")).get("session_id")
        or artifacts.get("session_id"),
        "instructions": (
            "Copy this file before editing. Generated decisions are examples only; "
            "set apply_decision=true only after explicit human review."
        ),
        "supported_decision_types": ALL_SUPPORTED_DECISION_TYPES,
        "supported_decision_types_by_target": SUPPORTED_DECISION_TYPES_BY_TARGET,
        "decisions": decisions,
        "generated_from_artifacts_only": True,
        "llm_called_for_review_productization": False,
        "search_called_for_review_productization": False,
        "fetch_called_for_review_productization": False,
    }


def build_review_decision_prefill(artifacts: dict) -> dict:
    items = build_top_review_items(artifacts)
    decisions = []
    for row in items:
        allowed = row.get("allowed_decision_types") or SUPPORTED_DECISION_TYPES_BY_TARGET["general"]
        decisions.append(
            {
                "decision_id": f"prefill_{row['review_item_id']}",
                "review_id": row["review_item_id"],
                "decision_type": allowed[0],
                "target_type": row["target_type"],
                "target_ids": list(row.get("target_ids") or []),
                "reviewer_id": "REPLACE_WITH_REVIEWER_ID",
                "decided_at": "REPLACE_WITH_ISO_TIMESTAMP",
                "reason": f"Review required: {row['issue_category']}",
                "notes": row["suggested_reviewer_question"],
                "patch": {},
                "corrected_fields": {},
                "confidence": None,
                "apply_decision": False,
                "issue_category": row["issue_category"],
                "priority_level": row["priority_level"],
                "short_title": row["short_title"],
                "source_url": row["source_url"],
                "evidence_quote": row["evidence_quote"],
            }
        )
    return {
        "template_type": "human_review_decision_prefill",
        "session_id": _as_dict(artifacts.get("task")).get("session_id")
        or artifacts.get("session_id"),
        "instructions": (
            "This prefill is non-applying. Human reviewers must edit each row and "
            "explicitly set apply_decision=true before the existing decision "
            "application path can modify post-review outputs."
        ),
        "decisions": decisions,
        "generated_from_artifacts_only": True,
        "llm_called_for_review_productization": False,
        "search_called_for_review_productization": False,
        "fetch_called_for_review_productization": False,
    }


def build_review_packet_index(artifacts: dict) -> dict:
    items = build_top_review_items(artifacts)
    packets = []
    for row in items:
        packets.append(
            {
                "review_id": row["review_item_id"],
                "priority_rank": row["priority_rank"],
                "priority_level": row["priority_level"],
                "issue_category": row["issue_category"],
                "target_type": row["target_type"],
                "target_ids": row["target_ids"],
                "recommended_artifacts_to_open": row["recommended_artifacts_to_open"],
                "artifact_paths": {
                    "human_review_items": "collection/human_review_items.json",
                    "top_review_items": "human_review/top_review_items.json",
                    "review_decision_prefill": "human_review/review_decision_prefill.json",
                    "review_action_guide": "human_review/review_action_guide.md",
                },
                "reason": row["why_it_matters"],
            }
        )
    return {
        "session_id": _as_dict(artifacts.get("task")).get("session_id")
        or artifacts.get("session_id"),
        "review_packets": packets,
        "generated_from_artifacts_only": True,
    }


def build_review_action_guide_markdown(artifacts: dict) -> str:
    summary = build_human_review_priority_summary(artifacts)
    return "\n".join(
        [
            "# Human Review Action Guide",
            "",
            "## 1. What to open first",
            "",
            "Start with `human_review/top_review_items.csv`. Review `P0_critical` and `P1_high` items first.",
            "",
            "## 2. How to inspect evidence",
            "",
            "Use each row's `recommended_artifacts_to_open` field to inspect source identity, record inclusion decisions, claims, corroborated events, anomaly results, validation results, and quarantined/context records.",
            "",
            "## 3. How to edit a decision file",
            "",
            "Copy `human_review/review_decision_prefill.json` to a new working file. For each decision you actually want to apply, edit `reviewer_id`, `decided_at`, `decision_type`, `target_ids`, `reason`, `notes`, and any safe `patch` or `corrected_fields` values.",
            "",
            "Set `apply_decision=true` only after a human reviewer has made an explicit decision. Generated template and prefill files intentionally keep `apply_decision=false`.",
            "",
            "## 4. How to apply decisions",
            "",
            "Use the existing workflow decision-application mechanism by providing the edited decision file through the configured `human_review.decisions_path` or CLI decision path and enabling decision application. Do not edit generated template files in place.",
            "",
            "## 5. Post-review outputs to inspect",
            "",
            "- `collection/final_dataset_post_review.json`",
            "- `collection/records_excluded_by_human_review.json`",
            "- `collection/applied_human_review_decisions.json`",
            "- `collection/rejected_human_review_decisions.json`",
            "- `collection/human_review_audit_trail.json`",
            "",
            "## 6. Current run summary",
            "",
            f"- Run quality status: `{summary.get('run_quality_status')}`",
            f"- Primary case dataset status: `{summary.get('primary_case_dataset_status')}`",
            f"- Review item count: `{summary.get('review_item_count')}`",
            f"- Prioritized item count: `{summary.get('prioritized_review_item_count')}`",
            f"- Final case dataset count: `{summary.get('final_case_dataset_count')}`",
            f"- Quarantined record count: `{summary.get('quarantined_record_count')}`",
            "",
            "## 7. Safety boundary",
            "",
            "This guide does not determine truth, provide medical advice, or make an official surveillance conclusion. It only helps a human reviewer act on existing workflow artifacts.",
            "",
        ]
    )


def _csv_ready(rows: list[dict]) -> list[dict]:
    ready = []
    for row in rows:
        ready.append(
            {
                key: "; ".join(row.get(key) or [])
                if isinstance(row.get(key), list)
                else row.get(key)
                for key in CSV_COLUMNS
            }
        )
    return ready


def _decision_csv_ready(decisions: list[dict]) -> list[dict]:
    columns = [
        "decision_id",
        "review_id",
        "decision_type",
        "target_type",
        "target_ids",
        "reviewer_id",
        "decided_at",
        "reason",
        "notes",
        "apply_decision",
        "issue_category",
        "priority_level",
        "short_title",
        "source_url",
        "evidence_quote",
    ]
    rows = []
    for decision in decisions:
        row = {}
        for column in columns:
            value = decision.get(column)
            if isinstance(value, list):
                value = "; ".join(str(item) for item in value)
            row[column] = value
        rows.append(row)
    return rows


def _artifact_manifest(session_dir: Path) -> dict[str, str]:
    human_dir = session_dir / "human_review"
    return {
        "human_review_priority_summary": str(human_dir / "human_review_priority_summary.json"),
        "human_review_priority_summary_json": str(human_dir / "human_review_priority_summary.json"),
        "human_review_priority_summary_md": str(human_dir / "human_review_priority_summary.md"),
        "top_review_items_csv": str(human_dir / "top_review_items.csv"),
        "top_review_items_json": str(human_dir / "top_review_items.json"),
        "review_decision_template_json": str(human_dir / "review_decision_template.json"),
        "review_decision_prefill_json": str(human_dir / "review_decision_prefill.json"),
        "review_decision_prefill_csv": str(human_dir / "review_decision_prefill.csv"),
        "review_packet_index_json": str(human_dir / "review_packet_index.json"),
        "review_action_guide_md": str(human_dir / "review_action_guide.md"),
    }


def write_human_review_workflow_artifacts(session_dir: Path) -> dict:
    """Write human review productization artifacts for a completed session."""

    session_dir = Path(session_dir)
    artifacts = load_human_review_artifacts(session_dir)
    human_dir = session_dir / "human_review"
    collection_dir = session_dir / "collection"
    diagnostics_dir = session_dir / "diagnostics"
    human_dir.mkdir(parents=True, exist_ok=True)
    collection_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    top_items = build_top_review_items(artifacts)
    summary = build_human_review_priority_summary(artifacts)
    summary_md = build_human_review_priority_summary_markdown(artifacts)
    template = build_review_decision_template(artifacts)
    prefill = build_review_decision_prefill(artifacts)
    packet_index = build_review_packet_index(artifacts)
    guide = build_review_action_guide_markdown(artifacts)

    write_json(summary, human_dir / "human_review_priority_summary.json")
    (human_dir / "human_review_priority_summary.md").write_text(
        summary_md,
        encoding="utf-8",
    )
    write_json(top_items, human_dir / "top_review_items.json")
    write_csv_rows(_csv_ready(top_items), human_dir / "top_review_items.csv")
    write_json(template, human_dir / "review_decision_template.json")
    write_json(prefill, human_dir / "review_decision_prefill.json")
    write_csv_rows(
        _decision_csv_ready(_as_list(prefill.get("decisions"))),
        human_dir / "review_decision_prefill.csv",
    )
    write_json(packet_index, human_dir / "review_packet_index.json")
    (human_dir / "review_action_guide.md").write_text(guide, encoding="utf-8")

    write_json(summary, collection_dir / "human_review_priority_summary.json")
    write_json(top_items, collection_dir / "top_review_items.json")
    write_csv_rows(_csv_ready(top_items), collection_dir / "top_review_items.csv")
    write_json(summary, diagnostics_dir / "human_review_priority_summary.json")
    write_json(top_items, diagnostics_dir / "top_review_items.json")

    manifest = _artifact_manifest(session_dir)
    run_summary_path = session_dir / "workflow_run_summary.json"
    run_summary = _as_dict(_read_json(run_summary_path, {}))
    artifact_paths = _as_dict(run_summary.get("artifact_paths"))
    artifact_paths.update(manifest)
    run_summary["artifact_paths"] = artifact_paths
    run_summary["review_reporting"] = {
        "human_review_priority_summary": str(
            human_dir / "human_review_priority_summary.json"
        ),
        "prioritized_review_item_count": summary.get("prioritized_review_item_count", 0),
        "top_n_recommended": summary.get("top_n_recommended", TOP_N_RECOMMENDED),
        "priority_level_counts": summary.get("priority_level_counts", {}),
        "issue_category_counts": summary.get("issue_category_counts", {}),
        "generated_from_artifacts_only": True,
        "llm_called_for_review_productization": False,
        "search_called_for_review_productization": False,
        "fetch_called_for_review_productization": False,
    }
    write_json(run_summary, run_summary_path)
    return manifest
