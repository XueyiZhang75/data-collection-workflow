"""Human review decision loading and deterministic application.

Only explicit structured decisions with ``apply_decision: true`` can modify
post-review workflow views. Original normalized records remain preserved.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import json
from copy import deepcopy
from pathlib import Path

from .models import (
    AppliedHumanReviewDecision,
    HumanReviewApplicationSummary,
    HumanReviewAuditEntry,
    HumanReviewDecisionInput,
    RejectedHumanReviewDecision,
)
from .state import DataCollectionState

_FIXED_APPLIED_AT = "2026-05-25T00:00:00Z"

RECORD_PATCH_FIELDS = {
    "disease",
    "disease_standard_name",
    "country",
    "subnational_location",
    "locality",
    "date_reported",
    "event_start_date",
    "event_end_date",
    "reporting_period",
    "as_of_date",
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "hospitalizations",
    "statistical_count_type",
    "count_semantics",
    "count_notes",
    "requires_human_review",
    "human_review_reason",
    "countable",
    "event_member_status",
    "duplicate_of_record_id",
    "representative_record_id",
    "validation_status",
    "anomaly_status",
    "review_status",
    "notes",
}

VALIDATION_PATCH_FIELDS = {
    "validation_status",
    "match_status",
    "reason",
    "needs_human_review",
    "human_review_reason",
    "review_status",
}

ANOMALY_PATCH_FIELDS = {
    "anomaly_status",
    "reason",
    "needs_human_review",
    "human_review_reason",
}

SOURCE_PATCH_FIELDS = {
    "source_role_final",
    "source_review_status",
    "source_excluded_by_human_review",
    "status",
}

ALLOWED_SOURCE_ROLES = {
    "collection",
    "collection_support",
    "validation",
    "validation_reserved",
    "context",
    "context_only",
    "excluded",
}


def _applied_at() -> str:
    return _FIXED_APPLIED_AT


def _require_reviewer_id() -> bool:
    raw = get_env("HUMAN_REVIEW_REQUIRE_REVIEWER_ID")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no"}


def _decision_path_from_state_or_env(state: DataCollectionState) -> str | None:
    apply_files = get_env("HUMAN_REVIEW_APPLY_DECISIONS")
    if apply_files is not None and apply_files.strip().lower() in {"0", "false", "no"}:
        return None
    raw = state.get("human_review_decisions_path") or get_env(
        "HUMAN_REVIEW_DECISIONS_PATH"
    )
    if not raw:
        return None
    return str(raw)


def has_human_review_decision_input(state: DataCollectionState) -> bool:
    """Return whether explicit review decision input is present."""

    if state.get("human_review_decisions"):
        return True
    path = _decision_path_from_state_or_env(state)
    return bool(path)


def _read_decision_file(path_value: str | None) -> list[dict]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rows.append(loaded)
        return rows
    loaded = json.loads(text)
    if isinstance(loaded, list):
        return [row for row in loaded if isinstance(row, dict)]
    if isinstance(loaded, dict):
        rows = loaded.get("decisions")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return [loaded]
    return []


def load_human_review_decision_inputs(state: DataCollectionState) -> list[dict]:
    """Load raw decisions from state and optional JSON/JSONL file."""

    rows = [dict(row) for row in (state.get("human_review_decisions") or []) if isinstance(row, dict)]
    rows.extend(_read_decision_file(_decision_path_from_state_or_env(state)))
    return rows


def _by_id(items: list[dict], key: str) -> dict[str, dict]:
    return {
        str(item.get(key)): item
        for item in items
        if isinstance(item, dict) and item.get(key) not in (None, "")
    }


def _reject(raw: dict, reason: str) -> dict:
    return RejectedHumanReviewDecision(
        decision_id=raw.get("decision_id"),
        review_id=raw.get("review_id"),
        decision_type=raw.get("decision_type"),
        target_type=raw.get("target_type"),
        target_ids=list(raw.get("target_ids") or []),
        rejection_reason=reason,
        raw_decision=dict(raw),
    ).model_dump()


def _patch_from_decision(decision: HumanReviewDecisionInput) -> dict:
    patch = dict(decision.corrected_fields or {})
    patch.update(decision.patch or {})
    return patch


def _new_audit(
    audits: list[dict],
    decision: HumanReviewDecisionInput,
    *,
    target_type: str,
    target_ids: list[str],
    field_name: str | None,
    before_value,
    after_value,
    apply_status: str = "applied",
    rejection_reason: str | None = None,
) -> dict:
    audit = HumanReviewAuditEntry(
        audit_id=f"audit_{len(audits) + 1:03d}",
        decision_id=decision.decision_id,
        review_id=decision.review_id,
        reviewer_id=decision.reviewer_id,
        decided_at=decision.decided_at,
        applied_at=_applied_at(),
        decision_type=decision.decision_type,
        target_type=target_type,
        target_ids=target_ids,
        field_name=field_name,
        before_value=before_value,
        after_value=after_value,
        apply_status=apply_status,
        rejection_reason=rejection_reason,
        reason=decision.reason,
        notes=decision.notes,
        provenance={
            "application_method": "deterministic_human_review_application",
            "apply_decision": decision.apply_decision,
        },
    ).model_dump()
    audits.append(audit)
    return audit


def _mark_target_review_metadata(target: dict, decision: HumanReviewDecisionInput, audit_ids: list[str]) -> None:
    ids = list(target.get("review_decision_ids") or [])
    if decision.decision_id not in ids:
        ids.append(decision.decision_id)
    target["review_decision_ids"] = ids
    existing_audits = list(target.get("human_review_audit_ids") or [])
    for audit_id in audit_ids:
        if audit_id not in existing_audits:
            existing_audits.append(audit_id)
    target["human_review_audit_ids"] = existing_audits
    target["human_review_applied"] = True


def _validate_patch(patch: dict, allowed: set[str]) -> str | None:
    for field in patch:
        if field not in allowed:
            return f"patch_field_not_allowed:{field}"
    return None


def _apply_record_decision(
    decision: HumanReviewDecisionInput,
    records_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    target_ids = list(decision.target_ids or [])
    if not target_ids:
        return False, "missing_target_ids", []
    missing = [rid for rid in target_ids if rid not in records_by_id]
    if missing:
        return False, f"target_not_found:{','.join(missing)}", []
    audit_ids: list[str] = []
    if decision.decision_type == "reject_record":
        for rid in target_ids:
            record = records_by_id[rid]
            updates = {
                "record_excluded_by_human_review": True,
                "review_status": "rejected",
                "final_dataset_included": False,
                "countable": False,
                "human_review_reason": decision.reason,
            }
            for field, value in updates.items():
                before = record.get(field)
                if before == value:
                    continue
                audit = _new_audit(
                    audits,
                    decision,
                    target_type="record",
                    target_ids=[rid],
                    field_name=field,
                    before_value=before,
                    after_value=value,
                )
                audit_ids.append(audit["audit_id"])
                record[field] = value
            _mark_target_review_metadata(record, decision, audit_ids)
        return True, None, audit_ids

    patch = _patch_from_decision(decision)
    if decision.decision_type == "mark_non_countable":
        patch.update({"countable": False, "event_member_status": "non_countable_by_review"})
    elif decision.decision_type == "mark_countable":
        patch.update({"countable": True, "event_member_status": "countable_by_review"})
    elif decision.decision_type == "accept_as_is":
        patch.update({"review_status": "accepted"})
    elif decision.decision_type in {"mark_requires_review", "needs_more_evidence", "defer_decision"}:
        patch.update({"requires_human_review": True, "review_status": "requires_follow_up"})
    elif decision.decision_type == "mark_review_resolved":
        patch.update({"requires_human_review": False, "review_status": "resolved"})
    elif decision.decision_type in {"mark_duplicate", "mark_not_duplicate"}:
        patch.setdefault(
            "event_member_status",
            "duplicate_by_review" if decision.decision_type == "mark_duplicate" else "not_duplicate_by_review",
        )
    elif decision.decision_type != "correct_fields":
        return False, f"unsupported_record_decision:{decision.decision_type}", []

    reason = _validate_patch(patch, RECORD_PATCH_FIELDS)
    if reason:
        return False, reason, []
    for rid in target_ids:
        record = records_by_id[rid]
        per_target_audits: list[str] = []
        for field, value in patch.items():
            before = record.get(field)
            if before == value:
                continue
            audit = _new_audit(
                audits,
                decision,
                target_type="record",
                target_ids=[rid],
                field_name=field,
                before_value=before,
                after_value=value,
            )
            per_target_audits.append(audit["audit_id"])
            audit_ids.append(audit["audit_id"])
            record[field] = value
        if decision.decision_type == "correct_fields":
            record["review_status"] = "corrected"
        record.setdefault("final_dataset_included", True)
        _mark_target_review_metadata(record, decision, per_target_audits)
    return True, None, audit_ids


def _apply_validation_decision(
    decision: HumanReviewDecisionInput,
    validations_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    target_ids = list(decision.target_ids or [])
    if not target_ids:
        return False, "missing_target_ids", []
    missing = [tid for tid in target_ids if tid not in validations_by_id]
    if missing:
        return False, f"target_not_found:{','.join(missing)}", []
    patch = _patch_from_decision(decision)
    if decision.decision_type == "accept_validation_result":
        patch.update({"validation_status": "validated", "review_status": "accepted"})
    elif decision.decision_type == "mark_validation_result_not_applicable":
        patch.update({"validation_status": "not_comparable", "review_status": "not_applicable"})
    elif decision.decision_type == "confirm_conflict":
        patch.update({"validation_status": "conflict", "review_status": "confirmed_conflict"})
    elif decision.decision_type == "resolve_conflict_as_left":
        patch.update({"validation_status": "validated", "match_status": "matched", "review_status": "resolved_as_left"})
    elif decision.decision_type == "resolve_conflict_as_right":
        patch.update({"validation_status": "validated", "match_status": "matched", "review_status": "resolved_as_right"})
    elif decision.decision_type == "mark_needs_more_evidence":
        patch.update({"validation_status": "needs_human_review", "review_status": "needs_more_evidence"})
    elif decision.decision_type != "override_validation_status":
        return False, f"unsupported_validation_decision:{decision.decision_type}", []
    reason = _validate_patch(patch, VALIDATION_PATCH_FIELDS)
    if reason:
        return False, reason, []
    audit_ids: list[str] = []
    for tid in target_ids:
        row = validations_by_id[tid]
        per_target_audits: list[str] = []
        for field, value in patch.items():
            before = row.get(field)
            if before == value:
                continue
            audit = _new_audit(
                audits,
                decision,
                target_type="validation_result",
                target_ids=[tid],
                field_name=field,
                before_value=before,
                after_value=value,
            )
            per_target_audits.append(audit["audit_id"])
            audit_ids.append(audit["audit_id"])
            row[field] = value
        _mark_target_review_metadata(row, decision, per_target_audits)
    return True, None, audit_ids


def _apply_anomaly_decision(
    decision: HumanReviewDecisionInput,
    anomalies_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    target_ids = list(decision.target_ids or [])
    if not target_ids:
        return False, "missing_target_ids", []
    missing = [tid for tid in target_ids if tid not in anomalies_by_id]
    if missing:
        return False, f"target_not_found:{','.join(missing)}", []
    status_by_type = {
        "accept_anomaly": "accepted",
        "dismiss_anomaly": "dismissed",
        "confirm_anomaly": "confirmed",
        "mark_anomaly_resolved": "resolved",
        "mark_anomaly_needs_more_evidence": "needs_more_evidence",
    }
    patch = _patch_from_decision(decision)
    if decision.decision_type in status_by_type:
        patch.update({"anomaly_status": status_by_type[decision.decision_type]})
    else:
        return False, f"unsupported_anomaly_decision:{decision.decision_type}", []
    reason = _validate_patch(patch, ANOMALY_PATCH_FIELDS)
    if reason:
        return False, reason, []
    audit_ids: list[str] = []
    for tid in target_ids:
        row = anomalies_by_id[tid]
        per_target_audits: list[str] = []
        for field, value in patch.items():
            before = row.get(field)
            if before == value:
                continue
            audit = _new_audit(
                audits,
                decision,
                target_type="anomaly",
                target_ids=[tid],
                field_name=field,
                before_value=before,
                after_value=value,
            )
            per_target_audits.append(audit["audit_id"])
            audit_ids.append(audit["audit_id"])
            row[field] = value
        _mark_target_review_metadata(row, decision, per_target_audits)
    return True, None, audit_ids


def _apply_source_decision(
    decision: HumanReviewDecisionInput,
    sources_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    target_ids = list(decision.target_ids or [])
    if not target_ids:
        return False, "missing_target_ids", []
    missing = [tid for tid in target_ids if tid not in sources_by_id]
    if missing:
        return False, f"target_not_found:{','.join(missing)}", []
    patch = _patch_from_decision(decision)
    if decision.decision_type == "approve_source_role":
        patch.setdefault("source_review_status", "approved")
    elif decision.decision_type == "override_source_role":
        patch.setdefault("source_review_status", "role_overridden")
    elif decision.decision_type == "exclude_source":
        patch.update(
            {
                "source_excluded_by_human_review": True,
                "source_review_status": "excluded",
                "source_role_final": "excluded",
            }
        )
    elif decision.decision_type == "mark_source_needs_review":
        patch.update({"source_review_status": "needs_review"})
    else:
        return False, f"unsupported_source_decision:{decision.decision_type}", []
    reason = _validate_patch(patch, SOURCE_PATCH_FIELDS)
    if reason:
        return False, reason, []
    if "source_role_final" in patch and patch["source_role_final"] not in ALLOWED_SOURCE_ROLES:
        return False, f"unsupported_source_role:{patch['source_role_final']}", []
    audit_ids: list[str] = []
    for tid in target_ids:
        row = sources_by_id[tid]
        per_target_audits: list[str] = []
        for field, value in patch.items():
            before = row.get(field)
            if before == value:
                continue
            audit = _new_audit(
                audits,
                decision,
                target_type="source",
                target_ids=[tid],
                field_name=field,
                before_value=before,
                after_value=value,
            )
            per_target_audits.append(audit["audit_id"])
            audit_ids.append(audit["audit_id"])
            row[field] = value
        _mark_target_review_metadata(row, decision, per_target_audits)
    return True, None, audit_ids


def _apply_cluster_decision(
    decision: HumanReviewDecisionInput,
    clusters_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    target_ids = list(decision.target_ids or [])
    if not target_ids:
        return False, "missing_target_ids", []
    missing = [tid for tid in target_ids if tid not in clusters_by_id]
    if missing:
        return False, f"target_not_found:{','.join(missing)}", []
    patch = _patch_from_decision(decision)
    if decision.decision_type == "merge_records_or_clusters":
        patch.setdefault("review_status", "merged_by_review")
    elif decision.decision_type == "split_cluster":
        patch.setdefault("review_status", "split_by_review")
    else:
        return False, f"unsupported_cluster_decision:{decision.decision_type}", []
    allowed = {"review_status", "cluster_status", "representative_record_id"}
    reason = _validate_patch(patch, allowed)
    if reason:
        return False, reason, []
    audit_ids: list[str] = []
    for tid in target_ids:
        row = clusters_by_id[tid]
        per_target_audits: list[str] = []
        for field, value in patch.items():
            before = row.get(field)
            if before == value:
                continue
            audit = _new_audit(
                audits,
                decision,
                target_type="event_cluster",
                target_ids=[tid],
                field_name=field,
                before_value=before,
                after_value=value,
            )
            per_target_audits.append(audit["audit_id"])
            audit_ids.append(audit["audit_id"])
            row[field] = value
        _mark_target_review_metadata(row, decision, per_target_audits)
    return True, None, audit_ids


def _apply_one(
    decision: HumanReviewDecisionInput,
    *,
    records_by_id: dict[str, dict],
    clusters_by_id: dict[str, dict],
    validations_by_id: dict[str, dict],
    anomalies_by_id: dict[str, dict],
    sources_by_id: dict[str, dict],
    audits: list[dict],
) -> tuple[bool, str | None, list[str]]:
    if decision.target_type == "record":
        return _apply_record_decision(decision, records_by_id, audits)
    if decision.target_type in {"event_cluster", "cluster"}:
        return _apply_cluster_decision(decision, clusters_by_id, audits)
    if decision.target_type == "validation_result":
        return _apply_validation_decision(decision, validations_by_id, audits)
    if decision.target_type == "anomaly":
        return _apply_anomaly_decision(decision, anomalies_by_id, audits)
    if decision.target_type == "source":
        return _apply_source_decision(decision, sources_by_id, audits)
    return False, f"unsupported_target_type:{decision.target_type}", []


def _base_post_review_records(records: list[dict]) -> list[dict]:
    result = []
    for record in records:
        row = dict(record)
        row.setdefault("review_status", "unreviewed")
        row.setdefault("review_decision_ids", [])
        row.setdefault("record_excluded_by_human_review", False)
        row.setdefault("final_dataset_included", True)
        row.setdefault("human_review_applied", False)
        row.setdefault("human_review_audit_ids", [])
        row.setdefault("anomaly_status", None)
        row.setdefault("anomaly_ids", [])
        result.append(row)
    return result


def _attach_anomaly_ids(records: list[dict], anomalies: list[dict]) -> None:
    by_record = _by_id(records, "record_id")
    for anomaly in anomalies:
        rid = anomaly.get("record_id")
        if not rid or rid not in by_record:
            continue
        record = by_record[rid]
        ids = list(record.get("anomaly_ids") or [])
        if anomaly.get("anomaly_id") and anomaly["anomaly_id"] not in ids:
            ids.append(anomaly["anomaly_id"])
        record["anomaly_ids"] = ids
        if not record.get("anomaly_status"):
            record["anomaly_status"] = "open"


def apply_human_review_decisions(state: DataCollectionState) -> dict:
    """Apply explicit human review decisions and return auditable state fields."""

    raw_decisions = load_human_review_decision_inputs(state)
    original_records = [deepcopy(row) for row in state.get("normalized_records") or []]
    records = _base_post_review_records([deepcopy(row) for row in state.get("normalized_records") or []])
    event_clusters = [deepcopy(row) for row in state.get("event_clusters") or []]
    validation_results = [deepcopy(row) for row in state.get("validation_results") or []]
    anomaly_results = [deepcopy(row) for row in state.get("anomaly_results") or []]
    source_registry = [deepcopy(row) for row in state.get("source_registry") or []]
    _attach_anomaly_ids(records, anomaly_results)

    records_by_id = _by_id(records, "record_id")
    clusters_by_id = _by_id(event_clusters, "event_cluster_id")
    validations_by_id = _by_id(validation_results, "validation_result_id")
    anomalies_by_id = _by_id(anomaly_results, "anomaly_id")
    sources_by_id = _by_id(source_registry, "source_id")

    applied: list[dict] = []
    rejected: list[dict] = []
    audits: list[dict] = []

    for raw in raw_decisions:
        try:
            decision = HumanReviewDecisionInput(**raw)
        except Exception as exc:  # pragma: no cover - message shape is tested via rejection.
            rejected.append(_reject(raw, f"invalid_decision_schema:{type(exc).__name__}"))
            continue
        if not decision.apply_decision:
            rejected.append(_reject(decision.model_dump(), "apply_decision_false"))
            continue
        if _require_reviewer_id() and not decision.reviewer_id:
            rejected.append(_reject(decision.model_dump(), "missing_reviewer_id"))
            continue
        ok, reason, audit_ids = _apply_one(
            decision,
            records_by_id=records_by_id,
            clusters_by_id=clusters_by_id,
            validations_by_id=validations_by_id,
            anomalies_by_id=anomalies_by_id,
            sources_by_id=sources_by_id,
            audits=audits,
        )
        if not ok:
            rejected.append(_reject(decision.model_dump(), reason or "decision_not_applied"))
            continue
        applied.append(
            AppliedHumanReviewDecision(
                decision_id=decision.decision_id,
                review_id=decision.review_id,
                decision_type=decision.decision_type,
                reviewer_id=decision.reviewer_id,
                decided_at=decision.decided_at,
                applied_at=_applied_at(),
                target_type=decision.target_type,
                target_ids=list(decision.target_ids),
                reason=decision.reason,
                notes=decision.notes,
                confidence=decision.confidence,
                audit_ids=audit_ids,
            ).model_dump()
        )

    final_dataset_post_review = [
        record for record in records if record.get("final_dataset_included") is not False
    ]
    excluded_records = [
        record
        for record in records
        if record.get("record_excluded_by_human_review")
        or record.get("final_dataset_included") is False
    ]
    corrected_records = [
        record
        for record in records
        if record.get("review_status") in {"corrected", "accepted", "resolved"}
        or record.get("human_review_applied")
    ]
    summary = HumanReviewApplicationSummary(
        records_before_review=len(original_records),
        records_after_review=len(final_dataset_post_review),
        records_excluded_by_review=len(excluded_records),
        records_corrected_by_review=sum(
            1 for record in corrected_records if record.get("review_status") == "corrected"
        ),
        clusters_modified_by_review=sum(
            1 for cluster in event_clusters if cluster.get("human_review_applied")
        ),
        validation_results_modified_by_review=sum(
            1 for row in validation_results if row.get("human_review_applied")
        ),
        sources_modified_by_review=sum(
            1 for row in source_registry if row.get("human_review_applied")
        ),
        anomalies_resolved_by_review=sum(
            1
            for row in anomaly_results
            if row.get("anomaly_status") in {"resolved", "dismissed", "confirmed", "accepted"}
        ),
        decisions_provided_count=len(raw_decisions),
        decisions_applied_count=len(applied),
        decisions_rejected_count=len(rejected),
        audit_entry_count=len(audits),
    ).model_dump()
    return {
        "human_review_decisions": raw_decisions,
        "normalized_records": original_records,
        "normalized_records_post_review": records,
        "event_clusters": event_clusters,
        "validation_results": validation_results,
        "anomaly_results": anomaly_results,
        "source_registry": source_registry,
        "applied_human_review_decisions": applied,
        "rejected_human_review_decisions": rejected,
        "human_review_audit_trail": audits,
        "human_review_application_summary": summary,
        "final_dataset_post_review": final_dataset_post_review,
        "records_excluded_by_human_review": excluded_records,
    }
