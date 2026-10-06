"""Human review packet construction and decision intake (Step 12).

Builds structured review packets for items already in `human_review_queue`,
attaches relevant context (source registry entries, records, linked events,
conflicts), and optionally records pre-supplied human decisions from
`state["human_review_decisions"]`.

This step does NOT build a UI, does NOT modify records or conflicts, and does
NOT resolve conflicts. It only structures items, attaches context, and
records decisions plus audit metadata.
"""

from __future__ import annotations

from collections import Counter

from ..config import load_human_review_policy
from ..human_review_application import apply_human_review_decisions
from ..models import (
    HumanReviewDecision,
    HumanReviewItem,
    HumanReviewPacket,
    HumanReviewPolicy,
)
from ..run_events import emit_workflow_progress
from ..state import DataCollectionState, append_trace

_FIXED_REVIEW_TIMESTAMP = "2026-05-25T00:00:00Z"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _fixed_review_timestamp() -> str:
    return _FIXED_REVIEW_TIMESTAMP


def _by_id(items: list[dict], key: str) -> dict[str, dict]:
    return {item.get(key): item for item in items if item.get(key)}


def _records_by_id(state: DataCollectionState) -> dict[str, dict]:
    """Build a record_id → record map, normalized > validated > raw."""

    out: dict[str, dict] = {}
    for list_name in ("raw_records", "validated_records", "normalized_records"):
        for r in (state.get(list_name) or []):
            rid = r.get("record_id")
            if rid:
                out[rid] = r
    return out


def _find_source_entry(state: DataCollectionState, source_id: str) -> dict | None:
    if not source_id:
        return None
    for entry in (state.get("source_registry") or []):
        if entry.get("source_id") == source_id:
            return entry
    return None


def _find_conflict(state: DataCollectionState, conflict_id: str) -> dict | None:
    if not conflict_id:
        return None
    for c in (state.get("conflicts") or []):
        if c.get("conflict_id") == conflict_id:
            return c
    return None


def _find_linked_event(
    state: DataCollectionState, linked_event_id: str
) -> dict | None:
    if not linked_event_id:
        return None
    for ev in (state.get("linked_events") or []):
        if ev.get("linked_event_id") == linked_event_id:
            return ev
    return None


def _related_records(
    state: DataCollectionState, related_ids: list[str]
) -> list[dict]:
    record_map = _records_by_id(state)
    seen: set[str] = set()
    result: list[dict] = []
    for rid in related_ids or []:
        record = record_map.get(rid)
        if record is not None and rid not in seen:
            seen.add(rid)
            result.append(record)
        else:
            # Maybe rid is actually a conflict_id; pull its record_ids.
            conflict = _find_conflict(state, rid)
            if conflict is None:
                continue
            for sub_rid in conflict.get("record_ids") or []:
                if sub_rid in seen:
                    continue
                sub_record = record_map.get(sub_rid)
                if sub_record is not None:
                    seen.add(sub_rid)
                    result.append(sub_record)
    return result


# ---------------------------------------------------------------------------
# Packet construction
# ---------------------------------------------------------------------------


def _build_packet_sections(
    item: dict,
    state: DataCollectionState,
    policy: HumanReviewPolicy,  # noqa: ARG001 — reserved for future per-policy section logic
) -> dict:
    item_type = item.get("item_type")
    related_ids = list(item.get("related_ids") or [])
    sections: dict = {}

    if item_type == "source_screening":
        source_id = related_ids[0] if related_ids else None
        entry = _find_source_entry(state, source_id)
        sections["source_registry_entry"] = entry
        return sections

    if item_type == "record_schema_validation":
        record_id = related_ids[0] if related_ids else None
        raw_map = _by_id(state.get("raw_records") or [], "record_id")
        validated_map = _by_id(state.get("validated_records") or [], "record_id")
        sections["raw_record"] = raw_map.get(record_id)
        sections["validated_record"] = validated_map.get(record_id)
        return sections

    if item_type == "record_normalization":
        record_id = related_ids[0] if related_ids else None
        validated_map = _by_id(state.get("validated_records") or [], "record_id")
        normalized_map = _by_id(state.get("normalized_records") or [], "record_id")
        sections["validated_record"] = validated_map.get(record_id)
        sections["normalized_record"] = normalized_map.get(record_id)
        return sections

    if item_type == "record_linking":
        linked_event = None
        for rid in related_ids:
            ev = _find_linked_event(state, rid)
            if ev is not None:
                linked_event = ev
                break
        sections["linked_event"] = linked_event
        sections["related_records"] = _related_records(state, related_ids)
        return sections

    if item_type == "cross_source_conflict":
        conflict = None
        for rid in related_ids:
            c = _find_conflict(state, rid)
            if c is not None:
                conflict = c
                break
        sections["conflict"] = conflict
        linked_event = None
        if conflict is not None and conflict.get("linked_event_id"):
            linked_event = _find_linked_event(state, conflict["linked_event_id"])
        sections["linked_event"] = linked_event
        # Prefer the conflict's record_ids when available.
        record_id_pool = (
            list(conflict.get("record_ids") or []) if conflict else list(related_ids)
        )
        sections["related_records"] = _related_records(state, record_id_pool)
        return sections

    # Fallback for unknown types.
    sections["related_records"] = _related_records(state, related_ids)
    return sections


_FIXTURE_TEXT_MARKER = "Fixture document for workflow testing only"


def _has_fixture_origin(packet_sections: dict) -> bool:
    return _section_has_fixture(packet_sections)


def _section_has_fixture(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, dict):
        if obj.get("is_fixture_document") is True:
            return True
        if obj.get("fixture_id"):
            return True
        metadata = obj.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("synthetic_fixture") is True:
                return True
            if metadata.get("not_real_public_health_data") is True:
                return True
            if metadata.get("fixture_id"):
                return True
        eq = obj.get("evidence_quote")
        if isinstance(eq, str) and _FIXTURE_TEXT_MARKER in eq:
            return True
        for value in obj.values():
            if _section_has_fixture(value):
                return True
        return False
    if isinstance(obj, list):
        return any(_section_has_fixture(v) for v in obj)
    if isinstance(obj, str):
        return _FIXTURE_TEXT_MARKER in obj
    return False


def _make_review_packet(
    item: dict,
    state: DataCollectionState,
    policy: HumanReviewPolicy,
) -> HumanReviewPacket:
    item_type = item.get("item_type") or "unknown"
    priority = int(policy.priority_by_item_type.get(item_type, 99))
    sections = _build_packet_sections(item, state, policy)
    fixture_warning = (
        policy.synthetic_fixture_warning if _has_fixture_origin(sections) else None
    )
    return HumanReviewPacket(
        review_id=item.get("review_id") or "",
        item_type=item_type,
        priority=priority,
        status=item.get("status") or "pending",
        reason=item.get("reason") or "",
        related_ids=list(item.get("related_ids") or []),
        packet_sections=sections,
        synthetic_fixture_warning=fixture_warning,
    )


# ---------------------------------------------------------------------------
# Decision handling
# ---------------------------------------------------------------------------


def _decision_map(state: DataCollectionState) -> dict[str, HumanReviewDecision]:
    out: dict[str, HumanReviewDecision] = {}
    for raw in (state.get("human_review_decisions") or []):
        try:
            decision = HumanReviewDecision(**raw)
        except Exception:
            continue
        # Last write wins for duplicate review_ids.
        out[decision.review_id] = decision
    return out


def _apply_decision(
    item: dict,
    decision: HumanReviewDecision,
    policy: HumanReviewPolicy,
) -> tuple[dict, list[str]]:
    updated = dict(item)
    warnings = list(updated.get("decision_warnings") or [])

    if decision.decision not in policy.allowed_decisions:
        updated["status"] = "invalid_decision"
        updated["human_decision"] = decision.decision
        updated["decision_applied"] = False
        if "invalid_decision" not in warnings:
            warnings.append("invalid_decision")
    else:
        new_status = policy.decision_to_status.get(
            decision.decision, updated.get("status") or "pending"
        )
        updated["status"] = new_status
        updated["human_decision"] = decision.decision
        if decision.notes:
            updated["notes"] = decision.notes
        if decision.reviewer_id:
            updated["reviewer_id"] = decision.reviewer_id
        updated["decided_at"] = decision.decided_at or _fixed_review_timestamp()
        updated["modified_values"] = dict(decision.modified_values or {})
        updated["decision_source"] = "state.human_review_decisions"
        updated["decision_applied"] = decision.decision != "no_action"

    updated["decision_warnings"] = warnings
    return updated, warnings


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def human_review(state: DataCollectionState) -> dict:
    """Build review packets and (optionally) record supplied human decisions."""

    policy = HumanReviewPolicy(**load_human_review_policy())
    incoming_queue = list(state.get("human_review_queue") or [])
    decisions = _decision_map(state)
    emit_workflow_progress(
        "human_review",
        "human review queue processing started",
        {
            "input_review_item_count": len(incoming_queue),
            "decision_count": len(decisions),
        },
    )
    invalid_decision_count = sum(
        1
        for raw in (state.get("human_review_decisions") or [])
        if raw.get("review_id") not in decisions
    )

    item_type_counter: Counter = Counter()
    status_counter: Counter = Counter()
    decision_counter: Counter = Counter()
    priority_counter: Counter = Counter()
    pending = reviewed = follow_up = deferred = invalid_status = 0
    decision_supplied = 0
    decision_applied = 0
    fixture_origin_review_count = 0

    processed: list[dict] = []
    for raw_item in incoming_queue:
        # Coerce shape via Pydantic and back to dict so optional fields exist.
        item = HumanReviewItem(**raw_item).model_dump()

        packet = _make_review_packet(item, state, policy)
        item["review_packet"] = packet.model_dump()
        item["priority"] = packet.priority
        if packet.synthetic_fixture_warning:
            fixture_origin_review_count += 1

        decision = decisions.get(item.get("review_id"))
        if decision is not None:
            decision_supplied += 1
            item, _w = _apply_decision(item, decision, policy)
            if item.get("decision_applied"):
                decision_applied += 1

        status = item.get("status") or "pending"
        decision = item.get("human_decision")
        item_type_counter[item.get("item_type") or "unknown"] += 1
        status_counter[status] += 1
        priority_counter[int(item.get("priority") or 99)] += 1
        if decision:
            decision_counter[decision] += 1

        if status == "pending":
            pending += 1
        elif status == "reviewed":
            reviewed += 1
        elif status == "requires_follow_up":
            follow_up += 1
        elif status == "deferred":
            deferred += 1
        elif status == "invalid_decision":
            invalid_status += 1

        processed.append(item)

    processed.sort(
        key=lambda x: (
            int(x.get("priority") if x.get("priority") is not None else 99),
            x.get("review_id") or "",
        )
    )

    summary = {
        "input_review_item_count": len(incoming_queue),
        "output_review_item_count": len(processed),
        "pending_count": pending,
        "reviewed_count": reviewed,
        "requires_follow_up_count": follow_up,
        "deferred_count": deferred,
        "invalid_decision_count": invalid_status + invalid_decision_count,
        "decision_supplied_count": decision_supplied,
        "decision_applied_count": decision_applied,
        "item_type_counts": dict(item_type_counter),
        "status_counts": dict(status_counter),
        "decision_counts": dict(decision_counter),
        "priority_counts": {str(k): v for k, v in priority_counter.items()},
        "fixture_origin_review_count": fixture_origin_review_count,
    }

    application_state = dict(state)
    application_state["human_review_queue"] = processed
    application_output = apply_human_review_decisions(application_state)
    app_summary = application_output.get("human_review_application_summary") or {}
    summary["human_review_application_summary"] = app_summary
    summary["decision_application_applied_count"] = app_summary.get(
        "decisions_applied_count", 0
    )
    summary["decision_application_rejected_count"] = app_summary.get(
        "decisions_rejected_count", 0
    )
    emit_workflow_progress(
        "human_review",
        "human review summary ready",
        {
            "output_review_item_count": len(processed),
            "pending_count": pending,
            "reviewed_count": reviewed,
            "requires_follow_up_count": follow_up,
            "deferred_count": deferred,
            "decision_applied_count": decision_applied,
            "item_type_counts": dict(item_type_counter),
        },
    )

    trace = append_trace(
        state,
        node_name="human_review",
        message=(
            f"Processed {len(processed)} review item(s): {pending} pending, "
            f"{reviewed} reviewed, {follow_up} follow-up, {deferred} deferred, "
            f"{decision_applied} decision(s) applied."
        ),
        metadata=summary,
    )
    return {
        "human_review_queue": processed,
        "human_review_summary": summary,
        **application_output,
        "collection_trace": trace,
    }
