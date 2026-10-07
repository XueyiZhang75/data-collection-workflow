"""Bounded discovery hints from dates, separate from observation coverage.

Publication metadata is a candidate lead. Supported observation cutoffs and
reporting dates describe observations, not verified document publication.
Neither track assumes a publication schedule or interprets missing dates as zero.
"""
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime
import hashlib
import re

from .evidence_products import METRICS, SCOPE, _documents, _resolved
from .evidence_qualification import _constraint_reasons, _supports, build_evidence_index
from .source_coverage import _entry_is_non_target_context, _generic_coverage_rejection


_POINT_FIELDS = {"as_of_date", "report_date", "date_reported"}


def _day(value, *, timestamp=False):
    if not isinstance(value, str):
        return None
    try:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            return date.fromisoformat(value)
        if timestamp and re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}[Tt ]", value):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    return None


def _exclusion(source, task, has_qualified_record):
    if source.get("blocked_from_fetch"):
        return "blocked_source"
    labels = {str(source.get(key) or "").lower() for key in
              ("source_role_final", "source_role", "triage_role", "status", "final_screening_decision")}
    flags = set(source.get("routing_flags") or []) | set(source.get("screening_flags") or [])
    if (labels & {"validation", "validation_reserved", "validation_source", "reserved_for_validation",
                  "search_endpoint", "needs_human_review", "context_source"}
            or source.get("source_excluded_by_human_review") or source.get("requires_human_review")
            or flags & {"context_only", "blocked_from_structured_extraction", "user_excluded",
                        "validation_reserved", "blocked_from_collection"}):
        return "collection_routing_boundary"
    if _entry_is_non_target_context(source):
        return "noncollection_source_role"
    if any(str(source.get(key) or "").lower() in {"exclude", "excluded", "do_not_fetch"}
           for key in ("status", "screening_decision", "critic_decision", "final_screening_decision")):
        return "excluded_source"
    reason = _generic_coverage_rejection(source, task)
    if reason:
        return reason
    if source.get("usable_for_task_collection") is False:
        return "not_usable_for_collection"
    if not has_qualified_record and not labels & {
        "collection", "task_record_collection_candidate", "include_for_content_fetch",
    }:
        return "collection_relevance_unconfirmed"
    return None


def _current_record(row, scope, index, documents):
    qualification = row.get("evidence_qualification") or {}
    if (qualification.get("status") != "qualified"
            or qualification.get("product_kind") not in {"aggregate", "case_level"}
            or _constraint_reasons(row, scope)):
        return False
    # Match evidence_products' current value/locator contract without requalifying.
    proofs = qualification.get("field_evidence") or []
    fields = {proof.get("field") for proof in proofs}
    return bool(proofs) and all(
        proof.get("value") == row.get(proof.get("field")) and _resolved(proof, index, documents=documents)
        for proof in proofs) and not any(
            row.get(key) not in (None, "") and key not in fields for key in SCOPE + METRICS + ("value_kind",))


def _track(points, source_ids, known_date_source_ids, start, end, basis, max_gaps):
    """Ordinal gaps use space proportional to known points, never window length."""
    dates = sorted(points)
    gaps = []
    if start is not None and end is not None:
        cursor, finish = start.toordinal(), end.toordinal()
        for point in dates:
            ordinal = date.fromisoformat(point).toordinal()
            if cursor < ordinal:
                gaps.append((cursor, ordinal - 1))
            cursor = ordinal + 1
        if cursor <= finish:
            gaps.append((cursor, finish))
    total_days = sum(right - left + 1 for left, right in gaps)
    gaps.sort(key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
    selected = []
    for left, right in gaps[:max_gaps]:
        first, last = date.fromordinal(left).isoformat(), date.fromordinal(right).isoformat()
        key = "|".join((basis, first, last))
        selected.append({"gap_id": "timeline_" + hashlib.sha256(key.encode()).hexdigest()[:20],
                         "date_basis": basis, "start_date": first, "end_date": last,
                         "day_count": right - left + 1})
    dated_sources = set().union(*points.values()) if points else set()
    duration = (end - start).days + 1 if start is not None and end is not None else None
    return {"date_basis": basis, "dates": dates, "distinct_date_count": len(dates),
            "dated_source_count": len(dated_sources),
            "unknown_date_source_count": len(source_ids - known_date_source_ids),
            "outside_window_source_count": len(known_date_source_ids - dated_sources) if duration else None,
            "point_day_fraction": len(dates) / duration if duration else None,
            "gap_day_count": total_days, "gap_count": len(gaps),
            "omitted_gap_count": len(gaps) - len(selected), "gaps": selected}


def build_report_timeline_inventory(state: dict, *, max_gaps: int = 24) -> dict:
    """Project current sources and qualified date proofs without I/O or mutation.

    Gaps are search opportunities between known point dates, including window
    boundaries. They are not missing-report counts or evidence completeness.
    The two date bases remain independent; archive captures never anchor either.
    """
    task = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    current_scope = {**spec, **{key: value for key, value in task.items() if value is not None}}
    start_value = task.get("start_date") or spec.get("start_date")
    end_value = task.get("end_date") or spec.get("end_date")
    start, end = _day(start_value), _day(end_value)
    reason = None
    if start is None or end is None or start > end:
        start = end = None
        reason = "invalid_task_window"
    if type(max_gaps) is not int or not 0 <= max_gaps <= 256:
        start = end = None
        max_gaps = 0
        reason = "invalid_gap_limit"

    index = build_evidence_index(state)
    documents = _documents(index)
    records = [row for row in state.get("qualified_records") or []
               if _current_record(row, current_scope, index, documents)]
    records_by_source = defaultdict(list)
    for record in records:
        if record.get("source_id"):
            records_by_source[str(record["source_id"])].append(record)
    registry = {str(source["source_id"]): source for source in state.get("source_registry") or [] if source.get("source_id")}
    publication_points, observation_points = defaultdict(set), defaultdict(set)
    publication_known, observation_known = set(), set()
    eligible_ids, entries = set(), []
    for sid in sorted(set(registry) | set(records_by_source)):
        source = registry.get(sid, {})
        excluded = _exclusion(source, task, bool(records_by_source[sid]))
        if excluded is None:
            eligible_ids.add(sid)
        raw_publication = source.get("published_date")
        publication = _day(raw_publication, timestamp=True)
        within = start is not None and publication is not None and start <= publication <= end
        snapshot = source.get("historical_snapshot") or {}
        entry = {"source_id": sid, "canonical_url": source.get("canonical_url") or source.get("url"),
                 "title": source.get("title"), "collection_eligible": excluded is None,
                 "exclusion_reason": excluded,
                 "publication_date": {"raw": raw_publication, "date": publication.isoformat() if publication else None,
                                      "status": "candidate_metadata" if publication else "unknown" if not raw_publication else "unresolved_date",
                                      "within_window": within if start is not None else None},
                 "archive_capture_at": snapshot.get("captured_at"), "historical_snapshot": deepcopy(snapshot),
                 "discovery_method": source.get("discovery_method"), "observation_dates": [],
                 "qualified_record_ids": sorted({str(row.get("record_id") or "") for row in records_by_source[sid]})}
        if excluded is None and within:
            publication_points[publication.isoformat()].add(sid)
        if excluded is None and publication is not None:
            publication_known.add(sid)
        seen = set()
        for row in records_by_source[sid]:
            for proof in (row.get("evidence_qualification") or {}).get("field_evidence") or []:
                field, value = proof.get("field"), proof.get("value")
                point = _day(value)
                chunk = index["evidence_chunks"].get(str((proof.get("locator") or {}).get("chunk_id"))) or {}
                if (field not in _POINT_FIELDS or point is None or row.get(field) != value
                        or str(chunk.get("source_id") or "") != sid
                        or not _resolved(proof, index, documents=documents)
                        or not _supports(field, value, proof.get("quote") or "", row)):
                    continue
                identity = (str(row.get("record_id") or ""), field, value, proof.get("document_hash"))
                if identity in seen:
                    continue
                seen.add(identity)
                in_window = start is not None and start <= point <= end
                entry["observation_dates"].append({"field": field, "date": value, "record_id": row.get("record_id"),
                                                   "status": "supported_observation_date", "evidence": deepcopy(proof),
                                                   "within_window": in_window if start is not None else None})
                if excluded is None and in_window:
                    observation_points[value].add(sid)
                if excluded is None:
                    observation_known.add(sid)
        entry["observation_dates"].sort(key=lambda item: (item["date"], item["field"], str(item["record_id"])))
        entries.append(entry)
    return {"schema_version": "report-timeline/1", "status": "not_evaluable" if reason else "evaluated", "reason": reason,
            "requested_window": {"start_date": start_value, "end_date": end_value},
            "window": {"start_date": start.isoformat(), "end_date": end.isoformat(), "inclusive_days": (end - start).days + 1} if start is not None else None,
            "semantics": "Date gaps are discovery opportunities, not missing-report counts, zero cases, or evidence completeness. Publication dates are candidate metadata; supported observation dates are not document publication dates; archive captures anchor neither track.",
            "source_count": len(entries), "collection_eligible_source_count": len(eligible_ids), "sources": entries,
            "publication_candidates": _track(publication_points, eligible_ids, publication_known, start, end, "candidate_publication_metadata", max_gaps),
            "supported_observation_dates": _track(observation_points, eligible_ids, observation_known, start, end, "supported_observation_points", max_gaps)}
