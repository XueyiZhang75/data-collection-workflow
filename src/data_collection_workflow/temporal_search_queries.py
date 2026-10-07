"""Bounded date-gap retrieval hints; never observation or publication coverage."""
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from urllib.parse import urlsplit, urlunsplit

from .report_timeline import _day, _exclusion, _track, build_report_timeline_inventory


MAX_TEMPORAL_PROBES = 4


def query_key(query):
    return (" ".join(str(query.get("query") or "").casefold().split()),
            str(query.get("provider_channel") or "web_search"))


def search_history(state, query_records=(), operation_history=()):
    """Project actual search outcomes from both existing history stores."""
    initial = [*((state.get("source_search_execution_summary") or {}).get("query_execution_records") or []),
               *query_records]
    history = []
    for row in initial:
        if row.get("selected_for_execution") and row.get("error") != "provider_unavailable":
            history.append({**row, "failed": row.get("execution_status") == "provider_error"})
    # The existing operation ledger survives an interruption before the graph
    # saves this node's query records. Denied calls have no ledger operation.
    for row in operation_history:
        if row.get("kind") == "search" and row.get("temporal_probe"):
            history.append({**row, "failed": row.get("status") == "failed"})
    for row in state.get("recovery_action_history") or []:
        if row.get("kind") != "search":
            continue
        query = row.get("query")
        value = dict(query) if isinstance(query, dict) else {"query": query}
        value.update({key: row[key] for key in ("provider_channel", "temporal_probe") if key in row})
        if (row.get("search_dispatched") or row.get("search_executed")
                or row.get("status") in {"completed", "completed_empty", "failed"}):
            history.append({**value, "failed": row.get("status") == "failed"})
    return history


def attempted_query_keys(state, query_records=(), *, retry_failed=False):
    history = search_history(state, query_records)
    failed = Counter(query_key(row) for row in history if row.get("failed"))
    return {query_key(row) for row in history
            if not retry_failed or not row.get("failed") or failed[query_key(row)] >= 2}


def representative_months(start, end):
    """At most three month indices, without arrays proportional to window length."""
    if start is None or end is None or start > end:
        return []
    first, last = start.year * 12 + start.month - 1, end.year * 12 + end.month - 1
    return list(dict.fromkeys((first, (first + last) // 2, last)))


def _probe_windows(first, last):
    # Breadth-first halves provide distinct bounded retries when a gap has no
    # new dates. They describe query hints, not an assumed reporting cadence.
    pending = [(first.toordinal(), last.toordinal())]
    if first != last:
        left, right = pending.pop()
        middle = (left + right) // 2
        pending = [(left, middle), (middle + 1, right)]
    for _ in range(2 * MAX_TEMPORAL_PROBES):
        if not pending:
            break
        left, right = pending.pop(0)
        yield date.fromordinal(left).isoformat(), date.fromordinal(right).isoformat()
        if left < right:
            middle = (left + right) // 2
            pending.extend(((left, middle), (middle + 1, right)))


def _source_urls(source):
    urls = set()
    for key in ("canonical_url", "url"):
        try:
            parsed = urlsplit(str(source.get(key) or ""))
            if parsed.hostname:
                urls.add(urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                                    parsed.path or "/", parsed.query, "")))
        except ValueError:
            continue
    return urls


def _candidate_allowed(source, task):
    return (_exclusion(source, task, False) in {None, "collection_relevance_unconfirmed"}
            and str(source.get("role_hint") or "").lower() not in {
                "validation", "validation_reference", "validation_reserved", "context", "context_only",
                "context_source", "background", "background_context", "reference"})


def temporal_search_queries(state, *, candidates=(), query_records=(), operation_history=(), channels=None):
    """Return unused probes; callers still enforce every existing search budget.

    Raw candidate publication dates are separate metadata-only hints. This does
    not relax the collection eligibility of the persisted timeline inventory.
    """
    task = {**(state.get("collection_spec") or {}),
            **{key: value for key, value in (state.get("structured_task") or {}).items() if value is not None}}
    first, last = _day(task.get("start_date")), _day(task.get("end_date"))
    if first is None or last is None or first > last or not task.get("disease"):
        return []
    history = search_history(state, query_records, operation_history)
    if len({query_key(row) for row in history if row.get("temporal_probe")}) >= MAX_TEMPORAL_PROBES:
        return []
    used = {query_key(row) for row in history}
    # Rebuild from current evidence rather than trusting a saved audit for a
    # previous task/window or a pre-recovery version of the source registry.
    inventory = build_report_timeline_inventory(state)
    tracks = [inventory.get("supported_observation_dates") or {},
              inventory.get("publication_candidates") or {}]
    points = defaultdict(set)
    excluded_ids, excluded_urls = set(), set()
    for source in state.get("source_registry") or []:
        if not _candidate_allowed(source, task):
            if source.get("source_id"):
                excluded_ids.add(str(source["source_id"]))
            excluded_urls.update(_source_urls(source))
    for source in candidates:
        if (not _candidate_allowed(source, task) or str(source.get("source_id") or "") in excluded_ids
                or _source_urls(source) & excluded_urls):
            continue
        point = _day(source.get("published_date"), timestamp=True)
        if point and first <= point <= last:
            points[point.isoformat()].add(str(source.get("source_id") or source.get("url") or "candidate"))
    if points:
        tracks.insert(0, _track(points, set(), set(), first, last, "candidate_publication_metadata", 24))
    # Prefer actual known date points; absence of dates still permits bounded
    # task-window retrieval without suggesting a missing weekly/daily report.
    track = next((value for value in tracks if value.get("distinct_date_count")), tracks[-1])
    basis = track.get("date_basis") if track.get("distinct_date_count") else "task_window"
    gaps = track.get("gaps") or []
    available = set(channels if channels is not None else ["official_site_search", "web_search"])
    templates = [("official_site_search", "official_public_health_agency", "surveillance reports cases"),
                 ("database_search", "structured_database", "surveillance data tables"),
                 ("literature_api", "peer_reviewed_literature", "surveillance study cases"),
                 ("news_search", "news_and_situation_report", "reported cases updates"),
                 ("web_search", "news_and_situation_report", "surveillance reports cases")]
    template = next((item for item in templates if item[0] in available), None)
    if not template:
        return []
    channel, source_type, terms = template
    result, seen = [], set(used)
    for gap in gaps:
        start, end = _day(gap.get("start_date")), _day(gap.get("end_date"))
        if start is None or end is None or start > end:
            continue
        start, end = max(first, start), min(last, end)
        if start > end:
            continue
        for left, right in _probe_windows(start, end):
            identity = [task.get("disease"), task.get("location") or task.get("geography"),
                        first.isoformat(), last.isoformat(), left, right]
            probe = {"probe_id": "temporal_" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20],
                     "start_date": left, "end_date": right, "date_basis": basis}
            query = {"query": f'"{task["disease"]}" "{task.get("location") or task.get("geography") or ""}" {terms} {left} {right}',
                     "source_type": source_type, "provider_channel": channel,
                     "role_hint": "collection_support", "time_terms": [left, right],
                     "disease_terms_used": [task["disease"]], "query_source": "temporal_gap_probe",
                     "query_rationale": "Bounded date-gap retrieval hint; dates do not establish coverage.",
                     "temporal_probe": probe, "search_direction": "temporal:" + probe["probe_id"]}
            key = query_key(query)
            if key not in seen:
                result.append(query)
                seen.add(key)
            if len(result) >= MAX_TEMPORAL_PROBES:
                return result
    return result
