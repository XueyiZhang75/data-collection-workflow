"""Bounded Wayback CDX metadata leads; archived page bodies are never fetched.

The exact task date window only selects archive-index leads. It does not establish
publication dates, reporting periods, observation eligibility, or page claims.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import ipaddress
import json
import math
import re
import time
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit

import requests

from .disease_identity import disease_names
from .environment import get_env
from .session_runtime import BudgetExceeded, external_call, get_runtime
from .source_identity import lookup_source_identity_registry


_INDEX_URL = "https://web.archive.org/cdx/search/cdx"
_PROVIDER = "internet_archive_wayback"
_MAX_BYTES = 2 * 1024 * 1024
_FIELDS = ("timestamp", "original", "statuscode", "mimetype", "digest")
_OFFICIAL_TYPES = {
    "official_public_health_agency", "national_public_health_agency",
    "state_or_local_public_health_agency", "international_public_health_agency",
    "government_report", "international_organization_report",
}
_NON_PAGE = re.compile(
    r"\.(?:pdf|csv|tsv|json|xml|xlsx?|docx?|pptx?|zip|gz|tar|rar|7z|"
    r"jpe?g|png|gif|webp|svg|ico|mp[34]|wav|avi|mov|css|js)(?:$|[/?&;=])|"
    r"(?:[?&])(?:format|type|filetype)=(?:pdf|csv|tsv|json|xml|xlsx?|docx?|pptx?)(?:&|$)",
    re.I,
)
_MONITORING = re.compile(
    r"\b(?:surveillance|monitoring|statistics|data|dashboard|cases?|counts?|"
    r"reports?|bulletins?|incidence|historical|archive)\b", re.I,
)
_MONITORING_PRODUCT = re.compile(
    r"\b(?:surveillance|monitoring|statistics|datasets?|data|dashboards?|reports?|"
    r"bulletins?|incidence|case[ -]counts?)\b", re.I,
)
_NOTES = (
    "Archive index-only candidate; capture time is not publication time or an "
    "observation reporting period; body not retrieved. CDX metadata does not "
    "verify the archived page's contents or factual authority."
)


class _IndexError(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


def _task_dates(state):
    task = state.get("structured_task") or {}
    parsed = []
    for key in ("start_date", "end_date"):
        value = task.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            raise ValueError("full task dates are required")
        parsed.append(date.fromisoformat(value))
    if parsed[0] > parsed[1]:
        raise ValueError("task start follows task end")
    return parsed


def _url_key(value):
    """Normalize only scheme/host case; never fold paths, escaping, or queries."""
    if not isinstance(value, str) or not value or re.search(r"[\s\\\x00-\x1f\x7f]", value):
        return None
    try:
        parts = urlsplit(value)
        host = parts.hostname
        if (parts.scheme.lower() not in {"http", "https"} or not host
                or parts.username is not None or parts.password is not None or parts.fragment):
            return None
        # Neither registry identity nor a monitoring label grants private URLs.
        address_host = host.rstrip(".")
        if ("." not in address_host or address_host.endswith(
                (".localhost", ".local", ".localdomain", ".internal", ".lan", ".private", ".home", ".corp"))
                or address_host == "localhost" or "%" in host):
            return None
        try:
            ipaddress.ip_address(address_host)
        except ValueError:
            pass
        else:
            return None
        port = parts.port  # Access also validates malformed/out-of-range ports.
        if port is not None and port not in {80, 443}:
            return None
        if not re.fullmatch(r"[a-z0-9.-]+", host, re.I) or ".." in host:
            return None
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    except ValueError:
        return None


def _origin_info(candidate, task):
    if not isinstance(candidate, dict):
        return None, "invalid_candidate"
    if (candidate.get("discovery_method") != "live_search_result"
            or str(candidate.get("search_provider") or "").lower() == "fixture"
            or candidate.get("seed_source_id")):
        return None, "not_live_search_result"
    original = candidate.get("url") or candidate.get("canonical_url")
    key = _url_key(original)
    if not key:
        return None, "unsafe_or_invalid_original_url"
    parts = urlsplit(key)
    host = parts.hostname
    if host == "archive.org" or host.endswith(".archive.org"):
        return None, "archive_url"
    identity = lookup_source_identity_registry(host)
    known_official = bool(identity and str(identity.get("source_type") or "").lower() in _OFFICIAL_TYPES)
    if identity and not known_official:
        return None, "registered_nonofficial_original_domain"
    media_type = str(candidate.get("content_type") or candidate.get("mimetype") or "").split(";")[0].strip().lower()
    if _NON_PAGE.search(unquote(original)) or media_type and media_type not in {"text/html", "application/xhtml+xml"}:
        return None, "non_html_page"
    # Search query/role/type metadata is not evidence that this page is on topic.
    text = " ".join((str(candidate.get("title") or ""), str(candidate.get("snippet") or ""),
                     unquote(parts.path).replace("-", " ").replace("_", " ")))
    disease = str(task.get("disease") or "").strip()
    aliases = [*sorted(disease_names(disease)), *(task.get("disease_aliases") or [])]
    if not disease or not any(
        isinstance(alias, str) and alias.strip() and
        re.search(r"(?<!\w)" + re.escape(alias.strip()) + r"(?!\w)", text, re.I)
        for alias in aliases
    ):
        return None, "missing_page_disease_evidence"
    if not known_official and not _MONITORING_PRODUCT.search(text):
        return None, "unknown_identity_without_monitoring_signal"
    return {"candidate": candidate, "original_url": original, "url_key": key,
            "monitoring": bool(_MONITORING.search(text)), "known_official": known_official,
            "original_identity_status": "registered_official" if known_official else "unknown",
            "original_registry_source_type": identity.get("source_type") if identity else None,
            "selection_reason": "registered_official_page" if known_official else "unregistered_monitoring_page"}, None


def _read_index(params, timeout_seconds):
    """One fixed-host request, no redirects, bounded decompressed bytes and time."""
    started = time.monotonic()
    with requests.get(_INDEX_URL, params=params, timeout=timeout_seconds,
                      stream=True, allow_redirects=False,
                      headers={"Accept": "application/json"}) as response:
        if 300 <= response.status_code < 400:
            raise _IndexError("redirect_refused")
        response.raise_for_status()
        if response.status_code != 200:
            raise _IndexError("http_error")
        length = response.headers.get("Content-Length")
        if length and str(length).isdigit() and int(length) > _MAX_BYTES:
            raise _IndexError("response_byte_limit")
        body = bytearray()
        # Requests' socket timeout measures inactivity. A large read buffer can
        # keep filling indefinitely under a bytewise drip without yielding here.
        # Check each decompressed byte; the socket timeout bounds waits between.
        for chunk in response.iter_content(chunk_size=1):
            if time.monotonic() - started > timeout_seconds:
                raise _IndexError("timeout")
            if len(body) + len(chunk) > _MAX_BYTES:
                raise _IndexError("response_byte_limit")
            body.extend(chunk)
        try:
            rows = json.loads(body)
        except (ValueError, UnicodeError) as exc:
            raise _IndexError("invalid_index_response") from exc
        if not isinstance(rows, list):
            raise _IndexError("invalid_index_response")
        if rows:
            header = rows[0]
            if (not isinstance(header, list) or not all(isinstance(key, str) for key in header)
                    or len(set(header)) != len(header) or not set(_FIELDS).issubset(header)):
                raise _IndexError("invalid_index_response")
        return {"rows": rows, "index_retrieved_at": _iso_now()}


def _records(rows, original_key, lookup_start, lookup_end, cap):
    rejected = Counter()
    if not rows:
        return [], rejected, 0, False
    header, raw = rows[0], rows[1:]
    valid = []
    for row in raw[:cap]:
        if not isinstance(row, list) or len(row) != len(header):
            rejected["malformed_row"] += 1
            continue
        item = dict(zip(header, row))
        if not all(isinstance(item[key], str) for key in _FIELDS):
            rejected["malformed_field"] += 1
            continue
        if _url_key(item["original"]) != original_key:
            rejected["different_original_url"] += 1
            continue
        if item["statuscode"] != "200" or item["mimetype"] != "text/html":
            rejected["non_html_or_non_success_capture"] += 1
            continue
        timestamp = item["timestamp"]
        try:
            if not re.fullmatch(r"[0-9]{14}", timestamp):
                raise ValueError("invalid timestamp")
            captured = datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            if not lookup_start <= captured.date() <= lookup_end:
                raise ValueError("capture outside retrieval-hint date window")
        except ValueError:
            rejected["invalid_or_out_of_range_capture"] += 1
            continue
        valid.append({**item, "captured_at": captured.isoformat()})
    valid.sort(key=lambda row: (row["timestamp"], row["original"], row["digest"]))
    return valid, rejected, len(raw), len(raw) > cap


def _candidate(parent, record, index_url, retrieved_at):
    original = record["original"]
    archive = f"https://web.archive.org/web/{record['timestamp']}/{original}"
    return {
        "source_id": "src_archive_" + hashlib.sha256(archive.encode("utf-8")).hexdigest()[:16],
        "url": archive, "canonical_url": archive, "domain": "web.archive.org",
        "title": "Archived version candidate: " + str(parent.get("title") or original),
        "source_type": "archived_web_page", "discovery_method": "historical_archive_index",
        "source_purpose": "historical_version_lead", "published_date": None,
        "snippet": None, "retrieved_at": retrieved_at, "search_provider": _PROVIDER,
        "provider_channel": "historical_archive_index", "notes": _NOTES,
        "blocked_from_fetch": True, "blocked_from_fetch_reason": "historical_index_only",
        "historical_snapshot": {
            "parent_source_id": parent.get("source_id"), "original_url": original,
            "archive_url": archive, "captured_at": record["captured_at"],
            "capture_timestamp": record["timestamp"], "index_url": index_url,
            "index_retrieved_at": retrieved_at, "archive_provider": _PROVIDER,
            "index_record": {"timestamp": record["timestamp"], "original": original,
                             "statuscode": record["statuscode"], "mimetype": record["mimetype"],
                             "cdx_digest": record["digest"]},
            "verification_status": "index_only",
        },
    }


def discover_historical_versions(state, candidates: list[dict], *, max_origins=4,
                                 max_records_per_origin=64, max_results=64,
                                 timeout_seconds=10):
    """Return candidate dicts, index metadata manifest rows, and bounded audit.

    The live evidence-mode caller supplies actual search candidates. Each CDX
    lookup uses the session's search budget/cache, and each unique admitted
    replay URL uses its search-result budget/cache. No fetch or model operation
    is performed. Failed metadata lookups are nonfatal and remain in the audit.
    The inclusive task dates bound capture lookup, never observation eligibility.
    """
    found, manifest = [], []
    summary = {"status": "pending", "index_only": True, "archive_provider": _PROVIDER,
               "eligible_origin_count": 0, "selected_origin_count": 0,
               "attempted_origin_count": 0, "index_query_count": 0,
               "http_dispatch_count": 0, "cache_hit_count": 0,
               "admitted_result_count": 0, "skipped_origin_count": 0,
               "no_capture_origin_count": 0, "error_origin_count": 0,
               "budget_deferred_origin_count": 0, "partial_origin_count": 0,
               "records_seen": 0, "valid_record_count": 0, "rejected_record_count": 0,
               "duplicate_record_count": 0, "record_rejection_reasons": {},
               "truncated": False, "budget_exhausted": False, "origins": []}
    try:
        start, end = _task_dates(state)
    except (ValueError, TypeError):
        summary["status"] = "invalid_task_dates"
        return found, manifest, summary
    summary["capture_lookup_from"] = start.isoformat().replace("-", "")
    summary["capture_lookup_to"] = end.isoformat().replace("-", "")
    summary["capture_lookup_basis"] = "task_dates_retrieval_hint_only"
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (max_origins, max_records_per_origin, max_results))
            or not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        summary["status"] = "invalid_limits"
        return found, manifest, summary
    runtime = get_runtime()
    if runtime is None or get_env("PIPELINE_MODE") != "evidence":
        summary["status"] = "runtime_unavailable"
        return found, manifest, summary

    def audit_row(**values):
        return {"provider": _PROVIDER, "discovery_method": "historical_archive_index",
                "index_only": True, "body_retrieved": False, **values}

    eligible, seen_origins = [], set()
    for candidate in candidates:
        info, reason = _origin_info(candidate, state.get("structured_task") or {})
        if info and info["url_key"] in seen_origins:
            info, reason = None, "duplicate_original_url"
        if info:
            seen_origins.add(info["url_key"])
            eligible.append(info)
        else:
            summary["skipped_origin_count"] += 1
            manifest.append(audit_row(record_type="origin_selection", result_status="skipped",
                parent_source_id=candidate.get("source_id") if isinstance(candidate, dict) else None,
                rejection_reason=reason))
    eligible.sort(key=lambda info: (not info["known_official"], not info["monitoring"]))
    summary["eligible_origin_count"] = len(eligible)
    summary["selected_origin_count"] = min(len(eligible), max_origins)
    seen_replays, rejection_counts = set(), Counter()

    for position, info in enumerate(eligible):
        parent, original = info["candidate"], info["original_url"]
        audit = {"parent_source_id": parent.get("source_id"), "original_url": original,
                 "original_identity_status": info["original_identity_status"],
                 "original_registry_source_type": info["original_registry_source_type"],
                 "selection_reason": info["selection_reason"], "metadata_only_identity": True,
                 "status": "pending", "reason": None, "records_seen": 0,
                 "valid_record_count": 0, "rejected_record_count": 0,
                 "duplicate_record_count": 0, "admitted_result_count": 0,
                 "record_limit_reached": False, "truncated": False}
        summary["origins"].append(audit)
        if position >= max_origins or len(found) >= max_results or max_records_per_origin == 0:
            audit["status"] = ("origin_limit" if position >= max_origins else
                               "result_limit" if len(found) >= max_results else "record_limit")
            audit["truncated"] = True
            manifest.append(audit_row(record_type="index_query", result_status="skipped", **audit))
            continue
        params = [("url", original), ("matchType", "exact"), ("output", "json"),
                  ("fl", ",".join(_FIELDS)), ("filter", "statuscode:200"),
                  ("filter", "mimetype:text/html"), ("from", summary["capture_lookup_from"]),
                  ("to", summary["capture_lookup_to"]), ("limit", str(max_records_per_origin + 1))]
        index_url = _INDEX_URL + "?" + urlencode(params)
        payload = {"provider": _PROVIDER, "operation": "historical_cdx_index",
                   "index_url": index_url, "timeout_seconds": timeout_seconds,
                   "max_response_bytes": _MAX_BYTES}
        audit["index_url"] = index_url
        cached = runtime.has_cached("search", payload)
        remaining = runtime.ledger.snapshot()["remaining"]
        # Cache replay is possible even when both budget dimensions are exhausted.
        if not cached and (remaining.get("search", 0) <= 0 or remaining.get("search_results", 0) <= 0):
            audit.update(status="budget_exhausted", reason="shared_search_budget_exhausted")
            summary["budget_exhausted"] = True
            manifest.append(audit_row(record_type="index_query", result_status="skipped", **audit))
            continue

        def fetch():
            summary["http_dispatch_count"] += 1
            return _read_index(params, timeout_seconds)

        summary["attempted_origin_count"] += 1
        try:
            response = external_call("search", payload, fetch)
            summary["index_query_count"] += 1
            summary["cache_hit_count"] += int(cached)
        except BudgetExceeded:
            audit.update(status="budget_exhausted", reason="shared_search_budget_exhausted")
            summary["budget_exhausted"] = True
            manifest.append(audit_row(record_type="index_query", result_status="skipped", **audit))
            continue
        except Exception as exc:  # Metadata-only network failures must not stop collection.
            summary["index_query_count"] += 1
            reason = (exc.reason if isinstance(exc, _IndexError) else "timeout"
                      if isinstance(exc, requests.Timeout) else "http_error"
                      if isinstance(exc, requests.HTTPError) else "index_request_error")
            audit.update(status="partial" if reason == "response_byte_limit" else "error",
                         reason=reason, error_type=type(exc).__name__,
                         truncated=reason == "response_byte_limit")
            manifest.append(audit_row(record_type="index_query", result_status="error", **audit))
            continue
        audit["index_retrieved_at"] = response["index_retrieved_at"]
        audit["cache_hit"] = cached
        valid, rejected, count, truncated = _records(response["rows"], info["url_key"],
                                                    start, end, max_records_per_origin)
        audit.update(records_seen=count, valid_record_count=len(valid),
                     rejected_record_count=sum(rejected.values()), record_limit_reached=truncated,
                     truncated=truncated, record_rejection_reasons=dict(rejected))
        rejection_counts.update(rejected)
        for record in valid:
            candidate = _candidate(parent, record, index_url, response["index_retrieved_at"])
            replay = candidate["canonical_url"]
            if replay in seen_replays:
                audit["duplicate_record_count"] += 1
                continue
            seen_replays.add(replay)
            if len(found) >= max_results:
                audit.update(status="partial", reason="result_limit", truncated=True)
                break
            try:
                external_call("search_results", {"canonical_url": replay}, lambda: True)
            except BudgetExceeded:
                audit.update(status="budget_exhausted", reason="shared_search_result_budget_exhausted")
                summary["budget_exhausted"] = True
                break
            except Exception as exc:  # In-doubt cached admissions also remain nonfatal.
                audit.update(status="error", reason="result_admission_error", error_type=type(exc).__name__)
                break
            found.append(candidate)
            audit["admitted_result_count"] += 1
            manifest.append(audit_row(record_type="archive_index_candidate", result_status="accepted",
                                     **candidate))
        if audit["status"] == "pending":
            audit["status"] = ("partial" if truncated else "completed" if valid else
                               "no_valid_captures" if count else "no_captures")
            if truncated:
                audit["reason"] = "record_limit"
        manifest.append(audit_row(record_type="index_query", result_status=audit["status"], **audit))

    origins = summary["origins"]
    for key in ("records_seen", "valid_record_count", "rejected_record_count", "duplicate_record_count"):
        summary[key] = sum(row[key] for row in origins)
    summary["record_rejection_reasons"] = dict(rejection_counts)
    summary["admitted_result_count"] = len(found)
    summary["no_capture_origin_count"] = sum(row["status"] == "no_captures" for row in origins)
    summary["error_origin_count"] = sum("error_type" in row for row in origins)
    summary["budget_deferred_origin_count"] = sum(row["status"] == "budget_exhausted" for row in origins)
    summary["partial_origin_count"] = sum(row["status"] == "partial" for row in origins)
    summary["truncated"] = any(row["truncated"] for row in origins)
    if not eligible:
        summary["status"] = "no_eligible_origins"
    elif summary["truncated"] or (found and (summary["error_origin_count"] or summary["budget_exhausted"])):
        summary["status"] = "partial"
    elif summary["budget_exhausted"]:
        summary["status"] = "budget_exhausted"
    elif summary["error_origin_count"]:
        summary["status"] = "error" if summary["error_origin_count"] == len(origins) else "partial"
    else:
        summary["status"] = ("completed" if found else "no_valid_captures"
                             if summary["rejected_record_count"] else "no_captures")
    return found, manifest, summary
