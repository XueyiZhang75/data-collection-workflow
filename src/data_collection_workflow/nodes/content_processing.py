"""Content fetching, parsing, and document quality (Step 5).

Default behavior is fully offline: when `ENABLE_LIVE_FETCH` is not set to
"true", the workflow generates deterministic metadata-stub documents and does
NOT contact the network. Live HTTP fetching is gated behind that environment
variable so tests stay offline-safe.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from time import monotonic
from urllib.parse import urljoin, urlsplit

import re


def _traceable_tool(name: str):
    try:
        from langsmith import traceable

        return traceable(name=name, run_type="tool")
    except Exception:
        return lambda fn: fn


from ..config import (
    get_collection_mode,
    load_content_fetch_policy,
    load_evidence_chunking_policy,
    load_hantavirus_fixture_documents,
    load_source_role_policy,
)
from ..disease_relevance import (
    REJECT_DISEASE_STATUSES,
    TARGET_DISEASE_MATCH,
    UNRELATED_DISEASE,
    AMBIGUOUS_DISEASE,
    assessment_fields,
    assess_chunk_disease_relevance,
    assess_document_disease_relevance,
    build_disease_relevance_context,
    update_disease_relevance_summary,
)
from ..models import (
    ContentFetchPolicy,
    ContentFetchRequest,
    Document,
    EvidenceChunk,
    EvidenceChunkingPolicy,
    FixtureDocument,
    FixtureDocumentCatalog,
)
from ..run_events import emit_workflow_progress
from ..source_identity import (
    enrich_source_identity_registry_post_fetch,
    lookup_source_identity_registry,
)
from ..source_product_profile import (
    apply_source_product_profiles,
    source_product_priority,
    source_product_skip_reason,
)
from ..source_coverage import annotate_source_coverage, build_source_coverage_requirements
from ..state import DataCollectionState, append_trace

_OFFLINE_FIXED_TIMESTAMP = "2026-05-25T00:00:00Z"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SEARCH_DERIVED_DISCOVERY_METHODS = {"fixture_search_result", "live_search_result"}
_SEARCH_DERIVED_FETCH_FINAL_ROLES = {
    "collection",
    "validation",
    "collection_support",
    "context",
}
_SEARCH_DERIVED_BLOCKED_FINAL_ROLES = {
    "excluded",
    "search_endpoint",
    "needs_human_review",
}
_OFFICIAL_SOURCE_TYPES = {
    "official_public_health_agency",
    "government_report",
    "international_public_health_agency",
    "national_public_health_agency",
    "state_public_health_agency",
    "state_or_local_public_health_agency",
    "local_public_health_agency",
}
_STRUCTURED_SOURCE_TYPES = {
    "structured_database",
    "dashboard",
    "public_health_dataset",
    "surveillance_database",
}
_LITERATURE_SOURCE_TYPES = {
    "academic_or_peer_reviewed_source",
    "peer_reviewed_literature",
    "literature_api",
    "journal_article",
}
_NEWS_SOURCE_TYPES = {
    "news_and_situation_report",
    "news",
    "media_report",
    "news_media",
    "secondary_media",
}

_SOURCE_TRUST_REVIEW_TYPES = {
    "blog",
    "community_forum",
    "forum",
    "media_report",
    "news",
    "news_and_situation_report",
    "news_media",
    "secondary_media",
    "social_media",
    "unknown",
}
_TASK_USABILITY_REVIEW_SOURCE_TYPES = (
    _SOURCE_TRUST_REVIEW_TYPES
    | _LITERATURE_SOURCE_TYPES
    | {
        "academic_or_peer_reviewed_source",
        "academic_secondary",
        "secondary_source",
    }
)
_NON_EXACT_DISEASE_FIT_VALUES = {
    "mismatch",
    "disease_mismatch",
    "non_target",
    "non_target_disease",
    "excluded",
}
_NON_EXACT_GEOGRAPHY_FIT_VALUES = {
    "broader",
    "broader_than_task",
    "context",
    "context_only",
    "country_context",
    "different_geography",
    "excluded",
    "geography_mismatch",
    "global",
    "national_context",
    "outside_task_geography",
    "regional",
    "regional_context",
    "supranational",
}
_NON_EXACT_DATE_FIT_VALUES = {
    "as_of",
    "broader",
    "broader_than_task",
    "campaign_period",
    "context",
    "current_page",
    "date_mismatch",
    "excluded",
    "partial_overlap",
    "period_mismatch",
    "season_to_date",
    "temporal_mismatch",
    "wrong_period",
    "ytd",
}


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = (get_env(name) or "").strip().lower()
    if not raw:
        return default
    return raw == "true"


def _parse_positive_int_env(name: str) -> int | None:
    raw = (get_env(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _live_fetch_enabled() -> bool:
    """True only if ENABLE_LIVE_FETCH is set to "true" (case-insensitive)."""

    return (get_env("ENABLE_LIVE_FETCH") or "").strip().lower() == "true"


def _fixture_documents_enabled() -> bool:
    """True only if USE_FIXTURE_DOCUMENTS is set to "true" (case-insensitive)."""

    return (get_env("USE_FIXTURE_DOCUMENTS") or "").strip().lower() == "true"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = get_env(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 or (allow_zero and value == 0) else default


def _env_float(name: str, default: float) -> float:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_csv(name: str, default: list[str] | None = None) -> list[str]:
    raw = get_env(name)
    if raw is None or not raw.strip():
        return list(default or [])
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(default or [])


def _fetch_config_from_env(policy: ContentFetchPolicy) -> dict:
    request_cfg = policy.request or {}
    return {
        "fetch_search_derived_sources": _env_bool(
            "FETCH_SEARCH_DERIVED_SOURCES", False
        ),
        "max_search_derived_sources": _env_int(
            "FETCH_MAX_SEARCH_DERIVED_SOURCES", 2, allow_zero=True
        ),
        "max_total_sources": _env_int("FETCH_MAX_TOTAL_SOURCES", 10, allow_zero=True),
        "min_credibility_score": _env_float(
            "FETCH_MIN_CREDIBILITY_SCORE", 0.55
        ),
        "allowed_final_roles": _env_csv(
            "FETCH_ALLOWED_FINAL_ROLES",
            sorted(_SEARCH_DERIVED_FETCH_FINAL_ROLES),
        ),
        "allow_needs_review": _env_bool("FETCH_ALLOW_NEEDS_REVIEW", False),
        "domain_allowlist": _env_csv("FETCH_DOMAIN_ALLOWLIST", []),
        "domain_blocklist": _env_csv("FETCH_DOMAIN_BLOCKLIST", []),
        "max_bytes": _env_int(
            "FETCH_MAX_BYTES",
            int(request_cfg.get("max_response_bytes") or 5_000_000),
        ),
        "user_agent": (
            get_env("FETCH_USER_AGENT")
            or request_cfg.get("user_agent")
            or "data-collection-workflow/0.1"
        ),
        "parse_pdf_text": _env_bool("FETCH_PARSE_PDF_TEXT", True),
        "parse_tables": _env_bool("FETCH_PARSE_TABLES", True),
        "store_raw_text": _env_bool("FETCH_STORE_RAW_TEXT", False),
        "content_fixture_map_path": (
            get_env("CONTENT_FIXTURE_MAP_PATH") or ""
        ).strip()
        or None,
        "external_fetch_enabled": _env_bool("EXTERNAL_FETCH_ENABLED", False),
        "external_fetch_provider_order": _env_csv(
            "EXTERNAL_FETCH_PROVIDER_ORDER",
            ["tavily_extract", "native_requests"],
        ),
        "tavily_extract_format": (
            get_env("TAVILY_EXTRACT_FORMAT") or "markdown"
        ).strip()
        or "markdown",
        "tavily_extract_depth": (
            get_env("TAVILY_EXTRACT_DEPTH") or "advanced"
        ).strip()
        or "advanced",
        "tavily_extract_timeout_seconds": _env_float(
            "TAVILY_EXTRACT_TIMEOUT_SECONDS", 45.0
        ),
        "tavily_extract_chunks_per_source": _env_int(
            "TAVILY_EXTRACT_CHUNKS_PER_SOURCE", 5
        ),
        "max_node_seconds": _env_float("FETCH_MAX_NODE_SECONDS", 0.0),
    }


def _parse_source_id_allowlist() -> set[str] | None:
    """Read SOURCE_ID_ALLOWLIST. Return a set of source IDs or None."""

    raw = get_env("SOURCE_ID_ALLOWLIST") or ""
    if not raw.strip():
        return None
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


def _load_fixture_document_map() -> dict[str, FixtureDocument]:
    """Load the fixture document catalog and map source_id -> FixtureDocument."""

    catalog = FixtureDocumentCatalog(**load_hantavirus_fixture_documents())
    result: dict[str, FixtureDocument] = {}
    for fixture in catalog.fixture_documents:
        if fixture.source_id not in result:
            result[fixture.source_id] = fixture
    return result


def _load_content_fixture_map(path_value: str | None) -> list[dict]:
    if not path_value:
        return []
    path = _resolve_project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"CONTENT_FIXTURE_MAP_PATH points to a missing file: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    fixtures = data.get("fixtures") if isinstance(data, dict) else None
    if not isinstance(fixtures, list):
        raise ValueError(f"Content fixture map must contain a fixtures list: {path}")
    result: list[dict] = []
    for item in fixtures:
        if not isinstance(item, dict):
            continue
        fixture_path = item.get("fixture_path")
        if not fixture_path:
            continue
        normalized = dict(item)
        normalized["fixture_path"] = str(_resolve_project_path(fixture_path))
        result.append(normalized)
    return result


def _fixture_for_request(
    request: ContentFetchRequest,
    source_entry: dict,
    fixture_map: list[dict],
) -> dict | None:
    if not fixture_map:
        return None
    canonical_url = request.canonical_url or request.url
    domain = _domain_for_entry(source_entry)
    for item in fixture_map:
        if item.get("source_id") and item.get("source_id") == request.source_id:
            return item
        if item.get("canonical_url") and item.get("canonical_url") == canonical_url:
            return item
        if item.get("url") and item.get("url") == request.url:
            return item
        if item.get("domain") and _domain_matches(domain, [item.get("domain")]):
            return item
    return None


def _now_fixed_or_utc(live: bool) -> str:
    if not live:
        return _OFFLINE_FIXED_TIMESTAMP
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _url_scheme(url: str) -> str:
    if not url:
        return ""
    return urlsplit(url).scheme.lower()


def _is_search_endpoint(entry: dict, policy: ContentFetchPolicy) -> bool:
    if get_env("PIPELINE_MODE") == "evidence":
        from ..source_identity import _is_search_endpoint as identity_search_endpoint
        url = str(entry.get("canonical_url") or entry.get("url") or "")
        return identity_search_endpoint(entry, urlsplit(url).hostname)
    return entry.get("publisher") in policy.search_endpoint_publishers


def _domain_for_entry(entry: dict) -> str:
    url = entry.get("canonical_url") or entry.get("url") or ""
    if not url:
        return ""
    netloc = urlsplit(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _domain_matches(domain: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    normalized_domain = (domain or "").lower()
    if normalized_domain.startswith("www."):
        normalized_domain = normalized_domain[4:]
    for pattern in patterns:
        value = str(pattern or "").strip().lower()
        if not value:
            continue
        if value.startswith("www."):
            value = value[4:]
        if normalized_domain == value or normalized_domain.endswith("." + value):
            return True
    return False


def _domain_matches_reserved(domain: str, reserved_domains: list[str]) -> bool:
    if not domain:
        return False
    for reserved in reserved_domains:
        normalized = str(reserved or "").strip().lower()
        if not normalized:
            continue
        if domain == normalized or domain.endswith("." + normalized):
            return True
    return False


def _is_validation_reserved_entry(
    entry: dict,
    collection_mode: str,
    role_policy: dict | None = None,
) -> bool:
    if collection_mode != "masked_validation":
        return False
    flags = set(entry.get("routing_flags") or [])
    has_reserved_marker = (
        entry.get("source_role") == "validation_reserved"
        or entry.get("final_screening_decision") == "reserved_for_validation"
        or "validation_reserved" in flags
        or "blocked_from_collection" in flags
    )
    if has_reserved_marker:
        return True
    if role_policy is None:
        return False

    reserved_ids = set(role_policy.get("validation_reserved_source_ids") or [])
    if entry.get("source_id") in reserved_ids:
        return True

    if not role_policy.get("domain_masking_enabled", False):
        return False
    reserved_domains = list(role_policy.get("validation_reserved_domains") or [])
    return _domain_matches_reserved(_domain_for_entry(entry), reserved_domains)


def _context_only_source_ids(role_policy: dict | None) -> set[str]:
    if not role_policy:
        return set()
    return {str(source_id) for source_id in role_policy.get("context_only_source_ids") or []}


def _routing_flags(obj: dict) -> list[str]:
    flags = list(obj.get("routing_flags") or [])
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    for flag in metadata.get("routing_flags") or []:
        if flag not in flags:
            flags.append(flag)
    return flags


def _is_search_derived_entry(entry: dict) -> bool:
    return entry.get("discovery_method") in _SEARCH_DERIVED_DISCOVERY_METHODS


def _source_provenance(source_entry: dict) -> dict:
    return {
        "discovery_method": source_entry.get("discovery_method"),
        "search_provider": source_entry.get("search_provider"),
        "query_id": source_entry.get("query_id"),
        "query_used": source_entry.get("query_used"),
        "planned_query_id": source_entry.get("planned_query_id"),
        "provider_channel": source_entry.get("provider_channel"),
        "role_hint": source_entry.get("role_hint"),
        "retrieved_at": source_entry.get("retrieved_at"),
        "source_role_final": source_entry.get("source_role_final"),
        "credibility_score": source_entry.get("credibility_score"),
        "credibility_level": source_entry.get("credibility_level"),
        "source_credibility_risk_flags": list(
            source_entry.get("risk_flags")
            or source_entry.get("credibility_flags")
            or []
        ),
        "actual_publisher": source_entry.get("actual_publisher"),
        "actual_publisher_normalized": source_entry.get(
            "actual_publisher_normalized"
        ),
        "source_type_final": source_entry.get("source_type_final"),
        "source_independence_group": source_entry.get(
            "source_independence_group"
        ),
        "claim_support_role": source_entry.get("claim_support_role"),
        "recommended_source_role": source_entry.get("recommended_source_role"),
        "recommended_fetch_use": source_entry.get("recommended_fetch_use"),
        "recommended_extraction_use": source_entry.get(
            "recommended_extraction_use"
        ),
        "likely_syndicated_or_aggregated": source_entry.get(
            "likely_syndicated_or_aggregated"
        ),
        "upstream_source_mentions": list(
            source_entry.get("upstream_source_mentions") or []
        ),
        "search_rank": source_entry.get("search_rank"),
        "search_result_id": source_entry.get("search_result_id"),
        "result_source": source_entry.get("result_source"),
        "query_type": source_entry.get("query_type"),
        "source_disease_relevance_status": source_entry.get(
            "source_disease_relevance_status"
        ),
        "source_disease_relevance_score": source_entry.get(
            "source_disease_relevance_score"
        ),
        "source_target_disease_terms_found": list(
            source_entry.get("source_target_disease_terms_found") or []
        ),
        "source_incompatible_disease_terms_found": list(
            source_entry.get("source_incompatible_disease_terms_found") or []
        ),
        "source_disease_relevance_reason": source_entry.get(
            "source_disease_relevance_reason"
        ),
        "source_disease_relevance_data_signal_count": source_entry.get(
            "source_disease_relevance_data_signal_count"
        ),
        "reporting_period_start": source_entry.get("reporting_period_start"),
        "reporting_period_end": source_entry.get("reporting_period_end"),
        "reporting_period_label": source_entry.get("reporting_period_label"),
        "period_basis": source_entry.get("period_basis"),
    }


def _is_context_only_entry(entry: dict, role_policy: dict | None = None) -> bool:
    source_id = entry.get("source_id")
    if source_id in _context_only_source_ids(role_policy):
        return True
    if str(entry.get("source_role_final") or "").strip().lower() in {
        "context",
        "context_only",
        "validation_only",
    }:
        return True
    flags = set(_routing_flags(entry))
    if "context_only" in flags or "blocked_from_structured_extraction" in flags:
        return True
    return (
        entry.get("source_role") in {"context_source", "context_only"}
        and entry.get("final_screening_decision") == "include_for_context_fetch"
    )


def _source_metadata(source_entry: dict) -> dict:
    return {
        "source_registry_status": source_entry.get("status"),
        "expected_fields": list(source_entry.get("expected_fields") or []),
        "matched_terms": list(source_entry.get("matched_terms") or []),
        "source_purpose": source_entry.get("source_purpose"),
        "notes": source_entry.get("notes"),
        "routing_flags": list(source_entry.get("routing_flags") or []),
        "requires_human_review": source_entry.get("requires_human_review"),
        "human_review_recommended": source_entry.get("human_review_recommended"),
        **_source_provenance(source_entry),
    }


def _source_type_for_entry(entry: dict) -> str:
    return str(
        entry.get("source_type_final")
        or entry.get("source_type")
        or entry.get("planned_query_source_type")
        or ""
    ).strip().lower()


def _registry_source_type_for_domain(domain: str) -> str:
    try:
        registry_entry = lookup_source_identity_registry(domain)
    except Exception:
        registry_entry = None
    return str((registry_entry or {}).get("source_type") or "").strip().lower()


def _entry_has_public_authority_identity(entry: dict) -> bool:
    domain = _domain_for_entry(entry)
    publisher = str(
        entry.get("publisher")
        or entry.get("actual_publisher")
        or entry.get("page_publisher_candidate")
        or ""
    ).lower()
    registry_source_type = _registry_source_type_for_domain(domain)
    if registry_source_type in _OFFICIAL_SOURCE_TYPES:
        return True
    if any(
        marker in domain
        for marker in (
            ".gov",
            ".gouv",
            ".gob",
            ".go.",
            ".gov.",
            ".gc.ca",
        )
    ):
        return True
    if domain in {
        "who.int",
        "ecdc.europa.eu",
        "paho.org",
        "rivm.nl",
        "canada.ca",
        "gov.uk",
        "nhs.uk",
    }:
        return True
    if any(
        term in publisher
        for term in (
            "department of health",
            "ministry of health",
            "public health agency",
            "public health institute",
            "health protection agency",
            "centers for disease control",
            "centre for disease control",
            "national institute",
            "world health organization",
            "pan american health organization",
            "european centre for disease prevention",
        )
    ):
        return True
    return False


def _fetch_bucket(entry: dict) -> str:
    role = str(entry.get("source_role_final") or "").strip().lower()
    source_type = _source_type_for_entry(entry)
    url = str(entry.get("canonical_url") or entry.get("url") or "").lower()
    publisher = str(entry.get("publisher") or entry.get("actual_publisher") or "").lower()
    flags = {str(flag).lower() for flag in entry.get("risk_flags") or []}
    domain = _domain_for_entry(entry)
    has_public_authority_identity = _entry_has_public_authority_identity(entry)
    if role == "validation":
        return "validation"
    if (
        source_type in {"social_media", "forum", "community_forum"}
        or any(
            host in domain
            for host in (
                "instagram.com",
                "facebook.com",
                "flutrackers.com",
                "x.com",
                "twitter.com",
                "tiktok.com",
                "reddit.com",
            )
        )
        or "/forum" in url
    ):
        return "forum_social"
    if "cdc.gov" in domain or publisher in {"cdc", "centers for disease control and prevention"}:
        return "national_context"
    if (
        source_type in _NEWS_SOURCE_TYPES
        and ".gov/" not in url
        and "cdc" not in publisher
        and "department of health" not in publisher
        and "ministry of health" not in publisher
    ):
        return "news_or_situation_report"
    if (
        has_public_authority_identity
        or url.endswith(".gov")
        or ".gov/" in url
        or "cdc" in publisher
        or "department of health" in publisher
        or "ministry of health" in publisher
    ):
        return "official_authority"
    path = urlsplit(url).path.lower()
    if (
        url.split("?")[0].endswith(".pdf")
        or "health-alert-network" in path
        or re.search(r"(^|/)(han)(/|[-_])", path)
    ):
        return "official_pdf_or_report"
    if source_type in _STRUCTURED_SOURCE_TYPES or "arcgis" in url:
        return "structured_database"
    if source_type in _LITERATURE_SOURCE_TYPES or "pubmed" in url:
        return "peer_reviewed_literature"
    if role in {"context", "collection_support"}:
        return role
    return "other"


def _fetch_bucket_rank(bucket: str) -> int:
    order = {
        "target_official_authority": 0,
        "official_authority": 1,
        "validation": 2,
        "official_pdf_or_report": 3,
        "structured_database": 4,
        "peer_reviewed_literature": 5,
        "national_context": 6,
        "news_or_situation_report": 7,
        "collection_support": 8,
        "context": 9,
        "forum_social": 10,
        "social_media": 10,
        "other": 11,
    }
    return order.get(bucket, 99)


def _direct_target_official_fast_path_active(
    registry: list[dict],
    *,
    collection_mode: str,
    allowlist: set[str] | None,
) -> bool:
    if get_env("PIPELINE_MODE") == "evidence":
        # Search metadata cannot prove usable target evidence or justify dropping
        # the remaining candidates before any page content has been acquired.
        return False
    if collection_mode != "direct_collection" or allowlist is not None:
        return False
    return any(_is_search_verified_target_entry(entry) for entry in registry)


def _is_search_verified_target_entry(entry: dict) -> bool:
    if not _is_search_derived_entry(entry):
        return False
    if _is_context_only_entry(entry, None):
        return False
    return str(entry.get("target_fit_status") or "").strip().lower() in {
        "verified_target",
        "verified_target_collection",
    } or str(entry.get("triage_role") or "").strip().lower() in {
        "verified_target_collection",
    } or str(entry.get("target_verification_status") or "").strip().lower() in {
        "verified",
        "search_verified",
    }


def _fit_value(entry: dict, key: str) -> str:
    return str(entry.get(key) or "").strip().lower()


def _entry_source_identity_requires_task_review(entry: dict) -> bool:
    source_types = {
        str(entry.get(key) or "").strip().lower()
        for key in (
            "source_type_final",
            "source_type",
            "planned_query_source_type",
            "identity_source_type",
            "original_source_type",
        )
        if str(entry.get(key) or "").strip()
    }
    if source_types & _TASK_USABILITY_REVIEW_SOURCE_TYPES:
        return True
    domain = _domain_for_entry(entry)
    if any(
        marker in domain
        for marker in (
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
        )
    ):
        return True
    return False


def _entry_has_non_exact_task_fit(entry: dict) -> str | None:
    disease_fit = _fit_value(entry, "disease_fit")
    if disease_fit in _NON_EXACT_DISEASE_FIT_VALUES:
        return "disease_not_exact_for_task"
    geography_fit = _fit_value(entry, "geography_fit")
    if geography_fit in _NON_EXACT_GEOGRAPHY_FIT_VALUES:
        return "geography_not_exact_for_task"
    date_fit = _fit_value(entry, "date_fit")
    if date_fit in _NON_EXACT_DATE_FIT_VALUES:
        return "period_not_exact_for_task"
    target_fit = _fit_value(entry, "target_fit_status")
    if target_fit in {
        "broader_than_task",
        "context_only",
        "geography_mismatch",
        "non_target_or_context",
        "partial_overlap",
        "temporal_mismatch",
        "wrong_period",
    }:
        if "geo" in target_fit or target_fit == "broader_than_task":
            return "geography_not_exact_for_task"
        return "period_not_exact_for_task"
    return None


def _is_task_record_collection_candidate_entry(entry: dict) -> bool:
    if not _is_search_derived_entry(entry) and str(
        entry.get("discovery_method") or ""
    ) != "fallback_link_discovery":
        return False
    target_status = str(entry.get("target_fit_status") or "").strip().lower()
    triage_role = str(entry.get("triage_role") or "").strip().lower()
    verification_status = str(
        entry.get("target_verification_status") or ""
    ).strip().lower()
    final_role = str(entry.get("source_role_final") or "").strip().lower()
    if final_role in {
        "context",
        "context_only",
        "validation",
        "validation_only",
        "excluded",
        "search_endpoint",
        "needs_human_review",
    }:
        return False
    if _entry_source_identity_requires_task_review(entry):
        return False
    if _fetch_bucket(entry) in {"forum_social", "news_or_situation_report", "peer_reviewed_literature"}:
        return False
    if _entry_has_non_exact_task_fit(entry):
        return False
    if target_status in {
        "context_only",
        "non_target_or_context",
        "temporal_mismatch",
        "geography_mismatch",
        "excluded",
    }:
        return False
    return (
        target_status
        in {
            "task_record_collection_candidate",
            "fetch_verified_target_collection",
            "verified_target_collection",
            "verified_target",
        }
        or triage_role
        in {
            "task_record_collection_candidate",
            "fetch_verified_target_collection",
            "verified_target_collection",
        }
        or verification_status
        in {
            "candidate_task_record_source",
            "fetch_verified",
            "search_verified",
            "verified",
        }
    )


def _document_attr(document, key: str):
    if isinstance(document, dict):
        return document.get(key)
    return getattr(document, key, None)


def _document_looks_like_error_page(document) -> bool:
    from ..page_status import page_failure_reason
    return page_failure_reason(document) is not None


def _task_location_for_geography_check(state: dict | None) -> str:
    if not isinstance(state, dict):
        return ""
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    return str(
        structured.get("location")
        or structured.get("geography")
        or spec.get("geography")
        or spec.get("location")
        or ""
    ).strip()


def _task_location_is_subnational(location: str) -> bool:
    value = str(location or "").strip().lower()
    if not value:
        return False
    national_terms = {
        "global",
        "world",
        "worldwide",
        "international",
        "united states",
        "united states of america",
        "usa",
        "u.s.",
        "us",
        "america",
    }
    return value not in national_terms


def _source_or_document_mentions_task_location(
    document,
    source_entry: dict,
    location: str,
) -> bool:
    location = str(location or "").strip().lower()
    if not location:
        return False
    haystack = " ".join(
        str(value or "")
        for value in (
            source_entry.get("title"),
            source_entry.get("publisher"),
            source_entry.get("canonical_url"),
            source_entry.get("url"),
            source_entry.get("query_used"),
            source_entry.get("target_verification_reason"),
            _document_attr(document, "title"),
            _document_attr(document, "canonical_url"),
            _document_attr(document, "url"),
            _document_attr(document, "clean_text"),
        )
    ).lower()
    return location in haystack


def _task_data_link_signal_terms(state: dict | None, required_ids: set[str]) -> set[str]:
    """Build generic task/data terms for one-hop fallback link discovery."""

    terms: set[str] = set()
    if isinstance(state, dict):
        structured = state.get("structured_task") or {}
        spec = state.get("collection_spec") or {}
        disease = str(
            structured.get("disease")
            or structured.get("virus")
            or spec.get("disease")
            or spec.get("virus")
            or ""
        ).strip()
        location = _task_location_for_geography_check(state)
        for value in (disease, location, structured.get("start_date"), structured.get("end_date")):
            if value:
                terms.add(str(value).strip().lower())
        if disease.lower() == "flu":
            terms.update({"influenza", "seasonal influenza", "ili"})
    for requirement_id in required_ids:
        for token in re.split(r"[^a-zA-Z0-9]+", requirement_id):
            if len(token) >= 4 or token.isdigit():
                terms.add(token.lower())
    terms.update(
        {
            "data",
            "dashboard",
            "archive",
            "api",
            "csv",
            "table",
            "surveillance",
            "report",
            "weekly",
            "respiratory",
        }
    )
    return {term for term in terms if term}


def _score_task_data_link(
    *,
    url: str,
    page_text: str,
    parent_url: str,
    state: dict | None,
    required_ids: set[str],
) -> tuple[int, list[str]]:
    haystack = f"{url} {page_text}".lower()
    score = 0
    reasons: list[str] = []
    task_location = _task_location_for_geography_check(state)
    if task_location and task_location.lower() in haystack:
        score += 3
        reasons.append("target_location_signal")
    task_terms = _task_data_link_signal_terms(state, required_ids)
    matched_terms = sorted(term for term in task_terms if term in haystack)
    if matched_terms:
        score += min(6, len(matched_terms))
        reasons.append("task_terms:" + ",".join(matched_terms[:6]))
    data_terms = {
        "data",
        "dashboard",
        "archive",
        "api",
        "csv",
        "table",
        "surveillance",
        "report",
        "weekly",
        "metric",
    }
    if any(term in haystack for term in data_terms):
        score += 3
        reasons.append("data_source_signal")
    bad_terms = {"facebook", "instagram", "youtube", "flu-trackers", "forum", "press-release"}
    if any(term in haystack for term in bad_terms):
        score -= 5
        reasons.append("low_collection_signal")
    parent_domain = urlsplit(parent_url).netloc.lower()
    child_domain = urlsplit(url).netloc.lower()
    if parent_domain and child_domain == parent_domain:
        score += 1
        reasons.append("same_domain")
    return score, reasons


def _extract_task_data_link_candidates(
    document,
    source_entry: dict,
    *,
    state: dict | None,
    required_ids: set[str],
) -> list[dict]:
    parent_url = str(
        _document_attr(document, "canonical_url")
        or _document_attr(document, "url")
        or source_entry.get("canonical_url")
        or source_entry.get("url")
        or ""
    )
    metadata = _document_attr(document, "metadata") or {}
    haystack = "\n".join(
        str(value or "")
        for value in (
            _document_attr(document, "title"),
            _document_attr(document, "clean_text"),
            metadata.get("raw_html") if isinstance(metadata, dict) else "",
            metadata.get("raw_text") if isinstance(metadata, dict) else "",
        )
    )
    candidates: list[str] = []
    for link in metadata.get("outbound_links") or []:
        if isinstance(link, dict) and link.get("href"):
            candidates.append(urljoin(parent_url, str(link.get("href")).strip()))
    for match in re.findall(r"""href=["']([^"']+)["']""", haystack, flags=re.I):
        candidates.append(urljoin(parent_url, match.strip()))
    for match in re.findall(r"https?://[^\s<>'\")]+", haystack):
        candidates.append(match.strip().rstrip(".,;:)]}"))

    seen: set[str] = set()
    ranked: list[dict] = []
    for candidate_url in candidates:
        if not candidate_url or candidate_url in seen:
            continue
        seen.add(candidate_url)
        if urlsplit(candidate_url).scheme not in {"http", "https"}:
            continue
        score, reasons = _score_task_data_link(
            url=candidate_url,
            page_text=haystack,
            parent_url=parent_url,
            state=state,
            required_ids=required_ids,
        )
        if score < 5:
            continue
        ranked.append(
            {
                "url": candidate_url,
                "score": score,
                "selection_reasons": reasons,
            }
        )
    ranked.sort(key=lambda row: (-int(row.get("score") or 0), row.get("url") or ""))
    return ranked


_EVENT_LINK_PRODUCT_TERMS = {
    "outbreak",
    "update",
    "risk assessment",
    "rapid risk assessment",
    "situation report",
    "case report",
    "technical note",
    "technical report",
    "surveillance report",
    "sequence",
    "genomic",
    "line list",
    "pdf",
}

_EVENT_LINK_EXCLUDED_TERMS = {
    "about us",
    "contact us",
    "privacy",
    "cookie",
    "facebook",
    "instagram",
    "linkedin",
    "share this",
    "subscribe",
    "sign in",
    "login",
    "careers",
    "donate",
}


def _extract_event_link_candidates(
    document,
    source_entry: dict,
    *,
    state: dict | None,
) -> list[dict]:
    """Rank one-hop report links from a verified event page."""

    parent_url = str(
        _document_attr(document, "canonical_url")
        or _document_attr(document, "url")
        or source_entry.get("canonical_url")
        or source_entry.get("url")
        or ""
    )
    metadata = _document_attr(document, "metadata") or {}
    raw_html = str(metadata.get("raw_html") or "") if isinstance(metadata, dict) else ""
    if not raw_html:
        raw_html = str(_document_attr(document, "clean_text") or "")
    links: list[tuple[str, str]] = []
    for link in metadata.get("outbound_links") or []:
        if not isinstance(link, dict) or not link.get("href"):
            continue
        links.append(
            (
                urljoin(parent_url, str(link.get("href")).strip()),
                str(link.get("text") or ""),
            )
        )
    for match in re.finditer(
        r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        raw_html,
        flags=re.I | re.S,
    ):
        label = re.sub(r"<[^>]+>", " ", match.group(2))
        links.append((urljoin(parent_url, match.group(1).strip()), label))
    for match in re.findall(r"https?://[^\s<>'\")]+", raw_html):
        links.append((match.strip().rstrip(".,;:)]}"), ""))

    structured = (state or {}).get("structured_task") or {}
    spec = (state or {}).get("collection_spec") or {}
    disease = str(
        structured.get("disease")
        or structured.get("virus")
        or spec.get("disease")
        or spec.get("virus")
        or ""
    ).strip().lower()
    disease_terms = {disease} if disease else set()
    if disease in {"flu", "influenza"}:
        disease_terms.update({"influenza", "seasonal influenza", "ili"})
    intelligence = (state or {}).get("disease_intelligence") or {}
    for value in (
        intelligence.get("aliases")
        or intelligence.get("disease_aliases")
        or intelligence.get("search_aliases")
        or []
    ):
        if str(value or "").strip():
            disease_terms.add(str(value).strip().lower())
    task_years = {
        str(value)[:4]
        for value in (structured.get("start_date"), structured.get("end_date"))
        if str(value or "")[:4].isdigit()
    }
    parent_domain = urlsplit(parent_url).netloc.lower()
    seen: set[str] = set()
    ranked: list[dict] = []
    for candidate_url, label in links:
        if not candidate_url or candidate_url in seen:
            continue
        seen.add(candidate_url)
        parsed = urlsplit(candidate_url)
        if parsed.scheme not in {"http", "https"}:
            continue
        haystack = f"{candidate_url} {label}".lower()
        if any(term in haystack for term in _EVENT_LINK_EXCLUDED_TERMS):
            continue
        product_terms = sorted(
            term for term in _EVENT_LINK_PRODUCT_TERMS if term in haystack
        )
        if not product_terms:
            continue
        matched_disease_terms = sorted(
            term for term in disease_terms if term and term in haystack
        )
        child_domain = parsed.netloc.lower()
        same_domain = bool(parent_domain and child_domain == parent_domain)
        score = 4 + min(3, len(product_terms))
        reasons = ["event_product_terms:" + ",".join(product_terms[:4])]
        if matched_disease_terms:
            score += 4
            reasons.append(
                "target_disease_terms:" + ",".join(matched_disease_terms[:4])
            )
        if any(year in haystack for year in task_years):
            score += 1
            reasons.append("task_year_signal")
        if same_domain:
            score += 2
            reasons.append("same_authority_domain")
        elif not matched_disease_terms:
            continue
        ranked.append(
            {
                "url": candidate_url,
                "score": score,
                "selection_reasons": reasons,
                "same_domain": same_domain,
                "link_text": " ".join(label.split()),
            }
        )
    ranked.sort(key=lambda row: (-int(row.get("score") or 0), row.get("url") or ""))
    return ranked


def _parse_task_date(value) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) >= 10:
        text = text[:10]
    if len(text) == 4 and text.isdigit():
        text = f"{text}-01-01"
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def _task_period_for_document_usability(
    state: dict | None,
) -> tuple[datetime | None, datetime | None, str]:
    if not isinstance(state, dict):
        return None, None, ""
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    contract = state.get("task_evidence_contract") or {}
    start = _parse_task_date(structured.get("start_date") or spec.get("start_date"))
    end = _parse_task_date(structured.get("end_date") or spec.get("end_date")) or start
    granularity = str(contract.get("time_granularity") or "").strip().lower()
    if not granularity:
        requirements = contract.get("requirements")
        if isinstance(requirements, list) and requirements:
            bases = {
                str(req.get("period_basis") or "").strip().lower()
                for req in requirements
                if isinstance(req, dict)
            }
            if bases == {"annual"}:
                granularity = "annual"
            elif bases == {"week_ending_saturday"}:
                granularity = "weekly"
    if not granularity:
        granularity = "task_window"
    if start and end and end < start:
        start, end = end, start
    return start, end, granularity


def _period_granularity_for_requirement(requirement: dict) -> str:
    basis = str(requirement.get("period_basis") or "").strip().lower()
    if basis == "annual":
        return "annual"
    if basis == "week_ending_saturday":
        return "weekly"
    return basis or "task_window"


def _candidate_periods_for_document_usability(
    state: dict | None,
    source_entry: dict,
) -> list[tuple[datetime, datetime, str]]:
    periods: list[tuple[datetime, datetime, str]] = []
    if isinstance(state, dict):
        requirement_ids = {
            str(value)
            for value in (source_entry.get("coverage_requirement_ids") or [])
            if value
        }
        requirements = (
            state.get("source_coverage_requirements")
            or (state.get("task_evidence_contract") or {}).get("requirements")
            or []
        )
        if not requirements:
            try:
                requirements = build_source_coverage_requirements(state)
            except Exception:
                requirements = []
        for requirement in requirements:
            if not isinstance(requirement, dict):
                continue
            req_id = str(requirement.get("requirement_id") or requirement.get("id") or "")
            if requirement_ids and req_id not in requirement_ids:
                continue
            start = _parse_task_date(
                requirement.get("period_start")
                or requirement.get("reporting_period_start")
                or requirement.get("start_date")
            )
            end = _parse_task_date(
                requirement.get("period_end")
                or requirement.get("reporting_period_end")
                or requirement.get("end_date")
            ) or start
            if start and end:
                if end < start:
                    start, end = end, start
                periods.append((start, end, _period_granularity_for_requirement(requirement)))
    if periods:
        return periods
    task_start, task_end, granularity = _task_period_for_document_usability(state)
    if task_start and task_end:
        return [(task_start, task_end, granularity)]
    return []


def _period_dates_from_source_or_document(
    document,
    source_entry: dict,
) -> tuple[datetime | None, datetime | None]:
    start = _parse_task_date(
        _document_attr(document, "reporting_period_start")
        or _document_attr(document, "metric_period_start")
        or source_entry.get("reporting_period_start")
        or source_entry.get("period_start")
    )
    end = _parse_task_date(
        _document_attr(document, "reporting_period_end")
        or _document_attr(document, "metric_period_end")
        or source_entry.get("reporting_period_end")
        or source_entry.get("period_end")
    )
    if start and end and end < start:
        start, end = end, start
    return start, end


def _date_phrase_tokens(value: datetime | None) -> set[str]:
    if not value:
        return set()
    month = value.strftime("%B").lower()
    month_abbr = value.strftime("%b").lower()
    day = str(value.day)
    year = str(value.year)
    return {
        value.date().isoformat(),
        f"{month} {day}, {year}",
        f"{month} {day} {year}",
        f"{month} {day}",
        f"{month_abbr} {day}, {year}",
        f"{month_abbr} {day} {year}",
        f"{month_abbr} {day}",
    }


def _task_week_tokens(task_start: datetime, task_end: datetime) -> set[str]:
    tokens: set[str] = set()
    for anchor in (task_start, task_end):
        try:
            iso = anchor.date().isocalendar()
        except (TypeError, ValueError):
            continue
        week = int(iso.week)
        year = int(iso.year)
        tokens.update(
            {
                f"week {week}",
                f"week-{week}",
                f"week_{week}",
                f"week {week:02d}",
                f"week-{week:02d}",
                f"week_{week:02d}",
                f"mmwr week {week}",
                f"mmwr week {week:02d}",
                f"week {week}, {year}",
                f"week {week:02d}, {year}",
            }
        )
    return tokens


_BROAD_PERIOD_DOCUMENT_MARKERS = (
    "annual report",
    "end of season",
    "end-of-season",
    "season report",
    "seasonal report",
    "season-to-date",
    "season to date",
    "year-to-date",
    "year to date",
    "full year",
)

_ANNUAL_EXACT_DOCUMENT_MARKERS = (
    "annual report",
    "annual surveillance",
    "full year",
    "full-year",
    "calendar year",
    "year-end",
    "year end",
)

_WEEKLY_PERIOD_DOCUMENT_MARKERS = (
    "weekly",
    "week ",
    "week-",
    "week_",
    "mmwr week",
    "current week",
    "previous week",
)
_NON_COLLECTION_PAGE_MARKERS = (
    "training",
    "course",
    "education",
    "contact",
    "faq",
    "frequently asked questions",
    "about ",
    "symptoms",
    "prevention",
    "what is",
    "resources",
)
_COLLECTION_DATA_PAGE_MARKERS = (
    "surveillance",
    "report",
    "dashboard",
    "dataset",
    "data set",
    "open data",
    "statistics",
    "table",
    "csv",
    "api",
    "case count",
    "cases reported",
    "incidence",
    "mortality",
    "death",
    "hospitalization",
    "laboratory",
    "positive",
    "positivity",
    "outbreak",
    "coverage",
)


def _document_identity_has_broad_period_marker(document, source_entry: dict) -> bool:
    text = " ".join(
        str(_document_attr(document, key) or source_entry.get(key) or "")
        for key in (
            "title",
            "source_title",
            "canonical_url",
            "url",
            "reporting_period_label",
        )
    ).lower()
    return any(marker in text for marker in _BROAD_PERIOD_DOCUMENT_MARKERS)


def _document_has_task_data_signal(document, source_entry: dict) -> bool:
    identity_text = " ".join(
        str(_document_attr(document, key) or source_entry.get(key) or "")
        for key in (
            "title",
            "source_title",
            "canonical_url",
            "url",
        )
    ).lower()
    body_text = " ".join(
        str(_document_attr(document, key) or "")
        for key in ("clean_text", "raw_text", "text", "excerpt")
    ).lower()
    if any(marker in identity_text for marker in _NON_COLLECTION_PAGE_MARKERS):
        identity_has_data_marker = any(
            marker in identity_text for marker in _COLLECTION_DATA_PAGE_MARKERS
        )
        if not identity_has_data_marker:
            return False
    haystack = f"{identity_text} {body_text}"
    return any(marker in haystack for marker in _COLLECTION_DATA_PAGE_MARKERS)


def _document_period_matches_task(
    document,
    source_entry: dict,
    state: dict | None,
) -> tuple[bool, str | None]:
    candidate_periods = _candidate_periods_for_document_usability(state, source_entry)
    if not candidate_periods:
        return True, None
    if str(source_entry.get("date_fit") or "").strip().lower() == "match":
        return True, None

    haystack = " ".join(
        str(_document_attr(document, key) or source_entry.get(key) or "")
        for key in (
            "title",
            "clean_text",
            "raw_text",
            "text",
            "excerpt",
            "source_title",
        )
    ).lower()

    source_start, source_end = _period_dates_from_source_or_document(document, source_entry)
    for task_start, task_end, granularity in candidate_periods:
        if granularity in {"annual", "year", "yearly"}:
            has_annual_marker = any(
                marker in haystack for marker in _ANNUAL_EXACT_DOCUMENT_MARKERS
            )
            has_weekly_marker = any(
                marker in haystack for marker in _WEEKLY_PERIOD_DOCUMENT_MARKERS
            )
            if has_weekly_marker and not has_annual_marker:
                continue
        elif (
            _document_identity_has_broad_period_marker(document, source_entry)
            or any(
                marker in haystack
                for marker in (
                    *_BROAD_PERIOD_DOCUMENT_MARKERS,
                    *_ANNUAL_EXACT_DOCUMENT_MARKERS,
                )
            )
        ):
            continue
        if source_start and source_end and (
            source_start.date() == task_start.date()
            and source_end.date() == task_end.date()
        ):
            return True, None
        if (
            granularity not in {"annual", "year", "yearly"}
            and _document_identity_has_broad_period_marker(document, source_entry)
            and not (
                source_start
                and source_end
                and source_start.date() == task_start.date()
                and source_end.date() == task_end.date()
            )
        ):
            continue
        start_tokens = _date_phrase_tokens(task_start)
        end_tokens = _date_phrase_tokens(task_end)
        if source_start and source_end and (
            source_start.date() != task_start.date()
            or source_end.date() != task_end.date()
        ):
            continue
        if granularity in {"annual", "year", "yearly"}:
            has_annual_marker = any(
                marker in haystack for marker in _ANNUAL_EXACT_DOCUMENT_MARKERS
            )
            has_weekly_marker = any(
                marker in haystack for marker in _WEEKLY_PERIOD_DOCUMENT_MARKERS
            )
            if has_weekly_marker and not has_annual_marker:
                continue
            if (
                str(task_start.year) in haystack
                and task_start.month == 1
                and task_start.day == 1
                and task_end.month == 12
                and task_end.day == 31
                and has_annual_marker
            ):
                return True, None
            continue
        if (
            task_start.date().isoformat() in haystack
            or task_end.date().isoformat() in haystack
            or (
                any(token in haystack for token in start_tokens)
                and any(token in haystack for token in end_tokens)
            )
            or (
                any(token in haystack for token in _task_week_tokens(task_start, task_end))
                and str(task_end.year) in haystack
            )
        ):
            return True, None
    return False, "period_not_exact_for_task"


def _document_task_usability(
    document,
    source_entry: dict,
    state: dict | None = None,
) -> tuple[bool, list[str]]:
    if _document_looks_like_error_page(document):
        return False, ["error_page_detected"]
    if str(_document_attr(document, "fetch_status") or "") == "fetch_failed":
        return False, ["fetch_failed"]
    parse_status = str(_document_attr(document, "parse_status") or "").lower()
    if not parse_status.startswith("parsed") and parse_status != "fixture_loaded":
        return False, ["parse_not_successful"]
    if str(_document_attr(document, "quality_status") or "").lower() == "unusable":
        return False, ["quality_unusable"]
    if len(str(_document_attr(document, "clean_text") or "")) < 80:
        return False, ["text_too_short"]
    final_role = str(source_entry.get("source_role_final") or "").strip().lower()
    if (
        final_role == "needs_human_review"
        or source_entry.get("requires_human_review") is True
        or source_entry.get("human_review_recommended") is True
    ):
        return False, ["source_trust_requires_review"]
    if final_role in {"excluded", "search_endpoint", "validation", "validation_only"}:
        return False, ["source_role_not_task_collection"]
    if _is_context_only_entry(source_entry, None):
        return False, ["context_only_source"]
    source_type = _source_type_for_entry(source_entry)
    if (
        source_type in _TASK_USABILITY_REVIEW_SOURCE_TYPES
        or _entry_source_identity_requires_task_review(source_entry)
        or _fetch_bucket(source_entry)
        in {"forum_social", "news_or_situation_report", "peer_reviewed_literature"}
    ):
        return False, ["source_trust_requires_review"]
    fit_reason = _entry_has_non_exact_task_fit(source_entry)
    if fit_reason:
        return False, [fit_reason]
    task_location = _task_location_for_geography_check(state)
    if (
        _task_location_is_subnational(task_location)
        and _fetch_bucket(source_entry) == "national_context"
        and not _source_or_document_mentions_task_location(
            document,
            source_entry,
            task_location,
        )
    ):
        return False, ["geography_not_exact_for_task"]
    if not bool(
        source_entry.get("must_fetch")
        or _is_search_verified_target_entry(source_entry)
        or _is_task_record_collection_candidate_entry(source_entry)
    ):
        return False, ["source_not_task_candidate"]
    if not _document_has_task_data_signal(document, source_entry):
        return False, ["document_lacks_task_data_signal"]
    period_ok, period_reason = _document_period_matches_task(document, source_entry, state)
    if not period_ok:
        return False, [period_reason or "period_not_exact_for_task"]
    return True, ["exact_task_document"]


def _document_is_usable_for_task_collection(
    document,
    source_entry: dict,
    state: dict | None = None,
) -> bool:
    usable, _ = _document_task_usability(document, source_entry, state)
    return usable


def _direct_fast_path_skip_reason(
    entry: dict,
    *,
    fast_path_active: bool,
) -> str | None:
    if not fast_path_active or entry.get("must_fetch"):
        return None
    if _is_search_verified_target_entry(entry):
        return None
    if _is_task_record_collection_candidate_entry(entry):
        return "direct_target_official_fast_path_deferred_task_candidate"
    if _is_search_derived_entry(entry):
        return "direct_target_official_fast_path"
    bucket = _fetch_bucket(entry)
    if bucket in {
        "national_context",
        "news_or_situation_report",
        "forum_social",
        "social_media",
        "context",
        "collection_support",
        "other",
    }:
        return "direct_target_official_fast_path"
    return None


def _fetch_selection_sort_key(entry: dict) -> tuple:
    bucket = _fetch_bucket(entry)
    priority = entry.get("priority")
    try:
        priority_value = int(priority)
    except (TypeError, ValueError):
        priority_value = 999
    rank = entry.get("search_rank")
    try:
        search_rank = int(rank)
    except (TypeError, ValueError):
        search_rank = 999
    score = entry.get("credibility_score")
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    target_status = str(entry.get("target_fit_status") or "").strip().lower()
    triage_role = str(entry.get("triage_role") or "").strip().lower()
    discovery_method = str(entry.get("discovery_method") or "").strip().lower()
    if discovery_method == "official_coverage_requirement" or (
        target_status == "predicted_target_candidate"
        or triage_role == "predicted_target_candidate"
    ):
        target_rank = 0
    elif target_status == "verified_target_collection" or (
        triage_role == "verified_target_collection"
    ):
        target_rank = 1
    else:
        target_rank = 2
    high_trust_fetch_rank = (
        0
        if bucket
        in {
            "target_official_authority",
            "official_authority",
            "validation",
            "official_pdf_or_report",
            "structured_database",
            "peer_reviewed_literature",
            "national_context",
        }
        else 1
    )
    return (
        target_rank,
        high_trust_fetch_rank,
        source_product_priority(entry),
        _fetch_bucket_rank(bucket),
        priority_value,
        -score_value,
        search_rank,
        str(entry.get("source_id") or ""),
    )


def _selection_manifest_product_fields(entry: dict) -> dict:
    return {
        "data_product_type": entry.get("data_product_type"),
        "task_specificity": entry.get("task_specificity"),
        "time_window_fit": entry.get("time_window_fit"),
        "machine_readability": entry.get("machine_readability"),
        "expected_evidence_role": entry.get("expected_evidence_role"),
        "source_product_skip_reason": source_product_skip_reason(entry),
    }


def _is_retry_high_trust_fetch_candidate(entry: dict, bucket: str | None = None) -> bool:
    if str(entry.get("query_source") or "").strip().lower() != "authority_gap_retry":
        return False
    bucket = bucket or _fetch_bucket(entry)
    return bucket in {
        "target_official_authority",
        "official_authority",
        "official_pdf_or_report",
        "structured_database",
        "peer_reviewed_literature",
        "national_context",
        "validation",
    }


_AUTHORITY_RETRY_EVENT_TERMS = {
    "advisory",
    "alert",
    "briefing",
    "brote",
    "case",
    "cases",
    "casos",
    "cluster",
    "confirmed",
    "death",
    "deaths",
    "epidemiological",
    "epidemiologico",
    "exposed",
    "exposure",
    "infection",
    "infections",
    "outbreak",
    "report",
    "response",
    "situation",
    "surveillance",
    "update",
}


def _is_authority_retry_event_fetch_candidate(
    entry: dict,
    bucket: str | None = None,
) -> bool:
    if not _is_retry_high_trust_fetch_candidate(entry, bucket):
        return False
    text = " ".join(
        str(entry.get(key) or "")
        for key in (
            "title",
            "snippet",
            "canonical_url",
            "url",
            "query_used",
            "publisher",
            "actual_publisher",
        )
    ).lower()
    return any(term in text for term in _AUTHORITY_RETRY_EVENT_TERMS)


def _is_verified_event_source_fetch_rescue(entry: dict) -> bool:
    """Allow a high-utility event source to survive an overbroad excluded role."""

    product_type = str(entry.get("data_product_type") or "").strip()
    task_specificity = str(entry.get("task_specificity") or "").strip()
    readability = str(entry.get("machine_readability") or "").strip()
    role = str(entry.get("expected_evidence_role") or "").strip()
    if source_product_skip_reason(entry):
        return False
    if product_type not in {
        "event_outbreak_report",
        "official_surveillance_report",
        "line_list_or_case_series",
        "case_report_article",
        "sequence_database_record",
        "news_case_report",
    }:
        return False
    if task_specificity == "unrelated":
        return False
    if readability in {"not_extractable", "manual_or_search_page"}:
        return False
    if role not in {
        "individual_case_evidence",
        "aggregate_event_evidence",
        "non_case_or_monitoring_evidence",
    }:
        return False
    source_type = _source_type_for_entry(entry)
    return (
        source_type in _OFFICIAL_SOURCE_TYPES
        or source_type in _STRUCTURED_SOURCE_TYPES
        or source_type in _LITERATURE_SOURCE_TYPES
        or str(entry.get("credibility_level") or "").strip().lower() == "high"
    )


def _is_high_trust_event_link_parent(
    entry: dict,
    document,
    state: dict | None,
) -> bool:
    if _is_verified_event_source_fetch_rescue(entry):
        return True
    source_type = _source_type_for_entry(entry)
    trusted = (
        source_type in _OFFICIAL_SOURCE_TYPES
        or source_type in _STRUCTURED_SOURCE_TYPES
        or source_type in _LITERATURE_SOURCE_TYPES
        or str(entry.get("credibility_level") or "").strip().lower() == "high"
    )
    if not trusted:
        return False
    structured = (state or {}).get("structured_task") or {}
    spec = (state or {}).get("collection_spec") or {}
    disease = str(
        structured.get("disease")
        or structured.get("virus")
        or spec.get("disease")
        or spec.get("virus")
        or ""
    ).strip().lower()
    text = " ".join(
        str(value or "")
        for value in (
            _document_attr(document, "title"),
            _document_attr(document, "clean_text"),
            entry.get("title"),
            entry.get("canonical_url"),
        )
    ).lower()
    disease_terms = {disease} if disease else set()
    if disease in {"flu", "influenza"}:
        disease_terms.update({"influenza", "seasonal influenza", "ili"})
    has_disease = any(term and term in text for term in disease_terms)
    has_event = any(
        term in text
        for term in (
            "outbreak",
            "disease outbreak news",
            "risk assessment",
            "situation report",
            "case report",
            "technical report",
            "cruise ship",
            "vessel-associated",
        )
    )
    return has_disease and has_event


def _is_context_only_document(doc: dict, role_policy: dict | None = None) -> bool:
    source_id = doc.get("source_id")
    if source_id in _context_only_source_ids(role_policy):
        return True
    flags = set(_routing_flags(doc))
    if "context_only" in flags or "blocked_from_structured_extraction" in flags:
        return True
    return (
        doc.get("source_role") in {"context_source", "context_only"}
        and doc.get("fetch_purpose") == "context_grounding"
    )


# ---------------------------------------------------------------------------
# Fetch-request construction
# ---------------------------------------------------------------------------


def _classify_skip_reason(
    entry: dict,
    policy: ContentFetchPolicy,
    allowlist: set[str] | None = None,
    collection_mode: str = "standard",
    role_policy: dict | None = None,
    fetch_config: dict | None = None,
) -> str | None:
    """Return a skip reason or None if the entry should be fetched.

    Order is intentional so the summary attributes each skipped source to the
    most informative reason: a PubMed search endpoint shows up as
    `search_endpoint` rather than the more generic `deferred`, and a seed://
    placeholder shows up as `blocked_scheme`.
    """

    # Allowlist takes priority: if the user opted into a small source list,
    # any other source_id should be skipped as `not_in_source_id_allowlist`
    # rather than falsely attributed to a downstream reason.
    if allowlist is not None and entry.get("source_id") not in allowlist:
        return "not_in_source_id_allowlist"
    if entry.get("source_excluded_by_human_review") is True:
        return "source_excluded_by_human_review"
    adaptive = _adaptive_acquisition_runtime() is not None
    provisional = adaptive and entry.get("task_fit_evidence_origin") == "discovery_metadata"
    if entry.get("source_disease_relevance_status") == UNRELATED_DISEASE and not provisional:
        return "disease_mismatch_source"
    if get_env("PIPELINE_MODE") == "evidence":
        # Priority is an ordering/budget permission, never permission to ignore
        # explicit source exclusions or the user's acquisition constraints.
        cfg = fetch_config or {}
        if _is_validation_reserved_entry(entry, collection_mode, role_policy):
            return "validation_reserved"
        final_role = str(entry.get("source_role_final") or "").strip().lower()
        if (final_role == "excluded" or entry.get("final_screening_decision") == "exclude") and not provisional:
            return "final_role_excluded"
        if entry.get("blocked_from_fetch"):
            return str(entry.get("blocked_from_fetch_reason") or "blocked_from_fetch")
        url = str(entry.get("canonical_url") or entry.get("url") or "")
        domain = urlsplit(url).hostname or ""
        if _domain_matches(domain, list(cfg.get("domain_blocklist") or [])):
            return "domain_blocklisted"
        allowed_domains = list(cfg.get("domain_allowlist") or [])
        if allowed_domains and not _domain_matches(domain, allowed_domains):
            return "domain_not_allowlisted"
        if _is_search_derived_entry(entry):
            if not bool(cfg.get("fetch_search_derived_sources", False)):
                return "search_derived_fetch_disabled"
            if adaptive:
                if _is_search_endpoint(entry, policy):
                    return "search_endpoint"
                if _url_scheme(url) not in {"http", "https"}:
                    return "unsupported_url_scheme"
                return None
            allowed_roles = set(cfg.get("allowed_final_roles") or [])
            review_allowed = final_role == "needs_human_review" and bool(cfg.get("allow_needs_review", False))
            if final_role == "needs_human_review" and not review_allowed:
                return "needs_review_not_allowed"
            if final_role and allowed_roles and final_role not in allowed_roles and not review_allowed:
                return "final_role_not_allowed"
        if adaptive:
            if _is_search_endpoint(entry, policy):
                return "search_endpoint"
            if _url_scheme(url) not in {"http", "https"}:
                return "unsupported_url_scheme"
            return None
    if entry.get("must_fetch"):
        url = entry.get("canonical_url") or entry.get("url") or ""
        scheme = _url_scheme(url)
        if scheme in {"http", "https"}:
            return None
    if _is_validation_reserved_entry(entry, collection_mode, role_policy):
        return "validation_reserved"
    url = entry.get("canonical_url") or entry.get("url") or ""
    scheme = _url_scheme(url)
    if _is_search_derived_entry(entry):
        cfg = fetch_config or {}
        if not bool(cfg.get("fetch_search_derived_sources", False)):
            return "search_derived_fetch_disabled"
        final_role = str(entry.get("source_role_final") or "").strip()
        allow_needs_review = bool(cfg.get("allow_needs_review", False))
        event_source_rescue = _is_verified_event_source_fetch_rescue(entry)
        if final_role in _SEARCH_DERIVED_BLOCKED_FINAL_ROLES:
            if final_role == "needs_human_review" and not allow_needs_review:
                return "needs_review_not_allowed"
            if final_role != "needs_human_review" and not event_source_rescue:
                return f"final_role_{final_role}"
        allowed_roles = set(cfg.get("allowed_final_roles") or [])
        if (
            final_role
            and allowed_roles
            and final_role not in allowed_roles
            and not (final_role == "needs_human_review" and allow_needs_review)
            and not event_source_rescue
        ):
            return "final_role_not_allowed"
        if not entry.get("source_type"):
            return "missing_source_type"
        if scheme not in {"http", "https"}:
            return "unsupported_url_scheme"
        bucket = _fetch_bucket(entry)
        authority_retry_event_candidate = _is_authority_retry_event_fetch_candidate(
            entry,
            bucket,
        )
        score = entry.get("credibility_score")
        threshold = float(cfg.get("min_credibility_score", 0.55))
        if score is None:
            return "missing_credibility_score"
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            return "invalid_credibility_score"
        if score_value < threshold and not authority_retry_event_candidate:
            return "credibility_score_below_threshold"
        level = str(entry.get("credibility_level") or "").strip().lower()
        # In evidence, explicit final permission to acquire content can coexist with
        # advisory metadata review. Fetching resolves that uncertainty; it does
        # not clear review or grant evidence qualification.
        explicit_evidence_fetch_permission = (
            get_env("PIPELINE_MODE") == "evidence"
            and entry.get("final_screening_decision") == "include_for_content_fetch"
            and entry.get("ready_for_content_fetch") is True
        )
        if (
            level not in {"high", "medium"}
            or (
                (entry.get("human_review_recommended") or entry.get("requires_human_review"))
                and not explicit_evidence_fetch_permission
            )
        ) and not (allow_needs_review or authority_retry_event_candidate):
            return "needs_review_not_allowed"
        if not entry.get("ready_for_content_fetch") and not (
            authority_retry_event_candidate
            or (
                allow_needs_review
                and (
                    final_role == "needs_human_review"
                    or entry.get("requires_human_review")
                    or entry.get("human_review_recommended")
                )
            )
        ):
            return "not_ready_for_content_fetch"
        final_decision = entry.get("final_screening_decision")
        if (
            final_decision not in policy.fetchable_final_decisions
            and not authority_retry_event_candidate
        ):
            return "final_screening_decision_not_fetchable"
        domain = _domain_for_entry(entry)
        if _domain_matches(domain, list(cfg.get("domain_blocklist") or [])):
            return "domain_blocklisted"
        allowlist_domains = list(cfg.get("domain_allowlist") or [])
        if allowlist_domains and not _domain_matches(domain, allowlist_domains):
            return "domain_not_allowlisted"
        return None
    if entry.get("requires_human_review"):
        return "human_review"
    if _is_search_endpoint(entry, policy):
        return "search_endpoint"
    if scheme in {s.lower() for s in policy.blocked_url_schemes}:
        return "blocked_scheme"
    if scheme not in {s.lower() for s in policy.allowed_url_schemes}:
        return "non_allowed_scheme"
    final_decision = entry.get("final_screening_decision")
    if final_decision in policy.deferred_final_decisions:
        return "deferred"
    if final_decision not in policy.fetchable_final_decisions:
        return "not_fetchable"
    return None


def _adaptive_acquisition_runtime():
    from ..acquisition_scheduling import adaptive_runtime
    return adaptive_runtime()


def _build_fetch_requests(
    state: DataCollectionState,
    policy: ContentFetchPolicy,
    live: bool,
    fetch_config: dict,
    allowlist: set[str] | None = None,
    collection_mode: str = "standard",
    role_policy: dict | None = None,
) -> tuple[list[ContentFetchRequest], dict[str, int], list[dict]]:
    registry, source_coverage_requirements, source_coverage_audit = (
        annotate_source_coverage(
            apply_source_product_profiles(list(state.get("source_registry") or []), state),
            state,
        )
    )
    adaptive = _adaptive_acquisition_runtime() is not None
    skip_counts: Counter = Counter()
    accepted: list[ContentFetchRequest] = []
    selection_manifest: list[dict] = []
    selected_search_derived_count = 0
    direct_fast_path_active = _direct_target_official_fast_path_active(
        registry,
        collection_mode=collection_mode,
        allowlist=allowlist,
    )

    if adaptive:
        direct_fast_path_active = False
    from ..acquisition_scheduling import acquisition_priority
    ordered = sorted(registry, key=(lambda row: -acquisition_priority(row, state=state)) if adaptive else _fetch_selection_sort_key)
    for entry in ordered:
        is_search_derived = _is_search_derived_entry(entry)
        bucket = _fetch_bucket(entry)
        skip_reason = _direct_fast_path_skip_reason(
            entry,
            fast_path_active=direct_fast_path_active,
        )
        if skip_reason is None:
            skip_reason = _classify_skip_reason(
                entry,
                policy,
                allowlist,
                collection_mode,
                role_policy,
                fetch_config,
            )
        if skip_reason is not None:
            skip_counts[skip_reason] += 1
            selection_manifest.append(
                {
                    "source_id": entry.get("source_id"),
                    "canonical_url": entry.get("canonical_url") or entry.get("url"),
                    "domain": _domain_for_entry(entry),
                    "discovery_method": entry.get("discovery_method"),
                    "source_role_final": entry.get("source_role_final"),
                    "source_type_final": entry.get("source_type_final"),
                    "actual_publisher": entry.get("actual_publisher"),
                    "target_fit_status": entry.get("target_fit_status"),
                    "triage_role": entry.get("triage_role"),
                    "disease_fit": entry.get("disease_fit"),
                    "geography_fit": entry.get("geography_fit"),
                    "date_fit": entry.get("date_fit"),
                    "requires_human_review": entry.get("requires_human_review"),
                    "human_review_reason": entry.get("human_review_reason"),
                    "fetch_bucket": bucket,
                    "credibility_score": entry.get("credibility_score"),
                    "credibility_level": entry.get("credibility_level"),
                    "must_fetch": bool(entry.get("must_fetch")),
                    "must_fetch_reason": entry.get("must_fetch_reason"),
                    "coverage_requirement_ids": entry.get("coverage_requirement_ids") or [],
                    "selected_for_fetch": False,
                    "skip_reason": skip_reason,
                    **_selection_manifest_product_fields(entry),
                }
            )
            continue
        is_must_fetch = bool(entry.get("must_fetch"))
        if not adaptive and is_search_derived and not is_must_fetch:
            max_search = int(fetch_config["max_search_derived_sources"] if fetch_config.get("max_search_derived_sources") is not None else 2)
            if selected_search_derived_count >= max_search:
                skip_reason = (
                    "fetch_cap_after_high_trust_floor"
                    if _is_retry_high_trust_fetch_candidate(entry, bucket)
                    else "max_search_derived_sources_reached"
                )
                skip_counts[skip_reason] += 1
                selection_manifest.append(
                    {
                        "source_id": entry.get("source_id"),
                        "canonical_url": entry.get("canonical_url") or entry.get("url"),
                        "domain": _domain_for_entry(entry),
                        "discovery_method": entry.get("discovery_method"),
                        "source_role_final": entry.get("source_role_final"),
                        "source_type_final": entry.get("source_type_final"),
                        "actual_publisher": entry.get("actual_publisher"),
                        "target_fit_status": entry.get("target_fit_status"),
                        "triage_role": entry.get("triage_role"),
                        "disease_fit": entry.get("disease_fit"),
                        "geography_fit": entry.get("geography_fit"),
                        "date_fit": entry.get("date_fit"),
                        "requires_human_review": entry.get("requires_human_review"),
                        "human_review_reason": entry.get("human_review_reason"),
                        "fetch_bucket": bucket,
                        "credibility_score": entry.get("credibility_score"),
                        "credibility_level": entry.get("credibility_level"),
                        "must_fetch": bool(entry.get("must_fetch")),
                        "must_fetch_reason": entry.get("must_fetch_reason"),
                        "coverage_requirement_ids": entry.get("coverage_requirement_ids") or [],
                        "selected_for_fetch": False,
                        "skip_reason": skip_reason,
                        **_selection_manifest_product_fields(entry),
                    }
                )
                continue
        max_total = int(fetch_config["max_total_sources"] if fetch_config.get("max_total_sources") is not None else 10)
        if not adaptive and len(accepted) >= max_total and not is_must_fetch:
            skip_reason = (
                "fetch_cap_after_high_trust_floor"
                if _is_retry_high_trust_fetch_candidate(entry, bucket)
                else "max_total_sources_reached"
            )
            skip_counts[skip_reason] += 1
            selection_manifest.append(
                {
                    "source_id": entry.get("source_id"),
                    "canonical_url": entry.get("canonical_url") or entry.get("url"),
                    "domain": _domain_for_entry(entry),
                    "discovery_method": entry.get("discovery_method"),
                    "source_role_final": entry.get("source_role_final"),
                    "source_type_final": entry.get("source_type_final"),
                    "actual_publisher": entry.get("actual_publisher"),
                    "target_fit_status": entry.get("target_fit_status"),
                    "triage_role": entry.get("triage_role"),
                    "disease_fit": entry.get("disease_fit"),
                    "geography_fit": entry.get("geography_fit"),
                    "date_fit": entry.get("date_fit"),
                    "requires_human_review": entry.get("requires_human_review"),
                    "human_review_reason": entry.get("human_review_reason"),
                    "fetch_bucket": bucket,
                    "credibility_score": entry.get("credibility_score"),
                    "credibility_level": entry.get("credibility_level"),
                    "must_fetch": bool(entry.get("must_fetch")),
                    "must_fetch_reason": entry.get("must_fetch_reason"),
                    "coverage_requirement_ids": entry.get("coverage_requirement_ids") or [],
                    "selected_for_fetch": False,
                    "skip_reason": skip_reason,
                    **_selection_manifest_product_fields(entry),
                }
            )
            continue

        final_decision = entry.get("final_screening_decision") or ""
        fetch_purpose = policy.fetch_purpose_by_decision.get(final_decision, "unknown")
        if _is_context_only_entry(entry, role_policy):
            fetch_purpose = "context_grounding"
        url = entry.get("canonical_url") or entry.get("url") or ""
        accepted.append(
            ContentFetchRequest(
                source_id=entry.get("source_id", ""),
                url=url,
                canonical_url=entry.get("canonical_url") or url,
                publisher=entry.get("publisher"),
                source_type=entry.get("source_type"),
                source_role=entry.get("source_role"),
                **_source_provenance(entry),
                final_screening_decision=final_decision,
                fetch_purpose=fetch_purpose,
                priority=entry.get("priority"),
                live_fetch_enabled=live,
            )
        )
        if is_search_derived and not is_must_fetch:
            selected_search_derived_count += 1
        selection_manifest.append(
            {
                "source_id": entry.get("source_id"),
                "canonical_url": entry.get("canonical_url") or entry.get("url"),
                "domain": _domain_for_entry(entry),
                "discovery_method": entry.get("discovery_method"),
                "source_role_final": entry.get("source_role_final"),
                "source_type_final": entry.get("source_type_final"),
                "actual_publisher": entry.get("actual_publisher"),
                "target_fit_status": entry.get("target_fit_status"),
                "triage_role": entry.get("triage_role"),
                "disease_fit": entry.get("disease_fit"),
                "geography_fit": entry.get("geography_fit"),
                "date_fit": entry.get("date_fit"),
                "requires_human_review": entry.get("requires_human_review"),
                "human_review_reason": entry.get("human_review_reason"),
                "fetch_bucket": bucket,
                "credibility_score": entry.get("credibility_score"),
                "credibility_level": entry.get("credibility_level"),
                "must_fetch": is_must_fetch,
                "must_fetch_reason": entry.get("must_fetch_reason"),
                "coverage_requirement_ids": entry.get("coverage_requirement_ids") or [],
                "selected_for_fetch": True,
                "skip_reason": None,
                **_selection_manifest_product_fields(entry),
            }
        )

    return accepted, dict(skip_counts), selection_manifest


def _canonical_url_key(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlsplit(str(value).strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    query = f"?{parsed.query}" if parsed.query else ""
    if not scheme and not netloc:
        return str(value).strip().rstrip("/")
    return f"{scheme}://{netloc}{path}{query}"


def _official_alias_urls_for_retry(
    request: ContentFetchRequest,
    source_entry: dict,
) -> list[str]:
    if not (
        source_entry.get("must_fetch")
        or source_entry.get("official_report_key")
        or source_entry.get("coverage_requirement_ids")
    ):
        return []
    seen = {
        _canonical_url_key(request.url),
        _canonical_url_key(request.canonical_url),
    }
    aliases: list[str] = []
    for raw_url in source_entry.get("official_report_alias_urls") or []:
        url = str(raw_url or "").strip()
        if not url:
            continue
        key = _canonical_url_key(url)
        if key in seen:
            continue
        seen.add(key)
        aliases.append(url)
    return aliases


def _request_for_alias(
    request: ContentFetchRequest,
    alias_url: str,
) -> ContentFetchRequest:
    return request.model_copy(
        update={
            "url": alias_url,
            "canonical_url": alias_url,
        }
    )


# ---------------------------------------------------------------------------
# Offline stub document
# ---------------------------------------------------------------------------


def _stub_clean_text(source_entry: dict) -> str:
    """Build a deterministic metadata summary for offline stub documents."""

    title = source_entry.get("title") or ""
    publisher = source_entry.get("publisher") or ""
    source_type = source_entry.get("source_type") or ""
    source_role = source_entry.get("source_role") or ""
    source_purpose = source_entry.get("source_purpose") or ""
    expected_fields = source_entry.get("expected_fields") or []
    matched_terms = source_entry.get("matched_terms") or []
    notes = source_entry.get("notes") or ""

    lines = [
        "Offline metadata stub document",
        f"Title: {title}",
        f"Publisher: {publisher}",
        f"Source type: {source_type}",
        f"Source role: {source_role}",
        f"Source purpose: {source_purpose}",
        f"Expected fields: {', '.join(expected_fields)}",
        f"Matched terms: {', '.join(matched_terms)}",
        f"Notes: {notes}",
    ]
    return "\n".join(lines).strip()


def _make_fixture_document(
    request: ContentFetchRequest,
    source_entry: dict,
    fixture: FixtureDocument,
) -> Document:
    """Build a Document from a synthetic local fixture (offline, opt-in only)."""

    clean_text = fixture.clean_text
    return Document(
        source_id=request.source_id,
        document_type=fixture.document_type,
        clean_text=clean_text,
        tables=list(fixture.tables or []),
        metadata={
            "fixture_mode_enabled": True,
            "synthetic_fixture": True,
            "not_real_public_health_data": True,
            **_source_metadata(source_entry),
            "fixture_id": fixture.fixture_id,
            "fixture_notes": fixture.notes,
        },
        parse_status="fixture_loaded",
        quality_status=None,
        quality_issues=[],
        url=request.url,
        canonical_url=request.canonical_url,
        title=fixture.title,
        publisher=source_entry.get("publisher"),
        source_type=source_entry.get("source_type"),
        source_role=source_entry.get("source_role"),
        **_source_provenance(source_entry),
        final_screening_decision=request.final_screening_decision,
        fetch_purpose=request.fetch_purpose,
        fetch_status="fixture_loaded",
        fetch_error=None,
        fetch_provider="fixture_document",
        provider_attempts=[{"provider": "fixture_document", "success": True}],
        http_status_code=None,
        content_type=None,
        fetched_at=_OFFLINE_FIXED_TIMESTAMP,
        parser_used="fixture_document_loader",
        text_char_count=len(clean_text or ""),
        table_count=len(fixture.tables or []),
        content_hash=_hash_text(clean_text),
        is_live_fetched=False,
        is_offline_stub=False,
        is_fixture_document=True,
        fixture_id=fixture.fixture_id,
        fixture_notes=fixture.notes,
    )


def _make_offline_stub_document(
    request: ContentFetchRequest,
    source_entry: dict,
) -> Document:
    clean_text = _stub_clean_text(source_entry)
    return Document(
        source_id=request.source_id,
        document_type="offline_metadata_stub",
        clean_text=clean_text,
        tables=[],
        metadata={
            "live_fetch_enabled": False,
            "offline_stub_reason": "Live fetching disabled; created metadata stub.",
            **_source_metadata(source_entry),
        },
        parse_status="offline_stub",
        quality_status=None,
        quality_issues=[],
        url=request.url,
        canonical_url=request.canonical_url,
        title=source_entry.get("title"),
        publisher=source_entry.get("publisher"),
        source_type=source_entry.get("source_type"),
        source_role=source_entry.get("source_role"),
        **_source_provenance(source_entry),
        final_screening_decision=request.final_screening_decision,
        fetch_purpose=request.fetch_purpose,
        fetch_status="offline_stub",
        fetch_error=None,
        fetch_provider="offline_stub",
        provider_attempts=[{"provider": "offline_stub", "success": True}],
        http_status_code=None,
        content_type=None,
        fetched_at=_OFFLINE_FIXED_TIMESTAMP,
        parser_used="offline_metadata_stub",
        text_char_count=len(clean_text or ""),
        table_count=0,
        content_hash=_hash_text(clean_text),
        is_live_fetched=False,
        is_offline_stub=True,
    )


# ---------------------------------------------------------------------------
# Live fetch (opt-in)
# ---------------------------------------------------------------------------


def _looks_like_pdf(url: str, content_type: str | None) -> bool:
    if content_type and "pdf" in content_type.lower():
        return True
    return (url or "").lower().split("?")[0].endswith(".pdf")


def _looks_like_textual_content_type(content_type: str | None) -> bool:
    lowered = (content_type or "").lower()
    return any(
        marker in lowered
        for marker in (
            "text/",
            "markdown",
            "json",
            "csv",
            "xml",
        )
    )


def _body_starts_like_pdf(body: bytes) -> bool:
    return body.lstrip().startswith(b"%PDF")


class _VisibleHTMLTextParser(HTMLParser):
    """Small stdlib parser for visible text, titles, meta dates, and tables."""

    _SKIP_TAGS = {"script", "style", "nav", "footer", "noscript"}

    def __init__(self, *, parse_tables: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self.parse_tables = parse_tables
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._current_text: list[str] = []
        self.title_parts: list[str] = []
        self.headings: list[str] = []
        self.text_parts: list[str] = []
        self.meta_description: str | None = None
        self.published_date: str | None = None
        self.tables: list[dict] = []
        self._in_table = False
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None
        self._latest_heading: str | None = None
        self._current_table_heading: str | None = None
        self.links: list[dict] = []
        self._current_link_href: str | None = None
        self._current_link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        attr_map = {key.lower(): value for key, value in attrs if key}
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "meta":
            name = (attr_map.get("name") or attr_map.get("property") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if not content:
                return
            if name in {"description", "og:description"} and not self.meta_description:
                self.meta_description = content
            if name in {
                "date",
                "dc.date",
                "article:published_time",
                "pubdate",
                "publishdate",
                "published_date",
            } and not self.published_date:
                self.published_date = content[:10]
        if tag == "a" and attr_map.get("href"):
            self._current_link_href = str(attr_map.get("href") or "").strip()
            self._current_link_parts = []
        if not self.parse_tables:
            return
        if tag == "table":
            self._in_table = True
            self._table_rows = []
            self._current_table_heading = self._latest_heading
        elif self._in_table and tag == "tr":
            self._current_row = []
        elif self._in_table and tag in {"td", "th"}:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a" and self._current_link_href:
            self.links.append(
                {
                    "href": self._current_link_href,
                    "text": _normalize_whitespace(" ".join(self._current_link_parts)),
                }
            )
            self._current_link_href = None
            self._current_link_parts = []
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if not self._tag_stack:
            return
        if self.parse_tables and self._in_table and tag in {"td", "th"}:
            if self._current_cell is not None and self._current_row is not None:
                cell = _normalize_whitespace(" ".join(self._current_cell))
                self._current_row.append(cell)
            self._current_cell = None
        elif self.parse_tables and self._in_table and tag == "tr":
            if self._current_row:
                self._table_rows.append(self._current_row)
            self._current_row = None
        elif self.parse_tables and self._in_table and tag == "table":
            if self._table_rows:
                self.tables.append(
                    {
                        "table_index": len(self.tables),
                        "rows": list(self._table_rows),
                        "heading_context": self._current_table_heading,
                    }
                )
            self._table_rows = []
            self._in_table = False
            self._current_table_heading = None
        while self._tag_stack:
            popped = self._tag_stack.pop()
            if popped == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = _normalize_whitespace(data)
        if not text:
            return
        current_tag = self._tag_stack[-1] if self._tag_stack else ""
        if current_tag == "title":
            self.title_parts.append(text)
        if current_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.headings.append(text)
            self._latest_heading = text
        if self.parse_tables and self._in_table and self._current_cell is not None:
            self._current_cell.append(text)
        if self._current_link_href:
            self._current_link_parts.append(text)
        self.text_parts.append(text)


def _decode_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _parse_pdf_text_with_pypdf(body: bytes) -> str | None:
    try:
        from io import BytesIO
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        raise RuntimeError(f"pypdf_unavailable:{type(exc).__name__}") from exc
    reader = PdfReader(BytesIO(body))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(part.strip() for part in pages if part.strip()) or None


def _parse_pdf_text_with_optional_fallback(body: bytes) -> dict | None:
    try:
        import fitz  # type: ignore

        doc = fitz.open(stream=body, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        clean_text = "\n".join(part.strip() for part in pages if part.strip()) or None
        if clean_text:
            return {
                "clean_text": clean_text,
                "parser_used": "pdf_pymupdf_fallback_parser",
                "parse_error": None,
            }
    except Exception:
        pass
    try:
        from io import BytesIO
        from pdfminer.high_level import extract_text  # type: ignore

        clean_text = extract_text(BytesIO(body)) or ""
        clean_text = "\n".join(
            part.strip() for part in clean_text.splitlines() if part.strip()
        ) or None
        if clean_text:
            return {
                "clean_text": clean_text,
                "parser_used": "pdf_pdfminer_fallback_parser",
                "parse_error": None,
            }
    except Exception:
        pass
    return None


def _parse_pdf_content(body: bytes, *, parse_pdf_text: bool) -> dict:
    if not parse_pdf_text:
        return {
            "document_type": "pdf",
            "clean_text": None,
            "tables": [],
            "parse_status": "parse_deferred",
            "parser_used": "pdf_parse_deferred",
            "title": None,
            "published_date": None,
            "content_type": "application/pdf",
            "parse_error": "pdf_text_parsing_disabled",
        }
    try:
        clean_text = _parse_pdf_text_with_pypdf(body)
        parser_used = "pdf_pypdf_parser"
        parse_error = None if clean_text else "pdf_no_extractable_text"
    except Exception as exc:  # noqa: BLE001 - malformed PDFs should not crash.
        fallback = _parse_pdf_text_with_optional_fallback(body)
        if fallback and fallback.get("clean_text"):
            clean_text = fallback.get("clean_text")
            parser_used = fallback.get("parser_used") or "pdf_fallback_parser"
            parse_error = fallback.get("parse_error")
        else:
            return {
                "document_type": "pdf",
                "clean_text": None,
                "tables": [],
                "parse_status": "parse_failed",
                "parser_used": "pdf_parse_failed",
                "title": None,
                "published_date": None,
                "content_type": "application/pdf",
                "parse_error": f"pypdf_parse_failed:{type(exc).__name__}",
            }
    return {
        "document_type": "pdf",
        "clean_text": clean_text,
        "tables": [],
        "parse_status": "parsed_pdf" if clean_text else "parse_failed",
        "parser_used": parser_used if clean_text else "pdf_parse_failed",
        "title": None,
        "published_date": None,
        "content_type": "application/pdf",
        "parse_error": parse_error,
    }


def _parse_document_content(
    body: bytes,
    *,
    url: str,
    content_type: str | None,
    parse_pdf_text: bool,
    parse_tables: bool,
) -> dict:
    """Parse fetched bytes into auditable clean text and parse metadata."""

    normalized_content_type = content_type or ""
    should_parse_as_pdf = (
        _looks_like_pdf(url, normalized_content_type)
        and not _looks_like_textual_content_type(normalized_content_type)
        and _body_starts_like_pdf(body)
    )
    if should_parse_as_pdf:
        result = _parse_pdf_content(body, parse_pdf_text=parse_pdf_text)
    elif "html" in normalized_content_type.lower() or body.lstrip().startswith(b"<"):
        parser = _VisibleHTMLTextParser(parse_tables=parse_tables)
        parser.feed(_decode_body(body))
        title = _normalize_whitespace(" ".join(parser.title_parts)) or None
        clean_parts = []
        if parser.meta_description:
            clean_parts.append(parser.meta_description)
        clean_parts.extend(parser.headings)
        clean_parts.extend(parser.text_parts)
        clean_text = "\n".join(
            line for line in (_normalize_whitespace(part) for part in clean_parts) if line
        )
        result = {
            "document_type": "html",
            "clean_text": clean_text or None,
            "tables": parser.tables if parse_tables else [],
            "parse_status": "parsed_html",
            "parser_used": "html_stdlib_parser",
            "title": title,
            "published_date": parser.published_date,
            "content_type": normalized_content_type or "text/html",
            "parse_error": None,
            "outbound_links": list(parser.links),
        }
    else:
        clean_text = _decode_body(body)
        lowered = normalized_content_type.lower()
        parser_used = "text_parser"
        if "json" in lowered:
            parser_used = "json_text_parser"
        elif "csv" in lowered:
            parser_used = "csv_text_parser"
        result = {
            "document_type": "text",
            "clean_text": clean_text,
            "tables": [],
            "parse_status": "parsed_text",
            "parser_used": parser_used,
            "title": None,
            "published_date": None,
            "content_type": normalized_content_type or "text/plain",
            "parse_error": None,
        }
    clean_text = result.get("clean_text") or ""
    tables = result.get("tables") or []
    result["text_char_count"] = len(clean_text)
    result["table_count"] = len(tables)
    return result




def _base_document_kwargs(
    request: ContentFetchRequest,
    source_entry: dict,
    *,
    live: bool,
    fetched_at: str,
) -> dict:
    return {
        "source_id": request.source_id,
        "url": request.url,
        "canonical_url": request.canonical_url,
        "title": source_entry.get("title"),
        "published_date": source_entry.get("published_date"),
        "publisher": source_entry.get("publisher"),
        "source_type": source_entry.get("source_type"),
        "source_role": source_entry.get("source_role"),
        **_source_provenance(source_entry),
        "final_screening_decision": request.final_screening_decision,
        "fetch_purpose": request.fetch_purpose,
        "is_live_fetched": live,
        "is_offline_stub": False,
        "fetched_at": fetched_at,
        "metadata": {
            "live_fetch_enabled": live,
            **_source_metadata(source_entry),
        },
    }


def _make_parsed_document(
    request: ContentFetchRequest,
    source_entry: dict,
    *,
    body: bytes,
    content_type: str | None,
    http_status_code: int | None,
    fetch_status: str,
    fetched_at: str,
    live: bool,
    fetch_config: dict,
    extra_metadata: dict | None = None,
    fetch_provider: str | None = None,
    provider_attempts: list[dict] | None = None,
) -> Document:
    parsed = _parse_document_content(
        body,
        url=request.url,
        content_type=content_type,
        parse_pdf_text=bool(fetch_config.get("parse_pdf_text", True)),
        parse_tables=bool(fetch_config.get("parse_tables", True)),
    )
    clean_text = parsed.get("clean_text")
    metadata = {
        **_source_metadata(source_entry),
        "parse_error": parsed.get("parse_error"),
        "raw_text_stored": bool(fetch_config.get("store_raw_text", False)),
        "outbound_links": list(parsed.get("outbound_links") or []),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    provider_attempt_rows = list(provider_attempts or [])
    if fetch_provider:
        metadata["fetch_provider"] = fetch_provider
    if provider_attempt_rows:
        metadata["provider_attempts"] = provider_attempt_rows
    if bool(fetch_config.get("store_raw_text", False)) and clean_text:
        metadata["raw_text"] = clean_text
    base_kwargs = _base_document_kwargs(
        request,
        source_entry,
        live=live,
        fetched_at=fetched_at,
    )
    base_kwargs["metadata"] = metadata
    base_kwargs["title"] = parsed.get("title") or source_entry.get("title")
    base_kwargs["published_date"] = (
        parsed.get("published_date") or source_entry.get("published_date")
    )
    return Document(
        **base_kwargs,
        document_type=parsed.get("document_type"),
        clean_text=clean_text,
        tables=list(parsed.get("tables") or []),
        parse_status=parsed.get("parse_status") or "parse_failed",
        quality_status=None,
        quality_issues=[],
        fetch_status=fetch_status,
        fetch_error=parsed.get("parse_error"),
        fetch_provider=fetch_provider,
        provider_attempts=provider_attempt_rows,
        http_status_code=http_status_code,
        content_type=parsed.get("content_type") or content_type,
        parser_used=parsed.get("parser_used"),
        text_char_count=parsed.get("text_char_count"),
        table_count=parsed.get("table_count"),
        content_hash=_hash_text(clean_text),
        is_fixture_document=bool(
            extra_metadata and extra_metadata.get("local_content_fixture")
        ),
        fixture_id=(extra_metadata or {}).get("fixture_id"),
        fixture_notes=(extra_metadata or {}).get("fixture_notes"),
    )


def _fetch_fixture_content_document(
    request: ContentFetchRequest,
    source_entry: dict,
    fixture_entry: dict,
    fetch_config: dict,
) -> Document:
    fixture_path = Path(fixture_entry["fixture_path"])
    body = fixture_path.read_bytes()
    max_bytes = int(fetch_config.get("max_bytes") or len(body))
    return _make_parsed_document(
        request,
        source_entry,
        body=body[:max_bytes],
        content_type=fixture_entry.get("content_type") or "text/html; charset=utf-8",
        http_status_code=int(fixture_entry.get("http_status_code") or 200),
        fetch_status="fixture_content_loaded",
        fetched_at=_OFFLINE_FIXED_TIMESTAMP,
        live=False,
        fetch_config=fetch_config,
        extra_metadata={
            "local_content_fixture": True,
            "synthetic_fixture": True,
            "not_real_public_health_data": True,
            "fixture_id": fixture_entry.get("fixture_id") or fixture_path.stem,
            "fixture_path": str(fixture_path),
            "fixture_notes": fixture_entry.get("notes"),
        },
        fetch_provider="content_fixture",
        provider_attempts=[{"provider": "content_fixture", "success": True}],
    )


def _fetch_live_document(
    request: ContentFetchRequest,
    source_entry: dict,
    policy: ContentFetchPolicy,
    fetch_config: dict,
) -> Document:
    """Perform a live HTTP fetch and parse through the generalized parser."""

    import requests  # local import; offline path never touches the network

    request_cfg = policy.request or {}
    timeout = float(
        get_env("FETCH_TIMEOUT_SECONDS")
        or request_cfg.get("timeout_seconds")
        or 15
    )
    max_bytes = int(fetch_config.get("max_bytes") or 5_000_000)
    user_agent = fetch_config.get("user_agent") or request_cfg.get(
        "user_agent", "data-collection-workflow/0.1"
    )
    headers = {"User-Agent": user_agent}
    fetched_at = _now_fixed_or_utc(live=True)

    try:
        response = requests.get(
            request.url,
            timeout=timeout,
            headers=headers,
            stream=True,
        )
        http_status = response.status_code
        content_type = response.headers.get("Content-Type")
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += len(chunks[-1])
        body = b"".join(chunks)
    except Exception as exc:  # pragma: no cover - manual live-fetch only
        return Document(
            **_base_document_kwargs(
                request,
                source_entry,
                live=True,
                fetched_at=fetched_at,
            ),
            document_type=None,
            clean_text=None,
            tables=[],
            parse_status="fetch_failed",
            fetch_status="fetch_failed",
            fetch_error=str(exc),
            fetch_provider="native_requests",
            provider_attempts=[
                {"provider": "native_requests", "success": False, "error": str(exc)}
            ],
            http_status_code=None,
            content_type=None,
            parser_used=None,
            text_char_count=0,
            table_count=0,
            content_hash=None,
        )

    fetch_status = "fetched" if http_status < 400 else "fetch_failed"
    return _make_parsed_document(
        request,
        source_entry,
        body=body,
        content_type=content_type,
        http_status_code=http_status,
        fetch_status=fetch_status,
        fetched_at=fetched_at,
        live=True,
        fetch_config=fetch_config,
        fetch_provider="native_requests",
        provider_attempts=[
            {
                "provider": "native_requests",
                "success": http_status < 400,
                "http_status_code": http_status,
                "error": None if http_status < 400 else f"http_status_{http_status}",
            }
        ],
    )


def _tavily_extract_text_from_item(item: dict) -> str:
    for key in ("raw_content", "content", "markdown", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _tavily_extract_results(data) -> list[dict]:
    if isinstance(data, dict):
        for key in ("results", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(key in data for key in ("raw_content", "content", "markdown", "text")):
            return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


@_traceable_tool("tavily_extract_fetch")
def _tavily_extract_fetch(
    request: ContentFetchRequest,
    source_entry: dict,
    fetch_config: dict,
) -> dict:
    """Fetch page content with Tavily Extract.

    This function is only called in live mode when explicitly enabled. It
    returns a small provider-neutral envelope so tests can mock it without
    touching the network.
    """

    api_key = (get_env("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return {
            "success": False,
            "provider": "tavily_extract",
            "error": "missing_api_key:TAVILY_API_KEY",
        }

    import requests  # local import; offline path never touches the network

    timeout = float(fetch_config.get("tavily_extract_timeout_seconds") or 45.0)
    payload = {
        "urls": [request.url],
        "format": fetch_config.get("tavily_extract_format") or "markdown",
        "extract_depth": fetch_config.get("tavily_extract_depth") or "advanced",
        "include_images": False,
    }
    chunks_per_source = int(fetch_config.get("tavily_extract_chunks_per_source") or 0)
    if chunks_per_source > 0:
        payload["chunks_per_source"] = chunks_per_source

    try:
        response = requests.post(
            "https://api.tavily.com/extract",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
            **({"allow_redirects": False} if get_env("PIPELINE_MODE") == "evidence" else {}),
        )
    except Exception as exc:  # pragma: no cover - live provider only
        return {
            "success": False,
            "provider": "tavily_extract",
            "error": f"provider_exception:{exc}",
        }

    http_status = response.status_code
    if 300 <= http_status < 400 and get_env("PIPELINE_MODE") == "evidence":
        # The session ledger reserved one provider request. A redirected API
        # request needs another reservation; never follow it invisibly.
        return {"success": False, "provider": "tavily_extract",
            "http_status_code": http_status, "error": f"provider_redirect_response:{http_status}",
            "redirect_location": response.headers.get("Location"),
            "response_body": response.content}
    try:
        data = response.json()
    except Exception as exc:  # pragma: no cover - live provider only
        return {
            "success": False,
            "provider": "tavily_extract",
            "http_status_code": http_status,
            "error": f"invalid_json:{exc}",
        }

    results = _tavily_extract_results(data)
    failed_results = data.get("failed_results") if isinstance(data, dict) else None
    first_result = results[0] if results else {}
    text = _tavily_extract_text_from_item(first_result)
    if http_status >= 400:
        return {
            "success": False,
            "provider": "tavily_extract",
            "http_status_code": http_status,
            "error": f"http_status_{http_status}",
            "failed_results": failed_results or [],
        }
    if not text.strip():
        return {
            "success": False,
            "provider": "tavily_extract",
            "http_status_code": http_status,
            "error": "empty_extracted_content",
            "failed_results": failed_results or [],
        }

    content_type = "text/markdown; charset=utf-8"
    if str(fetch_config.get("tavily_extract_format") or "").lower() == "text":
        content_type = "text/plain; charset=utf-8"
    return {
        "success": True,
        "provider": "tavily_extract",
        "body": text.encode("utf-8"),
        "content_type": content_type,
        "http_status_code": http_status,
        "metadata": {
            "tavily_extract_depth": fetch_config.get("tavily_extract_depth"),
            "tavily_extract_format": fetch_config.get("tavily_extract_format"),
            "tavily_extract_result_count": len(results),
            "failed_results": failed_results or [],
            "source_url": first_result.get("url") or request.url,
        },
    }


def _universal_priority_context(request: ContentFetchRequest, source_entry: dict) -> dict | None:
    """Bind a priority exception to maintained authority and explicit task fit."""
    if (source_entry.get("source_id") != request.source_id or
            source_entry.get("url") != request.url or
            source_entry.get("must_fetch") is not True or
            source_entry.get("source_identity_unverified") is not False or
            source_entry.get("metadata_only_identity") is not False):
        return None
    parsed = urlsplit(request.url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    # Never use claimed publisher/domain metadata to authorize an exception.
    identity = lookup_source_identity_registry(parsed.hostname)
    if not identity or identity.get("source_type") not in _OFFICIAL_SOURCE_TYPES:
        return None
    if (not _is_search_verified_target_entry(source_entry) or
            not str(source_entry.get("target_verification_reason") or "").strip() or
            any(_fit_value(source_entry, key) not in {"match", "exact", "exact_match"}
                for key in ("disease_fit", "geography_fit", "date_fit")) or
            source_entry.get("source_disease_relevance_status") in REJECT_DISEASE_STATUSES or
            _entry_has_non_exact_task_fit(source_entry)):
        return None
    return {"source_id": request.source_id, "verified_url": request.url,
            "source_identity_verified": True, "must_fetch": True, "task_fit_verified": True}


def _fetch_live_document_with_providers(
    request: ContentFetchRequest,
    source_entry: dict,
    policy: ContentFetchPolicy,
    fetch_config: dict,
) -> Document:
    if get_env("PIPELINE_MODE") == "evidence":
        from ..document_acquisition import acquire_document, parse_response, PartialResponseError, _mark_budget_exhausted
        from ..session_runtime import get_runtime
        runtime = get_runtime()
        if runtime is None:
            raise RuntimeError("evidence acquisition requires an active session runtime")
        acquisition_config = {**runtime.config,
            **((runtime.config.get("universal") or {}).get("acquisition") or {}),
            **fetch_config, "_source_target_id": request.canonical_url or request.url}
        acquisition_config["acquisition_strategy"] = (source_entry.get("acquisition_strategy")
            or acquisition_config.get("acquisition_strategy") or "native")
        result = acquire_document(
            request.url, source_id=request.source_id, session_dir=runtime.session_dir,
            config=acquisition_config,
            priority_context=_universal_priority_context(request, source_entry),
        )
        # Enabled alternative transports share the original target and request ledger.
        if (not result.get("content_readable") and fetch_config.get("external_fetch_enabled") and
                "tavily_extract" in (fetch_config.get("external_fetch_provider_order") or [])):
            from ..session_runtime import BudgetExceeded, ResumeMismatch
            from ..acquisition_budget import canonical_source_target
            def fetch_alternative():
                value = _tavily_extract_fetch(request, source_entry, fetch_config)
                if not value.get("success"):
                    raise PartialResponseError(value.get("error") or "alternative_fetch_failed", value)
                return value
            try:
                alternative = runtime.call("http_request" if runtime.ledger.adaptive else "fetch_ordinary",
                    {"provider": "tavily_extract", "url": request.url,
                     "max_bytes": fetch_config.get("max_bytes", 20_000_000)}, fetch_alternative,
                    source_target=canonical_source_target(request.canonical_url or request.url))
            except BudgetExceeded as exc:
                _mark_budget_exhausted(result, exc)
            except ResumeMismatch:
                raise
            except Exception as exc:
                result.setdefault("provider_attempts", []).append({"provider": "tavily_extract",
                    "success": False, "error": type(exc).__name__ + ": " + str(exc)})
            else:
                body = alternative.get("body") or b""
                if isinstance(body, str):
                    body = body.encode("utf-8")
                complete = len(body) <= int(fetch_config.get("max_bytes", 20_000_000))
                original = result
                result = parse_response(body[:int(fetch_config.get("max_bytes", 20_000_000))],
                    url=request.url, source_id=request.source_id, session_dir=runtime.session_dir,
                    config={**fetch_config, "_source_target_id": request.canonical_url or request.url},
                    content_type=alternative.get("content_type") or "text/plain",
                    status_code=alternative.get("http_status_code") or 200, raw_content_complete=complete)
                result.update(fetch_provider="tavily_extract",
                    response_artifact_path=original.get("raw_artifact_path"),
                    response_content_hash=original.get("content_hash"),
                    metadata={**result.get("metadata", {}), **alternative.get("metadata", {}),
                              "content_representation": "provider_extracted_text"},
                    provider_attempts=[*original.get("provider_attempts", []),
                        {"provider": "tavily_extract", "success": result.get("content_readable", False)}])
        base = _base_document_kwargs(request, source_entry, live=True, fetched_at=result["retrieved_at"])
        return Document(**{**base, **result})

    provider_order = list(
        fetch_config.get("external_fetch_provider_order")
        or ["tavily_extract", "native_requests"]
    )
    if not bool(fetch_config.get("external_fetch_enabled", False)):
        return _fetch_live_document(request, source_entry, policy, fetch_config)

    attempts: list[dict] = []
    for provider in provider_order:
        provider_name = str(provider or "").strip()
        if provider_name == "tavily_extract":
            result = _tavily_extract_fetch(request, source_entry, fetch_config)
            attempt = {
                "provider": "tavily_extract",
                "success": bool(result.get("success")),
                "http_status_code": result.get("http_status_code"),
                "error": result.get("error"),
            }
            if result.get("failed_results"):
                attempt["failed_results"] = result.get("failed_results")
            attempts.append(attempt)
            if result.get("success"):
                body_value = result.get("body") or b""
                if isinstance(body_value, str):
                    body_bytes = body_value.encode("utf-8")
                else:
                    body_bytes = bytes(body_value)
                return _make_parsed_document(
                    request,
                    source_entry,
                    body=body_bytes,
                    content_type=result.get("content_type"),
                    http_status_code=result.get("http_status_code"),
                    fetch_status="fetched",
                    fetched_at=_now_fixed_or_utc(live=True),
                    live=True,
                    fetch_config=fetch_config,
                    extra_metadata=result.get("metadata") or {},
                    fetch_provider="tavily_extract",
                    provider_attempts=attempts,
                )
            continue
        if provider_name == "native_requests":
            doc = _fetch_live_document(request, source_entry, policy, fetch_config)
            doc.fetch_provider = doc.fetch_provider or "native_requests"
            existing_attempts = list(doc.provider_attempts or [])
            if attempts and not any(attempt == attempts[0] for attempt in existing_attempts):
                doc.provider_attempts = attempts + existing_attempts
            else:
                doc.provider_attempts = existing_attempts
            if attempts:
                doc.metadata = {
                    **(doc.metadata or {}),
                    "native_fallback_after": attempts[-1].get("provider"),
                    "provider_attempts": doc.provider_attempts,
                    "fetch_provider": doc.fetch_provider,
                }
            return doc
        attempts.append(
            {
                "provider": provider_name,
                "success": False,
                "error": "unknown_provider",
            }
        )

    return Document(
        **_base_document_kwargs(
            request,
            source_entry,
            live=True,
            fetched_at=_now_fixed_or_utc(live=True),
        ),
        document_type=None,
        clean_text=None,
        tables=[],
        parse_status="fetch_failed",
        quality_status=None,
        quality_issues=[],
        fetch_status="fetch_failed",
        fetch_error="all_external_fetch_providers_failed",
        fetch_provider="external_fetch",
        provider_attempts=attempts,
        http_status_code=None,
        content_type=None,
        parser_used=None,
        text_char_count=0,
        table_count=0,
        content_hash=None,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def content_fetch_and_parse(state: DataCollectionState) -> dict:
    """Build fetch requests and produce Document objects.

    Offline by default; live HTTP only when ENABLE_LIVE_FETCH=true.
    """

    policy = ContentFetchPolicy(**load_content_fetch_policy())
    role_policy = load_source_role_policy()
    collection_mode = get_collection_mode(role_policy)
    live = _live_fetch_enabled()
    fetch_config = _fetch_config_from_env(policy)
    adaptive_runtime = _adaptive_acquisition_runtime()
    frontier = adaptive_runtime.frontier if adaptive_runtime else None
    fixture_enabled = _fixture_documents_enabled()
    fixture_map: dict[str, FixtureDocument] = (
        _load_fixture_document_map() if fixture_enabled else {}
    )
    content_fixture_map = _load_content_fixture_map(
        fetch_config.get("content_fixture_map_path")
    )
    allowlist = _parse_source_id_allowlist()

    fetch_requests, skip_counts, selection_manifest = _build_fetch_requests(
        state, policy, live, fetch_config, allowlist, collection_mode, role_policy
    )

    registry, source_coverage_requirements, source_coverage_audit = (
        annotate_source_coverage(
            apply_source_product_profiles(list(state.get("source_registry") or []), state),
            state,
        )
    )
    registry_by_id = {e.get("source_id"): e for e in registry}

    documents: list[Document] = [Document(**row) for row in state.get("documents") or []] if frontier else []
    if frontier:
        from ..acquisition_scheduling import load_frontier_results
        identities = {(doc.source_id, doc.content_hash, doc.text_hash) for doc in documents}
        for saved in load_frontier_results(adaptive_runtime):
            entry = saved['entry']
            if entry['source_id'] not in registry_by_id:
                registry.append(entry)
                registry_by_id[entry['source_id']] = entry
            doc = Document(**saved['document'])
            key = (doc.source_id, doc.content_hash, doc.text_hash)
            if key not in identities:
                documents.append(doc)
                identities.add(key)
    executed_fetch_requests: list[ContentFetchRequest] = []
    fixture_loaded_source_ids: list[str] = []
    alias_fetch_attempted_count = 0
    alias_fetch_success_count = 0
    alias_fetch_unusable_count = 0
    fallback_link_rows: list[dict] = []
    fallback_link_selected_source_ids: list[str] = []
    fallback_link_selected_urls: list[str] = []
    event_link_rows: list[dict] = []
    event_link_selected_source_ids: list[str] = []
    event_link_selected_urls: list[str] = []
    fetch_node_started_at = monotonic()
    fetch_fail_open_reason_counter: Counter = Counter()
    fetch_fail_open_skipped_source_ids: list[str] = []

    def _fetch_node_time_budget_exhausted() -> bool:
        max_seconds = float(fetch_config.get("max_node_seconds") or 0.0)
        return bool(max_seconds > 0 and monotonic() - fetch_node_started_at >= max_seconds)

    def _mark_fetch_fail_open_skipped(source_id: str, reason: str) -> None:
        if not source_id:
            return
        fetch_fail_open_reason_counter[reason] += 1
        fetch_fail_open_skipped_source_ids.append(source_id)
        skip_counts[reason] = int(skip_counts.get(reason, 0)) + 1
        for item in selection_manifest:
            if item.get("source_id") != source_id:
                continue
            item["selected_for_fetch"] = False
            item["skip_reason"] = reason
            item["fetch_fail_open_skipped"] = True
            item["fetch_status"] = "skipped"
            item["parse_status"] = "skipped"
            return

    def _fetch_document_for_request(
        request: ContentFetchRequest,
        source_entry: dict,
    ) -> tuple[Document, bool]:
        source_entry = registry_by_id.get(request.source_id) or {}
        emit_workflow_progress(
            "content_fetch_and_parse",
            "content fetch started",
            {
                "source_id": request.source_id,
                "url": request.url,
                "fetch_purpose": request.fetch_purpose,
                "live_fetch_enabled": live,
            },
        )
        fixture = fixture_map.get(request.source_id) if fixture_enabled else None
        content_fixture = _fixture_for_request(
            request, source_entry, content_fixture_map
        )
        if fixture is not None:
            doc = _make_fixture_document(request, source_entry, fixture)
            fixture_loaded_source_ids.append(request.source_id)
        elif content_fixture is not None:
            doc = _fetch_fixture_content_document(
                request, source_entry, content_fixture, fetch_config
            )
            fixture_loaded_source_ids.append(request.source_id)
        elif live:
            doc = _fetch_live_document_with_providers(
                request, source_entry, policy, fetch_config
            )
        else:
            doc = _make_offline_stub_document(request, source_entry)
        progress_status = "completed" if doc.fetch_status != "fetch_failed" else "error"
        if get_env("PIPELINE_MODE") == "evidence":
            if doc.budget_exhausted_kind or doc.acquisition_status == "budget_exhausted":
                progress_status = "deferred"
            elif doc.acquisition_incomplete:
                progress_status = "partial"
            elif not doc.content_readable or doc.fetch_status in {"fetch_failed", "failed"}:
                progress_status = "error"
        emit_workflow_progress(
            "content_fetch_and_parse",
            "content fetch completed" if progress_status == "completed" else "content fetch " + progress_status,
            {
                "source_id": doc.source_id,
                "fetch_status": doc.fetch_status,
                "acquisition_status": doc.acquisition_status,
                "acquisition_incomplete": doc.acquisition_incomplete,
                "budget_exhausted_kind": doc.budget_exhausted_kind,
                "fetch_provider": doc.fetch_provider,
                "parse_status": doc.parse_status,
                "parser_used": doc.parser_used,
                "text_char_count": doc.text_char_count,
                "table_count": doc.table_count,
            },
            status=progress_status,
        )
        return doc, fixture is not None or content_fixture is not None

    def _request_for_source_entry(entry: dict) -> ContentFetchRequest:
        final_decision = entry.get("final_screening_decision") or ""
        fetch_purpose = policy.fetch_purpose_by_decision.get(final_decision, "unknown")
        if _is_context_only_entry(entry, role_policy):
            fetch_purpose = "context_grounding"
        url = entry.get("canonical_url") or entry.get("url") or ""
        return ContentFetchRequest(
            source_id=entry.get("source_id", ""),
            url=url,
            canonical_url=entry.get("canonical_url") or url,
            publisher=entry.get("publisher"),
            source_type=entry.get("source_type"),
            source_role=entry.get("source_role"),
            **_source_provenance(entry),
            final_screening_decision=final_decision,
            fetch_purpose=fetch_purpose,
            priority=entry.get("priority"),
            live_fetch_enabled=live,
        )

    def _selected_primary_target_source_ids() -> set[str]:
        result: set[str] = set()
        for item in selection_manifest:
            source_id = item.get("source_id")
            if not source_id or not item.get("selected_for_fetch"):
                continue
            entry = registry_by_id.get(source_id) or {}
            if item.get("must_fetch") or _is_search_verified_target_entry(entry):
                result.add(source_id)
        return result

    def _has_usable_document_for_source_ids(source_ids: set[str]) -> bool:
        for document in documents:
            source_id = _document_attr(document, "source_id")
            if source_id not in source_ids:
                continue
            if _document_is_usable_for_task_collection(
                document,
                registry_by_id.get(source_id) or {},
                state,
            ):
                return True
        return False

    def _coverage_requirement_ids_for_source(source_id: str) -> set[str]:
        entry = registry_by_id.get(source_id) or {}
        return {
            str(value)
            for value in (entry.get("coverage_requirement_ids") or [])
            if value
        }

    def _primary_requirements_needing_fallback(
        primary_ids: set[str],
    ) -> tuple[set[str], bool]:
        attempted_requirement_ids: set[str] = set()
        usable_requirement_ids: set[str] = set()
        attempted_without_requirement = False
        usable_without_requirement = False
        for document in documents:
            source_id = str(_document_attr(document, "source_id") or "")
            if source_id not in primary_ids:
                continue
            requirement_ids = _coverage_requirement_ids_for_source(source_id)
            usable = _document_is_usable_for_task_collection(
                document,
                registry_by_id.get(source_id) or {},
                state,
            )
            if requirement_ids:
                attempted_requirement_ids.update(requirement_ids)
                if usable:
                    usable_requirement_ids.update(requirement_ids)
                continue
            attempted_without_requirement = True
            if usable:
                usable_without_requirement = True
        missing_requirement_ids = attempted_requirement_ids - usable_requirement_ids
        fallback_needed = bool(missing_requirement_ids) or (
            attempted_without_requirement and not usable_without_requirement
        )
        return missing_requirement_ids, fallback_needed

    def _candidate_matches_target_requirements(
        entry: dict,
        required_ids: set[str],
    ) -> bool:
        if not required_ids:
            return True
        candidate_ids = {
            str(value)
            for value in (entry.get("coverage_requirement_ids") or [])
            if value
        }
        if candidate_ids:
            return bool(candidate_ids & required_ids)
        if not _is_task_record_collection_candidate_entry(entry):
            return False
        target_status = str(entry.get("target_fit_status") or "").strip().lower()
        if target_status in {"temporal_mismatch", "geography_mismatch", "excluded"}:
            return False
        task_location = _task_location_for_geography_check(state)
        if _task_location_is_subnational(task_location):
            return _source_or_document_mentions_task_location(None, entry, task_location)
        return True

    def _mark_fallback_selected(source_id: str) -> None:
        for item in selection_manifest:
            if item.get("source_id") != source_id:
                continue
            old_reason = item.get("skip_reason")
            if old_reason:
                skip_counts[old_reason] = max(0, int(skip_counts.get(old_reason, 0)) - 1)
            item["selected_for_fetch"] = True
            item["skip_reason"] = None
            item["fallback_after_target_unusable"] = True
            return

    def _append_selected_manifest_for_discovered_link(
        entry: dict,
        parent_source_id: str,
        selection_reason: str,
    ) -> None:
        selection_manifest.append(
            {
                "source_id": entry.get("source_id"),
                "canonical_url": entry.get("canonical_url") or entry.get("url"),
                "domain": _domain_for_entry(entry),
                "discovery_method": entry.get("discovery_method"),
                "source_role_final": entry.get("source_role_final"),
                "fetch_bucket": _fetch_bucket(entry),
                "credibility_score": entry.get("credibility_score"),
                "credibility_level": entry.get("credibility_level"),
                "must_fetch": bool(entry.get("must_fetch")),
                "must_fetch_reason": entry.get("must_fetch_reason"),
                "coverage_requirement_ids": entry.get("coverage_requirement_ids") or [],
                "selected_for_fetch": True,
                "skip_reason": None,
                "fallback_after_target_unusable": True,
                "fallback_link_discovery": True,
                "parent_source_id": parent_source_id,
                "selection_reason": selection_reason,
            }
        )

    def _fetch_task_data_links_for_fallback_document(
        document: Document,
        parent_entry: dict,
        required_ids: set[str],
        fetched_source_ids: set[str],
    ) -> None:
        if len(fallback_link_selected_source_ids) >= _env_int(
            "DIRECT_FALLBACK_LINK_FETCH_LIMIT",
            3,
        ):
            return
        if _document_looks_like_error_page(document):
            return
        if str(_document_attr(document, "fetch_status") or "") == "fetch_failed":
            return

        parent_source_id = str(parent_entry.get("source_id") or "")
        candidates = _extract_task_data_link_candidates(
            document,
            parent_entry,
            state=state,
            required_ids=required_ids,
        )
        for rank, candidate in enumerate(candidates, start=1):
            if len(fallback_link_selected_source_ids) >= _env_int(
                "DIRECT_FALLBACK_LINK_FETCH_LIMIT",
                3,
            ):
                break
            url = str(candidate.get("url") or "")
            if not url:
                continue
            source_id = f"{parent_source_id}_link_{rank}"
            if source_id in registry_by_id or source_id in fetched_source_ids:
                continue
            entry = dict(parent_entry)
            entry.update(
                {
                    "source_id": source_id,
                    "canonical_url": url,
                    "url": url,
                    "title": f"Task data link from {parent_entry.get('title') or parent_source_id}",
                    "discovery_method": "fallback_link_discovery",
                    "parent_source_id": parent_source_id,
                    "parent_canonical_url": (
                        _document_attr(document, "canonical_url")
                        or _document_attr(document, "url")
                        or parent_entry.get("canonical_url")
                    ),
                    "fallback_link_selection_score": candidate.get("score"),
                    "fallback_link_selection_reasons": candidate.get(
                        "selection_reasons"
                    )
                    or [],
                    "target_fit_status": "task_record_collection_candidate",
                    "target_verification_status": "candidate_task_record_source",
                    "triage_role": "task_record_collection_candidate",
                    "source_role_final": "collection",
                    "final_screening_decision": "include_for_content_fetch",
                    "status": "ready_for_content_fetch",
                    "ready_for_content_fetch": True,
                    "coverage_requirement_ids": list(
                        required_ids
                        or set(parent_entry.get("coverage_requirement_ids") or [])
                    ),
                }
            )
            registry.append(entry)
            registry_by_id[source_id] = entry
            request = _request_for_source_entry(entry)
            child_doc, loaded_from_fixture = _fetch_document_for_request(request, entry)
            fetch_requests.append(request)
            executed_fetch_requests.append(request)
            documents.append(child_doc)
            fetched_source_ids.add(source_id)
            if loaded_from_fixture:
                fixture_loaded_source_ids.append(request.source_id)
            fallback_link_selected_source_ids.append(source_id)
            fallback_link_selected_urls.append(url)
            selection_reason = "; ".join(candidate.get("selection_reasons") or [])
            _append_selected_manifest_for_discovered_link(
                entry,
                parent_source_id,
                selection_reason,
            )
            fallback_link_rows.append(
                {
                    "parent_source_id": parent_source_id,
                    "parent_url": entry.get("parent_canonical_url"),
                    "child_source_id": source_id,
                    "child_url": url,
                    "score": candidate.get("score"),
                    "selection_reasons": candidate.get("selection_reasons") or [],
                    "fetch_status": child_doc.fetch_status,
                    "parse_status": child_doc.parse_status,
                    "quality_status": child_doc.quality_status,
                    "usable_for_task_collection": _document_is_usable_for_task_collection(
                        child_doc,
                        entry,
                        state,
                    ),
                }
            )

    def _fetch_task_level_fallbacks_after_unusable_target() -> list[str]:
        if collection_mode != "direct_collection":
            return []
        primary_ids = _selected_primary_target_source_ids()
        if not primary_ids:
            return []
        primary_documents = [
            document
            for document in documents
            if _document_attr(document, "source_id") in primary_ids
        ]
        if not primary_documents:
            return []

        required_ids, fallback_needed = _primary_requirements_needing_fallback(
            primary_ids
        )
        if not fallback_needed:
            return []

        fetched_source_ids = {request.source_id for request in executed_fetch_requests}
        fallback_limit = _env_int("DIRECT_FALLBACK_FETCH_LIMIT", 3)
        selected_source_ids: list[str] = []
        for entry in sorted(registry, key=_fetch_selection_sort_key):
            source_id = str(entry.get("source_id") or "")
            if not source_id or source_id in fetched_source_ids:
                continue
            if not (
                _is_search_verified_target_entry(entry)
                or _is_task_record_collection_candidate_entry(entry)
            ):
                continue
            if not _candidate_matches_target_requirements(entry, required_ids):
                continue
            request = _request_for_source_entry(entry)
            doc, loaded_from_fixture = _fetch_document_for_request(request, entry)
            fetch_requests.append(request)
            executed_fetch_requests.append(request)
            documents.append(doc)
            if loaded_from_fixture:
                fixture_loaded_source_ids.append(request.source_id)
            _mark_fallback_selected(source_id)
            selected_source_ids.append(source_id)
            if not _document_looks_like_error_page(doc):
                _fetch_task_data_links_for_fallback_document(
                    doc,
                    entry,
                    required_ids,
                    fetched_source_ids,
                )
            if len(selected_source_ids) >= fallback_limit:
                break
        return selected_source_ids

    def _fetch_related_links_from_high_trust_event_pages() -> list[str]:
        if not _env_flag("EVENT_LINK_EXPANSION_ENABLED", default=True):
            return []
        limit = max(0, _env_int("EVENT_LINK_EXPANSION_LIMIT", 8))
        if limit == 0:
            return []
        fetched_urls = {
            str(
                _document_attr(document, "canonical_url")
                or _document_attr(document, "url")
                or ""
            )
            for document in documents
        }
        registry_urls = {
            str(entry.get("canonical_url") or entry.get("url") or "")
            for entry in registry
        }
        parent_documents = list(documents)
        selected_ids: list[str] = []
        for document in parent_documents:
            if len(selected_ids) >= limit or _fetch_node_time_budget_exhausted():
                break
            parent_source_id = str(_document_attr(document, "source_id") or "")
            parent_entry = registry_by_id.get(parent_source_id) or {}
            if not parent_source_id or _document_looks_like_error_page(document):
                continue
            if str(_document_attr(document, "parse_status") or "").startswith(
                "parse_failed"
            ):
                continue
            if not _is_high_trust_event_link_parent(parent_entry, document, state):
                continue
            candidates = _extract_event_link_candidates(
                document,
                parent_entry,
                state=state,
            )
            for rank, candidate in enumerate(candidates, start=1):
                if len(selected_ids) >= limit or _fetch_node_time_budget_exhausted():
                    break
                url = str(candidate.get("url") or "")
                if not url or url in fetched_urls or url in registry_urls:
                    continue
                source_id = f"{parent_source_id}_event_link_{rank}"
                suffix = 1
                while source_id in registry_by_id:
                    suffix += 1
                    source_id = f"{parent_source_id}_event_link_{rank}_{suffix}"
                entry = dict(parent_entry)
                same_domain = bool(candidate.get("same_domain"))
                if not same_domain:
                    entry.update(
                        {
                            "publisher": urlsplit(url).netloc,
                            "actual_publisher": None,
                            "source_type": "institutional_event_source",
                            "source_type_final": "institutional_event_source",
                            "credibility_level": "medium",
                            "credibility_score": 0.7,
                            "source_identity_unverified": True,
                        }
                    )
                entry.update(
                    {
                        "source_id": source_id,
                        "canonical_url": url,
                        "url": url,
                        "title": candidate.get("link_text")
                        or f"Related event report from {parent_entry.get('title') or parent_source_id}",
                        "discovery_method": "event_link_expansion",
                        "parent_source_id": parent_source_id,
                        "parent_canonical_url": (
                            _document_attr(document, "canonical_url")
                            or _document_attr(document, "url")
                            or parent_entry.get("canonical_url")
                        ),
                        "event_link_selection_score": candidate.get("score"),
                        "event_link_selection_reasons": candidate.get(
                            "selection_reasons"
                        )
                        or [],
                        "target_fit_status": "task_record_collection_candidate",
                        "target_verification_status": "candidate_task_record_source",
                        "triage_role": "task_record_collection_candidate",
                        "source_role_final": "collection",
                        "final_screening_decision": "include_for_content_fetch",
                        "status": "ready_for_content_fetch",
                        "ready_for_content_fetch": True,
                        "source_exact_page_status": "event_page_found",
                    }
                )
                registry.append(entry)
                registry_by_id[source_id] = entry
                registry_urls.add(url)
                request = _request_for_source_entry(entry)
                child_doc, loaded_from_fixture = _fetch_document_for_request(
                    request,
                    entry,
                )
                fetch_requests.append(request)
                executed_fetch_requests.append(request)
                documents.append(child_doc)
                fetched_urls.add(url)
                if loaded_from_fixture:
                    fixture_loaded_source_ids.append(request.source_id)
                selected_ids.append(source_id)
                event_link_selected_source_ids.append(source_id)
                event_link_selected_urls.append(url)
                selection_manifest.append(
                    {
                        "source_id": source_id,
                        "canonical_url": url,
                        "domain": _domain_for_entry(entry),
                        "discovery_method": "event_link_expansion",
                        "source_role_final": "collection",
                        "fetch_bucket": _fetch_bucket(entry),
                        "credibility_score": entry.get("credibility_score"),
                        "credibility_level": entry.get("credibility_level"),
                        "selected_for_fetch": True,
                        "skip_reason": None,
                        "event_link_expansion": True,
                        "parent_source_id": parent_source_id,
                        "selection_reason": "; ".join(
                            candidate.get("selection_reasons") or []
                        ),
                    }
                )
                event_link_rows.append(
                    {
                        "parent_source_id": parent_source_id,
                        "parent_url": entry.get("parent_canonical_url"),
                        "child_source_id": source_id,
                        "child_url": url,
                        "score": candidate.get("score"),
                        "selection_reasons": candidate.get("selection_reasons") or [],
                        "fetch_status": child_doc.fetch_status,
                        "parse_status": child_doc.parse_status,
                        "quality_status": child_doc.quality_status,
                    }
                )
        return selected_ids

    universal_resources = get_env("PIPELINE_MODE") == "evidence"
    resource_link_rows: list[dict] = []
    resource_seen_urls: set[str] = set()
    resource_link_limit = (None if frontier else max(0, int(fetch_config["max_total_sources"] if fetch_config.get("max_total_sources") is not None else 10)))
    for limit_name in ("EVENT_LINK_EXPANSION_LIMIT", "RESOURCE_LINK_EXPANSION_LIMIT"):
        try:
            limit_value = get_env(limit_name)
            if limit_value is not None and (not frontier or limit_name == "RESOURCE_LINK_EXPANSION_LIMIT"):
                cap = max(0, int(limit_value))
                resource_link_limit = cap if resource_link_limit is None else min(resource_link_limit, cap)
        except ValueError:
            pass

    def _queue_task_resources(document: Document, parent: dict, after_index: int) -> None:
        from ..resource_discovery import (
            canonical_resource_url, resource_source_entry, task_resource_candidates,
        )
        if not universal_resources or allowlist is not None:
            return
        if not _env_flag("EVENT_LINK_EXPANSION_ENABLED", default=True):
            return
        limit = resource_link_limit
        depth = int(parent.get("resource_link_depth") or 0) + 1
        if depth > 3 or (limit is not None and len(resource_link_rows) >= limit):
            return
        for fetched in documents:
            for url in (fetched.url, fetched.canonical_url, fetched.final_url):
                resource_seen_urls.add(canonical_resource_url(url))
        pending_downloads, pending_navigation = [], []
        existing_by_url = {canonical_resource_url(e.get("canonical_url") or e.get("url")): e
                           for e in registry}
        resource_issues = []
        candidates = task_resource_candidates(document.model_dump(), parent, state, diagnostics=resource_issues)
        if resource_issues:
            document.metadata['resource_discovery_issues'] = resource_issues
        for candidate in candidates:
            if limit is not None and len(resource_link_rows) >= limit:
                break
            terminal = candidate["resource_type"] != "html"
            if depth > 2 and not terminal:
                continue
            url = candidate["url"]
            if url in resource_seen_urls:
                continue
            domain = urlsplit(url).hostname or ""
            if _domain_matches(domain, list(fetch_config.get("domain_blocklist") or [])):
                continue
            allowed = list(fetch_config.get("domain_allowlist") or [])
            if allowed and not _domain_matches(domain, allowed):
                continue
            entry = existing_by_url.get(url)
            if entry is not None:
                # Promote an already selected request without creating another
                # identity or bypassing an explicit upstream exclusion.
                request = next((r for r in (fetch_requests if frontier else fetch_requests[after_index + 1:])
                                if canonical_resource_url(r.url) == url), None)
                if request is None:
                    if not frontier or entry.get('blocked_from_fetch') or entry.get('source_role_final') == 'excluded':
                        continue
                    if _classify_skip_reason(entry, policy, allowlist, collection_mode, role_policy, fetch_config):
                        continue
                    request = _request_for_source_entry(entry)
                    fetch_requests.append(request)
                if terminal and not frontier:
                    fetch_requests.remove(request)
                entry["resource_link_depth"] = depth
            else:
                entry = resource_source_entry(candidate, parent=parent, depth=depth, state=state)
                registry.append(entry)
                registry_by_id[entry["source_id"]] = entry
                existing_by_url[url] = entry
                request = _request_for_source_entry(entry)
                selection_manifest.append({
                    "source_id": entry["source_id"], "canonical_url": url,
                    "domain": domain, "discovery_method": "task_resource_link",
                    "source_role_final": entry["source_role_final"],
                    "selected_for_fetch": True, "skip_reason": None, "must_fetch": False,
                    "parent_source_id": parent.get("source_id"), "resource_link_depth": depth,
                    "selection_reason": "; ".join(candidate["selection_reasons"]),
                })
            resource_seen_urls.add(url)
            if frontier:
                from ..acquisition_scheduling import acquisition_priority
                if request not in fetch_requests:
                    fetch_requests.append(request)
                frontier.enqueue(target_id=url, url=url, source_id=entry['source_id'],
                    priority=acquisition_priority(entry, state=state), payload={'entry':entry, 'request':request.model_dump()})
            elif terminal:
                pending_downloads.append(request)
            elif request not in fetch_requests:
                pending_navigation.append(request)
            resource_link_rows.append({
                "parent_source_id": parent.get("source_id"), "child_source_id": entry["source_id"],
                "child_url": url, "resource_link_depth": depth,
                "resource_type": candidate["resource_type"], "score": candidate["score"],
                "link_provenance": candidate["provenance"],
            })
        if frontier:
            return
        # Legacy strict scheduling retains its existing local limits.
        # Downloads remain first. If the existing queue would consume every
        # remaining source slot, give one new navigation child a chance within
        # that same budget; other archives stay behind the selected seeds.
        remaining_slots = max(0, int(fetch_config["max_total_sources"] if fetch_config.get("max_total_sources") is not None else 10)
                              - len(executed_fetch_requests))
        queued_ahead = len(fetch_requests) - after_index - 1 + len(pending_downloads)
        immediate_navigation = []
        if (pending_navigation and queued_ahead >= remaining_slots
                and remaining_slots > len(pending_downloads)):
            immediate_navigation.append(pending_navigation.pop(0))
        fetch_requests[after_index + 1:after_index + 1] = pending_downloads + immediate_navigation
        fetch_requests.extend(pending_navigation)

    if frontier:
        # A crash can occur after bytes are persisted but before their links are
        # registered. Re-expand saved pages idempotently, without another fetch.
        for restored_document in list(documents):
            if restored_document.content_readable:
                _queue_task_resources(restored_document, registry_by_id.get(restored_document.source_id) or {}, -1)
        from ..acquisition_scheduling import acquisition_priority, save_frontier_result, rerank_frontier
        from ..resource_discovery import canonical_resource_url
        for request in fetch_requests:
            entry = registry_by_id.get(request.source_id) or {}
            target_id = canonical_resource_url(request.canonical_url or request.url)
            frontier.enqueue(target_id=target_id, url=request.url, source_id=request.source_id,
                priority=acquisition_priority(entry, state=state), payload={'entry':entry, 'request':request.model_dump()})
        rerank_frontier(adaptive_runtime, registry, [doc.model_dump() for doc in documents], state=state)
    active_job = None
    soft_checkpoint = int(((adaptive_runtime.config.get('universal') or {}).get('budget_policy') or {}).get('soft_source_target',50)) if frontier else 0
    last_soft_checkpoint = 0
    default_acquisition_strategy = ({**adaptive_runtime.config,
        **((adaptive_runtime.config.get("universal") or {}).get("acquisition") or {}),
        **fetch_config}.get("acquisition_strategy") or "native") if frontier else "native"
    def _requests():
        nonlocal active_job
        if not frontier:
            yield from enumerate(fetch_requests)
            return
        index = 0
        while True:
            # Adaptive wall time bounds one batch, not the session allowance.
            # Leave unclaimed targets pending so recovery can continue them.
            # Permit one dispatch even if local preparation consumed the slice.
            if index and _fetch_node_time_budget_exhausted():
                emit_workflow_progress('content_fetch_and_parse',
                    'acquisition batch time reached; remaining targets stay pending',
                    {'processed_targets': index, 'frontier_counts': frontier.snapshot()['counts']})
                break
            active_job = frontier.claim_next(default_strategy=default_acquisition_strategy)
            if active_job is None:
                break
            entry = active_job['payload'].get('entry') or {}
            if entry.get('source_id') and entry['source_id'] not in registry_by_id:
                registry.append(entry)
                registry_by_id[entry['source_id']] = entry
            yield index, ContentFetchRequest(**active_job['payload']['request'])
            index += 1

    for request_index, request in _requests():
        source_entry = registry_by_id.get(request.source_id) or {}
        if frontier:
            reason = _classify_skip_reason(source_entry, policy, allowlist, collection_mode, role_policy, fetch_config)
            if reason:
                frontier.finish(active_job['target_id'], status='blocked', reason=reason, operation_started=False)
                _mark_fetch_fail_open_skipped(request.source_id, reason)
                continue
        if universal_resources and request.url in {d.url for d in documents if d.request_success and d.content_readable and not d.acquisition_incomplete and d.acquisition_status == "readable"}:
            if frontier:
                frontier.finish(active_job['target_id'], status='completed', reason='same_session_content', operation_started=False)
            continue
        if not frontier and _fetch_node_time_budget_exhausted():
            _mark_fetch_fail_open_skipped(
                request.source_id,
                "content_fetch_node_time_budget_exhausted",
            )
            continue
        source_entry = registry_by_id.get(request.source_id) or {}
        if (universal_resources and not frontier and
                len(executed_fetch_requests) >= int(fetch_config["max_total_sources"] if fetch_config.get("max_total_sources") is not None else 10) and
                _universal_priority_context(request, source_entry) is None):
            _mark_fetch_fail_open_skipped(
                request.source_id, "max_total_sources_reached_after_resource_priority"
            )
            continue
        before_used = adaptive_runtime.ledger.snapshot()['used'].get('http_requests', 0) if frontier else 0
        try:
            doc, loaded_from_fixture = _fetch_document_for_request(request, source_entry)
        except Exception as exc:
            from ..session_runtime import ResumeMismatch, BudgetExceeded
            if not frontier or isinstance(exc, ResumeMismatch):
                raise
            doc = Document(source_id=request.source_id, url=request.url, canonical_url=request.canonical_url,
                acquisition_status='request_error', fetch_status='failed', parse_status='not_started',
                fetch_error=type(exc).__name__ + ': ' + str(exc), request_success=False, content_readable=False)
            if isinstance(exc, BudgetExceeded):
                doc.acquisition_status = 'budget_exhausted'
                doc.budget_exhausted_kind = exc.kind
            loaded_from_fixture = False
        if universal_resources and doc.content_readable:
            from .source_screening import assess_source_task_fit
            assessment = assess_source_task_fit(source_entry, state, document=doc.model_dump())
            source_entry.update(assessment)
            for name, value in assessment.items():
                if name in Document.model_fields:
                    setattr(doc, name, value)
        if frontier:
            used = adaptive_runtime.ledger.snapshot()['used']
            started = used.get('http_requests',0) > before_used
            deferred = doc.acquisition_status == 'budget_exhausted'
            result_ref = save_frontier_result(adaptive_runtime, active_job['target_id'],
                document=doc.model_dump(), entry=source_entry, request=request.model_dump()) if not deferred or doc.raw_artifact_path else None
            frontier.finish(active_job['target_id'], status='budget_deferred' if deferred else 'completed' if (doc.content_readable and not doc.acquisition_incomplete and doc.acquisition_status == 'readable') else 'failed',
                reason=doc.budget_exhausted_kind if deferred else doc.fetch_error or doc.acquisition_status,
                operation_started=started, result_ref=result_ref)
            if deferred and not doc.raw_artifact_path:
                _mark_fetch_fail_open_skipped(request.source_id, 'budget_deferred:' + str(doc.budget_exhausted_kind))
                continue
        documents.append(doc)
        if not frontier or started or loaded_from_fixture:
            executed_fetch_requests.append(request)
        if loaded_from_fixture:
            fixture_loaded_source_ids.append(request.source_id)
        _queue_task_resources(doc, source_entry, request_index)
        if frontier:
            # Refresh both newly discovered and already queued siblings before
            # the next claim. Soft checkpoints report progress, not stale scores.
            assessment = rerank_frontier(adaptive_runtime, registry,
                [{'source_id': d.source_id, 'content_readable': d.content_readable}
                 for d in documents], state=state)
            if soft_checkpoint and used.get('source_targets',0) // soft_checkpoint > last_soft_checkpoint:
                last_soft_checkpoint = used.get('source_targets',0) // soft_checkpoint
                emit_workflow_progress('content_fetch_and_parse','acquisition soft checkpoint: continue ranked pending targets',assessment)

        should_retry_alias = (
            not frontier and (doc.fetch_status == "fetch_failed" or _document_looks_like_error_page(doc))
        )
        if not should_retry_alias:
            continue

        for alias_url in _official_alias_urls_for_retry(request, source_entry):
            if _fetch_node_time_budget_exhausted():
                _mark_fetch_fail_open_skipped(
                    alias_url,
                    "content_fetch_node_time_budget_exhausted",
                )
                break
            alias_fetch_attempted_count += 1
            alias_request = _request_for_alias(request, alias_url)
            alias_doc, alias_loaded_from_fixture = _fetch_document_for_request(
                alias_request,
                source_entry,
            )
            documents.append(alias_doc)
            executed_fetch_requests.append(alias_request)
            if alias_loaded_from_fixture:
                fixture_loaded_source_ids.append(alias_request.source_id)
            if (
                alias_doc.fetch_status == "fetch_failed"
                or _document_looks_like_error_page(alias_doc)
            ):
                alias_fetch_unusable_count += 1
                continue
            alias_fetch_success_count += 1
            break

    # evidence already expanded current-page resources within the budgeted queue;
    # legacy post-fetch expansion must not create a second, unbounded frontier.
    event_link_expansion_source_ids = ([] if universal_resources else
                                       _fetch_related_links_from_high_trust_event_pages())
    second_pass_fallback_source_ids = ([] if universal_resources else
                                      _fetch_task_level_fallbacks_after_unusable_target())

    document_dicts = [d.model_dump() for d in documents]
    (
        registry,
        source_coverage_requirements,
        source_coverage_audit,
    ) = annotate_source_coverage(registry, state, documents=document_dicts)
    source_identity_assessments = list(
        state.get("source_identity_assessments") or []
    )
    source_identity_summary = state.get("source_identity_summary") or {}
    if document_dicts:
        llm_identity_enabled = _env_flag("ENABLE_LLM_SOURCE_IDENTITY")
        post_fetch_enabled = _env_flag(
            "LLM_SOURCE_IDENTITY_POST_FETCH",
            default=True,
        )
        (
            registry,
            source_identity_assessments,
            source_identity_summary,
        ) = enrich_source_identity_registry_post_fetch(
            registry,
            source_identity_assessments,
            document_dicts,
            collection_spec=state.get("collection_spec"),
            llm_enabled=bool(llm_identity_enabled and post_fetch_enabled),
            max_sources=_parse_positive_int_env(
                "LLM_SOURCE_IDENTITY_MAX_SOURCES"
            ),
            require_llm=_env_flag("LLM_SOURCE_IDENTITY_REQUIRE_LLM"),
            allow_deterministic_fallback=_env_flag(
                "LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK",
                default=True,
            ),
        )
        identity_by_source = {
            row.get("source_id"): row for row in registry if row.get("source_id")
        }
        for doc in document_dicts:
            identity = identity_by_source.get(doc.get("source_id"))
            if not identity:
                continue
            for key in (
                "actual_publisher",
                "actual_publisher_normalized",
                "source_type_final",
                "source_independence_group",
                "claim_support_role",
                "recommended_source_role",
                "recommended_fetch_use",
                "recommended_extraction_use",
                "likely_syndicated_or_aggregated",
                "upstream_source_mentions",
            ):
                doc[key] = identity.get(key)
    request_dicts = [r.model_dump() for r in executed_fetch_requests]

    fetch_status_counts = dict(Counter(d.fetch_status or "unknown" for d in documents))
    fetch_provider_counts = dict(
        Counter(d.fetch_provider or "unknown" for d in documents)
    )
    failure_counter: Counter = Counter()
    for d in documents:
        seen_failures: set[tuple[str, str]] = set()
        for attempt in d.provider_attempts or []:
            if attempt.get("success"):
                continue
            provider = str(attempt.get("provider") or "unknown")
            error = str(attempt.get("error") or "")
            key = (provider, error)
            if key in seen_failures:
                continue
            seen_failures.add(key)
            failure_counter[provider] += 1
    external_fetch_failure_counts = dict(failure_counter)
    document_type_counts = dict(Counter(d.document_type or "unknown" for d in documents))
    parser_status_counts = dict(Counter(d.parse_status or "unknown" for d in documents))
    parser_used_counts = dict(Counter(d.parser_used or "unknown" for d in documents))
    content_type_counts = dict(Counter(d.content_type or "unknown" for d in documents))
    total_table_count = sum(int(d.table_count or len(d.tables or [])) for d in documents)
    total_text_char_count = sum(int(d.text_char_count or len(d.clean_text or "")) for d in documents)
    fetch_purpose_counts = dict(
        Counter(r.fetch_purpose or "unknown" for r in executed_fetch_requests)
    )
    selected_bucket_counter: Counter = Counter()
    for item in selection_manifest:
        if not item.get("selected_for_fetch"):
            continue
        bucket = item.get("fetch_bucket") or "unknown"
        selected_bucket_counter[bucket] += 1
        source_id = item.get("source_id")
        entry = registry_by_id.get(source_id) if source_id else None
        if entry and bucket == "validation":
            source_type = _source_type_for_entry(entry)
            publisher = str(
                entry.get("publisher") or entry.get("actual_publisher") or ""
            ).lower()
            url = str(entry.get("canonical_url") or entry.get("url") or "").lower()
            if (
                source_type in _OFFICIAL_SOURCE_TYPES
                or "cdc" in publisher
                or "department of health" in publisher
                or ".gov/" in url
            ):
                selected_bucket_counter["official_authority"] += 1
    selected_fetch_bucket_counts = dict(selected_bucket_counter)
    direct_fast_path_skipped_count = skip_counts.get(
        "direct_target_official_fast_path",
        0,
    )
    direct_fast_path_active = direct_fast_path_skipped_count > 0 or any(
        item.get("selected_for_fetch")
        and item.get("fetch_bucket") == "target_official_authority"
        for item in selection_manifest
    )
    must_fetch_selected_count = sum(
        1 for item in selection_manifest if item.get("selected_for_fetch") and item.get("must_fetch")
    )
    must_fetch_source_ids = [
        item.get("source_id")
        for item in selection_manifest
        if item.get("selected_for_fetch") and item.get("must_fetch")
    ]
    high_risk_fetch_source_count = sum(
        1
        for item in selection_manifest
        if item.get("selected_for_fetch")
        and (
            item.get("source_role_final") == "needs_human_review"
            or item.get("requires_human_review")
            or item.get("human_review_recommended")
        )
    )
    search_derived_registry = [e for e in registry if _is_search_derived_entry(e)]
    selected_search_derived_ids = [
        r.source_id for r in fetch_requests
        if r.discovery_method in _SEARCH_DERIVED_DISCOVERY_METHODS
    ]
    selected_search_verified_target_ids = [
        r.source_id
        for r in fetch_requests
        if r.source_id
        and _is_search_verified_target_entry(registry_by_id.get(r.source_id) or {})
    ]
    selected_task_candidate_ids = [
        r.source_id
        for r in fetch_requests
        if r.source_id
        and _is_task_record_collection_candidate_entry(
            registry_by_id.get(r.source_id) or {}
        )
    ]
    selected_generated_target_ids = [
        item.get("source_id")
        for item in selection_manifest
        if item.get("selected_for_fetch")
        and (
            registry_by_id.get(item.get("source_id")) or {}
        ).get("discovery_method") == "official_coverage_requirement"
    ]
    error_page_document_count = sum(
        1 for document in documents if _document_looks_like_error_page(document)
    )
    selected_target_source_ids = {
        item.get("source_id")
        for item in selection_manifest
        if item.get("selected_for_fetch")
        and (
            item.get("must_fetch")
            or _is_search_verified_target_entry(
                registry_by_id.get(item.get("source_id")) or {}
            )
        )
    }
    primary_target_source_ids = set(
        sid for sid in [*selected_generated_target_ids, *selected_search_verified_target_ids] if sid
    )
    fallback_task_candidate_ids = [
        sid
        for sid in selected_task_candidate_ids
        if sid and sid not in primary_target_source_ids
    ]
    usable_target_document_count = sum(
        1
        for document in documents
        if _document_attr(document, "source_id") in selected_target_source_ids
        and not _document_looks_like_error_page(document)
        and str(_document_attr(document, "parse_status") or "").startswith("parsed")
        and len(str(_document_attr(document, "clean_text") or "")) >= 80
    )
    usable_task_collection_documents = [
        document
        for document in documents
        if _document_is_usable_for_task_collection(
            document,
            registry_by_id.get(_document_attr(document, "source_id")) or {},
            state,
        )
    ]
    primary_target_documents = [
        document
        for document in documents
        if _document_attr(document, "source_id") in primary_target_source_ids
    ]
    primary_target_attempted = bool(primary_target_documents)
    usable_primary_target_document_count = sum(
        1
        for document in primary_target_documents
        if _document_is_usable_for_task_collection(
            document,
            registry_by_id.get(_document_attr(document, "source_id")) or {},
            state,
        )
    )
    primary_target_unusable_count = sum(
        1
        for document in primary_target_documents
        if not _document_is_usable_for_task_collection(
            document,
            registry_by_id.get(_document_attr(document, "source_id")) or {},
            state,
        )
    )
    primary_missing_requirement_ids, primary_requirement_fallback_needed = (
        _primary_requirements_needing_fallback(primary_target_source_ids)
        if primary_target_source_ids
        else (set(), False)
    )
    target_unusable_needs_fallback = (
        primary_target_attempted
        and primary_target_unusable_count > 0
        and (
            primary_requirement_fallback_needed
            or usable_primary_target_document_count == 0
        )
    )
    fallback_fetch_selected_source_ids = (
        fallback_task_candidate_ids
        if fallback_task_candidate_ids
        else [
            sid for sid in selected_search_verified_target_ids if sid
        ]
        if selected_generated_target_ids and selected_search_verified_target_ids
        else []
    )
    fallback_link_discovery_summary = {
        "link_discovery_enabled": collection_mode == "direct_collection",
        "candidate_parent_source_count": len(second_pass_fallback_source_ids),
        "link_fetch_attempted_count": len(fallback_link_selected_source_ids),
        "selected_child_source_ids": list(fallback_link_selected_source_ids),
        "selected_child_urls": list(fallback_link_selected_urls),
        "link_rows": list(fallback_link_rows),
    }
    event_link_expansion_summary = {
        "enabled": _env_flag("EVENT_LINK_EXPANSION_ENABLED", default=True),
        "limit": max(0, _env_int("EVENT_LINK_EXPANSION_LIMIT", 8)),
        "link_fetch_attempted_count": len(event_link_selected_source_ids),
        "selected_child_source_ids": list(event_link_selected_source_ids),
        "selected_child_urls": list(event_link_selected_urls),
        "link_rows": list(event_link_rows),
    }
    error_alias_urls: list[str] = []
    usable_target_alias_urls: list[str] = []
    usable_task_collection_urls: list[str] = []
    for document in documents:
        if _document_attr(document, "source_id") not in selected_target_source_ids:
            source_entry = registry_by_id.get(_document_attr(document, "source_id")) or {}
            if not _document_is_usable_for_task_collection(document, source_entry, state):
                continue
            url = str(
                _document_attr(document, "canonical_url")
                or _document_attr(document, "url")
                or ""
            )
            if url and url not in usable_task_collection_urls:
                usable_task_collection_urls.append(url)
            continue
        url = str(
            _document_attr(document, "canonical_url")
            or _document_attr(document, "url")
            or ""
        )
        if not url:
            continue
        if _document_looks_like_error_page(document):
            if url not in error_alias_urls:
                error_alias_urls.append(url)
            continue
        if (
            str(_document_attr(document, "parse_status") or "").startswith("parsed")
            and len(str(_document_attr(document, "clean_text") or "")) >= 80
            and url not in usable_target_alias_urls
        ):
            usable_target_alias_urls.append(url)
        source_entry = registry_by_id.get(_document_attr(document, "source_id")) or {}
        if _document_is_usable_for_task_collection(document, source_entry, state):
            if url and url not in usable_task_collection_urls:
                usable_task_collection_urls.append(url)
    skipped_search_derived_by_reason = {
        reason: count
        for reason, count in skip_counts.items()
        if reason
        in {
            "search_derived_fetch_disabled",
            "final_role_excluded",
            "final_role_search_endpoint",
            "needs_review_not_allowed",
            "final_role_not_allowed",
            "missing_source_type",
            "unsupported_url_scheme",
            "missing_credibility_score",
            "invalid_credibility_score",
            "credibility_score_below_threshold",
            "not_ready_for_content_fetch",
            "final_screening_decision_not_fetchable",
            "domain_blocklisted",
            "domain_not_allowlisted",
            "max_search_derived_sources_reached",
            "max_total_sources_reached",
            "fetch_cap_after_high_trust_floor",
            "content_fetch_node_time_budget_exhausted",
        }
    }
    skipped_validation_reserved_ids = [
        e.get("source_id")
        for e in registry
        if _is_validation_reserved_entry(e, collection_mode, role_policy)
    ]
    context_only_source_ids = sorted(
        {
            e.get("source_id")
            for e in registry
            if e.get("source_id") and _is_context_only_entry(e, role_policy)
        }
    )
    context_only_fetched_source_ids = sorted(
        {
            d.source_id
            for d in documents
            if d.source_id and d.source_id in set(context_only_source_ids)
        }
    )

    summary = {
        "collection_mode": collection_mode,
        "live_fetch_enabled": live,
        "fixture_documents_enabled": fixture_enabled,
        "content_fixture_map_enabled": bool(content_fixture_map),
        "fixture_document_count": len(fixture_loaded_source_ids),
        "fixture_source_ids": list(fixture_loaded_source_ids),
        "source_id_allowlist_enabled": allowlist is not None,
        "source_id_allowlist": sorted(allowlist) if allowlist else [],
        "search_derived_fetch_enabled": bool(
            fetch_config.get("fetch_search_derived_sources")
        ),
        "search_derived_input_count": len(search_derived_registry),
        "selected_search_derived_fetch_count": len(selected_search_derived_ids),
        "selected_search_derived_source_ids": selected_search_derived_ids,
        "generated_target_fetch_count": len(
            [sid for sid in selected_generated_target_ids if sid]
        ),
        "generated_target_source_ids": [
            sid for sid in selected_generated_target_ids if sid
        ],
        "search_verified_target_fetch_count": len(
            [sid for sid in selected_search_verified_target_ids if sid]
        ),
        "search_verified_target_fetch_source_ids": [
            sid for sid in selected_search_verified_target_ids if sid
        ],
        "target_unusable_needs_fallback": target_unusable_needs_fallback,
        "requirements_needing_fallback": sorted(primary_missing_requirement_ids),
        "fallback_search_executed": bool(fallback_fetch_selected_source_ids),
        "fallback_fetch_attempted": bool(fallback_fetch_selected_source_ids),
        "fallback_fetch_selected_source_ids": fallback_fetch_selected_source_ids,
        "second_pass_fallback_source_ids": second_pass_fallback_source_ids,
        "fallback_link_discovery_summary": fallback_link_discovery_summary,
        "event_link_expansion_source_ids": event_link_expansion_source_ids,
        "event_link_expansion_summary": event_link_expansion_summary,
        "alias_fetch_attempted_count": alias_fetch_attempted_count,
        "alias_fetch_success_count": alias_fetch_success_count,
        "alias_fetch_unusable_count": alias_fetch_unusable_count,
        "error_page_document_count": error_page_document_count,
        "usable_target_document_count": usable_target_document_count,
        "usable_task_collection_document_count": len(
            usable_task_collection_documents
        ),
        "usable_task_collection_urls": usable_task_collection_urls,
        "error_alias_urls": error_alias_urls,
        "usable_target_alias_urls": usable_target_alias_urls,
        "skipped_search_derived_fetch_disabled_count": skip_counts.get(
            "search_derived_fetch_disabled", 0
        ),
        "skipped_search_derived_fetch_limit_count": skip_counts.get(
            "max_search_derived_sources_reached", 0
        ),
        "skipped_search_derived_by_reason_counts": skipped_search_derived_by_reason,
        "max_search_derived_sources": fetch_config.get(
            "max_search_derived_sources"
        ),
        "max_total_sources": fetch_config.get("max_total_sources"),
        "max_node_seconds": fetch_config.get("max_node_seconds"),
        "min_credibility_score": fetch_config.get("min_credibility_score"),
        "allowed_final_roles": list(fetch_config.get("allowed_final_roles") or []),
        "allow_needs_review": bool(fetch_config.get("allow_needs_review")),
        "external_fetch_enabled": bool(fetch_config.get("external_fetch_enabled")),
        "external_fetch_provider_order": list(
            fetch_config.get("external_fetch_provider_order") or []
        ),
        "fetch_provider_counts": fetch_provider_counts,
        "external_fetch_failure_counts": external_fetch_failure_counts,
        "selected_fetch_bucket_counts": selected_fetch_bucket_counts,
        "direct_target_official_fast_path": direct_fast_path_active,
        "direct_target_official_fast_path_skipped_count": direct_fast_path_skipped_count,
        "source_coverage_requirement_count": len(source_coverage_requirements),
        "source_coverage_audit": source_coverage_audit,
        "must_fetch_selected_count": must_fetch_selected_count,
        "must_fetch_source_ids": [sid for sid in must_fetch_source_ids if sid],
        "high_risk_fetch_source_count": high_risk_fetch_source_count,
        "domain_allowlist": list(fetch_config.get("domain_allowlist") or []),
        "domain_blocklist": list(fetch_config.get("domain_blocklist") or []),
        "skipped_not_in_allowlist_count": skip_counts.get("not_in_source_id_allowlist", 0),
        "skipped_validation_reserved_count": skip_counts.get(
            "validation_reserved", 0
        ),
        "skipped_validation_reserved_source_ids": [
            sid for sid in skipped_validation_reserved_ids if sid
        ],
        "context_only_source_count": len(context_only_source_ids),
        "context_only_source_ids": context_only_source_ids,
        "context_only_fetched_source_ids": context_only_fetched_source_ids,
        "input_registry_count": len(registry),
        "fetch_request_count": len(fetch_requests),
        "executed_fetch_request_count": len(executed_fetch_requests),
        "document_count": len(documents),
        "fetch_fail_open_skipped_source_count": len(
            fetch_fail_open_skipped_source_ids
        ),
        "fetch_fail_open_skipped_source_ids": list(fetch_fail_open_skipped_source_ids),
        "fetch_fail_open_reason_counts": dict(fetch_fail_open_reason_counter),
        "fetch_status_counts": fetch_status_counts,
        "document_type_counts": document_type_counts,
        "parser_status_counts": parser_status_counts,
        "parser_used_counts": parser_used_counts,
        "content_type_counts": content_type_counts,
        "total_table_count": total_table_count,
        "total_text_char_count": total_text_char_count,
        "fetch_purpose_counts": fetch_purpose_counts,
        "skipped_deferred_count": skip_counts.get("deferred", 0)
        + skip_counts.get("not_fetchable", 0),
        "skipped_search_endpoint_count": skip_counts.get("search_endpoint", 0),
        "skipped_blocked_scheme_count": skip_counts.get("blocked_scheme", 0)
        + skip_counts.get("non_allowed_scheme", 0),
        "skipped_human_review_count": skip_counts.get("human_review", 0),
        "selection_manifest": selection_manifest,
    }

    fixture_summary = {
        "fixture_documents_enabled": fixture_enabled,
        "available_fixture_count": len(fixture_map),
        "loaded_fixture_count": len(fixture_loaded_source_ids),
        "loaded_fixture_source_ids": list(fixture_loaded_source_ids),
        "synthetic_fixture_notice": (
            "Fixture documents are synthetic and not real public health data."
        ),
    }
    parse_summary = {
        "document_count": len(documents),
        "parser_status_counts": parser_status_counts,
        "parser_used_counts": parser_used_counts,
        "content_type_counts": content_type_counts,
        "parsed_document_count": sum(
            count
            for status, count in parser_status_counts.items()
            if str(status).startswith("parsed")
        ),
        "parse_deferred_count": parser_status_counts.get("parse_deferred", 0)
        + parser_status_counts.get("pdf_parsing_deferred", 0),
        "parse_failed_count": parser_status_counts.get("parse_failed", 0)
        + parser_status_counts.get("fetch_failed", 0),
        "total_text_char_count": total_text_char_count,
        "total_table_count": total_table_count,
        "documents": [
            {
                "source_id": d.source_id,
                "canonical_url": d.canonical_url,
                "discovery_method": d.discovery_method,
                "source_role_final": d.source_role_final,
                "credibility_score": d.credibility_score,
                "credibility_level": d.credibility_level,
                "fetch_status": d.fetch_status,
                "parse_status": d.parse_status,
                "parser_used": d.parser_used,
                "content_type": d.content_type,
                "title": d.title,
                "published_date": d.published_date,
                "text_char_count": d.text_char_count,
                "table_count": d.table_count,
                "fetch_provider": d.fetch_provider,
                "provider_attempts": d.provider_attempts,
            }
            for d in documents
        ],
    }
    fetch_manifest = []
    document_by_source_id = {d.source_id: d for d in documents}
    source_by_id = {
        str(entry.get("source_id") or ""): entry
        for entry in registry
        if entry.get("source_id")
    }
    for item in selection_manifest:
        source_id = item.get("source_id")
        doc = document_by_source_id.get(source_id)
        source_entry = source_by_id.get(str(source_id or ""), {})
        usability_source_entry = {
            **source_entry,
            **{
                key: value
                for key, value in item.items()
                if value not in (None, "", [])
            },
        }
        enriched = dict(item)
        if doc is not None:
            doc_dict = doc.model_dump() if hasattr(doc, "model_dump") else dict(getattr(doc, "__dict__", {}))
            url = (
                enriched.get("url")
                or doc.url
                or source_entry.get("url")
                or doc.canonical_url
                or source_entry.get("canonical_url")
            )
            canonical_url = (
                enriched.get("canonical_url")
                or doc.canonical_url
                or source_entry.get("canonical_url")
                or url
            )
            http_status = doc.http_status_code
            if http_status is None and doc.fetch_status == "fetched":
                http_status = 200
            quality_status = doc.quality_status
            if not quality_status:
                if _document_looks_like_error_page(doc_dict):
                    quality_status = "unusable"
                elif str(doc.parse_status or "").lower().startswith("parsed") and int(doc.text_char_count or 0) >= 80:
                    quality_status = "usable"
                elif str(doc.parse_status or "").lower().startswith("parsed"):
                    quality_status = "partial"
            usable_for_task_collection, task_usability_reasons = _document_task_usability(
                doc_dict | {"quality_status": quality_status},
                usability_source_entry or enriched,
                state,
            )
            enriched.update(
                {
                    "url": url,
                    "canonical_url": canonical_url,
                    "fetch_status": doc.fetch_status,
                    "parse_status": doc.parse_status,
                    "parser_used": doc.parser_used,
                    "content_type": doc.content_type,
                    "http_status": http_status,
                    "http_status_code": doc.http_status_code,
                    "quality_status": quality_status,
                    "quality_issues": doc.quality_issues,
                    "text_char_count": doc.text_char_count,
                    "table_count": doc.table_count,
                    "fetch_provider": doc.fetch_provider,
                    "provider_attempts": doc.provider_attempts,
                    "source_role_final": (
                        doc.source_role_final
                        or enriched.get("source_role_final")
                        or source_entry.get("source_role_final")
                        or source_entry.get("source_role")
                    ),
                    "target_fit_status": enriched.get("target_fit_status")
                    or source_entry.get("target_fit_status"),
                    "triage_role": enriched.get("triage_role") or source_entry.get("triage_role"),
                    "disease_fit": enriched.get("disease_fit") or source_entry.get("disease_fit"),
                    "geography_fit": enriched.get("geography_fit") or source_entry.get("geography_fit"),
                    "date_fit": enriched.get("date_fit") or source_entry.get("date_fit"),
                    "coverage_requirement_ids": enriched.get("coverage_requirement_ids")
                    or source_entry.get("coverage_requirement_ids")
                    or [],
                    "usable_for_task_collection": usable_for_task_collection,
                    "usable_for_best_available_context": bool(
                        not usable_for_task_collection
                        and quality_status in {"usable", "partial"}
                        and not _is_context_only_entry(source_entry, state)
                    ),
                    "task_usability_reasons": task_usability_reasons,
                }
            )
        else:
            enriched.setdefault("url", source_entry.get("url") or source_entry.get("canonical_url"))
            enriched.setdefault("canonical_url", source_entry.get("canonical_url") or enriched.get("url"))
            enriched.setdefault("source_role_final", source_entry.get("source_role_final") or source_entry.get("source_role"))
            enriched.setdefault("target_fit_status", source_entry.get("target_fit_status"))
            enriched.setdefault("triage_role", source_entry.get("triage_role"))
            enriched.setdefault("disease_fit", source_entry.get("disease_fit"))
            enriched.setdefault("geography_fit", source_entry.get("geography_fit"))
            enriched.setdefault("date_fit", source_entry.get("date_fit"))
            enriched.setdefault("coverage_requirement_ids", source_entry.get("coverage_requirement_ids") or [])
            enriched.setdefault("usable_for_task_collection", False)
            enriched.setdefault("usable_for_best_available_context", False)
        fetch_manifest.append(enriched)
    fetch_failures_blocking = [
        item
        for item in fetch_manifest
        if item.get("must_fetch")
        and (
            str(item.get("fetch_status") or "") == "fetch_failed"
            or str(item.get("parse_status") or "") in {"parse_failed", "parse_deferred"}
        )
    ]
    target_official_fetch_plan = [
        item
        for item in fetch_manifest
        if item.get("must_fetch")
        or item.get("discovery_method") == "official_coverage_requirement"
    ]
    summary["resource_link_discovery"] = resource_link_rows
    summary["resource_link_selected_count"] = len(resource_link_rows)
    summary["resource_link_limit"] = resource_link_limit if universal_resources else None
    summary["source_coverage_audit"] = source_coverage_audit
    summary["coverage_status"] = source_coverage_audit.get("coverage_status")
    summary["target_official_fetch_plan_count"] = len(target_official_fetch_plan)
    summary["fetch_failures_blocking_count"] = len(fetch_failures_blocking)
    summary["fetch_failures_blocking_source_ids"] = [
        item.get("source_id") for item in fetch_failures_blocking if item.get("source_id")
    ]

    emit_workflow_progress(
        "content_fetch_and_parse",
        "content fetch summary ready",
        {
            "fetch_request_count": len(fetch_requests),
            "document_count": len(documents),
            "fetch_status_counts": fetch_status_counts,
            "fetch_provider_counts": fetch_provider_counts,
            "parser_status_counts": parser_status_counts,
            "total_text_char_count": total_text_char_count,
        },
    )

    trace = append_trace(
        state,
        node_name="content_fetch_and_parse",
        message=(
            f"Built {len(fetch_requests)} fetch requests, produced {len(documents)} "
            f"documents (live_fetch_enabled={live}, "
            f"fixture_documents_enabled={fixture_enabled}, "
            f"fixtures_loaded={len(fixture_loaded_source_ids)})."
        ),
        metadata=summary,
    )
    return {
        **({"acquisition_frontier": frontier.snapshot()} if frontier else {}),
        "content_fetch_requests": request_dicts,
        "documents": document_dicts,
        "source_registry": registry,
        "source_identity_assessments": source_identity_assessments,
        "source_identity_summary": source_identity_summary,
        "source_coverage_requirements": source_coverage_requirements,
        "source_coverage_audit": source_coverage_audit,
        "target_official_fetch_plan": target_official_fetch_plan,
        "must_fetch_sources": [
            {
                "source_id": row.get("source_id"),
                "canonical_url": row.get("canonical_url") or row.get("url"),
                "must_fetch_reason": row.get("must_fetch_reason"),
                "coverage_requirement_ids": row.get("coverage_requirement_ids") or [],
            }
            for row in registry
            if row.get("must_fetch")
        ],
        "fetch_failures_blocking": fetch_failures_blocking,
        "content_fetch_summary": summary,
        "document_parse_summary": parse_summary,
        "fetch_manifest": fetch_manifest,
        "fixture_document_summary": fixture_summary,
        "collection_trace": trace,
    }


def _post_fetch_readiness(doc: dict, disease_assessment: dict, quality_status: str) -> dict:
    universal = get_env("PIPELINE_MODE") == "evidence"
    title_value = ((doc.get("metadata") or {}).get("page_title") or (doc.get("metadata") or {}).get("page_heading")) if universal else doc.get("title")
    title = str(title_value or "").lower()
    url = str(doc.get("canonical_url") or doc.get("url") or "").lower()
    text = str(doc.get("clean_text") or "").lower()
    disease_status = str(disease_assessment.get("status") or "unknown")
    search_shell = bool(
        re.search(r"(?:^|/)search(?:[/?]|$)|[?&](?:q|query|search)=", url)
        or "search results" in title
        or "search results" in text
    )
    structured_signal = bool(doc.get("tables"))
    date_signal = bool(doc.get("published_date") or doc.get("reporting_period_start"))
    event_signal = bool(
        re.search(r"\b(?:case|patient|outbreak|cluster|confirmed|surveillance|incidence|death)\b", text)
    )
    if universal:
        from ..source_assertions import count_mentions
        event_signal = event_signal or bool(count_mentions(text))
    explicit_time_signal = bool(
        doc.get("reporting_period_start")
        or doc.get("reporting_period_end")
        or re.search(
            r"\b(?:19|20)\d{2}\b|\b(?:week|month|quarter)\s+\d{1,2}\b|"
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
            r"nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:,\s*(?:19|20)\d{2})?\b",
            text,
        )
    )
    specific_event_signal = bool(event_signal and explicit_time_signal)
    generic_background = bool(
        re.search(r"\b(?:about|what is|fact sheet|overview|background)\b", title)
        and not (structured_signal or specific_event_signal)
    )
    if disease_status == UNRELATED_DISEASE:
        return {
            "task_relevance_status": "wrong_disease",
            "page_identity_status": "search_shell" if search_shell else "wrong_disease_page",
            "extraction_readiness": "not_ready",
            "extraction_readiness_reason": "wrong disease for task; structured extraction is disabled",
        }
    if search_shell:
        return {
            "task_relevance_status": "target_disease_match" if disease_status == TARGET_DISEASE_MATCH else "unknown_task_relevance",
            "page_identity_status": "search_shell",
            "extraction_readiness": "not_ready",
            "extraction_readiness_reason": "search shell is navigational content, not an extraction page",
        }
    if generic_background:
        return {
            "task_relevance_status": "target_disease_match" if disease_status == TARGET_DISEASE_MATCH else "unknown_task_relevance",
            "page_identity_status": "generic_background",
            "extraction_readiness": "context_only",
            "extraction_readiness_reason": "generic background page lacks local event or structured-data signals",
        }
    if (universal and quality_status in {"usable", "partial"}
            and disease_status == AMBIGUOUS_DISEASE
            and (structured_signal or event_signal)):
        # A multi-disease report or a reference to another disease cannot decide
        # the identity of every local observation. Preserve page uncertainty;
        # each source-bound chunk must still prove target disease and data.
        return {
            "task_relevance_status": "unknown_task_relevance",
            "page_identity_status": "mixed_disease_page",
            "extraction_readiness": "local_assessment_required",
            "extraction_readiness_reason": "mixed disease page requires source-bound local evidence assessment",
        }
    if quality_status in {"usable", "partial"} and disease_status == TARGET_DISEASE_MATCH and (structured_signal or event_signal):
        signals = ", ".join(
            name for name, present in (("structured/table", structured_signal), ("date", date_signal), ("event", event_signal)) if present
        )
        return {
            "task_relevance_status": "target_disease_match",
            "page_identity_status": "task_specific_page",
            "extraction_readiness": "ready",
            "extraction_readiness_reason": f"task-aligned page with {signals} signals",
        }
    return {
        "task_relevance_status": "unknown_task_relevance",
        "page_identity_status": "unknown_page_identity",
        "extraction_readiness": "not_ready",
        "extraction_readiness_reason": "insufficient task-aligned page identity or data signals",
    }


def _content_routing_update(doc: dict, source: dict, context: dict, policy: dict) -> dict:
    """Correct advisory context routing after reading actual task evidence.

    This grants bounded extraction access, never trust or record qualification.
    Explicit exclusions/validation roles and policy restrictions remain binding.
    """
    roles = {str(item.get(key) or "") for item in (doc, source)
             for key in ("source_role", "source_role_final")}
    flags = set(_routing_flags(doc)) | set(_routing_flags(source))
    if (doc.get("source_id") in set(policy.get("context_only_source_ids") or [])
            or roles & {"excluded", "context_only", "validation", "validation_reserved", "validation_source", "search_endpoint"}
            or flags & {"context_only", "blocked_from_structured_extraction"}):
        return {}
    if not (roles & {"context", "context_source", "context_only"} or doc.get("fetch_purpose") == "context_grounding"):
        return {}
    if (doc.get("extraction_readiness") != "ready" or doc.get("quality_status") not in {"usable", "partial"}
            or doc.get("content_readable") is False or _document_looks_like_error_page(doc)):
        return {}
    body = str(doc.get("clean_text") or "")
    actual = assess_chunk_disease_relevance({"text": body, "title": ""}, context)
    if actual.get("status") != TARGET_DISEASE_MATCH or not re.search(r"\d", body):
        return {}
    return {"source_role": "data_source", "source_role_final": "collection", "fetch_purpose": "data_extraction"}


def document_quality_check(state: DataCollectionState) -> dict:
    """Annotate documents with quality status and surface issues."""

    policy = ContentFetchPolicy(**load_content_fetch_policy())
    documents = list(state.get("documents") or [])
    collection_mode = str(
        (state.get("structured_task") or {}).get("collection_mode")
        or (state.get("collection_spec") or {}).get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip()
    direct_collection = collection_mode == "direct_collection"

    quality_cfg = policy.document_quality or {}
    min_usable = int(quality_cfg.get("min_clean_text_chars_for_usable", 300))
    min_partial = int(quality_cfg.get("min_clean_text_chars_for_partial", 80))

    updated: list[dict] = []
    quality_counter: Counter = Counter()
    disease_status_counter: Counter = Counter()
    usable = partial = offline_stub = parse_deferred = unusable = 0
    not_task_relevant = 0
    context = build_disease_relevance_context(state)
    from ..evidence_qualification import evidence_qualification_enabled
    universal = evidence_qualification_enabled()
    routing_policy = load_source_role_policy() if universal else {}
    source_by_id = {row.get("source_id"): row for row in state.get("source_registry") or []}
    routing_updates = {}

    for doc in documents:
        new_doc = dict(doc)
        issues: list[str] = list(doc.get("quality_issues") or [])
        fetch_status = doc.get("fetch_status")
        parse_status = doc.get("parse_status")
        is_offline_stub = bool(doc.get("is_offline_stub"))
        is_fixture = bool(doc.get("is_fixture_document"))
        clean_text = doc.get("clean_text") or ""
        text_len = len(clean_text)
        error_page_detected = _document_looks_like_error_page(doc)

        if fetch_status == "fetch_failed":
            quality_status = "unusable"
            if "fetch_failed" not in issues:
                issues.append("fetch_failed")
            unusable += 1
        elif error_page_detected:
            quality_status = "unusable"
            if "error_page_detected" not in issues:
                issues.append("error_page_detected")
            unusable += 1
        elif is_offline_stub:
            quality_status = "offline_stub_pending_live_fetch"
            if "not_real_source_content" not in issues:
                issues.append("not_real_source_content")
            offline_stub += 1
        elif parse_status in {"pdf_parsing_deferred", "parse_deferred"}:
            quality_status = "parse_deferred"
            if "pdf_parsing_deferred" not in issues:
                issues.append("pdf_parsing_deferred")
            parse_deferred += 1
        elif text_len >= min_usable:
            quality_status = "usable"
            usable += 1
        elif text_len >= min_partial:
            quality_status = "partial"
            partial += 1
        elif (get_env("PIPELINE_MODE") == "evidence"
              and clean_text.strip() and doc.get("request_success")
              and doc.get("content_readable") and not doc.get("is_shell")
              and parse_status in {"parsed", "parsed_partial"}):
            # A short bulletin or a single table cell can be complete evidence.
            # This admits readable content to extraction, never a record to the
            # qualified dataset; field-level proof is checked downstream.
            quality_status = "partial"
            issues = [issue for issue in issues if issue != "insufficient_clean_text"]
            partial += 1
        else:
            quality_status = "unusable"
            if "insufficient_clean_text" not in issues:
                issues.append("insufficient_clean_text")
            unusable += 1

        if is_fixture and "synthetic_fixture_document" not in issues:
            issues.append("synthetic_fixture_document")

        relevance_document = new_doc
        if universal:
            # Search titles schedule discovery. Only the received page's title
            # and body may establish disease relevance for extraction.
            metadata = new_doc.get("metadata") or {}
            relevance_document = {**new_doc, "title": metadata.get("page_title") or metadata.get("page_heading") or ""}
        disease_assessment = assess_document_disease_relevance(relevance_document, context)
        new_doc.update(assessment_fields(disease_assessment, "document"))
        disease_status = disease_assessment.get("status") or "unknown"
        disease_status_counter[disease_status] += 1
        if (
            quality_status in {"usable", "partial"}
            and disease_status == UNRELATED_DISEASE
        ):
            if quality_status == "usable":
                usable = max(0, usable - 1)
            elif quality_status == "partial":
                partial = max(0, partial - 1)
            quality_status = "not_task_relevant"
            not_task_relevant += 1
            if "disease_mismatch_not_task_relevant" not in issues:
                issues.append("disease_mismatch_not_task_relevant")
            new_doc["not_extractable_for_task_disease"] = True
        else:
            new_doc["not_extractable_for_task_disease"] = False

        new_doc["quality_status"] = quality_status
        new_doc["quality_issues"] = issues
        new_doc["text_char_count"] = len(clean_text)
        new_doc["table_count"] = len(doc.get("tables") or [])
        new_doc.update(_post_fetch_readiness(new_doc, disease_assessment, quality_status))
        if universal:
            routing = _content_routing_update(new_doc, source_by_id.get(new_doc.get("source_id"), {}), context, routing_policy)
            if routing:
                audit = {"previous_source_role_final": new_doc.get("source_role_final"),
                         "previous_fetch_purpose": new_doc.get("fetch_purpose"),
                         "reason": "actual fetched task evidence supersedes advisory context metadata"}
                new_doc["metadata"] = {**(new_doc.get("metadata") or {}), "content_routing_revision": audit}
                new_doc.update(routing)
                routing_updates[new_doc.get("source_id")] = {**routing, "content_routing_revision": audit}
        updated.append(new_doc)
        quality_counter[quality_status] += 1

    summary = {
        "document_count": len(updated),
        "quality_status_counts": dict(quality_counter),
        "usable_count": usable,
        "partial_count": partial,
        "offline_stub_count": offline_stub,
        "parse_deferred_count": parse_deferred,
        "unusable_count": unusable,
        "error_page_document_count": sum(
            1 for doc in updated if "error_page_detected" in (doc.get("quality_issues") or [])
        ),
        "not_task_relevant_count": not_task_relevant,
        "disease_relevance_status_counts": dict(disease_status_counter),
    }

    trace = append_trace(
        state,
        node_name="document_quality_check",
        message=(
            f"Quality-checked {len(updated)} documents: "
            f"{usable} usable, {partial} partial, {offline_stub} offline stub, "
            f"{parse_deferred} parse deferred, {unusable} unusable."
        ),
        metadata=summary,
    )
    return {
        **({"source_registry": [{**row, **routing_updates.get(row.get("source_id"), {})}
                               for row in state.get("source_registry") or []]} if routing_updates else {}),
        "documents": updated,
        "document_quality_summary": summary,
        "disease_relevance_summary": update_disease_relevance_summary(
            {**state, "documents": updated}
        ),
        "collection_trace": trace,
    }


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _document_is_chunkable(
    doc: dict,
    policy: EvidenceChunkingPolicy,
) -> tuple[bool, str]:
    if doc.get("is_offline_stub"):
        return False, "offline_stub"
    if doc.get("parse_status") in {"pdf_parsing_deferred", "parse_deferred"}:
        return False, "parse_deferred"
    quality_status = doc.get("quality_status")
    if quality_status in policy.excluded_quality_statuses:
        return False, quality_status or "excluded_quality_status"
    if quality_status == "not_task_relevant" or doc.get(
        "not_extractable_for_task_disease"
    ):
        return False, "not_task_relevant"
    clean_text = doc.get("clean_text") or ""
    if not clean_text.strip():
        return False, "missing_clean_text"
    if quality_status in policy.chunkable_quality_statuses:
        return True, "chunkable"
    return False, "not_chunkable_quality_status"


def _split_text_into_chunks(
    text: str,
    max_chars: int,
    overlap_chars: int,
    min_chars: int,
) -> list[dict]:
    text = _normalize_whitespace(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [{"text": text, "char_start": 0, "char_end": len(text)}]

    chunks: list[dict] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chars, text_len)
        if end < text_len:
            # Try to end on a sentence boundary in the last ~200 chars.
            window_start = max(start, end - 200)
            last_period = text.rfind(". ", window_start, end)
            if last_period >= 0:
                end = last_period + 2
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {"text": chunk_text, "char_start": start, "char_end": end}
            )
        if end >= text_len:
            break
        start = max(end - overlap_chars, start + 1)

    if len(chunks) > 1:
        long_enough = [c for c in chunks if len(c["text"]) >= min_chars]
        if long_enough:
            chunks = long_enough
    return chunks


def _table_to_text(table: dict) -> str:
    rows = table.get("rows") if isinstance(table, dict) else None
    if rows:
        return "\n".join(
            " | ".join(str(cell) for cell in row) for row in rows
        )
    return str(table)


def _chunk_table(table: dict, table_cfg: dict) -> list[dict]:
    rows = table.get("rows") if isinstance(table, dict) else None
    if not rows:
        return [{"text": _table_to_text(table), "table_id": "0"}]

    max_rows = int(table_cfg.get("max_rows_per_chunk", 20) or 20)
    table_index = table.get("table_index", 0)
    heading_context = table.get("heading_context")
    table_header = " | ".join(str(c) for c in rows[0]) if rows else None
    if len(rows) <= max_rows:
        text = "\n".join(" | ".join(str(c) for c in row) for row in rows)
        return [
            {
                "text": text,
                "table_id": str(table_index),
                "table_header": table_header,
                "heading_context": heading_context,
                "row_context_type": "structured_table",
            }
        ]

    chunks: list[dict] = []
    for offset in range(0, len(rows), max_rows):
        slice_rows = rows[offset : offset + max_rows]
        text = "\n".join(" | ".join(str(c) for c in row) for row in slice_rows)
        chunks.append(
            {
                "text": text,
                "table_id": f"{table_index}_{offset // max_rows}",
                "table_header": table_header,
                "heading_context": heading_context,
                "row_context_type": "structured_table",
            }
        )
    return chunks


_METRIC_ROW_SIGNAL_RE = re.compile(
    r"\b("
    r"case|cases|death|deaths|positive|positives|specimen|specimens|test|tests|"
    r"hospital|hospitalization|hospitalizations|admission|admissions|outbreak|outbreaks|"
    r"ili|influenza-like|ed visit|ed visits|emergency department|percent|percentage|rate"
    r")\b",
    re.IGNORECASE,
)


def _is_markdown_separator_row(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in cells)


def _is_metric_row(line: str) -> bool:
    if "|" not in line or _is_markdown_separator_row(line):
        return False
    return bool(re.search(r"\d", line) and _METRIC_ROW_SIGNAL_RE.search(line))


def _markdown_cells(line: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", cell).strip()
        for cell in str(line or "").strip().strip("|").split("|")
        if re.sub(r"\s+", " ", cell).strip()
    ]


def _source_column_labels_from_header(header: str | None) -> list[str]:
    cells = _markdown_cells(header or "")
    if len(cells) <= 1:
        return []
    return cells[1:]


def _clean_plain_metric_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line or "").strip().lstrip("-* \t")).strip()


def _is_plain_metric_line(line: str, previous_heading: str | None = None) -> bool:
    cleaned = _clean_plain_metric_line(line)
    if not cleaned or "|" in cleaned:
        return False
    if len(cleaned) > 260:
        return False
    if not re.search(r"\d", cleaned):
        return False
    # A heading may describe a metric section, but a bare date, page number or
    # contact number underneath it is not itself a public-health metric row.
    return bool(_METRIC_ROW_SIGNAL_RE.search(cleaned))


_BOILERPLATE_TOKENS = {
    "about",
    "contact",
    "cookie",
    "copyright",
    "footer",
    "home",
    "menu",
    "navigation",
    "privacy",
    "share",
    "subscribe",
}


def _chunk_scheduler_signals(
    text: str,
    chunk_kind: str,
    contains_target_data: bool,
    confidence: float | None,
    disease_assessment: dict | None,
) -> tuple[str, str, float, int]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    lowered = normalized.lower()
    tokens = re.findall(r"[a-z]+", lowered)
    boilerplate_hits = sum(1 for token in tokens if token in _BOILERPLATE_TOKENS)
    boilerplate_ratio = (
        round(boilerplate_hits / len(tokens), 4) if tokens else 0.0
    )
    local_disease_status = str(
        (disease_assessment or {}).get("status") or "unknown"
    )

    explicit_case = bool(
        re.search(r"\b(?:case|patient)\s+(?:no\.?\s*)?\d+\b", lowered)
    )
    patient_narrative = bool(
        re.search(
            r"\b(?:patient|adult|male|female|crew|passenger|hospitali[sz]ed|"
            r"intensive care|developed symptoms|died|recovered)\b",
            lowered,
        )
    )
    aggregate_snapshot = bool(
        re.search(
            r"\b\d[\d,]*(?:\.\d+)?\s+(?:confirmed\s+|probable\s+|"
            r"suspected\s+)?(?:cases?|deaths?|patients?|hospitali[sz]ations?)\b",
            lowered,
        )
    )
    row_like = str(chunk_kind or "").lower() in {"table", "table_row", "metric_row"}
    if explicit_case and contains_target_data:
        priority = 0
    elif (patient_narrative or row_like) and contains_target_data:
        priority = 1
    elif aggregate_snapshot and contains_target_data:
        priority = 2
    elif contains_target_data:
        priority = 3
    else:
        priority = 4

    score = float(confidence or 0.0)
    if priority <= 1 and contains_target_data:
        signal_strength = "strong"
    elif priority <= 2 or score >= 0.70:
        signal_strength = "moderate"
    elif contains_target_data:
        signal_strength = "weak"
    else:
        signal_strength = "none"
    if boilerplate_ratio >= 0.30:
        signal_strength = "none"
        priority = max(priority, 5)
    return signal_strength, local_disease_status, boilerplate_ratio, priority


def _semantic_span_type(chunk_kind: str, text: str) -> str:
    lowered = str(text or "").lower()
    if str(chunk_kind or "").lower() in {"table", "table_row", "metric_row"}:
        return "table_row"
    if re.search(r"\b(?:case|patient)\s+(?:no\.?\s*)?\d+\b", lowered):
        return "explicit_case_span"
    if re.search(
        r"\b(?:patient|adult|male|female|hospitali[sz]ed|developed symptoms|died|recovered)\b",
        lowered,
    ):
        return "patient_narrative"
    if re.search(
        r"\b\d[\d,]*(?:\.\d+)?\s+(?:confirmed\s+|probable\s+|suspected\s+)?"
        r"(?:cases?|deaths?|patients?|hospitali[sz]ations?)\b",
        lowered,
    ):
        return "aggregate_snapshot"
    if re.search(r"\b(?:outbreak|cluster|event|reported|surveillance update)\b", lowered):
        return "event_narrative"
    return "context"


def _normalized_text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _generic_metric_row_data_types(text: str) -> list[str]:
    lowered = str(text or "").lower()
    if not re.search(r"\d", lowered) or not _METRIC_ROW_SIGNAL_RE.search(lowered):
        return []
    data_types = ["public_health_metric"]
    if any(term in lowered for term in ("positive", "positives", "specimen", "test")):
        data_types.append("testing_count")
    if any(term in lowered for term in ("percent", "percentage", "rate", "%")):
        data_types.append("rate_or_percent_metric")
    if any(term in lowered for term in ("hospital", "hospitalization", "admission")):
        data_types.append("hospitalization_count")
    if "death" in lowered or "fatalit" in lowered:
        data_types.append("death_count")
    if "outbreak" in lowered or "cluster" in lowered:
        data_types.append("outbreak")
    if "case" in lowered or "cases" in lowered:
        data_types.append("case_count")
    if any(term in lowered for term in ("ili", "influenza-like", "ed visit", "emergency department")):
        data_types.append("syndromic_surveillance_metric")
    return sorted(set(data_types))


def _extract_markdown_metric_row_chunks(text: str) -> list[dict]:
    """Extract row-sized metric chunks from markdown-like tables in text.

    Tavily and other extractors often return table-like markdown without
    structured `doc.tables`. These row chunks give the extraction agent precise
    evidence quotes and avoid spending one LLM call on every surrounding prose
    chunk.
    """

    rows: list[dict] = []
    lines = str(text or "").splitlines()
    header: str | None = None
    plain_heading: str | None = None
    last_heading: str | None = None
    plain_table_index = 0
    plain_row_index = 0
    table_index = 0
    row_index = 0
    in_table = False
    for line_number, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            next_nonblank = next(
                (
                    candidate.strip()
                    for candidate in lines[line_number + 1 :]
                    if candidate.strip()
                ),
                "",
            )
            preserve_heading_for_table = bool(
                not in_table and last_heading and "|" in next_nonblank
            )
            if in_table:
                table_index += 1
                row_index = 0
                header = None
                in_table = False
            plain_heading = None
            if not preserve_heading_for_table:
                last_heading = None
            continue
        if "|" not in line:
            if in_table:
                table_index += 1
                row_index = 0
                header = None
                in_table = False
                plain_heading = None
                last_heading = None
            cleaned = _clean_plain_metric_line(line)
            if re.match(r"^#{1,6}\s+\S", cleaned):
                plain_heading = cleaned
                last_heading = cleaned
                continue
            if _is_plain_metric_line(cleaned, plain_heading):
                plain_row_index += 1
                row_text = (
                    f"{plain_heading}\n{cleaned}"
                    if plain_heading
                    and not _METRIC_ROW_SIGNAL_RE.search(cleaned)
                    else cleaned
                )
                rows.append(
                    {
                        "text": row_text.strip(),
                        "table_id": f"markdown_metric_lines_{plain_table_index}",
                        "row_id": f"markdown_metric_lines_{plain_table_index}_{plain_row_index}",
                        "row_quote": cleaned,
                        "markdown_table_index": plain_table_index,
                        "markdown_row_index": plain_row_index,
                        "row_context_type": "markdown_metric_line",
                        "heading_context": plain_heading or last_heading,
                    }
                )
                continue
            if not re.search(r"\d", cleaned) and len(cleaned) <= 120:
                plain_heading = cleaned
                last_heading = cleaned
            continue
        in_table = True
        if _is_markdown_separator_row(line):
            continue
        if not re.search(r"\d", line):
            header = line
            continue
        if not _is_metric_row(line):
            continue
        row_index += 1
        row_text = f"{header}\n{line}" if header else line
        rows.append(
            {
                "text": row_text.strip(),
                "table_id": f"markdown_{table_index}",
                "row_id": f"markdown_{table_index}_{row_index}",
                "row_quote": line,
                "markdown_table_index": table_index,
                "markdown_row_index": row_index,
                "row_context_type": "markdown_table_row",
                "table_header": header,
                "source_column_labels": _source_column_labels_from_header(header),
                "heading_context": last_heading,
            }
        )
    return rows


def _detect_signal_categories(
    text: str,
    signal_map: dict[str, list[str]],
) -> dict[str, int]:
    lowered = (text or "").lower()
    hits: dict[str, int] = {}
    for category, terms in signal_map.items():
        count = 0
        for term in terms:
            if not term:
                continue
            if term.lower() in lowered:
                count += 1
        if count > 0:
            hits[category] = count
    return hits


def _flag_data_presence(
    text: str,
    doc: dict,
    policy: EvidenceChunkingPolicy,
) -> tuple[bool, list[str], list[str], float, str]:
    data_hits = _detect_signal_categories(text, policy.target_data_signals)
    context_hits = _detect_signal_categories(text, policy.context_signals)
    data_types = list(data_hits.keys())
    context_types = list(context_hits.keys())
    total_data = sum(data_hits.values())

    strong = int(policy.presence_thresholds.get("strong_data_signal_count", 3))
    moderate = int(policy.presence_thresholds.get("moderate_data_signal_count", 2))

    fetch_purpose = doc.get("fetch_purpose")
    if total_data >= strong:
        return True, data_types, context_types, 0.85, "strong deterministic data signal"
    if total_data >= moderate:
        return True, data_types, context_types, 0.70, "moderate deterministic data signal"
    if fetch_purpose == "data_extraction" and data_types:
        return (
            True,
            data_types,
            context_types,
            0.60,
            "weak data signal in a data-extraction document",
        )
    confidence = 0.40 if context_types else 0.20
    return False, data_types, context_types, confidence, "context only or no target data signal"


def _flag_context_only_presence(
    text: str,
    policy: EvidenceChunkingPolicy,
) -> tuple[bool, list[str], list[str], float, str, bool]:
    data_hits = _detect_signal_categories(text, policy.target_data_signals)
    context_hits = _detect_signal_categories(text, policy.context_signals)
    context_types = list(context_hits.keys())
    suppressed = bool(data_hits)
    confidence = 0.40 if context_types else 0.20
    return (
        False,
        [],
        context_types,
        confidence,
        "context-only source; target data suppressed",
        suppressed,
    )


def _metric_row_can_inherit_document_task_relevance(
    doc: dict,
    *,
    direct_collection: bool,
    doc_is_context_only: bool,
) -> bool:
    if doc_is_context_only:
        return False
    if doc.get("extraction_readiness") == "not_ready":
        return False
    relevance = str(
        doc.get("document_disease_relevance_status")
        or doc.get("task_relevance_status")
        or ""
    ).strip().lower()
    if relevance not in {
        "target_disease_match",
        "task_relevant",
        "relevant",
    }:
        return False
    role = str(doc.get("source_role_final") or doc.get("source_role") or "").strip().lower()
    if role not in {"collection", "validation", "collection_support"}:
        return False
    return str(doc.get("fetch_purpose") or "").strip().lower() == "data_extraction"


def _assess_metric_row_local_disease_relevance(row: dict, context: dict) -> dict:
    row_text = " ".join(
        str(row.get(key) or "") for key in ("text", "row_quote")
    ).strip()
    row_assessment = assess_chunk_disease_relevance(
        {"text": row_text, "title": ""}, context
    )
    row_target_terms = list(
        row_assessment.get("target_disease_terms_found") or []
    )
    row_incompatible_terms = list(
        row_assessment.get("incompatible_disease_terms_found") or []
    )
    if row_incompatible_terms and not row_target_terms:
        return row_assessment
    if row_assessment.get("status") in REJECT_DISEASE_STATUSES:
        return row_assessment

    contextual_text = " ".join(
        str(row.get(key) or "")
        for key in ("heading_context", "table_header", "text", "row_quote")
    )
    return assess_chunk_disease_relevance(
        {"text": contextual_text, "title": ""}, context
    )


def _apply_metric_row_document_task_relevance(
    *,
    doc: dict,
    direct_collection: bool,
    doc_is_context_only: bool,
    contains: bool,
    context_types: list[str],
    reason_text: str,
    disease_assessment: dict,
) -> tuple[dict, str]:
    if (
        not contains
        or disease_assessment.get("status") == TARGET_DISEASE_MATCH
        or disease_assessment.get("status") in REJECT_DISEASE_STATUSES
        or not _metric_row_can_inherit_document_task_relevance(
            doc,
            direct_collection=direct_collection,
            doc_is_context_only=doc_is_context_only,
        )
    ):
        return disease_assessment, reason_text
    inherited = dict(disease_assessment)
    inherited["status"] = TARGET_DISEASE_MATCH
    inherited["reason"] = (
        "metric row inherited task disease relevance from a verified "
        "collection document/source; row-level subtype exclusions remain "
        "record-level checks"
    )
    if "inherited_document_task_relevance" not in context_types:
        context_types.append("inherited_document_task_relevance")
    return (
        inherited,
        f"{reason_text}; metric row inherited document task relevance",
    )


def _promote_verified_metric_row_signal(
    *,
    doc: dict,
    direct_collection: bool,
    doc_is_context_only: bool,
    text: str,
    contains: bool,
    data_types: list[str],
    confidence: float,
    reason_text: str,
) -> tuple[bool, list[str], float, str]:
    if contains or not _metric_row_can_inherit_document_task_relevance(
        doc,
        direct_collection=direct_collection,
        doc_is_context_only=doc_is_context_only,
    ):
        return contains, data_types, confidence, reason_text
    generic_data_types = _generic_metric_row_data_types(text)
    if not generic_data_types:
        return contains, data_types, confidence, reason_text
    return (
        True,
        generic_data_types,
        max(float(confidence or 0.0), 0.65),
        "generic public-health metric row in verified collection document",
    )


def _make_evidence_chunk(
    doc: dict,
    chunk_index: int,
    chunk_kind: str,
    text: str,
    char_start: int | None,
    char_end: int | None,
    table_id: str | None,
    contains_target_data: bool,
    data_types: list[str],
    context_types: list[str],
    confidence: float,
    presence_reason: str,
    disease_assessment: dict | None = None,
    extraction_eligible_for_task_disease: bool | None = None,
    row_id: str | None = None,
    row_quote: str | None = None,
    source_column_label: str | None = None,
    metric_column_label: str | None = None,
    source_column_labels: list[str] | None = None,
    table_header: str | None = None,
    heading_context: str | None = None,
    row_context_type: str | None = None,
) -> EvidenceChunk:
    source_id = doc.get("source_id") or ""
    chunk_id = f"chunk_{source_id}_{chunk_index:03d}"
    document_hash = doc.get("content_hash")
    if get_env("PIPELINE_MODE") == "evidence":
        from ..session_runtime import fingerprint
        document_hash = document_hash or fingerprint(doc.get("clean_text") or doc.get("raw_text") or text)
        position = fingerprint([chunk_index, char_start, char_end, table_id, row_id, text])[:20]
        chunk_id = f"chunk_{source_id}_{document_hash}_{position}"
    (
        signal_strength,
        local_disease_status,
        boilerplate_ratio,
        extraction_priority,
    ) = _chunk_scheduler_signals(
        text,
        chunk_kind,
        contains_target_data,
        confidence,
        disease_assessment,
    )
    if doc.get("extraction_readiness") in {"context_only", "not_ready"}:
        extraction_priority = max(extraction_priority, 4)
        if signal_strength in {"strong", "moderate"}:
            signal_strength = "weak" if contains_target_data else "none"
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_hash=document_hash,
        document_id=doc.get("document_id"),
        source_id=source_id,
        text=text,
        section=None,
        page=None,
        table_id=table_id,
        row_id=row_id,
        row_quote=row_quote,
        contains_target_data=contains_target_data,
        data_types=list(data_types),
        confidence=confidence,
        document_type=doc.get("document_type"),
        fetch_purpose=doc.get("fetch_purpose"),
        source_url=doc.get("url"),
        canonical_url=doc.get("canonical_url"),
        title=doc.get("title"),
        publisher=doc.get("publisher"),
        source_type=doc.get("source_type"),
        source_role=doc.get("source_role"),
        source_role_final=doc.get("source_role_final"),
        credibility_score=doc.get("credibility_score"),
        credibility_level=doc.get("credibility_level"),
        actual_publisher=doc.get("actual_publisher"),
        actual_publisher_normalized=doc.get("actual_publisher_normalized"),
        source_type_final=doc.get("source_type_final"),
        source_independence_group=doc.get("source_independence_group"),
        claim_support_role=doc.get("claim_support_role"),
        recommended_source_role=doc.get("recommended_source_role"),
        recommended_fetch_use=doc.get("recommended_fetch_use"),
        recommended_extraction_use=doc.get("recommended_extraction_use"),
        likely_syndicated_or_aggregated=doc.get("likely_syndicated_or_aggregated"),
        upstream_source_mentions=list(doc.get("upstream_source_mentions") or []),
        discovery_method=doc.get("discovery_method"),
        search_provider=doc.get("search_provider"),
        query_id=doc.get("query_id"),
        query_used=doc.get("query_used"),
        planned_query_id=doc.get("planned_query_id"),
        provider_channel=doc.get("provider_channel"),
        role_hint=doc.get("role_hint"),
        quality_status=doc.get("quality_status"),
        chunk_index=chunk_index,
        chunk_kind=chunk_kind,
        char_start=char_start,
        char_end=char_end,
        context_types=list(context_types),
        presence_reason=presence_reason,
        task_relevance_status=doc.get("task_relevance_status"),
        page_identity_status=doc.get("page_identity_status"),
        extraction_readiness=doc.get("extraction_readiness"),
        extraction_readiness_reason=doc.get("extraction_readiness_reason"),
        **assessment_fields(disease_assessment or {}, ""),
        extraction_eligible_for_task_disease=extraction_eligible_for_task_disease,
        reporting_period_start=doc.get("reporting_period_start"),
        reporting_period_end=doc.get("reporting_period_end"),
        reporting_period_label=doc.get("reporting_period_label"),
        period_basis=doc.get("period_basis"),
        source_column_label=source_column_label,
        metric_column_label=metric_column_label,
        source_column_labels=list(source_column_labels or []),
        table_header=table_header,
        heading_context=heading_context,
        row_context_type=row_context_type,
        signal_strength=signal_strength,
        local_disease_status=local_disease_status,
        boilerplate_ratio=boilerplate_ratio,
        extraction_priority=extraction_priority,
    )


def evidence_chunking_and_data_presence_flagging(state: DataCollectionState) -> dict:
    """Split usable real documents into evidence chunks and flag data presence."""

    if get_env('PIPELINE_MODE') == 'evidence':
        from ..evidence_chunking import build_evidence_chunks
        return build_evidence_chunks(state)

    policy = EvidenceChunkingPolicy(**load_evidence_chunking_policy())
    role_policy = load_source_role_policy()
    documents = list(state.get("documents") or [])
    collection_mode = str(
        (state.get("structured_task") or {}).get("collection_mode")
        or (state.get("collection_spec") or {}).get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip()
    direct_collection = collection_mode == "direct_collection"

    evidence_chunks: list[EvidenceChunk] = []
    chunk_relevance_assessments: list[dict] = []
    skip_reason_counter: Counter = Counter()
    data_type_counter: Counter = Counter()
    context_type_counter: Counter = Counter()
    fetch_purpose_counter: Counter = Counter()
    skipped_count = 0
    chunkable_count = 0
    text_chunk_count = 0
    table_chunk_count = 0
    markdown_metric_row_chunk_count = 0
    target_data_count = 0
    context_only_count = 0
    no_signal_count = 0
    context_only_document_ids: set[str] = set()
    context_only_chunk_count = 0
    context_only_target_data_suppressed_count = 0
    disease_status_counter: Counter = Counter()
    disease_mismatch_chunk_count = 0
    disease_mismatch_source_ids: set[str] = set()
    context = build_disease_relevance_context(state)

    table_cfg = policy.table_chunking or {}
    table_enabled = bool(table_cfg.get("enabled", False))

    for doc in documents:
        is_chunkable, reason = _document_is_chunkable(doc, policy)
        if not is_chunkable:
            skip_reason_counter[reason] += 1
            skipped_count += 1
            continue
        chunkable_count += 1
        doc_is_context_only = _is_context_only_document(doc, role_policy)
        if doc_is_context_only and doc.get("source_id"):
            context_only_document_ids.add(doc.get("source_id"))
        chunk_index = 0

        # Text chunks.
        text_chunks = _split_text_into_chunks(
            doc.get("clean_text") or "",
            policy.max_chunk_chars,
            policy.chunk_overlap_chars,
            policy.min_chunk_chars,
        )
        for tc in text_chunks:
            chunk_index += 1
            if doc_is_context_only:
                contains, data_types, context_types, confidence, reason_text, suppressed = (
                    _flag_context_only_presence(tc["text"], policy)
                )
                context_only_chunk_count += 1
                if suppressed:
                    context_only_target_data_suppressed_count += 1
            else:
                contains, data_types, context_types, confidence, reason_text = (
                    _flag_data_presence(tc["text"], doc, policy)
                )
            disease_assessment = assess_chunk_disease_relevance(
                {**doc, "text": tc["text"], "title": doc.get("title")},
                context,
            )
            disease_status = disease_assessment.get("status") or "unknown"
            disease_status_counter[disease_status] += 1
            extraction_eligible = bool(
                contains and disease_status == TARGET_DISEASE_MATCH
            )
            if contains and disease_status != TARGET_DISEASE_MATCH:
                contains = False
                data_types = []
                if "disease_mismatch_context" not in context_types:
                    context_types.append("disease_mismatch_context")
                confidence = min(float(confidence or 0.0), 0.40)
                reason_text = (
                    "disease relevance gate suppressed extraction: "
                    f"{disease_assessment.get('reason')}"
                )
                disease_mismatch_chunk_count += 1
                if doc.get("source_id"):
                    disease_mismatch_source_ids.add(doc.get("source_id"))
            chunk = _make_evidence_chunk(
                doc=doc,
                chunk_index=chunk_index,
                chunk_kind="text",
                text=tc["text"],
                char_start=tc.get("char_start"),
                char_end=tc.get("char_end"),
                table_id=None,
                contains_target_data=contains,
                data_types=data_types,
                context_types=context_types,
                confidence=confidence,
                presence_reason=reason_text,
                disease_assessment=disease_assessment,
                extraction_eligible_for_task_disease=extraction_eligible,
            )
            evidence_chunks.append(chunk)
            chunk_relevance_assessments.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "decision_owner": (
                        "llm_chunk_relevance_agent"
                        if direct_collection
                        else "deterministic_chunk_relevance_gate"
                    ),
                    "contains_target_data": chunk.contains_target_data,
                    "extraction_eligible_for_task_disease": (
                        chunk.extraction_eligible_for_task_disease
                    ),
                    "disease_relevance_status": disease_status,
                    "presence_reason": reason_text,
                    "data_types": list(data_types),
                    "context_types": list(context_types),
                }
            )
            text_chunk_count += 1
            for cat in data_types:
                data_type_counter[cat] += 1
            for cat in context_types:
                context_type_counter[cat] += 1
            fp = doc.get("fetch_purpose") or "unknown"
            fetch_purpose_counter[fp] += 1
            if contains:
                target_data_count += 1
            elif context_types:
                context_only_count += 1
            else:
                no_signal_count += 1

        # Table chunks.
        if table_enabled:
            tables = doc.get("tables") or []
            for table in tables:
                for tc in _chunk_table(table, table_cfg):
                    chunk_index += 1
                    if doc_is_context_only:
                        (
                            contains,
                            data_types,
                            context_types,
                            confidence,
                            reason_text,
                            suppressed,
                        ) = _flag_context_only_presence(tc["text"], policy)
                        context_only_chunk_count += 1
                        if suppressed:
                            context_only_target_data_suppressed_count += 1
                    else:
                        contains, data_types, context_types, confidence, reason_text = (
                            _flag_data_presence(tc["text"], doc, policy)
                        )
                        contains, data_types, confidence, reason_text = (
                            _promote_verified_metric_row_signal(
                                doc=doc,
                                direct_collection=direct_collection,
                                doc_is_context_only=doc_is_context_only,
                                text=tc["text"],
                                contains=contains,
                                data_types=data_types,
                                confidence=confidence,
                                reason_text=reason_text,
                            )
                        )
                    disease_assessment = _assess_metric_row_local_disease_relevance(
                        tc,
                        context,
                    )
                    disease_assessment, reason_text = _apply_metric_row_document_task_relevance(
                        doc=doc,
                        direct_collection=direct_collection,
                        doc_is_context_only=doc_is_context_only,
                        contains=contains,
                        context_types=context_types,
                        reason_text=reason_text,
                        disease_assessment=disease_assessment,
                    )
                    disease_status = disease_assessment.get("status") or "unknown"
                    disease_status_counter[disease_status] += 1
                    extraction_eligible = bool(
                        contains and disease_status == TARGET_DISEASE_MATCH
                    )
                    if contains and disease_status != TARGET_DISEASE_MATCH:
                        contains = False
                        data_types = []
                        if "disease_mismatch_context" not in context_types:
                            context_types.append("disease_mismatch_context")
                        confidence = min(float(confidence or 0.0), 0.40)
                        reason_text = (
                            "disease relevance gate suppressed extraction: "
                            f"{disease_assessment.get('reason')}"
                        )
                        disease_mismatch_chunk_count += 1
                        if doc.get("source_id"):
                            disease_mismatch_source_ids.add(doc.get("source_id"))
                    chunk = _make_evidence_chunk(
                        doc=doc,
                        chunk_index=chunk_index,
                        chunk_kind="table",
                        text=tc["text"],
                        char_start=None,
                        char_end=None,
                        table_id=tc.get("table_id"),
                        row_id=tc.get("row_id"),
                        row_quote=tc.get("row_quote") or tc.get("text"),
                        table_header=tc.get("table_header"),
                        heading_context=tc.get("heading_context"),
                        row_context_type=tc.get("row_context_type"),
                        contains_target_data=contains,
                        data_types=data_types,
                        context_types=context_types,
                        confidence=confidence,
                        presence_reason=reason_text,
                        disease_assessment=disease_assessment,
                        extraction_eligible_for_task_disease=extraction_eligible,
                    )
                    evidence_chunks.append(chunk)
                    chunk_relevance_assessments.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "decision_owner": (
                                "llm_chunk_relevance_agent"
                                if direct_collection
                                else "deterministic_chunk_relevance_gate"
                            ),
                            "contains_target_data": chunk.contains_target_data,
                            "extraction_eligible_for_task_disease": (
                                chunk.extraction_eligible_for_task_disease
                            ),
                            "disease_relevance_status": disease_status,
                            "presence_reason": reason_text,
                            "data_types": list(data_types),
                            "context_types": list(context_types),
                        }
                    )
                    table_chunk_count += 1
                    for cat in data_types:
                        data_type_counter[cat] += 1
                    for cat in context_types:
                        context_type_counter[cat] += 1
                    fp = doc.get("fetch_purpose") or "unknown"
                    fetch_purpose_counter[fp] += 1
                    if contains:
                        target_data_count += 1
                    elif context_types:
                        context_only_count += 1
                    else:
                        no_signal_count += 1
            if not tables:
                for tc in _extract_markdown_metric_row_chunks(doc.get("clean_text") or ""):
                    chunk_index += 1
                    if doc_is_context_only:
                        (
                            contains,
                            data_types,
                            context_types,
                            confidence,
                            reason_text,
                            suppressed,
                        ) = _flag_context_only_presence(tc["text"], policy)
                        context_only_chunk_count += 1
                        if suppressed:
                            context_only_target_data_suppressed_count += 1
                    else:
                        contains, data_types, context_types, confidence, reason_text = (
                            _flag_data_presence(tc["text"], doc, policy)
                        )
                        contains, data_types, confidence, reason_text = (
                            _promote_verified_metric_row_signal(
                                doc=doc,
                                direct_collection=direct_collection,
                                doc_is_context_only=doc_is_context_only,
                                text=tc["text"],
                                contains=contains,
                                data_types=data_types,
                                confidence=confidence,
                                reason_text=reason_text,
                            )
                        )
                    row_context_type = tc.get("row_context_type") or "markdown_table_row"
                    if row_context_type not in context_types:
                        context_types.append(row_context_type)
                    disease_assessment = _assess_metric_row_local_disease_relevance(
                        tc,
                        context,
                    )
                    disease_assessment, reason_text = _apply_metric_row_document_task_relevance(
                        doc=doc,
                        direct_collection=direct_collection,
                        doc_is_context_only=doc_is_context_only,
                        contains=contains,
                        context_types=context_types,
                        reason_text=reason_text,
                        disease_assessment=disease_assessment,
                    )
                    disease_status = disease_assessment.get("status") or "unknown"
                    disease_status_counter[disease_status] += 1
                    extraction_eligible = bool(
                        contains and disease_status == TARGET_DISEASE_MATCH
                    )
                    if contains and disease_status != TARGET_DISEASE_MATCH:
                        contains = False
                        data_types = []
                        if "disease_mismatch_context" not in context_types:
                            context_types.append("disease_mismatch_context")
                        confidence = min(float(confidence or 0.0), 0.40)
                        reason_text = (
                            "disease relevance gate suppressed extraction: "
                            f"{disease_assessment.get('reason')}"
                        )
                        disease_mismatch_chunk_count += 1
                        if doc.get("source_id"):
                            disease_mismatch_source_ids.add(doc.get("source_id"))
                    chunk = _make_evidence_chunk(
                        doc=doc,
                        chunk_index=chunk_index,
                        chunk_kind="metric_row",
                        text=tc["text"],
                        char_start=None,
                        char_end=None,
                        table_id=tc.get("table_id"),
                        contains_target_data=contains,
                        data_types=data_types,
                        context_types=context_types,
                        confidence=confidence,
                        presence_reason=reason_text,
                        disease_assessment=disease_assessment,
                        extraction_eligible_for_task_disease=extraction_eligible,
                        row_id=tc.get("row_id"),
                        row_quote=tc.get("row_quote") or tc.get("text"),
                        source_column_labels=tc.get("source_column_labels") or [],
                        table_header=tc.get("table_header"),
                        heading_context=tc.get("heading_context"),
                        row_context_type=tc.get("row_context_type"),
                    )
                    evidence_chunks.append(chunk)
                    chunk_relevance_assessments.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_id": chunk.source_id,
                            "decision_owner": (
                                "llm_chunk_relevance_agent"
                                if direct_collection
                                else "deterministic_chunk_relevance_gate"
                            ),
                            "contains_target_data": chunk.contains_target_data,
                            "extraction_eligible_for_task_disease": (
                                chunk.extraction_eligible_for_task_disease
                            ),
                            "disease_relevance_status": disease_status,
                            "presence_reason": reason_text,
                            "data_types": list(data_types),
                            "context_types": list(context_types),
                            "table_row_source": row_context_type,
                        }
                    )
                    table_chunk_count += 1
                    markdown_metric_row_chunk_count += 1
                    for cat in data_types:
                        data_type_counter[cat] += 1
                    for cat in context_types:
                        context_type_counter[cat] += 1
                    fp = doc.get("fetch_purpose") or "unknown"
                    fetch_purpose_counter[fp] += 1
                    if contains:
                        target_data_count += 1
                    elif context_types:
                        context_only_count += 1
                    else:
                        no_signal_count += 1

    chunk_dicts: list[dict] = []
    seen_span_keys: set[tuple[str, str]] = set()
    duplicate_span_count = 0
    deduplicated_source_ids: set[str] = set()
    for chunk in evidence_chunks:
        row = chunk.model_dump()
        normalized_hash = _normalized_text_hash(row.get("text") or "")
        row["normalized_text_hash"] = normalized_hash
        row["semantic_span_type"] = _semantic_span_type(
            row.get("chunk_kind") or "",
            row.get("text") or "",
        )
        row["span_type"] = row["semantic_span_type"]
        dedupe_key = (str(row.get("source_id") or ""), normalized_hash)
        if dedupe_key in seen_span_keys:
            duplicate_span_count += 1
            deduplicated_source_ids.add(dedupe_key[0])
            continue
        seen_span_keys.add(dedupe_key)
        chunk_dicts.append(row)
    deduplication = {
        "scope": "within_source",
        "candidate_span_count": len(evidence_chunks),
        "duplicate_span_count": duplicate_span_count,
        "retained_span_count": len(chunk_dicts),
        "deduplicated_source_ids": sorted(deduplicated_source_ids),
    }

    chunking_summary = {
        "input_document_count": len(documents),
        "chunkable_document_count": chunkable_count,
        "skipped_document_count": skipped_count,
        "skip_reason_counts": dict(skip_reason_counter),
        "text_chunk_count": text_chunk_count,
        "table_chunk_count": table_chunk_count,
        "markdown_metric_row_chunk_count": markdown_metric_row_chunk_count,
        "total_chunk_count": len(chunk_dicts),
        "deduplication": deduplication,
        "context_only_document_count": len(context_only_document_ids),
        "context_only_source_ids": sorted(context_only_document_ids),
        "context_only_chunk_count": context_only_chunk_count,
        "disease_relevance_status_counts": dict(disease_status_counter),
        "disease_mismatch_chunk_count": disease_mismatch_chunk_count,
        "disease_mismatch_source_ids": sorted(disease_mismatch_source_ids),
    }
    presence_summary = {
        "total_chunk_count": len(chunk_dicts),
        "target_data_chunk_count": target_data_count,
        "context_only_chunk_count": context_only_count,
        "no_signal_chunk_count": no_signal_count,
        "context_only_document_count": len(context_only_document_ids),
        "context_only_source_ids": sorted(context_only_document_ids),
        "context_only_guardrail_chunk_count": context_only_chunk_count,
        "context_only_target_data_suppressed_count": (
            context_only_target_data_suppressed_count
        ),
        "data_type_counts": dict(data_type_counter),
        "context_type_counts": dict(context_type_counter),
        "fetch_purpose_counts": dict(fetch_purpose_counter),
        "disease_relevance_status_counts": dict(disease_status_counter),
        "disease_mismatch_chunk_count": disease_mismatch_chunk_count,
        "disease_mismatch_source_ids": sorted(disease_mismatch_source_ids),
    }

    trace = append_trace(
        state,
        node_name="evidence_chunking_and_data_presence_flagging",
        message=(
            f"Chunked {chunkable_count}/{len(documents)} documents into "
            f"{text_chunk_count + table_chunk_count} evidence chunks "
            f"({target_data_count} flagged as containing target data)."
        ),
        metadata={**chunking_summary, **presence_summary},
    )
    return {
        "evidence_chunks": chunk_dicts,
        "evidence_chunking_summary": chunking_summary,
        "data_presence_summary": presence_summary,
        "chunk_relevance_assessments": chunk_relevance_assessments,
        "disease_relevance_summary": update_disease_relevance_summary(
            {**state, "evidence_chunks": chunk_dicts},
            disease_mismatch_chunk_count=disease_mismatch_chunk_count,
        ),
        "collection_trace": trace,
    }
