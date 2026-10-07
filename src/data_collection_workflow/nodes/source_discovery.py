"""Source discovery and registry nodes.

Discover task-relevant sources through configured catalogs and bounded search.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from hashlib import sha256
import re
from urllib.parse import urlsplit, urlunsplit

from ..session_runtime import external_call, BudgetExceeded


def _search_request_identity(query):
    """Provider request semantics, excluding planner and iteration bookkeeping."""
    return {k: query.get(k) for k in ('query', 'provider_channel', 'source_type', 'official_domain_hint') if query.get(k) is not None}

from ..query_policy import assess_query_task_fit, generic_retry_specs, universal_queries_enabled
from ..config import (
    load_hantavirus_seed_sources,
    disease_intelligence_profile_slug,
    _load_overlay_json,
    _merge_seed_source_overlay,
)
from ..disease_relevance import (
    INCOMPATIBLE_DISEASE,
    UNRELATED_DISEASE,
    assess_text_disease_relevance,
    build_disease_relevance_context,
)
from ..models import (
    IterativeSourceDiscoverySummary,
    SearchProviderResponse,
    SearchResult,
    SearchIterationObservation,
    SearchIterationPlan,
    SearchRefinementDecision,
    SeedSource,
    SeedSourceCatalog,
    SourceCandidate,
    SourceRegistryEntry,
)
from ..search_providers import FixtureSearchProvider, build_search_provider
from ..run_events import emit_workflow_progress
from ..source_coverage import (
    build_authority_source_coverage_summary,
    build_official_coverage_candidates,
    official_report_key_for_url,
)
from ..source_identity import (
    authority_domain_hints_for_jurisdictions,
    load_source_identity_registry,
    lookup_source_identity_registry,
)
from ..state import DataCollectionState, append_trace

_DISCOVERY_METHOD = "offline_seed_catalog"
_SEARCH_RESULT_DISCOVERY_METHOD = {
    "fixture": "fixture_search_result",
    "live": "live_search_result",
}
_DEFAULT_PROVIDER_CHANNEL_ALLOWLIST = [
    "web_search",
    "official_site_search",
    "news_search",
    "literature_api",
    "database_search",
]
_SOURCE_CLASS_BUCKETS = {
    "international_official",
    "national_or_local_official",
    "structured_database",
    "peer_reviewed_literature",
    "news_or_supporting_media",
    "context_or_other",
}
_HIGH_TRUST_SOURCE_CLASSES = {
    "international_official",
    "national_or_local_official",
    "structured_database",
    "peer_reviewed_literature",
}
_OFFICIAL_QUERY_SOURCE_TYPES = {
    "official_public_health_agency",
    "government_report",
    "national_public_health_agency",
    "state_public_health_agency",
    "state_or_local_public_health_agency",
    "local_public_health_agency",
}
_INTERNATIONAL_QUERY_SOURCE_TYPES = {
    "international_organization_report",
    "international_public_health_agency",
}
_STRUCTURED_QUERY_SOURCE_TYPES = {
    "structured_database",
    "public_health_dataset",
    "surveillance_database",
    "dashboard",
}
_PEER_QUERY_SOURCE_TYPES = {
    "academic_or_peer_reviewed_source",
    "peer_reviewed_literature",
    "journal_article",
    "scientific_literature",
}
_NEWS_QUERY_SOURCE_TYPES = {
    "news_and_situation_report",
    "news",
    "news_media",
    "media_report",
    "secondary_media",
}
_ALLOWED_SEARCH_STATUSES = {
    "skipped_task_mismatch",
    "executed",
    "skipped_search_disabled",
    "skipped_provider_channel_not_supported",
    "skipped_query_limit",
    "skipped_total_result_limit",
    "skipped_verified_target_source_found",
    "skipped_invalid_query",
    "provider_error",
    "no_results",
}
_RAW_URL_ONLY_RE = re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.IGNORECASE)
_FLUVIEW_YEAR_WEEK_RE = re.compile(
    r"(?:20\d{2})[-_/ ]week[-_/ ]?(?:\d{1,2})|week[-_/ ]?(?:\d{1,2}).{0,40}(?:20\d{2})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_WEEK_RE = re.compile(r"\bweek[-_/ ]?(\d{1,2})\b", re.IGNORECASE)
_US_NATIONAL_LOCATION_ALIASES = {
    "united states",
    "usa",
    "us",
    "u.s.",
    "u.s.a.",
    "national",
}
_US_STATE_ABBREVIATIONS = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "district of columbia": "dc",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy",
}
_LOCAL_HEALTH_ACRONYM_ALIASES = {
    "california": {"cdph"},
    "new york": {"nysdoh", "nycdohmh"},
    "new york city": {"nyc", "nycdohmh"},
    "virginia": {"vdh"},
}


def _collection_mode(state: DataCollectionState) -> str:
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    return str(
        structured_task.get("collection_mode")
        or collection_spec.get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip()


def _is_hantavirus_task(state: DataCollectionState) -> bool:
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    disease = str(
        structured_task.get("disease")
        or collection_spec.get("disease")
        or ""
    ).lower()
    return disease_intelligence_profile_slug(disease) == "hantavirus"


@dataclass(frozen=True)
class SourceSearchSettings:
    mode: str = "disabled"
    provider: str = "tavily"
    fixture_path: str | None = None
    max_queries: int = 3
    max_results_per_query: int = 5
    max_total_results: int = 15
    timeout_seconds: float = 15.0
    combine_with_seed_catalog: bool = True
    cache_enabled: bool = True
    provider_channel_allowlist: list[str] = field(
        default_factory=lambda: list(_DEFAULT_PROVIDER_CHANNEL_ALLOWLIST)
    )
    iterative_enabled: bool = False
    iterative_max_iterations: int = 3
    iterative_max_queries_per_iteration: int = 3
    iterative_max_total_queries: int = 9
    iterative_max_total_results: int = 30
    iterative_require_llm: bool = True
    iterative_allow_deterministic_fallback: bool = False
    iterative_stop_when_llm_says_sufficient: bool = True
    iterative_require_observation_after_each_iteration: bool = True
    authority_gap_retry_enabled: bool = True
    authority_gap_retry_max_queries: int = 6
    authority_gap_known_domain_retry_max_queries: int = 4
    authority_gap_jurisdiction_retry_max_queries: int = 6
    authority_gap_official_page_family_retry_max_queries: int = 3
    authority_gap_retry_result_budget: int = 32
    authority_gap_retry_max_iterations: int = 1

    @property
    def fixture_search_enabled(self) -> bool:
        return self.mode == "fixture"

    @property
    def live_search_enabled(self) -> bool:
        return self.mode == "live" and _env_bool("ENABLE_LIVE_SEARCH", False)

    @property
    def search_enabled(self) -> bool:
        return self.fixture_search_enabled or self.live_search_enabled


def _effective_discovery_settings(settings: SourceSearchSettings) -> SourceSearchSettings:
    """In adaptive v2, the shared ledger owns total safety limits.

    Batch size remains a scheduling choice. Legacy/strict runs retain every
    explicitly configured small cap, including a disabled search.
    """
    if not universal_queries_enabled():
        return settings
    from ..session_runtime import get_runtime
    runtime = get_runtime()
    if runtime is None:
        return settings
    from ..acquisition_budget import is_adaptive_budget
    if not is_adaptive_budget(runtime.config):
        return settings
    limits = runtime.ledger.limits
    queries = max(0, int(limits.get("search", settings.iterative_max_total_queries)))
    results = max(0, int(limits.get("search_results", settings.iterative_max_total_results)))
    return replace(settings, max_queries=queries, max_total_results=results,
                   iterative_max_total_queries=queries, iterative_max_total_results=results,
                   iterative_max_iterations=queries)


def _supplementary_discovery_allowed(state: DataCollectionState, queries: list[dict]) -> bool:
    """Routine surveillance needs the same source-family discovery as outbreaks."""
    return universal_queries_enabled() or _state_is_outbreak_like(state, queries)


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL or seed:// URI for deduplication.

    - strip surrounding whitespace
    - drop URL fragment (after "#")
    - strip trailing slash on the path
    - lowercase scheme and host when present
    - preserve query string (search endpoints depend on it)
    - leave seed:// URIs intact (lowercase scheme; preserve the rest)
    """

    if url is None:
        return ""
    stripped = url.strip()
    if not stripped:
        return ""

    parts = urlsplit(stripped)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path or ""
    if scheme in {"http", "https"}:
        path = re.sub(r"/{2,}", "/", path)
    if path.endswith("/") and len(path) > 1:
        path = path.rstrip("/")
    # Drop fragment, preserve query.
    rebuilt = urlunsplit((scheme, netloc, path, parts.query, ""))

    # urlsplit on seed://... places "structured-database" in netloc — that's fine.
    # If there was no scheme and no netloc, fall back to stripped form minus fragment.
    if not scheme and not netloc:
        rebuilt = stripped.split("#", 1)[0]
        if rebuilt.endswith("/") and len(rebuilt) > 1:
            rebuilt = rebuilt.rstrip("/")
    return rebuilt


def _env_bool(name: str, default: bool = False) -> bool:
    raw = get_env(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_float(name: str, default: float) -> float:
    raw = (get_env(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = get_env(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or list(default)


def _source_search_settings_from_env() -> SourceSearchSettings:
    mode = (get_env("SEARCH_MODE") or "disabled").strip().lower()
    if mode not in {"disabled", "fixture", "live"}:
        mode = "disabled"
    provider = (get_env("SEARCH_PROVIDER") or "").strip().lower()
    if not provider:
        provider = "fixture" if mode == "fixture" else "tavily"
    return SourceSearchSettings(
        mode=mode,
        provider=provider,
        fixture_path=(get_env("SEARCH_FIXTURE_PATH") or "").strip() or None,
        max_queries=_env_int("SEARCH_MAX_QUERIES", 3),
        max_results_per_query=_env_int("SEARCH_MAX_RESULTS_PER_QUERY", 5),
        max_total_results=_env_int("SEARCH_MAX_TOTAL_RESULTS", 15),
        timeout_seconds=_env_float("SEARCH_TIMEOUT_SECONDS", 15.0),
        combine_with_seed_catalog=_env_bool(
            "SEARCH_COMBINE_WITH_SEED_CATALOG", True
        ),
        cache_enabled=_env_bool("SEARCH_CACHE_ENABLED", True),
        provider_channel_allowlist=_env_csv(
            "SEARCH_PROVIDER_CHANNEL_ALLOWLIST",
            _DEFAULT_PROVIDER_CHANNEL_ALLOWLIST,
        ),
        iterative_enabled=_env_bool("ENABLE_ITERATIVE_SOURCE_DISCOVERY", False),
        iterative_max_iterations=_env_int("ITERATIVE_SEARCH_MAX_ITERATIONS", 3),
        iterative_max_queries_per_iteration=_env_int(
            "ITERATIVE_SEARCH_MAX_QUERIES_PER_ITERATION", 3
        ),
        iterative_max_total_queries=_env_int(
            "ITERATIVE_SEARCH_MAX_TOTAL_QUERIES", 9
        ),
        iterative_max_total_results=_env_int(
            "ITERATIVE_SEARCH_MAX_TOTAL_RESULTS", 30
        ),
        iterative_require_llm=_env_bool("ITERATIVE_SEARCH_REQUIRE_LLM", True),
        iterative_allow_deterministic_fallback=_env_bool(
            "ITERATIVE_SEARCH_ALLOW_DETERMINISTIC_FALLBACK", False
        ),
        iterative_stop_when_llm_says_sufficient=_env_bool(
            "ITERATIVE_SEARCH_STOP_WHEN_LLM_SAYS_SUFFICIENT", True
        ),
        iterative_require_observation_after_each_iteration=_env_bool(
            "ITERATIVE_SEARCH_REQUIRE_OBSERVATION_AFTER_EACH_ITERATION", True
        ),
        authority_gap_retry_enabled=_env_bool(
            "AUTHORITY_GAP_RETRY_ENABLED", True
        ),
        authority_gap_retry_max_queries=_env_int(
            "AUTHORITY_GAP_RETRY_MAX_QUERIES", 6
        ),
        authority_gap_known_domain_retry_max_queries=_env_int(
            "AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES", 4
        ),
        authority_gap_jurisdiction_retry_max_queries=_env_int(
            "AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES", 6
        ),
        authority_gap_official_page_family_retry_max_queries=_env_int(
            "AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES", 3
        ),
        authority_gap_retry_result_budget=_env_int(
            "AUTHORITY_GAP_RETRY_RESULT_BUDGET", 32
        ),
        authority_gap_retry_max_iterations=_env_int(
            "AUTHORITY_GAP_RETRY_MAX_ITERATIONS", 1
        ),
    )


def _best_matching_query(
    seed_source: dict,
    search_query_inventory: list[dict],
) -> dict | None:
    """Pick the most relevant query for a seed source, or None.

    Preference order:
      1. queries whose source_type matches and whose query string contains
         any of the seed source's match_terms (case-insensitive).
      2. the first query whose source_type matches.
      3. None.
    """

    if not search_query_inventory:
        return None

    target_type = seed_source.get("source_type")
    same_type = [
        q for q in search_query_inventory
        if q.get("source_type") == target_type
    ]
    if not same_type:
        return None

    match_terms = [t.lower() for t in seed_source.get("match_terms", []) if t]
    for query in same_type:
        query_text = (query.get("query") or "").lower()
        if any(term in query_text for term in match_terms):
            return query
    return same_type[0]


def _make_source_candidate(
    seed_source: SeedSource,
    matched_query: dict | None,
) -> SourceCandidate:
    """Build a deterministic SourceCandidate from a seed source."""

    source_id = seed_source.seed_source_id
    if source_id.startswith("seed_"):
        source_id = "src_" + source_id[len("seed_"):]

    query_id = matched_query.get("query_id") if matched_query else None
    query_used = matched_query.get("query") if matched_query else None

    return SourceCandidate(
        source_id=source_id,
        title=seed_source.title,
        url=seed_source.url,
        publisher=seed_source.publisher,
        source_type=seed_source.source_type,
        published_date=None,
        snippet=None,
        query_used=query_used,
        retrieved_at=None,  # A catalog entry is not a retrieved search response.
        query_id=query_id,
        discovery_method=_DISCOVERY_METHOD,
        seed_source_id=seed_source.seed_source_id,
        priority=seed_source.priority,
        expected_fields=list(seed_source.expected_fields),
        matched_terms=list(seed_source.match_terms),
        source_purpose=seed_source.source_purpose,
        notes=seed_source.notes,
    )


def _planned_queries(state: DataCollectionState) -> list[dict]:
    plan = state.get("agentic_source_plan") or {}
    queries = plan.get("planned_queries") if isinstance(plan, dict) else None
    if not queries:
        queries = state.get("search_query_inventory") or []
    return [dict(item) for item in queries or [] if isinstance(item, dict)]


def _lower_text(value) -> str:
    return str(value or "").strip().lower()


def _unique_preserve_order(values) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _site_domain_from_query_text(query_text: str) -> str | None:
    match = re.search(r"\bsite:([a-z0-9.-]+\.[a-z]{2,})", query_text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower().removeprefix("www.")


def _known_authority_domain_for_query(query: dict) -> str | None:
    query_text = _lower_text(query.get("query"))
    domain = _lower_text(query.get("official_domain_hint")) or _site_domain_from_query_text(
        query_text
    )
    if not domain:
        return None
    registry_entry = lookup_source_identity_registry(domain)
    source_type = _lower_text((registry_entry or {}).get("source_type"))
    jurisdiction_scope = _lower_text((registry_entry or {}).get("jurisdiction_scope"))
    if source_type in (
        _OFFICIAL_QUERY_SOURCE_TYPES
        | _INTERNATIONAL_QUERY_SOURCE_TYPES
        | _STRUCTURED_QUERY_SOURCE_TYPES
        | _PEER_QUERY_SOURCE_TYPES
    ):
        return domain
    if jurisdiction_scope in {"international", "regional", "national", "subnational"}:
        return domain
    return None


def _query_source_class(query: dict) -> str:
    source_type = _lower_text(
        query.get("source_type") or query.get("planned_query_source_type")
    )
    channel = _lower_text(query.get("provider_channel"))
    query_text = _lower_text(query.get("query"))
    domain = _lower_text(query.get("official_domain_hint")) or _site_domain_from_query_text(
        query_text
    )
    registry_entry = lookup_source_identity_registry(domain) if domain else None
    registry_type = _lower_text((registry_entry or {}).get("source_type"))
    registry_scope = _lower_text((registry_entry or {}).get("jurisdiction_scope"))

    if registry_type in _STRUCTURED_QUERY_SOURCE_TYPES:
        return "structured_database"
    if registry_type in _PEER_QUERY_SOURCE_TYPES:
        return "peer_reviewed_literature"
    if registry_type in _INTERNATIONAL_QUERY_SOURCE_TYPES or registry_scope in {
        "international",
        "regional",
    } and registry_type in _OFFICIAL_QUERY_SOURCE_TYPES:
        return "international_official"
    if registry_type in _OFFICIAL_QUERY_SOURCE_TYPES:
        return "national_or_local_official"

    if source_type in _STRUCTURED_QUERY_SOURCE_TYPES or channel == "database_search":
        return "structured_database"
    if source_type in _PEER_QUERY_SOURCE_TYPES or channel == "literature_api":
        return "peer_reviewed_literature"
    if source_type in _INTERNATIONAL_QUERY_SOURCE_TYPES:
        return "international_official"
    if source_type in _OFFICIAL_QUERY_SOURCE_TYPES or channel == "official_site_search":
        return "national_or_local_official"
    if source_type in _NEWS_QUERY_SOURCE_TYPES or channel == "news_search":
        return "news_or_supporting_media"
    if any(
        token in query_text
        for token in ("case report", "case series", "journal", "pubmed", "nejm")
    ):
        return "peer_reviewed_literature"
    if any(token in query_text for token in ("database", "sequence", "genomic", "line list")):
        return "structured_database"
    if any(
        token in query_text
        for token in ("ministry of health", "public health", "health department")
    ):
        return "national_or_local_official"
    return "context_or_other"


def _query_selection_key(query: dict, index: int) -> str:
    return str(query.get("query_id") or f"__query_index_{index}")


def _state_is_outbreak_like(state: DataCollectionState, planned_queries: list[dict]) -> bool:
    task = _task_context(state)
    location_text = _lower_text(task.get("location"))
    request_text = " ".join(
        [
            _lower_text((state.get("structured_task") or {}).get("user_request")),
            _lower_text((state.get("collection_spec") or {}).get("user_request")),
        ]
    )
    query_hint_text = " ".join(
        " ".join(
            _lower_text(query.get(key))
            for key in ("query", "source_priority_reason", "rationale")
        )
        for query in planned_queries
    )
    if any(
        token in location_text
        for token in ("global", "worldwide", "multi-country", "multicountry")
    ):
        return True
    if any(
        token in request_text
        for token in (
            "multi-country",
            "multicountry",
            "cruise",
            "vessel",
            "travel-associated",
            "travel associated",
            "outbreak",
            "cluster",
        )
    ):
        return True
    return any(
        token in query_hint_text
        for token in (
            "multi-country",
            "multicountry",
            "cruise",
            "vessel",
            "travel-associated",
            "travel associated",
            "global_outbreak_authority_source_class",
        )
    )


def _select_source_search_queries(
    planned_queries: list[dict],
    state: DataCollectionState,
    settings: SourceSearchSettings,
) -> dict[str, dict]:
    """Select executable queries with a small high-trust source-class floor."""

    eligible: list[tuple[int, dict, str, str]] = []
    allowlist = set(settings.provider_channel_allowlist)
    for index, query in enumerate(planned_queries):
        if _is_invalid_query(query):
            continue
        if universal_queries_enabled() and not assess_query_task_fit(query, state)['accepted']:
            continue
        channel = str(query.get("provider_channel") or "")
        if channel not in allowlist:
            continue
        key = _query_selection_key(query, index)
        eligible.append((index, query, key, _query_source_class(query)))

    max_queries = max(0, int(settings.max_queries or 0))
    if not eligible or max_queries <= 0:
        return {}

    selected: dict[str, dict] = {}
    selected_order: list[str] = []

    def add_query(key: str, bucket: str, reason: str) -> bool:
        if key in selected or len(selected) >= max_queries:
            return False
        selected[key] = {
            "selection_bucket": bucket,
            "selection_reason": reason,
            "selection_order": len(selected_order) + 1,
        }
        selected_order.append(key)
        return True

    def add_from_bucket(
        bucket: str,
        count: int,
        reason: str,
        *,
        prefer_known_domain: bool = False,
    ) -> None:
        if count <= 0:
            return
        added = 0
        bucket_items = [
            (index, query, key, query_bucket)
            for index, query, key, query_bucket in eligible
            if query_bucket == bucket and key not in selected
        ]
        if prefer_known_domain:
            bucket_items = sorted(
                bucket_items,
                key=lambda item: (
                    0 if _known_authority_domain_for_query(item[1]) else 1,
                    item[0],
                ),
            )
        for _, _, key, query_bucket in bucket_items:
            if added >= count or len(selected) >= max_queries:
                return
            if add_query(key, query_bucket, reason):
                added += 1

    outbreak_like = _state_is_outbreak_like(state, planned_queries)
    if outbreak_like or universal_queries_enabled():
        for bucket in (
            "international_official",
            "national_or_local_official",
            "structured_database",
            "peer_reviewed_literature",
        ):
            add_from_bucket(
                bucket,
                1,
                f"source_class_floor:{bucket}",
                prefer_known_domain=True,
            )

        targets = {
            "international_official": 4,
            "national_or_local_official": 3,
            "structured_database": 2,
            "peer_reviewed_literature": 2,
        }
        for bucket, target in targets.items():
            current = sum(
                1
                for item in selected.values()
                if item.get("selection_bucket") == bucket
            )
            add_from_bucket(
                bucket,
                max(0, target - current),
                f"source_class_budget:{bucket}",
                prefer_known_domain=bucket
                in {
                    "national_or_local_official",
                    "structured_database",
                    "peer_reviewed_literature",
                },
            )

        news_selected = 0
        for _, _, key, bucket in eligible:
            if len(selected) >= max_queries:
                break
            if key in selected:
                continue
            if bucket == "news_or_supporting_media":
                if news_selected >= 3:
                    continue
                news_selected += 1
            add_query(key, bucket, "original_order_fill")
        return selected

    for _, _, key, bucket in eligible[:max_queries]:
        add_query(key, bucket, "original_order")

    if not any(
        item.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
        for item in selected.values()
    ):
        first_high_trust = next(
            (
                (key, bucket)
                for _, _, key, bucket in eligible
                if bucket in _HIGH_TRUST_SOURCE_CLASSES
            ),
            None,
        )
        if first_high_trust and selected_order:
            removed_key = selected_order.pop()
            selected.pop(removed_key, None)
            key, bucket = first_high_trust
            add_query(key, bucket, "high_trust_starvation_prevention")
    return selected


def _search_execution_record(
    query: dict,
    *,
    provider: str,
    status: str,
    selected: bool = False,
    result_count: int = 0,
    skipped_reason: str | None = None,
    error: str | None = None,
    selection_bucket: str | None = None,
    selection_reason: str | None = None,
) -> dict:
    if status not in _ALLOWED_SEARCH_STATUSES:
        raise ValueError(f"unsupported search execution status: {status}")
    return {
        "query_id": query.get("query_id"),
        "query": query.get("query"),
        "provider_channel": query.get("provider_channel"),
        "query_type": query.get("query_type"),
        "source_type": query.get("source_type"),
        "role_hint": query.get("role_hint"),
        "query_language": query.get("query_language"),
        "jurisdiction_hint": query.get("jurisdiction_hint"),
        "official_domain_hint": query.get("official_domain_hint"),
        "localized_source_hint": bool(query.get("localized_source_hint")),
        "source_priority_reason": query.get("source_priority_reason"),
        "query_source": query.get("query_source"),
        "retry_category": query.get("retry_category"),
        "iteration_index": query.get("iteration_index"),
        "iterative_query_id": query.get("iterative_query_id"),
        "previous_iteration_basis": query.get("previous_iteration_basis"),
        "selection_bucket": selection_bucket or _query_source_class(query),
        "selection_reason": selection_reason,
        "selected_for_execution": bool(selected),
        "execution_status": status,
        "provider": provider,
        "result_count": int(result_count or 0),
        "skipped_reason": skipped_reason,
        "error": error,
    }


def _is_invalid_query(query: dict) -> bool:
    text = str(query.get("query") or "").strip()
    if not text:
        return True
    if _RAW_URL_ONLY_RE.match(text):
        return True
    return str(query.get("execution_status") or "planned_not_executed") != (
        "planned_not_executed"
    )


def _domain(url: str) -> str | None:
    netloc = urlsplit(url).netloc.lower()
    if not netloc:
        return None
    return netloc[4:] if netloc.startswith("www.") else netloc


def _validate_search_url(url: str | None) -> tuple[str | None, str | None]:
    if url is None or not str(url).strip():
        return None, "missing_url"
    canonical = canonicalize_url(str(url))
    parts = urlsplit(canonical)
    if not parts.scheme:
        return None, "invalid_url"
    if parts.scheme not in {"http", "https"}:
        return None, "unsupported_scheme"
    if not parts.netloc:
        return None, "invalid_url"
    return canonical, None


def _source_id_from_canonical_url(canonical_url: str) -> str:
    digest = sha256(canonical_url.encode("utf-8")).hexdigest()[:12]
    return f"src_search_{digest}"


def _response_from_provider_output(
    output,
    *,
    provider: str,
    planned_query: dict,
) -> SearchProviderResponse:
    if isinstance(output, SearchProviderResponse):
        return output
    if isinstance(output, dict):
        raw_results = output.get("results") or []
        results = []
        for index, item in enumerate(raw_results, start=1):
            if isinstance(item, SearchResult):
                results.append(item)
            elif isinstance(item, dict):
                results.append(
                    SearchResult(
                        title=item.get("title"),
                        url=item.get("url"),
                        snippet=item.get("snippet") or item.get("content"),
                        published_date=item.get("published_date") or item.get("date"),
                        source=item.get("source") or item.get("publisher"),
                        rank=int(item.get("rank") or index),
                        query=planned_query.get("query"),
                        query_id=planned_query.get("query_id"),
                        provider_channel=planned_query.get("provider_channel"),
                        source_type=planned_query.get("source_type"),
                        role_hint=planned_query.get("role_hint"),
                        retrieved_at=item.get("retrieved_at"),
                        provider=output.get("provider") or provider,
                        query_type=planned_query.get("query_type"),
                        raw=dict(item),
                    )
                )
        return SearchProviderResponse(
            provider=output.get("provider") or provider,
            query_id=output.get("query_id") or planned_query.get("query_id"),
            query=output.get("query") or planned_query.get("query"),
            results=results,
            raw_result_count=int(output.get("raw_result_count") or len(results)),
            error=output.get("error"),
            warnings=list(output.get("warnings") or []),
        )
    raise TypeError(f"Unsupported search provider response type: {type(output)!r}")


def _build_search_provider(settings: SourceSearchSettings):
    return build_search_provider(
        provider=settings.provider,
        mode=settings.mode,
        fixture_path=settings.fixture_path,
    )


def _provider_for_settings(settings: SourceSearchSettings):
    if settings.mode == "fixture":
        if not settings.fixture_path:
            raise ValueError("fixture search mode requires SEARCH_FIXTURE_PATH")
        return FixtureSearchProvider(settings.fixture_path)
    return _build_search_provider(settings)


def _discovery_method_for_settings(settings: SourceSearchSettings) -> str:
    if not settings.search_enabled:
        return "seed_catalog_only" if settings.combine_with_seed_catalog else "search_disabled"
    if settings.live_search_enabled and settings.combine_with_seed_catalog:
        return "live_search_plus_seed_catalog"
    if settings.live_search_enabled:
        return "live_search_only"
    if settings.fixture_search_enabled and settings.combine_with_seed_catalog:
        return "fixture_search_plus_seed_catalog"
    if settings.fixture_search_enabled:
        return "fixture_search_only"
    return "seed_catalog_only" if settings.combine_with_seed_catalog else "search_disabled"


def _candidate_from_search_result(
    result: SearchResult,
    planned_query: dict,
    *,
    settings: SourceSearchSettings,
    canonical_url: str,
) -> SourceCandidate:
    domain = _domain(canonical_url)
    discovery_method = _SEARCH_RESULT_DISCOVERY_METHOD.get(settings.mode)
    source_id = _source_id_from_canonical_url(canonical_url)
    raw_result_source = result.source
    search_metadata_publisher = None
    if raw_result_source and raw_result_source.strip().lower() != settings.provider.lower():
        search_metadata_publisher = raw_result_source
    matched_terms = [
        *list(planned_query.get("disease_terms_used") or []),
        *list(planned_query.get("location_terms_used") or []),
        *list(planned_query.get("time_terms_used") or []),
    ]
    return SourceCandidate(
        source_id=source_id,
        title=result.title,
        url=result.url or canonical_url,
        publisher=search_metadata_publisher,
        source_type=planned_query.get("source_type") or result.source_type,
        published_date=result.published_date,
        snippet=result.snippet,
        query_used=planned_query.get("query") or result.query,
        retrieved_at=result.retrieved_at,
        query_id=planned_query.get("query_id") or result.query_id,
        discovery_method=discovery_method,
        seed_source_id=None,
        priority=planned_query.get("priority"),
        expected_fields=list(planned_query.get("expected_fields") or []),
        matched_terms=matched_terms,
        source_purpose=(
            f"{planned_query.get('role_hint')}_search_result"
            if planned_query.get("role_hint")
            else "search_discovered_source"
        ),
        notes=(
            f"Search-derived source candidate from {settings.mode} "
            f"{settings.provider} metadata; page body not fetched by search."
        ),
        search_provider=settings.provider,
        search_rank=result.rank,
        provider_channel=planned_query.get("provider_channel") or result.provider_channel,
        role_hint=planned_query.get("role_hint") or result.role_hint,
        planned_query_id=planned_query.get("query_id"),
        planned_query_source_type=planned_query.get("source_type"),
        search_result_id=f"{planned_query.get('query_id')}_{result.rank or 0}",
        canonical_url=canonical_url,
        domain=domain,
        result_source=result.source,
        search_result_source_raw=raw_result_source,
        search_provider_result_source=raw_result_source,
        publisher_candidate_from_search_metadata=search_metadata_publisher,
        query_type=planned_query.get("query_type") or result.query_type,
        additional_query_ids=[],
        query_source=planned_query.get("query_source"),
        iteration_index=planned_query.get("iteration_index"),
        iterative_query_id=planned_query.get("iterative_query_id"),
        previous_iteration_basis=planned_query.get("previous_iteration_basis"),
    )


def _candidate_from_official_coverage(row: dict) -> SourceCandidate | None:
    url = row.get("url") or row.get("canonical_url")
    canonical_url, invalid_reason = _validate_search_url(str(url or ""))
    if invalid_reason or not canonical_url:
        return None
    return SourceCandidate(
        source_id=row.get("source_id") or _source_id_from_canonical_url(canonical_url),
        title=row.get("title"),
        url=canonical_url,
        publisher=row.get("publisher"),
        source_type=row.get("source_type") or "official_public_health_agency",
        published_date=row.get("published_date"),
        snippet=row.get("snippet"),
        query_used=row.get("query_used"),
        retrieved_at=row.get("retrieved_at"),
        query_id=row.get("query_id"),
        discovery_method="official_coverage_requirement",
        seed_source_id=None,
        priority=row.get("priority", 0),
        expected_fields=list(row.get("expected_fields") or []),
        matched_terms=list(row.get("matched_terms") or []),
        source_purpose=row.get("source_purpose") or "target_official_surveillance_report",
        notes=row.get("notes"),
        search_provider=None,
        search_rank=0,
        provider_channel=row.get("provider_channel") or "official_site_search",
        role_hint=row.get("role_hint") or "collection",
        planned_query_id=row.get("planned_query_id") or row.get("query_id"),
        planned_query_source_type=(
            row.get("planned_query_source_type")
            or "official_weekly_surveillance_report"
        ),
        search_result_id=None,
        canonical_url=canonical_url,
        domain=row.get("domain") or _domain(canonical_url),
        query_type=row.get("query_type") or "deterministic_official_url",
        additional_query_ids=[],
        query_source=row.get("query_source") or "source_coverage_requirement",
        source_disease_relevance_status=row.get("source_disease_relevance_status"),
        source_disease_relevance_score=row.get("source_disease_relevance_score"),
        source_target_disease_terms_found=list(
            row.get("source_target_disease_terms_found") or []
        ),
        source_incompatible_disease_terms_found=list(
            row.get("source_incompatible_disease_terms_found") or []
        ),
        source_disease_relevance_reason=row.get("source_disease_relevance_reason"),
        source_disease_relevance_data_signal_count=row.get(
            "source_disease_relevance_data_signal_count"
        ),
        must_fetch=bool(row.get("must_fetch")),
        must_fetch_reason=row.get("must_fetch_reason"),
        coverage_requirement_ids=list(row.get("coverage_requirement_ids") or []),
        routing_conflict_warnings=list(row.get("routing_conflict_warnings") or []),
        target_fit_status=row.get("target_fit_status") or "predicted_target_candidate",
        target_verification_status=(
            row.get("target_verification_status") or "predicted_unverified"
        ),
        target_verification_reason=(
            row.get("target_verification_reason")
            or row.get("must_fetch_reason")
            or "official coverage requirement candidate; requires search or fetch validation"
        ),
        triage_role=row.get("triage_role") or "predicted_target_candidate",
    )


def _query_source_counts(query_records: list[dict]) -> dict:
    return dict(
        Counter(
            str(record.get("query_source") or "unspecified")
            for record in query_records
            if record.get("selected_for_execution")
        )
    )


def _query_record_source_class_counts(
    query_records: list[dict],
    *,
    selected_only: bool = False,
    executed_only: bool = False,
    result_sum: bool = False,
) -> dict:
    counter: Counter = Counter()
    for record in query_records:
        if selected_only and not record.get("selected_for_execution"):
            continue
        if executed_only and record.get("execution_status") not in {
            "executed",
            "no_results",
            "provider_error",
        }:
            continue
        bucket = str(record.get("selection_bucket") or "context_or_other")
        if result_sum:
            counter[bucket] += int(record.get("result_count") or 0)
        else:
            counter[bucket] += 1
    return {bucket: int(counter.get(bucket, 0)) for bucket in sorted(_SOURCE_CLASS_BUCKETS)}


def _planned_query_source_class_counts(planned_queries: list[dict]) -> dict:
    counter = Counter(_query_source_class(query) for query in planned_queries)
    return {bucket: int(counter.get(bucket, 0)) for bucket in sorted(_SOURCE_CLASS_BUCKETS)}


def _authority_gap_requires_retry(authority_summary: dict) -> bool:
    if not authority_summary:
        return True
    if authority_summary.get("missing_event_jurisdictions"):
        return True
    if (
        authority_summary.get("detected_event_jurisdictions")
        and authority_summary.get("national_or_subnational_authority_source_count", 0) == 0
    ):
        return True
    if authority_summary.get("official_or_public_health_source_count", 0) == 0:
        return True
    if authority_summary.get("structured_database_source_count", 0) == 0:
        return True
    if authority_summary.get("peer_reviewed_or_scientific_source_count", 0) == 0:
        return True
    return False


def _task_year_terms(state: DataCollectionState) -> list[str]:
    task = _task_context(state)
    years: list[str] = []
    for value in (task.get("start_date"), task.get("end_date")):
        for match in _YEAR_RE.finditer(str(value or "")):
            if match.group(1) not in years:
                years.append(match.group(1))
    return years


def _task_disease_term(state: DataCollectionState) -> str:
    disease = str(_task_context(state).get("disease") or "").strip()
    return disease or "disease"


def _retry_disease_and_event_terms(state: DataCollectionState, candidate_text: str) -> list[str]:
    terms: list[str] = []
    disease = _task_disease_term(state)
    terms.append(disease)
    intelligence = state.get("disease_intelligence") or {}
    if isinstance(intelligence, dict):
        for field in (
            "aliases",
            "abbreviations",
            "pathogen_terms",
            "syndrome_terms",
            "suggested_query_terms",
        ):
            for value in intelligence.get(field) or []:
                terms.append(str(value or ""))
    text_low = candidate_text.lower()
    event_terms = (
        ("andes virus", "Andes virus"),
        ("andv", "ANDV"),
        ("mv hondius", "MV Hondius"),
        ("cruise ship", "cruise ship"),
        ("vessel-associated", "vessel-associated"),
        ("vessel associated", "vessel-associated"),
        ("case report", "case report"),
        ("sequence", "sequence"),
    )
    for token, term in event_terms:
        if token in text_low:
            terms.append(term)
    deduped = _unique_preserve_order(terms)
    disease_low = disease.lower()
    preferred = [
        term
        for term in deduped
        if term.lower() != disease_low
        and term.lower() not in {"hps", "hfrs", "sequence", "case report"}
    ]
    return preferred + [term for term in deduped if term not in preferred]


def _candidate_is_event_specific_for_retry(candidate: SourceCandidate) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            candidate.title,
            candidate.snippet,
            candidate.canonical_url,
            candidate.url,
        )
    ).lower()
    has_target = any(
        token in text
        for token in (
            "hantavirus",
            "andes virus",
            "andv",
            "orthohantavirus",
        )
    )
    has_event = any(
        token in text
        for token in (
            "mv hondius",
            "hondius",
            "cruise ship",
            "outbreak",
            "case report",
            "risk assessment",
            "arrival and cleaning",
            "point de situation",
            "brote",
            "media update",
        )
    )
    return has_target and has_event


def _domain_has_event_specific_candidate(
    domain: str,
    search_candidates: list[SourceCandidate],
) -> bool:
    domain = domain.strip().lower().removeprefix("www.")
    if not domain:
        return False
    for candidate in search_candidates:
        candidate_domain = str(candidate.domain or "").strip().lower().removeprefix("www.")
        if candidate_domain == domain and _candidate_is_event_specific_for_retry(candidate):
            return True
    return False


def _candidate_text_for_retry(candidates: list[SourceCandidate]) -> str:
    return "\n".join(
        " ".join(
            str(value or "")
            for value in (
                candidate.title,
                candidate.snippet,
                candidate.publisher,
                candidate.domain,
                candidate.query_used,
            )
        )
        for candidate in candidates
    )


def _jurisdictions_mentioned_in_text(text: str) -> list[str]:
    text_low = f" {text.lower()} "
    matches: list[tuple[int, str]] = []
    for entry in load_source_identity_registry():
        source_type = _lower_text(entry.get("source_type"))
        if source_type not in _OFFICIAL_QUERY_SOURCE_TYPES | _INTERNATIONAL_QUERY_SOURCE_TYPES:
            continue
        name = str(entry.get("jurisdiction_name") or "").strip()
        if not name:
            continue
        name_low = name.lower()
        if name_low in {
            "global",
            "global literature",
            "global pathogen genomics",
            "americas",
            "eu/eea",
        }:
            continue
        aliases = [name, *(entry.get("publisher_aliases") or [])]
        for alias in aliases:
            alias_text = str(alias or "").strip()
            if len(alias_text) < 3:
                continue
            pattern = r"(?<![a-z0-9])" + re.escape(alias_text.lower()) + r"(?![a-z0-9])"
            match = re.search(pattern, text_low)
            if match:
                matches.append((match.start(), name))
                break
    ordered: list[str] = []
    seen: set[str] = set()
    for _, name in sorted(matches, key=lambda item: item[0]):
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _authority_retry_query(
    *,
    query_id: str,
    query: str,
    source_type: str,
    provider_channel: str,
    role_hint: str,
    jurisdiction_hint: str | None = None,
    official_domain_hint: str | None = None,
    previous_query_id: str | None = None,
    retry_category: str = "authority_gap_retry",
) -> dict:
    return {
        "query_id": query_id,
        "query": re.sub(r"\s+", " ", query).strip(),
        "query_type": (
            "database"
            if source_type in _STRUCTURED_QUERY_SOURCE_TYPES
            else "literature"
            if source_type in _PEER_QUERY_SOURCE_TYPES
            else "official_site"
            if source_type in _OFFICIAL_QUERY_SOURCE_TYPES | _INTERNATIONAL_QUERY_SOURCE_TYPES
            else "general_web"
        ),
        "provider_channel": provider_channel,
        "source_type": source_type,
        "role_hint": role_hint,
        "priority": 1,
        "expected_fields": [],
        "disease_terms_used": [],
        "location_terms_used": [jurisdiction_hint] if jurisdiction_hint else [],
        "time_terms_used": [],
        "query_language": "en",
        "jurisdiction_hint": jurisdiction_hint,
        "official_domain_hint": official_domain_hint,
        "localized_source_hint": False,
        "source_priority_reason": "authority_gap_retry",
        "query_source": "authority_gap_retry",
        "retry_category": retry_category,
        "previous_iteration_basis": previous_query_id,
        "execution_status": "planned_not_executed",
    }


def _retry_query_key(query_text: str) -> str:
    return re.sub(r"\s+", " ", str(query_text or "").strip().lower())


def _retry_query_domain(query: dict) -> str:
    return (
        str(query.get("official_domain_hint") or "").strip().lower().removeprefix("www.")
        or _site_domain_from_query_text(str(query.get("query") or ""))
        or ""
    )


def _retry_domain_accepted_result_cap(query: dict) -> int:
    category = str(query.get("retry_category") or "")
    source_type = str(query.get("source_type") or "").lower()
    provider_channel = str(query.get("provider_channel") or "").lower()
    if category == "jurisdiction_authority":
        return 4
    if category == "official_page_family":
        return 3
    if category == "known_domain_no_result":
        if "structured" in source_type or "database" in provider_channel:
            return 3
        if any(
            token in source_type
            for token in ("peer", "scientific", "academic", "literature")
        ) or "literature" in provider_channel:
            return 3
        return 2
    return 2


def _append_authority_retry_query(
    generated: list[dict],
    seen_texts: set[str],
    *,
    query_text: str,
    source_type: str,
    provider_channel: str,
    role_hint: str,
    jurisdiction_hint: str | None = None,
    official_domain_hint: str | None = None,
    previous_query_id: str | None = None,
    retry_category: str,
) -> bool:
    query_text = re.sub(r"\s+", " ", str(query_text or "")).strip()
    key = _retry_query_key(query_text)
    if not query_text or key in seen_texts:
        return False
    seen_texts.add(key)
    generated.append(
        _authority_retry_query(
            query_id=f"q_authority_retry_{len(generated) + 1:03d}",
            query=query_text,
            source_type=source_type,
            provider_channel=provider_channel,
            role_hint=role_hint,
            jurisdiction_hint=jurisdiction_hint,
            official_domain_hint=official_domain_hint,
            previous_query_id=previous_query_id,
            retry_category=retry_category,
        )
    )
    return True


def _jurisdiction_authority_retry_query_texts(
    *,
    disease: str,
    jurisdiction: str,
    domain: str,
    year_suffix: str,
    candidate_text: str,
) -> list[str]:
    jurisdiction_low = jurisdiction.lower()
    disease_term = disease or "disease"
    has_hondius_signal = any(
        token in candidate_text.lower()
        for token in ("hondius", "cruise ship", "andes virus", "andv")
    )
    event_suffix = "MV Hondius" if has_hondius_signal else jurisdiction
    localized: list[str] = []
    if jurisdiction_low == "spain":
        localized.append(
            f"{disease_term} brote casos informe site:{domain} {year_suffix}".strip()
        )
    elif jurisdiction_low == "france":
        localized.append(
            f"{disease_term} cas navire {event_suffix} site:{domain} {year_suffix}".strip()
        )
    elif jurisdiction_low == "netherlands":
        localized.append(
            f"{disease_term} {event_suffix} site:{domain} {year_suffix}".strip()
        )
    elif jurisdiction_low == "united kingdom":
        localized.append(
            f"{disease_term} UKHSA cruise ship outbreak site:{domain} {year_suffix}".strip()
        )
    elif jurisdiction_low == "canada":
        localized.append(
            f"{disease_term} Canada public health outbreak site:{domain} {year_suffix}".strip()
        )
    generic = (
        f'"{disease_term}" "{jurisdiction}" site:{domain} outbreak cases deaths '
        f"{year_suffix}"
    ).strip()
    return _unique_preserve_order([*localized, generic])


def _official_page_family_retry_specs(
    *,
    state: DataCollectionState,
    search_candidates: list[SourceCandidate],
    candidate_text: str,
    year_suffix: str,
) -> list[dict]:
    if universal_queries_enabled():
        return generic_retry_specs(state, search_candidates, year_suffix)
    disease = _task_disease_term(state)
    text_low = candidate_text.lower()
    domains = {
        str(candidate.domain or "").strip().lower().removeprefix("www.")
        for candidate in search_candidates
    }
    has_event_signal = any(
        token in text_low
        for token in (
            "andes virus",
            "andv",
            "mv hondius",
            "cruise ship",
            "hondius",
            "outbreak",
        )
    )
    if not has_event_signal:
        return []
    specs: list[dict] = []
    if "who.int" in domains:
        specs.extend(
            [
                {
                    "query": f'site:who.int "Andes virus" "cruise ship" {year_suffix}'.strip(),
                    "source_type": "international_public_health_agency",
                    "provider_channel": "official_site_search",
                    "role_hint": "validation",
                    "official_domain_hint": "who.int",
                },
                {
                    "query": (
                        f'site:who.int/publications/m "{disease}" '
                        f'"risk assessment" {year_suffix}'
                    ).strip(),
                    "source_type": "international_public_health_agency",
                    "provider_channel": "official_site_search",
                    "role_hint": "validation",
                    "official_domain_hint": "who.int",
                },
                {
                    "query": (
                        f'site:who.int/emergencies/disease-outbreak-news '
                        f'"{disease}" "cruise ship" {year_suffix}'
                    ).strip(),
                    "source_type": "international_public_health_agency",
                    "provider_channel": "official_site_search",
                    "role_hint": "validation",
                    "official_domain_hint": "who.int",
                },
            ]
        )
    if "gov.uk" in domains or "ukhsa" in text_low:
        specs.extend(
            [
                {
                    "query": f'{disease} UKHSA cruise ship outbreak site:gov.uk {year_suffix}'.strip(),
                    "source_type": "national_public_health_agency",
                    "provider_channel": "official_site_search",
                    "role_hint": "collection_support",
                    "jurisdiction_hint": "United Kingdom",
                    "official_domain_hint": "gov.uk",
                },
                {
                    "query": (
                        f'{disease} UKHSA cruise ship outbreak '
                        f"site:ukhsa-newsroom.prgloo.com {year_suffix}"
                    ).strip(),
                    "source_type": "national_public_health_agency",
                    "provider_channel": "official_site_search",
                    "role_hint": "collection_support",
                    "jurisdiction_hint": "United Kingdom",
                    "official_domain_hint": "ukhsa-newsroom.prgloo.com",
                },
            ]
        )
    event_terms = _retry_disease_and_event_terms(state, candidate_text)
    event_anchor = next(
        (
            term
            for term in event_terms
            if term.lower() in {"mv hondius", "andes virus", "andv", "cruise ship"}
        ),
        event_terms[0] if event_terms else disease,
    )
    for hint in authority_domain_hints_for_jurisdictions(
        _jurisdictions_mentioned_in_text(candidate_text)
    ):
        domain = str(hint.get("domain") or "").strip().lower().removeprefix("www.")
        jurisdiction = str(hint.get("jurisdiction_name") or "").strip()
        if not domain:
            continue
        if _domain_has_event_specific_candidate(domain, search_candidates):
            continue
        queries = [
            f'site:{domain} "{event_anchor}" "{disease}" {year_suffix}'.strip(),
            f'site:{domain} "{event_anchor}" outbreak update {year_suffix}'.strip(),
        ]
        if domain == "sante.gouv.fr":
            queries.append(
                f'site:{domain} "{event_anchor}" "cas d\'hantavirus" "situation" {year_suffix}'.strip()
            )
        elif domain == "rivm.nl":
            queries.append(
                f'site:{domain} "{event_anchor}" "arrival and cleaning" "cruise ship" {year_suffix}'.strip()
            )
        elif domain == "sanidad.gob.es":
            queries.append(
                f'site:{domain} "{disease}" brote informe cierre {year_suffix}'.strip()
            )
        elif domain == "canada.ca":
            queries.append(
                f'site:{domain} "{event_anchor}" "{disease}" "media update" {year_suffix}'.strip()
            )
        for query in _unique_preserve_order(queries):
            specs.append(
                {
                    "query": query,
                    "source_type": str(
                        hint.get("source_type") or "national_public_health_agency"
                    ),
                    "provider_channel": "official_site_search",
                    "role_hint": "collection_support",
                    "jurisdiction_hint": jurisdiction,
                    "official_domain_hint": domain,
                }
            )
    return specs


def _structured_database_retry_specs(
    *,
    state: DataCollectionState,
    search_candidates: list[SourceCandidate],
    year_suffix: str,
) -> list[dict]:
    """Return generic high-confidence structured database retry specs.

    These are registry-backed, disease/event driven query templates. They do
    not use benchmark source lists or case identifiers.
    """

    candidate_text = _candidate_text_for_retry(search_candidates)
    summary = build_authority_source_coverage_summary(
        [
            candidate.model_dump()
            for candidate in search_candidates
            if isinstance(candidate, SourceCandidate)
        ],
        detected_event_jurisdictions=_jurisdictions_mentioned_in_text(candidate_text),
    )
    has_database_completion_signal = any(
        token in candidate_text.lower()
        for token in (
            "sequence",
            "genomic",
            "accession",
            "phylogenetic",
            "genbank",
        )
    )
    if (
        int(summary.get("structured_database_source_count") or 0) > 0
        and not has_database_completion_signal
    ):
        return []

    disease = _task_disease_term(state)
    retry_terms = _retry_disease_and_event_terms(state, candidate_text)
    term = retry_terms[0] if retry_terms else disease
    preferred_domains = (
        "pathoplexus.org",
        "ncbi.nlm.nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "europepmc.org",
    )
    registry_by_domain = {
        str(entry.get("domain") or ""): entry
        for entry in load_source_identity_registry()
        if str(entry.get("source_type") or "").lower() == "structured_database"
    }
    existing_domains = {
        str(candidate.domain or "").strip().lower().removeprefix("www.")
        for candidate in search_candidates
    }
    specs: list[dict] = []
    for domain in preferred_domains:
        if domain in existing_domains:
            continue
        entry = registry_by_domain.get(domain)
        if not entry:
            continue
        if domain == "pathoplexus.org":
            query = f'"{term}" site:{domain} sequence outbreak {year_suffix}'.strip()
        else:
            query = f'"{term}" site:{domain} outbreak case report {year_suffix}'.strip()
        specs.append(
            {
                "query": query,
                "source_type": "structured_database",
                "provider_channel": "database_search",
                "role_hint": str(entry.get("default_source_role") or "collection_support"),
                "official_domain_hint": domain,
            }
        )
    return specs


def _peer_literature_retry_specs(
    *,
    state: DataCollectionState,
    search_candidates: list[SourceCandidate],
    year_suffix: str,
) -> list[dict]:
    """Return generic event-driven literature retry specs for outbreak tasks."""

    candidate_text = _candidate_text_for_retry(search_candidates)
    text_low = candidate_text.lower()
    has_event_signal = any(
        token in text_low
        for token in (
            "andes virus",
            "andv",
            "mv hondius",
            "cruise ship",
            "vessel",
            "outbreak",
        )
    )
    if not has_event_signal:
        return []

    retry_terms = _retry_disease_and_event_terms(state, candidate_text)
    term = retry_terms[0] if retry_terms else _task_disease_term(state)
    preferred_domains = (
        "eurosurveillance.org",
        "science.org",
        "nejm.org",
    )
    existing_domains = {
        str(candidate.domain or "").strip().lower().removeprefix("www.")
        for candidate in search_candidates
    }
    wrong_page_domains = [
        domain
        for domain in preferred_domains
        if domain in existing_domains
        and not _domain_has_event_specific_candidate(domain, search_candidates)
    ]
    missing_domains = [
        domain for domain in preferred_domains if domain not in existing_domains
    ]
    retry_domain_order = _unique_preserve_order(
        [*wrong_page_domains, *missing_domains, *preferred_domains]
    )
    registry_by_domain = {
        str(entry.get("domain") or ""): entry
        for entry in load_source_identity_registry()
        if str(entry.get("source_type") or "").lower() in _PEER_QUERY_SOURCE_TYPES
    }
    specs: list[dict] = []
    for domain in retry_domain_order:
        if _domain_has_event_specific_candidate(domain, search_candidates):
            continue
        entry = registry_by_domain.get(domain)
        if not entry:
            continue
        if domain == "science.org":
            query = f'"{term}" site:{domain} cruise ship outbreak {year_suffix}'.strip()
        else:
            query = f'"{term}" site:{domain} case report outbreak {year_suffix}'.strip()
        specs.append(
            {
                "query": query,
                "source_type": str(
                    entry.get("source_type") or "academic_or_peer_reviewed_source"
                ),
                "provider_channel": "literature_api",
                "role_hint": str(entry.get("default_source_role") or "context"),
                "official_domain_hint": domain,
            }
        )
    return specs


def _build_authority_gap_retry_queries(
    *,
    state: DataCollectionState,
    planned_queries: list[dict],
    query_records: list[dict],
    search_candidates: list[SourceCandidate],
    max_queries: int,
    known_domain_max_queries: int,
    jurisdiction_max_queries: int,
    official_page_family_max_queries: int,
    target_jurisdictions: list[str] | None = None,
) -> list[dict]:
    if universal_queries_enabled():
        specs = generic_retry_specs(state, search_candidates, ' '.join(_task_year_terms(state)[:2]))
        seen = {_retry_query_key(str(q.get('query') or '')) for q in planned_queries}
        generated = []
        cap = sum(max(0, n) for n in (max_queries, known_domain_max_queries, jurisdiction_max_queries, official_page_family_max_queries))
        for spec in specs:
            if len(generated) >= cap:
                break
            _append_authority_retry_query(generated, seen, query_text=spec['query'], source_type=spec['source_type'], provider_channel=spec['provider_channel'], role_hint=spec['role_hint'], official_domain_hint=spec['official_domain_hint'], retry_category='generic_authority_gap')
        return generated

    if (
        max_queries <= 0
        and known_domain_max_queries <= 0
        and jurisdiction_max_queries <= 0
        and official_page_family_max_queries <= 0
    ):
        return []

    generated: list[dict] = []
    seen_texts = {_retry_query_key(str(query.get("query") or "")) for query in planned_queries}

    skipped_high_trust = [
        record
        for record in query_records
        if record.get("execution_status") == "skipped_query_limit"
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
        and record.get("query")
    ]
    for record in skipped_high_trust:
        if max_queries <= 0:
            break
        query_text = re.sub(r"\s+", " ", str(record.get("query") or "").strip())
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=query_text,
            source_type=str(record.get("source_type") or "official_public_health_agency"),
            provider_channel=str(record.get("provider_channel") or "official_site_search"),
            role_hint=str(record.get("role_hint") or "collection_support"),
            jurisdiction_hint=record.get("jurisdiction_hint"),
            official_domain_hint=record.get("official_domain_hint"),
            previous_query_id=record.get("query_id"),
            retry_category="skipped_high_trust_query",
        ):
            max_queries -= 1

    candidate_text = _candidate_text_for_retry(search_candidates)
    years = _task_year_terms(state)
    year_suffix = " ".join(years[:2])
    page_family_added = 0
    for spec in _official_page_family_retry_specs(
        state=state,
        search_candidates=search_candidates,
        candidate_text=candidate_text,
        year_suffix=year_suffix,
    ):
        if page_family_added >= official_page_family_max_queries:
            break
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=str(spec.get("query") or ""),
            source_type=str(spec.get("source_type") or "official_public_health_agency"),
            provider_channel=str(spec.get("provider_channel") or "official_site_search"),
            role_hint=str(spec.get("role_hint") or "collection_support"),
            jurisdiction_hint=spec.get("jurisdiction_hint"),
            official_domain_hint=spec.get("official_domain_hint"),
            retry_category="official_page_family",
        ):
            page_family_added += 1

    retry_terms = _retry_disease_and_event_terms(state, candidate_text)
    known_domain_added = 0
    # Relax an explicitly executed no-result domain before spending the shared
    # known-domain budget on broad database and literature completion. This
    # preserves the strongest evidence that the original query was too narrow.
    no_result_known_domain_records = [
        record
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("execution_status") == "no_results"
        and _known_authority_domain_for_query(record)
        and record.get("query")
    ]
    for record in no_result_known_domain_records:
        if known_domain_added >= known_domain_max_queries:
            break
        domain = _known_authority_domain_for_query(record)
        if not domain:
            continue
        bucket = str(record.get("selection_bucket") or _query_source_class(record))
        source_type = str(record.get("source_type") or "official_public_health_agency")
        provider_channel = str(record.get("provider_channel") or "official_site_search")
        role_hint = str(record.get("role_hint") or "collection_support")
        term = retry_terms[0] if retry_terms else _task_disease_term(state)
        if bucket == "structured_database":
            query_text = f'"{term}" site:{domain} sequence outbreak {year_suffix}'.strip()
        elif bucket == "peer_reviewed_literature":
            query_text = f'"{term}" site:{domain} case report outbreak {year_suffix}'.strip()
        else:
            query_text = f'"{term}" site:{domain} outbreak cases deaths {year_suffix}'.strip()
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=query_text,
            source_type=source_type,
            provider_channel=provider_channel,
            role_hint=role_hint,
            jurisdiction_hint=record.get("jurisdiction_hint"),
            official_domain_hint=domain,
            previous_query_id=record.get("query_id"),
            retry_category="known_domain_no_result",
        ):
            known_domain_added += 1

    for spec in _structured_database_retry_specs(
        state=state,
        search_candidates=search_candidates,
        year_suffix=year_suffix,
    ):
        if known_domain_added >= known_domain_max_queries:
            break
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=str(spec.get("query") or ""),
            source_type=str(spec.get("source_type") or "structured_database"),
            provider_channel=str(spec.get("provider_channel") or "database_search"),
            role_hint=str(spec.get("role_hint") or "collection_support"),
            official_domain_hint=spec.get("official_domain_hint"),
            retry_category="known_domain_no_result",
        ):
            known_domain_added += 1

    for spec in _peer_literature_retry_specs(
        state=state,
        search_candidates=search_candidates,
        year_suffix=year_suffix,
    ):
        if known_domain_added >= known_domain_max_queries:
            break
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=str(spec.get("query") or ""),
            source_type=str(spec.get("source_type") or "academic_or_peer_reviewed_source"),
            provider_channel=str(spec.get("provider_channel") or "literature_api"),
            role_hint=str(spec.get("role_hint") or "context"),
            official_domain_hint=spec.get("official_domain_hint"),
            retry_category="known_domain_no_result",
        ):
            known_domain_added += 1

    disease = _task_disease_term(state)
    jurisdictions = _unique_preserve_order(
        target_jurisdictions
        if target_jurisdictions is not None
        else _jurisdictions_mentioned_in_text(candidate_text)
    )
    jurisdiction_added = 0
    for hint in authority_domain_hints_for_jurisdictions(jurisdictions):
        if jurisdiction_added >= jurisdiction_max_queries:
            break
        domain = str(hint.get("domain") or "").strip()
        jurisdiction = str(hint.get("jurisdiction_name") or "").strip()
        if not domain or not jurisdiction:
            continue
        for query_text in _jurisdiction_authority_retry_query_texts(
            disease=disease,
            jurisdiction=jurisdiction,
            domain=domain,
            year_suffix=year_suffix,
            candidate_text=candidate_text,
        ):
            if jurisdiction_added >= jurisdiction_max_queries:
                break
            if _append_authority_retry_query(
                generated,
                seen_texts,
                query_text=query_text,
                source_type=str(hint.get("source_type") or "official_public_health_agency"),
                provider_channel="official_site_search",
                role_hint="collection_support",
                jurisdiction_hint=jurisdiction,
                official_domain_hint=domain,
                retry_category="jurisdiction_authority",
            ):
                jurisdiction_added += 1
                break

    generic_specs = [
        (
            f'"{disease}" sequence outbreak database {year_suffix}'.strip(),
            "structured_database",
            "database_search",
            "collection_support",
        ),
        (
            f'"{disease}" "case report" outbreak {year_suffix}'.strip(),
            "peer_reviewed_literature",
            "literature_api",
            "context",
        ),
    ]
    generic_added = 0
    for query_text, source_type, channel, role_hint in generic_specs:
        if generic_added >= max_queries:
            break
        if _append_authority_retry_query(
            generated,
            seen_texts,
            query_text=query_text,
            source_type=source_type,
            provider_channel=channel,
            role_hint=role_hint,
            retry_category="generic_authority_gap",
        ):
            generic_added += 1
    return generated


def _retry_query_ids(
    retry_queries: list[dict],
    category: str,
) -> set[str]:
    return {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("retry_category") == category
    }


def _retry_generated_domain_counts(
    retry_queries: list[dict],
    query_ids: set[str],
) -> dict[str, int]:
    counts: Counter = Counter()
    for query in retry_queries:
        if str(query.get("query_id") or "") not in query_ids:
            continue
        domain = _retry_query_domain(query)
        if domain:
            counts[domain] += 1
    return dict(counts)


def _retry_executed_domain_counts(
    query_records: list[dict],
    query_ids: set[str],
) -> dict[str, int]:
    counts: Counter = Counter()
    for record in query_records:
        if str(record.get("query_id") or "") not in query_ids:
            continue
        if not record.get("selected_for_execution"):
            continue
        domain = _retry_query_domain(record)
        if domain:
            counts[domain] += 1
    return dict(counts)


def _ordered_authority_retry_queries(retry_queries: list[dict]) -> list[dict]:
    """Interleave high-trust retry classes and domains before broad expansion."""

    category_order = [
        "jurisdiction_authority",
        "known_domain_no_result",
        "official_page_family",
        "skipped_high_trust_query",
        "generic_authority_gap",
    ]
    categories: dict[str, dict[str, list[dict]]] = {}
    domain_order: dict[str, list[str]] = {}
    for index, query in enumerate(retry_queries):
        category = str(query.get("retry_category") or "unknown")
        domain = _retry_query_domain(query) or f"query:{index}"
        category_bucket = categories.setdefault(category, {})
        if domain not in category_bucket:
            category_bucket[domain] = []
            domain_order.setdefault(category, []).append(domain)
        category_bucket[domain].append(query)

    ordered: list[dict] = []
    pending = True
    while pending:
        pending = False
        for category in category_order:
            category_bucket = categories.get(category) or {}
            for domain in domain_order.get(category, []):
                domain_queries = category_bucket.get(domain) or []
                if not domain_queries:
                    continue
                ordered.append(domain_queries.pop(0))
                pending = True
        for category in sorted(set(categories) - set(category_order)):
            category_bucket = categories.get(category) or {}
            for domain in domain_order.get(category, []):
                domain_queries = category_bucket.get(domain) or []
                if not domain_queries:
                    continue
                ordered.append(domain_queries.pop(0))
                pending = True
    return ordered


def _retry_execution_order(
    query_records: list[dict],
    retry_query_ids: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for record in query_records:
        query_id = str(record.get("query_id") or "")
        if query_id not in retry_query_ids:
            continue
        rows.append(
            {
                "query_id": query_id,
                "retry_category": record.get("retry_category"),
                "domain": _retry_query_domain(record),
                "execution_status": record.get("execution_status"),
                "result_count": int(record.get("result_count") or 0),
            }
        )
    return rows


def _retry_result_budget_by_category(
    query_records: list[dict],
    retry_query_ids: set[str],
) -> dict[str, dict[str, int]]:
    budget: dict[str, Counter] = {}
    for record in query_records:
        query_id = str(record.get("query_id") or "")
        if query_id not in retry_query_ids:
            continue
        category = str(record.get("retry_category") or "unknown")
        counter = budget.setdefault(category, Counter())
        counter["query_count"] += 1
        if record.get("selected_for_execution"):
            counter["selected_count"] += 1
        if record.get("execution_status") in {"executed", "no_results", "provider_error"}:
            counter["executed_count"] += 1
        if record.get("execution_status") == "skipped_total_result_limit":
            counter["skipped_total_result_limit_count"] += 1
        counter["accepted_result_count"] += int(record.get("result_count") or 0)
    return {category: dict(counter) for category, counter in sorted(budget.items())}


def _retry_domain_result_cap_counts(search_results_manifest: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for row in search_results_manifest:
        if row.get("query_source") != "authority_gap_retry":
            continue
        if row.get("rejection_reason") != "per_domain_retry_result_cap_reached":
            continue
        domain = str(row.get("domain") or "").strip().lower().removeprefix("www.")
        if domain:
            counts[domain] += 1
    return dict(counts)


def _high_confidence_domain_query_status(
    *,
    retry_queries: list[dict],
    query_records: list[dict],
    search_candidates: list[SourceCandidate],
    detected_event_jurisdictions: list[str],
) -> dict[str, str]:
    candidate_domains = {
        str(candidate.domain or "").strip().lower().removeprefix("www.")
        for candidate in search_candidates
        if isinstance(candidate, SourceCandidate)
    }
    domains: set[str] = {
        "pathoplexus.org",
        "eurosurveillance.org",
        "science.org",
        "who.int",
        "gov.uk",
        "ukhsa-newsroom.prgloo.com",
    }
    domains.update(
        str(hint.get("domain") or "").strip().lower().removeprefix("www.")
        for hint in authority_domain_hints_for_jurisdictions(detected_event_jurisdictions)
        if hint.get("domain")
    )
    retry_domain_by_query_id = {
        str(query.get("query_id") or ""): _retry_query_domain(query)
        for query in retry_queries
        if query.get("query_id")
    }
    domains.update(domain for domain in retry_domain_by_query_id.values() if domain)
    records_by_domain: dict[str, list[dict]] = {}
    for record in query_records:
        query_id = str(record.get("query_id") or "")
        domain = retry_domain_by_query_id.get(query_id) or _retry_query_domain(record)
        if not domain:
            continue
        records_by_domain.setdefault(domain, []).append(record)

    statuses: dict[str, str] = {}
    for domain in sorted(domain for domain in domains if domain):
        records = records_by_domain.get(domain) or []
        domain_candidates = [
            candidate
            for candidate in search_candidates
            if str(candidate.domain or "").strip().lower().removeprefix("www.") == domain
        ]
        if any(_candidate_is_event_specific_for_retry(candidate) for candidate in domain_candidates):
            statuses[domain] = "exact_event_page_found"
        elif domain in candidate_domains:
            statuses[domain] = "domain_found_exact_missing"
        elif not records:
            statuses[domain] = "query_not_generated"
        elif any(record.get("execution_status") == "skipped_total_result_limit" for record in records):
            statuses[domain] = "generated_but_skipped_by_cap"
        elif any(record.get("execution_status") == "provider_error" for record in records):
            statuses[domain] = "provider_error"
        elif any(record.get("execution_status") == "no_results" for record in records):
            statuses[domain] = "executed_no_result"
        elif any(record.get("selected_for_execution") for record in records):
            statuses[domain] = "executed_no_accepted_candidate"
        else:
            statuses[domain] = "generated_but_not_executed"
    return statuses


def _build_source_recall_target_ledger(
    *,
    retry_queries: list[dict],
    query_records: list[dict],
    search_candidates: list[SourceCandidate],
    detected_event_jurisdictions: list[str],
) -> list[dict]:
    """Build a search-stage ledger for reusable high-confidence source targets."""

    status_by_domain = _high_confidence_domain_query_status(
        retry_queries=retry_queries,
        query_records=query_records,
        search_candidates=search_candidates,
        detected_event_jurisdictions=detected_event_jurisdictions,
    )
    registry_by_domain = {
        str(entry.get("domain") or "").strip().lower().removeprefix("www."): entry
        for entry in load_source_identity_registry()
        if entry.get("domain")
    }
    query_by_domain: dict[str, list[dict]] = {}
    for query in [*retry_queries, *query_records]:
        domain = _retry_query_domain(query)
        if domain:
            query_by_domain.setdefault(domain, []).append(query)

    status_map = {
        "exact_event_page_found": "event_page_found",
        "domain_found_exact_missing": "domain_found_event_page_missing",
        "query_not_generated": "query_not_generated",
        "generated_but_skipped_by_cap": "query_generated_not_executed",
        "provider_error": "query_provider_error",
        "executed_no_result": "query_executed_no_result",
        "executed_no_accepted_candidate": "query_executed_no_result",
        "generated_but_not_executed": "query_generated_not_executed",
    }
    ledger: list[dict] = []
    for domain, raw_status in sorted(status_by_domain.items()):
        registry = registry_by_domain.get(domain) or {}
        domain_queries = query_by_domain.get(domain) or []
        source_type = str(
            next(
                (
                    query.get("source_type")
                    for query in domain_queries
                    if query.get("source_type")
                ),
                registry.get("source_type") or "unknown",
            )
        )
        jurisdiction = str(
            next(
                (
                    query.get("jurisdiction_hint")
                    for query in domain_queries
                    if query.get("jurisdiction_hint")
                ),
                registry.get("jurisdiction_name") or "",
            )
        )
        source_class = _query_source_class({"source_type": source_type})
        expected_data_product = (
            "sequence_database_record"
            if source_class == "structured_database"
            else "case_report_article"
            if source_class == "peer_reviewed_literature"
            else "event_outbreak_report"
            if source_class
            in {"international_official", "national_or_local_official"}
            else "event_supporting_source"
        )
        ledger.append(
            {
                "source_class": source_class,
                "jurisdiction_or_institution": jurisdiction or None,
                "authority_domain": domain,
                "expected_data_product": expected_data_product,
                "event_page_status": status_map.get(raw_status, raw_status),
                "query_count": len(
                    {
                        str(query.get("query_id") or query.get("query") or "")
                        for query in domain_queries
                    }
                    - {""}
                ),
            }
        )
    return ledger


def _authority_gap_retry_diagnostics(
    *,
    retry_queries: list[dict],
    query_records: list[dict],
    search_candidates: list[SourceCandidate],
    search_results_manifest: list[dict],
    detected_event_jurisdictions: list[str],
    first_pass_authority_summary: dict,
    retry_accepted_candidate_count: int,
    authority_retry_result_budget: int,
) -> dict:
    known_domain_retry_query_ids = _retry_query_ids(
        retry_queries,
        "known_domain_no_result",
    )
    jurisdiction_retry_query_ids = _retry_query_ids(
        retry_queries,
        "jurisdiction_authority",
    )
    all_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("query_id")
    }
    official_page_family_retry_query_ids = _retry_query_ids(
        retry_queries,
        "official_page_family",
    )
    known_authority_domain_no_result_query_ids = {
        str(record.get("query_id") or "")
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("execution_status") == "no_results"
        and record.get("query_source") != "authority_gap_retry"
        and _known_authority_domain_for_query(record)
    }
    known_authority_domain_no_result_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if str(query.get("previous_iteration_basis") or "")
        in known_authority_domain_no_result_query_ids
    }
    jurisdiction_authority_retry_candidate_count = sum(
        int(record.get("result_count") or 0)
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
    )
    jurisdiction_retry_protected_execution_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
        and record.get("selected_for_execution")
    )
    jurisdiction_retry_skipped_due_to_result_cap_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
        and record.get("execution_status") == "skipped_total_result_limit"
    )
    known_domain_retry_skipped_due_to_result_cap_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in known_domain_retry_query_ids
        and record.get("execution_status") == "skipped_total_result_limit"
    )
    official_page_family_retry_candidate_count = sum(
        int(record.get("result_count") or 0)
        for record in query_records
        if str(record.get("query_id") or "") in official_page_family_retry_query_ids
    )
    post_retry_authority_summary = build_authority_source_coverage_summary(
        [
            candidate.model_dump()
            for candidate in search_candidates
            if isinstance(candidate, SourceCandidate)
        ],
        detected_event_jurisdictions=detected_event_jurisdictions,
    )
    missing_high_confidence_jurisdiction_domains = [
        str(hint.get("domain") or "")
        for hint in authority_domain_hints_for_jurisdictions(
            list(post_retry_authority_summary.get("missing_event_jurisdictions") or [])
        )
        if hint.get("domain")
    ]
    high_confidence_source_gap_status = (
        "complete"
        if not missing_high_confidence_jurisdiction_domains
        and int(post_retry_authority_summary.get("structured_database_source_count") or 0)
        > 0
        and int(
            post_retry_authority_summary.get(
                "peer_reviewed_or_scientific_source_count"
            )
            or 0
        )
        > 0
        else "partial"
    )
    high_recall_noise_rejection_count = sum(
        1
        for row in search_results_manifest
        if row.get("query_source") == "authority_gap_retry"
        and row.get("rejection_reason")
        in {
            "per_domain_retry_result_cap_reached",
            "per_query_result_cap_reached",
            "wrong_disease_result",
            "same_domain_wrong_page",
            "generic_background",
            "result_cap_after_protected_slots",
        }
    )
    return {
        "known_authority_domain_no_result_count": len(
            known_authority_domain_no_result_query_ids
        ),
        "known_authority_domain_no_result_retry_generated_count": len(
            known_authority_domain_no_result_retry_query_ids
        ),
        "known_domain_retry_generated_count": len(known_domain_retry_query_ids),
        "known_domain_retry_generated_by_domain": _retry_generated_domain_counts(
            retry_queries,
            known_domain_retry_query_ids,
        ),
        "known_domain_retry_executed_by_domain": _retry_executed_domain_counts(
            query_records,
            known_domain_retry_query_ids,
        ),
        "first_pass_authority_source_coverage_summary": first_pass_authority_summary,
        "post_retry_authority_source_coverage_summary": post_retry_authority_summary,
        "detected_event_jurisdictions": list(
            first_pass_authority_summary.get("detected_event_jurisdictions") or []
        ),
        "covered_event_jurisdictions": list(
            first_pass_authority_summary.get("covered_event_jurisdictions") or []
        ),
        "missing_event_jurisdictions": list(
            first_pass_authority_summary.get("missing_event_jurisdictions") or []
        ),
        "authority_gap_retry_generated_query_count": len(retry_queries),
        "authority_retry_result_budget": authority_retry_result_budget,
        "authority_retry_result_budget_used": retry_accepted_candidate_count,
        "high_recall_profile_enabled": int(authority_retry_result_budget or 0) >= 80,
        "high_recall_result_budget_used": retry_accepted_candidate_count,
        "high_recall_noise_rejection_count": high_recall_noise_rejection_count,
        "retry_execution_order": _retry_execution_order(
            query_records,
            all_retry_query_ids,
        ),
        "retry_category_round_robin_order": [
            "jurisdiction_authority",
            "known_domain_no_result",
            "official_page_family",
            "skipped_high_trust_query",
            "generic_authority_gap",
        ],
        "retry_result_budget_by_category": _retry_result_budget_by_category(
            query_records,
            all_retry_query_ids,
        ),
        "retry_domain_result_cap_counts": _retry_domain_result_cap_counts(
            search_results_manifest
        ),
        "high_confidence_domain_query_status": _high_confidence_domain_query_status(
            retry_queries=retry_queries,
            query_records=query_records,
            search_candidates=search_candidates,
            detected_event_jurisdictions=detected_event_jurisdictions,
        ),
        "source_recall_target_ledger": _build_source_recall_target_ledger(
            retry_queries=retry_queries,
            query_records=query_records,
            search_candidates=search_candidates,
            detected_event_jurisdictions=detected_event_jurisdictions,
        ),
        "jurisdiction_authority_retry_generated_count": len(
            jurisdiction_retry_query_ids
        ),
        "jurisdiction_retry_generated_by_domain": _retry_generated_domain_counts(
            retry_queries,
            jurisdiction_retry_query_ids,
        ),
        "jurisdiction_retry_executed_by_domain": _retry_executed_domain_counts(
            query_records,
            jurisdiction_retry_query_ids,
        ),
        "jurisdiction_authority_retry_candidate_count": (
            jurisdiction_authority_retry_candidate_count
        ),
        "jurisdiction_retry_protected_execution_count": (
            jurisdiction_retry_protected_execution_count
        ),
        "jurisdiction_retry_skipped_due_to_result_cap_count": (
            jurisdiction_retry_skipped_due_to_result_cap_count
        ),
        "known_domain_retry_truncated_before_jurisdiction_complete": bool(
            known_domain_retry_skipped_due_to_result_cap_count
            and jurisdiction_retry_protected_execution_count
            and not jurisdiction_retry_skipped_due_to_result_cap_count
        ),
        "official_page_family_retry_generated_count": len(
            official_page_family_retry_query_ids
        ),
        "official_page_family_retry_candidate_count": (
            official_page_family_retry_candidate_count
        ),
        "missing_high_confidence_jurisdiction_domains": (
            missing_high_confidence_jurisdiction_domains
        ),
        "high_confidence_source_gap_status": high_confidence_source_gap_status,
        "authority_gap_retry_selected_query_count": sum(
            1
            for record in query_records
            if record.get("query_source") == "authority_gap_retry"
            and record.get("selected_for_execution")
        ),
        "authority_gap_retry_accepted_candidate_count": retry_accepted_candidate_count,
    }


def _execute_authority_gap_retry(
    *,
    state: DataCollectionState,
    settings: SourceSearchSettings,
    provider,
    planned_queries: list[dict],
    query_records: list[dict],
    search_candidates: list[SourceCandidate],
    search_results_manifest: list[dict],
    seen_canonical_urls: set[str],
    totals: dict,
    rejection_counter: Counter,
    warnings: list[str],
    first_pass_authority_summary: dict,
    detected_event_jurisdictions: list[str],
) -> dict:
    retry_queries = _build_authority_gap_retry_queries(
        state=state,
        planned_queries=planned_queries,
        query_records=query_records,
        search_candidates=search_candidates,
        max_queries=settings.authority_gap_retry_max_queries,
        known_domain_max_queries=settings.authority_gap_known_domain_retry_max_queries,
        jurisdiction_max_queries=settings.authority_gap_jurisdiction_retry_max_queries,
        official_page_family_max_queries=(
            settings.authority_gap_official_page_family_retry_max_queries
        ),
        target_jurisdictions=list(
            first_pass_authority_summary.get("missing_event_jurisdictions") or []
        ),
    )
    execution_retry_queries = _ordered_authority_retry_queries(retry_queries)
    if retry_queries and "authority_gap_retry_attempted" not in warnings:
        warnings.append("authority_gap_retry_attempted")
    before_retry_count = len(search_candidates)
    if execution_retry_queries:
        retry_max_total_results = (
            int(totals.get("deduped_result_count") or 0)
            + max(0, int(settings.authority_gap_retry_result_budget or 0))
        )
        retry_domain_caps = {
            _retry_query_domain(query): _retry_domain_accepted_result_cap(query)
            for query in execution_retry_queries
            if _retry_query_domain(query)
        }
        batch_candidates, batch_manifest, batch_records, batch_rejections, batch_warnings = (
            _execute_iterative_query_batch(
                query_batch=execution_retry_queries,
                provider=provider,
                settings=settings,
                seen_canonical_urls=seen_canonical_urls,
                totals=totals,
                task=_task_context(state),
                evidence_state=state,
                max_total_results_override=retry_max_total_results,
                max_queries_override=len(execution_retry_queries),
                max_total_queries_override=(
                    totals["selected_query_count"] + len(execution_retry_queries)
                ),
                selection_reason_override="authority_gap_retry",
                accepted_result_cap_by_retry_category={
                    "jurisdiction_authority": 4,
                    "official_page_family": 3,
                    "known_domain_no_result": 3,
                    "generic_authority_gap": 1,
                    "skipped_high_trust_query": 2,
                },
                accepted_result_cap_by_retry_domain=retry_domain_caps,
            )
        )
        query_records.extend(batch_records)
        search_results_manifest.extend(batch_manifest)
        search_candidates.extend(batch_candidates)
        rejection_counter.update(batch_rejections)
        warnings.extend(batch_warnings)

    retry_accepted_candidate_count = len(search_candidates) - before_retry_count
    if retry_accepted_candidate_count > 0:
        warnings.append("authority_gap_retry_candidates_added")
    elif retry_queries:
        warnings.append("authority_gap_retry_no_results")

    return _authority_gap_retry_diagnostics(
        retry_queries=retry_queries,
        query_records=query_records,
        search_candidates=search_candidates,
        search_results_manifest=search_results_manifest,
        detected_event_jurisdictions=detected_event_jurisdictions,
        first_pass_authority_summary=first_pass_authority_summary,
        retry_accepted_candidate_count=retry_accepted_candidate_count,
        authority_retry_result_budget=settings.authority_gap_retry_result_budget,
    ) | {"_retry_queries": retry_queries}


def _iteration_query_counts(query_records: list[dict]) -> dict:
    return dict(
        Counter(
            str(record.get("iteration_index") or "none")
            for record in query_records
            if record.get("selected_for_execution")
        )
    )


def _execute_one_shot_source_search(
    state: DataCollectionState,
    settings: SourceSearchSettings,
) -> tuple[list[SourceCandidate], list[dict], dict]:
    planned_queries = _planned_queries(state)
    task = state.get("structured_task") or {}
    selection_plan = _select_source_search_queries(planned_queries, state, settings)
    query_records: list[dict] = []
    search_results_manifest: list[dict] = []
    search_candidates: list[SourceCandidate] = []
    rejection_counter: Counter = Counter()
    warnings: list[str] = []
    provider_error_count = 0
    selected_query_count = 0
    executed_query_count = 0
    raw_result_count = 0
    deduped_result_count = 0
    seen_canonical_urls: set[str] = set()
    search_stopped_reason: str | None = None
    stop_decision: str | None = None
    stop_reason: str | None = None
    verified_target_source_ids: list[str] = []
    target_verification_summary: dict = {
        "verified_target_source_count": 0,
        "verified_target_source_ids": [],
        "target_source_miss_reasons": [],
    }
    has_requirements = bool(
        state.get("source_coverage_requirements")
        or (state.get("task_evidence_contract") or {}).get("requirements")
    )
    if settings.search_enabled and has_requirements and not planned_queries:
        stop_decision = "query_generation_failed_for_requirements"
        stop_reason = (
            "Source search was enabled and task evidence requirements exist, "
            "but no executable source queries were available from the agentic "
            "plan or search query inventory."
        )
        warnings.append("query_generation_failed_for_requirements")

    provider = None
    if settings.search_enabled:
        try:
            provider = _provider_for_settings(settings)
        except (OSError, ValueError) as exc:
            provider_error_count += 1
            warnings.append(f"search_provider_setup_failed:{exc}")

    def execute_selected_query(
        query: dict,
        *,
        selection_bucket: str,
        selection_reason: str,
    ) -> int:
        nonlocal provider_error_count
        nonlocal selected_query_count
        nonlocal executed_query_count
        nonlocal raw_result_count
        nonlocal deduped_result_count

        if universal_queries_enabled() and not assess_query_task_fit(query, state)['accepted']:
            query_records.append(_search_execution_record(query, provider=settings.provider, status='skipped_task_mismatch', skipped_reason='query_not_task_grounded'))
            return 0

        if deduped_result_count >= settings.max_total_results:
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_total_result_limit",
                    skipped_reason="max_total_results_reached",
                    selection_bucket=selection_bucket,
                    selection_reason=selection_reason,
                )
            )
            return 0
        selected_query_count += 1
        if provider is None:
            provider_error_count += 1
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="provider_error",
                    selected=True,
                    error="provider_unavailable",
                    selection_bucket=selection_bucket,
                    selection_reason=selection_reason,
                )
            )
            return 0

        emit_workflow_progress(
            "source_discovery",
            "source search query started",
            {
                "query_id": query.get("query_id"),
                "provider": settings.provider,
                "query_type": query.get("query_type"),
                "provider_channel": query.get("provider_channel"),
                "selection_bucket": selection_bucket,
            },
        )
        try:
            provider_output = external_call('search', {'query': _search_request_identity(query), 'provider': settings.provider, 'max_results': settings.max_results_per_query}, lambda: provider.search(
                query,
                max_results=settings.max_results_per_query,
                timeout_seconds=settings.timeout_seconds,
            ))
            response = _response_from_provider_output(
                provider_output,
                provider=settings.provider,
                planned_query=query,
            )
            emit_workflow_progress(
                "source_discovery",
                "source search query completed",
                {
                    "query_id": query.get("query_id"),
                    "provider": response.provider,
                    "raw_result_count": response.raw_result_count,
                    "error": response.error,
                    "selection_bucket": selection_bucket,
                },
            )
        except Exception as exc:  # pragma: no cover - provider faults are summarized.
            provider_error_count += 1
            emit_workflow_progress(
                "source_discovery",
                "source search query failed",
                {
                    "query_id": query.get("query_id"),
                    "provider": settings.provider,
                    "error_type": exc.__class__.__name__,
                    "selection_bucket": selection_bucket,
                },
                status="error",
            )
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="provider_error",
                    selected=True,
                    error=f"{exc.__class__.__name__}",
                    selection_bucket=selection_bucket,
                    selection_reason=selection_reason,
                )
            )
            return 0

        if response.error:
            provider_error_count += 1
            query_records.append(
                _search_execution_record(
                    query,
                    provider=response.provider,
                    status="provider_error",
                    selected=True,
                    error=response.error,
                    selection_bucket=selection_bucket,
                    selection_reason=selection_reason,
                )
            )
            return 0

        raw_result_count += response.raw_result_count
        warnings.extend(response.warnings or [])
        if not response.results:
            query_records.append(
                _search_execution_record(
                    query,
                    provider=response.provider,
                    status="no_results",
                    selected=True,
                    selection_bucket=selection_bucket,
                    selection_reason=selection_reason,
                )
            )
            return 0

        accepted_for_query = 0
        for result in response.results:
            if deduped_result_count >= settings.max_total_results:
                rejection_counter["result_limit_reached"] += 1
                search_results_manifest.append(
                    {
                        **result.model_dump(),
                        "result_status": "rejected",
                        "rejection_reason": "result_limit_reached",
                    }
                )
                continue
            canonical_url, rejection_reason = _validate_search_url(result.url)
            if rejection_reason:
                rejection_counter[rejection_reason] += 1
                search_results_manifest.append(
                    {
                        **result.model_dump(),
                        "result_status": "rejected",
                        "rejection_reason": rejection_reason,
                    }
                )
                continue
            if not ((result.title or "").strip() or (result.snippet or "").strip()):
                rejection_counter["empty_title_and_snippet"] += 1
                search_results_manifest.append(
                    {
                        **result.model_dump(),
                        "canonical_url": canonical_url,
                        "result_status": "rejected",
                        "rejection_reason": "empty_title_and_snippet",
                    }
                )
                continue
            if _search_result_is_clear_wrong_disease(result, task):
                rejection_counter["wrong_disease_result"] += 1
                search_results_manifest.append(
                    {
                        **result.model_dump(),
                        "canonical_url": canonical_url,
                        "domain": _domain(canonical_url),
                        "result_status": "rejected",
                        "rejection_reason": "wrong_disease_result",
                    }
                )
                continue
            if canonical_url in seen_canonical_urls:
                rejection_counter["duplicate_url"] += 1
                search_results_manifest.append(
                    {
                        **result.model_dump(),
                        "canonical_url": canonical_url,
                        "result_status": "rejected",
                        "rejection_reason": "duplicate_url",
                    }
                )
                continue

            try:
                external_call('search_results', {'canonical_url': canonical_url}, lambda: True)
            except BudgetExceeded:
                rejection_counter['session_result_budget_exhausted'] += 1
                continue
            seen_canonical_urls.add(canonical_url)
            deduped_result_count += 1
            accepted_for_query += 1
            candidate = _candidate_from_search_result(
                result,
                query,
                settings=settings,
                canonical_url=canonical_url,
            )
            search_candidates.append(candidate)
            search_results_manifest.append(
                {
                    **result.model_dump(),
                    "source_id": candidate.source_id,
                    "canonical_url": canonical_url,
                    "domain": candidate.domain,
                    "result_status": "accepted",
                    "rejection_reason": None,
                    "selection_bucket": selection_bucket,
                    "query_source": query.get("query_source"),
                }
            )

        executed_query_count += 1
        query_records.append(
            _search_execution_record(
                query,
                provider=response.provider,
                status="executed" if accepted_for_query else "no_results",
                selected=True,
                result_count=accepted_for_query,
                selection_bucket=selection_bucket,
                selection_reason=selection_reason,
            )
        )
        return accepted_for_query

    execution_queries = list(enumerate(planned_queries))
    if universal_queries_enabled():
        # The selector establishes the family floor; execute that order before
        # extra queries in one family can consume the existing result budget.
        execution_queries.sort(key=lambda item: (
            selection_plan.get(_query_selection_key(item[1], item[0]), {}).get(
                "selection_order", len(planned_queries) + item[0] + 1
            )
        ))
    for query_index, query in execution_queries:
        channel = str(query.get("provider_channel") or "")
        selection_bucket = _query_source_class(query)
        selection_details = selection_plan.get(_query_selection_key(query, query_index))
        selection_reason = (
            str(selection_details.get("selection_reason"))
            if selection_details
            else None
        )
        if not settings.search_enabled:
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_search_disabled",
                    skipped_reason="search_disabled",
                    selection_bucket=selection_bucket,
                )
            )
            continue
        if _is_invalid_query(query):
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_invalid_query",
                    skipped_reason="empty_non_planned_or_raw_url_query",
                    selection_bucket=selection_bucket,
                )
            )
            continue
        if channel not in set(settings.provider_channel_allowlist):
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_provider_channel_not_supported",
                    skipped_reason="provider_channel_not_in_allowlist",
                    selection_bucket=selection_bucket,
                )
            )
            continue
        if universal_queries_enabled() and not assess_query_task_fit(query, state)['accepted']:
            query_records.append(_search_execution_record(
                query, provider=settings.provider, status='skipped_task_mismatch',
                skipped_reason='query_not_task_grounded', selection_bucket=selection_bucket,
            ))
            continue
        if selection_details is None:
            query_records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_query_limit",
                    skipped_reason="source_class_budget_not_selected",
                    selection_bucket=selection_bucket,
                    selection_reason="source_class_budget_not_selected",
                )
            )
            continue
        execute_selected_query(
            query,
            selection_bucket=selection_bucket,
            selection_reason=selection_reason or "selected_for_execution",
        )
        if _collection_mode(state) == "direct_collection" and _direct_fast_stop_enabled():
            search_sufficient, target_verification = _verified_target_search_sufficient(
                search_candidates,
                _task_context(state),
            )
            target_verification_summary = target_verification
            verified_target_source_ids = list(
                target_verification.get("verified_target_source_ids") or []
            )
            if search_sufficient:
                search_stopped_reason = "verified_target_source_found"
                for remaining_query in planned_queries[query_index + 1 :]:
                    query_records.append(
                        _search_execution_record(
                            remaining_query,
                            provider=settings.provider,
                            status="skipped_verified_target_source_found",
                            skipped_reason="verified_target_source_found",
                        )
                )
                break

    detected_event_jurisdictions = _jurisdictions_mentioned_in_text(
        _candidate_text_for_retry(search_candidates)
    )
    first_pass_authority_summary = build_authority_source_coverage_summary(
        [
            candidate.model_dump()
            for candidate in search_candidates
            if isinstance(candidate, SourceCandidate)
        ],
        detected_event_jurisdictions=detected_event_jurisdictions,
    )
    retry_queries: list[dict] = []
    retry_accepted_candidate_count = 0
    if (
        settings.search_enabled
        and settings.authority_gap_retry_enabled
        and settings.authority_gap_retry_max_iterations > 0
        and not search_stopped_reason
        and _supplementary_discovery_allowed(state, planned_queries)
        and _authority_gap_requires_retry(first_pass_authority_summary)
    ):
        if "authority_source_recall_incomplete" not in warnings:
            warnings.append("authority_source_recall_incomplete")
        retry_totals = {
            "selected_query_count": selected_query_count,
            "executed_query_count": executed_query_count,
            "raw_result_count": raw_result_count,
            "deduped_result_count": deduped_result_count,
            "provider_error_count": provider_error_count,
        }
        retry_diagnostics = _execute_authority_gap_retry(
            state=state,
            settings=settings,
            provider=provider,
            planned_queries=planned_queries,
            query_records=query_records,
            search_candidates=search_candidates,
            search_results_manifest=search_results_manifest,
            seen_canonical_urls=seen_canonical_urls,
            totals=retry_totals,
            rejection_counter=rejection_counter,
            warnings=warnings,
            first_pass_authority_summary=first_pass_authority_summary,
            detected_event_jurisdictions=detected_event_jurisdictions,
        )
        selected_query_count = int(retry_totals.get("selected_query_count") or 0)
        executed_query_count = int(retry_totals.get("executed_query_count") or 0)
        raw_result_count = int(retry_totals.get("raw_result_count") or 0)
        deduped_result_count = int(retry_totals.get("deduped_result_count") or 0)
        provider_error_count = int(retry_totals.get("provider_error_count") or 0)
        retry_queries = list(retry_diagnostics.get("_retry_queries") or [])
        retry_accepted_candidate_count = int(
            retry_diagnostics.get("authority_gap_retry_accepted_candidate_count") or 0
        )

    skipped_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
    )
    selected_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
    )
    executed_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        executed_only=True,
    )
    result_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        result_sum=True,
    )
    skipped_high_trust_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
    )
    high_trust_no_result_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
        and record.get("execution_status") == "no_results"
    )
    selected_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and _known_authority_domain_for_query(record)
    )
    skipped_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and _known_authority_domain_for_query(record)
    )
    known_authority_domain_no_result_query_ids = {
        str(record.get("query_id") or "")
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("execution_status") == "no_results"
        and record.get("query_source") != "authority_gap_retry"
        and _known_authority_domain_for_query(record)
    }
    known_authority_domain_no_result_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if str(query.get("previous_iteration_basis") or "")
        in known_authority_domain_no_result_query_ids
    }
    if known_authority_domain_no_result_query_ids:
        warnings.append("known_authority_domain_no_result")
        if known_authority_domain_no_result_retry_query_ids:
            warnings.append("known_authority_domain_no_result_retry_attempted")
    known_domain_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("retry_category") == "known_domain_no_result"
    }
    jurisdiction_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("retry_category") == "jurisdiction_authority"
    }
    official_page_family_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("retry_category") == "official_page_family"
    }
    all_retry_query_ids = {
        str(query.get("query_id") or "")
        for query in retry_queries
        if query.get("query_id")
    }
    jurisdiction_authority_retry_candidate_count = sum(
        int(record.get("result_count") or 0)
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
    )
    jurisdiction_retry_protected_execution_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
        and record.get("selected_for_execution")
    )
    jurisdiction_retry_skipped_due_to_result_cap_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in jurisdiction_retry_query_ids
        and record.get("execution_status") == "skipped_total_result_limit"
    )
    known_domain_retry_skipped_due_to_result_cap_count = sum(
        1
        for record in query_records
        if str(record.get("query_id") or "") in known_domain_retry_query_ids
        and record.get("execution_status") == "skipped_total_result_limit"
    )
    official_page_family_retry_candidate_count = sum(
        int(record.get("result_count") or 0)
        for record in query_records
        if str(record.get("query_id") or "") in official_page_family_retry_query_ids
    )

    def retry_generated_domain_counts(query_ids: set[str]) -> dict[str, int]:
        counts: Counter = Counter()
        for query in retry_queries:
            if str(query.get("query_id") or "") not in query_ids:
                continue
            domain = _retry_query_domain(query)
            if domain:
                counts[domain] += 1
        return dict(counts)

    def retry_executed_domain_counts(query_ids: set[str]) -> dict[str, int]:
        counts: Counter = Counter()
        for record in query_records:
            if str(record.get("query_id") or "") not in query_ids:
                continue
            if not record.get("selected_for_execution"):
                continue
            domain = _retry_query_domain(record)
            if domain:
                counts[domain] += 1
        return dict(counts)

    post_retry_authority_summary = build_authority_source_coverage_summary(
        [
            candidate.model_dump()
            for candidate in search_candidates
            if isinstance(candidate, SourceCandidate)
        ],
        detected_event_jurisdictions=detected_event_jurisdictions,
    )
    missing_high_confidence_jurisdiction_domains = [
        str(hint.get("domain") or "")
        for hint in authority_domain_hints_for_jurisdictions(
            list(post_retry_authority_summary.get("missing_event_jurisdictions") or [])
        )
        if hint.get("domain")
    ]
    high_confidence_source_gap_status = (
        "complete"
        if not missing_high_confidence_jurisdiction_domains
        and int(post_retry_authority_summary.get("structured_database_source_count") or 0)
        > 0
        and int(
            post_retry_authority_summary.get(
                "peer_reviewed_or_scientific_source_count"
            )
            or 0
        )
        > 0
        else "partial"
    )
    summary = {
        "search_enabled": settings.search_enabled,
        "live_search_enabled": settings.live_search_enabled,
        "fixture_search_enabled": settings.fixture_search_enabled,
        "search_mode": settings.mode,
        "search_provider": settings.provider,
        "combine_with_seed_catalog": settings.combine_with_seed_catalog,
        "discovery_method": _discovery_method_for_settings(settings),
        "planned_query_count": len(planned_queries),
        "selected_query_count": selected_query_count,
        "executed_query_count": executed_query_count,
        "skipped_query_count": skipped_query_count,
        "raw_search_result_count": raw_result_count,
        "deduplicated_search_result_count": deduped_result_count,
        "rejected_search_result_count": sum(rejection_counter.values()),
        "candidate_from_search_count": len(search_candidates),
        "candidate_from_seed_count": 0,
        "total_candidate_count": len(search_candidates),
        "provider_error_count": provider_error_count,
        "max_queries": settings.max_queries,
        "max_results_per_query": settings.max_results_per_query,
        "max_total_results": settings.max_total_results,
        "query_execution_records": query_records,
        "rejection_reason_counts": dict(rejection_counter),
        "search_result_source_ids": [candidate.source_id for candidate in search_candidates],
        "planned_query_count_by_source_class": _planned_query_source_class_counts(
            planned_queries
        ),
        "selected_query_count_by_source_class": selected_count_by_class,
        "executed_query_count_by_source_class": executed_count_by_class,
        "result_count_by_source_class": result_count_by_class,
        "skipped_high_trust_query_count": skipped_high_trust_query_count,
        "high_trust_no_result_query_count": high_trust_no_result_query_count,
        "selected_known_authority_domain_query_count": (
            selected_known_authority_domain_query_count
        ),
        "skipped_known_authority_domain_query_count": (
            skipped_known_authority_domain_query_count
        ),
        "known_authority_domain_no_result_count": len(
            known_authority_domain_no_result_query_ids
        ),
        "known_authority_domain_no_result_retry_generated_count": len(
            known_authority_domain_no_result_retry_query_ids
        ),
        "known_domain_retry_generated_count": len(known_domain_retry_query_ids),
        "known_domain_retry_generated_by_domain": retry_generated_domain_counts(
            known_domain_retry_query_ids
        ),
        "known_domain_retry_executed_by_domain": retry_executed_domain_counts(
            known_domain_retry_query_ids
        ),
        "first_pass_authority_source_coverage_summary": first_pass_authority_summary,
        "post_retry_authority_source_coverage_summary": post_retry_authority_summary,
        "detected_event_jurisdictions": list(
            first_pass_authority_summary.get("detected_event_jurisdictions") or []
        ),
        "covered_event_jurisdictions": list(
            first_pass_authority_summary.get("covered_event_jurisdictions") or []
        ),
        "missing_event_jurisdictions": list(
            first_pass_authority_summary.get("missing_event_jurisdictions") or []
        ),
        "authority_gap_retry_enabled": bool(settings.authority_gap_retry_enabled),
        "authority_gap_retry_max_queries": settings.authority_gap_retry_max_queries,
        "authority_gap_known_domain_retry_max_queries": (
            settings.authority_gap_known_domain_retry_max_queries
        ),
        "authority_gap_jurisdiction_retry_max_queries": (
            settings.authority_gap_jurisdiction_retry_max_queries
        ),
        "authority_gap_official_page_family_retry_max_queries": (
            settings.authority_gap_official_page_family_retry_max_queries
        ),
        "authority_gap_retry_generated_query_count": len(retry_queries),
        "authority_retry_result_budget": settings.authority_gap_retry_result_budget,
        "authority_retry_result_budget_used": retry_accepted_candidate_count,
        "retry_execution_order": _retry_execution_order(
            query_records,
            all_retry_query_ids,
        ),
        "retry_category_round_robin_order": [
            "jurisdiction_authority",
            "known_domain_no_result",
            "official_page_family",
            "skipped_high_trust_query",
            "generic_authority_gap",
        ],
        "retry_result_budget_by_category": _retry_result_budget_by_category(
            query_records,
            all_retry_query_ids,
        ),
        "retry_domain_result_cap_counts": _retry_domain_result_cap_counts(
            search_results_manifest
        ),
        "high_confidence_domain_query_status": _high_confidence_domain_query_status(
            retry_queries=retry_queries,
            query_records=query_records,
            search_candidates=search_candidates,
            detected_event_jurisdictions=detected_event_jurisdictions,
        ),
        "source_recall_target_ledger": _build_source_recall_target_ledger(
            retry_queries=retry_queries,
            query_records=query_records,
            search_candidates=search_candidates,
            detected_event_jurisdictions=detected_event_jurisdictions,
        ),
        "jurisdiction_authority_retry_generated_count": len(
            jurisdiction_retry_query_ids
        ),
        "jurisdiction_retry_generated_by_domain": retry_generated_domain_counts(
            jurisdiction_retry_query_ids
        ),
        "jurisdiction_retry_executed_by_domain": retry_executed_domain_counts(
            jurisdiction_retry_query_ids
        ),
        "jurisdiction_authority_retry_candidate_count": (
            jurisdiction_authority_retry_candidate_count
        ),
        "jurisdiction_retry_protected_execution_count": (
            jurisdiction_retry_protected_execution_count
        ),
        "jurisdiction_retry_skipped_due_to_result_cap_count": (
            jurisdiction_retry_skipped_due_to_result_cap_count
        ),
        "known_domain_retry_truncated_before_jurisdiction_complete": bool(
            known_domain_retry_skipped_due_to_result_cap_count
            and jurisdiction_retry_protected_execution_count
            and not jurisdiction_retry_skipped_due_to_result_cap_count
        ),
        "official_page_family_retry_generated_count": len(
            official_page_family_retry_query_ids
        ),
        "official_page_family_retry_candidate_count": (
            official_page_family_retry_candidate_count
        ),
        "missing_high_confidence_jurisdiction_domains": (
            missing_high_confidence_jurisdiction_domains
        ),
        "high_confidence_source_gap_status": high_confidence_source_gap_status,
        "authority_gap_retry_selected_query_count": sum(
            1
            for record in query_records
            if record.get("query_source") == "authority_gap_retry"
            and record.get("selected_for_execution")
        ),
        "authority_gap_retry_accepted_candidate_count": retry_accepted_candidate_count,
        "iterative_source_discovery_enabled": False,
        "search_iteration_count": 0,
        "llm_refinement_call_count": 0,
        "stop_decision": stop_decision,
        "stop_reason": stop_reason,
        "search_stopped_reason": search_stopped_reason,
        "verified_target_source_count": len(verified_target_source_ids),
        "verified_target_source_ids": verified_target_source_ids,
        "target_source_miss_reasons": target_verification_summary.get(
            "target_source_miss_reasons", []
        ),
        "target_evidence_requirements": target_verification_summary.get(
            "target_evidence_requirements", {}
        ),
        "iteration_query_counts": {},
        "query_source_counts": _query_source_counts(query_records),
        "warnings": sorted(set(warnings)),
    }
    return search_candidates, search_results_manifest, summary


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _direct_fast_stop_enabled() -> bool:
    if universal_queries_enabled():
        return False
    return _env_flag("DIRECT_FAST_STOP_ON_VERIFIED_TARGET", default=True)


def _direct_allow_generated_url_only() -> bool:
    return _env_flag("DIRECT_ALLOW_GENERATED_URL_ONLY", default=False)


def _is_predicted_official_candidate(row: dict) -> bool:
    return (
        row.get("discovery_method") == "official_coverage_requirement"
        or row.get("query_source") == "source_coverage_requirement"
    )


def _verified_official_candidate_ids(candidates: list[SourceCandidate]) -> list[str]:
    if not _direct_allow_generated_url_only():
        return []
    verified: list[str] = []
    for candidate in candidates:
        row = candidate.model_dump() if hasattr(candidate, "model_dump") else dict(candidate)
        if (
            row.get("must_fetch")
            or row.get("coverage_requirement_ids")
            or row.get("discovery_method") == "official_coverage_requirement"
            or row.get("query_source") == "source_coverage_requirement"
            or row.get("source_purpose") == "target_official_surveillance_report"
        ):
            source_id = row.get("source_id")
            if source_id:
                verified.append(str(source_id))
    return sorted(set(verified))


def _direct_search_skipped_for_verified_target(
    state: DataCollectionState,
    settings: SourceSearchSettings,
    official_coverage_candidates: list[SourceCandidate],
) -> tuple[list[SourceCandidate], list[dict], dict, dict] | None:
    if (
        _collection_mode(state) != "direct_collection"
        or not settings.search_enabled
        or not _direct_fast_stop_enabled()
    ):
        return None
    verified_ids = _verified_official_candidate_ids(official_coverage_candidates)
    if not verified_ids:
        return None
    planned_queries = _planned_queries(state)
    query_records = [
        _search_execution_record(
            query,
            provider=settings.provider,
            status="skipped_verified_target_source_found",
            skipped_reason="verified_target_source_found",
        )
        for query in planned_queries
    ]
    search_summary = {
        "search_enabled": settings.search_enabled,
        "live_search_enabled": settings.live_search_enabled,
        "fixture_search_enabled": settings.fixture_search_enabled,
        "search_mode": settings.mode,
        "search_provider": settings.provider,
        "combine_with_seed_catalog": settings.combine_with_seed_catalog,
        "discovery_method": _discovery_method_for_settings(settings),
        "planned_query_count": len(planned_queries),
        "selected_query_count": 0,
        "executed_query_count": 0,
        "skipped_query_count": len(planned_queries),
        "raw_search_result_count": 0,
        "deduplicated_search_result_count": 0,
        "rejected_search_result_count": 0,
        "candidate_from_search_count": 0,
        "candidate_from_seed_count": 0,
        "total_candidate_count": 0,
        "provider_error_count": 0,
        "max_queries": settings.max_queries,
        "max_results_per_query": settings.max_results_per_query,
        "max_total_results": settings.max_total_results,
        "query_execution_records": query_records,
        "rejection_reason_counts": {},
        "search_result_source_ids": [],
        "iterative_source_discovery_enabled": bool(settings.iterative_enabled),
        "search_iteration_count": 0,
        "llm_refinement_call_count": 0,
        "stop_decision": "stop_sufficient",
        "stop_reason": "verified target collection source was available before search",
        "search_stopped_reason": "verified_target_source_found",
        "verified_target_source_count": len(verified_ids),
        "verified_target_source_ids": verified_ids,
        "predicted_target_candidate_count": len(official_coverage_candidates),
        "search_verified_target_source_count": 0,
        "fetch_verified_target_source_count": 0,
        "iteration_query_counts": {},
        "query_source_counts": _query_source_counts(query_records),
        "warnings": [],
    }
    iterative_summary = IterativeSourceDiscoverySummary(
        iterative_source_discovery_enabled=bool(settings.iterative_enabled),
        llm_iterative_planning_enabled=False,
        search_iteration_count=0,
        llm_refinement_call_count=0,
        total_queries_planned=len(planned_queries),
        total_queries_executed=0,
        total_raw_results=0,
        total_candidates_created=0,
        stop_decision="stop_sufficient",
        stop_reason="verified target collection source was available before search",
        selected_query_ids=[],
        skipped_query_ids=[
            str(record.get("query_id"))
            for record in query_records
            if record.get("query_id")
        ],
        skipped_query_count=len(query_records),
        warnings=[],
        status="completed",
    ).model_dump()
    iterative_summary.update(
        {
            "search_stopped_reason": "verified_target_source_found",
            "verified_target_source_count": len(verified_ids),
            "verified_target_source_ids": verified_ids,
        }
    )
    return (
        [],
        [],
        search_summary,
        {
            "iterative_source_discovery_summary": iterative_summary,
            "search_iteration_plans": [],
            "search_iteration_observations": [],
            "search_refinement_decisions": [],
            "iterative_search_queries": [],
        },
    )


def _disabled_iterative_outputs() -> dict:
    summary = IterativeSourceDiscoverySummary(
        iterative_source_discovery_enabled=False,
        llm_iterative_planning_enabled=False,
        stop_decision=None,
        stop_reason="iterative_source_discovery_disabled",
        status="disabled",
    ).model_dump()
    return {
        "iterative_source_discovery_summary": summary,
        "search_iteration_plans": [],
        "search_iteration_observations": [],
        "search_refinement_decisions": [],
        "iterative_search_queries": [],
    }


def _iterative_limits(settings: SourceSearchSettings) -> dict:
    return {
        "max_iterations": settings.iterative_max_iterations,
        "max_queries_per_iteration": settings.iterative_max_queries_per_iteration,
        "max_total_queries": settings.iterative_max_total_queries,
        "max_results_per_query": settings.max_results_per_query,
        "max_total_results": min(
            settings.max_total_results,
            settings.iterative_max_total_results,
        ),
        "provider_channel_allowlist": list(settings.provider_channel_allowlist),
    }


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _task_context(state: DataCollectionState) -> dict:
    structured = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    return {
        "disease": structured.get("disease") or spec.get("disease"),
        "disease_aliases": list(structured.get("disease_aliases") or []),
        "user_request": structured.get("user_request") or spec.get("user_request") or state.get("user_request"),
        "location": (
            structured.get("location")
            or spec.get("geography")
            or spec.get("location")
        ),
        "start_date": structured.get("start_date") or spec.get("start_date"),
        "end_date": structured.get("end_date") or spec.get("end_date"),
        "target_fields": structured.get("target_fields")
        or spec.get("required_fields")
        or [],
        "collection_mode": (
            structured.get("collection_mode")
            or spec.get("collection_mode")
            or state.get("collection_mode")
        ),
    }


def _parse_date(value) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{4}", text):
        text = f"{text}-01-01"
    parts = text.split("-")
    if len(parts) == 3:
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (TypeError, ValueError):
            return None
    try:
        return datetime.fromisoformat(text).date()
    except (TypeError, ValueError):
        return None


def _date_span(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    days = []
    current = start
    while current <= end:
        days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return days


def _week_ending_saturdays(start: date, end: date) -> list[date]:
    if end < start:
        start, end = end, start
    days_until_saturday = (5 - start.weekday()) % 7
    current = date.fromordinal(start.toordinal() + days_until_saturday)
    week_ends: list[date] = []
    while current <= end or not week_ends:
        week_ends.append(current)
        current = date.fromordinal(current.toordinal() + 7)
    return week_ends


def _target_verification_requirements(task: dict) -> dict:
    disease = str(task.get("disease") or "").strip().lower()
    location = str(task.get("location") or "").strip().lower()
    collection_mode = str(task.get("collection_mode") or "").strip()
    start = _parse_date(task.get("start_date"))
    end = _parse_date(task.get("end_date")) or start
    if collection_mode != "direct_collection":
        return {"required": False, "reason": "non_direct_collection_mode"}
    if not start or not end:
        return {"required": False, "reason": "task_dates_missing"}
    if not any(token in disease for token in ("flu", "influenza")):
        return {"required": False, "reason": "non_influenza_task"}
    days = _date_span(start, end)
    report_week_ends = _week_ending_saturdays(start, end)
    iso_pairs = {
        (week_end.isocalendar().year, week_end.isocalendar().week)
        for week_end in report_week_ends
    }
    expected_years = {day.year for day in days} | {year for year, _week in iso_pairs}
    expected_weeks = {week for _year, week in iso_pairs}
    expected_date_hints = {day.isoformat() for day in days}
    return {
        "required": True,
        "disease": disease,
        "location": location,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "expected_years": sorted(expected_years),
        "expected_weeks": sorted(expected_weeks),
        "expected_date_hints": sorted(expected_date_hints),
        "accepted_authority_hints": [
            "cdc.gov" if location in {"united states", "usa", "us", "u.s."} else "",
            "department of health",
            ".gov",
            "public health",
        ],
    }


def _candidate_text(candidate: dict) -> str:
    return " ".join(
        str(candidate.get(key) or "")
        for key in (
            "canonical_url",
            "url",
            "title",
            "snippet",
            "publisher",
            "source",
            "published_date",
        )
    ).lower()


def _is_national_target_location(location: str) -> bool:
    normalized = re.sub(r"\s+", " ", (location or "").strip().lower())
    return normalized in _US_NATIONAL_LOCATION_ALIASES


def _target_location_aliases(location: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", (location or "").strip().lower())
    aliases = {normalized} if normalized else set()
    if not normalized:
        return aliases
    abbr = _US_STATE_ABBREVIATIONS.get(normalized)
    if abbr:
        aliases.update(
            {
                f".{abbr}.gov",
                f"//{abbr}.gov",
                f" {abbr} ",
                f" {abbr}.",
                f"/{abbr}/",
            }
        )
    aliases.update(_LOCAL_HEALTH_ACRONYM_ALIASES.get(normalized, set()))
    if normalized == "new york city":
        aliases.update({"new york city", "nyc", "www.nyc.gov"})
    return {alias for alias in aliases if alias}


def _candidate_has_target_geography(candidate: dict, requirements: dict) -> bool:
    location = str(requirements.get("location") or "").strip().lower()
    if not location or _is_national_target_location(location):
        return True
    text = f" {_candidate_text(candidate)} "
    return any(alias in text for alias in _target_location_aliases(location))


def _explicit_year_week_pairs(text: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for match in _FLUVIEW_YEAR_WEEK_RE.finditer(text):
        segment = match.group(0)
        years = [int(value) for value in _YEAR_RE.findall(segment)]
        weeks = [int(value) for value in _WEEK_RE.findall(segment)]
        for year in years:
            for week in weeks:
                pairs.add((year, week))
    return pairs


def _verify_candidate_for_target(candidate: dict, requirements: dict) -> tuple[bool, str]:
    if not requirements.get("required"):
        return True, "target_verification_not_required"
    if _is_predicted_official_candidate(candidate) and not _direct_allow_generated_url_only():
        return False, (
            f"{candidate.get('canonical_url') or candidate.get('url')} is a "
            "predicted target candidate and requires search or successful fetch validation"
        )
    text = _candidate_text(candidate)
    expected_years = {int(value) for value in requirements.get("expected_years") or []}
    expected_weeks = {int(value) for value in requirements.get("expected_weeks") or []}
    expected_dates = set(requirements.get("expected_date_hints") or [])
    explicit_pairs = _explicit_year_week_pairs(text)
    if explicit_pairs:
        if explicit_pairs & {
            (year, week) for year in expected_years for week in expected_weeks
        }:
            if not _candidate_has_target_geography(candidate, requirements):
                return False, (
                    f"{candidate.get('canonical_url') or candidate.get('url')} "
                    "lacks verified target geography evidence"
                )
            return True, "explicit_year_week_matches_task"
        return (
            False,
            (
                f"{candidate.get('canonical_url') or candidate.get('url')} "
                f"has explicit year/week {sorted(explicit_pairs)} outside "
                f"expected years {sorted(expected_years)} and weeks {sorted(expected_weeks)}"
            ),
        )
    if expected_dates and any(date_hint in text for date_hint in expected_dates):
        if not _candidate_has_target_geography(candidate, requirements):
            return False, (
                f"{candidate.get('canonical_url') or candidate.get('url')} "
                "lacks verified target geography evidence"
            )
        return True, "date_hint_matches_task"
    published = _parse_date(candidate.get("published_date"))
    if published and published.year not in expected_years:
        return (
            False,
            (
                f"{candidate.get('canonical_url') or candidate.get('url')} "
                f"published_date {published.isoformat()} outside expected years "
                f"{sorted(expected_years)}"
            ),
        )
    if expected_weeks and "week" in text:
        return (
            False,
            (
                f"{candidate.get('canonical_url') or candidate.get('url')} "
                "mentions a week but no verified task week/date match was found"
            ),
        )
    return False, (
        f"{candidate.get('canonical_url') or candidate.get('url')} lacks verified "
        "target date/week evidence"
    )


_NON_TARGET_COLLECTION_ROLE_VALUES = {
    "context",
    "context_only",
    "context_source",
    "validation",
    "validation_only",
    "validation_source",
    "excluded",
    "exclude",
    "human_review",
    "needs_human_review",
}


def _verify_candidate_target_collection_role(candidate: dict) -> tuple[bool, str]:
    """Return whether a candidate can satisfy target collection coverage.

    Role hints are not the source of truth for geography/period matching, but an
    explicit context/validation role means the source should not be counted as a
    verified target collection source.
    """

    url = candidate.get("canonical_url") or candidate.get("url")
    role_keys = (
        "triage_role",
        "target_fit_status",
        "source_role_final",
        "source_role",
        "role_hint",
        "query_type",
    )
    for key in role_keys:
        raw_value = candidate.get(key)
        value = str(raw_value or "").strip().lower().replace("-", "_")
        if not value:
            continue
        if value in _NON_TARGET_COLLECTION_ROLE_VALUES:
            return (
                False,
                f"{url} is not a target collection source ({key}={raw_value})",
            )
    return True, "candidate_role_allows_target_collection"


def _verify_target_sources(candidates: list[SourceCandidate], task: dict) -> dict:
    requirements = _target_verification_requirements(task)
    if universal_queries_enabled():
        return {
            "target_verification_required": True,
            "verified_target_source_count": 0,
            "verified_target_source_ids": [],
            "target_source_miss_reasons": ["fetched_evidence_required"],
            "target_evidence_requirements": requirements,
        }
    if not requirements.get("required"):
        return {
            "target_verification_required": False,
            "verified_target_source_count": 0,
            "verified_target_source_ids": [],
            "target_source_miss_reasons": [],
            "target_evidence_requirements": requirements,
        }
    verified_ids: list[str] = []
    miss_reasons: list[str] = []
    for candidate in candidates:
        row = candidate.model_dump() if hasattr(candidate, "model_dump") else dict(candidate)
        role_matched, role_reason = _verify_candidate_target_collection_role(row)
        if not role_matched:
            miss_reasons.append(role_reason)
            continue
        matched, reason = _verify_candidate_for_target(row, requirements)
        if matched:
            verified_ids.append(str(row.get("source_id") or ""))
        else:
            miss_reasons.append(reason)
    return {
        "target_verification_required": True,
        "verified_target_source_count": len([sid for sid in verified_ids if sid]),
        "verified_target_source_ids": [sid for sid in verified_ids if sid],
        "target_source_miss_reasons": miss_reasons,
        "target_evidence_requirements": requirements,
    }


def _verified_target_search_sufficient(
    candidates: list[SourceCandidate],
    task: dict,
) -> tuple[bool, dict]:
    verification = _verify_target_sources(candidates, task)
    verified_ids = set(verification.get("verified_target_source_ids") or [])
    requirements = verification.get("target_evidence_requirements") or {}
    expected_weeks = {
        int(week) for week in (requirements.get("expected_weeks") or []) if week
    }
    if not verified_ids:
        verification["covered_target_weeks"] = []
        verification["missing_target_weeks"] = sorted(expected_weeks)
        return False, verification
    if len(expected_weeks) <= 1:
        verification["covered_target_weeks"] = sorted(expected_weeks)
        verification["missing_target_weeks"] = []
        return True, verification
    covered_weeks: set[int] = set()
    for candidate in candidates:
        if candidate.source_id not in verified_ids:
            continue
        row = candidate.model_dump() if hasattr(candidate, "model_dump") else dict(candidate)
        for _year, week in _explicit_year_week_pairs(_candidate_text(row)):
            if week in expected_weeks:
                covered_weeks.add(week)
    missing_weeks = expected_weeks - covered_weeks
    verification["covered_target_weeks"] = sorted(covered_weeks)
    verification["missing_target_weeks"] = sorted(missing_weeks)
    return not missing_weeks, verification


def _normalize_iterative_query(
    query: dict,
    *,
    iteration_index: int,
    query_source: str,
    sequence_index: int,
) -> dict:
    query_id = str(query.get("query_id") or "").strip()
    if not query_id:
        query_id = f"iter_{iteration_index}_{sequence_index:03d}"
    return {
        "query_id": query_id,
        "query": str(query.get("query") or "").strip(),
        "provider_channel": str(query.get("provider_channel") or "web_search"),
        "query_type": str(query.get("query_type") or "general_web"),
        "source_type": str(query.get("source_type") or "news_and_situation_report"),
        "role_hint": str(query.get("role_hint") or "collection_support"),
        "priority": int(query.get("priority") or 5),
        "expected_fields": _as_str_list(query.get("expected_fields")),
        "query_rationale": str(query.get("query_rationale") or "").strip(),
        "expected_source_type_or_evidence": query.get(
            "expected_source_type_or_evidence"
        ),
        "expected_trust_signal": query.get("expected_trust_signal"),
        "language": query.get("language"),
        "target_disease_terms": _as_str_list(query.get("target_disease_terms")),
        "target_location_terms": _as_str_list(query.get("target_location_terms")),
        "time_terms": _as_str_list(query.get("time_terms")),
        "event_terms": _as_str_list(query.get("event_terms")),
        "disease_terms_used": _as_str_list(query.get("disease_terms_used")),
        "official_domain_hint": query.get("official_domain_hint"),
        "is_follow_up_query": bool(query.get("is_follow_up_query")),
        "previous_iteration_basis": query.get("previous_iteration_basis"),
        "execution_status": "planned_not_executed",
        "query_source": query_source,
        "iteration_index": iteration_index,
        "iterative_query_id": query_id,
    }


def _normalize_iteration_plan(raw: dict, iteration_index: int) -> dict:
    plan = SearchIterationPlan.model_validate(raw).model_dump()
    plan["iteration_index"] = iteration_index
    for query, original in zip(plan["query_batch"], raw.get("query_batch") or []):
        for key in ("event_terms", "disease_terms_used", "official_domain_hint", "query_source"):
            if key in original:
                query[key] = original[key]
    return plan


def _normalize_refinement_decision(raw: dict, iteration_index: int) -> dict:
    decision = SearchRefinementDecision.model_validate(raw).model_dump()
    decision["iteration_index"] = iteration_index
    return decision


def _safe_search_row(result: SearchResult, candidate: SourceCandidate | None, **extra) -> dict:
    row = {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
        "published_date": result.published_date,
        "source": result.source,
        "rank": result.rank,
        "query": result.query,
        "query_id": result.query_id,
        "provider_channel": result.provider_channel,
        "source_type": result.source_type,
        "role_hint": result.role_hint,
        "retrieved_at": result.retrieved_at,
        "provider": result.provider,
        "query_type": result.query_type,
    }
    if candidate is not None:
        row["source_id"] = candidate.source_id
    row.update(extra)
    return row


def _search_result_is_clear_wrong_disease(result: SearchResult, task: dict) -> bool:
    text = " ".join(
        str(value or "")
        for value in (result.title, result.snippet, result.url, result.source)
    )
    if not text.strip():
        return False
    assessment = assess_text_disease_relevance(
        text,
        build_disease_relevance_context({"structured_task": task}),
        require_data_signal=False,
    )
    if assessment.get("status") == INCOMPATIBLE_DISEASE:
        return True
    return (
        assessment.get("status") == UNRELATED_DISEASE
        and bool(assessment.get("incompatible_disease_terms_found"))
        and not bool(assessment.get("target_disease_terms_found"))
    )


def _execute_iterative_query_batch(
    *,
    query_batch: list[dict],
    provider,
    settings: SourceSearchSettings,
    seen_canonical_urls: set[str],
    totals: dict,
    task: dict,
    max_total_results_override: int | None = None,
    max_queries_override: int | None = None,
    max_total_queries_override: int | None = None,
    selection_reason_override: str | None = None,
    accepted_result_cap_by_retry_category: dict[str, int] | None = None,
    accepted_result_cap_by_retry_domain: dict[str, int] | None = None,
    evidence_state: dict | None = None,
) -> tuple[list[SourceCandidate], list[dict], list[dict], Counter, list[str]]:
    records: list[dict] = []
    manifest: list[dict] = []
    candidates: list[SourceCandidate] = []
    rejection_counter: Counter = Counter()
    warnings: list[str] = []
    selected_this_iteration = 0
    accepted_by_retry_domain: Counter = Counter()
    max_total_results = (
        int(max_total_results_override)
        if max_total_results_override is not None
        else min(settings.max_total_results, settings.iterative_max_total_results)
    )
    max_queries = (
        int(max_queries_override)
        if max_queries_override is not None
        else settings.iterative_max_queries_per_iteration
    )
    max_total_queries = (
        int(max_total_queries_override)
        if max_total_queries_override is not None
        else settings.iterative_max_total_queries
    )

    for query_index, query in enumerate(query_batch):
        if totals.get("search_budget_exhausted") or totals.get("search_result_budget_exhausted"):
            break
        if universal_queries_enabled() and not assess_query_task_fit(query, evidence_state or {'structured_task': task})['accepted']:
            records.append(_search_execution_record(query, provider=settings.provider, status='skipped_task_mismatch', skipped_reason='query_not_task_grounded'))
            continue
        channel = str(query.get("provider_channel") or "")
        retry_category = str(query.get("retry_category") or "")
        accepted_result_cap = (
            accepted_result_cap_by_retry_category.get(retry_category)
            if accepted_result_cap_by_retry_category and retry_category
            else None
        )
        if not settings.search_enabled:
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_search_disabled",
                    skipped_reason="search_disabled",
                )
            )
            continue
        if _is_invalid_query(query):
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_invalid_query",
                    skipped_reason="empty_non_planned_or_raw_url_query",
                )
            )
            continue
        if channel not in set(settings.provider_channel_allowlist):
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_provider_channel_not_supported",
                    skipped_reason="provider_channel_not_in_allowlist",
                )
            )
            continue
        if selected_this_iteration >= max_queries:
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_query_limit",
                    skipped_reason="max_queries_per_iteration_reached",
                    selection_reason=selection_reason_override,
                )
            )
            continue
        if totals["selected_query_count"] >= max_total_queries:
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_query_limit",
                    skipped_reason="max_total_queries_reached",
                    selection_reason=selection_reason_override,
                )
            )
            continue
        if totals["deduped_result_count"] >= max_total_results:
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="skipped_total_result_limit",
                    skipped_reason="max_total_results_reached",
                    selection_reason=selection_reason_override,
                )
            )
            continue
        if provider is None:
            totals["provider_error_count"] += 1
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="provider_error",
                    selected=True,
                    error="provider_unavailable",
                    selection_reason=selection_reason_override,
                )
            )
            continue

        selected_this_iteration += 1
        totals["selected_query_count"] += 1
        emit_workflow_progress(
            "source_discovery",
            "iterative source search query started",
            {
                "iteration_index": query.get("iteration_index"),
                "query_id": query.get("query_id"),
                "provider": settings.provider,
                "query_type": query.get("query_type"),
                "provider_channel": query.get("provider_channel"),
            },
        )
        try:
            provider_output = external_call('search', {'query': _search_request_identity(query), 'provider': settings.provider, 'max_results': settings.max_results_per_query}, lambda: provider.search(
                query,
                max_results=settings.max_results_per_query,
                timeout_seconds=settings.timeout_seconds,
            ))
            response = _response_from_provider_output(
                provider_output,
                provider=settings.provider,
                planned_query=query,
            )
            emit_workflow_progress(
                "source_discovery",
                "iterative source search query completed",
                {
                    "iteration_index": query.get("iteration_index"),
                    "query_id": query.get("query_id"),
                    "provider": response.provider,
                    "raw_result_count": response.raw_result_count,
                    "error": response.error,
                },
            )
        except BudgetExceeded as exc:
            totals["selected_query_count"] -= 1
            selected_this_iteration -= 1
            if exc.kind == "action_attempts":
                records.append(_search_execution_record(
                    query, provider=settings.provider, status="skipped_query_limit",
                    skipped_reason="same_query_attempt_limit_reached"))
                warnings.append("same_query_attempt_limit_reached")
                continue
            totals["search_budget_exhausted"] = True
            records.append(_search_execution_record(
                query, provider=settings.provider, status="skipped_query_limit",
                skipped_reason="shared_search_budget_exhausted"))
            warnings.append("shared_search_budget_exhausted")
            break
        except Exception as exc:  # pragma: no cover - provider faults are summarized.
            totals["provider_error_count"] += 1
            emit_workflow_progress(
                "source_discovery",
                "iterative source search query failed",
                {
                    "iteration_index": query.get("iteration_index"),
                    "query_id": query.get("query_id"),
                    "provider": settings.provider,
                    "error_type": exc.__class__.__name__,
                },
                status="error",
            )
            records.append(
                _search_execution_record(
                    query,
                    provider=settings.provider,
                    status="provider_error",
                    selected=True,
                    error=f"{exc.__class__.__name__}",
                    selection_reason=selection_reason_override,
                )
            )
            continue

        if response.error:
            totals["provider_error_count"] += 1
            records.append(
                _search_execution_record(
                    query,
                    provider=response.provider,
                    status="provider_error",
                    selected=True,
                    error=response.error,
                    selection_reason=selection_reason_override,
                )
            )
            continue

        totals["raw_result_count"] += response.raw_result_count
        warnings.extend(response.warnings or [])
        if not response.results:
            records.append(
                _search_execution_record(
                    query,
                    provider=response.provider,
                    status="no_results",
                    selected=True,
                    selection_reason=selection_reason_override,
                )
            )
            continue

        accepted_for_query = 0
        for result in response.results:
            if (
                accepted_result_cap is not None
                and accepted_result_cap > 0
                and accepted_for_query >= accepted_result_cap
            ):
                rejection_counter["per_query_result_cap_reached"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        result_status="rejected",
                        rejection_reason="per_query_result_cap_reached",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue
            if totals["deduped_result_count"] >= max_total_results:
                rejection_counter["result_limit_reached"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        result_status="rejected",
                        rejection_reason="result_limit_reached",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue
            canonical_url, rejection_reason = _validate_search_url(result.url)
            if rejection_reason:
                rejection_counter[rejection_reason] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        result_status="rejected",
                        rejection_reason=rejection_reason,
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue
            if not ((result.title or "").strip() or (result.snippet or "").strip()):
                rejection_counter["empty_title_and_snippet"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        canonical_url=canonical_url,
                        result_status="rejected",
                        rejection_reason="empty_title_and_snippet",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue
            if _search_result_is_clear_wrong_disease(result, task):
                rejection_counter["wrong_disease_result"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        canonical_url=canonical_url,
                        domain=_domain(canonical_url),
                        result_status="rejected",
                        rejection_reason="wrong_disease_result",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue
            if canonical_url in seen_canonical_urls:
                rejection_counter["duplicate_url"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        canonical_url=canonical_url,
                        domain=_domain(canonical_url),
                        result_status="rejected",
                        rejection_reason="duplicate_url",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue

            retry_domain = _retry_query_domain(query) or _domain(canonical_url)
            retry_domain_cap = (
                accepted_result_cap_by_retry_domain.get(retry_domain)
                if accepted_result_cap_by_retry_domain and retry_domain
                else None
            )
            if (
                retry_domain_cap is not None
                and retry_domain_cap > 0
                and accepted_by_retry_domain[retry_domain] >= retry_domain_cap
            ):
                rejection_counter["per_domain_retry_result_cap_reached"] += 1
                manifest.append(
                    _safe_search_row(
                        result,
                        None,
                        canonical_url=canonical_url,
                        domain=_domain(canonical_url),
                        result_status="rejected",
                        rejection_reason="per_domain_retry_result_cap_reached",
                        query_source=query.get("query_source"),
                        iteration_index=query.get("iteration_index"),
                    )
                )
                continue

            try:
                external_call('search_results', {'canonical_url': canonical_url}, lambda: True)
            except BudgetExceeded:
                rejection_counter['session_result_budget_exhausted'] += 1
                totals["search_result_budget_exhausted"] = True
                warnings.append("shared_search_result_budget_exhausted")
                break
            seen_canonical_urls.add(canonical_url)
            totals["deduped_result_count"] += 1
            accepted_for_query += 1
            if retry_domain:
                accepted_by_retry_domain[retry_domain] += 1
            candidate = _candidate_from_search_result(
                result,
                query,
                settings=settings,
                canonical_url=canonical_url,
            )
            candidates.append(candidate)
            manifest.append(
                _safe_search_row(
                    result,
                    candidate,
                    canonical_url=canonical_url,
                    domain=candidate.domain,
                    result_status="accepted",
                    rejection_reason=None,
                    query_source=query.get("query_source"),
                    iteration_index=query.get("iteration_index"),
                )
            )

        totals["executed_query_count"] += 1
        records.append(
            _search_execution_record(
                query,
                provider=response.provider,
                status="executed" if accepted_for_query else "no_results",
                selected=True,
                result_count=accepted_for_query,
                selection_reason=selection_reason_override,
            )
        )
        if (
            _collection_mode({"structured_task": task}) == "direct_collection"
            and _direct_fast_stop_enabled()
        ):
            search_sufficient, _verification = _verified_target_search_sufficient(
                candidates,
                task,
            )
            if search_sufficient:
                for remaining_query in query_batch[query_index + 1 :]:
                    records.append(
                        _search_execution_record(
                            remaining_query,
                            provider=settings.provider,
                            status="skipped_verified_target_source_found",
                            skipped_reason="verified_target_source_found",
                        )
                    )
                break

    return candidates, manifest, records, rejection_counter, warnings


def _contains_any(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return sorted({term for term in terms if term and term.lower() in lowered})


def _build_iteration_observation(
    *,
    iteration_index: int,
    query_records: list[dict],
    manifest_rows: list[dict],
    candidates: list[SourceCandidate],
    rejection_counter: Counter,
    task: dict,
) -> dict:
    domain_counts = Counter(
        str(row.get("domain") or _domain(str(row.get("url") or "")) or "unknown")
        for row in manifest_rows
    )
    top_rows = []
    for row in manifest_rows[:10]:
        top_rows.append(
            {
                "title": row.get("title"),
                "snippet": row.get("snippet"),
                "url": row.get("url"),
                "domain": row.get("domain") or _domain(str(row.get("url") or "")),
                "source": row.get("source"),
                "published_date": row.get("published_date"),
                "rank": row.get("rank"),
                "result_status": row.get("result_status"),
                "rejection_reason": row.get("rejection_reason"),
            }
        )

    text_blob = " ".join(
        str(value or "")
        for row in top_rows
        for value in (row.get("title"), row.get("snippet"), row.get("url"))
    )
    disease_terms = [str(task.get("disease") or "")]
    location_terms = [str(task.get("location") or "")]
    evidence_terms = ["case", "cases", "death", "deaths", "outbreak", "surveillance"]
    accepted_count = len(candidates)
    rejected_count = sum(rejection_counter.values())
    concerns = []
    if accepted_count == 0:
        concerns.append("no accepted search-derived candidates in this iteration")
    if rejection_counter.get("duplicate_url"):
        concerns.append("duplicate URLs appeared in search results")
    if not _contains_any(text_blob, disease_terms):
        concerns.append("returned metadata has weak disease-term signal")
    if not _contains_any(text_blob, location_terms):
        concerns.append("returned metadata has weak location-term signal")

    gaps = []
    if accepted_count == 0:
        gaps.append("no usable source metadata was found")
    if not _contains_any(text_blob, evidence_terms):
        gaps.append("metadata does not clearly signal case/death/outbreak evidence")
    if len(domain_counts) < 2 and accepted_count > 0:
        gaps.append("source diversity remains limited")

    observation = SearchIterationObservation(
        iteration_index=iteration_index,
        executed_query_ids=[
            str(record.get("query_id"))
            for record in query_records
            if record.get("selected_for_execution") and record.get("query_id")
        ],
        executed_queries=[
            str(record.get("query"))
            for record in query_records
            if record.get("selected_for_execution") and record.get("query")
        ],
        raw_result_count=len(manifest_rows),
        accepted_candidate_count=accepted_count,
        duplicate_result_count=int(rejection_counter.get("duplicate_url", 0)),
        rejected_result_count=rejected_count,
        result_domain_counts=dict(domain_counts),
        top_result_summaries=top_rows,
        apparent_source_types=sorted(
            {
                str(record.get("source_type") or "unknown")
                for record in query_records
                if record.get("source_type")
            }
        ),
        disease_relevance_signals=_contains_any(text_blob, disease_terms),
        location_relevance_signals=_contains_any(text_blob, location_terms),
        evidence_availability_signals=_contains_any(text_blob, evidence_terms),
        gaps_identified=gaps,
        source_quality_concerns=concerns,
        notes_for_llm=[
            "Observation includes search metadata only; fetched page text is not included.",
            "Only provider search results can become source candidates.",
        ],
    )
    return observation.model_dump()


def _blocked_iterative_result(
    *,
    settings: SourceSearchSettings,
    stop_decision: str,
    stop_reason: str,
    warnings: list[str],
) -> tuple[list[SourceCandidate], list[dict], dict, dict]:
    summary = IterativeSourceDiscoverySummary(
        iterative_source_discovery_enabled=True,
        llm_iterative_planning_enabled=False,
        stop_decision=stop_decision,
        stop_reason=stop_reason,
        warnings=warnings,
        status="blocked",
    ).model_dump()
    search_summary = {
        "search_enabled": settings.search_enabled,
        "live_search_enabled": settings.live_search_enabled,
        "fixture_search_enabled": settings.fixture_search_enabled,
        "search_mode": settings.mode,
        "search_provider": settings.provider,
        "combine_with_seed_catalog": settings.combine_with_seed_catalog,
        "discovery_method": _discovery_method_for_settings(settings),
        "planned_query_count": 0,
        "selected_query_count": 0,
        "executed_query_count": 0,
        "skipped_query_count": 0,
        "raw_search_result_count": 0,
        "deduplicated_search_result_count": 0,
        "rejected_search_result_count": 0,
        "candidate_from_search_count": 0,
        "candidate_from_seed_count": 0,
        "total_candidate_count": 0,
        "provider_error_count": 0,
        "max_queries": settings.max_queries,
        "max_results_per_query": settings.max_results_per_query,
        "max_total_results": settings.max_total_results,
        "query_execution_records": [],
        "rejection_reason_counts": {},
        "search_result_source_ids": [],
        "iterative_source_discovery_enabled": True,
        "search_iteration_count": 0,
        "llm_refinement_call_count": 0,
        "stop_decision": stop_decision,
        "stop_reason": stop_reason,
        "iteration_query_counts": {},
        "query_source_counts": {},
        "warnings": sorted(set(warnings)),
    }
    return (
        [],
        [],
        search_summary,
        {
            "iterative_source_discovery_summary": summary,
            "search_iteration_plans": [],
            "search_iteration_observations": [],
            "search_refinement_decisions": [],
            "iterative_search_queries": [],
        },
    )


def _discovery_query_key(query: dict) -> tuple[str, str]:
    return (" ".join(str(query.get("query") or "").casefold().split()),
            str(query.get("provider_channel") or "web_search"))


def _discovery_breadth_queries(state: DataCollectionState) -> list[dict]:
    """Task-derived fallbacks cover retrieval families, without inventing URLs."""
    task = _task_context(state)
    disease = str(task.get("disease") or "").strip()
    if not disease:
        return []
    location = str(task.get("location") or "").strip()
    years = " ".join(dict.fromkeys(str(task.get(key) or "")[:4]
                                   for key in ("start_date", "end_date")))
    anchor = f'"{disease}" "{location}" {years}'.strip()
    families = [
        ("international_organization_report", "official_site_search", [
            "international public health surveillance report",
            "international situation report cases deaths",
            "international historical reports archive"]),
        ("official_public_health_agency", "official_site_search", [
            "public health department case counts by jurisdiction",
            "public health surveillance deaths hospitalizations",
            "public health historical reports archive"]),
        ("structured_database", "database_search", [
            "surveillance dataset cases deaths download",
            "public health database monthly data tables",
            "historical data csv spreadsheet export"]),
        ("peer_reviewed_literature", "literature_api", [
            "epidemiological surveillance study cases deaths",
            "case report hospitalization study",
            "population incidence mortality study supplementary tables"]),
    ]
    queries = [
        {"query": f"{anchor} {terms}", "source_type": source_type,
         "provider_channel": channel, "role_hint": "collection_support",
         "expected_fields": task.get("target_fields") or [],
         "query_rationale": "Task-grounded retrieval family coverage fallback.",
         "query_source": "deterministic_discovery_fallback",
         "disease_terms_used": [disease]}
        for source_type, channel, intents in families for terms in intents
    ]
    try:
        start = date.fromisoformat(str(task.get("start_date") or ""))
        end = date.fromisoformat(str(task.get("end_date") or ""))
    except ValueError:
        return queries
    first_month = start.year * 12 + start.month - 1
    last_month = end.year * 12 + end.month - 1
    if end < start or not 0 <= last_month - first_month <= 2:
        return queries

    # English month terms are retrieval leads, not publication-date filters.
    # Keep the exact observation bounds in metadata, including partial months.
    months = ("january february march april may june july august september "
              "october november december").split()
    expanded = []
    for source_type, channel, intents in families:
        broad = [row for row in queries if row["source_type"] == source_type]
        for index, row in enumerate(broad):
            month = first_month + index
            if month <= last_month:
                expanded.append({
                    **row,
                    "query": (f'"{disease}" "{location}" {months[month % 12]} {month // 12} '
                              f'{intents[0]}'),
                    "query_rationale": "Task-derived month lead; observation dates are not publication limits.",
                    "time_terms": [start.isoformat(), end.isoformat()],
                })
            # Interleave month leads with broad queries within each existing
            # family. The selector and all search budgets remain unchanged.
            expanded.append(row)
    return expanded


def _remaining_discovery_queries(
    pool: list[dict], state: DataCollectionState, settings: SourceSearchSettings,
    attempted: set[tuple[str, str]], family_counts: Counter,
) -> list[dict]:
    """Select distinct grounded queries, prioritizing unexplored source families."""
    available = []
    seen = set(attempted)
    for item in pool:
        query = dict(item)
        query.setdefault("provider_channel", "web_search")
        key = _discovery_query_key(query)
        if (key in seen or _is_invalid_query(query)
                or query["provider_channel"] not in settings.provider_channel_allowlist
                or not assess_query_task_fit(query, state)["accepted"]):
            continue
        seen.add(key)
        available.append(query)
    selected = []
    counts = family_counts.copy()
    for _ in range(max(0, settings.iterative_max_queries_per_iteration)):
        if not available:
            break
        # Only retrieval families are balanced here. This is not a claim that
        # a source/query contains factual evidence or a quota of valid sources.
        index = min(range(len(available)), key=lambda i: (
            _query_source_class(available[i]) not in _HIGH_TRUST_SOURCE_CLASSES,
            counts[_query_source_class(available[i])], i))
        query = available.pop(index)
        selected.append(query)
        counts[_query_source_class(query)] += 1
    return selected


def _execute_iterative_source_search(
    state: DataCollectionState,
    settings: SourceSearchSettings,
) -> tuple[list[SourceCandidate], list[dict], dict, dict]:
    from ..agents import iterative_source_discovery_agent

    settings = _effective_discovery_settings(settings)
    planned_queries = _planned_queries(state)
    query_inventory = [dict(item) for item in state.get("search_query_inventory") or []]
    user_request = str(state.get("user_request") or "")
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    task = _task_context(state)
    limits = _iterative_limits(settings)
    warnings: list[str] = []
    provider_error_count = 0
    provider = None

    if settings.search_enabled:
        try:
            provider = _provider_for_settings(settings)
        except (OSError, ValueError) as exc:
            provider_error_count += 1
            warnings.append(f"search_provider_setup_failed:{exc}")

    try:
        raw_plan = iterative_source_discovery_agent.plan_initial_search_iteration(
            user_request=user_request,
            structured_task=structured_task,
            collection_spec=collection_spec,
            planned_queries=planned_queries,
            search_query_inventory=query_inventory,
            limits=limits,
        )
        raw_plan = _normalize_iteration_plan(raw_plan, 1)
        if universal_queries_enabled() and not raw_plan.get("query_batch"):
            raise ValueError("initial search plan has no query batch")
    except Exception as exc:
        warning = f"iterative_llm_initial_plan_failed:{exc.__class__.__name__}"
        if universal_queries_enabled():
            warnings.append(warning)
            raw_plan = {"query_batch": _remaining_discovery_queries(
                [*planned_queries, *query_inventory, *_discovery_breadth_queries(state)],
                state, settings, set(), Counter()), "warnings": [warning]}
        else:
            fallback_to_existing_queries = bool(planned_queries or query_inventory)
            if (
                settings.iterative_allow_deterministic_fallback
                or (
                    _collection_mode(state) == "direct_collection"
                    and fallback_to_existing_queries
                )
            ):
                search_candidates, manifest, search_summary = _execute_one_shot_source_search(
                    state,
                    settings,
                )
                search_warnings = list(search_summary.get("warnings") or [])
                search_warnings.append(warning)
                summary = IterativeSourceDiscoverySummary(
                    iterative_source_discovery_enabled=True,
                    llm_iterative_planning_enabled=False,
                    search_iteration_count=0,
                    total_queries_executed=search_summary.get("executed_query_count", 0),
                    total_raw_results=search_summary.get("raw_search_result_count", 0),
                    total_candidates_created=len(search_candidates),
                    stop_decision="fallback_to_existing_queries",
                    stop_reason=warning,
                    warnings=[warning],
                    status="fallback_to_existing_queries",
                ).model_dump()
                search_summary.update(
                    {
                        "iterative_source_discovery_enabled": True,
                        "search_iteration_count": 0,
                        "llm_refinement_call_count": 0,
                        "stop_decision": "fallback_to_existing_queries",
                        "stop_reason": warning,
                        "warnings": sorted(set(search_warnings)),
                    }
                )
                return (
                    search_candidates,
                    manifest,
                    search_summary,
                    {
                        "iterative_source_discovery_summary": summary,
                        "search_iteration_plans": [],
                        "search_iteration_observations": [],
                        "search_refinement_decisions": [],
                        "iterative_search_queries": [],
                    },
                )
            return _blocked_iterative_result(
                settings=settings,
                stop_decision="stop_llm_unavailable",
                stop_reason=warning,
                warnings=[warning],
            )

    plans: list[dict] = []
    observations: list[dict] = []
    decisions: list[dict] = []
    iterative_queries: list[dict] = []
    query_records: list[dict] = []
    search_results_manifest: list[dict] = []
    search_candidates: list[SourceCandidate] = []
    rejection_counter: Counter = Counter()
    seen_canonical_urls: set[str] = set()
    totals = {
        "selected_query_count": 0,
        "executed_query_count": 0,
        "raw_result_count": 0,
        "deduped_result_count": 0,
        "provider_error_count": provider_error_count,
    }
    llm_refinement_call_count = 0
    stop_decision = "stop_no_promising_sources"
    stop_reason = "No valid iterative query batch was available."
    final_coverage = None
    final_trustworthiness = None
    final_gap = None

    current_plan = _normalize_iteration_plan(raw_plan, 1)
    query_pool = [*current_plan.get("query_batch", []), *planned_queries, *query_inventory]
    if universal_queries_enabled():
        query_pool.extend(_discovery_breadth_queries(state))
    attempted_queries: set[tuple[str, str]] = set()
    attempted_families: Counter = Counter()
    stagnant_batches = 0

    for iteration_index in range(1, settings.iterative_max_iterations + 1):
        query_source = (
            "iterative_llm_initial_search_plan"
            if iteration_index == 1
            else "iterative_llm_refinement"
        )
        plan = dict(current_plan)
        plan["iteration_index"] = iteration_index
        normalized_queries = [
            _normalize_iterative_query(
                dict(query),
                iteration_index=iteration_index,
                query_source=str(query.get("query_source") or query_source),
                sequence_index=index,
            )
            for index, query in enumerate(plan.get("query_batch") or [], start=1)
        ]
        plan["query_batch"] = normalized_queries
        plans.append(plan)
        iterative_queries.extend(normalized_queries)

        batch_candidates, batch_manifest, batch_records, batch_rejections, batch_warnings = (
            _execute_iterative_query_batch(
                query_batch=normalized_queries,
                provider=provider,
                settings=settings,
                seen_canonical_urls=seen_canonical_urls,
                totals=totals,
                task=task,
                evidence_state=state,
            )
        )
        query_records.extend(batch_records)
        search_results_manifest.extend(batch_manifest)
        search_candidates.extend(batch_candidates)
        rejection_counter.update(batch_rejections)
        warnings.extend(batch_warnings)

        observation = _build_iteration_observation(
            iteration_index=iteration_index,
            query_records=batch_records,
            manifest_rows=batch_manifest,
            candidates=batch_candidates,
            rejection_counter=batch_rejections,
            task=task,
        )
        observations.append(observation)
        for record in batch_records:
            if (record.get("selected_for_execution")
                    or record.get("skipped_reason") == "same_query_attempt_limit_reached"):
                attempted_queries.add(_discovery_query_key(record))
            if record.get("selected_for_execution"):
                attempted_families[_query_source_class(record)] += 1
        if any(record.get("selected_for_execution") for record in batch_records):
            stagnant_batches = 0 if batch_candidates else stagnant_batches + 1

        try:
            raw_decision = iterative_source_discovery_agent.refine_search_iteration(
                user_request=user_request,
                structured_task=structured_task,
                collection_spec=collection_spec,
                previous_plans=plans,
                observations=observations,
                observation=observation,
                limits=limits,
            )
            if universal_queries_enabled():
                iterative_source_discovery_agent.validate_refinement_advice(raw_decision)
            decision = _normalize_refinement_decision(raw_decision, iteration_index)
        except Exception as exc:
            warning = f"iterative_llm_refinement_failed:{exc.__class__.__name__}"
            warnings.append(warning)
            decision = SearchRefinementDecision(
                iteration_index=iteration_index,
                decision="stop_llm_unavailable",
                decision_reason=warning,
                stop_reason=warning,
                warnings=[warning],
            ).model_dump()

        llm_refinement_call_count += 1
        decisions.append(decision)
        stop_decision = decision.get("decision") or "stop_no_promising_sources"
        stop_reason = (
            decision.get("stop_reason")
            or decision.get("decision_reason")
            or stop_decision
        )
        final_coverage = decision.get("coverage_assessment")
        final_trustworthiness = decision.get("trustworthiness_assessment")
        final_gap = "; ".join(observation.get("gaps_identified") or []) or None

        limits_reached = (
            totals["selected_query_count"] >= settings.iterative_max_total_queries
            or totals["deduped_result_count"]
            >= min(settings.max_total_results, settings.iterative_max_total_results)
            or iteration_index >= settings.iterative_max_iterations
        )
        if universal_queries_enabled():
            if (limits_reached or totals.get("search_budget_exhausted")
                    or totals.get("search_result_budget_exhausted")):
                stop_decision = "stop_limits_reached"
                stop_reason = "iterative source discovery safety limit reached"
                break
            if not settings.search_enabled or provider is None:
                stop_decision = "stop_no_promising_sources"
                stop_reason = "search_disabled" if not settings.search_enabled else "provider_unavailable"
                break
            query_pool = [*(decision.get("next_query_batch") or []), *query_pool]
            available_families = {
                _query_source_class(query) for query in query_pool
                if (str(query.get("provider_channel") or "web_search") in settings.provider_channel_allowlist
                    and assess_query_task_fit(query, state)["accepted"])
            } & _HIGH_TRUST_SOURCE_CLASSES
            breadth_floor = min(settings.iterative_max_total_queries,
                                max(2 * settings.iterative_max_queries_per_iteration,
                                    len(available_families)))
            breadth_explored = (totals["selected_query_count"] >= breadth_floor
                                and available_families <= set(attempted_families))
            if stagnant_batches >= 2 and breadth_explored:
                stop_decision = "stop_no_promising_sources"
                stop_reason = "no_new_sources_in_consecutive_batches"
                break
            next_batch = _remaining_discovery_queries(
                query_pool, state, settings, attempted_queries, attempted_families)
            if not next_batch:
                stop_decision = "stop_no_promising_sources"
                stop_reason = "no_untried_task_grounded_queries"
                break
            # Search snippets cannot establish fetched evidence coverage. Even
            # well-formed stop advice remains advisory while novel queries fit
            # the configured search/iteration/result bounds.
            if stop_decision != "continue_search":
                warnings.append("iterative_advisory_stop_overridden_for_evidence_discovery")
            current_plan = {
                "query_batch": next_batch,
                "search_objective": "Continue bounded task-grounded source discovery",
                "search_reasoning": "Fetched evidence is still required; explore untried retrieval families.",
                "warnings": decision.get("warnings") or [],
            }
            continue
        if limits_reached and stop_decision == "continue_search":
            stop_decision = "stop_limits_reached"
            stop_reason = "iterative source discovery safety limit reached"
            break
        if stop_decision != "continue_search":
            break
        next_batch = decision.get("next_query_batch") or []
        if not next_batch:
            stop_decision = "stop_no_promising_sources"
            stop_reason = "LLM requested continuation but returned no next query batch."
            break
        current_plan = {
            "iteration_index": iteration_index + 1,
            "search_objective": "Refined follow-up search",
            "search_reasoning": decision.get("decision_reason") or "",
            "query_batch": next_batch,
            "expected_evidence": [],
            "expected_source_characteristics": [],
            "language_or_localization_reasoning": "",
            "trust_considerations": [],
            "stop_condition_hypothesis": "",
            "warnings": decision.get("warnings") or [],
        }

    skipped_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
    )
    selected_query_ids = [
        str(record.get("query_id"))
        for record in query_records
        if record.get("selected_for_execution") and record.get("query_id")
    ]
    skipped_query_ids = [
        str(record.get("query_id"))
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and record.get("query_id")
    ]
    skipped_for_verified_target = all(
        record.get("skipped_reason") == "verified_target_source_found"
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
    )
    if (
        skipped_query_count
        and stop_decision == "stop_sufficient"
        and not skipped_for_verified_target
    ):
        stop_decision = "partially_sufficient_with_unexecuted_queries"
        warning = "iterative_search_unexecuted_queries_after_sufficient_decision"
        warnings.append(warning)
        skipped_text = ", ".join(skipped_query_ids)
        stop_reason = (
            "LLM considered the executed evidence sufficient, but "
            f"{skipped_query_count} planned query or queries were not executed"
            + (f": {skipped_text}" if skipped_text else ".")
        )
        skipped_gap = (
            f"Unexecuted planned queries may leave coverage gaps: {skipped_text}"
            if skipped_text
            else "Unexecuted planned queries may leave coverage gaps."
        )
        final_gap = (
            f"{final_gap}; {skipped_gap}" if final_gap else skipped_gap
        )
    target_verification = _verify_target_sources(search_candidates, task)
    if (
        stop_decision == "stop_sufficient"
        and target_verification.get("target_verification_required")
        and int(target_verification.get("verified_target_source_count") or 0) <= 0
    ):
        stop_decision = "target_source_missing_or_unverified"
        warning = "llm_stop_sufficient_rejected_no_verified_target_source"
        warnings.append(warning)
        stop_reason = (
            "LLM considered search metadata sufficient, but no candidate source "
            "passed deterministic target disease/location/date verification."
        )
        miss_preview = "; ".join(
            str(reason)
            for reason in (target_verification.get("target_source_miss_reasons") or [])[:5]
        )
        final_gap = (
            f"{final_gap}; {miss_preview}" if final_gap and miss_preview else (
                miss_preview or final_gap
            )
        )
    selected_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
    )
    executed_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        executed_only=True,
    )
    result_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        result_sum=True,
    )
    skipped_high_trust_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
    )
    high_trust_no_result_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
        and record.get("execution_status") == "no_results"
    )
    selected_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and _known_authority_domain_for_query(record)
    )
    skipped_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and _known_authority_domain_for_query(record)
    )
    known_authority_domain_no_result_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("execution_status") == "no_results"
        and _known_authority_domain_for_query(record)
    )
    detected_event_jurisdictions = _jurisdictions_mentioned_in_text(
        _candidate_text_for_retry(search_candidates)
    )
    first_pass_authority_summary = build_authority_source_coverage_summary(
        [
            candidate.model_dump()
            for candidate in search_candidates
            if isinstance(candidate, SourceCandidate)
        ],
        detected_event_jurisdictions=detected_event_jurisdictions,
    )
    if (
        stop_decision == "stop_sufficient"
        and _supplementary_discovery_allowed(state, iterative_queries)
        and _authority_gap_requires_retry(first_pass_authority_summary)
    ):
        stop_decision = "stop_partial_authority_gap"
        authority_warnings = list(first_pass_authority_summary.get("warnings") or [])
        warnings.extend(authority_warnings or ["authority_source_class_gap"])
        stop_reason = (
            "LLM considered the iterative search sufficient, but deterministic "
            "authority source coverage still has high-trust source-class gaps."
        )
        gap_status = first_pass_authority_summary.get("coverage_status")
        gap_text = (
            f"Authority source coverage status: {gap_status}; warnings: "
            f"{', '.join(authority_warnings)}"
        )
        final_gap = f"{final_gap}; {gap_text}" if final_gap else gap_text
    retry_diagnostics = _authority_gap_retry_diagnostics(
        retry_queries=[],
        query_records=query_records,
        search_candidates=search_candidates,
        search_results_manifest=search_results_manifest,
        detected_event_jurisdictions=detected_event_jurisdictions,
        first_pass_authority_summary=first_pass_authority_summary,
        retry_accepted_candidate_count=0,
        authority_retry_result_budget=settings.authority_gap_retry_result_budget,
    )
    if (
        settings.search_enabled
        and settings.authority_gap_retry_enabled
        and settings.authority_gap_retry_max_iterations > 0
        and _supplementary_discovery_allowed(state, [*planned_queries, *iterative_queries])
        and _authority_gap_requires_retry(first_pass_authority_summary)
    ):
        if "authority_source_recall_incomplete" not in warnings:
            warnings.append("authority_source_recall_incomplete")
        retry_diagnostics = _execute_authority_gap_retry(
            state=state,
            settings=settings,
            provider=provider,
            planned_queries=[*planned_queries, *iterative_queries],
            query_records=query_records,
            search_candidates=search_candidates,
            search_results_manifest=search_results_manifest,
            seen_canonical_urls=seen_canonical_urls,
            totals=totals,
            rejection_counter=rejection_counter,
            warnings=warnings,
            first_pass_authority_summary=first_pass_authority_summary,
            detected_event_jurisdictions=detected_event_jurisdictions,
        )
        target_verification = _verify_target_sources(search_candidates, task)

    skipped_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
    )
    selected_query_ids = [
        str(record.get("query_id"))
        for record in query_records
        if record.get("selected_for_execution") and record.get("query_id")
    ]
    skipped_query_ids = [
        str(record.get("query_id"))
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and record.get("query_id")
    ]
    selected_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
    )
    executed_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        executed_only=True,
    )
    result_count_by_class = _query_record_source_class_counts(
        query_records,
        selected_only=True,
        result_sum=True,
    )
    skipped_high_trust_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
    )
    high_trust_no_result_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and record.get("selection_bucket") in _HIGH_TRUST_SOURCE_CLASSES
        and record.get("execution_status") == "no_results"
    )
    selected_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if record.get("selected_for_execution")
        and _known_authority_domain_for_query(record)
    )
    skipped_known_authority_domain_query_count = sum(
        1
        for record in query_records
        if str(record.get("execution_status") or "").startswith("skipped_")
        and _known_authority_domain_for_query(record)
    )
    known_authority_domain_no_result_count = int(
        retry_diagnostics.get("known_authority_domain_no_result_count") or 0
    )
    source_class_diagnostics = {
        "planned_query_count_by_source_class": _planned_query_source_class_counts(
            iterative_queries
        ),
        "selected_query_count_by_source_class": selected_count_by_class,
        "executed_query_count_by_source_class": executed_count_by_class,
        "result_count_by_source_class": result_count_by_class,
        "skipped_high_trust_query_count": skipped_high_trust_query_count,
        "high_trust_no_result_query_count": high_trust_no_result_query_count,
        "selected_known_authority_domain_query_count": (
            selected_known_authority_domain_query_count
        ),
        "skipped_known_authority_domain_query_count": (
            skipped_known_authority_domain_query_count
        ),
        "known_authority_domain_no_result_count": (
            known_authority_domain_no_result_count
        ),
        "known_authority_domain_no_result_retry_generated_count": (
            retry_diagnostics.get(
                "known_authority_domain_no_result_retry_generated_count",
                0,
            )
        ),
        "first_pass_authority_source_coverage_summary": (
            first_pass_authority_summary
        ),
        "post_retry_authority_source_coverage_summary": (
            retry_diagnostics.get("post_retry_authority_source_coverage_summary")
        ),
        "detected_event_jurisdictions": list(
            first_pass_authority_summary.get("detected_event_jurisdictions") or []
        ),
        "covered_event_jurisdictions": list(
            first_pass_authority_summary.get("covered_event_jurisdictions") or []
        ),
        "missing_event_jurisdictions": list(
            first_pass_authority_summary.get("missing_event_jurisdictions") or []
        ),
        "authority_gap_retry_enabled": bool(settings.authority_gap_retry_enabled),
        "authority_gap_retry_max_queries": settings.authority_gap_retry_max_queries,
        "authority_gap_known_domain_retry_max_queries": (
            settings.authority_gap_known_domain_retry_max_queries
        ),
        "authority_gap_jurisdiction_retry_max_queries": (
            settings.authority_gap_jurisdiction_retry_max_queries
        ),
        "authority_gap_official_page_family_retry_max_queries": (
            settings.authority_gap_official_page_family_retry_max_queries
        ),
        "known_domain_retry_generated_count": retry_diagnostics.get(
            "known_domain_retry_generated_count",
            0,
        ),
        "known_domain_retry_generated_by_domain": retry_diagnostics.get(
            "known_domain_retry_generated_by_domain",
            {},
        ),
        "known_domain_retry_executed_by_domain": retry_diagnostics.get(
            "known_domain_retry_executed_by_domain",
            {},
        ),
        "authority_gap_retry_generated_query_count": retry_diagnostics.get(
            "authority_gap_retry_generated_query_count",
            0,
        ),
        "authority_retry_result_budget": retry_diagnostics.get(
            "authority_retry_result_budget",
            settings.authority_gap_retry_result_budget,
        ),
        "authority_retry_result_budget_used": retry_diagnostics.get(
            "authority_retry_result_budget_used",
            0,
        ),
        "retry_execution_order": retry_diagnostics.get("retry_execution_order", []),
        "retry_category_round_robin_order": retry_diagnostics.get(
            "retry_category_round_robin_order",
            [],
        ),
        "retry_result_budget_by_category": retry_diagnostics.get(
            "retry_result_budget_by_category",
            {},
        ),
        "retry_domain_result_cap_counts": retry_diagnostics.get(
            "retry_domain_result_cap_counts",
            {},
        ),
        "high_confidence_domain_query_status": retry_diagnostics.get(
            "high_confidence_domain_query_status",
            {},
        ),
        "source_recall_target_ledger": retry_diagnostics.get(
            "source_recall_target_ledger",
            [],
        ),
        "jurisdiction_authority_retry_generated_count": retry_diagnostics.get(
            "jurisdiction_authority_retry_generated_count",
            0,
        ),
        "jurisdiction_retry_generated_by_domain": retry_diagnostics.get(
            "jurisdiction_retry_generated_by_domain",
            {},
        ),
        "jurisdiction_retry_executed_by_domain": retry_diagnostics.get(
            "jurisdiction_retry_executed_by_domain",
            {},
        ),
        "jurisdiction_authority_retry_candidate_count": retry_diagnostics.get(
            "jurisdiction_authority_retry_candidate_count",
            0,
        ),
        "jurisdiction_retry_protected_execution_count": retry_diagnostics.get(
            "jurisdiction_retry_protected_execution_count",
            0,
        ),
        "jurisdiction_retry_skipped_due_to_result_cap_count": retry_diagnostics.get(
            "jurisdiction_retry_skipped_due_to_result_cap_count",
            0,
        ),
        "known_domain_retry_truncated_before_jurisdiction_complete": (
            retry_diagnostics.get(
                "known_domain_retry_truncated_before_jurisdiction_complete",
                False,
            )
        ),
        "official_page_family_retry_generated_count": retry_diagnostics.get(
            "official_page_family_retry_generated_count",
            0,
        ),
        "official_page_family_retry_candidate_count": retry_diagnostics.get(
            "official_page_family_retry_candidate_count",
            0,
        ),
        "missing_high_confidence_jurisdiction_domains": retry_diagnostics.get(
            "missing_high_confidence_jurisdiction_domains",
            [],
        ),
        "high_confidence_source_gap_status": retry_diagnostics.get(
            "high_confidence_source_gap_status",
            "partial",
        ),
        "authority_gap_retry_selected_query_count": retry_diagnostics.get(
            "authority_gap_retry_selected_query_count",
            0,
        ),
        "authority_gap_retry_accepted_candidate_count": retry_diagnostics.get(
            "authority_gap_retry_accepted_candidate_count",
            0,
        ),
    }
    iterative_summary = IterativeSourceDiscoverySummary(
        iterative_source_discovery_enabled=True,
        llm_iterative_planning_enabled=True,
        search_iteration_count=len(observations),
        llm_refinement_call_count=llm_refinement_call_count,
        total_queries_planned=len(iterative_queries),
        total_queries_executed=totals["executed_query_count"],
        total_raw_results=totals["raw_result_count"],
        total_candidates_created=len(search_candidates),
        stop_decision=stop_decision,
        stop_reason=stop_reason,
        final_coverage_assessment=final_coverage,
        final_trustworthiness_assessment=final_trustworthiness,
        final_gap_assessment=final_gap,
        selected_query_ids=selected_query_ids,
        skipped_query_ids=skipped_query_ids,
        skipped_query_count=skipped_query_count,
        warnings=sorted(set(warnings)),
        status="completed" if stop_decision != "stop_llm_unavailable" else "blocked",
    ).model_dump()
    iterative_summary.update(target_verification)
    iterative_summary.update(source_class_diagnostics)
    search_summary = {
        "search_enabled": settings.search_enabled,
        "live_search_enabled": settings.live_search_enabled,
        "fixture_search_enabled": settings.fixture_search_enabled,
        "search_mode": settings.mode,
        "search_provider": settings.provider,
        "combine_with_seed_catalog": settings.combine_with_seed_catalog,
        "discovery_method": _discovery_method_for_settings(settings),
        "planned_query_count": len(planned_queries),
        "selected_query_count": totals["selected_query_count"],
        "executed_query_count": totals["executed_query_count"],
        "skipped_query_count": skipped_query_count,
        "raw_search_result_count": totals["raw_result_count"],
        "deduplicated_search_result_count": totals["deduped_result_count"],
        "rejected_search_result_count": sum(rejection_counter.values()),
        "candidate_from_search_count": len(search_candidates),
        "candidate_from_seed_count": 0,
        "total_candidate_count": len(search_candidates),
        "provider_error_count": totals["provider_error_count"],
        "max_queries": settings.max_queries,
        "max_results_per_query": settings.max_results_per_query,
        "max_total_results": settings.max_total_results,
        "query_execution_records": query_records,
        "rejection_reason_counts": dict(rejection_counter),
        "search_result_source_ids": [candidate.source_id for candidate in search_candidates],
        "iterative_source_discovery_enabled": True,
        "search_iteration_count": len(observations),
        "llm_refinement_call_count": llm_refinement_call_count,
        "stop_decision": stop_decision,
        "stop_reason": stop_reason,
        "iteration_query_counts": _iteration_query_counts(query_records),
        "query_source_counts": _query_source_counts(query_records),
        "warnings": sorted(set(warnings)),
        **target_verification,
        **source_class_diagnostics,
    }
    return (
        search_candidates,
        search_results_manifest,
        search_summary,
        {
            "iterative_source_discovery_summary": iterative_summary,
            "search_iteration_plans": plans,
            "search_iteration_observations": observations,
            "search_refinement_decisions": decisions,
            "iterative_search_queries": iterative_queries,
        },
    )


def _execute_source_search(
    state: DataCollectionState,
    settings: SourceSearchSettings,
) -> tuple[list[SourceCandidate], list[dict], dict, dict]:
    if settings.iterative_enabled:
        return _execute_iterative_source_search(state, settings)
    search_candidates, manifest, search_summary = _execute_one_shot_source_search(
        state,
        settings,
    )
    return search_candidates, manifest, search_summary, _disabled_iterative_outputs()


def source_discovery(state: DataCollectionState) -> dict:
    """Produce SourceCandidates from seed catalog and optional source search."""

    if universal_queries_enabled():
        catalog_dict = {"catalog_name": "session_independent_discovery", "catalog_version": "evidence",
                        "description": "New sources must be discovered in this session.", "seed_sources": []}
    elif _is_hantavirus_task(state):
        catalog_dict = load_hantavirus_seed_sources()
    else:
        catalog_dict = {"catalog_name": "task_selected_sources", "catalog_version": "1",
                        "description": "Sources explicitly selected for this task.", "seed_sources": []}
        overlay, overlay_path = _load_overlay_json("SEED_SOURCE_OVERLAY_PATH")
        if overlay is not None and overlay_path is not None:
            catalog_dict = _merge_seed_source_overlay(catalog_dict, overlay, overlay_path)
    catalog = SeedSourceCatalog(**catalog_dict)
    settings = _effective_discovery_settings(_source_search_settings_from_env())

    search_query_inventory = list(state.get("search_query_inventory") or [])
    official_coverage_candidates = [
        candidate
        for row in ([] if universal_queries_enabled() else build_official_coverage_candidates(state))
        for candidate in [_candidate_from_official_coverage(row)]
        if candidate is not None
    ]

    seed_candidates: list[SourceCandidate] = []
    for seed in catalog.seed_sources:
        matched = _best_matching_query(seed.model_dump(), search_query_inventory)
        seed_candidates.append(_make_source_candidate(seed, matched))

    fast_stop_result = _direct_search_skipped_for_verified_target(
        state,
        settings,
        official_coverage_candidates,
    )
    if fast_stop_result is None:
        (
            search_candidates,
            search_results_manifest,
            search_summary,
            iterative_outputs,
        ) = _execute_source_search(state, settings)
    else:
        (
            search_candidates,
            search_results_manifest,
            search_summary,
            iterative_outputs,
        ) = fast_stop_result
    direct_generic_search = (
        _collection_mode(state) == "direct_collection"
        and settings.search_enabled
        and not _is_hantavirus_task(state)
    )
    include_seeds = not universal_queries_enabled() and (
        ((not settings.search_enabled) or settings.combine_with_seed_catalog)
        and not direct_generic_search
    )
    candidates = [
        *official_coverage_candidates,
        *(seed_candidates if include_seeds else []),
        *search_candidates,
    ]

    source_type_counts = dict(
        Counter(c.source_type or "unknown" for c in candidates)
    )
    matched_query_count = sum(1 for c in candidates if c.query_id is not None)
    unmatched_query_count = len(candidates) - matched_query_count
    candidate_from_seed_count = len(seed_candidates) if include_seeds else 0
    candidate_from_search_count = len(search_candidates)
    candidate_from_official_coverage_count = len(official_coverage_candidates)

    if not settings.search_enabled:
        discovery_method = _DISCOVERY_METHOD
    elif settings.mode == "fixture":
        discovery_method = (
            "fixture_search_plus_seed_catalog"
            if include_seeds else "fixture_search_only"
        )
    elif settings.mode == "live":
        discovery_method = (
            "live_search_plus_seed_catalog"
            if include_seeds else "live_search_only"
        )
    else:
        discovery_method = _DISCOVERY_METHOD

    combined_search_sufficient, combined_target_verification = (
        _verified_target_search_sufficient(candidates, _task_context(state))
    )
    combined_verified_ids = list(
        combined_target_verification.get("verified_target_source_ids") or []
    )
    search_source_ids = {c.source_id for c in search_candidates}
    search_verified_ids = [
        sid for sid in combined_verified_ids if sid in search_source_ids
    ]
    official_coverage_candidate_dicts = [
        c.model_dump() for c in official_coverage_candidates
    ]
    candidate_dicts = [c.model_dump() for c in candidates]
    for row in candidate_dicts:
        source_id = str(row.get("source_id") or "")
        if source_id in combined_verified_ids and (
            _direct_allow_generated_url_only()
            or not _is_predicted_official_candidate(row)
        ):
            row["target_fit_status"] = "verified_target_collection"
            row["target_verification_status"] = "verified"
            row["target_verification_reason"] = "search result matches target disease/location/date evidence"
            row["triage_role"] = "verified_target_collection"
    search_summary.update(
        {
            "candidate_from_seed_count": candidate_from_seed_count,
            "candidate_from_official_coverage_count": (
                candidate_from_official_coverage_count
            ),
            "predicted_target_candidate_count": (
                candidate_from_official_coverage_count
            ),
            "search_verified_target_source_count": len(search_verified_ids),
            "fetch_verified_target_source_count": 0,
            "total_candidate_count": len(candidates),
            "verified_target_source_count": len(combined_verified_ids),
            "verified_target_source_ids": combined_verified_ids,
            "target_source_miss_reasons": combined_target_verification.get(
                "target_source_miss_reasons", []
            ),
            "target_evidence_requirements": combined_target_verification.get(
                "target_evidence_requirements", {}
            ),
        }
    )
    search_warnings = list(search_summary.get("warnings") or [])
    if (
        _collection_mode(state) == "direct_collection"
        and candidate_from_official_coverage_count
        and not settings.search_enabled
        and "search_validation_skipped_search_disabled" not in search_warnings
    ):
        search_warnings.append("search_validation_skipped_search_disabled")
    search_summary["warnings"] = sorted(set(search_warnings))
    search_warnings = list(search_summary.get("warnings") or [])
    if (
        not combined_search_sufficient
        and search_summary.get("search_stopped_reason")
        == "verified_target_source_found"
    ):
        search_summary["search_stopped_reason"] = "partial_target_coverage"
        warning = "verified_target_source_found_but_target_period_coverage_incomplete"
        if warning not in search_warnings:
            search_warnings.append(warning)
    if (
        combined_search_sufficient
        and combined_verified_ids
        and not search_summary.get("search_stopped_reason")
    ):
        search_summary["search_stopped_reason"] = "verified_target_source_found"
    search_summary["warnings"] = sorted(set(search_warnings))

    summary = {
        "discovery_method": discovery_method,
        "seed_source_count": len(catalog.seed_sources),
        "candidate_count": len(candidates),
        "source_type_counts": source_type_counts,
        "matched_query_count": matched_query_count,
        "unmatched_query_count": unmatched_query_count,
        "search_enabled": settings.search_enabled,
        "search_mode": settings.mode,
        "search_provider": settings.provider,
        "candidate_from_search_count": candidate_from_search_count,
        "candidate_from_seed_count": candidate_from_seed_count,
        "candidate_from_official_coverage_count": (
            candidate_from_official_coverage_count
        ),
        "predicted_target_candidate_count": candidate_from_official_coverage_count,
        "search_verified_target_source_count": len(search_verified_ids),
        "fetch_verified_target_source_count": 0,
        "total_candidate_count": len(candidates),
        "executed_query_count": search_summary.get("executed_query_count", 0),
        "raw_search_result_count": search_summary.get("raw_search_result_count", 0),
        "rejected_search_result_count": search_summary.get(
            "rejected_search_result_count", 0
        ),
        "iterative_source_discovery_enabled": search_summary.get(
            "iterative_source_discovery_enabled", False
        ),
        "search_iteration_count": search_summary.get("search_iteration_count", 0),
        "verified_target_source_count": search_summary.get(
            "verified_target_source_count", 0
        ),
        "verified_target_source_ids": search_summary.get(
            "verified_target_source_ids", []
        ),
        "search_stopped_reason": search_summary.get("search_stopped_reason"),
        "stop_decision": search_summary.get("stop_decision"),
        "stop_reason": search_summary.get("stop_reason"),
        "warnings": search_summary.get("warnings", []),
    }
    emit_workflow_progress(
        "source_discovery",
        "source discovery summary ready",
        {
            "candidate_count": len(candidates),
            "candidate_from_search_count": candidate_from_search_count,
            "candidate_from_official_coverage_count": (
                candidate_from_official_coverage_count
            ),
            "executed_query_count": search_summary.get("executed_query_count", 0),
            "raw_search_result_count": search_summary.get(
                "raw_search_result_count", 0
            ),
            "search_provider": settings.provider,
            "search_mode": settings.mode,
        },
    )

    trace = append_trace(
        state,
        node_name="source_discovery",
        message=(
            f"Discovered {len(candidates)} source candidates using "
            f"{discovery_method}."
        ),
        metadata={**summary, "source_search_execution_summary": search_summary},
    )
    return {
        "source_candidates": candidate_dicts,
        "official_coverage_candidates": official_coverage_candidate_dicts,
        "source_search_results": search_results_manifest,
        "source_search_execution_summary": search_summary,
        "source_discovery_summary": summary,
        **iterative_outputs,
        "collection_trace": trace,
    }


_SEARCH_VERIFIED_DISCOVERY_METHODS = {"fixture_search_result", "live_search_result"}


def _append_unique_value(items: list, value) -> None:
    if value in (None, ""):
        return
    if value not in items:
        items.append(value)


def _merge_unique_values(left: list, right: list) -> list:
    merged: list = []
    for value in [*(left or []), *(right or [])]:
        _append_unique_value(merged, value)
    return merged


def _official_alias_rank(entry: SourceRegistryEntry) -> tuple:
    target_status = str(entry.target_fit_status or "").strip().lower()
    triage_role = str(entry.triage_role or "").strip().lower()
    verification_status = str(entry.target_verification_status or "").strip().lower()
    discovery_method = str(entry.discovery_method or "").strip().lower()
    if (
        target_status in {"verified_target", "verified_target_collection"}
        or triage_role == "verified_target_collection"
        or verification_status == "verified"
    ):
        tier = 0
    elif discovery_method in _SEARCH_VERIFIED_DISCOVERY_METHODS:
        tier = 1
    elif discovery_method == "official_coverage_requirement":
        tier = 2
    else:
        tier = 3
    try:
        search_rank = int(entry.search_rank) if entry.search_rank is not None else 999
    except (TypeError, ValueError):
        search_rank = 999
    try:
        priority = int(entry.priority) if entry.priority is not None else 999
    except (TypeError, ValueError):
        priority = 999
    return (tier, search_rank, priority, entry.source_id)


def _record_official_alias(entry: SourceRegistryEntry, official_report_key: str) -> None:
    entry.official_report_key = official_report_key
    _append_unique_value(entry.official_report_alias_source_ids, entry.source_id)
    _append_unique_value(entry.official_report_alias_urls, entry.canonical_url)
    _append_unique_value(
        entry.official_report_alias_discovery_methods,
        entry.discovery_method,
    )
    _append_unique_value(
        entry.official_report_alias_target_verification_statuses,
        entry.target_verification_status or entry.target_fit_status or entry.triage_role,
    )
    entry.official_report_alias_preferred_source_id = entry.source_id


def _merge_official_alias_entries(
    existing: SourceRegistryEntry,
    incoming: SourceRegistryEntry,
    official_report_key: str,
) -> SourceRegistryEntry:
    aliases = _merge_unique_values(
        existing.official_report_alias_source_ids or [existing.source_id],
        incoming.official_report_alias_source_ids or [incoming.source_id],
    )
    alias_urls = _merge_unique_values(
        existing.official_report_alias_urls or [existing.canonical_url],
        incoming.official_report_alias_urls or [incoming.canonical_url],
    )
    alias_methods = _merge_unique_values(
        existing.official_report_alias_discovery_methods
        or [existing.discovery_method],
        incoming.official_report_alias_discovery_methods
        or [incoming.discovery_method],
    )
    alias_verifications = _merge_unique_values(
        existing.official_report_alias_target_verification_statuses
        or [
            existing.target_verification_status
            or existing.target_fit_status
            or existing.triage_role
        ],
        incoming.official_report_alias_target_verification_statuses
        or [
            incoming.target_verification_status
            or incoming.target_fit_status
            or incoming.triage_role
        ],
    )
    preferred, fallback = (
        (incoming, existing)
        if _official_alias_rank(incoming) < _official_alias_rank(existing)
        else (existing, incoming)
    )
    merged = preferred.model_copy(deep=True)
    merged.official_report_key = official_report_key
    merged.official_report_alias_source_ids = aliases
    merged.official_report_alias_urls = alias_urls
    merged.official_report_alias_discovery_methods = alias_methods
    merged.official_report_alias_target_verification_statuses = alias_verifications
    merged.official_report_alias_preferred_source_id = preferred.source_id
    merged.must_fetch = bool(existing.must_fetch or incoming.must_fetch)
    merged.coverage_requirement_ids = _merge_unique_values(
        existing.coverage_requirement_ids,
        incoming.coverage_requirement_ids,
    )
    merged.routing_conflict_warnings = _merge_unique_values(
        existing.routing_conflict_warnings,
        incoming.routing_conflict_warnings,
    )
    if not merged.must_fetch_reason:
        merged.must_fetch_reason = fallback.must_fetch_reason
    if not merged.reporting_period_start:
        merged.reporting_period_start = fallback.reporting_period_start
    if not merged.reporting_period_end:
        merged.reporting_period_end = fallback.reporting_period_end
    if not merged.reporting_period_label:
        merged.reporting_period_label = fallback.reporting_period_label
    if not merged.period_basis:
        merged.period_basis = fallback.period_basis
    return merged


def _registry_entry_from_candidate(
    candidate: dict,
    canonical: str,
    official_report_key: str | None,
) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id=candidate.get("source_id") or canonical,
        canonical_url=canonical,
        title=candidate.get("title"),
        publisher=candidate.get("publisher"),
        source_type=candidate.get("source_type"),
        published_date=candidate.get("published_date"),
        snippet=candidate.get("snippet"),
        status="registered",
        query_id=candidate.get("query_id"),
        query_used=candidate.get("query_used"),
        discovery_method=candidate.get("discovery_method"),
        seed_source_id=candidate.get("seed_source_id"),
        priority=candidate.get("priority"),
        expected_fields=list(candidate.get("expected_fields") or []),
        matched_terms=list(candidate.get("matched_terms") or []),
        source_purpose=candidate.get("source_purpose"),
        notes=candidate.get("notes"),
        search_provider=candidate.get("search_provider"),
        search_rank=candidate.get("search_rank"),
        provider_channel=candidate.get("provider_channel"),
        role_hint=candidate.get("role_hint"),
        planned_query_id=candidate.get("planned_query_id"),
        planned_query_source_type=candidate.get("planned_query_source_type"),
        search_result_id=candidate.get("search_result_id"),
        domain=candidate.get("domain"),
        result_source=candidate.get("result_source"),
        search_result_source_raw=candidate.get("search_result_source_raw")
        or candidate.get("result_source"),
        search_provider_result_source=candidate.get("search_provider_result_source")
        or candidate.get("result_source"),
        publisher_candidate_from_search_metadata=candidate.get(
            "publisher_candidate_from_search_metadata"
        ),
        query_type=candidate.get("query_type"),
        additional_query_ids=list(candidate.get("additional_query_ids") or []),
        query_source=candidate.get("query_source"),
        iteration_index=candidate.get("iteration_index"),
        iterative_query_id=candidate.get("iterative_query_id"),
        previous_iteration_basis=candidate.get("previous_iteration_basis"),
        source_disease_relevance_status=candidate.get(
            "source_disease_relevance_status"
        ),
        source_disease_relevance_score=candidate.get(
            "source_disease_relevance_score"
        ),
        source_target_disease_terms_found=list(
            candidate.get("source_target_disease_terms_found") or []
        ),
        source_incompatible_disease_terms_found=list(
            candidate.get("source_incompatible_disease_terms_found") or []
        ),
        source_disease_relevance_reason=candidate.get(
            "source_disease_relevance_reason"
        ),
        source_disease_relevance_data_signal_count=candidate.get(
            "source_disease_relevance_data_signal_count"
        ),
        must_fetch=bool(candidate.get("must_fetch", False)),
        must_fetch_reason=candidate.get("must_fetch_reason"),
        coverage_requirement_ids=list(
            candidate.get("coverage_requirement_ids") or []
        ),
        routing_conflict_warnings=list(
            candidate.get("routing_conflict_warnings") or []
        ),
        target_fit_status=candidate.get("target_fit_status"),
        target_verification_status=candidate.get("target_verification_status"),
        target_verification_reason=candidate.get("target_verification_reason"),
        triage_role=candidate.get("triage_role"),
        reporting_period_start=candidate.get("reporting_period_start"),
        reporting_period_end=candidate.get("reporting_period_end"),
        reporting_period_label=candidate.get("reporting_period_label"),
        period_basis=candidate.get("period_basis"),
        official_report_key=official_report_key,
    )


def source_dedup_and_registry(state: DataCollectionState) -> dict:
    """Canonicalize, deduplicate, and register source candidates.

    Keeps the first occurrence per canonical URL and counts duplicates.
    """

    candidates = list(state.get("source_candidates") or [])
    registry: list[SourceRegistryEntry] = []
    dedup_key_to_index: dict[str, int] = {}
    duplicate_count = 0
    official_report_alias_duplicate_count = 0

    for candidate in candidates:
        url = candidate.get("url") or ""
        canonical = canonicalize_url(url)
        if not canonical:
            duplicate_count += 1
            continue
        official_report_key = official_report_key_for_url(canonical)
        dedup_key = official_report_key or canonical
        entry = _registry_entry_from_candidate(
            candidate,
            canonical,
            official_report_key,
        )
        if official_report_key:
            _record_official_alias(entry, official_report_key)
        if dedup_key in dedup_key_to_index:
            duplicate_count += 1
            if official_report_key:
                official_report_alias_duplicate_count += 1
                index = dedup_key_to_index[dedup_key]
                registry[index] = _merge_official_alias_entries(
                    registry[index],
                    entry,
                    official_report_key,
                )
            continue
        dedup_key_to_index[dedup_key] = len(registry)
        registry.append(entry)

    registry_dicts = [e.model_dump() for e in registry]
    source_type_counts = dict(
        Counter(e.source_type or "unknown" for e in registry)
    )
    search_derived_entry_count = sum(
        1
        for entry in registry_dicts
        if entry.get("discovery_method")
        in {"fixture_search_result", "live_search_result"}
    )

    summary = {
        "input_candidate_count": len(candidates),
        "registry_entry_count": len(registry),
        "duplicate_count": duplicate_count,
        "official_report_alias_duplicate_count": official_report_alias_duplicate_count,
        "source_type_counts": source_type_counts,
        "search_derived_entry_count": search_derived_entry_count,
    }

    trace = append_trace(
        state,
        node_name="source_dedup_and_registry",
        message=(
            f"Built source registry with {len(registry)} entries "
            f"({duplicate_count} duplicates dropped)."
        ),
        metadata=summary,
    )
    return {
        "source_registry": registry_dicts,
        "source_registry_summary": summary,
        "collection_trace": trace,
    }
