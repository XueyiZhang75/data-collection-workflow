"""Shared runtime profile helpers for Data Collection Workflow runs.

A workflow runtime profile controls the graph input, source overlays, live
fetch, LLM stages, source roles, validation evidence, and output paths.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env, normalize_env_updates

import os
import json
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_PROFILE_NAME = "data_collection"
DEFAULT_SEARCH_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "search_fixtures"
    / "example_search_results.json"
)
DEFAULT_SEARCH_PROVIDER_CHANNEL_ALLOWLIST = [
    "web_search",
    "official_site_search",
    "news_search",
    "literature_api",
    "database_search",
]

# Public defaults do not assign any source to a task or validation split.
COLLECTION_SOURCE_IDS: list[str] = []
CONTEXT_SOURCE_IDS: list[str] = []
VALIDATION_SOURCE_IDS: list[str] = []

DEFAULT_USER_REQUEST = ""
DEFAULT_TARGET_FIELDS = [
    "disease",
    "virus_or_syndrome",
    "country",
    "subnational_location",
    "date_reported",
    "event_start_date",
    "event_end_date",
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "case_definition",
    "source_url",
    "source_type",
    "evidence_quote",
]
DEFAULT_SOURCE_PREFERENCES = [
    "official_public_health_agency",
    "international_organization_report",
    "peer_reviewed_literature",
    "structured_database",
    "news_and_situation_report",
]
DEFAULT_STRUCTURED_TASK: dict = {}
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = ""
PIPELINE_MODE = "standard"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT
DEFAULT_CONSOLE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "workflow_console"
DEFAULT_WORKFLOW_RUN_CONFIG_PATH = PROJECT_ROOT / "configs" / "workflow.jsonc"


def load_project_env() -> bool:
    """Load the user's project credentials without replacing shell settings."""
    from dotenv import load_dotenv

    return load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_llm_selection(
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    """Resolve a provider and user-selected model without crossing providers."""
    environment_provider = str(get_env("LLM_PROVIDER") or "").strip().lower()
    selected_provider = str(provider or "").strip().lower() or environment_provider or DEFAULT_PROVIDER
    selected_model = str(model or "").strip()
    if not selected_model and environment_provider in {"", selected_provider}:
        selected_model = str(get_env("LLM_MODEL") or "").strip()
    return selected_provider, selected_model


ENV_KEYS = [
    "PIPELINE_MODE",
    "COLLECTION_MODE",
    "SEED_SOURCE_OVERLAY_PATH",
    "SOURCE_ROLE_POLICY_OVERLAY_PATH",
    "USE_FIXTURE_DOCUMENTS",
    "ENABLE_LIVE_FETCH",
    "ENABLE_LLM_DISEASE_INTELLIGENCE",
    "DISEASE_INTELLIGENCE_FORCE_LLM",
    "DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED",
    "ENABLE_LLM_SOURCE_PLANNING",
    "ENABLE_LLM_SOURCE_CRITIC",
    "ENABLE_LLM_SOURCE_CREDIBILITY",
    "ENABLE_LLM_SOURCE_IDENTITY",
    "ENABLE_LLM_EXTRACTION",
    "LLM_SOURCE_IDENTITY_MAX_SOURCES",
    "LLM_SOURCE_IDENTITY_POST_FETCH",
    "LLM_SOURCE_IDENTITY_REQUIRE_LLM",
    "LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK",
    "LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST",
    "LLM_SOURCE_CRITIC_MAX_SOURCES",
    "LLM_SOURCE_CRITIC_MAX_FAILURES",
    "LLM_SOURCE_CRITIC_MAX_NODE_SECONDS",
    "LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH",
    "LLM_SOURCE_CREDIBILITY_MAX_SOURCES",
    "LLM_SOURCE_CREDIBILITY_SOURCE_ID_ALLOWLIST",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_THINKING",
    "LLM_EFFORT",
    "LLM_STRUCTURED_OUTPUT_METHOD",
    "LLM_MAX_CHUNKS",
    "LLM_EXTRACTION_SCHEDULER_MODE",
    "LLM_EXTRACTION_MAX_CONCURRENCY",
    "LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS",
    "LLM_EXTRACTION_SAFETY_MAX_CALLS",
    "LLM_EXTRACTION_ROLLING_YIELD_WINDOW",
    "LLM_SOFT_PRIMARY_CALLS",
    "LLM_HARD_PRIMARY_CALLS",
    "LLM_FOCUSED_RECOVERY_RESERVED_CALLS",
    "LLM_MAX_DOMAIN_CALL_SHARE",
    "LLM_HIGH_VALUE_SOURCE_MIN_SPANS",
    "LLM_EVENT_SOURCE_MIN_SPANS",
    "LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE",
    "LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS",
    "LLM_FOCUSED_RECOVERY_MAX_CALLS",
    "LLM_FOCUSED_RECOVERY_MAX_SOURCES",
    "LLM_MAX_TOKENS",
    "LLM_FALLBACK_TO_RULE_BASED",
    "DIRECT_COLLECTION_ENABLE_AUDIT_VALIDATION",
    "SOURCE_ID_ALLOWLIST",
    "FETCH_TIMEOUT_SECONDS",
    "FETCH_SEARCH_DERIVED_SOURCES",
    "FETCH_MAX_SEARCH_DERIVED_SOURCES",
    "FETCH_MAX_TOTAL_SOURCES",
    "FETCH_MIN_CREDIBILITY_SCORE",
    "FETCH_ALLOWED_FINAL_ROLES",
    "FETCH_ALLOW_NEEDS_REVIEW",
    "FETCH_DOMAIN_ALLOWLIST",
    "FETCH_DOMAIN_BLOCKLIST",
    "FETCH_MAX_BYTES",
    "FETCH_USER_AGENT",
    "FETCH_PARSE_PDF_TEXT",
    "FETCH_PARSE_TABLES",
    "FETCH_STORE_RAW_TEXT",
    "FETCH_MAX_NODE_SECONDS",
    "CONTENT_FIXTURE_MAP_PATH",
    "EXTERNAL_FETCH_ENABLED",
    "EXTERNAL_FETCH_PROVIDER_ORDER",
    "TAVILY_EXTRACT_FORMAT",
    "TAVILY_EXTRACT_DEPTH",
    "TAVILY_EXTRACT_TIMEOUT_SECONDS",
    "TAVILY_EXTRACT_CHUNKS_PER_SOURCE",
    "ENABLE_LIVE_SEARCH",
    "SEARCH_MODE",
    "SEARCH_PROVIDER",
    "SEARCH_FIXTURE_PATH",
    "SEARCH_MAX_QUERIES",
    "SEARCH_MAX_RESULTS_PER_QUERY",
    "SEARCH_MAX_TOTAL_RESULTS",
    "SEARCH_TIMEOUT_SECONDS",
    "SEARCH_COMBINE_WITH_SEED_CATALOG",
    "SEARCH_PROVIDER_CHANNEL_ALLOWLIST",
    "ENABLE_ITERATIVE_SOURCE_DISCOVERY",
    "ITERATIVE_SEARCH_MAX_ITERATIONS",
    "ITERATIVE_SEARCH_MAX_QUERIES_PER_ITERATION",
    "ITERATIVE_SEARCH_MAX_TOTAL_QUERIES",
    "ITERATIVE_SEARCH_MAX_TOTAL_RESULTS",
    "ITERATIVE_SEARCH_REQUIRE_LLM",
    "ITERATIVE_SEARCH_ALLOW_DETERMINISTIC_FALLBACK",
    "ITERATIVE_SEARCH_STOP_WHEN_LLM_SAYS_SUFFICIENT",
    "ITERATIVE_SEARCH_REQUIRE_OBSERVATION_AFTER_EACH_ITERATION",
    "AUTHORITY_GAP_RETRY_ENABLED",
    "AUTHORITY_GAP_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES",
    "AUTHORITY_GAP_RETRY_MAX_ITERATIONS",
    "VALIDATION_MODE",
    "VALIDATION_HELD_OUT_RECORDS_PATH",
    "ALLOW_INCOMPATIBLE_VALIDATION_RECORDS",
    "HUMAN_REVIEW_ENABLED",
    "HUMAN_REVIEW_DECISIONS_PATH",
    "HUMAN_REVIEW_APPLY_DECISIONS",
    "HUMAN_REVIEW_REQUIRE_REVIEWER_ID",
    "ANOMALY_MAX_CASES_THRESHOLD",
    "ANOMALY_MAX_DEATHS_THRESHOLD",
    "ANOMALY_SPIKE_MULTIPLIER",
    "ANOMALY_MIN_PRIOR_RECORDS",
]


def workflow_run_env(
    *,
    pipeline_mode: str = PIPELINE_MODE,
    collection_mode: str = "standard",
    seed_source_overlay_path: str | Path | None = None,
    source_role_policy_overlay_path: str | Path | None = None,
    use_fixture_documents: bool = False,
    live_fetch: bool = True,
    llm_source_planning: bool = True,
    llm_source_critic: bool = True,
    llm_source_credibility: bool = False,
    llm_source_identity: bool = True,
    llm_extraction: bool = True,
    llm_disease_intelligence: bool = True,
    disease_intelligence_force_llm: bool = True,
    disease_intelligence_fallback_to_curated: bool = True,
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 30.0,
    llm_max_chunks: int = 8,
    llm_extraction_scheduler_mode: str = "quality_adaptive",
    llm_extraction_max_concurrency: int = 4,
    llm_extraction_soft_checkpoint_calls: int | None = None,
    llm_extraction_safety_max_calls: int | None = None,
    llm_extraction_rolling_yield_window: int = 40,
    llm_soft_primary_calls: int | None = None,
    llm_hard_primary_calls: int | None = None,
    llm_focused_recovery_reserved_calls: int = 120,
    llm_max_domain_call_share: float = 0.20,
    llm_high_value_source_min_spans: int = 2,
    llm_event_source_min_spans: int = 1,
    llm_must_fetch_min_chunks_per_source: int = 6,
    llm_official_extraction_max_chunks: int = 30,
    llm_focused_recovery_max_calls: int = 40,
    llm_focused_recovery_max_sources: int = 30,
    llm_source_critic_max_sources: int | None = 6,
    llm_source_critic_max_failures: int | None = None,
    llm_source_critic_max_node_seconds: float | None = None,
    llm_source_critic_review_blocks_fetch: bool = True,
    llm_source_credibility_max_sources: int | None = None,
    llm_source_credibility_source_id_allowlist: list[str] | None = None,
    llm_source_identity_max_sources: int | None = 30,
    llm_source_identity_post_fetch: bool = True,
    llm_source_identity_require_llm: bool = True,
    llm_source_identity_allow_deterministic_fallback: bool = False,
    llm_max_tokens: int = 4096,
    llm_timeout_seconds: float | None = None,
    llm_max_retries: int = 2,
    llm_thinking: str | None = None,
    llm_effort: str | None = None,
    llm_structured_output_method: str | None = None,
    fallback_to_rule_based: bool = False,
    source_id_allowlist: list[str] | None = None,
    source_id_allowlist_enabled: bool = False,
    llm_source_critic_source_id_allowlist: list[str] | None = None,
    live_search: bool = False,
    search_mode: str = "disabled",
    search_provider: str = "tavily",
    search_fixture_path: str | Path | None = None,
    search_max_queries: int = 3,
    search_max_results_per_query: int = 5,
    search_max_total_results: int = 15,
    search_timeout_seconds: float = 15.0,
    search_combine_with_seed_catalog: bool = False,
    search_provider_channel_allowlist: list[str] | None = None,
    iterative_source_discovery: bool = False,
    iterative_search_max_iterations: int = 3,
    iterative_search_max_queries_per_iteration: int = 3,
    iterative_search_max_total_queries: int = 9,
    iterative_search_max_total_results: int = 30,
    iterative_search_require_llm: bool = True,
    iterative_search_allow_deterministic_fallback: bool = False,
    iterative_search_stop_when_llm_says_sufficient: bool = True,
    iterative_search_require_observation_after_each_iteration: bool = True,
    authority_gap_retry_enabled: bool = True,
    authority_gap_retry_max_queries: int = 24,
    authority_gap_known_domain_retry_max_queries: int = 10,
    authority_gap_jurisdiction_retry_max_queries: int = 12,
    authority_gap_official_page_family_retry_max_queries: int = 10,
    authority_gap_retry_result_budget: int = 80,
    authority_gap_retry_max_iterations: int = 1,
    fetch_search_derived_sources: bool = False,
    fetch_max_search_derived_sources: int = 2,
    fetch_max_total_sources: int = 10,
    fetch_min_credibility_score: float = 0.55,
    fetch_allowed_final_roles: list[str] | None = None,
    fetch_allow_needs_review: bool = False,
    fetch_domain_allowlist: list[str] | None = None,
    fetch_domain_blocklist: list[str] | None = None,
    fetch_max_bytes: int = 1_000_000,
    fetch_parse_pdf_text: bool = True,
    fetch_parse_tables: bool = True,
    fetch_store_raw_text: bool = False,
    fetch_user_agent: str = "data-collection-workflow/0.1",
    fetch_max_node_seconds: float | None = None,
    content_fixture_map_path: str | Path | None = None,
    external_fetch_enabled: bool = True,
    external_fetch_provider_order: list[str] | None = None,
    tavily_extract_format: str = "markdown",
    tavily_extract_depth: str = "advanced",
    tavily_extract_timeout_seconds: float = 45.0,
    tavily_extract_chunks_per_source: int = 5,
    human_review_enabled: bool = True,
    human_review_decisions_path: str | Path | None = None,
    human_review_apply_decisions: bool = False,
    human_review_require_reviewer_id: bool = True,
    validation_mode: str = "live_cross_source",
    validation_held_out_records_path: str | Path | None = None,
    allow_incompatible_validation_records: bool = False,
    direct_collection_enable_audit_validation: bool = False,
    anomaly_max_cases_threshold: float = 1_000_000,
    anomaly_max_deaths_threshold: float = 100_000,
    anomaly_spike_multiplier: float = 10,
    anomaly_min_prior_records: int = 1,
) -> dict[str, str]:
    """Return the environment for a configured Data Collection Workflow run."""

    workflow_source_ids = list(source_id_allowlist or []) if source_id_allowlist_enabled else []
    critic_source_ids = list(llm_source_critic_source_id_allowlist or [])
    seed_overlay = seed_source_overlay_path
    role_overlay = source_role_policy_overlay_path
    resolved_soft_checkpoint_calls = int(
        llm_extraction_soft_checkpoint_calls
        if llm_extraction_soft_checkpoint_calls is not None
        else llm_soft_primary_calls
        if llm_soft_primary_calls is not None
        else 400
    )
    resolved_safety_max_calls = int(
        llm_extraction_safety_max_calls
        if llm_extraction_safety_max_calls is not None
        else llm_hard_primary_calls
        if llm_hard_primary_calls is not None
        else 2400
    )
    env = {
        "PIPELINE_MODE": str(pipeline_mode or PIPELINE_MODE),
        "COLLECTION_MODE": collection_mode,
        "SEED_SOURCE_OVERLAY_PATH": str(seed_overlay) if seed_overlay else "",
        "SOURCE_ROLE_POLICY_OVERLAY_PATH": (
            str(role_overlay) if role_overlay else ""
        ),
        "USE_FIXTURE_DOCUMENTS": "true" if use_fixture_documents else "false",
        "ENABLE_LIVE_FETCH": "true" if live_fetch else "false",
        "ENABLE_LLM_DISEASE_INTELLIGENCE": (
            "true" if llm_disease_intelligence else "false"
        ),
        "DISEASE_INTELLIGENCE_FORCE_LLM": (
            "true" if disease_intelligence_force_llm else "false"
        ),
        "DISEASE_INTELLIGENCE_FALLBACK_TO_CURATED": (
            "true" if disease_intelligence_fallback_to_curated else "false"
        ),
        "ENABLE_LLM_SOURCE_PLANNING": "true" if llm_source_planning else "false",
        "ENABLE_LLM_SOURCE_CRITIC": "true" if llm_source_critic else "false",
        "ENABLE_LLM_SOURCE_CREDIBILITY": (
            "true" if llm_source_credibility else "false"
        ),
        "ENABLE_LLM_SOURCE_IDENTITY": (
            "true" if llm_source_identity else "false"
        ),
        "LLM_SOURCE_IDENTITY_POST_FETCH": (
            "true" if llm_source_identity_post_fetch else "false"
        ),
        "LLM_SOURCE_IDENTITY_REQUIRE_LLM": (
            "true" if llm_source_identity_require_llm else "false"
        ),
        "LLM_SOURCE_IDENTITY_ALLOW_DETERMINISTIC_FALLBACK": (
            "true" if llm_source_identity_allow_deterministic_fallback else "false"
        ),
        "ENABLE_LLM_EXTRACTION": "true" if llm_extraction else "false",
        "SOURCE_ID_ALLOWLIST": ",".join(workflow_source_ids),
        "FETCH_TIMEOUT_SECONDS": str(timeout_seconds),
        "FETCH_SEARCH_DERIVED_SOURCES": (
            "true" if fetch_search_derived_sources else "false"
        ),
        "FETCH_MAX_SEARCH_DERIVED_SOURCES": str(
            fetch_max_search_derived_sources
        ),
        "FETCH_MAX_TOTAL_SOURCES": str(fetch_max_total_sources),
        "FETCH_MIN_CREDIBILITY_SCORE": str(fetch_min_credibility_score),
        "FETCH_ALLOWED_FINAL_ROLES": ",".join(
            fetch_allowed_final_roles
            or ["collection", "validation", "collection_support", "context"]
        ),
        "FETCH_ALLOW_NEEDS_REVIEW": (
            "true" if fetch_allow_needs_review else "false"
        ),
        "FETCH_DOMAIN_ALLOWLIST": ",".join(fetch_domain_allowlist or []),
        "FETCH_DOMAIN_BLOCKLIST": ",".join(fetch_domain_blocklist or []),
        "FETCH_MAX_BYTES": str(fetch_max_bytes),
        "FETCH_PARSE_PDF_TEXT": "true" if fetch_parse_pdf_text else "false",
        "FETCH_PARSE_TABLES": "true" if fetch_parse_tables else "false",
        "FETCH_STORE_RAW_TEXT": "true" if fetch_store_raw_text else "false",
        "FETCH_USER_AGENT": fetch_user_agent,
        "FETCH_MAX_NODE_SECONDS": (
            str(fetch_max_node_seconds) if fetch_max_node_seconds else ""
        ),
        "CONTENT_FIXTURE_MAP_PATH": (
            str(content_fixture_map_path) if content_fixture_map_path else ""
        ),
        "EXTERNAL_FETCH_ENABLED": "true" if external_fetch_enabled else "false",
        "EXTERNAL_FETCH_PROVIDER_ORDER": ",".join(
            external_fetch_provider_order or ["tavily_extract", "native_requests"]
        ),
        "TAVILY_EXTRACT_FORMAT": str(tavily_extract_format or "markdown"),
        "TAVILY_EXTRACT_DEPTH": str(tavily_extract_depth or "advanced"),
        "TAVILY_EXTRACT_TIMEOUT_SECONDS": str(tavily_extract_timeout_seconds),
        "TAVILY_EXTRACT_CHUNKS_PER_SOURCE": str(
            tavily_extract_chunks_per_source
        ),
        "LLM_MAX_CHUNKS": str(llm_max_chunks),
        "LLM_EXTRACTION_SCHEDULER_MODE": str(
            llm_extraction_scheduler_mode
        ),
        "LLM_EXTRACTION_MAX_CONCURRENCY": str(
            llm_extraction_max_concurrency
        ),
        "LLM_EXTRACTION_SOFT_CHECKPOINT_CALLS": str(
            resolved_soft_checkpoint_calls
        ),
        "LLM_EXTRACTION_SAFETY_MAX_CALLS": str(
            resolved_safety_max_calls
        ),
        "LLM_EXTRACTION_ROLLING_YIELD_WINDOW": str(
            llm_extraction_rolling_yield_window
        ),
        "LLM_SOFT_PRIMARY_CALLS": str(
            resolved_soft_checkpoint_calls
        ),
        "LLM_HARD_PRIMARY_CALLS": str(
            resolved_safety_max_calls
        ),
        "LLM_FOCUSED_RECOVERY_RESERVED_CALLS": str(
            llm_focused_recovery_reserved_calls
        ),
        "LLM_MAX_DOMAIN_CALL_SHARE": str(llm_max_domain_call_share),
        "LLM_HIGH_VALUE_SOURCE_MIN_SPANS": str(
            llm_high_value_source_min_spans
        ),
        "LLM_EVENT_SOURCE_MIN_SPANS": str(llm_event_source_min_spans),
        "LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE": str(
            llm_must_fetch_min_chunks_per_source
        ),
        "LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS": str(
            llm_official_extraction_max_chunks
        ),
        "LLM_FOCUSED_RECOVERY_MAX_CALLS": str(
            llm_focused_recovery_max_calls
        ),
        "LLM_FOCUSED_RECOVERY_MAX_SOURCES": str(
            llm_focused_recovery_max_sources
        ),
        "LLM_MAX_TOKENS": str(llm_max_tokens),
        "LLM_TIMEOUT_SECONDS": (
            str(llm_timeout_seconds) if llm_timeout_seconds else ""
        ),
        "LLM_MAX_RETRIES": str(llm_max_retries),
        "LLM_THINKING": str(llm_thinking or get_env("LLM_THINKING") or "auto"),
        "LLM_EFFORT": str(
            llm_effort if llm_effort is not None else get_env("LLM_EFFORT") or ""
        ),
        "LLM_STRUCTURED_OUTPUT_METHOD": str(
            llm_structured_output_method
            or get_env("LLM_STRUCTURED_OUTPUT_METHOD")
            or "auto"
        ),
        "LLM_SOURCE_CRITIC_SOURCE_ID_ALLOWLIST": ",".join(critic_source_ids),
        "LLM_SOURCE_CRITIC_REVIEW_BLOCKS_FETCH": (
            "true" if llm_source_critic_review_blocks_fetch else "false"
        ),
        "LLM_FALLBACK_TO_RULE_BASED": (
            "true" if fallback_to_rule_based else "false"
        ),
        "ENABLE_LIVE_SEARCH": "true" if live_search else "false",
        "SEARCH_MODE": search_mode,
        "SEARCH_PROVIDER": search_provider,
        "SEARCH_FIXTURE_PATH": str(
            search_fixture_path or DEFAULT_SEARCH_FIXTURE_PATH
        ),
        "SEARCH_MAX_QUERIES": str(search_max_queries),
        "SEARCH_MAX_RESULTS_PER_QUERY": str(search_max_results_per_query),
        "SEARCH_MAX_TOTAL_RESULTS": str(search_max_total_results),
        "SEARCH_TIMEOUT_SECONDS": str(search_timeout_seconds),
        "SEARCH_COMBINE_WITH_SEED_CATALOG": (
            "true" if search_combine_with_seed_catalog else "false"
        ),
        "SEARCH_PROVIDER_CHANNEL_ALLOWLIST": ",".join(
            search_provider_channel_allowlist
            or DEFAULT_SEARCH_PROVIDER_CHANNEL_ALLOWLIST
        ),
        "ENABLE_ITERATIVE_SOURCE_DISCOVERY": (
            "true" if iterative_source_discovery else "false"
        ),
        "ITERATIVE_SEARCH_MAX_ITERATIONS": str(
            iterative_search_max_iterations
        ),
        "ITERATIVE_SEARCH_MAX_QUERIES_PER_ITERATION": str(
            iterative_search_max_queries_per_iteration
        ),
        "ITERATIVE_SEARCH_MAX_TOTAL_QUERIES": str(
            iterative_search_max_total_queries
        ),
        "ITERATIVE_SEARCH_MAX_TOTAL_RESULTS": str(
            iterative_search_max_total_results
        ),
        "ITERATIVE_SEARCH_REQUIRE_LLM": (
            "true" if iterative_search_require_llm else "false"
        ),
        "ITERATIVE_SEARCH_ALLOW_DETERMINISTIC_FALLBACK": (
            "true" if iterative_search_allow_deterministic_fallback else "false"
        ),
        "ITERATIVE_SEARCH_STOP_WHEN_LLM_SAYS_SUFFICIENT": (
            "true" if iterative_search_stop_when_llm_says_sufficient else "false"
        ),
        "ITERATIVE_SEARCH_REQUIRE_OBSERVATION_AFTER_EACH_ITERATION": (
            "true"
            if iterative_search_require_observation_after_each_iteration
            else "false"
        ),
        "AUTHORITY_GAP_RETRY_ENABLED": (
            "true" if authority_gap_retry_enabled else "false"
        ),
        "AUTHORITY_GAP_RETRY_MAX_QUERIES": str(
            authority_gap_retry_max_queries
        ),
        "AUTHORITY_GAP_KNOWN_DOMAIN_RETRY_MAX_QUERIES": str(
            authority_gap_known_domain_retry_max_queries
        ),
        "AUTHORITY_GAP_JURISDICTION_RETRY_MAX_QUERIES": str(
            authority_gap_jurisdiction_retry_max_queries
        ),
        "AUTHORITY_GAP_OFFICIAL_PAGE_FAMILY_RETRY_MAX_QUERIES": str(
            authority_gap_official_page_family_retry_max_queries
        ),
        "AUTHORITY_GAP_RETRY_RESULT_BUDGET": str(
            authority_gap_retry_result_budget
        ),
        "AUTHORITY_GAP_RETRY_MAX_ITERATIONS": str(
            authority_gap_retry_max_iterations
        ),
        "VALIDATION_MODE": str(validation_mode or "live_cross_source"),
        "VALIDATION_HELD_OUT_RECORDS_PATH": (
            str(validation_held_out_records_path)
            if validation_held_out_records_path
            else ""
        ),
        "ALLOW_INCOMPATIBLE_VALIDATION_RECORDS": (
            "true" if allow_incompatible_validation_records else "false"
        ),
        "DIRECT_COLLECTION_ENABLE_AUDIT_VALIDATION": (
            "true" if direct_collection_enable_audit_validation else "false"
        ),
        "HUMAN_REVIEW_ENABLED": "true" if human_review_enabled else "false",
        "HUMAN_REVIEW_DECISIONS_PATH": (
            str(human_review_decisions_path) if human_review_decisions_path else ""
        ),
        "HUMAN_REVIEW_APPLY_DECISIONS": (
            "true" if human_review_apply_decisions else "false"
        ),
        "HUMAN_REVIEW_REQUIRE_REVIEWER_ID": (
            "true" if human_review_require_reviewer_id else "false"
        ),
        "ANOMALY_MAX_CASES_THRESHOLD": str(anomaly_max_cases_threshold),
        "ANOMALY_MAX_DEATHS_THRESHOLD": str(anomaly_max_deaths_threshold),
        "ANOMALY_SPIKE_MULTIPLIER": str(anomaly_spike_multiplier),
        "ANOMALY_MIN_PRIOR_RECORDS": str(anomaly_min_prior_records),
    }
    if llm_source_critic_max_sources is not None:
        env["LLM_SOURCE_CRITIC_MAX_SOURCES"] = str(
            llm_source_critic_max_sources
        )
    if llm_source_critic_max_failures is not None:
        env["LLM_SOURCE_CRITIC_MAX_FAILURES"] = str(
            llm_source_critic_max_failures
        )
    if llm_source_critic_max_node_seconds is not None:
        env["LLM_SOURCE_CRITIC_MAX_NODE_SECONDS"] = str(
            llm_source_critic_max_node_seconds
        )
    if llm_source_credibility_max_sources is not None:
        env["LLM_SOURCE_CREDIBILITY_MAX_SOURCES"] = str(
            llm_source_credibility_max_sources
        )
    if llm_source_credibility_source_id_allowlist:
        env["LLM_SOURCE_CREDIBILITY_SOURCE_ID_ALLOWLIST"] = ",".join(
            llm_source_credibility_source_id_allowlist
        )
    if llm_source_identity_max_sources is not None:
        env["LLM_SOURCE_IDENTITY_MAX_SOURCES"] = str(
            llm_source_identity_max_sources
        )
    resolved_provider, resolved_model = resolve_llm_selection(provider, model)
    env["LLM_PROVIDER"] = resolved_provider
    # Clear a stale process model when the selected provider has no model.
    env["LLM_MODEL"] = resolved_model
    return env


def default_workflow_run_config() -> dict:
    """Return built-in workflow runtime settings used when config keys are omitted."""

    return {
        "profile_name": DEFAULT_PROFILE_NAME,
        "pipeline_mode": PIPELINE_MODE,
        "description": (
            "Runtime profile for the Data Collection Workflow. It controls graph input, "
            "source overlays, live fetch, LLM stages, source roles, and outputs."
        ),
        "workflow": {
            "graph_name": "data_collection_workflow",
            "collection_mode": "standard",
            "seed_source_overlay_path": None,
            "source_role_policy_overlay_path": None,
            "use_fixture_documents": False,
        },
        "user_request": DEFAULT_USER_REQUEST,
        "structured_task": deepcopy(DEFAULT_STRUCTURED_TASK),
        "studio": {
            "port": None,
            "no_reload": True,
        },
        "live_web": {
            "enabled": True,
            "timeout_seconds": 30,
        },
        "source_search": {
            "enabled": True,
            "mode": "live",
            "provider": "tavily",
            "fixture_path": str(DEFAULT_SEARCH_FIXTURE_PATH.relative_to(PROJECT_ROOT)),
            "max_queries": 8,
            "max_results_per_query": 8,
            "max_total_results": 64,
            "timeout_seconds": 15,
            "combine_with_seed_catalog": False,
            "provider_channel_allowlist": list(
                DEFAULT_SEARCH_PROVIDER_CHANNEL_ALLOWLIST
            ),
            "iterative": {
                "enabled": True,
                "max_iterations": 3,
                "max_queries_per_iteration": 4,
                "max_total_queries": 20,
                "max_total_results": 120,
                "require_llm": True,
                "allow_deterministic_fallback": False,
                "stop_when_llm_says_sufficient": True,
                "require_observation_after_each_iteration": True,
            },
            "authority_gap_retry": {
                "enabled": True,
                    "max_queries": 24,
                    "known_domain_max_queries": 10,
                    "jurisdiction_max_queries": 12,
                    "official_page_family_max_queries": 10,
                    "result_budget": 80,
                "max_iterations": 1,
            },
        },
        "content_fetch": {
            "fetch_search_derived_sources": True,
            "max_search_derived_sources": 18,
            "max_total_sources": 18,
            "min_credibility_score": 0.55,
            "allowed_final_roles": [
                "collection",
                "validation",
                "collection_support",
                "context",
            ],
            "allow_needs_review": True,
            "domain_allowlist": [],
            "domain_blocklist": [],
            "max_bytes": 5_000_000,
            "parse_pdf_text": True,
            "parse_tables": True,
            "store_raw_text": False,
            "user_agent": "data-collection-workflow/0.1",
            "content_fixture_map_path": None,
            "external_fetch": {
                "enabled": True,
                "provider_order": ["tavily_extract", "native_requests"],
                "tavily_extract": {
                    "format": "markdown",
                    "extract_depth": "advanced",
                    "timeout_seconds": 45,
                    "chunks_per_source": 5,
                },
                "adaptive_budget": {
                    "max_candidate_urls": 120,
                    "max_fetch_urls": 18,
                    "min_usable_documents": 12,
                    "min_collection_sources": 6,
                    "min_validation_sources": 3,
                    "max_iterations": 3,
                    "stop_when_llm_says_sufficient": True,
                },
            },
        },
        "llm": {
            "provider": "",
            "model": DEFAULT_MODEL,
            "source_planning_enabled": True,
            "source_critic_enabled": True,
            "structured_extraction_enabled": True,
            "max_chunks": 30,
            "max_tokens": 4096,
            "extraction": {
                "focused_recovery_reserved_calls": 40,
                "max_domain_call_share": 0.20,
                "high_value_source_min_spans": 2,
                "event_source_min_spans": 1,
                "scheduler": {
                    "mode": "quality_adaptive",
                    "max_concurrency": 4,
                    "soft_checkpoint_calls": 400,
                    "safety_max_calls": 2400,
                    "rolling_yield_window": 40,
                },
            },
            "focused_recovery": {
                "max_calls": 40,
                "max_sources": 30,
            },
            "fallback_to_rule_based": False,
            "source_critic": {
                "max_sources": 30,
                "review_blocks_fetch": False,
            },
            "source_credibility": {
                "enabled": False,
                "max_sources": 6,
                "source_id_allowlist": [],
            },
            "source_identity": {
                "enabled": True,
                "max_sources": 30,
                "post_fetch": True,
                "require_llm": True,
                "allow_deterministic_fallback": False,
            },
            "must_fetch_min_chunks_per_source": 6,
            "official_extraction_max_chunks": 30,
        },
        "disease_intelligence": {
            "llm_enabled": True,
            "force_llm": True,
            "fallback_to_curated": True,
        },
        "anomaly_detection": {
            "enabled": True,
            "max_cases_threshold": 1000000,
            "max_deaths_threshold": 100000,
            "spike_multiplier": 10,
            "min_prior_records": 1,
        },
        "human_review": {
            "enabled": False,
            "decisions_path": None,
            "apply_decisions": False,
            "require_reviewer_id": True,
        },
        "validation": {
            "mode": "live_cross_source",
            "held_out_records_path": None,
            "prefer_authoritative_sources": True,
            "min_independent_validation_sources": 1,
            "allow_incompatible_validation_records": False,
        },
        "source_sets": {
            "source_id_allowlist_enabled": False,
            "collection_source_ids": list(COLLECTION_SOURCE_IDS),
            "context_source_ids": list(CONTEXT_SOURCE_IDS),
            "validation_reserved_source_ids": list(VALIDATION_SOURCE_IDS),
            "workflow_source_ids": [],
            "llm_source_critic_source_ids": [],
        },
        "output": {
            "run_output_root": str(DEFAULT_OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
            "sessionized": True,
            "session_id": None,
            "auto_build_console": True,
            "console_output_root": str(
                DEFAULT_CONSOLE_OUTPUT_ROOT.relative_to(PROJECT_ROOT)
            ),
            "write_latest_alias": True,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _map_legacy_scheduler_fields(config: dict, loaded: dict) -> None:
    loaded_llm = loaded.get("llm") if isinstance(loaded.get("llm"), dict) else {}
    loaded_extraction = (
        loaded_llm.get("extraction")
        if isinstance(loaded_llm.get("extraction"), dict)
        else {}
    )
    loaded_scheduler = (
        loaded_extraction.get("scheduler")
        if isinstance(loaded_extraction.get("scheduler"), dict)
        else {}
    )
    scheduler = config.setdefault("llm", {}).setdefault("extraction", {}).setdefault(
        "scheduler", {}
    )
    if (
        "soft_checkpoint_calls" not in loaded_scheduler
        and "soft_primary_calls" in loaded_extraction
    ):
        scheduler["soft_checkpoint_calls"] = loaded_extraction["soft_primary_calls"]
    if (
        "safety_max_calls" not in loaded_scheduler
        and "hard_primary_calls" in loaded_extraction
    ):
        scheduler["safety_max_calls"] = loaded_extraction["hard_primary_calls"]


def _strip_jsonc_comments(text: str) -> str:
    """Remove JSONC comments while preserving quoted string content."""

    result: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue

        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue

        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (
                text[index] == "*" and text[index + 1] == "/"
            ):
                index += 1
            index += 2
            continue

        result.append(char)
        index += 1
    return "".join(result)


def _load_json_or_jsonc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonc":
        text = _strip_jsonc_comments(text)
    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Workflow run config must be a JSON object: {path}")
    return loaded


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a workflow runtime config path relative to the project root."""

    path = Path(config_path) if config_path else DEFAULT_WORKFLOW_RUN_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path: str | Path | None = None) -> dict:
    """Load workflow runtime settings from JSON, with stable built-in defaults."""

    load_project_env()
    path = resolve_config_path(config_path)
    config = default_workflow_run_config()
    if path.exists():
        loaded = _load_json_or_jsonc(path)
        config = _deep_merge(config, loaded)
        _map_legacy_scheduler_fields(config, loaded)
        if path != DEFAULT_WORKFLOW_RUN_CONFIG_PATH:
            workflow_loaded = (
                loaded.get("workflow") if isinstance(loaded.get("workflow"), dict) else {}
            )
            if "seed_source_overlay_path" not in workflow_loaded:
                config.setdefault("workflow", {})["seed_source_overlay_path"] = None
            if "source_role_policy_overlay_path" not in workflow_loaded:
                config.setdefault("workflow", {})[
                    "source_role_policy_overlay_path"
                ] = None

            source_sets_loaded = (
                loaded.get("source_sets")
                if isinstance(loaded.get("source_sets"), dict)
                else {}
            )
            if not source_sets_loaded:
                config["source_sets"] = {
                    "source_id_allowlist_enabled": False,
                    "collection_source_ids": [],
                    "context_source_ids": [],
                    "validation_reserved_source_ids": [],
                    "workflow_source_ids": [],
                    "llm_source_critic_source_ids": [],
                }
            elif "workflow_source_ids" not in source_sets_loaded:
                config.setdefault("source_sets", {})["workflow_source_ids"] = []
    elif config_path:
        raise FileNotFoundError(f"Workflow run config not found: {path}")
    # Materialize inherited generation controls before session fingerprinting.
    llm = config.setdefault("llm", {})
    llm["provider"], llm["model"] = resolve_llm_selection(
        llm.get("provider"), llm.get("model")
    )
    for key, env_name, default in (
        ("thinking", "LLM_THINKING", "auto"),
        ("effort", "LLM_EFFORT", None),
        ("structured_output_method", "LLM_STRUCTURED_OUTPUT_METHOD", "auto"),
    ):
        if key == "effort" and key in llm:
            continue  # An explicit null clears inherited effort.
        llm[key] = llm.get(key) or (get_env(env_name) or "").strip() or default
    return config


def _resolve_project_path(
    value: str | Path | None,
    default: Path | None,
) -> Path | None:
    if not value:
        return default
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def workflow_run_env_from_config(config: dict) -> dict[str, str]:
    """Build workflow environment variables from a runtime profile."""

    workflow = config.get("workflow") or {}
    live_web = config.get("live_web") or {}
    llm = config.get("llm") or {}
    disease_intelligence = config.get("disease_intelligence") or {}
    source_search = config.get("source_search") or {}
    source_search_iterative = source_search.get("iterative") or {}
    source_search_authority_retry = source_search.get("authority_gap_retry") or {}
    content_fetch = config.get("content_fetch") or {}
    external_fetch = content_fetch.get("external_fetch") or {}
    tavily_extract = external_fetch.get("tavily_extract") or {}
    human_review = config.get("human_review") or {}
    validation = config.get("validation") or {}
    anomaly_detection = config.get("anomaly_detection") or {}
    source_critic = llm.get("source_critic") or {}
    source_credibility = llm.get("source_credibility") or {}
    source_identity = llm.get("source_identity") or {}
    focused_recovery = llm.get("focused_recovery") or {}
    extraction = llm.get("extraction") or {}
    scheduler = extraction.get("scheduler") or {}
    source_sets = config.get("source_sets") or {}
    live_fetch_enabled = bool(live_web.get("enabled", True))
    llm_source_planning_enabled = bool(llm.get("source_planning_enabled", True))
    llm_source_critic_enabled = bool(llm.get("source_critic_enabled", True))
    llm_extraction_enabled = bool(llm.get("structured_extraction_enabled", True))
    llm_source_credibility_enabled = bool(source_credibility.get("enabled", False))
    llm_source_identity_enabled = bool(source_identity.get("enabled", False))
    llm_source_identity_require_llm = bool(
        source_identity.get("require_llm", False)
    )
    llm_source_identity_allow_fallback = bool(
        source_identity.get("allow_deterministic_fallback", True)
    )
    llm_disease_intelligence_enabled = bool(
        disease_intelligence.get("llm_enabled", False)
    )
    disease_intelligence_force_llm = bool(
        disease_intelligence.get("force_llm", False)
    )
    disease_intelligence_fallback_to_curated = bool(
        disease_intelligence.get("fallback_to_curated", True)
    )
    offline_deterministic_run = (
        not live_fetch_enabled
        and not llm_source_planning_enabled
        and not llm_source_critic_enabled
        and not llm_extraction_enabled
        and not llm_source_credibility_enabled
    )
    if offline_deterministic_run:
        llm_disease_intelligence_enabled = False
        disease_intelligence_force_llm = False
        disease_intelligence_fallback_to_curated = True
        llm_source_identity_enabled = False
        llm_source_identity_require_llm = False
        llm_source_identity_allow_fallback = True
    human_review_enabled = bool(human_review.get("enabled", True))
    if human_review.get("decisions_path") and human_review.get("apply_decisions"):
        human_review_enabled = True
    search_enabled = bool(source_search.get("enabled", False))
    search_mode = str(source_search.get("mode") or "disabled").strip().lower()
    if not search_enabled:
        search_mode = "disabled"
    live_search_enabled = (
        search_enabled
        and search_mode == "live"
        and bool(source_search.get("live_search_enabled", True))
    )
    iterative_enabled = bool(source_search_iterative.get("enabled", False)) and (
        search_mode == "live"
    )
    return workflow_run_env(
        pipeline_mode=str(
            config.get("pipeline_mode") or PIPELINE_MODE
        ),
        collection_mode=workflow.get("collection_mode", "standard"),
        seed_source_overlay_path=_resolve_project_path(
            workflow.get("seed_source_overlay_path"), None
        ),
        source_role_policy_overlay_path=_resolve_project_path(
            workflow.get("source_role_policy_overlay_path"),
            None,
        ),
        use_fixture_documents=bool(workflow.get("use_fixture_documents", False)),
        live_fetch=live_fetch_enabled,
        llm_source_planning=llm_source_planning_enabled,
        llm_source_critic=llm_source_critic_enabled,
        llm_source_credibility=llm_source_credibility_enabled,
        llm_source_identity=llm_source_identity_enabled,
        llm_extraction=llm_extraction_enabled,
        llm_disease_intelligence=llm_disease_intelligence_enabled,
        disease_intelligence_force_llm=disease_intelligence_force_llm,
        disease_intelligence_fallback_to_curated=(
            disease_intelligence_fallback_to_curated
        ),
        provider=llm.get("provider"),
        model=llm.get("model"),
        timeout_seconds=float(live_web.get("timeout_seconds", 30)),
        llm_max_chunks=int(llm.get("max_chunks", 8)),
        llm_extraction_scheduler_mode=str(
            scheduler.get("mode", "quality_adaptive")
        ),
        llm_extraction_max_concurrency=int(
            scheduler.get("max_concurrency", 4)
        ),
        llm_extraction_soft_checkpoint_calls=int(
            scheduler.get(
                "soft_checkpoint_calls",
                extraction.get("soft_primary_calls", llm.get("max_chunks", 400)),
            )
        ),
        llm_extraction_safety_max_calls=int(
            scheduler.get(
                "safety_max_calls", extraction.get("hard_primary_calls", 2400)
            )
        ),
        llm_extraction_rolling_yield_window=int(
            scheduler.get("rolling_yield_window", 40)
        ),
        llm_soft_primary_calls=int(
            extraction.get("soft_primary_calls", llm.get("max_chunks", 8))
        ),
        llm_hard_primary_calls=int(
            extraction.get(
                "hard_primary_calls",
                extraction.get("soft_primary_calls", llm.get("max_chunks", 8)),
            )
        ),
        llm_focused_recovery_reserved_calls=int(
            extraction.get("focused_recovery_reserved_calls", 120)
        ),
        llm_max_domain_call_share=float(
            extraction.get("max_domain_call_share", 0.20)
        ),
        llm_high_value_source_min_spans=int(
            extraction.get("high_value_source_min_spans", 2)
        ),
        llm_event_source_min_spans=int(
            extraction.get("event_source_min_spans", 1)
        ),
        llm_must_fetch_min_chunks_per_source=int(
            llm.get("must_fetch_min_chunks_per_source", 6)
        ),
        llm_official_extraction_max_chunks=int(
            llm.get("official_extraction_max_chunks", 30)
        ),
        llm_focused_recovery_max_calls=int(
            focused_recovery.get("max_calls", 40)
        ),
        llm_focused_recovery_max_sources=int(
            focused_recovery.get("max_sources", 30)
        ),
        llm_source_critic_max_sources=source_critic.get("max_sources", 6),
        llm_source_critic_max_failures=source_critic.get("max_failures"),
        llm_source_critic_max_node_seconds=source_critic.get("max_node_seconds"),
        llm_source_critic_review_blocks_fetch=bool(
            source_critic.get("review_blocks_fetch", True)
        ),
        llm_source_credibility_max_sources=source_credibility.get("max_sources"),
        llm_source_credibility_source_id_allowlist=source_credibility.get(
            "source_id_allowlist"
        ),
        llm_source_identity_max_sources=source_identity.get("max_sources", 6),
        llm_source_identity_post_fetch=bool(source_identity.get("post_fetch", True)),
        llm_source_identity_require_llm=llm_source_identity_require_llm,
        llm_source_identity_allow_deterministic_fallback=(
            llm_source_identity_allow_fallback
        ),
        llm_max_tokens=int(llm.get("max_tokens", 4096)),
        llm_timeout_seconds=(
            float(llm.get("timeout_seconds"))
            if llm.get("timeout_seconds") is not None
            else None
        ),
        llm_max_retries=int(llm.get("max_retries", 2)),
        llm_thinking=llm.get("thinking"),
        llm_effort=(llm.get("effort") or "") if "effort" in llm else None,
        llm_structured_output_method=llm.get("structured_output_method"),
        fallback_to_rule_based=bool(llm.get("fallback_to_rule_based", False)),
        source_id_allowlist=source_sets.get("workflow_source_ids"),
        source_id_allowlist_enabled=bool(
            source_sets.get("source_id_allowlist_enabled", False)
        ),
        llm_source_critic_source_id_allowlist=source_sets.get(
            "llm_source_critic_source_ids"
        ),
        live_search=live_search_enabled,
        search_mode=search_mode,
        search_provider=str(source_search.get("provider") or "tavily"),
        search_fixture_path=_resolve_project_path(
            source_search.get("fixture_path"),
            DEFAULT_SEARCH_FIXTURE_PATH,
        ),
        search_max_queries=int(source_search.get("max_queries", 3)),
        search_max_results_per_query=int(
            source_search.get("max_results_per_query", 5)
        ),
        search_max_total_results=int(source_search.get("max_total_results", 15)),
        search_timeout_seconds=float(source_search.get("timeout_seconds", 15)),
        search_combine_with_seed_catalog=bool(
            source_search.get("combine_with_seed_catalog", False)
        ),
        search_provider_channel_allowlist=source_search.get(
            "provider_channel_allowlist"
        )
        or DEFAULT_SEARCH_PROVIDER_CHANNEL_ALLOWLIST,
        iterative_source_discovery=iterative_enabled,
        iterative_search_max_iterations=int(
            source_search_iterative.get("max_iterations", 3)
        ),
        iterative_search_max_queries_per_iteration=int(
            source_search_iterative.get("max_queries_per_iteration", 3)
        ),
        iterative_search_max_total_queries=int(
            source_search_iterative.get("max_total_queries", 9)
        ),
        iterative_search_max_total_results=int(
            source_search_iterative.get("max_total_results", 30)
        ),
        iterative_search_require_llm=bool(
            source_search_iterative.get("require_llm", True)
        ),
        iterative_search_allow_deterministic_fallback=bool(
            source_search_iterative.get("allow_deterministic_fallback", False)
        ),
        iterative_search_stop_when_llm_says_sufficient=bool(
            source_search_iterative.get("stop_when_llm_says_sufficient", True)
        ),
        iterative_search_require_observation_after_each_iteration=bool(
            source_search_iterative.get(
                "require_observation_after_each_iteration", True
            )
        ),
        authority_gap_retry_enabled=bool(
            source_search_authority_retry.get("enabled", True)
        ),
        authority_gap_retry_max_queries=int(
            source_search_authority_retry.get("max_queries", 6)
        ),
        authority_gap_known_domain_retry_max_queries=int(
            source_search_authority_retry.get("known_domain_max_queries", 4)
        ),
        authority_gap_jurisdiction_retry_max_queries=int(
            source_search_authority_retry.get("jurisdiction_max_queries", 6)
        ),
        authority_gap_official_page_family_retry_max_queries=int(
            source_search_authority_retry.get("official_page_family_max_queries", 3)
        ),
        authority_gap_retry_result_budget=int(
            source_search_authority_retry.get("result_budget", 32)
        ),
        authority_gap_retry_max_iterations=int(
            source_search_authority_retry.get("max_iterations", 1)
        ),
        fetch_search_derived_sources=bool(
            content_fetch.get("fetch_search_derived_sources", False)
        ),
        fetch_max_search_derived_sources=int(
            content_fetch.get("max_search_derived_sources", 2)
        ),
        fetch_max_total_sources=int(content_fetch.get("max_total_sources", 10)),
        fetch_min_credibility_score=float(
            content_fetch.get("min_credibility_score", 0.55)
        ),
        fetch_allowed_final_roles=content_fetch.get("allowed_final_roles")
        or ["collection", "validation", "collection_support", "context"],
        fetch_allow_needs_review=bool(
            content_fetch.get("allow_needs_review", False)
            or not human_review_enabled
        ),
        fetch_domain_allowlist=content_fetch.get("domain_allowlist") or [],
        fetch_domain_blocklist=content_fetch.get("domain_blocklist") or [],
        fetch_max_bytes=int(content_fetch.get("max_bytes", 1_000_000)),
        fetch_parse_pdf_text=bool(content_fetch.get("parse_pdf_text", True)),
        fetch_parse_tables=bool(content_fetch.get("parse_tables", True)),
        fetch_store_raw_text=bool(content_fetch.get("store_raw_text", False)),
        fetch_user_agent=str(
            content_fetch.get("user_agent") or "data-collection-workflow/0.1"
        ),
        fetch_max_node_seconds=content_fetch.get("max_node_seconds"),
        content_fixture_map_path=_resolve_project_path(
            content_fetch.get("content_fixture_map_path"),
            PROJECT_ROOT,
        )
        if content_fetch.get("content_fixture_map_path")
        else None,
        external_fetch_enabled=bool(external_fetch.get("enabled", True)),
        external_fetch_provider_order=external_fetch.get("provider_order")
        or ["tavily_extract", "native_requests"],
        tavily_extract_format=str(tavily_extract.get("format") or "markdown"),
        tavily_extract_depth=str(tavily_extract.get("extract_depth") or "advanced"),
        tavily_extract_timeout_seconds=float(
            tavily_extract.get("timeout_seconds", 45)
        ),
        tavily_extract_chunks_per_source=int(
            tavily_extract.get("chunks_per_source", 5)
        ),
        human_review_enabled=human_review_enabled,
        human_review_decisions_path=_resolve_project_path(
            human_review.get("decisions_path")
            or config.get("human_review_decisions_path"),
            PROJECT_ROOT,
        )
        if (human_review.get("decisions_path") or config.get("human_review_decisions_path"))
        else None,
        human_review_apply_decisions=bool(
            human_review.get("apply_decisions", False)
        ),
        human_review_require_reviewer_id=bool(
            human_review.get("require_reviewer_id", True)
        ),
        validation_mode=str(validation.get("mode") or "live_cross_source"),
        validation_held_out_records_path=_resolve_project_path(
            validation.get("held_out_records_path"),
            PROJECT_ROOT,
        )
        if validation.get("held_out_records_path")
        else None,
        allow_incompatible_validation_records=bool(
            validation.get("allow_incompatible_validation_records", False)
        ),
        direct_collection_enable_audit_validation=bool(
            config.get("direct_collection_enable_audit_validation")
            or validation.get("direct_collection_enable_audit_validation", False)
        ),
        anomaly_max_cases_threshold=float(
            anomaly_detection.get("max_cases_threshold", 1_000_000)
        ),
        anomaly_max_deaths_threshold=float(
            anomaly_detection.get("max_deaths_threshold", 100_000)
        ),
        anomaly_spike_multiplier=float(
            anomaly_detection.get("spike_multiplier", 10)
        ),
        anomaly_min_prior_records=int(
            anomaly_detection.get("min_prior_records", 1)
        ),
    )


def validate_workflow_task(config: dict) -> None:
    """Require explicit task scope before provider setup or workflow execution."""

    task = config.get("structured_task")
    if task is not None and not isinstance(task, dict):
        raise ValueError("workflow structured_task must be a JSON object.")
    task = task or {}
    required = ("disease", "location", "start_date", "end_date")
    missing = [
        field for field in required
        if not isinstance(task.get(field), str) or not task[field].strip()
    ]
    if missing:
        raise ValueError(
            "workflow structured_task requires non-empty fields: " + ", ".join(missing)
            + ". user_request supplies supplementary instructions only."
        )


def structured_task_from_config(config: dict) -> dict:
    """Return the structured task payload for graph initial state."""

    raw = deepcopy(config.get("structured_task"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("workflow structured_task must be a JSON object.")
    if not raw:
        return {}

    raw.setdefault("user_request", config.get("user_request") or DEFAULT_USER_REQUEST)
    raw.setdefault("run_label", config.get("profile_name") or DEFAULT_PROFILE_NAME)

    workflow = config.get("workflow") or {}
    collection_mode = workflow.get("collection_mode")
    if collection_mode:
        raw.setdefault("collection_mode", collection_mode)

    return {
        key: value
        for key, value in raw.items()
        if value not in (None, "", [], {})
    }


def workflow_run_config_with_overrides(
    config: dict,
    *,
    live_fetch: bool | None = None,
    all_llm: bool | None = None,
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    llm_max_chunks: int | None = None,
    port: int | None = None,
    output_dir: str | Path | None = None,
    session_id: str | None = None,
    user_request: str | None = None,
) -> dict:
    """Return a config copy with one-run command-line overrides applied."""

    updated = deepcopy(config)
    updated.setdefault("live_web", {})
    updated.setdefault("llm", {})
    updated.setdefault("studio", {})
    updated.setdefault("output", {})

    if live_fetch is not None:
        updated["live_web"]["enabled"] = bool(live_fetch)
    if all_llm is not None:
        updated["llm"]["source_planning_enabled"] = bool(all_llm)
        updated["llm"]["source_critic_enabled"] = bool(all_llm)
        updated["llm"]["structured_extraction_enabled"] = bool(all_llm)
        if not all_llm:
            updated["llm"].setdefault("source_credibility", {})["enabled"] = False
        source_identity = updated["llm"].setdefault("source_identity", {})
        source_identity["enabled"] = bool(all_llm)
        source_identity["require_llm"] = bool(all_llm)
        source_identity["allow_deterministic_fallback"] = not bool(all_llm)
        disease_intelligence = updated.setdefault("disease_intelligence", {})
        disease_intelligence["llm_enabled"] = bool(all_llm)
        disease_intelligence["force_llm"] = bool(all_llm)
        disease_intelligence["fallback_to_curated"] = True
    provider_override = str(provider or "").strip().lower()
    model_override = str(model or "").strip()
    current_provider, _ = resolve_llm_selection(
        updated["llm"].get("provider"), updated["llm"].get("model")
    )
    if provider_override:
        if provider_override != current_provider and not model_override:
            updated["llm"]["model"] = ""
        updated["llm"]["provider"] = provider_override
    if model_override:
        updated["llm"]["model"] = model_override
    if timeout_seconds is not None:
        updated["live_web"]["timeout_seconds"] = timeout_seconds
    if llm_max_chunks is not None:
        updated["llm"]["max_chunks"] = llm_max_chunks
    if port is not None:
        updated["studio"]["port"] = port
    if output_dir is not None:
        updated["output"]["run_output_root"] = str(output_dir)
        if not bool(updated["output"].get("sessionized", True)):
            updated["output"]["run_output_dir"] = str(output_dir)
    if session_id:
        updated["output"]["session_id"] = session_id
    if user_request:
        updated["user_request"] = user_request
        if isinstance(updated.get("structured_task"), dict) and updated["structured_task"]:
            updated["structured_task"]["user_request"] = user_request
    updated["llm"]["provider"], updated["llm"]["model"] = resolve_llm_selection(
        updated["llm"].get("provider"), updated["llm"].get("model")
    )
    return updated


def workflow_session_id(now: datetime | None = None) -> str:
    """Return a timestamped workflow session id."""

    timestamp = now or datetime.now(timezone.utc)
    return timestamp.strftime("%Y%m%d_%H%M%S_utc")


def workflow_output_root_from_config(config: dict) -> Path:
    """Resolve the workflow run output root from the centralized config."""

    output = config.get("output") or {}
    root_value = (
        output.get("run_output_root")
        or output.get("run_output_dir")
        or DEFAULT_OUTPUT_ROOT
    )
    root = Path(root_value)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def workflow_output_dir_from_config(
    config: dict,
    *,
    session_id: str | None = None,
) -> Path:
    """Resolve the run output directory from the centralized config."""

    output = config.get("output") or {}
    if bool(output.get("sessionized", True)):
        resolved_session_id = (
            session_id
            or output.get("session_id")
            or workflow_session_id()
        )
        return workflow_output_root_from_config(config) / "sessions" / str(
            resolved_session_id
        )

    output_dir = Path(
        output.get("run_output_dir")
        or output.get("run_output_root")
        or DEFAULT_OUTPUT_DIR
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    return output_dir


def workflow_console_output_dir_from_config(
    config: dict,
    *,
    run_output_dir: Path | None = None,
    session_id: str | None = None,
) -> Path:
    """Resolve where the HTML workflow console should be written."""

    output = config.get("output") or {}
    if bool(output.get("sessionized", True)) and run_output_dir is not None:
        return run_output_dir / "workflow_console"

    console_root = Path(output.get("console_output_root") or DEFAULT_CONSOLE_OUTPUT_ROOT)
    if not console_root.is_absolute():
        console_root = PROJECT_ROOT / console_root
    if session_id:
        return console_root / "sessions" / session_id
    return console_root


def validation_records_path_from_config(config: dict) -> Path | None:
    """Resolve an explicitly configured held-out validation CSV path.

    Live cross-source validation is the default and intentionally has no local
    held-out CSV. A profile may explicitly provide an independent reference file.
    """

    validation = config.get("validation") or {}
    workflow = config.get("workflow") or {}
    raw_path = (
        validation.get("held_out_records_path")
        or validation.get("validation_records_path")
        or workflow.get("validation_ground_truth_records_path")
    )
    if not raw_path:
        return None
    return _resolve_project_path(raw_path, PROJECT_ROOT)


def resolve_workflow_run_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the primary workflow runtime config path."""

    path = Path(config_path) if config_path else DEFAULT_WORKFLOW_RUN_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_workflow_run_config(config_path: str | Path | None = None) -> dict:
    """Load the primary workflow runtime profile."""

    return load_config(resolve_workflow_run_config_path(config_path))


def workflow_initial_state_from_config(
    config: dict,
    *,
    include_empty_fields: bool = True,
) -> dict:
    """Build the LangGraph Studio input payload from a runtime profile."""

    state = studio_initial_state(
        config.get("user_request") or DEFAULT_USER_REQUEST,
        structured_task=structured_task_from_config(config),
        include_empty_fields=include_empty_fields,
    )
    human_review = config.get("human_review") or {}
    validation = config.get("validation") or {}
    human_review_enabled = bool(human_review.get("enabled", True))
    if human_review.get("decisions_path") and human_review.get("apply_decisions"):
        human_review_enabled = True
    state["human_review_enabled"] = human_review_enabled
    state["validation_mode"] = str(validation.get("mode") or "live_cross_source")
    state["validation_config"] = {
        "mode": state["validation_mode"],
        "held_out_records_path": validation.get("held_out_records_path"),
        "prefer_authoritative_sources": bool(
            validation.get("prefer_authoritative_sources", True)
        ),
        "min_independent_validation_sources": int(
            validation.get("min_independent_validation_sources", 1)
        ),
    }
    decisions_path = human_review.get("decisions_path") or config.get(
        "human_review_decisions_path"
    )
    if decisions_path:
        state["human_review_decisions_path"] = str(
            _resolve_project_path(decisions_path, PROJECT_ROOT)
        )
    return state


def studio_initial_state(
    user_request: str | None = None,
    *,
    structured_task: dict | None = None,
    include_empty_fields: bool = True,
) -> dict:
    """Build the state payload a user can submit in LangGraph Studio."""

    state = {"user_request": user_request or DEFAULT_USER_REQUEST}
    if structured_task is not None:
        state["structured_task"] = deepcopy(structured_task)
    if not include_empty_fields:
        return state
    state.update(
        {
            "source_candidates": [],
            "source_search_results": [],
            "source_search_execution_summary": None,
            "source_discovery_summary": None,
            "iterative_source_discovery_summary": None,
            "search_iteration_plans": [],
            "search_iteration_observations": [],
            "search_refinement_decisions": [],
            "iterative_search_queries": [],
            "source_registry": [],
            "source_registry_summary": None,
            "source_screening_summary": None,
            "source_critic_summary": None,
            "source_routing_summary": None,
            "source_credibility_assessments": [],
            "source_credibility_summary": None,
            "documents": [],
            "evidence_chunks": [],
            "raw_records": [],
            "validated_records": [],
            "normalized_records": [],
            "linked_events": [],
            "event_clusters": [],
            "duplicate_clusters": [],
            "validation_records": [],
            "active_validation_records": [],
            "inactive_validation_records": [],
            "validation_source_compatibility_summary": None,
            "validation_cases": [],
            "validation_comparisons": [],
            "validation_results": [],
            "validation_summary": None,
            "trusted_source_validation_summary": None,
            "cross_source_validation_summary": None,
            "anomaly_results": [],
            "anomaly_summary": None,
            "anomaly_review_items": [],
            "conflicts": [],
            "human_review_queue": [],
            "human_review_enabled": True,
            "human_review_decisions": [],
            "human_review_decisions_path": None,
            "applied_human_review_decisions": [],
            "rejected_human_review_decisions": [],
            "human_review_audit_trail": [],
            "human_review_application_summary": None,
            "final_dataset_post_review": [],
            "records_excluded_by_human_review": [],
            "collection_trace": [],
            "collection_spec": None,
            "task_intake_summary": None,
            "disease_intelligence": None,
            "disease_intelligence_summary": None,
            "disease_profile": None,
            "collection_schema": None,
            "source_strategy": None,
            "screening_criteria": None,
            "profile_schema_summary": None,
            "search_queries": None,
            "search_query_inventory": [],
            "agentic_source_plan": None,
            "executable_source_plan_summary": None,
            "source_planning_agent_summary": None,
            "content_fetch_requests": [],
            "content_fetch_summary": None,
            "document_parse_summary": None,
            "fetch_manifest": [],
            "fixture_document_summary": None,
            "document_quality_summary": None,
            "structured_extraction_summary": None,
            "llm_extraction_summary": None,
            "schema_validation_summary": None,
            "record_normalization_summary": None,
            "record_linking_summary": None,
            "event_clustering_summary": None,
            "duplicate_detection_summary": None,
            "cross_source_consistency_summary": None,
            "human_review_summary": None,
            "final_data_package": None,
            "finalization_summary": None,
            "current_route": None,
        }
    )
    return state


@contextmanager
def temporary_workflow_env(updates: dict[str, str]):
    """Temporarily apply workflow runtime environment values."""

    updates = normalize_env_updates(updates)
    keys = set(ENV_KEYS).union(updates)
    # Snapshot the process keys that this scope actually changes.
    original = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def api_key_present(provider: str) -> bool:
    """Return whether the configured provider has a key in the environment."""

    provider_name = (provider or "").strip().lower()
    if provider_name == "anthropic":
        return bool(get_env("ANTHROPIC_API_KEY"))
    if provider_name == "openai":
        return bool(get_env("OPENAI_API_KEY"))
    return False


def safe_env_for_display(env: dict[str, str]) -> dict[str, str]:
    """Hide secrets before printing an environment preview."""

    return {
        key: value
        for key, value in env.items()
        if "API_KEY" not in key and "TOKEN" not in key and "SECRET" not in key
    }
