"""Optional LLM advisory agent for source identity and publisher assessment."""

from __future__ import annotations

import json

from .. import llm_clients


_SYSTEM_PROMPT = """You are a source identity assessor for the data collection workflow.

You must not browse, fetch, search, open URLs, or invent facts. Use only the
provided source metadata and, when present, already fetched page identity text.

Keep these concepts separate:
- search_provider: the search service that returned the result, such as Tavily.
- search_result_source_raw: the raw source label returned by that search service.
- actual_publisher: the organization that appears to publish the target page.
- source_owner: the owning or responsible organization when inferable.
- upstream_source_mentions: organizations whose reporting appears to be reused,
  syndicated, summarized, or quoted.

Never set actual_publisher to Tavily merely because Tavily returned the result.
If publisher identity is uncertain, use unknown/low confidence and explain why.
Return one JSON object only."""


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _clean(value) -> str | None:
    text = str(value or "").strip()
    return text or None


_SOURCE_TYPE_ALLOWED = {
    "unknown",
    "official_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "international_public_health_agency",
    "academic_or_peer_reviewed_source",
    "structured_database",
    "hospital_or_health_system",
    "news_media",
    "secondary_aggregator",
    "social_media",
    "personal_blog_or_forum",
    "commercial_site",
    "search_endpoint",
    "background_fact_sheet",
    "public_health_context_page",
}

_CLAIM_SUPPORT_ALLOWED = {
    "primary_case_claim_support",
    "corroboration_support",
    "zero_case_statement_support",
    "exposure_monitoring_support",
    "context_only",
    "search_discovery_only",
    "not_task_relevant",
    "insufficient_information",
}

_FETCH_ALLOWED = {
    "fetch_for_extraction",
    "fetch_for_context",
    "fetch_only_after_review",
    "do_not_fetch",
    "already_fetched_review_only",
    "insufficient_information",
}

_EXTRACTION_ALLOWED = {
    "extract_primary_case_claims",
    "extract_public_health_observations",
    "extract_context_only",
    "do_not_extract",
    "needs_human_review",
    "insufficient_information",
}


_LEGACY_VALUE_MAP = {
    "academic_literature": "academic_or_peer_reviewed_source",
    "zero_case_support": "zero_case_statement_support",
    "not_claim_support": "not_task_relevant",
    "extract_primary_claims": "extract_primary_case_claims",
    "extract_secondary_claims": "extract_public_health_observations",
    "fetch_for_validation_context": "fetch_for_context",
    "extract_validation_context": "extract_context_only",
    "unknown": "insufficient_information",
}


def _choice(value, allowed: set[str], default: str = "insufficient_information") -> str:
    text = str(value or "").strip().lower()
    text = _LEGACY_VALUE_MAP.get(text, text)
    return text if text in allowed else default


def _normalize_decision(raw: dict, source_entry: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected source identity JSON object, got {type(raw)!r}.")
    warnings = _as_str_list(raw.get("warnings"))
    errors = _as_str_list(raw.get("errors"))
    return {
        "source_id": _clean(raw.get("source_id"))
        or _clean(source_entry.get("source_id")),
        "actual_publisher": _clean(raw.get("actual_publisher")),
        "actual_publisher_normalized": _clean(
            raw.get("actual_publisher_normalized")
        ),
        "actual_publisher_confidence": _clean(
            raw.get("actual_publisher_confidence")
        ),
        "publisher_evidence_fields": _as_str_list(
            raw.get("publisher_evidence_fields")
        ),
        "publisher_evidence_quotes": _as_str_list(
            raw.get("publisher_evidence_quotes")
        ),
        "publisher_source": _clean(raw.get("publisher_source")),
        "source_owner": _clean(raw.get("source_owner")),
        "source_owner_confidence": _clean(raw.get("source_owner_confidence")),
        "source_type_llm": _clean(raw.get("source_type_llm"))
        or _clean(raw.get("source_type_final")),
        "source_type_final": _choice(
            raw.get("source_type_final"), _SOURCE_TYPE_ALLOWED, default="unknown"
        ),
        "source_type_confidence": _clean(raw.get("source_type_confidence")),
        "source_type_evidence": _as_str_list(raw.get("source_type_evidence")),
        "task_relevance_assessment": _clean(raw.get("task_relevance_assessment")),
        "disease_relevance_assessment": _clean(
            raw.get("disease_relevance_assessment")
        ),
        "geography_relevance_assessment": _clean(
            raw.get("geography_relevance_assessment")
        ),
        "time_relevance_assessment": _clean(raw.get("time_relevance_assessment")),
        "likely_contains_extractable_data": bool(
            raw.get("likely_contains_extractable_data", False)
        ),
        "supports_primary_case_claims": bool(
            raw.get("supports_primary_case_claims", False)
        ),
        "supports_zero_case_claims": bool(raw.get("supports_zero_case_claims", False)),
        "supports_exposure_monitoring_claims": bool(
            raw.get("supports_exposure_monitoring_claims", False)
        ),
        "supports_context_only": bool(raw.get("supports_context_only", False)),
        "claim_support_role": _choice(
            raw.get("claim_support_role"), _CLAIM_SUPPORT_ALLOWED
        ),
        "recommended_source_role": _clean(raw.get("recommended_source_role")),
        "recommended_fetch_use": _choice(raw.get("recommended_fetch_use"), _FETCH_ALLOWED),
        "recommended_extraction_use": _choice(
            raw.get("recommended_extraction_use"), _EXTRACTION_ALLOWED
        ),
        "credibility_level_llm": _clean(raw.get("credibility_level_llm")),
        "credibility_rationale": _clean(raw.get("credibility_rationale")),
        "trust_basis": _clean(raw.get("trust_basis")),
        "source_independence_group": _clean(raw.get("source_independence_group")),
        "independence_confidence": _clean(raw.get("independence_confidence")),
        "likely_syndicated_or_aggregated": bool(
            raw.get("likely_syndicated_or_aggregated", False)
        ),
        "aggregation_or_syndication_reason": _clean(
            raw.get("aggregation_or_syndication_reason")
        ),
        "upstream_source_mentions": _as_str_list(raw.get("upstream_source_mentions")),
        "warnings": warnings,
        "errors": errors,
    }


def assess_source_identity_with_llm(
    source_entry: dict,
    collection_spec: dict | None = None,
    parsed_page_identity: dict | None = None,
) -> dict:
    """Return normalized source identity advice from the configured LLM."""

    payload = {
        "source_entry": {
            key: source_entry.get(key)
            for key in (
                "source_id",
                "canonical_url",
                "url",
                "domain",
                "title",
                "snippet",
                "publisher",
                "source_type",
                "search_provider",
                "search_result_source_raw",
                "search_provider_result_source",
                "result_source",
                "search_rank",
                "query_used",
                "discovery_method",
                "role_hint",
                "source_purpose",
            )
        },
        "collection_spec": collection_spec or {},
        "parsed_page_identity": parsed_page_identity or {},
        "required_output_fields": [
            "actual_publisher",
            "actual_publisher_normalized",
            "actual_publisher_confidence",
            "publisher_evidence_fields",
            "publisher_evidence_quotes",
            "publisher_source",
            "source_owner",
            "source_owner_confidence",
            "source_type_final",
            "source_type_confidence",
            "source_type_evidence",
            "claim_support_role",
            "recommended_source_role",
            "recommended_fetch_use",
            "recommended_extraction_use",
            "credibility_level_llm",
            "credibility_rationale",
            "trust_basis",
            "source_independence_group",
            "likely_syndicated_or_aggregated",
            "upstream_source_mentions",
            "warnings",
            "errors",
        ],
    }
    raw = llm_clients.run_structured_llm_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=json.dumps(payload, ensure_ascii=True, indent=2),
        expected_schema_name="SourceIdentityAgentOutput",
        temperature=0.0,
    )
    return _normalize_decision(raw, source_entry)
