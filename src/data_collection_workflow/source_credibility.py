"""Source credibility scoring and final source-role assignment.

The default rubric is deterministic and metadata-only. Optional LLM review is
advisory, gated by explicit environment flags, and cannot browse or fetch.
"""

from __future__ import annotations

from data_collection_workflow.environment import get_env

from collections import Counter
from dataclasses import dataclass
import json
import re
from urllib.parse import urlsplit

from . import llm_clients
from .query_policy import universal_queries_enabled
from .disease_relevance import (
    UNRELATED_DISEASE,
    AMBIGUOUS_DISEASE,
    assessment_fields,
    assess_source_disease_relevance,
    build_disease_relevance_context,
)
from .models import LLMSourceCredibilitySuggestion, SourceCredibilityAssessment
from .state import DataCollectionState

RUBRIC_VERSION = "source_credibility_v2"
DETERMINISTIC_METHOD = "deterministic_source_credibility_v1"
ALLOWED_FINAL_ROLES = {
    "collection",
    "validation",
    "context",
    "collection_support",
    "search_endpoint",
    "excluded",
    "needs_human_review",
}
SEARCH_DISCOVERY_METHODS = {"fixture_search_result", "live_search_result"}
_COUNT_DATA_TERMS = (
    "case",
    "cases",
    "confirmed",
    "death",
    "deaths",
    "hospitalization",
    "surveillance",
    "dashboard",
    "open data",
    "dataset",
    "data",
    "table",
    "counts",
)
_CONTEXT_ONLY_TERMS = (
    "prevention",
    "treatment",
    "symptoms",
    "fact sheet",
    "factsheet",
    "clinical overview",
    "guidance",
    "diagnosis",
    "control",
)
_NEWS_TERMS = ("news", "media", "substack", "blog", "press", "outbreak news")
_SEARCH_ENDPOINT_TERMS = ("pubmed", "europe pmc", "openalex")
_UNRELATED_DISEASE_TERMS = {
    "covid": ["hantavirus", "hps", "dengue"],
    "dengue": ["hantavirus", "hps", "covid-19", "sars-cov-2"],
    "hantavirus": ["covid-19", "sars-cov-2", "dengue"],
}


@dataclass
class SourceCredibilityRuntime:
    llm_enabled: bool = False
    max_sources: int | None = None
    allowlist: set[str] | None = None
    attempted_count: int = 0
    assessed_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    skipped_reason_counts: Counter | None = None
    warnings: list[str] | None = None

    def warning_list(self) -> list[str]:
        if self.warnings is None:
            self.warnings = []
        return self.warnings

    def skipped_reasons(self) -> Counter:
        if self.skipped_reason_counts is None:
            self.skipped_reason_counts = Counter()
        return self.skipped_reason_counts


def _parse_csv_env(name: str) -> set[str] | None:
    raw = get_env(name) or ""
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or None


def _parse_positive_int_env(name: str) -> int | None:
    raw = (get_env(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def source_credibility_runtime_from_env() -> SourceCredibilityRuntime:
    """Read optional LLM source-credibility settings from the environment."""

    return SourceCredibilityRuntime(
        llm_enabled=llm_clients.llm_source_credibility_enabled(),
        max_sources=_parse_positive_int_env("LLM_SOURCE_CREDIBILITY_MAX_SOURCES"),
        allowlist=_parse_csv_env("LLM_SOURCE_CREDIBILITY_SOURCE_ID_ALLOWLIST"),
        warnings=[],
    )


def _text_parts(entry: dict, *, include_query: bool = False) -> list[str]:
    keys = [
        "title",
        "publisher",
        "domain",
        "source_type",
        "source_purpose",
        "notes",
        "snippet",
        "result_source",
        "canonical_url",
    ]
    parts = [str(entry.get(key) or "") for key in keys]
    for key in ("matched_terms", "expected_fields", "expected_extractable_fields"):
        parts.extend(str(item) for item in entry.get(key) or [])
    if include_query:
        parts.append(str(entry.get("query_used") or ""))
    return [part for part in parts if part.strip()]


def _text(entry: dict, *, include_query: bool = False) -> str:
    return " ".join(_text_parts(entry, include_query=include_query)).lower()


def _contains_any(text: str, terms: list[str] | tuple[str, ...] | set[str]) -> bool:
    return any(str(term).lower() in text for term in terms if str(term).strip())


def _is_search_endpoint_candidate(entry: dict) -> bool:
    """Return true for search portals, not for ordinary search-discovered pages."""

    if universal_queries_enabled():
        from .source_identity import _is_search_endpoint
        return _is_search_endpoint(entry, _domain(entry))
    internal_role = str(entry.get("source_role") or "").lower()
    if internal_role in {"search_endpoint", "placeholder_source"}:
        return True
    text = _text(entry, include_query=False)
    if _contains_any(text, _SEARCH_ENDPOINT_TERMS):
        return True
    url = str(entry.get("canonical_url") or entry.get("url") or "").lower()
    return "/search" in url or "search?" in url or "search=" in url


def _domain(entry: dict) -> str:
    domain = str(entry.get("domain") or "").strip().lower()
    if domain:
        return domain[4:] if domain.startswith("www.") else domain
    url = entry.get("canonical_url") or entry.get("url") or ""
    netloc = urlsplit(str(url)).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _location_terms(spec: dict) -> list[str]:
    location = str(spec.get("geography") or spec.get("location") or "").strip()
    terms = [location] if location else []
    lowered = location.lower()
    extras = {
        "new york": ["ny", "nyc", "new york city"],
        "florida": ["fl"],
        "new mexico": ["nm", "santa fe", "albuquerque"],
        "united states": ["usa", "us", "cdc"],
    }
    for key, values in extras.items():
        if key in lowered:
            terms.extend(values)
    return [term for term in terms if term]


def _year_terms(spec: dict) -> list[str]:
    values = [
        spec.get("time_window"),
        spec.get("start_date"),
        spec.get("end_date"),
    ]
    years: list[str] = []
    for value in values:
        for year in re.findall(r"\b(?:19|20)\d{2}\b", str(value or "")):
            if year not in years:
                years.append(year)
    return years


def _authority_score(entry: dict) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    domain = _domain(entry)
    source_type = str(entry.get("source_type") or "").lower()
    if (
        "official_public_health_agency" in source_type
        or domain.endswith(".gov")
        or domain in {"cdc.gov", "who.int", "paho.org", "ecdc.europa.eu"}
        or _contains_any(text, ["department of health", "ministry of health", "public health"])
    ):
        return 0.95, ["official_public_health_authority"]
    if "international_organization_report" in source_type:
        return 0.88, ["international_organization_authority"]
    if "structured_database" in source_type or _contains_any(text, ["open data", "dashboard", "dataset"]):
        return 0.78, ["structured_data_source"]
    if "peer_reviewed_literature" in source_type or _contains_any(text, ["pubmed", "university", "academic"]):
        return 0.74, ["academic_or_literature_source"]
    if "news_and_situation_report" in source_type or _contains_any(text, _NEWS_TERMS):
        return 0.48, ["secondary_news_or_media_source"]
    if not entry.get("publisher"):
        return 0.32, ["publisher_missing_authority_unclear"]
    return 0.45, ["authority_unclear"]


def _local_relevance_score(entry: dict, spec: dict) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    query_text = str(entry.get("query_used") or "").lower()
    terms = [term.lower() for term in _location_terms(spec)]
    domain = _domain(entry)
    if terms and (_contains_any(text, terms) or _contains_any(domain, terms)):
        return 0.95, ["local_source_matches_task_location"]
    if terms and _contains_any(query_text, terms):
        return 0.68, ["location_match_from_planned_query"]
    if any(domain.endswith(suffix) for suffix in ("cdc.gov", "who.int", "paho.org", "ecdc.europa.eu")):
        return 0.58, ["national_or_international_context"]
    if not terms:
        return 0.55, ["task_location_unspecified"]
    return 0.25, ["location_relevance_unclear"]


def _disease_relevance_score(
    entry: dict,
    spec: dict,
    disease_intelligence: dict,
) -> tuple[float, list[str]]:
    context = build_disease_relevance_context(
        {
            "collection_spec": spec,
            "disease_intelligence": disease_intelligence,
        }
    )
    assessment = assess_source_disease_relevance(entry, context)
    status = assessment.get("status")
    flags: list[str] = [f"source_disease_relevance:{status}"]
    if status == UNRELATED_DISEASE:
        flags.append("unrelated_disease_signal_in_source_metadata")
    elif status == AMBIGUOUS_DISEASE:
        flags.append("ambiguous_disease_signal_in_source_metadata")
    elif status == "target_disease_match":
        flags.append("source_metadata_matches_requested_disease")
    elif status == "related_context_only":
        flags.append("source_metadata_matches_requested_disease_context_only")
    else:
        flags.append("disease_relevance_unclear")
    return float(assessment.get("score") or 0.0), flags


def _timeliness_score(entry: dict, spec: dict) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    query_text = str(entry.get("query_used") or "").lower()
    published = str(entry.get("published_date") or "")
    years = _year_terms(spec)
    if years and any(year in published or year in text for year in years):
        return 0.95, ["source_time_matches_requested_window"]
    if years and any(year in query_text for year in years):
        return 0.72, ["time_window_match_from_planned_query"]
    if published:
        return 0.65, ["published_date_present_but_not_window_specific"]
    return 0.50, ["date_or_time_window_unclear"]


def _geographic_granularity_score(entry: dict, local_score: float) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    if local_score >= 0.80 and _contains_any(text, ["county", "city", "state", "zip", "local"]):
        return 0.95, ["local_or_subnational_granularity"]
    if local_score >= 0.80:
        return 0.82, ["task_location_granularity"]
    if local_score >= 0.55:
        return 0.58, ["national_or_international_granularity"]
    return 0.30, ["geographic_granularity_unclear"]


def _data_granularity_score(entry: dict) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    role = entry.get("source_role")
    fields = {str(field).lower() for field in entry.get("expected_extractable_fields") or entry.get("expected_fields") or []}
    if _is_search_endpoint_candidate(entry):
        return 0.20, ["not_directly_extractable_search_endpoint"]
    if _contains_any(text, _CONTEXT_ONLY_TERMS) and not ({"cases", "deaths"} & fields):
        return 0.25, ["context_or_prevention_only"]
    if {"cases", "deaths", "date", "location"} <= fields:
        return 0.92, ["case_death_date_location_expected"]
    if {"cases", "date", "location"} <= fields or {"deaths", "date", "location"} <= fields:
        return 0.78, ["partial_case_or_death_data_expected"]
    if _contains_any(text, _COUNT_DATA_TERMS):
        return 0.68, ["data_signal_in_source_metadata"]
    return 0.35, ["data_granularity_unclear"]


def _machine_readability_score(entry: dict) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    url = str(entry.get("canonical_url") or entry.get("url") or "").lower()
    source_type = str(entry.get("source_type") or "").lower()
    if _contains_any(text + " " + url + " " + source_type, ["api", "csv", "open data", "dashboard", "dataset", "structured_database"]):
        return 0.88, ["machine_readable_or_structured"]
    if ".pdf" in url or "[pdf]" in text or "pdf" in text:
        return 0.52, ["pdf_or_report_likely_medium_readability"]
    if url.startswith("http"):
        return 0.72, ["standard_web_page"]
    if url.startswith("seed://"):
        return 0.30, ["placeholder_uri_not_fetchable"]
    return 0.45, ["machine_readability_unclear"]


def _independence_score(entry: dict, authority: float) -> tuple[float, list[str]]:
    text = _text(entry, include_query=False)
    if authority >= 0.85:
        return 0.90, ["primary_or_authoritative_source"]
    if "peer_reviewed_literature" in str(entry.get("source_type") or ""):
        return 0.80, ["independent_literature_source"]
    if _contains_any(text, ["substack", "blog", "reposted", "syndicated"]):
        return 0.35, ["possible_derivative_or_syndicated_source"]
    if entry.get("publisher"):
        return 0.62, ["named_publisher"]
    return 0.42, ["independence_unclear"]


def _provenance_score(entry: dict) -> tuple[float, list[str]]:
    score = 1.0
    flags: list[str] = []
    for field, penalty, flag in (
        ("canonical_url", 0.25, "missing_canonical_url"),
        ("title", 0.15, "missing_title"),
        ("publisher", 0.15, "missing_publisher"),
        ("source_type", 0.10, "missing_source_type"),
        ("discovery_method", 0.10, "missing_discovery_method"),
    ):
        if not entry.get(field):
            score -= penalty
            flags.append(flag)
    if entry.get("discovery_method") in SEARCH_DISCOVERY_METHODS and not entry.get("query_id"):
        score -= 0.10
        flags.append("missing_search_query_id")
    return max(0.0, round(score, 4)), flags or ["complete_source_provenance"]


def _risk_penalty(
    entry: dict,
    *,
    authority: float,
    local_relevance: float,
    disease_relevance: float,
    timeliness: float,
    data_granularity: float,
    machine_readability: float,
) -> tuple[float, list[str], list[str]]:
    flags: list[str] = []
    warnings: list[str] = []
    text = _text(entry, include_query=False)
    url = str(entry.get("canonical_url") or entry.get("url") or "")
    source_type = str(entry.get("source_type") or "").lower()
    if not url:
        flags.append("missing_url")
    if url and not (url.startswith("http") or url.startswith("seed://")):
        flags.append("unsupported_url_scheme")
    if not entry.get("publisher"):
        flags.append("missing_publisher")
    if disease_relevance < 0.30:
        flags.append("ambiguous_disease")
    if local_relevance < 0.40:
        flags.append("ambiguous_location")
    if timeliness < 0.60:
        flags.append("ambiguous_date")
    if data_granularity < 0.45:
        flags.append("source_likely_not_extractable")
    if machine_readability < 0.55:
        flags.append("low_machine_readability")
    if _contains_any(text, _CONTEXT_ONLY_TERMS):
        flags.append("context_or_background_only")
    if authority < 0.55 and disease_relevance >= 0.50:
        flags.append("low_authority_relevant_source")
    if "news_and_situation_report" in source_type or (
        _contains_any(text, _NEWS_TERMS) and authority < 0.80
    ):
        flags.append("secondary_news_or_media_source")
    for flag in entry.get("routing_flags") or []:
        if flag not in flags and flag in {
            "validation_reserved",
            "blocked_from_collection",
            "screening_and_critic_disagree",
            "low_screening_confidence",
        }:
            flags.append(flag)
    for flag in entry.get("critic_flags") or []:
        if flag not in flags and "risk" in flag:
            flags.append(flag)

    penalty = 0.0
    penalty += 0.08 if "ambiguous_disease" in flags else 0.0
    penalty += 0.06 if "ambiguous_location" in flags else 0.0
    penalty += 0.04 if "ambiguous_date" in flags else 0.0
    penalty += 0.08 if "source_likely_not_extractable" in flags else 0.0
    penalty += 0.06 if "low_authority_relevant_source" in flags else 0.0
    penalty += 0.04 if "secondary_news_or_media_source" in flags else 0.0
    penalty += 0.06 if "low_machine_readability" in flags else 0.0
    penalty += 0.10 if "screening_and_critic_disagree" in flags else 0.0
    if "missing_url" in flags:
        warnings.append("source_credibility_missing_url")
    return min(0.45, round(penalty, 4)), flags, warnings


def _score_components(entry: dict, state: DataCollectionState) -> tuple[dict, list[str], list[str]]:
    spec = state.get("collection_spec") or {}
    disease_intelligence = state.get("disease_intelligence") or {}
    authority, authority_flags = _authority_score(entry)
    local_relevance, local_flags = _local_relevance_score(entry, spec)
    disease_relevance, disease_flags = _disease_relevance_score(entry, spec, disease_intelligence)
    timeliness, time_flags = _timeliness_score(entry, spec)
    geo_granularity, geo_flags = _geographic_granularity_score(entry, local_relevance)
    data_granularity, data_flags = _data_granularity_score(entry)
    machine_readability, machine_flags = _machine_readability_score(entry)
    independence, independence_flags = _independence_score(entry, authority)
    provenance, provenance_flags = _provenance_score(entry)
    risk_penalty, risk_flags, warnings = _risk_penalty(
        entry,
        authority=authority,
        local_relevance=local_relevance,
        disease_relevance=disease_relevance,
        timeliness=timeliness,
        data_granularity=data_granularity,
        machine_readability=machine_readability,
    )
    flags: list[str] = []
    for component_flags in (
        authority_flags,
        local_flags,
        disease_flags,
        time_flags,
        geo_flags,
        data_flags,
        machine_flags,
        independence_flags,
        provenance_flags,
        risk_flags,
    ):
        for flag in component_flags:
            if flag not in flags:
                flags.append(flag)
    components = {
        "authority_score": round(authority, 4),
        "local_relevance_score": round(local_relevance, 4),
        "disease_relevance_score": round(disease_relevance, 4),
        "timeliness_score": round(timeliness, 4),
        "geographic_granularity_score": round(geo_granularity, 4),
        "data_granularity_score": round(data_granularity, 4),
        "machine_readability_score": round(machine_readability, 4),
        "independence_score": round(independence, 4),
        "provenance_score": round(provenance, 4),
        "risk_penalty": round(risk_penalty, 4),
    }
    return components, flags, warnings


def _weighted_score(components: dict) -> float:
    weights = {
        "authority_score": 0.16,
        "local_relevance_score": 0.14,
        "disease_relevance_score": 0.16,
        "timeliness_score": 0.08,
        "geographic_granularity_score": 0.09,
        "data_granularity_score": 0.14,
        "machine_readability_score": 0.08,
        "independence_score": 0.07,
        "provenance_score": 0.08,
    }
    base = sum(float(components[key]) * weight for key, weight in weights.items())
    return max(0.0, min(1.0, round(base - float(components["risk_penalty"]), 4)))


def _role_recommendation(entry: dict, components: dict, risk_flags: list[str]) -> tuple[str, str]:
    internal_role = entry.get("source_role")
    source_type = str(entry.get("source_type") or "")
    role_hint = str(entry.get("role_hint") or "").strip()
    text = _text(entry, include_query=False)

    if "unrelated_disease_signal_in_source_metadata" in risk_flags:
        return "excluded", "Source metadata names an incompatible disease for the active task."
    if internal_role in {"validation_reserved", "validation_source", "validation_only"}:
        return "validation", "Source is reserved by source-role policy for validation."
    if role_hint == "validation" and not universal_queries_enabled():
        return "validation", "Planned source role hint marks this source as validation."
    if _is_search_endpoint_candidate(entry):
        return "search_endpoint", "Source is a search endpoint or placeholder, not a directly extractable data source."
    if internal_role == "context_source" or role_hint == "context" or _contains_any(text, _CONTEXT_ONLY_TERMS):
        return "context", "Source metadata indicates background/context rather than primary collection."
    if (
        internal_role == "data_source"
        and "official_public_health_agency" in source_type
        and components["disease_relevance_score"] >= 0.50
        and components["data_granularity_score"] >= 0.65
    ):
        return "collection", "Official public health source has task-relevant collection data signals."
    if "news_and_situation_report" in source_type or role_hint == "collection_support" or _contains_any(text, _NEWS_TERMS):
        return "collection_support", "Source may support event discovery but is secondary or lower authority."
    if components["disease_relevance_score"] < 0.30:
        return "excluded", "Source metadata does not match the requested disease closely enough."
    if components["data_granularity_score"] < 0.35 and components["disease_relevance_score"] >= 0.50:
        return "needs_human_review", "Disease appears relevant but data extractability is unclear."
    return "collection", "Source appears suitable for primary collection based on task-aware metadata."


def _human_review(
    role: str,
    components: dict,
    risk_flags: list[str],
    score: float,
) -> tuple[bool, str | None, str]:
    if role in {"validation", "context", "search_endpoint", "excluded"}:
        return False, None, role

    reasons: list[str] = []
    if role == "needs_human_review":
        reasons.append("role_assignment_ambiguous")
    if role == "collection_support" and "secondary_news_or_media_source" in risk_flags:
        reasons.append("secondary_source_reports_possible_case_data")
    if components["disease_relevance_score"] >= 0.50 and components["local_relevance_score"] < 0.40:
        reasons.append("disease_relevant_but_location_unclear")
    if components["disease_relevance_score"] >= 0.50 and components["timeliness_score"] < 0.60:
        reasons.append("disease_relevant_but_time_window_unclear")
    if "missing_publisher" in risk_flags:
        reasons.append("missing_publisher")
    if score < 0.55 and components["disease_relevance_score"] >= 0.40:
        reasons.append("low_score_but_potentially_relevant")
    if "screening_and_critic_disagree" in risk_flags:
        reasons.append("screening_and_critic_disagree")
    if not reasons:
        return False, None, role
    final_role = "needs_human_review" if role == "needs_human_review" else role
    return True, "; ".join(reasons), final_role


def _credibility_level(score: float, role: str, human_review: bool) -> str:
    if role == "excluded":
        return "excluded"
    if human_review and score < 0.70:
        return "needs_review"
    if score >= 0.78:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _legacy_component_map(components: dict) -> dict:
    return {
        "authority": components["authority_score"],
        "granularity": components["data_granularity_score"],
        "provenance": components["provenance_score"],
        "timeliness": components["timeliness_score"],
        "independence": components["independence_score"],
        "risk": round(1.0 - components["risk_penalty"], 4),
        **components,
    }


def _should_call_llm(entry: dict, runtime: SourceCredibilityRuntime) -> bool:
    if not runtime.llm_enabled:
        return False
    source_id = str(entry.get("source_id") or "")
    if runtime.allowlist is not None and source_id not in runtime.allowlist:
        runtime.skipped_count += 1
        return False
    if runtime.max_sources is not None and runtime.attempted_count >= runtime.max_sources:
        runtime.skipped_count += 1
        return False
    return True


def _collection_mode_from_state(state: DataCollectionState | dict) -> str:
    structured = state.get("structured_task") or {}
    collection = state.get("collection_spec") or {}
    workflow = state.get("workflow") or {}
    return str(
        structured.get("collection_mode")
        or collection.get("collection_mode")
        or workflow.get("collection_mode")
        or state.get("collection_mode")
        or ""
    ).strip()


def _is_direct_verified_target_source(entry: dict, state: DataCollectionState | dict) -> bool:
    if _collection_mode_from_state(state) != "direct_collection":
        return False
    if entry.get("must_fetch") is True:
        return True
    if str(entry.get("target_fit_status") or "").lower() in {
        "verified_target_collection",
        "verified_target_source",
    }:
        return True
    if str(entry.get("triage_role") or "").lower() == "verified_target_collection":
        return True
    return bool(entry.get("coverage_requirement_ids"))


def _direct_verified_target_fast_path_active(state: DataCollectionState | dict) -> bool:
    if _collection_mode_from_state(state) != "direct_collection":
        return False
    for entry in state.get("source_registry") or []:
        if isinstance(entry, dict) and _is_direct_verified_target_source(entry, state):
            return True
    for entry in state.get("must_fetch_sources") or []:
        if isinstance(entry, dict) and _is_direct_verified_target_source(entry, state):
            return True
    return False


def _apply_llm_advisory(
    entry: dict,
    state: DataCollectionState,
    assessment: dict,
    runtime: SourceCredibilityRuntime,
) -> dict:
    if not _should_call_llm(entry, runtime):
        return assessment
    runtime.attempted_count += 1
    payload = {
        "instruction": (
            "Review source metadata and deterministic credibility scores. Do not "
            "browse, fetch, create URLs, or decide final policy. Return advisory "
            "risk flags, role suggestion, and human review recommendation only."
        ),
        "source_entry": {
            key: entry.get(key)
            for key in (
                "source_id",
                "canonical_url",
                "title",
                "publisher",
                "domain",
                "source_type",
                "discovery_method",
                "query_id",
                "query_used",
                "role_hint",
                "snippet",
            )
        },
        "collection_spec": state.get("collection_spec") or {},
        "disease_intelligence_summary": state.get("disease_intelligence_summary") or {},
        "deterministic_assessment": assessment,
    }
    try:
        raw = llm_clients.run_pydantic_structured_llm(
            system_prompt=(
                "You are an advisory source credibility reviewer for the data "
                "collection workflow. Return only structured advisory output. "
                "Do not browse, fetch, or create URLs."
            ),
            user_prompt=json.dumps(payload, ensure_ascii=True, indent=2),
            schema_model=LLMSourceCredibilitySuggestion,
            temperature=0.0,
        )
        suggestion = LLMSourceCredibilitySuggestion(**raw)
    except Exception as exc:  # noqa: BLE001 - deterministic fallback is required.
        runtime.failure_count += 1
        warnings = list(assessment.get("warnings") or [])
        warnings.append(f"llm_source_credibility_failed:{type(exc).__name__}")
        assessment.update(
            {
                "llm_used": False,
                "llm_failed": True,
                "llm_error_type": type(exc).__name__,
                "warnings": warnings,
            }
        )
        return assessment

    runtime.assessed_count += 1
    risk_flags = list(assessment.get("risk_flags") or [])
    for flag in suggestion.risk_flags:
        if flag not in risk_flags:
            risk_flags.append(flag)
    human_review = bool(
        assessment.get("human_review_recommended") or suggestion.human_review_recommended
    )
    human_reason = assessment.get("human_review_reason")
    if suggestion.human_review_recommended and suggestion.explanation:
        human_reason = (
            f"{human_reason}; {suggestion.explanation}"
            if human_reason
            else suggestion.explanation
        )
    assessment.update(
        {
            "llm_used": True,
            "llm_failed": False,
            "llm_error_type": None,
            "llm_source_role_recommendation": suggestion.source_role_recommendation,
            "llm_credibility_level": suggestion.credibility_level,
            "llm_source_credibility_explanation": suggestion.explanation,
            "llm_source_credibility_confidence": suggestion.confidence,
            "risk_flags": risk_flags,
            "human_review_recommended": human_review,
            "human_review_reason": human_reason,
        }
    )
    return assessment


def apply_source_credibility_assessment(
    entry: dict,
    state: DataCollectionState,
    runtime: SourceCredibilityRuntime,
) -> tuple[dict, dict]:
    """Return an entry enriched with Stage 6 source credibility fields."""

    components, risk_flags, warnings = _score_components(entry, state)
    source_disease_assessment = assess_source_disease_relevance(
        entry,
        build_disease_relevance_context(state),
    )
    source_disease_fields = assessment_fields(source_disease_assessment, "source")
    score = _weighted_score(components)
    recommendation, role_reason = _role_recommendation(entry, components, risk_flags)
    human_review, human_reason, final_role = _human_review(
        recommendation,
        components,
        risk_flags,
        score,
    )
    level = _credibility_level(score, final_role, human_review)
    explanation = (
        f"{level} credibility by {RUBRIC_VERSION}: score={score:.2f}; "
        f"authority={components['authority_score']:.2f}; "
        f"local={components['local_relevance_score']:.2f}; "
        f"disease={components['disease_relevance_score']:.2f}; "
        f"data={components['data_granularity_score']:.2f}; "
        f"risk_penalty={components['risk_penalty']:.2f}."
    )

    assessment = {
        "source_id": entry.get("source_id", ""),
        "canonical_url": entry.get("canonical_url") or entry.get("url"),
        "title": entry.get("title"),
        "publisher": entry.get("publisher"),
        "domain": entry.get("domain") or _domain(entry),
        "source_type": entry.get("source_type"),
        "discovery_method": entry.get("discovery_method"),
        "query_id": entry.get("query_id"),
        "query_used": entry.get("query_used"),
        "role_hint": entry.get("role_hint"),
        "source_role_recommendation": recommendation,
        "source_role_final": final_role,
        "credibility_score": score,
        "credibility_level": level,
        **components,
        **source_disease_fields,
        "final_score_explanation": explanation,
        "role_assignment_reason": role_reason,
        "risk_flags": risk_flags,
        "human_review_recommended": human_review,
        "human_review_reason": human_reason,
        "assessment_method": DETERMINISTIC_METHOD,
        "llm_used": False,
        "llm_failed": False,
        "llm_error_type": None,
        "warnings": warnings,
    }
    if runtime.llm_enabled and (
        _is_direct_verified_target_source(entry, state)
        or _direct_verified_target_fast_path_active(state)
    ):
        reason = "direct_target_official_fast_path_skips_source_credibility"
        runtime.skipped_count += 1
        runtime.skipped_reasons()[reason] += 1
        warnings = list(assessment.get("warnings") or [])
        if reason not in warnings:
            warnings.append(reason)
        assessment.update(
            {
                "warnings": warnings,
                "source_credibility_llm_skipped_reason": reason,
            }
        )
    else:
        assessment = _apply_llm_advisory(entry, state, assessment, runtime)
    # Keep deterministic final role as the policy boundary, but allow LLM review
    # to add human-review risk.
    human_review = bool(assessment.get("human_review_recommended"))
    level = _credibility_level(score, final_role, human_review)
    assessment["credibility_level"] = level
    assessment["source_role_final"] = final_role

    validated = SourceCredibilityAssessment(**assessment).model_dump()
    updated = dict(entry)
    updated.update(validated)
    updated.update(
        {
            "credibility_rubric_version": RUBRIC_VERSION,
            "credibility_score_components": _legacy_component_map(components),
            "credibility_flags": list(validated.get("risk_flags") or []),
            "credibility_reason": validated.get("final_score_explanation"),
        }
    )
    if validated.get("source_disease_relevance_status") == UNRELATED_DISEASE:
        flags = list(updated.get("routing_flags") or [])
        if "disease_mismatch_excluded_from_collection" not in flags:
            flags.append("disease_mismatch_excluded_from_collection")
        updated["routing_flags"] = flags
        updated["source_role_final"] = "excluded"
        updated["credibility_level"] = "excluded"
        validated["source_role_final"] = "excluded"
        validated["credibility_level"] = "excluded"
        updated["ready_for_content_fetch"] = False
        updated["final_screening_decision"] = "exclude"
        updated["status"] = "excluded"
        updated["final_screening_reason"] = (
            "Source excluded because source metadata names an incompatible "
            "disease for the active task."
        )
    if validated.get("human_review_recommended"):
        flags = list(updated.get("routing_flags") or [])
        if "source_credibility_human_review_recommended" not in flags:
            flags.append("source_credibility_human_review_recommended")
        updated["routing_flags"] = flags
        updated["requires_human_review"] = True
        if final_role == "needs_human_review":
            updated["ready_for_content_fetch"] = False
            updated["final_screening_decision"] = "needs_human_review"
            updated["status"] = "needs_human_review"
            updated["final_screening_reason"] = validated.get("human_review_reason")
    return updated, validated


def build_source_credibility_summary(
    assessments: list[dict],
    runtime: SourceCredibilityRuntime,
) -> dict:
    level_counter: Counter = Counter()
    role_counter: Counter = Counter()
    discovery_counter: Counter = Counter()
    risk_counter: Counter = Counter()
    disease_status_counter: Counter = Counter()
    for assessment in assessments:
        level_counter[assessment.get("credibility_level") or "unknown"] += 1
        role_counter[assessment.get("source_role_final") or "unknown"] += 1
        discovery_counter[assessment.get("discovery_method") or "unknown"] += 1
        for flag in assessment.get("risk_flags") or []:
            risk_counter[flag] += 1
        disease_status_counter[
            assessment.get("source_disease_relevance_status") or "unknown"
        ] += 1
    return {
        "assessed_source_count": len(assessments),
        "high_credibility_count": level_counter.get("high", 0),
        "medium_credibility_count": level_counter.get("medium", 0),
        "low_credibility_count": level_counter.get("low", 0),
        "needs_review_count": level_counter.get("needs_review", 0)
        + role_counter.get("needs_human_review", 0),
        "excluded_count": level_counter.get("excluded", 0)
        + role_counter.get("excluded", 0),
        "credibility_level_counts": dict(level_counter),
        "role_counts": dict(role_counter),
        "discovery_method_counts": dict(discovery_counter),
        "search_derived_assessed_count": sum(
            discovery_counter.get(method, 0) for method in SEARCH_DISCOVERY_METHODS
        ),
        "seed_catalog_assessed_count": discovery_counter.get("offline_seed_catalog", 0),
        "llm_enabled": runtime.llm_enabled,
        "llm_assessed_count": runtime.assessed_count,
        "llm_failure_count": runtime.failure_count,
        "llm_skipped_count": runtime.skipped_count,
        "llm_skipped_reason_counts": dict(runtime.skipped_reasons()),
        "risk_flag_counts": dict(risk_counter),
        "source_disease_relevance_status_counts": dict(disease_status_counter),
        "warnings": runtime.warning_list(),
    }
