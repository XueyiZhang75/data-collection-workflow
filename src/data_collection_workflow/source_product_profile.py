"""Deterministic source product profiling.

The profiler classifies what a source *is useful for* before fetch/extraction.
It is intentionally lightweight and generic: it uses URL, title, snippet,
source type, and task terms, and it does not know about any benchmark rows.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from .source_identity import lookup_source_identity_registry


PROFILE_FIELDS = (
    "data_product_type",
    "task_specificity",
    "time_window_fit",
    "machine_readability",
    "expected_evidence_role",
    "source_product_profile_reason",
)

EVENT_TERMS = {
    "alert",
    "advisory",
    "case",
    "cases",
    "case report",
    "cluster",
    "confirmed",
    "death",
    "deaths",
    "don",
    "exposure",
    "line list",
    "line-list",
    "outbreak",
    "situation",
    "surveillance",
    "update",
}

_LEGACY_EVENT_PAGE_TERMS = {
    "arrival and cleaning",
    "brote",
    "cas navire",
    "cruise ship",
    "dgs urgent",
    "hondius",
    "informe cierre",
    "mv hondius",
    "navire",
    "outbreaks under monitoring",
    "point de situation",
    "rapid risk assessment",
    "ship hondius",
}

BACKGROUND_TERMS = {
    "about",
    "background",
    "biosafety",
    "clinical overview",
    "fact sheet",
    "frequently asked",
    "overview",
    "pathogen safety",
    "risk assessment sheet",
}

SEARCH_TERMS = {
    "browse",
    "current issue",
    "index",
    "search",
    "search results",
}

POLICY_TERMS = {
    "guidance",
    "protocol",
    "recommendations",
    "technical note",
}

OTHER_DISEASE_TERMS = {
    "avian influenza",
    "yellow fever",
    "covid",
    "dengue",
    "ebola",
    "influenza",
    "measles",
    "mpox",
    "polio",
    "tuberculosis",
    "zika",
}


def _text(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text(item) for item in value)
    return str(value or "")


def _lower_join(entry: dict) -> str:
    return " ".join(
        _text(entry.get(key))
        for key in (
            "canonical_url",
            "url",
            "title",
            "snippet",
            "publisher",
            "actual_publisher",
            "source_type",
            "source_type_final",
        )
    ).lower()


def _domain(entry: dict) -> str:
    value = str(entry.get("domain") or "").strip().lower()
    if value:
        return value.replace("www.", "")
    url = str(entry.get("canonical_url") or entry.get("url") or "")
    try:
        return urlsplit(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _path(entry: dict) -> str:
    url = str(entry.get("canonical_url") or entry.get("url") or "")
    try:
        return urlsplit(url).path.lower()
    except Exception:
        return url.lower()


def _contains(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _has_event_page_signal(text: str, state: dict | None = None) -> bool:
    from .query_policy import supported_event_terms, universal_queries_enabled
    if universal_queries_enabled():
        return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I)
                   for term in supported_event_terms(state or {}))
    return _contains(text, _LEGACY_EVENT_PAGE_TERMS) or (
        "cruise" in text and "ship" in text
    )


def _is_verified_authority_domain(entry: dict) -> bool:
    domain = _domain(entry)
    if not domain:
        return False
    registry_entry = lookup_source_identity_registry(domain)
    source_type = str((registry_entry or {}).get("source_type") or "").lower()
    if source_type in {
        "official_public_health_agency",
        "international_public_health_agency",
        "international_organization_report",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "state_public_health_agency",
        "local_public_health_agency",
        "government_report",
    }:
        return True
    source_type_final = str(
        entry.get("source_type_final") or entry.get("source_type") or ""
    ).lower()
    return source_type_final in {
        "official_public_health_agency",
        "international_public_health_agency",
        "international_organization_report",
        "national_public_health_agency",
        "state_or_local_public_health_agency",
        "state_public_health_agency",
        "local_public_health_agency",
        "government_report",
    }


def _task_terms(state: dict | None) -> set[str]:
    state = state or {}
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    disease_intelligence = state.get("disease_intelligence") or {}
    terms = {
        _text(structured_task.get("disease")).lower(),
        _text(collection_spec.get("disease")).lower(),
        _text(disease_intelligence.get("disease_input")).lower(),
        _text(disease_intelligence.get("disease_standard_name")).lower(),
    }
    for key in ("aliases", "abbreviations", "pathogen_terms", "syndrome_terms"):
        for value in disease_intelligence.get(key) or []:
            terms.add(_text(value).lower())
    cleaned = {term.strip() for term in terms if term and len(term.strip()) >= 3}
    if any("hanta" in term for term in cleaned):
        cleaned.update({"hantavirus", "orthohantavirus", "andv", "andes virus"})
    return cleaned


def _matches_task_disease(text: str, state: dict | None) -> bool:
    terms = _task_terms(state)
    return any(term in text for term in terms)


def _unrelated_to_task(text: str, state: dict | None) -> bool:
    if _matches_task_disease(text, state):
        return False
    return _contains(text, OTHER_DISEASE_TERMS)


def _task_years(state: dict | None) -> tuple[int | None, int | None]:
    state = state or {}
    structured_task = state.get("structured_task") or {}
    collection_spec = state.get("collection_spec") or {}
    start = (
        structured_task.get("start_date")
        or collection_spec.get("start_date")
        or state.get("task_start_date")
    )
    end = (
        structured_task.get("end_date")
        or collection_spec.get("end_date")
        or state.get("task_end_date")
    )

    def year(value: object) -> int | None:
        match = re.search(r"\b(20\d{2})\b", str(value or ""))
        return int(match.group(1)) if match else None

    return year(start), year(end)


def _time_window_fit(text: str, state: dict | None) -> str:
    years = [int(match) for match in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return "unknown"
    start_year, end_year = _task_years(state)
    if not start_year or not end_year:
        return "unknown"
    if any(start_year <= year <= end_year for year in years):
        return "in_window"
    if any(start_year - 1 <= year <= end_year + 1 for year in years):
        return "near_window"
    if max(years) < start_year:
        return "historical_context"
    if min(years) > end_year:
        return "future_or_out_of_scope"
    return "unknown"


def _data_product_type(entry: dict, text: str, state: dict | None) -> str:
    from .query_policy import universal_queries_enabled
    domain = _domain(entry)
    path = _path(entry)
    source_type = str(entry.get("source_type_final") or entry.get("source_type") or "").lower()

    if _unrelated_to_task(text, state):
        return "unrelated_or_other"
    if domain == "pathoplexus.org":
        if "/seq/" in path:
            return "sequence_database_record"
        if "search" in path or "browse" in text:
            return "sequence_database_search"
        if "/news/" in path and _contains(text, EVENT_TERMS):
            return "event_outbreak_report"
        return "sequence_database_search"
    if not universal_queries_enabled() and domain in {"nejm.org", "eurosurveillance.org", "science.org"}:
        if _contains(text, EVENT_TERMS) or "doi/" in path:
            return "case_report_article"
    if _has_event_page_signal(text, state) and (
        _is_verified_authority_domain(entry)
        or _matches_task_disease(text, state)
        or _contains(text, EVENT_TERMS)
    ):
        return "event_outbreak_report"
    if _contains(text, BACKGROUND_TERMS):
        return "background_fact_sheet"
    if _contains(text, POLICY_TERMS):
        return "policy_or_protocol"
    if universal_queries_enabled() and (
        source_type in {"academic_or_peer_reviewed_source", "peer_reviewed_literature"}
        or re.search(r"\b(?:study|epidemiolog\w*|cohort|genomic|cross-sectional)\b", text)
    ) and not any(term in text for term in ("line list", "line-list", "case series")):
        return "case_report_article" if "case report" in text else "research_article"
    if "line list" in text or "line-list" in text or "case series" in text:
        return "line_list_or_case_series"
    if _contains(text, SEARCH_TERMS) or path.rstrip("/").endswith("/search"):
        return "search_or_index_page"
    if any(token in text for token in ("surveillance report", "annual epidemiological", "reported cases", "dashboard")):
        return "official_surveillance_report"
    if _contains(text, EVENT_TERMS):
        if "news" in source_type or "media" in source_type:
            return "news_case_report"
        return "event_outbreak_report"
    if _matches_task_disease(text, state):
        if universal_queries_enabled():
            return "unknown"
        return "background_fact_sheet"
    return "unrelated_or_other"


def _task_specificity(product_type: str, text: str, state: dict | None) -> str:
    if product_type == "unrelated_or_other":
        return "unrelated"
    if product_type in {
        "background_fact_sheet",
        "policy_or_protocol",
        "search_or_index_page",
        "sequence_database_search",
    }:
        if _matches_task_disease(text, state):
            return "disease_specific_but_context"
        return "generic_background"
    if product_type in {
        "event_outbreak_report",
        "official_surveillance_report",
        "case_report_article",
        "line_list_or_case_series",
    } and _has_event_page_signal(text, state):
        return "event_specific"
    if _contains(text, EVENT_TERMS) and _matches_task_disease(text, state):
        return "event_specific"
    if _matches_task_disease(text, state):
        return "disease_specific_but_context"
    return "generic_background"


def _machine_readability(product_type: str, text: str, entry: dict) -> str:
    path = _path(entry)
    if product_type == "unrelated_or_other":
        return "not_extractable"
    if path.endswith(".pdf"):
        return "pdf_extractable"
    if product_type in {"sequence_database_record", "line_list_or_case_series"}:
        return "structured_table"
    if product_type in {"search_or_index_page", "sequence_database_search"}:
        return "manual_or_search_page"
    if product_type in {"background_fact_sheet", "policy_or_protocol"}:
        return "not_extractable"
    if any(token in text for token in ("table", "csv", "download", "dashboard")):
        return "structured_table"
    return "narrative_extractable"


def _expected_evidence_role(product_type: str, text: str) -> str:
    if product_type == "unrelated_or_other":
        return "do_not_extract"
    if product_type == "research_article":
        return "potential_public_health_evidence"
    if any(token in text for token in ("tested negative", "negative", "monitoring", "quarantine", "contact")):
        return "non_case_or_monitoring_evidence"
    if product_type in {
        "case_report_article",
        "line_list_or_case_series",
        "news_case_report",
        "sequence_database_record",
    }:
        return "individual_case_evidence"
    if product_type in {"event_outbreak_report", "official_surveillance_report"}:
        return "aggregate_event_evidence"
    return "source_context"


def _profile_reason(product_type: str, task_specificity: str, text: str) -> str:
    if product_type == "unrelated_or_other":
        return "unrelated_to_task_disease_or_event"
    if product_type == "background_fact_sheet":
        return "background_or_fact_sheet"
    if product_type == "search_or_index_page":
        return "search_or_index_page"
    if product_type == "policy_or_protocol":
        return "policy_or_protocol"
    if product_type == "sequence_database_record":
        return "structured_sequence_record"
    if product_type == "sequence_database_search":
        return "structured_database_search"
    if product_type == "research_article":
        return "research_product_requires_content_assessment"
    if product_type == "case_report_article":
        return "peer_reviewed_case_report"
    if product_type == "line_list_or_case_series":
        return "line_list_or_case_series"
    if product_type == "official_surveillance_report":
        return "official_surveillance_or_report"
    if product_type == "news_case_report":
        return "news_or_supporting_case_report"
    if product_type == "event_outbreak_report":
        if _has_event_page_signal(text):
            return "official_event_page_signal"
        if task_specificity == "event_specific":
            return "task_disease_event_terms_present"
        return "event_terms_present_but_task_specificity_limited"
    return "deterministic_source_product_profile"


def _is_upstream_target_source(entry: dict) -> bool:
    target_status = str(entry.get("target_fit_status") or "").strip().lower()
    triage_role = str(entry.get("triage_role") or "").strip().lower()
    return bool(
        target_status
        in {
            "verified_target_collection",
            "predicted_target_candidate",
            "task_record_collection_candidate",
        }
        or triage_role
        in {
            "verified_target_collection",
            "predicted_target_candidate",
            "task_record_collection_candidate",
        }
    )


def _target_source_product_type(entry: dict, text: str) -> str:
    path = _path(entry)
    source_type = str(entry.get("source_type_final") or entry.get("source_type") or "").lower()
    if "line list" in text or "line-list" in text or "case series" in text:
        return "line_list_or_case_series"
    if path.endswith(".pdf") or "surveillance" in text or "report" in text:
        return "official_surveillance_report"
    if "news" in source_type or "media" in source_type:
        return "news_case_report"
    return "event_outbreak_report"


def profile_source_product(entry: dict, state: dict | None = None) -> dict:
    """Return deterministic source product profile fields for a registry entry."""

    text = _lower_join(entry)
    product_type = _data_product_type(entry, text, state)
    if product_type == "unrelated_or_other" and _is_upstream_target_source(entry):
        product_type = _target_source_product_type(entry, text)
    task_specificity = _task_specificity(product_type, text, state)
    time_window_fit = _time_window_fit(text, state)
    machine_readability = _machine_readability(product_type, text, entry)
    expected_evidence_role = _expected_evidence_role(product_type, text)
    return {
        "data_product_type": product_type,
        "task_specificity": task_specificity,
        "time_window_fit": time_window_fit,
        "machine_readability": machine_readability,
        "expected_evidence_role": expected_evidence_role,
        "source_product_profile_reason": _profile_reason(
            product_type,
            task_specificity,
            text,
        ),
    }


def apply_source_product_profiles(entries: list[dict], state: dict | None = None) -> list[dict]:
    """Attach source product profile fields to source registry entries."""

    profiled: list[dict] = []
    for entry in entries:
        row = dict(entry)
        profile = profile_source_product(row, state)
        for field in PROFILE_FIELDS:
            row[field] = profile[field]
        profiled.append(row)
    return profiled


def source_product_priority(entry: dict) -> int:
    """Lower rank means a source is more useful for fetch/extraction."""

    product_type = str(entry.get("data_product_type") or "")
    task_specificity = str(entry.get("task_specificity") or "")
    readability = str(entry.get("machine_readability") or "")
    role = str(entry.get("expected_evidence_role") or "")
    if task_specificity == "event_specific" and product_type in {
        "event_outbreak_report",
        "line_list_or_case_series",
        "case_report_article",
        "sequence_database_record",
    }:
        return 0
    if role in {"individual_case_evidence", "aggregate_event_evidence"} and readability in {
        "narrative_extractable",
        "pdf_extractable",
        "structured_table",
    }:
        return 1
    if product_type in {"news_case_report", "official_surveillance_report", "research_article"}:
        return 2
    if product_type in {"sequence_database_search", "policy_or_protocol"}:
        return 4
    if product_type == "background_fact_sheet":
        return 6
    if product_type == "search_or_index_page":
        return 7
    if product_type == "unrelated_or_other":
        return 9
    return 5


def source_product_skip_reason(entry: dict) -> str | None:
    """Return a human-readable product reason when a high-authority source is low utility."""

    product_type = str(entry.get("data_product_type") or "")
    task_specificity = str(entry.get("task_specificity") or "")
    readability = str(entry.get("machine_readability") or "")
    time_fit = str(entry.get("time_window_fit") or "")
    if product_type == "unrelated_or_other" or task_specificity == "unrelated":
        return "unrelated"
    if time_fit in {"historical_context", "future_or_out_of_scope"}:
        return "out_of_scope"
    high_utility_products = {
        "event_outbreak_report",
        "official_surveillance_report",
        "line_list_or_case_series",
        "case_report_article",
        "sequence_database_record",
        "news_case_report",
    }
    if product_type == "background_fact_sheet" or (
        task_specificity == "generic_background"
        and product_type not in high_utility_products
    ):
        return "generic_background"
    if product_type in {"search_or_index_page", "sequence_database_search"}:
        return "search_or_index_page"
    if readability == "not_extractable":
        return "not_extractable"
    return None
