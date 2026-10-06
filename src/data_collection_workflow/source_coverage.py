"""Task-aware official source coverage requirements.

This module is deliberately deterministic. It protects task-critical official
sources from later LLM/source-role routing mistakes and gives diagnostics a
stable coverage table to audit against.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import date, datetime
from hashlib import sha256
from urllib.parse import urlsplit


_US_STATE_OFFICIAL_DOMAINS = {
    "united_states": {
        "slug": "united_states",
        "canonical_location": "United States",
        "aliases": {
            "united states",
            "united states of america",
            "usa",
            "us",
            "u.s.",
            "u.s.a.",
        },
        "official_domains": ["cdc.gov"],
        "agency": "Centers for Disease Control and Prevention",
        "report_title_hints": [
            "FluView",
            "Weekly US Influenza Surveillance Report",
            "Key Updates",
        ],
    },
    "virginia": {
        "slug": "virginia",
        "canonical_location": "Virginia",
        "aliases": {"virginia", "va"},
        "official_domains": ["vdh.virginia.gov"],
        "agency": "Virginia Department of Health",
        "report_title_hints": [
            "Weekly-RDS-Report",
            "Respiratory Disease Surveillance",
        ],
    },
    "new_york": {
        "slug": "new_york",
        "canonical_location": "New York",
        "aliases": {"new york", "new york state", "ny", "nys"},
        "official_domains": [
            "health.ny.gov",
            "health.state.ny.us",
            "nyshc.health.ny.gov",
        ],
        "agency": "New York State Department of Health",
        "report_title_hints": [
            "New York State Influenza Surveillance Report",
            "flu_report",
            "Respiratory Surveillance and Reports",
            "New York State Flu Tracker",
        ],
    },
}

_INFLUENZA_TERMS = {"flu", "influenza", "seasonal influenza"}
_GENERIC_METRIC_CATEGORIES = [
    "case_count",
    "death_count",
    "incidence_rate",
    "mortality_rate",
    "hospitalization_count",
    "hospitalization_rate",
    "lab_test_count",
    "lab_positive_count",
    "lab_positivity_percent",
    "ili_percent",
    "ed_visit_percent",
    "outbreak_count",
    "treatment_coverage_percent",
    "treatment_success_percent",
    "vaccination_coverage_percent",
    "public_health_metric",
]
_STRICT_FINAL_CONDITIONS = [
    "task_disease_match",
    "task_geography_match",
    "task_period_or_explicit_reporting_period_match",
    "interpretable_public_health_metric",
    "source_provenance_verified",
    "trusted_or_human_reviewed_source_provenance",
    "evidence_quote_or_source_row_binding",
]
_BEST_AVAILABLE_CONDITIONS = [
    "wrong_period_or_broader_than_task",
    "near_match_wrong_or_broader_period",
    "broader_than_task_geography_context",
    "season_or_multi_year_context_for_short_window",
]
_HUMAN_REVIEW_CONDITIONS = [
    "source_trust_boundary",
    "low_or_social_source_with_task_metric",
    "high_impact_source_aware_anomaly",
    "unresolved_period_or_column_semantics_but_potentially_useful",
    "borderline_source_trust",
]
_LOCATION_ALIAS_MAP = {
    "german": "Germany",
    "germany": "Germany",
    "deutschland": "Germany",
    "indian": "India",
    "india": "India",
    "american": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "british": "United Kingdom",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
}
_NY_FLU_REPORT_RE = re.compile(
    r"/influenza/surveillance/(?P<season>20\d{2}-(?:\d{2}|20\d{2}))/"
    r"archive/(?P<date>20\d{2}-\d{2}-\d{2})_flu_report\.pdf$",
    re.IGNORECASE,
)
_VDH_RDS_WEEK_RE = re.compile(
    r"weekly-rds-report[_-]week[_-]?(?P<week>\d{1,2})\.pdf$",
    re.IGNORECASE,
)
_CDC_FLUVIEW_WEEK_RE = re.compile(
    r"/fluview/surveillance/(?P<year>20\d{2})-week-(?P<week>\d{1,2})\.html$",
    re.IGNORECASE,
)
_AUTHORITY_OFFICIAL_SOURCE_TYPES = {
    "official_public_health_agency",
    "international_organization_report",
    "international_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "state_public_health_agency",
    "local_public_health_agency",
    "government_report",
}
_AUTHORITY_STRUCTURED_SOURCE_TYPES = {
    "structured_database",
    "public_health_dataset",
    "surveillance_database",
    "dashboard",
}
_AUTHORITY_PEER_SOURCE_TYPES = {
    "academic_or_peer_reviewed_source",
    "peer_reviewed_literature",
    "journal_article",
    "scientific_literature",
}
_AUTHORITY_NEWS_SOURCE_TYPES = {
    "news_media",
    "news",
    "media_report",
    "news_and_situation_report",
    "secondary_media",
}
_AUTHORITY_TRACKER_CONTEXT_SOURCE_TYPES = {
    "secondary_aggregator",
    "search_endpoint",
    "background_fact_sheet",
    "public_health_context_page",
    "social_media",
    "personal_blog_or_forum",
    "commercial_site",
    "unknown",
}


def _lower(value) -> str:
    return str(value or "").strip().lower()


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _domain_for_source(entry: dict) -> str:
    direct = str(entry.get("domain") or "").strip().lower()
    if direct:
        return direct.removeprefix("www.")
    url = str(entry.get("canonical_url") or entry.get("url") or "").strip()
    if not url:
        return ""
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _authority_source_type(entry: dict) -> str:
    domain = _domain_for_source(entry)
    registry_entry = _domain_registry_entry(domain)
    if registry_entry and registry_entry.get("source_type"):
        return _lower(registry_entry.get("source_type"))

    final_type = _lower(entry.get("source_type_final"))
    if final_type:
        if final_type in {"unknown", "unclassified", "unspecified"}:
            if _domain_looks_public_health_authority(domain):
                return "national_public_health_agency"
            return final_type
        if (
            final_type in _AUTHORITY_OFFICIAL_SOURCE_TYPES
            and not _domain_looks_public_health_authority(domain)
        ):
            return "unknown"
        if final_type in _AUTHORITY_STRUCTURED_SOURCE_TYPES:
            if _domain_looks_structured_database(domain):
                return final_type
            return "unknown"
        if final_type in _AUTHORITY_PEER_SOURCE_TYPES:
            if _domain_looks_peer_scientific(domain):
                return final_type
            return "unknown"
        return final_type

    source_type = _lower(entry.get("source_type") or entry.get("planned_query_source_type"))
    if source_type in _AUTHORITY_OFFICIAL_SOURCE_TYPES:
        if _domain_looks_public_health_authority(domain):
            return source_type
        return "unknown"
    if source_type in _AUTHORITY_STRUCTURED_SOURCE_TYPES:
        if _domain_looks_structured_database(domain):
            return source_type
        return "unknown"
    if source_type in _AUTHORITY_PEER_SOURCE_TYPES:
        if _domain_looks_peer_scientific(domain):
            return source_type
        return "unknown"
    if source_type:
        return source_type
    if _domain_looks_public_health_authority(domain):
        return "national_public_health_agency"
    return "unknown"


def _domain_looks_public_health_authority(domain: str) -> bool:
    if not domain:
        return False
    domain = domain.lower().removeprefix("www.")
    if any(
        domain == known or domain.endswith("." + known)
        for known in (
            "cdc.gov",
            "who.int",
            "ecdc.europa.eu",
            "paho.org",
            "gov.uk",
            "canada.ca",
            "rivm.nl",
            "sanidad.gob.es",
            "sante.gouv.fr",
        )
    ):
        return True
    if domain.endswith((".gov", ".gov.uk", ".gob.es", ".gouv.fr")) and any(
        token in domain for token in ("health", "doh", "dhhs", "cdc", "sanidad", "sante")
    ):
        return True
    return False


def _domain_registry_entry(domain: str | None) -> dict | None:
    if not domain:
        return None
    try:
        from .source_identity import lookup_source_identity_registry

        return lookup_source_identity_registry(domain)
    except Exception:
        return None


def _registry_source_type(domain: str | None) -> str:
    registry_entry = _domain_registry_entry(domain)
    if registry_entry and registry_entry.get("source_type"):
        return _lower(registry_entry.get("source_type"))
    return ""


def _domain_looks_structured_database(domain: str) -> bool:
    source_type = _registry_source_type(domain)
    if source_type in _AUTHORITY_STRUCTURED_SOURCE_TYPES:
        return True
    domain = (domain or "").lower().removeprefix("www.")
    return any(
        domain == known or domain.endswith("." + known)
        for known in (
            "pathoplexus.org",
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "europepmc.org",
            "data.cdc.gov",
            "healthdata.gov",
        )
    )


def _domain_looks_peer_scientific(domain: str) -> bool:
    source_type = _registry_source_type(domain)
    if source_type in _AUTHORITY_PEER_SOURCE_TYPES:
        return True
    domain = (domain or "").lower().removeprefix("www.")
    return any(
        domain == known or domain.endswith("." + known)
        for known in (
            "nejm.org",
            "eurosurveillance.org",
            "science.org",
            "cidrap.umn.edu",
        )
    )


def _domain_jurisdiction_scope(domain: str | None) -> str:
    registry_entry = _domain_registry_entry(domain)
    if registry_entry and registry_entry.get("jurisdiction_scope"):
        return _lower(registry_entry.get("jurisdiction_scope"))
    normalized = (domain or "").lower().removeprefix("www.")
    if normalized in {"who.int", "paho.org", "ecdc.europa.eu"}:
        return "regional" if normalized != "who.int" else "international"
    if _domain_looks_public_health_authority(normalized):
        return "national"
    return ""


def _domain_jurisdiction_name(domain: str | None) -> str:
    registry_entry = _domain_registry_entry(domain)
    if registry_entry and registry_entry.get("jurisdiction_name"):
        return str(registry_entry.get("jurisdiction_name") or "").strip()
    return ""


def _normalize_jurisdiction(value: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    aliases = {
        "uk": "united kingdom",
        "u k": "united kingdom",
        "usa": "united states",
        "us": "united states",
        "u s": "united states",
    }
    return aliases.get(text, text)


_EVENT_JURISDICTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Canada", ("canada", "canadian")),
    ("France", ("france", "french")),
    ("Spain", ("spain", "spanish", "madrid")),
    ("Netherlands", ("netherlands", "dutch", "holland")),
    ("United Kingdom", ("united kingdom", "uk", "u.k.", "britain", "british")),
    ("South Africa", ("south africa", "south african", "johannesburg")),
    ("Argentina", ("argentina", "argentine", "ushuaia")),
    ("Cabo Verde", ("cabo verde", "cape verde")),
    ("Saint Helena", ("saint helena", "st helena")),
    ("Switzerland", ("switzerland", "swiss", "zurich")),
    ("Germany", ("germany", "german")),
    ("Singapore", ("singapore",)),
    ("Italy", ("italy", "italian", "milan")),
    ("United States", ("united states", "usa", "u.s.", "american")),
)


def _source_event_text(entry: dict) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in (
            "title",
            "snippet",
            "description",
            "canonical_url",
            "url",
            "query_used",
            "matched_terms",
        )
    ).lower()


def _jurisdictions_from_source_text(registry: list[dict]) -> list[str]:
    detected: list[str] = []
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        text = _source_event_text(entry)
        if not text:
            continue
        for display, aliases in _EVENT_JURISDICTION_ALIASES:
            for alias in aliases:
                if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", text):
                    detected.append(display)
                    break
    return _unique_preserve_order(detected)


def _row_source_id(row: dict) -> str:
    return str(row.get("source_id") or "").strip()


def _stage_source_sets(
    registry: list[dict],
    documents: list[dict] | None,
    evidence_rows: list[dict] | None,
) -> tuple[set[str], set[str], set[str], set[str]]:
    fetched: set[str] = set()
    parsed: set[str] = set()
    extracted: set[str] = set()
    retained_source_only: set[str] = set()
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        source_id = _row_source_id(entry)
        if not source_id:
            continue
        if _lower(entry.get("fetch_status")) == "fetched":
            fetched.add(source_id)
        parse_status = _lower(entry.get("parse_status"))
        if parse_status.startswith("parsed") or parse_status in {"fixture_loaded"}:
            parsed.add(source_id)
        try:
            extracted_count = int(entry.get("extracted_record_count") or 0)
        except (TypeError, ValueError):
            extracted_count = 0
        if extracted_count > 0:
            extracted.add(source_id)
    for doc in documents or []:
        if not isinstance(doc, dict):
            continue
        source_id = _row_source_id(doc)
        if not source_id:
            continue
        if _lower(doc.get("fetch_status")) == "fetched":
            fetched.add(source_id)
        parse_status = _lower(doc.get("parse_status"))
        if parse_status.startswith("parsed") or parse_status in {"fixture_loaded"}:
            parsed.add(source_id)
    for row in evidence_rows or []:
        if not isinstance(row, dict):
            continue
        source_id = _row_source_id(row)
        if not source_id:
            continue
        if _lower(row.get("quality_status")) == "source_only":
            retained_source_only.add(source_id)
            continue
        if row.get("record_id") or row.get("case_candidate_id") or row.get("evidence_quote"):
            extracted.add(source_id)
    return fetched, parsed, extracted, retained_source_only


def _authority_source_family(entry: dict) -> str:
    source_type = _authority_source_type(entry)
    if source_type in _AUTHORITY_OFFICIAL_SOURCE_TYPES:
        return "official_or_public_health"
    if source_type in _AUTHORITY_STRUCTURED_SOURCE_TYPES:
        return "structured_database"
    if source_type in _AUTHORITY_PEER_SOURCE_TYPES:
        return "peer_reviewed_or_scientific"
    if source_type in _AUTHORITY_NEWS_SOURCE_TYPES:
        return "news_media"
    if source_type in _AUTHORITY_TRACKER_CONTEXT_SOURCE_TYPES:
        return "unknown_tracker_context"
    return "unknown_tracker_context"


def _jurisdictions_from_verified_authority_domains(registry: list[dict]) -> list[str]:
    jurisdictions: list[str] = []
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        domain = _domain_for_source(entry)
        if not domain or _authority_source_family(entry) != "official_or_public_health":
            continue
        scope = _domain_jurisdiction_scope(domain)
        if scope not in {"national", "subnational", "state", "local"}:
            continue
        jurisdiction_name = (
            str(entry.get("jurisdiction_name") or "").strip()
            or _domain_jurisdiction_name(domain)
        )
        if jurisdiction_name:
            jurisdictions.append(jurisdiction_name)
    return _unique_preserve_order(jurisdictions)


def build_authority_source_coverage_summary(
    registry: list[dict],
    *,
    detected_event_jurisdictions: list[str] | None = None,
    documents: list[dict] | None = None,
    evidence_rows: list[dict] | None = None,
) -> dict:
    """Summarize high-trust source-class recall without gating final inclusion."""

    counts = {
        "official_or_public_health": 0,
        "structured_database": 0,
        "peer_reviewed_or_scientific": 0,
        "news_media": 0,
        "unknown_tracker_context": 0,
    }
    source_ids = {key: [] for key in counts}
    domains = {key: [] for key in counts}
    international_or_regional_authority_count = 0
    national_or_subnational_authority_count = 0
    fetched_source_ids, parsed_source_ids, extracted_source_ids, retained_source_only_ids = (
        _stage_source_sets(registry, documents, evidence_rows)
    )
    stage_aware = (
        documents is not None
        or evidence_rows is not None
        or bool(fetched_source_ids or parsed_source_ids or extracted_source_ids)
    )
    text_jurisdictions = _jurisdictions_from_source_text(registry)
    inferred_jurisdictions = (
        _unique_preserve_order(
            _jurisdictions_from_verified_authority_domains(registry)
            + text_jurisdictions
        )
        if detected_event_jurisdictions is None
        else []
    )
    detected_jurisdictions = _unique_preserve_order(
        [
            str(value or "").strip()
            for value in (detected_event_jurisdictions or inferred_jurisdictions)
            if str(value or "").strip()
        ]
    )
    detected_jurisdiction_keys = {
        _normalize_jurisdiction(value): value
        for value in detected_jurisdictions
        if _normalize_jurisdiction(value)
    }
    covered_jurisdiction_keys: set[str] = set()
    for entry in registry or []:
        if not isinstance(entry, dict):
            continue
        family = _authority_source_family(entry)
        counts[family] += 1
        source_id = entry.get("source_id")
        if source_id:
            source_ids[family].append(source_id)
        domain = _domain_for_source(entry)
        if domain and domain not in domains[family]:
            domains[family].append(domain)
        if family != "official_or_public_health":
            continue
        scope = _domain_jurisdiction_scope(domain)
        if scope in {"international", "regional"}:
            international_or_regional_authority_count += 1
        elif scope in {"national", "subnational", "state", "local"}:
            national_or_subnational_authority_count += 1
        elif _domain_looks_public_health_authority(domain):
            national_or_subnational_authority_count += 1

        jurisdiction_name = (
            str(entry.get("jurisdiction_name") or "").strip()
            or _domain_jurisdiction_name(domain)
        )
        jurisdiction_key = _normalize_jurisdiction(jurisdiction_name)
        stage_covers_jurisdiction = True
        if stage_aware:
            source_id = _row_source_id(entry)
            stage_covers_jurisdiction = bool(
                source_id
                and (
                    source_id in parsed_source_ids
                    or source_id in extracted_source_ids
                )
            )
        if (
            jurisdiction_key
            and jurisdiction_key in detected_jurisdiction_keys
            and stage_covers_jurisdiction
        ):
            covered_jurisdiction_keys.add(jurisdiction_key)

    authority_total = (
        counts["official_or_public_health"]
        + counts["structured_database"]
        + counts["peer_reviewed_or_scientific"]
    )
    covered_event_jurisdictions = [
        display
        for key, display in detected_jurisdiction_keys.items()
        if key in covered_jurisdiction_keys
    ]
    missing_event_jurisdictions = [
        display
        for key, display in detected_jurisdiction_keys.items()
        if key not in covered_jurisdiction_keys
    ]
    if detected_jurisdictions and missing_event_jurisdictions:
        event_jurisdiction_authority_coverage_status = (
            "event_jurisdiction_authority_missing"
        )
    elif detected_jurisdictions:
        event_jurisdiction_authority_coverage_status = (
            "event_jurisdiction_authority_complete"
        )
    else:
        event_jurisdiction_authority_coverage_status = "not_applicable"

    warnings: list[str] = []
    if not registry:
        coverage_status = "no_sources_discovered"
    elif authority_total == 0:
        coverage_status = "authority_source_recall_incomplete"
        warnings.append("authority_source_recall_incomplete")
    elif (
        counts["official_or_public_health"] == 0
        or counts["structured_database"] == 0
        or counts["peer_reviewed_or_scientific"] == 0
    ):
        coverage_status = "authority_source_recall_partial"
        warnings.append("authority_source_class_gap")
        if counts["structured_database"] == 0:
            warnings.append("structured_database_source_missing")
        if counts["peer_reviewed_or_scientific"] == 0:
            warnings.append("peer_reviewed_source_missing")
    else:
        coverage_status = "authority_source_recall_sufficient"

    if missing_event_jurisdictions:
        warnings.append("event_jurisdiction_authority_missing")
        if stage_aware and national_or_subnational_authority_count > 0:
            warnings.append("event_jurisdiction_authority_not_extracted")
        if coverage_status == "authority_source_recall_sufficient":
            coverage_status = "authority_source_recall_partial"
        elif coverage_status == "no_sources_discovered":
            warnings.append("authority_source_recall_incomplete")

    if stage_aware and authority_total > 0 and not extracted_source_ids:
        warnings.append("authority_source_stage_gap")
        if coverage_status == "authority_source_recall_sufficient":
            coverage_status = "authority_source_recall_partial"

    if counts["official_or_public_health"] == 0 and registry:
        warnings.append("official_public_health_source_missing")
    elif detected_jurisdictions and national_or_subnational_authority_count == 0:
        warnings.append("national_or_subnational_authority_source_missing")
    if (
        counts["structured_database"] == 0
        and counts["peer_reviewed_or_scientific"] == 0
        and registry
    ):
        warnings.append("structured_or_peer_reviewed_source_missing")

    return {
        "coverage_status": coverage_status,
        "source_count": len(registry or []),
        "discovered_count": len(registry or []),
        "fetched_count": len(fetched_source_ids),
        "parsed_count": len(parsed_source_ids),
        "extracted_evidence_count": len(extracted_source_ids),
        "retained_source_only_count": len(retained_source_only_ids),
        "official_or_public_health_source_count": counts["official_or_public_health"],
        "structured_database_source_count": counts["structured_database"],
        "peer_reviewed_or_scientific_source_count": counts[
            "peer_reviewed_or_scientific"
        ],
        "international_or_regional_authority_source_count": (
            international_or_regional_authority_count
        ),
        "national_or_subnational_authority_source_count": (
            national_or_subnational_authority_count
        ),
        "news_media_source_count": counts["news_media"],
        "unknown_tracker_context_source_count": counts["unknown_tracker_context"],
        "authority_source_count": authority_total,
        "detected_event_jurisdictions": detected_jurisdictions,
        "covered_event_jurisdictions": covered_event_jurisdictions,
        "missing_event_jurisdictions": missing_event_jurisdictions,
        "event_jurisdiction_authority_coverage_status": (
            event_jurisdiction_authority_coverage_status
        ),
        "source_ids_by_family": source_ids,
        "domains_by_family": domains,
        "warnings": _unique_preserve_order(warnings),
    }


def _as_date(value) -> date | None:
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


def _task_field(state: dict, key: str, *fallbacks: str) -> str:
    structured = state.get("structured_task") or {}
    collection = state.get("collection_spec") or {}
    for name in (key, *fallbacks):
        value = structured.get(name)
        if value not in (None, ""):
            return str(value)
        value = collection.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _canonical_location_label(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _LOCATION_ALIAS_MAP.get(_lower(text), text)


def _state_profile(location: str) -> dict | None:
    loc = _lower(location)
    for profile in _US_STATE_OFFICIAL_DOMAINS.values():
        if loc in profile["aliases"]:
            return profile
    return None


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _lower(value)).strip("_")
    return text or "unknown"



def _annual_periods_for_range(start: date, end: date) -> list[tuple[int, date, date]]:
    """Return full natural-year periods covered by an inclusive or exclusive range.

    User-facing date ranges often express a natural year either as
    2023-01-01..2023-12-31 or as the half-open interval
    2023-01-01..2024-01-01. Treating the latter as a generic task window makes
    otherwise equivalent annual requests behave differently.
    """

    if end < start:
        start, end = end, start
    if start.month != 1 or start.day != 1:
        return []
    if end.month == 12 and end.day == 31:
        final_year = end.year
    elif end.month == 1 and end.day == 1 and end.year > start.year:
        final_year = end.year - 1
    else:
        return []
    if final_year < start.year:
        return []
    return [
        (year, date(year, 1, 1), date(year, 12, 31))
        for year in range(start.year, final_year + 1)
    ]


def _generic_requirement(
    *,
    disease_label: str,
    canonical_location: str,
    disease_slug: str,
    location_slug: str,
    period_start: date,
    period_end: date,
    period_basis: str,
    label: str,
    requirement_suffix: str,
) -> dict:
    time_granularity = _time_granularity_for_period_basis(period_basis)
    return {
        "requirement_id": f"{location_slug}_{disease_slug}_{requirement_suffix}",
        "disease": disease_label.lower(),
        "location": canonical_location,
        "geography": canonical_location,
        "year": period_start.year if period_start.year == period_end.year else None,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "reporting_period_start": period_start.isoformat(),
        "reporting_period_end": period_end.isoformat(),
        "reporting_period_label": label,
        "period_basis": period_basis,
        "time_granularity": time_granularity,
        "source_type": "task_relevant_public_health_evidence",
        "official_domains": [],
        "accepted_source_roles": [
            "official_public_health_agency",
            "national_public_health_agency",
            "state_or_local_public_health_agency",
            "international_public_health_agency",
            "academic_or_peer_reviewed_source",
            "public_health_dataset",
            "task_record_collection_candidate",
        ],
        "accepted_metric_categories": list(_GENERIC_METRIC_CATEGORIES),
        "accepted_metric_families": list(_GENERIC_METRIC_CATEGORIES),
        "strict_final_conditions": list(_STRICT_FINAL_CONDITIONS),
        "best_available_conditions": list(_BEST_AVAILABLE_CONDITIONS),
        "human_review_conditions": list(_HUMAN_REVIEW_CONDITIONS),
        "official_candidate_urls": [],
        "agency": "",
        "title_hints": [
            disease_label,
            canonical_location,
            str(period_start.year),
            str(period_end.year),
            "surveillance",
            "report",
            "dashboard",
            "data",
            "epidemiology",
            "statistics",
        ],
        "reason": (
            f"Target {canonical_location} {disease_label} task requires "
            f"task-relevant public health evidence for {label}."
        ),
    }


def _time_granularity_for_period_basis(period_basis: str) -> str:
    basis = str(period_basis or "").strip().lower()
    if basis == "week_ending_saturday":
        return "weekly"
    if basis == "annual":
        return "annual"
    if basis:
        return basis
    return "task_window"


def _is_influenza_task(disease: str, state: dict) -> bool:
    text = " ".join(
        [
            disease,
            str((state.get("structured_task") or {}).get("user_request") or ""),
            str((state.get("collection_spec") or {}).get("user_request") or ""),
        ]
    ).lower()
    return any(term in text for term in _INFLUENZA_TERMS)


def _week_numbers_between(start: date, end: date) -> list[tuple[int, int]]:
    if end < start:
        start, end = end, start
    weeks: set[tuple[int, int]] = set()
    current = start
    while current <= end:
        # CDC FluView, state influenza reports, and many US respiratory
        # surveillance reports use a week-ending-Saturday convention. Using the
        # Saturday anchor keeps a Sunday-Saturday task window such as
        # 2024-09-29..2024-10-05 attached to Week 40, not split into ISO weeks
        # 39 and 40.
        days_until_saturday = (5 - current.weekday()) % 7
        week_anchor = date.fromordinal(current.toordinal() + days_until_saturday)
        iso = week_anchor.isocalendar()
        weeks.add((iso.year, iso.week))
        current = date.fromordinal(current.toordinal() + 1)
    return sorted(weeks)


def _dates_for_iso_week(year: int, week: int) -> list[str]:
    try:
        monday = date.fromisocalendar(int(year), int(week), 1)
    except (TypeError, ValueError):
        return []
    return [
        date.fromordinal(monday.toordinal() + offset).isoformat()
        for offset in range(7)
    ]


def _week_ending_saturday(year: int, week: int) -> date | None:
    try:
        return date.fromisocalendar(int(year), int(week), 6)
    except (TypeError, ValueError):
        return None


def _week_reporting_period(year: int, week: int) -> tuple[str | None, str | None, str | None]:
    ending = _week_ending_saturday(year, week)
    if not ending:
        return None, None, None
    start = date.fromordinal(ending.toordinal() - 6)
    return (
        start.isoformat(),
        ending.isoformat(),
        f"MMWR week {int(week)}, {int(year)}",
    )


def _influenza_season_for_week(year: int, week: int) -> str:
    ending = _week_ending_saturday(year, week)
    if not ending:
        return f"{year}-{str(year + 1)[-2:]}"
    if ending.month >= 7:
        return f"{ending.year}-{str(ending.year + 1)[-2:]}"
    return f"{ending.year - 1}-{str(ending.year)[-2:]}"


def _official_candidate_urls(profile: dict, year: int, week: int) -> list[str]:
    slug = str(profile.get("slug") or "")
    ending = _week_ending_saturday(year, week)
    if slug == "united_states":
        return [
            f"https://www.cdc.gov/fluview/surveillance/{year}-week-{week}.html"
        ]
    if slug == "new_york" and ending:
        season = _influenza_season_for_week(year, week)
        return [
            (
                "https://www.health.ny.gov/diseases/communicable/influenza/"
                f"surveillance/{season}/archive/{ending.isoformat()}_flu_report.pdf"
            )
        ]
    if slug == "virginia" and ending:
        season = _influenza_season_for_week(year, week)
        month = f"{ending.month:02d}"
        return [
            (
                "https://www.vdh.virginia.gov/content/uploads/sites/13/"
                f"{ending.year}/{month}/Weekly-RDS-Report_Week-{week}.pdf"
            ),
            (
                "https://www.vdh.virginia.gov/content/uploads/sites/3/"
                f"{ending.year}/{month}/{season}_Weekly-RDS-Report_Week-{week}.pdf"
            ),
        ]
    return []


def _build_period_coverage_requirements(state: dict) -> list[dict]:
    """Build task-specific source coverage requirements for a run."""

    disease = _task_field(state, "disease")
    location = _canonical_location_label(_task_field(state, "location", "geography"))
    start = _as_date(_task_field(state, "start_date"))
    end = _as_date(_task_field(state, "end_date")) or start
    state_profile = _state_profile(location)
    if not start or not end:
        return []
    if not state_profile or not _is_influenza_task(disease, state):
        disease_slug = _slug(disease)
        location_slug = _slug(location)
        canonical_location = str(location or "").strip() or location_slug
        disease_label = str(disease or "").strip() or disease_slug
        requirements: list[dict] = []
        annual_periods = _annual_periods_for_range(start, end)
        if annual_periods:
            for year, period_start, period_end in annual_periods:
                requirements.append(
                    _generic_requirement(
                        disease_label=disease_label,
                        canonical_location=canonical_location,
                        disease_slug=disease_slug,
                        location_slug=location_slug,
                        period_start=period_start,
                        period_end=period_end,
                        period_basis="annual",
                        label=str(year),
                        requirement_suffix=f"annual_{year}",
                    )
                )
            return requirements

        period_start = min(start, end)
        period_end = max(start, end)
        requirements.append(
            _generic_requirement(
                disease_label=disease_label,
                canonical_location=canonical_location,
                disease_slug=disease_slug,
                location_slug=location_slug,
                period_start=period_start,
                period_end=period_end,
                period_basis="task_window",
                label=f"{period_start.isoformat()} to {period_end.isoformat()}",
                requirement_suffix=(
                    "task_window_"
                    f"{period_start.isoformat().replace('-', '_')}_"
                    f"{period_end.isoformat().replace('-', '_')}"
                ),
            )
        )
        return requirements

    requirements: list[dict] = []
    for year, week in _week_numbers_between(start, end):
        slug = state_profile.get("slug") or _lower(location).replace(" ", "_")
        canonical_location = state_profile.get("canonical_location") or location
        report_hints = list(state_profile.get("report_title_hints") or [])
        candidate_urls = _official_candidate_urls(state_profile, year, week)
        period_start, period_end, period_label = _week_reporting_period(year, week)
        requirements.append(
            {
                "requirement_id": f"{slug}_influenza_official_week_{week}_{year}",
                "disease": "influenza",
                "location": canonical_location,
                "geography": canonical_location,
                "year": year,
                "week": week,
                "date_hints": _dates_for_iso_week(year, week),
                "period_start": period_start,
                "period_end": period_end,
                "reporting_period_start": period_start,
                "reporting_period_end": period_end,
                "reporting_period_label": period_label,
                "period_basis": "week_ending_saturday",
                "time_granularity": "weekly",
                "source_type": "official_weekly_surveillance_report",
                "official_domains": list(state_profile["official_domains"]),
                "accepted_metric_categories": list(_GENERIC_METRIC_CATEGORIES),
                "accepted_metric_families": list(_GENERIC_METRIC_CATEGORIES),
                "strict_final_conditions": list(_STRICT_FINAL_CONDITIONS),
                "best_available_conditions": list(_BEST_AVAILABLE_CONDITIONS),
                "human_review_conditions": list(_HUMAN_REVIEW_CONDITIONS),
                "official_candidate_urls": candidate_urls,
                "agency": state_profile["agency"],
                "title_hints": [
                    f"Week-{week}",
                    f"Week {week}",
                    *[url.rsplit("/", 1)[-1] for url in candidate_urls],
                    *report_hints,
                ],
                "reason": (
                    f"Target {canonical_location} seasonal influenza task requires "
                    f"the {state_profile['agency']} weekly surveillance report "
                    f"for week {week}, {year}."
                ),
            }
        )
    return requirements


_REQUESTED_METRIC_FAMILIES = {
    "cases_confirmed": "case_count", "cases_probable": "case_count",
    "cases_suspected": "case_count", "cases_unspecified": "case_count",
    "deaths": "death_count", "hospitalizations": "hospitalization_count",
    "icu_admissions": "hospitalization_count", "tests_positive": "lab_test_count",
    "tests_total": "lab_test_count", "positivity_rate": "positivity_rate",
    "incidence_rate": "incidence_rate", "cumulative_count": "public_health_metric",
    "new_count": "public_health_metric",
}


def _requested_metric_fields(state: dict) -> list[str]:
    task = state.get("structured_task") or {}
    spec = state.get("collection_spec") or {}
    fields = (task.get("target_fields") if "target_fields" in task else spec.get("target_fields")) or []
    return list(dict.fromkeys(str(field).strip().casefold() for field in fields
                             if str(field).strip().casefold() in _REQUESTED_METRIC_FAMILIES))


def build_source_coverage_requirements(state: dict) -> list[dict]:
    requirements = _build_period_coverage_requirements(state)
    from .query_policy import universal_queries_enabled
    if not universal_queries_enabled():
        return requirements
    # A supported individual case is retained as a case product; it does not
    # establish an aggregate total for the requested population and period.
    requirements = [{**row, "accepted_product_kinds": ["aggregate"]} for row in requirements]
    fields = _requested_metric_fields(state)
    if not fields:
        return requirements
    # Separate obligations let independently qualified observations supply
    # different requested metrics without requiring every row to contain all.
    return [{**requirement,
             "requirement_id": requirement["requirement_id"] + "_metric_" + field,
             "required_metric_fields": [field],
             "accepted_metric_categories": [_REQUESTED_METRIC_FAMILIES[field]],
             "accepted_metric_families": [_REQUESTED_METRIC_FAMILIES[field]]}
            for requirement in requirements for field in fields]


def build_task_evidence_contract(state: dict) -> dict:
    """Build the generic evidence contract shared by discovery, extraction, and gate.

    The contract is intentionally a thin deterministic layer over coverage
    requirements. Its job is to keep every downstream agent aligned to the same
    disease/location/time/metric semantics without encoding source-specific
    shortcuts as the workflow's primary mechanism.
    """

    requirements = build_source_coverage_requirements(state)
    disease = _task_field(state, "disease")
    location = _canonical_location_label(_task_field(state, "location", "geography"))
    start = _as_date(_task_field(state, "start_date"))
    end = _as_date(_task_field(state, "end_date")) or start
    period_bases = {
        str(req.get("period_basis") or "task_window")
        for req in requirements
        if isinstance(req, dict)
    }
    if not period_bases:
        time_granularity = "unknown"
    elif len(period_bases) == 1:
        only = next(iter(period_bases))
        if only == "week_ending_saturday":
            time_granularity = "weekly"
        elif only == "annual":
            time_granularity = "annual"
        else:
            time_granularity = only
    else:
        time_granularity = "mixed"
    contract = {
        "contract_version": "hdc_task_evidence_contract_v1",
        "disease": str(disease or "").strip(),
        "location": str(location or "").strip(),
        "time_granularity": time_granularity,
        "task_period_start": start.isoformat() if start else None,
        "task_period_end": end.isoformat() if end else None,
        "requirements": [dict(row) for row in requirements],
        "accepted_metric_families": list(dict.fromkeys(
            family for requirement in requirements
            for family in requirement.get("accepted_metric_families", []))) or list(_GENERIC_METRIC_CATEGORIES),
        "strict_final_conditions": list(_STRICT_FINAL_CONDITIONS),
        "best_available_conditions": list(_BEST_AVAILABLE_CONDITIONS),
        "human_review_conditions": list(_HUMAN_REVIEW_CONDITIONS),
        "partial_output_allowed": True,
    }
    from .query_policy import universal_queries_enabled
    if universal_queries_enabled():
        # Admission preserves each source-supported observation in task scope.
        # Coverage requirements remain separate period/metric obligations.
        contract["record_scope"] = {
            "disease": contract["disease"],
            "location": contract["location"],
            "task_period_start": contract["task_period_start"],
            "task_period_end": contract["task_period_end"],
        }
    return contract


def build_official_coverage_candidates(state: dict) -> list[dict]:
    """Build deterministic target official source candidates for a task."""

    candidates: list[dict] = []
    for requirement in build_source_coverage_requirements(state):
        urls = list(requirement.get("official_candidate_urls") or [])
        for index, url in enumerate(urls, start=1):
            canonical = str(url).strip()
            if not canonical:
                continue
            digest = sha256(canonical.encode("utf-8")).hexdigest()[:12]
            filename = canonical.rsplit("/", 1)[-1]
            candidates.append(
                {
                    "source_id": f"src_official_{digest}",
                    "title": (
                        f"{requirement.get('agency')} {requirement.get('source_type')} "
                        f"week {requirement.get('week')}, {requirement.get('year')}"
                    ),
                    "url": canonical,
                    "canonical_url": canonical,
                    "publisher": requirement.get("agency"),
                    "source_type": "official_public_health_agency",
                    "published_date": (
                        (requirement.get("date_hints") or [None])[-2]
                        if len(requirement.get("date_hints") or []) >= 2
                        else (requirement.get("date_hints") or [None])[-1]
                    ),
                    "snippet": requirement.get("reason"),
                    "query_used": None,
                    "retrieved_at": None,  # Generated URL; no retrieval has occurred.
                    "query_id": requirement.get("requirement_id"),
                    "discovery_method": "official_coverage_requirement",
                    "priority": index - 1,
                    "expected_fields": [
                        "cases",
                        "deaths",
                        "hospitalizations",
                        "tests_positive",
                        "reporting_period",
                        "source_url",
                    ],
                    "matched_terms": [
                        "influenza",
                        str(requirement.get("location") or ""),
                        f"week {requirement.get('week')}",
                        str(requirement.get("year") or ""),
                        filename,
                    ],
                    "source_purpose": "target_official_surveillance_report",
                    "notes": requirement.get("reason"),
                    "provider_channel": "official_site_search",
                    "role_hint": "collection",
                    "planned_query_id": requirement.get("requirement_id"),
                    "planned_query_source_type": requirement.get("source_type"),
                    "domain": _domain_for_url(canonical),
                    "query_type": "deterministic_official_url",
                    "query_source": "source_coverage_requirement",
                    "source_disease_relevance_status": "target_disease_match",
                    "source_disease_relevance_score": 1.0,
                    "source_target_disease_terms_found": ["influenza", "flu"],
                    "source_disease_relevance_reason": requirement.get("reason"),
                    "source_disease_relevance_data_signal_count": 1,
                    "coverage_requirement_ids": [requirement.get("requirement_id")],
                    "must_fetch": True,
                    "must_fetch_reason": requirement.get("reason"),
                    "reporting_period_start": requirement.get("reporting_period_start"),
                    "reporting_period_end": requirement.get("reporting_period_end"),
                    "reporting_period_label": requirement.get("reporting_period_label"),
                    "period_basis": requirement.get("period_basis"),
                }
            )
    return candidates


def _domain_for_url(url: str) -> str:
    domain = urlsplit(str(url or "")).netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


def official_report_key_for_url(url: str | None) -> str | None:
    """Return a stable key for equivalent target official report URLs.

    Some state health sites expose the same influenza PDF under both short and
    long season aliases (for example NYSDOH `2024-25` and `2024-2025`). The
    workflow should treat those as one official report so fetch/extraction
    diagnostics do not contradict themselves.
    """

    if not url:
        return None
    parts = urlsplit(str(url).strip())
    domain = parts.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path = parts.path.lower()
    if domain in {"health.ny.gov", "health.state.ny.us", "nyshc.health.ny.gov"}:
        match = _NY_FLU_REPORT_RE.search(path)
        if match:
            return f"ny_influenza_weekly_report:{match.group('date')}"
    if domain == "vdh.virginia.gov":
        match = _VDH_RDS_WEEK_RE.search(path)
        if match:
            year_match = re.search(r"/(20\d{2})/", path)
            year = year_match.group(1) if year_match else "unknown_year"
            week = int(match.group("week"))
            return f"virginia_rds_weekly_report:{year}:week_{week:02d}"
    if domain == "cdc.gov":
        match = _CDC_FLUVIEW_WEEK_RE.search(path)
        if match:
            year = int(match.group("year"))
            week = int(match.group("week"))
            return f"cdc_fluview_weekly_report:{year}:week_{week:02d}"
    return None


def _domain_for_entry(entry: dict) -> str:
    url = entry.get("canonical_url") or entry.get("url") or ""
    if not url:
        return ""
    domain = urlsplit(str(url)).netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


def _matches_domain(domain: str, official_domains: list[str]) -> bool:
    for official in official_domains:
        official = _lower(official)
        if domain == official or domain.endswith("." + official):
            return True
    return False


def _entry_text(entry: dict) -> str:
    return " ".join(
        str(entry.get(key) or "")
        for key in (
            "canonical_url",
            "url",
            "title",
            "name",
            "source_title",
            "snippet",
            "publisher",
        )
    ).lower()


def _explicit_year_week_pairs(text: str) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for match in re.finditer(
        r"\b(?P<year>20\d{2})[-_/ ]+week[-_/ ]?(?P<week>\d{1,2})\b",
        text,
        re.IGNORECASE,
    ):
        try:
            pairs.add((int(match.group("year")), int(match.group("week"))))
        except (TypeError, ValueError):
            continue
    for match in re.finditer(
        r"\bweek[-_/ ]?(?P<week>\d{1,2})\b.{0,40}?\b(?P<year>20\d{2})\b",
        text,
        re.IGNORECASE,
    ):
        try:
            pairs.add((int(match.group("year")), int(match.group("week"))))
        except (TypeError, ValueError):
            continue
    return pairs


def _explicit_years(text: str) -> set[int]:
    years: set[int] = set()
    for value in re.findall(r"\b20\d{2}\b", text or ""):
        try:
            years.add(int(value))
        except (TypeError, ValueError):
            continue
    return years


_GENERIC_TARGET_ROLES = {
    "verified_target_collection",
    "search_verified_target_collection",
    "fetch_verified_target_collection",
    "task_record_collection_candidate",
}
_GENERIC_COLLECTION_ROLES = {
    "collection",
    "data_source",
    "official_authority",
    "official_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "international_public_health_agency",
    "academic_or_peer_reviewed_source",
    "public_health_dataset",
}
_GENERIC_REJECT_FITS = {
    "mismatch",
    "wrong_period",
    "wrong_year",
    "wrong_week",
    "wrong_geography",
    "outside_scope",
    "excluded",
    "non_target",
    "context_only",
}


def _fit_rejected(value) -> bool:
    return _lower(value) in _GENERIC_REJECT_FITS


def _period_tokens_for_requirement(requirement: dict) -> set[str]:
    tokens: set[str] = set()
    for key in ("reporting_period_start", "reporting_period_end", "reporting_period_label"):
        value = str(requirement.get(key) or "").strip().lower()
        if value:
            tokens.add(value)
    year = requirement.get("year")
    if year:
        tokens.add(str(year))
    for value in (requirement.get("date_hints") or []):
        text = str(value or "").strip().lower()
        if text:
            tokens.add(text)
    return tokens


def _generic_coverage_rejection(entry: dict, requirement: dict) -> str | None:
    """A candidate task label cannot reverse an explicit relevance rejection."""
    if any(
        _lower(entry.get(key)) == "irrelevant_source"
        for key in ("source_role", "source_role_fit")
    ):
        return "irrelevant_source"
    if _lower(entry.get("source_disease_relevance_status")) in {
        "unrelated_disease", "incompatible_disease"
    }:
        return "disease_mismatch"
    if any(
        _lower(entry.get(key)) in (_GENERIC_REJECT_FITS - {"context_only"})
        for key in ("disease_fit", "geography_fit", "date_fit", "period_fit")
    ):
        return "task_fit_mismatch"
    explicitly_excluded = any(
        _lower(entry.get(key)) in {"exclude", "excluded", "do_not_fetch"}
        for key in ("screening_decision", "critic_decision")
    )
    if explicitly_excluded:
        # Keep the existing repair for a concrete target mistakenly rejected on
        # source identity/review grounds. Candidate/possible fit flags, task
        # hints and the institution's authority do not supply this evidence.
        text = _entry_text(entry)
        disease = _lower(requirement.get("disease"))
        location = _lower(requirement.get("location"))
        concrete_target = (
            bool(disease and disease in text)
            and bool(location and location in text)
            and any(token in text for token in _period_tokens_for_requirement(requirement))
        )
        if not concrete_target:
            return "explicit_exclusion_without_target_evidence"
    return None


def _matches_generic_requirement(entry: dict, requirement: dict) -> bool:
    if _generic_coverage_rejection(entry, requirement):
        return False
    labels = {
        _lower(entry.get("target_fit_status")),
        _lower(entry.get("triage_role")),
        _lower(entry.get("source_role_final")),
        _lower(entry.get("source_role")),
        _lower(entry.get("source_type")),
        _lower(entry.get("source_type_final")),
    }
    if labels & _NON_TARGET_COVERAGE_STATUSES:
        return False
    if any(
        _fit_rejected(entry.get(key))
        for key in ("disease_fit", "geography_fit", "date_fit", "period_fit")
    ):
        return False

    text = _entry_text(entry)
    disease = _lower(requirement.get("disease"))
    location = _lower(requirement.get("location"))
    disease_signal = (
        _lower(entry.get("disease_fit")) == "match"
        or any(token and token in text for token in {disease, disease.replace("flu", "influenza")})
    )
    geography_signal = (
        _lower(entry.get("geography_fit")) == "match"
        or (location and location in text)
    )
    period_signal = (
        _lower(entry.get("date_fit")) == "match"
        or any(token and token in text for token in _period_tokens_for_requirement(requirement))
    )
    role_signal = bool(labels & (_GENERIC_TARGET_ROLES | _GENERIC_COLLECTION_ROLES))

    if labels & _GENERIC_TARGET_ROLES:
        return disease_signal and geography_signal and period_signal
    return role_signal and disease_signal and geography_signal and period_signal


def _matches_requirement(entry: dict, requirement: dict) -> bool:
    if requirement.get("source_type") == "task_relevant_public_health_evidence":
        return _matches_generic_requirement(entry, requirement)
    domain = _domain_for_entry(entry)
    if not _matches_domain(domain, list(requirement.get("official_domains") or [])):
        return False
    text = _entry_text(entry)
    try:
        week_int = int(requirement.get("week"))
        year_int = int(requirement.get("year"))
    except (TypeError, ValueError):
        return False
    explicit_pairs = _explicit_year_week_pairs(text)
    if explicit_pairs and (year_int, week_int) not in explicit_pairs:
        return False
    explicit_year_values = _explicit_years(text)
    if explicit_year_values and year_int not in explicit_year_values:
        return False
    week = str(week_int)
    year = str(year_int)
    has_week = (
        f"week-{week}" in text
        or f"week_{week}" in text
        or f"week {week}" in text
        or f"week-{int(week):02d}" in text
        or f"week_{int(week):02d}" in text
        or f"week {int(week):02d}" in text
    )
    has_report = "weekly-rds-report" in text or "respiratory disease surveillance" in text
    title_hints = [
        str(value or "").strip().lower()
        for value in (requirement.get("title_hints") or [])
        if str(value or "").strip()
    ]
    has_profile_report_hint = any(hint in text for hint in title_hints)
    has_date_hint = any(
        str(value or "").strip().lower() in text
        for value in (requirement.get("date_hints") or [])
        if str(value or "").strip()
    )
    return (has_week or has_date_hint) and (
        year in text or has_report or has_profile_report_hint
    )


def _append_unique(items: list, value: str) -> None:
    if value and value not in items:
        items.append(value)


def _http_status_ok(value) -> bool:
    try:
        status = int(value)
    except (TypeError, ValueError):
        return True
    return status < 400


def _looks_like_error_page(doc: dict) -> bool:
    from .page_status import page_failure_reason
    return page_failure_reason(doc) is not None


def _fetch_succeeded(doc: dict) -> bool:
    if _lower(doc.get("fetch_status")) == "fetch_failed":
        return False
    if not _http_status_ok(doc.get("http_status_code")):
        return False
    return True


def _parse_succeeded(doc: dict) -> bool:
    if not _fetch_succeeded(doc):
        return False
    if _lower(doc.get("quality_status")) == "unusable":
        return False
    if _looks_like_error_page(doc):
        return False
    parse_status = _lower(doc.get("parse_status"))
    return parse_status not in {"", "parse_failed", "parse_deferred", "fetch_failed"}


_NON_TARGET_COVERAGE_STATUSES = {
    "best_available_context_candidate",
    "context_only",
    "wrong_period_context",
    "validation_only",
    "excluded",
}


def _entry_is_non_target_context(entry: dict) -> bool:
    labels = {
        _lower(entry.get("target_fit_status")),
        _lower(entry.get("triage_role")),
        _lower(entry.get("source_role_final")),
        _lower(entry.get("source_role")),
    }
    return bool(labels & _NON_TARGET_COVERAGE_STATUSES) or "context" in labels


def _coverage_parse_succeeded(entry: dict, doc: dict) -> bool:
    if not _parse_succeeded(doc):
        return False
    if doc.get("usable_for_task_collection") is False:
        return False
    if _entry_is_non_target_context(entry) and doc.get("usable_for_task_collection") is not True:
        return False
    return True


def _doc_text(doc: dict) -> str:
    return _lower(
        " ".join(
            str(doc.get(key) or "")
            for key in (
                "title",
                "clean_text",
                "raw_text",
                "text",
                "excerpt",
                "url",
                "canonical_url",
                "source_url",
                "reporting_period_label",
            )
        )
    )


def _iso_date(value) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0))
    except ValueError:
        return None


def _requirement_years(requirement: dict) -> set[int]:
    years: set[int] = set()
    try:
        if requirement.get("year") not in (None, ""):
            years.add(int(requirement.get("year")))
    except (TypeError, ValueError):
        pass
    for key in ("reporting_period_start", "reporting_period_end"):
        parsed = _iso_date(requirement.get(key))
        if parsed:
            years.add(parsed.year)
    return years


def _date_ranges_overlap(
    left_start: date | None,
    left_end: date | None,
    right_start: date | None,
    right_end: date | None,
) -> bool | None:
    if not (left_start and left_end and right_start and right_end):
        return None
    return left_start <= right_end and right_start <= left_end


def _entry_doc_requirement_period_mismatch(
    entry: dict,
    doc: dict,
    requirement: dict,
) -> bool:
    """Reject stale coverage ids when the fetched document clearly says another period."""

    text = f"{_entry_text(entry)} {_doc_text(doc)}"
    if requirement.get("source_type") != "task_relevant_public_health_evidence":
        try:
            week_int = int(requirement.get("week"))
            year_int = int(requirement.get("year"))
        except (TypeError, ValueError):
            week_int = None
            year_int = None
        if week_int is not None and year_int is not None:
            explicit_pairs = _explicit_year_week_pairs(text)
            if explicit_pairs and (year_int, week_int) not in explicit_pairs:
                return True
            explicit_year_values = _explicit_years(text)
            if explicit_year_values and year_int not in explicit_year_values:
                return True

    requirement_years = _requirement_years(requirement)
    explicit_year_values = _explicit_years(text)
    if explicit_year_values and requirement_years and not (explicit_year_values & requirement_years):
        return True

    req_start = _iso_date(requirement.get("reporting_period_start"))
    req_end = _iso_date(requirement.get("reporting_period_end"))
    source_start = _iso_date(
        doc.get("reporting_period_start")
        or doc.get("metric_period_start")
        or entry.get("reporting_period_start")
    )
    source_end = _iso_date(
        doc.get("reporting_period_end")
        or doc.get("metric_period_end")
        or entry.get("reporting_period_end")
    )
    overlap = _date_ranges_overlap(req_start, req_end, source_start, source_end)
    if overlap is False:
        return True
    return False


def _not_task_collection_document(entry: dict, doc: dict) -> bool:
    if not _parse_succeeded(doc):
        return False
    if doc.get("usable_for_task_collection") is False:
        return True
    return _entry_is_non_target_context(entry) and doc.get("usable_for_task_collection") is not True


def _coverage_missing_reason(row: dict) -> str | None:
    if row.get("accepted"):
        return None
    if row.get("parsed"):
        return "parsed_no_records"
    if row.get("period_mismatch"):
        return "source_period_mismatch"
    if row.get("not_task_collection_document"):
        return "no_task_collection_document"
    if row.get("unusable"):
        return "target_alias_error_page"
    if row.get("fetch_failed"):
        return "target_fetch_failed"
    if row.get("fetch_attempted") and not row.get("fetched"):
        return "target_fetch_failed"
    if row.get("discovered"):
        return "target_source_discovered_not_fetched"
    return "target_source_missing"


def _repair_must_fetch_routing(entry: dict, requirement_ids: list[str], reason: str) -> dict:
    out = dict(entry)
    warnings = list(out.get("routing_conflict_warnings") or [])
    flags = list(out.get("routing_flags") or [])
    disease_status = _lower(out.get("source_disease_relevance_status"))
    if disease_status in {"unrelated_disease", "incompatible_disease"}:
        _append_unique(warnings, "must_fetch_does_not_override_disease_mismatch")
        out.update(
            {
                "must_fetch": True,
                "must_fetch_reason": reason,
                "coverage_requirement_ids": requirement_ids,
                "routing_conflict_warnings": warnings,
            }
        )
        return out
    old_role = _lower(out.get("source_role_final"))
    old_level = _lower(out.get("credibility_level"))
    if old_role in {"excluded", "search_endpoint", "needs_human_review"}:
        _append_unique(warnings, f"source_role_final:{old_role}")
    if out.get("blocked_from_fetch"):
        _append_unique(warnings, "blocked_from_fetch:true")
    if _lower(out.get("final_screening_decision")) not in {
        "include_for_content_fetch",
        "include_for_context_fetch",
    }:
        _append_unique(
            warnings,
            f"final_screening_decision:{out.get('final_screening_decision')}",
        )
    out.update(
        {
            "must_fetch": True,
            "must_fetch_reason": reason,
            "coverage_requirement_ids": requirement_ids,
            "routing_conflict_warnings": warnings,
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "blocked_from_fetch": False,
            "blocked_from_fetch_reason": None,
            "status": "ready_for_content_fetch",
        }
    )
    if old_level in {"", "excluded", "low", "needs_review"}:
        out["credibility_level"] = "high"
    if not out.get("credibility_score"):
        out["credibility_score"] = 0.95
    _append_unique(flags, "target_official_must_fetch")
    out["routing_flags"] = flags
    return out


def build_source_coverage_audit(
    requirements: list[dict],
    registry: list[dict],
    documents: list[dict] | None = None,
    evidence_rows: list[dict] | None = None,
) -> dict:
    from .evidence_qualification import evidence_qualification_enabled, qualified_coverage
    if evidence_qualification_enabled():
        return qualified_coverage(requirements, evidence_rows or [])
    documents_provided = documents is not None
    documents = documents or []
    authority_summary = build_authority_source_coverage_summary(
        registry,
        documents=documents if documents_provided else None,
        evidence_rows=evidence_rows,
    )
    docs_by_source = {str(doc.get("source_id")): doc for doc in documents if isinstance(doc, dict)}
    rows: list[dict] = []
    for requirement in requirements:
        rid = requirement["requirement_id"]
        matches = [
            entry for entry in registry
            if rid in (entry.get("coverage_requirement_ids") or [])
        ]
        attempted = [
            entry
            for entry in matches
            if str(entry.get("source_id")) in docs_by_source
        ]
        fetched = [
            entry
            for entry in attempted
            if _fetch_succeeded(docs_by_source[str(entry.get("source_id"))])
        ]
        fetch_failed = [
            entry
            for entry in attempted
            if not _fetch_succeeded(docs_by_source[str(entry.get("source_id"))])
        ]
        period_mismatch = [
            entry
            for entry in fetched
            if _entry_doc_requirement_period_mismatch(
                entry,
                docs_by_source[str(entry.get("source_id"))],
                requirement,
            )
        ]
        period_mismatch_ids = {str(entry.get("source_id")) for entry in period_mismatch}
        parsed = [
            entry for entry in fetched
            if _coverage_parse_succeeded(entry, docs_by_source[str(entry.get("source_id"))])
            and str(entry.get("source_id")) not in period_mismatch_ids
        ]
        not_task_collection = [
            entry
            for entry in fetched
            if _not_task_collection_document(
                entry,
                docs_by_source[str(entry.get("source_id"))],
            )
        ]
        unusable = [
            entry
            for entry in attempted
            if not _coverage_parse_succeeded(
                entry,
                docs_by_source[str(entry.get("source_id"))],
            )
        ]
        row = {
                **requirement,
                "discovered": bool(matches),
                "matched_source_ids": [m.get("source_id") for m in matches],
                "fetched": bool(fetched),
                "fetched_source_ids": [m.get("source_id") for m in fetched],
                "fetch_attempted": bool(attempted),
                "fetch_attempted_source_ids": [m.get("source_id") for m in attempted],
                "fetch_failed": bool(fetch_failed),
                "fetch_failed_source_ids": [m.get("source_id") for m in fetch_failed],
                "parsed": bool(parsed),
                "parsed_source_ids": [m.get("source_id") for m in parsed],
                "period_mismatch": bool(period_mismatch),
                "period_mismatch_source_ids": [
                    m.get("source_id") for m in period_mismatch
                ],
                "not_task_collection_document": bool(not_task_collection),
                "not_task_collection_source_ids": [
                    m.get("source_id") for m in not_task_collection
                ],
                "unusable": bool(unusable),
                "unusable_source_ids": [m.get("source_id") for m in unusable],
            }
        row["missing_reason"] = _coverage_missing_reason(row)
        rows.append(row)
    requirement_count = len(requirements)
    discovered_count = sum(1 for row in rows if row["discovered"])
    fetched_count = sum(1 for row in rows if row["fetched"])
    fetch_failed_count = sum(1 for row in rows if row["fetch_failed"])
    unusable_count = sum(1 for row in rows if row["unusable"])
    period_mismatch_count = sum(1 for row in rows if row.get("period_mismatch"))
    parsed_count = sum(1 for row in rows if row["parsed"])
    # Initial source coverage is intentionally pre-extraction and therefore
    # cannot prove strict target coverage. Finalization refreshes this audit
    # with accepted exact record ids; until then parsed documents mean
    # "available for extraction", not "requirement complete".
    complete_requirement_count = 0
    partial_requirement_count = (
        requirement_count - complete_requirement_count if requirement_count else 0
    )
    missing_requirement_ids = [
        row.get("requirement_id")
        for row in rows
        if row.get("requirement_id") and not row.get("accepted")
    ]
    if not requirement_count:
        coverage_completeness_status = "not_required"
    else:
        coverage_completeness_status = "no_target_coverage"
    if not requirement_count:
        coverage_status = "not_required"
    elif parsed_count:
        coverage_status = "parsed_no_records"
    elif period_mismatch_count:
        coverage_status = "target_source_period_mismatch"
    elif fetch_failed_count and not fetched_count:
        coverage_status = "target_official_source_fetch_failed"
    elif fetched_count and unusable_count and not parsed_count:
        coverage_status = "target_official_source_unusable"
    elif fetched_count:
        coverage_status = "fetched_not_parsed"
    elif discovered_count:
        coverage_status = "target_official_source_discovered_not_fetched"
    else:
        coverage_status = "target_official_source_missing"
    return {
        "requirement_count": len(requirements),
        "discovered_requirement_count": discovered_count,
        "fetched_requirement_count": fetched_count,
        "fetch_failed_requirement_count": fetch_failed_count,
        "unusable_requirement_count": unusable_count,
        "period_mismatch_requirement_count": period_mismatch_count,
        "parsed_requirement_count": parsed_count,
        "complete_requirement_count": complete_requirement_count,
        "partial_requirement_count": partial_requirement_count,
        "missing_requirement_ids": missing_requirement_ids,
        "coverage_completeness_status": coverage_completeness_status,
        "coverage_status": coverage_status,
        "authority_source_coverage_summary": authority_summary,
        "requirements": rows,
    }


def annotate_source_coverage(
    registry: list[dict],
    state: dict,
    *,
    documents: list[dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Annotate task-critical official sources and return coverage diagnostics."""

    from .evidence_qualification import evidence_qualification_enabled

    requirements = build_source_coverage_requirements(state)
    if not requirements:
        updated = [dict(row) for row in registry]
        return updated, [], build_source_coverage_audit([], updated, documents)

    docs_by_source: dict[str, list[dict]] = {}
    for doc in documents or []:
        if isinstance(doc, dict) and doc.get("source_id"):
            docs_by_source.setdefault(str(doc["source_id"]), []).append(doc)
    updated: list[dict] = []
    for entry in registry:
        row = deepcopy(entry)
        match_entries = [dict(row)]
        for doc in docs_by_source.get(str(row.get("source_id")), []):
            if (
                not _parse_succeeded(doc)
                or doc.get("usable_for_task_collection") is False
                or doc.get("content_readable") is False
            ):
                continue
            # Evaluate each readable version independently: partial disease,
            # location and period mentions in different versions must not form
            # an artificial complete match. Metadata stays source-bound.
            match_entry = dict(row)
            match_entry["snippet"] = " ".join(
                str(value or "") for value in (
                    row.get("snippet"), doc.get("title"), doc.get("clean_text"),
                    doc.get("raw_text"), doc.get("text"), doc.get("excerpt"),
                )
            )
            match_entries.append(match_entry)
        matched = [req for req in requirements
                   if any(_matches_requirement(candidate, req) for candidate in match_entries)]
        rejections = []
        for req in requirements:
            if req.get("source_type") == "task_relevant_public_health_evidence":
                reasons = [_generic_coverage_rejection(candidate, req) for candidate in match_entries]
                rejections.append(reasons[0] if all(reasons) else None)
        if not matched and len(rejections) == len(requirements) and row.get("must_fetch"):
            # Uncertain matches remain eligible for ordinary fetching, but a
            # previous candidate label does not retain mandatory priority.
            row.update({
                "must_fetch": False,
                "must_fetch_reason": None,
                "coverage_requirement_ids": [],
                "routing_flags": [flag for flag in (row.get("routing_flags") or [])
                                  if flag != "target_official_must_fetch"],
            })
        provisional_discovery = (evidence_qualification_enabled() and
                                 row.get("task_fit_evidence_origin") == "discovery_metadata")
        if not matched and len(rejections) == len(requirements) and all(rejections) and not provisional_discovery:
            # Annotation is repeated after screening. Revoke an earlier
            # candidate-based priority so later fast paths cannot revive it.
            warnings = list(row.get("routing_conflict_warnings") or [])
            _append_unique(warnings, "must_fetch_does_not_override_task_rejection")
            row.update({
                "must_fetch": False,
                "must_fetch_reason": None,
                "coverage_requirement_ids": [],
                "source_role_final": "excluded",
                "final_screening_decision": "exclude",
                "ready_for_content_fetch": False,
                "blocked_from_fetch": True,
                "blocked_from_fetch_reason": rejections[0],
                "status": "excluded",
                "routing_conflict_warnings": warnings,
                "routing_flags": [flag for flag in (row.get("routing_flags") or [])
                                  if flag != "target_official_must_fetch"],
            })
        if matched and evidence_qualification_enabled():
            # Task fit schedules a candidate; it does not verify its publisher,
            # raise credibility, or supply the source's reporting period.
            row["coverage_requirement_ids"] = [req["requirement_id"] for req in matched]
            flags = list(row.get("routing_flags") or [])
            _append_unique(flags, "task_coverage_candidate")
            row["routing_flags"] = flags
        elif matched:
            ids = [req["requirement_id"] for req in matched]
            reason = " ".join(req["reason"] for req in matched)
            row = _repair_must_fetch_routing(row, ids, reason)
            first = matched[0]
            for key in (
                "reporting_period_start",
                "reporting_period_end",
                "reporting_period_label",
                "period_basis",
            ):
                if not row.get(key):
                    row[key] = first.get(key)
        updated.append(row)
    audit = build_source_coverage_audit(requirements, updated, documents)
    return updated, requirements, audit
