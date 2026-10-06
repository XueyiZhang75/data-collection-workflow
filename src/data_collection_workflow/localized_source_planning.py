"""Localized multilingual source-planning helpers.

The helper is deterministic and advisory: it only creates bounded search-query
hints. It does not crawl, fetch, or add discovered source URLs.
"""

from __future__ import annotations

import re
from typing import Any

_DEFAULT_EXPECTED_FIELDS = [
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
    "deaths",
    "date_reported",
    "subnational_location",
    "source_url",
    "source_type",
    "evidence_quote",
]

_SHANGHAI_LOCATION_TERMS = ["上海", "上海市", "Shanghai"]
_SHANGHAI_MATCH_TERMS = ["shanghai", "上海", "上海市"]

_HANTAVIRUS_DISEASE_TERMS = [
    "汉坦病毒",
    "肾综合征出血热",
    "流行性出血热",
    "HFRS",
    "hantavirus",
    "hemorrhagic fever with renal syndrome",
]
_HANTAVIRUS_MATCH_TERMS = [
    "hantavirus",
    "hantavirus disease",
    "hfrs",
    "hemorrhagic fever with renal syndrome",
    "汉坦病毒",
    "肾综合征出血热",
    "流行性出血热",
]

_SHANGHAI_OFFICIAL_AGENCY_TERMS = [
    "上海市卫生健康委员会",
    "上海市疾病预防控制中心",
    "中国疾病预防控制中心",
    "国家卫生健康委员会",
    "Shanghai Municipal Health Commission",
    "Shanghai CDC",
    "China CDC",
    "National Health Commission",
]

_SHANGHAI_OFFICIAL_DOMAINS = [
    "wsjkw.sh.gov.cn",
    "shcdc.sh.cn",
    "chinacdc.cn",
    "nhc.gov.cn",
]

_US_HANTAVIRUS_DISEASE_TERMS = [
    "hantavirus",
    "HPS",
    "hantavirus pulmonary syndrome",
]

_US_STATE_OFFICIAL_SOURCE_HINTS = {
    "new mexico": {
        "jurisdiction": "New Mexico / United States",
        "location_terms": ["New Mexico", "NM"],
        "official_agency_terms": [
            "New Mexico Department of Health",
            "NMDOH",
            "New Mexico Department of Health Epidemiology",
        ],
        "official_domains": ["nmhealth.org"],
    },
}


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _joined_terms(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            for nested in value.values():
                parts.extend(_as_str_list(nested))
        else:
            parts.extend(_as_str_list(value))
    return "\n".join(parts).casefold()


def _years_from_window(*values: Any) -> list[str]:
    joined = " ".join(_as_str_list(values))
    years = [int(match.group(0)) for match in re.finditer(r"\b(?:19|20)\d{2}\b", joined)]
    if len(years) >= 2:
        start, end = min(years), max(years)
        if 1900 <= start <= end <= 2100 and end - start <= 10:
            return [str(year) for year in range(start, end + 1)]
    return _unique([str(year) for year in years])


def _disabled_summary(reason: str) -> dict:
    return {
        "enabled": False,
        "location_detected": False,
        "jurisdiction": None,
        "localized_language_count": 0,
        "localized_query_count": 0,
        "official_domain_hint_count": 0,
        "official_domain_hints": [],
        "localized_disease_terms": [],
        "localized_location_terms": [],
        "official_agency_terms": [],
        "query_language_hints": [],
        "warnings": [reason],
        "planned_query_specs": [],
    }


def _query_spec(
    *,
    query: str,
    disease_terms: list[str],
    location_terms: list[str],
    time_terms: list[str],
    official_domain_hint: str | None,
    query_language: str,
    jurisdiction_hint: str = "Shanghai / China",
    source_priority_reason: str | None = None,
    priority: int = 1,
) -> dict:
    reason = source_priority_reason or (
        "Localized official source planning prioritizes Shanghai / China "
        "public-health agencies and Chinese HFRS terminology before broad web "
        "or news queries."
    )
    return {
        "query": query,
        "query_type": "official_site" if official_domain_hint else "general_web",
        "provider_channel": "official_site_search" if official_domain_hint else "web_search",
        "source_type": "official_public_health_agency",
        "role_hint": "collection",
        "priority": priority,
        "expected_fields": list(_DEFAULT_EXPECTED_FIELDS),
        "disease_terms_used": list(disease_terms),
        "location_terms_used": list(location_terms),
        "time_terms_used": list(time_terms),
        "query_language": query_language,
        "jurisdiction_hint": jurisdiction_hint,
        "official_domain_hint": official_domain_hint,
        "localized_source_hint": True,
        "source_priority_reason": reason,
        "rationale": (
            "Localized official source planning. "
            f"jurisdiction_hint={jurisdiction_hint}; query_language={query_language}; "
            f"official_domain_hint={official_domain_hint or 'none'}; "
            "localized_source_hint=true; "
            f"source_priority_reason={reason}"
        ),
    }


def _shanghai_hantavirus_query_specs(years: list[str]) -> list[dict]:
    years = years or ["2024", "2025", "2026"]
    first_year = years[0]
    second_year = years[1] if len(years) > 1 else first_year
    last_year = years[-1]
    return [
        _query_spec(
            query=f"site:wsjkw.sh.gov.cn 肾综合征出血热 上海 {first_year}",
            disease_terms=["肾综合征出血热"],
            location_terms=["上海"],
            time_terms=[first_year],
            official_domain_hint="wsjkw.sh.gov.cn",
            query_language="zh",
            priority=1,
        ),
        _query_spec(
            query=f"site:wsjkw.sh.gov.cn 汉坦病毒 上海 {first_year}",
            disease_terms=["汉坦病毒"],
            location_terms=["上海"],
            time_terms=[first_year],
            official_domain_hint="wsjkw.sh.gov.cn",
            query_language="zh",
            priority=1,
        ),
        _query_spec(
            query="site:shcdc.sh.cn 肾综合征出血热 上海",
            disease_terms=["肾综合征出血热"],
            location_terms=["上海"],
            time_terms=[],
            official_domain_hint="shcdc.sh.cn",
            query_language="zh",
            priority=1,
        ),
        _query_spec(
            query="site:chinacdc.cn 肾综合征出血热 上海",
            disease_terms=["肾综合征出血热"],
            location_terms=["上海"],
            time_terms=[],
            official_domain_hint="chinacdc.cn",
            query_language="zh",
            priority=1,
        ),
        _query_spec(
            query="site:nhc.gov.cn 肾综合征出血热 法定传染病",
            disease_terms=["肾综合征出血热"],
            location_terms=["中国"],
            time_terms=[],
            official_domain_hint="nhc.gov.cn",
            query_language="zh",
            priority=1,
        ),
        _query_spec(
            query=f"上海 肾综合征出血热 病例 {first_year}",
            disease_terms=["肾综合征出血热"],
            location_terms=["上海"],
            time_terms=[first_year],
            official_domain_hint=None,
            query_language="zh",
            priority=2,
        ),
        _query_spec(
            query=f"上海 汉坦病毒 病例 {first_year}",
            disease_terms=["汉坦病毒"],
            location_terms=["上海"],
            time_terms=[first_year],
            official_domain_hint=None,
            query_language="zh",
            priority=2,
        ),
        _query_spec(
            query=f"上海市 法定传染病 肾综合征出血热 {second_year}",
            disease_terms=["肾综合征出血热", "法定传染病"],
            location_terms=["上海市"],
            time_terms=[second_year],
            official_domain_hint=None,
            query_language="zh",
            priority=2,
        ),
        _query_spec(
            query=f"Shanghai HFRS hemorrhagic fever with renal syndrome {last_year}",
            disease_terms=["HFRS", "hemorrhagic fever with renal syndrome"],
            location_terms=["Shanghai"],
            time_terms=[last_year],
            official_domain_hint=None,
            query_language="en",
            priority=3,
        ),
    ]


def _us_state_source_hint(location_text: str) -> dict | None:
    for state_name, hint in _US_STATE_OFFICIAL_SOURCE_HINTS.items():
        location_terms = [state_name, *hint.get("location_terms", [])]
        if any(str(term).casefold() in location_text for term in location_terms):
            return hint
    return None


def _us_state_hantavirus_query_specs(hint: dict, years: list[str]) -> list[dict]:
    years = years or ["2024", "2025", "2026"]
    year_terms = [years[0]] if len(years) == 1 else [years[0], years[-1]]
    first_year = years[0]
    last_year = years[-1]
    jurisdiction = str(hint["jurisdiction"])
    location = str(hint["location_terms"][0])
    agency = str(hint["official_agency_terms"][0])
    abbreviation = str(hint["official_agency_terms"][1])
    domain = str(hint["official_domains"][0])
    reason = (
        "Localized official source planning prioritizes United States state "
        "health department sources for subnational hantavirus surveillance "
        "before broad web or news queries."
    )
    return [
        _query_spec(
            query=f'site:{domain} hantavirus "{location}" cases {first_year}',
            disease_terms=["hantavirus"],
            location_terms=[location],
            time_terms=[first_year],
            official_domain_hint=domain,
            query_language="en",
            jurisdiction_hint=jurisdiction,
            source_priority_reason=reason,
            priority=1,
        ),
        _query_spec(
            query=f'site:{domain} HPS "{location}" deaths {last_year}',
            disease_terms=["HPS"],
            location_terms=[location],
            time_terms=[last_year],
            official_domain_hint=domain,
            query_language="en",
            jurisdiction_hint=jurisdiction,
            source_priority_reason=reason,
            priority=1,
        ),
        _query_spec(
            query=f'"{agency}" hantavirus cases deaths {first_year}',
            disease_terms=["hantavirus"],
            location_terms=[location],
            time_terms=[first_year],
            official_domain_hint=None,
            query_language="en",
            jurisdiction_hint=jurisdiction,
            source_priority_reason=reason,
            priority=2,
        ),
        _query_spec(
            query=f'"{abbreviation}" hantavirus HPS {last_year}',
            disease_terms=["hantavirus", "HPS"],
            location_terms=[location],
            time_terms=[last_year],
            official_domain_hint=None,
            query_language="en",
            jurisdiction_hint=jurisdiction,
            source_priority_reason=reason,
            priority=2,
        ),
        _query_spec(
            query=(
                f'"{location}" "hantavirus pulmonary syndrome" '
                f'"Department of Health" {" ".join(year_terms)}'
            ),
            disease_terms=["hantavirus pulmonary syndrome"],
            location_terms=[location],
            time_terms=year_terms,
            official_domain_hint=None,
            query_language="en",
            jurisdiction_hint=jurisdiction,
            source_priority_reason=reason,
            priority=3,
        ),
    ]


def build_localized_source_planning_hints(
    *,
    structured_task: dict | None = None,
    collection_spec: dict | None = None,
    disease_intelligence: dict | None = None,
    preferred_source_categories: list[str] | None = None,  # noqa: ARG001
) -> dict:
    """Return deterministic localized official-source hints for known jurisdictions."""

    structured_task = structured_task or {}
    collection_spec = collection_spec or {}
    disease_intelligence = disease_intelligence or {}

    location_text = _joined_terms(
        structured_task.get("location"),
        collection_spec.get("geography"),
    )
    disease_text = _joined_terms(
        structured_task.get("disease"),
        collection_spec.get("disease"),
        disease_intelligence.get("disease_standard_name"),
        disease_intelligence.get("aliases"),
        disease_intelligence.get("abbreviations"),
        disease_intelligence.get("pathogen_terms"),
        disease_intelligence.get("syndrome_terms"),
        disease_intelligence.get("suggested_query_terms"),
    )
    shanghai_detected = any(
        term.casefold() in location_text for term in _SHANGHAI_MATCH_TERMS
    )
    us_state_hint = _us_state_source_hint(location_text)
    location_detected = shanghai_detected or us_state_hint is not None
    disease_detected = any(term.casefold() in disease_text for term in _HANTAVIRUS_MATCH_TERMS)

    if not location_detected:
        return _disabled_summary("no_localized_jurisdiction_hints_available")
    if not disease_detected:
        summary = _disabled_summary("localized_jurisdiction_present_but_disease_not_supported")
        summary["location_detected"] = True
        summary["jurisdiction"] = (
            str(us_state_hint["jurisdiction"]) if us_state_hint else "Shanghai / China"
        )
        return summary

    years = _years_from_window(
        structured_task.get("start_date"),
        structured_task.get("end_date"),
        collection_spec.get("time_window"),
        collection_spec.get("start_date"),
        collection_spec.get("end_date"),
    )
    if us_state_hint:
        query_specs = _us_state_hantavirus_query_specs(us_state_hint, years)
        return {
            "enabled": True,
            "location_detected": True,
            "jurisdiction": str(us_state_hint["jurisdiction"]),
            "localized_language_count": 1,
            "localized_query_count": len(query_specs),
            "official_domain_hint_count": len(us_state_hint["official_domains"]),
            "official_domain_hints": list(us_state_hint["official_domains"]),
            "localized_disease_terms": _unique(_US_HANTAVIRUS_DISEASE_TERMS),
            "localized_location_terms": list(us_state_hint["location_terms"]),
            "official_agency_terms": list(us_state_hint["official_agency_terms"]),
            "query_language_hints": ["en"],
            "warnings": [],
            "planned_query_specs": query_specs,
        }

    query_specs = _shanghai_hantavirus_query_specs(years)
    return {
        "enabled": True,
        "location_detected": True,
        "jurisdiction": "Shanghai / China",
        "localized_language_count": 2,
        "localized_query_count": len(query_specs),
        "official_domain_hint_count": len(_SHANGHAI_OFFICIAL_DOMAINS),
        "official_domain_hints": list(_SHANGHAI_OFFICIAL_DOMAINS),
        "localized_disease_terms": _unique(_HANTAVIRUS_DISEASE_TERMS),
        "localized_location_terms": list(_SHANGHAI_LOCATION_TERMS),
        "official_agency_terms": list(_SHANGHAI_OFFICIAL_AGENCY_TERMS),
        "query_language_hints": ["zh", "en"],
        "warnings": [],
        "planned_query_specs": query_specs,
    }


def public_summary(hints: dict | None) -> dict:
    """Strip bulky query specs from the summary stored in workflow diagnostics."""

    hints = hints or _disabled_summary("no_localized_jurisdiction_hints_available")
    return {
        key: value
        for key, value in hints.items()
        if key != "planned_query_specs"
    }
