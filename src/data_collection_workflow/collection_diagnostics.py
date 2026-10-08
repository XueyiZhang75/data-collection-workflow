"""Deterministic source-corroboration labels for report facts."""

from __future__ import annotations

import re


_MISSING = {"", "none", "null", "nan", "n/a", "na", "unknown", "publisher_unknown"}
_OFFICIAL_SOURCE_TYPES = {
    "official_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "international_public_health_agency",
    "international_organization_report",
    "structured_database",
}


def classify_single_source_record(record: dict) -> str:
    """Return a credibility-aware single-source taxonomy label."""

    status = _norm(record.get("corroboration_status"))
    if status in {"corroborated", "cross_source_supported", "multi_source_corroborated"}:
        return "multi_source_corroborated"
    if status in {"conflicting_claims", "conflicting_sources"}:
        return "conflicting_sources"
    if record.get("requires_human_review") or record.get("human_review_reason"):
        return "needs_human_review"
    has_provenance = _has_value(record.get("source_url")) and _has_value(
        record.get("evidence_quote")
    )
    publisher = _publisher(record)
    source_type = _norm(record.get("source_type_final") or record.get("source_type"))
    official = source_type in _OFFICIAL_SOURCE_TYPES or _known_official_publisher(
        publisher
    )
    if has_provenance and official:
        return "official_single_source_not_cross_validated"
    return "non_official_single_source_unverified"


def _publisher(record: dict) -> str:
    for key in ("actual_publisher", "publisher", "source_publisher"):
        value = _norm(record.get(key))
        if value and value not in _MISSING:
            return value
    return ""


def _known_official_publisher(publisher: str) -> bool:
    return publisher in {
        "centers for disease control and prevention",
        "world health organization",
        "pan american health organization",
        "european centre for disease prevention and control",
        "new mexico department of health",
        "california department of public health",
        "virginia department of health",
    }


def _norm(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    return " ".join(text.split())


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _norm(value) not in _MISSING
    return True
