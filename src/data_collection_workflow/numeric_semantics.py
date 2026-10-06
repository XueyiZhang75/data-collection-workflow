"""Deterministic numeric semantics guardrails for case/death fields."""

from __future__ import annotations

from typing import Any


CASE_FIELDS = (
    "cases_confirmed",
    "cases_probable",
    "cases_suspected",
    "cases_unspecified",
)
DEATH_FIELDS = ("deaths",)

CASE_LABELS = (
    "case",
    "cases",
    "confirmed",
    "probable",
    "suspected",
    "reported case",
    "reported cases",
    "human case",
    "human cases",
)
DEATH_LABELS = ("death", "deaths", "fatality", "fatalities", "died")
NON_CASE_NUMBER_MARKERS = (
    "county",
    "counties",
    "state",
    "states",
    "country",
    "countries",
    "week",
    "weeks",
    "percentage",
    "percent",
    "%",
    "age",
    "aged",
    "table",
    "row",
    "page",
    "year",
    "population",
    "hospital",
    "hospitalization",
    "vector",
    "animal",
    "rodent",
)


def _text(record: dict) -> str:
    fields = (
        "evidence_quote",
        "evidence_context",
        "metric_name",
        "metric_category",
        "metric_unit",
        "count_semantics",
        "source_column_label",
        "metric_column_label",
        "source_column_labels",
        "table_header",
        "heading_context",
        "count_notes",
        "source_row_id",
    )
    return " ".join(str(record.get(field) or "") for field in fields).lower()


def _has_label(text: str, field: str) -> bool:
    labels = DEATH_LABELS if field in DEATH_FIELDS else CASE_LABELS
    return any(label in text for label in labels)


def _has_contextual_non_case_number(text: str) -> bool:
    return any(marker in text for marker in NON_CASE_NUMBER_MARKERS)


def _append_unique(values: list[str], item: str) -> list[str]:
    if item not in values:
        values.append(item)
    return values


def sanitize_case_death_numeric_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Clear case/death fields whose evidence lacks a case/death label.

    This does not decide whether generic metrics are useful; it only prevents
    contextual numbers such as counties, weeks, percentages, page numbers, ages,
    and populations from being accepted as case/death counts.
    """

    out = dict(record)
    text = _text(out)
    warnings = list(out.get("semantic_warnings") or [])
    for field in (*CASE_FIELDS, *DEATH_FIELDS):
        if out.get(field) in (None, ""):
            continue
        if _has_label(text, field):
            continue
        _append_unique(warnings, "case_death_count_requires_explicit_label")
        if _has_contextual_non_case_number(text):
            out[field] = None
            _append_unique(warnings, f"numeric_semantics_rejected:{field}")
    out["semantic_warnings"] = warnings
    return out
