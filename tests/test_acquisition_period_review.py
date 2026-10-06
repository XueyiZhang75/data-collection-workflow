"""Independent period-adapter admission and coverage boundary controls."""
from copy import deepcopy
from datetime import date

import pytest

from data_collection_workflow.evidence_qualification import (
    _constraint_reasons, _period, assess_record_evidence, qualified_coverage,
)
from test_acquisition_evidence import evidence

YEAR = {"task_period_start": "2025-01-01", "task_period_end": "2025-12-31"}


def _qualified(**values):
    return {"record_id": "r1", "disease": "measles", "country": "Canada",
            "cases_confirmed": 4,
            "evidence_qualification": {"status": "qualified", "product_kind": "aggregate"},
            **values}


@pytest.mark.parametrize("values", [
    {"date_reported": "2025-05-23"},
    {"report_date": "2025-05-23"},
    {"metric_period_start": "2025-05-23", "report_date": "2025-12-31"},
    {"reporting_period": "January to May 2025"},
    {"reporting_period": "May 2025"},
])
def test_admissible_point_or_partial_period_does_not_complete_annual_coverage(values):
    row = _qualified(**values)
    original = deepcopy(row)
    assert _constraint_reasons(row, YEAR) == []
    coverage = qualified_coverage([{"requirement_id": "year", **YEAR}], [row])
    assert coverage["coverage_complete"] is False
    assert coverage["requirements"][0]["coverage_status"] == "evidence_gap"
    assert row == original


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
def test_report_anchor_completes_only_its_single_day(field):
    row = _qualified(**{field: "2025-05-23"})
    point = {"task_period_start": "2025-05-23", "task_period_end": "2025-05-23"}
    assert qualified_coverage([point], [row])["coverage_complete"] is True
    assert qualified_coverage([{**point, "task_period_end": "2025-05-24"}], [row])["coverage_complete"] is False


def test_month_precision_union_requires_every_calendar_month():
    rows = [_qualified(record_id=f"m{month}", reporting_period=f"2025-{month:02d}")
            for month in range(1, 13)]
    original = deepcopy(rows)
    assert qualified_coverage([YEAR], rows)["coverage_complete"] is True
    assert qualified_coverage([YEAR], [r for r in rows if r["record_id"] != "m6"])["coverage_complete"] is False
    assert rows == original


@pytest.mark.parametrize("period", ["23 May 2025", "23 mai 2025", "2025-05-23"])
def test_explicit_day_label_stays_a_point(period):
    row = {"reporting_period": period, "report_date": "2025-12-31"}
    assert _period(row) == (date(2025, 5, 23), date(2025, 5, 23))


@pytest.mark.parametrize("values", [
    {"reporting_period": "May to January 2025", "date_reported": "2025-05-23"},
    {"reporting_period": "January to May", "report_date": "2025-05-23"},
    {"reporting_period": "May", "report_date": "2025-05-23"},
    {"reporting_period": "2025 season to date", "as_of_date": "2025-05-23"},
    {"metric_period_start": "2025-05", "metric_period_end": "2025-05-23", "report_date": "2025-05-23"},
    {"metric_period_end": "2025-05-23", "reporting_period": "2025"},
    {"metric_period_start": "2025-02-30", "report_date": "2025-03-01"},
    {"publication_date": "2025-05-23"},
])
def test_unresolved_observation_never_borrows_task_or_report_scope(values):
    row = _qualified(**values)
    original = deepcopy(row)
    assert _period(row) == (None, None)
    assert _constraint_reasons(row, YEAR)
    assert qualified_coverage([YEAR], [row])["coverage_complete"] is False
    assert row == original


@pytest.mark.parametrize("field", ["as_of_date", "date_onset", "date_confirmation"])
def test_report_anchor_does_not_replace_prior_year_observation(field):
    row = {field: "2024-12-31", "report_date": "2025-01-16"}
    assert _period(row) == (date(2024, 12, 31), date(2024, 12, 31))
    assert "task_period_start:contract_mismatch" in _constraint_reasons(row, YEAR)


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
def test_report_fallback_still_needs_report_role_evidence(field):
    text = "This report was published on 16 January 2025. Canada confirmed 4 measles cases."
    row, doc, chunk = evidence(text, disease="measles", country="Canada", cases_confirmed=4,
                               **{field: "2025-01-16"})
    result = assess_record_evidence(
        row, contract={"record_scope": {"disease": "measles", "location": "Canada", **YEAR}},
        evidence_index={"documents": [doc], "evidence_chunks": {"span": chunk}},
    )
    assert result.status != "qualified"
    assert any(item.field == field and not item.supported for item in result.field_evidence)
