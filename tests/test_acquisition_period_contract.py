"""Contract comparisons reuse source periods without publishing inferred boundary days."""
from copy import deepcopy
from datetime import date
import pytest
from data_collection_workflow.evidence_qualification import _period, _constraint_reasons, assess_record_evidence
from test_acquisition_evidence import evidence


FULL_YEAR = {"task_period_start": "2025-01-01", "task_period_end": "2025-12-31"}


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
@pytest.mark.parametrize("disease,country", [("measles", "Canada"), ("pertussis", "France")])
def test_supported_reported_date_satisfies_actual_task_period(field, disease, country):
    text = f"{country} reported 4 confirmed {disease} cases on 16 January 2025."
    row, doc, chunk = evidence(text, disease=disease, country=country, cases_confirmed=4, **{field: "2025-01-16"})
    result = assess_record_evidence(row, contract={"record_scope": {"disease": disease, "location": country, **FULL_YEAR}},
        evidence_index={"documents": [doc], "evidence_chunks": {"span": chunk}})
    assert all(item.supported for item in result.field_evidence), result.reasons
    assert result.status == "qualified", result.reasons


@pytest.mark.parametrize("period,start,end", [
    ("January to May 2025", date(2025, 1, 1), date(2025, 5, 31)),
    ("janvier au mai 2025", date(2025, 1, 1), date(2025, 5, 31)),
    ("January to 23 May 2025", date(2025, 1, 1), date(2025, 5, 23)),
    ("February 2024", date(2024, 2, 1), date(2024, 2, 29)),
    ("2025-05", date(2025, 5, 1), date(2025, 5, 31)),
])
def test_source_month_precision_has_conservative_comparison_bounds_without_mutation(period, start, end):
    row = {"reporting_period": period}
    original = deepcopy(row)
    assert _period(row) == (start, end)
    assert row == original
    assert "metric_period_start" not in row and "metric_period_end" not in row


@pytest.mark.parametrize("observation", [
    {"metric_period_start": "2024-01-01", "metric_period_end": "2024-12-31"},
    {"reporting_period": "January to May 2024"},
    {"as_of_date": "2024-12-31"},
    {"date_onset": "2024-12-31"},
    {"date_confirmation": "2024-12-31"},
])
def test_observation_period_is_not_overridden_by_later_report_date(observation):
    row = {**observation, "date_reported": "2025-01-16", "report_date": "2025-01-16"}
    start, end = _period(row)
    assert start and end and start.year == end.year == 2024
    assert "task_period_start:contract_mismatch" in _constraint_reasons(row, FULL_YEAR)


@pytest.mark.parametrize("row", [
    {"publication_date": "2025-01-16"},
    {"reporting_period": "January to May"},
    {"reporting_period": "January to May", "date_reported": "2025-05-31"},
    {"date_reported": "2025-02-30"},
    {"reporting_period": "2025-13"},
    {"reporting_period": "February 30 to March 2 2025"},
])
def test_unresolved_or_invalid_source_period_cannot_borrow_task_or_report_year(row):
    assert _period(row) == (None, None)
    reasons = _constraint_reasons(row, FULL_YEAR)
    assert "task_period_start:contract_mismatch" in reasons
    assert "task_period_end:contract_mismatch" in reasons


@pytest.mark.parametrize("period,contract", [
    ("December 2024 to January 2025", FULL_YEAR),
    ("January 2025", {"task_period_start": "2025-01-10", "task_period_end": "2025-01-20"}),
])
def test_conservative_interval_must_fit_whole_requested_window(period, contract):
    assert _constraint_reasons({"reporting_period": period}, contract)


def test_supported_natural_period_satisfies_contract_without_inventing_exact_fact_dates():
    text = "Canada reported 4 confirmed measles cases from January to May 2025."
    row, doc, chunk = evidence(text, disease="measles", country="Canada", reporting_period="January to May 2025", cases_confirmed=4)
    original = deepcopy(row)
    result = assess_record_evidence(row, contract={"record_scope": {"disease": "measles", "location": "Canada", **FULL_YEAR}},
        evidence_index={"documents": [doc], "evidence_chunks": {"span": chunk}})
    assert all(item.supported for item in result.field_evidence), result.reasons
    assert result.status == "qualified", result.reasons
    assert row == original


@pytest.mark.parametrize("row", [
    {"metric_period_start": "2025-05-23", "metric_period_end": "2025-01-01"},
    {"reporting_period": "May to January 2025"},
    {"reporting_period": "2025 season to date", "report_date": "2025-05-23"},
])
def test_reversed_or_open_ended_source_period_is_not_a_supported_interval(row):
    assert _period(row) == (None, None)


def test_report_date_and_partial_explicit_start_remain_points_not_full_years():
    assert _period({"report_date": "2025-05-23"}) == (date(2025, 5, 23), date(2025, 5, 23))
    assert _period({"metric_period_start": "2025-01-10", "report_date": "2025-05-23"}) == (date(2025, 1, 10), date(2025, 1, 10))
    assert _period({"metric_period_end": "2025-05-23", "report_date": "2025-05-23"}) == (None, None)
