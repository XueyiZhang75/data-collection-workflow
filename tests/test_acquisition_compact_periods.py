"""Explicit same-month day ranges preserve source dates and source text."""
import pytest

from data_collection_workflow.source_assertions import date_intervals, observation_dates, typed_date_support
from data_collection_workflow.evidence_qualification import assess_record_evidence
from test_acquisition_evidence import evidence


@pytest.mark.parametrize('source,label', [
    ('17 au 23 mars 2025', '17-23 mars 2025'),
    ('17 to 23 March 2025', '17-23 March 2025'),
    ('17\u201323 March 2025', '17 to 23 March 2025'),
    ('17th through 23rd March 2025', '17-23 March 2025'),
    ('17 au 23 f\u00e9vrier 2024', '17-23 fevrier 2024'),
])
def test_same_month_interval_equivalence_preserves_exact_source_quote(source, label):
    month = 2 if 'vrier' in source else 3
    year = 2024 if month == 2 else 2025
    expected = ((year, month, 17), (year, month, 23))
    text = f'France reported four cases from {source}.'
    assert date_intervals(text) == [expected]
    assert date_intervals(label) == [expected]
    result = observation_dates(text)
    assert result['reporting_period'] == source
    assert result['date_text_span'] == source
    assert result['metric_period_start'] == f'{year}-{month:02d}-17'
    assert result['metric_period_end'] == f'{year}-{month:02d}-23'
    assert typed_date_support('reporting_period', label, text) is True


@pytest.mark.parametrize('source', [
    '32 au 23 mars 2025',
    '24 au 23 mars 2025',
    '29 au 30 f\u00e9vrier 2025',
    '31 au 2 mars 2025',
    '17 au 23 mars',
    '17.5 au 23 mars 2025',
    '2025-13-17 au 23 mars 2025',
    '2025-03-17 au 2025-02-30',
])
def test_invalid_reversed_yearless_and_iso_fragment_ranges_are_not_inferred(source):
    assert date_intervals(source) == []
    assert 'metric_period_start' not in observation_dates(source)
    assert 'metric_period_end' not in observation_dates(source)


@pytest.mark.parametrize('source,expected', [
    ('28 February to 2 March 2025', ((2025, 2, 28), (2025, 3, 2))),
    ('30 December 2024 to 2 January 2025', ((2024, 12, 30), (2025, 1, 2))),
    ('2025-03-17 to 2025-03-23', ((2025, 3, 17), (2025, 3, 23))),
    ('January to 23 May 2025', ((2025, 1, None), (2025, 5, 23))),
])
def test_explicit_cross_month_iso_and_month_precision_are_preserved(source, expected):
    assert date_intervals(source) == [expected]


@pytest.mark.parametrize('disease,country,text,period,count', [
    ('measles', 'Canada', 'Canada reported 3011 confirmed measles cases from 17 to 23 March 2025.', '17-23 March 2025', 3011),
    ('chikungunya', 'France', 'France a signal\u00e9 4156 cas confirm\u00e9s de chikungunya du 17 au 23 mars 2025.', '17-23 mars 2025', 4156),
])
def test_source_supported_compact_period_qualifies_without_task_inference(disease, country, text, period, count):
    row, doc, chunk = evidence(text, disease=disease, country=country, reporting_period=period, cases_confirmed=count)
    result = assess_record_evidence(row,
        contract={'record_scope': {'disease': disease, 'location': country, 'task_period_start': '2025-01-01', 'task_period_end': '2025-12-31'}},
        evidence_index={'documents': [doc], 'evidence_chunks': {'span': chunk}})
    assert result.status == 'qualified', result.reasons
    assert 'metric_period_start' not in row
    assert 'metric_period_end' not in row


def test_observation_day_range_does_not_establish_event_start_or_end():
    source = 'Canada reported 3011 confirmed measles cases from 17 to 23 March 2025.'
    assert typed_date_support('event_start_date', '2025-03-17', source) is False
    assert typed_date_support('event_end_date', '2025-03-23', source) is False
