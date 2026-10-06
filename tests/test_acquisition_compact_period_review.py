"""Independent compact-period scope and offset controls."""
import pytest
from data_collection_workflow.source_assertions import date_intervals, observation_dates

@pytest.mark.parametrize('text',[
    'Updates were issued on 17 and 23 March 2025.',
    'Le rapport est disponible le 17 et le 23 mars 2025.',
    '2025\u201313\u201317 au 23 mars 2025',
    '2025\u201413\u201417 au 23 mars 2025',
])
def test_discrete_dates_and_malformed_date_fragments_are_not_contiguous_periods(text):
    assert date_intervals(text) == []
    assert 'metric_period_start' not in observation_dates(text)
    assert 'metric_period_end' not in observation_dates(text)

@pytest.mark.parametrize('text',[
    'Canada reported 4 cases between 17 and 23 March 2025.',
    'Canada reported 4 cases from 17 to 23 March 2025.',
    'Le Canada a déclaré 4 cas du 17 au 23 mars 2025.',
    '17-23 March 2025',
])
def test_explicit_date_intervals_still_match(text):
    assert date_intervals(text) == [((2025,3,17),(2025,3,23))]

@pytest.mark.parametrize('text,quote',[
    ('Rapport : du 17 au 23 février 2024, quatre cas.', '17 au 23 février 2024'),
    ('During 17\u201323 March 2025: counts.', '17\u201323 March 2025'),
])
def test_projection_preserves_exact_source_quote(text,quote):
    result=observation_dates(text)
    assert result['date_text_span'] == quote
    assert result['reporting_period'] == quote
    assert quote in text
