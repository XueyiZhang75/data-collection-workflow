"""Exact counts and annual totals cannot be inferred from bounded observations."""
import pytest
from test_acquisition_evidence import assess


@pytest.mark.parametrize('expression,value', [
    ('18-20',20), ('18\u201320',20), ('between 18 and 20',20),
    ('18 to 20',20), ('>= 20',20), ('at least 20',20),
])
def test_count_range_or_lower_bound_cannot_become_exact(expression,value):
    text=f'Canada reported {expression} confirmed pertussis cases during 2025.'
    q=assess(text,disease='pertussis',country='Canada',reporting_period='2025',cases_confirmed=value)
    assert q.status=='candidate',q.to_dict()
    assert not next(field.supported for field in q.field_evidence if field.field=='cases_confirmed')


@pytest.mark.parametrize('semantics', ['statistical_count_type','count_semantics'])
@pytest.mark.parametrize('period', ['Between January and 23 May 2025', 'As of 23 May 2025'])
def test_partial_period_cannot_claim_annual_total(semantics,period):
    text=f'{period}, Canada reported 3,011 confirmed pertussis cases.'
    q=assess(text,disease='pertussis',country='Canada',reporting_period='2025',cases_confirmed=3011,**{semantics:'annual'})
    assert q.status=='candidate',q.to_dict()


@pytest.mark.parametrize('text',[
    'Canada reported an annual total of 3,011 confirmed pertussis cases during 2025.',
    'From 1 January to 31 December 2025, Canada reported 3,011 confirmed pertussis cases.',
])
def test_explicit_full_year_annual_count_stays_qualified(text):
    q=assess(text,disease='pertussis',country='Canada',reporting_period='2025',cases_confirmed=3011,statistical_count_type='annual')
    assert q.status=='qualified',q.to_dict()
