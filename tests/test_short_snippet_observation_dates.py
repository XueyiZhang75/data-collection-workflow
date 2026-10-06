"""Current short metadata: event dates cannot close an observation interval."""
import pytest
from data_collection_workflow.nodes.source_screening import _source_reporting_date_fit

@pytest.fixture(autouse=True)
def evidence(monkeypatch):monkeypatch.setenv('PIPELINE_MODE','evidence')

def task(disease='measles',place='Canada'):
    return {'structured_task':{'disease':disease,'location':place,'start_date':'2025-01-01','end_date':'2025-12-31'}}

@pytest.mark.parametrize('disease,place',[('measles','Canada'),('dengue','Brazil')])
def test_emergency_declaration_date_is_not_an_observation_period(disease,place):
    text=f'On 14 August 2024, the health authority declared the {disease} outbreak a Public Health Emergency.'
    assert _source_reporting_date_fit([text],task(disease,place))=='candidate'

@pytest.mark.parametrize('disease,place',[('measles','Canada'),('dengue','Brazil')])
def test_outbreak_start_cannot_date_recent_undated_count_trend(disease,place):
    text=f'The {disease} outbreak in {place} began in August 2024, but the number of cases has increased sharply in recent weeks.'
    assert _source_reporting_date_fit([text],task(disease,place))=='candidate'

@pytest.mark.parametrize('text',[
    'In 2024, Canada reported 12 measles cases.',
    'In 2024, measles cases increased sharply during recent weeks.',
])
def test_explicit_observation_year_still_outside_task(text):
    assert _source_reporting_date_fit([text],task())=='mismatch'

def test_explicit_current_task_year_still_matches():
    assert _source_reporting_date_fit(['Canada reported 12 measles cases during 2025.'],task())=='match'
