"""Retrieval periods require local observation scope, not incidental years."""
import pytest

from data_collection_workflow.nodes.source_screening import _direct_target_verification, _source_positive_target_verification
from data_collection_workflow.acquisition_scheduling import acquisition_priority


def state(disease='example fever', location='Example Region', start='2025-01-01', end='2025-12-31'):
    return {'structured_task': {'disease': disease, 'location': location, 'start_date': start, 'end_date': end}}


@pytest.mark.parametrize('disease,place', [('measles', 'Canada'), ('dengue', 'Brazil'), ('example fever', 'Example Region')])
@pytest.mark.parametrize('snippet', [
    'Wednesday, 23 September 2026\n\nHome\n\nDownload App\nCases were reported in the region.',
    '2026-09-23\nCases were reported in the region.',
    'On 14 August 2024, the region now stands better prepared to detect and respond to health threats.',
    'The agency was established in 2023 and protects public health. Current case reports are available.',
    'The surveillance agency was founded in 2023. Disease updates and counts are available.',
    'Copyright 2026. Latest surveillance reports.',
])
def test_incidental_dates_remain_unknown(disease, place, snippet):
    entry = {'title': f'{disease} in {place}', 'snippet': snippet}
    fit = _source_positive_target_verification(entry, state(disease, place))
    assert fit['date_fit'] == 'candidate', fit
    assert fit['target_verification_status'] == 'unverified_candidate'


@pytest.mark.parametrize('phrase', [
    '12 cases reported since 2024', '12 cases reported since January 2024',
    '12 cases reported since 2024-07-01', '12 cas signalés depuis 2024',
    '12 cases reported from 2024 onwards', '12 cases reported from 2024 to present',
])
def test_open_observation_period_not_bounded_to_its_start_year(phrase):
    entry = {'title': 'Example fever in Example Region', 'snippet': phrase}
    assert _source_positive_target_verification(entry, state())['date_fit'] == 'candidate'


def test_open_period_start_after_task_is_explicitly_outside():
    entry = {'title': 'Example fever in Example Region', 'snippet': '12 cases reported since 2026'}
    assert _source_positive_target_verification(entry, state())['date_fit'] == 'mismatch'


@pytest.mark.parametrize('text', [
    'Example fever in Example Region during 2024: 12 confirmed cases.',
    'Example fever surveillance, Example Region, 2024.',
    'Surveillance period: 2024-01-01 to 2024-12-31.',
    '12 cas de example fever en Example Region en 2024.',
])
def test_explicit_observation_outside_task_stays_mismatch(text):
    assert _source_positive_target_verification({'title': text}, state())['date_fit'] == 'mismatch'


@pytest.mark.parametrize('text', [
    'Example fever in Example Region during 2025: 12 confirmed cases.',
    'Example fever surveillance, Example Region, 2024–2026.',
    'Surveillance period: 2024-01-01 to 2026-12-31.',
    '12 cas de example fever en Example Region en 2025.',
])
def test_explicit_observation_overlap_is_preserved(text):
    assert _source_positive_target_verification({'title': text}, state())['date_fit'] == 'match'


@pytest.mark.parametrize('publisher', ['Public Health Agency', 'Unverified author'])
def test_header_year_and_badge_cannot_change_candidate_priority(publisher):
    entry = {'url': 'https://source.example/one', 'title': 'Example fever in Example Region',
        'snippet': 'Wednesday, 23 September 2026\n\n12 confirmed cases.', 'publisher': publisher,
        'target_verification_status': 'temporal_mismatch', 'disease_fit': 'match',
        'geography_fit': 'match', 'task_fit_evidence_origin': 'discovery_metadata',
        'data_product_type': 'surveillance_report'}
    unknown = {'url': entry['url'], 'data_product_type': 'surveillance_report'}
    assert acquisition_priority(entry, state=state()) == acquisition_priority(unknown, state=state())
    assert acquisition_priority({**entry, 'task_fit_evidence_origin': 'explicit_constraint'}, state=state()) < acquisition_priority(unknown, state=state())


def test_legacy_week_verification_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    assert _direct_target_verification({'title': 'Weekly bulletin, week 40, 2025'}, state())['date_fit'] == 'match'
    assert _direct_target_verification({'title': 'Example fever in Example Region during 2024'}, state())['date_fit'] == 'candidate'


def test_evidence_strict_budget_uses_same_source_semantics(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    entry = {'title': 'Example fever in Example Region', 'snippet': 'Established in 2023.'}
    assert _direct_target_verification(entry, {**state(), 'universal': {'budget_policy': {'mode': 'strict'}}})['date_fit'] == 'candidate'


@pytest.mark.parametrize('phrase', ['Depuis le 23 août 2024, 12 cas signalés.', '12 cas signalés depuis février 2024.'])
def test_open_french_date_with_article_is_not_closed_to_start_year(phrase):
    assert _source_positive_target_verification({'title': 'Example fever in Example Region', 'snippet': phrase}, state())['date_fit'] == 'candidate'


@pytest.mark.parametrize('current', [
    'The current outbreak has 850 cases.',
    "L'épidémie s'intensifie avec plus de 850 cas recensés.",
    'Cases have increased in recent weeks.',
])
def test_historical_statistics_cannot_date_distinct_current_undated_observations(current):
    entry = {'title': 'Example fever in Example Region',
        'snippet': 'In 2005–2006, an outbreak caused 120 cases. ' + current}
    fit = _source_positive_target_verification(entry, state())
    assert fit['date_fit'] == 'candidate'
    assert fit['target_verification_status'] != 'verified_target'


def test_dated_current_word_does_not_rescue_explicit_old_observation():
    entry = {'title': 'Example fever in Example Region', 'snippet': 'The current outbreak in 2024 caused 120 cases.'}
    assert _source_positive_target_verification(entry, state())['date_fit'] == 'mismatch'


@pytest.mark.parametrize('navigation', ['Latest surveillance reports', 'Current case reports are available', 'Recent outbreak information'])
def test_undated_navigation_does_not_override_explicit_reporting_period(navigation):
    entry = {'title': 'Example fever in Example Region during 2024: 12 confirmed cases.', 'snippet': navigation}
    assert _source_positive_target_verification(entry, state())['date_fit'] == 'mismatch'


@pytest.mark.parametrize('snippet', ['2025 confirmed cases were reported.', 'The number of cases: 2025.', '2024 deaths were recorded.', 'Case counts are available | 2026 | Copyright'])
def test_counts_and_separate_header_columns_are_not_observation_years(snippet):
    entry = {'title': 'Example fever in Example Region', 'snippet': snippet}
    assert _source_positive_target_verification(entry, state())['date_fit'] == 'candidate'


@pytest.mark.parametrize('text', [
    'Example fever reporting period 2023–2025, with 40 cases.',
    'Example fever report for 2023–2025.',
    'Example fever 2024-W21 report and observations from 2025-01-01 to 2025-12-31.',
    'Example fever 2024-W21 report; Example fever data in 2025.',
])
def test_independent_explicit_periods_survive_week_and_count_masking(text):
    assert _source_positive_target_verification({'title': text}, state())['date_fit'] == 'match'


@pytest.mark.parametrize('text', [
    '2023–2025 cases were estimated.', '2023 to 2025 confirmed cases.',
    'Estimated deaths: 2023–2025.', '2024-W21 report with 2023–2025 cases estimated.',
])
def test_numeric_ranges_do_not_leave_a_false_reporting_start_year(text):
    expected = 'mismatch' if 'W21' in text else 'candidate'
    assert _source_positive_target_verification({'title': text}, state())['date_fit'] == expected


def test_legacy_actual_screening_route_does_not_call_evidence_date_parser(monkeypatch):
    import importlib
    module = importlib.import_module('data_collection_workflow.nodes.source_screening')
    monkeypatch.setenv('PIPELINE_MODE', 'legacy')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_CRITIC', 'false')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_IDENTITY', 'false')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_CREDIBILITY', 'false')
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy route reached evidence date parser')
    monkeypatch.setattr(module, '_source_reporting_date_fit', forbidden)
    task = {**state()['structured_task'], 'collection_mode': 'direct_collection'}
    result = module.source_screening({'structured_task': task,
        'collection_spec': {**task, 'geography': task['location']}, 'collection_trace': [],
        'source_registry': [{'source_id':'legacy_source', 'canonical_url':'https://report.example/a',
            'title':'Example fever in Example Region, week 40, 2025', 'snippet':'12 cases.',
            'publisher':'Public Health Agency', 'source_type':'official_public_health_agency',
            'status':'registered', 'priority':1, 'discovery_method':'offline_fixture'}]})
    assert result['source_triage_results'][0]['date_fit'] == 'match'
