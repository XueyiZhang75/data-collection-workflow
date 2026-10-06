from copy import deepcopy

import pytest

from data_collection_workflow.source_coverage import annotate_source_coverage
from data_collection_workflow.source_identity import assess_source_identity, apply_source_identity_to_registry


def _state():
    return {'structured_task': {'disease': 'Example fever', 'location': 'Canada',
            'start_date': '2025-01-01', 'end_date': '2025-12-31'}}


def _candidate(**overrides):
    return dict(source_id='candidate', url='https://files.invalid/cases.csv',
                canonical_url='https://files.invalid/cases.csv',
                title='Example fever Canada 2025 Download CSV', source_type='unknown',
                source_role='data_source', source_role_final='collection',
                target_fit_status='task_record_collection_candidate', date_fit='candidate',
                ready_for_content_fetch=True, final_screening_decision='include_for_content_fetch',
                source_identity_unverified=True, must_fetch=False, **overrides)


@pytest.mark.parametrize('credibility', [{}, {'credibility_score': 0.2, 'credibility_level': 'low'}])
def test_evidence_task_match_is_a_candidate_not_source_trust_or_period(monkeypatch, credibility):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    source = _candidate(**credibility)
    original = deepcopy(source)
    annotated, requirements, audit = annotate_source_coverage([source], _state())
    result = annotated[0]
    assert result['coverage_requirement_ids'] == [requirements[0]['requirement_id']]
    assert not result['must_fetch']
    assert result.get('credibility_score') == source.get('credibility_score')
    assert result.get('credibility_level') == source.get('credibility_level')
    assert result['source_identity_unverified'] is True
    assert result['date_fit'] == 'candidate'
    assert not result.get('reporting_period_start')
    assert not result.get('reporting_period_end')
    assert not result.get('reporting_period_label')
    assert not result.get('period_basis')
    assert result['ready_for_content_fetch'] is True
    assert not result.get('blocked_from_fetch')
    assert source == original


def test_evidence_annotation_preserves_source_supplied_dates_and_priority(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    source = _candidate(published_date='2025-04-09', publication_date='2025-04-09',
                        reporting_period_start='2025-01-01', reporting_period_end='2025-03-31',
                        reporting_period_label='2025 Q1', period_basis='quarterly')
    source['must_fetch'] = True
    source['must_fetch_reason'] = 'explicit user source selection'
    result = annotate_source_coverage([source], _state())[0][0]
    for key in ('published_date', 'publication_date', 'reporting_period_start',
                'reporting_period_end', 'reporting_period_label', 'period_basis',
                'must_fetch', 'must_fetch_reason'):
        assert result[key] == source[key]


@pytest.mark.parametrize('metadata', [{}, {'result_source': 'Unverified Organization'}])
def test_evidence_unknown_publisher_remains_unverified_and_fetchable(monkeypatch, metadata):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    source = _candidate(**metadata)
    identity = assess_source_identity(source)
    assert identity['source_identity_unverified'] is True
    assert identity['source_type_final'] == 'unknown'
    assert identity['actual_publisher_confidence'] == 'low'
    registry, _, _ = apply_source_identity_to_registry([source], llm_enabled=False)
    assert registry[0]['source_identity_unverified'] is True
    assert registry[0]['ready_for_content_fetch'] is True
    assert not registry[0].get('blocked_from_fetch')


def test_evidence_maintained_authority_identity_is_still_verified(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    source = _candidate()
    source.update(url='https://www.canada.ca/en/public-health/report',
                  canonical_url='https://www.canada.ca/en/public-health/report')
    identity = assess_source_identity(source)
    assert identity['actual_publisher'] == 'Government of Canada'
    assert identity['source_identity_unverified'] is False
    assert identity['actual_publisher_confidence'] == 'high'
    assert identity['source_type_final'] == 'national_public_health_agency'


def test_legacy_task_source_repair_and_identity_are_unchanged(monkeypatch):
    monkeypatch.delenv('PIPELINE_MODE', raising=False)
    source = _candidate()
    result = annotate_source_coverage([source], _state())[0][0]
    assert result['must_fetch'] is True
    assert result['credibility_score'] == 0.95
    assert result['credibility_level'] == 'high'
    assert result['reporting_period_start'] == '2025-01-01'
    assert assess_source_identity(source)['source_identity_unverified'] is False
