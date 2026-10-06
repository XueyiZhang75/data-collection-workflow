"""Built-in disease sources must not leak into a different user's task."""
import importlib
import json

import pytest

discovery = importlib.import_module('data_collection_workflow.nodes.source_discovery')


def _discover(monkeypatch, disease, mode='disabled'):
    monkeypatch.setenv('PIPELINE_MODE', 'standard')
    settings = discovery.SourceSearchSettings(mode=mode, combine_with_seed_catalog=True)
    monkeypatch.setattr(discovery, '_source_search_settings_from_env', lambda: settings)
    monkeypatch.setattr(discovery, '_effective_discovery_settings', lambda value: value)
    monkeypatch.setattr(discovery, 'build_official_coverage_candidates', lambda state: [])
    monkeypatch.setattr(discovery, '_execute_source_search',
                        lambda state, settings: ([], [], {}, discovery._disabled_iterative_outputs()))
    return discovery.source_discovery({'structured_task': {
        'disease': disease, 'location': 'Canada', 'start_date': '2024',
        'end_date': '2024', 'collection_mode': 'standard'}})


@pytest.mark.parametrize('mode', ['disabled', 'fixture'])
@pytest.mark.parametrize('disease', ['measles', 'dengue'])
def test_other_diseases_do_not_inherit_hantavirus_sources(monkeypatch, disease, mode):
    monkeypatch.delenv('SEED_SOURCE_OVERLAY_PATH', raising=False)
    result = _discover(monkeypatch, disease, mode)
    assert result['source_candidates'] == []
    assert result['source_discovery_summary']['candidate_from_seed_count'] == 0


def test_explicit_other_disease_catalog_is_used_without_unrelated_builtin_sources(monkeypatch, tmp_path):
    overlay = tmp_path / 'sources.json'
    overlay.write_text(json.dumps({'seed_sources': [{
        'seed_source_id': 'seed_user_measles', 'title': 'Synthetic measles report',
        'url': 'https://example.org/measles', 'publisher': 'Example authority',
        'source_type': 'official_public_health_agency', 'priority': 1,
        'source_purpose': 'collection', 'expected_fields': ['cases_confirmed'],
        'match_terms': ['measles'], 'notes': 'Synthetic test source'}]}), encoding='utf-8')
    monkeypatch.setenv('SEED_SOURCE_OVERLAY_PATH', str(overlay))
    result = _discover(monkeypatch, 'measles')
    assert [row['url'] for row in result['source_candidates']] == ['https://example.org/measles']


@pytest.mark.parametrize('disease', ['hantavirus', 'HPS', 'HFRS'])
def test_matching_disease_catalog_remains_available(monkeypatch, disease):
    monkeypatch.delenv('SEED_SOURCE_OVERLAY_PATH', raising=False)
    result = _discover(monkeypatch, disease)
    assert result['source_discovery_summary']['candidate_from_seed_count'] > 0
