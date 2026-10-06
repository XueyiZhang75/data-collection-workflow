"""Offline regressions for task grounding and bounded discovery continuation."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@pytest.fixture(autouse=True)
def universal_pipeline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def _state():
    return {'structured_task': {
        'disease': 'mpox', 'location': 'United States',
        'start_date': '2025-01-01', 'end_date': '2025-12-31',
        'target_fields': ['cases_confirmed', 'deaths', 'hospitalizations', 'test_positivity'],
        'collection_mode': 'direct_collection',
    }}


@pytest.mark.parametrize('query', [
    'state health department mpox update 2025 cases hospitalizations deaths',
    'CDC data.cdc.gov mpox dataset United States 2025 case counts by jurisdiction',
    'mpox United States test positivity by jurisdictions 2025',
    'mpox United States hospitalization deaths 2025',
])
def test_target_fields_and_reporting_inflections_are_grounded(query):
    from data_collection_workflow.query_policy import assess_query_task_fit
    assert assess_query_task_fit({'query': query}, _state())['accepted']


@pytest.mark.parametrize('query', [
    'mpox United States hantavirus Andes cruise 2025',
    'mpox United States ship Alpha 2025',
    'mpox Spain hospitalizations 2025',
    'measles United States hospitalizations 2025',
])
def test_reporting_vocabulary_does_not_license_other_entities(query):
    from data_collection_workflow.query_policy import assess_query_task_fit
    assert not assess_query_task_fit({'query': query}, _state())['accepted']


def _query(number):
    return {'query': f'mpox United States official cases {number}',
            'provider_channel': 'official_site_search',
            'source_type': 'official_public_health_agency'}


def _run(monkeypatch, *, refinement='stop', initial='valid', empty=False,
         total=20, iterations=3, per_iteration=4, result_limit=260,
         duplicate=False, disabled=False, first_failure=None, real_runtime=False):
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    from data_collection_workflow.agents import iterative_source_discovery_agent as agent
    calls = []
    class Provider:
        def search(self, query, **kwargs):
            calls.append(query)
            if len(calls) == 1 and first_failure == 'exception':
                raise TimeoutError('temporary search service failure')
            if len(calls) == 1 and first_failure == 'response':
                return {'results': [], 'error': 'temporary search service failure'}
            if empty:
                return {'results': []}
            return {'results': [{'title': 'mpox United States 2025 cases',
                                 'url': f'https://public-health.example/report/{1 if duplicate else len(calls)}'}]}
    monkeypatch.setattr(mod, '_provider_for_settings', lambda settings: Provider())
    if real_runtime:
        from data_collection_workflow.session_runtime import external_call
        monkeypatch.setattr(mod, 'external_call', external_call)
    else:
        monkeypatch.setattr(mod, 'external_call', lambda kind, payload, call: call())
    def plan(**kwargs):
        if initial == 'error':
            raise ValueError('invalid model plan')
        if initial == 'malformed':
            return {'search_reasoning': 'embedded query_batch is not a top-level field'}
        return {'query_batch': [_query(n) for n in range(per_iteration)]}
    def refine(**kwargs):
        if refinement == 'error':
            raise ValueError('invalid model advice')
        if refinement == 'malformed':
            return {'iteration_index': 2, 'coverage_assessment': 'embedded decision and queries'}
        return {'iteration_index': 1, 'decision': 'stop_sufficient',
                'decision_reason': 'Search snippets appear sufficient.', 'next_query_batch': []}
    monkeypatch.setattr(agent, 'plan_initial_search_iteration', plan)
    monkeypatch.setattr(agent, 'refine_search_iteration', refine)
    settings = mod.SourceSearchSettings(
        mode='disabled' if disabled else 'fixture', max_queries=16,
        max_total_results=result_limit, iterative_max_total_results=result_limit,
        iterative_enabled=True, iterative_max_iterations=iterations,
        iterative_max_queries_per_iteration=per_iteration,
        iterative_max_total_queries=total, authority_gap_retry_enabled=False)
    result = mod._execute_iterative_source_search(_state(), settings)
    return calls, result


@pytest.mark.parametrize('refinement', ['stop', 'malformed', 'error'])
def test_search_metadata_or_bad_advice_cannot_end_novel_discovery(monkeypatch, refinement):
    calls, (_, _, summary, details) = _run(monkeypatch, refinement=refinement)
    assert len(calls) == 12  # The existing 3 iterations x 4 queries, not the ledger ceiling.
    assert len({row['query'] for row in calls}) == 12
    assert summary['stop_decision'] == 'stop_limits_reached'
    assert len(details['search_iteration_observations']) == 3
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    assert {mod._query_source_class(row) for row in calls} >= {
        'international_official', 'national_or_local_official',
        'structured_database', 'peer_reviewed_literature'}


@pytest.mark.parametrize('initial', ['malformed', 'error'])
def test_bad_initial_plan_still_uses_bounded_task_grounded_search(monkeypatch, initial):
    calls, (_, _, summary, _) = _run(monkeypatch, initial=initial)
    assert len(calls) == 12
    assert summary['stop_decision'] == 'stop_limits_reached'
    from data_collection_workflow.query_policy import assess_query_task_fit
    assert all(assess_query_task_fit(query, _state())['accepted'] for query in calls)


@pytest.mark.parametrize('total,iterations,per_iteration,result_limit,expected', [
    (2, 3, 4, 260, 2), (20, 1, 4, 260, 4), (20, 3, 1, 260, 3),
    (20, 3, 4, 2, 2), (0, 3, 4, 260, 0),
])
def test_explicit_small_caps_are_hard_limits(monkeypatch, total, iterations,
                                           per_iteration, result_limit, expected):
    calls, _ = _run(monkeypatch, total=total, iterations=iterations,
                    per_iteration=per_iteration, result_limit=result_limit)
    assert len(calls) == expected


def test_genuinely_empty_providers_stop_after_bounded_diverse_attempts(monkeypatch):
    calls, (_, _, summary, _) = _run(monkeypatch, empty=True, iterations=20, total=80)
    assert len(calls) == 8
    assert len({row['query'] for row in calls}) == len(calls)
    assert summary['stop_decision'] == 'stop_no_promising_sources'
    assert summary['stop_reason'] == 'no_new_sources_in_consecutive_batches'


def test_duplicate_only_batches_exhaust_novelty(monkeypatch):
    calls, (_, _, summary, _) = _run(monkeypatch, duplicate=True, iterations=20, total=80)
    assert len(calls) == 12  # One productive batch followed by two without a new URL.
    assert summary['stop_reason'] == 'no_new_sources_in_consecutive_batches'


def test_search_disabled_never_tries_provider(monkeypatch):
    calls, _ = _run(monkeypatch, disabled=True)
    assert calls == []


def test_agent_rejects_incomplete_refinement_instead_of_default_stop(monkeypatch):
    from data_collection_workflow.agents import iterative_source_discovery_agent as agent
    monkeypatch.setattr(agent.llm_clients, 'run_pydantic_structured_llm', lambda **kwargs: {
        'iteration_index': 2, 'decision': 'stop_sufficient', 'decision_reason': '',
        'coverage_assessment': 'entire remaining JSON was swallowed here',
        'next_query_batch': [],
    })
    with pytest.raises(ValueError):
        agent.refine_search_iteration(user_request='', structured_task=_state()['structured_task'],
            collection_spec={}, previous_plans=[], observations=[], observation={}, limits={})


@pytest.mark.parametrize('failure', ['exception', 'response'])
def test_recoverable_provider_errors_do_not_abort_discovery(monkeypatch, failure):
    calls, (candidates, _, summary, _) = _run(monkeypatch, first_failure=failure)
    assert len(calls) == 12
    assert len(candidates) == 11
    assert summary['provider_error_count'] == 1
    assert summary['stop_decision'] == 'stop_limits_reached'


def test_shared_ledger_search_cap_stops_before_next_provider_call(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': {'search': 2}}})
    with runtime.activate():
        calls, (_, _, summary, _) = _run(monkeypatch, real_runtime=True)
    assert len(calls) == summary['selected_query_count'] == 2
    assert runtime.ledger.snapshot()['used']['search'] == 2
    assert summary['stop_decision'] == 'stop_limits_reached'
    assert 'shared_search_budget_exhausted' in summary['warnings']


def test_shared_ledger_result_cap_does_not_keep_searching(monkeypatch, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': {'search_results': 1}}})
    with runtime.activate():
        calls, (candidates, _, summary, _) = _run(monkeypatch, real_runtime=True)
    assert len(calls) <= 2  # At most the response that first reveals exhaustion.
    assert len(candidates) == 1
    assert runtime.ledger.snapshot()['used']['search_results'] == 1
    assert summary['stop_decision'] == 'stop_limits_reached'


def test_interactive_budget_maps_to_original_stage_caps(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.collect import _real_run_config
    from data_collection_workflow.runtime_profile import temporary_workflow_env, workflow_run_env_from_config
    from data_collection_workflow.session_runtime import derive_budget_limits
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    config = _real_run_config(disease='measles', location='Canada',
        start_date='2025-01-01', end_date='2025-12-31', target_fields=['cases'],
        session_id='offline-config-test', provider='anthropic', model='claude-sonnet-4-6',
        output_dir=None, no_llm=False)
    budget = derive_budget_limits(config)
    assert (budget['search'], budget['search_results']) == (88, 380)
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        settings = mod._source_search_settings_from_env()
    assert settings.max_queries == 16  # The separate one-shot cap.
    assert settings.iterative_max_total_queries == 20
    assert settings.iterative_max_iterations * settings.iterative_max_queries_per_iteration == 12
    assert min(settings.max_total_results, settings.iterative_max_total_results) == 260
    config['universal'] = {'budget_limits': {'search': 2, 'search_results': 3}}
    assert derive_budget_limits(config)['search'] == 2
    assert derive_budget_limits(config)['search_results'] == 3


@pytest.mark.parametrize('query', [
    'CDC mpox case count United States 2025 data tracker',
    'CDC MMWR mpox 2025 United States surveillance report',
    'WHO mpox multi-country outbreak external situation report United States 2025',
])
def test_legitimate_reporting_products_do_not_become_unbound_events(query):
    from data_collection_workflow.query_policy import assess_query_task_fit
    assert assess_query_task_fit({'query': query}, _state())['accepted']


@pytest.mark.parametrize('query', [
    'CDC mpox United States data tracker ship Alpha 2025',
    'CDC MMWR mpox Pakistan surveillance report 2025',
    'WHO mpox Spain multi-country external situation report 2025',
])
def test_reporting_product_terms_still_cannot_introduce_unbound_facts(query):
    from data_collection_workflow.query_policy import assess_query_task_fit
    assert not assess_query_task_fit({'query': query}, _state())['accepted']


def test_generic_retry_rejects_inherited_trust_from_search_query(monkeypatch):
    from types import SimpleNamespace
    from data_collection_workflow.query_policy import generic_retry_specs, assess_query_task_fit
    import data_collection_workflow.source_identity as identity
    monkeypatch.setattr(identity, 'lookup_source_identity_registry', lambda domain: None)
    candidates = [
        SimpleNamespace(domain='custom-map.example', source_type='peer_reviewed_literature'),
        SimpleNamespace(domain='entertainment.example', source_type='official_public_health_agency'),
    ]
    queries = generic_retry_specs(_state(), candidates, '2025')
    assert queries  # Rejected identities must not eliminate generic exploration.
    assert not any('site:custom-map.example' in query['query'] or
                   'site:entertainment.example' in query['query'] for query in queries)
    assert all(assess_query_task_fit(query, _state())['accepted'] for query in queries)


def test_generic_retry_uses_independent_domain_identity_for_type(monkeypatch):
    from types import SimpleNamespace
    from data_collection_workflow.query_policy import generic_retry_specs
    import data_collection_workflow.source_identity as identity
    curated = {'domain': 'health.example', 'publisher_name': 'Example Public Health Agency',
               'source_type': 'national_public_health_agency',
               'jurisdiction_scope': 'national', 'jurisdiction_name': 'United States'}
    monkeypatch.setattr(identity, 'lookup_source_identity_registry',
                        lambda domain: curated if domain == 'health.example' else None)
    queries = generic_retry_specs(_state(), [
        SimpleNamespace(domain='health.example', source_type='peer_reviewed_literature'),
        SimpleNamespace(domain='custom-map.example', source_type='peer_reviewed_literature'),
    ], '2025')
    selected = [query for query in queries if 'site:health.example' in query['query']]
    assert selected
    assert all(query['source_type'] == 'national_public_health_agency' and
               query['provider_channel'] == 'official_site_search' for query in selected)
    assert not any('site:custom-map.example' in query['query'] for query in queries)


@pytest.mark.parametrize('bound_to_current_page,page_location', [(True, 'United States'), (False, 'United States'), (True, 'Pakistan')])
def test_retry_can_use_task_relevant_post_fetch_identity_only_when_bound(monkeypatch, bound_to_current_page, page_location):
    from types import SimpleNamespace
    from data_collection_workflow.query_policy import generic_retry_specs
    import data_collection_workflow.source_identity as identity
    monkeypatch.setattr(identity, 'lookup_source_identity_registry', lambda domain: None)
    state = _state()
    quote = 'Example University Public Health Research Centre'
    state['source_identity_assessments'] = [{
        'source_id': 'research', 'domain': 'research.example',
        'source_type_final': 'academic_or_peer_reviewed_source',
        'actual_publisher': quote, 'post_fetch_identity_assessed': True,
        'metadata_only_identity': False, 'source_identity_unverified': False,
        'publisher_domain_mismatch': False, 'page_identity_evidence': [quote],
    }]
    state['documents'] = [{'source_id': 'research', 'url': 'https://research.example/report',
        'content_readable': True,
        'clean_text': ('Published by ' + quote + '. ' if bound_to_current_page else '') +
                      f'Mpox {page_location} 2025 surveillance case report.'}]
    queries = generic_retry_specs(state, [SimpleNamespace(
        source_id='research', domain='research.example', source_type='official_public_health_agency')], '2025')
    selected = [q for q in queries if 'site:research.example' in q['query']]
    assert bool(selected) is (bound_to_current_page and page_location == 'United States')
    if selected:
        assert all(q['source_type'] == 'academic_or_peer_reviewed_source' and
                   q['provider_channel'] == 'literature_api' for q in selected)
