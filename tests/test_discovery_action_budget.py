"""A single exhausted search action must not consume the global search budget."""
import importlib

import pytest

from data_collection_workflow.session_runtime import RunContext


def query(metric='cases'):
    return {'query': f'mpox United States {metric} 2025', 'provider_channel': 'official_site_search',
            'source_type': 'official_public_health_agency'}


def task():
    return {'disease': 'mpox', 'location': 'United States', 'start_date': '2025-01-01',
            'end_date': '2025-12-31', 'target_fields': ['cases', 'deaths']}


def prime_failed_action(context, mod, settings):
    payload = {'query': mod._search_request_identity(query()), 'provider': settings.provider,
               'max_results': settings.max_results_per_query}
    def fail():
        raise TimeoutError('fixture failure')
    for _ in range(2):
        with pytest.raises(TimeoutError):
            context.call('search', payload, fail)


def run_batch(mod, context, settings, rows, calls):
    class Provider:
        def search(self, item, **kwargs):
            calls.append(item['query'])
            return {'results': []}
    totals = {'selected_query_count': 0, 'executed_query_count': 0, 'raw_result_count': 0,
              'deduped_result_count': 0, 'provider_error_count': 0}
    with context.activate():
        result = mod._execute_iterative_query_batch(query_batch=rows, provider=Provider(), settings=settings,
                    seen_canonical_urls=set(), totals=totals, task=task())
    return totals, result


def test_exhausted_action_is_skipped_while_novel_query_uses_remaining_budget(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    settings = mod.SourceSearchSettings(mode='fixture', iterative_max_queries_per_iteration=1)
    context = RunContext(tmp_path, {'universal': {'budget_limits': {'search': 10}}})
    prime_failed_action(context, mod, settings)
    calls = []
    totals, result = run_batch(mod, context, settings, [query(), query('deaths')], calls)
    assert calls == [query('deaths')['query']]
    assert totals['selected_query_count'] == 1
    assert not totals.get('search_budget_exhausted')
    assert result[2][0]['skipped_reason'] == 'same_query_attempt_limit_reached'
    assert not result[2][0]['selected_for_execution']
    assert context.ledger.snapshot()['used']['search'] == 3


def test_global_search_budget_still_stops_before_next_provider_dispatch(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    settings = mod.SourceSearchSettings(mode='fixture', iterative_max_queries_per_iteration=4)
    context = RunContext(tmp_path, {'universal': {'budget_limits': {'search': 1}}})
    calls = []
    totals, result = run_batch(mod, context, settings, [query(), query('deaths')], calls)
    assert calls == [query()['query']]
    assert totals['search_budget_exhausted']
    assert result[2][-1]['skipped_reason'] == 'shared_search_budget_exhausted'
    assert context.ledger.snapshot()['used']['search'] == 1


def test_action_exhaustion_is_remembered_by_following_discovery_iterations(monkeypatch, tmp_path):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    from data_collection_workflow.agents import iterative_source_discovery_agent as agent
    settings = mod.SourceSearchSettings(mode='fixture', iterative_max_iterations=3,
                   iterative_max_queries_per_iteration=2, iterative_max_total_queries=10,
                   authority_gap_retry_enabled=False)
    context = RunContext(tmp_path, {'universal': {'budget_limits': {'search': 10}}})
    prime_failed_action(context, mod, settings)
    calls = []
    class Provider:
        def search(self, item, **kwargs):
            calls.append(item['query'])
            return {'results': []}
    monkeypatch.setattr(mod, '_provider_for_settings', lambda settings: Provider())
    monkeypatch.setattr(agent, 'plan_initial_search_iteration', lambda **kwargs: {'query_batch': [query()]})
    monkeypatch.setattr(agent, 'refine_search_iteration', lambda **kwargs: {
        'iteration_index': 1, 'decision': 'stop_sufficient', 'decision_reason': 'Fixture advice.',
        'next_query_batch': []})
    with context.activate():
        _, _, summary, details = mod._execute_iterative_source_search({'structured_task': task()}, settings)
    planned = [row for plan in details['search_iteration_plans'] for row in plan['query_batch']]
    assert sum(row['query'] == query()['query'] for row in planned) == 1
    assert calls and query()['query'] not in calls
    assert 'shared_search_budget_exhausted' not in summary['warnings']
    assert context.ledger.snapshot()['used']['search'] == len(calls) + 2
