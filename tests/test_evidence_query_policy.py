import importlib

import pytest

from data_collection_workflow.models import SourceCandidate


@pytest.mark.parametrize('disease', ['mpox', 'measles', 'dengue', 'novel disease'])
def test_universal_retries_do_not_invent_event_or_disease(monkeypatch, disease):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    state = {'structured_task': {'disease': disease, 'location': 'United States', 'start_date': '2025-01-01', 'end_date': '2025-12-31'}}
    candidates = [SourceCandidate(source_id='s1', url='https://www.who.int/example', canonical_url='https://www.who.int/example', domain='who.int', title=disease + ' outbreak update', source_type='international_public_health_agency')]
    specs = mod._official_page_family_retry_specs(state=state, search_candidates=candidates, candidate_text=disease + ' outbreak France', year_suffix='2025')
    queries = [s['query'].lower() for s in specs]
    assert queries
    assert all(disease in q for q in queries)
    assert not any(term in q for q in queries for term in ['andes', 'hantavirus', 'hondius', 'cruise', 'navire'])


def test_metadata_cannot_seed_unrelated_event(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    state = {'structured_task': {'disease': 'mpox', 'location': 'Canada'}}
    candidate = SourceCandidate(source_id='s1', url='https://who.int/example', domain='who.int', title='Andes virus cruise ship', source_type='international_public_health_agency')
    specs = mod._official_page_family_retry_specs(state=state, search_candidates=[candidate], candidate_text='Andes virus cruise ship outbreak', year_suffix='2025')
    assert all('andes' not in s['query'].lower() and 'cruise' not in s['query'].lower() for s in specs)


def test_query_requires_task_disease_and_grounded_event():
    from data_collection_workflow.query_policy import assess_query_task_fit
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    assert assess_query_task_fit({'query': 'measles Canada cases'}, state)['accepted']
    assert not assess_query_task_fit({'query': 'hantavirus cruise ship'}, state)['accepted']
    assert not assess_query_task_fit({'query': 'measles ship Alpha', 'event_terms': ['ship Alpha']}, state)['accepted']
    state['structured_task']['user_request'] = 'Collect measles on ship Alpha in Canada'
    assert assess_query_task_fit({'query': 'measles ship Alpha', 'event_terms': ['ship Alpha']}, state)['accepted']


def test_unannotated_invented_event_is_rejected():
    from data_collection_workflow.query_policy import assess_query_task_fit
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    assert not assess_query_task_fit({'query': 'measles Andes cruise ship'}, state)['accepted']
    assert assess_query_task_fit({'query': 'measles Canada historical surveillance cases data site:canada.ca 2025'}, state)['accepted']


def test_retry_keeps_independently_known_secondary_domain_identity(monkeypatch):
    from data_collection_workflow.query_policy import generic_retry_specs
    from data_collection_workflow import source_identity
    monkeypatch.setattr(source_identity, 'lookup_source_identity_registry', lambda domain: {
        'domain': 'example.org', 'publisher_name': 'Example Health News',
        'source_type': 'news_and_situation_report',
    } if domain == 'example.org' else None)
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    source = SourceCandidate(source_id='news', url='https://example.org/story', domain='example.org', source_type='official_public_health_agency')
    specs = generic_retry_specs(state, [source], '2025')
    selected = [row for row in specs if row['official_domain_hint'] == 'example.org']
    assert selected
    assert all(s['source_type'] == 'news_and_situation_report' for s in selected)
    assert all(s['provider_channel'] == 'news_search' for s in selected)


def test_rejected_iterative_query_is_audited_without_calling_provider(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    settings = mod.SourceSearchSettings()
    totals = {'selected_query_count': 0, 'deduped_result_count': 0, 'provider_error_count': 0}
    result = mod._execute_iterative_query_batch(query_batch=[{'query_id': 'bad', 'query': 'measles Andes ship', 'provider_channel': 'web_search'}], provider=None, settings=settings, seen_canonical_urls=set(), totals=totals, task={'disease': 'measles', 'location': 'Canada'})
    assert result[2][0]['execution_status'] == 'skipped_task_mismatch'
    assert totals['selected_query_count'] == 0


def test_empty_discovery_still_plans_task_grounded_search():
    from data_collection_workflow.query_policy import generic_retry_specs
    specs = generic_retry_specs({'structured_task': {'disease': 'measles', 'location': 'Canada'}}, [], '2025')
    assert specs
    assert all('measles' in s['query'] and 'Canada' in s['query'] for s in specs)


def _task_plan(monkeypatch, disease='hantavirus', location='New Mexico', **task_fields):
    from data_collection_workflow.nodes import task_scope
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_PLANNING', 'false')
    monkeypatch.setenv('ENABLE_LLM_DISEASE_INTELLIGENCE', 'false')
    state = {'structured_task': {
        'disease': disease, 'location': location, 'start_date': '2024-01-01',
        'end_date': '2024-12-31', 'collection_mode': 'direct_collection', **task_fields,
    }}
    for node in (task_scope.task_intake_and_scope_planning,
                 task_scope.disease_intelligence_builder,
                 task_scope.profile_and_schema_setup,
                 task_scope.executable_source_planning,
                 task_scope.query_strategy_builder):
        state.update(node(state))
    return state


@pytest.mark.parametrize('disease,location', [
    ('hantavirus', 'New Mexico'), ('COVID-19', 'New York'),
    ('dengue', 'Brazil'), ('measles', 'Canada'),
])
def test_evidence_fallback_plan_uses_executable_task_terms(monkeypatch, disease, location):
    from data_collection_workflow.query_policy import assess_query_task_fit
    state = _task_plan(monkeypatch, disease, location)
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    queries = mod._planned_queries(state)
    assert queries
    assert all(assess_query_task_fit(query, state)['accepted'] for query in queries)
    assert all(location in query['query'] for query in queries)
    assert not any(word in query['query'].casefold() for query in queries
                   for word in ('cruise', 'vessel', 'andes', 'hondius'))


def test_evidence_user_alias_survives_intake_and_query_guard(monkeypatch):
    from data_collection_workflow.query_policy import assess_query_task_fit
    state = _task_plan(monkeypatch, 'measles', 'Canada', disease_aliases=['rubeola'])
    assert state['structured_task']['disease_aliases'] == ['rubeola']
    assert assess_query_task_fit({'query': 'rubeola Canada cases',
                                  'disease_terms_used': ['rubeola']}, state)['accepted']


def test_evidence_query_selection_does_not_spend_slots_on_rejected_terms(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    queries = [
        {'query_id': 'bad', 'query': 'measles Andes cruise',
         'source_type': 'official_public_health_agency', 'provider_channel': 'official_site_search'},
        {'query_id': 'good', 'query': 'measles Canada cases',
         'source_type': 'official_public_health_agency', 'provider_channel': 'official_site_search'},
    ]
    selected = mod._select_source_search_queries(queries, state, mod.SourceSearchSettings(max_queries=1))
    assert list(selected) == ['good']


def test_evidence_routine_task_preserves_source_family_query_floor(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    specs = [('official_public_health_agency', 'official_site_search')] * 5 + [
        ('international_organization_report', 'official_site_search'),
        ('structured_database', 'database_search'),
        ('peer_reviewed_literature', 'literature_api'),
    ]
    queries = [{'query_id': str(i), 'query': 'measles Canada cases',
                'source_type': kind, 'provider_channel': channel}
               for i, (kind, channel) in enumerate(specs)]
    selected = mod._select_source_search_queries(queries, state, mod.SourceSearchSettings(max_queries=4))
    assert {row['selection_bucket'] for row in selected.values()} == {
        'international_official', 'national_or_local_official',
        'structured_database', 'peer_reviewed_literature',
    }


def _offline_search(monkeypatch, mod, *, title='Measles Canada cases', url='https://example.org/cases'):
    class Provider:
        def search(self, query, **kwargs):
            return {'results': [{'title': title, 'url': url, 'snippet': title}]}
    monkeypatch.setattr(mod, '_provider_for_settings', lambda settings: Provider())
    # The provider is in-memory; keep the real discovery/validation path.
    monkeypatch.setattr(mod, 'external_call', lambda kind, payload, fn: fn())
    return Provider()


@pytest.mark.parametrize('grounding', ['user_request', 'fetched_span'])
def test_evidence_iterative_search_keeps_current_task_event_context(monkeypatch, grounding):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    from data_collection_workflow.agents import iterative_source_discovery_agent as agent
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'}}
    if grounding == 'user_request':
        state['structured_task']['user_request'] = 'Collect measles cases on ship Alpha in Canada'
    else:
        state['documents'] = [{'document_id': 'doc', 'content_hash': 'hash',
                               'text': 'Measles cases were on ship Alpha.'}]
        state['supported_event_terms'] = [{'term': 'ship Alpha', 'document_id': 'doc',
                                            'quote': 'Measles cases were on ship Alpha.'}]
    monkeypatch.setattr(agent, 'plan_initial_search_iteration', lambda **kwargs: {
        'query_batch': [{'query': 'measles Canada ship Alpha cases'}],
    })
    monkeypatch.setattr(agent, 'refine_search_iteration', lambda **kwargs: {
        'iteration_index': 1, 'decision': 'stop_sufficient',
    })
    _offline_search(monkeypatch, mod)
    candidates, _, summary, _ = mod._execute_iterative_source_search(
        state, mod.SourceSearchSettings(mode='fixture', iterative_max_iterations=1,
                                        authority_gap_retry_enabled=False))
    assert len(candidates) == 1
    assert summary['executed_query_count'] == 1


def test_evidence_does_not_stop_search_on_influenza_snippet_match(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    state = {'structured_task': {'disease': 'influenza', 'location': 'United States',
             'start_date': '2024-09-29', 'end_date': '2024-10-05',
             'collection_mode': 'direct_collection'}, 'search_query_inventory': [
        {'query_id': str(i), 'query': 'influenza United States cases 2024',
         'provider_channel': 'official_site_search', 'source_type': 'official_public_health_agency'}
        for i in range(2)
    ]}
    _offline_search(monkeypatch, mod, title='United States influenza week 40 2024 cases',
                    url='https://www.cdc.gov/fluview/surveillance/2024-week-40.html')
    _, _, summary = mod._execute_one_shot_source_search(
        state, mod.SourceSearchSettings(mode='fixture', max_queries=2,
                                        authority_gap_retry_enabled=False))
    assert summary['selected_query_count'] == 2
    assert summary['search_stopped_reason'] != 'verified_target_source_found'


def test_evidence_executes_family_floors_before_extra_queries_fill_result_budget(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    mod = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    specs = [('official_public_health_agency', 'official_site_search')] * 2 + [
        ('international_organization_report', 'official_site_search'),
        ('structured_database', 'database_search'),
        ('peer_reviewed_literature', 'literature_api'),
    ]
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada'},
             'search_query_inventory': [
        {'query_id': str(i), 'query': 'measles Canada cases',
         'source_type': kind, 'provider_channel': channel}
        for i, (kind, channel) in enumerate(specs)
    ]}
    class Provider:
        def search(self, query, **kwargs):
            return {'results': [{'title': 'Measles Canada cases',
                                 'url': 'https://example.org/' + query['query_id']}]}
    monkeypatch.setattr(mod, '_provider_for_settings', lambda settings: Provider())
    monkeypatch.setattr(mod, 'external_call', lambda kind, payload, fn: fn())
    _, _, summary = mod._execute_one_shot_source_search(
        state, mod.SourceSearchSettings(mode='fixture', max_queries=5, max_total_results=4,
                                        authority_gap_retry_enabled=False))
    assert summary['result_count_by_source_class'] == {
        'international_official': 1, 'national_or_local_official': 1,
        'structured_database': 1, 'peer_reviewed_literature': 1,
        'news_or_supporting_media': 0, 'context_or_other': 0,
    }
