"""Real configured graph and acquisition readiness with offline external endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3

import pytest

from test_evidence_workflow_integration import ROOT, _offline_config, _runner


@pytest.mark.skipif(not (ROOT / '.runtime/acquisition-paths.json').exists(),
                    reason='requires installed local browser/PDF/OCR runtime')
@pytest.mark.parametrize('include_distractors', [False, True])
def test_configured_graph_completes_after_partial_external_failures(tmp_path, monkeypatch, include_distractors):
    import anthropic
    import httpx
    import requests
    from langchain_anthropic import ChatAnthropic
    from data_collection_workflow import llm_clients
    import importlib
    discovery = importlib.import_module('data_collection_workflow.nodes.source_discovery')

    runner = _runner()
    config = _offline_config(tmp_path)
    config['user_request'] = 'Collect confirmed pertussis cases in United States during 2024.'
    config['structured_task'] = {
        'disease': 'Pertussis', 'location': 'United States',
        'start_date': '2024-01-01', 'end_date': '2024-12-31',
        'collection_mode': 'direct_collection',
        'target_fields': ['disease', 'country', 'subnational_location', 'reporting_period', 'cases_confirmed'],
    }
    config['workflow'].update(collection_mode='direct_collection', use_fixture_documents=False)
    config['source_sets']['source_id_allowlist_enabled'] = False
    config['live_web']['enabled'] = True
    config['source_search'].update(enabled=True, mode='live', combine_with_seed_catalog=False,
                                   max_queries=4, max_results_per_query=3, max_total_results=6)
    config['source_search']['iterative'].update(enabled=True, max_iterations=2,
        max_queries_per_iteration=2, max_total_queries=4, max_total_results=6)
    config['source_search']['authority_gap_retry']['enabled'] = False
    config['content_fetch'].update(max_total_sources=4, max_search_derived_sources=4,
        min_credibility_score=0, content_fixture_map_path=None)
    config['content_fetch']['external_fetch']['enabled'] = False
    config['validation']['mode'] = 'diagnostic_only'
    config['llm'].update(provider='anthropic', model='claude-sonnet-4-6', thinking='disabled',
        structured_output_method='function_calling', structured_extraction_enabled=True,
        max_tokens=4096, max_chunks=6, fallback_to_rule_based=False)
    config['llm']['source_identity']['require_llm'] = False
    config['llm']['source_identity']['post_fetch'] = False
    config['llm']['extraction'].update(focused_recovery_reserved_calls=2, max_domain_call_share=1)
    config['llm']['extraction']['scheduler'].update(max_concurrency=1, safety_max_calls=8)
    config['universal'] = {'budget_limits': {'search': 4, 'search_results': 6, 'fetch': 8,
        'fetch_ordinary': 4, 'browser': 2, 'ocr': 2, 'extraction': 8}, 'extraction_reserve': 2}
    if include_distractors:
        config['source_search']['max_results_per_query'] = 6
        config['content_fetch'].update(max_total_sources=6, max_search_derived_sources=6)
        config['llm']['max_chunks'] = 10
        config['llm']['extraction']['scheduler']['safety_max_calls'] = 12
        config['universal']['budget_limits'].update(fetch=14, fetch_ordinary=6, extraction=12)
    config['output']['session_id'] = 'offline-resilience'
    config['output']['auto_build_console'] = False
    monkeypatch.setattr(runner, '_config_with_cli_overrides', lambda args: (None, config))
    monkeypatch.setenv('TAVILY_API_KEY', 'offline-placeholder')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'offline-placeholder')
    monkeypatch.setenv('LLM_MAX_RETRIES', '0')
    monkeypatch.setenv('ENABLE_LANGSMITH_TRACE', 'false')

    search_calls, fetched_urls, model_calls = [], [], Counter()
    base = 'https://www.cdc.gov/offline-integration/'
    quotes = {
        'report-a': 'Pertussis in Washington, United States: 12 confirmed cases during 2024.',
        'report-b': 'Pertussis in Oregon, United States: 7 confirmed cases during 2024.',
    }
    if include_distractors:
        quotes.update({
            'wrong-country': 'Pertussis in Ontario, Canada: 90 confirmed cases during 2024.',
            'wrong-year': 'Pertussis in Oregon, United States: 91 confirmed cases during 2023.',
            'wrong-disease': 'Measles in Oregon, United States: 92 confirmed cases during 2024.',
        })
    class Provider:
        def search(self, query, **kwargs):
            search_calls.append(query['query'])
            if len(search_calls) == 1:
                raise TimeoutError('offline transient search failure')
            return {'results': [
                {'title': 'Pertussis United States surveillance 2024 ' + key,
                 'url': base + key, 'snippet': quotes.get(key, 'Pertussis United States 2024 confirmed cases'),
                 'source': 'Centers for Disease Control and Prevention'}
                for key in (('missing', 'report-a', 'report-b', 'wrong-country', 'wrong-year', 'wrong-disease')
                            if include_distractors else ('missing', 'report-a', 'report-b'))]}
    monkeypatch.setattr(discovery, '_provider_for_settings', lambda settings: Provider())

    class Response:
        def __init__(self, url):
            self.url = url
            self.status_code = 404 if url.endswith('/missing') else 200
            self.headers = {'content-type': 'text/plain'}
            key = url.rsplit('/', 1)[-1]
            self.content = ('Page not found' if self.status_code == 404 else quotes[key]).encode()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield self.content
    def remote_response(url, **kwargs):
        assert url.startswith(base), 'unexpected external HTTP destination: ' + url
        fetched_urls.append(url)
        return Response(url)
    monkeypatch.setattr(requests, 'get', remote_response)

    http_clients = []
    def respond(request):
        payload = json.loads(request.content)
        schema = payload['tools'][0]['name']
        user = payload['messages'][-1]['content']
        if isinstance(user, list):
            user = ' '.join(item.get('text', '') for item in user)
        if 'Synthetic source:' in user:
            stage = 'preflight'
            result = {'records': [{'disease': 'measles', 'cases_confirmed': 3,
                       'evidence_quote': 'There were 3 confirmed measles cases.'}], 'chunk_is_relevant': True}
        elif schema == 'SearchIterationPlan':
            stage = schema
            result = {'query_batch': [
                {'query': 'Pertussis United States public health cases 2024',
                 'provider_channel': 'official_site_search', 'source_type': 'official_public_health_agency', 'role_hint': 'collection'},
                {'query': 'Pertussis United States surveillance report 2024',
                 'provider_channel': 'official_site_search', 'source_type': 'official_public_health_agency', 'role_hint': 'collection'}]}
        elif schema == 'SearchRefinementDecision':
            stage = schema
            result = {'iteration_index': 1, 'decision': 'stop_sufficient',
                      'decision_reason': 'The result snippets appear sufficient.', 'next_query_batch': []}
        elif schema == 'LLMExtractionOutput':
            stage = 'extraction'
            if model_calls[stage] == 0:
                result = {'records': '{"records":['}  # Deliberately incomplete; must remain a failed operation.
            else:
                if include_distractors and any(quote in user for key, quote in quotes.items() if key.startswith('wrong-')):
                    key, quote = next((key, quote) for key, quote in quotes.items() if key.startswith('wrong-') and quote in user)
                    result = {'records': [{'disease': 'Measles' if key == 'wrong-disease' else 'Pertussis',
                        'country': 'Canada' if key == 'wrong-country' else 'United States',
                        'subnational_location': 'Ontario' if key == 'wrong-country' else 'Oregon',
                        'reporting_period': '2023' if key == 'wrong-year' else '2024',
                        'cases_confirmed': {'wrong-country': 90, 'wrong-year': 91, 'wrong-disease': 92}[key],
                        'evidence_quote': quote}], 'chunk_is_relevant': True}
                    model_calls[stage] += 1
                    return httpx.Response(200, json={
                        'id': 'msg_offline_distractor', 'type': 'message', 'role': 'assistant',
                        'model': 'claude-sonnet-4-6', 'content': [{'type': 'tool_use', 'id': 'distractor',
                            'name': schema, 'input': result}], 'stop_reason': 'tool_use', 'stop_sequence': None,
                        'usage': {'input_tokens': 31, 'output_tokens': 17}})
                region, count = ('Oregon', 7) if 'Oregon' in user else ('Washington', 12)
                quote = quotes['report-b' if region == 'Oregon' else 'report-a']
                result = {'records': [{'disease': 'Pertussis', 'country': 'United States',
                    'subnational_location': region, 'reporting_period': '2024', 'cases_confirmed': count,
                    'evidence_quote': quote}], 'chunk_is_relevant': True}
        else:
            raise AssertionError('Unexpected paid model stage: ' + schema)
        model_calls[stage] += 1
        return httpx.Response(200, json={
            'id': 'msg_offline_integration', 'type': 'message', 'role': 'assistant',
            'model': 'claude-sonnet-4-6', 'content': [{'type': 'tool_use', 'id': 'offline_tool',
                'name': schema, 'input': result}], 'stop_reason': 'tool_use', 'stop_sequence': None,
            'usage': {'input_tokens': 31, 'output_tokens': 17}})
    def build_model(settings=None):
        model = ChatAnthropic(**llm_clients.chat_model_kwargs(settings or llm_clients.get_llm_settings()),
            api_key='offline-placeholder', base_url='https://offline.invalid')
        http_client = httpx.Client(transport=httpx.MockTransport(respond))
        http_clients.append(http_client)
        monkeypatch.setitem(model.__dict__, '_client', anthropic.Anthropic(
            api_key='offline-placeholder', base_url='https://offline.invalid',
            max_retries=0, http_client=http_client))
        return model
    monkeypatch.setattr(llm_clients, 'build_chat_model', build_model)

    try:
        summary = runner.run_workflow(argparse.Namespace(config=None, resume_session=None,
            live_status=False, write_run_notebook=False))
    finally:
        for client in http_clients:
            client.close()

    session = tmp_path / 'sessions/offline-resilience'
    readiness = json.loads((session / 'acquisition-preflight.json').read_text())
    assert readiness['ready'] and readiness['dynamic_page'] and readiness['scanned_page']
    manifest = json.loads((session / 'result_manifest.json').read_text())
    package = json.loads((session / 'collection/final_package.json').read_text())
    assert manifest == summary['result_manifest'] == package['result_manifest']
    assert manifest['technical_completion'] == 'completed'
    assert manifest['counts']['qualified_observations'] >= 1
    assert model_calls['preflight'] == 1 and model_calls['extraction'] >= 2
    assert 2 <= len(search_calls) <= 4
    assert base + 'missing' in fetched_urls
    budget = manifest['budget']
    assert budget['used']['search'] == len(search_calls)
    assert budget['used']['extraction'] == model_calls['extraction']
    assert all(amount <= budget['limits'].get(kind, budget['model_default_limit'])
               for kind, amount in budget['used'].items())
    progress = manifest['source_progress']
    assert 1 <= progress['qualified_evidence_sources'] <= 2
    assert progress['qualified_evidence_sources'] < progress['discovered_unique_sources']
    assert package['aggregate_dataset']
    assert {row['cases_confirmed'] for row in package['aggregate_dataset']} <= {7, 12}
    assert all(row['disease'].lower() == 'pertussis' for row in package['aggregate_dataset'])
    assert manifest['counts']['qualified_observations'] == len(package['qualified_records'])
    source_urls = {row['source_id']: row['canonical_url'] for row in package['source_registry']}
    if include_distractors:
        assert {base + key for key in ('wrong-country', 'wrong-year', 'wrong-disease')} <= set(source_urls.values())
    expected = {'report-a': ('Washington', 12), 'report-b': ('Oregon', 7)}
    for record in package['qualified_records']:
        region, count = expected[source_urls[record['source_id']].rsplit('/', 1)[-1]]
        assert (record['country'], record['subnational_location'], record['cases_confirmed']) == (
            'United States', region, count)
    with sqlite3.connect(session / '.universal/operations.sqlite') as db:
        failures = db.execute("SELECT kind,error,response FROM operations WHERE status='failed'").fetchall()
    assert any(kind == 'search' and 'TimeoutError' in error for kind, error, _ in failures)
    assert any(kind == 'extraction' and response for kind, _, response in failures)
    acquisition_artifacts = [json.loads(path.read_text(encoding='utf-8')) for path in (session / 'acquisition').rglob('*.json')]
    failed_documents = [doc for doc in acquisition_artifacts if isinstance(doc, dict) and doc.get('http_status_code') == 404]
    assert failed_documents and all(not doc['content_readable'] for doc in failed_documents)
    assert all((session / doc['raw_artifact_path']).is_file() for doc in failed_documents)
