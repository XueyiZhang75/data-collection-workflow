"""Real local HTTP acquisition survives exhausted session fetch budgets."""
import hashlib
import json
import os
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from data_collection_workflow.document_acquisition import acquire_document
from data_collection_workflow.session_runtime import RunContext


@pytest.fixture
def reports_server():
    requests = Counter()
    text = 'Pertussis in Washington, United States: 12 confirmed cases during 2024.'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests[self.path] += 1
            body = text.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests, text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _runtime(tmp_path):
    return RunContext(tmp_path / 'session', {
        'pipeline_mode': 'evidence',
        'universal': {'budget_limits': {'fetch_ordinary': 1, 'fetch': 3}},
    })


def _entry(base, number):
    return {
        'source_id': f's{number}',
        'canonical_url': f'{base}/report{number}',
        'title': 'Pertussis surveillance in Washington, United States, 2024',
        'publisher': 'Local fixture publisher',
        'source_type': 'official_public_health_agency',
        'status': 'ready_for_content_fetch',
        'final_screening_decision': 'include_for_content_fetch',
        'ready_for_content_fetch': True,
        'requires_human_review': False,
        'discovery_method': 'web_search',
        'provider_channel': 'web_search',
        'search_rank': number,
        'source_role': 'data_source',
        'source_role_final': 'collection',
        'credibility_score': 0.88,
        'credibility_level': 'high',
        'risk_flags': [],
    }


def _live_node_env(monkeypatch):
    from data_collection_workflow.environment import WORKFLOW_ENV_NAMES
    for key in list(os.environ):
        if key in WORKFLOW_ENV_NAMES or key.startswith('HDC_'):
            monkeypatch.delenv(key, raising=False)
    for key, value in {
        'PIPELINE_MODE': 'evidence',
        'ENABLE_LIVE_FETCH': 'true',
        'FETCH_SEARCH_DERIVED_SOURCES': 'true',
        'FETCH_MAX_SEARCH_DERIVED_SOURCES': '3',
        'FETCH_MAX_TOTAL_SOURCES': '3',
        'USE_FIXTURE_DOCUMENTS': 'false',
        'ENABLE_LLM_SOURCE_IDENTITY': 'false',
        'LLM_SOURCE_IDENTITY_POST_FETCH': 'false',
        'ENABLE_LLM_SOURCE_CREDIBILITY': 'false',
    }.items():
        monkeypatch.setenv(key, value)


def test_content_node_retains_fetched_evidence_and_exports_after_budget_exhaustion(reports_server, tmp_path, monkeypatch):
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    from data_collection_workflow.export import export_final_data_package

    _live_node_env(monkeypatch)
    base, requests, text = reports_server
    runtime = _runtime(tmp_path)
    state = {'source_registry': [_entry(base, n) for n in range(1, 4)], 'collection_trace': []}
    with runtime.activate():
        result = content_fetch_and_parse(state)
        docs = result['documents']
        readable = [doc for doc in docs if doc['content_readable']]
        deferred = [doc for doc in docs if doc['acquisition_status'] == 'budget_exhausted']
        assert len(readable) == 1
        assert len(deferred) == 2
        assert all(doc['budget_exhausted_kind'] == 'fetch_ordinary' for doc in deferred)
        assert all(not doc['request_success'] and not doc['content_readable'] for doc in deferred)
        assert sum(requests.values()) == 1
        doc = readable[0]
        assert doc['clean_text'] == text
        assert doc['text_hash'] == hashlib.sha256(text.encode()).hexdigest()
        assert (runtime.session_dir / doc['raw_artifact_path']).is_file()
        assert runtime.ledger.snapshot()['used'] == {'fetch': 1, 'fetch_ordinary': 1}
        chunk = {'chunk_id': 'c1', 'source_id': doc['source_id'], 'document_hash': doc['content_hash'],
                 'text': text, 'char_start': 0, 'char_end': len(text)}
        row = {'record_id': 'r1', 'source_id': doc['source_id'], 'chunk_id': 'c1', 'disease': 'Pertussis',
               'country': 'United States', 'subnational_location': 'Washington', 'reporting_period': '2024',
               'cases_confirmed': 12}
        finalized = final_data_package_builder({**state, **result, 'evidence_chunks': [chunk],
                                                'normalized_records': [row], 'human_review_enabled': False})
        package = finalized['final_data_package']
        export_final_data_package(package, tmp_path / 'export')
    saved = json.loads((tmp_path / 'export' / 'final_package.json').read_text(encoding='utf-8'))
    assert saved['result_manifest']['counts']['qualified_observations'] == 1
    assert saved['aggregate_dataset'][0]['cases_confirmed'] == 12
    assert saved['result_manifest']['budget']['used'] == {'fetch': 1, 'fetch_ordinary': 1}
    assert saved['result_manifest']['acquisition']['budget_deferred_document_count'] == 2
    assert saved['result_manifest']['acquisition']['budget_exhausted_causes'] == {'fetch_ordinary': 2}
    assert 'Acquisition incomplete' in (tmp_path / 'export' / 'workflow_console.html').read_text(encoding='utf-8')


def test_completed_response_cache_remains_usable_after_fetch_budget_exhaustion(reports_server, tmp_path):
    base, requests, text = reports_server
    runtime = _runtime(tmp_path)
    with runtime.activate():
        first = acquire_document(base + '/first', source_id='first', session_dir=runtime.session_dir)
        denied = acquire_document(base + '/second', source_id='second', session_dir=runtime.session_dir)
        assert denied['acquisition_status'] == 'budget_exhausted'
        cached = acquire_document(base + '/first', source_id='first', session_dir=runtime.session_dir)
    assert cached['content_readable']
    assert cached['clean_text'] == first['clean_text'] == text
    assert cached['content_hash'] == first['content_hash']
    assert requests == {'/first': 1}
    assert runtime.ledger.snapshot()['used'] == {'fetch': 1, 'fetch_ordinary': 1}


def test_verified_priority_still_runs_when_ordinary_pool_is_exhausted(reports_server, tmp_path):
    base, requests, _ = reports_server
    runtime = _runtime(tmp_path)
    url = base + '/priority'
    verified = {'source_id': 'priority', 'verified_url': url, 'source_identity_verified': True,
                'must_fetch': True, 'task_fit_verified': True}
    with runtime.activate():
        acquire_document(base + '/ordinary', source_id='ordinary', session_dir=runtime.session_dir)
        denied = acquire_document(url, source_id='priority', session_dir=runtime.session_dir,
                                  priority_context={**verified, 'source_identity_verified': False})
        assert denied['acquisition_status'] == 'budget_exhausted'
        priority = acquire_document(url, source_id='priority', session_dir=runtime.session_dir,
                                    priority_context=verified)
    assert priority['content_readable']
    assert requests == {'/ordinary': 1, '/priority': 1}
    assert runtime.ledger.snapshot()['used'] == {'fetch': 2, 'fetch_ordinary': 1}
