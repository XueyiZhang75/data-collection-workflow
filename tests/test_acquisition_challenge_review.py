"""Independent challenge-page boundaries and durable-budget integration."""
import base64
import hashlib

import pytest
import requests

from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.nodes import content_processing as cp
from data_collection_workflow.session_runtime import RunContext
from test_acquisition_challenge import CHALLENGE
from test_evidence_resource_discovery import _env, _source

GATE = b'Please enable cookies to continue.'
REPORT = b'<main><h1>Example fever report</h1><p>During 2025, Canada reported 12 Example fever cases.</p></main>'
URL = 'https://publisher.example/report'
TASK = {'disease': 'Example fever', 'location': 'Canada', 'start_date': '2025-01-01', 'end_date': '2025-12-31'}


@pytest.mark.parametrize('body', [
    GATE,
    b'<!doctype html>' + GATE,
    CHALLENGE.replace(b'<body>', b'<body><header>National Research Library</header>'),
    'Veuillez activer les cookies et recharger cette page pour continuer.'.encode('utf-8'),
])
def test_entire_gate_with_html_fragments_chrome_or_french_is_blocked(tmp_path, body):
    doc = acquisition.parse_response(body, url=URL, source_id='s1', session_dir=tmp_path,
                                      content_type='text/html', status_code=203)
    assert doc['acquisition_status'] == 'blocked'
    assert not doc['parse_eligible'] and not doc['content_readable']
    assert doc['fetch_status'] == 'failed'
    assert doc['content_hash'] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / doc['raw_artifact_path']).read_bytes() == body


@pytest.mark.parametrize('body', [
    b'During 2025, Canada reported 12 Example fever cases.',
    ('<aside>Veuillez activer les cookies et recharger cette page pour continuer.</aside>'
     '<main>En 2025, 12 cas de chikungunya ont ete signales a La Reunion.</main>').encode('utf-8'),
    b'<main>The report quotes the instruction "Please enable cookies to continue."</main>',
    b'<article><header>During 2025, Canada reported 12 Example fever cases.</header><p>Please enable cookies to continue.</p></article>',
])
def test_real_content_with_cookie_notice_or_quote_stays_readable(tmp_path, body):
    doc = acquisition.parse_response(body, url=URL, source_id='s1', session_dir=tmp_path, content_type='text/html')
    assert doc['content_readable'] and doc['parse_eligible']
    assert doc['acquisition_status'] == 'readable'
    assert doc['content_hash'] == hashlib.sha256(body).hexdigest()
    for span in doc['locator_spans']:
        assert doc['clean_text'][span['char_start']:span['char_end']]


def _node(tmp_path, monkeypatch, *, mode, requests_limit=5, browser_limit=20):
    _env(monkeypatch)
    monkeypatch.setenv('EXTERNAL_FETCH_ENABLED', 'true' if mode == 'provider_success' else 'false')
    monkeypatch.setenv('EXTERNAL_FETCH_PROVIDER_ORDER', 'native_requests,tavily_extract')
    runtime = RunContext(tmp_path, {'pipeline_mode': 'evidence',
        'universal': {'budget_policy': {'version': 2, 'mode': 'adaptive', 'soft_source_target': 50},
                      'budget_limits': {'source_targets': 1, 'http_requests': requests_limit, 'browser': browser_limit}}})
    class Response:
        status_code = 203
        headers = {'content-type': 'text/html'}
        url = URL
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield CHALLENGE
    monkeypatch.setattr(requests, 'get', lambda *a, **kw: Response())
    browser_calls = []
    def browser(url, config):
        browser_calls.append(url)
        body = REPORT if mode == 'browser_success' else CHALLENGE
        return runtime.call('http_request', {'url': url, 'transport': 'browser-test'},
            lambda: {'body': base64.b64encode(body).decode(), 'content_type': 'text/html',
                     'status_code': 200, 'final_url': url}, source_target=URL)
    monkeypatch.setattr(acquisition, '_browser', browser)
    provider_calls = []
    def provider(*args):
        provider_calls.append(1)
        return {'success': True, 'body': REPORT, 'content_type': 'text/html', 'http_status_code': 200,
                'metadata': {'source_url': URL}}
    monkeypatch.setattr(cp, '_tavily_extract_fetch', provider)
    source = _source(URL)
    source.update(disease_fit='match', geography_fit='match', date_fit='match',
                  target_verification_status='unverified', task_fit_evidence_origin='discovery_metadata',
                  task_fit_assessment_version=1)
    with runtime.activate():
        result = cp.content_fetch_and_parse({'source_registry': [source], 'collection_trace': [], 'structured_task': TASK})
    return runtime, result, browser_calls, provider_calls


def test_unresolved_challenge_does_not_become_fetched_scope_evidence(tmp_path, monkeypatch):
    runtime, result, browser, provider = _node(tmp_path, monkeypatch, mode='blocked')
    row = result['source_registry'][0]
    assert row['task_fit_evidence_origin'] == 'discovery_metadata'
    assert row['disease_fit'] == 'match' and row['geography_fit'] == 'match'
    assert not row.get('task_fit_content_hash')
    assert row.get('target_verification_status') != 'verified_target'
    assert result['documents'][0]['acquisition_status'] == 'blocked'
    assert result['acquisition_frontier']['counts'].get('completed', 0) == 0
    assert result['acquisition_frontier']['counts']['failed'] == 1
    assert browser == [URL] and not provider
    assert runtime.ledger.snapshot()['used']['source_targets'] == 1


@pytest.mark.parametrize('mode,http_calls', [('browser_success', 2), ('provider_success', 3)])
def test_only_successful_actual_fallback_completes_target_and_verifies_body(tmp_path, monkeypatch, mode, http_calls):
    runtime, result, browser, provider = _node(tmp_path, monkeypatch, mode=mode)
    row = result['source_registry'][0]
    doc = result['documents'][0]
    assert result['acquisition_frontier']['counts']['completed'] == 1
    assert doc['content_readable'] and doc['acquisition_status'] == 'readable'
    assert row['task_fit_evidence_origin'] == 'fetched_content'
    assert row['task_fit_content_hash'] == doc['content_hash']
    assert row['target_verification_status'] == 'verified_target'
    used = runtime.ledger.snapshot()['used']
    assert used['source_targets'] == 1 and used['http_requests'] == http_calls and used['browser'] == 1
    assert browser == [URL]
    assert len(provider) == (1 if mode == 'provider_success' else 0)


@pytest.mark.parametrize('http_limit,browser_limit', [(1, 20), (5, 0)])
def test_denied_fallback_keeps_target_deferred_and_discovery_scope_pending(tmp_path, monkeypatch, http_limit, browser_limit):
    runtime, result, browser, provider = _node(tmp_path, monkeypatch, mode='blocked',
        requests_limit=http_limit, browser_limit=browser_limit)
    assert result['acquisition_frontier']['counts'].get('completed', 0) == 0
    assert result['acquisition_frontier']['counts']['budget_deferred'] == 1
    row = result['source_registry'][0]
    assert row['task_fit_evidence_origin'] == 'discovery_metadata'
    assert row['disease_fit'] == 'match' and row['geography_fit'] == 'match'
    assert not row.get('task_fit_content_hash')
    used = runtime.ledger.snapshot()['used']
    assert used['source_targets'] == 1 and used['http_requests'] == 1
    assert used.get('browser', 0) == (1 if browser_limit else 0)
    assert not provider
