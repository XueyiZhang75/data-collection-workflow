"""A successful transport is not proof that the requested content was served."""
import base64
import hashlib

import pytest
import requests

from data_collection_workflow import document_acquisition as acquisition
from test_acquisition_transport import _adaptive_runtime


CHALLENGE = (b'<!doctype html><html><head><title>publisher.example</title></head>'
    b'<body><div hidden><h1>Cookies must be enabled</h1>'
    b'<p>Enable cookies for <span>publisher.example</span> and reload this page to continue.</p>'
    b'</div><footer><a href="/security">Vulnerability Disclosure</a></footer></body></html>')
REPORT = b'<main><h1>Surveillance update</h1><p>During 2025, Canada reported 12 cases.</p></main>'


def _parse(tmp_path, body, mime='text/html', status=200):
    return acquisition.parse_response(body, url='https://publisher.example/article',
        source_id='source', session_dir=tmp_path, content_type=mime, status_code=status)


@pytest.mark.parametrize('status', [200, 203])
def test_cookie_gate_is_blocked_without_discarding_received_artifact(tmp_path, status):
    doc = _parse(tmp_path, CHALLENGE, status=status)
    assert doc['acquisition_status'] == 'blocked'
    assert not doc['parse_eligible'] and not doc['content_readable']
    assert doc['fetch_status'] == 'failed' and doc['parse_status'] == 'failed'
    assert doc['raw_content_complete']
    assert doc['content_hash'] == hashlib.sha256(CHALLENGE).hexdigest()
    assert (tmp_path / doc['raw_artifact_path']).read_bytes() == CHALLENGE
    assert 'Cookies must be enabled' in doc['clean_text']
    for span in doc['locator_spans']:
        assert doc['clean_text'][span['char_start']:span['char_end']]


@pytest.mark.parametrize('body', [
    b'Cookies must be enabled. Enable cookies for publisher.example and reload this page to continue.',
    b'Please enable JavaScript and cookies to continue.',
])
def test_plain_text_access_gate_is_not_readable_evidence(tmp_path, body):
    doc = _parse(tmp_path, body, 'text/plain')
    assert doc['acquisition_status'] == 'blocked'
    assert not doc['parse_eligible']


@pytest.mark.parametrize('body', [
    REPORT,
    b'<aside>We use cookies. Accept cookies to improve your experience.</aside>' + REPORT,
    b'<aside><h2>Cookies must be enabled</h2><p>Enable cookies to continue.</p></aside>' + REPORT,
    b'<p>Enable cookies to continue. During 2025, Canada reported 12 cases.</p>',
    b'<main><h1>Cookie troubleshooting</h1><p>The message "Cookies must be enabled" means that browser settings need review.</p></main>',
])
def test_short_content_and_cookie_notice_with_content_remain_readable(tmp_path, body):
    doc = _parse(tmp_path, body, status=203)
    assert doc['acquisition_status'] == 'readable'
    assert doc['parse_eligible'] and doc['content_readable']


def _transport(monkeypatch, rendered_body):
    url = 'https://publisher.example/article'
    class Response:
        status_code = 203
        headers = {'content-type': 'text/html'}
        def __init__(self): self.url = url
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield CHALLENGE
    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: Response())
    def browser(*args):
        return {'body': base64.b64encode(rendered_body).decode(), 'content_type': 'text/html',
                'status_code': 200, 'final_url': url}
    monkeypatch.setattr(acquisition, '_browser', browser)
    return url


def test_cookie_gate_reaches_existing_browser_transport_and_shared_ledger(tmp_path, monkeypatch):
    from test_evidence_resource_discovery import _env
    _env(monkeypatch)
    url = _transport(monkeypatch, REPORT)
    runtime = _adaptive_runtime(tmp_path, targets=1, requests=3)
    with runtime.activate():
        doc = acquisition.acquire_document(url, source_id='source', session_dir=tmp_path)
    assert doc['fetch_provider'] == 'chromium' and doc['content_readable']
    assert (tmp_path / doc['response_artifact_path']).read_bytes() == CHALLENGE
    used = runtime.ledger.snapshot()['used']
    assert used['source_targets'] == 1 and used['browser'] == 1 and used['http_requests'] == 1


def test_unresolved_cookie_gate_is_failed_frontier_work_not_completed(tmp_path, monkeypatch):
    from test_evidence_resource_discovery import _env, _source
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery
    _env(monkeypatch)
    url = _transport(monkeypatch, CHALLENGE)
    runtime = _adaptive_runtime(tmp_path, targets=1, requests=3)
    state = {'source_registry': [_source(url)], 'collection_trace': [],
        'structured_task': {'disease': 'Example fever', 'location': 'Canada',
                            'start_date': '2025-01-01', 'end_date': '2025-12-31'}}
    with runtime.activate():
        result = content_fetch_and_parse(state)
        merged = {**state, **result}
        plan = plan_recovery(assess_collection_gaps(merged), state=merged, budget=runtime.ledger)
    assert result['acquisition_frontier']['counts'].get('completed', 0) == 0
    assert result['acquisition_frontier']['counts']['failed'] == 1
    assert result['documents'][0]['acquisition_status'] == 'blocked'
    assert not any(action.kind == 'reparse' for action in plan.actions)


def test_unresolved_gate_reaches_enabled_provider_without_new_source_charge(tmp_path, monkeypatch):
    from test_evidence_resource_discovery import _env, _source
    from data_collection_workflow.nodes import content_processing as cp
    from data_collection_workflow.models import ContentFetchRequest
    _env(monkeypatch)
    url = _transport(monkeypatch, CHALLENGE)
    monkeypatch.setattr(cp, '_tavily_extract_fetch', lambda *args: {
        'success': True, 'body': REPORT, 'content_type': 'text/html', 'http_status_code': 200,
        'metadata': {'source_url': url}})
    runtime = _adaptive_runtime(tmp_path, targets=1, requests=3)
    source = _source(url)
    request = ContentFetchRequest(source_id=source['source_id'], url=url, canonical_url=url,
        final_screening_decision='include_for_content_fetch', fetch_purpose='data_extraction')
    with runtime.activate():
        doc = cp._fetch_live_document_with_providers(request, source, None, {
            'external_fetch_enabled': True, 'external_fetch_provider_order': ['native_requests', 'tavily_extract']})
    assert doc.fetch_provider == 'tavily_extract' and doc.content_readable
    used = runtime.ledger.snapshot()['used']
    assert used['source_targets'] == 1 and used['http_requests'] == 2 and used['browser'] == 1
