"""UI-only shell routing is structural and preserves actual source content."""
import base64
import hashlib
import socket

import pytest
import requests
from data_collection_workflow import document_acquisition as acquisition


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LANGSMITH_TRACING','false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2','false')
    def denied(*args, **kwargs): raise AssertionError('External network forbidden')
    monkeypatch.setattr(socket.socket,'connect',denied)


def _parse(tmp_path, body):
    return acquisition.parse_response(body, url='https://renamed.example/dashboard',
        source_id='source', session_dir=tmp_path, content_type='text/html')


@pytest.mark.parametrize('placeholder', ['Loading real-time data...', 'Loading data...', 'Loading results...', 'Loading...'])
def test_scripted_ui_only_title_and_placeholder_need_rendering(tmp_path, placeholder):
    raw=('<!doctype html><html><head><title>Example Tracker - Real-time information</title></head>'
        '<body><div>Example Tracker</div><div>'+placeholder+'</div><div id="app"></div>'
        '<script src="/app.js"></script></body></html>').encode()
    doc=_parse(tmp_path, raw)
    assert doc['acquisition_status']=='shell' and doc['is_shell']
    assert not doc['content_readable'] and not doc['parse_eligible']
    assert doc['content_hash']==hashlib.sha256(raw).hexdigest()
    assert (tmp_path/doc['raw_artifact_path']).read_bytes()==raw
    assert placeholder in doc['clean_text']
    for span in doc['locator_spans']:
        assert doc['clean_text'][span['char_start']:span['char_end']]


@pytest.mark.parametrize('content', [
    '<p>Canada reported 12 confirmed cases during 2025.</p>',
    '<p>The patient developed fever and acute renal failure after rodent exposure.</p>',
    '<h1>12 hospitalized patients in Canada</h1>',
    '<h1>2025 cases in Canada</h1>',
    '<table><tr><th>Year</th><th>Cases</th></tr><tr><td>2025</td><td>12</td></tr></table>',
    '<p>The page displays the message Loading real-time data while the interface initializes.</p>',
    '<p>Loading doses were administered to 12 patients.</p>',
])
def test_loading_widget_does_not_hide_actual_short_content(tmp_path, content):
    doc=_parse(tmp_path, ('<title>Clinical report</title><div>Loading...</div>'+content+'<script>analytics()</script>').encode())
    assert doc['content_readable'] and doc['parse_eligible']
    assert doc['acquisition_status']=='readable' and not doc.get('is_shell')


def test_non_scripted_short_report_stays_readable(tmp_path):
    doc=_parse(tmp_path,b'<h1>Situation update</h1><p>12 cases; no deaths.</p>')
    assert doc['content_readable']


def test_loading_words_in_clinical_title_are_not_a_placeholder(tmp_path):
    doc=_parse(tmp_path,b'<h1>Loading doses in clinical practice</h1><p>Fever resolved after treatment.</p><script>analytics()</script>')
    assert doc['content_readable']


def test_ui_shell_uses_existing_browser_and_one_source_target(tmp_path,monkeypatch):
    from test_acquisition_transport import _adaptive_runtime
    from test_evidence_resource_discovery import _env
    _env(monkeypatch)
    raw=b'<title>Observation Tracker</title><div>Observation Tracker</div><div>Loading real-time data...</div><script src="/app.js"></script>'
    url='https://renamed.example/dashboard'
    class Response:
        status_code=200
        headers={'content-type':'text/html'}
        def __init__(self): self.url=url
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def iter_content(self,size): yield raw
    monkeypatch.setattr(requests,'get',lambda *a,**kw:Response())
    calls=[]
    def browser(*args):
        calls.append(args)
        return {'body':base64.b64encode(b'<p>Canada reported 12 confirmed cases during 2025.</p>').decode(),
            'content_type':'text/html','status_code':200,'final_url':url}
    monkeypatch.setattr(acquisition,'_browser',browser)
    runtime=_adaptive_runtime(tmp_path,targets=1,requests=2)
    with runtime.activate():
        doc=acquisition.acquire_document(url,source_id='source',session_dir=tmp_path)
    assert len(calls)==1 and doc['fetch_provider']=='chromium' and doc['content_readable']
    assert (tmp_path/doc['response_artifact_path']).read_bytes()==raw
    used=runtime.ledger.snapshot()['used']
    assert used['source_targets']==1 and used['http_requests']==1 and used['browser']==1
