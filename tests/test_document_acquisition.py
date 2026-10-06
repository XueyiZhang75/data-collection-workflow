"""Offline acquisition contracts; no historical or paid sources."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from data_collection_workflow.document_acquisition import acquire_document, parse_response, preflight_acquisition

@pytest.fixture
def local_url():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            fixtures = {
                '/table': ('text/html', b'<h1>Units: cases</h1><table><tr><th>Place</th><th>Count</th></tr><tr><td>Example</td><td>12</td></tr></table><p>* provisional</p>'),
                '/csv': ('text/csv', b'Place,Count (cases)\nExample,12\n'),
                '/json': ('application/json', b'{"units":"cases","count":12}'),
                '/text': ('text/plain', b'Example observation 12 cases.'),
                '/headed-dashboard': ('text/html', b'<h1>Surveillance dashboard</h1><div>Loading...</div><script>fetch("/json").then(r=>r.json()).then(x=>window.dashboard=x)</script>'),
                '/dashboard': ('text/html', b'<div>Loading...</div><script>fetch("/json").then(r=>r.json()).then(x=>window.dashboard=x)</script>'),
                '/shell': ('text/html', b'<div id="root">Loading...</div><script>setTimeout(()=>document.getElementById("root").textContent="DYNAMIC PREFLIGHT 731",100)</script>'),
            }
            kind, body = fixtures.get(self.path, ('text/html', b'<h1>Access denied</h1>'))
            self.send_response(302 if self.path == '/redirect' else 404 if self.path == '/error' else 200)
            if self.path == '/redirect':
                self.send_header('Location', '/csv')
            self.send_header('Content-Type', kind)
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()

@pytest.mark.parametrize('path', ['/table', '/csv', '/json', '/text'])
def test_local_formats_preserve_hashes_and_artifacts(local_url, tmp_path, path):
    doc = acquire_document(local_url + path, source_id='s', session_dir=tmp_path)
    assert doc['request_success'] and doc['content_readable']
    assert doc['target_data_status'] == 'unassessed'
    assert doc['text_hash'] == hashlib.sha256(doc['clean_text'].encode()).hexdigest()
    raw = (tmp_path / doc['raw_artifact_path']).read_bytes()
    assert doc['content_hash'] == hashlib.sha256(raw).hexdigest()
    assert doc['locator_spans']
    if path == '/table':
        assert 'Units: cases' in doc['clean_text'] and '* provisional' in doc['clean_text']
        assert doc['tables'][0]['headers'] == ['Place', 'Count']

@pytest.mark.parametrize('status,body,expected', [(200,b'<h1>Access denied</h1>','error_page'), (200,b'<div>Loading...</div><script>x()</script>','shell'), (404,b'not found','http_error')])
def test_success_is_not_readable_target_evidence(tmp_path,status,body,expected):
    doc = parse_response(body, url='https://example.invalid', source_id='s', session_dir=tmp_path, content_type='text/html', status_code=status)
    assert doc['acquisition_status'] == expected
    assert not doc['content_readable']
    assert doc['target_data_status'] == 'unavailable'


def test_session_directory_required():
    with pytest.raises(ValueError, match='session_dir'):
        parse_response(b'text', url='x', source_id='s', session_dir=None)


def test_model_dump_preserves_acquisition_fields(tmp_path):
    from data_collection_workflow.models import Document
    original = parse_response(b'Local evidence.', url='x', source_id='s', session_dir=tmp_path)
    restored = Document(**original).model_dump()
    for field in ['text_hash','raw_artifact_path','locator_spans','acquisition_status','content_readable']:
        assert restored[field] == original[field]


def test_actual_local_dynamic_and_scanned_preflight(tmp_path):
    from pathlib import Path
    paths = Path(__file__).resolve().parents[1] / '.runtime/acquisition-paths.json'
    if not paths.exists():
        pytest.skip('local runtime not installed')
    report = preflight_acquisition({'acquisition_paths_file': str(paths)}, tmp_path)
    assert report['ready'] and report['dynamic_page'] and report['scanned_page']
    assert set(report['languages']) >= {'eng','osd','fra','spa','por','chi_sim'}


def test_dynamic_page_falls_back_to_real_chromium(local_url, tmp_path):
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / '.runtime/acquisition-paths.json').exists():
        pytest.skip('local runtime not installed')
    doc = acquire_document(local_url + '/shell', source_id='s', session_dir=tmp_path)
    assert 'DYNAMIC PREFLIGHT 731' in doc['clean_text']
    assert doc['fetch_provider'] == 'chromium'
    assert doc['response_content_hash'] != doc['content_hash']


def test_unsupported_ocr_language_is_explicit(tmp_path):
    from PIL import Image
    from data_collection_workflow.document_acquisition import _ocr, _paths
    with pytest.raises(ValueError, match='unsupported OCR languages: klingon'):
        _ocr(Image.new('RGB', (100,100), 'white'), _paths({'ocr_languages':'klingon'}))


def test_scanned_pdf_has_word_boxes_and_low_confidence_flag(tmp_path):
    import io
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (1200,250), 'white')
    ImageDraw.Draw(image).text((20,60),'LOCAL SCAN 731',font=ImageFont.truetype('arial.ttf',64),fill='black')
    stream = io.BytesIO()
    image.save(stream,format='PDF')
    doc = parse_response(stream.getvalue(),url='local:scan',source_id='s',session_dir=tmp_path,config={'ocr_min_confidence':101})
    assert 'LOCAL SCAN 731' in doc['clean_text']
    assert doc['ocr_words'] and all(len(w['bbox']) == 4 and w['page']==1 for w in doc['ocr_words'])
    assert 'low_ocr_confidence_candidate' in doc['quality_issues']


def test_adapter_uses_runtime_for_fetch_and_browser(local_url,tmp_path,monkeypatch):
    from data_collection_workflow import session_runtime
    calls=[]
    class Runtime:
        def call(self,kind,payload,fn,**kwargs):
            calls.append(kind)
            return fn()
    monkeypatch.setattr(session_runtime,'get_runtime',lambda:Runtime())
    doc=acquire_document(local_url+'/shell',source_id='s',session_dir=tmp_path)
    assert doc['content_readable']
    assert calls == ['fetch_ordinary','browser_fetch']


def test_network_failure_has_explicit_failed_document(tmp_path):
    doc=acquire_document('http://127.0.0.1:1',source_id='s',session_dir=tmp_path,config={'timeout_seconds':0.1})
    assert doc['acquisition_status']=='request_error'
    assert not doc['request_success'] and not doc['content_readable']
    assert doc['fetch_error']


def test_same_session_cache_and_new_session_fetch(local_url,tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    import sqlite3
    first=RunContext(tmp_path/'first',{})
    with first.activate():
        for _ in range(2):
            acquire_document(local_url+'/csv',source_id='s',session_dir=first.session_dir)
    second=RunContext(tmp_path/'second',{})
    with second.activate():
        acquire_document(local_url+'/csv',source_id='s',session_dir=second.session_dir)
    for context in [first,second]:
        with sqlite3.connect(context.session_dir/'.universal/operations.sqlite') as db:
            assert db.execute("select count(*) from operations where kind='fetch_ordinary'").fetchone()[0] == 1


def test_runtime_cannot_write_into_another_session(local_url,tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime=RunContext(tmp_path/'one',{})
    with runtime.activate(), pytest.raises(ValueError,match='session'):
        acquire_document(local_url+'/csv',source_id='s',session_dir=tmp_path/'two')


def test_dashboard_preserves_xhr_json_bytes_and_locators(local_url,tmp_path):
    doc=acquire_document(local_url+'/dashboard',source_id='s',session_dir=tmp_path)
    assert doc['content_readable']
    response=doc['browser_responses'][0]
    assert response['url'].endswith('/json')
    assert (tmp_path/response['raw_artifact_path']).read_bytes() == b'{"units":"cases","count":12}'
    span=next(span for span in doc['locator_spans'] if 'response_url' in span)
    assert doc['clean_text'][span['char_start']:span['char_end']] == response['clean_text']
    assert doc['text_hash']==hashlib.sha256(doc['clean_text'].encode()).hexdigest()


def test_table_caption_and_row_locators_survive(tmp_path):
    body=b'<h1>Report</h1><table><caption>Confirmed cases, Country X, 2025</caption><thead><tr><th>Age</th><th>Cases</th></tr></thead><tbody><tr><td>Children</td><td>3</td></tr></tbody><tfoot><tr><td colspan="2">* provisional</td></tr></tfoot></table>'
    doc=parse_response(body,url='https://example.invalid',source_id='s',session_dir=tmp_path,content_type='text/html')
    assert 'Confirmed cases, Country X, 2025' in doc['clean_text']
    spans=[s for s in doc['locator_spans'] if s.get('table_id')=='table_1']
    assert any(doc['clean_text'][s['char_start']:s['char_end']]=='Children | 3' and s['row_id']==1 for s in spans)
    assert any(doc['clean_text'][s['char_start']:s['char_end']]=='Age | Cases' and s['row_id']==0 for s in spans)
    assert '* provisional' in doc['clean_text']


def test_headed_loading_dashboard_captures_real_xhr(local_url,tmp_path):
    doc=acquire_document(local_url+'/headed-dashboard',source_id='s',session_dir=tmp_path)
    assert doc.get('fetch_provider')=='chromium'
    assert doc['browser_responses'][0]['clean_text'].find('12') >= 0


@pytest.mark.parametrize('verified', [True, False])
def test_fetch_priority_requires_identity_and_task_fit(local_url,tmp_path,monkeypatch,verified):
    from data_collection_workflow import session_runtime
    calls=[]
    class Runtime:
        def call(self,kind,payload,fn,**kwargs):
            calls.append(kind)
            return fn()
    monkeypatch.setattr(session_runtime,'get_runtime',lambda:Runtime())
    url=local_url+'/csv'
    priority=dict(source_id='s',verified_url=url,source_identity_verified=verified,must_fetch=True,task_fit_verified=True)
    doc=acquire_document(url,source_id='s',session_dir=tmp_path,priority_context=priority)
    assert doc['content_readable']
    assert calls==['fetch_priority' if verified else 'fetch_ordinary']


def test_embedded_table_footnote_survives_exactly(tmp_path):
    body=b'<table><caption>Confirmed cases</caption><tr><th>Cases</th></tr><tr><td>3</td></tr><div>* Counts exclude imported cases.</div></table>'
    doc=parse_response(body,url='x',source_id='s',session_dir=tmp_path,content_type='text/html')
    assert '* Counts exclude imported cases.' in doc['clean_text']
    assert any(s.get('table_id')=='table_1' and doc['clean_text'][s['char_start']:s['char_end']]=='* Counts exclude imported cases.' for s in doc['locator_spans'])


@pytest.mark.parametrize('change', [{'source_id':'other'}, {'verified_url':'https://other.invalid'}, {'must_fetch':False}, {'task_fit_verified':False}, {'source_identity_verified':'true'}])
def test_priority_context_must_match_every_verified_condition(local_url,tmp_path,monkeypatch,change):
    from data_collection_workflow import session_runtime
    calls=[]
    class Runtime:
        def call(self,kind,payload,fn,**kwargs):
            calls.append(kind)
            return fn()
    monkeypatch.setattr(session_runtime,'get_runtime',lambda:Runtime())
    url=local_url+'/csv'
    context=dict(source_id='s',verified_url=url,source_identity_verified=True,must_fetch=True,task_fit_verified=True)
    context.update(change)
    acquire_document(url,source_id='s',session_dir=tmp_path,priority_context=context)
    assert calls==['fetch_ordinary']


@pytest.mark.parametrize('limits,kind', [({'fetch_ordinary': 0}, 'fetch_ordinary'), ({'fetch': 0}, 'fetch')])
def test_unstarted_fetch_budget_deferral_has_no_fabricated_response(local_url, tmp_path, limits, kind):
    from data_collection_workflow.models import Document
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': limits}})
    with runtime.activate():
        doc = acquire_document(local_url + '/csv', source_id='s', session_dir=runtime.session_dir)
    saved = Document(**doc).model_dump()
    assert saved['acquisition_status'] == 'budget_exhausted'
    assert saved['budget_exhausted_kind'] == kind
    assert not saved['is_live_fetched'] and not saved['request_success']
    assert saved['raw_artifact_path'] is None and saved['content_hash'] is None
    assert saved['http_status_code'] is None and saved['retrieved_at'] is None
    assert runtime.ledger.snapshot()['used'] == {}


@pytest.mark.parametrize('limits,kind', [({'browser': 0}, 'browser'), ({'fetch': 1}, 'fetch')])
def test_browser_budget_deferral_preserves_native_response(local_url, tmp_path, limits, kind):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': limits}})
    with runtime.activate():
        doc = acquire_document(local_url + '/shell', source_id='s', session_dir=runtime.session_dir)
    assert doc['acquisition_status'] == 'budget_exhausted'
    from data_collection_workflow.models import Document
    doc = Document(**doc).model_dump()
    assert doc['is_shell']
    assert doc['budget_exhausted_kind'] == kind
    assert doc['request_success'] and not doc['content_readable']
    assert b'Loading...' in (runtime.session_dir / doc['raw_artifact_path']).read_bytes()
    assert doc['content_hash'] and doc['clean_text'] == 'Loading...'
    assert runtime.ledger.snapshot()['used'] == {'fetch': 1, 'fetch_ordinary': 1}


def _two_page_pdf(page_text="Example surveillance: 12 confirmed cases during 2024."):
    # Self-contained PDF: one text page and one blank scan-like page; no PDF writer dependency.
    text = b'BT /F1 16 Tf 50 700 Td (' + page_text.encode('ascii') + b') Tj ET'
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length ' + str(len(text)).encode() + b' >>\nstream\n' + text + b'\nendstream']
    body = bytearray(b'%PDF-1.4\n')
    offsets = []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body.extend(str(number).encode() + b' 0 obj\n' + obj + b'\nendobj\n')
    xref = len(body)
    body.extend(b'xref\n0 7\n0000000000 65535 f \n')
    for offset in offsets:
        body.extend(f'{offset:010d} 00000 n \n'.encode())
    body.extend(b'trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n' + str(xref).encode() + b'\n%%EOF')
    return bytes(body)


def test_ocr_budget_deferral_preserves_other_pdf_pages(tmp_path):
    from data_collection_workflow.models import Document
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': {'ocr': 0}}})
    body = _two_page_pdf()
    with runtime.activate():
        doc = parse_response(body, url='local:two-pages', source_id='s', session_dir=runtime.session_dir)
    doc = Document(**doc).model_dump()
    assert doc['acquisition_status'] == 'budget_exhausted'
    assert doc['budget_exhausted_kind'] == 'ocr'
    assert doc['unprocessed_pages'] == [2]
    from data_collection_workflow.nodes.content_processing import _document_task_usability
    native_usability = _document_task_usability({**doc, 'parse_status': 'parsed'}, {})
    assert _document_task_usability(doc, {}) == native_usability
    assert doc['parse_status'] == 'parsed_partial' and doc['content_readable']
    assert '12 confirmed cases during 2024' in doc['clean_text']
    assert doc['locator_spans'][0]['page'] == 1
    assert (runtime.session_dir / doc['raw_artifact_path']).read_bytes() == body
    assert runtime.ledger.snapshot()['used'] == {}



def test_redirect_destination_budget_deferral_keeps_only_received_response(local_url, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': {'fetch_ordinary': 1}}})
    with runtime.activate():
        doc = acquire_document(local_url + '/redirect', source_id='s', session_dir=runtime.session_dir)
    assert doc['acquisition_status'] == 'budget_exhausted'
    assert doc['budget_exhausted_kind'] == 'fetch_ordinary'
    assert doc['final_url'] == local_url + '/redirect' and doc['http_status_code'] == 302
    assert doc['request_success'] and not doc['content_readable']
    assert doc['clean_text'] == '' and doc['locator_spans'] == []
    assert (runtime.session_dir / doc['raw_artifact_path']).read_bytes() == b'<h1>Access denied</h1>'
    assert runtime.ledger.snapshot()['used'] == {'fetch': 1, 'fetch_ordinary': 1}


def test_failed_action_attempt_cap_defers_without_new_charge(local_url, tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {})
    with runtime.activate():
        for _ in range(2):
            failed = acquire_document(local_url + '/csv', source_id='s', session_dir=runtime.session_dir, config={'max_bytes': 1})
            assert failed['acquisition_status'] == 'request_error'
        deferred = acquire_document(local_url + '/csv', source_id='s', session_dir=runtime.session_dir, config={'max_bytes': 1})
    assert deferred['acquisition_status'] == 'budget_exhausted'
    assert deferred['budget_exhausted_kind'] == 'action_attempts'
    assert not deferred['is_live_fetched'] and not deferred['request_success']
    assert runtime.ledger.snapshot()['used'] == {'fetch': 2, 'fetch_ordinary': 2}


def test_in_doubt_dispatch_is_not_swallowed_as_budget_deferral(local_url, tmp_path):
    from data_collection_workflow.document_acquisition import PARSER_VERSION
    from data_collection_workflow.session_runtime import RunContext, ResumeMismatch
    runtime = RunContext(tmp_path / 'session', {})
    url = local_url + '/csv'
    runtime.ledger.begin('fetch_ordinary', {'fingerprint': runtime.fingerprint, 'input': {
        'parser_version': PARSER_VERSION, 'url': url, 'max_bytes': 20_000_000}})
    with runtime.activate(), pytest.raises(ResumeMismatch, match='in_doubt'):
        acquire_document(url, source_id='s', session_dir=runtime.session_dir)
    assert runtime.ledger.snapshot()['used'] == {'fetch': 1, 'fetch_ordinary': 1}



def test_ocr_budget_keeps_short_native_page_evidence(tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    runtime = RunContext(tmp_path / 'session', {'universal': {'budget_limits': {'ocr': 0}}})
    with runtime.activate():
        doc = parse_response(_two_page_pdf('Cases: 12'), url='local:short-text', source_id='s', session_dir=runtime.session_dir)
    assert doc['clean_text'] == 'Cases: 12'
    assert doc['unprocessed_pages'] == [1, 2]
    assert doc['locator_spans'] == [{'char_start': 0, 'char_end': 9, 'page': 1}]
    assert doc['acquisition_incomplete'] and doc['parse_status'] == 'parsed_partial'
    assert runtime.ledger.snapshot()['used'] == {}
