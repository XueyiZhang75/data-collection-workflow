"""Acquisition failures preserve provenance and do not abort other sources."""
import hashlib
import io
import pytest
import requests
from PIL import Image

from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.session_runtime import RunContext, ResumeMismatch


@pytest.fixture
def response_transport(monkeypatch):
    responses = {}
    class Response:
        def __init__(self, url):
            self.url = url
            self.status_code = 200
            self.headers = {'content-type': responses[url][0]}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size): yield responses[self.url][1]
    monkeypatch.setattr(requests, 'get', lambda url, **kwargs: Response(url))
    return responses


@pytest.mark.parametrize('content_type,body', [
    ('application/json', b'not JSON'),
    ('application/pdf', b'%PDF-broken'),
])
def test_bad_document_returns_parse_failure_and_next_source_survives(tmp_path, response_transport, content_type, body):
    response_transport['https://example.test/bad'] = (content_type, body)
    response_transport['https://example.test/good'] = ('text/plain', b'France reported 12 cases.')
    runtime = RunContext(tmp_path, {})
    with runtime.activate():
        bad = acquisition.acquire_document('https://example.test/bad', source_id='bad', session_dir=tmp_path)
        good = acquisition.acquire_document('https://example.test/good', source_id='good', session_dir=tmp_path)
    assert bad['request_success'] is True
    assert bad['acquisition_status'] == 'parse_error'
    assert bad['parse_status'] == 'failed'
    assert bad['content_readable'] is False
    assert bad['parse_error']
    assert bad['content_hash'] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / bad['raw_artifact_path']).read_bytes() == body
    assert good['content_readable'] is True
    assert runtime.ledger.snapshot()['used']['fetch'] == 2


def test_failed_browser_returns_original_response_and_remains_charged(tmp_path, response_transport, monkeypatch):
    body = b'<p>Loading...</p><script>app()</script>'
    response_transport['https://example.test/shell'] = ('text/html', body)
    def fail(*args): raise TimeoutError('navigation timed out')
    monkeypatch.setattr(acquisition, '_browser', fail)
    runtime = RunContext(tmp_path, {})
    with runtime.activate():
        doc = acquisition.acquire_document('https://example.test/shell', source_id='shell', session_dir=tmp_path)
    assert doc['acquisition_status'] == 'browser_error'
    assert doc['request_success'] is True
    assert doc['content_readable'] is False
    assert (tmp_path / doc['raw_artifact_path']).read_bytes() == body
    assert 'navigation timed out' in doc['fetch_error']
    assert runtime.ledger.snapshot()['used'] == {'fetch': 2, 'fetch_ordinary': 1, 'browser': 1}
    assert runtime.ledger.snapshot()['operations'] == {'completed': 1, 'failed': 1}


def test_failed_ocr_returns_raw_pdf_and_failed_charge(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new('RGB', (300, 100), 'white').save(buffer, format='PDF')
    body = buffer.getvalue()
    def fail(*args): raise RuntimeError('OCR subprocess failed')
    monkeypatch.setattr(acquisition, '_ocr', fail)
    runtime = RunContext(tmp_path, {})
    with runtime.activate():
        doc = acquisition.parse_response(body, url='local:scan', source_id='scan', session_dir=tmp_path)
    assert doc['acquisition_status'] == 'parse_error'
    assert 'OCR subprocess failed' in doc['parse_error']
    assert (tmp_path / doc['raw_artifact_path']).read_bytes() == body
    assert runtime.ledger.snapshot()['used']['ocr'] == 1
    assert runtime.ledger.snapshot()['operations'] == {'failed': 1}


def test_request_error_does_not_fabricate_response_artifact(tmp_path, monkeypatch):
    def fail(*args, **kwargs): raise requests.ConnectionError('connection refused')
    monkeypatch.setattr(requests, 'get', fail)
    runtime = RunContext(tmp_path, {})
    with runtime.activate():
        doc = acquisition.acquire_document('https://example.test/missing', source_id='missing', session_dir=tmp_path)
    assert doc['acquisition_status'] == 'request_error'
    assert doc['request_success'] is False
    assert doc['is_live_fetched'] is False
    for field in ('content_hash', 'raw_artifact_path', 'http_status_code', 'retrieved_at'):
        assert not doc.get(field)
    assert runtime.ledger.snapshot()['used']['fetch'] == 1


def test_parse_keeps_resume_mismatch_fatal(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    Image.new('RGB', (300, 100), 'white').save(buffer, format='PDF')
    def fail(*args, **kwargs): raise ResumeMismatch('in_doubt operation')
    monkeypatch.setattr(acquisition, '_call', fail)
    with pytest.raises(ResumeMismatch, match='in_doubt'):
        acquisition.parse_response(buffer.getvalue(), url='local:scan', source_id='scan', session_dir=tmp_path)


def test_partial_parse_failure_keeps_explicit_reparse_gap_after_span_extraction():
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery
    doc = {'document_id': 'd', 'source_id': 's', 'content_hash': 'raw', 'raw_artifact_path': 'acquisition/raw',
           'content_readable': True, 'clean_text': 'France reported 12 cases.',
           'acquisition_status': 'parse_error', 'parse_status': 'parsed_partial', 'fetch_error': 'OCR subprocess failed'}
    state = {'source_registry': [{'source_id': 's'}], 'documents': [doc],
             'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': 'raw'}],
             'extraction_attempted_chunk_ids': ['c']}
    gaps = assess_collection_gaps(state)
    assert len(gaps) == 1
    assert gaps[0].kind == 'parse_failed'
    assert 'budget' not in gaps[0].reason
    plan = plan_recovery(gaps, state=state, budget={'remaining': {'fetch': 0, 'extraction': 0, 'ocr': 1}})
    assert [(action.kind, action.document_hash) for action in plan.actions] == [('reparse', 'raw')]
    state['recovery_action_history'] = [{'action_id': plan.actions[0].action_id, 'status': 'completed'}]
    assert not plan_recovery(gaps, state=state, budget={'remaining': {'ocr': 1}}).actions


@pytest.mark.parametrize('completed_hash,expected_gap', [('raw', False), ('other-version', True)])
def test_only_completed_same_version_settles_parse_failure(completed_hash, expected_gap):
    from data_collection_workflow.workflow_recovery import assess_collection_gaps
    partial = {'document_id': 'partial', 'source_id': 's', 'content_hash': 'raw',
               'raw_artifact_path': 'acquisition/raw', 'clean_text': '12 cases', 'content_readable': True,
               'acquisition_status': 'parse_error', 'parse_status': 'parsed_partial'}
    complete = {**partial, 'document_id': 'complete', 'content_hash': completed_hash,
                'acquisition_status': 'readable', 'parse_status': 'parsed'}
    for documents in ([partial, complete], [complete, partial]):
        state = {'source_registry': [{'source_id': 's'}], 'documents': documents,
                 'evidence_chunks': [{'source_id': 's', 'chunk_id': 'a', 'document_hash': 'raw'},
                                     {'source_id': 's', 'chunk_id': 'b', 'document_hash': completed_hash}],
                 'extraction_attempted_chunk_ids': ['a', 'b']}
        assert any(g.kind == 'parse_failed' for g in assess_collection_gaps(state)) is expected_gap


def test_same_text_completed_reparse_replaces_failed_metadata_idempotently():
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, merge_recovery_delta, RecoveryDelta
    partial = {'document_id': 'old', 'source_id': 's', 'content_hash': 'raw', 'text_hash': 'same-text',
               'parser_version': 'same-parser', 'raw_artifact_path': 'acquisition/raw', 'clean_text': '12 cases',
               'content_readable': True, 'acquisition_status': 'parse_error', 'parse_status': 'parsed_partial',
               'fetch_error': 'OCR failed'}
    complete = {**partial, 'document_id': 'new', 'acquisition_status': 'readable', 'parse_status': 'parsed', 'fetch_error': None}
    state = {'source_registry': [{'source_id': 's'}], 'documents': [partial],
             'evidence_chunks': [{'source_id': 's', 'document_id': 'old', 'chunk_id': 'c', 'document_hash': 'raw'}],
             'extraction_attempted_chunk_ids': ['c']}
    merged = merge_recovery_delta(state, RecoveryDelta(documents=[complete]))
    assert len(merged['documents']) == 1
    assert merged['documents'][0]['acquisition_status'] == 'readable'
    assert merged['documents'][0]['document_id'] == 'old'
    assert not assess_collection_gaps(merged)
    assert merge_recovery_delta(merged, RecoveryDelta(documents=[partial])) == merged


def test_recovery_returned_parse_failure_is_not_a_completed_action(tmp_path, monkeypatch):
    from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery, execute_recovery, merge_recovery_delta
    from data_collection_workflow.nodes import content_processing
    monkeypatch.setattr(content_processing, 'document_quality_check', lambda state: {})
    monkeypatch.setattr(content_processing, 'evidence_chunking_and_data_presence_flagging', lambda state: {'evidence_chunks': []})
    runtime = RunContext(tmp_path, {})
    doc = acquisition.parse_response(b'{', url='local:invalid-json', source_id='s', session_dir=tmp_path, content_type='application/json')
    doc['document_id'] = 'd'
    state = {'source_registry': [{'source_id': 's', 'source_role_final': 'excluded'}], 'documents': [doc]}
    with runtime.activate():
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=runtime.ledger)
        assert [action.kind for action in plan.actions] == ['reparse']
        delta = execute_recovery(plan, context=runtime, artifacts=state, budget=runtime.ledger)
        assert delta.documents[0]['acquisition_status'] == 'parse_error'
        assert delta.actions[0]['status'] == 'failed'
        assert 'JSONDecodeError' in delta.actions[0]['error']
        merged = {**state, **merge_recovery_delta(state, delta), 'recovery_action_history': delta.actions}
        retry = plan_recovery(assess_collection_gaps(merged), state=merged, budget=runtime.ledger)
        assert len(retry.actions) == 1
        merged['recovery_action_history'] *= 2
        assert not plan_recovery(assess_collection_gaps(merged), state=merged, budget=runtime.ledger).actions
    assert runtime.ledger.snapshot()['used'] == {}


@pytest.mark.parametrize('status', ['parse_error', 'budget_exhausted'])
def test_extract_action_does_not_inherit_an_unrelated_parse_failure(tmp_path, monkeypatch, status):
    from data_collection_workflow.workflow_recovery import execute_recovery, RecoveryAction, RecoveryPlan
    from data_collection_workflow.nodes import extraction
    monkeypatch.setattr(extraction, 'structured_extraction', lambda state: {'raw_records': [], 'extraction_attempted_chunk_ids': ['c']})
    runtime = RunContext(tmp_path, {})
    state = {'documents': [{'source_id': 'other', 'acquisition_status': status, 'fetch_error': 'other document',
                            'acquisition_incomplete': status == 'budget_exhausted', 'budget_exhausted_kind': 'ocr'}],
             'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'text': '12 cases'}]}
    delta = execute_recovery(RecoveryPlan([RecoveryAction('extract', 'c', 'a', 'gain')]), context=runtime, artifacts=state, budget=runtime.ledger)
    assert delta.actions[0]['status'] == 'completed'
    assert delta.attempted_chunk_ids == ['c']
