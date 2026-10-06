"""Independent transport review regressions; all requests are local doubles."""
import io
import json
from pathlib import Path

import pytest
import requests
from PIL import Image

from data_collection_workflow import document_acquisition as acquisition
from test_acquisition_transport import _adaptive_runtime


class Response:
    url = "https://data.example/report"
    def __init__(self, body=b"Dengue Canada 2025: 12 cases.", status=200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = {"content-type":"text/plain", **(headers or {})}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def iter_content(self,size): yield self.body


@pytest.mark.parametrize("statuses", [(500,200),(500,500)])
def test_transient_http_statuses_have_two_billed_attempts_and_retain_failed_raw_bytes(tmp_path, monkeypatch, statuses):
    runtime = _adaptive_runtime(tmp_path)
    hits = []
    def get(url,**kwargs):
        hits.append(url)
        return Response(status=statuses[len(hits)-1])
    monkeypatch.setattr(requests,"get",get)
    with runtime.activate():
        doc=acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
    assert len(hits)==2
    assert doc["content_readable"] == (statuses[-1]==200)
    assert runtime.ledger.snapshot()["used"]=={"source_targets":1,"http_requests":2}
    with runtime.ledger._db() as db:
        rows=db.execute("SELECT status,response FROM operations ORDER BY id").fetchall()
    assert [row["status"] for row in rows]==["failed","completed" if statuses[-1]==200 else "failed"]
    assert json.loads(rows[0]["response"])["status_code"]==500


def test_completed_empty_http_response_is_cached_without_becoming_parse_input(tmp_path, monkeypatch):
    runtime=_adaptive_runtime(tmp_path)
    hits=[]
    def get(url,**kwargs):hits.append(url);return Response(body=b"")
    monkeypatch.setattr(requests,"get",get)
    with runtime.activate():
        first=acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
        second=acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
    assert hits==[Response.url]
    assert first["raw_content_complete"] and second["raw_content_complete"]
    assert not second["parse_eligible"] and not second["content_readable"]


def test_content_length_truncation_is_audited_and_never_text_evidence(tmp_path, monkeypatch):
    runtime=_adaptive_runtime(tmp_path)
    hits=[]
    def get(url,**kwargs):hits.append(url);return Response(body=b"short",headers={"content-length":"100"})
    monkeypatch.setattr(requests,"get",get)
    with runtime.activate():doc=acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
    assert len(hits)==2
    assert not doc["raw_content_complete"] and not doc["parse_eligible"] and not doc["content_readable"]
    assert (tmp_path/doc["raw_artifact_path"]).read_bytes()==b"short"
    assert runtime.ledger.snapshot()["operations"]=={"failed":2}


def test_exhausted_retry_allowance_is_terminal_not_a_budget_extension_candidate(tmp_path, monkeypatch):
    runtime=_adaptive_runtime(tmp_path)
    hits=[]
    def get(url,**kwargs):hits.append(url);return Response(status=500)
    monkeypatch.setattr(requests,"get",get)
    with runtime.activate():
        acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
        second=acquisition.acquire_document(Response.url,source_id="source",session_dir=tmp_path)
    assert len(hits)==2
    assert second["acquisition_status"]!="budget_exhausted"
    assert not second["content_readable"]


def test_low_confidence_image_ocr_retains_candidate_issue_and_bbox(tmp_path, monkeypatch):
    runtime=_adaptive_runtime(tmp_path)
    stream=io.BytesIO();Image.new("RGB",(10,10),"white").save(stream,format="PNG")
    monkeypatch.setattr(acquisition,"_ocr",lambda *_:{"text":"Dengue Canada 2025: 12 cases.","words":[{"text":"12","confidence":40,"bbox":[1,2,3,4],"line":[1,1,1]}]})
    with runtime.activate():doc=acquisition.parse_response(stream.getvalue(),url=Response.url,source_id="image",session_dir=tmp_path,content_type="image/png")
    assert "low_ocr_confidence_candidate" in doc["quality_issues"]
    assert "low_ocr_confidence_candidate" in doc["locator_spans"][0]["quality_issues"]
    assert doc["ocr_words"][0]["bbox"]==[1,2,3,4]
    assert runtime.ledger.snapshot()["used"]=={"ocr":1}


def test_alternative_fetch_uses_original_canonical_target(tmp_path, monkeypatch):
    from test_evidence_resource_discovery import _env,_source
    from data_collection_workflow.nodes import content_processing as cp
    from data_collection_workflow.models import ContentFetchRequest
    _env(monkeypatch)
    runtime=_adaptive_runtime(tmp_path)
    monkeypatch.setattr(requests,"get",lambda *args,**kwargs:Response(status=403))
    monkeypatch.setattr(acquisition,"_browser",lambda *args: (_ for _ in ()).throw(RuntimeError("blocked")))
    monkeypatch.setattr(cp,"_tavily_extract_fetch",lambda *args:{"success":True,"body":b"Dengue Canada 2025: 12 cases.","content_type":"text/plain","http_status_code":200})
    canonical=Response.url
    url=canonical+"?download=1"
    entry=_source(canonical)
    request=ContentFetchRequest(source_id=entry["source_id"],url=url,canonical_url=canonical,
        final_screening_decision="include_for_content_fetch",fetch_purpose="data_extraction")
    with runtime.activate():doc=cp._fetch_live_document_with_providers(request,entry,None,{
        "external_fetch_enabled":True,"external_fetch_provider_order":["native_requests","tavily_extract"]})
    assert doc.content_readable
    assert runtime.ledger.snapshot()["used"]["source_targets"]==1
    assert doc.source_target_id==canonical


def test_readable_partial_browser_result_remains_retryable_in_persistent_frontier(tmp_path, monkeypatch):
    import base64
    from test_acquisition_queue import setup_node
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    runtime,state,_=setup_node(monkeypatch,tmp_path)
    state['source_registry']=state['source_registry'][:1]
    runtime.config['universal']['acquisition']={'acquisition_strategy':'browser'}
    calls=[]
    def browser(url,config):
        calls.append(url)
        runtime.call('http_request',{'url':url,'try':len(calls)},lambda:{'body':'dispatched'},source_target=config['_source_target_id'])
        response={'body':base64.b64encode(b'<h1>Example fever Canada 2025: 12 cases.</h1>').decode(),
            'final_url':url,'status_code':200,'content_type':'text/html','browser_responses':[]}
        if len(calls)==1:response['request_errors']=[{'reason':'request_failed','url':url+'/data'}]
        return response
    monkeypatch.setattr(acquisition,'_browser',browser)
    with runtime.activate():first=content_fetch_and_parse(state)
    row=runtime.frontier.snapshot()['items'][0]
    assert row['status']=='failed' and row['result_ref']
    assert first['documents'][0]['content_readable'] and first['documents'][0]['acquisition_incomplete']
    assert runtime.frontier.retry(row['target_id'])
    with runtime.activate():second=content_fetch_and_parse(first)
    assert len(calls)==2
    assert runtime.frontier.snapshot()['items'][0]['status']=='completed'
    assert any(doc['content_readable'] and not doc['acquisition_incomplete'] for doc in second['documents'])
    assert [row['status'] for row in runtime.ledger.operation_audit() if row['kind']=='browser_navigation']==['failed','completed']
