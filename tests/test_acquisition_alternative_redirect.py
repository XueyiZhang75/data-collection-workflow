"""Alternative acquisition cannot hide redirect HTTP requests from the ledger."""
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import pytest
import requests
from data_collection_workflow.models import ContentFetchRequest
from data_collection_workflow.nodes import content_processing as content


@pytest.mark.parametrize('revision',['evidence','legacy'])
def test_alternative_api_redirect_is_not_an_unmetered_evidence_request(tmp_path,monkeypatch,revision):
    visits=[]
    target='https://public.example/report'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            visits.append('POST')
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            self.send_response(302)
            self.send_header('Location','/redirected')
            self.end_headers()
        def do_GET(self):
            visits.append('GET')
            body=json.dumps({'results':[{'url':target,'raw_content':'Canada 2025: 12 cases.'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    real_post=requests.post
    def local_post(url,**kwargs):
        assert url=='https://api.tavily.com/extract'
        return real_post(f'http://127.0.0.1:{server.server_port}/extract',**kwargs)
    monkeypatch.setattr(requests,'post',local_post)
    monkeypatch.setenv('TAVILY_API_KEY','offline-fake-key')
    monkeypatch.setenv('PIPELINE_MODE',revision)
    request=ContentFetchRequest(source_id='s',url=target,canonical_url=target,
        final_screening_decision='include_for_content_fetch',fetch_purpose='data_extraction')
    try:
        result=content._tavily_extract_fetch(request,{}, {'tavily_extract_timeout_seconds':3})
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    if revision=='evidence':
        assert visits==['POST'], 'Redirect hop bypassed the per-request ledger.'
        assert result['success'] is False
        assert result['http_status_code']==302
        assert result['error']=='provider_redirect_response:302'
    else:
        assert visits==['POST','GET']
        assert result['success'] is True
