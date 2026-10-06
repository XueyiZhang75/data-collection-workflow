"""Loopback Chromium tests: redirects and script requests use the real ledger."""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.session_runtime import RunContext

@pytest.fixture
def site():
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            if self.path.startswith('/hop/'):
                step=int(self.path.rsplit('/',1)[-1]);self.send_response(302)
                self.send_header('Location',f'/hop/{step+1}' if step<2 else '/data.json')
                self.end_headers();return
            body=(b'<html><head><link rel="icon" href="data:,"></head><body><div id="data">Loading</div>'
                  b'<script>fetch("/hop/0").then(r=>r.json()).then(d=>document.getElementById("data").textContent=JSON.stringify(d));</script></body></html>')
            if self.path=='/data.json':body=b'{"disease":"dengue","cases":12}'
            self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/data.json' else 'text/html')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/',received
    finally:server.shutdown();server.server_close();thread.join()


def run_browser(tmp_path,url,limit):
    context=RunContext(tmp_path,{'pipeline_mode':'evidence','universal':{
        'budget_policy':{'version':2,'mode':'adaptive','soft_source_target':50},
        'budget_limits':{'source_targets':200,'http_requests':limit}}})
    with context.activate():
        result=acquisition._browser(url,acquisition._paths({'render_wait_ms':200,'timeout_ms':5000}))
    return result,context.ledger.snapshot()


def test_browser_counts_script_requests_and_every_redirect(tmp_path,site):
    url,received=site
    result,snapshot=run_browser(tmp_path,url,20)
    assert '12' in base64.b64decode(result['body']).decode()
    assert '/data.json' in received
    assert snapshot['used'].get('http_requests')==len(received)==5
    assert snapshot['used'].get('source_targets')==1


def test_browser_aborts_before_request_exceeds_hard_limit(tmp_path,site):
    url,received=site
    result,snapshot=run_browser(tmp_path,url,2)
    assert len(received)==2
    assert '/data.json' not in received
    assert snapshot['used'].get('http_requests')==2
    assert result['budget_exhausted_kind']=='http_requests'
