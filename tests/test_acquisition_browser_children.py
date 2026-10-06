"""Real loopback coverage for child-process Chromium request reservations."""
import base64
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from data_collection_workflow import document_acquisition as acquisition
from data_collection_workflow.session_runtime import RunContext


@pytest.fixture
def child_site(monkeypatch):
    from playwright.sync_api import BrowserType
    launch = BrowserType.launch
    def isolated_launch(self, *args, **kwargs):
        kwargs["args"] = [*(kwargs.get("args") or []), "--site-per-process"]
        return launch(self, *args, **kwargs)
    monkeypatch.setattr(BrowserType, "launch", isolated_launch)
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.path)
            if self.path.startswith("/bounded/"):
                hop = int(self.path.rsplit("/", 1)[1])
                if hop < 6:
                    self.send_response(302)
                    self.send_header("Location", f"/bounded/{hop + 1}")
                    self.end_headers()
                    return
            if self.path == "/broken":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if self.path in {"/worker-hop", "/frame-hop"}:
                self.send_response(302)
                self.send_header("Location", self.path.replace("-hop", "-data.json"))
                self.end_headers()
                return
            bodies = {
                "/": (f'<link rel="icon" href="data:,"><div id="data">Loading</div>'
                      '<script>const w=new Worker("/worker.js");w.onmessage=e=>document.getElementById("data").textContent=JSON.stringify(e.data);</script>'
                      f'<iframe src="http://localhost:{self.server.server_port}/frame"></iframe>').encode(),
                "/bounded/6": b'<link rel="icon" href="data:,"><p>Unexpected sixth redirect</p>',
                "/redirect-script-page": b'<link rel="icon" href="data:,"><script>fetch("/bounded/0");</script>',
                "/redirect-worker-page": b'<link rel="icon" href="data:,"><script>new Worker("/redirect-worker.js");</script>',
                "/redirect-worker.js": b'fetch("/bounded/0");',
                "/redirect-frame-page": (f'<link rel="icon" href="data:,"><iframe src="http://localhost:{self.server.server_port}/redirect-frame"></iframe>').encode(),
                "/redirect-frame": b'<script>fetch("/bounded/0");</script>',
                "/worker.js": b'fetch("/worker-hop").then(r=>r.json()).then(d=>postMessage(d));',
                "/frame": b'<script>fetch("/frame-hop").then(r=>r.json()).then(d=>document.body.textContent=JSON.stringify(d));</script>',
                "/worker-data.json": b'{"disease":"Example fever","cases":731}',
                "/frame-data.json": b'{"disease":"Example fever","cases":412}',
                "/serviceworker-page": b'<link rel="icon" href="data:,"><script>navigator.serviceWorker.register("/service-worker.js");</script>',
                "/broken-page": b'<link rel="icon" href="data:,"><script>fetch("/broken");</script>',
                "/worker-websocket-page": b'<link rel="icon" href="data:,"><script>new Worker("/socket-worker.js");</script>',
                "/socket-worker.js": (f'new WebSocket("ws://127.0.0.1:{self.server.server_port}/hidden-worker-socket");').encode(),
                "/websocket-page": (f'<link rel="icon" href="data:,"><script>new WebSocket("ws://127.0.0.1:{self.server.server_port}/hidden-socket");</script>').encode(),
                "/unsupported-page": b'<link rel="icon" href="data:,"><script>try {new SharedWorker("/shared-worker.js");} catch(e) {} window.open("/popup");</script>',
                "/shared-worker.js": b'onconnect=()=>fetch("/hidden-worker-data");',
                "/popup": b'<script>fetch("/hidden-popup-data")</script>',
                "/service-worker.js": b'self.addEventListener("activate",e=>e.waitUntil(fetch("/worker-data.json")));',
            }
            body = bodies.get(self.path, b"missing")
            content_type = "application/json" if self.path.endswith(".json") else "application/javascript" if self.path.endswith(".js") else "text/html"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def context(tmp_path, limit):
    return RunContext(tmp_path, {"pipeline_mode": "evidence", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 50},
        "budget_limits": {"source_targets": 200, "http_requests": limit}}})


def test_browser_counts_out_of_process_iframe_and_worker_redirects(tmp_path, child_site):
    url, received = child_site
    run = context(tmp_path, 30)
    with run.activate():
        result = acquisition._browser(url, acquisition._paths({"render_wait_ms": 300, "timeout_ms": 5000}))
    assert "731" in base64.b64decode(result["body"]).decode()
    assert "/frame-data.json" in received
    assert run.ledger.snapshot()["used"].get("http_requests") == len(received) == 7
    assert run.ledger.snapshot()["used"].get("source_targets") == 1


@pytest.mark.parametrize("limit", [0, 3])
def test_browser_child_requests_cannot_escape_hard_limit(tmp_path, child_site, limit):
    url, received = child_site
    run = context(tmp_path, limit)
    with run.activate():
        result = acquisition._browser(url, acquisition._paths({"render_wait_ms": 300, "timeout_ms": 5000}))
    assert len(received) == limit
    assert run.ledger.snapshot()["used"].get("http_requests", 0) == limit
    assert run.ledger.snapshot()["used"].get("source_targets", 0) == int(limit > 0)
    assert result["budget_exhausted_kind"] == "http_requests"


def test_browser_blocks_service_worker_registration_before_hidden_requests(tmp_path, child_site):
    url, received = child_site
    run = context(tmp_path, 20)
    with run.activate():
        acquisition._browser(url + "serviceworker-page", acquisition._paths({"render_wait_ms": 500, "timeout_ms": 5000}))
    assert received == ["/serviceworker-page"]
    assert run.ledger.snapshot()["used"].get("http_requests") == 1


def test_budget_amendment_retries_partial_browser_navigation_instead_of_caching_it(tmp_path, child_site):
    url, received = child_site
    run = context(tmp_path, 2)
    config = {"render_wait_ms": 300, "timeout_ms": 5000}
    with run.activate():
        first = acquisition.acquire_document(url, source_id="target", session_dir=tmp_path, config=config)
        first_used = run.ledger.snapshot()["used"].copy()
        assert first.get("budget_exhausted_kind") == "http_requests"
        assert first_used["http_requests"] == len(received) == 2
        run.amend_budget(amendment_id="continue-browser", increases={"http_requests": 30}, reason="Continue remaining browser resources")
        second = acquisition.acquire_document(url, source_id="target", session_dir=tmp_path, config=config)
    assert "731" in second["clean_text"]
    assert second.get("budget_exhausted_kind") is None
    used = run.ledger.snapshot()["used"]
    assert used["source_targets"] == 1
    assert used["browser"] == 2
    assert used["http_requests"] == len(received) == 9
    navigation = [row for row in run.ledger.operation_audit() if row["kind"] in {"browser_fetch", "browser_navigation"}]
    assert [row["status"] for row in navigation] == ["failed", "completed"]


def test_uninterceptable_browser_contexts_are_blocked_and_reported(tmp_path, child_site):
    url, received = child_site
    run = context(tmp_path, 20)
    with run.activate():
        result = acquisition._browser(url + "unsupported-page", acquisition._paths({"render_wait_ms": 500, "timeout_ms": 5000}))
    assert received == ["/unsupported-page"]
    assert run.ledger.snapshot()["used"].get("http_requests") == 1
    assert any("unsupported_browser_context" in str(error) for error in result.get("request_errors", []))


@pytest.mark.parametrize("page, expected", [("websocket-page", ["/websocket-page"]), ("worker-websocket-page", ["/worker-websocket-page", "/socket-worker.js"])])
def test_websocket_handshake_cannot_bypass_the_http_cap(tmp_path, child_site, page, expected):
    url, received = child_site
    run = context(tmp_path, len(expected))
    with run.activate():
        result = acquisition._browser(url + page, acquisition._paths({"render_wait_ms": 500, "timeout_ms": 5000}))
    assert received == expected
    assert run.ledger.snapshot()["used"].get("http_requests") == len(expected)
    assert any("unsupported_browser_context" in str(error) for error in result.get("request_errors", []))


def test_failed_subrequest_is_reported_instead_of_a_complete_navigation(tmp_path, child_site):
    url, received = child_site
    run = context(tmp_path, 20)
    with run.activate():
        result = acquisition._browser(url + "broken-page", acquisition._paths({"render_wait_ms": 300, "timeout_ms": 5000}))
    assert "/broken" in received
    assert run.ledger.snapshot()["used"]["http_requests"] == len(received)
    assert any(error.get("reason") == "request_failed" and error.get("url", "").endswith("/broken")
               for error in result.get("request_errors", []))


@pytest.mark.parametrize("entry, overhead", [
    ("bounded/0", 0), ("redirect-script-page", 1),
    ("redirect-worker-page", 2), ("redirect-frame-page", 2),
])
def test_browser_stops_each_request_chain_after_five_redirects(tmp_path, child_site, entry, overhead):
    url, received = child_site
    run = context(tmp_path, 30)
    with run.activate():
        result = acquisition._browser(url + entry, acquisition._paths({"render_wait_ms": 300, "timeout_ms": 5000}))
    assert [path for path in received if path.startswith("/bounded/")] == [f"/bounded/{hop}" for hop in range(6)]
    assert run.ledger.snapshot()["used"]["http_requests"] == len(received) == overhead + 6
    assert run.ledger.snapshot()["used"]["source_targets"] == 1
    assert any(error.get("reason") == "redirect_limit_exceeded" and error.get("url", "").endswith("/bounded/6")
               for error in result.get("request_errors", []))
