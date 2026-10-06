"""Provider-extracted text must remain text without accepting binary artifacts."""
import hashlib
import json

import pytest
import requests

from data_collection_workflow import document_acquisition as acquisition
from test_acquisition_transport import _adaptive_runtime


def _parse(tmp_path, body, content_type):
    return acquisition.parse_response(body, url="https://publisher.example/report",
        source_id="source", session_dir=tmp_path, content_type=content_type)


@pytest.mark.parametrize("control", ["\x02", "\x1f"])
@pytest.mark.parametrize("mime", ["text/markdown; charset=utf-8", "text/plain; charset=utf-8"])
def test_sparse_provider_controls_preserve_original_text_bytes_and_locator(tmp_path, control, mime):
    text = ("Example surveillance report. " * 15 +
            f"\nSeparated tokens: 12{control}34.\nDuring 2025, Canada reported 17 measles cases.")
    body = text.encode()
    doc = _parse(tmp_path, body, mime)
    assert doc["content_readable"] and doc["parse_eligible"]
    assert doc["document_type"] == "text"
    assert doc["clean_text"] == text
    assert doc["content_hash"] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / doc["raw_artifact_path"]).read_bytes() == body
    span = doc["locator_spans"][0]
    assert doc["clean_text"][span["char_start"]:span["char_end"]] == text
    assert "1234" not in doc["clean_text"]


@pytest.mark.parametrize("mime", ["text/markdown", "text/plain", "application/octet-stream"])
def test_bracket_led_markdown_is_not_a_json_parse_error(tmp_path, mime):
    text = '[Skip to Main Content](https://publisher.example/report#main "Skip")\n\n## Report\n17 measles cases.'
    doc = _parse(tmp_path, text.encode(), mime)
    assert doc["content_readable"]
    assert doc["document_type"] == "text"
    assert doc["clean_text"] == text
    assert not doc.get("parse_error")


@pytest.mark.parametrize("mime", ["application/json", "text/plain", "application/octet-stream"])
def test_actual_json_is_still_parsed(tmp_path, mime):
    body = b'[{"metric":"cases","count":17}]'
    doc = _parse(tmp_path, body, mime)
    assert doc["content_readable"] and doc["document_type"] == "json"
    assert json.loads(doc["clean_text"]) == json.loads(body)


@pytest.mark.parametrize(("body", "mime"), [
    (b"\x00" + b"readable looking words " * 30, "text/plain"),
    (b"\x02\x1f" * 30 + b"report", "text/markdown"),
    (b"report \xff\xfe invalid", "text/plain; charset=utf-8"),
    (b"word " * 100 + b"\x02", "application/octet-stream"),
])
def test_binary_invalid_and_dense_control_responses_stay_unreadable(tmp_path, body, mime):
    doc = _parse(tmp_path, body, mime)
    assert not doc["content_readable"] and not doc["parse_eligible"]
    assert doc["acquisition_status"] == "unsupported_format"
    assert (tmp_path / doc["raw_artifact_path"]).read_bytes() == body


def test_declared_malformed_json_still_fails_parsing(tmp_path):
    doc = _parse(tmp_path, b'[not valid JSON]', "application/json")
    assert doc["acquisition_status"] == "parse_error"
    assert not doc["content_readable"]
    assert "JSONDecodeError" in doc["parse_error"]


def test_real_provider_adapter_accepts_markdown_with_sparse_control(tmp_path, monkeypatch):
    from test_evidence_resource_discovery import _env, _source
    from data_collection_workflow.nodes import content_processing as cp
    from data_collection_workflow.models import ContentFetchRequest
    _env(monkeypatch)
    url = "https://publisher.example/report"
    class Response:
        status_code = 403
        headers = {"content-type": "text/html"}
        def __init__(self):
            self.url = url
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter_content(self, size):
            yield b"<h1>Access denied</h1>"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    def blocked(*args):
        raise RuntimeError("Synthetic unavailable browser")
    monkeypatch.setattr(acquisition, "_browser", blocked)
    text = ('[Skip to Content](https://publisher.example/report#main)\n' +
            'Generic surveillance report. ' * 15 +
            '\nAuthor\x1fName\nDuring 2025, Canada reported 17 measles cases.')
    calls = []
    def provider(*args):
        calls.append(True)
        return {"success": True, "body": text, "content_type": "text/markdown; charset=utf-8",
                "http_status_code": 200, "metadata": {"source_url": url}}
    monkeypatch.setattr(cp, "_tavily_extract_fetch", provider)
    runtime = _adaptive_runtime(tmp_path, targets=2, requests=5)
    entry = _source(url)
    request = ContentFetchRequest(source_id=entry["source_id"], url=url, canonical_url=url,
        final_screening_decision="include_for_content_fetch", fetch_purpose="data_extraction")
    with runtime.activate():
        doc = cp._fetch_live_document_with_providers(request, entry, None, {
            "external_fetch_enabled": True,
            "external_fetch_provider_order": ["native_requests", "tavily_extract"]})
    assert calls == [True]
    assert doc.content_readable and doc.parse_eligible
    assert doc.fetch_provider == "tavily_extract"
    assert doc.clean_text == text
    assert doc.content_hash == hashlib.sha256(text.encode()).hexdigest()
    assert (runtime.session_dir / doc.raw_artifact_path).read_bytes() == text.encode()
    assert doc.response_artifact_path != doc.raw_artifact_path
    assert runtime.ledger.snapshot()["used"]["http_requests"] == 2
