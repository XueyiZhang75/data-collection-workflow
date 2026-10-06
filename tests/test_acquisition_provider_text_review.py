"""Independent declared-text decoding and format-routing boundaries, offline."""
import hashlib
import socket
import pytest
from data_collection_workflow.document_acquisition import parse_response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    def forbidden(*args, **kwargs):
        raise AssertionError("External network is forbidden in provider-text review")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("separator", ["\x02", "\x1f"])
def test_declared_markdown_controls_keep_raw_evidence_and_separate_tokens(tmp_path, separator):
    body = ("Source content for the report. " * 40 + f"\n\nAlpha{separator}Beta observed 12{separator}18 records.").encode()
    doc = parse_response(body, url="https://example.invalid/report", source_id="s", session_dir=tmp_path,
                         content_type="text/markdown; charset=utf-8")
    assert doc["content_readable"] is True
    assert "AlphaBeta" not in doc["clean_text"] and "1218" not in doc["clean_text"]
    assert "Alpha" in doc["clean_text"] and "Beta" in doc["clean_text"]
    assert doc["content_hash"] == hashlib.sha256(body).hexdigest()
    assert (tmp_path / doc["raw_artifact_path"]).read_bytes() == body
    assert doc["text_hash"] == hashlib.sha256(doc["clean_text"].encode()).hexdigest()
    assert all(doc["clean_text"][span["char_start"]:span["char_end"]] for span in doc["locator_spans"])


def test_leading_markdown_link_is_text_even_when_original_url_is_pdf(tmp_path):
    body = b'[Skip to main content](https://example.invalid/report#main)\n\n# Generic report\nValid prose.'
    doc = parse_response(body, url="https://example.invalid/report.pdf", source_id="s", session_dir=tmp_path,
                         content_type="text/markdown; charset=utf-8")
    assert doc["content_readable"] is True
    assert doc["document_type"] == "text"
    assert "Valid prose." in doc["clean_text"]


@pytest.mark.parametrize("body,mime,expected", [
    (b"PK\x03\x04\x00\x00\x00\x00binary", "text/markdown; charset=utf-8", "unsupported_format"),
    (b'{"records": nope}', "application/json", "parse_error"),
    (b'{"records": nope}', "Application/JSON; charset=utf-8", "parse_error"),
    (b'[{"records": 12}]', "application/json", "readable"),
])
def test_real_binary_and_json_validation_keep_their_existing_boundaries(tmp_path, body, mime, expected):
    doc = parse_response(body, url="https://example.invalid/response", source_id="s", session_dir=tmp_path,
                         content_type=mime)
    assert doc["acquisition_status"] == expected
    if expected != "readable":
        assert not doc["content_readable"]
