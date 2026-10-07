"""Archive-index discovery must never turn capture metadata into observations."""
import importlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from data_collection_workflow.session_runtime import RunContext


CDX = "https://web.archive.org/cdx/search/cdx"
ORIGINAL = "https://www.cdc.gov/dengue/Surveillance%2FData.html?view=annual&year=2020"
HEADER = ["timestamp", "original", "statuscode", "mimetype", "digest"]


@pytest.fixture
def module(monkeypatch):
    name = "data_collection_workflow.historical_source_discovery"
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    return importlib.import_module(name)


def test_archive_discovery_public_contract_rejects_missing_task_dates(module):
    found, manifest, summary = module.discover_historical_versions({}, [])
    assert found == [] and manifest == [] and summary["status"] == "invalid_task_dates"


def task(**overrides):
    return {"structured_task": {"disease": "dengue", "start_date": "2020-01-01",
                                "end_date": "2021-12-31", **overrides}}


def origin(url=ORIGINAL, **overrides):
    return {"source_id": "parent_search", "url": url, "canonical_url": url,
            "title": "Dengue surveillance data", "snippet": "Reported dengue cases: 12",
            "discovery_method": "live_search_result", "search_provider": "tavily",
            "source_type": "national_public_health_agency", **overrides}


def record(timestamp="20200102030405", original=ORIGINAL, **overrides):
    values = dict(timestamp=timestamp, original=original, statuscode="200",
                  mimetype="text/html", digest="CDX-SHA1-BASE32")
    values.update(overrides)
    return [values[key] for key in HEADER]


class Response:
    def __init__(self, rows=None, *, body=None, status=200, headers=None, chunks=None):
        self.body = body if body is not None else json.dumps(rows if rows is not None else [HEADER]).encode()
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks
        self.closed = False
        self.bytes_yielded = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        chunks = self.chunks if self.chunks is not None else [self.body[i:i+chunk_size]
                    for i in range(0, len(self.body), chunk_size)]
        for chunk in chunks:
            self.bytes_yielded += len(chunk)
            yield chunk


def transport(monkeypatch, module, *responses):
    calls = []
    pending = iter(responses)

    def get(url, **kwargs):
        calls.append((url, kwargs))
        response = next(pending)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(module.requests, "get", get)
    return calls


def runtime(tmp_path, *, search=8, results=64):
    return RunContext(tmp_path, {"pipeline_mode": "evidence",
                      "universal": {"budget_limits": {"search": search,
                                                       "search_results": results}}})


def test_exact_lookup_preserves_query_and_capture_is_only_an_index_lead(module, monkeypatch, tmp_path):
    response = Response([HEADER, record()])
    calls = transport(monkeypatch, module, response)
    parent = origin(published_date="2025-01-01", publisher="CDC", reporting_period_start="2020-01-01",
                    reporting_period_end="2020-12-31", expected_fields=["case_count"], must_fetch=True,
                    target_verification_status="verified", credibility_tier="tier_1")
    ctx = runtime(tmp_path)
    with ctx.activate():
        found, manifest, summary = module.discover_historical_versions(task(), [parent], timeout_seconds=7)
    assert len(calls) == 1 and calls[0][0] == CDX
    request = calls[0][1]
    params = dict(request["params"])
    assert params["url"] == ORIGINAL
    assert params["matchType"] == "exact" and params["output"] == "json"
    assert params["fl"] == "timestamp,original,statuscode,mimetype,digest"
    assert str(params["from"]) == "20200101" and str(params["to"]) == "20211231"
    assert str(params["limit"]) == "65"
    assert {value for key, value in request["params"] if key == "filter"} == {"statuscode:200", "mimetype:text/html"}
    assert request["stream"] is True and request["allow_redirects"] is False
    assert request["timeout"] == 7 and response.closed
    replay = "https://web.archive.org/web/20200102030405/" + ORIGINAL
    assert len(found) == 1 and found[0]["url"] == replay
    candidate = found[0]
    assert candidate["canonical_url"] == replay and candidate["domain"] == "web.archive.org"
    assert candidate["source_type"] == "archived_web_page"
    assert candidate["source_purpose"] == "historical_version_lead"
    assert candidate["discovery_method"] == "historical_archive_index"
    assert candidate["published_date"] is None and not candidate.get("snippet")
    assert candidate["blocked_from_fetch"] is True
    assert candidate["blocked_from_fetch_reason"] == "historical_index_only"
    for key in ("reporting_period_start", "reporting_period_end", "case_count", "publisher",
                "credibility_tier", "must_fetch", "target_verification_status"):
        assert not candidate.get(key), key
    assert not candidate.get("expected_fields")
    assert candidate["title"].startswith("Archived version candidate: ")
    assert "index-only" in candidate["notes"] and "not publication" in candidate["notes"]
    assert "body not retrieved" in candidate["notes"]
    snapshot = candidate["historical_snapshot"]
    assert snapshot["parent_source_id"] == "parent_search" and snapshot["original_url"] == ORIGINAL
    assert snapshot["capture_timestamp"] == "20200102030405"
    assert snapshot["captured_at"] == "2020-01-02T03:04:05+00:00"
    assert snapshot["archive_url"] == replay and snapshot["verification_status"] == "index_only"
    assert snapshot["index_retrieved_at"] and snapshot["archive_provider"] == "internet_archive_wayback"
    assert snapshot["index_record"]["cdx_digest"] == "CDX-SHA1-BASE32"
    assert "raw_sha256" not in snapshot["index_record"]
    assert parse_qs(urlsplit(snapshot["index_url"]).query)["url"] == [ORIGINAL]
    assert all(row["index_only"] is True and row["body_retrieved"] is False for row in manifest)
    assert summary["status"] == "completed" and summary["admitted_result_count"] == 1
    assert ctx.ledger.snapshot()["used"] == {"search": 1, "search_results": 1}


def test_two_same_digest_captures_survive_exact_replay_duplicates(module, monkeypatch, tmp_path):
    rows = [HEADER, record("20210601000000"), record(), record(), record("20200102030405", digest="OTHER")]
    transport(monkeypatch, module, Response(rows))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin(), origin(source_id="duplicate")])
    assert [item["historical_snapshot"]["capture_timestamp"] for item in found] == ["20200102030405", "20210601000000"]
    assert len({item["source_id"] for item in found}) == 2
    assert summary["index_query_count"] == 1 and summary["duplicate_record_count"] == 2


def test_header_driven_rows_reject_other_originals_dates_statuses_and_formats(module, monkeypatch, tmp_path):
    reordered = ["digest", "mimetype", "original", "timestamp", "statuscode"]
    rows = [record(), record("20191231235959"), record("20220101000000"), record("20200230000000"),
            record("20200101000060"), record("20200101"), record(original=ORIGINAL.lower()),
            record(original=ORIGINAL.replace("%2F", "/")), record(original=ORIGINAL + "&extra=1"),
            record(original=ORIGINAL.replace("year=2020", "year=2021")), record(statuscode="302"),
            record(mimetype="application/pdf"), record(timestamp=20200102030405)]
    payload = [reordered] + [[row[HEADER.index(key)] for key in reordered] for row in rows]
    payload.extend([["short"], {"timestamp": "20200102030405"}])
    transport(monkeypatch, module, Response(payload))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert len(found) == 1
    assert summary["rejected_record_count"] == 14
    assert summary["origins"][0]["status"] == "completed"


def test_only_scheme_and_host_case_normalize_for_original_matching(module, monkeypatch, tmp_path):
    changed = ORIGINAL.replace("https://www.cdc.gov", "HTTPS://WWW.CDC.GOV")
    transport(monkeypatch, module, Response([HEADER, record(original=changed),
        record(original=ORIGINAL.replace("https:", "http:"))]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert len(found) == 1 and found[0]["historical_snapshot"]["original_url"] == changed
    assert summary["rejected_record_count"] == 1


@pytest.mark.parametrize("updates", [
    {"start_date": "2020"}, {"start_date": "2020-2-01"}, {"start_date": "2020-02-30"},
    {"end_date": None}, {"end_date": "2019-12-31"}, {"start_date": "2020-01-01T00:00:00"},
])
def test_invalid_full_task_dates_never_dispatch(module, monkeypatch, tmp_path, updates):
    calls = transport(monkeypatch, module)
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(**updates), [origin()])
    assert found == [] and calls == [] and summary["status"] == "invalid_task_dates"


def test_admission_uses_actual_domain_and_own_disease_evidence(module, monkeypatch, tmp_path):
    candidates = [origin("https://reuters.com/dengue/data", domain="cdc.gov"),
                  origin("https://www.cdc.gov/measles/data", title="Measles data", snippet="Measles cases",
                         query_used="dengue surveillance", matched_terms=["dengue"]),
                  origin(discovery_method="seed_catalog", seed_source_id="seed_dengue"),
                  origin(discovery_method="fixture_search_result", search_provider="fixture"),
                  origin("https://www.cdc.gov/dengue/report.pdf"),
                  origin("https://www.cdc.gov/export?format=csv", title="Dengue data"),
                  origin("https://web.archive.org/web/20200101/https://www.cdc.gov/dengue/"),
                  origin("ftp://www.cdc.gov/dengue"), origin("https://user:pass@www.cdc.gov/dengue"),
                  origin("http://localhost/dengue"), origin("http://127.0.0.1/dengue"),
                  origin("http://10.0.0.1/dengue"), origin("http://[::1]/dengue"),
                  origin("https://www.cdc.gov.private/dengue"), origin("https://local/dengue"),
                  origin("https://www.cdc.gov\\@127.0.0.1/dengue")]
    calls = transport(monkeypatch, module)
    with runtime(tmp_path).activate():
        found, manifest, summary = module.discover_historical_versions(task(), candidates)
    assert not found and not calls
    assert summary["status"] == "no_eligible_origins"
    assert summary["skipped_origin_count"] == len(candidates)
    assert all(row["result_status"] == "skipped" for row in manifest)


def test_monitoring_preferred_and_task_alias_supported_with_independent_identity(module, monkeypatch, tmp_path):
    general = origin("https://www.cdc.gov/pertussis/about", title="Pertussis overview", snippet="")
    monitor_url = "https://www.cdc.gov/whooping-cough/surveillance"
    monitor = origin(monitor_url, title="Whooping cough case monitoring", snippet="", source_type="news_media")
    calls = transport(monkeypatch, module, Response([HEADER, record(original=monitor_url)]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(disease="pertussis"), [general, monitor], max_origins=1)
    assert dict(calls[0][1]["params"])["url"] == monitor_url
    assert len(found) == 1 and summary["eligible_origin_count"] == 2
    assert summary["status"] == "partial" and summary["truncated"] is True


def test_result_and_origin_limits_stop_unneeded_requests(module, monkeypatch, tmp_path):
    second = origin("https://www.cdc.gov/dengue/statistics", source_id="second")
    calls = transport(monkeypatch, module, Response([HEADER, record(), record("20210101000000")]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin(), second], max_results=1)
    assert len(found) == 1 and len(calls) == 1
    assert summary["status"] == "partial" and summary["truncated"] is True
    assert summary["origins"][0]["status"] == "partial"
    assert summary["origins"][1]["status"] == "result_limit"


def test_record_sentinel_is_reported_as_partial_not_complete(module, monkeypatch, tmp_path):
    calls = transport(monkeypatch, module, Response([HEADER, record(), record("20210101000000"), record("20210601000000")]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()], max_records_per_origin=2)
    assert dict(calls[0][1]["params"])["limit"] == "3"
    assert len(found) == 2 and summary["status"] == "partial"
    assert summary["partial_origin_count"] == 1 and summary["truncated"] is True
    assert summary["origins"][0]["record_limit_reached"] is True


@pytest.mark.parametrize("kind", ["search", "search_results"])
def test_exhausted_shared_budget_stops_before_http(module, monkeypatch, tmp_path, kind):
    calls = transport(monkeypatch, module)
    ctx = runtime(tmp_path, search=0 if kind == "search" else 4, results=0 if kind == "search_results" else 64)
    with ctx.activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and not calls
    assert summary["status"] == "budget_exhausted" and summary["budget_deferred_origin_count"] == 1
    assert ctx.ledger.snapshot()["used"] == {}


def test_result_budget_accepts_only_available_unique_snapshots(module, monkeypatch, tmp_path):
    transport(monkeypatch, module, Response([HEADER, record(), record("20210101000000")]))
    ctx = runtime(tmp_path, results=1)
    with ctx.activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert len(found) == 1 and summary["status"] == "partial"
    assert summary["budget_exhausted"] is True
    assert summary["origins"][0]["status"] == "budget_exhausted"
    assert ctx.ledger.snapshot()["used"] == {"search": 1, "search_results": 1}


def test_cached_index_and_admissions_replay_at_exhausted_budget(module, monkeypatch, tmp_path):
    calls = transport(monkeypatch, module, Response([HEADER, record()]))
    ctx = runtime(tmp_path, search=1, results=1)
    with ctx.activate():
        first, _, one = module.discover_historical_versions(task(), [origin()])
        second, _, two = module.discover_historical_versions(task(), [origin()])
    assert first == second and len(calls) == 1
    assert one["http_dispatch_count"] == 1 and two["http_dispatch_count"] == 0
    assert two["cache_hit_count"] == 1 and two["status"] == "completed"
    assert ctx.ledger.snapshot()["used"] == {"search": 1, "search_results": 1}


@pytest.mark.parametrize("response,reason", [
    (Response(status=302, headers={"Location": "http://127.0.0.1/private"}), "redirect_refused"),
    (Response(status=503), "http_error"),
    (Response(body=b"not json"), "invalid_index_response"),
    (Response([["timestamp", "original"], ["20200102030405", ORIGINAL]]), "invalid_index_response"),
    (requests.Timeout("timed out"), "timeout"),
])
def test_index_errors_are_nonfatal_audited_and_never_fetch_replay(module, monkeypatch, tmp_path, response, reason):
    calls = transport(monkeypatch, module, response)
    ctx = runtime(tmp_path)
    with ctx.activate():
        found, manifest, summary = module.discover_historical_versions(task(), [origin()])
    assert found == [] and summary["status"] == "error"
    assert summary["error_origin_count"] == 1 and summary["origins"][0]["reason"] == reason
    assert len(calls) == 1 and calls[0][1]["allow_redirects"] is False
    assert any(row.get("result_status") == "error" for row in manifest)
    assert ctx.ledger.snapshot()["used"] == {"search": 1}


def test_stream_byte_cap_aborts_and_closes_response(module, monkeypatch, tmp_path):
    response = Response(chunks=[b" " * 65536] * 40)
    transport(monkeypatch, module, response)
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and summary["origins"][0]["reason"] == "response_byte_limit"
    assert summary["status"] == "partial" and summary["truncated"] is True
    assert response.bytes_yielded == 2 * 1024 * 1024 + 65536 and response.closed


def test_content_length_limit_aborts_without_reading_body(module, monkeypatch, tmp_path):
    response = Response(headers={"Content-Length": str(2 * 1024 * 1024 + 1)})
    transport(monkeypatch, module, response)
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and summary["origins"][0]["reason"] == "response_byte_limit"
    assert response.bytes_yielded == 0 and response.closed


def test_empty_valid_index_is_distinct_from_errors(module, monkeypatch, tmp_path):
    transport(monkeypatch, module, Response([]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and summary["status"] == "no_captures"
    assert summary["no_capture_origin_count"] == 1 and summary["error_origin_count"] == 0


def test_missing_runtime_never_dispatches(module, monkeypatch):
    calls = transport(monkeypatch, module)
    found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and not calls and summary["status"] == "runtime_unavailable"


def test_in_doubt_result_admission_is_audited_without_escaping(module, monkeypatch, tmp_path):
    calls = transport(monkeypatch, module, Response([HEADER, record()]))
    ctx = runtime(tmp_path)
    replay = "https://web.archive.org/web/20200102030405/" + ORIGINAL
    ctx.ledger.begin("search_results", {"fingerprint": ctx.fingerprint, "input": {"canonical_url": replay}})
    with ctx.activate():
        found, manifest, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and len(calls) == 1
    assert summary["status"] == "error" and summary["error_origin_count"] == 1
    assert summary["origins"][0]["reason"] == "result_admission_error"
    assert any(row.get("error_type") == "ResumeMismatch" for row in manifest)


def test_rejected_index_rows_are_distinct_from_a_genuinely_empty_index(module, monkeypatch, tmp_path):
    transport(monkeypatch, module, Response([HEADER, record("20100101000000")]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()])
    assert not found and summary["status"] == "no_valid_captures"
    assert summary["no_capture_origin_count"] == 0
    assert summary["origins"][0]["status"] == "no_valid_captures"
    assert summary["rejected_record_count"] == 1


def test_one_origin_error_does_not_discard_another_origins_capture(module, monkeypatch, tmp_path):
    second = "https://www.ecdc.europa.eu/en/dengue/surveillance"
    calls = transport(monkeypatch, module, requests.Timeout("timed out"),
                      Response([HEADER, record(original=second)]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin(), origin(second)])
    assert len(found) == 1 and found[0]["historical_snapshot"]["original_url"] == second
    assert len(calls) == 2 and summary["status"] == "partial"
    assert summary["error_origin_count"] == 1 and summary["admitted_result_count"] == 1


@pytest.mark.parametrize("url,disease,title", [
    ("https://www.alberta.ca/hantavirus", "hantavirus", "Hantavirus monitoring reports"),
    ("https://health.example/dengue/annual-report", "dengue", "Dengue annual surveillance report"),
])
def test_registry_unknown_monitoring_page_is_an_untrusted_index_lead(module, monkeypatch, tmp_path,
                                                                   url, disease, title):
    # The real bundled registry is intentionally used: fallback must not depend
    # on adding each jurisdiction or accepting the search query's desired type.
    assert module.lookup_source_identity_registry(urlsplit(url).hostname) is None
    parent = origin(url, title=title, snippet="", domain="cdc.gov", publisher="Unverified publisher",
                    source_type="national_public_health_agency", credibility_tier="tier_1_official")
    calls = transport(monkeypatch, module, Response([HEADER, record(original=url)]))
    with runtime(tmp_path).activate():
        found, manifest, summary = module.discover_historical_versions(task(disease=disease), [parent])
    assert len(calls) == 1 and dict(calls[0][1]["params"])["url"] == url
    assert len(found) == 1 and found[0]["source_type"] == "archived_web_page"
    assert found[0]["blocked_from_fetch"] is True
    assert found[0]["historical_snapshot"]["verification_status"] == "index_only"
    assert not found[0].get("publisher") and not found[0].get("credibility_tier")
    assert not found[0].get("original_registry_source_type")
    audit = summary["origins"][0]
    assert audit["original_identity_status"] == "unknown"
    assert audit["selection_reason"] == "unregistered_monitoring_page"
    assert audit["metadata_only_identity"] is True
    assert any(row.get("selection_reason") == "unregistered_monitoring_page" for row in manifest)


def test_registered_official_origins_precede_unknown_monitoring_fallback(module, monkeypatch, tmp_path):
    unknown = "https://health.example/dengue/data"
    calls = transport(monkeypatch, module, Response([]), Response([HEADER, record(original=unknown)]))
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin(unknown), origin()], max_origins=2)
    assert [dict(kwargs["params"])["url"] for _, kwargs in calls] == [ORIGINAL, unknown]
    assert len(found) == 1 and found[0]["historical_snapshot"]["original_url"] == unknown
    assert summary["origins"][0]["original_identity_status"] == "registered_official"
    assert summary["origins"][1]["original_identity_status"] == "unknown"


def test_unknown_faq_does_not_inherit_monitoring_or_official_type_from_query(module, monkeypatch, tmp_path):
    calls = transport(monkeypatch, module)
    parent = origin("https://health.example/dengue/faq", title="Dengue frequently asked questions",
                    snippet="What symptoms occur in dengue cases?", query_used="Dengue surveillance report",
                    source_type="national_public_health_agency")
    with runtime(tmp_path).activate():
        found, manifest, summary = module.discover_historical_versions(task(), [parent])
    assert not found and not calls and summary["status"] == "no_eligible_origins"
    assert manifest[0]["rejection_reason"] == "unknown_identity_without_monitoring_signal"


def test_quarterly_lookup_uses_exact_dates_as_retrieval_hints_only(module, monkeypatch, tmp_path):
    rows = [HEADER, record("20200102030405"), record("20200701000000"),
            record("20200930235959"), record("20201001000000")]
    calls = transport(monkeypatch, module, Response(rows), Response(rows))
    with runtime(tmp_path).activate():
        first, _, first_summary = module.discover_historical_versions(
            task(start_date="2020-01-01", end_date="2020-03-31"), [origin()])
        third, _, third_summary = module.discover_historical_versions(
            task(start_date="2020-07-01", end_date="2020-09-30"), [origin()])
    params = [dict(kwargs["params"]) for _, kwargs in calls]
    assert [(p["from"], p["to"]) for p in params] == [("20200101", "20200331"), ("20200701", "20200930")]
    assert [c["historical_snapshot"]["capture_timestamp"] for c in first] == ["20200102030405"]
    assert [c["historical_snapshot"]["capture_timestamp"] for c in third] == ["20200701000000", "20200930235959"]
    assert first_summary["capture_lookup_from"] == "20200101" and first_summary["capture_lookup_to"] == "20200331"
    assert third_summary["capture_lookup_from"] == "20200701" and third_summary["capture_lookup_to"] == "20200930"
    assert third_summary["capture_lookup_basis"] == "task_dates_retrieval_hint_only"
    assert third_summary["rejected_record_count"] == 2
    for candidate in first + third:
        assert candidate["published_date"] is None
        assert not candidate.get("reporting_period_start") and not candidate.get("reporting_period_end")
        assert candidate["historical_snapshot"]["verification_status"] == "index_only"


def test_slow_stream_deadline_is_checked_before_a_large_chunk_can_fill(module, monkeypatch, tmp_path):
    elapsed = {"seconds": 0.0}

    class DrippingResponse(Response):
        def iter_content(self, chunk_size):
            buffered = bytearray()
            for byte in b" " * 1000:
                elapsed["seconds"] += 0.04
                buffered.append(byte)
                if len(buffered) >= chunk_size:
                    self.bytes_yielded += len(buffered)
                    yield bytes(buffered)
                    buffered.clear()
            if buffered:
                self.bytes_yielded += len(buffered)
                yield bytes(buffered)

    response = DrippingResponse()
    transport(monkeypatch, module, response)
    monkeypatch.setattr(module.time, "monotonic", lambda: elapsed["seconds"])
    with runtime(tmp_path).activate():
        found, _, summary = module.discover_historical_versions(task(), [origin()], timeout_seconds=0.1)
    assert not found and summary["origins"][0]["reason"] == "timeout"
    assert response.bytes_yielded == 3 and elapsed["seconds"] == pytest.approx(0.12)
    assert response.closed
