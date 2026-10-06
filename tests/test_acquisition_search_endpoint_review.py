"""Independent URL semantics and shared queue/recovery endpoint controls."""
from types import SimpleNamespace

import pytest

from data_collection_workflow.nodes import content_processing as content

POLICY = SimpleNamespace(search_endpoint_publishers=["PubMed", "Europe PMC", "OpenAlex"])


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


@pytest.mark.parametrize("url", [
    "https://pubmed.ncbi.nlm.nih.gov/12345678/?utm_source=reference",
    "https://europepmc.org/article/MED/12345678?pdf=render",
    "https://api.openalex.org/works/W123456789?select=id,title",
    "https://data.example/reports/2025?format=json",
    "https://journal.example/research/report?display=full",
])
def test_record_and_data_landing_identity_does_not_depend_on_publisher_label(url):
    for publisher in ("PubMed", "Europe PMC", "OpenAlex", "Unverified publisher", None):
        entry = {"url": url, "canonical_url": url, "publisher": publisher,
                 "title": "Research results from surveillance",
                 "source_role_final": "collection", "ready_for_content_fetch": True}
        assert not content._is_search_endpoint(entry, POLICY), (url, publisher)


@pytest.mark.parametrize("url", [
    "https://pubmed.ncbi.nlm.nih.gov/?term=example",
    "https://europepmc.org/search?query=example",
    "https://api.openalex.org/works?search=example",
])
def test_actual_search_index_is_not_disguised_by_publisher_label(url):
    for publisher in ("PubMed", "Europe PMC", "OpenAlex", "Unverified publisher", None):
        assert content._is_search_endpoint(
            {"url": url, "canonical_url": url, "publisher": publisher,
             "title": "Search results"}, POLICY), (url, publisher)


@pytest.mark.parametrize("article", [True, False])
def test_queue_and_recovery_share_the_article_versus_index_decision(tmp_path, monkeypatch, article):
    from test_acquisition_queue import setup_node
    from data_collection_workflow.workflow_recovery import _recovery_fetch_skip_reason
    runtime, state, visits = setup_node(monkeypatch, tmp_path)
    source = state["source_registry"][0]
    url = ("https://pubmed.ncbi.nlm.nih.gov/12345678/?utm_source=reference" if article else
           "https://pubmed.ncbi.nlm.nih.gov/?term=example")
    source.update(url=url, canonical_url=url, publisher="PubMed")
    if not article:
        source["source_role"] = "search_endpoint"
    state["source_registry"] = [source]
    with runtime.activate():
        reason = _recovery_fetch_skip_reason(source, state)
        result = content.content_fetch_and_parse(state)
    assert reason == (None if article else "search_endpoint")
    assert visits == ([url] if article else [])
    if article:
        assert len(result["documents"]) == 1
        assert result["documents"][0]["content_readable"]
        assert runtime.frontier.snapshot()["counts"] == {"completed": 1}
    else:
        assert not result.get("documents")
        assert runtime.ledger.snapshot()["used"].get("http_requests", 0) == 0


@pytest.mark.parametrize(("path", "expected"), [
    ("/", False),
    ("/search?query=example", True),
    ("/works?search=example", True),
    ("/data/cases.csv?filter=region:example", False),
])
def test_renamed_host_cannot_change_url_search_or_data_semantics(path, expected):
    for host in ("pubmed.ncbi.nlm.nih.gov", "openalex.org", "catalog.example", "renamed.example"):
        url = "https://" + host + path
        entry = {"url": url, "canonical_url": url, "publisher": "Unverified publisher"}
        if path == "/works?search=example":
            entry["title"] = "Search results"
        assert content._is_search_endpoint(entry, POLICY) is expected, url


@pytest.mark.parametrize("path", [
    "/data/records?query=region:example",
    "/articles/12345678?query=example",
    "/?q=region:example",
    "/works?search=example",
])
def test_query_parameters_alone_do_not_exclude_unknown_or_data_resources(path):
    for host in ("pubmed.ncbi.nlm.nih.gov", "openalex.org", "data.example"):
        url = "https://" + host + path
        entry = {"url": url, "canonical_url": url, "publisher": "PubMed",
                 "title": "Surveillance data and research results"}
        assert not content._is_search_endpoint(entry, POLICY), url


@pytest.mark.parametrize("route", ["credibility", "critic"])
def test_evidence_data_role_priority_is_shared_before_legacy_endpoint_guards(route):
    from importlib import import_module
    from data_collection_workflow import source_credibility
    entry = {"url": "https://catalog.example/search?query=example",
             "source_role": "search_endpoint", "data_product_type": "dataset"}
    assert not content._is_search_endpoint(entry, POLICY)
    predicate = (source_credibility._is_search_endpoint_candidate if route == "credibility" else
                 import_module("data_collection_workflow.nodes.source_screening")._is_search_endpoint_for_source_critic)
    assert not predicate(entry)


def test_explicit_opaque_placeholder_role_keeps_existing_skip_semantics():
    from importlib import import_module
    from data_collection_workflow import source_credibility
    entry = {"url": "https://catalog.example/opaque", "source_role": "placeholder_source"}
    assert content._is_search_endpoint(entry, POLICY)
    assert source_credibility._is_search_endpoint_candidate(entry)
    assert import_module("data_collection_workflow.nodes.source_screening")._is_search_endpoint_for_source_critic(entry)
