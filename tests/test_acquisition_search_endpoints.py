"""Endpoint roles describe URLs, not every resource held by a publisher."""
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from data_collection_workflow import source_identity, source_credibility
from data_collection_workflow.nodes import content_processing
from importlib import import_module
source_screening = import_module("data_collection_workflow.nodes.source_screening")

POLICY = SimpleNamespace(search_endpoint_publishers=["PubMed", "Europe PMC", "OpenAlex"])

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")

@pytest.mark.parametrize("url, expected", [
    ("https://pubmed.ncbi.nlm.nih.gov/12345678/", False),
    ("https://pubmed.ncbi.nlm.nih.gov/12345678/?utm_source=search", False),
    ("https://europepmc.org/article/MED/12345678", False),
    ("https://api.openalex.org/works/W123456789?select=id,title", False),
    ("https://openalex.org/W123456789", False),
    ("https://data.example/research-results/report-2025?format=csv", False),
    ("https://reports.example/archive", False),
    ("https://pubmed.ncbi.nlm.nih.gov/?term=example", False),
    ("https://pubmed.ncbi.nlm.nih.gov/", False),
    ("https://europepmc.org/search?query=example", True),
    ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=example", True),
    ("https://api.openalex.org/works?search=example", False),
    ("https://api.openalex.org/works?filter=publication_year:2025", False),
    ("https://catalog.example/search?q=example", True),
])
def test_shared_endpoint_identity_uses_resource_shape_not_research_title(url, expected):
    entry = {"url": url, "title": "Research results from surveillance", "publisher": "Unknown"}
    assert source_identity._is_search_endpoint(entry, urlsplit(url).hostname) is expected
    assert source_credibility._is_search_endpoint_candidate(entry) is expected
    assert source_screening._is_search_endpoint_for_source_critic(entry) is expected

@pytest.mark.parametrize("field", ["source_role", "source_role_final", "role_hint"])
def test_explicit_endpoint_role_blocks_opaque_url(field):
    entry = {"url": "https://catalog.example/opaque", field: "search_endpoint"}
    assert content_processing._is_search_endpoint(entry, POLICY)

@pytest.mark.parametrize("url, expected", [
    ("https://pubmed.ncbi.nlm.nih.gov/12345678/?utm_source=reference", False),
    ("https://catalog.example/search?q=example", True),
    ("https://data.example/archive", False),
])
def test_evidence_initial_screening_agrees_with_url_identity_for_unknown_publishers(url, expected):
    from data_collection_workflow.models import SourceScreeningPolicy
    from data_collection_workflow.config import load_source_screening_policy
    policy = SourceScreeningPolicy(**load_source_screening_policy())
    entry = {"url": url, "publisher": "PubMed" if "archive" in url else "Unverified publisher",
             "title": "Research results from surveillance"}
    role, _ = source_screening._classify_source_role(entry, policy)
    assert (role == "search_endpoint") is expected


def test_legacy_publisher_policy_keeps_its_opt_in_contract(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "legacy")
    assert content_processing._is_search_endpoint(
        {"url": "https://pubmed.ncbi.nlm.nih.gov/12345678/", "publisher": "PubMed"}, POLICY)


@pytest.mark.parametrize("url", [
    "https://data.example/data/records?query=example",
    "https://data.example/data/records?q=example",
    "https://catalog.example/?q=example",
    "https://catalog.example/12345678/?query=example",
])
def test_query_parameter_without_interface_evidence_remains_acquirable(url):
    assert not content_processing._is_search_endpoint({"url": url}, POLICY)

@pytest.mark.parametrize("field, value", [
    ("page_function", "dashboard_or_database"),
    ("data_product_type", "dataset"),
    ("source_role", "data_source"),
    ("role_hint", "navigation_landing"),
])
def test_explicit_data_and_navigation_products_are_not_discarded_for_search_route(field, value):
    entry = {"url": "https://data.example/search?query=example", field: value}
    assert not content_processing._is_search_endpoint(entry, POLICY)

@pytest.mark.parametrize("query, expected", [("term=example", True), ("term=", False), ("format=json", False)])
def test_search_results_title_corroborates_nonempty_search_query(query, expected):
    entry = {"url": "https://catalog.example/?" + query, "title": "Search results"}
    assert content_processing._is_search_endpoint(entry, POLICY) is expected
