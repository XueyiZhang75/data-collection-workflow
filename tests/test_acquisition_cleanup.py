"""Offline regressions for residual acquisition routing replaced by evidence."""
import socket

import pytest

from data_collection_workflow.resource_discovery import task_resource_candidates
from data_collection_workflow.source_product_profile import profile_source_product


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("ENABLE_LIVE_SEARCH", "false")
    monkeypatch.setenv("ENABLE_LIVE_FETCH", "false")
    def forbidden(*args, **kwargs):
        raise AssertionError("External network is forbidden in cleanup regressions")
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def task(disease="Example fever"):
    return {"structured_task": {"disease": disease, "location": "Example Region",
            "start_date": "2025-01-01", "end_date": "2025-12-31"}}


def linked_document(link, disease="Example fever"):
    return {"source_id": "parent", "url": "https://health.example/report",
            "content_readable": True,
            "clean_text": f"{disease} surveillance in Example Region during 2025.",
            "metadata": {"outbound_links": [link]}}


@pytest.mark.parametrize("disease", ["Example fever", "Measles"])
@pytest.mark.parametrize("label", ["dashboard", "case counts dashboard", "report archive"])
def test_task_navigation_products_remain_discoverable(disease, label):
    link = {"href": "https://health.example/products/current",
            "text": f"{disease} {label}", "navigation": True}
    rows = task_resource_candidates(linked_document(link, disease), {}, task(disease))
    assert [row["url"] for row in rows] == [link["href"]]
    assert rows[0]["resource_type"] == "html"
    assert "link_task_match" in rows[0]["selection_reasons"]


def test_navigation_data_availability_link_keeps_explicit_parent_relationship():
    link = {"href": "https://repository.example/record/123", "text": "Underlying data",
            "heading": "Data availability", "navigation": True,
            "source_url": "https://health.example/report", "source_content_hash": "parent-hash"}
    rows = task_resource_candidates(linked_document(link), {}, task())
    assert [row["url"] for row in rows] == [link["href"]]
    assert "scoped_data_availability_link" in rows[0]["selection_reasons"]
    assert rows[0]["provenance"] == link


@pytest.mark.parametrize("label,path", [
    ("Privacy policy", "privacy"),
    ("Reports", "reports"),
    ("Data", "data"),
    ("Other fever dashboard", "other/dashboard"),
    ("Example fever privacy", "privacy"),
])
def test_generic_or_utility_navigation_does_not_inherit_parent_disease(label, path):
    link = {"href": f"https://health.example/{path}", "text": label, "navigation": True}
    assert task_resource_candidates(linked_document(link), {}, task()) == []


@pytest.mark.parametrize("domain", ["nejm.org", "eurosurveillance.org", "science.org", "journal.example"])
def test_journal_domain_and_doi_do_not_assert_individual_case_evidence(domain):
    entry = {"domain": domain, "url": f"https://{domain}/doi/10.9999/example",
             "title": "Genomic analysis of Example fever in Example Region during 2025",
             "source_type": "academic_or_peer_reviewed_source"}
    result = profile_source_product(entry, task())
    assert result["data_product_type"] == "research_article"
    assert result["expected_evidence_role"] == "potential_public_health_evidence"


def test_legacy_journal_profile_contract_remains_unchanged(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "legacy")
    entry = {"domain": "nejm.org", "url": "https://nejm.org/doi/10.9999/example",
             "title": "Genomic analysis of Example fever in Example Region during 2025",
             "source_type": "academic_or_peer_reviewed_source"}
    assert profile_source_product(entry, task())["data_product_type"] == "case_report_article"


def test_navigation_cannot_borrow_task_from_surrounding_menu_context():
    link = {"href": "https://health.example/reports", "text": "Reports", "navigation": True,
            "context": "Example fever surveillance reports in Example Region during 2025"}
    assert task_resource_candidates(linked_document(link), {}, task()) == []


def test_discovery_uses_shared_budget_revision_contract(monkeypatch):
    from importlib import import_module
    from types import SimpleNamespace
    discovery = import_module("data_collection_workflow.nodes.source_discovery")
    runtime = SimpleNamespace(config={"pipeline_mode": "legacy", "universal": {
        "budget_policy": {"version": 2, "mode": "adaptive", "soft_source_target": 50}}},
        ledger=SimpleNamespace(limits={"search": 40, "search_results": 400}))
    monkeypatch.setattr("data_collection_workflow.session_runtime.get_runtime", lambda: runtime)
    settings = discovery.SourceSearchSettings(max_queries=2, max_total_results=5)
    assert discovery._effective_discovery_settings(settings) == settings
