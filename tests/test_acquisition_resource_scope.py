"""Actual HTML/link and queue regressions for source-local resource admission."""
import socket
from html import escape

import pytest

from data_collection_workflow.resource_discovery import task_resource_candidates, resource_source_entry
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.acquisition_scheduling import acquisition_priority

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    def denied(*a, **kw): raise AssertionError("external network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)


def candidates(tmp_path, html, disease="mpox", location="Sierra Leone", host="journal.example"):
    doc = parse_response(html.encode(), url="https://"+host+"/issue", source_id="parent",
                         session_dir=tmp_path, content_type="text/html")
    state = {"structured_task": {"disease": disease, "location": location,
             "start_date": "2025-01-01", "end_date": "2025-12-31"}}
    return task_resource_candidates(doc, {}, state)


@pytest.mark.parametrize("task,place,other,foreign", [
    ("mpox", "Sierra Leone", "measles", "United States"),
    ("measles", "Canada", "dengue", "Brazil"),
    ("dengue", "Peru", "cholera", "France"),
])
@pytest.mark.parametrize("host", ["journal.example", "renamed.example"])
def test_journal_sibling_wrong_scope_cannot_borrow_page_task(tmp_path, task, place, other, foreign, host):
    html = f'<h1>Journal issue</h1><p>{task} outbreak in {place}.</p><h2>Research articles</h2>'
    html += f'<li><a href="/article/other">Whole-genome sequencing of {other} in {foreign}</a> We strengthen genomic surveillance.</li>'
    assert candidates(tmp_path, html, task, place, host) == []


@pytest.mark.parametrize("host", ["journal.example", "renamed.example"])
def test_switching_task_admits_the_same_relevant_research_article_without_product_keywords(tmp_path, host):
    html = '<h1>Journal issue</h1><p>Mpox in Sierra Leone.</p><li><a href="/article/other">Whole-genome sequencing of measles in United States</a></li>'
    rows = candidates(tmp_path, html, "measles", "United States", host)
    assert [r["url"] for r in rows] == ["https://"+host+"/article/other"]


@pytest.mark.parametrize("label,path", [("Alex Researcher", "/people/alex"), ("Institute of Science", "/organization/science")])
def test_person_or_organization_anchor_does_not_inherit_entire_post_purpose(tmp_path, label, path):
    html = '<h1>Research post</h1><p>Our mpox report presents Sierra Leone surveillance data. '
    html += f'Thanks to <a href="{path}">{label}</a> for the collaboration.</p>'
    assert candidates(tmp_path, html) == []


@pytest.mark.parametrize("location", ["Brazil", "France"])
def test_same_disease_foreign_report_cannot_inherit_parent_geography(tmp_path, location):
    html = '<h1>Dengue Peru report</h1><ul>'
    html += f'<li><a href="/reports/foreign">Dengue case counts in {location}</a></li></ul>'
    assert candidates(tmp_path, html, "dengue", "Peru") == []


def test_scoped_anonymous_data_and_report_navigation_remain_acquirable(tmp_path):
    html = '<h1>Measles Canada observations 2025</h1><h2>Data availability</h2>'
    html += '<p>The underlying observations: <a href="https://unknown.example/export/42" download type="text/csv"></a></p>'
    html += '<h2>Measles annual reports</h2><a href="/archive">Report series</a>'
    rows = candidates(tmp_path, html, "measles", "Canada")
    assert {r["url"] for r in rows} == {"https://unknown.example/export/42", "https://journal.example/archive"}
    assert rows[0]["url"] == "https://unknown.example/export/42"


def test_data_relation_cannot_spread_to_next_person_link_in_same_paragraph(tmp_path):
    html = '<h1>Measles Canada report</h1><h2>Data availability</h2><p>Underlying data '
    html += '<a href="https://data.example/cases.csv">Download CSV</a>. Contact <a href="/people/alex">Alex Researcher</a>.</p>'
    rows = candidates(tmp_path, html, "measles", "Canada")
    assert [r["url"] for r in rows] == ["https://data.example/cases.csv"]


def test_french_scoped_data_keeps_unicode_and_country_boundary(tmp_path):
    html = '<h1>Cholera France 2025</h1><h2>Disponibilit\u00e9 des donn\u00e9es</h2>'
    html += '<a href="https://fichiers.example/42" download type="text/csv">T\u00e9l\u00e9charger les donn\u00e9es</a>'
    html += '<h2>Autres rapports</h2><a href="/foreign">Cholera au Br\u00e9sil: rapport de surveillance</a>'
    rows = candidates(tmp_path, html, "cholera", "France")
    assert [r["url"] for r in rows] == ["https://fichiers.example/42"]


def test_unproven_link_bonus_cannot_outrank_verified_task_report():
    verified = {"url":"https://authority.example/report", "target_fit_status":"verified_target",
                "data_product_type":"official_surveillance_report"}
    unproven = {"url":"https://other.example/opaque", "resource_link_selection_score":60,
                "resource_link_selection_reasons":["task_page_same_host_resource"],
                "target_verification_status":"candidate_task_record_source"}
    assert acquisition_priority(verified) > acquisition_priority(unproven)


def test_scoped_real_data_keeps_priority_over_unknown_navigation(tmp_path):
    rows = candidates(tmp_path, '<h1>Measles Canada report</h1><h2>Data availability</h2><a href="https://files.example/cases.csv">Download data</a>', "measles", "Canada")
    data = resource_source_entry(rows[0], parent={"source_id":"parent"}, depth=1)
    unknown = {"url":"https://other.example/archive", "target_fit_status":"task_record_collection_candidate"}
    assert acquisition_priority(data) > acquisition_priority(unknown)


@pytest.mark.parametrize("label,href", [("PDF", "/files/article.pdf"), ("Supplementary material", "/supplement/42"), ("Table 1", "/table/1")])
def test_scoped_article_attachments_do_not_need_disease_in_anchor(tmp_path, label, href):
    html = '<article><h1>Measles transmission in Canada during 2025</h1>'
    html += f'<a href="{href}">{label}</a></article>'
    assert [r["url"] for r in candidates(tmp_path, html, "measles", "Canada")] == ["https://journal.example"+href]


def test_bounded_queue_spends_second_target_on_report_not_wrong_sibling(tmp_path, monkeypatch):
    import requests
    from test_acquisition_queue import setup_node
    from test_evidence_resource_discovery import _source
    from data_collection_workflow.nodes.content_processing import content_fetch_and_parse
    runtime, state, visits = setup_node(monkeypatch, tmp_path, targets=2)
    index = _source("https://journal.example/issue", 1)
    index.update(target_fit_status="verified_target", data_product_type="surveillance_dashboard")
    report = _source("https://authority.example/report", 2)
    report.update(target_fit_status="verified_target", data_product_type="official_surveillance_report")
    state["source_registry"] = [index, report]
    class Response:
        status_code = 200
        def __init__(self, url):
            self.url = url
            self.headers = {"content-type": "text/html" if url == index["url"] else "text/plain"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def iter_content(self, size):
            yield (b'<h1>Journal issue</h1><p>Example fever in Canada.</p><li><a href="/wrong">Dengue transmission in Brazil</a> Genomic surveillance.</li>'
                   if self.url == index["url"] else b'Example fever in Canada during 2025: 12 confirmed cases.')
    def get(url, **kwargs): visits.append(url); return Response(url)
    monkeypatch.setattr(requests, "get", get)
    with runtime.activate(): result = content_fetch_and_parse(state)
    assert visits == [index["url"], report["url"]]
    assert runtime.ledger.snapshot()["used"]["source_targets"] == 2
    assert not any(s["url"].endswith("/wrong") for s in result["source_registry"])


def test_publication_inference_does_not_bury_current_period_report():
    state = {"structured_task": {"disease":"measles", "location":"Canada", "start_date":"2025-01-01", "end_date":"2025-12-31"}}
    base = {"url":"https://report.example/one", "title":"Measles surveillance in Canada",
        "snippet":"Published 2026. Measles in Canada during 2025: 12 confirmed cases.",
        "published_date":"2026-03-01", "target_verification_status":"temporal_mismatch",
        "date_fit":"mismatch", "disease_fit":"match", "geography_fit":"match",
        "task_fit_evidence_origin":"discovery_metadata", "data_product_type":"surveillance_report"}
    wrong = {**base, "snippet":"Measles in Canada during 2024: 12 confirmed cases."}
    explicit = {**base, "task_fit_evidence_origin":"explicit_constraint"}
    unknown = {"url":"https://unknown.example/", "data_product_type":"surveillance_report"}
    assert acquisition_priority(base, state=state) >= acquisition_priority(unknown, state=state)
    assert acquisition_priority(wrong, state=state) < acquisition_priority(unknown, state=state)
    assert acquisition_priority(explicit, state=state) < acquisition_priority(unknown, state=state)


@pytest.mark.parametrize("heading", ["Related articles", "Recommended reading"])
def test_related_section_allows_own_target_paper_and_rejects_foreign_sibling(tmp_path, heading):
    html = '<h1>Current issue</h1><h2>'+heading+'</h2>'
    html += '<a href="/target">Measles transmission in Canada in 2025</a>'
    html += '<a href="/wrong">Dengue transmission in Brazil in 2025</a>'
    assert [r["url"] for r in candidates(tmp_path, html, "measles", "Canada")] == ["https://journal.example/target"]


def test_recommended_target_navigation_is_not_an_editorial_file(tmp_path):
    html = '<h1>Index</h1><a href="/collection">Related articles: measles in Canada</a>'
    assert [r["url"] for r in candidates(tmp_path, html, "measles", "Canada")] == ["https://journal.example/collection"]


@pytest.mark.parametrize("wrapper", ["tr", "li"])
def test_anonymous_attachment_uses_its_report_row_without_promoting_row_author(tmp_path, wrapper):
    html = '<h1>Publications</h1><'+wrapper+'><a href="/paper">Measles in Canada during 2025</a> '
    html += '<a href="/paper.pdf">PDF</a> by <a href="/people/alex">Alex Researcher</a></'+wrapper+'>'
    urls = {r["url"] for r in candidates(tmp_path, html, "measles", "Canada")}
    assert urls == {"https://journal.example/paper", "https://journal.example/paper.pdf"}
