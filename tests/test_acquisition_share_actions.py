"""Offline regressions for explicit social actions beneath data headings."""
import socket
from html import escape
from urllib.parse import quote

import pytest

from data_collection_workflow.resource_discovery import task_resource_candidates
from data_collection_workflow.document_acquisition import parse_response


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    def denied(*args, **kwargs):
        raise AssertionError("external network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)


def discover(tmp_path, html, disease, location):
    doc = parse_response(html.encode(), url="https://journal.example/paper", source_id="parent",
                         session_dir=tmp_path, content_type="text/html")
    state = {"structured_task": {"disease": disease, "location": location,
             "start_date": "2025-01-01", "end_date": "2025-12-31"}}
    return task_resource_candidates(doc, {}, state)


@pytest.mark.parametrize("disease,location", [("measles", "Canada"), ("Example fever", "Example Region")])
@pytest.mark.parametrize("label_location", ["text", "title", "aria-label"])
def test_explicit_share_action_does_not_inherit_data_availability(tmp_path, disease, location, label_location):
    paper = f"{disease} observations in {location} during 2025"
    action_url = "https://social.example/action?title=" + quote(paper)
    attrs = "" if label_location == "text" else f' {label_location}="Share on Example Network"'
    content = "Share on Example Network" if label_location == "text" else ""
    html = f'<article><h1>{paper}</h1><h2>Data Availability</h2>'
    html += '<a href="https://data.example/cases.csv" download>Download data</a>'
    html += f'<a href="{escape(action_url)}"{attrs}>{content}</a></article>'
    rows = discover(tmp_path, html, disease, location)
    assert {row["url"] for row in rows} == {"https://data.example/cases.csv"}


@pytest.mark.parametrize("href", [
    "https://repository.example/share/opaque-token",
    "https://repository.example/files/cases.csv?share_token=opaque",
    "https://repository.example/export/42?access=shared",
])
def test_actual_dataset_with_share_access_url_remains_acquirable(tmp_path, href):
    html = '<article><h1>Measles observations in Canada during 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="{escape(href)}" download type="text/csv">Download underlying data</a></article>'
    rows = discover(tmp_path, html, "measles", "Canada")
    assert [row["url"] for row in rows] == [href]


def test_repository_html_dataset_entry_is_not_a_social_share_action(tmp_path):
    href = "https://repository.example/share/opaque-token"
    html = '<article><h1>Measles observations in Canada during 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="{href}">Underlying dataset</a></article>'
    rows = discover(tmp_path, html, "measles", "Canada")
    assert [row["url"] for row in rows] == [href]


def test_aria_action_is_preserved_when_title_describes_parent_data(tmp_path):
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += '<a href="https://social.example/action" title="Measles Canada data" aria-label="Share on Example Network">icon</a></article>'
    assert discover(tmp_path, html, "measles", "Canada") == []


@pytest.mark.parametrize("attribute", ["title", "aria-label"])
def test_explicit_french_share_action_cannot_borrow_data_section(tmp_path, attribute):
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="https://social.example/action" {attribute}="Partager sur un r\u00e9seau"></a></article>'
    assert discover(tmp_path, html, "measles", "Canada") == []


def test_dataset_title_and_accessible_label_are_preserved_without_becoming_action(tmp_path):
    href = "https://repository.example/share/opaque-token"
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="{href}" title="Underlying dataset" aria-label="Open underlying dataset">Data</a></article>'
    rows = discover(tmp_path, html, "measles", "Canada")
    assert [row["url"] for row in rows] == [href]
    assert rows[0]["provenance"]["title"] == "Underlying dataset"
    assert rows[0]["provenance"]["aria_label"] == "Open underlying dataset"


@pytest.mark.parametrize("path", ["/login", "/privacy"])
def test_data_words_do_not_override_explicit_login_or_policy_endpoint(tmp_path, path):
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="https://portal.example{path}" title="Surveillance data"></a></article>'
    assert discover(tmp_path, html, "measles", "Canada") == []


def test_survey_results_still_remain_a_data_product(tmp_path):
    href = "https://repository.example/survey/results"
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="{href}">Surveillance data</a></article>'
    assert [row["url"] for row in discover(tmp_path, html, "measles", "Canada")] == [href]


def test_parent_share_prose_does_not_change_download_link_purpose(tmp_path):
    href = "https://repository.example/cases.csv"
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<p>Share this paper with colleagues. <a href="{href}">Download underlying data</a></p></article>'
    assert [row["url"] for row in discover(tmp_path, html, "measles", "Canada")] == [href]


@pytest.mark.parametrize("path", ["/share/login", "/login/share/opaque-token", "/share/privacy"])
def test_nested_utility_endpoint_cannot_use_share_data_exception(tmp_path, path):
    html = '<article><h1>Measles Canada 2025</h1><h2>Data Availability</h2>'
    html += f'<a href="https://portal.example{path}">Surveillance data</a></article>'
    assert discover(tmp_path, html, "measles", "Canada") == []
